"""Supervised fine-tuning example for NV-Reason-CT.

The custom collator keeps NIfTI paths out of the chat-template text and routes
them through the downloaded processor's ``images3d=`` interface. The stock
Qwen3.5 2D vision tower remains frozen; the 3D vision encoder, multimodal
projector, and language model can be trained or frozen independently.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import datasets
import torch
import transformers
from accelerate import PartialState
from datasets import disable_caching
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser


TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from vlm_callbacks import (  # noqa: E402
    PerModalityProcessorSaveCallback,
    write_per_modality_processor_configs,
)


disable_caching()
logger = logging.getLogger(__name__)


@dataclass
class VLMScriptArguments(ScriptArguments):
    """NV-Reason-CT-specific command-line arguments."""

    dataset_path: str = field(
        default="sft.jsonl", metadata={"help": "Dataset JSONL filename or path."}
    )
    dataset_root: str | None = field(
        default="datalists", metadata={"help": "Root for a relative dataset_path."}
    )
    image_dir: str = field(
        default="images", metadata={"help": "Root for relative NIfTI paths."}
    )

    freeze_llm: bool = field(
        default=False, metadata={"help": "Freeze the language model and LM head."}
    )
    freeze_head: bool = field(
        default=False,
        metadata={"help": "Freeze input token embeddings and the output LM head."},
    )
    freeze_visual: bool = field(
        default=False, metadata={"help": "Freeze the 3D vision encoder."}
    )
    freeze_merger: bool = field(
        default=False, metadata={"help": "Freeze the 3D multimodal projector."}
    )
    freeze_show: bool = field(
        default=False, metadata={"help": "Log every frozen parameter name."}
    )
    completion_only: bool = field(
        default=True,
        metadata={"help": "Compute loss only on the final assistant response."},
    )
    per_group_lr: bool = field(
        default=True,
        metadata={
            "help": "Use paper component LRs: 3D vision encoder 0.1x, "
            "projector 5x, LLM 1x."
        },
    )


def print_input_config() -> None:
    args = sys.argv[1:]
    if "--config" in args and PartialState().is_main_process:
        with open(args[args.index("--config") + 1], encoding="utf-8") as file:
            print("Input config:", file.read().strip())


def _resolve_relative(root: str | None, path: str) -> str:
    if os.path.isabs(path) or not root:
        return path
    return os.path.join(root, path)


def _format_sample(sample: Dict[str, Any], image_dir: str) -> Dict[str, Any]:
    """Resolve nested image entries while preserving the public JSONL schema."""
    for message in sample["messages"]:
        for content in message.get("content", []):
            if content.get("type") == "image":
                image_path = content.get("image") or sample.get("image")
                if not image_path:
                    raise ValueError(f"Sample {sample.get('id')} has an image item without a path")
                content["image"] = _resolve_relative(image_dir, image_path)
                content.pop("text", None)
            elif content.get("type") == "text":
                content.pop("image", None)
    return sample


def load_training_dataset(script_args: VLMScriptArguments):
    dataset_file = _resolve_relative(script_args.dataset_root, script_args.dataset_path)
    train_dataset = datasets.load_dataset(
        "json",
        data_files=dataset_file,
        split="train",
        streaming=script_args.dataset_streaming,
    )
    required = {"messages", "anatomy_region"}
    missing = required - set(train_dataset.column_names)
    if missing:
        raise ValueError(f"SFT dataset is missing required columns: {sorted(missing)}")
    train_dataset = train_dataset.map(
        _format_sample,
        fn_kwargs={"image_dir": script_args.image_dir},
    )
    if not isinstance(train_dataset, datasets.IterableDataset):
        logger.info("Loaded %d SFT examples from %s", len(train_dataset), dataset_file)
    return train_dataset


def get_padding_token_ids(processor) -> torch.Tensor:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokens = [
        "<|image|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    if hasattr(tokenizer, "image_token"):
        tokens.append(tokenizer.image_token)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if tokenizer.pad_token_id is not None:
        token_ids.append(tokenizer.pad_token_id)
    return torch.tensor(sorted({token_id for token_id in token_ids if token_id is not None}))


class VLM_SFT_DataCollator:
    """Build Qwen chat text and a batched 3D CT tensor."""

    def __init__(self, processor, completion_only: bool = True):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.padding_token_ids = get_padding_token_ids(processor)
        self.completion_only = completion_only
        self.assistant_token_ids = self.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages_batch = [example["messages"] for example in examples]
        texts = [
            self.processor.apply_chat_template(messages, tokenize=False)
            for messages in messages_batch
        ]

        image_paths: list[str] = []
        anatomy_regions: list[str | None] = []
        for example, messages in zip(examples, messages_batch):
            region = example.get("anatomy_region")
            for message in messages:
                for content in message.get("content", []):
                    if content.get("type") == "image":
                        image_paths.append(content["image"])
                        anatomy_regions.append(region)

        if image_paths:
            batch = self.processor(
                text=texts,
                images3d=image_paths,
                anatomy_region=anatomy_regions,
                return_tensors="pt",
                padding=True,
            )
        else:
            batch = self.processor(text=texts, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[torch.isin(labels, self.padding_token_ids)] = -100

        if self.completion_only:
            marker = self.assistant_token_ids
            marker_len = len(marker)
            for row in range(labels.shape[0]):
                input_ids = batch["input_ids"][row].tolist()
                assistant_start = None
                for index in range(len(input_ids) - marker_len, -1, -1):
                    if input_ids[index : index + marker_len] == marker:
                        assistant_start = index + marker_len
                        break
                if assistant_start is None:
                    logger.warning(
                        "No assistant marker in batch row %d; masking the complete sample", row
                    )
                    labels[row] = -100
                else:
                    labels[row, :assistant_start] = -100

        batch["labels"] = labels
        return batch


def create_param_groups(model, base_lr: float, weight_decay: float):
    """Create the component-wise learning-rate groups used in the paper."""

    def split_decay(named_parameters):
        decay, no_decay = [], []
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            lower_name = name.lower()
            if (
                parameter.ndim < 2
                or name.endswith(".bias")
                or "norm" in lower_name
                or "conv1d.weight" in lower_name
            ):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return decay, no_decay

    llm_parameters = list(model.model.language_model.named_parameters())
    llm_parameter_ids = {id(parameter) for _, parameter in llm_parameters}
    llm_parameters.extend(
        (f"lm_head.{name}", parameter)
        for name, parameter in model.lm_head.named_parameters()
        if id(parameter) not in llm_parameter_ids
    )
    components = [
        ("projector", model.model.vision3d.merger.named_parameters(), base_lr * 5.0),
        (
            "vision_encoder",
            model.model.vision3d.sub_vision.named_parameters(),
            base_lr * 0.1,
        ),
        ("language", llm_parameters, base_lr),
    ]

    groups = []
    for component, named_parameters, learning_rate in components:
        decay, no_decay = split_decay(named_parameters)
        if decay:
            groups.append(
                {
                    "params": decay,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "name": f"{component}_decay",
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                    "name": f"{component}_no_decay",
                }
            )
    return groups


def _set_trainable_components(model, script_args: VLMScriptArguments) -> None:
    # CT volumes never use the stock Qwen 2D tower.
    for parameter in model.model.visual.parameters():
        parameter.requires_grad = False

    if script_args.freeze_llm:
        for parameter in model.model.language_model.parameters():
            parameter.requires_grad = False
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = False
    if script_args.freeze_head:
        for parameter in model.model.language_model.embed_tokens.parameters():
            parameter.requires_grad = False
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = False
    if script_args.freeze_visual:
        for parameter in model.model.vision3d.sub_vision.parameters():
            parameter.requires_grad = False
    if script_args.freeze_merger:
        for parameter in model.model.vision3d.merger.parameters():
            parameter.requires_grad = False

    if script_args.freeze_show:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                logger.info("Frozen: %s", name)


def _full_numel(parameter) -> int:
    return getattr(parameter, "ds_numel", None) or parameter.numel()


def main(script_args, training_args, model_args) -> None:
    set_seed(training_args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint and training_args.resume_from_checkpoint is None:
        logger.info("Resuming from detected checkpoint %s", last_checkpoint)

    train_dataset = load_training_dataset(script_args)
    if isinstance(train_dataset, datasets.IterableDataset):
        training_args.dataloader_drop_last = True
        training_args.accelerator_config.dispatch_batches = False
        training_args.ignore_data_skip = True

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        device_map=None,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=False,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    processor.tokenizer.padding_side = "right"
    if getattr(processor, "image_processor_3d", None) is None:
        raise RuntimeError("The model repository does not contain image_processor_3d")

    # Keep the processor and 3D tower normalization synchronized with the
    # released checkpoint. Do not silently override it from a training config.
    normalize_mode = processor.image_processor_3d.normalize_mode
    model.config.normalize_mode = normalize_mode
    model.model.vision3d.normalize_mode = normalize_mode

    _set_trainable_components(model, script_args)
    total_parameters = sum(_full_numel(parameter) for parameter in model.parameters())
    trainable_parameters = sum(
        _full_numel(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters == 0:
        raise ValueError("All model parameters are frozen")
    logger.info(
        "Parameters: %.1fM total, %.1fM trainable (%.2f%%)",
        total_parameters / 1e6,
        trainable_parameters / 1e6,
        100 * trainable_parameters / total_parameters,
    )

    optimizer = None
    if script_args.per_group_lr:
        groups = create_param_groups(
            model,
            base_lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
        for group in groups:
            logger.info(
                "Optimizer group %s: lr=%g, weight_decay=%g",
                group["name"],
                group["lr"],
                group["weight_decay"],
            )
        optimizer = torch.optim.AdamW(groups, fused=True)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VLM_SFT_DataCollator(
            processor, completion_only=script_args.completion_only
        ),
        processing_class=processor,
        optimizers=(optimizer, None),
        callbacks=[PerModalityProcessorSaveCallback(processor)],
    )

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(
            dataset_name=script_args.dataset_name,
            tags=["ct", "3d-vlm", "nv-reason-ct"],
        )
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        if getattr(trainer.model, "generation_config", None) is not None:
            trainer.model.generation_config.use_cache = True
            trainer.model.generation_config.save_pretrained(training_args.output_dir)
        write_per_modality_processor_configs(processor, training_args.output_dir)


if __name__ == "__main__":
    print_input_config()
    parser = TrlParser((VLMScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.run_name is None:
        training_args.run_name = Path(training_args.output_dir).name
    if not training_args.use_cpu and torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA devices are visible")
    main(script_args, training_args, model_args)
