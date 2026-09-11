# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Group Relative Policy Optimization example for NV-Reason-CT.

The companion trainer adapts TRL GRPO to route NIfTI paths through the custom
``images3d=`` processor argument and to preserve 5D CT tensors throughout
generation and policy-log-probability computation.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import datasets
import torch
import transformers
from accelerate import PartialState
from datasets import disable_caching
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, ModelConfig, TrlParser


TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from vlm_callbacks import (  # noqa: E402
    PerModalityProcessorSaveCallback,
    write_per_modality_processor_configs,
)
from vlm_grpo_trainer import VLM3D_GRPOTrainer  # noqa: E402
from vlm_rewards import (  # noqa: E402
    accuracy_reward_hard,
    soft_band_punishment,
    structured_report_format_reward,
)


disable_caching()
logger = logging.getLogger(__name__)


@dataclass
class VLMScriptArguments:
    dataset_path: str = field(
        default="datalists/grpo.jsonl", metadata={"help": "Path to the dataset JSONL file."}
    )
    image_dir: str = field(
        default="images", metadata={"help": "Root for relative NIfTI paths."}
    )

    freeze_llm: bool = field(
        default=False, metadata={"help": "Freeze the language model and LM head."}
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
    reward_funcs: list[str] = field(
        default_factory=lambda: [
            "accuracy_reward_hard",
            "structured_report_format",
            "soft_band_punishment",
        ],
        metadata={"help": "Reward names from REWARD_REGISTRY."},
    )
    prompt: str | None = field(
        default=None, metadata={"help": "Override every dataset user prompt."}
    )


REWARD_REGISTRY = {
    "accuracy_reward_hard": accuracy_reward_hard,
    "structured_report_format": structured_report_format_reward,
    "soft_band_punishment": soft_band_punishment,
}


def print_input_config() -> None:
    args = sys.argv[1:]
    if "--config" in args and PartialState().is_main_process:
        with open(args[args.index("--config") + 1], encoding="utf-8") as file:
            print("Input config:", file.read().strip())


def _resolve_relative(root: str | None, path: str) -> str:
    if os.path.isabs(path) or not root:
        return path
    return os.path.join(root, path)


def _extract_user_prompt_text(sample: Dict[str, Any]) -> str:
    for message in sample["messages"]:
        if message.get("role") != "user":
            continue
        for content in message.get("content", []):
            if content.get("type") == "text" and content.get("text", "").strip():
                return content["text"].strip()
    raise ValueError(f"Sample {sample.get('id')} has no user text prompt")


def _format_grpo_sample(
    sample: Dict[str, Any],
    image_dir: str,
    prompt_override: str | None,
) -> Dict[str, Any]:
    prompt_text = prompt_override or _extract_user_prompt_text(sample)
    return {
        "id": sample["id"],
        "solution": sample["solution"],
        "prompt": [{"role": "user", "content": prompt_text}],
        "chat_template_kwargs": {"enable_thinking": True},
        "image": _resolve_relative(image_dir, sample["image"]),
        "anatomy_region": sample["anatomy_region"],
    }


def load_training_dataset(script_args: VLMScriptArguments):
    train_dataset = datasets.load_dataset("json", data_files=script_args.dataset_path, split="train")
    required = {"id", "image", "anatomy_region", "solution", "messages"}
    missing = required - set(train_dataset.column_names)
    if missing:
        raise ValueError(f"GRPO dataset is missing required columns: {sorted(missing)}")
    train_dataset = train_dataset.map(
        _format_grpo_sample,
        fn_kwargs={
            "image_dir": script_args.image_dir,
            "prompt_override": script_args.prompt,
        },
        remove_columns=train_dataset.column_names,
    )
    logger.info("Loaded %d GRPO examples from %s", len(train_dataset), script_args.dataset_path)
    return train_dataset


def _set_trainable_components(model, script_args: VLMScriptArguments) -> None:
    for parameter in model.model.visual.parameters():
        parameter.requires_grad = False
    if script_args.freeze_llm:
        for parameter in model.model.language_model.parameters():
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

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        device_map=None,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    model.register_for_auto_class("AutoModelForImageTextToText")
    model.config.text_config.use_cache = False
    _set_trainable_components(model, script_args)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    processor.tokenizer.padding_side = "left"
    if getattr(processor, "image_processor_3d", None) is None:
        raise RuntimeError("The model repository does not contain image_processor_3d")

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

    unknown_rewards = sorted(set(script_args.reward_funcs) - set(REWARD_REGISTRY))
    if unknown_rewards:
        raise ValueError(
            f"Unknown reward functions {unknown_rewards}; choices: {sorted(REWARD_REGISTRY)}"
        )
    reward_functions = [REWARD_REGISTRY[name] for name in script_args.reward_funcs]

    trainer = VLM3D_GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_functions,
        train_dataset=train_dataset,
        processing_class=processor,
        callbacks=[PerModalityProcessorSaveCallback(processor)],
    )

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.model.config.text_config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        if getattr(trainer.model, "generation_config", None) is not None:
            trainer.model.generation_config.use_cache = True
            trainer.model.generation_config.save_pretrained(training_args.output_dir)
        write_per_modality_processor_configs(processor, training_args.output_dir)


if __name__ == "__main__":
    print_input_config()
    parser = TrlParser((VLMScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.run_name is None:
        training_args.run_name = Path(training_args.output_dir).name
    if not training_args.use_cpu and torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA devices are visible")
    main(script_args, training_args, model_args)
