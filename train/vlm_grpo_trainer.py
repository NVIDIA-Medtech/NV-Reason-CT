# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GRPO trainer for NV-Reason-CT.

Adapted from the TRL 1.2.0 GRPO trainer:
https://github.com/huggingface/trl/blob/v1.2.0/trl/trainer/grpo_trainer.py

This adaptation supports the VLM3D dual-encoder model, which uses the
**image** pathway with 3D volumes routed through the downloaded processor's
`images3d=` interface.

What this subclass changes vs the stock TRL 1.2.0 GRPOTrainer:

* Calls the processor with `images3d=` (our custom 3D kwarg) instead of
  `images=` (the stock 2D kwarg). The 3D loader (`ImageLoader3D` with
  `merge_size=1`) produces `pixel_values` shape `(B, 1, T, H, W)` plus
  `image_grid_thw=[24,24,24]` and `mm_token_type_ids` — same OUTPUT keys
  as the stock image path, so most downstream trainer logic is reused.

* Per-sample slicing for the 5D `pixel_values` shape during chunked
  log-prob computation. Stock code assumes a flattened `(num_tokens,
  channels)` layout indexed by `image_grid_thw.prod(-1).cumsum()`; our
  5D tensor needs simple `pixel_values[start:start+batch_size]`.

* `prepare_image_messages` inserts `{type: image}` placeholders into the
  user message content -- the same placeholder shape stock TRL uses for
  ordinary 2D images, since Qwen3.5's chat template renders `<|image_pad|>`
  for both. The 3D-vs-2D dispatch happens inside our processor based on
  the kwarg name, not on the chat-template tokens.

* Reserved multimodal control-token IDs are suppressed while sampling text
  completions. The existing post-generation check remains fatal as an
  invariant guard in case a generation path bypasses suppression.
"""

import copy
import logging
import time
from contextlib import nullcontext
from typing import Any, Optional, Union

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate.utils import gather_object, is_peft_model
from transformers import Trainer

from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import unwrap_model_for_generation
from trl.trainer import grpo_trainer as trl_grpo_module
from trl.trainer.utils import (
    entropy_from_logits,
    nanmax,
    nanmin,
    nanstd,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_tensor_dict,
)

logger = logging.getLogger(__name__)


FORBIDDEN_MULTIMODAL_COMPLETION_TOKENS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def prepare_image_messages(messages: list[dict[str, Any]], num_images: int = 1) -> None:
    """Insert {"type": "image"} placeholders into the user message content list.

    Mirrors stock TRL's `prepare_multimodal_messages` shape but is invoked
    explicitly by us so we can keep the trainer's own auto-detection logic
    inactive for our custom `images3d=` flow. Qwen3.5's chat template
    renders `<|image_pad|>` for `{type: image}` -- which is what we want
    for the 3D path (the processor's `images3d=` kwarg expands those
    placeholders into 13824 vision tokens via `image_processor_3d.merge_size=1`).
    """
    image_included = False
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            if isinstance(content, str):
                message["content"] = [{"type": "text", "text": content}]
        elif role == "user":
            if isinstance(content, str) and not image_included:
                placeholders = [{"type": "image"}] * num_images
                message["content"] = [*placeholders, {"type": "text", "text": content}]
                image_included = True
            elif isinstance(content, str):
                message["content"] = [{"type": "text", "text": content}]
        elif role == "assistant":
            if isinstance(content, str):
                message["content"] = [{"type": "text", "text": content}]
        else:
            raise ValueError(f"Invalid role in message: {role}")


class VLM3D_GRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs.get("args")
        policy_model = kwargs.get("model")
        needs_vlm3d_reference = (
            training_args is not None
            and policy_model is not None
            and training_args.beta != 0.0
            and not is_peft_model(policy_model)
        )

        original_create_model_from_path = None
        if needs_vlm3d_reference:
            # TRL reconstructs the reference policy from the checkpoint's
            # architecture name. NV-Reason-CT is a dynamic remote-code class,
            # not a class exported by Transformers, so reuse the policy class
            # resolved by AutoModel. Patch only the construction performed by
            # super().__init__, then restore TRL's process-global factory.
            policy_model_class = type(policy_model)
            try:
                policy_dtype = next(policy_model.parameters()).dtype
            except StopIteration:
                policy_dtype = torch.bfloat16
            policy_attn_implementation = getattr(
                policy_model.config,
                "_attn_implementation",
                None,
            )
            policy_revision = getattr(policy_model.config, "_commit_hash", None)
            reference_model_defaults = {
                "dtype": policy_dtype,
                "trust_remote_code": True,
                "attn_implementation": (
                    policy_attn_implementation or "flash_attention_2"
                ),
            }
            if policy_revision:
                reference_model_defaults["revision"] = policy_revision

            def create_vlm3d_reference_model(model_id, **model_kwargs):
                for key, value in reference_model_defaults.items():
                    model_kwargs.setdefault(key, value)
                return policy_model_class.from_pretrained(
                    model_id,
                    **model_kwargs,
                )

            original_create_model_from_path = (
                trl_grpo_module.create_model_from_path
            )
            trl_grpo_module.create_model_from_path = (
                create_vlm3d_reference_model
            )

        try:
            super().__init__(*args, **kwargs)
        finally:
            if original_create_model_from_path is not None:
                trl_grpo_module.create_model_from_path = (
                    original_create_model_from_path
                )

        if needs_vlm3d_reference:
            if self.ref_model is None:
                raise RuntimeError(
                    "beta > 0 requires an NV-Reason-CT reference model, but TRL did not "
                    "create one."
                )

            reference_model = self.ref_model
            while hasattr(reference_model, "module"):
                reference_model = reference_model.module
            if not isinstance(reference_model, policy_model_class):
                raise TypeError(
                    "TRL created the wrong beta > 0 reference model type: "
                    f"{type(reference_model).__name__}"
                )

            policy_dtype = next(policy_model.parameters()).dtype
            reference_dtype = next(reference_model.parameters()).dtype
            if reference_dtype != policy_dtype:
                raise TypeError(
                    "NV-Reason-CT beta > 0 reference model dtype does not match the "
                    f"policy: reference={reference_dtype}, policy={policy_dtype}."
                )
            logger.info(
                "[vlm3d-grpo] beta > 0 reference model loaded: class=%s "
                "dtype=%s beta=%s",
                type(reference_model).__name__,
                reference_dtype,
                training_args.beta,
            )

        self.image_token = getattr(self.processing_class, "image_token", None)
        self.image_token_id = getattr(self.processing_class, "image_token_id", None)
        self.generation_config.use_cache = True
        self._logged_generation_sampling = False

        tokenizer = self.processing_class.tokenizer
        resolved_forbidden_tokens = {
            tokenizer.convert_tokens_to_ids(token): token
            for token in FORBIDDEN_MULTIMODAL_COMPLETION_TOKENS
        }
        if None in resolved_forbidden_tokens:
            missing_tokens = [
                token
                for token in FORBIDDEN_MULTIMODAL_COMPLETION_TOKENS
                if tokenizer.convert_tokens_to_ids(token) is None
            ]
            raise RuntimeError(
                "Could not resolve required multimodal control tokens in the tokenizer: "
                f"{missing_tokens}"
            )
        self.forbidden_completion_token_ids = resolved_forbidden_tokens

        # Suppression is applied only to scores for newly generated tokens;
        # it does not modify the legitimate multimodal control IDs already in
        # the prompt. Preserve any checkpoint/config-provided suppression list
        # and add the VLM3D multimodal IDs that are invalid in assistant text.
        existing_suppress_tokens = set(
            getattr(self.generation_config, "suppress_tokens", None) or []
        )
        existing_suppress_tokens.update(
            self.forbidden_completion_token_ids.keys()
        )
        self.generation_config.suppress_tokens = sorted(
            existing_suppress_tokens
        )

        if self.accelerator.is_main_process:
            logger.info(
                "[GRPO][vlm3d][generation] suppressing multimodal completion "
                "tokens: %s",
                {
                    token_text: token_id
                    for token_id, token_text in sorted(
                        self.forbidden_completion_token_ids.items()
                    )
                },
            )

    def _log_timing_rank0(self, message: str) -> None:
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None or accelerator.process_index == 0:
            logger.info(message)

    def _raise_on_forbidden_multimodal_completion_tokens(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        completion_ids_list: list[list[int]],
        inputs: list[dict[str, Any]],
    ) -> None:
        """Abort if text generation emits a reserved multimodal control token."""
        for token_id, token_text in self.forbidden_completion_token_ids.items():
            matches = torch.nonzero(
                (completion_ids == token_id) & completion_mask.bool(),
                as_tuple=False,
            )
            if matches.numel() == 0:
                continue

            row_idx, token_offset = matches[0].tolist()
            sample_id = inputs[row_idx].get("id", "<unknown>")
            tokenizer = self.processing_class.tokenizer
            raw_completion = tokenizer.decode(
                completion_ids_list[row_idx],
                skip_special_tokens=False,
            )
            readable_completion = tokenizer.decode(
                completion_ids_list[row_idx],
                skip_special_tokens=True,
            )

            logger.critical(
                "[GRPO][vlm3d][FATAL] Model generated forbidden multimodal control "
                "token %s (id=%d). global_step=%d rank=%d sample_id=%s "
                "local_completion=%d token_offset=%d\n"
                "Completion with special tokens:\n%s\n"
                "Completion with special tokens removed:\n%s",
                token_text,
                token_id,
                self.state.global_step,
                self.accelerator.process_index,
                sample_id,
                row_idx,
                token_offset,
                raw_completion,
                readable_completion,
            )
            raise RuntimeError(
                "GRPO generated a forbidden multimodal control token "
                f"{token_text} (id={token_id}) at completion offset {token_offset}. "
                "The policy may have degenerated or the generation configuration is "
                "invalid. Aborting before multimodal loss replay."
            )

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Standard buffered-iteration logic with one tweak: our 5D pixel_values
        is already per-sample, so `shuffle_sequence_dict` + `split_tensor_dict`
        work directly without the video-style split/unsplit helpers."""
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation
                )
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        # VLM3D image kwargs (5D pixel_values from images3d= route)
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        # Accept (and ignore) the kwargs stock TRL might pass through; we
        # don't use them in the VLM3D image-path forward.
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        image_position_ids=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Per-sample chunking that matches our 5D `pixel_values` layout.
        Stock TRL's image-path chunking assumes a flattened `(num_tokens, C)`
        tensor sliced by `image_grid_thw.prod(-1).cumsum()`; our 3D pixel_values
        is `(B, 1, T, H, W)` per-sample, so we slice along dim 0 directly.
        """
        batch_size = batch_size or input_ids.size(0)
        all_logps: list[torch.Tensor] = []
        all_entropies: list[torch.Tensor] = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs: dict[str, Any] = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
            }

            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw[start : start + batch_size]
            if mm_token_type_ids is not None:
                model_inputs["mm_token_type_ids"] = mm_token_type_ids[start : start + batch_size]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        # Stock kwargs we don't use, accepted for compatibility
        pixel_attention_mask=None,
        image_sizes=None,
        image_position_ids=None,
    ):
        """Liger-loss helper: forward through model.model and return the last
        hidden state. Same logic as stock TRL but only threads our VLM3D image
        kwargs (no video). Liger is disabled by default in our YAML
        (`use_liger_kernel: false` -- liger has no Qwen3.5 integration as of
        writing), so this method is mostly here for completeness; it'll fire
        only if the user overrides the YAML.
        """
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        model_inputs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        if mm_token_type_ids is not None:
            model_inputs["mm_token_type_ids"] = mm_token_type_ids

        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        model_inputs["use_cache"] = False

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]
        return last_hidden_state

    def compute_liger_loss(self, unwrapped_model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )

        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if mean_kl is not None:
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).mean().item()
            )
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        normalizer = (
            self.current_gradient_accumulation_steps
            if mode == "train"
            else 1.0
        )
        return loss / normalizer

    def _compute_loss(self, model, inputs):
        mode = "train" if self.model.training else "eval"
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """VLM3D generation + reward computation. Routes 3D volumes through
        `processor(text=..., images3d=[paths])` so the chat template's
        `<|image_pad|>` placeholders expand to 13824 vision tokens
        (24*24*24, with image_processor_3d.merge_size=1)."""
        if self.use_vllm:
            raise NotImplementedError("VLM3D_GRPOTrainer only supports use_vllm=False.")
        if self.use_transformers_paged:
            raise NotImplementedError(
                "VLM3D_GRPOTrainer only supports use_transformers_paged=False."
            )

        t_start = time.time()
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        original_prompts = copy.deepcopy(prompts)

        # Pull the 3D volume path from each sample. `image` is the SFT-style
        # field name (single .nii.gz path per sample); `images` is the plural
        # form. Either is accepted.
        if "images" in inputs[0]:
            image_paths = [example.get("images") for example in inputs]
            # Flatten if list-of-lists
            image_paths = [
                p if not isinstance(p, list) else p[0] if p else None for p in image_paths
            ]
        elif "image" in inputs[0]:
            image_paths = [example.get("image") for example in inputs]
        else:
            image_paths = None

        kwargs: dict[str, Any] = {}
        has_images3d = image_paths is not None and any(p is not None for p in image_paths)
        if has_images3d:
            anatomy_regions = [example.get("anatomy_region") for example in inputs]
            # Insert {type: image} placeholders for the chat-template render.
            for prompt in prompts:
                if isinstance(prompt, list):
                    prepare_image_messages(prompt, num_images=1)
            kwargs = {
                "images3d": image_paths,
                "anatomy_region": anatomy_regions,
            }

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]

        t_processor_start = time.time()
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        t_processor_end = time.time()
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        t_prompt = time.time()
        self._log_timing_rank0(
            f"[TIMING][GRPO] mode={mode} processor_call={t_processor_end-t_processor_start:.2f}s "
            f"prompt_tokens_shape={tuple(prompt_ids.shape)}"
        )
        self._log_timing_rank0(
            f"[TIMING][GRPO] mode={mode} prompt_prep={t_prompt-t_start:.2f}s "
            f"batch={len(inputs)} has_images3d={has_images3d}"
        )

        # TRL 1.2 dropped both `truncate_with_protected_tokens` and the
        # `max_prompt_length` GRPOTrainer attribute. NV-Reason-CT prompts are
        # short (single CT volume + a single instruction), so truncation
        # isn't exercised in practice.
        max_prompt_length = getattr(self, "max_prompt_length", None) or getattr(
            self.args, "max_prompt_length", None
        )
        if max_prompt_length is not None:
            logger.warning(
                "[GRPO][vlm3d] max_prompt_length is set but TRL 1.2.0 removed the truncation helper "
                "we used to protect vision tokens; this trainer ignores max_prompt_length."
            )

        t_generate_start = time.time()
        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False)
            if self.is_fsdp_enabled
            else nullcontext(),
        ):
            prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
            if not self._logged_generation_sampling:
                logger.info(
                    "[GRPO][vlm3d][generation] "
                    f"use_cache={self.generation_config.use_cache} "
                    f"temperature={self.temperature} top_p={self.top_p} top_k={self.top_k} "
                    f"min_p={self.min_p} repetition_penalty={self.repetition_penalty} "
                    f"max_new_tokens={self.max_completion_length}"
                )
                self._logged_generation_sampling = True
            # transformers 5 dropped `use_model_defaults` from generate's
            # accepted kwargs; pass sampling overrides directly.
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                use_cache=True,
                disable_compile=True,
            )
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        t_generate_end = time.time()
        self._log_timing_rank0(
            f"[TIMING][GRPO] mode={mode} generate={t_generate_end-t_generate_start:.2f}s "
            f"prompt_tokens={prompt_ids.size(1)} completion_tokens={completion_ids.size(1)}"
        )

        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())
        ]
        self._raise_on_forbidden_multimodal_completion_tokens(
            completion_ids,
            completion_mask,
            completion_ids_list,
            inputs,
        )

        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = agg_completion_lengths.sum()

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        # Extend mm_token_type_ids to cover the completion portion with zeros
        # (completion tokens are pure text -- never multimodal). Mirrors the
        # block at trl/trainer/grpo_trainer.py:1962-1974 in stock TRL 1.2.
        # Without this, downstream forward passes get a mask of length
        # prompt_len while input_ids has length prompt_len+completion_len,
        # causing the IndexError inside get_rope_index.
        if "mm_token_type_ids" in prompt_inputs and prompt_inputs["mm_token_type_ids"] is not None:
            mm_token_type_ids = prompt_inputs["mm_token_type_ids"]
            prompt_inputs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)],
                dim=1,
            )

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            t_logps_start = time.time()
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    mm_token_type_ids=prompt_inputs.get("mm_token_type_ids"),
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        mm_token_type_ids=prompt_inputs.get("mm_token_type_ids"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            mm_token_type_ids=prompt_inputs.get("mm_token_type_ids"),
                        )
            else:
                ref_per_token_logps = None
            t_logps_end = time.time()
            self._log_timing_rank0(
                f"[TIMING][GRPO] mode={mode} logps_and_ref={t_logps_end-t_logps_start:.2f}s"
            )

        t_decode_start = time.time()
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        t_decode_end = time.time()
        self._log_timing_rank0(f"[TIMING][GRPO] mode={mode} decode={t_decode_end-t_decode_start:.2f}s")

        t_rewards_start = time.time()
        rewards_per_func = self._calculate_rewards(
            inputs, original_prompts, completions, completion_ids_list
        )
        t_rewards_end = time.time()
        self._log_timing_rank0(f"[TIMING][GRPO] mode={mode} rewards={t_rewards_end-t_rewards_start:.2f}s")

        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. "
                "Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())
        if has_images3d:
            # Do not populate self._logs["images"] for CT volumes. TRL treats
            # that field as a list of 2D images and W&B will try to open each
            # entry with wandb.Image(...). Our values are .nii.gz paths, so log
            # them as text metadata in the completions table instead.
            self._logs["extra"]["image3d_path"].extend(gather_object(image_paths))

        output: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "mm_token_type_ids" in prompt_inputs:
            output["mm_token_type_ids"] = prompt_inputs["mm_token_type_ids"]
        t_end = time.time()
        self._log_timing_rank0(f"[TIMING][GRPO] mode={mode} total_prepare_inputs={t_end-t_start:.2f}s")
        return output
