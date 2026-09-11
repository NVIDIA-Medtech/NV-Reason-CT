# NV-Reason-CT

NV-Reason-CT is a generative vision-language model for native 3D chest and
abdominal CT interpretation. It combines a Qwen3.5-4B language model with a 3D Vision Transformer
(Primus, initialized with Colipri weights) and supports abnormality classification,
structured report generation, visual question answering, and interactive
reasoning from NIfTI volumes.

This repository provides inference, supervised fine-tuning (SFT), and Group
Relative Policy Optimization (GRPO) examples for the
[NV-Reason-CT model](https://huggingface.co/nvidia/NV-Reason-CT). The complete
model package—including weights, custom model and processor code,
configuration, tokenizer, and preprocessing metadata—is maintained on Hugging
Face and is intentionally not duplicated here.

> NV-Reason-CT is for research and development only. It is not a medical
> device and must not be used as a substitute for professional clinical
> judgment, diagnosis, or treatment decisions. Model outputs can be incorrect
> and should be reviewed by qualified medical professionals.

## Contents

- [`inference.py`](inference.py): one-volume command-line inference
- [`train/vlm_sft_train.py`](train/vlm_sft_train.py): SFT example
- [`train/vlm_grpo_train.py`](train/vlm_grpo_train.py): GRPO entrypoint
- [`train/vlm_grpo_trainer.py`](train/vlm_grpo_trainer.py): 3D adaptation of
  the TRL GRPO trainer
- [`train/vlm_rewards.py`](train/vlm_rewards.py): region-aware verifiable
  rewards
- [`datalists/`](datalists): illustrative JSONL schemas; CT volumes are not
  included
- [`configs/`](configs): example SFT and GRPO configurations
- [`THIRD-PARTY-NOTICES`](THIRD-PARTY-NOTICES): PyPI dependencies and the
  TRL-adapted trainer attribution

## Model source

All examples load `nvidia/NV-Reason-CT` with
`AutoModelForImageTextToText` and `AutoProcessor`. Because the model defines a
custom 3D architecture and NIfTI processor, `trust_remote_code=True` is
required. On first use, Transformers downloads the model package from Hugging
Face and stores it in the local Hugging Face cache.

The command-line inference example accepts `--model` when a local model copy or
a different Hugging Face repository is needed. The training configurations use
`model_name_or_path: nvidia/NV-Reason-CT` by default.

## Model overview

- Language backbone: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- 3D vision encoder: Primus, initialized from COLIPRI
- Input: one-channel chest or abdominal CT in `.nii` or `.nii.gz` format
- Preprocessing: LPS orientation, 2 mm isotropic resampling, and an
  anatomy-aware `192 x 192 x 192` voxel crop
- Patch grid: `24 x 24 x 24` from non-overlapping `8 x 8 x 8` patches
- Visual interface: all 13,824 projected visual tokens are passed to the
  language model without spatial merging
- Position encoding: depth, height, and width are represented with 3D MRoPE

CT volumes are routed through the custom `images3d=` processor argument and
the Primus 3D tower.

![NV-Reason-CT overview](assets/nv_reason_ct_overview.png)

## Installation

Python 3.11 and a CUDA-capable PyTorch installation are recommended. A fresh
environment can be created with [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --seed --python 3.11 nvreasonct
source nvreasonct/bin/activate
```

For inference:

```bash
uv pip install -r requirements.txt
```

For training:

```bash
uv pip install -r requirements-train.txt
uv pip install flash-attn==2.8.3 --no-build-isolation
```

The training code and configs are pinned to Transformers 5.6.2, TRL 1.2.0,
and Accelerate 1.13.0 because the custom GRPO trainer follows those TRL
internals.

## Quick start: inference

The input CT must contain valid Hounsfield-unit values. Select `"chest"` or
`"abdomen"` so the processor applies the corresponding deterministic crop.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "nvidia/NV-Reason-CT"
ct_path = "path/to/volume.nii.gz"
region = "chest"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).eval().to("cuda")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Write a structured chest CT report."},
        ],
    }
]
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
inputs = processor(
    text=prompt,
    images3d=[ct_path],
    anatomy_region=region,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        use_cache=True,
    )

new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

The equivalent CLI command is:

```bash
python inference.py path/to/volume.nii.gz \
  --region chest \
  --prompt "Write a structured chest CT report."
```

Thinking is enabled by default for reports, questions, and reasoning tasks.
Use `--disable-thinking` only when a non-thinking response is required.


## Inspect the model input crop

The same preprocessing used for inference can be inspected without running the
model:

```python
import nibabel as nib
import numpy as np
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "nvidia/NV-Reason-CT",
    trust_remote_code=True,
)
cropped = processor.image_processor_3d.load_image(
    "path/to/volume.nii.gz",
    normalize_mode=0,  # preserve Hounsfield units for inspection
    anatomy_region="chest",
)

affine = np.diag([-2.0, -2.0, 2.0, 1.0])
cropped_volume = cropped[0].permute(2, 1, 0).detach().cpu().numpy()
nib.save(nib.Nifti1Image(cropped_volume, affine), "crop.nii.gz")
```

The inspection file has the correct 2 mm voxel size and LPS axis directions,
but `load_image()` does not retain the source scan's world-space origin. If the
anatomy heuristic is unsuitable for a dataset, crop the volume before model
input and call the processor with `anatomy_region=None` in custom code.

## Training data

The full training corpus is not distributed here. The two small manifests are
schema examples only; their image paths are placeholders and no patient data
are included. Replace them with NIfTI files that you are authorized to use.

Each SFT row contains the conversation and repeats the volume path at the top
level so the same record is easy to adapt for GRPO:

```json
{
  "id": "case-001",
  "image": "dataset/case-001.nii.gz",
  "anatomy_region": "chest",
  "solution": "Atelectasis, Pleural effusion",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "dataset/case-001.nii.gz"},
        {"type": "text", "text": "Write a structured chest CT report."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "<think>...</think>\n...\n<answer>Atelectasis, Pleural effusion</answer>"}
      ]
    }
  ]
}
```

For GRPO, `messages` contains the user turn only. `solution` is the verified,
comma-separated finding set used by the reward, and `anatomy_region` selects
the matching chest or abdominal ontology. Allowed labels are defined in
[`train/vlm_labels.py`](train/vlm_labels.py). No-finding cases must use exactly
`No Chest Finding` or `No Abdominal Finding`.

## SFT example

The SFT collator computes loss only on the final assistant response, sends the
NIfTI volume through `images3d=`, and preserves Qwen3.5's intended multi-turn
chat behavior. By default, all CT components are trained end to end.

```bash
accelerate launch --config_file accelerate/zero3.yaml \
  train/vlm_sft_train.py \
  --config configs/sft_config.yaml
```

All trainable components use the learning rate specified in the configuration.

Use `--freeze_llm`, `--freeze_visual`, or `--freeze_merger` for controlled
adaptation experiments. 

## GRPO example

GRPO generates a structured report and one `<answer>...</answer>` block. The
example combines three rewards:

```text
2.0 * abnormality-set F1
+ 0.5 * region-specific report-structure score
+ 1.0 * soft completion-length penalty
```

Run the single-node example with:

```bash
accelerate launch --config_file accelerate/zero3.yaml \
  train/vlm_grpo_train.py \
  --config configs/grpo_config.yaml
```

The custom trainer currently uses standard Transformers generation rather
than vLLM because the 5D CT tensor and `images3d=` route require specialized
handling.

The supplied configurations are single-node training examples. Adapt them to
your own data and hardware.

## Acknowledgements

NV-Reason-CT builds on
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B),
[Primus](https://openreview.net/forum?id=x4vZE4PDEu),
[COLIPRI](https://arxiv.org/abs/2510.15042),
[MONAI](https://monai.io/),
[Transformers](https://github.com/huggingface/transformers), and
[TRL](https://github.com/huggingface/trl). It extends the reasoning-centered
training strategy introduced in
[NV-Reason-CXR](https://github.com/NVIDIA-Medtech/NV-Reason-CXR).

## License

This repository is released under the [OpenMDW-1.1 license](LICENSE), the same
license used by the
[Hugging Face model repository](https://huggingface.co/nvidia/NV-Reason-CT).
Third-party package licenses and the TRL 1.2.0 GRPO trainer adaptation are
listed in [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES).

## Citation

This repository accompanies *NV-Reason-CT: 3D Visual Language Model for CT
Analysis*. Full citation details will be added with the public paper release.
