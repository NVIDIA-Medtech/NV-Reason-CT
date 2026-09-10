---
name: nv-reason-ct
description: Used for mock or live NV-Reason-CT inference on 3D NIfTI chest or abdominal CT volumes. Not for diagnosis, treatment, or clinical reporting.
license: Apache-2.0
allowed-tools: Bash
metadata:
  author: "NVIDIA MedTech <noreply@nvidia.com>"
  tags:
    - medtech
    - ct
    - reasoning
---

# NV-Reason-CT

## Purpose

- Runs the documented `nvidia/NV-Reason-CT` Hugging Face Transformers path on one user-provided 3D NIfTI CT volume and text prompt.
- Emits `result_json` with input geometry and hash, anatomy-region selection, response text, runtime identity, dependency versions, and limitations.
- Manifest I/O: inputs are `ct_request_or_fixture` and optional direct `ct_volume`; output is `result_json` on stdout.
- Supports deterministic mock wiring checks. Not for diagnosis, treatment, triage, patient-facing output, or clinical reporting.

## Instructions

1. Read `skill_manifest.yaml` before changing arguments, dependencies, side effects, or validation gates.
2. Run `scripts/run_nv_reason_ct.py`; do not write replacement inference code or use the Gradio demo for normal runs.
3. For hosts exposing a helper, use `run_script("scripts/run_nv_reason_ct.py", args=["PATH_TO_CT.nii.gz", "--anatomy-region", "chest", "--prompt", "PROMPT", "--out-dir", "OUT_DIR"])`.
4. Inspect stdout JSON and the exit status. Preserve JSON for the caller or eval harness, including partial output on failure; do not redirect it away with `>`. Exit code `3` means generation hit the token ceiling without EOS: the text is incomplete and the completion gate fails. Partial JSON is also saved as a unique `partial_result_*.json` under `--out-dir` (path in `output.partial_json_path` and stderr), so callers that discard stdout on failure can retain it.

Direct mock smoke test from the repository root:

```bash
python skills/nv-reason-ct/scripts/run_nv_reason_ct.py \
  skills/nv-reason-ct/fixtures/synthetic_ct_input.json \
  --mock \
  --out-dir runs/nv_reason_ct_smoke
```

Live inference on a chest CT:

```bash
python skills/nv-reason-ct/scripts/run_nv_reason_ct.py PATH_TO_CT.nii.gz \
  --anatomy-region chest \
  --prompt "Write a structured chest CT report." \
  --out-dir runs/nv_reason_ct_case
```

For an abdominal crop, pass `--anatomy-region abdomen`. Pass `--anatomy-region none` only for a manually cropped input; upstream preprocessing then uses a centered crop. Pass `--no-thinking` for a concise response without thinking output.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/run_nv_reason_ct.py` | Primary NIfTI preflight, mock, setup-check, and live inference entrypoint. | `CT_OR_FIXTURE [--anatomy-region chest|abdomen|none] [--prompt TEXT] [--thinking|--no-thinking] [--mock] [--out-dir OUT_DIR]` |

## Prerequisites

- Python 3.11+, a CUDA-capable NVIDIA GPU with bfloat16 support for live inference, and enough memory for the model plus volumetric visual tokens.
- Activate a fresh environment before installing the model-card requirements. For a private or gated repository, supply an authorized `HF_TOKEN` through the environment first; do not put its value in commands or logs. Download with the authenticated Hugging Face client because pip does not use `HF_TOKEN` for a requirements URL:

```bash
python -m pip install "huggingface_hub>=1.5,<2" &&
nv_reason_ct_prereqs="$(mktemp -d)" &&
hf download nvidia/NV-Reason-CT requirements.txt \
  --revision "${NV_REASON_CT_REVISION:-main}" --local-dir "$nv_reason_ct_prereqs" &&
python -m pip install -r "$nv_reason_ct_prereqs/requirements.txt"
```

- The upstream requirements pin `transformers==5.6.2`, `dynamic-network-architectures==0.4.3`, `monai>=1.5.1`, and `nibabel>=5.3.3`; they do not pin PyTorch or every numerical dependency.
- Model assets may download from `https://huggingface.co` under `~/.cache/huggingface/`. Set `HF_TOKEN` when repository access requires authentication.
- Run a read-only setup check before downloading weights or starting inference. It checks required files (including all indexed weight shards) in the requested revision only and reports `missing_files`; it does not download assets or prove their integrity:

```bash
python skills/nv-reason-ct/scripts/run_nv_reason_ct.py --check-setup
```

### Verified environment baseline

The live CUDA path was exercised successfully end to end on 2026-09-09 with the versions below. Treat this as a known-good verification baseline, not an upstream compatibility guarantee or a complete cross-platform lockfile. In particular, select a PyTorch CUDA build compatible with the host driver; the verified Python distribution version was `torch==2.12.0`, and `torch.__version__` reported `2.12.0+cu130`.

| Component | Verified version or setting |
|---|---|
| Python | `3.12.3` |
| PyTorch | `2.12.0` (`2.12.0+cu130` runtime build) |
| CUDA reported by PyTorch | `13.0` |
| Transformers | `5.6.2` |
| dynamic-network-architectures | `0.4.3` |
| MONAI | `1.6.0` |
| NiBabel | `5.4.2` |
| SciPy | `1.16.0` |
| NumPy | `2.5.2` |
| Pillow | `11.2.1` |
| timm | `1.0.22` |
| einops | `0.8.2` |
| safetensors | `0.7.0` |
| huggingface-hub | `1.30.0` |

The verified inference settings were one 48 GB NVIDIA RTX 6000 Ada GPU, bfloat16 model weights, SDPA attention, deterministic decoding (`do_sample=False`), chest anatomy cropping, thinking enabled, and `max_new_tokens=4096`. The run completed normally without reaching the token ceiling. Lower-memory GPUs and other PyTorch/CUDA combinations have not yet been verified by this skill.

Environment variables:

| Variable | When to use |
|---|---|
| `MOCK_NV_REASON_CT` | Set to `1` for deterministic mock checks without model inference. |
| `NV_REASON_CT_MODEL` | Override the Hugging Face model id for an explicit compatibility probe. |
| `NV_REASON_CT_REVISION` | Select a reviewed model revision; prefer an immutable revision for evidence runs. |
| `HF_HOME` | Point to a caller-managed Hugging Face cache. |
| `HF_TOKEN` | Authenticate model access when required. |
| `TRANSFORMERS_OFFLINE` | Set to `1` only after all model and custom-code files are cached. |
| `HF_HUB_OFFLINE` | Set to `1` only after all Hugging Face assets are cached. |
| `CUDA_VISIBLE_DEVICES` | Restrict which GPU the wrapper may use. |

Live mode uses the upstream contract: `AutoModelForImageTextToText` and `AutoProcessor` with `trust_remote_code=True`, bfloat16, SDPA attention, `images3d=[CT_PATH]`, deterministic generation, and anatomy-aware chest or abdomen cropping. Each run resolves the requested revision once and uses that immutable commit for model weights, processor assets, and custom code. Output retains the requested `runtime.revision` and records the commit in `runtime.resolved_revision` (`null` in mock mode). Keep private revision identifiers in local evidence until approved for disclosure.

## Limitations

- The wrapper loads upstream custom code. Use an isolated environment and an immutable reviewed revision for evidence runs.
- It checks readable 3D NIfTI structure, positive voxel spacing, a finite affine, and file identity. It does not verify de-identification, Hounsfield-unit calibration, anatomy coverage, crop quality, or clinical correctness.
- Model output can hallucinate, omit findings, or present unreliable reasoning. Reviewable reasoning text is generated output, not proof of the model's internal computation.
- The committed fixture generates compact synthetic NIfTI wiring data and a mock response under `--out-dir`. It is not private data, a model-quality demo, or a substitute for live inference.
- The moving `main` revision is suitable only for development. Replace it with the reviewed immutable public release revision before publication.
- This wrapper does not cover the Gradio UI, finetuning, retraining, clinical deployment, or patient-facing use.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| Dependency mismatch in `--check-setup` | The environment differs from the model-card requirements. | Create a fresh environment and install the upstream requirements exactly. |
| Authentication or repository-not-found error | Model assets are not public or the current account lacks access. | Set an authorized `HF_TOKEN`; do not copy tokens into commands, fixtures, or logs. |
| Cache is incomplete in `--check-setup` | The selected revision or some of its required assets are absent. | Inspect `missing_files`, then download the selected revision with authorized access before enabling offline mode. Files cached for another revision do not satisfy this check. |
| CUDA unavailable or bfloat16 unsupported | Live inference is running on an unsupported device or isolated GPU context. | Use a compatible CUDA host; use `--mock` only for wiring checks. |
| NIfTI shape or spacing error | Input is unreadable, 4D, or has invalid geometry metadata. | Supply one 3D `.nii` or `.nii.gz` CT volume with valid spacing and affine metadata. |
| Poor chest or abdomen crop | Automatic anatomy heuristics do not fit the scan geometry. | Inspect upstream preprocessing, manually crop the CT, and pass `--anatomy-region none`. |
| Empty or truncated response | Generation stopped without text, or reached the token limit without EOS (exit `3`). | Preserve any partial JSON and stderr; do not treat it as a completed response. Verify setup and input, then adjust `--max-new-tokens` if truncation is reported. |

## License

The wrapper is Apache-2.0. The upstream model repository and weights use OpenMDW-1.1; users are responsible for complying with those terms.
