# Upstream CT examples

Use the volumes in [NV-Reason-CT/examples](https://github.com/NVIDIA-Medtech/NV-Reason-CT/tree/main/examples) for end-to-end inference. The upstream README uses `example_1.nii.gz` for both chest and abdomen inference. This skill does not assume the anatomy coverage of the other two volumes or provide reference diagnoses.

Run these commands from the skills repository root using the isolated Python environment described in `SKILL.md`. Git/Git LFS, authorized GitHub/model access, and a compatible CUDA GPU must already be available. No scan, weight, or generated medical response belongs in a commit.

## Get the actual example volumes

Use a separate disposable checkout; do not reset an existing development checkout or run a global `git lfs install`:

```bash
nv_reason_ct_demo="$(mktemp -d)"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA-Medtech/NV-Reason-CT.git \
  "$nv_reason_ct_demo/upstream"
git -C "$nv_reason_ct_demo/upstream" lfs install --local
git -C "$nv_reason_ct_demo/upstream" lfs pull --include="examples/*.nii.gz"
git -C "$nv_reason_ct_demo/upstream" lfs fsck
git -C "$nv_reason_ct_demo/upstream" rev-parse HEAD
git -C "$nv_reason_ct_demo/upstream" lfs ls-files --long
```

Keep the Git commit and LFS content hashes with local evidence. The default checkout follows upstream `main`; retain an immutable reviewed commit when replaying an earlier run. GitHub archive downloads or clones with LFS smudging disabled may contain only pointer files. Missing authorization, missing volumes, or pointers block inference until resolved.

## Run chest and abdomen end to end

Keep downloaded model assets and custom-code caches in the disposable directory, not the user's shared cache:

```bash
export HF_HOME="$nv_reason_ct_demo/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$nv_reason_ct_demo/cache/modules"
export XDG_CACHE_HOME="$nv_reason_ct_demo/cache/xdg"
export TORCH_HOME="$nv_reason_ct_demo/cache/torch"
export NV_REASON_CT_REVISION=REVIEWED_IMMUTABLE_MODEL_COMMIT
```

Replace the revision placeholder with the reviewed 40-character Hugging Face commit. With any required `HF_TOKEN` already supplied securely, stage the assets once and inspect the read-only setup report:

```bash
hf download nvidia/NV-Reason-CT --revision "$NV_REASON_CT_REVISION"
python skills/nv-reason-ct/scripts/run_nv_reason_ct.py \
  --check-setup --fail-on-not-ready --revision "$NV_REASON_CT_REVISION"
```

Proceed only when setup exits `0` and reports `ready_for_live_cuda_inference`; exit `1` means the reported setup blocker must be resolved first. Enable offline mode only after the download completes. Unset `MOCK_NV_REASON_CT` for these live examples, and retain stdout JSON and stderr through the calling tool or harness:

```bash
env -u MOCK_NV_REASON_CT python skills/nv-reason-ct/scripts/run_nv_reason_ct.py \
  "$nv_reason_ct_demo/upstream/examples/example_1.nii.gz" \
  --anatomy-region chest --prompt "write a structured chest CT report" \
  --thinking --max-new-tokens 2048 --local-files-only \
  --revision "$NV_REASON_CT_REVISION" --out-dir runs/nv_reason_ct_example_1_chest

env -u MOCK_NV_REASON_CT python skills/nv-reason-ct/scripts/run_nv_reason_ct.py \
  "$nv_reason_ct_demo/upstream/examples/example_1.nii.gz" \
  --anatomy-region abdomen --prompt "write a structured abdominal CT report" \
  --thinking --max-new-tokens 2048 --local-files-only \
  --revision "$NV_REASON_CT_REVISION" --out-dir runs/nv_reason_ct_example_1_abdomen
```

Require exit `0`, `runtime.mode == "hf_transformers"`, `runtime.mock == false`, the selected `runtime.resolved_revision`, nonempty text, and `runtime.truncated_by_max_new_tokens == false`. Verify the recorded volume hash and crop selection. Exit `3` is incomplete generation, not a successful end-to-end run. These are engineering checks, not checks of the generated medical content.

## Cleanup

After inference and when cleanup is authorized, remove only the task-created checkout, environment, and caches. Preserve result JSON, source/content hashes, and setup versions separately; do not purge shared caches or delete a caller-owned input scan.
