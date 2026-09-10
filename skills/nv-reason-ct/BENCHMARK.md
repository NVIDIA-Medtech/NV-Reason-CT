# nv-reason-ct Benchmark

## Scope

This report covers the `nv-reason-ct` engineering wrapper for 3D NIfTI CT inference through `nvidia/NV-Reason-CT`. It does not measure diagnostic performance, clinical utility, or model ranking.

## Current Evidence

- `SKILL.md` defines the agent-facing trigger, exact wrapper command, NIfTI and anatomy-region contract, prerequisites, and limitations.
- `skill_manifest.yaml` declares inputs, stdout JSON, runtime side effects, dependency gates, model identity, and deterministic mock repeatability.
- `tests/test_nv_reason_ct.py` exercises generated-fixture and direct-NIfTI mock paths plus input failures without downloading weights.
- `evals/evals.json` covers correct 3D routing, mock/live separation, exclusion of UI and finetuning paths, stdout preservation, and clinical-scope refusal.

## With-Skill vs Without-Skill Evaluation

| Arm | Document surface | Expected behavior | Status |
|---|---|---|---|
| With skill | `SKILL.md`, wrapper, manifest, and evals | Agent should use the committed `images3d` wrapper, preserve the requested chest/abdomen crop and prompt, retain stdout JSON, and state engineering-only boundaries. | Authored; measured agent study pending. |
| Without skill | User request plus model card or general Transformers knowledge | Agent may use the 2D `images` route, omit required custom-code or crop settings, call the UI, or miss scope and evidence fields. | Baseline run pending. |

## Remaining Evidence Gaps

- The committed fixture is synthetic mock-only wiring data; it does not exercise model preprocessing or inference.
- Live end-to-end validation requires authorized model access, a separately obtained de-identified CT evaluation volume, a compatible CUDA environment, and an immutable reviewed model revision.
- Future measured results should record agent/runtime version, task completion, invocation correctness, token and wall-clock cost, GPU memory, and evidence-pack links. They must not grade clinical correctness without a separately approved qualified-review protocol.
