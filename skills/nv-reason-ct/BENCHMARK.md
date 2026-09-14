# nv-reason-ct Benchmark

## Scope

This report covers the `nv-reason-ct` engineering wrapper for 3D NIfTI CT inference through `nvidia/NV-Reason-CT`. It does not measure diagnostic performance, clinical utility, or model ranking.

## Current Evidence

- `SKILL.md` defines the agent-facing trigger, exact wrapper command, NIfTI and anatomy-region contract, prerequisites, and limitations.
- `skill_manifest.yaml` declares inputs, stdout JSON, runtime side effects, dependency gates, model identity, and deterministic mock repeatability.
- `tests/test_nv_reason_ct.py` has 46 passing tests covering generated-fixture and direct-NIfTI mock paths, input failures, revision-specific cache checks, dependency minima and exact pins, prompt defaults, and generation completion without downloading weights.
- `evals/evals.json` covers correct 3D routing, mock/live separation, exclusion of UI and finetuning paths, stdout preservation, single-turn/history boundaries, rejection of training records as demo volumes, and clinical-scope refusal. These are authored cases, not measured agent-study results.

## Live Engineering Verification (2026-09-14)

An authorized private chest CT demo was run through both the refreshed upstream `inference.py` and the skill's full evidence-pack harness, using the same immutable model revision, exact default prompt, and offline model assets. The responses matched after trimming surrounding whitespace. Setup, output schema, sanity, completion, runtime, cost, environment-pin, and evidence-integrity checks all passed.

| Measurement | Observed result |
|---|---|
| Generated tokens / limit | `587 / 2048`; EOS reached, not truncated |
| Wrapper inference time | `21.97 s` |
| Upstream CLI wall time | `23.92 s` |
| Skill plus harness wall time | `23.79 s` |
| Sampled GPU memory above idle baseline | `14,558 MiB` |
| Host peak RSS | `11,131 MiB` |
| Mock reproducibility | Two matching runs |

`SKILL.md` records the tested package and GPU settings. Private input data, immutable revisions, logs, and the evidence pack remain local and are not committed. The run used a disposable environment; it did not upgrade the user's development environment. Timing and memory are single-run observations, not hardware requirements or model-quality scores. The upstream reference was launched with Python safe-path mode (`-P`) from outside its checkout because its `accelerate/` configuration directory otherwise shadowed the absent optional Python package; no upstream code or extra package was needed for this workaround.

## With-Skill vs Without-Skill Evaluation

| Arm | Document surface | Expected behavior | Status |
|---|---|---|---|
| With skill | `SKILL.md`, wrapper, manifest, and evals | Agent should use the committed `images3d` wrapper, preserve the requested chest/abdomen crop and prompt, retain stdout JSON, and state engineering-only boundaries. | Authored; measured agent study pending. |
| Without skill | User request plus model card or general Transformers knowledge | Agent may use the 2D `images` route, omit required custom-code or crop settings, call the UI, or miss scope and evidence fields. | Baseline run pending. |

## Remaining Evidence Gaps

- The committed fixture is synthetic mock-only wiring data; it does not exercise model preprocessing or inference.
- The successful private live run is not yet publicly reproducible: model access and demo data require authorization, and neither weights nor CT volumes are committed. Independent validation requires a separately obtained authorized CT evaluation volume, a compatible CUDA environment, and an immutable reviewed model revision.
- This run verified one chest scan with thinking enabled. Abdominal, manually cropped, multi-turn, lower-memory, and other runtime combinations were not exercised by this live check.
- Future measured results should record agent/runtime version, task completion, invocation correctness, token and wall-clock cost, GPU memory, and evidence-pack links. They must not grade clinical correctness without a separately approved qualified-review protocol.
