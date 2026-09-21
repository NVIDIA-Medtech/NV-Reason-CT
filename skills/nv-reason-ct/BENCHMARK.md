# Skill Benchmark: nv-reason-ct

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nv-reason-ct`
- Evaluation date: 2026-09-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:6d62762b34d1e9d57a0e4ca861b612029444be807f43210c7968f1d6fdc7d2c8` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.9% — baseline ran, but no comparable score was available; uplift unavailable | 79.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 66.7% → 57.1% (-9.6 points) |
| Correctness | 41.5% → 91.4% (+49.9 points) | 46.7% → 91.4% (+44.7 points) |
| Discoverability | 80.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 41.3% → 84.6% (+43.3 points) | 37.1% → 83.6% (+46.5 points) |
| Efficiency | 83.2% — baseline ran, but no comparable score was available; uplift unavailable | 77.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,258,216 | 5,236,479 | N/A | N/A | skill 7/7; base 13/13 |
| claude-code | diagnosis-treatment-refusal | 30,542 | 123,457 | -92,915 | -75.26% | skill 1/1; base 1/1 |
| claude-code | git-lfs-pointer-is-not-volume | 105,339 | 157,215 | -51,876 | -33.00% | skill 1/1; base 1/1 |
| claude-code | live-abdominal-nifti-routing | 596,915 | 493,501 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | mock-synthetic-ct-fixture | 697,545 | 2,488,859 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | no-silent-history-loss | 102,201 | 359,209 | -257,008 | -71.55% | skill 1/1; base 1/1 |
| claude-code | training-records-are-not-demo-volumes | 150,224 | 665,134 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | upstream-example-end-to-end | 575,450 | 949,104 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 2,051,416 | 4,253,758 | N/A | N/A | skill 7/7; base 12/12 |
| codex | diagnosis-treatment-refusal | 34,344 | 17,914 | +16,430 | +91.72% | skill 1/1; base 1/1 |
| codex | git-lfs-pointer-is-not-volume | 130,531 | 1,807,431 | -1,676,900 | -92.78% | skill 1/1; base 1/1 |
| codex | live-abdominal-nifti-routing | 131,391 | 80,317 | +51,074 | +63.59% | skill 1/1; base 1/1 |
| codex | mock-synthetic-ct-fixture | 240,151 | 519,078 | N/A | N/A | skill 1/1; base 3/3 |
| codex | no-silent-history-loss | 176,699 | 41,479 | +135,220 | +326.00% | skill 1/1; base 1/1 |
| codex | training-records-are-not-demo-volumes | 1,006,020 | 981,717 | N/A | N/A | skill 1/1; base 2/2 |
| codex | upstream-example-end-to-end | 332,280 | 805,822 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 4,309,632 | 9,490,237 | N/A | N/A | skill 14/14; base 25/25 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 17 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/Unknown (SQP-2): The use of trust_remote_code=True in both AutoModelForImageTextToText.from_pretrained() and AutoProcessor.from_pretraine (`scripts/run_nv_reason_ct.py:686`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:     return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        capture_output=True,
        text=True,
        timeout=30,
    ) (`tests/test_nv_reason_ct.py:104`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:     proc = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=10,
        env={
            "PATH": str(fake_bin) + os.pathsep + os.enviro (`tests/test_nv_reason_ct.py:718`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:     result = subprocess.run(
        command,
        cwd=output_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    ) (`tests/test_upstream_examples.py:60`)
- **MEDIUM** SECURITY/Tainted flow: 'env' from os.environ.get (line 54, credential/environment) → subprocess.run (code execution) (TT2): Data Flow:     result = subprocess.run(
        command,
        cwd=output_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    ) (`tests/test_upstream_examples.py:60`)
- 12 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
