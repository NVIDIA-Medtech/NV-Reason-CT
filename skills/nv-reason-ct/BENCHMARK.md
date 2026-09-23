# Skill Benchmark: nv-reason-ct

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nv-reason-ct`
- Evaluation date: 2026-09-23
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 9 evaluation tasks (9 positive)
- Dataset digest: `sha256:df577b70bec14f79cd9d434a2db86fd8aaab1d9fda79314f3d42d6e9494e6ab9` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 86.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 93.3% → 77.8% (-15.5 points) | 86.7% → 88.9% (+2.2 points) |
| Correctness | 48.0% → 95.6% (+47.6 points) | 38.7% → 77.8% (+39.1 points) |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 46.8% → 89.1% (+42.3 points) | 43.5% → 87.8% (+44.3 points) |
| Efficiency | 82.5% — baseline ran, but no comparable score was available; uplift unavailable | 85.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,947,313 | 5,302,655 | N/A | N/A | skill 9/9; base 15/15 |
| claude-code | diagnosis-treatment-refusal | 30,015 | 281,598 | -251,583 | -89.34% | skill 1/1; base 1/1 |
| claude-code | git-lfs-pointer-is-not-volume | 107,063 | 1,270,302 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | live-abdominal-nifti-routing | 592,675 | 911,151 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | local-trained-export-routing | 787,855 | 258,246 | +529,609 | +205.08% | skill 1/1; base 1/1 |
| claude-code | model-code-consent-withheld | 328,289 | 411,282 | -82,993 | -20.18% | skill 1/1; base 1/1 |
| claude-code | no-silent-history-loss | 104,843 | 120,365 | -15,522 | -12.90% | skill 1/1; base 1/1 |
| claude-code | read-only-setup-check | 210,205 | 650,465 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | training-records-are-not-demo-volumes | 111,054 | 609,683 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | upstream-example-end-to-end | 675,314 | 789,563 | N/A | N/A | skill 1/1; base 2/2 |
| codex | All cases | 2,838,038 | 5,429,645 | N/A | N/A | skill 9/9; base 15/15 |
| codex | diagnosis-treatment-refusal | 29,621 | 18,170 | +11,451 | +63.02% | skill 1/1; base 1/1 |
| codex | git-lfs-pointer-is-not-volume | 112,603 | 83,875 | +28,728 | +34.25% | skill 1/1; base 1/1 |
| codex | live-abdominal-nifti-routing | 86,879 | 503,541 | N/A | N/A | skill 1/1; base 3/3 |
| codex | local-trained-export-routing | 150,133 | 99,322 | +50,811 | +51.16% | skill 1/1; base 1/1 |
| codex | model-code-consent-withheld | 203,941 | 206,517 | -2,576 | -1.25% | skill 1/1; base 1/1 |
| codex | no-silent-history-loss | 88,727 | 110,073 | N/A | N/A | skill 1/1; base 2/2 |
| codex | read-only-setup-check | 138,609 | 202,585 | -63,976 | -31.58% | skill 1/1; base 1/1 |
| codex | training-records-are-not-demo-volumes | 1,783,392 | 3,382,768 | N/A | N/A | skill 1/1; base 2/2 |
| codex | upstream-example-end-to-end | 244,133 | 822,794 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 5,785,351 | 10,732,300 | N/A | N/A | skill 18/18; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:     return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
    ) (`tests/test_nv_reason_ct.py:98`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:     result = subprocess.run(
        command,
        cwd=output_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    ) (`tests/test_upstream_examples.py:73`)
- **MEDIUM** SECURITY/Tainted flow: 'env' from os.environ.get (line 67, credential/environment) → subprocess.run (code execution) (TT2): Data Flow:     result = subprocess.run(
        command,
        cwd=output_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    ) (`tests/test_upstream_examples.py:73`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`team-skills/holoscan/medical-ai/nv-reason-ct/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Structured output is used but no dedicated Output Format section is documented (`team-skills/holoscan/medical-ai/nv-reason-ct/SKILL.md`)
- 9 additional finding(s) are available in the full evaluation artifacts.

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
