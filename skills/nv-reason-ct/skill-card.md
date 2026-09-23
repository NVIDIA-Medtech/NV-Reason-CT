## Description: <br>
Run NV-Reason-CT inference on user-provided 3D NIfTI chest or abdominal CT volumes for engineering and research workflows. Not for diagnosis, treatment, or clinical reporting. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
OpenMDW-1.1 <br>
## Use Case: <br>
Developers and engineers running NV-Reason-CT model inference on 3D NIfTI chest or abdominal CT volumes for engineering validation and research analysis workflows. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NV-Reason-CT Installation Guide](https://github.com/NVIDIA-Medtech/NV-Reason-CT#installation) <br>
- [EXAMPLES.md](EXAMPLES.md) <br>
- [BENCHMARK.md](BENCHMARK.md) <br>


## Skill Output: <br>
**Output Type(s):** [JSON, Shell commands] <br>
**Output Format:** [JSON (stdout result_json with input geometry, anatomy-region selection, response text, runtime identity, dependency versions, and limitations)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Partial JSON saved to --out-dir on exit code 3 (truncated generation); exit code 2 for input/dependency/inference errors] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
9 evaluation tasks (9 positive) with 3 attempts per task in isolated k8s-sandbox pods, covering static validation (Tier 1), semantic deduplication (Tier 2), and live agent evaluation (Tier 3). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer produced by the skill is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and activated when needed, and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 86.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 93.3% → 77.8% (-15.5 points) | 86.7% → 88.9% (+2.2 points) |
| Correctness | 48.0% → 95.6% (+47.6 points) | 38.7% → 77.8% (+39.1 points) |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 46.8% → 89.1% (+42.3 points) | 43.5% → 87.8% (+44.3 points) |
| Efficiency | 82.5% — baseline ran, but no comparable score was available; uplift unavailable | 85.2% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
23f8e87f (source: git SHA, committed 2026-09-23) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
