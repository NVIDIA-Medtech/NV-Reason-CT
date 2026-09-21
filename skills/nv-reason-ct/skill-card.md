## Description: <br>
Run NV-Reason-CT inference on user-provided 3D NIfTI chest or abdominal CT volumes for engineering and research workflows. Not for diagnosis, treatment, or clinical reporting. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
OpenMDW-1.1 <br>
## Use Case: <br>
Developers and engineers running NV-Reason-CT model inference on 3D NIfTI chest or abdominal CT volumes for engineering validation, research analysis, and model evaluation workflows. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NV-Reason-CT upstream inference repository](https://github.com/NVIDIA-Medtech/NV-Reason-CT) <br>
- [EXAMPLES.md](EXAMPLES.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, JSON data] <br>
**Output Format:** [JSON] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Structured result_json on stdout with input geometry, anatomy-region selection, response text, runtime identity, dependency versions, and limitations] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), 3 attempts per task, each in an isolated k8s-sandbox pod. Dataset digest: sha256:adcb6a27. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.5% | 83.8% |
| Security | 94.4% → 100.0% (+5.6 pts) | 50.0% → 78.6% (+28.6 pts) |
| Correctness | 53.3% → 97.1% (+43.8 pts) | 46.2% → 91.4% (+45.2 pts) |
| Discoverability | 81.4% | 88.6% |
| Effectiveness | 60.9% → 90.0% (+29.1 pts) | 47.6% → 82.9% (+35.3 pts) |
| Efficiency | 83.9% | 77.7% |

## Skill Version(s): <br>
d94c50b4 (source: git SHA, committed 2026-09-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
