## Description: <br>
Used for mock or live NV-Reason-CT inference on 3D NIfTI chest or abdominal CT volumes. Not for diagnosis, treatment, or clinical reporting. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
OpenMDW-1.1 <br>
## Use Case: <br>
Developers and engineers use this skill to run NV-Reason-CT model inference on 3D NIfTI chest or abdominal CT volumes for engineering evaluation, mock wiring checks, and structured report generation. <br>

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
- [NV-Reason-CT upstream repository](https://github.com/NVIDIA-Medtech/NV-Reason-CT) <br>
- [End-to-end examples guide](EXAMPLES.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, JSON] <br>
**Output Format:** [JSON] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Structured result including input geometry, anatomy-region selection, response text, runtime identity, and dependency versions] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks with 3 attempts per task, each in an isolated sandbox pod. Dataset digest: sha256:6d62762b34d1e9d57a0e4ca861b612029444be807f43210c7968f1d6fdc7d2c8. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code | Codex |
|---|---:|---:|
| Overall | 87.9% | 79.0% |
| Security | 100.0% | 57.1% |
| Correctness | 91.4% | 91.4% |
| Discoverability | 80.0% | 85.7% |
| Effectiveness | 84.6% | 83.6% |
| Efficiency | 83.2% | 77.1% |

## Skill Version(s): <br>
0f8cd79c (source: git SHA, committed 2026-09-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
