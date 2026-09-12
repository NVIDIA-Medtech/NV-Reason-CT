# Security Policy: NV-Reason-CT

NVIDIA is dedicated to the security and trust of our software products and
services, including all source code repositories managed through our
organization.

If you need to report a security issue, please use the appropriate contact
points outlined below. **Please do not report security vulnerabilities through
GitHub issues or pull requests.**

## Reporting a Vulnerability

If you discover a potential security vulnerability, please **do not open a
public issue.**

* **Web (preferred):** [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/)
* **E-Mail:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - We encourage you to use the following PGP key for secure email
    communication: [NVIDIA public PGP Key](https://www.nvidia.com/en-us/security/pgp-key)
* **GitHub:** Use this repository's **Security** tab > **Report a
  vulnerability** to submit a report directly.

Please include the following information:

- Product/project name and version or branch (this repository and, if
  applicable, the Hugging Face model revision)
- Type of vulnerability
- Step-by-step reproduction instructions
- Proof-of-concept code (if available)
- Impact assessment

**Detailed reports help NVIDIA evaluate and address issues faster.**

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge
receipt, validate severity, develop fixes, and publish security bulletins as
appropriate.

## Security Architecture & Context

NV-Reason-CT is a research vision-language model for 3D chest and abdominal CT
interpretation. This repository provides **command-line inference** and
**local supervised fine-tuning (SFT) and GRPO training examples**. Model
weights, custom model and processor code, tokenizer, and preprocessing are
published as a Hugging Face model package and are not duplicated here.

This software operates at the **CLI / training-script** level. It is not a
network service. Its primary security responsibility is to keep
operator-supplied CT volumes, prompts, training manifests, and generated
reports under the operator's control, and to load only the intended model
package when Hugging Face remote code execution is enabled.

**Repository Exposure Classification:** Public.
Basis: origin remote is GitHub and the project is an open-source research
release; this document is written for public consumption.

**Service Exposure Classification:** External / Regulated (medium confidence).
Basis: externally distributed research model materials, a medical-imaging
workload that operators may run on sensitive clinical data, and an explicit
research-only (not a medical device) constraint. This repository has no
network listener.

Key boundaries and interfaces:

- **Operator-supplied inputs:** local NIfTI volume paths, free-text prompts
  and dataset conversations, JSONL training manifests, YAML training
  configs, and the model identifier or local model directory (including
  revision when set).
- **Remote code and weights:** inference and training load the model and
  processor with Hugging Face `trust_remote_code` so the custom 3D
  architecture and NIfTI processor from the model package can run. That
  code executes in the same process as the CLI or trainer.
- **No in-repo authentication or API surface:** there are no HTTP or RPC
  endpoints and no application-level auth. Transport and access control are
  expected from the host, the operator's filesystems, and the Hugging Face
  Hub.
- **Data not shipped here:** checked-in JSONL files are schema examples with
  placeholder paths. Volume files and checkpoints are excluded from version
  control. Operators must supply imaging data they are authorized to use.
- **Outputs:** generated text goes to standard output; training writes
  checkpoints and processor configuration into the configured output
  directory.

NV-Reason-CT is for research and development only. It is not a medical device
and must not be used as a substitute for professional clinical judgment.

### Threat Model

The following scenarios represent the primary security concerns for this
project (including auxiliary/support code):

1. **Untrusted model or processor code execution:** inference and training
   load Python from the selected Hugging Face repository or local model
   directory. A substituted, malicious, or unverified package can run in
   the inference or training process.
2. **Exposure of clinical volumes and generated reports:** the tools read
   operator CT files and emit reports, logs, and checkpoints. Shared
   storage, overly broad logging, or an unintended output directory can
   disclose sensitive imaging data or derived clinical text.
3. **Unexpected local file access from training manifests:** dataset
   records include filesystem paths to NIfTI volumes. A malicious or
   mistaken manifest can cause the loader to read files the process is
   allowed to open but that were not intended as training inputs.
4. **Training-data or reward manipulation:** GRPO rewards depend on dataset
   labels, prompts, and anatomy region. Tampered manifests can bias the
   learned policy or produce checkpoints that reflect attacker-chosen
   targets.
5. **Checkpoint and processor-config integrity:** saved trainer outputs
   include model weights and preprocessor settings. Replacing those
   artifacts before a later load can change preprocessing or model
   behavior without changing this repository's scripts.
6. **Unsafe clinical or production use:** model outputs can be incorrect.
   Using this software as a diagnostic service, or feeding outputs into
   clinical systems without qualified review, is outside the research
   design and can cause patient-safety harm even when the code behaves as
   implemented.

### Critical Security Assumptions

- This repository does **not** authenticate callers, encrypt data at rest,
  or terminate TLS. Access control is assumed to come from the host OS,
  job scheduler, and Hugging Face Hub credentials.
- Hub downloads and authentication tokens are assumed to be configured and
  protected by the operator. Callers are expected to use a verified model
  source rather than an untrusted identifier or directory.
- `trust_remote_code` is assumed to load **only** the intended NVIDIA model
  package or a local copy the operator has reviewed.
- CT volumes, JSONL manifests, and prompts are assumed to be supplied by a
  trusted operator who is authorized to use that data. Example datalists in
  this repository do not contain patient images.
- NIfTI parsing and 3D cropping are performed by the **model package**
  processor, not by the scripts in this repository. That remote code is
  trusted once the package itself is trusted.
- There is no network service in this repository. The software is assumed
  to run as a local CLI or distributed training job on machines the
  operator controls.
- The host GPU stack, CUDA isolation, and Python environment are assumed to
  be administered securely. This project does not sandbox model code.
- Clinical safety, regulatory clearance, and policy for sensitive health
  data (including logging and retention of reports) are assumed to be
  handled **outside** this software. Research-only use is a condition of
  intended operation.

## Scope

**In scope:** vulnerabilities in this repository's inference CLI, training
scripts, collators, GRPO trainer adaptations, reward helpers, and example
configs that allow unexpected code execution, unauthorized access to
operator data, or integrity loss of training artifacts.

**Out of scope:**

- Model quality, hallucinated findings, or clinical accuracy (unless they
  result from a security bug such as poisoned inputs or substituted weights)
- Issues solely in third-party libraries (PyTorch, Transformers, TRL,
  Accelerate, DeepSpeed, MONAI, nibabel) except where this repo misuses them
- Hugging Face Hub, cluster, or OS vulnerabilities
- Public GitHub issues used as a disclosure channel

## Deployment Assumptions

Operators run inference or training on a GPU machine they control, install
the published Python dependencies, and point the model path at a verified
local tree or the official NVIDIA Hub repository. Example training configs
do not push checkpoints to the Hub. The checked-in Accelerate files are
single-node training templates, not a hosted service.
