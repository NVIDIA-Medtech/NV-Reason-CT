#!/usr/bin/env python3
"""Run NV-Reason-CT inference on one NIfTI CT volume.

The live path follows the upstream Hugging Face Transformers model card. The
mock path exists for CI and command-contract checks; it never calls the model
and must not be treated as model or clinical output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "nvidia/NV-Reason-CT"
DEFAULT_REVISION = "main"
DEFAULT_ANATOMY_REGION = "chest"
DEFAULT_MAX_NEW_TOKENS = 2048
ANATOMY_REGIONS = ("chest", "abdomen", "none")
GENERATED_VOLUME_SENTINELS = {
    "generated://synthetic_ct_volume",
    "generated://synthetic_ct",
}
TRUTHY = {"1", "true", "yes", "on"}

# This compact volume is generated only under a caller-selected output
# directory. It is deterministic wiring data, not a clinical or demo case.
SYNTHETIC_SHAPE = (64, 64, 96)
SYNTHETIC_SPACING_MM = 2.0
SYNTHETIC_BACKGROUND_HU = -1000
SYNTHETIC_SOFT_TISSUE_HU = 40
SYNTHETIC_LUNG_HU = -800
SYNTHETIC_BONE_HU = 300
HASH_BLOCK_BYTES = 1024 * 1024

EXACT_UPSTREAM_VERSIONS = {
    "transformers": "5.6.2",
    "dynamic-network-architectures": "0.4.3",
}
DEPENDENCY_IMPORTS = {
    "torch": "torch",
    "transformers": "transformers",
    "monai": "monai",
    "nibabel": "nibabel",
    "dynamic-network-architectures": "dynamic_network_architectures",
    "scipy": "scipy",
    "numpy": "numpy",
    "Pillow": "PIL",
    "timm": "timm",
    "einops": "einops",
    "safetensors": "safetensors",
    "huggingface-hub": "huggingface_hub",
}
ENVIRONMENT_PACKAGES = tuple(DEPENDENCY_IMPORTS)
# Assets used by the reviewed upstream composite processor and generation path.
# Check these in one snapshot; weights alone do not make offline inference ready.
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.py",
    "processor.py",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "image_processor_3d/preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
)
HUB_COMMIT_PATTERN = r"[0-9a-f]{40}"

LIMITATIONS = [
    "Output is engineering evidence only; it is not a diagnosis or treatment recommendation.",
    "NV-Reason-CT output can hallucinate, omit findings, or present unreliable reasoning.",
    "The wrapper does not verify CT de-identification, Hounsfield-unit calibration, crop quality, or clinical correctness.",
    "A qualified professional must review any use involving medical interpretation.",
]


class SkillError(Exception):
    """Input, dependency, or runtime error that should be shown cleanly."""


@dataclass(frozen=True)
class VolumeInfo:
    """Inspectable NIfTI metadata used by the wrapper contract."""

    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    dtype: str
    sha256: str


@dataclass(frozen=True)
class InputSpec:
    """Resolved input and inference settings."""

    volume_path: Path
    source: str
    prompt: str
    case_id: str
    anatomy_region: str
    enable_thinking: bool
    fixture_mock: bool
    fixture_mock_response: str | None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUTHY


def _installed_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_status(dist_name: str, import_name: str) -> dict[str, Any]:
    version = _installed_version(dist_name)
    if version is None:
        return {"installed": False, "importable": False, "version": None}
    try:
        __import__(import_name)
        importable = True
        error = None
    except Exception as exc:  # setup probes must report broken binary imports
        importable = False
        error = str(exc)
    result: dict[str, Any] = {
        "installed": True,
        "importable": importable,
        "version": version,
    }
    if error:
        result["error"] = error
    return result


def _cuda_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "available": False,
            "bfloat16_supported": False,
            "devices": [],
            "error": f"torch import failed: {exc}",
        }

    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_total_mb": round(properties.total_memory / 1024 / 1024),
                }
            )
    try:
        bfloat16_supported = bool(available and torch.cuda.is_bf16_supported())
    except Exception:
        bfloat16_supported = False
    return {
        "available": available,
        "bfloat16_supported": bfloat16_supported,
        "device_count": len(devices),
        "devices": devices,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }


def _cached_weight_requirements(files: dict[str, Path]) -> set[str]:
    """Return the monolithic weight file or every shard named by its index."""
    if "model.safetensors" in files:
        return {"model.safetensors"}
    index_name = "model.safetensors.index.json"
    if index_name not in files:
        return {"model.safetensors"}
    index = json.loads(files[index_name].read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index must contain a nonempty weight_map")
    if not all(isinstance(name, str) and name for name in weight_map.values()):
        raise ValueError("safetensors index contains an invalid shard filename")
    return {index_name, *weight_map.values()}


def _model_cache_report(model_id: str, revision: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "inspectable": False,
        "cached": False,
        "repo_id": model_id,
        "revision": revision,
        "resolved_revision": None,
        "revisions": 0,
        "size_on_disk_mb": 0,
        "has_config": False,
        "has_custom_model_code": False,
        "has_custom_processor_code": False,
        "has_safetensors": False,
        "complete": False,
        "missing_files": sorted((*REQUIRED_MODEL_FILES, "model.safetensors")),
    }
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
    except Exception as exc:
        report["error"] = str(exc)
        return report

    report["inspectable"] = True
    for repo in cache.repos:
        if repo.repo_id != model_id or repo.repo_type != "model":
            continue
        report["revisions"] = len(repo.revisions)
        snapshot = next(
            (
                candidate
                for candidate in repo.revisions
                if revision == candidate.commit_hash or revision in candidate.refs
            ),
            None,
        )
        if snapshot is None:
            return report
        files = {
            str(
                cached_file.file_path.relative_to(snapshot.snapshot_path)
            ): cached_file.file_path
            for cached_file in snapshot.files
            if cached_file.size_on_disk > 0
        }
        report.update(
            cached=True,
            resolved_revision=snapshot.commit_hash,
            size_on_disk_mb=round(snapshot.size_on_disk / 1024 / 1024),
            has_config="config.json" in files,
            has_custom_model_code="model.py" in files,
            has_custom_processor_code="processor.py" in files,
            missing_files=sorted(set(REQUIRED_MODEL_FILES) - files.keys()),
        )
        try:
            weight_files = _cached_weight_requirements(files)
        except (OSError, ValueError) as exc:
            report["error"] = f"could not inspect cached weights: {exc}"
            return report
        report["has_safetensors"] = weight_files.issubset(files)
        report["missing_files"] = sorted(
            (set(REQUIRED_MODEL_FILES) | weight_files) - files.keys()
        )
        report["complete"] = not report["missing_files"]
        return report
    return report


def _setup_report(model_id: str, revision: str) -> dict[str, Any]:
    dependencies = {
        name: _package_status(name, import_name)
        for name, import_name in DEPENDENCY_IMPORTS.items()
    }
    missing_or_broken = [
        name
        for name, status in dependencies.items()
        if not status["installed"] or not status["importable"]
    ]
    exact_version_mismatches = {
        name: {
            "installed": dependencies[name]["version"],
            "required": required,
        }
        for name, required in EXACT_UPSTREAM_VERSIONS.items()
        if dependencies[name]["version"] != required
    }
    cuda = _cuda_report()
    cache = _model_cache_report(model_id, revision)

    if missing_or_broken:
        recommendation = "install_or_repair_upstream_dependencies"
    elif exact_version_mismatches:
        recommendation = "use_the_upstream_exact_dependency_versions"
    elif not cuda["available"]:
        recommendation = "use_a_cuda_host_or_mock_mode"
    elif not cuda["bfloat16_supported"]:
        recommendation = "use_a_cuda_gpu_with_bfloat16_support"
    elif not cache["complete"]:
        recommendation = "download_model_assets_or_disable_local_files_only"
    else:
        recommendation = "ready_for_live_cuda_inference"

    return {
        "skill": "nv_reason_ct",
        "setup": {
            "python": sys.executable,
            "model": model_id,
            "revision": revision,
            "recommended_torch_dtype": "bfloat16",
            "recommended_attention_implementation": "sdpa",
            "dependencies": dependencies,
            "exact_version_mismatches": exact_version_mismatches,
            "cuda": cuda,
            "model_cache": cache,
            "recommendation": recommendation,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_nifti_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")


def _nifti_modules() -> tuple[Any, Any]:
    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:
        raise SkillError(
            "NIfTI input handling requires nibabel and numpy. Install the "
            f"SKILL.md prerequisites. Import error: {exc}"
        ) from exc
    return nib, np


def _volume_info(path: Path) -> VolumeInfo:
    if not path.exists():
        raise SkillError(f"volume not found: {path}")
    if not path.is_file():
        raise SkillError(f"volume path is not a file: {path}")
    if not _is_nifti_path(path):
        raise SkillError(
            f"unsupported volume format for {path}: expected .nii or .nii.gz"
        )

    nib, np = _nifti_modules()
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise SkillError(f"could not read NIfTI volume {path}: {exc}") from exc

    shape = tuple(int(value) for value in image.shape)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise SkillError(
            f"NV-Reason-CT requires one 3D NIfTI volume; got shape {shape}"
        )
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if len(spacing) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in spacing
    ):
        raise SkillError(
            f"NIfTI voxel spacing must contain three positive values; got {spacing}"
        )
    if not bool(np.isfinite(np.asarray(image.affine)).all()):
        raise SkillError("NIfTI affine contains non-finite values")

    return VolumeInfo(
        shape=(shape[0], shape[1], shape[2]),
        spacing_mm=(spacing[0], spacing[1], spacing[2]),
        dtype=str(image.get_data_dtype()),
        sha256=_sha256(path),
    )


def _write_synthetic_nifti(path: Path) -> None:
    """Write deterministic mock-only CT-like wiring data."""
    nib, np = _nifti_modules()
    data = np.full(SYNTHETIC_SHAPE, SYNTHETIC_BACKGROUND_HU, dtype=np.int16)

    # Rectangular regions are intentionally simple and deterministic. They
    # exercise NIfTI I/O and metadata gates without claiming anatomical realism.
    data[6:-6, 6:-6, 8:-8] = SYNTHETIC_SOFT_TISSUE_HU
    data[14:30, 14:30, 20:82] = SYNTHETIC_LUNG_HU
    data[34:50, 14:30, 20:82] = SYNTHETIC_LUNG_HU
    data[30:34, 30:34, 12:88] = SYNTHETIC_BONE_HU

    spacing = SYNTHETIC_SPACING_MM
    affine = np.diag((-spacing, -spacing, spacing, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _normalize_region(value: Any) -> str:
    region = str(value or DEFAULT_ANATOMY_REGION).strip().lower()
    if region not in ANATOMY_REGIONS:
        raise SkillError("anatomy_region must be 'chest', 'abdomen', or 'none'")
    return region


def _default_prompt(region: str) -> str:
    if region == "chest":
        return "Write a structured chest CT report."
    if region == "abdomen":
        return "Write a structured abdominal CT report."
    return "Describe this CT volume."


def _fixture_bool(fixture: dict[str, Any], key: str, default: bool) -> bool:
    value = fixture.get(key, default)
    if not isinstance(value, bool):
        raise SkillError(f"fixture field {key} must be a boolean")
    return value


def _load_json_fixture(
    path: Path,
    out_dir: Path,
    cli_prompt: str | None,
    cli_region: str | None,
    cli_enable_thinking: bool | None,
) -> InputSpec:
    try:
        fixture = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SkillError(f"fixture is not valid JSON: {path}: {exc}") from exc
    if not isinstance(fixture, dict):
        raise SkillError(f"fixture must be a JSON object: {path}")

    volume_value = str(
        fixture.get("volume_path") or fixture.get("ct_path") or ""
    ).strip()
    region = _normalize_region(
        cli_region if cli_region is not None else fixture.get("anatomy_region")
    )
    prompt = cli_prompt or str(fixture.get("prompt") or _default_prompt(region)).strip()
    if not prompt:
        raise SkillError("prompt must not be empty")
    enable_thinking = (
        cli_enable_thinking
        if cli_enable_thinking is not None
        else _fixture_bool(fixture, "enable_thinking", True)
    )
    case_id = str(fixture.get("case_id") or path.stem).strip()
    if not case_id:
        raise SkillError("case_id must not be empty")
    fixture_mock = _fixture_bool(fixture, "mock", False)
    fixture_mock_response = fixture.get("mock_response")
    if fixture_mock_response is not None and not isinstance(fixture_mock_response, str):
        raise SkillError("fixture field mock_response must be a string when present")

    if volume_value in GENERATED_VOLUME_SENTINELS:
        volume_path = out_dir / "input_synthetic_ct.nii"
        _write_synthetic_nifti(volume_path)
        source = "generated_fixture"
    elif volume_value:
        volume_path = Path(volume_value)
        if not volume_path.is_absolute():
            volume_path = (path.parent / volume_path).resolve()
        source = "fixture_file"
    else:
        raise SkillError(
            "fixture must include volume_path or use generated://synthetic_ct_volume"
        )

    return InputSpec(
        volume_path=volume_path,
        source=source,
        prompt=prompt,
        case_id=case_id,
        anatomy_region=region,
        enable_thinking=enable_thinking,
        fixture_mock=fixture_mock,
        fixture_mock_response=fixture_mock_response,
    )


def _load_input(
    path: Path,
    out_dir: Path,
    cli_prompt: str | None,
    cli_region: str | None,
    cli_enable_thinking: bool | None,
) -> InputSpec:
    if not path.exists():
        raise SkillError(f"input not found: {path}")
    if path.suffix.lower() == ".json":
        return _load_json_fixture(
            path,
            out_dir,
            cli_prompt,
            cli_region,
            cli_enable_thinking,
        )

    region = _normalize_region(cli_region)
    prompt = str(cli_prompt or _default_prompt(region)).strip()
    if not prompt:
        raise SkillError("prompt must not be empty")
    enable_thinking = True if cli_enable_thinking is None else cli_enable_thinking
    return InputSpec(
        volume_path=path.resolve(),
        source="file",
        prompt=prompt,
        case_id=path.name.removesuffix(".gz").removesuffix(".nii"),
        anatomy_region=region,
        enable_thinking=enable_thinking,
        fixture_mock=False,
        fixture_mock_response=None,
    )


def _mock_response(spec: InputSpec, info: VolumeInfo) -> str:
    if spec.fixture_mock_response:
        return spec.fixture_mock_response
    return (
        "Mock NV-Reason-CT response for volume "
        f"{spec.case_id!r} with shape {list(info.shape)}. "
        "This deterministic response verifies NIfTI loading, prompt and region "
        "wiring, and JSON output only; it asserts no clinical finding."
    )


def _require_exact_live_versions() -> None:
    mismatches = {
        name: {"installed": _installed_version(name), "required": required}
        for name, required in EXACT_UPSTREAM_VERSIONS.items()
        if _installed_version(name) != required
    }
    if mismatches:
        details = ", ".join(
            f"{name}={values['installed'] or 'missing'} (required {values['required']})"
            for name, values in sorted(mismatches.items())
        )
        raise SkillError(
            "live inference requires the model-card exact dependency versions: "
            f"{details}. Create a fresh environment and run --check-setup."
        )


def _select_cuda_device(requested: str) -> str:
    try:
        import torch
    except Exception as exc:
        raise SkillError(f"live inference requires torch: {exc}") from exc
    if not torch.cuda.is_available():
        raise SkillError(
            "CUDA is unavailable for live NV-Reason-CT inference. Use a CUDA "
            "host or --mock for a command-contract check."
        )
    try:
        if not torch.cuda.is_bf16_supported():
            raise SkillError("the visible CUDA device does not report bfloat16 support")
    except AttributeError as exc:
        raise SkillError("installed torch cannot verify CUDA bfloat16 support") from exc
    return "cuda" if requested == "auto" else requested


def _resolve_model_revision(
    model_id: str, revision: str, *, local_files_only: bool, token: str | None
) -> str:
    """Resolve a Hub ref once, without executing any model-provided code."""
    try:
        from huggingface_hub import hf_hub_download

        config_path = Path(
            hf_hub_download(
                model_id,
                "config.json",
                revision=revision,
                local_files_only=local_files_only,
                token=token,
            )
        )
    except Exception as exc:
        raise SkillError(
            f"could not resolve model revision {revision!r}: {exc}"
        ) from exc
    # Keep the snapshot path, not Path.resolve(): the file is a symlink to a blob.
    commit = config_path.parent.name
    if config_path.parent.parent.name != "snapshots" or not re.fullmatch(
        HUB_COMMIT_PATTERN, commit
    ):
        raise SkillError(
            "could not identify the immutable Hub snapshot for config.json"
        )
    return commit


def _run_transformers_inference(
    *,
    volume_path: Path,
    prompt: str,
    anatomy_region: str,
    enable_thinking: bool,
    model_id: str,
    revision: str,
    device_request: str,
    max_new_tokens: int,
    local_files_only: bool,
) -> tuple[str, dict[str, Any]]:
    _require_exact_live_versions()
    try:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise SkillError(
            "live inference requires the packages in the upstream requirements. "
            f"Run --check-setup first. Import error: {exc}"
        ) from exc

    device = _select_cuda_device(device_request)
    token = os.environ.get("HF_TOKEN") or None
    processor_region = None if anatomy_region == "none" else anatomy_region
    resolved_revision = _resolve_model_revision(
        model_id, revision, local_files_only=local_files_only, token=token
    )

    try:
        model = (
            AutoModelForImageTextToText.from_pretrained(
                model_id,
                revision=resolved_revision,
                code_revision=resolved_revision,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=local_files_only,
                token=token,
            )
            .eval()
            .to(device)
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=resolved_revision,
            code_revision=resolved_revision,
            trust_remote_code=True,
            local_files_only=local_files_only,
            token=token,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = processor(
            text=formatted_prompt,
            images3d=[str(volume_path)],
            anatomy_region=processor_region,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
        generated_tokens = int(new_tokens.shape[-1])
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        ended_with_eos = bool(
            generated_tokens and int(new_tokens[0, -1].item()) in eos_ids
        )
        response_text = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    except Exception as exc:
        raise SkillError(f"NV-Reason-CT inference failed: {exc}") from exc

    if not response_text.strip():
        raise SkillError("NV-Reason-CT returned empty text")
    return response_text, {
        "resolved_revision": resolved_revision,
        "device": str(getattr(model, "device", device)),
        "torch_dtype": "bfloat16",
        "transformers_version": getattr(transformers, "__version__", None),
        "torch_version": getattr(torch, "__version__", None),
        "generated_tokens": generated_tokens,
        "truncated_by_max_new_tokens": (
            generated_tokens >= max_new_tokens and not ended_with_eos
        ),
    }


def _environment_packages() -> dict[str, str]:
    return {
        name: version
        for name in ENVIRONMENT_PACKAGES
        if (version := _installed_version(name)) is not None
    }


def _preserve_partial_result(payload: dict[str, Any], out_dir: Path) -> None:
    """Retain partial JSON even when a caller discards stdout on nonzero exit."""
    try:
        # A unique name preserves results from earlier attempts in the same out-dir.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="partial_result_",
            suffix=".json",
            dir=out_dir,
            delete=False,
        ) as stream:
            payload["output"]["partial_json_path"] = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except OSError as exc:
        payload["output"].pop("partial_json_path", None)
        print(
            f"warning: could not save partial JSON; retain stdout: {exc}",
            file=sys.stderr,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NV-Reason-CT inference on one 3D NIfTI CT volume."
    )
    parser.add_argument(
        "volume_or_fixture",
        nargs="?",
        type=Path,
        help=".nii/.nii.gz volume path, or JSON fixture for mock smoke tests.",
    )
    parser.add_argument("--prompt", default=None, help="Text prompt for the model.")
    parser.add_argument(
        "--anatomy-region",
        choices=ANATOMY_REGIONS,
        default=None,
        help="Chest/abdomen crop, or none for a manually cropped volume.",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        dest="enable_thinking",
        action="store_true",
        help="Request reviewable reasoning text (default).",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking output for concise responses.",
    )
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/nv_reason_ct"))
    parser.add_argument(
        "--model-id",
        default=os.environ.get("NV_REASON_CT_MODEL", DEFAULT_MODEL),
        help=f"Hugging Face model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("NV_REASON_CT_REVISION", DEFAULT_REVISION),
        help="Hugging Face model revision (default: main).",
    )
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Skip model inference.")
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Print dependency/GPU/cache readiness JSON and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_new_tokens < 1:
        print("error: --max-new-tokens must be >= 1", file=sys.stderr)
        return 2
    if not str(args.model_id).strip():
        print("error: --model-id must not be empty", file=sys.stderr)
        return 2
    if not str(args.revision).strip():
        print("error: --revision must not be empty", file=sys.stderr)
        return 2

    try:
        if args.check_setup:
            print(
                json.dumps(
                    _setup_report(args.model_id, args.revision),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.volume_or_fixture is None:
            print(
                "error: volume_or_fixture is required unless --check-setup is used",
                file=sys.stderr,
            )
            return 2

        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = _load_input(
            args.volume_or_fixture.resolve(),
            out_dir,
            args.prompt,
            args.anatomy_region,
            args.enable_thinking,
        )
        info = _volume_info(spec.volume_path)
        local_files_only = bool(
            args.local_files_only
            or _truthy(os.environ.get("TRANSFORMERS_OFFLINE"))
            or _truthy(os.environ.get("HF_HUB_OFFLINE"))
        )
        mock = bool(
            args.mock
            or spec.fixture_mock
            or _truthy(os.environ.get("MOCK_NV_REASON_CT"))
        )

        started = time.perf_counter()
        if mock:
            response_text = _mock_response(spec, info)
            runtime_extra = {
                "resolved_revision": None,
                "device": "none",
                "torch_dtype": "none",
                "transformers_version": _installed_version("transformers"),
                "torch_version": _installed_version("torch"),
                "generated_tokens": 0,
                "truncated_by_max_new_tokens": False,
            }
            mode = "mock"
        else:
            response_text, runtime_extra = _run_transformers_inference(
                volume_path=spec.volume_path,
                prompt=spec.prompt,
                anatomy_region=spec.anatomy_region,
                enable_thinking=spec.enable_thinking,
                model_id=args.model_id,
                revision=args.revision,
                device_request=args.device,
                max_new_tokens=args.max_new_tokens,
                local_files_only=local_files_only,
            )
            mode = "hf_transformers"
        elapsed = time.perf_counter() - started

        payload = {
            "skill": "nv_reason_ct",
            "input": {
                "case_id": spec.case_id,
                "prompt": spec.prompt,
                "anatomy_region": spec.anatomy_region,
                "enable_thinking": spec.enable_thinking,
                "volume": {
                    "path": str(spec.volume_path),
                    "source": spec.source,
                    "format": "nifti",
                    "ndim": 3,
                    "shape": list(info.shape),
                    "spacing_mm": [round(value, 6) for value in info.spacing_mm],
                    "dtype": info.dtype,
                    "sha256": info.sha256,
                },
            },
            "output": {
                "response_text": response_text,
                "text_chars": len(response_text),
            },
            "runtime": {
                "model": args.model_id,
                "revision": args.revision,
                "resolved_revision": runtime_extra["resolved_revision"],
                "mode": mode,
                "mock": mock,
                "device": str(runtime_extra["device"]),
                "torch_dtype": str(runtime_extra["torch_dtype"]),
                "attention_implementation": "sdpa",
                "trust_remote_code": True,
                "deterministic_generation": True,
                "max_new_tokens": args.max_new_tokens,
                "inference_seconds": round(elapsed, 6),
                "local_files_only": local_files_only,
                "transformers_version": runtime_extra["transformers_version"],
                "torch_version": runtime_extra["torch_version"],
                "generated_tokens": runtime_extra["generated_tokens"],
                "truncated_by_max_new_tokens": runtime_extra[
                    "truncated_by_max_new_tokens"
                ],
            },
            "environment": {"packages": _environment_packages()},
            "limitations": LIMITATIONS,
        }
        if runtime_extra["truncated_by_max_new_tokens"]:
            _preserve_partial_result(payload, out_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if runtime_extra["truncated_by_max_new_tokens"]:
            print(
                "error: generation reached --max-new-tokens without EOS; partial "
                "output is preserved in stdout JSON. Review it before retrying "
                "with a larger token limit.",
                file=sys.stderr,
            )
            if "partial_json_path" in payload["output"]:
                print(
                    f"partial JSON: {payload['output']['partial_json_path']}",
                    file=sys.stderr,
                )
            return 3
        return 0
    except SkillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
