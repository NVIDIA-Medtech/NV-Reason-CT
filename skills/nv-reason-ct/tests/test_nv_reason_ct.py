"""Smoke tests for the NV-Reason-CT wrapper.

These tests exercise input handling, deterministic mock output, and clean
failure paths. They do not download or run model weights.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import nibabel as nib
import numpy as np
import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SKILL_DIR / "scripts" / "run_nv_reason_ct.py"
FIXTURE = SKILL_DIR / "fixtures" / "synthetic_ct_input.json"
TEST_COMMIT = "a" * 40


@pytest.fixture
def wrapper(monkeypatch):
    spec = importlib.util.spec_from_file_location("nv_reason_ct_test_wrapper", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ready_setup(wrapper, monkeypatch):
    monkeypatch.setattr(
        wrapper,
        "_package_status",
        lambda name, _: {
            "installed": True,
            "importable": True,
            "version": wrapper.EXACT_UPSTREAM_VERSIONS.get(
                name, wrapper.MINIMUM_UPSTREAM_VERSIONS.get(name, "1.0")
            ),
        },
    )
    monkeypatch.setattr(
        wrapper,
        "_cuda_report",
        lambda: {"available": True, "bfloat16_supported": True},
    )
    return wrapper


def _cache_snapshot(tmp_path, commit, refs, filenames):
    snapshot_path = tmp_path / "snapshots" / commit
    files = []
    for filename in filenames:
        path = snapshot_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        files.append(SimpleNamespace(file_path=path, size_on_disk=path.stat().st_size))
    return SimpleNamespace(
        commit_hash=commit,
        refs=set(refs),
        snapshot_path=snapshot_path,
        files=files,
        size_on_disk=sum(f.size_on_disk for f in files),
    )


def _stub_cache(monkeypatch, *snapshots):
    repo = SimpleNamespace(
        repo_id="nvidia/NV-Reason-CT", repo_type="model", revisions=snapshots
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(scan_cache_dir=lambda: SimpleNamespace(repos=[repo])),
    )


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _write_nifti(path: Path, shape: tuple[int, ...] = (8, 9, 10)) -> None:
    data = np.zeros(shape, dtype=np.int16)
    nib.save(nib.Nifti1Image(data, np.diag((-2.0, -2.0, 2.0, 1.0))), path)


def test_json_fixture_mock_generates_valid_output(tmp_path: Path) -> None:
    proc = _run(FIXTURE, "--out-dir", tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["skill"] == "nv_reason_ct"
    assert payload["runtime"]["mock"] is True
    assert payload["runtime"]["mode"] == "mock"
    assert payload["runtime"]["model"] == "nvidia/NV-Reason-CT"
    assert payload["runtime"]["resolved_revision"] is None
    assert payload["runtime"]["attention_implementation"] == "sdpa"
    assert payload["runtime"]["trust_remote_code"] is True
    assert payload["runtime"]["generated_tokens"] == 0
    assert payload["input"]["anatomy_region"] == "chest"
    assert payload["input"]["enable_thinking"] is True
    assert payload["input"]["volume"]["source"] == "generated_fixture"
    assert payload["input"]["volume"]["format"] == "nifti"
    assert payload["input"]["volume"]["ndim"] == 3
    assert len(payload["input"]["volume"]["shape"]) == 3
    assert Path(payload["input"]["volume"]["path"]).exists()
    assert payload["output"]["response_text"]


def test_check_setup_reports_json() -> None:
    proc = _run("--check-setup")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["skill"] == "nv_reason_ct"
    assert payload["setup"]["model"] == "nvidia/NV-Reason-CT"
    assert payload["setup"]["revision"] == "main"
    assert "dependencies" in payload["setup"]
    assert "recommendation" in payload["setup"]


def test_direct_nifti_mock_path(tmp_path: Path) -> None:
    volume = tmp_path / "ct.nii.gz"
    _write_nifti(volume)
    proc = _run(
        volume,
        "--mock",
        "--anatomy-region",
        "abdomen",
        "--prompt",
        "Summarize the model response for engineering review.",
        "--no-thinking",
        "--out-dir",
        tmp_path / "out",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["input"]["volume"]["source"] == "file"
    assert payload["input"]["volume"]["shape"] == [8, 9, 10]
    assert payload["input"]["anatomy_region"] == "abdomen"
    assert payload["input"]["enable_thinking"] is False
    assert payload["input"]["prompt"].startswith("Summarize")


def test_cli_overrides_fixture_region_and_thinking(tmp_path: Path) -> None:
    proc = _run(
        FIXTURE,
        "--anatomy-region",
        "none",
        "--no-thinking",
        "--out-dir",
        tmp_path / "out",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["input"]["anatomy_region"] == "none"
    assert payload["input"]["enable_thinking"] is False


def test_rejects_non_nifti_file(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_volume.txt"
    bad.write_text("not a volume")
    proc = _run(bad, "--mock", "--out-dir", tmp_path / "out")
    assert proc.returncode == 2
    assert "expected .nii or .nii.gz" in proc.stderr


def test_rejects_4d_nifti(tmp_path: Path) -> None:
    volume = tmp_path / "four_dimensional.nii.gz"
    _write_nifti(volume, shape=(4, 5, 6, 2))
    proc = _run(volume, "--mock", "--out-dir", tmp_path / "out")
    assert proc.returncode == 2
    assert "requires one 3D NIfTI volume" in proc.stderr


def test_missing_input_fails_cleanly(tmp_path: Path) -> None:
    proc = _run(tmp_path / "missing.nii.gz", "--mock", "--out-dir", tmp_path / "out")
    assert proc.returncode == 2
    assert "input not found" in proc.stderr


def test_empty_cache_reports_download_guidance(ready_setup, monkeypatch, tmp_path):
    cache_manager = pytest.importorskip("huggingface_hub.utils._cache_manager")
    monkeypatch.setattr(cache_manager, "HF_HUB_CACHE", str(tmp_path / "absent-cache"))
    setup = ready_setup._setup_report("nvidia/NV-Reason-CT", "main")["setup"]
    assert (
        setup["recommendation"] == "download_model_assets_or_disable_local_files_only"
    )
    cache = setup["model_cache"]
    assert cache["inspectable"] is False
    assert cache["has_safetensors"] is False
    assert cache["complete"] is False
    assert "config.json" in cache["missing_files"]
    assert not (tmp_path / "absent-cache").exists()


def test_uncached_revision_is_not_ready(ready_setup, monkeypatch, tmp_path):
    snapshot = _cache_snapshot(
        tmp_path,
        TEST_COMMIT,
        ["main"],
        (*ready_setup.REQUIRED_MODEL_FILES, "model.safetensors"),
    )
    _stub_cache(monkeypatch, snapshot)
    setup = ready_setup._setup_report("nvidia/NV-Reason-CT", "missing-revision")[
        "setup"
    ]
    assert setup["recommendation"] != "ready_for_live_cuda_inference"
    assert setup["model_cache"]["cached"] is False
    assert setup["model_cache"]["resolved_revision"] is None


@pytest.mark.parametrize("revision", ["main", TEST_COMMIT])
def test_complete_selected_revision_is_ready(
    ready_setup, monkeypatch, tmp_path, revision
):
    snapshot = _cache_snapshot(
        tmp_path,
        TEST_COMMIT,
        ["main"],
        (*ready_setup.REQUIRED_MODEL_FILES, "model.safetensors"),
    )
    _stub_cache(monkeypatch, snapshot)
    setup = ready_setup._setup_report("nvidia/NV-Reason-CT", revision)["setup"]
    assert setup["recommendation"] == "ready_for_live_cuda_inference"
    assert setup["model_cache"]["resolved_revision"] == TEST_COMMIT
    assert setup["model_cache"]["missing_files"] == []


@pytest.mark.parametrize(
    "missing_file",
    [
        "config.json",
        "processor.py",
        "tokenizer.json",
        "chat_template.jinja",
        "image_processor_3d/preprocessor_config.json",
    ],
)
def test_assets_from_another_revision_do_not_complete_cache(
    ready_setup, monkeypatch, tmp_path, missing_file
):
    selected = _cache_snapshot(
        tmp_path,
        TEST_COMMIT,
        ["main"],
        (set(ready_setup.REQUIRED_MODEL_FILES) | {"model.safetensors"})
        - {missing_file},
    )
    other = _cache_snapshot(tmp_path, "b" * 40, ["other"], [missing_file])
    _stub_cache(monkeypatch, selected, other)
    setup = ready_setup._setup_report("nvidia/NV-Reason-CT", "main")["setup"]
    assert setup["recommendation"] != "ready_for_live_cuda_inference"
    assert setup["model_cache"]["missing_files"] == [missing_file]
    assert setup["model_cache"]["has_safetensors"] is True


@pytest.mark.parametrize("complete", [False, True])
def test_every_indexed_weight_shard_is_required(
    ready_setup, monkeypatch, tmp_path, complete
):
    index_name = "model.safetensors.index.json"
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    selected = _cache_snapshot(
        tmp_path,
        TEST_COMMIT,
        ["main"],
        [
            *ready_setup.REQUIRED_MODEL_FILES,
            index_name,
            *(shards if complete else shards[:1]),
        ],
    )
    (selected.snapshot_path / index_name).write_text(
        json.dumps({"weight_map": {"layer1": shards[0], "layer2": shards[1]}})
    )
    _stub_cache(monkeypatch, selected)
    cache = ready_setup._model_cache_report("nvidia/NV-Reason-CT", "main")
    assert cache["complete"] is complete
    assert cache["has_safetensors"] is complete
    assert cache["missing_files"] == ([] if complete else shards[1:])


@pytest.mark.parametrize("index_text", ["not json", "[]", '{"weight_map": {}}'])
def test_invalid_weight_index_is_not_ready(
    ready_setup, monkeypatch, tmp_path, index_text
):
    index_name = "model.safetensors.index.json"
    selected = _cache_snapshot(
        tmp_path, TEST_COMMIT, ["main"], [*ready_setup.REQUIRED_MODEL_FILES, index_name]
    )
    (selected.snapshot_path / index_name).write_text(index_text)
    _stub_cache(monkeypatch, selected)
    setup = ready_setup._setup_report("nvidia/NV-Reason-CT", "main")["setup"]
    assert setup["recommendation"] != "ready_for_live_cuda_inference"
    assert "could not inspect cached weights" in setup["model_cache"]["error"]


@pytest.mark.parametrize(
    "new_tokens,eos_ids,truncated",
    [
        ([7, 8, 9], [2, 3], True),
        ([7, 8, 2], [2, 3], False),
        ([7, 8, 3], [2, 3], False),
        ([7, 8, 2], 2, False),
        ([7, 2], [2, 3], False),
        ([7, 8, 9], None, True),
    ],
)
def test_live_route_pins_both_loaders_and_detects_truncation(
    wrapper, monkeypatch, tmp_path, new_tokens, eos_ids, truncated
):
    calls = {}
    monkeypatch.setattr(wrapper, "_require_live_versions", lambda: None)
    monkeypatch.setattr(wrapper, "_select_cuda_device", lambda _: "cuda")
    monkeypatch.setenv("HF_TOKEN", "unit-test-token")

    def download(model_id, filename, **kwargs):
        calls.setdefault("downloads", []).append((model_id, filename, kwargs))
        return str(tmp_path / "snapshots" / TEST_COMMIT / filename)

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            assert device == "cuda"
            return self

    class Model:
        device = "cuda"
        generation_config = SimpleNamespace(eos_token_id=eos_ids)

        def eval(self):
            return self

        def to(self, device):
            assert device == "cuda"
            return self

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return np.array([[10, 11, *new_tokens]])

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = (messages, kwargs)
            return "formatted prompt"

        def __call__(self, **kwargs):
            calls["preprocess"] = kwargs
            return Inputs(input_ids=np.array([[10, 11]]))

        def batch_decode(self, tokens, **kwargs):
            assert tokens.tolist() == [new_tokens]
            return ["partial or complete model text"]

    def load_model(model_id, **kwargs):
        calls["model"] = (model_id, kwargs)
        return Model()

    def load_processor(model_id, **kwargs):
        calls["processor"] = (model_id, kwargs)
        return Processor()

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            bfloat16="bfloat16",
            __version__="test",
            inference_mode=nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            __version__="test",
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load_model),
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        ),
    )
    text, runtime = wrapper._run_transformers_inference(
        volume_path=tmp_path / "ct.nii",
        prompt="engineering prompt",
        anatomy_region="chest",
        enable_thinking=True,
        model_id="nvidia/NV-Reason-CT",
        revision="main",
        device_request="auto",
        max_new_tokens=3,
        local_files_only=True,
    )
    assert text
    assert calls["downloads"] == [
        (
            "nvidia/NV-Reason-CT",
            "config.json",
            {
                "revision": "main",
                "local_files_only": True,
                "token": "unit-test-token",
            },
        )
    ]
    for name in ("model", "processor"):
        model_id, kwargs = calls[name]
        assert model_id == "nvidia/NV-Reason-CT"
        assert kwargs["revision"] == kwargs["code_revision"] == TEST_COMMIT
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is True
        assert kwargs["token"] == "unit-test-token"
    assert calls["model"][1]["dtype"] == "bfloat16"
    assert calls["model"][1]["attn_implementation"] == "sdpa"
    assert calls["preprocess"]["images3d"] == [str(tmp_path / "ct.nii")]
    assert calls["preprocess"]["anatomy_region"] == "chest"
    assert calls["template"][0][0]["content"][1]["text"] == "engineering prompt"
    assert calls["template"][1]["enable_thinking"] is True
    assert calls["generate"]["do_sample"] is False
    assert calls["generate"]["max_new_tokens"] == 3
    assert runtime["resolved_revision"] == TEST_COMMIT
    assert runtime["truncated_by_max_new_tokens"] is truncated


def test_unresolvable_revision_fails_cleanly(wrapper, monkeypatch):
    def download(*args, **kwargs):
        raise FileNotFoundError("requested revision is not cached")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    with pytest.raises(wrapper.SkillError, match="could not resolve model revision"):
        wrapper._resolve_model_revision(
            "nvidia/NV-Reason-CT", "missing", local_files_only=True, token=None
        )


@pytest.mark.parametrize("truncated", [False, True])
def test_live_cli_preserves_json_and_returns_completion_status(
    wrapper, monkeypatch, tmp_path, capsys, truncated
):
    volume = tmp_path / "ct.nii"
    _write_nifti(volume)
    monkeypatch.delenv("MOCK_NV_REASON_CT", raising=False)
    monkeypatch.setattr(
        wrapper,
        "_run_transformers_inference",
        lambda **_: (
            "x",
            {
                "resolved_revision": TEST_COMMIT,
                "device": "cuda",
                "torch_dtype": "bfloat16",
                "transformers_version": "test",
                "torch_version": "test",
                "generated_tokens": 1,
                "truncated_by_max_new_tokens": truncated,
            },
        ),
    )
    code = wrapper.main(
        [str(volume), "--out-dir", str(tmp_path / "out"), "--max-new-tokens", "1"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == (3 if truncated else 0)
    assert payload["output"]["response_text"] == "x"
    assert payload["runtime"]["revision"] == "main"
    assert payload["runtime"]["resolved_revision"] == TEST_COMMIT
    assert payload["runtime"]["truncated_by_max_new_tokens"] is truncated
    assert ("partial output" in captured.err) is truncated
    if truncated:
        partial_path = Path(payload["output"]["partial_json_path"])
        assert partial_path.parent == tmp_path / "out"
        assert json.loads(partial_path.read_text()) == payload
        assert str(partial_path) in captured.err
    else:
        assert "partial_json_path" not in payload["output"]
        assert not list((tmp_path / "out").glob("partial_result_*.json"))
    schema = json.loads((SKILL_DIR / "validators/output_schema.json").read_text())
    jsonschema.validate(payload, schema)
    checks = yaml.safe_load((SKILL_DIR / "skill_manifest.yaml").read_text())[
        "validation"
    ]["sanity_checks"]
    completion_gate = next(
        c for c in checks if c["path"] == "runtime.truncated_by_max_new_tokens"
    )
    assert completion_gate["eq"] is False
    assert (
        payload["runtime"]["truncated_by_max_new_tokens"] == completion_gate["eq"]
    ) is not truncated
    for invalid_revision in (None, "main"):
        payload["runtime"]["resolved_revision"] = invalid_revision
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, schema)


def test_partial_results_do_not_overwrite_previous_attempts(wrapper, tmp_path):
    payload = {"output": {"response_text": "first attempt"}}
    wrapper._preserve_partial_result(payload, tmp_path)
    first_path = Path(payload["output"]["partial_json_path"])
    first_bytes = first_path.read_bytes()
    payload["output"]["response_text"] = "second attempt"
    wrapper._preserve_partial_result(payload, tmp_path)
    assert Path(payload["output"]["partial_json_path"]) != first_path
    assert first_path.read_bytes() == first_bytes


def test_partial_save_failure_still_allows_stdout_payload(
    wrapper, monkeypatch, tmp_path, capsys
):
    def fail_save(**kwargs):
        raise PermissionError("test output directory is not writable")

    monkeypatch.setattr(wrapper.tempfile, "NamedTemporaryFile", fail_save)
    payload = {"output": {"response_text": "partial text"}}
    wrapper._preserve_partial_result(payload, tmp_path)
    assert payload == {"output": {"response_text": "partial text"}}
    assert "retain stdout" in capsys.readouterr().err


@pytest.mark.parametrize("download_fails", [False, True])
def test_documented_install_downloads_before_pip_without_exposing_token(
    tmp_path, download_fails
):
    command = yaml.safe_load((SKILL_DIR / "skill_manifest.yaml").read_text())[
        "runtime"
    ]["external_assets"][0]["install_command"]
    assert command.strip() in (SKILL_DIR / "SKILL.md").read_text()
    # Execute the actual documented shell flow with fake installers: no network,
    # package installation, or changes to the caller's environment are permitted.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("python", "hf"):
        executable = fake_bin / name
        executable.write_text(
            f"#!{sys.executable}\n"
            + """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
name = Path(sys.argv[0]).name
with Path(os.environ["INSTALL_TEST_LOG"]).open("a") as stream:
    stream.write(json.dumps([name, args]) + "\\n")
if name == "hf":
    assert os.environ["HF_TOKEN"] == "unit-test-token"
    if os.environ["INSTALL_TEST_FAIL"] == "1":
        sys.exit(17)
    Path(args[args.index("--local-dir") + 1], "requirements.txt").write_text("# test\\n")
elif "-r" in args:
    assert Path(args[args.index("-r") + 1]).is_file()
"""
        )
        executable.chmod(0o700)
    log_path = tmp_path / "commands.jsonl"
    proc = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "TMPDIR": str(tmp_path),
            "HF_TOKEN": "unit-test-token",
            "NV_REASON_CT_REVISION": "reviewed-revision",
            "INSTALL_TEST_LOG": str(log_path),
            "INSTALL_TEST_FAIL": "1" if download_fails else "0",
        },
    )
    assert proc.returncode == (17 if download_fails else 0), proc.stderr
    calls = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(calls) == (2 if download_fails else 3)
    assert calls[0] == ["python", ["-m", "pip", "install", "huggingface_hub>=1.5,<2"]]
    assert calls[1][0] == "hf"
    assert calls[1][1][:5] == [
        "download",
        "nvidia/NV-Reason-CT",
        "requirements.txt",
        "--revision",
        "reviewed-revision",
    ]
    assert "unit-test-token" not in log_path.read_text() + proc.stdout + proc.stderr
    if not download_fails:
        assert calls[2] == [
            "python",
            [
                "-m",
                "pip",
                "install",
                "-r",
                str(Path(calls[1][1][-1]) / "requirements.txt"),
                "torch>=2.9.0",
            ],
        ]


@pytest.mark.parametrize(
    "package,version",
    [
        ("torch", "2.8.0+cu128"),
        ("monai", "1.5.0"),
        ("nibabel", "5.3.2"),
        ("torch", "not-a-version"),
    ],
)
def test_setup_reports_upstream_minimum_version_mismatch(
    ready_setup, monkeypatch, package, version
):
    original = ready_setup._package_status

    def package_status(name, import_name):
        status = original(name, import_name)
        if name == package:
            status["version"] = version
        return status

    monkeypatch.setattr(ready_setup, "_package_status", package_status)
    _stub_cache(monkeypatch)
    setup = ready_setup._setup_report(ready_setup.DEFAULT_MODEL, "main")["setup"]
    assert setup["recommendation"] == "use_the_upstream_minimum_dependency_versions"
    assert setup["minimum_version_mismatches"][package]["installed"] == version


@pytest.mark.parametrize("torch_version", ["2.9.0", "2.9.0+cu128", "2.12.0+cu130"])
def test_live_version_check_accepts_supported_torch_builds(
    wrapper, monkeypatch, torch_version
):
    versions = {
        **wrapper.EXACT_UPSTREAM_VERSIONS,
        **wrapper.MINIMUM_UPSTREAM_VERSIONS,
        "torch": torch_version,
    }
    monkeypatch.setattr(wrapper, "_installed_version", versions.get)
    wrapper._require_live_versions()


@pytest.mark.parametrize("package", ["torch", "monai", "nibabel"])
def test_live_version_check_rejects_missing_or_old_minimum(
    wrapper, monkeypatch, package
):
    versions = {**wrapper.EXACT_UPSTREAM_VERSIONS, **wrapper.MINIMUM_UPSTREAM_VERSIONS}
    versions[package] = "1.0.0"
    monkeypatch.setattr(wrapper, "_installed_version", versions.get)
    with pytest.raises(wrapper.SkillError, match=package):
        wrapper._require_live_versions()


@pytest.mark.parametrize(
    "region,prompt",
    [
        ("chest", "write a structured chest CT report"),
        ("abdomen", "write a structured abdominal CT report"),
    ],
)
def test_default_prompt_matches_upstream_cli(wrapper, tmp_path, region, prompt):
    volume = tmp_path / "ct.nii"
    _write_nifti(volume)
    spec = wrapper._load_input(volume, tmp_path, None, region, None)
    assert spec.prompt == prompt
    assert spec.enable_thinking is True
