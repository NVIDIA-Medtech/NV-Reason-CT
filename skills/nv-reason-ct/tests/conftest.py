"""Opt-in upstream example and live checks; never download or install assets."""

from pathlib import Path
import re

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("nv-reason-ct")
    group.addoption(
        "--nv-reason-ct-upstream",
        help="Authorized NV-Reason-CT checkout with examples downloaded via Git LFS.",
    )
    group.addoption(
        "--nv-reason-ct-live",
        action="store_true",
        help="Run offline CUDA inference and compare with the upstream CLI.",
    )
    group.addoption(
        "--nv-reason-ct-revision",
        help="Reviewed, already cached immutable Hugging Face commit for live tests.",
    )


def pytest_generate_tests(metafunc):
    if "ct_input" in metafunc.fixturenames:
        cases = (
            ["example_1", "example_2", "example_3"]
            if metafunc.config.getoption("--nv-reason-ct-upstream")
            else ["synthetic"]
        )
        metafunc.parametrize("ct_input", cases, indirect=True)


@pytest.fixture(scope="session")
def upstream_checkout(pytestconfig):
    value = pytestconfig.getoption("--nv-reason-ct-upstream")
    if not value:
        pytest.fail("Provide --nv-reason-ct-upstream with an authorized checkout")
    checkout = Path(value).expanduser().resolve()
    if not (checkout / "inference.py").is_file():
        pytest.fail(f"Missing upstream inference.py in {checkout}")
    return checkout


@pytest.fixture
def nifti_factory(request):
    """Accept a test-infrastructure provider without importing repository tools."""
    try:
        return request.getfixturevalue("synthetic_nifti_factory")
    except pytest.FixtureLookupError as exc:
        if exc.argname != "synthetic_nifti_factory":
            raise
        pytest.skip(
            "Synthetic fixture provider not loaded; run the repository test "
            "target or supply the optional pytest provider. Upstream example "
            "tests do not require it."
        )


@pytest.fixture
def ct_input(request, tmp_path):
    """Use real examples for success paths when the caller supplies the data."""
    if request.param == "synthetic":
        return request.getfixturevalue("nifti_factory")("ct.nii.gz")
    checkout = request.getfixturevalue("upstream_checkout")
    volume = checkout / "examples" / f"{request.param}.nii.gz"
    if not volume.is_file():
        pytest.fail(f"Missing example volume: {volume}; run git lfs pull")
    return volume


@pytest.fixture(scope="session")
def live_revision(pytestconfig):
    if not pytestconfig.getoption("--nv-reason-ct-live"):
        pytest.skip("Live model inference requires --nv-reason-ct-live")
    revision = pytestconfig.getoption("--nv-reason-ct-revision") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        pytest.fail("Live tests require a reviewed immutable --nv-reason-ct-revision")
    return revision
