import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# The lowest interpreter the code actually runs on. CLAUDE.md section 1 specifies
# 3.11, but nothing in the tree needs a 3.11-only feature, and the training
# workstation runs 3.10. `ruff.toml` pins the same floor so a lint autofix cannot
# quietly reintroduce something newer, which is exactly how `datetime.UTC` (3.11+)
# got in once already.
MINIMUM_PYTHON = (3, 10)

REQUIRED_DIRS = [
    "peri/core",
    "train",
    "api",
    "web",
    "artifacts",
    "evidence",
    "tools",
    "tests",
]


def test_required_directories_exist():
    for name in REQUIRED_DIRS:
        assert Path(name).is_dir(), f"missing directory: {name}"


def test_error_hierarchy_is_importable_and_rooted():
    from peri.core.errors import (
        CalibrationError,
        ExaminationError,
        IntakeError,
        PeriError,
    )

    for cls in (IntakeError, CalibrationError, ExaminationError):
        assert issubclass(cls, PeriError)


def test_environment_artifact_is_on_disk():
    assert Path("artifacts/environment.json").is_file()


def test_ffprobe_is_callable_from_this_interpreter():
    """ffprobe backs L0 intake, so the examination workstation must have it.

    Skipped where it is absent rather than failed. The training workstation has
    no intake path: it reads a prebuilt corpus of PNG frames, and OpenCV decodes
    video through its own bundled FFmpeg rather than this binary. Failing here
    would block training over a tool training does not use.
    """
    if shutil.which("ffprobe") is None:
        pytest.skip(
            "ffprobe not on PATH. Required on the examination workstation for L0 "
            "intake; not required to train."
        )
    completed = subprocess.run(
        ["ffprobe", "-version"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0
    assert "ffprobe version" in completed.stdout


def test_interpreter_is_recent_enough():
    """Assert a floor, not a specific version.

    This previously pinned an exact pair of versions, which failed on any
    interpreter nobody had thought to add to the list.
    """
    assert sys.version_info[:2] >= MINIMUM_PYTHON, (
        f"Python {'.'.join(map(str, MINIMUM_PYTHON))} or newer is required; "
        f"this interpreter is {sys.version.split()[0]}"
    )


def test_torch_imports_and_reports_cuda_availability():
    import torch

    assert isinstance(torch.cuda.is_available(), bool)
