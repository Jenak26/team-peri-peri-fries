import subprocess
import sys
from pathlib import Path

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
    completed = subprocess.run(
        ["ffprobe", "-version"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0
    assert "ffprobe version" in completed.stdout

def test_torch_imports_without_a_cuda_device():
    import torch
    assert isinstance(torch.cuda.is_available(), bool)
    assert sys.version_info[:2] == (3, 12) or sys.version_info[:2] == (3, 13)
