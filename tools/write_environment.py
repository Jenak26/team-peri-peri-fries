"""Write artifacts/environment.json - the pinned environment the report cites.

Generated, never hand-edited: a hand-edited record drifts from the interpreter
that actually produced the findings, and the Methods page would then be false.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from peri.core.canon import hash_obj, utc_now_iso

SCHEMA = "peri.environment/1"

TRACKED_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "opencv-python-headless",
    "pillow",
    "fastapi",
    "uvicorn",
    "python-multipart",
    "reportlab",
    "c2pa-python",
    "pytest",
]

SPECIFIED_PYTHON = "3.11"


def _derive_deviations(python_version: str, torch_version: str) -> list[str]:
    """Describe how THIS machine departs from the specified build environment.

    Derived, never hard-coded. A fixed list of deviations goes stale the moment
    the interpreter or the wheel index changes, and the report's Methods page
    would then assert something about the run that is not true of it. The two
    facts below are exactly the two that differ between the examination
    workstation and the training workstation, so both must be read at runtime.
    """
    deviations: list[str] = []

    running = ".".join(python_version.split(".")[:2])
    if running != SPECIFIED_PYTHON:
        deviations.append(
            f"CLAUDE.md section 1 specifies Python {SPECIFIED_PYTHON}; this build "
            f"runs Python {python_version}, because {SPECIFIED_PYTHON} is not "
            f"installed on this workstation and every pinned dependency publishes "
            f"wheels for {running}."
        )

    if torch_version == "not-installed":
        deviations.append(
            "torch is not installed on this workstation. It performs no model "
            "inference; findings produced here are limited to the layers that "
            "require no learned model."
        )
    elif "+cu" in torch_version:
        deviations.append(
            f"torch {torch_version} is a CUDA build. This workstation is capable "
            f"of model training; checkpoint SHA-256 values for any weights it "
            f"produces are recorded in the examination manifest."
        )
    else:
        deviations.append(
            f"torch {torch_version} is a CPU build, so this workstation has no "
            f"CUDA device. Model training was performed on a separate CUDA 12.8 "
            f"workstation; checkpoint SHA-256 values are recorded in the "
            f"examination manifest."
        )

    return deviations


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _binary_version(argv: list[str]) -> str:
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "not-available"
    first = (completed.stdout or completed.stderr or "").strip().splitlines()
    return first[0] if first else "not-available"


def build_environment_record() -> dict:
    record: dict = {
        "schema": SCHEMA,
        "generated_utc": utc_now_iso(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
        "binaries": {
            "ffmpeg": _binary_version(["ffmpeg", "-version"]),
            "ffprobe": _binary_version(["ffprobe", "-version"]),
        },
        "deviations": [],
    }
    record["deviations"] = _derive_deviations(
        record["python"]["version"], record["packages"]["torch"]
    )
    # generated_utc is deliberately excluded: the hash identifies the environment,
    # not the moment the record was written.
    hashable = {k: v for k, v in record.items() if k != "generated_utc"}
    record["record_hash"] = hash_obj(hashable)
    return record


def main() -> None:
    record = build_environment_record()
    out = Path("artifacts/environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out} record_hash={record['record_hash'][:16]}")


if __name__ == "__main__":
    main()
