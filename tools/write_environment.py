"""Write artifacts/environment.json — the pinned environment the report cites.

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

DEVIATIONS = [
    "CLAUDE.md section 1 specifies Python 3.11; this build runs Python 3.12 because "
    "3.11 is not installed on the examination workstation and every pinned "
    "dependency publishes 3.12 wheels.",
    "torch is installed from the CPU wheel index on the examination workstation, "
    "which has no CUDA device. Model training was performed on a separate CUDA 12.8 "
    "workstation; checkpoint SHA-256 values are recorded in the examination manifest.",
]


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
        "deviations": list(DEVIATIONS),
    }
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
