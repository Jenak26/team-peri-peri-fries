# Phase 0 - Environment & Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the repo, the virtualenv, the pinned dependency record, and the
determinism helpers every later phase imports - so no later task has to stop and
invent a JSON convention.

**Architecture:** A Python 3.12 venv with CPU torch. A `peri` package with the file
tree from CLAUDE.md section 2. One tiny shared module, `peri/core/canon.py`, holding the
canonical-JSON and float-quantisation functions that make the replay hash reproducible.
`artifacts/environment.json` is written by a script, not by hand, so it cannot drift.

**Tech Stack:** Python 3.12, pytest, torch (CPU wheels), FastAPI, ReportLab, OpenCV,
NumPy, SciPy, scikit-learn, c2pa-python, ffmpeg 9.0 (already installed).

**Spec:** `CLAUDE.md` (repo root), section 1 and section 2. Roadmap:
`docs/superpowers/plans/2026-08-20-ppf-00-ROADMAP.md`.

## Global Constraints

- Python **3.12** (`py -3.12`), a stated deviation from CLAUDE.md's 3.11. Reason
  recorded in `artifacts/environment.json` under `deviations`.
- torch installs from the **CPU** index on this machine. Never import
  `torch.cuda.*` unguarded in CPU-track code.
- All hashed JSON uses `json.dumps(obj, sort_keys=True, separators=(",", ":"),
  ensure_ascii=False)`.
- All floats that reach a hashed structure pass through `q(x) = float(f"{x:.6f}")`.
- Global seed constant: `PERI_SEED = 20260820`.
- Forbidden strings (CLAUDE.md section 8) must not appear in any file authored in this phase.
- Every module gets a one-line docstring naming its CLAUDE.md layer (L0…L8).

---

### Task 1: Repository skeleton and virtualenv

**Files:**
- Create: `.gitignore`
- Create: `peri/__init__.py`
- Create: `peri/core/__init__.py`
- Create: `train/__init__.py`
- Create: `api/__init__.py`
- Create: `tools/__init__.py`
- Create: `tests/__init__.py`
- Create: `pytest.ini`
- Create: `requirements-cpu.txt`
- Create: `requirements-gpu.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: the importable package root `peri`, and a `pytest` invocation that
  discovers tests under `tests/`.

- [ ] **Step 1: Initialise git and create the directory tree**

```bash
cd "/c/Users/Janak/OneDrive/Desktop/Innohack"
git init
mkdir -p peri/core train api web/vendor artifacts evidence tools tests data/authentic data/corpus docs/superpowers/plans
touch peri/__init__.py peri/core/__init__.py train/__init__.py api/__init__.py tools/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
*.pt
*.onnx
evidence/
data/authentic/*
data/corpus/*
!data/authentic/.gitkeep
!data/corpus/.gitkeep
web/vendor/*.js
.pytest_cache/
```

Then:

```bash
touch data/authentic/.gitkeep data/corpus/.gitkeep
```

- [ ] **Step 3: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q --strict-markers
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 4: Write `requirements-cpu.txt`**

This is the dev-box file. torch comes from the CPU index (see Step 6).

```
numpy==2.1.3
scipy==1.14.1
scikit-learn==1.5.2
pandas==2.2.3
opencv-python-headless==4.10.0.84
pillow==11.0.0
fastapi==0.115.5
uvicorn==0.32.1
python-multipart==0.0.19
reportlab==4.2.5
c2pa-python==0.10.0
pytest==8.3.4
```

- [ ] **Step 5: Write `requirements-gpu.txt`**

This is the file that ships to the training laptop. Same core, plus the model libs
that only training needs.

```
numpy==2.1.3
scipy==1.14.1
scikit-learn==1.5.2
opencv-python-headless==4.10.0.84
pillow==11.0.0
timm==1.0.11
transformers==4.46.3
open_clip_torch==2.29.0
pytest==8.3.4
```

With a header comment inside the file:

```
# torch/torchvision are NOT pinned here. On the training box install them first:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# then:  pip install -r requirements-gpu.txt
```

- [ ] **Step 6: Create the venv and install**

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements-cpu.txt
```

- [ ] **Step 7: Verify the install**

Run:

```bash
.venv/Scripts/python.exe -c "import torch, numpy, scipy, sklearn, cv2, fastapi, reportlab; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Expected: prints a torch version and `cuda False`. `cuda False` is correct on this
machine and is not an error.

If `c2pa-python` fails to build, do not fight it - remove it from
`requirements-cpu.txt`, and Phase 3 will report `c2pa: unavailable` through its
already-planned fallback path. Note the removal in `artifacts/environment.json`
under `deviations`.

- [ ] **Step 8: Commit**

```bash
git add .gitignore pytest.ini requirements-cpu.txt requirements-gpu.txt peri train api tools tests data
git commit -m "chore: repo skeleton, venv requirements, pytest config"
```

---

### Task 2: Determinism helpers (`peri/core/canon.py`)

**Files:**
- Create: `peri/core/canon.py`
- Test: `tests/test_canon.py`

**Interfaces:**
- Consumes: nothing.
- Produces, relied on by every later phase:
  - `PERI_SEED: int = 20260820`
  - `q(x: float) -> float` - quantise a float to 6 decimal places
  - `qdeep(obj: Any) -> Any` - recursively apply `q` to every float in a nested
    structure of dict / list / tuple / float / int / str / bool / None
  - `canonical_json(obj: Any) -> str` - sorted-key, tight-separator JSON of `qdeep(obj)`
  - `sha256_hex(data: bytes) -> str`
  - `hash_obj(obj: Any) -> str` - `sha256_hex(canonical_json(obj).encode("utf-8"))`
  - `sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str`
  - `utc_now_iso() -> str` - `2026-08-20T11:09:33Z` form, second resolution
  - `ist_now_iso() -> str` - same instant, `+05:30` offset
  - `seed_everything(seed: int = PERI_SEED) -> None` - seeds `random`, `numpy`,
    and `torch` if torch is importable

- [ ] **Step 1: Write the failing test**

Create `tests/test_canon.py`:

```python
import math

from peri.core import canon


def test_q_rounds_to_six_places():
    assert canon.q(1 / 3) == 0.333333
    assert canon.q(2.0) == 2.0
    assert canon.q(-1e-9) == 0.0 or canon.q(-1e-9) == -0.0


def test_qdeep_walks_nested_structures():
    src = {"b": [1 / 3, {"c": 2 / 3}], "a": 1, "s": "x", "n": None, "t": True}
    out = canon.qdeep(src)
    assert out["b"][0] == 0.333333
    assert out["b"][1]["c"] == 0.666667
    assert out["a"] == 1 and out["s"] == "x" and out["n"] is None and out["t"] is True


def test_canonical_json_is_key_sorted_and_tight():
    assert canon.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_hash_obj_is_insertion_order_independent():
    assert canon.hash_obj({"a": 1, "b": 2}) == canon.hash_obj({"b": 2, "a": 1})


def test_hash_obj_is_float_noise_independent():
    left = {"score": 0.1 + 0.2}
    right = {"score": 0.3}
    assert left["score"] != right["score"]
    assert canon.hash_obj(left) == canon.hash_obj(right)


def test_sha256_hex_known_vector():
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert canon.sha256_hex(b"") == empty


def test_sha256_file_matches_sha256_hex(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"peri")
    assert canon.sha256_file(p) == canon.sha256_hex(b"peri")


def test_utc_and_ist_timestamps_have_expected_suffixes():
    assert canon.utc_now_iso().endswith("Z")
    assert canon.ist_now_iso().endswith("+05:30")


def test_nan_and_inf_are_rejected():
    for bad in (math.nan, math.inf, -math.inf):
        try:
            canon.canonical_json({"x": bad})
        except ValueError:
            continue
        raise AssertionError("canonical_json must reject non-finite floats")


def test_seed_everything_makes_random_reproducible():
    import random

    canon.seed_everything()
    a = [random.random() for _ in range(5)]
    canon.seed_everything()
    b = [random.random() for _ in range(5)]
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_canon.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'peri.core.canon'`

- [ ] **Step 3: Write the implementation**

Create `peri/core/canon.py`:

```python
"""Determinism helpers shared by every PERI layer (L0-L8).

The replay hash (L8) is only reproducible if every structure that reaches a hash
goes through exactly one serialisation convention. That convention lives here and
nowhere else.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any

PERI_SEED: int = 20260820

_FLOAT_PLACES = 6
_IST = timezone(timedelta(hours=5, minutes=30))


def q(x: float) -> float:
    """Quantise a float to a fixed decimal precision.

    Float arithmetic that differs in the last bits between two runs must not
    change a hash. Six places is far below any precision we report.
    """
    value = float(x)
    if not math.isfinite(value):
        raise ValueError(f"non-finite float cannot be canonicalised: {value!r}")
    rounded = round(value, _FLOAT_PLACES)
    # Collapse negative zero so -0.0 and 0.0 hash identically.
    return rounded + 0.0


def qdeep(obj: Any) -> Any:
    """Recursively quantise every float inside a JSON-shaped structure."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return q(obj)
    if isinstance(obj, dict):
        return {str(k): qdeep(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [qdeep(v) for v in obj]
    raise TypeError(f"unsupported type in canonical structure: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Sorted-key, tight-separator JSON of the quantised structure."""
    return json.dumps(
        qdeep(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json(obj).encode("utf-8"))


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ist_now_iso() -> str:
    return datetime.now(_IST).replace(microsecond=0).isoformat()


def seed_everything(seed: int = PERI_SEED) -> None:
    """Seed every RNG we might touch. Safe to call when torch is absent."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_canon.py -v`
Expected: PASS, 10 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/canon.py tests/test_canon.py
git commit -m "feat(core): canonical JSON, hashing, and seeding helpers"
```

---

### Task 3: Environment record (`tools/write_environment.py`)

**Files:**
- Create: `tools/write_environment.py`
- Test: `tests/test_environment_record.py`
- Output: `artifacts/environment.json`

**Interfaces:**
- Consumes: `peri.core.canon.canonical_json`, `hash_obj`.
- Produces: `build_environment_record() -> dict` and the on-disk
  `artifacts/environment.json`, read by Phase 9 (report Methods page) and Phase 10
  (manifest). Keys, exactly: `schema`, `generated_utc`, `python`, `platform`,
  `packages`, `binaries`, `deviations`, `record_hash`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_environment_record.py`:

```python
from tools.write_environment import build_environment_record

REQUIRED_KEYS = {
    "schema",
    "generated_utc",
    "python",
    "platform",
    "packages",
    "binaries",
    "deviations",
    "record_hash",
}


def test_record_has_every_required_key():
    record = build_environment_record()
    assert REQUIRED_KEYS <= set(record)


def test_record_pins_the_packages_the_report_cites():
    record = build_environment_record()
    for name in ("numpy", "torch", "reportlab", "opencv-python-headless"):
        assert name in record["packages"], f"{name} missing from environment record"


def test_record_names_ffmpeg_and_ffprobe():
    record = build_environment_record()
    assert "ffmpeg" in record["binaries"]
    assert "ffprobe" in record["binaries"]


def test_python_deviation_is_declared():
    record = build_environment_record()
    joined = " ".join(record["deviations"]).lower()
    assert "3.11" in joined and "3.12" in joined


def test_record_hash_is_stable_across_two_builds():
    first = build_environment_record()
    second = build_environment_record()
    assert first["record_hash"] == second["record_hash"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_environment_record.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'tools.write_environment'`

- [ ] **Step 3: Write the implementation**

Create `tools/write_environment.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_environment_record.py -v`
Expected: PASS, 5 passed.

- [ ] **Step 5: Generate the artifact and read it**

Run:

```bash
.venv/Scripts/python.exe -m tools.write_environment
cat artifacts/environment.json
```

Expected: a JSON file where `packages.torch` is a real version, `packages.numpy` is
a real version, and `binaries.ffmpeg` starts with `ffmpeg version 9.0-full_build`.
If `c2pa-python` reads `not-installed`, that is the Task 1 Step 7 fallback and is
acceptable - confirm Phase 3 knows.

- [ ] **Step 6: Commit**

```bash
git add tools/write_environment.py tests/test_environment_record.py artifacts/environment.json
git commit -m "feat(tools): generated environment record with declared deviations"
```

---

### Task 4: Package layout placeholders and phase gate

**Files:**
- Create: `peri/core/errors.py`
- Test: `tests/test_phase0_gate.py`

**Interfaces:**
- Consumes: `peri.core.canon`.
- Produces: `class PeriError(Exception)`, `class IntakeError(PeriError)`,
  `class CalibrationError(PeriError)`, `class ExaminationError(PeriError)` -
  the exception hierarchy every later phase raises from, so the API can map one base
  class to an HTTP 4xx/5xx and nothing leaks a raw traceback into a findings file.

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase0_gate.py`:

```python
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
    assert sys.version_info[:2] == (3, 12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_phase0_gate.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'peri.core.errors'`

- [ ] **Step 3: Write the implementation**

Create `peri/core/errors.py`:

```python
"""Exception hierarchy for PERI (all layers).

One base class so api/main.py can translate any expected failure into a response
without letting a raw traceback reach a findings file or the examination record.
"""

from __future__ import annotations


class PeriError(Exception):
    """Base class for every expected PERI failure."""


class IntakeError(PeriError):
    """L0: the exhibit could not be received, hashed, sealed, or probed."""


class CalibrationError(PeriError):
    """L4: calibration data is missing, malformed, or too small to fit."""


class ExaminationError(PeriError):
    """L1-L5: a stage failed while examining a sealed exhibit."""
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: PASS - all tests from Tasks 2, 3, 4 green (20 passed).

- [ ] **Step 5: Commit**

```bash
git add peri/core/errors.py tests/test_phase0_gate.py
git commit -m "feat(core): exception hierarchy and phase-0 gate test"
```

---

## Phase 0 acceptance test

Run, and paste the actual output into the phase log:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m tools.write_environment
.venv/Scripts/python.exe -c "from peri.core.canon import hash_obj; print(hash_obj({'a':1,'b':[0.1+0.2]}))"
```

**Pass criteria, all four:**
1. `pytest -q` reports 20 passed, 0 failed.
2. `artifacts/environment.json` exists and pins torch, numpy, reportlab, ffmpeg.
3. The `hash_obj` line prints a 64-character hex digest.
4. Running the `hash_obj` line a second time prints the **same** digest.

**Phase 0 is green when all four hold. Do not start Phase 1 or Phase 2 before that.**
