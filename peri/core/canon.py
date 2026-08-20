"""Determinism helpers shared by every layer (L0-L8).

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
    return round(value, _FLOAT_PLACES) + 0.0


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
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
