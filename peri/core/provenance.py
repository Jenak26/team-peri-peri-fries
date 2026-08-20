"""Rule-based provenance stream."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from peri.core.canon import q
from peri.core.forensic_lr import StreamObservation

C2PA_STATUSES = ("present", "absent", "invalid", "unavailable")
PROVENANCE_FEATURE_DIM = 6


def read_c2pa(path: str | Path) -> dict:
    """Best-effort C2PA read; never raises."""
    try:
        import c2pa  # type: ignore
    except Exception:
        return {"status": "unavailable", "validation_errors": [], "claims": []}
    try:
        # c2pa-python APIs have changed across releases; keep this defensive.
        reader = getattr(c2pa, "Reader", None)
        if reader is None:
            return {"status": "unavailable", "validation_errors": [], "claims": []}
        manifest = reader(str(path)).json()
        return {"status": "present", "validation_errors": [], "claims": [manifest]}
    except Exception as exc:
        text = str(exc)
        if "not found" in text.lower() or "no manifest" in text.lower():
            return {"status": "absent", "validation_errors": [], "claims": []}
        return {"status": "invalid", "validation_errors": [text[:200]], "claims": []}


def collect_facts(path: str | Path, container: dict | None = None, now_utc=None) -> dict:
    target = Path(path)
    tags = dict((container or {}).get("tags") or {})
    video = dict((container or {}).get("video") or {})
    now = now_utc or datetime.now(timezone.utc)
    return {
        "path": str(target),
        "suffix": target.suffix.lower(),
        "size_bytes": target.stat().st_size if target.is_file() else 0,
        "container": container or {},
        "tags": tags,
        "video": video,
        "c2pa": read_c2pa(target),
        "now_utc": now.isoformat(),
    }


def _rule(code: str, evaluated: bool, triggered: bool, weight: float, statement: str) -> dict:
    return {
        "code": code,
        "evaluated": bool(evaluated),
        "triggered": bool(triggered),
        "weight": q(weight),
        "statement": statement,
    }


def evaluate_rules(facts: dict) -> list[dict]:
    tags = {str(k).lower(): str(v) for k, v in facts.get("tags", {}).items()}
    video = facts.get("video", {})
    c2pa = facts.get("c2pa", {})
    status = c2pa.get("status", "unavailable")
    has_device = any(k in tags for k in ("make", "model", "com.apple.quicktime.make"))
    has_encoder = any(k in tags for k in ("encoder", "software", "major_brand"))
    duration = float((facts.get("container") or {}).get("duration_s", 0.0) or 0.0)
    width = int(video.get("width", 0) or 0)
    height = int(video.get("height", 0) or 0)
    suffix = str(facts.get("suffix", ""))
    errors = c2pa.get("validation_errors") or []
    return [
        _rule(
            "PRV01",
            True,
            status == "absent",
            1.0,
            "No C2PA manifest was found; absence is recorded but is not by itself a finding.",
        ),
        _rule(
            "PRV02",
            True,
            status == "invalid" or bool(errors),
            2.0,
            "A C2PA manifest failed validation; forensic findings are still evaluated independently.",
        ),
        _rule(
            "PRV03",
            True,
            not has_device,
            1.0,
            "Device make or model tags were not available in the container metadata.",
        ),
        _rule(
            "PRV04",
            True,
            has_encoder and "lavf" in " ".join(tags.values()).lower(),
            0.8,
            "The container reports a software encoder in its metadata.",
        ),
        _rule(
            "PRV05",
            True,
            duration <= 0.0,
            1.5,
            "The reported duration is missing or zero.",
        ),
        _rule(
            "PRV06",
            True,
            width <= 0 or height <= 0,
            1.5,
            "The reported video dimensions are missing or zero.",
        ),
        _rule(
            "PRV07",
            True,
            suffix not in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"},
            0.5,
            "The file extension is outside the expected video set.",
        ),
        _rule(
            "PRV08",
            bool(tags),
            "creation_time" not in tags,
            0.7,
            "Container tags exist but no creation time tag was available.",
        ),
        _rule(
            "PRV09",
            True,
            int(facts.get("size_bytes", 0)) < 1024,
            1.0,
            "The exhibit file is too small for the stated video container.",
        ),
        _rule(
            "PRV10",
            True,
            False,
            0.2,
            "C2PA verifies that provenance claims have not been tampered with, not that they are truthful.",
        ),
    ]


def provenance_score(rule_results: list[dict]) -> float:
    evaluated = [r for r in rule_results if r["evaluated"]]
    denom = sum(float(r["weight"]) for r in evaluated)
    if denom <= 0:
        return 0.0
    return q(sum(float(r["weight"]) for r in evaluated if r["triggered"]) / denom)


def provenance_feature(facts: dict, rule_results: list[dict]) -> tuple[float, ...]:
    tags = facts.get("tags", {})
    c2pa = facts.get("c2pa", {})
    return (
        provenance_score(rule_results),
        q(len(tags) / 20.0),
        1.0 if c2pa.get("status") == "present" else 0.0,
        1.0 if c2pa.get("status") == "invalid" else 0.0,
        q(float((facts.get("container") or {}).get("duration_s", 0.0) or 0.0) / 60.0),
        q(float((facts.get("video") or {}).get("width", 0) or 0) / 4096.0),
    )


def provenance_stress_scores(facts: dict) -> tuple[float, ...]:
    rules = evaluate_rules(facts)
    base = provenance_score(rules)
    return (max(0.0, q(base - 0.05)), base, min(1.0, q(base + 0.05)))


def analyse(path: str | Path, container: dict | None = None, now_utc=None) -> dict:
    facts = collect_facts(path, container=container, now_utc=now_utc)
    rules = evaluate_rules(facts)
    return {
        "facts": facts,
        "rules": rules,
        "score": provenance_score(rules),
        "feature": provenance_feature(facts, rules),
        "stress_scores": provenance_stress_scores(facts),
    }


def to_observation(analysis: dict, weight: float = 1.0) -> StreamObservation:
    return StreamObservation(
        name="provenance",
        score=float(analysis["score"]),
        feature=tuple(float(v) for v in analysis["feature"]),
        stress_scores=tuple(float(v) for v in analysis["stress_scores"]),
        weight=weight,
    )
