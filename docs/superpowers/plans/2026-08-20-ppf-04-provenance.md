# Phase 3 - Provenance Stream (S4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `peri/core/provenance.py` - a **rule-based, zero-ML** stream that
reads a C2PA manifest if one exists, extracts structured container facts, applies a
fixed set of named metadata-contradiction rules, and emits a score, a feature vector,
and stress replicas in the shape the LR layer already expects.

**Architecture:** Pure functions over the `container` dict that Phase 2's
`probe_container` already produces, plus an optional C2PA read. Each rule is a frozen
dataclass with an ID, a plain-English statement, a weight, and a pure predicate. The
score is the weighted fraction of triggered rules. Nothing here learns anything, so
nothing here can fail because training failed - which is exactly why CLAUDE.md
requires it to work when everything else does not.

**Tech Stack:** Python 3.12, `c2pa` (optional - degrades to `unavailable`), stdlib.
No torch, no network, no model files.

**Spec:** `CLAUDE.md` sections 2 (S4), 3 (CPU-track step 3), 8 (C2PA precedence).
Roadmap: `docs/superpowers/plans/2026-08-20-ppf-00-ROADMAP.md`.
Depends on: Phase 2 (`probe_container`).

## Global Constraints

- **Zero ML.** No torch import, no model file, no learned threshold. If a reviewer
  greps this module for `torch`, they find nothing.
- **Must survive total training failure.** This module is the floor of the system.
- **C2PA precedence, verbatim policy (CLAUDE.md section 8):** C2PA verifies that
  provenance claims have not been tampered with, **not** that they are truthful.
  Where forensic findings contradict a manifest, forensic findings take precedence and
  **both are reported**. Rule `PRV-09` exists solely to surface that contradiction.
- Score orientation: **higher means Hd**. The provenance score is a weighted fraction
  in `[0.0, 1.0]`; more triggered contradiction rules means a higher score.
- Rule IDs are stable strings. Renaming one after Phase 6 calibration invalidates the
  calibration - treat IDs as an on-disk format.
- All floats through `peri.core.canon.q`. No RNG.
- Forbidden strings (CLAUDE.md section 8) must not appear. A rule statement says
  *"is inconsistent with"*, never *"proves"*.

## The rule set (fixed; ten rules, IDs frozen)

| ID | Statement | Weight |
|---|---|---|
| `PRV-01` | The container carries no creation timestamp. | 1.0 |
| `PRV-02` | The encoder or handler string names a software rendering or editing tool rather than a capture device. | 3.0 |
| `PRV-03` | The container carries no capture-device make or model tag. | 1.5 |
| `PRV-04` | The declared creation timestamp is later than the moment of examination. | 2.0 |
| `PRV-05` | The declared frame count is inconsistent with duration multiplied by frame rate. | 2.0 |
| `PRV-06` | The container major brand is a generic muxer brand rather than a device brand. | 1.0 |
| `PRV-07` | No C2PA provenance manifest is attached to the exhibit. | 1.0 |
| `PRV-08` | A C2PA manifest is attached but its validation reported errors. | 3.0 |
| `PRV-09` | A C2PA manifest asserts a capture device while the container encoder names a rendering tool; these two claims are inconsistent. | 3.0 |
| `PRV-10` | The exhibit carries no audio stream, which is inconsistent with an ordinary camera capture. | 1.0 |

`score = sum(weight of triggered rules) / sum(all evaluable weights)`. A rule that
cannot be evaluated (for example `PRV-08` when no manifest exists) is excluded from
**both** the numerator and the denominator, and is recorded as `not-evaluated` - it is
never silently counted as passing.

---

### Task 1: C2PA reader with a hard fallback

**Files:**
- Create: `peri/core/provenance.py`
- Test: `tests/test_provenance_c2pa.py`

**Interfaces:**
- Consumes: `peri.core.canon.q`.
- Produces:
  - `C2PA_STATUSES: tuple[str, ...] = ("present", "absent", "invalid", "unavailable")`
  - `read_c2pa(path: str | Path) -> dict` - returns
    `{"status", "library_available", "claim_generator", "assertions",
      "validation_errors", "raw_excerpt"}`. Never raises: an exhibit with no manifest
    is the normal case, and a missing library is a stated limitation, not a crash.

- [ ] **Step 1: Write the failing test**

Create `tests/test_provenance_c2pa.py`:

```python
from peri.core.provenance import C2PA_STATUSES, read_c2pa
from tools.make_demo_clip import make_demo_clip


def test_status_is_one_of_the_declared_values(tmp_path):
    clip = make_demo_clip(tmp_path / "demo.mp4", seconds=2)
    assert read_c2pa(clip)["status"] in C2PA_STATUSES


def test_a_plain_ffmpeg_clip_has_no_manifest(tmp_path):
    clip = make_demo_clip(tmp_path / "demo.mp4", seconds=2)
    result = read_c2pa(clip)
    assert result["status"] in ("absent", "unavailable")
    assert result["validation_errors"] == []


def test_reader_never_raises_on_a_non_media_file(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("not a video", encoding="utf-8")
    result = read_c2pa(junk)
    assert result["status"] in ("absent", "invalid", "unavailable")


def test_reader_never_raises_on_a_missing_file(tmp_path):
    result = read_c2pa(tmp_path / "gone.mp4")
    assert result["status"] in ("absent", "invalid", "unavailable")


def test_result_always_carries_every_key(tmp_path):
    clip = make_demo_clip(tmp_path / "demo.mp4", seconds=2)
    result = read_c2pa(clip)
    for key in (
        "status", "library_available", "claim_generator", "assertions",
        "validation_errors", "raw_excerpt",
    ):
        assert key in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_provenance_c2pa.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'peri.core.provenance'`

- [ ] **Step 3: Write the implementation**

Create `peri/core/provenance.py`:

```python
"""S4: provenance and metadata contradiction analysis. Rule-based, no machine learning.

This stream is the floor of the system. It has no model file, no learned threshold,
and no training dependency, so it produces structured findings even if every other
stage is unavailable.

On C2PA: a valid manifest establishes that the recorded provenance claims have not
been altered since they were signed. It does not establish that those claims are
truthful. Where the findings of the other streams contradict a manifest, the
forensic findings take precedence, and both are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from peri.core.canon import q

C2PA_STATUSES = ("present", "absent", "invalid", "unavailable")

_EMPTY_C2PA = {
    "status": "absent",
    "library_available": False,
    "claim_generator": "",
    "assertions": [],
    "validation_errors": [],
    "raw_excerpt": "",
}


def read_c2pa(path: str | Path) -> dict:
    """Read a C2PA manifest if one is attached. Never raises.

    Four outcomes: present, absent, invalid (a manifest exists but could not be
    parsed or validated), unavailable (the reader library is not installed on this
    workstation). 'unavailable' is reported as a limitation on the report, never
    silently treated as 'absent'.
    """
    result = dict(_EMPTY_C2PA)
    target = Path(path)

    try:
        import c2pa  # type: ignore
    except ImportError:
        result["status"] = "unavailable"
        result["library_available"] = False
        return result

    result["library_available"] = True
    if not target.is_file():
        result["status"] = "absent"
        return result

    try:
        reader = c2pa.Reader.from_file(str(target))
        raw = reader.json()
    except Exception:  # noqa: BLE001 - any reader failure means "no usable manifest"
        result["status"] = "absent"
        return result

    if not raw:
        result["status"] = "absent"
        return result

    result["raw_excerpt"] = str(raw)[:2000]
    try:
        import json as _json

        parsed = _json.loads(raw)
    except Exception:  # noqa: BLE001
        result["status"] = "invalid"
        result["validation_errors"] = ["manifest present but could not be parsed"]
        return result

    manifests = parsed.get("manifests") or {}
    active_id = parsed.get("active_manifest") or (next(iter(manifests), None))
    active = manifests.get(active_id, {}) if active_id else {}

    result["status"] = "present"
    result["claim_generator"] = str(active.get("claim_generator", ""))
    result["assertions"] = [
        str(a.get("label", "")) for a in (active.get("assertions") or [])
    ]
    result["validation_errors"] = [
        str(item.get("code", "")) for item in (parsed.get("validation_status") or [])
    ]
    if result["validation_errors"]:
        result["status"] = "present"
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_provenance_c2pa.py -v`
Expected: PASS, 5 passed. `status` will be `unavailable` if `c2pa-python` did not
install in Phase 0 - that is a passing state, not a failure.

- [ ] **Step 5: Commit**

```bash
git add peri/core/provenance.py tests/test_provenance_c2pa.py
git commit -m "feat(provenance): C2PA reader with unavailable/absent/invalid fallbacks"
```

---

### Task 2: The ten contradiction rules

**Files:**
- Modify: `peri/core/provenance.py` (append)
- Test: `tests/test_provenance_rules.py`

**Interfaces:**
- Consumes: Task 1; Phase 2 `probe_container` output shape.
- Produces:
  - `class Rule` - frozen dataclass: `rule_id: str`, `statement: str`,
    `weight: float`, `evaluable: Callable[[dict], bool]`,
    `triggered: Callable[[dict], bool]`
  - `RULES: tuple[Rule, ...]` - the ten rules, in ID order
  - `SOFTWARE_ENCODERS: frozenset[str]` - lowercase substrings that identify a
    rendering or editing tool: `{"lavf", "ffmpeg", "handbrake", "adobe", "premiere",
    "after effects", "blender", "moviepy", "obs", "avidemux", "shotcut", "davinci",
    "vegas", "x264", "x265", "mencoder", "imovie", "kdenlive"}`
  - `DEVICE_BRANDS: frozenset[str]` - major brands a capture device writes:
    `{"qt", "3gp4", "3gp5", "mp41", "avc1", "heic", "mif1", "msnv"}`
  - `evaluate_rules(facts: dict) -> list[dict]` - one dict per rule with keys
    `rule_id`, `statement`, `weight`, `evaluated`, `triggered`

**`facts` shape** (built in Task 3, but the rules are written against it now):

```python
{
  "container": {...},          # exactly probe_container()'s return value
  "c2pa": {...},               # exactly read_c2pa()'s return value
  "examined_at_utc": "2026-08-20T11:09:33Z",
}
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_provenance_rules.py`:

```python
import pytest

from peri.core import provenance as prov


def base_container(**overrides):
    container = {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "format_long_name": "QuickTime / MOV",
        "duration_s": 4.0,
        "bit_rate": 500_000,
        "size_bytes": 250_000,
        "stream_count": 2,
        "video": {
            "codec": "h264",
            "profile": "High",
            "width": 320,
            "height": 240,
            "fps": 25.0,
            "nb_frames": 100,
            "pix_fmt": "yuv420p",
        },
        "audio": {"codec": "aac", "sample_rate": 44100, "channels": 2},
        "tags": {
            "creation_time": "2026-08-19T10:00:00.000000Z",
            "major_brand": "qt",
            "encoder": "Apple iPhone 15",
            "com.apple.quicktime.make": "Apple",
            "com.apple.quicktime.model": "iPhone 15",
        },
    }
    container.update(overrides)
    return container


def facts(container=None, c2pa=None, now="2026-08-20T11:00:00Z"):
    return {
        "container": container or base_container(),
        "c2pa": c2pa or {
            "status": "absent",
            "library_available": True,
            "claim_generator": "",
            "assertions": [],
            "validation_errors": [],
            "raw_excerpt": "",
        },
        "examined_at_utc": now,
    }


def triggered_ids(f):
    return {r["rule_id"] for r in prov.evaluate_rules(f) if r["triggered"]}


def evaluated_ids(f):
    return {r["rule_id"] for r in prov.evaluate_rules(f) if r["evaluated"]}


def test_there_are_exactly_ten_rules_with_unique_ids():
    ids = [r.rule_id for r in prov.RULES]
    assert len(ids) == 10
    assert len(set(ids)) == 10
    assert ids == sorted(ids)


def test_a_clean_camera_capture_triggers_only_the_c2pa_absence_rule():
    assert triggered_ids(facts()) == {"PRV-07"}


def test_missing_creation_time_triggers_prv01():
    container = base_container()
    container["tags"].pop("creation_time")
    assert "PRV-01" in triggered_ids(facts(container))


def test_software_encoder_triggers_prv02():
    container = base_container()
    container["tags"]["encoder"] = "Lavf61.7.100"
    assert "PRV-02" in triggered_ids(facts(container))


def test_missing_device_tags_trigger_prv03():
    container = base_container()
    container["tags"] = {"creation_time": "2026-08-19T10:00:00.000000Z", "major_brand": "qt"}
    assert "PRV-03" in triggered_ids(facts(container))


def test_future_creation_time_triggers_prv04():
    container = base_container()
    container["tags"]["creation_time"] = "2027-01-01T00:00:00.000000Z"
    assert "PRV-04" in triggered_ids(facts(container))


def test_frame_count_inconsistency_triggers_prv05():
    container = base_container()
    container["video"]["nb_frames"] = 40  # duration 4.0s at 25fps implies 100
    assert "PRV-05" in triggered_ids(facts(container))


def test_frame_count_within_tolerance_does_not_trigger_prv05():
    container = base_container()
    container["video"]["nb_frames"] = 101
    assert "PRV-05" not in triggered_ids(facts(container))


def test_zero_frame_count_makes_prv05_not_evaluable():
    container = base_container()
    container["video"]["nb_frames"] = 0
    assert "PRV-05" not in evaluated_ids(facts(container))


def test_generic_muxer_brand_triggers_prv06():
    container = base_container()
    container["tags"]["major_brand"] = "isom"
    assert "PRV-06" in triggered_ids(facts(container))


def test_present_manifest_clears_prv07():
    present = {
        "status": "present",
        "library_available": True,
        "claim_generator": "Acme Camera 1.0",
        "assertions": ["c2pa.actions"],
        "validation_errors": [],
        "raw_excerpt": "{}",
    }
    assert "PRV-07" not in triggered_ids(facts(c2pa=present))


def test_manifest_validation_errors_trigger_prv08():
    bad = {
        "status": "present",
        "library_available": True,
        "claim_generator": "Acme Camera 1.0",
        "assertions": [],
        "validation_errors": ["signingCredential.untrusted"],
        "raw_excerpt": "{}",
    }
    assert "PRV-08" in triggered_ids(facts(c2pa=bad))


def test_prv08_is_not_evaluated_when_no_manifest_exists():
    assert "PRV-08" not in evaluated_ids(facts())


def test_manifest_contradicting_the_encoder_triggers_prv09():
    container = base_container()
    container["tags"]["encoder"] = "Lavf61.7.100"
    manifest = {
        "status": "present",
        "library_available": True,
        "claim_generator": "Acme Camera Capture 2.1",
        "assertions": ["c2pa.actions.v2"],
        "validation_errors": [],
        "raw_excerpt": "{}",
    }
    assert "PRV-09" in triggered_ids(facts(container, manifest))


def test_missing_audio_triggers_prv10():
    container = base_container(audio=None)
    assert "PRV-10" in triggered_ids(facts(container))


def test_unavailable_c2pa_library_makes_all_c2pa_rules_unevaluable():
    unavailable = {
        "status": "unavailable",
        "library_available": False,
        "claim_generator": "",
        "assertions": [],
        "validation_errors": [],
        "raw_excerpt": "",
    }
    ids = evaluated_ids(facts(c2pa=unavailable))
    assert "PRV-07" not in ids and "PRV-08" not in ids and "PRV-09" not in ids


def test_every_statement_avoids_overclaiming_language():
    banned = ("proves", "guaranteed", "certified", "admissible")
    for rule in prov.RULES:
        lowered = rule.statement.lower()
        for word in banned:
            assert word not in lowered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_provenance_rules.py -v`
Expected: FAIL - `AttributeError: module 'peri.core.provenance' has no attribute 'RULES'`

- [ ] **Step 3: Write the implementation**

Append to `peri/core/provenance.py`:

```python
SOFTWARE_ENCODERS = frozenset(
    {
        "lavf", "ffmpeg", "handbrake", "adobe", "premiere", "after effects",
        "blender", "moviepy", "obs", "avidemux", "shotcut", "davinci", "vegas",
        "x264", "x265", "mencoder", "imovie", "kdenlive",
    }
)

DEVICE_BRANDS = frozenset({"qt", "3gp4", "3gp5", "mp41", "avc1", "heic", "mif1", "msnv"})

_DEVICE_TAG_HINTS = ("make", "model", "manufacturer", "device", "camera")
_FRAME_COUNT_TOLERANCE = 0.02


def _tags(facts: dict) -> dict:
    return {str(k).lower(): str(v) for k, v in (facts["container"].get("tags") or {}).items()}


def _encoder_text(facts: dict) -> str:
    tags = _tags(facts)
    parts = [tags.get(key, "") for key in ("encoder", "handler_name", "writing_library")]
    return " ".join(parts).lower()


def _is_software_encoder(text: str) -> bool:
    return any(token in text for token in SOFTWARE_ENCODERS)


def _parse_iso(value: str) -> datetime | None:
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _c2pa_evaluable(facts: dict) -> bool:
    return bool(facts["c2pa"].get("library_available"))


def _c2pa_present(facts: dict) -> bool:
    return facts["c2pa"].get("status") == "present"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    statement: str
    weight: float
    evaluable: Callable[[dict], bool]
    triggered: Callable[[dict], bool]


def _prv05_evaluable(facts: dict) -> bool:
    video = facts["container"]["video"]
    return bool(video.get("nb_frames")) and video.get("fps", 0) > 0 and facts["container"].get("duration_s", 0) > 0


def _prv05_triggered(facts: dict) -> bool:
    video = facts["container"]["video"]
    expected = facts["container"]["duration_s"] * video["fps"]
    if expected <= 0:
        return False
    return abs(video["nb_frames"] - expected) / expected > _FRAME_COUNT_TOLERANCE


def _prv04_triggered(facts: dict) -> bool:
    declared = _parse_iso(_tags(facts).get("creation_time", ""))
    examined = _parse_iso(facts["examined_at_utc"])
    if declared is None or examined is None:
        return False
    return declared > examined


RULES: tuple[Rule, ...] = (
    Rule(
        "PRV-01",
        "The container carries no creation timestamp.",
        1.0,
        lambda f: True,
        lambda f: "creation_time" not in _tags(f),
    ),
    Rule(
        "PRV-02",
        "The encoder or handler string names a software rendering or editing tool "
        "rather than a capture device.",
        3.0,
        lambda f: bool(_encoder_text(f).strip()),
        lambda f: _is_software_encoder(_encoder_text(f)),
    ),
    Rule(
        "PRV-03",
        "The container carries no capture-device make or model tag.",
        1.5,
        lambda f: True,
        lambda f: not any(
            hint in key for key in _tags(f) for hint in _DEVICE_TAG_HINTS
        ),
    ),
    Rule(
        "PRV-04",
        "The declared creation timestamp is later than the moment of examination.",
        2.0,
        lambda f: _parse_iso(_tags(f).get("creation_time", "")) is not None,
        _prv04_triggered,
    ),
    Rule(
        "PRV-05",
        "The declared frame count is inconsistent with duration multiplied by frame rate.",
        2.0,
        _prv05_evaluable,
        _prv05_triggered,
    ),
    Rule(
        "PRV-06",
        "The container major brand is a generic muxer brand rather than a device brand.",
        1.0,
        lambda f: "major_brand" in _tags(f),
        lambda f: _tags(f).get("major_brand", "").strip().lower() not in DEVICE_BRANDS,
    ),
    Rule(
        "PRV-07",
        "No C2PA provenance manifest is attached to the exhibit.",
        1.0,
        _c2pa_evaluable,
        lambda f: not _c2pa_present(f),
    ),
    Rule(
        "PRV-08",
        "A C2PA manifest is attached but its validation reported errors.",
        3.0,
        lambda f: _c2pa_evaluable(f) and _c2pa_present(f),
        lambda f: bool(f["c2pa"].get("validation_errors")),
    ),
    Rule(
        "PRV-09",
        "A C2PA manifest asserts a capture device while the container encoder names a "
        "rendering tool; these two claims are inconsistent.",
        3.0,
        lambda f: _c2pa_evaluable(f) and _c2pa_present(f),
        lambda f: (
            not _is_software_encoder(str(f["c2pa"].get("claim_generator", "")).lower())
            and _is_software_encoder(_encoder_text(f))
        ),
    ),
    Rule(
        "PRV-10",
        "The exhibit carries no audio stream, which is inconsistent with an ordinary "
        "camera capture.",
        1.0,
        lambda f: True,
        lambda f: f["container"].get("audio") is None,
    ),
)


def evaluate_rules(facts: dict) -> list[dict]:
    """Apply every rule. A rule that cannot be evaluated is recorded, not assumed."""
    out: list[dict] = []
    for rule in RULES:
        evaluated = bool(rule.evaluable(facts))
        out.append(
            {
                "rule_id": rule.rule_id,
                "statement": rule.statement,
                "weight": q(rule.weight),
                "evaluated": evaluated,
                "triggered": bool(rule.triggered(facts)) if evaluated else False,
            }
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_provenance_rules.py -v`
Expected: PASS, 17 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/provenance.py tests/test_provenance_rules.py
git commit -m "feat(provenance): ten frozen metadata contradiction rules"
```

---

### Task 3: Stream assembly - facts, score, feature vector, stress replicas

**Files:**
- Modify: `peri/core/provenance.py` (append)
- Test: `tests/test_provenance_stream.py`

**Interfaces:**
- Consumes: Tasks 1–2, Phase 2 `probe_container`, Phase 1 `StreamObservation`.
- Produces:
  - `PROVENANCE_FEATURE_DIM: int = 6`
  - `collect_facts(path, container=None, now_utc=None) -> dict`
  - `provenance_score(rule_results: list[dict]) -> float` - weighted fraction in `[0,1]`
  - `provenance_feature(facts, rule_results) -> tuple[float, ...]` - 6 dimensions:
    `(score, n_triggered, tag_count, log10(bit_rate + 1), fps, duration_s)`
  - `provenance_stress_scores(facts) -> tuple[float, ...]` - leave-one-tag-out
    replicas, at most 8, always at least 1
  - `analyse(path, container=None, now_utc=None) -> dict` - the stream's full record:
    `{"stream": "provenance", "score", "feature", "stress_scores", "rules",
      "c2pa", "facts_summary", "n_triggered", "n_evaluated"}`
  - `to_observation(analysis: dict, weight: float = 1.0) -> StreamObservation`

**Why leave-one-tag-out is the right stress for this stream.** The other streams are
stressed by degrading pixels. Degrading pixels does not perturb metadata, so a pixel
stress would report this stream as perfectly stable and overstate its reliability. The
analogous perturbation for a metadata stream is the loss of one metadata field, which
is exactly what a remux or an upload pipeline does. This reasoning goes in the report's
Methods page.

- [ ] **Step 1: Write the failing test**

Create `tests/test_provenance_stream.py`:

```python
import pytest

from peri.core import provenance as prov
from peri.core.intake import probe_container
from tools.make_demo_clip import make_demo_clip


@pytest.fixture()
def clip(tmp_path):
    return make_demo_clip(tmp_path / "demo.mp4", seconds=2)


def test_score_is_a_fraction_between_zero_and_one(clip):
    analysis = prov.analyse(clip)
    assert 0.0 <= analysis["score"] <= 1.0


def test_score_rises_when_more_rules_trigger():
    from tests.test_provenance_rules import base_container, facts

    clean = prov.provenance_score(prov.evaluate_rules(facts()))
    dirty_container = base_container()
    dirty_container["tags"]["encoder"] = "Lavf61.7.100"
    dirty_container["tags"]["major_brand"] = "isom"
    dirty = prov.provenance_score(prov.evaluate_rules(facts(dirty_container)))
    assert dirty > clean


def test_unevaluated_rules_are_excluded_from_the_denominator():
    from tests.test_provenance_rules import facts

    results = prov.evaluate_rules(facts())
    evaluable_weight = sum(r["weight"] for r in results if r["evaluated"])
    triggered_weight = sum(r["weight"] for r in results if r["triggered"])
    assert prov.provenance_score(results) == pytest.approx(
        triggered_weight / evaluable_weight
    )


def test_feature_vector_has_the_declared_dimension(clip):
    analysis = prov.analyse(clip)
    assert len(analysis["feature"]) == prov.PROVENANCE_FEATURE_DIM


def test_feature_vector_is_all_finite_floats(clip):
    import math

    for value in prov.analyse(clip)["feature"]:
        assert isinstance(value, float) and math.isfinite(value)


def test_stress_scores_are_produced_and_bounded(clip):
    analysis = prov.analyse(clip)
    assert 1 <= len(analysis["stress_scores"]) <= 8
    assert all(0.0 <= s <= 1.0 for s in analysis["stress_scores"])


def test_analysis_is_deterministic(clip):
    assert prov.analyse(clip) == prov.analyse(clip)


def test_analysis_reuses_a_supplied_container_probe(clip):
    container = probe_container(clip)
    analysis = prov.analyse(clip, container=container)
    assert analysis["facts_summary"]["video_codec"] == container["video"]["codec"]


def test_observation_carries_the_score_feature_and_stress(clip):
    analysis = prov.analyse(clip)
    obs = prov.to_observation(analysis)
    assert obs.name == "provenance"
    assert obs.score == analysis["score"]
    assert tuple(obs.feature) == tuple(analysis["feature"])
    assert tuple(obs.stress_scores) == tuple(analysis["stress_scores"])


def test_module_imports_no_machine_learning_library():
    import pathlib

    source = pathlib.Path("peri/core/provenance.py").read_text(encoding="utf-8")
    for banned in ("import torch", "from torch", "sklearn", "timm"):
        assert banned not in source


def test_analysis_still_works_when_c2pa_is_unavailable(clip, monkeypatch):
    monkeypatch.setattr(prov, "read_c2pa", lambda p: dict(prov._EMPTY_C2PA, status="unavailable"))
    analysis = prov.analyse(clip)
    assert analysis["c2pa"]["status"] == "unavailable"
    assert 0.0 <= analysis["score"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_provenance_stream.py -v`
Expected: FAIL - `AttributeError: module 'peri.core.provenance' has no attribute 'analyse'`

- [ ] **Step 3: Write the implementation**

Append to `peri/core/provenance.py`:

```python
import copy
import math

from peri.core.canon import utc_now_iso
from peri.core.forensic_lr import StreamObservation
from peri.core.intake import probe_container

PROVENANCE_FEATURE_DIM = 6
_MAX_STRESS_REPLICAS = 8


def collect_facts(
    path: str | Path,
    container: dict | None = None,
    now_utc: str | None = None,
) -> dict:
    return {
        "container": container if container is not None else probe_container(path),
        "c2pa": read_c2pa(path),
        "examined_at_utc": now_utc or utc_now_iso(),
    }


def provenance_score(rule_results: list[dict]) -> float:
    """Weighted fraction of triggered rules among the rules that could be evaluated."""
    evaluable = sum(r["weight"] for r in rule_results if r["evaluated"])
    if evaluable <= 0:
        return 0.0
    triggered = sum(r["weight"] for r in rule_results if r["triggered"])
    return q(triggered / evaluable)


def provenance_feature(facts: dict, rule_results: list[dict]) -> tuple[float, ...]:
    container = facts["container"]
    return (
        q(provenance_score(rule_results)),
        q(float(sum(1 for r in rule_results if r["triggered"]))),
        q(float(len(container.get("tags") or {}))),
        q(math.log10(float(container.get("bit_rate", 0) or 0) + 1.0)),
        q(float(container["video"].get("fps", 0.0) or 0.0)),
        q(float(container.get("duration_s", 0.0) or 0.0)),
    )


def provenance_stress_scores(facts: dict) -> tuple[float, ...]:
    """Re-score with one metadata tag removed at a time.

    Pixel degradation does not perturb metadata, so a pixel-based stress would
    report this stream as perfectly stable and overstate its reliability. The
    analogous perturbation for a metadata stream is the loss of a metadata field,
    which is what an ordinary remux or upload pipeline does.
    """
    baseline = provenance_score(evaluate_rules(facts))
    tags = list((facts["container"].get("tags") or {}).keys())
    replicas: list[float] = [baseline]
    for tag in sorted(tags)[:_MAX_STRESS_REPLICAS - 1]:
        perturbed = copy.deepcopy(facts)
        perturbed["container"]["tags"].pop(tag, None)
        replicas.append(provenance_score(evaluate_rules(perturbed)))
    return tuple(replicas)


def analyse(
    path: str | Path,
    container: dict | None = None,
    now_utc: str | None = None,
) -> dict:
    facts = collect_facts(path, container=container, now_utc=now_utc)
    rules = evaluate_rules(facts)
    score = provenance_score(rules)
    video = facts["container"]["video"]
    return {
        "stream": "provenance",
        "score": score,
        "feature": list(provenance_feature(facts, rules)),
        "stress_scores": list(provenance_stress_scores(facts)),
        "rules": rules,
        "c2pa": facts["c2pa"],
        "n_triggered": sum(1 for r in rules if r["triggered"]),
        "n_evaluated": sum(1 for r in rules if r["evaluated"]),
        "facts_summary": {
            "format_name": facts["container"]["format_name"],
            "video_codec": video["codec"],
            "width": video["width"],
            "height": video["height"],
            "fps": video["fps"],
            "duration_s": facts["container"]["duration_s"],
            "tag_count": len(facts["container"].get("tags") or {}),
            "has_audio": facts["container"].get("audio") is not None,
            "examined_at_utc": facts["examined_at_utc"],
        },
        "c2pa_precedence_note": (
            "A C2PA manifest establishes that the recorded provenance claims have not "
            "been altered since signing. It does not establish that those claims are "
            "truthful. Where the findings of the other streams contradict a manifest, "
            "the forensic findings take precedence and both are reported."
        ),
    }


def to_observation(analysis: dict, weight: float = 1.0) -> StreamObservation:
    return StreamObservation(
        name="provenance",
        score=float(analysis["score"]),
        feature=tuple(float(v) for v in analysis["feature"]),
        stress_scores=tuple(float(v) for v in analysis["stress_scores"]),
        weight=float(weight),
    )
```

**Note on `examined_at_utc`:** it appears in `facts_summary` and therefore in findings.
Phase 10's `canonical_findings()` strips it before hashing. Do not remove it from the
record - the report needs it.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_provenance_stream.py -v`
Expected: PASS, 11 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/provenance.py tests/test_provenance_stream.py
git commit -m "feat(provenance): stream assembly with leave-one-tag-out stress replicas"
```

---

### Task 4: Provenance CLI and phase acceptance

**Files:**
- Create: `tools/provenance_demo.py`
- Test: `tests/test_provenance_acceptance.py`

**Interfaces:**
- Consumes: Task 3.
- Produces: `run_provenance_demo(path) -> dict` (the `analyse` record) and a
  `__main__` printing a rule table plus the score.

- [ ] **Step 1: Write the failing test**

Create `tests/test_provenance_acceptance.py`:

```python
from tools.make_demo_clip import make_demo_clip
from tools.provenance_demo import run_provenance_demo


def test_provenance_produces_structured_facts_and_a_score_with_no_model(tmp_path):
    clip = make_demo_clip(tmp_path / "demo.mp4", seconds=2)
    record = run_provenance_demo(clip)

    assert record["stream"] == "provenance"
    assert 0.0 <= record["score"] <= 1.0
    assert len(record["rules"]) == 10
    assert record["n_evaluated"] >= 7
    assert record["facts_summary"]["video_codec"] == "h264"


def test_provenance_needs_no_artifacts_directory(tmp_path, monkeypatch):
    clip = make_demo_clip(tmp_path / "demo.mp4", seconds=2)
    monkeypatch.chdir(tmp_path)  # no artifacts/ here at all
    record = run_provenance_demo(clip)
    assert record["score"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_provenance_acceptance.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'tools.provenance_demo'`

- [ ] **Step 3: Write the implementation**

Create `tools/provenance_demo.py`:

```python
"""Phase 3 acceptance driver: print the provenance rule table for one exhibit."""

from __future__ import annotations

import sys
from pathlib import Path

from peri.core.provenance import analyse


def run_provenance_demo(path: str | Path) -> dict:
    return analyse(path)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m tools.provenance_demo <path-to-exhibit>")
        return 2
    record = run_provenance_demo(sys.argv[1])
    print(f"C2PA status : {record['c2pa']['status']}")
    print(f"score       : {record['score']:.4f}  "
          f"({record['n_triggered']} of {record['n_evaluated']} evaluated rules triggered)")
    print("-- rules --")
    for rule in record["rules"]:
        if not rule["evaluated"]:
            mark = "n/a "
        elif rule["triggered"]:
            mark = "HIT "
        else:
            mark = "ok  "
        print(f"  {mark}{rule['rule_id']}  w={rule['weight']:.1f}  {rule['statement']}")
    print(f"stress replicas: {['%.3f' % s for s in record['stress_scores']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_provenance_acceptance.py -v`
Expected: PASS, 2 passed.

- [ ] **Step 5: Run the driver by hand**

```bash
.venv/Scripts/python.exe -m tools.provenance_demo evidence/_fixtures/demo.mp4
```

Expected: a C2PA status line, a score line, ten rule lines each marked `HIT`, `ok`, or
`n/a`, and a stress replica list. For the ffmpeg-generated fixture, expect `PRV-02`
(encoder `Lavf`), `PRV-03`, `PRV-06`, and `PRV-10` to be `HIT` - a synthetically muxed
clip is exactly what this rule set is meant to notice.

- [ ] **Step 6: Commit**

```bash
git add tools/provenance_demo.py tests/test_provenance_acceptance.py
git commit -m "feat(tools): provenance acceptance driver"
```

---

## Phase 3 acceptance test

```bash
.venv/Scripts/python.exe -m pytest tests/test_provenance_c2pa.py tests/test_provenance_rules.py tests/test_provenance_stream.py tests/test_provenance_acceptance.py -q
.venv/Scripts/python.exe -m tools.provenance_demo evidence/_fixtures/demo.mp4
grep -rnE "import torch|from torch|sklearn|timm" peri/core/provenance.py
```

**Pass criteria, all five:**
1. 35 tests pass, 0 fail.
2. The driver prints a ten-row rule table with a score in `[0,1]`.
3. The `grep` returns nothing - this stream imports no ML library.
4. Deleting `artifacts/` entirely and re-running the driver still produces the same
   rule table. **This is the "must work even if all training fails" criterion.**
5. `analyse()` called twice on the same file returns equal dicts, except for
   `facts_summary.examined_at_utc`.

**Phase 3 is green when all five hold.** Phase 7 (API) may consume this stream.
