# Phase 4 — Evidence Fragility Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `peri/core/fragility.py` — an adversarial search that finds, on three
independent laundering axes, the minimum degradation strength at which **our own
verdict flips**, reports it in court-legible units, assigns a band, and forces
`INCONCLUSIVE` when the verdict does not survive ordinary social-media recompression.

**Architecture:** The search is decoupled from the examination. `search_fragility`
takes a `verdict_fn(axis, level) -> str` callable and knows nothing about video; the
Phase 7 wiring supplies a callable that re-encodes the working copy and re-examines
it. That decoupling is why this phase can be built and tested against a stub scorer
before any model exists, which is CLAUDE.md section 3, CPU-track step 5.

Each axis is an **ordered ladder** from mildest to harshest laundering. The search is a
binary search over ladder indices under a stated monotonicity assumption: if the
verdict survives level *k*, it is assumed to survive every level milder than *k*. That
assumption is stated in the report, not hidden — it is what makes the search
logarithmic instead of linear, and a reviewer is entitled to know it.

**Tech Stack:** Python 3.12, ffmpeg (re-encode / rescale), OpenCV + Pillow (JPEG frame
recompression). No torch.

**Spec:** `CLAUDE.md` sections 3 (CPU-track step 5), 6 (bands and the disjointness hard rule).
Roadmap: `docs/superpowers/plans/2026-08-20-ppf-00-ROADMAP.md`.
Depends on: Phase 1 (outcome strings).

## Global Constraints

- **HARD RULE (CLAUDE.md section 6):** training augmentations and fragility-search
  transforms must be **disjoint** — different families, different parameter ranges.
  This is asserted in code by `assert_transform_disjointness()`, which is called at
  import time of both this module and every training script. If it ever raises, the
  robustness claim is circular and the build stops. Both sets are printed on the
  report's Methods page.
- Three axes, no more: re-encode CRF, rescale factor, JPEG quality.
- Bands: `LOW` (survives heavy laundering) · `MODERATE` · `HIGH` (flips under ordinary
  social-media recompression). `HIGH` forces `INCONCLUSIVE`.
- Report units are court-legible: `CRF 34`, `41% rescale`, `JPEG q38`. Never an index,
  never a normalised score.
- Determinism: fixed ladders, no RNG, `-bitexact` on every ffmpeg call.
- Forbidden strings (CLAUDE.md section 8) must not appear.

---

### Task 1: Transform families, ladders, and the disjointness assertion

**Files:**
- Create: `peri/core/fragility.py`
- Test: `tests/test_fragility_disjointness.py`

**Interfaces:**
- Consumes: nothing (this task is pure data plus one assertion).
- Produces:
  - `TRAINING_AUGMENTATIONS: dict[str, dict]` — the **only** augmentations any
    training script may use. Each entry: `{"family": str, "params": dict}`.
    Families: `gaussian_blur`, `additive_gaussian_noise`, `horizontal_flip`,
    `random_crop`.
  - `FRAGILITY_AXES: dict[str, dict]` — three axes. Each entry:
    `{"family": str, "unit": str, "ladder": tuple[float, ...], "label": Callable}`.
    Families: `codec_reencode`, `spatial_rescale`, `jpeg_recompression`.
  - `AXIS_NAMES: tuple[str, ...] = ("reencode_crf", "rescale", "jpeg_quality")`
  - `assert_transform_disjointness() -> None` — raises `AssertionError` naming the
    overlap if any family appears in both sets, or if any training augmentation
    performs codec re-encoding, spatial rescaling, or JPEG recompression.
  - `axis_label(axis: str, level: float) -> str` — `"CRF 34"`, `"41% rescale"`,
    `"JPEG q38"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fragility_disjointness.py`:

```python
import pytest

from peri.core import fragility as frg


def test_there_are_exactly_three_axes():
    assert frg.AXIS_NAMES == ("reencode_crf", "rescale", "jpeg_quality")
    assert set(frg.FRAGILITY_AXES) == set(frg.AXIS_NAMES)


def test_training_and_fragility_families_do_not_intersect():
    training = {v["family"] for v in frg.TRAINING_AUGMENTATIONS.values()}
    search = {v["family"] for v in frg.FRAGILITY_AXES.values()}
    assert training.isdisjoint(search)


def test_disjointness_assertion_passes_on_the_shipped_sets():
    frg.assert_transform_disjointness()  # must not raise


def test_disjointness_assertion_catches_an_injected_overlap(monkeypatch):
    poisoned = dict(frg.TRAINING_AUGMENTATIONS)
    poisoned["sneaky_recompress"] = {
        "family": "codec_reencode",
        "params": {"crf": [20, 30]},
    }
    monkeypatch.setattr(frg, "TRAINING_AUGMENTATIONS", poisoned)
    with pytest.raises(AssertionError) as excinfo:
        frg.assert_transform_disjointness()
    assert "codec_reencode" in str(excinfo.value)


def test_no_training_augmentation_touches_a_fragility_operation():
    banned = ("crf", "jpeg", "quality", "rescale", "resize", "bitrate")
    for name, spec in frg.TRAINING_AUGMENTATIONS.items():
        blob = f"{name} {spec['family']} {' '.join(spec['params'])}".lower()
        for token in banned:
            assert token not in blob, f"{name} looks like a fragility transform"


def test_every_ladder_runs_from_mildest_to_harshest():
    assert list(frg.FRAGILITY_AXES["reencode_crf"]["ladder"]) == sorted(
        frg.FRAGILITY_AXES["reencode_crf"]["ladder"]
    )
    assert list(frg.FRAGILITY_AXES["rescale"]["ladder"]) == sorted(
        frg.FRAGILITY_AXES["rescale"]["ladder"], reverse=True
    )
    assert list(frg.FRAGILITY_AXES["jpeg_quality"]["ladder"]) == sorted(
        frg.FRAGILITY_AXES["jpeg_quality"]["ladder"], reverse=True
    )


def test_ladders_are_long_enough_for_a_meaningful_binary_search():
    for axis in frg.AXIS_NAMES:
        assert len(frg.FRAGILITY_AXES[axis]["ladder"]) >= 8


def test_axis_labels_are_court_legible():
    assert frg.axis_label("reencode_crf", 34) == "CRF 34"
    assert frg.axis_label("rescale", 0.41) == "41% rescale"
    assert frg.axis_label("jpeg_quality", 38) == "JPEG q38"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_disjointness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'peri.core.fragility'`

- [ ] **Step 3: Write the implementation**

Create `peri/core/fragility.py`:

```python
"""L3: the Evidence Fragility Index.

We attack our own conclusion and report where it breaks. Three laundering axes,
each an ordered ladder from mildest to harshest, searched for the minimum strength
at which our verdict changes.

Hard rule, asserted below: the transforms used here and the augmentations used to
train the models are drawn from disjoint families with non-overlapping parameter
ranges. If a model were trained on the same degradation we then use to test it, the
robustness claim would be circular.

Stated assumption: the search treats each ladder as monotone - if the verdict
survives a given strength it is taken to survive every milder strength on that
ladder. This is what makes the search logarithmic rather than exhaustive. It is
stated on the report's Methods page rather than assumed silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# The only augmentations any training script may use. train/ imports this constant
# and must not add to it locally.
TRAINING_AUGMENTATIONS: dict[str, dict] = {
    "blur": {"family": "gaussian_blur", "params": {"sigma_min": 0.5, "sigma_max": 1.5}},
    "noise": {
        "family": "additive_gaussian_noise",
        "params": {"sigma_min_255": 1.0, "sigma_max_255": 5.0},
    },
    "flip": {"family": "horizontal_flip", "params": {"probability": 0.5}},
    "crop": {"family": "random_crop", "params": {"min_fraction": 0.85, "max_fraction": 1.0}},
}

AXIS_NAMES = ("reencode_crf", "rescale", "jpeg_quality")

FRAGILITY_AXES: dict[str, dict] = {
    "reencode_crf": {
        "family": "codec_reencode",
        "unit": "CRF",
        "ladder": (18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44, 48, 51),
    },
    "rescale": {
        "family": "spatial_rescale",
        "unit": "scale factor",
        "ladder": (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.41, 0.35, 0.25, 0.15, 0.10),
    },
    "jpeg_quality": {
        "family": "jpeg_recompression",
        "unit": "JPEG quality",
        "ladder": (95, 90, 85, 80, 70, 60, 50, 40, 38, 30, 20, 10, 5),
    },
}

_FRAGILITY_OPERATION_TOKENS = ("crf", "jpeg", "quality", "rescale", "resize", "bitrate")


def assert_transform_disjointness() -> None:
    """Enforce CLAUDE.md section 6's hard rule. Called at import time, and by every
    training script, so an overlap stops the build instead of quietly invalidating
    the robustness claim.
    """
    training_families = {spec["family"] for spec in TRAINING_AUGMENTATIONS.values()}
    search_families = {spec["family"] for spec in FRAGILITY_AXES.values()}
    overlap = training_families & search_families
    assert not overlap, (
        "training augmentations and fragility transforms share the families "
        f"{sorted(overlap)}; the robustness claim would be circular"
    )
    for name, spec in TRAINING_AUGMENTATIONS.items():
        blob = f"{name} {spec['family']} {' '.join(spec['params'])}".lower()
        for token in _FRAGILITY_OPERATION_TOKENS:
            assert token not in blob, (
                f"training augmentation {name!r} performs a fragility-axis operation "
                f"({token!r}); the two sets must stay disjoint"
            )


assert_transform_disjointness()


def axis_label(axis: str, level: float) -> str:
    """Court-legible rendering of one ladder rung."""
    if axis == "reencode_crf":
        return f"CRF {int(level)}"
    if axis == "rescale":
        return f"{int(round(level * 100))}% rescale"
    if axis == "jpeg_quality":
        return f"JPEG q{int(level)}"
    raise ValueError(f"unknown fragility axis: {axis!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_disjointness.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/fragility.py tests/test_fragility_disjointness.py
git commit -m "feat(fragility): transform ladders and the disjointness hard rule"
```

---

### Task 2: Binary search for the breaking point

**Files:**
- Modify: `peri/core/fragility.py` (append)
- Test: `tests/test_fragility_search.py`

**Interfaces:**
- Consumes: Task 1; Phase 1 outcome strings.
- Produces:
  - `class AxisResult` — frozen dataclass: `axis: str`, `unit: str`,
    `baseline_outcome: str`, `survives_to: float | None`,
    `survives_to_label: str | None`, `flips_at: float | None`,
    `flips_at_label: str | None`, `flipped_outcome: str | None`,
    `evaluations: int`, `band: str`; plus `to_dict()`
  - `search_axis(axis, baseline_outcome, verdict_fn) -> AxisResult` — binary search
    over ladder indices; `verdict_fn(axis: str, level: float) -> str`
  - `AXIS_BAND_RULES: dict[str, dict]` — per-axis `LOW` / `HIGH` cut points

**Band rules per axis** (a flip at or before the `high_at` rung means the conclusion
does not survive ordinary social-media recompression):

| Axis | `HIGH` when it flips at | `LOW` when it survives past | else |
|---|---|---|---|
| `reencode_crf` | CRF ≤ 28 | never flips, or flips at CRF ≥ 34 | `MODERATE` |
| `rescale` | scale ≥ 0.75 | never flips, or flips at scale ≤ 0.50 | `MODERATE` |
| `jpeg_quality` | q ≥ 70 | never flips, or flips at q ≤ 40 | `MODERATE` |

- [ ] **Step 1: Write the failing test**

Create `tests/test_fragility_search.py`:

```python
import pytest

from peri.core import fragility as frg

MANIP = "MANIPULATION INDICATED"
INCONC = "INCONCLUSIVE"


def flip_after(axis, threshold_index):
    """verdict_fn that keeps the baseline verdict for the first N ladder rungs."""
    ladder = frg.FRAGILITY_AXES[axis]["ladder"]
    calls = []

    def verdict_fn(axis_name, level):
        calls.append(level)
        index = ladder.index(level)
        return MANIP if index < threshold_index else INCONC

    return verdict_fn, calls


def test_never_flipping_axis_reports_no_breaking_point():
    verdict_fn, _ = flip_after("reencode_crf", threshold_index=99)
    result = frg.search_axis("reencode_crf", MANIP, verdict_fn)
    assert result.flips_at is None
    assert result.survives_to == frg.FRAGILITY_AXES["reencode_crf"]["ladder"][-1]
    assert result.band == "LOW"


def test_immediately_flipping_axis_is_high_fragility():
    verdict_fn, _ = flip_after("reencode_crf", threshold_index=0)
    result = frg.search_axis("reencode_crf", MANIP, verdict_fn)
    assert result.flips_at == 18
    assert result.survives_to is None
    assert result.band == "HIGH"


def test_search_finds_the_exact_ladder_rung_where_the_verdict_flips():
    ladder = frg.FRAGILITY_AXES["reencode_crf"]["ladder"]
    target = ladder.index(36)
    verdict_fn, _ = flip_after("reencode_crf", threshold_index=target)
    result = frg.search_axis("reencode_crf", MANIP, verdict_fn)
    assert result.flips_at == 36
    assert result.survives_to == 34
    assert result.flips_at_label == "CRF 36"
    assert result.survives_to_label == "CRF 34"


def test_search_is_logarithmic_not_exhaustive():
    ladder = frg.FRAGILITY_AXES["reencode_crf"]["ladder"]
    verdict_fn, calls = flip_after("reencode_crf", threshold_index=ladder.index(36))
    frg.search_axis("reencode_crf", MANIP, verdict_fn)
    assert len(calls) <= 5
    assert len(calls) == len(set(calls)), "the same rung was evaluated twice"


def test_crf_band_boundaries():
    ladder = frg.FRAGILITY_AXES["reencode_crf"]["ladder"]
    for level, expected in ((26, "HIGH"), (28, "HIGH"), (30, "MODERATE"), (34, "LOW"), (40, "LOW")):
        verdict_fn, _ = flip_after("reencode_crf", threshold_index=ladder.index(level))
        assert frg.search_axis("reencode_crf", MANIP, verdict_fn).band == expected


def test_rescale_band_boundaries():
    ladder = frg.FRAGILITY_AXES["rescale"]["ladder"]
    for level, expected in ((0.9, "HIGH"), (0.7, "MODERATE"), (0.5, "LOW"), (0.25, "LOW")):
        verdict_fn, _ = flip_after("rescale", threshold_index=ladder.index(level))
        assert frg.search_axis("rescale", MANIP, verdict_fn).band == expected


def test_jpeg_band_boundaries():
    ladder = frg.FRAGILITY_AXES["jpeg_quality"]["ladder"]
    for level, expected in ((80, "HIGH"), (60, "MODERATE"), (40, "LOW"), (20, "LOW")):
        verdict_fn, _ = flip_after("jpeg_quality", threshold_index=ladder.index(level))
        assert frg.search_axis("jpeg_quality", MANIP, verdict_fn).band == expected


def test_rescale_labels_are_percentages():
    ladder = frg.FRAGILITY_AXES["rescale"]["ladder"]
    verdict_fn, _ = flip_after("rescale", threshold_index=ladder.index(0.35))
    result = frg.search_axis("rescale", MANIP, verdict_fn)
    assert result.flips_at_label == "35% rescale"
    assert result.survives_to_label == "41% rescale"


def test_flipped_outcome_is_recorded():
    ladder = frg.FRAGILITY_AXES["reencode_crf"]["ladder"]
    verdict_fn, _ = flip_after("reencode_crf", threshold_index=ladder.index(36))
    assert frg.search_axis("reencode_crf", MANIP, verdict_fn).flipped_outcome == INCONC


def test_a_verdict_fn_that_errors_is_treated_as_a_flip():
    def exploding(axis, level):
        if level >= 30:
            raise RuntimeError("transcode failed")
        return MANIP

    result = frg.search_axis("reencode_crf", MANIP, exploding)
    assert result.flips_at is not None
    assert result.flipped_outcome == "EVALUATION FAILED"


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError):
        frg.search_axis("brightness", MANIP, lambda a, l: MANIP)


def test_axis_result_dict_is_json_ready():
    from peri.core.canon import canonical_json

    verdict_fn, _ = flip_after("reencode_crf", threshold_index=5)
    payload = frg.search_axis("reencode_crf", MANIP, verdict_fn).to_dict()
    assert "band" in canonical_json(payload)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_search.py -v`
Expected: FAIL — `AttributeError: module 'peri.core.fragility' has no attribute 'search_axis'`

- [ ] **Step 3: Write the implementation**

Append to `peri/core/fragility.py`:

```python
from peri.core.canon import q

BAND_LOW = "LOW"
BAND_MODERATE = "MODERATE"
BAND_HIGH = "HIGH"
BANDS = (BAND_LOW, BAND_MODERATE, BAND_HIGH)

EVALUATION_FAILED = "EVALUATION FAILED"

# "high_at" is the mildest rung at which a flip means the conclusion does not
# survive ordinary social-media recompression. "low_at" is the rung a conclusion
# must reach before we call it robust. Compared with `harsher_or_equal`, which
# knows each ladder's direction.
AXIS_BAND_RULES: dict[str, dict] = {
    "reencode_crf": {"high_at": 28, "low_at": 34},
    "rescale": {"high_at": 0.75, "low_at": 0.50},
    "jpeg_quality": {"high_at": 70, "low_at": 40},
}


def _harsher_or_equal(axis: str, level: float, reference: float) -> bool:
    """True when `level` is at least as much laundering as `reference`."""
    if axis == "reencode_crf":
        return level >= reference
    return level <= reference  # rescale and jpeg ladders descend


@dataclass(frozen=True)
class AxisResult:
    axis: str
    unit: str
    baseline_outcome: str
    survives_to: float | None
    survives_to_label: str | None
    flips_at: float | None
    flips_at_label: str | None
    flipped_outcome: str | None
    evaluations: int
    band: str

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "unit": self.unit,
            "baseline_outcome": self.baseline_outcome,
            "survives_to": None if self.survives_to is None else q(self.survives_to),
            "survives_to_label": self.survives_to_label,
            "flips_at": None if self.flips_at is None else q(self.flips_at),
            "flips_at_label": self.flips_at_label,
            "flipped_outcome": self.flipped_outcome,
            "evaluations": self.evaluations,
            "band": self.band,
        }


def _band_for(axis: str, flips_at: float | None) -> str:
    rules = AXIS_BAND_RULES[axis]
    if flips_at is None:
        return BAND_LOW
    if not _harsher_or_equal(axis, flips_at, rules["high_at"]):
        # Flipped before reaching even the ordinary-recompression rung.
        return BAND_HIGH
    if _harsher_or_equal(axis, flips_at, rules["high_at"]) and not _harsher_or_equal(
        axis, flips_at, rules["low_at"]
    ):
        return BAND_HIGH if flips_at == rules["high_at"] else BAND_MODERATE
    return BAND_LOW


def search_axis(
    axis: str,
    baseline_outcome: str,
    verdict_fn: Callable[[str, float], str],
) -> AxisResult:
    """Binary-search one ladder for the mildest rung at which the verdict changes.

    A verdict_fn that raises is treated as a flip: if the exhibit cannot even be
    evaluated at that strength, the conclusion certainly does not survive it.
    """
    if axis not in FRAGILITY_AXES:
        raise ValueError(f"unknown fragility axis: {axis!r}")

    spec = FRAGILITY_AXES[axis]
    ladder = spec["ladder"]
    cache: dict[int, str] = {}

    def outcome_at(index: int) -> str:
        if index not in cache:
            try:
                cache[index] = verdict_fn(axis, ladder[index])
            except Exception:  # noqa: BLE001 - an unevaluable exhibit is a flip
                cache[index] = EVALUATION_FAILED
        return cache[index]

    lo, hi = 0, len(ladder) - 1
    first_flip: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if outcome_at(mid) == baseline_outcome:
            lo = mid + 1
        else:
            first_flip = mid
            hi = mid - 1

    if first_flip is None:
        survives_index: int | None = len(ladder) - 1
        flips_at = flipped_outcome = None
    else:
        survives_index = first_flip - 1 if first_flip > 0 else None
        flips_at = ladder[first_flip]
        flipped_outcome = cache[first_flip]

    survives_to = ladder[survives_index] if survives_index is not None else None

    return AxisResult(
        axis=axis,
        unit=spec["unit"],
        baseline_outcome=baseline_outcome,
        survives_to=survives_to,
        survives_to_label=None if survives_to is None else axis_label(axis, survives_to),
        flips_at=flips_at,
        flips_at_label=None if flips_at is None else axis_label(axis, flips_at),
        flipped_outcome=flipped_outcome,
        evaluations=len(cache),
        band=_band_for(axis, flips_at),
    )
```

**Implementer note on `_band_for`.** The three-way comparison above is fiddly; write it
as an explicit table walk instead if the boundary tests fail. The required behaviour,
stated once more so it cannot be misread:

- `flips_at is None` → `LOW`
- axis flipped at a rung **milder than or equal to** `high_at` → `HIGH`
- axis flipped at a rung **harsher than or equal to** `low_at` → `LOW`
- otherwise → `MODERATE`

For `reencode_crf`, "milder" means a smaller CRF; for `rescale` and `jpeg_quality`,
"milder" means a larger number. `_harsher_or_equal` already encodes that direction, so
the clean form is:

```python
def _band_for(axis: str, flips_at: float | None) -> str:
    if flips_at is None:
        return BAND_LOW
    rules = AXIS_BAND_RULES[axis]
    if not _harsher_or_equal(axis, flips_at, rules["high_at"]) or flips_at == rules["high_at"]:
        return BAND_HIGH
    if _harsher_or_equal(axis, flips_at, rules["low_at"]):
        return BAND_LOW
    return BAND_MODERATE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_search.py -v`
Expected: PASS, 12 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/fragility.py tests/test_fragility_search.py
git commit -m "feat(fragility): binary-search breaking point with per-axis bands"
```

---

### Task 3: Whole-index assembly and the forced-abstention rule

**Files:**
- Modify: `peri/core/fragility.py` (append)
- Test: `tests/test_fragility_index.py`

**Interfaces:**
- Consumes: Task 2.
- Produces:
  - `class FragilityIndex` — frozen dataclass: `axes: tuple[AxisResult, ...]`,
    `band: str`, `forces_inconclusive: bool`, `statement: str`,
    `monotonicity_assumption: str`; plus `to_dict()`
  - `assess_fragility(baseline_outcome, verdict_fn, axes=AXIS_NAMES) -> FragilityIndex`
  - The statement format, exactly:
    `"Conclusion survives to CRF 34 / 41% rescale / JPEG q38. Flips at CRF 36. FRAGILITY: LOW."`
    When an axis never flips its clause is omitted from the "Flips at" sentence; when
    no axis flips at all the second sentence is
    `"No axis flipped within the searched range."`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fragility_index.py`:

```python
from peri.core import fragility as frg

MANIP = "MANIPULATION INDICATED"
INCONC = "INCONCLUSIVE"


def make_verdict_fn(flip_levels):
    """flip_levels: {axis: first ladder level that flips, or None}"""

    def verdict_fn(axis, level):
        threshold = flip_levels.get(axis)
        if threshold is None:
            return MANIP
        ladder = frg.FRAGILITY_AXES[axis]["ladder"]
        return MANIP if ladder.index(level) < ladder.index(threshold) else INCONC

    return verdict_fn


def test_robust_conclusion_is_low_and_does_not_force_abstention():
    index = frg.assess_fragility(MANIP, make_verdict_fn({}))
    assert index.band == "LOW"
    assert index.forces_inconclusive is False
    assert "No axis flipped within the searched range." in index.statement


def test_statement_matches_the_specified_format():
    index = frg.assess_fragility(
        MANIP, make_verdict_fn({"reencode_crf": 36, "rescale": 0.35, "jpeg_quality": 30})
    )
    assert index.statement.startswith(
        "Conclusion survives to CRF 34 / 41% rescale / JPEG q38."
    )
    assert "FRAGILITY: LOW." in index.statement


def test_overall_band_is_the_worst_axis():
    index = frg.assess_fragility(
        MANIP, make_verdict_fn({"reencode_crf": 36, "rescale": 0.9, "jpeg_quality": 30})
    )
    assert index.band == "HIGH"


def test_high_band_forces_inconclusive():
    index = frg.assess_fragility(MANIP, make_verdict_fn({"jpeg_quality": 80}))
    assert index.band == "HIGH"
    assert index.forces_inconclusive is True


def test_moderate_band_does_not_force_inconclusive():
    index = frg.assess_fragility(
        MANIP, make_verdict_fn({"reencode_crf": 30, "rescale": 0.25, "jpeg_quality": 20})
    )
    assert index.band == "MODERATE"
    assert index.forces_inconclusive is False


def test_all_three_axes_are_always_searched():
    index = frg.assess_fragility(MANIP, make_verdict_fn({}))
    assert tuple(a.axis for a in index.axes) == frg.AXIS_NAMES


def test_monotonicity_assumption_is_stated_in_the_record():
    index = frg.assess_fragility(MANIP, make_verdict_fn({}))
    assert "monoton" in index.monotonicity_assumption.lower()


def test_dict_carries_every_axis_and_the_band():
    payload = frg.assess_fragility(MANIP, make_verdict_fn({"reencode_crf": 36})).to_dict()
    assert len(payload["axes"]) == 3
    assert payload["band"] in frg.BANDS
    assert isinstance(payload["forces_inconclusive"], bool)


def test_flips_sentence_names_only_the_axes_that_flipped():
    index = frg.assess_fragility(MANIP, make_verdict_fn({"reencode_crf": 36}))
    assert "Flips at CRF 36." in index.statement
    assert "rescale." not in index.statement.split("Flips at")[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_index.py -v`
Expected: FAIL — `AttributeError: module 'peri.core.fragility' has no attribute 'assess_fragility'`

- [ ] **Step 3: Write the implementation**

Append to `peri/core/fragility.py`:

```python
_MONOTONICITY_ASSUMPTION = (
    "The search treats each laundering axis as monotone: a conclusion that survives a "
    "given strength is taken to survive every milder strength on the same axis. This "
    "assumption makes the search logarithmic in the length of the ladder. It is stated "
    "here so that it can be examined rather than assumed."
)

_BAND_SEVERITY = {BAND_LOW: 0, BAND_MODERATE: 1, BAND_HIGH: 2}


@dataclass(frozen=True)
class FragilityIndex:
    axes: tuple[AxisResult, ...]
    band: str
    forces_inconclusive: bool
    statement: str
    monotonicity_assumption: str

    def to_dict(self) -> dict:
        return {
            "axes": [a.to_dict() for a in self.axes],
            "band": self.band,
            "forces_inconclusive": self.forces_inconclusive,
            "statement": self.statement,
            "monotonicity_assumption": self.monotonicity_assumption,
        }


def _build_statement(axes: tuple[AxisResult, ...], band: str) -> str:
    survived = [a.survives_to_label or "no laundering" for a in axes]
    first = "Conclusion survives to " + " / ".join(survived) + "."
    flipped = [a.flips_at_label for a in axes if a.flips_at_label]
    second = (
        "Flips at " + " / ".join(flipped) + "."
        if flipped
        else "No axis flipped within the searched range."
    )
    return f"{first} {second} FRAGILITY: {band}."


def assess_fragility(
    baseline_outcome: str,
    verdict_fn: Callable[[str, float], str],
    axes: tuple[str, ...] = AXIS_NAMES,
) -> FragilityIndex:
    """Search every axis and combine the results into one reportable index.

    The overall band is the worst axis, not the average: a conclusion that survives
    heavy recompression but dies under a mild rescale is fragile, and averaging
    would conceal exactly the failure a court needs to know about.
    """
    results = tuple(search_axis(axis, baseline_outcome, verdict_fn) for axis in axes)
    band = max((r.band for r in results), key=lambda b: _BAND_SEVERITY[b])
    return FragilityIndex(
        axes=results,
        band=band,
        forces_inconclusive=band == BAND_HIGH,
        statement=_build_statement(results, band),
        monotonicity_assumption=_MONOTONICITY_ASSUMPTION,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_index.py -v`
Expected: PASS, 9 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/fragility.py tests/test_fragility_index.py
git commit -m "feat(fragility): whole-index assembly and forced abstention on HIGH"
```

---

### Task 4: Real transforms — ffmpeg re-encode, rescale, JPEG frame recompression

**Files:**
- Modify: `peri/core/fragility.py` (append)
- Test: `tests/test_fragility_transforms.py`

**Interfaces:**
- Consumes: Task 1 ladders; ffmpeg on PATH; OpenCV + Pillow.
- Produces:
  - `apply_axis_transform(src, dst, axis, level) -> Path` — writes a laundered copy
  - `make_verdict_fn(working_path, examine_fn, workdir) -> Callable[[str, float], str]`
    — the production adapter Phase 7 passes to `assess_fragility`. `examine_fn` takes
    a path and returns an outcome string. Laundered copies are written into `workdir`
    and are never written back into the evidence directory.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fragility_transforms.py`:

```python
import cv2
import pytest

from peri.core import fragility as frg
from tools.make_demo_clip import make_demo_clip


@pytest.fixture()
def clip(tmp_path):
    return make_demo_clip(tmp_path / "demo.mp4", seconds=2)


def dims(path):
    cap = cv2.VideoCapture(str(path))
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def test_reencode_produces_a_readable_file_of_the_same_size(clip, tmp_path):
    out = frg.apply_axis_transform(clip, tmp_path / "crf40.mp4", "reencode_crf", 40)
    assert out.is_file() and out.stat().st_size > 0
    assert dims(out) == dims(clip)


def test_harsher_crf_produces_a_smaller_file(clip, tmp_path):
    mild = frg.apply_axis_transform(clip, tmp_path / "crf20.mp4", "reencode_crf", 20)
    harsh = frg.apply_axis_transform(clip, tmp_path / "crf48.mp4", "reencode_crf", 48)
    assert harsh.stat().st_size < mild.stat().st_size


def test_rescale_changes_the_frame_size_then_restores_it(clip, tmp_path):
    out = frg.apply_axis_transform(clip, tmp_path / "half.mp4", "rescale", 0.5)
    # The laundered copy is returned at the original size so the examiner sees the
    # same geometry; the information loss is real, the dimensions are restored.
    assert dims(out) == dims(clip)
    assert out.stat().st_size != clip.stat().st_size


def test_jpeg_recompression_produces_a_readable_file(clip, tmp_path):
    out = frg.apply_axis_transform(clip, tmp_path / "q20.mp4", "jpeg_quality", 20)
    assert out.is_file()
    assert dims(out) == dims(clip)


def test_lower_jpeg_quality_loses_more_detail(clip, tmp_path):
    import numpy as np

    def first_frame(path):
        cap = cv2.VideoCapture(str(path))
        try:
            ok, frame = cap.read()
            assert ok
            return frame.astype(np.float64)
        finally:
            cap.release()

    base = first_frame(clip)
    mild = first_frame(frg.apply_axis_transform(clip, tmp_path / "q90.mp4", "jpeg_quality", 90))
    harsh = first_frame(frg.apply_axis_transform(clip, tmp_path / "q10.mp4", "jpeg_quality", 10))
    assert np.abs(harsh - base).mean() > np.abs(mild - base).mean()


def test_transforms_are_deterministic(clip, tmp_path):
    from peri.core.canon import sha256_file

    a = frg.apply_axis_transform(clip, tmp_path / "a.mp4", "reencode_crf", 30)
    b = frg.apply_axis_transform(clip, tmp_path / "b.mp4", "reencode_crf", 30)
    assert sha256_file(a) == sha256_file(b)


def test_unknown_axis_is_rejected(clip, tmp_path):
    with pytest.raises(ValueError):
        frg.apply_axis_transform(clip, tmp_path / "x.mp4", "brightness", 1.0)


def test_verdict_fn_adapter_calls_the_examiner_on_a_laundered_copy(clip, tmp_path):
    seen = []

    def examine_fn(path):
        seen.append(path)
        return "MANIPULATION INDICATED"

    verdict_fn = frg.make_verdict_fn(clip, examine_fn, tmp_path / "work")
    assert verdict_fn("reencode_crf", 30) == "MANIPULATION INDICATED"
    assert len(seen) == 1
    assert seen[0] != clip
    assert seen[0].is_file()


def test_verdict_fn_never_writes_into_the_source_directory(clip, tmp_path):
    before = sorted(p.name for p in clip.parent.iterdir())
    verdict_fn = frg.make_verdict_fn(clip, lambda p: "INCONCLUSIVE", tmp_path / "work")
    verdict_fn("jpeg_quality", 30)
    assert sorted(p.name for p in clip.parent.iterdir()) == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_transforms.py -v`
Expected: FAIL — `AttributeError: module 'peri.core.fragility' has no attribute 'apply_axis_transform'`

- [ ] **Step 3: Write the implementation**

Append to `peri/core/fragility.py`:

```python
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

from peri.core.errors import ExaminationError

_FFMPEG_BITEXACT = ("-bitexact", "-fflags", "+bitexact", "-flags:v", "+bitexact")


def _run_ffmpeg(argv: list[str]) -> None:
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ExaminationError(f"ffmpeg failed: {completed.stderr.strip()[:400]}")


def _reencode(src: Path, dst: Path, crf: int) -> None:
    _run_ffmpeg(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-an", *_FFMPEG_BITEXACT, str(dst),
        ]
    )


def _rescale(src: Path, dst: Path, factor: float) -> None:
    """Downscale then restore the original geometry.

    The information the downscale destroys is gone; restoring the dimensions means
    the examiner is handed a file with the same geometry as the exhibit, so the
    verdict change is attributable to the information loss and not to a change of
    input shape.
    """
    scale = (
        f"scale=trunc(iw*{factor}/2)*2:trunc(ih*{factor}/2)*2:flags=bicubic,"
        f"scale=iw/{factor}:ih/{factor}:flags=bicubic"
    )
    _run_ffmpeg(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src), "-vf", scale,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", *_FFMPEG_BITEXACT, str(dst),
        ]
    )


def _jpeg_recompress(src: Path, dst: Path, quality: int) -> None:
    """Recompress every frame as JPEG at the given quality, then remux losslessly."""
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ExaminationError(f"could not open exhibit for JPEG recompression: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    staging = Path(tempfile.mkdtemp(prefix="peri_jpeg_"))
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=int(quality), optimize=False)
            buffer.seek(0)
            recompressed = np.array(Image.open(buffer).convert("RGB"))
            cv2.imwrite(
                str(staging / f"{index:06d}.png"),
                cv2.cvtColor(recompressed, cv2.COLOR_RGB2BGR),
            )
            index += 1
        cap.release()
        if index == 0:
            raise ExaminationError(f"exhibit produced no decodable frames: {src}")
        _run_ffmpeg(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", f"{fps}", "-i", str(staging / "%06d.png"),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
                *_FFMPEG_BITEXACT, str(dst),
            ]
        )
    finally:
        cap.release()
        shutil.rmtree(staging, ignore_errors=True)


def apply_axis_transform(
    src: str | Path, dst: str | Path, axis: str, level: float
) -> Path:
    """Write a laundered copy of `src` at one ladder rung of `axis`."""
    if axis not in FRAGILITY_AXES:
        raise ValueError(f"unknown fragility axis: {axis!r}")
    source, target = Path(src), Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    if axis == "reencode_crf":
        _reencode(source, target, int(level))
    elif axis == "rescale":
        _rescale(source, target, float(level))
    else:
        _jpeg_recompress(source, target, int(level))

    if not target.is_file() or target.stat().st_size == 0:
        raise ExaminationError(f"fragility transform produced no output: {axis} {level}")
    return target


def make_verdict_fn(
    working_path: str | Path,
    examine_fn: Callable[[Path], str],
    workdir: str | Path,
) -> Callable[[str, float], str]:
    """Adapter that turns an examiner into the verdict_fn assess_fragility expects.

    Laundered copies go to `workdir`, never into the evidence directory: an
    adversarial artefact we generated is not part of the chain of custody.
    """
    source = Path(working_path)
    scratch = Path(workdir)
    scratch.mkdir(parents=True, exist_ok=True)

    def verdict_fn(axis: str, level: float) -> str:
        suffix = source.suffix or ".mp4"
        stem = f"{axis}_{str(level).replace('.', 'p')}"
        laundered = apply_axis_transform(source, scratch / f"{stem}{suffix}", axis, level)
        return examine_fn(laundered)

    return verdict_fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_transforms.py -v`
Expected: PASS, 9 passed. This is the slowest test file in the project (real
transcodes); 60–120 s on this CPU is normal.

- [ ] **Step 5: Commit**

```bash
git add peri/core/fragility.py tests/test_fragility_transforms.py
git commit -m "feat(fragility): ffmpeg and JPEG laundering transforms with an examiner adapter"
```

---

### Task 5: Fragility CLI on a stub scorer (the phase acceptance)

**Files:**
- Create: `tools/fragility_demo.py`
- Test: `tests/test_fragility_acceptance.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `run_stub_fragility(flip_crf=36, flip_scale=0.41, flip_jpeg=38)
  -> FragilityIndex` using a synthetic scorer, and a `__main__` printing the axis
  table and the statement. This is CLAUDE.md section 3, CPU-track step 5: *"Binary search
  returns critical CRF / rescale / JPEG-q on a stub scorer."*

- [ ] **Step 1: Write the failing test**

Create `tests/test_fragility_acceptance.py`:

```python
from peri.core import fragility as frg
from tools.fragility_demo import run_stub_fragility


def test_stub_search_returns_a_critical_value_on_all_three_axes():
    index = run_stub_fragility(flip_crf=36, flip_scale=0.35, flip_jpeg=30)
    by_axis = {a.axis: a for a in index.axes}
    assert by_axis["reencode_crf"].flips_at == 36
    assert by_axis["rescale"].flips_at == 0.35
    assert by_axis["jpeg_quality"].flips_at == 30


def test_stub_search_reports_survival_thresholds_in_court_units():
    index = run_stub_fragility(flip_crf=36, flip_scale=0.35, flip_jpeg=30)
    assert "CRF 34" in index.statement
    assert "41% rescale" in index.statement
    assert "JPEG q38" in index.statement


def test_stub_search_assigns_a_band_from_the_declared_set():
    assert run_stub_fragility().band in frg.BANDS


def test_a_fragile_conclusion_forces_abstention():
    index = run_stub_fragility(flip_crf=22, flip_scale=0.9, flip_jpeg=85)
    assert index.band == "HIGH"
    assert index.forces_inconclusive is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_acceptance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.fragility_demo'`

- [ ] **Step 3: Write the implementation**

Create `tools/fragility_demo.py`:

```python
"""Phase 4 acceptance driver: run the fragility search against a stub scorer.

No model is required. The stub declares, per axis, the ladder rung at which the
verdict would flip; the search must rediscover exactly that rung.
"""

from __future__ import annotations

from peri.core import fragility as frg
from peri.core.forensic_lr import OUTCOME_INCONCLUSIVE, OUTCOME_MANIPULATION


def run_stub_fragility(
    flip_crf: float | None = 36,
    flip_scale: float | None = 0.41,
    flip_jpeg: float | None = 38,
) -> frg.FragilityIndex:
    thresholds = {
        "reencode_crf": flip_crf,
        "rescale": flip_scale,
        "jpeg_quality": flip_jpeg,
    }

    def verdict_fn(axis: str, level: float) -> str:
        threshold = thresholds.get(axis)
        if threshold is None:
            return OUTCOME_MANIPULATION
        ladder = frg.FRAGILITY_AXES[axis]["ladder"]
        flipped = ladder.index(level) >= ladder.index(threshold)
        return OUTCOME_INCONCLUSIVE if flipped else OUTCOME_MANIPULATION

    return frg.assess_fragility(OUTCOME_MANIPULATION, verdict_fn)


def main() -> int:
    index = run_stub_fragility()
    print(f"{'axis':<16}{'survives to':<18}{'flips at':<18}{'evals':<7}band")
    for axis in index.axes:
        print(
            f"{axis.axis:<16}"
            f"{(axis.survives_to_label or '-'):<18}"
            f"{(axis.flips_at_label or 'never'):<18}"
            f"{axis.evaluations:<7}{axis.band}"
        )
    print()
    print(index.statement)
    print(f"forces INCONCLUSIVE: {index.forces_inconclusive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fragility_acceptance.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 5: Run the driver by hand**

Run: `.venv/Scripts/python.exe -m tools.fragility_demo`
Expected, exactly this shape:

```
axis            survives to       flips at          evals  band
reencode_crf    CRF 34            CRF 36            4      LOW
rescale         41% rescale       41% rescale       ...    ...
jpeg_quality    JPEG q40          JPEG q38          ...    ...

Conclusion survives to CRF 34 / ... . Flips at CRF 36 / ... . FRAGILITY: ... .
forces INCONCLUSIVE: False
```

- [ ] **Step 6: Commit**

```bash
git add tools/fragility_demo.py tests/test_fragility_acceptance.py
git commit -m "feat(tools): fragility acceptance driver on a stub scorer"
```

---

## Phase 4 acceptance test

```bash
.venv/Scripts/python.exe -m pytest tests/test_fragility_disjointness.py tests/test_fragility_search.py tests/test_fragility_index.py tests/test_fragility_transforms.py tests/test_fragility_acceptance.py -q
.venv/Scripts/python.exe -m tools.fragility_demo
```

**Pass criteria, all six:**
1. 42 tests pass, 0 fail.
2. The driver prints a critical value for all three axes and a band from
   `{LOW, MODERATE, HIGH}`.
3. `assert_transform_disjointness()` runs at import and does not raise.
4. Injecting an overlapping family into `TRAINING_AUGMENTATIONS` makes it raise with
   the offending family named (covered by
   `test_disjointness_assertion_catches_an_injected_overlap`).
5. Two runs of `apply_axis_transform` with identical arguments produce identical
   SHA-256 values.
6. No laundered file is ever written inside an `evidence/` directory (covered by
   `test_verdict_fn_never_writes_into_the_source_directory`).

**Phase 4 is green when all six hold.** Phase 7 (API) may wire `make_verdict_fn` to
the real examiner.
