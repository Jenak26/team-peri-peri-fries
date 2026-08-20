# Phase 1 — Likelihood-Ratio Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `peri/core/forensic_lr.py` — the layer that turns raw stream scores
into a log₁₀ likelihood ratio between two stated propositions, gates on the validated
domain, excludes unstable streams with machine reason codes, fuses under a stated
dependence discount, and returns one of exactly three outcomes.

**Architecture:** Pure functions over frozen dataclasses. No I/O, no model imports, no
global state. Densities are fitted per stream on the held-out `cal` split — Gaussian
KDE with a shared Silverman bandwidth, logistic fallback under 15 samples per class.
The validated-domain gate is a Mahalanobis distance fitted on **feature vectors**, not
on scores. Everything is unit-tested against synthetic scores **before any model
exists**, which is why this is Phase 1 and not Phase 6.

**Tech Stack:** Python 3.12, NumPy, SciPy (`chi2` only), dataclasses. No torch.

**Spec:** `CLAUDE.md` section 5 (authoritative for every constant and every decision
rule), section 2 layers L2 and L4. Roadmap:
`docs/superpowers/plans/2026-08-20-ppf-00-ROADMAP.md`.

## Global Constants (copied verbatim from CLAUDE.md section 5)

```
LOG10LR_DECISION_THRESHOLD = 1.0
STABILITY_IQR_MAX          = 1.0
MAHALANOBIS_QUANTILE       = 0.99
DEPENDENCE_SHRINKAGE       = 0.5
LR_CLIP                    = 6.0
```

## Global Constraints

- **Score orientation:** every stream score is oriented **higher means Hd**. The LR
  layer assumes this and never re-orients; wrappers negate at their own boundary.
- **Fusion formula, verbatim:** `log10LR_total = clip(λ · Σ wₛ · median(stressₛ))`.
  The weights `wₛ` are per-stream weights defaulting to `1.0` and are **not**
  normalised — `Σ` is a sum, and `λ = 0.5` is the dependence discount applied to that
  sum. This is deliberately conservative: with one usable stream it halves that
  stream's own LR. That conservatism is stated in the report, never hidden.
- **Never claim independence** anywhere in code comments, docstrings, or output text.
- Three outcomes only: `MANIPULATION INDICATED`, `AUTHENTICITY SUPPORTED`,
  `INCONCLUSIVE`. There is no fourth.
- Reason codes are fixed strings from a frozen set — never free text.
- Determinism: all floats reaching a serialised structure go through
  `peri.core.canon.q`. No RNG in this module at all.
- Forbidden strings (CLAUDE.md section 8) must not appear. In particular the verbal
  scale says *"support for"*, never *"proves"*.

## Reason-code vocabulary (frozen; nothing else may be emitted)

| Code | Emitted when |
|---|---|
| `out-of-validated-domain` | stream feature vector fails the Mahalanobis gate |
| `sign-unstable-under-degradation` | stream LR changes sign under stress with meaningful magnitude |
| `unstable-under-degradation` | stress LR IQR exceeds `STABILITY_IQR_MAX` |
| `no-usable-stream` | every stream was excluded |
| `cross-stream-contradiction` | two usable streams with `abs(LR) > 1` point opposite ways |
| `evidence-strength-below-reporting-threshold` | `abs(total) < LOG10LR_DECISION_THRESHOLD` |

---

### Task 1: Propositions, constants, and the ENFSI verbal scale

**Files:**
- Create: `peri/core/forensic_lr.py`
- Test: `tests/test_forensic_lr_scale.py`

**Interfaces:**
- Consumes: `peri.core.canon.q`.
- Produces, relied on by Phases 6, 7, 8, 9:
  - `HP_TEXT: str`, `HD_TEXT: str` — the two propositions, verbatim
  - `LOG10LR_DECISION_THRESHOLD`, `STABILITY_IQR_MAX`, `MAHALANOBIS_QUANTILE`,
    `DEPENDENCE_SHRINKAGE`, `LR_CLIP` — module-level floats
  - `OUTCOME_MANIPULATION`, `OUTCOME_AUTHENTIC`, `OUTCOME_INCONCLUSIVE` — the three
    outcome strings
  - `REASON_CODES: frozenset[str]`
  - `enfsi_verbal(log10lr: float) -> str` — the strength band alone, lowercase
  - `enfsi_sentence(log10lr: float) -> str` — band plus the named supported proposition

- [ ] **Step 1: Write the failing test**

Create `tests/test_forensic_lr_scale.py`:

```python
import pytest

from peri.core import forensic_lr as flr


def test_constants_match_the_specification():
    assert flr.LOG10LR_DECISION_THRESHOLD == 1.0
    assert flr.STABILITY_IQR_MAX == 1.0
    assert flr.MAHALANOBIS_QUANTILE == 0.99
    assert flr.DEPENDENCE_SHRINKAGE == 0.5
    assert flr.LR_CLIP == 6.0


def test_propositions_are_verbatim():
    assert flr.HP_TEXT == "the exhibit is an unmanipulated recording of a real event"
    assert flr.HD_TEXT == (
        "the exhibit is synthetically generated or materially manipulated in the "
        "facial region"
    )


def test_there_are_exactly_three_outcomes():
    assert flr.OUTCOMES == (
        "MANIPULATION INDICATED",
        "AUTHENTICITY SUPPORTED",
        "INCONCLUSIVE",
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "no support"),
        (0.99, "no support"),
        (1.0, "moderate support"),
        (1.99, "moderate support"),
        (2.0, "moderately strong support"),
        (3.0, "strong support"),
        (4.0, "very strong support"),
        (5.0, "extremely strong support"),
        (7.5, "extremely strong support"),
    ],
)
def test_verbal_scale_bands(value, expected):
    assert flr.enfsi_verbal(value) == expected
    assert flr.enfsi_verbal(-value) == expected


def test_sentence_names_the_supported_proposition():
    manip = flr.enfsi_sentence(2.5)
    auth = flr.enfsi_sentence(-2.5)
    assert flr.HD_TEXT in manip and flr.HP_TEXT not in manip
    assert flr.HP_TEXT in auth and flr.HD_TEXT not in auth
    assert "moderately strong support" in manip


def test_sentence_below_threshold_supports_neither():
    text = flr.enfsi_sentence(0.4)
    assert "neither proposition" in text


def test_reason_codes_are_frozen_and_complete():
    assert flr.REASON_CODES == frozenset(
        {
            "out-of-validated-domain",
            "sign-unstable-under-degradation",
            "unstable-under-degradation",
            "no-usable-stream",
            "cross-stream-contradiction",
            "evidence-strength-below-reporting-threshold",
        }
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_scale.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'peri.core.forensic_lr'`

- [ ] **Step 3: Write the implementation**

Create `peri/core/forensic_lr.py`:

```python
"""L2 + L4: validated-domain gate, likelihood ratios, fusion, and the decision gate.

This module is deliberately free of I/O, model imports, and randomness. It is the
one place where a number becomes a statement about two propositions, so it is the
one place that must be testable without any model in existence.

Scores entering this module are oriented so that a higher score is more consistent
with Hd. Wrappers are responsible for that orientation; this module never re-orients.
"""

from __future__ import annotations

import math

HP_TEXT = "the exhibit is an unmanipulated recording of a real event"
HD_TEXT = (
    "the exhibit is synthetically generated or materially manipulated in the "
    "facial region"
)

LOG10LR_DECISION_THRESHOLD = 1.0
STABILITY_IQR_MAX = 1.0
MAHALANOBIS_QUANTILE = 0.99
DEPENDENCE_SHRINKAGE = 0.5
LR_CLIP = 6.0

OUTCOME_MANIPULATION = "MANIPULATION INDICATED"
OUTCOME_AUTHENTIC = "AUTHENTICITY SUPPORTED"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
OUTCOMES = (OUTCOME_MANIPULATION, OUTCOME_AUTHENTIC, OUTCOME_INCONCLUSIVE)

REASON_OUT_OF_DOMAIN = "out-of-validated-domain"
REASON_SIGN_UNSTABLE = "sign-unstable-under-degradation"
REASON_UNSTABLE = "unstable-under-degradation"
REASON_NO_USABLE_STREAM = "no-usable-stream"
REASON_CONTRADICTION = "cross-stream-contradiction"
REASON_BELOW_THRESHOLD = "evidence-strength-below-reporting-threshold"

REASON_CODES = frozenset(
    {
        REASON_OUT_OF_DOMAIN,
        REASON_SIGN_UNSTABLE,
        REASON_UNSTABLE,
        REASON_NO_USABLE_STREAM,
        REASON_CONTRADICTION,
        REASON_BELOW_THRESHOLD,
    }
)

_BANDS = (
    (1.0, "no support"),
    (2.0, "moderate support"),
    (3.0, "moderately strong support"),
    (4.0, "strong support"),
    (5.0, "very strong support"),
)


def enfsi_verbal(log10lr: float) -> str:
    """Verbal equivalent of an LR magnitude on the ENFSI scale."""
    magnitude = abs(float(log10lr))
    for upper, label in _BANDS:
        if magnitude < upper:
            return label
    return "extremely strong support"


def enfsi_sentence(log10lr: float) -> str:
    """Verbal equivalent with the supported proposition named explicitly."""
    value = float(log10lr)
    band = enfsi_verbal(value)
    if abs(value) < LOG10LR_DECISION_THRESHOLD:
        return (
            f"The findings provide {band} for either proposition over the other; "
            f"they distinguish neither proposition at the reporting threshold."
        )
    supported = HD_TEXT if value > 0 else HP_TEXT
    other = HP_TEXT if value > 0 else HD_TEXT
    return (
        f"The findings provide {band} for the proposition that {supported}, "
        f"rather than the proposition that {other}."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_scale.py -v`
Expected: PASS, 16 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/forensic_lr.py tests/test_forensic_lr_scale.py
git commit -m "feat(lr): propositions, spec constants, ENFSI verbal scale"
```

---

### Task 2: Density fitting — KDE with logistic fallback

**Files:**
- Modify: `peri/core/forensic_lr.py` (append)
- Test: `tests/test_forensic_lr_density.py`

**Interfaces:**
- Consumes: Task 1 constants; `peri.core.errors.CalibrationError`.
- Produces:
  - `MIN_KDE_SAMPLES_PER_CLASS: int = 15`
  - `silverman_bandwidth(values: Sequence[float]) -> float`
  - `class StreamCalibration` — frozen dataclass with fields
    `name: str`, `method: str`, `hp_scores: tuple[float, ...]`,
    `hd_scores: tuple[float, ...]`, `bandwidth: float`, `logistic_coef: float`,
    `logistic_intercept: float`, `prior_log_odds: float`,
    `feature_mean: tuple[float, ...]`,
    `feature_cov_inv: tuple[tuple[float, ...], ...]`,
    `mahalanobis_threshold: float`, `feature_dim: int`;
    plus `to_dict() -> dict` and classmethod `from_dict(d: dict) -> StreamCalibration`
  - `fit_stream_calibration(name, hp_scores, hd_scores, features) -> StreamCalibration`
  - `log10_lr(cal: StreamCalibration, score: float) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_forensic_lr_density.py`:

```python
import numpy as np
import pytest

from peri.core import forensic_lr as flr
from peri.core.errors import CalibrationError

RNG = np.random.default_rng(20260820)


def make_cal(name="s", sep=3.0, n=200, dim=4, seed=1):
    rng = np.random.default_rng(seed)
    hp = rng.normal(0.0, 1.0, n).tolist()
    hd = rng.normal(sep, 1.0, n).tolist()
    features = rng.normal(0.0, 1.0, (2 * n, dim)).tolist()
    return flr.fit_stream_calibration(name, hp, hd, features)


def test_kde_is_chosen_when_both_classes_are_large_enough():
    cal = make_cal()
    assert cal.method == "kde"
    assert cal.bandwidth > 0.0


def test_logistic_is_chosen_below_the_sample_floor():
    rng = np.random.default_rng(7)
    hp = rng.normal(0.0, 1.0, 8).tolist()
    hd = rng.normal(3.0, 1.0, 8).tolist()
    features = rng.normal(0.0, 1.0, (16, 3)).tolist()
    cal = flr.fit_stream_calibration("small", hp, hd, features)
    assert cal.method == "logistic"


def test_lr_is_positive_for_hd_like_scores_and_negative_for_hp_like():
    cal = make_cal()
    assert flr.log10_lr(cal, 4.0) > 1.0
    assert flr.log10_lr(cal, -1.0) < -1.0


def test_lr_is_monotone_increasing_in_score_across_the_bulk():
    cal = make_cal()
    values = [flr.log10_lr(cal, s) for s in (-1.0, 0.0, 1.5, 3.0, 4.0)]
    assert values == sorted(values)


def test_lr_is_clipped_to_the_specified_bound():
    cal = make_cal(sep=12.0)
    assert flr.log10_lr(cal, 40.0) == pytest.approx(flr.LR_CLIP)
    assert flr.log10_lr(cal, -40.0) == pytest.approx(-flr.LR_CLIP)


def test_logistic_lr_removes_the_fitted_prior():
    # Deliberately imbalanced classes: a posterior-derived LR must not inherit
    # the class imbalance of the calibration corpus.
    rng = np.random.default_rng(11)
    hp = rng.normal(0.0, 1.0, 10).tolist()
    hd = rng.normal(3.0, 1.0, 4).tolist()
    features = rng.normal(0.0, 1.0, (14, 3)).tolist()
    cal = flr.fit_stream_calibration("imb", hp, hd, features)
    assert cal.method == "logistic"
    assert cal.prior_log_odds < 0.0  # fewer Hd than Hp samples
    midpoint = flr.log10_lr(cal, 1.5)
    assert -1.0 < midpoint < 1.0


def test_calibration_round_trips_through_dict():
    cal = make_cal()
    restored = flr.StreamCalibration.from_dict(cal.to_dict())
    assert restored == cal
    assert flr.log10_lr(restored, 2.0) == flr.log10_lr(cal, 2.0)


def test_empty_class_is_a_calibration_error():
    with pytest.raises(CalibrationError):
        flr.fit_stream_calibration("bad", [], [1.0, 2.0], [[0.0], [1.0]])


def test_degenerate_zero_variance_scores_do_not_crash():
    hp = [1.0] * 20
    hd = [1.0] * 20
    features = RNG.normal(0.0, 1.0, (40, 2)).tolist()
    cal = flr.fit_stream_calibration("flat", hp, hd, features)
    assert abs(flr.log10_lr(cal, 1.0)) < 0.5


def test_silverman_bandwidth_is_positive_even_for_constant_input():
    assert flr.silverman_bandwidth([2.0] * 30) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_density.py -v`
Expected: FAIL — `AttributeError: module 'peri.core.forensic_lr' has no attribute 'fit_stream_calibration'`

- [ ] **Step 3: Write the implementation**

Append to `peri/core/forensic_lr.py` (add the new imports at the top of the file
alongside the existing ones):

```python
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from peri.core.canon import q
from peri.core.errors import CalibrationError

MIN_KDE_SAMPLES_PER_CLASS = 15
_DENSITY_FLOOR = 1e-12
_LN10 = math.log(10.0)


def silverman_bandwidth(values: Sequence[float]) -> float:
    """Silverman rule-of-thumb bandwidth, floored so it is never zero.

    A shared bandwidth is used for both classes so that the ratio of the two
    densities is not an artefact of two different smoothing choices.
    """
    arr = np.asarray(values, dtype=float)
    n = max(arr.size, 1)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    iqr = float(np.subtract(*np.percentile(arr, [75, 25]))) if arr.size > 1 else 0.0
    spread = min(x for x in (std, iqr / 1.349) if x > 0.0) if (std > 0 or iqr > 0) else 0.0
    if spread <= 0.0:
        spread = max(float(np.ptp(arr)), 1.0) * 0.05
    return float(0.9 * spread * n ** (-1.0 / 5.0))


def _kde_density(x: float, samples: np.ndarray, bandwidth: float) -> float:
    z = (float(x) - samples) / bandwidth
    return float(np.exp(-0.5 * z * z).sum() / (samples.size * bandwidth * math.sqrt(2 * math.pi)))


def _fit_logistic(hp: np.ndarray, hd: np.ndarray) -> tuple[float, float]:
    """Single-feature logistic fit by Newton-Raphson. No sklearn dependency here:
    a two-parameter fit with a fixed iteration count is deterministic, which the
    replay hash requires.
    """
    x = np.concatenate([hp, hd])
    y = np.concatenate([np.zeros(hp.size), np.ones(hd.size)])
    scale = float(x.std(ddof=0)) or 1.0
    xs = x / scale
    beta = np.zeros(2, dtype=float)
    design = np.column_stack([xs, np.ones_like(xs)])
    for _ in range(50):
        eta = design @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        gradient = design.T @ (y - p)
        hessian = design.T @ (design * w[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if float(np.abs(step).max()) < 1e-10:
            break
    return float(beta[0] / scale), float(beta[1])


def _fit_mahalanobis(
    features: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...], float, int]:
    """Fit the validated-domain gate on the calibration feature population.

    Both classes are pooled: the declared validated domain is the set of exhibits
    the calibration corpus actually covers, regardless of which proposition each
    calibration item belonged to.
    """
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise CalibrationError("at least two calibration feature vectors are required")
    dim = int(matrix.shape[1])
    mean = matrix.mean(axis=0)
    cov = np.cov(matrix, rowvar=False)
    cov = np.atleast_2d(cov)
    ridge = 1e-6 * (float(np.trace(cov)) / dim if dim else 1.0)
    cov_inv = np.linalg.pinv(cov + ridge * np.eye(dim))
    centred = matrix - mean
    md2 = np.einsum("ij,jk,ik->i", centred, cov_inv, centred)
    if matrix.shape[0] >= 10:
        threshold = float(np.quantile(md2, MAHALANOBIS_QUANTILE))
    else:
        from scipy.stats import chi2

        threshold = float(chi2.ppf(MAHALANOBIS_QUANTILE, df=dim))
    return (
        tuple(float(v) for v in mean),
        tuple(tuple(float(v) for v in row) for row in cov_inv),
        max(threshold, 1e-9),
        dim,
    )


@dataclass(frozen=True)
class StreamCalibration:
    """Everything needed to turn one stream's score into a log10 LR, plus its gate."""

    name: str
    method: str
    hp_scores: tuple[float, ...]
    hd_scores: tuple[float, ...]
    bandwidth: float
    logistic_coef: float
    logistic_intercept: float
    prior_log_odds: float
    feature_mean: tuple[float, ...]
    feature_cov_inv: tuple[tuple[float, ...], ...]
    mahalanobis_threshold: float
    feature_dim: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "method": self.method,
            "hp_scores": [q(v) for v in self.hp_scores],
            "hd_scores": [q(v) for v in self.hd_scores],
            "bandwidth": q(self.bandwidth),
            "logistic_coef": q(self.logistic_coef),
            "logistic_intercept": q(self.logistic_intercept),
            "prior_log_odds": q(self.prior_log_odds),
            "feature_mean": [q(v) for v in self.feature_mean],
            "feature_cov_inv": [[q(v) for v in row] for row in self.feature_cov_inv],
            "mahalanobis_threshold": q(self.mahalanobis_threshold),
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StreamCalibration":
        return cls(
            name=str(data["name"]),
            method=str(data["method"]),
            hp_scores=tuple(float(v) for v in data["hp_scores"]),
            hd_scores=tuple(float(v) for v in data["hd_scores"]),
            bandwidth=float(data["bandwidth"]),
            logistic_coef=float(data["logistic_coef"]),
            logistic_intercept=float(data["logistic_intercept"]),
            prior_log_odds=float(data["prior_log_odds"]),
            feature_mean=tuple(float(v) for v in data["feature_mean"]),
            feature_cov_inv=tuple(
                tuple(float(v) for v in row) for row in data["feature_cov_inv"]
            ),
            mahalanobis_threshold=float(data["mahalanobis_threshold"]),
            feature_dim=int(data["feature_dim"]),
        )


def fit_stream_calibration(
    name: str,
    hp_scores: Sequence[float],
    hd_scores: Sequence[float],
    features: Sequence[Sequence[float]],
) -> StreamCalibration:
    """Fit one stream's densities and validated-domain gate on the cal split.

    The cal split is never trained on. It exists solely for this function.
    """
    hp = np.asarray(list(hp_scores), dtype=float)
    hd = np.asarray(list(hd_scores), dtype=float)
    if hp.size == 0 or hd.size == 0:
        raise CalibrationError(
            f"stream {name!r}: both propositions need at least one calibration score "
            f"(Hp={hp.size}, Hd={hd.size})"
        )

    mean, cov_inv, threshold, dim = _fit_mahalanobis(features)
    use_kde = min(hp.size, hd.size) >= MIN_KDE_SAMPLES_PER_CLASS

    if use_kde:
        bandwidth = silverman_bandwidth(np.concatenate([hp, hd]))
        coef = intercept = 0.0
    else:
        bandwidth = 0.0
        coef, intercept = _fit_logistic(hp, hd)

    prior_log_odds = math.log(max(hd.size, 1) / max(hp.size, 1))

    return StreamCalibration(
        name=name,
        method="kde" if use_kde else "logistic",
        hp_scores=tuple(float(v) for v in hp),
        hd_scores=tuple(float(v) for v in hd),
        bandwidth=float(bandwidth),
        logistic_coef=float(coef),
        logistic_intercept=float(intercept),
        prior_log_odds=float(prior_log_odds),
        feature_mean=mean,
        feature_cov_inv=cov_inv,
        mahalanobis_threshold=threshold,
        feature_dim=dim,
    )


def log10_lr(cal: StreamCalibration, score: float) -> float:
    """log10 of f(score | Hd) / f(score | Hp), clipped to the reportable range."""
    if cal.method == "kde":
        hp = np.asarray(cal.hp_scores, dtype=float)
        hd = np.asarray(cal.hd_scores, dtype=float)
        f_hp = max(_kde_density(score, hp, cal.bandwidth), _DENSITY_FLOOR)
        f_hd = max(_kde_density(score, hd, cal.bandwidth), _DENSITY_FLOOR)
        value = math.log10(f_hd / f_hp)
    else:
        # Logistic gives a posterior. Subtracting the fitted prior log-odds
        # recovers the likelihood ratio, so the calibration corpus's own class
        # balance does not leak into the reported LR.
        logit = cal.logistic_coef * float(score) + cal.logistic_intercept
        value = (logit - cal.prior_log_odds) / _LN10
    return float(max(-LR_CLIP, min(LR_CLIP, value)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_density.py -v`
Expected: PASS, 10 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/forensic_lr.py tests/test_forensic_lr_density.py
git commit -m "feat(lr): KDE and logistic density fitting with prior removal"
```

---

### Task 3: Validated-domain gate and per-stream evaluation

**Files:**
- Modify: `peri/core/forensic_lr.py` (append)
- Test: `tests/test_forensic_lr_stream.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces:
  - `class StreamObservation` — frozen dataclass:
    `name: str`, `score: float`, `feature: tuple[float, ...]`,
    `stress_scores: tuple[float, ...]`, `weight: float = 1.0`
  - `class StreamResult` — frozen dataclass:
    `name: str`, `baseline_log10lr: float`, `stress_log10lrs: tuple[float, ...]`,
    `median_log10lr: float`, `iqr: float`, `mahalanobis: float`,
    `mahalanobis_threshold: float`, `in_domain: bool`, `weight: float`,
    `usable: bool`, `exclusion_reason: str | None`; plus `to_dict() -> dict`
  - `mahalanobis_distance(cal: StreamCalibration, feature: Sequence[float]) -> float`
    — returns the **distance** (square root of the squared form)
  - `evaluate_stream(cal: StreamCalibration, obs: StreamObservation) -> StreamResult`

**Exclusion precedence (checked in this order, first hit wins):**
1. `out-of-validated-domain` — the gate is asked first because an out-of-domain
   exhibit's stability is not meaningful.
2. `sign-unstable-under-degradation` — the stress LRs straddle zero **and** at least
   one of them reaches `LOG10LR_DECISION_THRESHOLD` in magnitude. A stream that
   wobbles between −0.2 and +0.2 is weak, not sign-unstable.
3. `unstable-under-degradation` — stress LR IQR exceeds `STABILITY_IQR_MAX`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_forensic_lr_stream.py`:

```python
import numpy as np
import pytest

from peri.core import forensic_lr as flr


def build_cal(dim=3, seed=5, n=120):
    rng = np.random.default_rng(seed)
    hp = rng.normal(0.0, 1.0, n).tolist()
    hd = rng.normal(3.0, 1.0, n).tolist()
    features = rng.normal(0.0, 1.0, (2 * n, dim)).tolist()
    return flr.fit_stream_calibration("stream", hp, hd, features)


def obs(score, feature=(0.0, 0.0, 0.0), stress=None, weight=1.0):
    return flr.StreamObservation(
        name="stream",
        score=score,
        feature=tuple(feature),
        stress_scores=tuple(stress if stress is not None else (score,)),
        weight=weight,
    )


def test_in_domain_feature_passes_the_gate():
    cal = build_cal()
    result = flr.evaluate_stream(cal, obs(3.2))
    assert result.in_domain is True
    assert result.usable is True
    assert result.exclusion_reason is None
    assert result.median_log10lr > 1.0


def test_far_feature_is_excluded_as_out_of_domain():
    cal = build_cal()
    result = flr.evaluate_stream(cal, obs(3.2, feature=(60.0, -60.0, 60.0)))
    assert result.in_domain is False
    assert result.usable is False
    assert result.exclusion_reason == "out-of-validated-domain"


def test_sign_flipping_stress_scores_are_sign_unstable():
    cal = build_cal()
    result = flr.evaluate_stream(cal, obs(3.2, stress=(4.0, -1.5, 3.5, -1.0)))
    assert result.usable is False
    assert result.exclusion_reason == "sign-unstable-under-degradation"


def test_wide_but_same_signed_stress_is_merely_unstable():
    cal = build_cal()
    # All positive LRs, but spread far wider than STABILITY_IQR_MAX.
    result = flr.evaluate_stream(cal, obs(3.2, stress=(1.2, 3.0, 5.0, 6.0)))
    assert result.usable is False
    assert result.exclusion_reason == "unstable-under-degradation"


def test_tiny_wobble_around_zero_is_not_sign_instability():
    cal = build_cal()
    result = flr.evaluate_stream(cal, obs(1.5, stress=(1.45, 1.5, 1.55, 1.5)))
    assert result.exclusion_reason is None


def test_out_of_domain_takes_precedence_over_instability():
    cal = build_cal()
    result = flr.evaluate_stream(
        cal, obs(3.2, feature=(60.0, -60.0, 60.0), stress=(4.0, -1.5, 3.5, -1.0))
    )
    assert result.exclusion_reason == "out-of-validated-domain"


def test_median_is_taken_over_stress_replicas_not_the_baseline():
    cal = build_cal()
    result = flr.evaluate_stream(cal, obs(3.2, stress=(3.0, 3.1, 3.2)))
    expected = float(
        np.median([flr.log10_lr(cal, s) for s in (3.0, 3.1, 3.2)])
    )
    assert result.median_log10lr == pytest.approx(expected)


def test_empty_stress_falls_back_to_the_baseline_score():
    cal = build_cal()
    result = flr.evaluate_stream(
        cal, flr.StreamObservation("stream", 3.2, (0.0, 0.0, 0.0), (), 1.0)
    )
    assert result.stress_log10lrs == (result.baseline_log10lr,)
    assert result.iqr == 0.0


def test_wrong_feature_dimension_is_rejected():
    cal = build_cal(dim=3)
    with pytest.raises(ValueError):
        flr.evaluate_stream(cal, obs(3.2, feature=(0.0, 0.0)))


def test_result_dict_only_emits_known_reason_codes():
    cal = build_cal()
    for observation in (obs(3.2), obs(3.2, feature=(60.0, -60.0, 60.0))):
        payload = flr.evaluate_stream(cal, observation).to_dict()
        reason = payload["exclusion_reason"]
        assert reason is None or reason in flr.REASON_CODES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_stream.py -v`
Expected: FAIL — `AttributeError: module 'peri.core.forensic_lr' has no attribute 'StreamObservation'`

- [ ] **Step 3: Write the implementation**

Append to `peri/core/forensic_lr.py`:

```python
@dataclass(frozen=True)
class StreamObservation:
    """One stream's output for one exhibit, plus its degradation replicas.

    `stress_scores` are the same stream re-scored under mild degradations. They
    exist so that a conclusion resting on a knife-edge can be detected and
    excluded rather than reported.
    """

    name: str
    score: float
    feature: tuple[float, ...]
    stress_scores: tuple[float, ...] = ()
    weight: float = 1.0


@dataclass(frozen=True)
class StreamResult:
    name: str
    baseline_log10lr: float
    stress_log10lrs: tuple[float, ...]
    median_log10lr: float
    iqr: float
    mahalanobis: float
    mahalanobis_threshold: float
    in_domain: bool
    weight: float
    usable: bool
    exclusion_reason: str | None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "baseline_log10lr": q(self.baseline_log10lr),
            "stress_log10lrs": [q(v) for v in self.stress_log10lrs],
            "median_log10lr": q(self.median_log10lr),
            "iqr": q(self.iqr),
            "mahalanobis": q(self.mahalanobis),
            "mahalanobis_threshold": q(self.mahalanobis_threshold),
            "in_domain": self.in_domain,
            "weight": q(self.weight),
            "usable": self.usable,
            "exclusion_reason": self.exclusion_reason,
        }


def mahalanobis_distance(cal: StreamCalibration, feature: Sequence[float]) -> float:
    """Distance from the calibration feature population, in standard deviations."""
    vector = np.asarray(list(feature), dtype=float)
    if vector.size != cal.feature_dim:
        raise ValueError(
            f"stream {cal.name!r}: feature has dimension {vector.size}, "
            f"calibration expects {cal.feature_dim}"
        )
    centred = vector - np.asarray(cal.feature_mean, dtype=float)
    cov_inv = np.asarray(cal.feature_cov_inv, dtype=float)
    md2 = float(centred @ cov_inv @ centred)
    return float(math.sqrt(max(md2, 0.0)))


def evaluate_stream(cal: StreamCalibration, obs: StreamObservation) -> StreamResult:
    """Score one stream, gate it on the validated domain, and test its stability."""
    baseline = log10_lr(cal, obs.score)
    stress_scores = obs.stress_scores or (obs.score,)
    stress = tuple(log10_lr(cal, s) for s in stress_scores)
    arr = np.asarray(stress, dtype=float)
    median = float(np.median(arr))
    iqr = float(np.subtract(*np.percentile(arr, [75, 25]))) if arr.size > 1 else 0.0

    distance = mahalanobis_distance(cal, obs.feature)
    in_domain = distance <= math.sqrt(cal.mahalanobis_threshold)

    reason: str | None = None
    if not in_domain:
        reason = REASON_OUT_OF_DOMAIN
    else:
        straddles_zero = float(arr.min()) < 0.0 < float(arr.max())
        meaningful = float(np.abs(arr).max()) >= LOG10LR_DECISION_THRESHOLD
        if straddles_zero and meaningful:
            reason = REASON_SIGN_UNSTABLE
        elif iqr > STABILITY_IQR_MAX:
            reason = REASON_UNSTABLE

    return StreamResult(
        name=obs.name,
        baseline_log10lr=baseline,
        stress_log10lrs=stress,
        median_log10lr=median,
        iqr=iqr,
        mahalanobis=distance,
        mahalanobis_threshold=float(math.sqrt(cal.mahalanobis_threshold)),
        in_domain=in_domain,
        weight=float(obs.weight),
        usable=reason is None,
        exclusion_reason=reason,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_stream.py -v`
Expected: PASS, 10 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/forensic_lr.py tests/test_forensic_lr_stream.py
git commit -m "feat(lr): Mahalanobis domain gate and per-stream stability evaluation"
```

---

### Task 4: Fusion and the three-way decision gate

**Files:**
- Modify: `peri/core/forensic_lr.py` (append)
- Test: `tests/test_forensic_lr_decision.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `class Decision` — frozen dataclass:
    `outcome: str`, `log10lr_total: float`, `verbal: str`, `sentence: str`,
    `reason_codes: tuple[str, ...]`, `streams: tuple[StreamResult, ...]`,
    `usable_stream_names: tuple[str, ...]`, `shrinkage: float`;
    plus `to_dict() -> dict`
  - `fuse_and_decide(results: Sequence[StreamResult]) -> Decision`

**Decision order (CLAUDE.md section 5, evaluated top to bottom):**
1. no usable stream → `INCONCLUSIVE` + `no-usable-stream` + each stream's own code
2. two usable streams with `abs(median LR) > 1` in opposing directions →
   `INCONCLUSIVE` + `cross-stream-contradiction`
3. `abs(total) < 1.0` → `INCONCLUSIVE` + `evidence-strength-below-reporting-threshold`
4. `total > 0` → `MANIPULATION INDICATED`
5. `total < 0` → `AUTHENTICITY SUPPORTED`

- [ ] **Step 1: Write the failing test**

Create `tests/test_forensic_lr_decision.py`:

```python
import pytest

from peri.core import forensic_lr as flr


def result(name, median, usable=True, reason=None, weight=1.0):
    return flr.StreamResult(
        name=name,
        baseline_log10lr=median,
        stress_log10lrs=(median,),
        median_log10lr=median,
        iqr=0.0,
        mahalanobis=1.0,
        mahalanobis_threshold=3.0,
        in_domain=usable or reason != "out-of-validated-domain",
        weight=weight,
        usable=usable,
        exclusion_reason=reason,
    )


def test_strong_agreeing_streams_indicate_manipulation():
    decision = flr.fuse_and_decide([result("a", 3.0), result("b", 2.6)])
    assert decision.outcome == "MANIPULATION INDICATED"
    assert decision.log10lr_total == pytest.approx(0.5 * (3.0 + 2.6))
    assert decision.reason_codes == ()


def test_strong_agreeing_negative_streams_support_authenticity():
    decision = flr.fuse_and_decide([result("a", -3.0), result("b", -2.6)])
    assert decision.outcome == "AUTHENTICITY SUPPORTED"
    assert decision.log10lr_total < 0


def test_no_usable_stream_abstains_and_keeps_each_reason():
    decision = flr.fuse_and_decide(
        [
            result("a", 3.0, usable=False, reason="out-of-validated-domain"),
            result("b", 2.0, usable=False, reason="unstable-under-degradation"),
        ]
    )
    assert decision.outcome == "INCONCLUSIVE"
    assert "no-usable-stream" in decision.reason_codes
    assert "out-of-validated-domain" in decision.reason_codes
    assert "unstable-under-degradation" in decision.reason_codes
    assert decision.log10lr_total == 0.0


def test_opposing_strong_streams_abstain_on_contradiction():
    decision = flr.fuse_and_decide([result("a", 3.0), result("b", -2.5)])
    assert decision.outcome == "INCONCLUSIVE"
    assert decision.reason_codes == ("cross-stream-contradiction",)


def test_opposing_but_weak_streams_do_not_trigger_contradiction():
    decision = flr.fuse_and_decide([result("a", 0.4), result("b", -0.3)])
    assert decision.outcome == "INCONCLUSIVE"
    assert decision.reason_codes == ("evidence-strength-below-reporting-threshold",)


def test_total_below_threshold_abstains():
    decision = flr.fuse_and_decide([result("a", 1.2)])
    # 0.5 * 1.2 = 0.6, below the reporting threshold of 1.0
    assert decision.log10lr_total == pytest.approx(0.6)
    assert decision.outcome == "INCONCLUSIVE"
    assert decision.reason_codes == ("evidence-strength-below-reporting-threshold",)


def test_shrinkage_is_applied_and_reported():
    decision = flr.fuse_and_decide([result("a", 4.0)])
    assert decision.shrinkage == 0.5
    assert decision.log10lr_total == pytest.approx(2.0)
    assert decision.outcome == "MANIPULATION INDICATED"


def test_weights_scale_contributions():
    decision = flr.fuse_and_decide([result("a", 4.0, weight=0.5), result("b", 2.0)])
    assert decision.log10lr_total == pytest.approx(0.5 * (0.5 * 4.0 + 1.0 * 2.0))


def test_total_is_clipped_to_the_reportable_bound():
    streams = [result(f"s{i}", 6.0) for i in range(8)]
    decision = flr.fuse_and_decide(streams)
    assert decision.log10lr_total == pytest.approx(flr.LR_CLIP)


def test_excluded_streams_do_not_contribute_to_the_total():
    decision = flr.fuse_and_decide(
        [result("a", 3.0), result("b", -5.0, usable=False, reason="unstable-under-degradation")]
    )
    assert decision.log10lr_total == pytest.approx(1.5)
    assert decision.usable_stream_names == ("a",)


def test_every_emitted_reason_code_is_in_the_frozen_vocabulary():
    cases = [
        [result("a", 3.0)],
        [result("a", 0.1)],
        [result("a", 3.0), result("b", -2.5)],
        [result("a", 1.0, usable=False, reason="out-of-validated-domain")],
    ]
    for streams in cases:
        for code in flr.fuse_and_decide(streams).reason_codes:
            assert code in flr.REASON_CODES


def test_outcome_is_always_one_of_the_three():
    cases = [
        [result("a", 3.0)],
        [result("a", -3.0)],
        [result("a", 0.1)],
        [],
    ]
    for streams in cases:
        assert flr.fuse_and_decide(streams).outcome in flr.OUTCOMES


def test_decision_dict_is_json_ready():
    from peri.core.canon import canonical_json

    decision = flr.fuse_and_decide([result("a", 3.0), result("b", 2.6)])
    text = canonical_json(decision.to_dict())
    assert "MANIPULATION INDICATED" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_decision.py -v`
Expected: FAIL — `AttributeError: module 'peri.core.forensic_lr' has no attribute 'fuse_and_decide'`

- [ ] **Step 3: Write the implementation**

Append to `peri/core/forensic_lr.py`:

```python
@dataclass(frozen=True)
class Decision:
    outcome: str
    log10lr_total: float
    verbal: str
    sentence: str
    reason_codes: tuple[str, ...]
    streams: tuple[StreamResult, ...]
    usable_stream_names: tuple[str, ...]
    shrinkage: float

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "log10lr_total": q(self.log10lr_total),
            "verbal": self.verbal,
            "sentence": self.sentence,
            "reason_codes": list(self.reason_codes),
            "streams": [s.to_dict() for s in self.streams],
            "usable_stream_names": list(self.usable_stream_names),
            "dependence_shrinkage": q(self.shrinkage),
            "propositions": {"Hp": HP_TEXT, "Hd": HD_TEXT},
        }


def fuse_and_decide(results: Sequence[StreamResult]) -> Decision:
    """Fuse usable streams under a stated dependence discount and apply the gate.

    The streams are correlated: they observe the same pixels through different
    operators. Summing their log LRs as if they were independent would overstate
    the evidence, so the sum carries an explicit discount factor. The discount is
    a stated conservatism, not an estimate of the true dependence structure, and
    the report says exactly that.
    """
    results = tuple(results)
    usable = tuple(r for r in results if r.usable)

    if not usable:
        codes = [REASON_NO_USABLE_STREAM]
        for r in results:
            if r.exclusion_reason and r.exclusion_reason not in codes:
                codes.append(r.exclusion_reason)
        return Decision(
            outcome=OUTCOME_INCONCLUSIVE,
            log10lr_total=0.0,
            verbal=enfsi_verbal(0.0),
            sentence=enfsi_sentence(0.0),
            reason_codes=tuple(codes),
            streams=results,
            usable_stream_names=(),
            shrinkage=DEPENDENCE_SHRINKAGE,
        )

    total = DEPENDENCE_SHRINKAGE * sum(r.weight * r.median_log10lr for r in usable)
    total = float(max(-LR_CLIP, min(LR_CLIP, total)))

    strong_positive = any(r.median_log10lr > LOG10LR_DECISION_THRESHOLD for r in usable)
    strong_negative = any(r.median_log10lr < -LOG10LR_DECISION_THRESHOLD for r in usable)
    names = tuple(r.name for r in usable)

    if strong_positive and strong_negative:
        return Decision(
            outcome=OUTCOME_INCONCLUSIVE,
            log10lr_total=total,
            verbal=enfsi_verbal(total),
            sentence=enfsi_sentence(total),
            reason_codes=(REASON_CONTRADICTION,),
            streams=results,
            usable_stream_names=names,
            shrinkage=DEPENDENCE_SHRINKAGE,
        )

    if abs(total) < LOG10LR_DECISION_THRESHOLD:
        outcome, codes = OUTCOME_INCONCLUSIVE, (REASON_BELOW_THRESHOLD,)
    elif total > 0:
        outcome, codes = OUTCOME_MANIPULATION, ()
    else:
        outcome, codes = OUTCOME_AUTHENTIC, ()

    return Decision(
        outcome=outcome,
        log10lr_total=total,
        verbal=enfsi_verbal(total),
        sentence=enfsi_sentence(total),
        reason_codes=codes,
        streams=results,
        usable_stream_names=names,
        shrinkage=DEPENDENCE_SHRINKAGE,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_decision.py -v`
Expected: PASS, 13 passed.

- [ ] **Step 5: Commit**

```bash
git add peri/core/forensic_lr.py tests/test_forensic_lr_decision.py
git commit -m "feat(lr): shrunk fusion and the three-way decision gate"
```

---

### Task 5: The three synthetic acceptance assertions

This is CLAUDE.md section 3, CPU-track step 1: *"Three assertions pass: clear-manipulation,
unstable-abstains, out-of-domain-abstains."* It is an end-to-end exercise of the whole
module against synthetic scores, with no model in existence.

**Files:**
- Create: `tools/selftest_lr.py`
- Test: `tests/test_forensic_lr_acceptance.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `run_selftest() -> dict` returning
  `{"clear_manipulation": Decision, "unstable": Decision, "out_of_domain": Decision}`,
  and a `__main__` that prints a three-line human-readable result. Phase 10's
  acceptance script calls this module.

- [ ] **Step 1: Write the failing test**

Create `tests/test_forensic_lr_acceptance.py`:

```python
from peri.core import forensic_lr as flr
from tools.selftest_lr import run_selftest


def test_assertion_one_clear_manipulation_is_indicated():
    decision = run_selftest()["clear_manipulation"]
    assert decision.outcome == "MANIPULATION INDICATED"
    assert decision.log10lr_total >= flr.LOG10LR_DECISION_THRESHOLD
    assert decision.reason_codes == ()
    assert flr.HD_TEXT in decision.sentence


def test_assertion_two_unstable_stream_abstains():
    decision = run_selftest()["unstable"]
    assert decision.outcome == "INCONCLUSIVE"
    assert "no-usable-stream" in decision.reason_codes
    assert any(
        code in decision.reason_codes
        for code in ("unstable-under-degradation", "sign-unstable-under-degradation")
    )


def test_assertion_three_out_of_domain_exhibit_abstains():
    decision = run_selftest()["out_of_domain"]
    assert decision.outcome == "INCONCLUSIVE"
    assert "out-of-validated-domain" in decision.reason_codes


def test_selftest_is_deterministic_across_runs():
    first = {k: v.to_dict() for k, v in run_selftest().items()}
    second = {k: v.to_dict() for k, v in run_selftest().items()}
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_acceptance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.selftest_lr'`

- [ ] **Step 3: Write the implementation**

Create `tools/selftest_lr.py`:

```python
"""L4 self-test: exercise the likelihood-ratio layer with no model in existence.

Three synthetic exhibits, three required behaviours:

  1. clear manipulation  -> MANIPULATION INDICATED
  2. unstable stream     -> INCONCLUSIVE, stream excluded for instability
  3. out-of-domain       -> INCONCLUSIVE, stream excluded by the domain gate

If any of these regress, the abstention path is broken, and the abstention path
is the product.
"""

from __future__ import annotations

import numpy as np

from peri.core import forensic_lr as flr

_CAL_SEED = 20260820
_N_PER_CLASS = 200
_FEATURE_DIM = 4


def _build_calibrations() -> dict[str, flr.StreamCalibration]:
    rng = np.random.default_rng(_CAL_SEED)
    cals: dict[str, flr.StreamCalibration] = {}
    for name, separation in (("videoprint", 3.0), ("temporal", 2.5), ("provenance", 2.0)):
        hp = rng.normal(0.0, 1.0, _N_PER_CLASS).tolist()
        hd = rng.normal(separation, 1.0, _N_PER_CLASS).tolist()
        features = rng.normal(0.0, 1.0, (2 * _N_PER_CLASS, _FEATURE_DIM)).tolist()
        cals[name] = flr.fit_stream_calibration(name, hp, hd, features)
    return cals


def run_selftest() -> dict[str, flr.Decision]:
    cals = _build_calibrations()
    inside = (0.1, -0.2, 0.05, 0.15)
    outside = (25.0, -30.0, 28.0, -26.0)

    clear = flr.fuse_and_decide(
        [
            flr.evaluate_stream(
                cals["videoprint"],
                flr.StreamObservation("videoprint", 4.2, inside, (4.1, 4.2, 4.3, 4.15)),
            ),
            flr.evaluate_stream(
                cals["temporal"],
                flr.StreamObservation("temporal", 3.6, inside, (3.5, 3.6, 3.7, 3.55)),
            ),
        ]
    )

    unstable = flr.fuse_and_decide(
        [
            flr.evaluate_stream(
                cals["videoprint"],
                flr.StreamObservation("videoprint", 3.5, inside, (4.5, -1.0, 3.9, -0.8)),
            ),
            flr.evaluate_stream(
                cals["temporal"],
                flr.StreamObservation("temporal", 3.0, inside, (5.0, 0.5, 4.0, 1.0)),
            ),
        ]
    )

    out_of_domain = flr.fuse_and_decide(
        [
            flr.evaluate_stream(
                cals["videoprint"],
                flr.StreamObservation("videoprint", 4.2, outside, (4.1, 4.2, 4.3, 4.15)),
            ),
            flr.evaluate_stream(
                cals["temporal"],
                flr.StreamObservation("temporal", 3.6, outside, (3.5, 3.6, 3.7, 3.55)),
            ),
        ]
    )

    return {
        "clear_manipulation": clear,
        "unstable": unstable,
        "out_of_domain": out_of_domain,
    }


def main() -> int:
    expectations = {
        "clear_manipulation": flr.OUTCOME_MANIPULATION,
        "unstable": flr.OUTCOME_INCONCLUSIVE,
        "out_of_domain": flr.OUTCOME_INCONCLUSIVE,
    }
    failures = 0
    for key, decision in run_selftest().items():
        expected = expectations[key]
        status = "OK " if decision.outcome == expected else "FAIL"
        failures += decision.outcome != expected
        codes = ",".join(decision.reason_codes) or "-"
        print(
            f"{status} {key:20s} log10LR={decision.log10lr_total:+.3f} "
            f"outcome={decision.outcome:22s} reasons={codes}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_acceptance.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 5: Run the self-test as a human would**

Run: `.venv/Scripts/python.exe -m tools.selftest_lr`
Expected, three lines all starting `OK `:

```
OK  clear_manipulation   log10LR=+... outcome=MANIPULATION INDICATED reasons=-
OK  unstable             log10LR=+0.000 outcome=INCONCLUSIVE           reasons=no-usable-stream,...
OK  out_of_domain        log10LR=+0.000 outcome=INCONCLUSIVE           reasons=no-usable-stream,out-of-validated-domain
```

Exit code must be 0 (`echo $?`).

- [ ] **Step 6: Commit**

```bash
git add tools/selftest_lr.py tests/test_forensic_lr_acceptance.py
git commit -m "test(lr): three synthetic acceptance assertions for the LR layer"
```

---

## Phase 1 acceptance test

```bash
.venv/Scripts/python.exe -m pytest tests/test_forensic_lr_scale.py tests/test_forensic_lr_density.py tests/test_forensic_lr_stream.py tests/test_forensic_lr_decision.py tests/test_forensic_lr_acceptance.py -q
.venv/Scripts/python.exe -m tools.selftest_lr; echo "exit=$?"
```

**Pass criteria, all five:**
1. 53 tests pass, 0 fail.
2. `tools.selftest_lr` prints three `OK ` lines and exits 0.
3. `grep -rniE "\b(court-admissible|legally valid|legally admissible|certified evidence|meets Section 63|proves|guaranteed authentic)\b" peri/core/forensic_lr.py tools/selftest_lr.py` returns nothing.
4. `grep -rn "independen" peri/core/forensic_lr.py` returns only the docstring in
   `fuse_and_decide` that explains why we do **not** assume independence.
5. `peri/core/forensic_lr.py` imports no torch, no file I/O, and no `random`.

**Phase 1 is green when all five hold.** Phase 4 (fragility) and Phase 6
(calibration) may then start.
