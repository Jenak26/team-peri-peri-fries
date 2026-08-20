"""L2 + L4: validated-domain gate, likelihood ratios, fusion, and the decision gate.

This module is deliberately free of I/O, model imports, and randomness. It is the
one place where a number becomes a statement about two propositions, so it is the
one place that must be testable without any model in existence.

Scores entering this module are oriented so that a higher score is more consistent
with Hd. Wrappers are responsible for that orientation; this module never re-orients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from peri.core.canon import q
from peri.core.errors import CalibrationError

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
    """Distance from the calibration population. Returns the non-squared distance."""
    if len(feature) != cal.feature_dim:
        raise ValueError(
            f"expected feature dim {cal.feature_dim}, got {len(feature)}"
        )
    x = np.asarray(feature, dtype=float) - np.asarray(cal.feature_mean, dtype=float)
    inv = np.asarray(cal.feature_cov_inv, dtype=float)
    md2 = float(np.einsum("i,ij,j->", x, inv, x))
    return float(math.sqrt(max(md2, 0.0)))


def evaluate_stream(cal: StreamCalibration, obs: StreamObservation) -> StreamResult:
    baseline_lr = log10_lr(cal, obs.score)
    stress_lrs = tuple(log10_lr(cal, s) for s in obs.stress_scores) if obs.stress_scores else (baseline_lr,)
    
    arr = np.asarray(stress_lrs)
    median_lr = float(np.median(arr))
    iqr = float(np.subtract(*np.percentile(arr, [75, 25]))) if arr.size > 1 else 0.0
    
    md = mahalanobis_distance(cal, obs.feature)
    in_domain = md <= math.sqrt(cal.mahalanobis_threshold)

    exclusion_reason = None
    if not in_domain:
        exclusion_reason = REASON_OUT_OF_DOMAIN
    else:
        # Check sign instability
        # Both positive and negative values exist AND at least one exceeds threshold
        min_lr = float(arr.min())
        max_lr = float(arr.max())
        if min_lr < 0 < max_lr and max(abs(min_lr), abs(max_lr)) >= LOG10LR_DECISION_THRESHOLD:
            exclusion_reason = REASON_SIGN_UNSTABLE
        elif iqr > STABILITY_IQR_MAX:
            exclusion_reason = REASON_UNSTABLE

    return StreamResult(
        name=obs.name,
        baseline_log10lr=baseline_lr,
        stress_log10lrs=stress_lrs,
        median_log10lr=median_lr,
        iqr=iqr,
        mahalanobis=md,
        mahalanobis_threshold=cal.mahalanobis_threshold,
        in_domain=in_domain,
        weight=obs.weight,
        usable=(exclusion_reason is None),
        exclusion_reason=exclusion_reason,
    )


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
