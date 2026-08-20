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
    # Scores chosen so every replica LR stays positive - the stream never changes
    # its answer, it only changes how loudly it gives it. Asserted rather than
    # assumed: with this calibration a score of 1.2 already sits below the
    # crossover and would make this sign instability instead.
    stress = (2.0, 3.0, 4.0, 5.0)
    assert all(flr.log10_lr(cal, s) > 0.0 for s in stress)
    result = flr.evaluate_stream(cal, obs(3.2, stress=stress))
    assert result.iqr > flr.STABILITY_IQR_MAX
    assert result.usable is False
    assert result.exclusion_reason == "unstable-under-degradation"


def test_crossover_scores_are_reported_as_sign_instability():
    cal = build_cal()
    # The mirror of the case above: a replica ladder that crosses the crossover
    # point while one end is strong enough to have been reported.
    result = flr.evaluate_stream(cal, obs(3.2, stress=(1.2, 3.0, 5.0, 6.0)))
    assert result.usable is False
    assert result.exclusion_reason == "sign-unstable-under-degradation"

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
