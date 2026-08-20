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
    rng = np.random.default_rng(11)
    hp = rng.normal(0.0, 1.0, 10).tolist()
    hd = rng.normal(3.0, 1.0, 4).tolist()
    features = rng.normal(0.0, 1.0, (14, 3)).tolist()
    cal = flr.fit_stream_calibration("imb", hp, hd, features)
    assert cal.method == "logistic"
    assert cal.prior_log_odds < 0.0  
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
