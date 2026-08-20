import pytest

from peri.core import forensic_lr as flr

def stub_result(name, usable, median, reason=None):
    return flr.StreamResult(
        name=name,
        baseline_log10lr=median,
        stress_log10lrs=(median,),
        median_log10lr=median,
        iqr=0.0,
        mahalanobis=1.0,
        mahalanobis_threshold=10.0,
        in_domain=True,
        weight=1.0,
        usable=usable,
        exclusion_reason=reason,
    )

def test_fusion_with_no_usable_streams_is_inconclusive():
    res = flr.fuse_and_decide([stub_result("a", False, 5.0, "unstable-under-degradation")])
    assert res.outcome == "INCONCLUSIVE"
    assert res.primary_reason == "no-usable-stream"

def test_fusion_with_contradicting_strong_streams_is_inconclusive():
    res = flr.fuse_and_decide([
        stub_result("a", True, 2.5),
        stub_result("b", True, -1.5)
    ])
    assert res.outcome == "INCONCLUSIVE"
    assert res.primary_reason == "cross-stream-contradiction"

def test_weak_contradiction_is_allowed():
    res = flr.fuse_and_decide([
        stub_result("a", True, 2.5),
        stub_result("b", True, -0.5)
    ])
    assert res.outcome == "MANIPULATION INDICATED"
    assert res.primary_reason is None

def test_fused_lr_is_shrunk_sum():
    res = flr.fuse_and_decide([
        stub_result("a", True, 3.0),
        stub_result("b", True, 2.0)
    ])
    assert res.total_log10lr == pytest.approx(2.5)

def test_below_threshold_total_is_inconclusive():
    res = flr.fuse_and_decide([stub_result("a", True, 1.5)])
    assert res.outcome == "INCONCLUSIVE"
    assert res.primary_reason == "evidence-strength-below-reporting-threshold"
