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
