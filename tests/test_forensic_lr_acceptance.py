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
