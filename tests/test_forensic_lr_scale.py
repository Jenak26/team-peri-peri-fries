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
