import pytest

from src import config
from src.flagging_logic import CSFlagger, contains_emergency_keyword, flag_review

URGENT_TEXT = "The product arrived completely broken and smells dangerous, I want a refund now!"


def test_urgent_when_negative_confident_and_keyword():
    result = flag_review("Negative", 0.92, URGENT_TEXT)
    assert result.is_urgent
    assert result.label == "🔴 URGENT: TRANSFER TO CS"
    assert result.matched_keywords == ("broken", "dangerous", "refund")


def test_normal_when_positive_even_with_keyword():
    result = flag_review("Positive", 0.99, "The refund process was simple and the support team was great.")
    assert not result.is_urgent
    assert result.label == "✅ Normal"


def test_normal_when_negative_without_keyword():
    result = flag_review("Negative", 0.95, "The product quality is disappointing. I don't really like it.")
    assert not result.is_urgent
    assert result.reason == "No emergency keyword found."


@pytest.mark.parametrize("confidence", [0.5, 0.79, 0.80])
def test_threshold_is_strictly_greater_than(confidence):
    assert not flag_review("Negative", confidence, "fake item").is_urgent


def test_keyword_matches_word_start_variants_and_case():
    assert contains_emergency_keyword("I was SCAMMED and it got Refunded late")
    assert not contains_emergency_keyword("A perfectly fine product.")
    assert not contains_emergency_keyword("unbroken seal")  # no match mid-word


def test_custom_keywords_and_threshold():
    flagger = CSFlagger(keywords=["never arrived", "rude"], threshold=0.6)
    result = flagger.flag("Negative", 0.7, "Package never arrived and the shipper was rude.")
    assert result.is_urgent
    assert result.matched_keywords == ("never arrived", "rude")


def test_invalid_configuration():
    with pytest.raises(ValueError):
        CSFlagger(keywords=[])
    with pytest.raises(ValueError):
        CSFlagger(threshold=1.5)


def test_default_keywords_come_from_config():
    assert set(CSFlagger().keywords) == set(config.EMERGENCY_KEYWORDS)
