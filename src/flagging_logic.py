"""CS flagging business rules, independent of any model or UI.

A review is escalated to Customer Service when ALL of these hold:

    sentiment == "Negative"
    AND confidence > FLAG_CONFIDENCE_THRESHOLD (default 0.80)
    AND the text contains at least one emergency keyword
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class FlagResult:
    is_urgent: bool
    label: str
    matched_keywords: tuple[str, ...]
    reason: str


class CSFlagger:
    """Configurable flagger: swap keywords or threshold without touching models.

    Keywords match at the start of a word, so ``refund`` also catches
    ``refunded``/``refunds`` and ``scam`` catches ``scammed``, while ``fake``
    does not fire inside unrelated words such as ``bifake``.
    """

    def __init__(
        self,
        keywords: Iterable[str] = config.EMERGENCY_KEYWORDS,
        threshold: float = config.FLAG_CONFIDENCE_THRESHOLD,
        negative_label: str = config.ID2LABEL[0],
    ):
        self.keywords = tuple(dict.fromkeys(k.strip().lower() for k in keywords if k.strip()))
        if not self.keywords:
            raise ValueError("At least one emergency keyword is required.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a probability in [0, 1].")
        self.threshold = threshold
        self.negative_label = negative_label
        # Longest first so multi-word phrases win over their prefixes.
        alternatives = "|".join(re.escape(k) for k in sorted(self.keywords, key=len, reverse=True))
        self._pattern = re.compile(rf"\b({alternatives})\w*", re.IGNORECASE)

    def find_keywords(self, text: str) -> tuple[str, ...]:
        """Distinct emergency keywords found in ``text``, in order of appearance."""
        found = (match.group(1).lower() for match in self._pattern.finditer(text or ""))
        return tuple(dict.fromkeys(found))

    def contains_emergency_keyword(self, text: str) -> bool:
        return bool(self.find_keywords(text))

    def flag(self, sentiment: str, confidence: float, text: str) -> FlagResult:
        matched = self.find_keywords(text)
        is_negative = sentiment == self.negative_label
        is_confident = confidence > self.threshold

        if is_negative and is_confident and matched:
            reason = (
                f"Negative with {confidence:.0%} confidence (> {self.threshold:.0%}) "
                f"and emergency keyword(s): {', '.join(matched)}."
            )
            return FlagResult(True, config.URGENT_FLAG, matched, reason)

        if not is_negative:
            reason = "Sentiment is not negative."
        elif not is_confident:
            reason = f"Confidence {confidence:.0%} is not above {self.threshold:.0%}."
        else:
            reason = "No emergency keyword found."
        return FlagResult(False, config.NORMAL_FLAG, matched, reason)


_default_flagger = CSFlagger()


def contains_emergency_keyword(text: str) -> bool:
    """Module-level shortcut using the default keyword list."""
    return _default_flagger.contains_emergency_keyword(text)


def flag_review(sentiment: str, confidence: float, text: str) -> FlagResult:
    """Module-level shortcut using the default keywords and threshold."""
    return _default_flagger.flag(sentiment, confidence, text)
