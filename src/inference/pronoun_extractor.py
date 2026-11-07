"""Count explicit third-person pronoun evidence in supplied biography text."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


def extract_pronoun_label(
    text: str, target_pronouns: Mapping[str, Sequence[str]]
) -> tuple[str, int]:
    """Return the uniquely best label and its count, or unknown on ties/no evidence."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    counts: dict[str, int] = {}
    for label, pronouns in target_pronouns.items():
        count = 0
        for pronoun in pronouns:
            pattern = rf"\b{re.escape(pronoun)}\b"
            count += len(re.findall(pattern, text, flags=re.IGNORECASE))
        counts[label] = count
    if not counts:
        return "unknown", 0
    maximum = max(counts.values())
    winners = [label for label, count in counts.items() if count == maximum]
    return (winners[0], maximum) if maximum > 0 and len(winners) == 1 else ("unknown", maximum)
