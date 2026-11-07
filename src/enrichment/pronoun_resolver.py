"""Extract explicit pronoun evidence from caller-verified academic profile URLs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

@dataclass(frozen=True)
class PronounEvidence:
    label: str
    count: int
    source_url: str
    margin: int = 0
    explicit: bool = False


class BioPronounExtractor:
    """Extract a leading explicit pronoun signal from supplied verified profiles.

    URLs must be verified by the caller. This resolver fetches only the supplied
    pages and does not discover or follow links to additional profiles.
    """

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds

    def resolve(
        self,
        *,
        faculty_bio_url: str | None,
        personal_homepage_url: str | None,
        target_pronouns: Mapping[str, Sequence[str]],
        csrankings_profile_url: str | None = None,
    ) -> PronounEvidence:
        first_evidence = PronounEvidence("unknown", 0, "", 0)
        for url in (
            faculty_bio_url,
            csrankings_profile_url,
            personal_homepage_url,
        ):
            if url is None or not url.strip():
                continue
            parsed_url = urlsplit(url)
            if (
                parsed_url.scheme not in {"http", "https"}
                or not parsed_url.hostname
                or parsed_url.username is not None
                or parsed_url.password is not None
            ):
                raise ValueError("profile URLs must be absolute HTTP(S) URLs")

            response = self._session.get(url, timeout=self._timeout_seconds)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "noscript"]):
                element.decompose()
            text = soup.get_text(" ", strip=True)
            explicit_patterns = {
                "female": r"(?<!\w)she\s*/\s*her(?:\s*/\s*hers)?(?!\w)",
                "male": r"(?<!\w)he\s*/\s*him(?:\s*/\s*his)?(?!\w)",
                "non_binary": (
                    r"(?<!\w)they\s*/\s*them(?:\s*/\s*theirs?)?(?!\w)"
                ),
            }
            explicit_counts = {
                label: len(re.findall(pattern, text, flags=re.IGNORECASE))
                for label, pattern in explicit_patterns.items()
            }
            explicit_labels = [
                label for label, count in explicit_counts.items() if count
            ]
            if len(explicit_labels) == 1:
                label = explicit_labels[0]
                count = explicit_counts[label]
                return PronounEvidence(label, count, url, count, True)
            if explicit_labels:
                first_evidence = PronounEvidence(
                    "unknown",
                    max(explicit_counts.values()),
                    url,
                    0,
                    True,
                )
                continue
            counts = {
                label: sum(
                    len(
                        re.findall(
                            rf"\b{re.escape(pronoun)}\b",
                            text,
                            flags=re.IGNORECASE,
                        )
                    )
                    for pronoun in pronouns
                )
                for label, pronouns in target_pronouns.items()
            }
            if not counts:
                continue
            ordered_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            best_label, best_count = ordered_counts[0]
            second_count = ordered_counts[1][1] if len(ordered_counts) > 1 else 0
            margin = best_count - second_count
            if best_count:
                if first_evidence.count == 0:
                    first_evidence = PronounEvidence("unknown", best_count, url, margin)
                label = (
                    best_label
                    if margin >= 2 and best_label in {"female", "male", "non_binary"}
                    else "unknown"
                )
                if label != "unknown":
                    return PronounEvidence(label, best_count, url, margin)
        return first_evidence


PronounProfileResolver = BioPronounExtractor
