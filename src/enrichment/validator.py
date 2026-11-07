"""Conservative validation gates for demographic and geographic metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from src.enrichment.countries import is_iso_country_code
from src.enrichment.pronoun_resolver import (
    BioPronounExtractor,
    PronounEvidence,
)
from src.inference.probabilistic_tagger import TagResult

class GenderInferenceProtocol(Protocol):
    def infer(self, given_name: str) -> TagResult: ...


class AuthorValidator:
    """Prefer explicit profile pronouns; fail closed below 95% confidence."""

    def __init__(
        self,
        *,
        gender_engine: GenderInferenceProtocol | None = None,
        bio_pronoun_extractor: BioPronounExtractor | None = None,
        gender_min_confidence: float = 0.95,
        gender_min_sample_count: int = 100,
        require_country_validation: bool = True,
    ) -> None:
        if (
            isinstance(gender_min_confidence, bool)
            or not isinstance(gender_min_confidence, (int, float))
            or not math.isfinite(gender_min_confidence)
            or not 0.95 <= gender_min_confidence <= 1
        ):
            raise ValueError("gender_min_confidence must be between 0.95 and 1")
        if (
            not isinstance(gender_min_sample_count, int)
            or isinstance(gender_min_sample_count, bool)
            or gender_min_sample_count < 0
        ):
            raise ValueError("gender_min_sample_count must be non-negative")
        if not isinstance(require_country_validation, bool):
            raise TypeError("require_country_validation must be a boolean")
        self._gender_engine = gender_engine
        self._bio_pronoun_extractor = bio_pronoun_extractor
        self._minimum_confidence = float(gender_min_confidence)
        self._minimum_sample_count = gender_min_sample_count
        self._require_country_validation = require_country_validation

    @staticmethod
    def is_valid_country_code(country_code: object) -> bool:
        return is_iso_country_code(country_code)

    def validate_author(
        self,
        given_name: str,
        *,
        country_code: str | None,
        target_pronouns: Mapping[str, Sequence[str]],
        faculty_bio_url: str | None = None,
        personal_homepage_url: str | None = None,
        csrankings_profile_url: str | None = None,
        inference_result: TagResult | None = None,
    ) -> dict[str, Any]:
        """Return the chosen label, tier/source, and evidence with safe fallbacks."""
        pronoun_evidence = (
            self._bio_pronoun_extractor.resolve(
                faculty_bio_url=faculty_bio_url,
                personal_homepage_url=personal_homepage_url,
                csrankings_profile_url=csrankings_profile_url,
                target_pronouns=target_pronouns,
            )
            if self._bio_pronoun_extractor is not None
            and (
                faculty_bio_url
                or personal_homepage_url
                or csrankings_profile_url
            )
            else PronounEvidence("unknown", 0, "", 0)
        )
        if pronoun_evidence.label in {"female", "male", "non_binary"} and (
            pronoun_evidence.explicit or pronoun_evidence.margin >= 2
        ):
            return {
                "inferred_gender": pronoun_evidence.label,
                "inference_source": "verified_bio_pronouns",
                "confidence": 1.0,
                "inference_count": pronoun_evidence.count,
                "validation_tier": 1,
                "pronoun_label": pronoun_evidence.label,
                "pronoun_count": pronoun_evidence.count,
                "pronoun_margin": pronoun_evidence.margin,
                "pronoun_source_url": pronoun_evidence.source_url,
            }

        if inference_result is None:
            if self._gender_engine is None:
                raise RuntimeError(
                    "gender_engine is required when no inference result is supplied"
                )
            result = self._gender_engine.infer(given_name)
        else:
            result = inference_result
        label = self.validate_gender(
            result.inferred_gender,
            result.confidence,
            sample_count=result.count,
            country_code=country_code,
        )
        return {
            "inferred_gender": label,
            "inference_source": (
                result.inference_source if label != "unknown" else "unknown"
            ),
            "confidence": result.confidence,
            "inference_count": result.count,
            "validation_tier": 2 if label != "unknown" else 3,
            "pronoun_label": pronoun_evidence.label,
            "pronoun_count": pronoun_evidence.count,
            "pronoun_margin": pronoun_evidence.margin,
            "pronoun_source_url": pronoun_evidence.source_url,
        }

    def validate_gender(
        self,
        label: str,
        confidence: float,
        *,
        sample_count: int | None = None,
        country_code: str | None = None,
    ) -> str:
        """Validate a probabilistic label; absent sample counts fail closed."""
        if label not in {"female", "male"}:
            return "unknown"
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or confidence < self._minimum_confidence
            or confidence > 1
        ):
            return "unknown"
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < self._minimum_sample_count
        ):
            return "unknown"
        if self._require_country_validation and not self.is_valid_country_code(
            country_code
        ):
            return "unknown"
        return label


class DemographicValidator(AuthorValidator):
    """Compatibility name for the strict probabilistic and country guard."""
