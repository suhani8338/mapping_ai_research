"""Genderize.io client returning explicit unknowns when evidence is insufficient."""

from __future__ import annotations

from dataclasses import dataclass

import requests

GENDERIZE_URL = "https://api.genderize.io"


@dataclass(frozen=True)
class TagResult:
    inferred_gender: str
    confidence: float
    count: int
    inference_source: str = "genderize_api"


class GenderizeTagger:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        session: requests.Session | None = None,
        timeout_seconds: float = 15.0,
        confidence_threshold: float = 0.90,
        min_sample_count: int = 100,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if min_sample_count < 0:
            raise ValueError("min_sample_count must be non-negative")
        self._api_key = api_key
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._confidence_threshold = confidence_threshold
        self._min_sample_count = min_sample_count

    def infer(self, given_name: str) -> TagResult:
        if not given_name.strip():
            return TagResult("unknown", 0.0, 0)
        params = {"name": given_name}
        if self._api_key:
            params["apikey"] = self._api_key
        response = self._session.get(
            GENDERIZE_URL, params=params, timeout=self._timeout_seconds
        )
        response.raise_for_status()
        payload = response.json()
        gender = payload.get("gender")
        probability = payload.get("probability")
        count = payload.get("count")
        if (
            gender not in {"female", "male"}
            or not isinstance(probability, (int, float))
            or not isinstance(count, int)
        ):
            return TagResult("unknown", 0.0, count if isinstance(count, int) else 0)
        if probability < self._confidence_threshold or count < self._min_sample_count:
            return TagResult("unknown", float(probability), count)
        return TagResult(gender, float(probability), count)


class GenderInferenceEngine:
    """Classify names locally first, with an optional Genderize fallback."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.90,
        min_sample_count: int = 100,
        api_key: str | None = None,
        api_tagger: GenderizeTagger | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if min_sample_count < 0:
            raise ValueError("min_sample_count must be non-negative")
        import gender_guesser.detector as gender

        self._threshold = confidence_threshold
        self._detector = gender.Detector(case_sensitive=False)
        self._api_tagger = api_tagger
        if api_key and self._api_tagger is None:
            self._api_tagger = GenderizeTagger(
                api_key=api_key,
                confidence_threshold=confidence_threshold,
                min_sample_count=min_sample_count,
            )

    def infer(self, given_name: str) -> TagResult:
        if not given_name.strip():
            return TagResult("unknown", 0.0, 0, "gender_guesser")

        guess = self._detector.get_gender(given_name)
        offline_results = {
            "female": ("female", 1.0),
            "mostly_female": ("female", 0.75),
            "male": ("male", 1.0),
            "mostly_male": ("male", 0.75),
        }
        gender, confidence = offline_results.get(guess, ("unknown", 0.0))
        if confidence >= self._threshold:
            return TagResult(gender, confidence, 0, "gender_guesser")

        offline_result = TagResult("unknown", confidence, 0, "gender_guesser")
        if self._api_tagger is None:
            return offline_result
        api_result = self._api_tagger.infer(given_name)
        return TagResult(
            api_result.inferred_gender,
            api_result.confidence,
            api_result.count,
            "genderize_api",
        )
