"""Load the central YAML configuration."""

from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml


DEFAULT_ENRICHMENT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "openalex_mailto": "",
    "rate_limit_delay": 0.1,
    "academic_age_brackets": {
        "junior": [0, 3],
        "mid": [4, 7],
        "senior": [8, None],
    },
    "classification_thresholds": {
        "gender_min_confidence": 0.95,
        "gender_min_sample_count": 100,
        "require_country_validation": True,
    },
}


def _validated_enrichment_config(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("enrichment must be a YAML mapping")

    config = {**DEFAULT_ENRICHMENT_CONFIG, **value}
    configured_brackets = value.get("academic_age_brackets")
    if configured_brackets is not None and not isinstance(configured_brackets, Mapping):
        raise ValueError("enrichment.academic_age_brackets must be a mapping")
    config["academic_age_brackets"] = {
        **DEFAULT_ENRICHMENT_CONFIG["academic_age_brackets"],
        **(configured_brackets or {}),
    }
    if not isinstance(config["enabled"], bool):
        raise ValueError("enrichment.enabled must be a boolean")
    mailto = config["openalex_mailto"]
    if not isinstance(mailto, str):
        raise ValueError("enrichment.openalex_mailto must be a string")
    delay = config["rate_limit_delay"]
    if (
        isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not math.isfinite(delay)
        or delay < 0
    ):
        raise ValueError("enrichment.rate_limit_delay must be a finite non-negative number")

    brackets = config["academic_age_brackets"]
    if not isinstance(brackets, Mapping) or not brackets:
        raise ValueError("enrichment.academic_age_brackets must be a non-empty mapping")
    validated_brackets: dict[str, list[int | None]] = {}
    for cohort, bounds in brackets.items():
        if (
            not isinstance(cohort, str)
            or not cohort
            or not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or not isinstance(bounds[0], int)
            or isinstance(bounds[0], bool)
            or (
                bounds[1] is not None
                and (
                    not isinstance(bounds[1], int)
                    or isinstance(bounds[1], bool)
                )
            )
            or bounds[0] < 0
            or (
                bounds[1] is not None
                and bounds[0] > bounds[1]
            )
        ):
            raise ValueError(
                "each enrichment academic age bracket must be "
                "[non-negative minimum, maximum or null]"
            )
        validated_brackets[cohort] = [bounds[0], bounds[1]]
    ordered_brackets = sorted(
        validated_brackets.values(),
        key=lambda bounds: (
            bounds[0],
            float("inf") if bounds[1] is None else bounds[1],
        ),
    )
    for previous, following in zip(ordered_brackets, ordered_brackets[1:]):
        if previous[1] is None or following[0] <= previous[1]:
            raise ValueError("enrichment academic age brackets must not overlap")
    config["academic_age_brackets"] = validated_brackets

    thresholds = config["classification_thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ValueError("enrichment.classification_thresholds must be a mapping")
    thresholds = {
        **DEFAULT_ENRICHMENT_CONFIG["classification_thresholds"],
        **thresholds,
    }
    confidence = thresholds["gender_min_confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.95 <= confidence <= 1
    ):
        raise ValueError(
            "enrichment.classification_thresholds.gender_min_confidence "
            "must be between 0.95 and 1"
        )
    sample_count = thresholds["gender_min_sample_count"]
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
        raise ValueError(
            "enrichment.classification_thresholds.gender_min_sample_count must be non-negative"
        )
    if not isinstance(thresholds["require_country_validation"], bool):
        raise ValueError(
            "enrichment.classification_thresholds.require_country_validation must be a boolean"
        )
    config["classification_thresholds"] = dict(thresholds)
    return config


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {config_path} must contain a YAML mapping")
    config["enrichment"] = _validated_enrichment_config(config.get("enrichment"))
    return config
