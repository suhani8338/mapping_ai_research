"""Deterministic stratified sampling helpers for manual review."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from typing import TypeVar

T = TypeVar("T")


def stratified_sample(
    records: Iterable[T],
    *,
    stratum: Callable[[T], Hashable],
    sample_rate: float,
    seed: int = 0,
) -> list[T]:
    """Sample each stratum independently, with at least one item per nonempty stratum."""
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be greater than 0 and at most 1")
    groups: dict[Hashable, list[T]] = defaultdict(list)
    for record in records:
        groups[stratum(record)].append(record)
    rng = random.Random(seed)
    selected: list[T] = []
    for group in groups.values():
        sample_size = min(len(group), max(1, round(len(group) * sample_rate)))
        selected.extend(rng.sample(group, sample_size))
    return selected
