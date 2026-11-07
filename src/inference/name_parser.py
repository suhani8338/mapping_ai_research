"""Normalize person names into given and family name components."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TITLES = {"dr", "prof", "professor", "mr", "mrs", "ms", "mx"}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_TOKEN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*\.?", re.UNICODE)


@dataclass(frozen=True)
class ParsedName:
    full_name: str
    given_name: str | None
    family_name: str | None


def parse_name(name: str) -> ParsedName:
    """Parse a display name conservatively; initials are not given names."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    full_name = " ".join(name.split())
    tokens = _TOKEN.findall(full_name)
    while tokens and tokens[0].rstrip(".").casefold() in _TITLES:
        tokens.pop(0)
    while tokens and tokens[-1].rstrip(".").casefold() in _SUFFIXES:
        tokens.pop()
    if not tokens:
        return ParsedName(full_name, None, None)

    given_name = next(
        (token.rstrip(".") for token in tokens[:-1] if len(token.rstrip(".")) > 1),
        None,
    )
    family_name = tokens[-1].rstrip(".") if len(tokens) > 1 else None
    return ParsedName(full_name, given_name, family_name)
