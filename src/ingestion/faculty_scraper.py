"""Extract faculty names from supplied directory HTML; this module does not crawl."""

from __future__ import annotations

from bs4 import BeautifulSoup


def extract_faculty_names(html: str, *, selector: str) -> list[str]:
    """Return unique, non-empty text for elements matched by a CSS selector."""
    if not selector.strip():
        raise ValueError("selector must not be empty")
    soup = BeautifulSoup(html, "html.parser")
    names: list[str] = []
    seen: set[str] = set()
    for element in soup.select(selector):
        name = " ".join(element.get_text(" ", strip=True).split())
        key = name.casefold()
        if name and key not in seen:
            names.append(name)
            seen.add(key)
    return names
