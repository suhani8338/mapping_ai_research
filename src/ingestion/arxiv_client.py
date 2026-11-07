"""Small arXiv Atom API client with pagination and polite request spacing."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlencode

import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_NAMESPACE = {"arxiv": "http://arxiv.org/schemas/atom"}


class ArxivClient:
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        delay_seconds: float = 3.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        self._session = session or requests.Session()
        self._delay_seconds = delay_seconds
        self._timeout_seconds = timeout_seconds
        self._last_request_at: float | None = None

    def query(
        self,
        search_query: str,
        *,
        start: int = 0,
        max_results: int = 100,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[dict[str, Any]]:
        """Fetch one result page and return normalized paper and author fields."""
        if start < 0 or max_results < 1:
            raise ValueError("start must be non-negative and max_results positive")
        if sort_by not in {"relevance", "lastUpdatedDate", "submittedDate"}:
            raise ValueError("sort_by must be relevance, lastUpdatedDate, or submittedDate")
        if sort_order not in {"ascending", "descending"}:
            raise ValueError("sort_order must be ascending or descending")
        params = urlencode(
            {
                "search_query": search_query,
                "start": start,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }
        )
        for attempt in range(self.MAX_ATTEMPTS):
            self._wait_for_rate_limit()
            response = self._session.get(
                f"{ARXIV_API_URL}?{params}", timeout=self._timeout_seconds
            )
            self._last_request_at = time.monotonic()
            if response.status_code == 503 and attempt < self.MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return self._parse_feed(response.text)
        raise RuntimeError("arXiv request retry loop exited unexpectedly")

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self._delay_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)

    @staticmethod
    def _parse_feed(xml_text: str) -> list[dict[str, Any]]:
        root = ET.fromstring(xml_text)
        entries: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ATOM_NAMESPACE):
            identifier = entry.findtext("atom:id", default="", namespaces=ATOM_NAMESPACE)
            paper_id = identifier.rsplit("/", 1)[-1].split("v", 1)[0]
            author_elements = entry.findall("atom:author", ATOM_NAMESPACE)
            authors = [
                author.findtext("atom:name", default="", namespaces=ATOM_NAMESPACE)
                for author in author_elements
            ]
            author_records = []
            for index, (name, author) in enumerate(zip(authors, author_elements)):
                author_record: dict[str, str] = {
                    "full_name": name,
                    "author_order": (
                        "first"
                        if len(authors) == 1 or index == 0
                        else "last"
                        if index == len(authors) - 1
                        else "middle"
                    ),
                }
                email = author.findtext(
                    "atom:email",
                    default="",
                    namespaces={**ATOM_NAMESPACE, **ARXIV_NAMESPACE},
                ).strip() or author.findtext(
                    "arxiv:email", default="", namespaces=ARXIV_NAMESPACE
                ).strip()
                if email:
                    author_record["email"] = email
                author_records.append(author_record)
            categories = [
                category.attrib["term"]
                for category in entry.findall("atom:category", ATOM_NAMESPACE)
                if "term" in category.attrib
            ]
            primary_category = entry.find(
                "arxiv:primary_category", ARXIV_NAMESPACE
            )
            doi = entry.findtext(
                "arxiv:doi",
                default="",
                namespaces=ARXIV_NAMESPACE,
            ).strip() or entry.findtext(
                "atom:doi",
                default="",
                namespaces=ATOM_NAMESPACE,
            ).strip()
            entries.append(
                {
                    "paper_id": paper_id,
                    **({"doi": doi} if doi else {}),
                    "title": " ".join(
                        entry.findtext(
                            "atom:title", default="", namespaces=ATOM_NAMESPACE
                        ).split()
                    ),
                    "published": entry.findtext(
                        "atom:published", default="", namespaces=ATOM_NAMESPACE
                    ),
                    "authors": authors,
                    "author_records": author_records,
                    "categories": categories,
                    "primary_category": (
                        primary_category.attrib.get("term", "")
                        if primary_category is not None
                        else ""
                    ),
                }
            )
        return entries
