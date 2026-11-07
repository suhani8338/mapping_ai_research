"""OpenAlex works and author lookup for paper-level author enrichment."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, Protocol

import requests

from src.enrichment.countries import is_iso_country_code
from src.ingestion.arxiv_client import ArxivClient

OPENALEX_API_URL = "https://api.openalex.org"
LOGGER = logging.getLogger(__name__)
UNKNOWN_AUTHOR_METADATA: dict[str, Any] = {
    "openalex_author_id": "UNKNOWN",
    "country": "UNKNOWN",
    "ror_id": "UNKNOWN",
    "first_publication_year": -1,
    "academic_age": -1,
    "academic_cohort": "unknown",
}


class ArxivAuthorSearch(Protocol):
    def query(
        self,
        search_query: str,
        *,
        start: int = 0,
        max_results: int = 100,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[dict[str, Any]]: ...


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


class OpenAlexClient:
    """Resolve paper identifiers and exact author-name matches from OpenAlex."""

    def __init__(
        self,
        *,
        mailto: str = "",
        rate_limit_delay: float = 0.1,
        academic_age_brackets: Mapping[str, Sequence[int | None]] | None = None,
        session: requests.Session | None = None,
        arxiv_client: ArxivAuthorSearch | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not isinstance(mailto, str):
            raise TypeError("mailto must be a string")
        if (
            isinstance(rate_limit_delay, bool)
            or not isinstance(rate_limit_delay, (int, float))
            or not isfinite(rate_limit_delay)
            or rate_limit_delay < 0
        ):
            raise ValueError("rate_limit_delay must be finite and non-negative")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self._mailto = mailto
        self._rate_limit_delay = float(rate_limit_delay)
        self._session = session or requests.Session()
        if self._mailto and hasattr(self._session, "headers"):
            self._session.headers.update(
                {"User-Agent": f"gendered-ai-research (mailto:{self._mailto})"}
            )
        self._timeout_seconds = timeout_seconds
        self._last_request_at: float | None = None
        self._works: dict[str, Mapping[str, Any] | None] = {}
        self._first_publication_years: dict[str, int | None] = {}
        self._arxiv_first_publication_years: dict[str, int | None] = {}
        self._arxiv_client = (
            arxiv_client if arxiv_client is not None else ArxivClient()
        )
        self._brackets = dict(
            academic_age_brackets
            or {"junior": (0, 3), "mid": (4, 7), "senior": (8, None)}
        )
        self._validate_brackets(self._brackets)

    @staticmethod
    def _validate_brackets(
        brackets: Mapping[str, Sequence[int | None]],
    ) -> None:
        if not brackets or any(
            not isinstance(name, str)
            or not name
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
            for name, bounds in brackets.items()
        ):
            raise ValueError("academic_age_brackets must map names to valid [min, max] pairs")
        ordered = sorted(
            (
                bounds[0],
                float("inf") if bounds[1] is None else bounds[1],
                name,
            )
            for name, bounds in brackets.items()
        )
        for previous, following in zip(ordered, ordered[1:]):
            if following[0] <= previous[1]:
                raise ValueError("academic_age_brackets must not overlap")

    def _get_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        if self._last_request_at is not None:
            remaining = self._rate_limit_delay - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        query = dict(params or {})
        if self._mailto:
            query["mailto"] = self._mailto
        response = self._session.get(
            f"{OPENALEX_API_URL}{path}",
            params=query,
            timeout=self._timeout_seconds,
        )
        self._last_request_at = time.monotonic()
        if response.status_code in {404, 429}:
            LOGGER.warning(
                "OpenAlex returned HTTP %d for %s; retaining unknown metadata",
                response.status_code,
                path,
            )
            return None
        response.raise_for_status()
        return response.json()

    def _lookup_work(
        self, *, arxiv_id: str, title: str, doi: str = ""
    ) -> Mapping[str, Any] | None:
        return self._lookup_work_by_identifier(
            arxiv_id=arxiv_id,
            title=title,
            doi=doi,
        )

    def _lookup_work_by_identifier(
        self, *, arxiv_id: str, title: str, doi: str = ""
    ) -> Mapping[str, Any] | None:
        normalized_doi = doi.strip().lower()
        if normalized_doi:
            normalized_doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", normalized_doi)
            payload = self._get_json(
                "/works",
                params={"filter": f"doi:{normalized_doi}", "per-page": 50},
            )
            results = payload.get("results") if isinstance(payload, Mapping) else None
            matches = [
                work for work in results or [] if isinstance(work, Mapping)
            ] if isinstance(results, list) else []
            if len(matches) == 1:
                return matches[0]
            if matches:
                exact_doi_matches = [
                    work
                    for work in matches
                    if isinstance(work.get("doi"), str)
                    and work["doi"].rstrip("/").lower().endswith(normalized_doi)
                ]
                if len(exact_doi_matches) == 1:
                    return exact_doi_matches[0]

        arxiv_suffix = arxiv_id.strip().lower().removeprefix("arxiv:")
        arxiv_suffix = re.sub(r"v\d+$", "", arxiv_suffix)
        if arxiv_suffix:
            payload = self._get_json(
                "/works",
                params={
                    "filter": (
                        "locations.landing_page_url:"
                        f"https://arxiv.org/abs/{arxiv_suffix}"
                    ),
                    "per-page": 50,
                },
            )
            if payload is None:
                return None
            results = payload.get("results") if isinstance(payload, Mapping) else None
            candidates = [
                work for work in results or [] if isinstance(work, Mapping)
            ] if isinstance(results, list) else []
            exact_matches = [
                work
                for work in candidates
                if self._work_matches_arxiv(work, arxiv_suffix)
                or (
                    title.strip()
                    and isinstance(work.get("title"), str)
                    and _normalized_text(work["title"]) == _normalized_text(title)
                )
            ]
            if len(exact_matches) == 1:
                return exact_matches[0]
            if candidates:
                return None

        if not title.strip():
            return None
        payload = self._get_json(
            "/works",
            params={"search": title.strip(), "per-page": 50},
        )
        results = payload.get("results") if isinstance(payload, Mapping) else None
        exact_matches = [
            work
            for work in results or []
            if isinstance(work, Mapping)
            and isinstance(work.get("title"), str)
            and _normalized_text(work["title"]) == _normalized_text(title)
        ] if isinstance(results, list) else []
        return exact_matches[0] if len(exact_matches) == 1 else None

    @staticmethod
    def _work_matches_arxiv(work: Mapping[str, Any], arxiv_suffix: str) -> bool:
        urls = [
            str(location.get("landing_page_url", "")).lower()
            for location in work.get("locations", [])
            if isinstance(location, Mapping)
        ] if isinstance(work.get("locations"), list) else []
        return any(
            re.search(
                rf"(?:arxiv\.org/(?:abs|pdf)/){re.escape(arxiv_suffix)}(?:v\d+)?(?:\.pdf)?(?:$|[?#])",
                url,
            )
            for url in urls
        )

    def _find_work(self, paper: Mapping[str, Any]) -> Mapping[str, Any] | None:
        paper_id = str(paper.get("paper_id", "")).strip()
        if paper_id in self._works:
            return self._works[paper_id]
        title = paper.get("title")
        title = title if isinstance(title, str) else ""
        arxiv_id = str(paper.get("arxiv_id", paper_id))
        doi = paper.get("doi")
        doi = doi if isinstance(doi, str) else ""
        work = self._lookup_work_by_identifier(
            arxiv_id=arxiv_id,
            title=title,
            doi=doi,
        )
        self._works[paper_id] = work
        return work

    def _first_publication_year(self, author_id: str) -> int | None:
        if author_id in self._first_publication_years:
            return self._first_publication_years[author_id]
        payload = self._get_json(
            f"/authors/{author_id.rsplit('/', 1)[-1]}",
        )
        stats = payload.get("summary_stats") if isinstance(payload, Mapping) else None
        year = (
            stats.get("first_publication_year")
            if isinstance(stats, Mapping)
            else None
        )
        if year is None and isinstance(payload, Mapping):
            year = payload.get("first_publication_year")
        if year is None and isinstance(payload, Mapping):
            counts_by_year = payload.get("counts_by_year")
            years = [
                row.get("year")
                for row in counts_by_year or []
                if isinstance(row, Mapping)
                and isinstance(row.get("year"), int)
                and not isinstance(row.get("year"), bool)
            ]
            year = min(years) if years else None
        result = year if isinstance(year, int) and not isinstance(year, bool) else None
        self._first_publication_years[author_id] = result
        return result

    def _arxiv_first_publication_year(self, author_name: str) -> int | None:
        cache_key = _normalized_text(author_name)
        if cache_key in self._arxiv_first_publication_years:
            return self._arxiv_first_publication_years[cache_key]
        escaped_name = author_name.replace("\\", "\\\\").replace('"', '\\"')
        papers = self._arxiv_client.query(
            f'au:"{escaped_name}"',
            start=0,
            max_results=100,
            sort_by="submittedDate",
            sort_order="ascending",
        )
        matching_years = [
            int(str(paper.get("published", ""))[:4])
            for paper in papers
            if isinstance(paper, Mapping)
            and isinstance(paper.get("authors"), list)
            and any(
                isinstance(name, str)
                and _normalized_text(name) == cache_key
                for name in paper["authors"]
            )
            and str(paper.get("published", ""))[:4].isdigit()
        ]
        result = min(matching_years) if matching_years else None
        self._arxiv_first_publication_years[cache_key] = result
        return result

    @staticmethod
    def _country_from_email(email: object) -> str:
        if not isinstance(email, str):
            return "UNKNOWN"
        addresses = re.findall(
            r"(?<![\w.+-])[\w.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![\w.-])",
            email,
        )
        for domain in addresses:
            top_level_domain = domain.rsplit(".", 1)[-1].upper()
            country_code = {"UK": "GB"}.get(top_level_domain, top_level_domain)
            if is_iso_country_code(country_code):
                return country_code
        return "UNKNOWN"

    @classmethod
    def _country_from_affiliation(
        cls, authorship: Mapping[str, Any], email: object
    ) -> tuple[str, str]:
        institutions = authorship.get("institutions")
        first_institution = (
            institutions[0]
            if isinstance(institutions, list)
            and institutions
            and isinstance(institutions[0], Mapping)
            else None
        )
        country_code = (
            first_institution.get("country_code")
            if isinstance(first_institution, Mapping)
            else None
        )
        ror_id = (
            first_institution.get("ror")
            if isinstance(first_institution, Mapping)
            else None
        )
        valid_ror_id = (
            ror_id.strip()
            if isinstance(ror_id, str)
            and re.fullmatch(
                r"https://ror\.org/[0-9a-z]{9}",
                ror_id.strip(),
                flags=re.IGNORECASE,
            )
            else "UNKNOWN"
        )
        return (
            country_code
            if is_iso_country_code(country_code)
            else cls._country_from_email(email),
            valid_ror_id,
        )

    def get_paper_affiliations(
        self, arxiv_id: str, title: str
    ) -> list[dict[str, Any]]:
        """Return matched OpenAlex authorships or an empty list when unresolved."""
        work = self._lookup_work(arxiv_id=arxiv_id, title=title)
        if work is None:
            return []
        authorships = work.get("authorships")
        return (
            [dict(authorship) for authorship in authorships if isinstance(authorship, Mapping)]
            if isinstance(authorships, list)
            else []
        )

    def get_author_career_stats(
        self,
        openalex_author_id: str,
        *,
        pub_year: int | None = None,
    ) -> dict[str, Any]:
        """Fetch first publication year and derive age/cohort for a paper year."""
        first_year = self._first_publication_year(openalex_author_id)
        academic_age = (
            pub_year - first_year
            if isinstance(pub_year, int)
            and not isinstance(pub_year, bool)
            and first_year is not None
            and pub_year >= first_year
            else -1
        )
        return {
            "first_publication_year": first_year if first_year is not None else -1,
            "academic_age": academic_age,
            "academic_cohort": (
                self._cohort_for_age(academic_age) if academic_age >= 0 else "unknown"
            ),
        }

    def _cohort_for_age(self, academic_age: int) -> str:
        for cohort, bounds in self._brackets.items():
            if bounds[0] <= academic_age and (
                bounds[1] is None or academic_age <= bounds[1]
            ):
                return cohort
        return "unknown"

    def enrich_authors(
        self,
        papers: Sequence[Mapping[str, Any]],
        authors: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Return OpenAlex metadata keyed by each pipeline author's record_id."""
        papers_by_id = {
            str(paper.get("paper_id", "")): paper
            for paper in papers
            if paper.get("paper_id") is not None
        }
        work_cache: dict[str, Mapping[str, Any] | None] = {}
        metadata_by_record: dict[str, dict[str, Any]] = {}

        for author in authors:
            record_id = str(author.get("record_id", ""))
            paper_id = str(author.get("paper_id", ""))
            paper = papers_by_id.get(paper_id)
            if paper is None:
                metadata_by_record[record_id] = dict(UNKNOWN_AUTHOR_METADATA)
                continue
            if paper_id not in work_cache:
                work_cache[paper_id] = self._find_work(paper)
            work = work_cache[paper_id]
            full_name = author.get("full_name")
            authorships = work.get("authorships") if isinstance(work, Mapping) else []
            matched_authorships = [
                authorship
                for authorship in authorships or []
                if isinstance(authorship, Mapping)
                and isinstance(authorship.get("author"), Mapping)
                and isinstance(authorship["author"].get("display_name"), str)
                and isinstance(full_name, str)
                and _normalized_text(authorship["author"]["display_name"])
                == _normalized_text(full_name)
            ]
            matched_authorship = (
                matched_authorships[0] if len(matched_authorships) == 1 else None
            )
            openalex_author = (
                matched_authorship.get("author")
                if isinstance(matched_authorship, Mapping)
                else None
            )
            candidate_author_id = (
                openalex_author.get("id")
                if isinstance(openalex_author, Mapping)
                else None
            )
            author_id = (
                candidate_author_id
                if isinstance(candidate_author_id, str)
                and candidate_author_id.startswith("https://openalex.org/A")
                else None
            )

            email = author.get("email")
            country, ror_id = (
                self._country_from_affiliation(matched_authorship, email)
                if matched_authorship is not None
                else (self._country_from_email(email), "UNKNOWN")
            )
            paper_year_value = str(paper.get("published", ""))[:4]
            paper_year = int(paper_year_value) if paper_year_value.isdigit() else None
            first_year: int | None = None
            academic_age = -1
            cohort = "unknown"

            if author_id is not None:
                first_year = self._first_publication_year(author_id)
                if (
                    first_year is not None
                    and paper_year is not None
                    and paper_year >= first_year
                ):
                    academic_age = paper_year - first_year
                    cohort = self._cohort_for_age(academic_age)

            if (
                academic_age < 0
                and isinstance(full_name, str)
                and full_name.strip()
                and paper_year is not None
            ):
                arxiv_year = self._arxiv_first_publication_year(full_name)
                if arxiv_year is not None and paper_year >= arxiv_year:
                    first_year = arxiv_year
                    academic_age = paper_year - arxiv_year
                    cohort = self._cohort_for_age(academic_age)
                else:
                    first_year = None

            metadata_by_record[record_id] = {
                "openalex_author_id": author_id or "UNKNOWN",
                "country": country,
                "ror_id": ror_id,
                "first_publication_year": first_year if first_year is not None else -1,
                "academic_age": academic_age,
                "academic_cohort": cohort,
            }
        return metadata_by_record


class OpenAlexEnricher(OpenAlexClient):
    """Compatibility name for the Phase 1 OpenAlex enrichment API."""
