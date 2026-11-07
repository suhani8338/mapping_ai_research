from __future__ import annotations

from typing import Any

import pytest

from src.enrichment.openalex_client import OpenAlexClient, OpenAlexEnricher
from src.enrichment.pronoun_resolver import (
    PronounEvidence,
    PronounProfileResolver,
)
from src.enrichment.validator import DemographicValidator
from src.utils.config_loader import load_config


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> FakeResponse:
        self.requests.append((url, params))
        return self.responses.pop(0)


class FakePageResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


class FakePageSession:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.requested_urls: list[str] = []

    def get(self, url: str, *, timeout: float) -> FakePageResponse:
        self.requested_urls.append(url)
        return FakePageResponse(self.pages[url])


class FakeArxivClient:
    def __init__(self, papers: list[dict[str, Any]] | None = None) -> None:
        self.papers = papers or []
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def query(
        self,
        search_query: str,
        *,
        start: int = 0,
        max_results: int = 100,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[dict[str, Any]]:
        self.queries.append(
            (
                search_query,
                {
                    "start": start,
                    "max_results": max_results,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                },
            )
        )
        return self.papers


def test_openalex_enriches_exact_author_and_computes_career_cohort():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Exact Research Title",
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A123",
                                        "display_name": "A. Researcher",
                                    },
                                    "institutions": [
                                        {
                                            "country_code": "IN",
                                            "ror": "https://ror.org/012345678",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ),
            FakeResponse({"summary_stats": {"first_publication_year": 2018}}),
        ]
    )
    client = OpenAlexClient(
        mailto="research@example.org",
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=FakeArxivClient(),
    )

    result = client.enrich_authors(
        [{"paper_id": "2501.00001", "title": "Exact Research Title", "published": "2025-02-01"}],
        [
            {
                "record_id": "record-1",
                "paper_id": "2501.00001",
                "full_name": "A. Researcher",
            }
        ],
    )["record-1"]

    assert result == {
        "openalex_author_id": "https://openalex.org/A123",
        "country": "IN",
        "ror_id": "https://ror.org/012345678",
        "first_publication_year": 2018,
        "academic_age": 7,
        "academic_cohort": "mid",
    }
    assert all(params["mailto"] == "research@example.org" for _, params in session.requests)
    assert session.headers["User-Agent"] == (
        "gendered-ai-research (mailto:research@example.org)"
    )


def test_openalex_uses_unknown_for_ambiguous_paper_and_author_matches():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "results": [
                        {"title": "Repeated Title", "authorships": []},
                        {"title": "Repeated Title", "authorships": []},
                    ]
                }
            )
        ]
    )
    client = OpenAlexClient(
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=FakeArxivClient(),
    )

    result = client.enrich_authors(
        [{"paper_id": "2501.00001", "title": "Repeated Title", "published": "2025"}],
        [
            {
                "record_id": "record-1",
                "paper_id": "2501.00001",
                "full_name": "Same Name",
            }
        ],
    )["record-1"]

    assert result["country"] == "UNKNOWN"
    assert result["academic_age"] == -1
    assert result["academic_cohort"] == "unknown"
    assert len(session.requests) == 1


def test_openalex_uses_the_paper_authorships_first_institution():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Exact Research Title",
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A123",
                                        "display_name": "A. Researcher",
                                    },
                                    "institutions": [
                                        {
                                            "country_code": "US",
                                            "ror": "https://ror.org/012345678",
                                        },
                                        {
                                            "country_code": "DE",
                                            "ror": "https://ror.org/987654321",
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ),
            FakeResponse({"first_publication_year": 2020}),
        ]
    )
    client = OpenAlexClient(
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=FakeArxivClient(),
    )

    result = client.enrich_authors(
        [{"paper_id": "2501.00001", "title": "Exact Research Title", "published": "2025"}],
        [
            {
                "record_id": "record-1",
                "paper_id": "2501.00001",
                "full_name": "A. Researcher",
            }
        ],
    )["record-1"]

    assert result["country"] == "US"
    assert result["ror_id"] == "https://ror.org/012345678"
    assert result["academic_age"] == 5


def test_openalex_uses_affiliation_country_without_requiring_ror():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Exact Research Title",
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A123",
                                        "display_name": "A. Researcher",
                                    },
                                    "institutions": [{"country_code": "IN"}],
                                }
                            ],
                        }
                    ]
                }
            ),
            FakeResponse({"first_publication_year": 2020}),
        ]
    )
    client = OpenAlexClient(
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=FakeArxivClient(),
    )

    result = client.enrich_authors(
        [{"paper_id": "2501.00001", "title": "Exact Research Title", "published": "2025"}],
        [
            {
                "record_id": "record-1",
                "paper_id": "2501.00001",
                "full_name": "A. Researcher",
            }
        ],
    )["record-1"]

    assert result["country"] == "IN"
    assert result["ror_id"] == "UNKNOWN"


@pytest.mark.parametrize("status_code", [404, 429])
def test_openalex_returns_unknown_fields_for_missing_or_rate_limited_data(status_code):
    session = FakeSession([FakeResponse({}, status_code=status_code)])
    client = OpenAlexEnricher(
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=FakeArxivClient(),
    )

    assert client.get_paper_affiliations("2501.00001", "Unknown Paper") == []


def test_openalex_career_stats_returns_unknown_age_when_author_lookup_is_unavailable():
    session = FakeSession([FakeResponse({}, status_code=429)])
    client = OpenAlexEnricher(
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=FakeArxivClient(),
    )

    assert client.get_author_career_stats(
        "https://openalex.org/A123", pub_year=2025
    ) == {
        "first_publication_year": -1,
        "academic_age": -1,
        "academic_cohort": "unknown",
    }


def test_author_rate_limit_preserves_paper_affiliation_but_unknowns_academic_age():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Exact Research Title",
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A123",
                                        "display_name": "A. Researcher",
                                    },
                                    "institutions": [
                                        {
                                            "country_code": "IN",
                                            "ror": "https://ror.org/012345678",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ),
            FakeResponse({}, status_code=429),
        ]
    )
    client = OpenAlexClient(
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=FakeArxivClient(),
    )

    result = client.enrich_authors(
        [
            {
                "paper_id": "2501.00001",
                "title": "Exact Research Title",
                "published": "2025-01-01",
            }
        ],
        [
            {
                "record_id": "record-1",
                "paper_id": "2501.00001",
                "full_name": "A. Researcher",
            }
        ],
    )["record-1"]

    assert result == {
        "openalex_author_id": "https://openalex.org/A123",
        "country": "IN",
        "ror_id": "https://ror.org/012345678",
        "first_publication_year": -1,
        "academic_age": -1,
        "academic_cohort": "unknown",
    }


def test_openalex_uses_earliest_counts_by_year_when_first_year_is_not_top_level():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "counts_by_year": [
                        {"year": 2021, "works_count": 1},
                        {"year": 2018, "works_count": 1},
                    ]
                }
            )
        ]
    )
    client = OpenAlexClient(
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=FakeArxivClient(),
    )

    assert client.get_author_career_stats(
        "https://openalex.org/A123", pub_year=2025
    ) == {
        "first_publication_year": 2018,
        "academic_age": 7,
        "academic_cohort": "mid",
    }


def test_senior_cohort_has_no_upper_age_limit():
    client = OpenAlexClient(rate_limit_delay=0, arxiv_client=FakeArxivClient())

    assert client._cohort_for_age(150) == "senior"


def test_openalex_falls_back_to_earliest_arxiv_paper_and_email_country():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Exact Research Title",
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A123",
                                        "display_name": "A. Researcher",
                                    },
                                    "institutions": [],
                                }
                            ],
                        }
                    ]
                }
            ),
            FakeResponse({"summary_stats": {}}),
        ]
    )
    arxiv = FakeArxivClient(
        [
            {
                "authors": ["A. Researcher"],
                "published": "2017-01-01T00:00:00Z",
            },
            {
                "authors": ["A. Researcher"],
                "published": "2021-01-01T00:00:00Z",
            },
        ]
    )
    client = OpenAlexClient(
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=arxiv,
    )

    result = client.enrich_authors(
        [{"paper_id": "2501.00001", "title": "Exact Research Title", "published": "2025"}],
        [
            {
                "record_id": "record-1",
                "paper_id": "2501.00001",
                "full_name": "A. Researcher",
                "email": "researcher@university.de",
            }
        ],
    )["record-1"]

    assert result["country"] == "DE"
    assert result["first_publication_year"] == 2017
    assert result["academic_age"] == 8
    assert result["academic_cohort"] == "senior"
    assert arxiv.queries[0][1]["sort_by"] == "submittedDate"
    assert arxiv.queries[0][1]["sort_order"] == "ascending"


def test_openalex_doi_lookup_and_generic_email_fall_back_to_unknown_country():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "Exact Research Title",
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A123",
                                        "display_name": "A. Researcher",
                                    },
                                    "institutions": [],
                                }
                            ],
                        }
                    ]
                }
            ),
            FakeResponse({"first_publication_year": 2020}),
        ]
    )
    client = OpenAlexClient(
        rate_limit_delay=0,
        session=session,  # type: ignore[arg-type]
        arxiv_client=FakeArxivClient(),
    )

    result = client.enrich_authors(
        [
            {
                "paper_id": "2501.00001",
                "doi": "https://doi.org/10.1234/example",
                "title": "Exact Research Title",
                "published": "2025",
            }
        ],
        [
            {
                "record_id": "record-1",
                "paper_id": "2501.00001",
                "full_name": "A. Researcher",
                "email": "researcher@gmail.com",
            }
        ],
    )["record-1"]

    assert result["country"] == "UNKNOWN"
    assert session.requests[0][1]["filter"] == "doi:10.1234/example"


def test_demographic_validator_applies_all_configured_gates():
    validator = DemographicValidator(
        gender_min_confidence=0.95,
        gender_min_sample_count=100,
        require_country_validation=True,
    )

    assert validator.validate_gender(
        "female", 0.95, sample_count=100, country_code="IN"
    ) == "female"
    assert validator.validate_gender(
        "female", 0.949, sample_count=100, country_code="IN"
    ) == "unknown"
    assert validator.validate_gender(
        "male", 0.99, sample_count=99, country_code="IN"
    ) == "unknown"
    assert validator.validate_gender(
        "male", 0.99, sample_count=100, country_code="UNKNOWN"
    ) == "unknown"
    assert validator.validate_gender(
        "female", float("nan"), sample_count=100, country_code="IN"
    ) == "unknown"
    assert validator.validate_gender(
        "female", 0.99, country_code="IN"
    ) == "unknown"
    assert validator.validate_gender(
        "female", 0.99, sample_count=100, country_code="ZZ"
    ) == "unknown"
    with pytest.raises(ValueError, match="between 0.95 and 1"):
        DemographicValidator(gender_min_confidence=0.94)


def test_author_validator_prioritizes_strong_pronoun_evidence_and_requires_count():
    from src.enrichment.validator import AuthorValidator
    from src.inference.probabilistic_tagger import TagResult

    class FakeInference:
        def infer(self, given_name):
            return TagResult("female", 0.99, 150)

    class FakeBioExtractor:
        def resolve(self, **kwargs):
            return PronounEvidence("non_binary", 3, "https://profile.example", 2)

    validator = AuthorValidator(
        gender_engine=FakeInference(),
        bio_pronoun_extractor=FakeBioExtractor(),  # type: ignore[arg-type]
    )

    result = validator.validate_author(
        "Alex",
        country_code="ZZ",
        target_pronouns={"female": ["she"], "male": ["he"], "non_binary": ["they"]},
        personal_homepage_url="https://profile.example",
    )

    assert result["inferred_gender"] == "non_binary"
    assert result["inference_source"] == "verified_bio_pronouns"
    assert result["validation_tier"] == 1


def test_pronoun_resolver_checks_bio_then_homepage_without_link_crawling():
    bio_url = "https://university.example.edu/bio"
    homepage_url = "https://researcher.example.com"
    session = FakePageSession(
        {
            bio_url: "<html><body>No pronouns listed.</body></html>",
            homepage_url: "<html><body><p>They are a researcher. They publish papers.</p></body></html>",
        }
    )
    resolver = PronounProfileResolver(session=session)  # type: ignore[arg-type]

    result = resolver.resolve(
        faculty_bio_url=bio_url,
        personal_homepage_url=homepage_url,
        target_pronouns={
            "female": ["she", "her", "hers"],
            "male": ["he", "him", "his"],
            "non_binary": ["they", "them", "their"],
        },
    )

    assert result == PronounEvidence("non_binary", 2, homepage_url, 2)
    assert session.requested_urls == [bio_url, homepage_url]


def test_pronoun_resolver_accepts_explicit_pronoun_pairs_without_repetition():
    url = "https://researcher.example.edu/about"
    resolver = PronounProfileResolver(
        session=FakePageSession({url: "<p>Pronouns: they/them</p>"})
    )  # type: ignore[arg-type]

    evidence = resolver.resolve(
        faculty_bio_url=url,
        personal_homepage_url=None,
        target_pronouns={
            "female": ["she", "her", "hers"],
            "male": ["he", "him", "his"],
            "non_binary": ["they", "them", "their"],
        },
    )

    assert evidence.label == "non_binary"
    assert evidence.explicit


def test_pronoun_resolver_stops_on_conflicting_evidence_and_rejects_bad_urls():
    bio_url = "https://university.example.edu/bio"
    homepage_url = "https://researcher.example.com"
    session = FakePageSession(
        {
            bio_url: "<p>She thanked him.</p>",
            homepage_url: "<p>She is a researcher.</p>",
        }
    )
    resolver = PronounProfileResolver(session=session)  # type: ignore[arg-type]
    pronouns = {"female": ["she", "her"], "male": ["he", "him"]}

    result = resolver.resolve(
        faculty_bio_url=bio_url,
        personal_homepage_url=homepage_url,
        target_pronouns=pronouns,
    )

    assert result == PronounEvidence("unknown", 1, bio_url, 0)
    assert session.requested_urls == [bio_url, homepage_url]
    with pytest.raises(ValueError, match="absolute HTTP"):
        resolver.resolve(
            faculty_bio_url="file:///private/profile",
            personal_homepage_url=None,
            target_pronouns=pronouns,
        )


def test_enrichment_config_defaults_and_rejects_overlapping_age_brackets(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment: {}\n", encoding="utf-8")

    loaded = load_config(config_path)
    assert loaded["enrichment"]["enabled"] is False
    assert loaded["enrichment"]["academic_age_brackets"]["senior"] == [8, None]
    config_path.write_text(
        """\
enrichment:
  academic_age_brackets:
    junior: [0, 4]
    mid: [4, 7]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not overlap"):
        load_config(config_path)
