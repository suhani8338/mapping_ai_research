import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.inference.probabilistic_tagger import GenderInferenceEngine, TagResult
from src.enrichment.pronoun_resolver import PronounEvidence
from src.pipeline import (
    _macro_cluster_contrast,
    collect_arxiv_records,
    infer_author_names,
    run_pipeline,
)
from src.utils.config_loader import load_config


class FakeArxivClient:
    def __init__(self, pages):
        self.pages = pages
        self.queries = []

    def query(self, search_query, *, start=0, max_results=100):
        self.queries.append((search_query, start, max_results))
        return self.pages.get(start, [])


class FakeTagger:
    def infer(self, given_name):
        return TagResult("female", 0.98, 250)


class FakeOpenAlexClient:
    def __init__(self, country="IN"):
        self.country = country
        self.calls = []

    def enrich_authors(self, papers, authors):
        self.calls.append((papers, authors))
        return {
            author["record_id"]: {
                "openalex_author_id": "https://openalex.org/A123",
                "country": self.country,
                "ror_id": "https://ror.org/012345678",
                "first_publication_year": 2018,
                "academic_age": 7,
                "academic_cohort": "mid",
            }
            for author in authors
        }


class FakePronounResolver:
    def resolve(
        self,
        *,
        faculty_bio_url,
        personal_homepage_url,
        csrankings_profile_url=None,
        target_pronouns,
    ):
        source_url = (
            faculty_bio_url
            or csrankings_profile_url
            or personal_homepage_url
            or ""
        )
        return PronounEvidence("female", 2, source_url)


class FakeGenderizeTagger:
    def __init__(self):
        self.names = []

    def infer(self, given_name):
        self.names.append(given_name)
        return TagResult("male", 0.97, 250)


def test_default_config_contains_granular_taxonomy_and_cohort_metadata():
    config = load_config(Path(__file__).parents[1] / "config.yaml")

    assert set(config["subfields"]) == {
        "multilingual_nlp",
        "multicultural_alignment",
        "ai_governance_policy",
        "healthcare_ai",
        "human_ai_interaction_edtech",
        "hardware_systems",
        "cybersecurity_ai",
        "robotics_kinematics",
        "core_theory_statistical_optimization",
        "pretraining_foundations_architectures",
        "technical_safety_interp",
    }
    clusters = [field["macro_cluster"] for field in config["subfields"].values()]
    assert clusters.count("applied_socio_technical") == 5
    assert clusters.count("core_infrastructure") == 5
    assert clusters.count("methodological_bridge") == 1
    assert config["experiment"]["sample_size_per_subfield"] == 800
    assert config["experiment"]["date_range"] == {
        "start": "2020-01-01",
        "end": "present",
    }


def test_collect_arxiv_records_uses_parsed_author_order_and_present_date():
    config = {
        "experiment": {
            "sample_size_per_subfield": 1,
            "date_range": {"start": "2020-01-01", "end": "present"},
        },
        "subfields": {"nlp": ["cs.CL"]},
    }
    client = FakeArxivClient(
        {
            0: [
                {
                    "paper_id": "2501.00001",
                    "doi": "10.1234/example",
                    "published": "2025-02-01T00:00:00Z",
                    "primary_category": "cs.CL",
                    "authors": ["First Author", "Middle Author", "Last Author"],
                    "author_records": [
                        {
                            "full_name": "First Author",
                            "author_order": "first",
                            "email": "first@university.jp",
                        },
                        {"full_name": "Middle Author", "author_order": "middle"},
                        {"full_name": "Last Author", "author_order": "last"},
                    ],
                }
            ]
        }
    )

    papers, authors = collect_arxiv_records(config, client)

    assert papers[0]["primary_category"] == "cs.CL"
    assert papers[0]["doi"] == "10.1234/example"
    assert [author["author_order"] for author in authors] == [
        "first",
        "middle",
        "last",
    ]
    assert authors[0]["email"] == "first@university.jp"


def test_collect_arxiv_records_filters_dates_and_assigns_author_order():
    config = {
        "experiment": {
            "sample_size_per_subfield": 1,
            "date_range": {"start": "2025-01-01", "end": "2025-12-31"},
        },
        "subfields": {"nlp": ["cs.CL", "cs.SD"]},
    }
    client = FakeArxivClient(
        {
            0: [
                {
                    "paper_id": "2501.00001",
                    "published": "2025-02-01T00:00:00Z",
                    "authors": ["First Author", "Middle Author", "Last Author"],
                }
            ]
        }
    )

    papers, authors = collect_arxiv_records(config, client)

    assert len(papers) == 1
    assert [author["author_order"] for author in authors] == [
        "first",
        "middle",
        "last",
    ]
    assert len({author["record_id"] for author in authors}) == 3
    assert "cat:cs.CL OR cat:cs.SD" in client.queries[0][0]
    assert "202501010000 TO 202512312359" in client.queries[0][0]


def test_collect_arxiv_records_uses_granular_query_and_preserves_cohort_metadata():
    config = {
        "experiment": {
            "sample_size_per_subfield": 1,
            "date_range": {"start": "2025-01-01", "end": "2025-12-31"},
        },
        "subfields": {
            "multilingual_low_resource_nlp": {
                "macro_cluster": "applied_socio_technical",
                "hypothesized_female_representation": ">25%",
                "target_scope": "Low-resource translation and dialect adaptation.",
                "query": '(cat:cs.CL) AND (ti:"multilingual" OR abs:"low-resource")',
            }
        },
    }
    client = FakeArxivClient(
        {
            0: [
                {
                    "paper_id": "2501.00001",
                    "published": "2025-02-01T00:00:00Z",
                    "authors": ["First Author"],
                }
            ]
        }
    )

    papers, authors = collect_arxiv_records(config, client)

    assert '(ti:"multilingual" OR abs:"low-resource")' in client.queries[0][0]
    assert "202501010000 TO 202512312359" in client.queries[0][0]
    assert papers[0]["macro_cluster"] == "applied_socio_technical"
    assert papers[0]["target_scope"] == "Low-resource translation and dialect adaptation."
    assert authors[0]["hypothesized_female_representation"] == ">25%"


def test_macro_cluster_contrast_compares_first_authors_and_excludes_bridge():
    authors = pd.DataFrame(
        [
            {
                "author_order": "first",
                "inferred_gender": "female",
                "macro_cluster": "applied_socio_technical",
            },
            {
                "author_order": "first",
                "inferred_gender": "male",
                "macro_cluster": "applied_socio_technical",
            },
            {
                "author_order": "first",
                "inferred_gender": "male",
                "macro_cluster": "core_infrastructure",
            },
            {
                "author_order": "first",
                "inferred_gender": "male",
                "macro_cluster": "core_infrastructure",
            },
            {
                "author_order": "first",
                "inferred_gender": "female",
                "macro_cluster": "methodological_bridge",
            },
            {
                "author_order": "last",
                "inferred_gender": "female",
                "macro_cluster": "applied_socio_technical",
            },
        ]
    )

    result = _macro_cluster_contrast(authors).iloc[0]

    assert result["applied_female_rate"] == 0.5
    assert result["applied_known_first_authors"] == 2
    assert result["core_female_rate"] == 0
    assert result["core_known_first_authors"] == 2
    assert result["rate_difference"] == 0.5
    assert result["p_value"] == pytest.approx(0.2482, abs=0.001)


def test_offline_inference_preserves_ambiguous_names_as_unknown():
    authors = [
        {
            "record_id": "a",
            "paper_id": "2501.00001",
            "subfield": "nlp",
            "author_order": "first",
            "full_name": "Xyzzy Smith",
        }
    ]

    result = infer_author_names(
        authors,
        confidence_threshold=0.7,
        min_sample_count=100,
        api_key=None,
    )

    assert result.loc[0, "inferred_gender"] == "unknown"
    assert result.loc[0, "inference_source"] == "gender_guesser"
    assert result.loc[0, "given_name"] == "Xyzzy"


def test_offline_inference_classifies_common_names_without_api():
    authors = [
        {
            "record_id": "a",
            "paper_id": "2501.00001",
            "subfield": "nlp",
            "author_order": "first",
            "full_name": "Mary Smith",
        }
    ]

    result = infer_author_names(
        authors,
        confidence_threshold=0.7,
        min_sample_count=100,
        api_key=None,
    )

    assert result.loc[0, "inferred_gender"] == "female"
    assert result.loc[0, "inference_source"] == "gender_guesser"


def test_genderize_is_used_only_for_names_without_confident_offline_results():
    api_tagger = FakeGenderizeTagger()
    engine = GenderInferenceEngine(api_tagger=api_tagger)

    result = engine.infer("Xyzzy")

    assert result.inferred_gender == "male"
    assert result.inference_source == "genderize_api"
    assert api_tagger.names == ["Xyzzy"]


def test_pipeline_writes_outputs_and_preserves_completed_reviews(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
experiment:
  sample_size_per_subfield: 2
  date_range:
    start: "2025-01-01"
    end: "2025-12-31"
subfields:
  nlp: ["cs.CL"]
inference:
  confidence_threshold: 0.90
  min_sample_count: 100
validation:
  audit_sample_rate: 1.0
""",
        encoding="utf-8",
    )
    client = FakeArxivClient(
        {
            0: [
                {
                    "paper_id": "2501.00001",
                    "published": "2025-02-01T00:00:00Z",
                    "authors": ["Mary Smith", "Alex Smith"],
                },
                {
                    "paper_id": "2501.00002",
                    "published": "2025-03-01T00:00:00Z",
                    "authors": ["Jane Jones", "Alex Smith"],
                },
            ]
        }
    )

    outputs = run_pipeline(
        config_path,
        client=client,
        tagger=FakeTagger(),
        make_plots=True,
    )
    authors_path = outputs["authors"]
    authors = pd.read_csv(authors_path)
    assert len(authors) == 4
    assert set(authors["inferred_gender"]) == {"female"}
    assert set(authors["macro_cluster"]) == {"unspecified"}
    assert outputs["macro_cluster_contrast"].exists()
    assert all(
        outputs[f"{plot}_{format}"].is_file()
        for plot in (
            "pipeline_dropoff",
            "odds_ratio_forest",
            "author_coverage",
            "career_stage_comparison",
        )
        for format in ("png", "pdf")
    )

    audit_path = outputs["audit_queue"]
    audit = pd.read_csv(audit_path, dtype=str)
    audit.loc[0, "reviewer_label"] = "female"
    audit.loc[0, "reviewed_at"] = "2026-11-04"
    audit.to_csv(audit_path, index=False)

    outputs = run_pipeline(
        config_path,
        client=client,
        tagger=FakeTagger(),
        make_plots=False,
    )
    audit = pd.read_csv(outputs["audit_queue"], dtype=str)
    assert "female" in set(audit["reviewer_label"].dropna())
    authors = pd.read_csv(outputs["authors"])
    assert authors["is_audited"].sum() == 1


@pytest.mark.parametrize(
    ("country", "expected_gender"),
    [("IN", "female"), ("UNKNOWN", "unknown")],
)
def test_pipeline_applies_openalex_metadata_and_country_guardrail(
    tmp_path, country, expected_gender
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
experiment:
  sample_size_per_subfield: 1
  date_range:
    start: "2025-01-01"
    end: "2025-12-31"
subfields:
  nlp: ["cs.CL"]
inference:
  confidence_threshold: 0.90
  min_sample_count: 100
validation:
  audit_sample_rate: 1.0
enrichment:
  enabled: true
""",
        encoding="utf-8",
    )
    arxiv_client = FakeArxivClient(
        {
            0: [
                {
                    "paper_id": "2501.00001",
                    "published": "2025-02-01T00:00:00Z",
                    "title": "Research Title",
                    "authors": ["Mary Smith"],
                }
            ]
        }
    )
    openalex_client = FakeOpenAlexClient(country=country)

    outputs = run_pipeline(
        config_path,
        client=arxiv_client,
        tagger=FakeTagger(),
        openalex_client=openalex_client,
        make_plots=False,
    )

    authors = pd.read_csv(outputs["authors"])
    assert len(openalex_client.calls) == 1
    assert authors.loc[0, "country"] == country
    assert authors.loc[0, "academic_cohort"] == "mid"
    assert authors.loc[0, "inferred_gender"] == expected_gender


def test_pipeline_resolves_pronouns_from_supplied_verified_profile_urls(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
experiment:
  sample_size_per_subfield: 1
  date_range:
    start: "2025-01-01"
    end: "2025-12-31"
subfields:
  nlp: ["cs.CL"]
inference:
  confidence_threshold: 0.90
  min_sample_count: 100
  target_pronouns:
    female: ["she", "her", "hers"]
    male: ["he", "him", "his"]
    non_binary: ["they", "them", "their"]
validation:
  audit_sample_rate: 1.0
enrichment:
  enabled: true
""",
        encoding="utf-8",
    )
    client = FakeArxivClient(
        {
            0: [
                {
                    "paper_id": "2501.00001",
                    "published": "2025-02-01T00:00:00Z",
                    "title": "Research Title",
                    "authors": ["Mary Smith"],
                }
            ]
        }
    )
    record_id = hashlib.sha256(b"nlp:2501.00001:0").hexdigest()[:20]

    outputs = run_pipeline(
        config_path,
        client=client,
        tagger=FakeTagger(),
        openalex_client=FakeOpenAlexClient(),
        author_profile_urls={
            record_id: {
                "faculty_bio_url": "https://university.example.edu/mary-smith",
                "personal_homepage_url": "https://mary.example.com",
            }
        },
        pronoun_resolver=FakePronounResolver(),
        make_plots=False,
    )

    author = pd.read_csv(outputs["authors"]).iloc[0]
    assert author["pronoun_label"] == "female"
    assert author["pronoun_count"] == 2
    assert author["pronoun_source_url"] == "https://university.example.edu/mary-smith"
