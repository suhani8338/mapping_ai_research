"""Orchestrate collection, cautious inference, auditing, and aggregate reporting."""

from __future__ import annotations

import hashlib
import logging
import math
import os
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.analysis.plotting import plot_analysis_exports
from src.enrichment.openalex_client import UNKNOWN_AUTHOR_METADATA, OpenAlexEnricher
from src.enrichment.pronoun_resolver import BioPronounExtractor, PronounEvidence
from src.enrichment.validator import AuthorValidator
from src.inference.name_parser import parse_name
from src.inference.probabilistic_tagger import GenderInferenceEngine, TagResult
from src.ingestion.arxiv_client import ArxivClient
from src.utils.config_loader import load_config
from src.validation.audit_sampler import stratified_sample

LOGGER = logging.getLogger(__name__)

AUTHOR_COLUMNS = [
    "record_id",
    "paper_id",
    "subfield",
    "macro_cluster",
    "hypothesized_female_representation",
    "author_order",
    "full_name",
    "given_name",
    "inferred_gender",
    "inference_source",
    "confidence",
    "openalex_author_id",
    "country",
    "ror_id",
    "first_publication_year",
    "academic_age",
    "academic_cohort",
    "pronoun_label",
    "pronoun_count",
    "pronoun_margin",
    "pronoun_source_url",
    "inference_count",
    "validation_tier",
    "is_audited",
]
AUDIT_COLUMNS = [
    "record_id",
    "reviewer_label",
    "reviewed_at",
    "review_notes",
    "full_name",
    "given_name",
    "subfield",
    "macro_cluster",
    "author_order",
    "inferred_gender",
    "confidence",
]


class ArxivQueryClient(Protocol):
    def query(
        self,
        search_query: str,
        *,
        start: int = 0,
        max_results: int = 100,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[dict[str, Any]]: ...


class OpenAlexEnrichmentClient(Protocol):
    def enrich_authors(
        self,
        papers: list[dict[str, Any]],
        authors: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]: ...


class PronounEvidenceResolver(Protocol):
    def resolve(
        self,
        *,
        faculty_bio_url: str | None,
        personal_homepage_url: str | None,
        csrankings_profile_url: str | None,
        target_pronouns: Mapping[str, list[str]],
    ) -> PronounEvidence: ...


def _validated_study_window(config: Mapping[str, Any]) -> tuple[date, date]:
    try:
        start = date.fromisoformat(config["experiment"]["date_range"]["start"])
        end_value = config["experiment"]["date_range"]["end"]
        end = (
            date.today()
            if isinstance(end_value, str) and end_value.lower() == "present"
            else date.fromisoformat(end_value)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("config must define valid experiment date_range start/end dates") from error
    if start > end:
        raise ValueError("experiment date_range start must not be after end")
    return start, end


def collect_arxiv_records(
    config: Mapping[str, Any], client: ArxivQueryClient
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch the configured number of dated papers per subfield and expand authors."""
    start_date, end_date = _validated_study_window(config)
    sample_size = config.get("experiment", {}).get("sample_size_per_subfield")
    if not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("experiment.sample_size_per_subfield must be a positive integer")
    subfields = config.get("subfields")
    if not isinstance(subfields, dict) or not subfields:
        raise ValueError("config.subfields must be a non-empty mapping")

    all_authors: list[dict[str, Any]] = []
    all_papers: list[dict[str, Any]] = []
    date_filter = (
        f"submittedDate:[{start_date:%Y%m%d}0000 TO {end_date:%Y%m%d}2359]"
    )

    for subfield, subfield_config in subfields.items():
        if isinstance(subfield_config, list):
            if not subfield_config or not all(
                isinstance(category, str) and category.strip()
                for category in subfield_config
            ):
                raise ValueError(
                    f"subfields.{subfield} must contain non-empty arXiv categories"
                )
            category_filter = " OR ".join(
                f"cat:{category}" for category in subfield_config
            )
            query = f"({category_filter})"
            macro_cluster = "unspecified"
            hypothesis = ""
            target_scope = ""
        elif isinstance(subfield_config, Mapping):
            query_value = subfield_config.get("query")
            macro_cluster_value = subfield_config.get("macro_cluster")
            if not isinstance(query_value, str) or not query_value.strip():
                raise ValueError(f"subfields.{subfield}.query must be a non-empty string")
            if not isinstance(macro_cluster_value, str) or not macro_cluster_value.strip():
                raise ValueError(
                    f"subfields.{subfield}.macro_cluster must be a non-empty string"
                )
            if macro_cluster_value not in {
                "applied_socio_technical",
                "core_infrastructure",
                "methodological_bridge",
            }:
                raise ValueError(
                    f"subfields.{subfield}.macro_cluster is not a recognized cohort"
                )
            query = f"({query_value.strip()})"
            macro_cluster = macro_cluster_value.strip()
            hypothesis_value = subfield_config.get(
                "hypothesized_female_representation", ""
            )
            scope_value = subfield_config.get("target_scope", "")
            if not isinstance(hypothesis_value, str) or not isinstance(scope_value, str):
                raise ValueError(
                    f"subfields.{subfield} hypothesis and target_scope must be strings"
                )
            hypothesis = hypothesis_value
            target_scope = scope_value
        else:
            raise ValueError(
                f"subfields.{subfield} must be a category list or query mapping"
            )
        search_query = f"{query} AND {date_filter}"
        papers: dict[str, dict[str, Any]] = {}
        start_index = 0

        while len(papers) < sample_size:
            page_size = min(100, sample_size - len(papers))
            page = client.query(
                search_query, start=start_index, max_results=page_size
            )
            if not page:
                break

            for paper in page:
                paper_id = paper.get("paper_id")
                published = paper.get("published", "")
                authors = paper.get("authors")
                if not paper_id or not isinstance(authors, list):
                    LOGGER.warning("Skipping malformed arXiv record in %s", subfield)
                    continue
                try:
                    publication_date = date.fromisoformat(str(published)[:10])
                except ValueError:
                    LOGGER.warning("Skipping arXiv record with invalid date: %s", paper_id)
                    continue
                if start_date <= publication_date <= end_date:
                    papers.setdefault(str(paper_id), paper)

            start_index += len(page)
            if len(page) < page_size:
                break

        selected_papers = list(papers.values())[:sample_size]
        if len(selected_papers) < sample_size:
            LOGGER.warning(
                "%s: collected %d of %d requested papers",
                subfield,
                len(selected_papers),
                sample_size,
            )

        for paper in selected_papers:
            paper_id = str(paper["paper_id"])
            authors = paper["authors"]
            author_records = paper.get("author_records")
            if author_records is not None and (
                not isinstance(author_records, list)
                or len(author_records) != len(authors)
            ):
                raise ValueError(
                    f"arXiv author_records must match authors for paper {paper_id}"
                )
            normalized_paper = {
                "paper_id": paper_id,
                "subfield": subfield,
                "macro_cluster": macro_cluster,
                "hypothesized_female_representation": hypothesis,
                "target_scope": target_scope,
                **(
                    {"doi": paper["doi"]}
                    if isinstance(paper.get("doi"), str) and paper["doi"].strip()
                    else {}
                ),
                "published": str(paper.get("published", "")),
                "primary_category": str(paper.get("primary_category", "")),
                "title": str(paper.get("title", "")),
                "authors": " | ".join(str(name) for name in authors),
            }
            all_papers.append(normalized_paper)
            for author_index, full_name in enumerate(authors):
                if not isinstance(full_name, str) or not full_name.strip():
                    continue
                if author_records is not None:
                    author_record = author_records[author_index]
                    if (
                        not isinstance(author_record, Mapping)
                        or author_record.get("full_name") != full_name
                        or author_record.get("author_order")
                        not in {"first", "middle", "last"}
                    ):
                        raise ValueError(
                            f"Invalid arXiv author record for paper {paper_id}"
                        )
                    author_order = author_record["author_order"]
                elif author_index == 0:
                    author_order = "first"
                elif author_index == len(authors) - 1:
                    author_order = "last"
                else:
                    author_order = "middle"
                record_key = f"{subfield}:{paper_id}:{author_index}"
                all_authors.append(
                    {
                        "record_id": hashlib.sha256(record_key.encode()).hexdigest()[:20],
                        "paper_id": paper_id,
                        "subfield": subfield,
                        "macro_cluster": macro_cluster,
                        "hypothesized_female_representation": hypothesis,
                        "author_order": author_order,
                        "full_name": full_name.strip(),
                        **(
                            {"email": author_record["email"]}
                            if author_records is not None
                            and isinstance(author_record.get("email"), str)
                            else {}
                        ),
                    }
                )

    if not all_papers:
        raise RuntimeError("arXiv returned no papers in the configured date range")
    return all_papers, all_authors


def infer_author_names(
    authors: list[dict[str, Any]],
    *,
    confidence_threshold: float,
    min_sample_count: int,
    api_key: str | None,
    tagger: GenderInferenceEngine | None = None,
    validator: AuthorValidator | None = None,
    pronoun_resolver: PronounEvidenceResolver | None = None,
    target_pronouns: Mapping[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Parse names, infer offline, and use the API only as an opt-in fallback."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("inference.confidence_threshold must be between 0 and 1")
    if min_sample_count < 0:
        raise ValueError("inference.min_sample_count must not be negative")
    inference_engine = tagger
    if inference_engine is None:
        inference_engine = GenderInferenceEngine(
            api_key=api_key,
            confidence_threshold=confidence_threshold,
            min_sample_count=min_sample_count,
        )
    if not api_key:
        LOGGER.warning(
            "GENDERIZE_API_KEY is unset; using offline name classifications only"
        )

    parsed_rows: list[dict[str, Any]] = []
    result_cache: dict[str, TagResult] = {}
    for author in authors:
        parsed = parse_name(author["full_name"])
        given_name = parsed.given_name
        if not given_name:
            result = TagResult("unknown", 0.0, 0)
            source = "name_unavailable"
        else:
            cache_key = given_name.casefold()
            if cache_key not in result_cache:
                result_cache[cache_key] = inference_engine.infer(given_name)
            result = result_cache[cache_key]
            source = result.inference_source
        if validator is not None:
            classification = validator.validate_author(
                given_name,
                country_code=author.get("country"),
                target_pronouns=target_pronouns or {},
                faculty_bio_url=author.get("faculty_bio_url"),
                personal_homepage_url=author.get("personal_homepage_url"),
                csrankings_profile_url=author.get("csrankings_profile_url"),
                inference_result=result,
            )
        else:
            classification = {
                "inferred_gender": result.inferred_gender,
                "inference_source": source,
                "confidence": result.confidence,
                "inference_count": result.count,
                "validation_tier": 0,
            }
        if validator is not None:
            pronoun_evidence = PronounEvidence(
                classification["pronoun_label"],
                classification["pronoun_count"],
                classification["pronoun_source_url"],
                classification["pronoun_margin"],
            )
        elif pronoun_resolver is not None and target_pronouns is not None:
            pronoun_evidence = pronoun_resolver.resolve(
                faculty_bio_url=author.get("faculty_bio_url"),
                personal_homepage_url=author.get("personal_homepage_url"),
                csrankings_profile_url=author.get("csrankings_profile_url"),
                target_pronouns=target_pronouns or {},
            )
        else:
            pronoun_evidence = PronounEvidence("unknown", 0, "")
        parsed_rows.append(
            {
                **author,
                "given_name": given_name or "",
                **classification,
                **{
                    key: author.get(key, default)
                    for key, default in UNKNOWN_AUTHOR_METADATA.items()
                },
                "pronoun_label": pronoun_evidence.label,
                "pronoun_count": pronoun_evidence.count,
                "pronoun_margin": pronoun_evidence.margin,
                "pronoun_source_url": pronoun_evidence.source_url,
                "is_audited": False,
            }
        )
    return pd.DataFrame(parsed_rows, columns=AUTHOR_COLUMNS)


def _load_existing_audit(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    previous = pd.read_csv(path, dtype={"record_id": str})
    previous = previous.astype(object).where(previous.notna(), "")
    if "record_id" not in previous or "reviewer_label" not in previous:
        raise ValueError(f"Existing audit file has an invalid schema: {path}")
    return {
        str(row["record_id"]): row.to_dict()
        for _, row in previous.iterrows()
        if str(row.get("reviewer_label", "")).strip()
    }


def _create_audit_file(
    authors: pd.DataFrame, output_path: Path, sample_rate: float
) -> pd.DataFrame:
    reviewed = _load_existing_audit(output_path)
    pending = stratified_sample(
        authors.to_dict(orient="records"),
        stratum=lambda row: (
            row["subfield"],
            row["inferred_gender"],
            row["author_order"],
        ),
        sample_rate=sample_rate,
        seed=42,
    )
    rows_by_id: dict[str, dict[str, Any]] = dict(reviewed)
    for author in pending:
        record_id = str(author["record_id"])
        previous = reviewed.get(record_id, {})
        rows_by_id[record_id] = {
            "record_id": record_id,
            "reviewer_label": previous.get("reviewer_label", ""),
            "reviewed_at": previous.get("reviewed_at", ""),
            "review_notes": previous.get("review_notes", ""),
            "full_name": author["full_name"],
            "given_name": author["given_name"],
            "subfield": author["subfield"],
            "macro_cluster": author.get("macro_cluster", "unspecified"),
            "author_order": author["author_order"],
            "inferred_gender": author["inferred_gender"],
            "confidence": author["confidence"],
        }

    audit = pd.DataFrame(rows_by_id.values(), columns=AUDIT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)
    reviewed_ids = {
        record_id for record_id, row in rows_by_id.items()
        if str(row.get("reviewer_label", "")).strip()
    }
    authors.loc[:, "is_audited"] = authors["record_id"].isin(reviewed_ids)
    return audit


def _representation_summary(authors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subfield, group in authors.groupby("subfield", sort=True):
        row: dict[str, Any] = {
            "subfield": subfield,
            "macro_cluster": (
                group["macro_cluster"].iloc[0]
                if "macro_cluster" in group and not group.empty
                else "unspecified"
            ),
            "hypothesized_female_representation": (
                group["hypothesized_female_representation"].iloc[0]
                if "hypothesized_female_representation" in group and not group.empty
                else ""
            ),
        }
        known_authors = group[group["inferred_gender"].isin(["female", "male"])]
        row["known_author_count"] = len(known_authors)
        row["author_coverage"] = len(known_authors) / len(group) if len(group) else math.nan
        for role, prefix in (("first", "first_author"), ("last", "last_author")):
            role_rows = group[group["author_order"] == role]
            known = role_rows[role_rows["inferred_gender"].isin(["female", "male"])]
            row[f"{prefix}_female_pct"] = (
                100 * (known["inferred_gender"] == "female").mean()
                if not known.empty
                else math.nan
            )
            row[f"{prefix}_coverage"] = (
                len(known) / len(role_rows) if not role_rows.empty else math.nan
            )
            row[f"{prefix}_known_count"] = len(known)
        rows.append(row)
    return pd.DataFrame(rows)


def _macro_cluster_contrast(authors: pd.DataFrame) -> pd.DataFrame:
    """Compare first-author female proportions in applied and core cohorts."""
    first_authors = authors[
        (authors["author_order"] == "first")
        & authors["inferred_gender"].isin(["female", "male"])
    ]
    cohorts = {
        cluster: first_authors[first_authors["macro_cluster"] == cluster]
        for cluster in ("applied_socio_technical", "core_infrastructure")
    }
    applied = cohorts["applied_socio_technical"]
    core = cohorts["core_infrastructure"]
    applied_n = len(applied)
    core_n = len(core)
    applied_female = int((applied["inferred_gender"] == "female").sum())
    core_female = int((core["inferred_gender"] == "female").sum())
    applied_rate = applied_female / applied_n if applied_n else math.nan
    core_rate = core_female / core_n if core_n else math.nan
    z_stat = math.nan
    p_value = math.nan
    if applied_n and core_n:
        pooled_rate = (applied_female + core_female) / (applied_n + core_n)
        standard_error = math.sqrt(
            pooled_rate * (1 - pooled_rate) * (1 / applied_n + 1 / core_n)
        )
        if standard_error:
            z_stat = (applied_rate - core_rate) / standard_error
            p_value = math.erfc(abs(z_stat) / math.sqrt(2))
    return pd.DataFrame(
        [
            {
                "contrast": "applied_socio_technical_vs_core_infrastructure",
                "applied_female_rate": applied_rate,
                "applied_known_first_authors": applied_n,
                "core_female_rate": core_rate,
                "core_known_first_authors": core_n,
                "rate_difference": (
                    applied_rate - core_rate
                    if applied_n and core_n
                    else math.nan
                ),
                "z_statistic": z_stat,
                "p_value": p_value,
            }
        ]
    )


def _chi_square_by_role(authors: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import chi2_contingency

    results: list[dict[str, Any]] = []
    for role in ("first", "last"):
        known = authors[
            (authors["author_order"] == role)
            & authors["inferred_gender"].isin(["female", "male"])
        ]
        table = (
            pd.crosstab(known["subfield"], known["inferred_gender"])
            .reindex(columns=["female", "male"], fill_value=0)
        )
        table = table.loc[table.sum(axis=1) > 0]
        if len(table) >= 2 and (table.sum(axis=0) > 0).all():
            statistic, p_value, degrees, _ = chi2_contingency(table)
            results.append(
                {
                    "author_order": role,
                    "chi2_statistic": statistic,
                    "p_value": p_value,
                    "degrees_of_freedom": degrees,
                    "subfield_count": len(table),
                }
            )
        else:
            results.append(
                {
                    "author_order": role,
                    "chi2_statistic": math.nan,
                    "p_value": math.nan,
                    "degrees_of_freedom": math.nan,
                    "subfield_count": len(table),
                }
            )
    return pd.DataFrame(results)


def _seniority_dropoff(authors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subfield, group in authors.groupby("subfield", sort=True):
        counts: dict[str, tuple[int, int]] = {}
        for role in ("first", "last"):
            role_rows = group[
                (group["author_order"] == role)
                & group["inferred_gender"].isin(["female", "male"])
            ]
            female_count = int((role_rows["inferred_gender"] == "female").sum())
            counts[role] = (female_count, len(role_rows))
        first_female, first_total = counts["first"]
        last_female, last_total = counts["last"]
        p_first = first_female / first_total if first_total else math.nan
        p_last = last_female / last_total if last_total else math.nan
        z_stat: float = math.nan
        p_value: float = math.nan
        if first_total and last_total:
            pooled = (first_female + last_female) / (first_total + last_total)
            standard_error = math.sqrt(
                pooled * (1 - pooled) * (1 / first_total + 1 / last_total)
            )
            if standard_error:
                z_stat = (p_first - p_last) / standard_error
                p_value = math.erfc(abs(z_stat) / math.sqrt(2))
        rows.append(
            {
                "subfield": subfield,
                "p_first": p_first,
                "p_last": p_last,
                "delta": p_first - p_last if first_total and last_total else math.nan,
                "z_stat": z_stat,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def _faculty_summary(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "faculty_final.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["subfield", "faculty_count", "known_count", "female_pct", "coverage"]
        )
    faculty = pd.read_csv(path)
    if not {"inferred_gender"}.issubset(faculty.columns):
        raise ValueError(f"Faculty data is missing inferred_gender: {path}")
    if "subfield" not in faculty.columns:
        faculty["subfield"] = "all"
    rows = []
    for subfield, group in faculty.groupby("subfield", sort=True):
        known = group[group["inferred_gender"].isin(["female", "male"])]
        rows.append(
            {
                "subfield": subfield,
                "faculty_count": len(group),
                "known_count": len(known),
                "female_pct": (
                    100 * (known["inferred_gender"] == "female").mean()
                    if not known.empty
                    else math.nan
                ),
                "coverage": len(known) / len(group) if len(group) else math.nan,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["subfield", "faculty_count", "known_count", "female_pct", "coverage"],
    )


def run_pipeline(
    config_path: str | Path = "config.yaml",
    *,
    client: ArxivQueryClient | None = None,
    tagger: GenderInferenceEngine | None = None,
    openalex_client: OpenAlexEnrichmentClient | None = None,
    author_profile_urls: Mapping[str, Mapping[str, str]] | None = None,
    pronoun_resolver: PronounEvidenceResolver | None = None,
    make_plots: bool = True,
) -> dict[str, Path]:
    """Run the configured research workflow and return paths to generated outputs."""
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    if author_profile_urls is not None and not isinstance(author_profile_urls, Mapping):
        raise ValueError("author_profile_urls must be a mapping keyed by record_id")
    min_known_authors = int(
        config.get("validation", {}).get("min_known_authors_per_subfield", 350)
    )
    if min_known_authors < 1:
        raise ValueError("validation.min_known_authors_per_subfield must be positive")
    project_root = config_file.parent
    arxiv_client = client or ArxivClient()

    papers, raw_authors = collect_arxiv_records(config, arxiv_client)
    interim_path = project_root / "data" / "interim" / "arxiv_papers.csv"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(papers).to_csv(interim_path, index=False)

    inference_config = config.get("inference", {})
    target_pronouns = inference_config.get(
        "target_pronouns",
        {
            "female": ["she", "her", "hers"],
            "male": ["he", "him", "his"],
            "non_binary": ["they", "them", "their"],
        },
    )
    api_key = os.environ.get("GENDERIZE_API_KEY")
    confidence_threshold = float(inference_config.get("confidence_threshold", 0.9))
    min_sample_count = int(inference_config.get("min_sample_count", 100))

    enrichment_config = config["enrichment"]
    validator: AuthorValidator | None = None
    resolved_pronoun_resolver: PronounEvidenceResolver | None = None
    inference_engine = tagger
    if enrichment_config["enabled"]:
        enrichment_client = openalex_client or OpenAlexEnricher(
            mailto=enrichment_config["openalex_mailto"],
            rate_limit_delay=enrichment_config["rate_limit_delay"],
            academic_age_brackets=enrichment_config["academic_age_brackets"],
        )
        metadata = enrichment_client.enrich_authors(papers, raw_authors)
        for author in raw_authors:
            author.update(
                metadata.get(
                    str(author["record_id"]),
                    dict(UNKNOWN_AUTHOR_METADATA),
                )
            )
        thresholds = enrichment_config["classification_thresholds"]
        confidence_threshold = float(thresholds["gender_min_confidence"])
        min_sample_count = int(thresholds["gender_min_sample_count"])
        if inference_engine is None:
            inference_engine = GenderInferenceEngine(
                api_key=api_key,
                confidence_threshold=confidence_threshold,
                min_sample_count=min_sample_count,
            )
        if author_profile_urls:
            resolved_pronoun_resolver = pronoun_resolver or BioPronounExtractor()
            for author in raw_authors:
                urls = author_profile_urls.get(str(author["record_id"]), {})
                if not isinstance(urls, Mapping):
                    raise ValueError(
                        "author_profile_urls values must map profile URL fields to URLs"
                    )
                unexpected_fields = set(urls) - {
                    "faculty_bio_url",
                    "personal_homepage_url",
                    "csrankings_profile_url",
                }
                if unexpected_fields:
                    raise ValueError(
                        "author_profile_urls supports only faculty_bio_url, "
                        "personal_homepage_url, and csrankings_profile_url"
                    )
                for field in (
                    "faculty_bio_url",
                    "personal_homepage_url",
                    "csrankings_profile_url",
                ):
                    url = urls.get(field)
                    if url is not None and not isinstance(url, str):
                        raise ValueError(f"author profile URL {field} must be a string")
                    author[field] = url

        validator = AuthorValidator(
            gender_engine=inference_engine,
            bio_pronoun_extractor=resolved_pronoun_resolver,
            **thresholds,
        )
    authors = infer_author_names(
        raw_authors,
        confidence_threshold=confidence_threshold,
        min_sample_count=min_sample_count,
        api_key=api_key,
        tagger=inference_engine,
        validator=validator,
        pronoun_resolver=resolved_pronoun_resolver,
        target_pronouns=target_pronouns,
    )
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    audit_path = processed_dir / "manual_audit_ground_truth.csv"
    audit_rate = float(config.get("validation", {}).get("audit_sample_rate", 0.1))
    if not 0 < audit_rate <= 1:
        raise ValueError("validation.audit_sample_rate must be greater than 0 and at most 1")
    _create_audit_file(authors, audit_path, audit_rate)

    author_path = processed_dir / "arxiv_authors_final.csv"
    authors.reindex(columns=AUTHOR_COLUMNS).to_csv(author_path, index=False)
    summary = _representation_summary(authors)
    for _, row in summary.iterrows():
        if row["known_author_count"] < min_known_authors:
            LOGGER.warning(
                "%s: only %d known-gender authors; target is at least %d",
                row["subfield"],
                row["known_author_count"],
                min_known_authors,
            )
    summary_path = processed_dir / "subfield_representation_summary.csv"
    summary.to_csv(summary_path, index=False)
    contrast_path = processed_dir / "macro_cluster_contrast.csv"
    _macro_cluster_contrast(authors).to_csv(contrast_path, index=False)
    chi_square_path = processed_dir / "subfield_chi_square_tests.csv"
    _chi_square_by_role(authors).to_csv(chi_square_path, index=False)
    dropoff_path = processed_dir / "seniority_dropoff.csv"
    dropoff = _seniority_dropoff(authors)
    dropoff.to_csv(dropoff_path, index=False)
    faculty_path = processed_dir / "faculty_representation_summary.csv"
    faculty_summary = _faculty_summary(project_root)
    faculty_summary.to_csv(faculty_path, index=False)

    outputs = {
        "interim_papers": interim_path,
        "authors": author_path,
        "audit_queue": audit_path,
        "representation_summary": summary_path,
        "macro_cluster_contrast": contrast_path,
        "chi_square_tests": chi_square_path,
        "seniority_dropoff": dropoff_path,
        "faculty_summary": faculty_path,
    }
    if make_plots:
        outputs.update(
            plot_analysis_exports(
                summary,
                dropoff,
                faculty_summary,
                project_root / "figures",
            )
        )

    LOGGER.info("Pipeline complete: %d author records from %d papers", len(authors), len(papers))
    return outputs
