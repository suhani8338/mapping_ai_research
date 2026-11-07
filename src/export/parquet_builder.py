"""Normalize pipeline CSV outputs into compact, validated Parquet datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.enrichment.countries import is_iso_country_code
from src.export.schemas import (
    AUTHORS_ENRICHED_SCHEMA,
    GENDER_LABELS,
    PAPERS_SCHEMA,
)

LOGGER = logging.getLogger(__name__)
AUTHOR_ORDER_LABELS = frozenset({"first", "middle", "last", "sole"})
MAX_AUTHORS_PARQUET_BYTES = 15_000_000


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return frame


def _clean_text(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    result = str(value).strip()
    return result if result else default


def _integer(value: Any, *, field: str, default: int) -> int:
    text = _clean_text(value)
    if not text:
        return default
    try:
        number = int(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer, got {value!r}") from error
    return number


def _bounded_integer(
    value: Any, *, field: str, default: int, minimum: int, maximum: int
) -> int:
    number = _integer(value, field=field, default=default)
    if not minimum <= number <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}, got {number}")
    return number


def _gender(value: Any, *, row_number: int) -> str:
    label = _clean_text(value, "unknown").lower()
    if label not in GENDER_LABELS:
        raise ValueError(
            f"Invalid inferred_gender at author CSV row {row_number}: {label!r}"
        )
    return label


def _confidence(value: Any, *, row_number: int) -> float:
    text = _clean_text(value)
    if not text:
        return 0.0
    try:
        confidence = float(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid confidence at author CSV row {row_number}: {value!r}"
        ) from error
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Confidence at author CSV row {row_number} must be between 0 and 1"
        )
    return confidence


def _publication_date(value: Any, *, row_number: int) -> str:
    text = _clean_text(value)
    try:
        return pd.Timestamp(text).date().isoformat()
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid published date at paper CSV row {row_number}: {value!r}"
        ) from error


def _arrow_table(
    rows: list[dict[str, Any]], schema: pa.Schema, *, dataset: str
) -> pa.Table:
    try:
        return pa.Table.from_pylist(rows, schema=schema)
    except (pa.ArrowException, OverflowError) as error:
        raise ValueError(f"Could not convert {dataset} rows to the declared schema") from error


def _merge_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in (*existing_rows, *new_rows):
        merged[tuple(row[field] for field in key_fields)] = row
    return list(merged.values())


def build_parquet(
    project_root: str | Path = ".",
    *,
    authors_csv: str | Path | None = None,
    papers_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    append: bool = False,
) -> tuple[Path, Path]:
    """Build Parquet datasets from pipeline CSVs, optionally merging an existing batch."""
    root = Path(project_root)
    author_path = Path(authors_csv) if authors_csv is not None else (
        root / "data" / "processed" / "arxiv_authors_final.csv"
    )
    paper_path = Path(papers_csv) if papers_csv is not None else (
        root / "data" / "interim" / "arxiv_papers.csv"
    )
    destination = Path(output_dir) if output_dir is not None else (
        root / "frontend" / "public" / "data"
    )
    author_output = destination / "authors_enriched.parquet"
    paper_output = destination / "papers.parquet"

    authors = _read_csv(
        author_path,
        {"paper_id", "subfield", "macro_cluster", "author_order", "full_name"},
    )
    papers = _read_csv(
        paper_path,
        {"paper_id", "subfield", "macro_cluster", "published", "title", "authors"},
    )

    paper_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    paper_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(papers.to_dict(orient="records"), start=2):
        paper_id = _clean_text(row["paper_id"])
        subfield = _clean_text(row["subfield"])
        key = (paper_id, subfield)
        if not paper_id or not subfield:
            raise ValueError(f"Paper CSV row {row_number} has an empty paper_id or subfield")
        if key in paper_by_key:
            raise ValueError(
                f"Duplicate paper_id/subfield at paper CSV row {row_number}: {key}"
            )
        publication_date = _publication_date(row["published"], row_number=row_number)
        listed_authors = [
            name.strip()
            for name in _clean_text(row["authors"]).split("|")
            if name.strip()
        ]
        paper = {
            "paper_id": paper_id,
            "subfield": subfield,
            "macro_cluster": _clean_text(row["macro_cluster"], "unknown"),
            "publication_year": int(publication_date[:4]),
        }
        paper_by_key[key] = paper
        paper_rows.append(
            {
                "paper_id": paper_id,
                "title": _clean_text(row["title"], "Unknown"),
                "primary_category": _clean_text(row.get("primary_category"), "UNKNOWN"),
                "subfield": subfield,
                "macro_cluster": paper["macro_cluster"],
                "publication_date": publication_date,
                "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
                "total_authors": _bounded_integer(
                    len(listed_authors),
                    field=f"total_authors at paper CSV row {row_number}",
                    default=0,
                    minimum=0,
                    maximum=32767,
                ),
            }
        )

    author_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(authors.to_dict(orient="records"), start=2):
        key = (_clean_text(row["paper_id"]), _clean_text(row["subfield"]))
        paper = paper_by_key.get(key)
        if paper is None:
            raise ValueError(
                f"Author CSV row {row_number} does not match a paper by paper_id/subfield: {key}"
            )
        order = _clean_text(row["author_order"]).lower()
        if order not in AUTHOR_ORDER_LABELS:
            raise ValueError(
                f"Invalid author_order at author CSV row {row_number}: {order!r}"
            )
        country = _clean_text(row.get("country_code", row.get("country")), "UNKNOWN").upper()
        country = country if is_iso_country_code(country) else "UNKNOWN"
        age = _bounded_integer(
            row.get("academic_age"),
            field=f"academic_age at author CSV row {row_number}",
            default=-1,
            minimum=-1,
            maximum=32767,
        )
        author_rows.append(
            {
                "paper_id": key[0],
                "subfield": key[1],
                "macro_cluster": _clean_text(
                    row["macro_cluster"], paper["macro_cluster"]
                ),
                "publication_year": paper["publication_year"],
                "author_order": order,
                "full_name": _clean_text(row["full_name"], "unknown"),
                "gender": _gender(row.get("inferred_gender"), row_number=row_number),
                "gender_confidence": _confidence(
                    row.get("confidence"), row_number=row_number
                ),
                "country_code": country,
                "academic_age": age,
                "academic_cohort": _clean_text(
                    row.get("academic_cohort"), "unknown"
                ).lower(),
            }
        )

    if append:
        existing_outputs = (author_output.is_file(), paper_output.is_file())
        if existing_outputs[0] != existing_outputs[1]:
            raise ValueError(
                "Cannot append unless both existing author and paper Parquet files are present"
            )
        if all(existing_outputs):
            existing_authors = pq.read_table(author_output).to_pylist()
            existing_papers = pq.read_table(paper_output).to_pylist()
            author_rows = _merge_rows(
                existing_authors,
                author_rows,
                ("paper_id", "subfield", "author_order", "full_name"),
            )
            paper_rows = _merge_rows(
                existing_papers, paper_rows, ("paper_id", "subfield")
            )

    for dataset, rows, schema in (
        ("authors", author_rows, AUTHORS_ENRICHED_SCHEMA),
        ("papers", paper_rows, PAPERS_SCHEMA),
    ):
        for field in schema:
            if field.type == pa.int16():
                for row_number, row in enumerate(rows, start=2):
                    value = row[field.name]
                    if not -32768 <= value <= 32767:
                        raise ValueError(
                            f"{field.name} at {dataset} row {row_number} exceeds int16 range"
                        )

    author_table = _arrow_table(author_rows, AUTHORS_ENRICHED_SCHEMA, dataset="authors")
    paper_table = _arrow_table(paper_rows, PAPERS_SCHEMA, dataset="papers")
    destination.mkdir(parents=True, exist_ok=True)
    dictionary_columns = [
        "subfield",
        "macro_cluster",
        "author_order",
        "gender",
        "country_code",
        "academic_cohort",
    ]
    pq.write_table(
        author_table,
        author_output,
        compression="snappy",
        use_dictionary=dictionary_columns,
    )
    pq.write_table(
        paper_table,
        paper_output,
        compression="snappy",
        use_dictionary=["primary_category", "subfield", "macro_cluster"],
    )
    author_output_size = author_output.stat().st_size
    if author_output_size >= MAX_AUTHORS_PARQUET_BYTES:
        raise ValueError(
            "authors_enriched.parquet must be below "
            f"{MAX_AUTHORS_PARQUET_BYTES} bytes; got {author_output_size}"
        )

    author_frame = pd.DataFrame(author_rows)
    unknown_rates = (
        author_frame.assign(is_unknown=author_frame["gender"].eq("unknown"))
        .groupby("subfield", sort=True)["is_unknown"]
        .mean()
    )
    print(f"Authors: {len(author_rows)} rows -> {author_output}")
    print(f"Papers: {len(paper_rows)} rows -> {paper_output}")
    print("Unclassified author rates by subfield:")
    for subfield, rate in unknown_rates.items():
        print(f"  {subfield}: {rate:.1%}")
    return author_output, paper_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frontend Parquet datasets from processed pipeline CSVs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data/interim and data/processed.",
    )
    parser.add_argument("--authors-csv", type=Path)
    parser.add_argument("--papers-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge the CSV batch into existing Parquet files without duplicate records.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_parquet(
        args.project_root,
        authors_csv=args.authors_csv,
        papers_csv=args.papers_csv,
        output_dir=args.output_dir,
        append=args.append,
    )


if __name__ == "__main__":
    main()
