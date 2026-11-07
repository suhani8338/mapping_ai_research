import pyarrow.parquet as pq
import pytest

import src.export.parquet_builder as parquet_builder
from src.export.parquet_builder import build_parquet


def _write_inputs(tmp_path, *, gender="unknown", country="IN"):
    authors_path = tmp_path / "authors.csv"
    papers_path = tmp_path / "papers.csv"
    authors_path.write_text(
        "paper_id,subfield,macro_cluster,author_order,full_name,inferred_gender,"
        "confidence,country,academic_age,academic_cohort\n"
        f"0012.34000,nlp,applied_socio_technical,sole,Ada Lovelace,{gender},"
        f"0.95,{country},4,mid\n",
        encoding="utf-8",
    )
    papers_path.write_text(
        "paper_id,subfield,macro_cluster,published,title,authors\n"
        '0012.34000,nlp,applied_socio_technical,2025-03-04T12:00:00Z,Paper title,'
        '"Ada Lovelace"\n',
        encoding="utf-8",
    )
    return authors_path, papers_path


def test_build_parquet_normalizes_and_preserves_output_schema(tmp_path, capsys):
    authors, papers = _write_inputs(tmp_path, country="")
    output = tmp_path / "frontend" / "public" / "data"

    author_path, paper_path = build_parquet(
        tmp_path, authors_csv=authors, papers_csv=papers, output_dir=output
    )

    author_table = pq.read_table(author_path)
    paper_table = pq.read_table(paper_path)
    author = author_table.to_pylist()[0]
    paper = paper_table.to_pylist()[0]
    assert author_table.schema.field("publication_year").type == "int32"
    assert author_table.schema.field("academic_age").type == "int16"
    assert author["paper_id"] == "0012.34000"
    assert author["gender"] == "unknown"
    assert author["country_code"] == "UNKNOWN"
    assert author["publication_year"] == 2025
    assert paper["primary_category"] == "UNKNOWN"
    assert paper["publication_date"] == "2025-03-04"
    assert paper["arxiv_url"] == "https://arxiv.org/abs/0012.34000"
    assert paper["total_authors"] == 1
    assert "nlp: 100.0%" in capsys.readouterr().out


def test_build_parquet_rejects_gender_outside_allowed_labels(tmp_path):
    authors, papers = _write_inputs(tmp_path, gender="possibly_female")

    with pytest.raises(ValueError, match="Invalid inferred_gender"):
        build_parquet(
            tmp_path,
            authors_csv=authors,
            papers_csv=papers,
            output_dir=tmp_path / "out",
        )


def test_build_parquet_rejects_author_dataset_at_size_limit(tmp_path, monkeypatch):
    authors, papers = _write_inputs(tmp_path)
    monkeypatch.setattr(parquet_builder, "MAX_AUTHORS_PARQUET_BYTES", 1)

    with pytest.raises(ValueError, match="must be below"):
        build_parquet(
            tmp_path,
            authors_csv=authors,
            papers_csv=papers,
            output_dir=tmp_path / "out",
        )


def test_build_parquet_joins_repeated_paper_ids_by_subfield(tmp_path):
    authors_path = tmp_path / "authors.csv"
    papers_path = tmp_path / "papers.csv"
    authors_path.write_text(
        "paper_id,subfield,macro_cluster,author_order,full_name,inferred_gender\n"
        "0012.34000,nlp,applied_socio_technical,sole,Ada,unknown\n"
        "0012.34000,vision,core_infrastructure,sole,Grace,female\n",
        encoding="utf-8",
    )
    papers_path.write_text(
        "paper_id,subfield,macro_cluster,published,title,authors\n"
        '0012.34000,nlp,applied_socio_technical,2025-03-04,Paper A,Ada\n'
        '0012.34000,vision,core_infrastructure,2024-06-07,Paper B,Grace\n',
        encoding="utf-8",
    )

    author_path, _ = build_parquet(
        tmp_path,
        authors_csv=authors_path,
        papers_csv=papers_path,
        output_dir=tmp_path / "out",
    )

    rows = pq.read_table(author_path).to_pylist()
    assert [(row["subfield"], row["publication_year"]) for row in rows] == [
        ("nlp", 2025),
        ("vision", 2024),
    ]
