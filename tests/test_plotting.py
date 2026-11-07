import pandas as pd
import pytest

from src.analysis.plotting import plot_analysis_exports


def test_analysis_exports_create_four_png_and_pdf_plots(tmp_path):
    representation = pd.DataFrame(
        [
            {
                "subfield": "ethics_hci",
                "first_author_female_pct": 39.3,
                "first_author_coverage": 0.54,
                "first_author_known_count": 229,
                "last_author_female_pct": 30.6,
                "last_author_coverage": 0.62,
                "last_author_known_count": 265,
            },
            {
                "subfield": "nlp",
                "first_author_female_pct": 20.3,
                "first_author_coverage": 0.32,
                "first_author_known_count": 148,
                "last_author_female_pct": 15.7,
                "last_author_coverage": 0.46,
                "last_author_known_count": 216,
            },
        ]
    )
    dropoff = pd.DataFrame(
        [
            {
                "subfield": "ethics_hci",
                "p_first": 0.393,
                "p_last": 0.306,
                "delta": 0.087,
                "p_value": 0.0418,
            },
            {
                "subfield": "nlp",
                "p_first": 0.203,
                "p_last": 0.157,
                "delta": 0.046,
                "p_value": 0.265,
            },
        ]
    )
    faculty = pd.DataFrame(
        [
            {"subfield": "nlp", "known_count": 20, "female_pct": 25.0},
            {"subfield": "all", "known_count": 80, "female_pct": 30.0},
        ]
    )

    outputs = plot_analysis_exports(representation, dropoff, faculty, tmp_path)

    assert set(outputs) == {
        "pipeline_dropoff_png",
        "pipeline_dropoff_pdf",
        "odds_ratio_forest_png",
        "odds_ratio_forest_pdf",
        "author_coverage_png",
        "author_coverage_pdf",
        "career_stage_comparison_png",
        "career_stage_comparison_pdf",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())


def test_analysis_exports_accept_empty_faculty_summary(tmp_path):
    representation = pd.DataFrame(
        [
            {
                "subfield": "vision",
                "first_author_female_pct": 16.3,
                "first_author_coverage": 0.26,
                "first_author_known_count": 123,
                "last_author_female_pct": 17.4,
                "last_author_coverage": 0.36,
                "last_author_known_count": 172,
            }
        ]
    )
    dropoff = pd.DataFrame(
        [
            {
                "subfield": "vision",
                "p_first": 0.163,
                "p_last": 0.174,
                "delta": -0.011,
                "p_value": 0.8,
            }
        ]
    )

    outputs = plot_analysis_exports(
        representation,
        dropoff,
        pd.DataFrame(columns=["subfield", "known_count", "female_pct"]),
        tmp_path,
    )

    assert outputs["career_stage_comparison_png"].is_file()
    assert outputs["career_stage_comparison_pdf"].is_file()


def test_invalid_coverage_data_is_rejected(tmp_path):
    representation = pd.DataFrame(
        [
            {
                "subfield": "vision",
                "first_author_female_pct": 16.3,
                "first_author_coverage": 1.2,
                "first_author_known_count": 123,
            }
        ]
    )

    from src.analysis.plotting import plot_author_coverage_exports

    with pytest.raises(ValueError, match="coverage must be"):
        plot_author_coverage_exports(representation, tmp_path / "coverage")
