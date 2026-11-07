"""Statistical plotting helpers; matplotlib is imported only when a plot is requested."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(sorted(missing))}")


def _save_exports(figure: Any, output_stem: str | Path) -> tuple[Path, Path]:
    stem = Path(output_stem)
    destinations = (stem.with_suffix(".png"), stem.with_suffix(".pdf"))
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        for destination in destinations:
            figure.savefig(destination, dpi=300, bbox_inches="tight")
    finally:
        _pyplot().close(figure)
    return destinations


def plot_pipeline_dumbbell_exports(
    dropoff: pd.DataFrame, output_stem: str | Path
) -> tuple[Path, Path]:
    """Plot first-to-last-author female representation by subfield."""
    _require_columns(
        dropoff, {"subfield", "p_first", "p_last", "delta", "p_value"}, "dropoff"
    )
    df = dropoff.dropna(subset=["p_first", "p_last", "delta"]).sort_values(
        "delta", ascending=False
    )
    if df.empty:
        raise ValueError("dropoff must contain at least one complete subfield row")

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, max(4.5, 0.62 * len(df))))
    y_positions = range(len(df))
    for y, (_, row) in zip(y_positions, df.iterrows()):
        first = float(row["p_first"]) * 100
        last = float(row["p_last"]) * 100
        p_value = float(row["p_value"])
        significant = math.isfinite(p_value) and p_value < 0.05
        color = "#d95f02" if significant else "#7570b3"
        axis.plot([last, first], [y, y], color=color, linewidth=2.5, alpha=0.85)
        delta = float(row["delta"]) * 100
        label = f"\u0394={delta:+.1f}%" + ("*" if significant else "")
        axis.text((first + last) / 2, y + 0.18, label, ha="center", fontsize=8.5, color=color)

    axis.scatter(
        df["p_first"].astype(float) * 100,
        list(y_positions),
        color="#1b9e77",
        s=65,
        label="First author",
        zorder=3,
    )
    axis.scatter(
        df["p_last"].astype(float) * 100,
        list(y_positions),
        color="#d95f02",
        s=65,
        label="Last author",
        zorder=3,
    )
    axis.set_yticks(list(y_positions), df["subfield"].astype(str))
    axis.set_xlabel("Female representation among known-gender authors (%)")
    axis.set_xlim(5, 45)
    axis.grid(axis="x", linestyle="--", alpha=0.45)
    axis.legend(frameon=True, loc="lower left")
    axis.set_title("First-to-last author representation by subfield")
    axis.set_ylim(-0.5, len(df) - 0.15)
    figure.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.15)
    return _save_exports(figure, output_stem)


def plot_odds_ratio_forest_exports(
    representation: pd.DataFrame, output_stem: str | Path
) -> tuple[Path, Path]:
    """Plot last-author odds ratios against the pooled AI sample with Wald intervals."""
    _require_columns(
        representation,
        {"subfield", "last_author_female_pct", "last_author_known_count"},
        "representation",
    )
    rows: list[dict[str, float | str]] = []
    for _, row in representation.iterrows():
        count = int(row["last_author_known_count"])
        pct = float(row["last_author_female_pct"])
        if count <= 0 or not math.isfinite(pct):
            continue
        female = round(count * pct / 100)
        rows.append(
            {
                "subfield": str(row["subfield"]),
                "female": float(female),
                "male": float(count - female),
            }
        )
    if not rows:
        raise ValueError("representation must contain known last-author counts")

    total_female = sum(row["female"] for row in rows)
    total_male = sum(row["male"] for row in rows)
    overall_odds = (total_female + 0.5) / (total_male + 0.5)
    for row in rows:
        female = row["female"] + 0.5
        male = row["male"] + 0.5
        odds_ratio = (female / male) / overall_odds
        standard_error = math.sqrt(
            1 / female + 1 / male + 1 / (total_female + 0.5) + 1 / (total_male + 0.5)
        )
        row["odds_ratio"] = odds_ratio
        row["lower"] = math.exp(math.log(odds_ratio) - 1.96 * standard_error)
        row["upper"] = math.exp(math.log(odds_ratio) + 1.96 * standard_error)
    rows.sort(key=lambda result: float(result["odds_ratio"]))

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, max(4.5, 0.58 * len(rows))))
    y_positions = list(range(len(rows)))
    for y, row in zip(y_positions, rows):
        odds_ratio = float(row["odds_ratio"])
        lower = float(row["lower"])
        upper = float(row["upper"])
        axis.errorbar(
            odds_ratio,
            y,
            xerr=[[odds_ratio - lower], [upper - odds_ratio]],
            fmt="o",
            color="#2c7fb8",
            ecolor="#7a9eaf",
            capsize=3,
        )
    axis.axvline(1, color="#555555", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yticks(y_positions, [str(row["subfield"]) for row in rows])
    axis.set_xlabel("Odds ratio vs. pooled AI sample (95% Wald CI)")
    axis.set_title("Last-author female representation relative to the AI average")
    axis.grid(axis="x", which="both", linestyle=":", alpha=0.4)
    figure.tight_layout()
    return _save_exports(figure, output_stem)


def plot_author_coverage_exports(
    representation: pd.DataFrame, output_stem: str | Path
) -> tuple[Path, Path]:
    """Show inferred female/male and unknown shares of all first-author names."""
    _require_columns(
        representation,
        {
            "subfield",
            "first_author_female_pct",
            "first_author_coverage",
            "first_author_known_count",
        },
        "representation",
    )
    rows: list[dict[str, float | str | int]] = []
    for _, row in representation.iterrows():
        known = int(row["first_author_known_count"])
        coverage = float(row["first_author_coverage"])
        female_pct = float(row["first_author_female_pct"])
        if known <= 0 or not math.isfinite(coverage) or coverage <= 0:
            continue
        if coverage > 1 or not math.isfinite(female_pct) or not 0 <= female_pct <= 100:
            raise ValueError("coverage must be in (0, 1] and female percentages in [0, 100]")
        total = known / coverage
        female = known * female_pct / 100
        male = known - female
        rows.append(
            {
                "subfield": str(row["subfield"]),
                "known": known,
                "total": total,
                "female_pct": female / total * 100,
                "male_pct": male / total * 100,
                "unknown_pct": (total - known) / total * 100,
            }
        )
    if not rows:
        raise ValueError("representation must contain first-author coverage data")
    rows.sort(key=lambda result: float(result["unknown_pct"]), reverse=True)

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, max(4.2, 0.55 * len(rows))))
    y_positions = list(range(len(rows)))
    left = [0.0] * len(rows)
    for key, label, color in (
        ("female_pct", "Female (inferred)", "#2ca25f"),
        ("male_pct", "Male (inferred)", "#43a2ca"),
        ("unknown_pct", "Unknown / ambiguous", "#d9d9d9"),
    ):
        values = [float(row[key]) for row in rows]
        axis.barh(y_positions, values, left=left, color=color, label=label)
        left = [prior + value for prior, value in zip(left, values)]
    axis.set_yticks(
        y_positions,
        [f"{row['subfield']} (known N={row['known']})" for row in rows],
    )
    axis.set_xlabel("Share of all first-author names (%)")
    axis.set_xlim(0, 100)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    axis.set_title("First-author classification coverage")
    figure.tight_layout()
    return _save_exports(figure, output_stem)


def plot_career_stage_exports(
    representation: pd.DataFrame,
    faculty: pd.DataFrame,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    """Compare first/last arXiv authors and faculty with binomial SE error bars."""
    _require_columns(
        representation,
        {
            "subfield",
            "first_author_female_pct",
            "first_author_known_count",
            "last_author_female_pct",
            "last_author_known_count",
        },
        "representation",
    )
    _require_columns(
        faculty, {"subfield", "female_pct", "known_count"}, "faculty"
    )
    has_pooled_faculty = any(
        str(row["subfield"]) == "all"
        and int(row["known_count"]) > 0
        and math.isfinite(float(row["female_pct"]))
        for _, row in faculty.iterrows()
    )
    subfields = list(representation["subfield"].astype(str))
    if has_pooled_faculty and "all" not in subfields:
        subfields.append("all")
    subfields = list(dict.fromkeys(subfields))
    if not subfields:
        raise ValueError("representation must contain at least one subfield")
    x_by_subfield = {subfield: index for index, subfield in enumerate(subfields)}

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(max(9, 1.25 * len(subfields)), 5.2))
    stage_specs = (
        ("first_author_female_pct", "first_author_known_count", "First author", "D", -0.2, "#1b9e77"),
        ("last_author_female_pct", "last_author_known_count", "Last author", "o", 0, "#7570b3"),
    )
    plotted = False
    for pct_column, count_column, label, marker, offset, color in stage_specs:
        positions: list[float] = []
        percentages: list[float] = []
        errors: list[float] = []
        for _, row in representation.iterrows():
            count = int(row[count_column])
            pct = float(row[pct_column])
            if count <= 0 or not math.isfinite(pct):
                continue
            proportion = pct / 100
            if not 0 <= proportion <= 1:
                raise ValueError("female percentages must be between 0 and 100")
            positions.append(x_by_subfield[str(row["subfield"])] + offset)
            percentages.append(pct)
            errors.append(math.sqrt(proportion * (1 - proportion) / count) * 100)
        if has_pooled_faculty:
            total_count = int(representation[count_column].sum())
            if total_count > 0:
                total_female = sum(
                    int(row[count_column]) * float(row[pct_column]) / 100
                    for _, row in representation.iterrows()
                    if int(row[count_column]) > 0 and math.isfinite(float(row[pct_column]))
                )
                proportion = total_female / total_count
                positions.append(x_by_subfield["all"] + offset)
                percentages.append(proportion * 100)
                errors.append(
                    math.sqrt(proportion * (1 - proportion) / total_count) * 100
                )
        if percentages:
            axis.errorbar(
                positions,
                percentages,
                yerr=errors,
                fmt=marker,
                color=color,
                capsize=3,
                label=label,
            )
            plotted = True

    faculty_positions: list[float] = []
    faculty_percentages: list[float] = []
    faculty_errors: list[float] = []
    for _, row in faculty.iterrows():
        count = int(row["known_count"])
        pct = float(row["female_pct"])
        subfield = str(row["subfield"])
        if count <= 0 or not math.isfinite(pct):
            continue
        if subfield not in x_by_subfield:
            continue
        proportion = pct / 100
        if not 0 <= proportion <= 1:
            raise ValueError("faculty female percentages must be between 0 and 100")
        faculty_positions.append(x_by_subfield[subfield] + 0.2)
        faculty_percentages.append(pct)
        faculty_errors.append(math.sqrt(proportion * (1 - proportion) / count) * 100)
    if faculty_percentages:
        axis.errorbar(
            faculty_positions,
            faculty_percentages,
            yerr=faculty_errors,
            fmt="s",
            color="#d95f02",
            capsize=3,
            label="Faculty",
        )
        plotted = True

    if not plotted:
        raise ValueError("no known-gender observations are available to plot")
    if not faculty_percentages:
        axis.text(
            0.99,
            0.97,
            "Faculty data unavailable",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#666666",
        )
    axis.set_xticks(range(len(subfields)), subfields, rotation=30, ha="right")
    axis.set_ylabel("Female representation among known-gender observations (%)")
    axis.set_ylim(0, 45)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(frameon=False)
    axis.set_title("Female representation by career stage")
    figure.subplots_adjust(bottom=0.25, top=0.88, left=0.08, right=0.98)
    return _save_exports(figure, output_stem)


def plot_analysis_exports(
    representation: pd.DataFrame,
    dropoff: pd.DataFrame,
    faculty: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save all four analysis figures as PNG and PDF."""
    directory = Path(output_dir)
    plots = (
        ("pipeline_dropoff", plot_pipeline_dumbbell_exports, (dropoff,)),
        ("odds_ratio_forest", plot_odds_ratio_forest_exports, (representation,)),
        ("author_coverage", plot_author_coverage_exports, (representation,)),
        ("career_stage_comparison", plot_career_stage_exports, (representation, faculty)),
    )
    outputs: dict[str, Path] = {}
    for stem, plot_function, data in plots:
        png_path, pdf_path = plot_function(*data, directory / stem)
        outputs[f"{stem}_png"] = png_path
        outputs[f"{stem}_pdf"] = pdf_path
    return outputs
