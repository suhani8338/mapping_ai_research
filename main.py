"""Command-line entry point for the research data pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gendered AI Research pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the study configuration YAML file.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating figures.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    outputs = run_pipeline(args.config, make_plots=not args.no_plots)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
