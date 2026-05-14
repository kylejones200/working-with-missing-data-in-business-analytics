#!/usr/bin/env python3
"""Working with missing data — Polars + DuckDB rewrite (no sklearn imputers)."""

import sys
import argparse
import yaml
import logging
import numpy as np
import polars as pl
from datetime import date
from dateutil.relativedelta import relativedelta
from pathlib import Path

from core import analyze_missing_data, impute_missing_values, plot_missing_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_synthetic(seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = 144
    start = date(2020, 1, 1)
    months = [start + relativedelta(months=i) for i in range(n)]
    passengers = 100 + 20 * np.sin(np.arange(n) / 12) + rng.normal(0, 10, n)

    # inject missing values
    null_pass = set(range(10, 16))
    null_rev  = set(range(5, 11))
    pass_list = [None if i in null_pass else float(passengers[i]) for i in range(n)]
    rev_list  = [
        None if i in null_rev
        else float(passengers[i] * 10 + rng.normal(0, 50))
        for i in range(n)
    ]

    return pl.DataFrame({"Month": months, "Passengers": pass_list, "Revenue": rev_list})


def main():
    parser = argparse.ArgumentParser(description="Missing data — Polars + DuckDB")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["figures_dir"])
    output_dir.mkdir(exist_ok=True)

    df = make_synthetic(seed=config["data"]["seed"])
    numeric_cols = ["Passengers", "Revenue"]
    strategy = config["imputation"]["strategy"]

    analysis = analyze_missing_data(df)
    logging.info("Missing Data Summary:")
    for col, count in analysis["missing_counts"].items():
        if count > 0:
            logging.info(f"  {col}: {count} ({analysis['missing_percentages'][col]:.2f}%)")

    logging.info(f"Imputing with strategy: {strategy!r}")
    df_imputed = impute_missing_values(df, numeric_cols, strategy=strategy)

    after = analyze_missing_data(df_imputed)
    logging.info(f"Missing after imputation: {after['total_missing']}")

    df_imputed.write_csv(output_dir / "imputed_output.csv")
    logging.info(f"Imputed data saved → {output_dir / 'imputed_output.csv'}")

    plot_missing_analysis(df, "Missing Data (before)", output_dir / "missing_analysis.png")

    logging.info(f"Analysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
