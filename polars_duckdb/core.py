"""Missing data analysis and imputation using Polars and DuckDB.

sklearn SimpleImputer (mean/median) is replaced with DuckDB window aggregates.
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import polars as pl


def analyze_missing_data(df: pl.DataFrame) -> dict:
    null_counts = df.null_count()
    total = df.height
    counts = {c: null_counts[0, c] for c in df.columns}
    return {
        "missing_counts": counts,
        "missing_percentages": {c: v / total * 100 for c, v in counts.items()},
        "total_missing": sum(counts.values()),
    }


def impute_missing_values(
    df: pl.DataFrame,
    numeric_cols: list[str],
    strategy: str = "mean",
) -> pl.DataFrame:
    """
    Impute nulls in numeric_cols.
    mean/median → single DuckDB query using window aggregates.
    forward     → Polars fill_null (equivalent to sklearn's nearest/ffill).
    """
    if strategy in ("mean", "median"):
        fn = "AVG" if strategy == "mean" else "MEDIAN"
        imputed_exprs = ", ".join(
            f'COALESCE("{c}", {fn}("{c}") OVER ()) AS "{c}"' for c in numeric_cols
        )
        passthrough = ", ".join(f'"{c}"' for c in df.columns if c not in numeric_cols)
        select_clause = ", ".join(filter(None, [passthrough, imputed_exprs]))
        # Preserve original column order
        result = duckdb.sql(f"SELECT {select_clause} FROM df").pl()
        return result.select(df.columns)

    if strategy == "forward":
        return df.with_columns(
            [
                pl.col(c).fill_null(strategy="forward").fill_null(strategy="backward")
                for c in numeric_cols
            ]
        )

    raise ValueError(
        f"Unknown strategy: {strategy!r}. Use 'mean', 'median', or 'forward'."
    )


def plot_missing_analysis(
    df: pl.DataFrame, title: str, output_path: Path, plot: bool = False
):
    null_counts = df.null_count()
    cols = df.columns
    counts = [null_counts[0, c] for c in cols]
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        if any(c > 0 for c in counts):
            ax.bar(
                range(len(cols)), counts, color="#4A90A4", alpha=0.7, edgecolor="none"
            )
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha="right")
        else:
            ax.text(
                0.5,
                0.5,
                "No missing values",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )

        ax.set_xlabel("Column")
        ax.set_ylabel("Missing Count")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
