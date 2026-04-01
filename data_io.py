"""Common data ingestion utilities for time series examples."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with basic logging.

    Args:
        path: File path to the CSV.
        **kwargs: Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns:
        Loaded ``pandas.DataFrame``.
    """
    csv_path = Path(path)
    logger.info("Reading CSV from '%s'", csv_path)
    return pd.read_csv(csv_path, **kwargs)


