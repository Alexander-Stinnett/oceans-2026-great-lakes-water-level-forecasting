from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from oceans_glwl.config import (
    DATE_COLUMN,
    RAW_BASE_COLUMNS,
    RAW_EXOG_COLUMNS,
    SMOOTHED_EXOG_COLUMNS,
    TARGET_COLUMN,
    TRAILING_30_DAY_COLUMNS,
)

TRAILING_MEAN_PAIRS = (("wl_lake", TARGET_COLUMN), *zip(RAW_EXOG_COLUMNS, SMOOTHED_EXOG_COLUMNS, strict=True))


def load_frame(path: str | Path, *, verify_causal: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)
    validate_frame(frame, verify_causal=verify_causal)
    return frame


def validate_frame(frame: pd.DataFrame, *, verify_causal: bool = True) -> None:
    required = [DATE_COLUMN, *RAW_BASE_COLUMNS, *TRAILING_30_DAY_COLUMNS]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}.")
    assert_daily_contiguous(frame)
    numeric = frame[[*RAW_BASE_COLUMNS, *TRAILING_30_DAY_COLUMNS]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Required target/exogenous columns must be finite.")
    if verify_causal:
        verify_causal_rolling_columns(frame)


def assert_daily_contiguous(frame: pd.DataFrame) -> None:
    dates = pd.to_datetime(frame[DATE_COLUMN])
    if dates.duplicated().any():
        raise ValueError("Dataset contains duplicate dates.")
    deltas = dates.diff().dropna().dt.days.to_numpy(dtype=np.int64)
    if deltas.size and not np.all(deltas == 1):
        raise ValueError("Dataset must be daily and contiguous.")


def verify_causal_rolling_columns(frame: pd.DataFrame, *, window: int = 30, atol: float = 1e-10) -> None:
    if int(window) <= 0:
        raise ValueError("window must be positive.")
    start = int(window) - 1
    for raw_column, smooth_column in TRAILING_MEAN_PAIRS:
        expected = pd.to_numeric(frame[raw_column], errors="coerce").rolling(window).mean().to_numpy(dtype=np.float64)
        actual = pd.to_numeric(frame[smooth_column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.allclose(expected[start:], actual[start:], atol=atol, rtol=0.0):
            raise ValueError(f"{smooth_column} is not a causal trailing-{window}-day mean of {raw_column}.")


def synthetic_daily_frame(rows: int = 720) -> pd.DataFrame:
    """Create deterministic contract-valid data for smoke tests only."""
    if rows < 400:
        raise ValueError("Synthetic smoke data requires at least 400 rows.")
    index = np.arange(rows, dtype=np.float64)
    frame = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2000-01-01", periods=rows, freq="D"),
            "wl_lake": 1.5 + 0.001 * index + 0.05 * np.sin(index / 20.0),
            "temp_lake": 10.0 + np.sin(index / 30.0),
            "temp_land": 11.0 + np.cos(index / 31.0),
            "wspd_lake": 4.0 + 0.2 * np.sin(index / 8.0),
            "wspd_land": 3.0 + 0.2 * np.cos(index / 9.0),
            "lst": 9.0 + np.sin(index / 27.0),
            "Precp": 1.0 + 0.1 * np.cos(index / 11.0),
        }
    )
    for raw_column, smooth_column in TRAILING_MEAN_PAIRS:
        frame[smooth_column] = frame[raw_column].rolling(30, min_periods=1).mean()
    return frame
