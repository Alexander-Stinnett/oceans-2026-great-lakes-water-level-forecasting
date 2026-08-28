from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

RAW_COLUMNS = ("wl_lake", "temp_lake", "temp_land", "wspd_lake", "wspd_land", "lst", "Precp")
TEMPERATURE_COLUMNS = ("temp_lake", "temp_land")
UPSTREAM_FILENAMES = ("finaldata.csv", "test_2022.csv", "test_2023.csv")
WINDOWS = (30, 60, 90, 120, 150, 180)
CANONICAL_SHA256 = "cdb8c4e0ad99f9c3363e193306580c6403cd30079e42a6e4f8a748555edbe0b9"


def avg_column(raw_column: str, window: int) -> str:
    feature = "precp" if raw_column == "Precp" else raw_column
    return f"avg_{feature}_{window}"


def read_upstream(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(frame.columns) != len(RAW_COLUMNS) + 1:
        raise ValueError(f"{path} has unexpected columns: {list(frame.columns)}")
    if frame.columns[0] != "date":
        frame = frame.rename(columns={frame.columns[0]: "date"})
    expected = ["date", *RAW_COLUMNS]
    if list(frame.columns) != expected:
        raise ValueError(f"{path} columns must be {expected}, found {list(frame.columns)}")
    return frame


def assert_daily_contiguous(frame: pd.DataFrame, label: str) -> None:
    dates = pd.to_datetime(frame["date"], errors="raise")
    expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError(f"{label} dates are not unique and sorted")
    if len(dates) != len(expected) or not (dates.to_numpy() == expected.to_numpy()).all():
        raise ValueError(f"{label} is not daily contiguous")


def load_upstream_directory(data_dir: str | Path) -> pd.DataFrame:
    root = Path(data_dir)
    frame = pd.concat([read_upstream(root / name) for name in UPSTREAM_FILENAMES], ignore_index=True)
    assert_daily_contiguous(frame, "upstream concatenation")
    if len(frame) != 15_705:
        raise ValueError(f"Expected 15,705 upstream rows, found {len(frame)}")
    for column in RAW_COLUMNS:
        pd.to_numeric(frame[column], errors="raise")
    return frame


def build_canonical_frame(frame: pd.DataFrame, *, kelvin_threshold: float = 100.0) -> pd.DataFrame:
    output = frame.copy()
    for column in RAW_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="raise")
    for column in TEMPERATURE_COLUMNS:
        output[column] = output[column].mask(output[column] > kelvin_threshold, output[column] - 273.15)
    for window in WINDOWS:
        for raw_column in RAW_COLUMNS:
            output[avg_column(raw_column, window)] = output[raw_column].rolling(window).mean()
    columns = ["date", *RAW_COLUMNS, *[avg_column(column, window) for window in WINDOWS for column in RAW_COLUMNS]]
    output = output.loc[:, columns].iloc[max(WINDOWS) - 1 :].reset_index(drop=True)
    if output.isna().any().any() or len(output) != 15_526:
        raise ValueError("Canonical dataset has an invalid row count or null values")
    return output


def write_canonical(frame: pd.DataFrame, destination: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(destination).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_dataset(data_dir: str | Path, destination: str | Path, *, overwrite: bool = False) -> Path:
    result = write_canonical(build_canonical_frame(load_upstream_directory(data_dir)), destination, overwrite=overwrite)
    observed = sha256_file(result)
    if observed != CANONICAL_SHA256:
        raise RuntimeError(f"Canonical byte hash mismatch: {observed}")
    return result

