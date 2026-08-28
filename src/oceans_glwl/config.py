from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

EXPERIMENT_NAME = "oceans_final_search"
PROTOCOL_VERSION = "oceans_final_search_v1"
DATE_COLUMN = "date"
TARGET_COLUMN = "avg_wl_lake_30"
RAW_EXOG_COLUMNS = ("temp_lake", "temp_land", "wspd_lake", "wspd_land", "lst", "Precp")
SMOOTHED_EXOG_COLUMNS = (
    "avg_temp_lake_30",
    "avg_temp_land_30",
    "avg_wspd_lake_30",
    "avg_wspd_land_30",
    "avg_lst_30",
    "avg_precp_30",
)
RAW_BASE_COLUMNS = ("wl_lake", *RAW_EXOG_COLUMNS)
TRAILING_30_DAY_COLUMNS = (TARGET_COLUMN, *SMOOTHED_EXOG_COLUMNS)
INPUT_VARIANTS = {
    "daily_smoothed_y_raw_exog",
    "daily_smoothed_y_smoothed_exog",
    "sparse_30d_smoothed_all",
}
OUTPUT_MODES = {
    "seq2one_30d",
    "seq2one_60d",
    "seq2one_90d",
    "seq2one_120d",
    "seq2one_150d",
    "seq2one_180d",
    "seq2seq_6x30d",
}
INPUT_GRID_POLICIES = {"origin_relative", "row_zero_anchored"}
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.10
    test_fraction: float = 0.20
    train_size: int | None = None
    validation_size: int | None = None
    test_size: int | None = None

    def uses_explicit_sizes(self) -> bool:
        return any(value is not None for value in (self.train_size, self.validation_size, self.test_size))


@dataclass(frozen=True)
class WindowConfig:
    input_variant: str = "sparse_30d_smoothed_all"
    context_days: int = 180
    input_gap_days: int = 30
    horizon_days: int = 180
    output_mode: str = "seq2seq_6x30d"
    output_gap_days: int = 30
    input_grid: str = "origin_relative"
    target_col: str = TARGET_COLUMN
    date_col: str = DATE_COLUMN

    def __post_init__(self) -> None:
        validate_window_config(self)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(feature_columns_for(self.input_variant, target_col=self.target_col))

    @property
    def lead_days(self) -> tuple[int, ...]:
        return tuple(lead_days_for_output_mode(self.output_mode, horizon_days=self.horizon_days))

    @property
    def context_rows(self) -> int:
        return context_row_count(context_days=self.context_days, input_gap_days=self.input_gap_days)


def validate_window_config(config: WindowConfig) -> None:
    if config.input_variant not in INPUT_VARIANTS:
        raise ValueError(f"Unsupported input_variant={config.input_variant!r}.")
    if config.output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unsupported output_mode={config.output_mode!r}.")
    if config.input_grid not in INPUT_GRID_POLICIES:
        raise ValueError(f"Unsupported input_grid={config.input_grid!r}.")
    for name, value in {
        "context_days": config.context_days,
        "input_gap_days": config.input_gap_days,
        "horizon_days": config.horizon_days,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")
    if _parse_seq2one_days(config.output_mode) not in {None, int(config.horizon_days)}:
        raise ValueError(f"{config.output_mode} does not end at horizon_days={config.horizon_days}.")
    if config.output_mode == "seq2seq_6x30d" and int(config.horizon_days) != 180:
        raise ValueError("seq2seq_6x30d requires horizon_days=180.")


def feature_columns_for(input_variant: str, *, target_col: str = TARGET_COLUMN) -> list[str]:
    if input_variant == "daily_smoothed_y_raw_exog":
        return [target_col, *RAW_EXOG_COLUMNS]
    if input_variant in {"daily_smoothed_y_smoothed_exog", "sparse_30d_smoothed_all"}:
        return [target_col, *SMOOTHED_EXOG_COLUMNS]
    raise ValueError(f"Unsupported input_variant={input_variant!r}.")


def lead_days_for_output_mode(output_mode: str, *, horizon_days: int) -> list[int]:
    point_lead = _parse_seq2one_days(output_mode)
    if point_lead is not None:
        if point_lead != int(horizon_days):
            raise ValueError(f"{output_mode} does not match horizon_days={horizon_days}.")
        return [point_lead]
    if output_mode == "seq2seq_6x30d" and int(horizon_days) == 180:
        return [30, 60, 90, 120, 150, 180]
    raise ValueError(f"Unsupported output_mode={output_mode!r}.")


def input_offsets_for_window(*, context_days: int, input_gap_days: int) -> np.ndarray:
    context = int(context_days)
    gap = int(input_gap_days)
    if context <= 0 or gap <= 0:
        raise ValueError("context_days and input_gap_days must be positive.")
    max_offset = ((context - 1) // gap) * gap
    return np.arange(max_offset, -1, -gap, dtype=np.int64)


def context_row_count(*, context_days: int, input_gap_days: int) -> int:
    return int(input_offsets_for_window(context_days=context_days, input_gap_days=input_gap_days).shape[0])


def _parse_seq2one_days(output_mode: str) -> int | None:
    match = re.fullmatch(r"seq2one_(\d+)d", str(output_mode))
    return None if match is None else int(match.group(1))
