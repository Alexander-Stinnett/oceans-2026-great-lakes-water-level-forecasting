from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from oceans_glwl.config import SPLIT_NAMES, WindowConfig, input_offsets_for_window
from oceans_glwl.data.splits import ChronologicalSplits, normalize_split_name
from oceans_glwl.data.validate import assert_daily_contiguous


@dataclass(frozen=True)
class ForecastSamples:
    metadata: pd.DataFrame
    input_values: np.ndarray
    input_observed_mask: np.ndarray
    target_values: np.ndarray
    feature_columns: tuple[str, ...]
    target_column: str
    window_config: WindowConfig

    def for_split(self, split: str) -> ForecastSamples:
        normalized = normalize_split_name(split)
        indices = self.metadata.index[self.metadata["split"].astype(str).eq(normalized)].to_numpy(dtype=np.int64)
        return ForecastSamples(
            self.metadata.loc[indices].reset_index(drop=True),
            self.input_values[indices],
            self.input_observed_mask[indices],
            self.target_values[indices],
            self.feature_columns,
            self.target_column,
            self.window_config,
        )


def concatenate_samples(items: Sequence[ForecastSamples]) -> ForecastSamples:
    if not items:
        raise ValueError("At least one sample set is required.")
    first = items[0]
    for item in items[1:]:
        if item.window_config != first.window_config or item.feature_columns != first.feature_columns:
            raise ValueError("Sample sets must share window and feature contracts.")
    return ForecastSamples(
        pd.concat([item.metadata for item in items], ignore_index=True),
        np.concatenate([item.input_values for item in items], axis=0),
        np.concatenate([item.input_observed_mask for item in items], axis=0),
        np.concatenate([item.target_values for item in items], axis=0),
        first.feature_columns,
        first.target_column,
        first.window_config,
    )


def build_samples(
    frame: pd.DataFrame,
    *,
    window_config: WindowConfig,
    splits: ChronologicalSplits,
    split: str | None = None,
    max_samples_per_split: int | None = None,
) -> ForecastSamples:
    assert_daily_contiguous(frame)
    requested = list(SPLIT_NAMES) if split is None else [normalize_split_name(split)]
    if max_samples_per_split is not None and int(max_samples_per_split) <= 0:
        raise ValueError("max_samples_per_split must be positive.")
    required = {window_config.date_col, window_config.target_col, *window_config.feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Data frame is missing required sample columns: {missing}.")

    offsets = input_offsets_for_window(
        context_days=window_config.context_days,
        input_gap_days=window_config.input_gap_days,
    )
    leads = np.asarray(window_config.lead_days, dtype=np.int64)
    candidates: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {name: [] for name in requested}
    first_origin = int(offsets[0])
    last_origin = len(frame) - int(leads.max()) - 1
    for origin in _candidate_origins(first_origin, last_origin, window_config):
        input_indices = origin - offsets
        target_indices = origin + leads
        target_splits = [splits.split_for_index(int(index)) for index in target_indices]
        endpoint_split = target_splits[-1]
        if endpoint_split in candidates and len(set(target_splits)) == 1:
            candidates[endpoint_split].append((origin, input_indices, target_indices))

    dates = pd.to_datetime(frame[window_config.date_col]).reset_index(drop=True)
    features = frame.loc[:, list(window_config.feature_columns)].apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(frame[window_config.target_col], errors="coerce")
    records: list[dict[str, Any]] = []
    input_blocks: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for split_name in requested:
        for ordinal, (origin, input_indices, target_indices) in enumerate(
            _sample_candidates(candidates[split_name], max_samples_per_split)
        ):
            origin_date = pd.Timestamp(dates.iloc[origin])
            input_dates = [pd.Timestamp(dates.iloc[int(index)]) for index in input_indices]
            target_dates = [pd.Timestamp(dates.iloc[int(index)]) for index in target_indices]
            block = features.iloc[input_indices].to_numpy(dtype=np.float32)
            target_block = target.iloc[target_indices].to_numpy(dtype=np.float32)
            if not np.isfinite(target_block).all():
                raise ValueError("Sample target values must be finite.")
            input_blocks.append(np.nan_to_num(block, nan=0.0))
            masks.append(np.isfinite(block).astype(np.float32))
            target_blocks.append(target_block)
            records.append(
                {
                    "sample_id": f"{split_name}_{ordinal:06d}",
                    "split": split_name,
                    "split_assignment_rule": "complete_target_block",
                    "forecast_origin_index": int(origin),
                    "forecast_origin_date": origin_date,
                    "input_indices": [int(value) for value in input_indices],
                    "input_dates": input_dates,
                    "input_end_index": int(input_indices[-1]),
                    "input_end_date": input_dates[-1],
                    "target_indices": [int(value) for value in target_indices],
                    "target_dates": target_dates,
                    "target_splits": [split_name] * len(target_indices),
                    "endpoint_target_date": target_dates[-1],
                    "endpoint_target_split": split_name,
                    "lead_days": list(window_config.lead_days),
                    "horizon_days": int(window_config.horizon_days),
                    "input_grid": window_config.input_grid,
                    "input_gap_days": int(window_config.input_gap_days),
                }
            )
    if not records:
        raise ValueError("No valid samples were created.")
    result = ForecastSamples(
        pd.DataFrame(records),
        np.stack(input_blocks).astype(np.float32),
        np.stack(masks).astype(np.float32),
        np.stack(target_blocks).astype(np.float32),
        window_config.feature_columns,
        window_config.target_col,
        window_config,
    )
    assert_sample_contract(result, splits=splits)
    return result


def assert_sample_contract(samples: ForecastSamples, *, splits: ChronologicalSplits) -> None:
    if samples.input_values.shape != samples.input_observed_mask.shape:
        raise AssertionError("Input values and masks must share a shape.")
    expected_input = (len(samples.metadata), samples.window_config.context_rows, len(samples.feature_columns))
    expected_target = (len(samples.metadata), len(samples.window_config.lead_days))
    if samples.input_values.shape != expected_input or samples.target_values.shape != expected_target:
        raise AssertionError("Sample arrays do not match the window contract.")
    for row in samples.metadata.to_dict("records"):
        split = normalize_split_name(str(row["split"]))
        origin_index = int(row["forecast_origin_index"])
        origin_date = pd.Timestamp(row["forecast_origin_date"])
        input_indices = [int(value) for value in row["input_indices"]]
        target_indices = [int(value) for value in row["target_indices"]]
        input_dates = [pd.Timestamp(value) for value in row["input_dates"]]
        target_dates = [pd.Timestamp(value) for value in row["target_dates"]]
        if input_indices[-1] != origin_index or pd.Timestamp(row["input_end_date"]) != origin_date:
            raise AssertionError("Input window must end at the forecast origin.")
        if max(input_dates) > origin_date or any(value <= origin_date for value in target_dates):
            raise AssertionError("Sample contains temporal leakage.")
        observed_leads = [(value - origin_date).days for value in target_dates]
        if observed_leads != list(samples.window_config.lead_days):
            raise AssertionError("Target dates do not match configured leads.")
        if any(splits.split_for_index(index) != split for index in target_indices):
            raise AssertionError("Target block crosses split boundaries.")


def _candidate_origins(first: int, last: int, config: WindowConfig) -> range:
    if config.input_grid == "row_zero_anchored" and config.input_gap_days > 1:
        gap = config.input_gap_days
        first = first if first % gap == 0 else first + gap - first % gap
        return range(first, last + 1, gap)
    return range(first, last + 1)


def _sample_candidates(
    candidates: Sequence[tuple[int, np.ndarray, np.ndarray]],
    maximum: int | None,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if maximum is None or len(candidates) <= int(maximum):
        return list(candidates)
    positions = np.linspace(0, len(candidates) - 1, num=int(maximum), dtype=np.int64)
    return [candidates[int(position)] for position in positions]
