from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from oceans_glwl.config import DATE_COLUMN, SPLIT_NAMES, SplitConfig

SPLIT_ALIASES = {"val": "validation", "valid": "validation"}


@dataclass(frozen=True)
class SplitRange:
    name: str
    start_index: int
    end_index: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    def contains_index(self, index: int) -> bool:
        return self.start_index <= int(index) <= self.end_index


@dataclass(frozen=True)
class ChronologicalSplits:
    train: SplitRange
    validation: SplitRange
    test: SplitRange

    def range(self, split: str) -> SplitRange:
        return getattr(self, normalize_split_name(split))

    def split_for_index(self, index: int) -> str:
        for name in SPLIT_NAMES:
            if self.range(name).contains_index(index):
                return name
        raise ValueError(f"Index {index} is outside the split bounds.")

    def to_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "start_index": item.start_index,
                "end_index": item.end_index,
                "start_date": item.start_date.strftime("%Y-%m-%d"),
                "end_date": item.end_date.strftime("%Y-%m-%d"),
            }
            for name in SPLIT_NAMES
            for item in [self.range(name)]
        }


def normalize_split_name(split: str) -> str:
    normalized = SPLIT_ALIASES.get(str(split), str(split))
    if normalized not in SPLIT_NAMES:
        raise ValueError(f"Unknown split={split!r}.")
    return normalized


def build_chronological_splits(
    frame: pd.DataFrame,
    *,
    split_config: SplitConfig | None = None,
    date_col: str = DATE_COLUMN,
) -> ChronologicalSplits:
    config = split_config or SplitConfig()
    n_rows = len(frame)
    if n_rows < 10:
        raise ValueError("At least ten rows are required for chronological splitting.")
    if config.uses_explicit_sizes():
        train_size, validation_size, test_size = _explicit_sizes(config, n_rows)
    else:
        fractions = (config.train_fraction, config.validation_fraction, config.test_fraction)
        if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("Split fractions must be positive and sum to one.")
        train_size = int(n_rows * config.train_fraction)
        validation_end = int(n_rows * (config.train_fraction + config.validation_fraction))
        validation_size = validation_end - train_size
        test_size = n_rows - validation_end
    if min(train_size, validation_size, test_size) <= 0 or train_size + validation_size + test_size != n_rows:
        raise ValueError("Chronological split sizes must be positive and cover every row exactly once.")
    bounds = {
        "train": (0, train_size - 1),
        "validation": (train_size, train_size + validation_size - 1),
        "test": (train_size + validation_size, n_rows - 1),
    }
    ranges = {
        name: SplitRange(name, start, end, pd.Timestamp(frame[date_col].iloc[start]), pd.Timestamp(frame[date_col].iloc[end]))
        for name, (start, end) in bounds.items()
    }
    return ChronologicalSplits(ranges["train"], ranges["validation"], ranges["test"])


def _explicit_sizes(config: SplitConfig, n_rows: int) -> tuple[int, int, int]:
    sizes = [config.train_size, config.validation_size, config.test_size]
    if sum(value is not None for value in sizes) < 2:
        raise ValueError("At least two explicit split sizes are required.")
    resolved = [None if value is None else int(value) for value in sizes]
    missing = resolved.index(None) if None in resolved else None
    if missing is not None:
        resolved[missing] = n_rows - sum(value for value in resolved if value is not None)
    return tuple(int(value) for value in resolved)  # type: ignore[arg-type,return-value]
