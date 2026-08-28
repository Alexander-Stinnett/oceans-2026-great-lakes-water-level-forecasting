from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def final_search_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "artifacts" / "frozen" / "final_search"


def rq3_raw_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "artifacts" / "frozen" / "rq3" / "raw"


def canonical_data_path(root: Path | None = None) -> Path:
    return (root or repository_root()) / "data" / "canonical" / "combined_full.csv"

