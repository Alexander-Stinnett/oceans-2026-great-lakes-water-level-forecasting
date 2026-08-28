import math
from pathlib import Path

import pandas as pd

from oceans_glwl.data.build import (
    CANONICAL_SHA256,
    RAW_COLUMNS,
    build_canonical_frame,
    load_upstream_directory,
    rebuild_dataset,
    sha256_file,
)
from oceans_glwl.data.splits import build_chronological_splits


def test_upstream_and_canonical_hashes(repo_root: Path) -> None:
    expected = {
        "finaldata.csv": "98e21a38cecbbbe629ced269551823bf448caa2f0413b57cb804beed9cdbf053",
        "test_2022.csv": "c0b4b45ed75ffea073e0aad118139a0e88084aa205ea7fa627d96377d54b5511",
        "test_2023.csv": "c7e66a43c7f0f539c186b3a05921420fe255ff80a347c7bfd743ee172f29de2c",
    }
    for name, digest in expected.items():
        assert sha256_file(repo_root / "data/upstream" / name) == digest
    assert sha256_file(repo_root / "data/canonical/combined_full.csv") == CANONICAL_SHA256


def test_deterministic_dataset_rebuild_is_byte_exact(repo_root: Path, tmp_path: Path) -> None:
    output = rebuild_dataset(repo_root / "data/upstream", tmp_path / "combined_full.csv")
    assert sha256_file(output) == CANONICAL_SHA256
    assert output.read_bytes() == (repo_root / "data/canonical/combined_full.csv").read_bytes()


def test_target_and_trailing_averages_are_causal(repo_root: Path) -> None:
    raw = load_upstream_directory(repo_root / "data/upstream")
    canonical = build_canonical_frame(raw)
    assert len(raw) == 15_705
    assert list(raw.columns) == ["date", *RAW_COLUMNS]
    assert canonical.iloc[0]["date"] == raw.iloc[179]["date"]
    raw_levels = pd.to_numeric(raw["wl_lake"])
    for canonical_index in (0, 1, len(canonical) - 1):
        raw_index = canonical_index + 179
        expected = raw_levels.iloc[raw_index - 29 : raw_index + 1].mean()
        assert math.isclose(canonical.iloc[canonical_index]["avg_wl_lake_30"], expected, abs_tol=1e-12)


def test_canonical_split_contract(canonical_frame: pd.DataFrame) -> None:
    assert len(canonical_frame) == 15_526
    splits = build_chronological_splits(canonical_frame)
    assert (splits.train.start_date.strftime("%Y-%m-%d"), splits.train.end_date.strftime("%Y-%m-%d")) == ("1981-06-29", "2011-03-31")
    assert (splits.validation.start_date.strftime("%Y-%m-%d"), splits.validation.end_date.strftime("%Y-%m-%d")) == ("2011-04-01", "2015-06-30")
    assert (splits.test.start_date.strftime("%Y-%m-%d"), splits.test.end_date.strftime("%Y-%m-%d")) == ("2015-07-01", "2023-12-31")
    assert [splits.train.end_index + 1, splits.validation.end_index - splits.validation.start_index + 1, splits.test.end_index - splits.test.start_index + 1] == [10_868, 1_552, 3_106]
