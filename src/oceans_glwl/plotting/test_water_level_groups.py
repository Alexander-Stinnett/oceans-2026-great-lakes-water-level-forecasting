"""Render test-set water level with saved 180-day RQ3 regime shading."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oceans_glwl.artifacts.io import canonical_data_path, repository_root
from oceans_glwl.config import DATE_COLUMN, TARGET_COLUMN
from oceans_glwl.data.splits import build_chronological_splits
from oceans_glwl.data.validate import load_frame

OUTPUT_STEM = "test_water_level_groups_180_day_change"
LEAD_DAYS = 180
EXPECTED_TARGET_START = pd.Timestamp("2015-12-28")
EXPECTED_TARGET_END = pd.Timestamp("2023-12-31")
EXPECTED_TARGET_COUNT = 2_926
REGIME_COLORS = {"rising": "#D1495B", "stable": "#2A9D8F", "falling": "#277DA1"}
REGIME_SHADE_ALPHA = 0.22


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_saved_thresholds(path: Path) -> tuple[float, float]:
    """Read the frozen RQ3 terciles for the selected forecast lead."""

    thresholds = pd.read_parquet(path)
    row = thresholds.loc[thresholds["lead_days"].eq(LEAD_DAYS)]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one saved threshold row for {LEAD_DAYS} days.")
    return float(row.iloc[0]["lower_tercile_m"]), float(row.iloc[0]["upper_tercile_m"])


def prepare_test_groups(data_path: Path, threshold_path: Path) -> tuple[pd.DataFrame, float, float]:
    """Apply the saved 180-day observed-change regimes to canonical test targets."""

    frame = load_frame(data_path)
    splits = build_chronological_splits(frame)
    lower, upper = load_saved_thresholds(threshold_path)
    prepared = frame[[DATE_COLUMN, TARGET_COLUMN]].copy()
    prepared["observed_change_m"] = prepared[TARGET_COLUMN] - prepared[TARGET_COLUMN].shift(LEAD_DAYS)
    prepared = prepared.loc[
        prepared[DATE_COLUMN].between(EXPECTED_TARGET_START, EXPECTED_TARGET_END, inclusive="both")
    ].copy()
    if len(prepared) != EXPECTED_TARGET_COUNT:
        raise ValueError(f"Expected {EXPECTED_TARGET_COUNT} 180-day test targets, found {len(prepared)}.")
    origins = prepared[DATE_COLUMN] - pd.Timedelta(days=LEAD_DAYS)
    if origins.min() != splits.test.start_date or origins.max() != pd.Timestamp("2023-07-04"):
        raise ValueError("The selected target dates do not map to the canonical test-origin surface.")
    if prepared["observed_change_m"].isna().any():
        raise ValueError("The test-target surface contains an undefined 180-day change.")
    prepared["regime"] = np.where(
        prepared["observed_change_m"] < lower,
        "falling",
        np.where(prepared["observed_change_m"] > upper, "rising", "stable"),
    )
    return prepared, lower, upper


def contiguous_regime_spans(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Collapse adjacent daily target dates with the same regime into one span."""

    boundaries = frame["regime"].ne(frame["regime"].shift()) | frame[DATE_COLUMN].diff().dt.days.ne(1)
    groups = frame.groupby(boundaries.cumsum(), sort=False)
    return [
        (group[DATE_COLUMN].iloc[0], group[DATE_COLUMN].iloc[-1], str(group["regime"].iloc[0]))
        for _, group in groups
    ]


def save_figure(data_path: Path, threshold_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Write presentation PNG/PDF exports and a provenance manifest."""

    series, lower, upper = prepare_test_groups(data_path, threshold_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    spans = contiguous_regime_spans(series)

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
        for start, end, regime in spans:
            ax.axvspan(
                start,
                end + pd.Timedelta(days=1),
                color=REGIME_COLORS[regime],
                alpha=REGIME_SHADE_ALPHA,
                linewidth=0,
                zorder=0,
            )
        ax.plot(series[DATE_COLUMN], series[TARGET_COLUMN], color="#1F4E79", linewidth=1.5, zorder=2)
        ax.set_title("Test-Set Water Level Groups (180-Day Change)", pad=12)
        ax.set_xlabel("Target date")
        ax.set_ylabel("Lake Superior water level (m)")
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", metadata={"Creator": "OCEANS test water-level regime figure", "CreationDate": None})
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        plt.close(fig)

    root = repository_root()
    manifest = {
        "figure": OUTPUT_STEM,
        "source_data": data_path.resolve().relative_to(root).as_posix(),
        "source_data_sha256": _sha256(data_path),
        "saved_thresholds": threshold_path.resolve().relative_to(root).as_posix(),
        "saved_thresholds_sha256": _sha256(threshold_path),
        "lead_days": LEAD_DAYS,
        "target_period": {"start": EXPECTED_TARGET_START.date().isoformat(), "end": EXPECTED_TARGET_END.date().isoformat(), "count": len(series)},
        "thresholds_m": {"lower_tercile": lower, "upper_tercile": upper},
        "classification": "falling: change < lower; stable: lower <= change <= upper; rising: change > upper",
        "regime_shading": {"colors": REGIME_COLORS, "alpha": REGIME_SHADE_ALPHA},
        "regime_counts": {str(name): int(count) for name, count in series["regime"].value_counts().sort_index().items()},
        "contiguous_span_count": len(spans),
        "outputs": {
            "pdf": {"path": pdf_path.name, "sha256": _sha256(pdf_path)},
            "png": {"path": png_path.name, "sha256": _sha256(png_path)},
        },
    }
    (output_dir / f"{OUTPUT_STEM}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return pdf_path, png_path


def run(data_path: Path | None = None, threshold_path: Path | None = None, output_dir: Path | None = None) -> dict[str, str]:
    root = repository_root()
    pdf_path, png_path = save_figure(
        data_path or canonical_data_path(root),
        threshold_path or root / "artifacts" / "frozen" / "rq3" / "manuscript_analysis" / "regime_thresholds.parquet",
        output_dir or root / "additional_figures",
    )
    return {"pdf": str(pdf_path.resolve()), "png": str(png_path.resolve())}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="Canonical combined dataset CSV.")
    parser.add_argument("--thresholds", type=Path, default=None, help="Saved RQ3 threshold parquet.")
    parser.add_argument("--output-dir", type=Path, default=None, help="PNG/PDF destination directory.")
    args = parser.parse_args(argv)
    print(run(args.data, args.thresholds, args.output_dir))


if __name__ == "__main__":
    main()
