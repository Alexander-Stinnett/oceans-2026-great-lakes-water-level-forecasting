"""Plot the frozen RQ1 180-day test predictions for the OCEANS paper.

The four condition IDs below were frozen using validation RMSE. Test metrics are
used only as integrity checks; this script performs no model selection. The
evaluation surface contains forecast origins on or after the final-refit training
cutoff, plotted at their target timestamps 180 days later.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREDICTIONS_FILENAME = "test_predictions_ALL.parquet"
OUTPUT_STEM = "rq1_seq2one_180d_test_predictions"
EXPECTED_COUNT = 2_926
EXPECTED_ORIGIN_START = pd.Timestamp("2015-07-01")
EXPECTED_ORIGIN_END = pd.Timestamp("2023-07-04")
EXPECTED_TARGET_START = pd.Timestamp("2015-12-28")
EXPECTED_TARGET_END = pd.Timestamp("2023-12-31")
ACTUAL_ATOL = 1e-8
RMSE_ATOL = 1e-6


@dataclass(frozen=True)
class FrozenCondition:
    """Scientific identity and display metadata for one frozen RQ1 condition."""

    label: str
    condition_id: str
    model_name: str
    input_variant: str
    context_days: int
    expected_test_rmse: float


FROZEN_CONDITIONS: tuple[FrozenCondition, ...] = (
    FrozenCondition(
        label="LSTM",
        condition_id=(
            "ofs1__autolstm__daily_smoothed_y_smoothed_exog__ctx540__"
            "seq2one_180d__5377ce08298a"
        ),
        model_name="AutoLSTM",
        input_variant="daily_smoothed_y_smoothed_exog",
        context_days=540,
        expected_test_rmse=0.1283629361,
    ),
    FrozenCondition(
        label="N-BEATSx",
        condition_id=(
            "ofs1__autonbeatsx__sparse_30d_smoothed_all__ctx720__"
            "seq2one_180d__a6d7a9c161de"
        ),
        model_name="AutoNBEATSx",
        input_variant="sparse_30d_smoothed_all",
        context_days=720,
        expected_test_rmse=0.1220241487,
    ),
    FrozenCondition(
        label="NHITS",
        condition_id=(
            "ofs1__autonhits__daily_smoothed_y_smoothed_exog__ctx720__"
            "seq2one_180d__cb22e7679fc3"
        ),
        model_name="AutoNHITS",
        input_variant="daily_smoothed_y_smoothed_exog",
        context_days=720,
        expected_test_rmse=0.1203348944,
    ),
    FrozenCondition(
        label="TFT",
        condition_id=(
            "ofs1__autotft__sparse_30d_smoothed_all__ctx360__"
            "seq2one_180d__13941b3fe1c7"
        ),
        model_name="AutoTFT",
        input_variant="sparse_30d_smoothed_all",
        context_days=360,
        expected_test_rmse=0.1555865080,
    ),
)

REQUIRED_COLUMNS = {
    "condition_id",
    "source_host",
    "sample_id",
    "unique_id",
    "split",
    "forecast_origin",
    "target_timestamp",
    "horizon_index",
    "lead_days",
    "is_endpoint",
    "actual",
    "model_name",
    "input_variant",
    "output_mode",
    "context_days",
    "prediction",
}


@dataclass(frozen=True)
class PreparedPredictions:
    """Validated, target-aligned series and their recomputed test RMSE values."""

    frame: pd.DataFrame
    rmse_by_model: dict[str, float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    configured_exports = os.environ.get("OCEANS_FINAL_ANALYSIS_EXPORTS")
    parser = argparse.ArgumentParser(
        description="Plot frozen RQ1 seq2one_180d test predictions at target dates."
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=Path(configured_exports) if configured_exports else None,
        required=configured_exports is None,
        help=f"Directory containing {PREDICTIONS_FILENAME}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure destination (default: the experiment's repository-local figures directory).",
    )
    return parser.parse_args(argv)


def load_frozen_predictions(exports_dir: Path) -> pd.DataFrame:
    """Read only the frozen conditions from the authoritative merged export."""

    source = exports_dir / PREDICTIONS_FILENAME
    if not source.is_file():
        raise FileNotFoundError(f"Authoritative prediction export not found: {source}")

    frame = pd.read_parquet(
        source,
        columns=sorted(REQUIRED_COLUMNS),
        filters=[
            (
                "condition_id",
                "in",
                [condition.condition_id for condition in FROZEN_CONDITIONS],
            )
        ],
    )
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise KeyError(f"{source} is missing required columns: {missing}")
    return frame


def prepare_frozen_predictions(
    predictions: pd.DataFrame,
    *,
    conditions: Sequence[FrozenCondition] = FROZEN_CONDITIONS,
    expected_count: int = EXPECTED_COUNT,
    expected_origin_start: pd.Timestamp = EXPECTED_ORIGIN_START,
    expected_origin_end: pd.Timestamp = EXPECTED_ORIGIN_END,
    expected_target_start: pd.Timestamp = EXPECTED_TARGET_START,
    expected_target_end: pd.Timestamp = EXPECTED_TARGET_END,
    actual_atol: float = ACTUAL_ATOL,
    rmse_atol: float = RMSE_ATOL,
) -> PreparedPredictions:
    """Select the post-cutoff origin surface and align it on target timestamp."""

    missing = sorted(REQUIRED_COLUMNS - set(predictions.columns))
    if missing:
        raise KeyError(f"Prediction data is missing required columns: {missing}")

    expected_origins = pd.date_range(expected_origin_start, expected_origin_end, freq="D")
    expected_targets = pd.date_range(expected_target_start, expected_target_end, freq="D")
    if len(expected_origins) != expected_count or len(expected_targets) != expected_count:
        raise ValueError(
            "Configured origin/target ranges disagree with the expected count: "
            f"origins={len(expected_origins)}, targets={len(expected_targets)}, "
            f"expected={expected_count}."
        )
    if not ((expected_targets - expected_origins) == pd.Timedelta(days=180)).all():
        raise ValueError("Configured target dates must be exactly 180 days after origins.")

    aligned: pd.DataFrame | None = None
    rmse_by_model: dict[str, float] = {}

    for condition in conditions:
        condition_frame = predictions.loc[
            predictions["condition_id"].eq(condition.condition_id)
        ].copy()
        condition_frame["forecast_origin"] = pd.to_datetime(
            condition_frame["forecast_origin"], errors="coerce"
        )
        if condition_frame["forecast_origin"].isna().any():
            raise ValueError(f"{condition.label} condition has invalid forecast origins.")
        condition_frame = condition_frame.loc[
            condition_frame["forecast_origin"].ge(expected_origin_start)
        ].copy()
        if len(condition_frame) != expected_count:
            raise ValueError(
                f"{condition.label} condition {condition.condition_id} has "
                f"{len(condition_frame)} post-cutoff rows; expected exactly "
                f"{expected_count}."
            )

        _validate_single_condition(
            condition_frame,
            condition,
            expected_origins,
            expected_targets,
        )
        condition_frame = condition_frame.sort_values("target_timestamp")

        if aligned is None:
            aligned = condition_frame[["target_timestamp", "actual"]].rename(
                columns={"actual": "Ground Truth"}
            )
        else:
            timestamps = condition_frame["target_timestamp"].reset_index(drop=True)
            if not timestamps.equals(aligned["target_timestamp"].reset_index(drop=True)):
                raise ValueError(
                    f"{condition.label} target-date coverage differs from the other "
                    "frozen conditions."
                )
            actual = condition_frame["actual"].to_numpy(dtype=float)
            reference = aligned["Ground Truth"].to_numpy(dtype=float)
            if not np.allclose(actual, reference, rtol=0.0, atol=actual_atol):
                mismatches = int(
                    (~np.isclose(actual, reference, rtol=0.0, atol=actual_atol)).sum()
                )
                max_difference = float(np.max(np.abs(actual - reference)))
                raise ValueError(
                    f"{condition.label} ground truth differs from the shared series: "
                    f"{mismatches} mismatches, max absolute difference "
                    f"{max_difference:.12g} m."
                )

        prediction = condition_frame["prediction"].to_numpy(dtype=float)
        actual = condition_frame["actual"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(np.square(prediction - actual))))
        if not np.isclose(
            rmse,
            condition.expected_test_rmse,
            rtol=0.0,
            atol=rmse_atol,
        ):
            raise ValueError(
                f"{condition.label} recomputed test RMSE {rmse:.12f} m does not "
                f"match expected {condition.expected_test_rmse:.12f} m within "
                f"{rmse_atol:g} m."
            )
        rmse_by_model[condition.label] = rmse
        assert aligned is not None
        aligned[condition.label] = prediction

    if aligned is None:
        raise ValueError("No frozen conditions were configured.")
    return PreparedPredictions(frame=aligned.reset_index(drop=True), rmse_by_model=rmse_by_model)


def _validate_single_condition(
    frame: pd.DataFrame,
    condition: FrozenCondition,
    expected_origins: pd.DatetimeIndex,
    expected_targets: pd.DatetimeIndex,
) -> None:
    """Fail closed on malformed identity, values, or temporal alignment."""

    for column, expected in (
        ("model_name", condition.model_name),
        ("input_variant", condition.input_variant),
        ("context_days", condition.context_days),
        ("output_mode", "seq2one_180d"),
        ("split", "test"),
        ("lead_days", 180),
        ("horizon_index", 1),
        ("is_endpoint", True),
    ):
        values = frame[column].drop_duplicates().tolist()
        if values != [expected]:
            raise ValueError(
                f"{condition.label} condition has unexpected {column}: "
                f"{values!r}; expected only {expected!r}."
            )

    if frame["source_host"].isna().any() or frame["source_host"].nunique() != 1:
        raise ValueError(
            f"{condition.label} condition must have exactly one non-null source_host; "
            f"found {frame['source_host'].nunique(dropna=False)}."
        )
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"{condition.label} condition has null or duplicate sample_id values.")
    if frame["unique_id"].isna().any() or frame["unique_id"].duplicated().any():
        raise ValueError(f"{condition.label} condition has null or duplicate unique_id values.")
    if not frame["sample_id"].astype(str).equals(frame["unique_id"].astype(str)):
        raise ValueError(
            f"{condition.label} condition has sample_id/unique_id identity mismatches."
        )

    frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"], errors="coerce")
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], errors="coerce")
    if frame[["forecast_origin", "target_timestamp"]].isna().any(axis=None):
        raise ValueError(f"{condition.label} condition has invalid origin or target timestamps.")
    if frame["target_timestamp"].duplicated().any():
        raise ValueError(f"{condition.label} condition has duplicate target timestamps.")

    offsets = frame["target_timestamp"] - frame["forecast_origin"]
    if not offsets.eq(pd.Timedelta(days=180)).all():
        bad_count = int((~offsets.eq(pd.Timedelta(days=180))).sum())
        raise ValueError(
            f"{condition.label} has {bad_count} rows not aligned at target = origin + 180 days."
        )

    observed_origins = pd.DatetimeIndex(frame["forecast_origin"].sort_values())
    if not observed_origins.equals(expected_origins):
        missing = expected_origins.difference(observed_origins)
        extra = observed_origins.difference(expected_origins)
        raise ValueError(
            f"{condition.label} does not cover the complete expected forecast-origin "
            f"range {expected_origins[0].date()} to {expected_origins[-1].date()}; "
            f"missing={len(missing)}, extra={len(extra)}."
        )

    observed_targets = pd.DatetimeIndex(frame["target_timestamp"].sort_values())
    if not observed_targets.equals(expected_targets):
        missing = expected_targets.difference(observed_targets)
        extra = observed_targets.difference(expected_targets)
        raise ValueError(
            f"{condition.label} does not cover the complete expected target range "
            f"{expected_targets[0].date()} to {expected_targets[-1].date()}; "
            f"missing={len(missing)}, extra={len(extra)}."
        )

    for column in ("actual", "prediction"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{condition.label} condition has non-finite {column} values.")


def save_figure(prepared: PreparedPredictions, output_dir: Path) -> tuple[Path, Path]:
    """Render publication PDF/PNG files and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    frame = prepared.frame

    prediction_color = "#4477AA"
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "normal",
        "legend.fontsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.45), sharex=True, sharey=True)
        panel_labels = ("(a)", "(b)", "(c)", "(d)")
        for ax, condition, panel_label in zip(
            axes.flat, FROZEN_CONDITIONS, panel_labels, strict=True
        ):
            ax.plot(
                frame["target_timestamp"],
                frame["Ground Truth"],
                color="#111111",
                linewidth=1.25,
                linestyle="-",
                label="Ground Truth",
                zorder=4,
            )
            ax.plot(
                frame["target_timestamp"],
                frame[condition.label],
                color=prediction_color,
                linewidth=0.95,
                linestyle="-",
                label="Prediction",
                zorder=3,
            )
            ax.set_title(f"{panel_label} {condition.label}", pad=2.0)
            ax.set_xlim(
                frame["target_timestamp"].iloc[0], frame["target_timestamp"].iloc[-1]
            )
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.grid(axis="y", color="#DEDEDE", linewidth=0.4, alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plotted_values = frame[["Ground Truth", *[item.label for item in FROZEN_CONDITIONS]]]
        lower = float(plotted_values.min(axis=None))
        upper = float(plotted_values.max(axis=None))
        padding = 0.035 * (upper - lower)
        axes[0, 0].set_ylim(lower - padding, upper + padding)
        fig.supxlabel("Target Date", y=0.025)
        fig.supylabel("Water Level (m)", x=0.023)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.5,
        )
        fig.tight_layout(
            rect=(0.04, 0.045, 1.0, 0.93),
            pad=0.3,
            h_pad=0.15,
            w_pad=0.55,
        )
        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            metadata={"Creator": "OCEANS final-search figure script", "CreationDate": None},
        )
        fig.savefig(
            png_path,
            format="png",
            dpi=400,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib"},
        )
        plt.close(fig)
    return pdf_path, png_path


def run(exports_dir: Path, output_dir: Path | None = None) -> dict[str, object]:
    """Load, validate, plot, and summarize the frozen test predictions."""

    predictions = load_frozen_predictions(exports_dir)
    prepared = prepare_frozen_predictions(predictions)
    destination = (
        output_dir if output_dir is not None else Path(__file__).resolve().parent / "figures"
    )
    pdf_path, png_path = save_figure(prepared, destination)
    frame = prepared.frame
    return {
        "source": str(exports_dir / PREDICTIONS_FILENAME),
        "pdf": str(pdf_path.resolve()),
        "png": str(png_path.resolve()),
        "first_target_date": frame["target_timestamp"].iloc[0].date().isoformat(),
        "last_target_date": frame["target_timestamp"].iloc[-1].date().isoformat(),
        "first_forecast_origin": EXPECTED_ORIGIN_START.date().isoformat(),
        "last_forecast_origin": EXPECTED_ORIGIN_END.date().isoformat(),
        "forecast_origins": int(len(frame)),
        "observations": int(len(frame)),
        "ground_truth_min_m": float(frame["Ground Truth"].min()),
        "ground_truth_max_m": float(frame["Ground Truth"].max()),
        "test_rmse_m": prepared.rmse_by_model,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    print(json.dumps(run(args.exports_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
