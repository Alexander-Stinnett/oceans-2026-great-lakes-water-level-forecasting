"""Render the RQ2 test-RMSE comparison of best Seq2One and Seq2Seq models."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from oceans_glwl.analysis.contracts import validate_final_search_exports
from oceans_glwl.analysis.rq2 import analyze_rq2
from oceans_glwl.artifacts.io import final_search_root, repository_root

OUTPUT_STEM = "best_seq2one_vs_seq2seq_test_rmse"
SEQ2ONE_COLOR = "#1F4E79"
SEQ2SEQ_COLOR = "#D55E00"


def _sha256(path: Path) -> str:
    """Return a content digest for the provenance manifest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rq2_comparison(final_root: Path) -> tuple[pd.DataFrame, dict[str, object], dict[str, Path]]:
    """Recompute the validation-selected RQ2 comparison from frozen exports."""

    sources = {
        "condition_results": final_root / "condition_results_ALL.parquet",
        "test_horizon_metrics": final_root / "test_horizon_metrics_ALL.parquet",
        "test_predictions": final_root / "test_predictions_ALL.parquet",
        "validation_trials": final_root / "validation_trials_ALL.parquet",
    }
    conditions = pd.read_parquet(sources["condition_results"])
    horizons = pd.read_parquet(sources["test_horizon_metrics"])
    predictions = pd.read_parquet(sources["test_predictions"])
    trials = pd.read_parquet(sources["validation_trials"])
    validate_final_search_exports(conditions, horizons, predictions, trials)
    comparison, summary = analyze_rq2(conditions, predictions)
    return comparison, summary, sources


def save_figure(final_root: Path, output_dir: Path) -> tuple[Path, Path]:
    """Write poster-ready PNG/PDF exports and their provenance manifest."""

    comparison, summary, sources = load_rq2_comparison(final_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"

    horizons = comparison["horizon_days"].to_numpy(dtype=int)
    y_axis_lower_bound = float(
        min(comparison["seq2one_rmse_cm"].min(), comparison["seq2seq_rmse_cm"].min()) - 0.35
    )
    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.labelsize": 21,
        "axes.titlesize": 25,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 22,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(13.5, 8.0), constrained_layout=True)
        ax.plot(
            horizons,
            comparison["seq2one_rmse_cm"],
            color=SEQ2ONE_COLOR,
            linewidth=5.2,
            marker="o",
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=1.4,
            label="Best Seq2One",
            zorder=3,
        )
        ax.plot(
            horizons,
            comparison["seq2seq_rmse_cm"],
            color=SEQ2SEQ_COLOR,
            linewidth=5.2,
            marker="s",
            markersize=11,
            markeredgecolor="white",
            markeredgewidth=1.4,
            label="Best Seq2Seq",
            zorder=3,
        )
        ax.set_title("Best Seq2One VS Seq2Seq Models", pad=16, fontweight="bold")
        ax.set_xlabel("Forecast horizon (days)", labelpad=10)
        ax.set_ylabel("Test RMSE (cm)", labelpad=10)
        ax.set_xticks(horizons)
        ax.set_ylim(bottom=y_axis_lower_bound)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.9, zorder=0)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.9, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", frameon=False, handlelength=3.0, markerscale=1.2)

        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            metadata={"Creator": "OCEANS RQ2 best-model RMSE figure", "CreationDate": None},
        )
        fig.savefig(
            png_path,
            format="png",
            dpi=300,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib"},
        )
        plt.close(fig)

    root = repository_root()
    manifest = {
        "figure": OUTPUT_STEM,
        "source_artifacts": {
            name: {"path": path.resolve().relative_to(root).as_posix(), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "selection": "lowest validation RMSE per output formulation",
        "metric": "test RMSE (cm)",
        "y_axis_lower_bound_cm": y_axis_lower_bound,
        "evaluation_surface": {"forecast_origins": int(comparison["forecast_origins"].iloc[0])},
        "horizons_days": horizons.tolist(),
        "series": {
            "seq2one": {
                "models": comparison["seq2one_model"].tolist(),
                "condition_ids": comparison["seq2one_condition_id"].tolist(),
                "test_rmse_cm": comparison["seq2one_rmse_cm"].tolist(),
            },
            "seq2seq": {
                "model": str(comparison["seq2seq_model"].iloc[0]),
                "condition_id": str(comparison["seq2seq_condition_id"].iloc[0]),
                "test_rmse_cm": comparison["seq2seq_rmse_cm"].tolist(),
            },
        },
        "mean_relative_difference_percent": summary["mean_relative_difference_percent"],
        "outputs": {
            "pdf": {"path": pdf_path.name, "sha256": _sha256(pdf_path)},
            "png": {"path": png_path.name, "sha256": _sha256(png_path)},
        },
    }
    (output_dir / f"{OUTPUT_STEM}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return pdf_path, png_path


def run(final_root: Path | None = None, output_dir: Path | None = None) -> dict[str, str]:
    """Write the RQ2 comparison figure using repository defaults."""

    root = repository_root()
    pdf_path, png_path = save_figure(
        final_root or final_search_root(root),
        output_dir or root / "additional_figures",
    )
    return {"pdf": str(pdf_path.resolve()), "png": str(png_path.resolve())}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, default=None, help="Frozen final-search export directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="PNG/PDF destination directory.")
    args = parser.parse_args(argv)
    print(run(args.final_root, args.output_dir))


if __name__ == "__main__":
    main()
