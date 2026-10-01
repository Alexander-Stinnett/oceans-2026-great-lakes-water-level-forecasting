"""Render a presentation-ready historical Lake Superior water-level figure."""

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

from oceans_glwl.artifacts.io import canonical_data_path, repository_root
from oceans_glwl.config import DATE_COLUMN, TARGET_COLUMN
from oceans_glwl.data.splits import ChronologicalSplits, build_chronological_splits
from oceans_glwl.data.validate import load_frame

OUTPUT_STEM = "lake_superior_water_level_30_day_moving_average"
PARTITION_COLORS = {
    "train": "#2A9D8F",
    "validation": "#E69F00",
    "test": "#C73E1D",
}
PARTITION_LABELS = {
    "train": "Train",
    "validation": "Validation",
    "test": "Test",
}


def _partition_starts(splits: ChronologicalSplits) -> dict[str, object]:
    """Return the three canonical partition start dates for direct plot labels."""

    return {
        "train": splits.train.start_date,
        "validation": splits.validation.start_date,
        "test": splits.test.start_date,
    }


def _sha256(path: Path) -> str:
    """Return a content digest for the figure provenance manifest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Load the canonical target series and write PNG/PDF presentation exports."""

    frame = load_frame(data_path)
    splits = build_chronological_splits(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"

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
        ax.plot(
            frame[DATE_COLUMN],
            frame[TARGET_COLUMN],
            color="#1F4E79",
            linewidth=1.45,
        )
        ax.set_title("Lake Superior Water Level (30-Day Moving Average)", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Lake Superior water level (m)")
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for name, date in _partition_starts(splits).items():
            color = PARTITION_COLORS[name]
            ax.axvline(date, color=color, linewidth=1.8, linestyle="--", zorder=3)
            ax.annotate(
                PARTITION_LABELS[name],
                xy=(date, 1),
                xycoords=("data", "axes fraction"),
                xytext=(5, -8),
                textcoords="offset points",
                color=color,
                fontweight="bold",
                ha="left",
                va="top",
            )

        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            metadata={"Creator": "OCEANS historical water-level figure", "CreationDate": None},
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
        "source_data": data_path.resolve().relative_to(root).as_posix(),
        "source_data_sha256": _sha256(data_path),
        "series": TARGET_COLUMN,
        "split_starts": {
            name: str(date.date()) for name, date in _partition_starts(splits).items()
        },
        "outputs": {
            "pdf": {"path": pdf_path.name, "sha256": _sha256(pdf_path)},
            "png": {"path": png_path.name, "sha256": _sha256(png_path)},
        },
    }
    (output_dir / f"{OUTPUT_STEM}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return pdf_path, png_path


def run(data_path: Path | None = None, output_dir: Path | None = None) -> dict[str, str]:
    """Write the figure using canonical repository defaults."""

    root = repository_root()
    pdf_path, png_path = save_figure(
        data_path or canonical_data_path(root),
        output_dir or root / "additional_figures",
    )
    return {"pdf": str(pdf_path.resolve()), "png": str(png_path.resolve())}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="Canonical combined dataset CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="PNG/PDF destination directory.")
    args = parser.parse_args(argv)
    print(run(args.data, args.output_dir))


if __name__ == "__main__":
    main()
