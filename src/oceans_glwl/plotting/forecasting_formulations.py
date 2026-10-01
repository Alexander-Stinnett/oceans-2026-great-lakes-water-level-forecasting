"""Render a presentation schematic of the two OCEANS forecasting formulations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from oceans_glwl.artifacts.io import repository_root

OUTPUT_STEM = "forecasting_formulations"
HORIZONS = (30, 60, 90, 120, 150, 180)
HISTORY_FILL = "#DCEAF7"
HISTORY_EDGE = "#1F4E79"
OUTPUT_FILL = "#FFF1D6"
OUTPUT_EDGE = "#C87500"
TEXT = "#202124"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _box(ax: plt.Axes, x: float, y: float, width: float, height: float, *, text: str, fill: str, edge: str, fontsize: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.8,
        )
    )
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", color=TEXT, fontsize=fontsize)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=16,
            linewidth=1.8, color="#4D5966",
        )
    )


def _panel_title(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.text(6, 6.65, title, ha="center", va="center", fontsize=19, fontweight="bold", color=TEXT)
    ax.text(6, 6.1, subtitle, ha="center", va="center", fontsize=13, color="#5F6368")


def _draw_single_horizon(ax: plt.Axes) -> None:
    _panel_title(ax, "Single-Horizon", "Sequence-to-One")
    _box(
        ax, 0.8, 2.7, 3.7, 1.35,
        text="Historical input\ncontext", fill=HISTORY_FILL, edge=HISTORY_EDGE, fontsize=15,
    )
    _arrow(ax, (4.65, 3.375), (7.0, 3.375))
    _box(
        ax, 7.15, 2.7, 3.55, 1.35,
        text=r"$y_{t+h}$", fill=OUTPUT_FILL, edge=OUTPUT_EDGE, fontsize=25,
    )
    ax.text(
        6, 1.55,
        r"$h \in \{30, 60, 90, 120, 150, 180\}\ \mathrm{days}$",
        ha="center", va="center", fontsize=16, color="#5F6368",
    )


def _draw_multi_horizon(ax: plt.Axes) -> None:
    _panel_title(ax, "Multi-Horizon", "Sequence-to-Sequence")
    _box(
        ax, 0.55, 2.7, 3.5, 1.35,
        text="Historical input\ncontext", fill=HISTORY_FILL, edge=HISTORY_EDGE, fontsize=15,
    )
    trunk_x = 5.0
    _arrow(ax, (4.2, 3.375), (trunk_x, 3.375))
    output_y = (5.15, 4.25, 3.35, 2.45, 1.55, 0.65)
    ax.plot([trunk_x, trunk_x], [output_y[-1] + 0.32, output_y[0] + 0.32], color="#4D5966", linewidth=1.8)
    for horizon, y in zip(HORIZONS, output_y, strict=True):
        _arrow(ax, (trunk_x, y + 0.32), (5.85, y + 0.32))
        _box(
            ax, 6.0, y, 4.8, 0.65,
            text=rf"$y_{{t+{horizon}}}$", fill=OUTPUT_FILL, edge=OUTPUT_EDGE, fontsize=17,
        )


def save_figure(output_dir: Path) -> tuple[Path, Path]:
    """Write PNG/PDF exports and a manifest identifying the configuration source."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
        for ax in axes:
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 7)
            ax.axis("off")
        _draw_single_horizon(axes[0])
        _draw_multi_horizon(axes[1])
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", metadata={"Creator": "OCEANS forecasting formulations figure", "CreationDate": None})
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        plt.close(fig)

    root = repository_root()
    config_path = root / "configs" / "historical_oceans_final_search.yaml"
    manifest = {
        "figure": OUTPUT_STEM,
        "concept_source": config_path.relative_to(root).as_posix(),
        "concept_source_sha256": _sha256(config_path),
        "single_horizon_outputs": [f"y_(t+{lead})" for lead in HORIZONS],
        "single_horizon_available_leads_days": list(HORIZONS),
        "multi_horizon_output_mode": "seq2seq_6x30d",
        "multi_horizon_leads_days": list(HORIZONS),
        "outputs": {
            "pdf": {"path": pdf_path.name, "sha256": _sha256(pdf_path)},
            "png": {"path": png_path.name, "sha256": _sha256(png_path)},
        },
    }
    (output_dir / f"{OUTPUT_STEM}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return pdf_path, png_path


def run(output_dir: Path | None = None) -> dict[str, str]:
    root = repository_root()
    pdf_path, png_path = save_figure(output_dir or root / "additional_figures")
    return {"pdf": str(pdf_path.resolve()), "png": str(png_path.resolve())}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="PNG/PDF destination directory.")
    args = parser.parse_args(argv)
    print(run(args.output_dir))


if __name__ == "__main__":
    main()
