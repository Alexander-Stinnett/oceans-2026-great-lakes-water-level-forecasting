import hashlib
import json
from pathlib import Path

import pytest

from oceans_glwl.artifacts.io import canonical_data_path, final_search_root
from oceans_glwl.plotting import (
    forecasting_formulations,
    historical_water_level,
    rq2_best_models_rmse,
    test_water_level_groups,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("module", "save"),
    [
        (forecasting_formulations, lambda output, root: forecasting_formulations.save_figure(output)),
        (
            historical_water_level,
            lambda output, root: historical_water_level.save_figure(canonical_data_path(root), output),
        ),
        (
            rq2_best_models_rmse,
            lambda output, root: rq2_best_models_rmse.save_figure(final_search_root(root), output),
        ),
        (
            test_water_level_groups,
            lambda output, root: test_water_level_groups.save_figure(
                canonical_data_path(root),
                root / "artifacts" / "frozen" / "rq3" / "manuscript_analysis" / "regime_thresholds.parquet",
                output,
            ),
        ),
    ],
)
def test_additional_figure_manifest_is_portable_and_matches_outputs(
    module: object, save: object, repo_root: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / module.OUTPUT_STEM
    pdf_path, png_path = save(output_dir, repo_root)
    manifest = json.loads((output_dir / f"{module.OUTPUT_STEM}.manifest.json").read_text(encoding="utf-8"))

    assert manifest["figure"] == module.OUTPUT_STEM
    assert str(repo_root.resolve()) not in json.dumps(manifest)
    assert manifest["outputs"]["pdf"] == {"path": pdf_path.name, "sha256": _sha256(pdf_path)}
    assert manifest["outputs"]["png"] == {"path": png_path.name, "sha256": _sha256(png_path)}
