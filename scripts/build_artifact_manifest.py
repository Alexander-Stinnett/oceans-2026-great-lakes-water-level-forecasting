from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from oceans_glwl.artifacts.manifests import artifact_record

GLM_MIGRATION_COMMIT = "832c29035d0478adaf3a78e3c2afb57e7281c8b9"
FINAL_SEARCH_COMMIT = "14943f0d6a923cc529ea467195edfa07a3e348c0"
RQ3_ANALYSIS_COMMIT = "a877c3a1eb89b4ccad8bd9b896c170066efcbf49"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    upstream = [root / "data/upstream" / name for name in ("finaldata.csv", "test_2022.csv", "test_2023.csv")]
    combined = root / "data/canonical/combined_full.csv"
    final_search = sorted((root / "artifacts/frozen/final_search").glob("*.parquet"))
    rq3_raw = sorted((root / "artifacts/frozen/rq3/raw").glob("*"))
    historical = sorted((root / "artifacts/frozen/rq3/historical_analysis").glob("*"))
    manuscript_analysis = sorted((root / "artifacts/frozen/rq3/manuscript_analysis").glob("*"))
    paper = [root / "paper/main.tex", root / "paper/references.bib"]
    figures = sorted((root / "paper/figures").glob("rq1_seq2one_180d_test_predictions*"))
    records = []
    for path in upstream:
        records.append(
            artifact_record(
                path,
                root=root,
                category="upstream_data",
                schema_version="dual_transformer_lake_superior_v1",
                provenance={"zenodo_doi": "10.5281/zenodo.15276228", "license": "GPL-3.0-or-later record-level"},
            )
        )
    records.append(
        artifact_record(
            combined,
            root=root,
            category="canonical_dataset",
            schema_version="lake_superior_daily_v1",
            parents=[path.relative_to(root).as_posix() for path in upstream],
            provenance={"glm_commit": GLM_MIGRATION_COMMIT},
        )
    )
    for path in final_search:
        records.append(
            artifact_record(
                path,
                root=root,
                category="frozen_final_search",
                schema_version="oceans_final_search_exports_v1",
                parents=[combined.relative_to(root).as_posix()],
                provenance={"final_attempt_commit": FINAL_SEARCH_COMMIT},
            )
        )
    for path in rq3_raw:
        records.append(
            artifact_record(
                path,
                root=root,
                category="frozen_rq3_raw",
                schema_version="oceans_final_search_rq3_v1",
                parents=["artifacts/frozen/final_search/condition_results_ALL.parquet"],
                provenance={"analysis_commit": RQ3_ANALYSIS_COMMIT},
            )
        )
    for path in historical:
        records.append(
            artifact_record(
                path,
                root=root,
                category="historical_rq3_2956_origin_analysis",
                schema_version="oceans_final_search_rq3_v1_historical",
                parents=["artifacts/frozen/rq3/raw/rq3_refit_predictions.parquet"],
                provenance={"analysis_commit": RQ3_ANALYSIS_COMMIT, "forecast_origins": 2956},
            )
        )
    for path in manuscript_analysis:
        records.append(
            artifact_record(
                path,
                root=root,
                category="manuscript_rq3_2926_origin_analysis",
                schema_version="oceans_final_search_rq3_v1_manuscript_surface",
                parents=[
                    "data/canonical/combined_full.csv",
                    "artifacts/frozen/rq3/raw/rq3_refit_predictions.parquet",
                    "artifacts/frozen/rq3/raw/analysis_policy.json",
                ],
                provenance={
                    "generated_by": "scripts/reproduce_paper.py",
                    "forecast_origin_start": "2015-07-01",
                    "forecast_origin_end": "2023-07-04",
                    "forecast_origins": 2926,
                },
            )
        )
    for path in paper:
        records.append(
            artifact_record(path, root=root, category="submitted_manuscript", schema_version="submitted_2026-08-28")
        )
    for path in figures:
        records.append(
            artifact_record(
                path,
                root=root,
                category="submitted_figure",
                schema_version="rq1_2926_origin_v1",
                parents=["artifacts/frozen/final_search/test_predictions_ALL.parquet"],
            )
        )
    payload = {
        "manifest_version": "1.0",
        "created_for_migration_date": "2026-08-28",
        "source_glm_commit": GLM_MIGRATION_COMMIT,
        "artifacts": sorted(records, key=lambda item: item["path"]),
    }
    destination = root / "artifacts/manifests/artifacts.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=destination.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, destination)
    print(f"Wrote {destination} with {len(records)} artifacts")


if __name__ == "__main__":
    main()
