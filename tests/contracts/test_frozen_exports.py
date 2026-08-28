import pandas as pd

from oceans_glwl.analysis.contracts import validate_final_search_exports
from oceans_glwl.artifacts.manifests import verify_manifest


def test_artifact_manifest_verifies() -> None:
    assert verify_manifest() == []


def test_final_search_contracts(final_exports: dict[str, pd.DataFrame]) -> None:
    summary = validate_final_search_exports(
        final_exports["conditions"], final_exports["horizons"], final_exports["predictions"], final_exports["trials"]
    )
    assert summary == {
        "conditions": 336, "trials": 17_204, "complete_trials": 16_800,
        "failed_trials": 403, "running_trials": 1, "predictions": 1_745_856, "horizon_metrics": 576,
    }

