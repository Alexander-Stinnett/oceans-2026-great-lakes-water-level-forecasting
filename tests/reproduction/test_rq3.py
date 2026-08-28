import json
from pathlib import Path

import pandas as pd

from oceans_glwl.analysis.goldens import RQ3_PRINTED, assert_manuscript_goldens


def test_rq3_frozen_refit_provenance(repo_root: Path) -> None:
    root = repo_root / "artifacts/frozen/rq3/raw"
    frozen = json.loads((root / "frozen_condition_index.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "seed_refit_plan.json").read_text(encoding="utf-8"))
    status = pd.read_parquet(root / "rq3_refit_status.parquet")
    assert len(frozen["selection_slots"]) == 11
    assert len({slot["condition_id"] for slot in frozen["selection_slots"]}) == 10
    assert plan["seeds"] == list(range(1337, 1347))
    assert plan["task_count"] == len(plan["tasks"]) == 100
    assert plan["max_steps"] == 750
    assert all(task["max_steps"] == 750 and task["new_hpo"] is False for task in plan["tasks"])
    assert len(status) == status["rq3_refit_id"].nunique() == 100


def test_rq3_manuscript_surface_and_regime_counts(rq3_result: dict[str, pd.DataFrame]) -> None:
    predictions = rq3_result["manuscript_predictions"]
    assert predictions["forecast_origin"].nunique() == 2_926
    assert predictions["forecast_origin"].min() == pd.Timestamp("2015-07-01")
    assert predictions["forecast_origin"].max() == pd.Timestamp("2023-07-04")
    counts = (
        predictions.loc[predictions["system"].eq("global_seq2one") & predictions["seed"].eq(1337)]
        .groupby(["lead_days", "regime"])["forecast_origin"].nunique().to_dict()
    )
    assert counts == {
        (30, "falling"): 921, (30, "rising"): 831, (30, "stable"): 1174,
        (60, "falling"): 929, (60, "rising"): 871, (60, "stable"): 1126,
        (90, "falling"): 823, (90, "rising"): 907, (90, "stable"): 1196,
        (120, "falling"): 825, (120, "rising"): 942, (120, "stable"): 1159,
        (150, "falling"): 927, (150, "rising"): 967, (150, "stable"): 1032,
        (180, "falling"): 949, (180, "rising"): 944, (180, "stable"): 1033,
    }


def test_all_90_rq3_printed_values(rq1_result: pd.DataFrame, rq2_result: tuple[pd.DataFrame, dict[str, object]], rq3_result: dict[str, pd.DataFrame]) -> None:
    table, summary = rq2_result
    result = assert_manuscript_goldens(
        rq1_result, table, summary, rq3_result["regime_rmse_aggregate"], rq3_result["persistence_regime_rmse"]
    )
    assert result["rq3_printed_values"] == 90 == len(RQ3_PRINTED) * 5


def test_rq3_uses_sample_standard_deviation(rq3_result: dict[str, pd.DataFrame]) -> None:
    by_seed = rq3_result["regime_rmse_by_seed"]
    aggregate = rq3_result["regime_rmse_aggregate"]
    key = ("global_seq2one", 90, "stable")
    values = by_seed.loc[
        by_seed["system"].eq(key[0]) & by_seed["lead_days"].eq(key[1]) & by_seed["regime"].eq(key[2]), "rmse_cm"
    ]
    observed = aggregate.loc[
        aggregate["system"].eq(key[0]) & aggregate["lead_days"].eq(key[1]) & aggregate["regime"].eq(key[2]), "rmse_cm_seed_std"
    ].iloc[0]
    assert observed == values.std(ddof=1)
    assert observed != values.std(ddof=0)
