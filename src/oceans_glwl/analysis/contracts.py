from __future__ import annotations

import numpy as np
import pandas as pd

from oceans_glwl.analysis.selection import validate_condition_coverage


def validate_final_search_exports(
    conditions: pd.DataFrame,
    horizons: pd.DataFrame,
    predictions: pd.DataFrame,
    trials: pd.DataFrame,
) -> dict[str, object]:
    validate_condition_coverage(conditions)
    expected_ids = set(conditions["condition_id"].astype(str))
    for label, frame in (("horizons", horizons), ("predictions", predictions), ("trials", trials)):
        observed = set(frame["condition_id"].astype(str))
        if observed != expected_ids:
            raise ValueError(f"{label} condition coverage differs from condition results")
    states = trials["state"].value_counts().to_dict()
    expected_states = {"COMPLETE": 16_800, "FAIL": 403, "RUNNING": 1}
    for state, count in expected_states.items():
        if int(states.get(state, 0)) != count:
            raise ValueError(f"Expected {count} {state} trials, found {states.get(state, 0)}")
    if int(states.get("PRUNED", 0)) != 0:
        raise ValueError("The frozen trial export unexpectedly contains PRUNED trials")
    complete = trials.loc[trials["state"].eq("COMPLETE")]
    per_condition = complete.groupby("condition_id").size()
    if len(per_condition) != 336 or not per_condition.eq(50).all():
        raise ValueError("Every condition must contain exactly 50 COMPLETE trials")
    best_trials = complete.sort_values(["validation_rmse", "trial_number"], kind="stable").groupby(
        "condition_id", as_index=False
    ).first()
    joined = conditions[["condition_id", "best.trial_number", "best.validation_value"]].merge(
        best_trials[["condition_id", "trial_number", "validation_rmse"]], on="condition_id", validate="one_to_one"
    )
    if not (joined["best.trial_number"].astype(int) == joined["trial_number"].astype(int)).all():
        raise ValueError("A selected trial is not the minimum validation objective")
    if not np.allclose(joined["best.validation_value"], joined["validation_rmse"], atol=0, rtol=0):
        raise ValueError("Selected validation values disagree with trial history")
    identity = ["condition_id", "forecast_origin", "target_timestamp", "lead_days"]
    if predictions.duplicated(identity).any():
        raise ValueError("Final-search predictions contain duplicate identities")
    origins = pd.to_datetime(predictions["forecast_origin"], errors="raise")
    targets = pd.to_datetime(predictions["target_timestamp"], errors="raise")
    offsets = (targets - origins).dt.days
    if not (offsets.to_numpy() == predictions["lead_days"].to_numpy()).all():
        raise ValueError("A prediction target timestamp does not equal origin plus lead")
    if set(predictions["lead_days"].astype(int)) != {30, 60, 90, 120, 150, 180}:
        raise ValueError("Unexpected prediction horizons")
    if len(predictions) != 1_745_856 or len(horizons) != 576:
        raise ValueError("Frozen prediction or horizon-metric row count changed")
    return {
        "conditions": len(conditions),
        "trials": len(trials),
        "complete_trials": int(states["COMPLETE"]),
        "failed_trials": int(states["FAIL"]),
        "running_trials": int(states["RUNNING"]),
        "predictions": len(predictions),
        "horizon_metrics": len(horizons),
    }

