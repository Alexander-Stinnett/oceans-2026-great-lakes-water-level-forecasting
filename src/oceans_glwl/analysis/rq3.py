from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from oceans_glwl.analysis.common_origins import manuscript_surface
from oceans_glwl.config import TARGET_COLUMN, input_offsets_for_window
from oceans_glwl.data.splits import build_chronological_splits

HORIZONS = (30, 60, 90, 120, 150, 180)
SEEDS = tuple(range(1337, 1347))
GLOBAL_REASON = "rq2_global_winner"
ARCHITECTURE_REASON = "rq1_180d_architecture_winner"
PREDICTION_IDENTITY = ("forecast_origin", "target_timestamp", "lead_days")


def validate_rq3_provenance(
    frozen: dict[str, Any], policy: dict[str, Any], plan: dict[str, Any], statuses: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    slots = frozen["selection_slots"]
    if len(slots) != 11 or len({slot["condition_id"] for slot in slots}) != 10:
        raise ValueError("RQ3 freeze must contain 11 slots and 10 unique conditions")
    reasons = pd.Series([slot["reason"] for slot in slots]).value_counts().to_dict()
    if reasons != {GLOBAL_REASON: 7, ARCHITECTURE_REASON: 4}:
        raise ValueError(f"Unexpected RQ3 selection-slot coverage: {reasons}")
    policy_seeds = tuple(int(seed) for seed in policy["refits"]["seeds"])
    if policy_seeds != SEEDS:
        raise ValueError(f"Unexpected RQ3 seed policy: {policy_seeds}")
    if int(policy["refits"]["optimizer_updates"]) != 750:
        raise ValueError("RQ3 refits must use 750 optimizer updates")
    tasks = plan["tasks"]
    if len(tasks) != 100 or int(plan["max_steps"]) != 750:
        raise ValueError("RQ3 plan must contain 100 fixed 750-step refits")
    if any(bool(task["new_hpo"]) or int(task["max_steps"]) != 750 for task in tasks):
        raise ValueError("RQ3 tasks must be fixed configurations with no new HPO and 750 updates")
    selected_ids = {slot["condition_id"] for slot in slots}
    expected_pairs = {(condition_id, seed) for condition_id in selected_ids for seed in SEEDS}
    task_pairs = {(task["source_condition_id"], int(task["seed"])) for task in tasks}
    if task_pairs != expected_pairs:
        raise ValueError("RQ3 task plan does not cover every selected condition and seed exactly once")
    if len(statuses) != 100 or statuses["rq3_refit_id"].nunique() != 100:
        raise ValueError("RQ3 status export must contain 100 unique refits")
    if set(statuses["seed"].astype(int)) != set(SEEDS):
        raise ValueError("RQ3 status seeds do not match the frozen policy")
    status_pairs = set(zip(statuses["source_condition_id"], statuses["seed"].astype(int), strict=True))
    if status_pairs != expected_pairs:
        raise ValueError("RQ3 status export does not cover every selected condition and seed exactly once")
    if not statuses["status"].eq("satisfied_refit_attempt").all():
        raise ValueError("RQ3 status export contains an unsatisfied refit")
    required = {"rq3_refit_id", "source_condition_id", "seed", "forecast_origin", "target_timestamp", "lead_days", "actual", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"RQ3 predictions are missing columns: {sorted(missing)}")
    if predictions.duplicated(["rq3_refit_id", *PREDICTION_IDENTITY]).any():
        raise ValueError("RQ3 predictions contain duplicate identities")
    origins = pd.to_datetime(predictions["forecast_origin"], errors="raise")
    targets = pd.to_datetime(predictions["target_timestamp"], errors="raise")
    if not ((targets - origins).dt.days.to_numpy() == predictions["lead_days"].to_numpy()).all():
        raise ValueError("An RQ3 prediction target timestamp does not equal origin plus lead")
    if set(predictions["rq3_refit_id"].astype(str)) != set(statuses["rq3_refit_id"].astype(str)):
        raise ValueError("RQ3 prediction and status refit coverage differs")


def training_regime_thresholds(frame: pd.DataFrame, frozen: dict[str, Any]) -> pd.DataFrame:
    splits = build_chronological_splits(frame)
    minimum_origin = max(
        int(input_offsets_for_window(context_days=int(record["condition"]["context_days"]), input_gap_days=int(record["condition"]["input_gap_days"]))[0])
        for record in frozen["conditions"]
    )
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy(dtype=np.float64)
    dates = pd.to_datetime(frame["date"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for lead in HORIZONS:
        origins = np.arange(max(minimum_origin, splits.train.start_index), splits.validation.end_index - lead + 1)
        changes = target[origins + lead] - target[origins]
        lower, upper = np.quantile(changes, [1 / 3, 2 / 3])
        rows.append(
            {
                "lead_days": lead,
                "lower_tercile_m": float(lower),
                "upper_tercile_m": float(upper),
                "development_origin_count": len(origins),
                "first_development_origin": dates.iloc[int(origins[0])],
                "last_development_origin": dates.iloc[int(origins[-1])],
                "threshold_period": "combined_train_plus_validation",
            }
        )
    return pd.DataFrame(rows)


def _decode(frame: pd.DataFrame, predictions: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["forecast_origin"] = pd.to_datetime(result["forecast_origin"])
    result["target_timestamp"] = pd.to_datetime(result["target_timestamp"])
    lookup = frame[["date", TARGET_COLUMN]].copy()
    lookup["date"] = pd.to_datetime(lookup["date"])
    result = result.merge(
        lookup.rename(columns={"date": "forecast_origin", TARGET_COLUMN: "origin_actual"}),
        on="forecast_origin",
        validate="many_to_one",
    ).merge(
        lookup.rename(columns={"date": "target_timestamp", TARGET_COLUMN: "frame_target_actual"}),
        on="target_timestamp",
        validate="many_to_one",
    )
    if not np.allclose(result["actual"], result["frame_target_actual"], atol=1e-4, rtol=0):
        raise ValueError("RQ3 actual values disagree with the canonical target")
    result = result.merge(thresholds, on="lead_days", validate="many_to_one")
    result["observed_change_m"] = result["frame_target_actual"] - result["origin_actual"]
    result["regime"] = np.where(
        result["observed_change_m"] < result["lower_tercile_m"],
        "falling",
        np.where(result["observed_change_m"] > result["upper_tercile_m"], "rising", "stable"),
    )
    return result


def _seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["system", "seed", "lead_days", "regime"], sort=True):
        error = group["actual"].to_numpy(dtype=np.float64) - group["prediction"].to_numpy(dtype=np.float64)
        rmse_m = float(np.sqrt(np.mean(np.square(error))))
        rows.append(
            {
                "system": keys[0],
                "seed": int(keys[1]),
                "lead_days": int(keys[2]),
                "regime": keys[3],
                "row_count": len(group),
                "origin_count": group["forecast_origin"].nunique(),
                "rmse_m": rmse_m,
                "rmse_cm": rmse_m * 100,
                "mae_m": float(np.mean(np.abs(error))),
                "bias_m": float(np.mean(-error)),
            }
        )
    return pd.DataFrame(rows)


def _aggregate(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    result = seed_metrics.groupby(["system", "lead_days", "regime"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        origin_count=("origin_count", "max"),
        rmse_cm_seed_mean=("rmse_cm", "mean"),
        rmse_cm_seed_std=("rmse_cm", "std"),
        rmse_cm_seed_min=("rmse_cm", "min"),
        rmse_cm_seed_max=("rmse_cm", "max"),
    )
    if not result["seed_count"].eq(10).all():
        raise ValueError("Every RQ3 aggregate must contain all ten seeds")
    return result


def analyze_rq3(
    frame: pd.DataFrame,
    frozen: dict[str, Any],
    policy: dict[str, Any],
    plan: dict[str, Any],
    statuses: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    validate_rq3_provenance(frozen, policy, plan, statuses, predictions)
    thresholds = training_regime_thresholds(frame, frozen)
    decoded = manuscript_surface(_decode(frame, predictions, thresholds))
    global_slots = {slot["output_mode"]: slot["condition_id"] for slot in frozen["selection_slots"] if slot["reason"] == GLOBAL_REASON}
    parts: list[pd.DataFrame] = []
    for lead in HORIZONS:
        point = decoded.loc[
            decoded["source_condition_id"].eq(global_slots[f"seq2one_{lead}d"])
            & decoded["lead_days"].eq(lead)
        ].copy()
        sequence = decoded.loc[
            decoded["source_condition_id"].eq(global_slots["seq2seq_6x30d"])
            & decoded["lead_days"].eq(lead)
        ].copy()
        point["system"] = "global_seq2one"
        sequence["system"] = "global_seq2seq_6x30d"
        parts.extend([point, sequence])
    matched = pd.concat(parts, ignore_index=True)
    expected_rows = 2 * len(HORIZONS) * len(SEEDS) * 2_926
    if len(matched) != expected_rows:
        raise ValueError(f"Expected {expected_rows} matched neural predictions, found {len(matched)}")
    seed_metrics = _seed_metrics(matched)
    aggregate = _aggregate(seed_metrics)

    reference = matched.loc[matched["system"].eq("global_seq2one") & matched["seed"].eq(SEEDS[0])].copy()
    reference["system"] = "persistence"
    reference["prediction"] = reference["origin_actual"]
    persistence_rows: list[dict[str, object]] = []
    for keys, group in reference.groupby(["lead_days", "regime"], sort=True):
        error = group["actual"].to_numpy(dtype=np.float64) - group["prediction"].to_numpy(dtype=np.float64)
        rmse_m = float(np.sqrt(np.mean(np.square(error))))
        persistence_rows.append(
            {
                "system": "persistence",
                "lead_days": int(keys[0]),
                "regime": keys[1],
                "row_count": len(group),
                "origin_count": group["forecast_origin"].nunique(),
                "rmse_m": rmse_m,
                "rmse_cm": rmse_m * 100,
            }
        )
    persistence = pd.DataFrame(persistence_rows)

    architecture_ids = {
        record["architecture"]: record["condition_id"]
        for record in frozen["conditions"]
        if ARCHITECTURE_REASON in record["inclusion_reasons"]
    }
    architecture_parts: list[pd.DataFrame] = []
    for architecture, condition_id in architecture_ids.items():
        selected = decoded.loc[decoded["source_condition_id"].eq(condition_id) & decoded["lead_days"].eq(180)].copy()
        selected["system"] = architecture
        architecture_parts.append(selected)
    architecture_by_seed = _seed_metrics(pd.concat(architecture_parts, ignore_index=True))
    architecture_variability = _aggregate(architecture_by_seed)
    return {
        "regime_thresholds": thresholds,
        "manuscript_predictions": matched,
        "regime_rmse_by_seed": seed_metrics,
        "regime_rmse_aggregate": aggregate,
        "persistence_regime_rmse": persistence,
        "architecture_180d_rmse_by_seed": architecture_by_seed,
        "architecture_180d_seed_variability": architecture_variability,
    }
