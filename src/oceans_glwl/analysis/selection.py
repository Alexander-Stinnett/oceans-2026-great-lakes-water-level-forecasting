from __future__ import annotations

from itertools import product

import pandas as pd

from oceans_glwl.conditions import FINAL_MODEL_NAMES, FinalSearchCondition

CONTEXT_DAYS = (180, 360, 540, 720)
INPUT_FORMULATIONS = (
    ("daily_smoothed_y_raw_exog", 1),
    ("daily_smoothed_y_smoothed_exog", 1),
    ("sparse_30d_smoothed_all", 30),
)
OUTPUT_FORMULATIONS = (
    ("seq2one_30d", 30),
    ("seq2one_60d", 60),
    ("seq2one_90d", 90),
    ("seq2one_120d", 120),
    ("seq2one_150d", 150),
    ("seq2one_180d", 180),
    ("seq2seq_6x30d", 180),
)


def enumerate_conditions() -> list[FinalSearchCondition]:
    conditions = [
        FinalSearchCondition(
            model_name=model,
            input_variant=input_variant,
            input_gap_days=input_gap,
            context_days=context,
            output_mode=output_mode,
            horizon_days=horizon,
        )
        for model, (input_variant, input_gap), context, (output_mode, horizon) in product(
            FINAL_MODEL_NAMES, INPUT_FORMULATIONS, CONTEXT_DAYS, OUTPUT_FORMULATIONS
        )
    ]
    if len(conditions) != 336 or len({item.condition_id for item in conditions}) != 336:
        raise AssertionError("Canonical condition enumeration must contain 336 unique conditions")
    return conditions


def validate_condition_coverage(results: pd.DataFrame) -> None:
    expected = {item.condition_id for item in enumerate_conditions()}
    observed = set(results["condition_id"].astype(str))
    if observed != expected or len(results) != 336:
        raise ValueError(f"Condition coverage mismatch: missing={len(expected-observed)}, extra={len(observed-expected)}")


def validation_winners(results: pd.DataFrame, *, output_mode: str) -> pd.DataFrame:
    eligible = results.loc[results["condition.output_mode"].eq(output_mode)].copy()
    if eligible.empty:
        raise ValueError(f"No conditions for {output_mode}")
    return eligible.sort_values(["best.validation_value", "condition_id"], kind="stable")


def architecture_winners(results: pd.DataFrame, *, output_mode: str = "seq2one_180d") -> pd.DataFrame:
    eligible = results.loc[results["condition.output_mode"].eq(output_mode)].copy()
    selected = eligible.sort_values(["best.validation_value", "condition_id"], kind="stable").groupby(
        "condition.model_name", as_index=False, sort=True
    ).first()
    if len(selected) != 4:
        raise ValueError("Expected one winner for each of four architectures")
    return selected


def global_winner(results: pd.DataFrame, output_mode: str) -> pd.Series:
    return validation_winners(results, output_mode=output_mode).iloc[0]

