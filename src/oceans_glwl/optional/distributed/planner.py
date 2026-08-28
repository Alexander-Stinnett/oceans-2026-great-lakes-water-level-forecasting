from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from typing import Any

import optuna

from oceans_glwl.conditions import FINAL_MODEL_NAMES, FinalSearchCondition
from oceans_glwl.training.search_space import SEARCH_SPACE_VERSION

EXPECTED_CONDITION_COUNT = 336
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
            FINAL_MODEL_NAMES,
            INPUT_FORMULATIONS,
            CONTEXT_DAYS,
            OUTPUT_FORMULATIONS,
        )
    ]
    ids = [condition.condition_id for condition in conditions]
    studies = [condition.study_name for condition in conditions]
    if len(conditions) != EXPECTED_CONDITION_COUNT:
        raise AssertionError(f"Expected {EXPECTED_CONDITION_COUNT} conditions, got {len(conditions)}.")
    if len(set(ids)) != len(ids) or len(set(studies)) != len(studies):
        raise AssertionError("Final-search condition or study identity collision detected.")
    return conditions


def plan_production(
    *,
    queue: Any,
    optuna_storage: Any,
    target_n_trials: int,
    conditions: Iterable[FinalSearchCondition] | None = None,
) -> dict[str, int]:
    if int(target_n_trials) <= 0:
        raise ValueError("target_n_trials must be positive.")
    items = list(conditions or enumerate_conditions())
    if len(items) != EXPECTED_CONDITION_COUNT:
        raise AssertionError(f"Production planning requires exactly {EXPECTED_CONDITION_COUNT} conditions.")
    inserted = queue.insert_conditions(items, target_n_trials=int(target_n_trials))
    for condition in items:
        study = optuna.create_study(
            study_name=condition.study_name,
            storage=optuna_storage,
            direction="minimize",
            load_if_exists=True,
        )
        expected_attributes = {
            "condition_id": condition.condition_id,
            "scientific_identity": condition.identity_payload,
            "search_space_version": SEARCH_SPACE_VERSION,
        }
        for name, expected in expected_attributes.items():
            existing = study.user_attrs.get(name)
            if existing is not None and existing != expected:
                raise RuntimeError(
                    f"Study {condition.study_name} has incompatible {name}: {existing!r} != {expected!r}."
                )
            study.set_user_attr(name, expected)
    return {"conditions": len(items), "inserted": int(inserted), "existing": len(items) - int(inserted)}
