from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from oceans_glwl.conditions import FinalSearchCondition
from oceans_glwl.data.samples import ForecastSamples


def regression_metrics(actual: Any, prediction: Any) -> dict[str, float | int]:
    y = np.asarray(actual, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != y_hat.shape or y.size == 0:
        raise ValueError("Metric arrays must share a non-empty shape.")
    if not np.isfinite(y).all() or not np.isfinite(y_hat).all():
        raise ValueError("Metric arrays must be finite.")
    error = y_hat - y
    mse = float(np.mean(error**2))
    return {
        "n": int(y.size),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def prediction_frame_from_samples(
    *,
    samples: ForecastSamples,
    predictions: np.ndarray,
    condition: FinalSearchCondition,
    model_name: str | None = None,
) -> pd.DataFrame:
    predicted = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(samples.target_values, dtype=np.float64)
    if predicted.shape != actual.shape:
        raise ValueError(f"Predictions must match targets: {predicted.shape} != {actual.shape}.")
    rows: list[dict[str, Any]] = []
    for sample_index, metadata in samples.metadata.reset_index(drop=True).iterrows():
        for horizon_index, (lead_day, target_date) in enumerate(
            zip(metadata["lead_days"], metadata["target_dates"], strict=True),
            start=1,
        ):
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "sample_id": metadata["sample_id"],
                    "split": metadata["split"],
                    "forecast_origin": pd.Timestamp(metadata["forecast_origin_date"]),
                    "target_timestamp": pd.Timestamp(target_date),
                    "horizon_index": horizon_index,
                    "lead_days": int(lead_day),
                    "is_endpoint": int(lead_day) == condition.horizon_days,
                    "actual": float(actual[sample_index, horizon_index - 1]),
                    "prediction": float(predicted[sample_index, horizon_index - 1]),
                    "model_name": model_name or condition.model_name,
                    "input_variant": condition.input_variant,
                    "output_mode": condition.output_mode,
                    "context_days": condition.context_days,
                }
            )
    return pd.DataFrame(rows)


def prediction_metrics(predictions: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    required = {"actual", "prediction", "horizon_index", "lead_days", "is_endpoint"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise KeyError(f"Predictions missing metric columns: {missing}.")
    endpoint = predictions.loc[predictions["is_endpoint"].astype(bool)]
    if endpoint.empty:
        raise ValueError("No endpoint rows are available.")
    summary: dict[str, float | int] = {
        **{f"endpoint_{key}": value for key, value in regression_metrics(endpoint["actual"], endpoint["prediction"]).items()},
        **{f"full_horizon_{key}": value for key, value in regression_metrics(predictions["actual"], predictions["prediction"]).items()},
    }
    rows = []
    for (index, lead), group in predictions.groupby(["horizon_index", "lead_days"], sort=True):
        rows.append({"horizon_index": int(index), "lead_days": int(lead), **regression_metrics(group["actual"], group["prediction"])})
    by_horizon = pd.DataFrame(rows)
    summary["mean_horizon_rmse"] = float(by_horizon["rmse"].mean())
    return summary, by_horizon
