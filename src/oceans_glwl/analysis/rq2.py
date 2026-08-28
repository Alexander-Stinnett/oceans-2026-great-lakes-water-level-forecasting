from __future__ import annotations

import numpy as np
import pandas as pd

from oceans_glwl.analysis.common_origins import manuscript_surface
from oceans_glwl.analysis.selection import global_winner

HORIZONS = (30, 60, 90, 120, 150, 180)


def _rmse_cm(frame: pd.DataFrame) -> float:
    errors = frame["actual"].to_numpy(dtype=np.float64) - frame["prediction"].to_numpy(dtype=np.float64)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Cannot compute RMSE from empty or non-finite predictions")
    return float(np.sqrt(np.mean(np.square(errors))) * 100.0)


def analyze_rq2(condition_results: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    sequence = global_winner(condition_results, "seq2seq_6x30d")
    sequence_id = str(sequence["condition_id"])
    rows: list[dict[str, object]] = []
    for lead in HORIZONS:
        point = global_winner(condition_results, f"seq2one_{lead}d")
        point_id = str(point["condition_id"])
        point_rows = manuscript_surface(
            predictions.loc[predictions["condition_id"].eq(point_id) & predictions["lead_days"].eq(lead)]
        )
        sequence_rows = manuscript_surface(
            predictions.loc[predictions["condition_id"].eq(sequence_id) & predictions["lead_days"].eq(lead)]
        )
        point_rmse = _rmse_cm(point_rows)
        sequence_rmse = _rmse_cm(sequence_rows)
        rows.append(
            {
                "horizon_days": lead,
                "seq2one_condition_id": point_id,
                "seq2one_model": point["condition.model_name"],
                "seq2one_rmse_cm": point_rmse,
                "seq2seq_condition_id": sequence_id,
                "seq2seq_model": sequence["condition.model_name"],
                "seq2seq_rmse_cm": sequence_rmse,
                "relative_difference_percent": (sequence_rmse - point_rmse) / point_rmse * 100.0,
                "forecast_origins": point_rows["forecast_origin"].nunique(),
            }
        )
    result = pd.DataFrame(rows)
    summary = {
        "seq2seq_condition_id": sequence_id,
        "mean_relative_difference_percent": float(result["relative_difference_percent"].mean()),
        "horizons": list(HORIZONS),
    }
    return result, summary

