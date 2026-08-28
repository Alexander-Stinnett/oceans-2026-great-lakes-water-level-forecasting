from __future__ import annotations

import math
from typing import cast

import pandas as pd

RQ1_EXPECTED = {
    "AutoLSTM": (9.55446843214644, 12.83629361007153),
    "AutoNBEATSx": (8.88252421890696, 12.202414865920387),
    "AutoNHITS": (9.027645479564494, 12.033489439348163),
    "AutoTFT": (8.899207068363102, 15.558650799326298),
}

RQ2_EXPECTED = {
    30: (3.748127, 4.335231, 15.663916),
    60: (6.545398, 6.781731, 3.610671),
    90: (9.051267, 8.459413, -6.538907),
    120: (10.379907, 9.485790, -8.613919),
    150: (10.807644, 10.401400, -3.758860),
    180: (12.202415, 11.557564, -5.284615),
}

# lead, regime, seq2one mean/sd, seq2seq mean/sd, persistence RMSE (cm)
RQ3_PRINTED = (
    (30, "falling", 3.18, 0.21, 3.97, 0.32, 6.67),
    (30, "stable", 3.45, 0.40, 3.86, 0.24, 2.06),
    (30, "rising", 4.78, 0.29, 5.15, 0.16, 9.19),
    (60, "falling", 6.40, 0.73, 5.79, 0.34, 12.14),
    (60, "stable", 5.83, 0.56, 5.90, 0.29, 4.06),
    (60, "rising", 8.02, 0.57, 8.26, 0.35, 16.00),
    (90, "falling", 6.72, 0.94, 6.73, 0.34, 17.44),
    (90, "stable", 8.88, 1.23, 8.51, 0.29, 6.34),
    (90, "rising", 10.54, 0.80, 9.54, 0.57, 20.96),
    (120, "falling", 8.49, 0.82, 8.20, 0.51, 21.59),
    (120, "stable", 10.89, 0.65, 9.88, 0.37, 7.61),
    (120, "rising", 11.73, 0.85, 10.39, 0.67, 24.28),
    (150, "falling", 10.75, 1.08, 9.58, 0.55, 23.88),
    (150, "stable", 12.20, 1.14, 11.15, 0.52, 7.66),
    (150, "rising", 12.42, 1.22, 10.93, 0.69, 26.22),
    (180, "falling", 11.08, 0.29, 10.65, 0.51, 25.69),
    (180, "stable", 13.23, 0.45, 12.46, 0.62, 7.68),
    (180, "rising", 12.16, 0.33, 11.91, 0.87, 27.37),
)


def assert_manuscript_goldens(
    rq1: pd.DataFrame,
    rq2: pd.DataFrame,
    rq2_summary: dict[str, object],
    rq3_aggregate: pd.DataFrame,
    persistence: pd.DataFrame,
) -> dict[str, int]:
    for architecture, expected_rq1 in RQ1_EXPECTED.items():
        row = rq1.loc[rq1["architecture"].eq(architecture)].iloc[0]
        if not math.isclose(float(row["validation_rmse_cm"]), expected_rq1[0], abs_tol=1e-10):
            raise AssertionError(f"RQ1 validation mismatch for {architecture}")
        if not math.isclose(float(row["test_rmse_cm"]), expected_rq1[1], abs_tol=1e-8):
            raise AssertionError(f"RQ1 test mismatch for {architecture}")
    for lead, expected_rq2 in RQ2_EXPECTED.items():
        row = rq2.loc[rq2["horizon_days"].eq(lead)].iloc[0]
        for column, value in zip(
            ("seq2one_rmse_cm", "seq2seq_rmse_cm", "relative_difference_percent"), expected_rq2, strict=True
        ):
            if not math.isclose(float(row[column]), value, abs_tol=1e-5):
                raise AssertionError(f"RQ2 {lead}-day {column} mismatch")
    if not math.isclose(
        cast(float, rq2_summary["mean_relative_difference_percent"]), -0.8202857064, abs_tol=1e-8
    ):
        raise AssertionError("RQ2 mean relative difference mismatch")
    checked = 0
    for lead, regime, one_mean, one_sd, seq_mean, seq_sd, baseline in RQ3_PRINTED:
        one = rq3_aggregate.loc[
            rq3_aggregate["system"].eq("global_seq2one")
            & rq3_aggregate["lead_days"].eq(lead)
            & rq3_aggregate["regime"].eq(regime)
        ].iloc[0]
        seq = rq3_aggregate.loc[
            rq3_aggregate["system"].eq("global_seq2seq_6x30d")
            & rq3_aggregate["lead_days"].eq(lead)
            & rq3_aggregate["regime"].eq(regime)
        ].iloc[0]
        persistence_row = persistence.loc[
            persistence["lead_days"].eq(lead) & persistence["regime"].eq(regime)
        ].iloc[0]
        observed = (
            one["rmse_cm_seed_mean"], one["rmse_cm_seed_std"],
            seq["rmse_cm_seed_mean"], seq["rmse_cm_seed_std"], persistence_row["rmse_cm"],
        )
        expected_rq3 = (one_mean, one_sd, seq_mean, seq_sd, baseline)
        for actual, printed in zip(observed, expected_rq3, strict=True):
            checked += 1
            if f"{float(actual):.2f}" != f"{printed:.2f}":
                raise AssertionError(
                    f"RQ3 {lead}-day {regime} mismatch: expected {printed:.2f}, got {float(actual):.8f}"
                )
    return {"rq1_values": 8, "rq2_values": 19, "rq3_printed_values": checked}
