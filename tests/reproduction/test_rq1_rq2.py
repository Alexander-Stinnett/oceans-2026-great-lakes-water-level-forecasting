import math
from typing import cast

import pandas as pd

from oceans_glwl.analysis.goldens import RQ1_EXPECTED, RQ2_EXPECTED

RQ1_CONDITION_IDS = {
    "AutoLSTM": "ofs1__autolstm__daily_smoothed_y_smoothed_exog__ctx540__seq2one_180d__5377ce08298a",
    "AutoNBEATSx": "ofs1__autonbeatsx__sparse_30d_smoothed_all__ctx720__seq2one_180d__a6d7a9c161de",
    "AutoNHITS": "ofs1__autonhits__daily_smoothed_y_smoothed_exog__ctx720__seq2one_180d__cb22e7679fc3",
    "AutoTFT": "ofs1__autotft__sparse_30d_smoothed_all__ctx360__seq2one_180d__13941b3fe1c7",
}

RQ2_POINT_IDS = {
    30: "ofs1__autotft__sparse_30d_smoothed_all__ctx360__seq2one_30d__46c32a89a066",
    60: "ofs1__autolstm__sparse_30d_smoothed_all__ctx360__seq2one_60d__a7650318b632",
    90: "ofs1__autotft__daily_smoothed_y_raw_exog__ctx720__seq2one_90d__3858209314dd",
    120: "ofs1__autolstm__sparse_30d_smoothed_all__ctx360__seq2one_120d__c24073a2022a",
    150: "ofs1__autotft__sparse_30d_smoothed_all__ctx720__seq2one_150d__f9b8d69e831a",
    180: "ofs1__autonbeatsx__sparse_30d_smoothed_all__ctx720__seq2one_180d__a6d7a9c161de",
}
RQ2_SEQUENCE_ID = "ofs1__autonhits__daily_smoothed_y_raw_exog__ctx540__seq2seq_6x30d__9414a1b54ccd"


def test_rq1_exact_values(rq1_result: pd.DataFrame) -> None:
    assert set(rq1_result["architecture"]) == set(RQ1_EXPECTED)
    for architecture, (validation, test) in RQ1_EXPECTED.items():
        row = rq1_result.loc[rq1_result["architecture"].eq(architecture)].iloc[0]
        assert math.isclose(row["validation_rmse_cm"], validation, abs_tol=1e-10)
        assert math.isclose(row["test_rmse_cm"], test, abs_tol=1e-8)
        assert row["condition_id"] == RQ1_CONDITION_IDS[architecture]
        assert row["forecast_origins"] == 2_926


def test_rq2_exact_values(rq2_result: tuple[pd.DataFrame, dict[str, object]]) -> None:
    table, summary = rq2_result
    for lead, expected in RQ2_EXPECTED.items():
        row = table.loc[table["horizon_days"].eq(lead)].iloc[0]
        assert math.isclose(row["seq2one_rmse_cm"], expected[0], abs_tol=1e-5)
        assert math.isclose(row["seq2seq_rmse_cm"], expected[1], abs_tol=1e-5)
        assert math.isclose(row["relative_difference_percent"], expected[2], abs_tol=1e-5)
        assert row["seq2one_condition_id"] == RQ2_POINT_IDS[lead]
        assert row["seq2seq_condition_id"] == RQ2_SEQUENCE_ID
        assert row["forecast_origins"] == 2_926
    assert math.isclose(
        cast(float, summary["mean_relative_difference_percent"]), -0.8202857064, abs_tol=1e-8
    )
