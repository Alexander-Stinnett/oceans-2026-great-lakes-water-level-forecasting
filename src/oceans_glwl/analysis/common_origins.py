from __future__ import annotations

import pandas as pd

MANUSCRIPT_ORIGIN_START = pd.Timestamp("2015-07-01")
MANUSCRIPT_ORIGIN_END = pd.Timestamp("2023-07-04")
MANUSCRIPT_ORIGIN_COUNT = 2_926


def manuscript_surface(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["forecast_origin"] = pd.to_datetime(result["forecast_origin"])
    result = result.loc[
        result["forecast_origin"].between(MANUSCRIPT_ORIGIN_START, MANUSCRIPT_ORIGIN_END, inclusive="both")
    ].copy()
    origins = result["forecast_origin"].drop_duplicates().sort_values()
    if len(origins) != MANUSCRIPT_ORIGIN_COUNT:
        raise ValueError(f"Expected 2,926 manuscript origins, found {len(origins)}")
    if origins.iloc[0] != MANUSCRIPT_ORIGIN_START or origins.iloc[-1] != MANUSCRIPT_ORIGIN_END:
        raise ValueError("Manuscript origin boundaries do not match the frozen protocol")
    return result

