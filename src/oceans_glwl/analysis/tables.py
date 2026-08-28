from __future__ import annotations

from pathlib import Path

import pandas as pd


def manuscript_rq3_table(aggregate: pd.DataFrame, persistence: pd.DataFrame) -> pd.DataFrame:
    regimes = ("falling", "stable", "rising")
    rows: list[dict[str, object]] = []
    for lead in (30, 60, 90, 120, 150, 180):
        for regime in regimes:
            one = aggregate.loc[
                aggregate["system"].eq("global_seq2one")
                & aggregate["lead_days"].eq(lead)
                & aggregate["regime"].eq(regime)
            ].iloc[0]
            seq = aggregate.loc[
                aggregate["system"].eq("global_seq2seq_6x30d")
                & aggregate["lead_days"].eq(lead)
                & aggregate["regime"].eq(regime)
            ].iloc[0]
            base = persistence.loc[persistence["lead_days"].eq(lead) & persistence["regime"].eq(regime)].iloc[0]
            rows.append(
                {
                    "horizon_days": lead,
                    "regime": regime.title(),
                    "seq2one_rmse_cm_mean": one["rmse_cm_seed_mean"],
                    "seq2one_rmse_cm_sd": one["rmse_cm_seed_std"],
                    "seq2seq_rmse_cm_mean": seq["rmse_cm_seed_mean"],
                    "seq2seq_rmse_cm_sd": seq["rmse_cm_seed_std"],
                    "persistence_rmse_cm": base["rmse_cm"],
                }
            )
    return pd.DataFrame(rows)


def write_tables(output_dir: str | Path, rq1: pd.DataFrame, rq2: pd.DataFrame, rq3: pd.DataFrame) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in (("rq1", rq1), ("rq2", rq2), ("rq3", rq3)):
        csv_path = root / f"{name}.csv"
        tex_path = root / f"{name}.tex"
        frame.to_csv(csv_path, index=False)
        tex_path.write_text(frame.to_latex(index=False, float_format=lambda value: f"{value:.2f}"), encoding="utf-8")
        written.extend([csv_path, tex_path])
    return written

