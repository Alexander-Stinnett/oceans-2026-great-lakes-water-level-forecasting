# OCEANS 2026 Great Lakes water-level forecasting

This is the standalone scientific-reproduction repository for the submitted
paper *Long-Horizon and Multi-Horizon Deep Learning for Great Lakes Water-Level
Forecasting*. It studies Lake Superior forecasts from 30 to 180 days using
LSTM, N-BEATSx, NHITS, and TFT models across input representations, historical
contexts, and sequence-to-one versus sequence-to-sequence formulations.

The repository contains immutable final-search exports, raw ten-seed RQ3 refit
predictions, exact upstream input files, deterministic preprocessing, analysis
code, manuscript source, table generators, and the publication figure. The
historical GLM development repository is provenance only and is not a runtime
dependency.

## Reproduce the paper

Python 3.11 or 3.12 is supported. Install the lightweight default environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\verify_frozen_artifacts.py
.\.venv\Scripts\python.exe scripts\reproduce_paper.py
```

On Linux or macOS, replace `.\.venv\Scripts\python.exe` with
`.venv/bin/python`.

`requirements-lock.txt` records the complete Windows/Python 3.12 environment
used for the final base, development, and CPU-training validation. It is a
current reproduction lock, not a reconstruction of the historical cluster
environment.

Paper reproduction verifies every frozen hash, validates the 336-condition HPO
exports, regenerates RQ1/RQ2/RQ3 tables, regenerates the RQ1 figure, and checks
all manuscript numerical goldens. It does not import or require Torch,
NeuralForecast, Optuna, SQLAlchemy, psycopg, Horizon, or PostgreSQL.

Generated files are written to `paper/generated_tables`,
`paper/figures/generated`, and `artifacts/frozen/rq3/manuscript_analysis`.

## Dataset

The study uses seven daily Lake Superior variables from 1981 through 2023. The
three exact upstream files are in `data/upstream`. Deterministic preprocessing
converts temperature units where required, constructs causal trailing averages
at 30/60/90/120/150/180 days, and drops the first 179 incomplete rows. The
canonical dataset has 15,526 rows and SHA-256
`cdb8c4e0ad99f9c3363e193306580c6403cd30079e42a6e4f8a748555edbe0b9`.

Rebuild it without overwriting the frozen copy:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_dataset.py --output tmp\combined_full.csv
```

The build fails unless its output is byte-identical to the canonical artifact.

## Scientific design

- Four model families: AutoLSTM, AutoNBEATSx, AutoNHITS, AutoTFT.
- Three input representations.
- Contexts of 180, 360, 540, and 720 days.
- Six point horizons and one six-output formulation.
- 336 conditions and 50 successful Optuna trials per condition.
- Validation-only model selection; test metrics are never used for selection.
- Final refit on combined train and validation.

The exact historical configuration and search space are preserved in
`configs/historical_oceans_final_search.yaml` and
`src/oceans_glwl/training/search_space.py`.

## RQ3 historical-analysis note

The historical RQ3 analysis implementation did not explicitly enforce the
lower held-out test-period boundary when constructing the common
forecast-origin surface. Its archived derived products include 30 pre-test
origins from June 2015. The submitted manuscript uses the declared held-out
interval from July 1, 2015 through July 4, 2023, exactly 2,926 origins. This
repository enforces that interval and reproduces all manuscript RQ3 values from
the frozen row-level refit predictions. Historical 2,956-origin products remain
unchanged under `artifacts/frozen/rq3/historical_analysis`.

## Local training and HPO

Install training support:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[training]"
```

Training accepts `device=auto|cpu|cuda|mps` without hostname restrictions.
`auto` chooses CUDA, then MPS, then CPU. CPU support is intended for tests,
modification, and small experiments; recreating the complete historical HPO
search on CPU is not practical.

A minimal fixed-configuration run uses the public Python API:

```python
from oceans_glwl.conditions import FinalSearchCondition
from oceans_glwl.training.fixed import run_fixed_experiment

condition = FinalSearchCondition(
    "AutoLSTM",
    input_variant="sparse_30d_smoothed_all",
    context_days=180,
    input_gap_days=30,
    output_mode="seq2one_30d",
    horizon_days=30,
)
result = run_fixed_experiment(
    "data/canonical/combined_full.csv",
    condition=condition,
    training_config={
        "max_steps": 1,
        "learning_rate": 0.001,
        "batch_size": 2,
        "windows_batch_size": 2,
        "scaler_type": "identity",
        "model_kwargs": {
            "encoder_hidden_size": 8,
            "decoder_hidden_size": 8,
            "encoder_n_layers": 1,
            "encoder_dropout": 0.0,
        },
    },
    device="cpu",
    max_samples_per_split=2,
)
```

For local single-process HPO, install `.[training,hpo]` and call
`oceans_glwl.training.hpo.run_local_hpo`. An in-memory study is the default;
SQLite or another Optuna URL can be supplied explicitly.

```python
from oceans_glwl.conditions import FinalSearchCondition
from oceans_glwl.training.hpo import run_local_hpo

condition = FinalSearchCondition(
    "AutoLSTM",
    input_variant="sparse_30d_smoothed_all",
    context_days=180,
    input_gap_days=30,
    output_mode="seq2one_30d",
    horizon_days=30,
)
study = run_local_hpo(
    "data/canonical/combined_full.csv",
    condition=condition,
    n_trials=1,
    device="cpu",
    max_steps=1,
    max_samples_per_split=2,
)
```

## Optional distributed execution

Historical PostgreSQL queue, leasing, fencing, planning, worker, and host-export
support lives under `oceans_glwl.optional.distributed`. Install
`.[training,distributed]` only if needed. No default path imports it.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Training and distributed integration tests are separately marked. The default
suite covers hashes, deterministic preprocessing, condition IDs, HPO exports,
prediction identities, RQ1/RQ2/RQ3 goldens, origin boundaries, and portable
device resolution.

## Provenance, licensing, and citation

Machine-readable hashes are in `artifacts/manifests/artifacts.json`. See
`docs/provenance.md` and `docs/reproducibility.md` for lineage.

Original code is provisionally MIT-licensed. The Zenodo-derived files are not
MIT-licensed; see `DATA_LICENSE.md`. The identifiable upstream release is
Zenodo DOI `10.5281/zenodo.15276228`, labeled GPL-3.0-or-later at record level.
The three preserved input CSVs have been verified byte-for-byte against the
current official archive. No unsupported claim is made about underlying
NOAA/GLSEA rights.

Citation metadata are provided in `CITATION.cff`.
