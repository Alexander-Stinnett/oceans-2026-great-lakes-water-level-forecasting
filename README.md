# OCEANS 2026 Great Lakes Water-Level Forecasting

Code and reproducibility materials for the paper *Long-Horizon and Multi-Horizon Deep Learning for Great Lakes Water-Level Forecasting*.

This study evaluates Lake Superior water-level forecasting from 30 to 180 days using LSTM, N-BEATSx, NHITS, and TFT models. The experiments compare multiple input representations, historical context lengths, and sequence-to-one versus sequence-to-sequence forecasting.

This repository contains the final experiment outputs used in the paper, the data-processing pipeline, RQ1–RQ3 analysis code, manuscript source, and scripts for regenerating the reported tables and figures.

## Reproduce the paper

Python 3.11 and 3.12 are supported.

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\verify_frozen_artifacts.py
.\.venv\Scripts\python.exe scripts\reproduce_paper.py
```

On Linux or macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

Paper reproduction:

* verifies the stored experiment artifacts;
* regenerates the RQ1, RQ2, and RQ3 results;
* regenerates the paper figure;
* checks the numerical results against the submitted manuscript.

Rerunning the full hyperparameter search is **not required** to reproduce the paper.

`requirements-lock.txt` records the environment used for final repository validation. It should not be interpreted as a reconstruction of the original distributed experiment environment.

## Supplemental figures

`additional_figures/` contains reproducible presentation figures derived from
the canonical dataset and frozen experiment artifacts. Each PNG/PDF pair has a
JSON manifest recording repository-relative source paths and SHA-256 hashes.
Regenerate the set with:

```powershell
.\.venv\Scripts\python.exe -m oceans_glwl.plotting.forecasting_formulations
.\.venv\Scripts\python.exe -m oceans_glwl.plotting.historical_water_level
.\.venv\Scripts\python.exe -m oceans_glwl.plotting.rq2_best_models_rmse
.\.venv\Scripts\python.exe -m oceans_glwl.plotting.test_water_level_groups
```

Generated outputs are written under:

```text
paper/generated_tables/
paper/figures/generated/
artifacts/frozen/rq3/manuscript_analysis/
```

## Dataset

The study uses seven daily Lake Superior variables spanning 1981–2023.

The three original input CSVs are stored in `data/upstream/` and were obtained from the Dual Transformer Zenodo release. The preprocessing pipeline performs the transformations used in the final experiment, including causal trailing averages at 30, 60, 90, 120, 150, and 180 days.

The final dataset contains 15,526 daily rows.

To rebuild it:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_dataset.py --output tmp\combined_full.csv
```

The resulting file is checked against the dataset used in the final experiments.

Canonical SHA-256:

```text
cdb8c4e0ad99f9c3363e193306580c6403cd30079e42a6e4f8a748555edbe0b9
```

## Experimental design

The final search evaluated:

* four architectures: LSTM, N-BEATSx, NHITS, and TFT;
* three input representations;
* context lengths of 180, 360, 540, and 720 days;
* sequence-to-one forecasts at 30, 60, 90, 120, 150, and 180 days;
* one sequence-to-sequence formulation predicting all six horizons.

This produced 336 experimental conditions with 50 completed Optuna trials per condition.

Hyperparameters were selected using validation performance only. The selected configurations were then refit using the combined training and validation data before final test evaluation.

The exact search configuration is preserved in:

```text
configs/historical_oceans_final_search.yaml
src/oceans_glwl/training/search_space.py
```

## RQ3 reproduction note

The original RQ3 analysis code included 30 forecast origins from June 2015 before the declared test interval. The submitted paper uses the intended held-out evaluation period from July 1, 2015 through July 4, 2023, giving 2,926 forecast origins.

The reproduction code explicitly applies this interval and reproduces all RQ3 values reported in the manuscript from the saved row-level refit predictions.

The original 2,956-origin analysis outputs are retained for provenance under:

```text
artifacts/frozen/rq3/historical_analysis/
```

## Local training

Install the training dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[training]"
```

Training supports:

```text
auto
cpu
cuda
mps
```

`auto` selects CUDA when available, followed by MPS and then CPU.

A minimal example:

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

CPU support is intended primarily for testing and smaller experiments. Reproducing the complete historical HPO search requires substantially more compute.

## Local HPO

Install the HPO dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[training,hpo]"
```

A local Optuna search can be run through `run_local_hpo`:

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

The default uses an in-memory Optuna study. Other Optuna storage backends can also be supplied.

## Distributed execution

The final experiments were executed using distributed infrastructure built around PostgreSQL-backed experiment coordination.

That implementation is retained under:

```text
oceans_glwl.optional.distributed
```

It is optional and is not required for reproducing the paper or running local experiments.

Install it with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[training,distributed]"
```

## Tests

Install development dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite covers dataset reconstruction, experimental-condition generation, frozen HPO outputs, prediction identities, RQ1–RQ3 reproduction, evaluation boundaries, and device selection.

## Repository structure

```text
src/oceans_glwl/       Scientific implementation
configs/               Historical and reproduction configurations
data/                  Upstream and processed data
artifacts/frozen/      Final experiment outputs
paper/                 Submitted manuscript and generated outputs
scripts/               Reproduction and verification commands
tests/                 Automated tests
docs/                  Reproducibility and provenance documentation
```

## Provenance

Artifact hashes and metadata are recorded in:

```text
artifacts/manifests/artifacts.json
```

Additional details are available in:

* `docs/provenance.md`
* `docs/reproducibility.md`
* `docs/artifact_inventory.md`

## Licensing

Repository source code is licensed under the MIT License.

The upstream dataset and other preserved external materials are not covered by the repository's MIT license. See `DATA_LICENSE.md` for details.

The Lake Superior input files were obtained from:

**Chen, Y. and Xue, P.**
*Dual-Transformer Deep Learning Framework for Seasonal Forecasting of Great Lakes Water Levels*
Zenodo: `10.5281/zenodo.15276228`

The three input CSVs preserved here were verified byte-for-byte against the corresponding files in the current Zenodo archive.

## Citation

Citation metadata are provided in `CITATION.cff`.
