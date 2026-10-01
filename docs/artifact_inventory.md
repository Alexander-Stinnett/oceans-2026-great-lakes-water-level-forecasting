# Migration and artifact inventory

The machine-readable source of truth for immutable bytes, sizes, row counts,
schemas, parent artifacts, and SHA-256 values is
`artifacts/manifests/artifacts.json`. This document explains each retained
group and the file-level migration boundary.

## Frozen scientific evidence

| Path | Role | Authority |
| --- | --- | --- |
| `data/upstream/finaldata.csv` | 1981-2021 upstream observations | Exact historical input; never normalized in place |
| `data/upstream/test_2022.csv` | 2022 extension | Exact historical input |
| `data/upstream/test_2023.csv` | 2023 extension | Exact historical input |
| `data/canonical/combined_full.csv` | Deterministic seven-variable daily dataset with causal averages | Canonical dataset golden |
| `artifacts/frozen/final_search/condition_results_ALL.parquet` | Condition-level validation selection and test summary | Authoritative main-search export |
| `artifacts/frozen/final_search/validation_trials_ALL.parquet` | All 17,204 historical Optuna rows | Authoritative HPO provenance |
| `artifacts/frozen/final_search/test_predictions_ALL.parquet` | Row-level test predictions | Authoritative RQ1/RQ2 predictions |
| `artifacts/frozen/final_search/test_horizon_metrics_ALL.parquet` | Per-horizon test metrics | Authoritative exported metric evidence |
| `artifacts/frozen/rq3/raw/frozen_condition_index.json` | Ten selected fixed-refit conditions | Authoritative RQ3 selection provenance |
| `artifacts/frozen/rq3/raw/analysis_policy.json` | Historical analysis policy | Frozen policy provenance; manuscript code additionally enforces the test lower bound |
| `artifacts/frozen/rq3/raw/seed_refit_plan.json` | Seeds 1337-1346 and intended 100 refits | Authoritative refit plan |
| `artifacts/frozen/rq3/raw/rq3_refit_status.parquet` | Fixed-refit status and execution metadata | Authoritative refit status |
| `artifacts/frozen/rq3/raw/rq3_refit_predictions.parquet` | Row-level predictions for all fixed refits | Lowest-level authoritative RQ3 numerical evidence |

Every file under `artifacts/frozen/rq3/historical_analysis/` is retained
byte-for-byte as provenance for the archived 2,956-origin implementation. It
is not a manuscript golden. Every file under
`artifacts/frozen/rq3/manuscript_analysis/` is deterministically regenerated
from the raw predictions over exactly 2,926 held-out origins by
`scripts/reproduce_paper.py`.

## Manuscript evidence

`paper/main.tex` and `paper/references.bib` are exact copies of the submitted
sources. The three `rq1_seq2one_180d_test_predictions_styled_wip.*`/clean-name
files under `paper/figures` retain the submitted RQ1 figure and a stable clean
alias. `paper/figures/generated/` and `paper/generated_tables/` are ignored
working outputs regenerated from frozen evidence.

`additional_figures/` contains tracked supplemental presentation figures. Their
generators live in `src/oceans_glwl/plotting/`; each figure is accompanied by a
manifest with repository-relative source paths and content hashes. These
figures are supplemental and do not replace the submitted manuscript evidence.

## Reviewed code migration

| Publication file | Historical GLM source or disposition |
| --- | --- |
| `src/oceans_glwl/config.py` | Refactored from the final-search configuration contract |
| `src/oceans_glwl/conditions.py` | Ported canonical `ofs1` condition serialization and enumeration |
| `src/oceans_glwl/data/{validate,splits,samples}.py` | Ported scientific data, split, and window contracts |
| `src/oceans_glwl/data/{build,schema,views}.py` | Publication-local deterministic build and explicit schemas/views |
| `src/oceans_glwl/models/neuralforecast.py` | Ported backend adapter with portable device policy |
| `src/oceans_glwl/models/protocol.py` | Publication-local adapter protocol |
| `src/oceans_glwl/training/search_space.py` | Exact historical HPO parameter domains |
| `src/oceans_glwl/training/{fixed,hpo,devices}.py` | Publication-local portable fixed/HPO execution interfaces |
| `src/oceans_glwl/evaluation/metrics.py` | Ported metric definitions and validation |
| `src/oceans_glwl/analysis/*.py` | New frozen-evidence selection, RQ1/RQ2/RQ3, table, contract, and golden layer |
| `src/oceans_glwl/plotting/rq1_predictions.py` | Ported and path-parameterized submitted-figure generator |
| `src/oceans_glwl/artifacts/*.py` | New hash manifest, path, IO, and artifact contracts |
| `src/oceans_glwl/paper.py`, `cli.py` | New lightweight public reproduction entrypoints |
| `src/oceans_glwl/optional/distributed/*.py` | Reviewed copy/refactor of PostgreSQL queue, planner, worker, and export support |

All package imports are repository-local; no module imports GLM or Horizon.
Distributed code is isolated behind the `distributed` extra and is not loaded
by the default reproduction path.

## Intentionally excluded GLM material

The migration excludes every unrelated experiment, legacy model prototype,
exploratory notebook, transient log, checkpoint, database dump, cache,
Horizon checkout, host deployment wrapper, and non-OCEANS production output.
It also excludes historical generated duplicates when the authoritative ALL
export or raw RQ3 prediction artifact supersedes them. These files remain in
the read-only GLM development repository and are not runtime dependencies.

The historical production environment was not fully frozen, so this repository
does not claim to recreate its exact Python, CUDA, driver, hostname, or database
state. Known execution facts are retained in `docs/historical_execution.md` and
`configs/distributed_historical.yaml`; they do not constrain ordinary use.
