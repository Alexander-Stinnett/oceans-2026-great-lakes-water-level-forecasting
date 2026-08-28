from __future__ import annotations

from pathlib import Path
from typing import Any

from oceans_glwl.conditions import FinalSearchCondition
from oceans_glwl.data.samples import build_samples
from oceans_glwl.data.splits import build_chronological_splits
from oceans_glwl.data.validate import load_frame
from oceans_glwl.models.neuralforecast import NeuralForecastAdapter, NeuralForecastAdapterConfig
from oceans_glwl.training.devices import resolve_device
from oceans_glwl.training.search_space import suggest_training_config


def run_local_hpo(
    data_path: str | Path,
    *,
    condition: FinalSearchCondition,
    n_trials: int,
    device: str = "auto",
    max_steps: int = 750,
    storage: str | None = None,
    max_samples_per_split: int | None = None,
) -> Any:
    """Run an ordinary single-process Optuna study; no PostgreSQL is required."""
    try:
        import optuna
    except ImportError:
        raise RuntimeError("Install the 'hpo' extra to run hyperparameter optimization") from None
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    resolved_device = resolve_device(device)
    frame = load_frame(data_path)
    splits = build_chronological_splits(frame)
    samples = build_samples(
        frame,
        window_config=condition.window_config,
        splits=splits,
        max_samples_per_split=max_samples_per_split,
    )
    adapter = NeuralForecastAdapter(condition.model_name)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=condition.seed, n_startup_trials=10, constant_liar=True),
        storage=storage,
        study_name=condition.study_name if storage else None,
        load_if_exists=bool(storage),
    )

    def objective(trial: Any) -> float:
        sampled = suggest_training_config(
            trial,
            condition,
            max_steps=max_steps,
            accelerator=resolved_device.lightning_accelerator,
            devices=resolved_device.devices,
        )
        result = adapter.fit_predict_splits(
            train_samples=samples.for_split("train"),
            evaluation_samples=samples.for_split("validation"),
            condition=condition,
            config=NeuralForecastAdapterConfig(**sampled),
        )
        return float(result["metrics"]["full_horizon_rmse"])

    complete = len([trial for trial in study.trials if trial.state.name == "COMPLETE"])
    remaining = max(0, int(n_trials) - complete) if storage else int(n_trials)
    if remaining:
        study.optimize(objective, n_trials=remaining)
    return study
