from __future__ import annotations

from pathlib import Path
from typing import Any

from oceans_glwl.conditions import FinalSearchCondition
from oceans_glwl.data.samples import build_samples
from oceans_glwl.data.splits import build_chronological_splits
from oceans_glwl.data.validate import load_frame
from oceans_glwl.models.neuralforecast import NeuralForecastAdapter, NeuralForecastAdapterConfig


def run_fixed_experiment(
    data_path: str | Path,
    *,
    condition: FinalSearchCondition,
    training_config: dict[str, Any],
    device: str = "auto",
    max_samples_per_split: int | None = None,
) -> dict[str, Any]:
    """Train a fixed configuration locally without HPO or distributed services."""
    frame = load_frame(data_path)
    splits = build_chronological_splits(frame)
    samples = build_samples(
        frame,
        window_config=condition.window_config,
        splits=splits,
        max_samples_per_split=max_samples_per_split,
    )
    config = NeuralForecastAdapterConfig(
        max_steps=int(training_config.get("max_steps", 750)),
        learning_rate=float(training_config["learning_rate"]),
        batch_size=int(training_config["batch_size"]),
        windows_batch_size=int(training_config["windows_batch_size"]),
        scaler_type=str(training_config.get("scaler_type", "standard")),
        accelerator=device,
        devices=int(training_config.get("devices", 1)),
        enable_progress_bar=bool(training_config.get("enable_progress_bar", False)),
        logger=bool(training_config.get("logger", False)),
        model_kwargs=dict(training_config.get("model_kwargs", {})),
    )
    adapter = NeuralForecastAdapter(condition.model_name)
    validation = adapter.fit_predict_splits(
        train_samples=samples.for_split("train"),
        evaluation_samples=samples.for_split("validation"),
        condition=condition,
        config=config,
    )
    test = adapter.final_refit_and_evaluate(
        train_samples=samples.for_split("train"),
        validation_samples=samples.for_split("validation"),
        test_samples=samples.for_split("test"),
        condition=condition,
        config=config,
    )
    return {"condition": condition.to_dict(), "validation": validation, "test": test}

