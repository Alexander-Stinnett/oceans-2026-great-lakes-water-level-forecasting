from pathlib import Path

import pytest

from oceans_glwl.conditions import FinalSearchCondition
from oceans_glwl.training.fixed import run_fixed_experiment


@pytest.mark.training
def test_tiny_cpu_training_smoke(repo_root: Path) -> None:
    pytest.importorskip("neuralforecast")
    condition = FinalSearchCondition(
        "AutoLSTM",
        input_variant="sparse_30d_smoothed_all",
        context_days=180,
        input_gap_days=30,
        output_mode="seq2one_30d",
        horizon_days=30,
    )
    result = run_fixed_experiment(
        repo_root / "data/canonical/combined_full.csv",
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
    assert result["validation"]["metrics"]["full_horizon_n"] == 2
    assert result["test"]["metrics"]["full_horizon_n"] == 2

