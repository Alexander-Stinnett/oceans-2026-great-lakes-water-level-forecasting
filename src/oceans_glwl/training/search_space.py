from __future__ import annotations

from typing import Any

from oceans_glwl.conditions import FinalSearchCondition

SEARCH_SPACE_VERSION = "oceans_final_search_space_v1"
SHARED_SEARCH_SPACE: dict[str, Any] = {
    "learning_rate": {"kind": "float", "low": 1e-4, "high": 3e-3, "log": True},
    "batch_size": {"kind": "categorical", "choices": [16, 32]},
    "windows_batch_size": {"kind": "categorical", "choices": [64, 128]},
    "scaler_type": {"kind": "fixed", "value": "standard"},
}
MODEL_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "AutoLSTM": {
        "encoder_hidden_size": {"kind": "categorical", "choices": [32, 64, 128]},
        "decoder_hidden_size": {"kind": "categorical", "choices": [32, 64, 128]},
        "encoder_n_layers": {"kind": "int", "low": 1, "high": 2},
        "encoder_dropout": {"kind": "categorical", "choices": [0.0, 0.1, 0.2]},
    },
    "AutoNBEATSx": {
        "blocks_per_stack": {"kind": "int", "low": 1, "high": 2},
        "mlp_width": {"kind": "categorical", "choices": [64, 128]},
        "dropout_prob_theta": {"kind": "categorical", "choices": [0.0, 0.1, 0.2]},
    },
    "AutoNHITS": {
        "blocks_per_stack": {"kind": "int", "low": 1, "high": 2},
        "mlp_width": {"kind": "categorical", "choices": [64, 128]},
        "dropout_prob_theta": {"kind": "categorical", "choices": [0.0, 0.1, 0.2]},
    },
    "AutoTFT": {
        "hidden_size": {"kind": "categorical", "choices": [32, 64, 128]},
        "n_head": {"kind": "categorical", "choices": [1, 2, 4]},
        "n_rnn_layers": {"kind": "int", "low": 1, "high": 2},
        "dropout": {"kind": "categorical", "choices": [0.0, 0.1, 0.2]},
    },
}


def suggest_training_config(
    trial: Any,
    condition: FinalSearchCondition,
    *,
    max_steps: int,
    accelerator: str,
    devices: int,
) -> dict[str, Any]:
    shared = {
        "max_steps": int(max_steps),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "windows_batch_size": trial.suggest_categorical("windows_batch_size", [64, 128]),
        "scaler_type": "standard",
        "accelerator": accelerator,
        "devices": int(devices),
        "enable_progress_bar": False,
        "logger": False,
    }
    sampled: dict[str, Any] = {}
    for name, spec in MODEL_SEARCH_SPACES[condition.model_name].items():
        if spec["kind"] == "categorical":
            sampled[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif spec["kind"] == "int":
            sampled[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        else:
            raise ValueError(f"Unsupported search-space specification for {name}: {spec}.")
    shared["model_kwargs"] = resolve_model_kwargs(condition, sampled)
    return shared


def resolve_model_kwargs(condition: FinalSearchCondition, sampled: dict[str, Any]) -> dict[str, Any]:
    if condition.model_name == "AutoLSTM":
        return {
            "encoder_hidden_size": int(sampled["encoder_hidden_size"]),
            "decoder_hidden_size": int(sampled["decoder_hidden_size"]),
            "encoder_n_layers": int(sampled["encoder_n_layers"]),
            "encoder_dropout": float(sampled["encoder_dropout"]),
        }
    if condition.model_name == "AutoNBEATSx":
        blocks = int(sampled["blocks_per_stack"])
        width = int(sampled["mlp_width"])
        return {
            "stack_types": ["identity", "identity", "identity"],
            "n_blocks": [blocks, blocks, blocks],
            "mlp_units": [[width, width], [width, width], [width, width]],
            "dropout_prob_theta": float(sampled["dropout_prob_theta"]),
        }
    if condition.model_name == "AutoNHITS":
        blocks = int(sampled["blocks_per_stack"])
        width = int(sampled["mlp_width"])
        return {
            "n_blocks": [blocks, blocks, blocks],
            "mlp_units": [[width, width], [width, width], [width, width]],
            "dropout_prob_theta": float(sampled["dropout_prob_theta"]),
        }
    if condition.model_name == "AutoTFT":
        hidden = int(sampled["hidden_size"])
        heads = int(sampled["n_head"])
        if hidden % heads:
            raise ValueError(f"AutoTFT hidden_size={hidden} is not divisible by n_head={heads}.")
        return {
            "hidden_size": hidden,
            "n_head": heads,
            "n_rnn_layers": int(sampled["n_rnn_layers"]),
            "dropout": float(sampled["dropout"]),
        }
    raise ValueError(f"Unsupported final-search model: {condition.model_name!r}.")
