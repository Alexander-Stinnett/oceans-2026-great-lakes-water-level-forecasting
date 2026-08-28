from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from oceans_glwl.config import PROTOCOL_VERSION, WindowConfig

FINAL_MODEL_NAMES = ("AutoLSTM", "AutoNBEATSx", "AutoNHITS", "AutoTFT")


@dataclass(frozen=True)
class FinalSearchCondition:
    model_name: str
    input_variant: str = "sparse_30d_smoothed_all"
    context_days: int = 180
    input_gap_days: int = 30
    output_mode: str = "seq2seq_6x30d"
    horizon_days: int = 180
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.model_name not in FINAL_MODEL_NAMES:
            raise ValueError(f"Unsupported final-search model: {self.model_name!r}.")
        _ = self.window_config  # validate the complete I/O contract

    @property
    def window_config(self) -> WindowConfig:
        return WindowConfig(
            input_variant=self.input_variant,
            context_days=int(self.context_days),
            input_gap_days=int(self.input_gap_days),
            output_mode=self.output_mode,
            horizon_days=int(self.horizon_days),
        )

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "model_name": self.model_name,
            "input_variant": self.input_variant,
            "context_days": int(self.context_days),
            "context_rows": int(self.window_config.context_rows),
            "input_gap_days": int(self.input_gap_days),
            "input_grid": self.window_config.input_grid,
            "output_mode": self.output_mode,
            "horizon_days": int(self.horizon_days),
            "lead_days": list(self.window_config.lead_days),
            "seed": int(self.seed),
            "objective": "validation_rmse",
        }

    @property
    def condition_id(self) -> str:
        digest = hashlib.sha256(_canonical_json(self.identity_payload).encode("utf-8")).hexdigest()[:12]
        return (
            f"ofs1__{_slug(self.model_name)}__{_slug(self.input_variant)}__"
            f"ctx{self.context_days}__{_slug(self.output_mode)}__{digest}"
        )

    @property
    def study_name(self) -> str:
        return f"glm::oceans_final_search_v1::{self.condition_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"condition_id": self.condition_id, "study_name": self.study_name, **self.identity_payload}


def condition_from_dict(payload: dict[str, Any]) -> FinalSearchCondition:
    condition = FinalSearchCondition(
        model_name=str(payload["model_name"]),
        input_variant=str(payload["input_variant"]),
        context_days=int(payload["context_days"]),
        input_gap_days=int(payload["input_gap_days"]),
        output_mode=str(payload["output_mode"]),
        horizon_days=int(payload["horizon_days"]),
        seed=int(payload.get("seed", 1337)),
    )
    expected_id = payload.get("condition_id")
    if expected_id is not None and str(expected_id) != condition.condition_id:
        raise ValueError(f"Condition identity mismatch: {expected_id!r} != {condition.condition_id!r}.")
    expected_study = payload.get("study_name")
    if expected_study is not None and str(expected_study) != condition.study_name:
        raise ValueError(f"Study identity mismatch: {expected_study!r} != {condition.study_name!r}.")
    return condition


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _slug(value: object) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))
