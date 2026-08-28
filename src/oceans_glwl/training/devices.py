from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DevicePolicy:
    backend: str
    lightning_accelerator: str
    devices: int = 1


def resolve_device(
    requested: str = "auto", *, devices: int = 1, torch_module: Any | None = None
) -> DevicePolicy:
    """Resolve a portable CPU, CUDA, or MPS training backend."""
    if int(devices) <= 0:
        raise ValueError("devices must be positive")
    normalized = str(requested).strip().lower()
    aliases = {"gpu": "cuda", "cuda": "cuda", "mps": "mps", "cpu": "cpu", "auto": "auto"}
    if normalized not in aliases:
        raise ValueError("device must be one of auto, cpu, cuda, or mps")
    normalized = aliases[normalized]
    resolved_torch: Any = torch_module
    if resolved_torch is None:
        try:
            import torch

            resolved_torch = torch
        except ImportError:
            if normalized in {"auto", "cpu"}:
                return DevicePolicy("cpu", "cpu", int(devices))
            raise RuntimeError(f"{normalized} was requested but torch is not installed") from None
    if normalized == "auto":
        if bool(resolved_torch.cuda.is_available()):
            normalized = "cuda"
        elif bool(getattr(getattr(resolved_torch.backends, "mps", None), "is_available", lambda: False)()):
            normalized = "mps"
        else:
            normalized = "cpu"
    if normalized == "cuda" and not bool(resolved_torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if normalized == "mps":
        available = getattr(getattr(resolved_torch.backends, "mps", None), "is_available", lambda: False)
        if not bool(available()):
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    accelerator = "gpu" if normalized == "cuda" else normalized
    return DevicePolicy(normalized, accelerator, int(devices))
