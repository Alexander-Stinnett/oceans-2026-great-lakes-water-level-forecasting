from types import SimpleNamespace

import pytest

from oceans_glwl.training.devices import resolve_device


def fake_torch(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_cpu_is_always_accepted() -> None:
    assert resolve_device("cpu", torch_module=fake_torch(cuda=False, mps=False)).backend == "cpu"


def test_auto_prefers_cuda_then_mps_then_cpu() -> None:
    assert resolve_device("auto", torch_module=fake_torch(cuda=True, mps=True)).backend == "cuda"
    assert resolve_device("auto", torch_module=fake_torch(cuda=False, mps=True)).backend == "mps"
    assert resolve_device("auto", torch_module=fake_torch(cuda=False, mps=False)).backend == "cpu"


def test_unavailable_explicit_accelerators_fail() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda", torch_module=fake_torch(cuda=False, mps=False))
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps", torch_module=fake_torch(cuda=False, mps=False))

