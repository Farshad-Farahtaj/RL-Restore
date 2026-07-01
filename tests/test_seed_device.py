import numpy as np
import torch

from rlrestore.common.device import get_device
from rlrestore.common.seed import set_seed


def test_set_seed_makes_torch_and_numpy_deterministic():
    set_seed(123)
    a1 = torch.randn(4)
    n1 = np.random.rand(4)
    set_seed(123)
    a2 = torch.randn(4)
    n2 = np.random.rand(4)
    assert torch.equal(a1, a2)
    assert np.allclose(n1, n2)


def test_get_device_env_override(monkeypatch):
    monkeypatch.setenv("RLR_DEVICE", "cpu")
    assert get_device().type == "cpu"


def test_get_device_arg_beats_default(monkeypatch):
    monkeypatch.delenv("RLR_DEVICE", raising=False)
    assert get_device("cpu").type == "cpu"


def test_get_device_default_is_mps_or_cpu(monkeypatch):
    monkeypatch.delenv("RLR_DEVICE", raising=False)
    dev = get_device()
    assert dev.type == ("mps" if torch.backends.mps.is_available() else "cpu")
