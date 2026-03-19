"""
tests/conftest.py
~~~~~~~~~~~~~~~~~
Shared pytest fixtures for the R3-FL test suite.

Provides:
  - torch_seed: Sets a deterministic random seed for reproducible tensor ops.
  - simple_ndarrays: A list of simple numpy arrays for use as model parameters.
  - mock_flower_client_proxy: A lightweight mock for fl.server.client_proxy.ClientProxy.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import flwr as fl


@pytest.fixture(autouse=False)
def torch_seed():
    """Set a deterministic PyTorch seed (42) for the duration of the test.

    This fixture is not ``autouse`` — opt in by requesting it explicitly.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    yield


@pytest.fixture()
def simple_ndarrays() -> List[np.ndarray]:
    """Return a list of simple float32 numpy arrays mimicking a two-layer model.

    Layer 0: (4, 4) weight matrix.
    Layer 1: (4,)  bias vector.
    """
    return [
        np.full((4, 4), 0.5, dtype=np.float32),
        np.full((4,), 0.1, dtype=np.float32),
    ]


@pytest.fixture()
def mock_flower_client_proxy() -> MagicMock:
    """Return a mock ClientProxy with a deterministic cid of 'test_client_0'."""
    proxy = MagicMock(spec=fl.server.client_proxy.ClientProxy)
    proxy.cid = "test_client_0"
    return proxy
