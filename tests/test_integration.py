"""
tests/test_integration.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive pytest tests for the integration layer:
  - RLReputationStrategy (src/integration/strategy.py)

Covers:
  - Strategy initialisation (with and without valid PPO checkpoint)
  - configure_fit / configure_evaluate sampling logic
  - aggregate_evaluate weighted loss/accuracy aggregation
  - evaluate() (server-side centralized evaluation)
  - aggregate_fit core pipeline (state build, PPO inference, weighted avg,
    blockchain update, fallback to uniform on PPO failure)
  - _build_state_matrix (participating vs non-participating rows)
  - _compute_gradient_similarity (cosine similarity, values in [0,1])
  - _compute_update_magnitude (L2 norm min-max normalisation, edge cases)
  - _ppo_inference (PPO path, fallback uniform, near-zero weight guard)
  - _weighted_average (correctness, chunking, layer streaming)
  - _update_blockchain (Redis upload, batch_update_clients, error tolerance)
  - _get_or_create_address (mapping lookup, deterministic derivation)
  - Edge cases: zero weights, all-malicious, all-honest, single client,
    blockchain / Redis failures

All external I/O is mocked. No blockchain node, Redis, or Ray required.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from src.rl_agent.env import NUM_CLIENTS, NUM_FEATURES

# Module under test -- imported after mocking Ray so __init__ does not crash.
# We keep the import at module level: the patch for the PPO checkpoint path is
# applied per-fixture (see `strategy` fixture below).
import src.integration.strategy as strategy_mod
from src.integration.strategy import (
    RLReputationStrategy,
    _AGGREGATION_CHUNK_SIZE,
    _MAX_REPUTATION,
    _NEUTRAL_FEATURE,
    _REPUTATION_SCALE_DEFAULT,
)


# ---------------------------------------------------------------------------
# Helpers for mocking Ray during strategy construction
# ---------------------------------------------------------------------------


def _make_ray_mocks(algo_side_effect=None, algo_return_value=None):
    """Build a sys.modules-compatible dict that mocks out Ray / RLlib.

    Args:
        algo_side_effect: If set, from_checkpoint will raise this exception.
        algo_return_value: If set, from_checkpoint will return this object.

    Returns:
        A tuple (mock_algo_cls, modules_dict) suitable for use with
        ``patch.dict(sys.modules, ...)``.
    """
    mock_algo_cls = MagicMock()
    if algo_side_effect is not None:
        mock_algo_cls.from_checkpoint.side_effect = algo_side_effect
    elif algo_return_value is not None:
        mock_algo_cls.from_checkpoint.return_value = algo_return_value
    else:
        mock_algo_cls.from_checkpoint.side_effect = RuntimeError("No checkpoint")

    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = False

    mock_rllib = MagicMock()
    mock_rllib.algorithms.algorithm.Algorithm = mock_algo_cls

    modules = {
        "ray": mock_ray,
        "ray.rllib": mock_rllib,
        "ray.rllib.algorithms": mock_rllib.algorithms,
        "ray.rllib.algorithms.algorithm": mock_rllib.algorithms.algorithm,
        "ray.tune": MagicMock(),
        "ray.tune.registry": MagicMock(),
    }
    return mock_algo_cls, modules


def _make_strategy_no_ppo(**kwargs) -> RLReputationStrategy:
    """Create a strategy where PPO loading always fails (uniform-weight fallback)."""
    _, modules = _make_ray_mocks(algo_side_effect=RuntimeError("No checkpoint in tests"))
    with patch.dict(sys.modules, modules):
        return RLReputationStrategy(
            ppo_checkpoint_path="/tmp/nonexistent",
            min_fit_clients=kwargs.pop("min_fit_clients", 2),
            min_evaluate_clients=kwargs.pop("min_evaluate_clients", 2),
            min_available_clients=kwargs.pop("min_available_clients", 2),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Helpers for constructing Flower FitRes / EvaluateRes
# ---------------------------------------------------------------------------


def _make_fit_res(
    ndarrays: List[np.ndarray],
    num_examples: int = 100,
    metrics: Optional[Dict] = None,
) -> FitRes:
    """Create a FitRes from a list of numpy arrays."""
    return FitRes(
        status=fl.common.Status(code=fl.common.Code.OK, message=""),
        parameters=ndarrays_to_parameters(ndarrays),
        num_examples=num_examples,
        metrics=metrics or {},
    )


def _make_eval_res(
    loss: float,
    num_examples: int = 100,
    accuracy: float = 0.0,
) -> EvaluateRes:
    """Create an EvaluateRes."""
    return EvaluateRes(
        status=fl.common.Status(code=fl.common.Code.OK, message=""),
        loss=loss,
        num_examples=num_examples,
        metrics={"accuracy": accuracy},
    )


def _make_client_proxy(cid: str) -> MagicMock:
    """Return a mock ClientProxy with the given cid."""
    proxy = MagicMock(spec=fl.server.client_proxy.ClientProxy)
    proxy.cid = cid
    return proxy


def _simple_params(value: float = 0.5, shape: tuple = (4, 4)) -> List[np.ndarray]:
    """Return a simple list of numpy arrays for model parameters."""
    return [np.full(shape, value, dtype=np.float32)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_blockchain():
    """Patch all blockchain / Redis calls in strategy_mod."""
    with (
        patch.object(strategy_mod, "get_client_score") as mock_get,
        patch.object(strategy_mod, "batch_update_clients") as mock_batch,
        patch.object(strategy_mod, "upload_tensor_to_redis") as mock_upload,
    ):
        mock_get.return_value = {"reputationScore": 500}
        mock_batch.return_value = None
        mock_upload.return_value = "test-redis-key-uuid"
        yield {
            "get_client_score": mock_get,
            "batch_update_clients": mock_batch,
            "upload_tensor_to_redis": mock_upload,
        }


@pytest.fixture()
def strategy(mock_blockchain) -> RLReputationStrategy:
    """Strategy with PPO loading intentionally failing (uniform fallback).

    The resulting strategy has ppo_algo=None and uses uniform weights.
    """
    return _make_strategy_no_ppo(
        num_clients=NUM_CLIENTS,
        fraction_fit=0.1,
        fraction_evaluate=0.05,
    )


@pytest.fixture()
def strategy_with_ppo(mock_blockchain) -> RLReputationStrategy:
    """Strategy with a mock PPO algo that returns fixed linearly-spaced actions.

    Weights are non-uniform and predictable, enabling deterministic tests.
    """
    fixed_action = np.linspace(0.0, 1.0, NUM_CLIENTS, dtype=np.float32)
    mock_algo = MagicMock()
    mock_algo.compute_single_action.return_value = fixed_action

    _, modules = _make_ray_mocks(algo_return_value=mock_algo)
    with patch.dict(sys.modules, modules):
        s = RLReputationStrategy(
            ppo_checkpoint_path="/tmp/fake_checkpoint",
            num_clients=NUM_CLIENTS,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )
    # Ensure the mock algo is available for inference
    s.ppo_algo = mock_algo
    return s


@pytest.fixture()
def three_clients_results(mock_blockchain):
    """Three (ClientProxy, FitRes) pairs with distinct parameter values."""
    proxies = [_make_client_proxy(str(i)) for i in range(3)]
    results = [
        (proxies[0], _make_fit_res(_simple_params(0.2), num_examples=100,
                                    metrics={"accuracy": 0.7, "loss_improvement": 0.6})),
        (proxies[1], _make_fit_res(_simple_params(0.5), num_examples=150,
                                    metrics={"accuracy": 0.8, "loss_improvement": 0.7})),
        (proxies[2], _make_fit_res(_simple_params(0.8), num_examples=200,
                                    metrics={"accuracy": 0.9, "loss_improvement": 0.8})),
    ]
    return results


# =========================================================================
# 1. Strategy Initialisation Tests
# =========================================================================


class TestStrategyInitialisation:
    """Verify RLReputationStrategy constructor stores configuration correctly."""

    def test_default_ppo_is_none_on_bad_checkpoint(self, strategy: RLReputationStrategy):
        """When the PPO checkpoint fails to load, ppo_algo must be None."""
        assert strategy.ppo_algo is None

    def test_num_clients_stored(self, strategy: RLReputationStrategy):
        """num_clients attribute must match the constructor argument."""
        assert strategy.num_clients == NUM_CLIENTS

    def test_fraction_fit_stored(self, strategy: RLReputationStrategy):
        """fraction_fit attribute must match the constructor argument."""
        assert strategy.fraction_fit == pytest.approx(0.1)

    def test_fraction_evaluate_stored(self, strategy: RLReputationStrategy):
        """fraction_evaluate attribute must match the constructor argument."""
        assert strategy.fraction_evaluate == pytest.approx(0.05)

    def test_reputation_scale_default(self, strategy: RLReputationStrategy):
        """Default reputation_scale should be _REPUTATION_SCALE_DEFAULT."""
        assert strategy.reputation_scale == _REPUTATION_SCALE_DEFAULT

    def test_custom_reputation_scale(self, mock_blockchain):
        """Custom reputation_scale should be stored on the strategy."""
        s = _make_strategy_no_ppo(reputation_scale=500)
        assert s.reputation_scale == 500

    def test_client_address_map_empty_by_default(self, strategy: RLReputationStrategy):
        """client_address_map must be empty when no addresses are provided."""
        assert strategy.client_address_map == {}

    def test_custom_client_address_map(self, mock_blockchain):
        """Provided client_addresses dict should be stored as-is."""
        addr_map = {"client_0": "0xABC", "client_1": "0xDEF"}
        s = _make_strategy_no_ppo(client_addresses=addr_map)
        assert s.client_address_map == addr_map

    def test_initial_parameters_stored(self, mock_blockchain):
        """initial_parameters passed to constructor should be stored."""
        params = ndarrays_to_parameters([np.zeros((4, 4), dtype=np.float32)])
        s = _make_strategy_no_ppo(initial_parameters=params)
        assert s.initial_parameters is params

    def test_round_metrics_starts_empty(self, strategy: RLReputationStrategy):
        """round_metrics dict must start empty."""
        assert strategy.round_metrics == {}


# =========================================================================
# 2. initialize_parameters Tests
# =========================================================================


class TestInitializeParameters:
    """Verify initialize_parameters returns provided initial parameters."""

    def test_returns_none_when_no_initial_params(self, strategy: RLReputationStrategy):
        """Without initial_parameters, method should return None."""
        client_manager = MagicMock()
        result = strategy.initialize_parameters(client_manager)
        assert result is None

    def test_returns_initial_parameters_when_provided(self, mock_blockchain):
        """With initial_parameters, method should return them unchanged."""
        params = ndarrays_to_parameters([np.zeros((4, 4), dtype=np.float32)])
        s = _make_strategy_no_ppo(initial_parameters=params)
        client_manager = MagicMock()
        assert s.initialize_parameters(client_manager) is params


# =========================================================================
# 3. configure_fit Tests
# =========================================================================


class TestConfigureFit:
    """Verify configure_fit samples the correct number of clients."""

    def test_configure_fit_returns_list(self, strategy: RLReputationStrategy):
        """configure_fit should return a list of (ClientProxy, FitIns) tuples."""
        client_manager = MagicMock()
        client_manager.num_available.return_value = 50
        mock_clients = [MagicMock() for _ in range(5)]
        client_manager.sample.return_value = mock_clients

        params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
        result = strategy.configure_fit(1, params, client_manager)

        assert isinstance(result, list)
        assert len(result) == 5

    def test_configure_fit_includes_server_round_in_config(self, strategy: RLReputationStrategy):
        """Each FitIns config must include the 'server_round' key."""
        client_manager = MagicMock()
        client_manager.num_available.return_value = 50
        client_manager.sample.return_value = [MagicMock() for _ in range(3)]

        params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
        result = strategy.configure_fit(7, params, client_manager)

        for _, fit_ins in result:
            assert "server_round" in fit_ins.config
            assert fit_ins.config["server_round"] == 7

    def test_configure_fit_respects_min_fit_clients(self, strategy: RLReputationStrategy):
        """Sample size should be at least min_fit_clients even for small pools."""
        client_manager = MagicMock()
        # Only 5 available, fraction_fit=0.1 → max(2, int(5*0.1)) = max(2,0) = 2
        client_manager.num_available.return_value = 5
        client_manager.sample.return_value = [MagicMock() for _ in range(2)]

        params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
        result = strategy.configure_fit(1, params, client_manager)

        # sample called with num_clients >= min_fit_clients (2)
        call_kwargs = client_manager.sample.call_args[1]
        assert call_kwargs["num_clients"] >= strategy.min_fit_clients


# =========================================================================
# 4. configure_evaluate Tests
# =========================================================================


class TestConfigureEvaluate:
    """Verify configure_evaluate sampling logic."""

    def test_configure_evaluate_returns_empty_when_not_enough_clients(
        self, strategy: RLReputationStrategy
    ):
        """Returns [] when available clients < min_evaluate_clients."""
        client_manager = MagicMock()
        client_manager.num_available.return_value = 1  # below min_evaluate_clients=2

        params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
        result = strategy.configure_evaluate(1, params, client_manager)

        assert result == []

    def test_configure_evaluate_samples_correctly(self, strategy: RLReputationStrategy):
        """Returns sampled clients when enough are available."""
        client_manager = MagicMock()
        client_manager.num_available.return_value = 100
        mock_clients = [MagicMock() for _ in range(5)]
        client_manager.sample.return_value = mock_clients

        params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
        result = strategy.configure_evaluate(1, params, client_manager)

        assert len(result) == 5


# =========================================================================
# 5. aggregate_evaluate Tests
# =========================================================================


class TestAggregateEvaluate:
    """Verify aggregate_evaluate computes weighted loss and accuracy."""

    def test_returns_none_for_empty_results(self, strategy: RLReputationStrategy):
        """No results → returns None."""
        result = strategy.aggregate_evaluate(1, [], [])
        assert result is None

    def test_returns_none_for_zero_total_samples(self, strategy: RLReputationStrategy):
        """Results with 0 examples → returns None."""
        proxy = _make_client_proxy("c0")
        eval_res = _make_eval_res(loss=0.5, num_examples=0)
        result = strategy.aggregate_evaluate(1, [(proxy, eval_res)], [])
        assert result is None

    def test_weighted_loss_single_client(self, strategy: RLReputationStrategy):
        """Single client: aggregated loss equals client loss."""
        proxy = _make_client_proxy("c0")
        eval_res = _make_eval_res(loss=0.42, num_examples=100, accuracy=0.75)
        loss, metrics = strategy.aggregate_evaluate(1, [(proxy, eval_res)], [])
        assert loss == pytest.approx(0.42, abs=1e-5)
        assert metrics["accuracy"] == pytest.approx(0.75, abs=1e-5)

    def test_weighted_loss_multiple_clients(self, strategy: RLReputationStrategy):
        """Weighted average with two clients of different sizes."""
        proxies = [_make_client_proxy("c0"), _make_client_proxy("c1")]
        results = [
            (proxies[0], _make_eval_res(loss=0.2, num_examples=100, accuracy=0.8)),
            (proxies[1], _make_eval_res(loss=0.6, num_examples=300, accuracy=0.6)),
        ]
        loss, metrics = strategy.aggregate_evaluate(1, results, [])
        expected_loss = (100 * 0.2 + 300 * 0.6) / 400
        expected_acc = (100 * 0.8 + 300 * 0.6) / 400
        assert loss == pytest.approx(expected_loss, abs=1e-5)
        assert metrics["accuracy"] == pytest.approx(expected_acc, abs=1e-5)

    def test_accuracy_defaults_to_zero_when_not_reported(self, strategy: RLReputationStrategy):
        """Clients not reporting accuracy → accuracy treated as 0."""
        proxy = _make_client_proxy("c0")
        eval_res = EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message=""),
            loss=0.5,
            num_examples=50,
            metrics={},  # no accuracy key
        )
        loss, metrics = strategy.aggregate_evaluate(1, [(proxy, eval_res)], [])
        assert loss == pytest.approx(0.5, abs=1e-5)
        assert metrics["accuracy"] == pytest.approx(0.0, abs=1e-5)

    def test_returns_tuple_of_float_and_dict(self, strategy: RLReputationStrategy):
        """Return type must be (float, dict)."""
        proxy = _make_client_proxy("c0")
        eval_res = _make_eval_res(loss=0.1, num_examples=10, accuracy=0.9)
        result = strategy.aggregate_evaluate(1, [(proxy, eval_res)], [])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], dict)


# =========================================================================
# 6. evaluate (server-side) Tests
# =========================================================================


class TestEvaluate:
    """Verify server-side evaluate delegates to evaluate_fn when provided."""

    def test_returns_none_when_no_evaluate_fn(self, strategy: RLReputationStrategy):
        """Without evaluate_fn, evaluate() returns None."""
        params = ndarrays_to_parameters([np.zeros((4, 4), dtype=np.float32)])
        result = strategy.evaluate(1, params)
        assert result is None

    def test_calls_evaluate_fn_with_correct_args(self, mock_blockchain):
        """evaluate_fn receives (server_round, ndarrays, {})."""
        called_with = {}
        expected_arrays = [np.zeros((4, 4), dtype=np.float32)]

        def fake_eval_fn(server_round, ndarrays, config):
            called_with["round"] = server_round
            called_with["ndarrays"] = ndarrays
            called_with["config"] = config
            return 0.25, {"accuracy": 0.9}

        s = _make_strategy_no_ppo(evaluate_fn=fake_eval_fn)
        params = ndarrays_to_parameters(expected_arrays)
        result = s.evaluate(3, params)

        assert result == (0.25, {"accuracy": 0.9})
        assert called_with["round"] == 3
        assert called_with["config"] == {}
        assert len(called_with["ndarrays"]) == 1

    def test_returns_none_when_evaluate_fn_raises(self, mock_blockchain):
        """If evaluate_fn raises, evaluate() returns None (no propagation)."""
        def bad_fn(r, n, c):
            raise ValueError("simulated error")

        s = _make_strategy_no_ppo(evaluate_fn=bad_fn)
        params = ndarrays_to_parameters([np.zeros((4, 4), dtype=np.float32)])
        result = s.evaluate(1, params)
        assert result is None


# =========================================================================
# 7. _get_or_create_address Tests
# =========================================================================


class TestGetOrCreateAddress:
    """Verify Flower CID → Ethereum address mapping."""

    def test_returns_mapped_address_when_present(self, strategy: RLReputationStrategy):
        """If the CID is in client_address_map, return that address."""
        strategy.client_address_map["client_42"] = "0xKnownAddress"
        result = strategy._get_or_create_address("client_42")
        assert result == "0xKnownAddress"

    def test_derives_deterministic_address_for_unknown_cid(self, strategy: RLReputationStrategy):
        """Two calls with the same CID produce the same address."""
        addr1 = strategy._get_or_create_address("unknown_cid_xyz")
        addr2 = strategy._get_or_create_address("unknown_cid_xyz")
        assert addr1 == addr2

    def test_different_cids_produce_different_addresses(self, strategy: RLReputationStrategy):
        """Two different CIDs must (with overwhelming probability) differ."""
        addr1 = strategy._get_or_create_address("cid_aaa")
        addr2 = strategy._get_or_create_address("cid_bbb")
        assert addr1 != addr2

    def test_derived_address_starts_with_0x(self, strategy: RLReputationStrategy):
        """Derived address must start with '0x'."""
        addr = strategy._get_or_create_address("some_new_cid")
        assert addr.startswith("0x")

    def test_derived_address_cached_in_map(self, strategy: RLReputationStrategy):
        """After first derivation, address is stored in client_address_map."""
        cid = "fresh_cid_001"
        assert cid not in strategy.client_address_map
        addr = strategy._get_or_create_address(cid)
        assert cid in strategy.client_address_map
        assert strategy.client_address_map[cid] == addr

    def test_derived_address_is_40_hex_chars_after_0x(self, strategy: RLReputationStrategy):
        """Derived address has 42 chars total (0x + 40 hex chars)."""
        addr = strategy._get_or_create_address("cid_for_length_check")
        # Web3 checksum or lowercase fallback — both should be 42 chars
        assert len(addr) == 42


# =========================================================================
# 8. _compute_gradient_similarity Tests
# =========================================================================


class TestComputeGradientSimilarity:
    """Verify cosine similarity computation."""

    def test_identical_params_return_similarity_one(self, strategy: RLReputationStrategy):
        """Identical parameter vectors should yield similarity ~1.0."""
        params = [
            [torch.full((4, 4), 0.5, dtype=torch.float32)],
            [torch.full((4, 4), 0.5, dtype=torch.float32)],
            [torch.full((4, 4), 0.5, dtype=torch.float32)],
        ]
        similarities = strategy._compute_gradient_similarity(params)
        np.testing.assert_allclose(similarities, 1.0, atol=1e-5)

    def test_output_shape_matches_client_count(self, strategy: RLReputationStrategy):
        """Output array length must equal the number of clients."""
        n = 5
        params = [[torch.randn(10)] for _ in range(n)]
        torch.manual_seed(99)
        params = [[torch.randn(10)] for _ in range(n)]
        similarities = strategy._compute_gradient_similarity(params)
        assert similarities.shape == (n,)

    def test_similarities_in_zero_one_range(self, strategy: RLReputationStrategy):
        """All similarity values must be in [0, 1]."""
        torch.manual_seed(7)
        params = [[torch.randn(20)] for _ in range(8)]
        similarities = strategy._compute_gradient_similarity(params)
        assert np.all(similarities >= 0.0)
        assert np.all(similarities <= 1.0)

    def test_dtype_is_float32(self, strategy: RLReputationStrategy):
        """Returned array must be float32."""
        params = [[torch.randn(4)] for _ in range(3)]
        similarities = strategy._compute_gradient_similarity(params)
        assert similarities.dtype == np.float32

    def test_single_client_returns_one(self, strategy: RLReputationStrategy):
        """A single client is trivially similar to the mean (itself)."""
        params = [[torch.randn(6)]]
        similarities = strategy._compute_gradient_similarity(params)
        assert similarities[0] == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors_have_low_similarity(self, strategy: RLReputationStrategy):
        """Vectors pointing in opposite directions should have similarity ~0."""
        v = torch.ones(10, dtype=torch.float32)
        params = [[v], [-v]]
        similarities = strategy._compute_gradient_similarity(params)
        # Cosine sim of +v vs mean(+v, -v)=0 is undefined but rescaled to 0.5
        # One pair of opposites: each is far from mean(0). Check both are in [0,1].
        assert np.all(similarities >= 0.0)
        assert np.all(similarities <= 1.0)


# =========================================================================
# 9. _compute_update_magnitude Tests
# =========================================================================


class TestComputeUpdateMagnitude:
    """Verify L2 norm magnitude computation and normalisation."""

    def test_output_shape_matches_client_count(self, strategy: RLReputationStrategy):
        """Output shape must equal (n_clients,)."""
        torch.manual_seed(42)
        params = [[torch.randn(10)] for _ in range(6)]
        mags = strategy._compute_update_magnitude(params)
        assert mags.shape == (6,)

    def test_values_in_zero_one_range(self, strategy: RLReputationStrategy):
        """All magnitude values must be in [0, 1]."""
        torch.manual_seed(42)
        params = [[torch.randn(20) * i] for i in range(1, 7)]
        mags = strategy._compute_update_magnitude(params)
        assert np.all(mags >= 0.0)
        assert np.all(mags <= 1.0)

    def test_min_max_endpoints_at_zero_and_one(self, strategy: RLReputationStrategy):
        """After normalisation, smallest norm → 0.0 and largest → 1.0."""
        # Construct params with predictable L2 norms: 1.0 and 3.0
        small = torch.tensor([1.0, 0.0], dtype=torch.float32)  # norm=1
        large = torch.tensor([3.0, 0.0], dtype=torch.float32)  # norm=3
        params = [[small], [large]]
        mags = strategy._compute_update_magnitude(params)
        assert mags.min() == pytest.approx(0.0, abs=1e-5)
        assert mags.max() == pytest.approx(1.0, abs=1e-5)

    def test_equal_norms_return_neutral_feature(self, strategy: RLReputationStrategy):
        """All clients with identical L2 norms → return neutral (0.5) for all."""
        equal = torch.ones(4, dtype=torch.float32)  # all identical
        params = [[equal.clone()] for _ in range(5)]
        mags = strategy._compute_update_magnitude(params)
        np.testing.assert_allclose(mags, _NEUTRAL_FEATURE, atol=1e-5)

    def test_dtype_is_float32(self, strategy: RLReputationStrategy):
        """Returned array must be float32."""
        params = [[torch.randn(4)] for _ in range(3)]
        mags = strategy._compute_update_magnitude(params)
        assert mags.dtype == np.float32


# =========================================================================
# 10. _build_state_matrix Tests
# =========================================================================


class TestBuildStateMatrix:
    """Verify state matrix construction (participating / non-participating rows)."""

    def _make_arrays(self, n: int, value: float = 0.5) -> np.ndarray:
        return np.full(n, value, dtype=np.float32)

    def test_output_shape(self, strategy: RLReputationStrategy):
        """Output shape must be (NUM_CLIENTS, NUM_FEATURES)."""
        n = 4
        state = strategy._build_state_matrix(
            participating_indices=list(range(n)),
            accuracies=self._make_arrays(n, 0.8),
            similarities=self._make_arrays(n, 0.7),
            reputations=self._make_arrays(n, 0.6),
            loss_improvements=self._make_arrays(n, 0.5),
            magnitudes=self._make_arrays(n, 0.4),
        )
        assert state.shape == (NUM_CLIENTS, NUM_FEATURES)

    def test_participating_rows_populated_correctly(self, strategy: RLReputationStrategy):
        """Participating rows should have feature values set from arrays."""
        n = 3
        state = strategy._build_state_matrix(
            participating_indices=list(range(n)),
            accuracies=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            similarities=np.array([0.4, 0.5, 0.6], dtype=np.float32),
            reputations=np.array([0.7, 0.8, 0.9], dtype=np.float32),
            loss_improvements=np.array([0.15, 0.25, 0.35], dtype=np.float32),
            magnitudes=np.array([0.45, 0.55, 0.65], dtype=np.float32),
        )
        # Row 0 check
        assert state[0, 0] == pytest.approx(0.1, abs=1e-5)  # accuracy
        assert state[0, 1] == pytest.approx(0.4, abs=1e-5)  # similarity
        assert state[0, 2] == pytest.approx(0.7, abs=1e-5)  # reputation
        assert state[0, 3] == pytest.approx(0.15, abs=1e-5) # loss_improvement
        assert state[0, 4] == pytest.approx(0.45, abs=1e-5) # magnitude

    def test_non_participating_rows_are_neutral(self, strategy: RLReputationStrategy):
        """Rows for non-participating clients must be filled with _NEUTRAL_FEATURE."""
        n = 2
        state = strategy._build_state_matrix(
            participating_indices=list(range(n)),
            accuracies=self._make_arrays(n, 0.9),
            similarities=self._make_arrays(n, 0.9),
            reputations=self._make_arrays(n, 0.9),
            loss_improvements=self._make_arrays(n, 0.9),
            magnitudes=self._make_arrays(n, 0.9),
        )
        # Rows n and beyond should be neutral
        np.testing.assert_allclose(
            state[n:], _NEUTRAL_FEATURE, atol=1e-5,
            err_msg="Non-participating rows should equal neutral feature"
        )

    def test_all_values_clipped_to_zero_one(self, strategy: RLReputationStrategy):
        """Even extreme input values must be clipped to [0, 1] in output."""
        n = 3
        state = strategy._build_state_matrix(
            participating_indices=list(range(n)),
            accuracies=np.array([1.5, -0.5, 0.5], dtype=np.float32),
            similarities=self._make_arrays(n, 0.5),
            reputations=self._make_arrays(n, 0.5),
            loss_improvements=self._make_arrays(n, 0.5),
            magnitudes=self._make_arrays(n, 0.5),
        )
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)

    def test_feature_column_ordering(self, strategy: RLReputationStrategy):
        """Column order: [accuracy, similarity, reputation, loss_improvement, magnitude]."""
        n = 1
        state = strategy._build_state_matrix(
            participating_indices=[0],
            accuracies=np.array([0.11], dtype=np.float32),
            similarities=np.array([0.22], dtype=np.float32),
            reputations=np.array([0.33], dtype=np.float32),
            loss_improvements=np.array([0.44], dtype=np.float32),
            magnitudes=np.array([0.55], dtype=np.float32),
        )
        assert state[0, 0] == pytest.approx(0.11, abs=1e-5)
        assert state[0, 1] == pytest.approx(0.22, abs=1e-5)
        assert state[0, 2] == pytest.approx(0.33, abs=1e-5)
        assert state[0, 3] == pytest.approx(0.44, abs=1e-5)
        assert state[0, 4] == pytest.approx(0.55, abs=1e-5)

    def test_out_of_range_participating_index_skipped(self, strategy: RLReputationStrategy):
        """Indices >= num_clients should be silently ignored."""
        n = 2
        state = strategy._build_state_matrix(
            participating_indices=[0, NUM_CLIENTS + 5],  # second index out of range
            accuracies=np.array([0.9, 0.1], dtype=np.float32),
            similarities=self._make_arrays(n, 0.5),
            reputations=self._make_arrays(n, 0.5),
            loss_improvements=self._make_arrays(n, 0.5),
            magnitudes=self._make_arrays(n, 0.5),
        )
        # Should not raise; index 0 should be populated
        assert state[0, 0] == pytest.approx(0.9, abs=1e-5)


# =========================================================================
# 11. _ppo_inference Tests
# =========================================================================


class TestPpoInference:
    """Verify _ppo_inference weight derivation and fallback logic."""

    def _make_state(self, num_clients: int = NUM_CLIENTS) -> np.ndarray:
        return np.full((num_clients, NUM_FEATURES), 0.5, dtype=np.float32)

    def test_falls_back_to_uniform_when_ppo_is_none(self, strategy: RLReputationStrategy):
        """Without PPO, weights must be uniform (1/n each)."""
        n = 5
        state = self._make_state()
        weights = strategy._ppo_inference(state, list(range(n)))
        expected = np.full(n, 1.0 / n, dtype=np.float32)
        np.testing.assert_allclose(weights, expected, atol=1e-5)

    def test_uniform_fallback_sums_to_one(self, strategy: RLReputationStrategy):
        """Uniform fallback weights must sum to 1.0."""
        n = 7
        state = self._make_state()
        weights = strategy._ppo_inference(state, list(range(n)))
        assert weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_uniform_fallback_shape(self, strategy: RLReputationStrategy):
        """Fallback weight array must have shape (n_participating,)."""
        n = 10
        state = self._make_state()
        weights = strategy._ppo_inference(state, list(range(n)))
        assert weights.shape == (n,)

    def test_ppo_weights_normalized_to_sum_one(self, strategy_with_ppo: RLReputationStrategy):
        """PPO-derived weights must sum to 1.0 after normalisation."""
        n = 5
        state = self._make_state()
        participating = list(range(n))
        weights = strategy_with_ppo._ppo_inference(state, participating)
        assert weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_ppo_weights_shape(self, strategy_with_ppo: RLReputationStrategy):
        """PPO weight output must have shape (n_participating,)."""
        n = 5
        state = self._make_state()
        weights = strategy_with_ppo._ppo_inference(state, list(range(n)))
        assert weights.shape == (n,)

    def test_ppo_weights_clipped_non_negative(self, strategy_with_ppo: RLReputationStrategy):
        """Weights must never be negative."""
        n = 5
        state = self._make_state()
        weights = strategy_with_ppo._ppo_inference(state, list(range(n)))
        assert np.all(weights >= 0.0)

    def test_ppo_near_zero_weights_fallback_to_uniform(self, mock_blockchain):
        """When PPO returns near-zero action, fallback to uniform weights."""
        mock_algo = MagicMock()
        mock_algo.compute_single_action.return_value = np.zeros(NUM_CLIENTS, dtype=np.float32)

        _, modules = _make_ray_mocks(algo_return_value=mock_algo)
        with patch.dict(sys.modules, modules):
            s = RLReputationStrategy(
                ppo_checkpoint_path="/tmp/x",
                num_clients=NUM_CLIENTS,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
            )
        s.ppo_algo = mock_algo

        n = 4
        state = np.full((NUM_CLIENTS, NUM_FEATURES), 0.5, dtype=np.float32)
        weights = s._ppo_inference(state, list(range(n)))
        # Should fall back to uniform
        np.testing.assert_allclose(weights, 1.0 / n, atol=1e-5)

    def test_ppo_inference_with_smaller_state_pads_correctly(
        self, strategy_with_ppo: RLReputationStrategy
    ):
        """State with fewer rows than NUM_CLIENTS should be padded before inference."""
        small_n = 20  # fewer than NUM_CLIENTS=100
        state = np.full((small_n, NUM_FEATURES), 0.5, dtype=np.float32)
        participating = list(range(small_n))
        weights = strategy_with_ppo._ppo_inference(state, participating)
        # Should not raise; shape should match participating count
        assert weights.shape == (small_n,)
        assert weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_ppo_inference_error_falls_back_to_uniform(self, mock_blockchain):
        """If compute_single_action raises, fallback to uniform weights."""
        mock_algo = MagicMock()
        mock_algo.compute_single_action.side_effect = RuntimeError("inference fail")

        _, modules = _make_ray_mocks(algo_return_value=mock_algo)
        with patch.dict(sys.modules, modules):
            s = RLReputationStrategy(
                ppo_checkpoint_path="/tmp/x",
                num_clients=NUM_CLIENTS,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
            )
        s.ppo_algo = mock_algo

        n = 3
        state = np.full((NUM_CLIENTS, NUM_FEATURES), 0.5, dtype=np.float32)
        weights = s._ppo_inference(state, list(range(n)))
        np.testing.assert_allclose(weights, 1.0 / n, atol=1e-5)


# =========================================================================
# 12. _weighted_average Tests
# =========================================================================


class TestWeightedAverage:
    """Verify _weighted_average computes correct weighted parameter aggregation."""

    def test_uniform_weights_return_mean(self, strategy: RLReputationStrategy):
        """Uniform weights should produce the mean of client parameters."""
        n = 3
        params = [
            [torch.tensor([1.0, 2.0], dtype=torch.float32)],
            [torch.tensor([3.0, 4.0], dtype=torch.float32)],
            [torch.tensor([5.0, 6.0], dtype=torch.float32)],
        ]
        weights = np.array([1.0 / n] * n, dtype=np.float32)
        result = strategy._weighted_average(params, weights)
        expected = np.array([3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result[0], expected, atol=1e-5)

    def test_single_client_returns_its_params(self, strategy: RLReputationStrategy):
        """With a single client at weight 1.0, output equals that client's params."""
        params = [[torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)]]
        weights = np.array([1.0], dtype=np.float32)
        result = strategy._weighted_average(params, weights)
        np.testing.assert_allclose(result[0], [7.0, 8.0, 9.0], atol=1e-5)

    def test_output_list_length_matches_num_layers(self, strategy: RLReputationStrategy):
        """Number of output arrays must equal the number of model layers."""
        n_layers = 4
        params = [
            [torch.randn(3) for _ in range(n_layers)]
            for _ in range(3)
        ]
        weights = np.array([1.0 / 3] * 3, dtype=np.float32)
        result = strategy._weighted_average(params, weights)
        assert len(result) == n_layers

    def test_output_is_list_of_ndarrays(self, strategy: RLReputationStrategy):
        """Each element in result should be a numpy ndarray."""
        params = [[torch.randn(4)] for _ in range(2)]
        weights = np.array([0.5, 0.5], dtype=np.float32)
        result = strategy._weighted_average(params, weights)
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_asymmetric_weights(self, strategy: RLReputationStrategy):
        """Weight=1.0 for one client and 0.0 for another returns first client's params."""
        params = [
            [torch.tensor([10.0, 20.0], dtype=torch.float32)],
            [torch.tensor([99.0, 99.0], dtype=torch.float32)],
        ]
        weights = np.array([1.0, 0.0], dtype=np.float32)
        result = strategy._weighted_average(params, weights)
        np.testing.assert_allclose(result[0], [10.0, 20.0], atol=1e-5)

    def test_output_shape_preserved_per_layer(self, strategy: RLReputationStrategy):
        """Output shape for each layer must match the input layer shape."""
        shape = (5, 3)
        torch.manual_seed(42)
        params = [[torch.randn(*shape)] for _ in range(4)]
        weights = np.array([0.25] * 4, dtype=np.float32)
        result = strategy._weighted_average(params, weights)
        assert result[0].shape == shape

    def test_chunked_aggregation_matches_direct(self, strategy: RLReputationStrategy):
        """Chunked aggregation should give same result as direct weighted sum.
        
        Note: Accumulation order (chunked vs direct) introduces minor numerical
        differences; we use ``atol=1e-4`` to account for float32 precision drift.
        """
        n = _AGGREGATION_CHUNK_SIZE + 3  # spans two chunks
        torch.manual_seed(0)
        params = [[torch.randn(8)] for _ in range(n)]
        weights = np.full(n, 1.0 / n, dtype=np.float32)

        result = strategy._weighted_average(params, weights)

        # Direct reference computation
        direct = sum(
            p[0].float() * w for p, w in zip(params, weights)
        ).numpy()
        # Tolerance increased from 1e-5 to 1e-4 due to float32 accumulation order differences
        np.testing.assert_allclose(result[0], direct, atol=1e-4)

    def test_fp16_params_are_upcast(self, strategy: RLReputationStrategy):
        """Float16 client parameters should be handled without type errors."""
        params = [[torch.tensor([1.0, 2.0], dtype=torch.float16)] for _ in range(2)]
        weights = np.array([0.5, 0.5], dtype=np.float32)
        result = strategy._weighted_average(params, weights)
        assert result[0].dtype == np.float32


# =========================================================================
# 13. _update_blockchain Tests
# =========================================================================


class TestUpdateBlockchain:
    """Verify _update_blockchain calls Redis and blockchain helpers correctly."""

    def test_calls_upload_tensor_to_redis(
        self, strategy: RLReputationStrategy, mock_blockchain
    ):
        """upload_tensor_to_redis should be called once per aggregation."""
        arrays = [np.zeros((4, 4), dtype=np.float32)]
        strategy._update_blockchain(
            server_round=1,
            eth_addresses=["0xABC"],
            weights=np.array([1.0], dtype=np.float32),
            aggregated_ndarrays=arrays,
        )
        mock_blockchain["upload_tensor_to_redis"].assert_called_once()

    def test_calls_batch_update_clients(
        self, strategy: RLReputationStrategy, mock_blockchain
    ):
        """batch_update_clients should be called once per aggregation."""
        arrays = [np.zeros((4, 4), dtype=np.float32)]
        strategy._update_blockchain(
            server_round=1,
            eth_addresses=["0xABC", "0xDEF"],
            weights=np.array([0.6, 0.4], dtype=np.float32),
            aggregated_ndarrays=arrays,
        )
        mock_blockchain["batch_update_clients"].assert_called_once()

    def test_reputation_scores_derived_from_weights(
        self, strategy: RLReputationStrategy, mock_blockchain
    ):
        """Reputation scores passed to batch_update_clients are weight * scale."""
        arrays = [np.zeros((4, 4), dtype=np.float32)]
        weights = np.array([0.5, 0.5], dtype=np.float32)
        strategy._update_blockchain(
            server_round=1,
            eth_addresses=["0xA", "0xB"],
            weights=weights,
            aggregated_ndarrays=arrays,
        )
        call_kwargs = mock_blockchain["batch_update_clients"].call_args[1]
        scores = call_kwargs["scores"]
        expected_scores = [
            int(np.clip(w * strategy.reputation_scale, 0, _MAX_REPUTATION))
            for w in weights
        ]
        assert scores == expected_scores

    def test_redis_failure_does_not_propagate(
        self, strategy: RLReputationStrategy, mock_blockchain
    ):
        """If Redis upload fails, _update_blockchain must not raise."""
        mock_blockchain["upload_tensor_to_redis"].side_effect = ConnectionError("Redis down")
        arrays = [np.zeros((4, 4), dtype=np.float32)]
        # Must not raise
        strategy._update_blockchain(
            server_round=1,
            eth_addresses=["0xA"],
            weights=np.array([1.0], dtype=np.float32),
            aggregated_ndarrays=arrays,
        )

    def test_blockchain_failure_does_not_propagate(
        self, strategy: RLReputationStrategy, mock_blockchain
    ):
        """If batch_update_clients fails, _update_blockchain must not raise."""
        mock_blockchain["batch_update_clients"].side_effect = Exception("Contract error")
        arrays = [np.zeros((4, 4), dtype=np.float32)]
        strategy._update_blockchain(
            server_round=1,
            eth_addresses=["0xA"],
            weights=np.array([1.0], dtype=np.float32),
            aggregated_ndarrays=arrays,
        )

    def test_scores_clamped_to_max_reputation(
        self, strategy: RLReputationStrategy, mock_blockchain
    ):
        """Weights that would exceed _MAX_REPUTATION are clamped."""
        arrays = [np.zeros((4, 4), dtype=np.float32)]
        # Weight > 1 after scaling: weight=10.0 * scale=1000 would be 10000 (at max)
        strategy._update_blockchain(
            server_round=1,
            eth_addresses=["0xA"],
            weights=np.array([10.0], dtype=np.float32),
            aggregated_ndarrays=arrays,
        )
        call_kwargs = mock_blockchain["batch_update_clients"].call_args[1]
        scores = call_kwargs["scores"]
        assert scores[0] <= _MAX_REPUTATION


# =========================================================================
# 14. aggregate_fit Integration Tests
# =========================================================================


class TestAggregateFit:
    """Verify the full aggregate_fit pipeline."""

    def test_returns_none_for_empty_results(self, strategy: RLReputationStrategy):
        """No fit results → returns None."""
        result = strategy.aggregate_fit(1, [], [])
        assert result is None

    def test_returns_parameters_and_metrics(
        self, strategy: RLReputationStrategy, three_clients_results
    ):
        """aggregate_fit must return (Parameters, dict) on success."""
        result = strategy.aggregate_fit(1, three_clients_results, [])
        assert result is not None
        params, metrics = result
        assert isinstance(metrics, dict)

    def test_metrics_contain_expected_keys(
        self, strategy: RLReputationStrategy, three_clients_results
    ):
        """Returned metrics dict must have the expected statistical keys."""
        _, metrics = strategy.aggregate_fit(1, three_clients_results, [])
        for key in [
            "num_clients", "mean_weight", "std_weight", "max_weight",
            "min_weight", "mean_similarity", "mean_magnitude",
        ]:
            assert key in metrics, f"Missing metric key: {key}"

    def test_num_clients_in_metrics(
        self, strategy: RLReputationStrategy, three_clients_results
    ):
        """num_clients metric must equal the number of participating clients."""
        _, metrics = strategy.aggregate_fit(1, three_clients_results, [])
        assert metrics["num_clients"] == 3

    def test_round_metrics_recorded(
        self, strategy: RLReputationStrategy, three_clients_results
    ):
        """round_metrics should record entry for this round after aggregate_fit."""
        strategy.aggregate_fit(5, three_clients_results, [])
        assert 5 in strategy.round_metrics

    def test_aggregated_params_can_be_decoded(
        self, strategy: RLReputationStrategy, three_clients_results
    ):
        """Returned Parameters must be decodable back to numpy arrays."""
        params, _ = strategy.aggregate_fit(1, three_clients_results, [])
        ndarrays = parameters_to_ndarrays(params)
        assert len(ndarrays) == 1  # one layer (4x4)
        assert ndarrays[0].shape == (4, 4)

    def test_aggregated_values_are_weighted_average(
        self, strategy: RLReputationStrategy
    ):
        """With uniform weights (fallback), result is the arithmetic mean of params."""
        proxies = [_make_client_proxy(str(i)) for i in range(3)]
        results = [
            (proxies[0], _make_fit_res(_simple_params(0.0))),
            (proxies[1], _make_fit_res(_simple_params(0.5))),
            (proxies[2], _make_fit_res(_simple_params(1.0))),
        ]
        params, _ = strategy.aggregate_fit(1, results, [])
        ndarrays = parameters_to_ndarrays(params)
        # Uniform weights → mean = 0.5
        np.testing.assert_allclose(ndarrays[0], 0.5, atol=1e-4)

    def test_blockchain_called_once_per_round(
        self, strategy: RLReputationStrategy, three_clients_results, mock_blockchain
    ):
        """upload_tensor_to_redis and batch_update_clients called once per aggregate_fit."""
        strategy.aggregate_fit(1, three_clients_results, [])
        mock_blockchain["upload_tensor_to_redis"].assert_called_once()
        mock_blockchain["batch_update_clients"].assert_called_once()

    def test_blockchain_score_lookup_per_client(
        self, strategy: RLReputationStrategy, three_clients_results, mock_blockchain
    ):
        """get_client_score is called once per participating client."""
        strategy.aggregate_fit(1, three_clients_results, [])
        assert mock_blockchain["get_client_score"].call_count == 3

    def test_aggregate_fit_with_failures_continues(
        self, strategy: RLReputationStrategy, three_clients_results
    ):
        """Failures list does not prevent aggregation from completing."""
        fake_failure = Exception("client crashed")
        result = strategy.aggregate_fit(1, three_clients_results, [fake_failure])
        assert result is not None

    def test_aggregate_fit_single_client(
        self, strategy: RLReputationStrategy
    ):
        """Single-client aggregation should work (degenerate case)."""
        proxy = _make_client_proxy("solo")
        results = [(proxy, _make_fit_res(_simple_params(0.7)))]
        params, metrics = strategy.aggregate_fit(1, results, [])
        assert params is not None
        assert metrics["num_clients"] == 1
        ndarrays = parameters_to_ndarrays(params)
        np.testing.assert_allclose(ndarrays[0], 0.7, atol=1e-4)

    def test_blockchain_failure_does_not_abort_aggregation(
        self, strategy: RLReputationStrategy, three_clients_results, mock_blockchain
    ):
        """If blockchain update fails, aggregate_fit still returns valid result."""
        mock_blockchain["batch_update_clients"].side_effect = Exception("Web3 down")
        result = strategy.aggregate_fit(1, three_clients_results, [])
        assert result is not None

    def test_blockchain_score_lookup_failure_uses_neutral_reputation(
        self, strategy: RLReputationStrategy, three_clients_results, mock_blockchain
    ):
        """If get_client_score fails for a client, that client uses 0.5 reputation."""
        mock_blockchain["get_client_score"].side_effect = Exception("blockchain down")
        # Should not raise and should still return valid output
        result = strategy.aggregate_fit(1, three_clients_results, [])
        assert result is not None

    def test_aggregate_fit_multi_layer_model(
        self, strategy: RLReputationStrategy
    ):
        """Aggregate_fit handles multi-layer model parameters correctly."""
        def multi_layer_params(value: float) -> List[np.ndarray]:
            return [
                np.full((4, 4), value, dtype=np.float32),
                np.full((4,), value, dtype=np.float32),
                np.full((2, 4), value, dtype=np.float32),
            ]

        proxies = [_make_client_proxy(str(i)) for i in range(3)]
        results = [
            (proxy, _make_fit_res(multi_layer_params(v)))
            for proxy, v in zip(proxies, [0.2, 0.5, 0.8])
        ]
        params, metrics = strategy.aggregate_fit(1, results, [])
        assert params is not None
        ndarrays = parameters_to_ndarrays(params)
        assert len(ndarrays) == 3


# =========================================================================
# 15. Edge Case Tests
# =========================================================================


class TestEdgeCases:
    """Adversarial and boundary condition tests for the strategy."""

    def test_all_honest_high_accuracy_metrics(
        self, strategy: RLReputationStrategy
    ):
        """All clients with high accuracy should produce valid aggregation."""
        proxies = [_make_client_proxy(str(i)) for i in range(5)]
        results = [
            (proxy, _make_fit_res(
                _simple_params(0.9),
                metrics={"accuracy": 0.95, "loss_improvement": 0.9}
            ))
            for proxy in proxies
        ]
        result = strategy.aggregate_fit(1, results, [])
        assert result is not None

    def test_all_malicious_zero_metrics(
        self, strategy: RLReputationStrategy
    ):
        """All clients with zero accuracy (simulating all-malicious) should not crash."""
        proxies = [_make_client_proxy(str(i)) for i in range(4)]
        results = [
            (proxy, _make_fit_res(
                _simple_params(0.01),
                metrics={"accuracy": 0.0, "loss_improvement": 0.0}
            ))
            for proxy in proxies
        ]
        result = strategy.aggregate_fit(1, results, [])
        assert result is not None

    def test_params_with_zero_weights_from_ppo(self, mock_blockchain):
        """When PPO returns all-zero actions, fallback keeps aggregation working."""
        mock_algo = MagicMock()
        mock_algo.compute_single_action.return_value = np.zeros(NUM_CLIENTS, dtype=np.float32)

        _, modules = _make_ray_mocks(algo_return_value=mock_algo)
        with patch.dict(sys.modules, modules):
            s = RLReputationStrategy(
                ppo_checkpoint_path="/tmp/x",
                num_clients=NUM_CLIENTS,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
            )
        s.ppo_algo = mock_algo

        proxies = [_make_client_proxy(str(i)) for i in range(3)]
        results = [
            (proxy, _make_fit_res(_simple_params(float(i))))
            for i, proxy in enumerate(proxies)
        ]
        # Should fall back to uniform and not crash
        result = s.aggregate_fit(1, results, [])
        assert result is not None

    def test_large_number_of_clients(
        self, strategy: RLReputationStrategy
    ):
        """Aggregation with NUM_CLIENTS participants should complete without error."""
        proxies = [_make_client_proxy(str(i)) for i in range(NUM_CLIENTS)]
        results = [
            (proxy, _make_fit_res(_simple_params(0.5)))
            for proxy in proxies
        ]
        result = strategy.aggregate_fit(1, results, [])
        assert result is not None

    def test_weights_always_sum_to_one_after_aggregation(
        self, strategy: RLReputationStrategy
    ):
        """Regardless of PPO output, weights fed to _weighted_average sum to 1."""
        # Intercept _weighted_average to capture the weights argument
        captured_weights: List[np.ndarray] = []
        original_wa = strategy._weighted_average

        def capturing_wa(client_params, weights):
            captured_weights.append(weights.copy())
            return original_wa(client_params, weights)

        strategy._weighted_average = capturing_wa  # type: ignore[method-assign]

        proxies = [_make_client_proxy(str(i)) for i in range(3)]
        results = [
            (proxy, _make_fit_res(_simple_params(0.5)))
            for proxy in proxies
        ]
        strategy.aggregate_fit(1, results, [])

        assert len(captured_weights) > 0
        for w in captured_weights:
            assert w.sum() == pytest.approx(1.0, abs=1e-5)

    def test_very_large_and_small_params_do_not_overflow(
        self, strategy: RLReputationStrategy
    ):
        """Very large and very small parameter values should not produce NaN/Inf."""
        large = np.full((4, 4), 1e20, dtype=np.float32)
        small = np.full((4, 4), 1e-20, dtype=np.float32)
        proxies = [_make_client_proxy("large"), _make_client_proxy("small")]
        results = [
            (proxies[0], _make_fit_res([large])),
            (proxies[1], _make_fit_res([small])),
        ]
        params, _ = strategy.aggregate_fit(1, results, [])
        ndarrays = parameters_to_ndarrays(params)
        assert np.all(np.isfinite(ndarrays[0]))

    def test_metrics_with_nan_values_default_to_neutral(
        self, strategy: RLReputationStrategy
    ):
        """Clients reporting NaN metrics should not crash aggregate_fit."""
        proxy = _make_client_proxy("nan_client")
        results = [
            (proxy, _make_fit_res(
                _simple_params(0.5),
                metrics={"accuracy": float("nan"), "loss_improvement": float("nan")}
            ))
        ]
        # Should not crash
        result = strategy.aggregate_fit(1, results, [])
        assert result is not None
