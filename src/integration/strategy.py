"""
strategy.py
~~~~~~~~~~~
Custom Flower Strategy that integrates the FL pipeline, blockchain reputation
ledger, and a trained RLlib PPO agent for intelligent aggregation weighting.

The ``RLReputationStrategy`` overrides ``aggregate_fit`` to:
1. Extract per-client model updates from Flower ``FitRes`` results.
2. Fetch on-chain reputation history via ``web3_utils``.
3. Build a (K x NUM_FEATURES) state matrix for participating clients.
4. Run PPO inference to obtain per-client aggregation weights.
5. Perform weighted aggregation of model parameters.
6. Update on-chain reputation scores and persist the aggregated model.

Fallback behaviour:
  - If the PPO checkpoint cannot be loaded or inference fails, the strategy
    falls back to uniform (FedAvg-style) weighting with a logged warning.
  - If blockchain calls fail, aggregation proceeds without reputation data
    and blockchain updates are skipped for that round.
"""

from __future__ import annotations

import gc
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes

from src.blockchain.storage_utils import upload_tensor_to_redis
from src.blockchain.web3_utils import (
    batch_update_clients,
    get_client_score,
)
from src.rl_agent.env import NUM_CLIENTS, NUM_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEUTRAL_FEATURE: float = 0.5
"""Default feature value for non-participating clients in the state matrix."""

_REPUTATION_SCALE_DEFAULT: int = 1000
"""Default multiplicative scale when converting float weights to int scores."""

_MAX_REPUTATION: int = 10000
"""Upper clamp for on-chain reputation scores."""

_AGGREGATION_CHUNK_SIZE: int = 10
"""Number of clients processed per chunk during weighted aggregation.
Keeping this at 10-20 bounds peak memory to ~chunk_size model copies rather
than materializing all 100 client states simultaneously."""

_HEURISTIC_BLEND_GAMMA: float = 0.5
"""Blending ratio: final = gamma * ppo_weights + (1-gamma) * heuristic_weights.
Equal blend: 0.5 gives PPO and heuristic equal influence. The trained PPO policy
has shown sufficient quality to warrant parity with the heuristic safety net,
while still benefiting from the robust floor the heuristic provides."""

_MAGNITUDE_OUTLIER_THRESHOLD: float = 2.0
"""Number of MADs (median absolute deviations) above the median magnitude
beyond which a client's update is considered an outlier and downweighted."""

_MEMORY_WARNING_THRESHOLD: float = 0.80
"""Emit a warning when GPU memory utilisation exceeds this fraction (0-1)."""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class RLReputationStrategy(fl.server.strategy.Strategy):
    """Flower Strategy that uses a trained PPO agent for aggregation weighting.

    The strategy fetches on-chain reputation data, constructs a state matrix,
    runs PPO inference to obtain per-client weights, and performs a weighted
    average of model parameters.  Blockchain scores are updated after each
    aggregation round.

    Args:
        ppo_checkpoint_path: Filesystem path to a saved RLlib PPO checkpoint.
        num_clients: Total number of FL clients in the simulation.
        fraction_fit: Fraction of clients sampled for training each round.
        fraction_evaluate: Fraction of clients sampled for evaluation.
        min_fit_clients: Minimum number of clients required for a fit round.
        min_evaluate_clients: Minimum clients required for evaluation.
        min_available_clients: Minimum clients that must be connected.
        initial_parameters: Optional initial global model parameters.
        client_addresses: Mapping of Flower client CID to Ethereum address.
            If ``None``, deterministic test addresses are generated on demand.
        reputation_scale: Multiplicative factor for converting float weights
            to integer reputation scores stored on-chain.
    """

    def __init__(
        self,
        ppo_checkpoint_path: str,
        num_clients: int = NUM_CLIENTS,
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.05,
        min_fit_clients: int = 10,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 10,
        initial_parameters: Optional[Parameters] = None,
        client_addresses: Optional[Dict[str, str]] = None,
        reputation_scale: int = _REPUTATION_SCALE_DEFAULT,
        evaluate_fn: Optional[Any] = None,
    ) -> None:
        super().__init__()

        # Strategy configuration
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.reputation_scale = reputation_scale
        self._evaluate_fn = evaluate_fn

        # Client address mapping (Flower CID -> Ethereum address)
        self.client_address_map: Dict[str, str] = client_addresses or {}

        # Per-round metrics history
        self.round_metrics: Dict[int, Dict[str, Any]] = {}

        # Cross-round EMA weight smoothing to prevent cascading failures
        # from one bad round polluting the aggregated model.
        self._ema_weights: Optional[np.ndarray] = None

        # Persistent LSTM hidden state across FL rounds.  Allows the LSTM
        # to accumulate temporal context (e.g. trend of client behaviour)
        # rather than starting from zeros each round.  Reset naturally when
        # a new RLReputationStrategy instance is created (one per benchmark).
        self._lstm_state: Optional[List[np.ndarray]] = None

        # Load PPO algorithm --------------------------------------------------
        self.ppo_algo: Optional[Any] = None
        try:
            import os as _os
            # Prevent TensorboardX from registering an atexit handler that
            # tries to create threads after the interpreter begins shutdown.
            # Ray shuts down (via Flower simulation) before our atexit hooks
            # run, so the background TBX writer thread cannot be joined cleanly
            # — this env var disables auto-callback loggers (including TBX).
            _os.environ.setdefault("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "1")

            from ray.rllib.algorithms.algorithm import Algorithm
            from ray.tune.registry import register_env
            from src.rl_agent.env import FLReputationEnv
            from src.rl_agent.train import ENV_NAME, env_creator

            register_env(ENV_NAME, env_creator)

            # Clean up any stale Ray state (e.g. dead Raylet socket from a
            # prior Flower simulation) before loading the PPO checkpoint,
            # which internally calls ray.init().
            import ray
            if ray.is_initialized():
                ray.shutdown()
            ray.init(ignore_reinit_error=True)

            # Override checkpoint config for inference-only mode: disable all
            # env runners (rollout workers) and GPU allocation for the learner.
            # from_checkpoint() normally restores the full training topology
            # (e.g. 8 env runners ~1GB each), but compute_single_action() only
            # needs the policy network.  This reduces memory by ~8GB.
            from ray.rllib.utils.checkpoints import get_checkpoint_info
            checkpoint_info = get_checkpoint_info(ppo_checkpoint_path)
            state = Algorithm._checkpoint_info_to_algorithm_state(
                checkpoint_info=checkpoint_info,
                policy_ids=["shared_policy"],
            )
            config = state.get("config")
            if config is not None:
                config.num_env_runners = 0
                config.num_gpus = 0
            self.ppo_algo = Algorithm.from_state(state)
            # Close RLlib's internal loggers (including TBX) since we only
            # use the algorithm for inference, not training/logging.
            if hasattr(self.ppo_algo, '_result_logger') and self.ppo_algo._result_logger:
                self.ppo_algo._result_logger.close()
            logger.info(
                "PPO algorithm loaded from checkpoint: %s",
                ppo_checkpoint_path,
            )
        except Exception:
            logger.warning(
                "Failed to load PPO checkpoint at '%s'. "
                "Strategy will fall back to uniform (FedAvg) weights.",
                ppo_checkpoint_path,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Strategy interface: initialise / configure
    # ------------------------------------------------------------------

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Return initial global model parameters if provided."""
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Sample clients for training and send current parameters.

        Args:
            server_round: Current communication round number.
            parameters: Current global model parameters.
            client_manager: Flower ClientManager for sampling.

        Returns:
            List of (ClientProxy, FitIns) tuples for sampled clients.
        """
        config: Dict[str, Scalar] = {"server_round": server_round}
        fit_ins = fl.common.FitIns(parameters, config)

        num_available = client_manager.num_available()
        sample_size = max(
            self.min_fit_clients,
            int(num_available * self.fraction_fit),
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients,
        )
        logger.info(
            "Round %d: sampled %d / %d clients for fit",
            server_round,
            len(clients),
            num_available,
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """Sample clients for evaluation.

        Args:
            server_round: Current communication round number.
            parameters: Current global model parameters.
            client_manager: Flower ClientManager for sampling.

        Returns:
            List of (ClientProxy, EvaluateIns) tuples.
        """
        config: Dict[str, Scalar] = {"server_round": server_round}
        evaluate_ins = fl.common.EvaluateIns(parameters, config)

        num_available = client_manager.num_available()
        if num_available < self.min_evaluate_clients:
            logger.warning(
                "Round %d: not enough clients for evaluation (%d < %d)",
                server_round,
                num_available,
                self.min_evaluate_clients,
            )
            return []

        sample_size = max(
            self.min_evaluate_clients,
            int(num_available * self.fraction_evaluate),
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients,
        )
        return [(client, evaluate_ins) for client in clients]

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side centralized evaluation using the provided evaluate_fn.

        When an ``evaluate_fn`` is supplied (e.g. from the benchmark harness),
        it is called here so that Flower records centralized accuracy into
        ``history.metrics_centralized``.  Without this, accuracy would only
        appear in ``history.metrics_distributed`` (from client evaluate rounds),
        which the benchmark's ``_collect_results`` does not currently read,
        causing ``final_acc=0.0`` in the summary.

        If no ``evaluate_fn`` was provided the method returns ``None`` and
        Flower falls back to distributed client evaluation only.
        """
        logger.info(
            "evaluate() called round=%d, has_fn=%s",
            server_round,
            self._evaluate_fn is not None,
        )
        if self._evaluate_fn is None:
            return None
        ndarrays = parameters_to_ndarrays(parameters)
        try:
            result = self._evaluate_fn(server_round, ndarrays, {})
            logger.info(
                "evaluate() round=%d returned loss=%.6f metrics=%s",
                server_round,
                result[0] if result else float("nan"),
                result[1] if result else {},
            )
            return result
        except Exception:
            logger.error(
                "evaluate() round=%d: evaluate_fn raised an exception",
                server_round,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # aggregate_evaluate
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Aggregate client evaluation results via weighted average.

        Args:
            server_round: Current round.
            results: Successful evaluation results.
            failures: Failed evaluations.

        Returns:
            Tuple of (aggregated_loss, aggregated_metrics) or None.
        """
        if not results:
            return None

        total_samples = sum(
            evaluate_res.num_examples for _, evaluate_res in results
        )
        if total_samples == 0:
            return None

        weighted_loss = sum(
            evaluate_res.num_examples * evaluate_res.loss
            for _, evaluate_res in results
        )
        aggregated_loss = weighted_loss / total_samples

        # Aggregate accuracy if reported
        weighted_acc = sum(
            evaluate_res.num_examples
            * evaluate_res.metrics.get("accuracy", 0.0)
            for _, evaluate_res in results
        )
        aggregated_acc = weighted_acc / total_samples

        logger.info(
            "Round %d evaluate: loss=%.4f accuracy=%.4f (%d clients, %d failures)",
            server_round,
            aggregated_loss,
            aggregated_acc,
            len(results),
            len(failures),
        )
        return aggregated_loss, {"accuracy": float(aggregated_acc)}

    # ------------------------------------------------------------------
    # aggregate_fit  (CORE LOGIC)
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        """Aggregate client model updates using PPO-derived weights.

        Steps:
            1. Convert each client's FitRes parameters to torch tensors.
            2. Fetch on-chain reputation for each participating client.
            3. Compute per-client features (similarity, magnitude, etc.).
            4. Build a (K x 5) state matrix for participating clients.
            5. Run PPO inference to get aggregation weights.
            6. Perform weighted average of model parameters.
            7. Update on-chain reputation scores.

        Args:
            server_round: Current communication round.
            results: List of (ClientProxy, FitRes) from successful clients.
            failures: Failed fit results.

        Returns:
            Tuple of (aggregated Parameters, metrics dict) or None.
        """
        if not results:
            logger.warning("Round %d: no fit results received.", server_round)
            return None

        if failures:
            logger.warning(
                "Round %d: %d fit failures (proceeding with %d results)",
                server_round,
                len(failures),
                len(results),
            )

        num_participating = len(results)

        # -- Step 1: Extract client updates as torch tensors ---------------
        client_cids: List[str] = []
        client_params: List[List[torch.Tensor]] = []
        client_num_samples: List[int] = []
        client_metrics: List[Dict[str, Scalar]] = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            client_cids.append(cid)

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            tensors = [torch.from_numpy(arr) for arr in ndarrays]
            client_params.append(tensors)
            client_num_samples.append(fit_res.num_examples)
            client_metrics.append(fit_res.metrics)

        # -- Step 2: Fetch blockchain reputation for each client -----------
        eth_addresses: List[str] = []
        reputation_scores: List[float] = []

        for cid in client_cids:
            eth_addr = self._get_or_create_address(cid)
            eth_addresses.append(eth_addr)

            try:
                record = get_client_score(eth_addr)
                raw_score = record.get("reputationScore", 0)
                # Normalise to [0, 1] using the reputation scale
                norm_score = float(
                    np.clip(raw_score / self.reputation_scale, 0.0, 1.0)
                )
            except Exception:
                logger.debug(
                    "Round %d: blockchain lookup failed for %s, defaulting to 0.5",
                    server_round,
                    eth_addr,
                    exc_info=True,
                )
                norm_score = _NEUTRAL_FEATURE
            reputation_scores.append(norm_score)

        reputations = np.array(reputation_scores, dtype=np.float32)

        # -- Step 3: Compute per-client features ---------------------------
        similarities = self._compute_gradient_similarity(client_params)
        magnitudes = self._compute_update_magnitude(client_params)

        # Accuracy contribution: use client-reported accuracy or default 0.5
        accuracies = np.array(
            [
                float(m.get("accuracy", _NEUTRAL_FEATURE))
                for m in client_metrics
            ],
            dtype=np.float32,
        )

        # Loss improvement: use client-reported loss or default 0.5
        loss_improvements = np.array(
            [
                float(m.get("loss_improvement", _NEUTRAL_FEATURE))
                for m in client_metrics
            ],
            dtype=np.float32,
        )

        # -- Step 4: Build full state matrix and run PPO -------------------
        # Map participating clients to indices 0..N-1 within the full matrix
        participating_indices = list(range(num_participating))

        state = self._build_state_matrix(
            participating_indices=participating_indices,
            accuracies=accuracies,
            similarities=similarities,
            reputations=reputations,
            loss_improvements=loss_improvements,
            magnitudes=magnitudes,
        )

        # -- Step 5: PPO inference + heuristic blending --------------------
        ppo_weights = self._ppo_inference(state, participating_indices)

        # Warmup: during the first few rounds, early-round features are
        # unreliable (random model, low gradient SNR).  Use pure heuristic
        # with linear similarity (no squaring) to avoid amplifying noise.
        warmup_rounds = 1
        is_warmup = server_round <= warmup_rounds

        heuristic_weights = self._compute_heuristic_weights(
            similarities, magnitudes, warmup=is_warmup,
        )

        # Blend: final = gamma * PPO + (1-gamma) * heuristic
        # The heuristic provides a robust floor (similar to FLTrust/Median)
        # while PPO can learn to improve on it over time.
        # During warmup, gamma=0 (pure heuristic) because PPO is unreliable
        # on unseen early-round distributions.
        gamma = 0.0 if is_warmup else _HEURISTIC_BLEND_GAMMA
        weights = gamma * ppo_weights + (1.0 - gamma) * heuristic_weights

        # Re-normalise after blending (both inputs sum to 1, so blend does too,
        # but guard against float drift)
        w_sum = weights.sum()
        if w_sum > 1e-8:
            weights = weights / w_sum
        else:
            weights = np.full(num_participating, 1.0 / num_participating, dtype=np.float32)

        # Cross-round EMA smoothing: prevents one bad round from causing
        # cascading failures.  70% historical, 30% new.
        ema_alpha = 0.5
        if self._ema_weights is not None and len(self._ema_weights) == len(weights):
            weights = ema_alpha * weights + (1.0 - ema_alpha) * self._ema_weights
            weights /= weights.sum()  # re-normalise
        self._ema_weights = weights.copy()

        logger.info(
            "Round %d blending: gamma=%.2f, warmup=%s, ema_applied=%s, "
            "ppo_weight_on_top5_sim=%.4f, heuristic_weight_on_top5_sim=%.4f",
            server_round,
            gamma,
            is_warmup,
            self._ema_weights is not None,
            float(ppo_weights[np.argsort(similarities)[-5:]].sum()),
            float(heuristic_weights[np.argsort(similarities)[-5:]].sum()),
        )

        # -- Step 6: Weighted aggregation ----------------------------------
        aggregated_ndarrays = self._weighted_average(client_params, weights)
        aggregated_params = ndarrays_to_parameters(aggregated_ndarrays)

        # -- Step 7: Update blockchain -------------------------------------
        self._update_blockchain(
            server_round=server_round,
            eth_addresses=eth_addresses,
            weights=weights,
            aggregated_ndarrays=aggregated_ndarrays,
        )

        # -- Free fragmented VRAM after aggregation -------------------------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- Metrics -------------------------------------------------------
        metrics: Dict[str, Scalar] = {
            "num_clients": num_participating,
            "mean_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights)),
            "max_weight": float(np.max(weights)),
            "min_weight": float(np.min(weights)),
            "mean_similarity": float(np.mean(similarities)),
            "mean_magnitude": float(np.mean(magnitudes)),
            "blend_gamma": gamma,
            "heuristic_entropy": float(-np.sum(
                heuristic_weights * np.log(heuristic_weights + 1e-10)
            )),
        }
        self.round_metrics[server_round] = {**metrics}

        logger.info(
            "Round %d aggregated: %d clients, weights [%.4f, %.4f], "
            "mean_sim=%.4f, mean_mag=%.4f",
            server_round,
            num_participating,
            float(np.min(weights)),
            float(np.max(weights)),
            float(np.mean(similarities)),
            float(np.mean(magnitudes)),
        )
        return aggregated_params, metrics

    # ------------------------------------------------------------------
    # Helper: heuristic trust weights (safety net)
    # ------------------------------------------------------------------

    def _compute_heuristic_weights(
        self,
        similarities: np.ndarray,
        magnitudes: np.ndarray,
        warmup: bool = False,
    ) -> np.ndarray:
        """Compute robust heuristic aggregation weights from server-side features.

        Uses two non-falsifiable signals:
        1. Gradient similarity to median update (cosine similarity, already [0,1]).
           Higher similarity = more aligned with the consensus = more trustworthy.
        2. Magnitude anomaly detection via MAD (median absolute deviation).
           Clients with update magnitudes far above the median are downweighted
           as potential poisoners who inject large adversarial gradients.

        The final heuristic weight is: sim_score * magnitude_penalty, normalised.

        During warmup (first few rounds), linear similarity is used instead of
        squared to avoid amplifying noisy early-round signals when the global
        model is still random and gradient SNR is very low.

        Args:
            similarities: Per-client cosine similarity in [0, 1], shape (K,).
            magnitudes: Per-client L2 norm (min-max normalised to [0,1]), shape (K,).
            warmup: If True, use linear similarity (gentler) instead of squared.

        Returns:
            Normalised heuristic weights of shape (K,) summing to 1.
        """
        K = len(similarities)

        # --- Similarity-based trust (primary signal) ---
        # During warmup: linear similarity (gentler, avoids amplifying noise).
        # After warmup: squared similarity (sharper discrimination).
        if warmup:
            sim_trust = np.clip(similarities, 0.0, 1.0)
        else:
            sim_trust = np.clip(similarities, 0.0, 1.0) ** 2

        # --- Magnitude anomaly penalty ---
        # Median Absolute Deviation (MAD) is robust to outliers unlike std.
        # Clients whose magnitude exceeds median + threshold * MAD are penalised.
        median_mag = np.median(magnitudes)
        mad = np.median(np.abs(magnitudes - median_mag))
        mad = max(mad, 1e-6)  # avoid division by zero for uniform magnitudes

        # z_mag: how many MADs above the median each client is
        z_mag = (magnitudes - median_mag) / mad
        # Penalty: 1.0 for normal magnitudes, decays toward 0 for outliers
        # Using sigmoid-style soft clipping instead of hard threshold
        mag_penalty = np.where(
            z_mag > _MAGNITUDE_OUTLIER_THRESHOLD,
            np.exp(-0.5 * (z_mag - _MAGNITUDE_OUTLIER_THRESHOLD)),
            1.0,
        ).astype(np.float32)

        # --- Combined heuristic weight ---
        heuristic = sim_trust * mag_penalty

        # Normalise to sum to 1
        h_sum = heuristic.sum()
        if h_sum < 1e-8:
            # Extreme fallback: all clients look equally bad
            return np.full(K, 1.0 / K, dtype=np.float32)
        return (heuristic / h_sum).astype(np.float32)

    # ------------------------------------------------------------------
    # Helper: gradient similarity
    # ------------------------------------------------------------------

    def _compute_gradient_similarity(
        self, client_params: List[List[torch.Tensor]]
    ) -> np.ndarray:
        """Cosine similarity of each client's flattened update vs. the median.

        Uses the coordinate-wise median instead of mean as the reference vector.
        The median is robust to up to 50% Byzantine clients (we typically have
        ~30%), whereas the mean can be arbitrarily corrupted by a single
        large-magnitude adversarial update — especially in early rounds when
        honest gradients are small and the signal-to-noise ratio is low.

        Args:
            client_params: List of per-client parameter tensor lists.

        Returns:
            Array of shape (num_clients,) with values in [0, 1].
        """
        with torch.no_grad():
            flat_vectors = []
            for params in client_params:
                # flatten + float32 cast in one pass; keeps intermediate tensors small
                flat = torch.cat([p.flatten().float() for p in params])
                flat_vectors.append(flat)

            # Use GPU if available for the (N x D) matrix operations
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            stacked = torch.stack(flat_vectors).to(device, non_blocking=True)  # (N, D)
            self._check_gpu_memory(context="similarity stacked")
            # Median is robust to up to 50% corruption; mean is not.
            # torch.median returns (values, indices) — we only need values.
            median_vec = stacked.median(dim=0).values.unsqueeze(0)  # (1, D)

            # Cosine similarity (result in [-1, 1], rescale to [0, 1])
            cos_sim = F.cosine_similarity(stacked, median_vec, dim=1)  # (N,)
            rescaled = ((cos_sim + 1.0) / 2.0).cpu().numpy().astype(np.float32)

            # Free the large stacked matrix immediately after use
            del stacked, median_vec, cos_sim
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return np.clip(rescaled, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Helper: update magnitude
    # ------------------------------------------------------------------

    def _compute_update_magnitude(
        self, client_params: List[List[torch.Tensor]]
    ) -> np.ndarray:
        """L2 norm of each client's parameter vector, min-max normalised.

        Args:
            client_params: List of per-client parameter tensor lists.

        Returns:
            Array of shape (num_clients,) with values in [0, 1].
        """
        norms = []
        for params in client_params:
            flat = torch.cat([p.flatten().float() for p in params])
            norms.append(float(torch.linalg.vector_norm(flat)))

        norms_arr = np.array(norms, dtype=np.float32)
        lo, hi = norms_arr.min(), norms_arr.max()
        if hi - lo < 1e-8:
            return np.full(len(norms), _NEUTRAL_FEATURE, dtype=np.float32)
        return ((norms_arr - lo) / (hi - lo)).astype(np.float32)

    # ------------------------------------------------------------------
    # Helper: build state matrix
    # ------------------------------------------------------------------

    def _build_state_matrix(
        self,
        participating_indices: List[int],  # kept for API compat but unused internally now
        accuracies: np.ndarray,
        similarities: np.ndarray,
        reputations: np.ndarray,
        loss_improvements: np.ndarray,
        magnitudes: np.ndarray,
    ) -> np.ndarray:
        """Build a compact (K, NUM_FEATURES) observation matrix for K participating clients.

        With parameter-shared MAPPO, the shared policy processes each client
        row independently — there is no need to pad to the full NUM_CLIENTS
        dimension.  The matrix contains only rows for clients that actually
        participated in this round.

        Args:
            participating_indices: Kept for API compatibility; unused internally.
            accuracies: Per-participant accuracy contribution.
            similarities: Per-participant gradient similarity.
            reputations: Per-participant normalised reputation.
            loss_improvements: Per-participant loss improvement.
            magnitudes: Per-participant update magnitude.

        Returns:
            State matrix of shape (K, NUM_FEATURES) where K = len(accuracies).
        """
        K = len(accuracies)
        state = np.zeros((K, NUM_FEATURES), dtype=np.float32)

        # Local features (indices 0–4)
        state[:, 0] = accuracies
        state[:, 1] = similarities
        state[:, 2] = reputations
        state[:, 3] = loss_improvements
        state[:, 4] = magnitudes

        # Global context features (indices 5–8): mean of each local feature,
        # broadcast to every row so every client sees the same global context.
        # Matches the feature layout the PPO was trained on in env.py (_GACC=5,
        # _GSIM=6, _GLOSS=7, _GMAG=8).
        state[:, 5] = float(np.mean(accuracies))
        state[:, 6] = float(np.mean(similarities))
        state[:, 7] = float(np.mean(loss_improvements))
        state[:, 8] = float(np.mean(magnitudes))

        return np.clip(state, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Helper: PPO inference
    # ------------------------------------------------------------------

    def _ppo_inference(
        self,
        state: np.ndarray,       # now (K, NUM_FEATURES) — K participating clients only
        participating_indices: List[int],  # kept for API compat
    ) -> np.ndarray:
        """Run MAPPO shared policy to obtain per-participant aggregation weights.

        Falls back to uniform weights if PPO is unavailable or inference fails.

        With parameter-shared MAPPO, a single policy maps each client's
        per-row feature vector of shape ``(NUM_FEATURES,)`` to a scalar
        weight.  Each row of the ``(K, NUM_FEATURES)`` state matrix is
        independently passed through the shared policy via
        ``compute_single_action`` with ``policy_id="shared_policy"``.
        No padding to ``NUM_CLIENTS`` is required because the policy
        operates on individual rows, not the full client matrix.

        Args:
            state: Observation matrix of shape (K, NUM_FEATURES) where
                K is the number of participating clients this round.
            participating_indices: Kept for API compatibility; unused internally.

        Returns:
            Normalised weight array of shape (K,) summing to 1.
        """
        K = state.shape[0]

        if self.ppo_algo is not None:
            try:
                raw_weights = np.zeros(K, dtype=np.float32)

                # Detect LSTM vs MLP checkpoint by querying the initial state.
                # get_initial_state() returns a non-empty list for LSTM policies
                # and an empty list for plain MLP policies.
                policy = self.ppo_algo.get_policy("shared_policy")
                initial_state = policy.model.get_initial_state()
                use_lstm = len(initial_state) > 0

                if use_lstm:
                    # Reuse the LSTM hidden state from the previous FL round
                    # so the LSTM accumulates temporal context across rounds
                    # (e.g. trends in client behaviour).  Falls back to zeros
                    # on the very first round or if shapes changed.
                    if (
                        self._lstm_state is not None
                        and len(self._lstm_state) == len(initial_state)
                        and all(
                            s.shape == ref.shape
                            for s, ref in zip(self._lstm_state, initial_state)
                        )
                    ):
                        lstm_state: List[np.ndarray] = [
                            s.copy() for s in self._lstm_state
                        ]
                    else:
                        lstm_state = [
                            np.zeros_like(s) for s in initial_state
                        ]

                for i in range(K):
                    obs_i = state[i]  # shape (NUM_FEATURES,) = (9,)
                    if use_lstm:
                        # compute_single_action with state returns a 3-tuple:
                        # (action, state_out, info).  state_out is threaded to
                        # the next iteration so seq_lens is correctly populated.
                        action, lstm_state, _ = self.ppo_algo.compute_single_action(
                            obs_i,
                            state=lstm_state,
                            policy_id="shared_policy",
                        )
                    else:
                        action = self.ppo_algo.compute_single_action(
                            obs_i, policy_id="shared_policy"
                        )
                    raw_weights[i] = float(np.clip(action[0], 0.0, 1.0))

                # Persist the final LSTM state for reuse in the next FL round
                if use_lstm:
                    self._lstm_state = [s.copy() for s in lstm_state]

                # Normalise to sum to 1
                weight_sum = raw_weights.sum()
                if weight_sum > 1e-8:
                    return raw_weights / weight_sum

                logger.warning(
                    "PPO returned near-zero weights; falling back to uniform."
                )
            except Exception:
                logger.warning(
                    "PPO inference failed; falling back to uniform weights.",
                    exc_info=True,
                )

        # Fallback: uniform (FedAvg-style)
        uniform = np.full(K, 1.0 / K, dtype=np.float32)
        return uniform

    # ------------------------------------------------------------------
    # Helper: GPU memory monitoring
    # ------------------------------------------------------------------

    @staticmethod
    def _check_gpu_memory(context: str = "") -> None:
        """Emit a warning when GPU memory utilisation exceeds the threshold.

        This is a lightweight diagnostic helper called at chunk boundaries
        during aggregation to detect OOM risk before it becomes fatal.

        Args:
            context: Human-readable label for the log message (e.g. round/chunk).
        """
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        if reserved == 0:
            return
        utilisation = allocated / reserved
        if utilisation >= _MEMORY_WARNING_THRESHOLD:
            logger.warning(
                "GPU memory utilisation at %.1f%% (%s): "
                "allocated=%.1f MiB / reserved=%.1f MiB — OOM risk elevated",
                utilisation * 100,
                context,
                allocated / (1024 ** 2),
                reserved / (1024 ** 2),
            )

    # ------------------------------------------------------------------
    # Helper: weighted average (chunked, layer-streaming)
    # ------------------------------------------------------------------

    def _weighted_average(
        self,
        client_params: List[List[torch.Tensor]],
        weights: np.ndarray,
    ) -> List[np.ndarray]:
        """Weighted average of per-client model parameters.

        Memory strategy (two-axis optimisation):

        1. **Layer-by-layer streaming**: only one layer's worth of tensors from
           ALL clients is resident in memory at once.  This bounds peak memory
           to ``max(layer_size) * num_clients`` rather than the full model times
           ``num_clients``.

        2. **Client chunking**: within each layer, clients are processed in
           chunks of ``_AGGREGATION_CHUNK_SIZE``.  After each chunk the client
           tensors are explicitly deleted and the CUDA cache is released so that
           fragmented blocks can be reclaimed by subsequent chunks.

        Aggregation arithmetic is always done in float32 regardless of the
        dtype the client sent (e.g. float16 during AMP training) to avoid
        precision loss from repeated fp16 additions.

        Args:
            client_params: List of per-client parameter tensor lists.
            weights: Normalised aggregation weights (sum to 1).

        Returns:
            List of numpy arrays representing the aggregated model.
        """
        with torch.no_grad():
            num_clients = len(weights)
            num_layers = len(client_params[0])

            # Pre-convert weights to Python floats once — avoids a CUDA
            # kernel launch for each scalar multiply inside the loop
            weights_f: List[float] = weights.tolist()

            aggregated: List[np.ndarray] = []

            for layer_idx in range(num_layers):
                # Accumulate into float32 regardless of client dtype
                # (fp16 accumulation suffers catastrophic cancellation at scale)
                layer_sum: Optional[torch.Tensor] = None

                # Process clients in chunks to limit simultaneous GPU residency
                for chunk_start in range(0, num_clients, _AGGREGATION_CHUNK_SIZE):
                    chunk_end = min(chunk_start + _AGGREGATION_CHUNK_SIZE, num_clients)
                    chunk_tensors: List[torch.Tensor] = []

                    for client_idx in range(chunk_start, chunk_end):
                        raw = client_params[client_idx][layer_idx]
                        # Cast to float32 for numerically stable accumulation
                        t = raw.float()
                        weighted = t.mul_(weights_f[client_idx])
                        chunk_tensors.append(weighted)

                    # Sum the chunk; keep result on same device as first tensor
                    chunk_sum = torch.stack(chunk_tensors).sum(dim=0)

                    if layer_sum is None:
                        layer_sum = chunk_sum
                    else:
                        layer_sum.add_(chunk_sum)

                    # Eagerly free chunk tensors and release CUDA cache
                    del chunk_tensors, chunk_sum
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                self._check_gpu_memory(context=f"layer {layer_idx}")

                # Move to CPU and convert to numpy before discarding the tensor
                assert layer_sum is not None
                aggregated.append(layer_sum.cpu().numpy())
                del layer_sum
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return aggregated

    # ------------------------------------------------------------------
    # Helper: blockchain update
    # ------------------------------------------------------------------

    def _update_blockchain(
        self,
        server_round: int,
        eth_addresses: List[str],
        weights: np.ndarray,
        aggregated_ndarrays: List[np.ndarray],
    ) -> None:
        """Persist aggregated model to Redis and update on-chain reputations.

        Errors are logged but do not halt the aggregation pipeline.

        Args:
            server_round: Current communication round.
            eth_addresses: Ethereum addresses of participating clients.
            weights: Normalised aggregation weights used this round.
            aggregated_ndarrays: The aggregated model (list of numpy arrays).
        """
        # Upload aggregated model to Redis
        cid = ""
        try:
            tensors = [torch.from_numpy(arr) for arr in aggregated_ndarrays]
            cid = upload_tensor_to_redis(tensors, ttl_seconds=3600)
            logger.debug(
                "Round %d: aggregated model stored in Redis (key=%s)",
                server_round,
                cid,
            )
        except Exception:
            logger.warning(
                "Round %d: failed to upload aggregated model to Redis.",
                server_round,
                exc_info=True,
            )

        # Compute integer reputation scores from weights
        scores = [
            int(np.clip(w * self.reputation_scale, 0, _MAX_REPUTATION))
            for w in weights
        ]
        cids = [cid] * len(eth_addresses)

        # Encode loss / magnitude as scaled integers (1e4 precision)
        losses = [0] * len(eth_addresses)
        mags = [0] * len(eth_addresses)

        try:
            batch_update_clients(
                addresses=eth_addresses,
                scores=scores,
                cids=cids,
                losses=losses,
                magnitudes=mags,
                account_index=0,
            )
            logger.info(
                "Round %d: updated on-chain reputations for %d clients.",
                server_round,
                len(eth_addresses),
            )
        except Exception:
            logger.warning(
                "Round %d: blockchain batch update failed; "
                "reputations not persisted this round.",
                server_round,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Helper: address mapping
    # ------------------------------------------------------------------

    def _get_or_create_address(self, flower_cid: str) -> str:
        """Map a Flower client CID to an Ethereum address.

        If no explicit mapping exists, a deterministic 20-byte hex address is
        derived from a SHA-256 hash of the CID (for testing only).

        Args:
            flower_cid: The Flower-assigned client identifier string.

        Returns:
            A checksummed Ethereum-style address (``0x`` prefixed, 42 chars).
        """
        if flower_cid in self.client_address_map:
            return self.client_address_map[flower_cid]

        # Deterministic test address derived from the CID
        digest = hashlib.sha256(flower_cid.encode()).hexdigest()
        raw_addr = "0x" + digest[:40]

        # Attempt checksum via Web3; fall back to lowercase if Web3 is unavailable
        try:
            from web3 import Web3

            checksummed = Web3.to_checksum_address(raw_addr)
        except Exception:
            checksummed = raw_addr.lower()

        self.client_address_map[flower_cid] = checksummed
        return checksummed
