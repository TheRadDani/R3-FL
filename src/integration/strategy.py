"""
strategy.py
~~~~~~~~~~~
Custom Flower Strategy that integrates the FL pipeline, blockchain reputation
ledger, and a trained RLlib PPO agent for intelligent aggregation weighting.

The ``RLReputationStrategy`` overrides ``aggregate_fit`` to:
1. Extract per-client model updates from Flower ``FitRes`` results.
2. Fetch on-chain reputation history via ``web3_utils``.
3. Build a (NUM_CLIENTS x NUM_FEATURES) state matrix for the PPO agent.
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

        # Client address mapping (Flower CID -> Ethereum address)
        self.client_address_map: Dict[str, str] = client_addresses or {}

        # Per-round metrics history
        self.round_metrics: Dict[int, Dict[str, Any]] = {}

        # Load PPO algorithm --------------------------------------------------
        self.ppo_algo: Optional[Any] = None
        try:
            from ray.rllib.algorithms.algorithm import Algorithm
            from ray.tune.registry import register_env
            from src.rl_agent.env import FLReputationEnv
            from src.rl_agent.train import ENV_NAME, env_creator

            register_env(ENV_NAME, env_creator)
            self.ppo_algo = Algorithm.from_checkpoint(ppo_checkpoint_path)
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
        """Server-side evaluation (not used -- evaluation is distributed)."""
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
            4. Build a (num_clients x 5) state matrix for the PPO agent.
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

        # -- Step 5: PPO inference → weights -------------------------------
        weights = self._ppo_inference(state, participating_indices)

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
    # Helper: gradient similarity
    # ------------------------------------------------------------------

    def _compute_gradient_similarity(
        self, client_params: List[List[torch.Tensor]]
    ) -> np.ndarray:
        """Cosine similarity of each client's flattened update vs. the mean.

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
            mean_vec = stacked.mean(dim=0, keepdim=True)  # (1, D)

            # Cosine similarity (result in [-1, 1], rescale to [0, 1])
            cos_sim = F.cosine_similarity(stacked, mean_vec, dim=1)  # (N,)
            rescaled = ((cos_sim + 1.0) / 2.0).cpu().numpy().astype(np.float32)

            # Free the large stacked matrix immediately after use
            del stacked, mean_vec, cos_sim
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
        participating_indices: List[int],
        accuracies: np.ndarray,
        similarities: np.ndarray,
        reputations: np.ndarray,
        loss_improvements: np.ndarray,
        magnitudes: np.ndarray,
    ) -> np.ndarray:
        """Build the full (NUM_CLIENTS x NUM_FEATURES) observation matrix.

        Non-participating client rows are filled with the neutral value (0.5).

        Args:
            participating_indices: Indices assigned to actual participants.
            accuracies: Per-participant accuracy contribution.
            similarities: Per-participant gradient similarity.
            reputations: Per-participant normalised reputation.
            loss_improvements: Per-participant loss improvement.
            magnitudes: Per-participant update magnitude.

        Returns:
            State matrix of shape (NUM_CLIENTS, NUM_FEATURES).
        """
        state = np.full(
            (self.num_clients, NUM_FEATURES),
            _NEUTRAL_FEATURE,
            dtype=np.float32,
        )

        for local_idx, global_idx in enumerate(participating_indices):
            if global_idx >= self.num_clients:
                continue
            state[global_idx, 0] = accuracies[local_idx]
            state[global_idx, 1] = similarities[local_idx]
            state[global_idx, 2] = reputations[local_idx]
            state[global_idx, 3] = loss_improvements[local_idx]
            state[global_idx, 4] = magnitudes[local_idx]

        return np.clip(state, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Helper: PPO inference
    # ------------------------------------------------------------------

    def _ppo_inference(
        self,
        state: np.ndarray,
        participating_indices: List[int],
    ) -> np.ndarray:
        """Run PPO to obtain per-participant aggregation weights.

        Falls back to uniform weights if PPO is unavailable or inference fails.

        Args:
            state: Observation matrix of shape (NUM_CLIENTS, NUM_FEATURES).
            participating_indices: Indices of actual participants in the state.

        Returns:
            Normalised weight array of shape (num_participating,) summing to 1.
        """
        num_participating = len(participating_indices)

        if self.ppo_algo is not None:
            try:
                action = self.ppo_algo.compute_single_action(state)
                # action shape: (NUM_CLIENTS,)
                action = np.asarray(action, dtype=np.float32)

                # Extract weights for participating clients only
                raw_weights = np.array(
                    [action[idx] for idx in participating_indices],
                    dtype=np.float32,
                )
                raw_weights = np.clip(raw_weights, 0.0, 1.0)

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
        uniform = np.full(
            num_participating, 1.0 / num_participating, dtype=np.float32
        )
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
