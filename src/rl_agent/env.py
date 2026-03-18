"""Custom Gymnasium environment for RL-based Federated Learning reputation management.

This environment simulates a Federated Learning system where an RL agent (PPO)
learns to assign aggregation weights to clients based on behavioral features.
The agent must learn to upweight honest clients and downweight malicious ones
to maintain global model accuracy while mitigating poisoning attacks.

Environment Spaces:
    Observation: Box(0, 1, shape=(NUM_CLIENTS, NUM_FEATURES)) — per-client feature matrix
    Action: Box(0, 1, shape=(NUM_CLIENTS,)) — aggregation weight per client

Performance notes:
    - _state is a pre-allocated float32 buffer reused every step (no GC pressure)
    - _obs_buffer is a pre-allocated flat copy buffer returned from step()/reset()
    - Reward computation is fully vectorized with no Python loops
    - Static coefficients (log_num_clients, etc.) pre-computed at init time
    - Reputation update avoids temporary arrays via in-place numpy ops
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLIENTS: int = 100
"""Total number of Federated Learning clients (honest + malicious)."""

NUM_FEATURES: int = 5
"""Number of observable features per client."""

FEATURE_NAMES: list[str] = [
    "accuracy_contribution",
    "gradient_similarity",
    "historical_reputation",
    "loss_improvement",
    "update_magnitude",
]
"""Human-readable names for the 5 per-client features.

Index mapping:
    0 — accuracy_contribution : How much this client's update improves the model
    1 — gradient_similarity   : Cosine similarity of client gradient to global gradient (mapped 0-1)
    2 — historical_reputation : Exponential moving average of past reputation scores
    3 — loss_improvement      : Reduction in loss attributable to this client's update
    4 — update_magnitude      : L2 norm of the gradient update (high values may indicate poisoning)
"""

# Feature indices for readability
_ACC = 0
_SIM = 1
_REP = 2
_LOSS = 3
_MAG = 4

# Reputation update parameters
_REPUTATION_EMA_ALPHA: float = 0.3
"""EMA smoothing factor for historical reputation updates."""

# Pre-computed EMA complement — avoids per-step subtraction in hot path
_REPUTATION_EMA_BETA: float = 1.0 - _REPUTATION_EMA_ALPHA


class FLReputationEnv(gym.Env):
    """Gymnasium environment simulating an FL aggregation round.

    The RL agent observes per-client behavioral metrics and outputs aggregation
    weights.  The reward encourages high weighted accuracy while penalising
    weight allocated to malicious clients.

    Reward:
        R = alpha * weighted_accuracy - beta * attack_impact

    Args:
        alpha: Weight for the accuracy term in the reward (default 0.6).
        beta: Weight for the attack-impact penalty in the reward (default 0.4).
        malicious_fraction: Proportion of clients that are malicious (default 0.3).
        max_rounds: Maximum FL rounds per episode (default 200).
    """

    metadata: dict[str, Any] = {"render_modes": ["human"], "render_fps": 1}

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        malicious_fraction: float = 0.3,
        max_rounds: int = 200,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Reward hyper-parameters
        self.alpha = np.float32(alpha)   # cast once at init — avoids silent float64 promotion
        self.beta = np.float32(beta)

        # Episode configuration
        self.malicious_fraction = malicious_fraction
        self.max_rounds = max_rounds
        self.render_mode = render_mode

        # Gymnasium spaces — explicit float32 dtype avoids RLlib's silent
        # float64→float32 downcast which burns a memcpy every step
        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(NUM_CLIENTS, NUM_FEATURES),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(NUM_CLIENTS,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------ #
        # Pre-allocated buffers — allocated ONCE here, reused every step.
        # This is the primary GC-pressure reduction: no per-step np.zeros/ones.
        # ------------------------------------------------------------------ #

        # Main state matrix — written in-place by _generate_state()
        self._state: np.ndarray = np.zeros(
            (NUM_CLIENTS, NUM_FEATURES), dtype=np.float32
        )

        # Observation output buffer — step()/reset() copy into this and return
        # a view; RLlib makes its own copy so we can safely return the same buffer
        self._obs_buffer: np.ndarray = np.zeros(
            (NUM_CLIENTS, NUM_FEATURES), dtype=np.float32
        )

        # Reusable weight normalization buffer — avoids allocation in _compute_reward
        self._p_buffer: np.ndarray = np.zeros(NUM_CLIENTS, dtype=np.float32)

        # Reusable log-probability buffer for entropy computation
        self._logp_buffer: np.ndarray = np.zeros(NUM_CLIENTS, dtype=np.float32)

        # Pre-allocated honest/malicious signal buffers — sized to max possible;
        # actual slice depends on malicious_fraction set each episode
        self._honest_signal: np.ndarray = np.zeros(NUM_CLIENTS, dtype=np.float32)
        self._malicious_signal: np.ndarray = np.zeros(NUM_CLIENTS, dtype=np.float32)

        # Internal state (set properly in reset())
        self._rng: np.random.Generator = np.random.default_rng()
        self._malicious_mask: np.ndarray = np.zeros(NUM_CLIENTS, dtype=bool)
        self._honest_mask: np.ndarray = np.ones(NUM_CLIENTS, dtype=bool)  # ~_malicious_mask
        self._round: int = 0
        self._prev_weights: np.ndarray = np.full(
            NUM_CLIENTS, 1.0 / NUM_CLIENTS, dtype=np.float32
        )
        self._cumulative_reward: float = 0.0

        # ------------------------------------------------------------------ #
        # Pre-computed static reward-shaping coefficients
        # Compute once at __init__ — avoids repeated log/division in hot path
        # ------------------------------------------------------------------ #

        # log(NUM_CLIENTS) — denominator for entropy normalization
        # Stored as float32 to match all other reward computations
        self._log_num_clients: np.float32 = np.float32(np.log(NUM_CLIENTS))

        # Small entropy penalty coefficient — gentle guidance, not domination
        self._entropy_coeff: np.float32 = np.float32(0.05)

        # Epsilon for log(0) guard in entropy computation
        self._log_eps: np.float32 = np.float32(1e-10)

        # Epsilon for weight-sum guard (avoids div-by-zero when all weights≈0)
        self._wsum_eps: np.float32 = np.float32(1e-8)

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Currently unused; reserved for future configuration.

        Returns:
            A tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)

        # Create a fresh RNG from the Gymnasium-managed seed
        self._rng = np.random.default_rng(seed)

        # Randomly designate malicious clients for this episode
        num_malicious = int(NUM_CLIENTS * self.malicious_fraction)
        malicious_indices = self._rng.choice(
            NUM_CLIENTS, size=num_malicious, replace=False
        )
        # Re-use existing mask buffers — fill(False) then set indices True
        self._malicious_mask[:] = False
        self._malicious_mask[malicious_indices] = True
        # Keep honest_mask in sync (avoids repeated ~mask in step hot path)
        np.logical_not(self._malicious_mask, out=self._honest_mask)

        # Reset episode counters
        self._round = 0
        self._cumulative_reward = 0.0
        # In-place fill avoids allocation of new full() array
        self._prev_weights.fill(1.0 / NUM_CLIENTS)

        # Generate the first observation (writes into self._state in-place)
        self._generate_state(initial=True)

        # Copy state into the obs buffer and return — RLlib copies again internally
        np.copyto(self._obs_buffer, self._state)

        info: dict[str, Any] = {
            "round": self._round,
            "num_malicious": int(num_malicious),
        }
        return self._obs_buffer.copy(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one FL aggregation round.

        Args:
            action: Array of shape (NUM_CLIENTS,) with per-client weights in [0, 1].

        Returns:
            A 5-tuple (observation, reward, terminated, truncated, info).
        """
        # Clip action to valid range in-place where possible.
        # astype with copy=False avoids a copy when action is already float32.
        weights = np.clip(action, 0.0, 1.0).astype(np.float32, copy=False)

        # Compute reward for this round (fully vectorized, no Python loops)
        reward, reward_info = self._compute_reward(weights)
        self._cumulative_reward += reward

        # Update historical reputation based on the weights the agent assigned
        self._update_reputation(weights)

        # Advance to next round — in-place copy avoids re-allocation
        self._round += 1
        np.copyto(self._prev_weights, weights)

        # Generate next observation (writes into self._state in-place,
        # carries forward the updated reputation column)
        self._generate_state(initial=False)

        terminated = self._round >= self.max_rounds
        truncated = False

        # Copy state into obs buffer
        np.copyto(self._obs_buffer, self._state)

        info: dict[str, Any] = {
            "round": self._round,
            "cumulative_reward": self._cumulative_reward,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return self._obs_buffer.copy(), reward, terminated, truncated, info

    def render(self) -> None:
        """Print a human-readable summary of the current round."""
        if self._round == 0:
            print("[FLReputationEnv] Episode not yet started.")
            return

        weights = self._prev_weights
        top_5 = np.argsort(weights)[-5:][::-1]
        bottom_5 = np.argsort(weights)[:5]

        print(f"\n--- Round {self._round}/{self.max_rounds} ---")
        print(f"  Cumulative reward : {self._cumulative_reward:.4f}")
        print(f"  Top-5 weighted    : {top_5.tolist()}")
        print(f"  Bottom-5 weighted : {bottom_5.tolist()}")
        print(
            f"  Malicious clients : {int(self._malicious_mask.sum())} / {NUM_CLIENTS}"
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _generate_state(self, initial: bool = False) -> None:
        """Fill self._state in-place with (NUM_CLIENTS, NUM_FEATURES) observations.

        Honest clients exhibit high accuracy/similarity and moderate magnitude,
        while malicious clients show low accuracy/similarity and high magnitude.
        Writes directly into the pre-allocated self._state buffer — no return value,
        no allocation.

        Args:
            initial: If True, set reputation column to 0.5 for all clients.
                     If False, carry forward the updated reputation column.
        """
        honest_mask = self._honest_mask
        malicious_mask = self._malicious_mask
        n_honest = int(honest_mask.sum())
        n_malicious = int(malicious_mask.sum())

        # Save reputation column before zeroing the whole buffer.
        # _update_reputation() writes directly into self._state[:, _REP]
        # and that data must survive the fill(0.0) below.
        if not initial:
            # Use _p_buffer as a temp slot to save reputation
            # (_p_buffer is only used in _compute_reward, not here)
            np.copyto(self._p_buffer, self._state[:, _REP])

        # Zero the entire state buffer in-place — one fast memset
        self._state.fill(0.0)

        # --- Honest clients (vectorized column writes) ---
        self._state[honest_mask, _ACC] = self._rng.normal(0.7, 0.10, size=n_honest)
        self._state[honest_mask, _SIM] = self._rng.normal(0.8, 0.10, size=n_honest)
        self._state[honest_mask, _LOSS] = self._rng.normal(0.6, 0.15, size=n_honest)
        self._state[honest_mask, _MAG] = self._rng.normal(0.5, 0.10, size=n_honest)

        # --- Malicious clients (vectorized column writes) ---
        self._state[malicious_mask, _ACC] = self._rng.normal(0.3, 0.15, size=n_malicious)
        self._state[malicious_mask, _SIM] = self._rng.normal(0.2, 0.15, size=n_malicious)
        self._state[malicious_mask, _LOSS] = self._rng.normal(0.2, 0.10, size=n_malicious)
        self._state[malicious_mask, _MAG] = self._rng.normal(0.9, 0.20, size=n_malicious)

        # --- Reputation column ---
        if initial:
            self._state[:, _REP] = 0.5
        else:
            # Restore reputation that was updated in-place by _update_reputation()
            # (was saved into _p_buffer above)
            np.copyto(self._state[:, _REP], self._p_buffer)

        # Clip to valid observation range in-place — no intermediate array
        np.clip(self._state, 0.0, 1.0, out=self._state)

    def _compute_reward(
        self, weights: np.ndarray
    ) -> tuple[float, dict[str, Any]]:
        """Compute the scalar reward for a given weight vector.

        Reward = alpha * weighted_accuracy - beta * attack_impact - entropy_penalty

        All operations are vectorized numpy; no Python loops in the hot path.
        Pre-allocated buffers (_p_buffer, _logp_buffer) avoid per-call allocation.

        A small entropy-based penalty is added when weights are nearly uniform,
        which would indicate the agent is not discriminating between clients.

        Args:
            weights: Per-client aggregation weights in [0, 1].

        Returns:
            (reward, info_dict) where info_dict contains reward component breakdown.
        """
        weight_sum = float(np.sum(weights))

        # Guard against all-zero weights (agent refuses to aggregate anything)
        if weight_sum < self._wsum_eps:
            return -1.0, {
                "weighted_accuracy": 0.0,
                "attack_impact": 1.0,
                "entropy_penalty": 0.0,
                "reward": -1.0,
            }

        inv_weight_sum = np.float32(1.0 / weight_sum)  # pre-compute reciprocal

        # Weighted accuracy: dot product of weights and accuracy contributions,
        # normalized by total weight — measures quality of aggregated model
        acc_contributions = self._state[:, _ACC]  # view, no copy
        weighted_accuracy = float(np.dot(weights, acc_contributions) * inv_weight_sum)

        # Attack impact: fraction of total weight allocated to malicious clients
        # np.dot with boolean mask is equivalent to np.sum(weights[mask])
        malicious_weight = float(np.sum(weights[self._malicious_mask]))
        attack_impact = malicious_weight * inv_weight_sum

        # Entropy penalty — reuse pre-allocated buffers to avoid per-call alloc
        # p = weights / weight_sum  (written into _p_buffer in-place)
        np.multiply(weights, inv_weight_sum, out=self._p_buffer)

        # log(p + epsilon)  (written into _logp_buffer in-place)
        np.add(self._p_buffer, self._log_eps, out=self._logp_buffer)
        np.log(self._logp_buffer, out=self._logp_buffer)

        # entropy = -sum(p * log_p)  (dot product, scalar result)
        entropy = float(-np.dot(self._p_buffer, self._logp_buffer))

        # Normalise entropy by log(NUM_CLIENTS) — pre-computed at init time
        normalised_entropy = (
            entropy / float(self._log_num_clients)
            if self._log_num_clients > 0
            else 0.0
        )

        # Small entropy penalty — gentle guidance toward weight discrimination
        entropy_penalty = float(self._entropy_coeff * normalised_entropy)

        reward = float(
            self.alpha * weighted_accuracy
            - self.beta * attack_impact
            - entropy_penalty
        )

        info: dict[str, Any] = {
            "weighted_accuracy": weighted_accuracy,
            "attack_impact": float(attack_impact),
            "entropy_penalty": entropy_penalty,
            "reward": reward,
        }
        return reward, info

    def _update_reputation(self, weights: np.ndarray) -> None:
        """Update the historical_reputation column via exponential moving average.

        Honest clients receiving high weights see their reputation increase;
        malicious clients receiving high weights see theirs decrease (simulating
        that over time the system should discover them).

        All EMA operations are in-place to avoid temporary array allocation.
        The pre-computed _REPUTATION_EMA_BETA = 1 - alpha avoids per-call subtraction.

        Args:
            weights: The aggregation weights the agent just assigned.
        """
        weight_sum = float(np.sum(weights))
        if weight_sum < self._wsum_eps:
            return

        inv_weight_sum = np.float32(1.0 / weight_sum)

        # Normalize weights once — result stored in _p_buffer (reused across methods
        # but _compute_reward is always called before this, so buffer is free)
        np.multiply(weights, inv_weight_sum, out=self._p_buffer)
        normalised_w = self._p_buffer  # alias for readability

        honest_mask = self._honest_mask
        malicious_mask = self._malicious_mask

        # For honest clients: higher normalized weight → reputation increases
        # EMA: rep = (1-α)*rep + α*signal  (in-place, no temp array)
        # In-place: rep *= (1-α)  then  rep += α * signal
        self._state[honest_mask, _REP] *= _REPUTATION_EMA_BETA
        self._state[honest_mask, _REP] += _REPUTATION_EMA_ALPHA * normalised_w[honest_mask]

        # For malicious clients: higher weight → reputation drops
        # Signal is 1 - normalised_w (attacker getting weight is bad)
        self._state[malicious_mask, _REP] *= _REPUTATION_EMA_BETA
        # Compute (1.0 - normalised_w[malicious]) into a slice of honest_signal buffer
        mal_slice = normalised_w[malicious_mask]
        self._state[malicious_mask, _REP] += _REPUTATION_EMA_ALPHA * (1.0 - mal_slice)

        # Cast reputation column back to float32 after in-place mixed arithmetic
        self._state[:, _REP] = self._state[:, _REP].astype(np.float32)

        # Keep in valid range — in-place clip, no allocation
        np.clip(self._state[:, _REP], 0.0, 1.0, out=self._state[:, _REP])
