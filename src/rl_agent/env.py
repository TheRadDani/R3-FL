"""Multi-Agent Gymnasium environment for RL-based Federated Learning reputation management.

Location: src/rl_agent/env.py
Summary: Defines FLReputationEnv, a MultiAgentEnv (Ray RLlib) where each FL client
is modeled as an independent MAPPO agent with parameter sharing. Each agent observes
its own 5-feature vector and outputs a single scalar aggregation weight. The global
reward is shared cooperatively across all agents.

Used by: src/rl_agent/train.py (training loop), src/rl_agent/config.py (RLlib config).

Architecture: MAPPO with parameter sharing
    - NUM_CLIENTS agents, each identified as "agent_0" .. "agent_{N-1}"
    - Observation per agent: Box(0, 1, shape=(NUM_FEATURES,)) — local feature vector
    - Action per agent: Box(0, 1, shape=(1,)) — scalar aggregation weight
    - Reward: cooperative (identical global reward broadcast to every agent)

Performance notes:
    - _state is a pre-allocated float32 buffer reused every step (no GC pressure)
    - Reward computation is fully vectorized with no Python loops
    - Static coefficients (log_num_clients, etc.) pre-computed at init time
    - Reputation update avoids temporary arrays via in-place numpy ops
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLIENTS: int = 100
"""Total number of Federated Learning clients (honest + malicious)."""

NUM_FEATURES: int = 9
"""Number of observable features per client (5 local + 4 global context)."""

FEATURE_NAMES: list[str] = [
    "accuracy_contribution",
    "gradient_similarity",
    "historical_reputation",
    "loss_improvement",
    "update_magnitude",
    "global_mean_accuracy",
    "global_mean_similarity",
    "global_mean_loss_improvement",
    "global_mean_magnitude",
]
"""Human-readable names for the 9 per-client features.

Index mapping:
    0 — accuracy_contribution      : How much this client's update improves the model
    1 — gradient_similarity        : Cosine similarity of client gradient to global gradient (mapped 0-1)
    2 — historical_reputation      : Exponential moving average of past reputation scores
    3 — loss_improvement           : Reduction in loss attributable to this client's update
    4 — update_magnitude           : L2 norm of the gradient update (high values may indicate poisoning)
    5 — global_mean_accuracy       : Mean accuracy_contribution across all clients (broadcast)
    6 — global_mean_similarity     : Mean gradient_similarity across all clients (broadcast)
    7 — global_mean_loss_improvement : Mean loss_improvement across all clients (broadcast)
    8 — global_mean_magnitude      : Mean update_magnitude across all clients (broadcast)
"""

# Feature indices for readability — local features
_ACC = 0
_SIM = 1
_REP = 2
_LOSS = 3
_MAG = 4

# Feature indices for readability — global context features
_GACC = 5
_GSIM = 6
_GLOSS = 7
_GMAG = 8

# Reputation update parameters
_REPUTATION_EMA_ALPHA: float = 0.3
"""EMA smoothing factor for historical reputation updates."""

# Pre-computed EMA complement — avoids per-step subtraction in hot path
_REPUTATION_EMA_BETA: float = 1.0 - _REPUTATION_EMA_ALPHA


class FLReputationEnv(MultiAgentEnv):
    """Multi-agent environment simulating an FL aggregation round (MAPPO).

    Each of the num_clients FL clients is modeled as an independent RL agent.
    All agents share the same policy network (parameter sharing). Each agent
    observes its own 9-feature behavioral vector and outputs a single scalar
    aggregation weight. The reward is cooperative: every agent receives the
    same global reward signal.

    Reward:
        R = alpha * weighted_accuracy - beta * attack_impact

    Args:
        alpha: Weight for the accuracy term in the reward (default 0.6).
        beta: Weight for the attack-impact penalty in the reward (default 0.4).
        malicious_fraction: Proportion of clients that are malicious (default 0.3).
        max_rounds: Maximum FL rounds per episode (default 200).
        num_clients: Number of FL clients in the federation (default NUM_CLIENTS=100).
            Can be overridden via ``env_config["num_clients"]`` in RLlib to run
            benchmarks or evaluations with a different client count.
        render_mode: Optional rendering mode; ``"human"`` prints a round summary.
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
        num_clients: int = NUM_CLIENTS,
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

        # Configurable client count — stored before any buffer allocation that
        # depends on it.  The module-level NUM_CLIENTS constant is preserved for
        # backward-compatible imports (e.g. strategy.py).
        self.num_clients: int = num_clients

        # Agent IDs — required by MultiAgentEnv
        self._agent_ids = {f"agent_{i}" for i in range(self.num_clients)}

        # Pre-computed ordered agent ID list — avoids f-string formatting on every
        # step() call.  Use this list (not the set above) wherever iteration order matters.
        self._agent_id_list: list[str] = [f"agent_{i}" for i in range(self.num_clients)]

        # Per-agent spaces — each agent sees its own feature vector and
        # outputs a single scalar weight. Explicit float32 dtype avoids
        # RLlib's silent float64→float32 downcast which burns a memcpy.
        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(NUM_FEATURES,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(1,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------ #
        # Pre-allocated buffers — allocated ONCE here, reused every step.
        # This is the primary GC-pressure reduction: no per-step np.zeros/ones.
        # ------------------------------------------------------------------ #

        # Main state matrix — written in-place by _generate_state()
        self._state: np.ndarray = np.zeros(
            (self.num_clients, NUM_FEATURES), dtype=np.float32
        )

        # Reusable weight normalization buffer — avoids allocation in _compute_reward
        # and _update_reputation (written in-place for EMA updates)
        self._p_buffer: np.ndarray = np.zeros(self.num_clients, dtype=np.float32)

        # Pre-allocated honest/malicious signal buffers — sized to max possible;
        # actual slice depends on malicious_fraction set each episode
        self._honest_signal: np.ndarray = np.zeros(self.num_clients, dtype=np.float32)
        self._malicious_signal: np.ndarray = np.zeros(self.num_clients, dtype=np.float32)

        # Internal state (set properly in reset())
        self._rng: np.random.Generator = np.random.default_rng()
        self._malicious_mask: np.ndarray = np.zeros(self.num_clients, dtype=bool)
        self._honest_mask: np.ndarray = np.ones(self.num_clients, dtype=bool)  # ~_malicious_mask
        self._round: int = 0
        self._prev_weights: np.ndarray = np.full(
            self.num_clients, 1.0 / self.num_clients, dtype=np.float32
        )
        self._cumulative_reward: float = 0.0

        # ------------------------------------------------------------------ #
        # Pre-computed static reward-shaping coefficients
        # Compute once at __init__ — avoids repeated operations in hot path
        # ------------------------------------------------------------------ #

        # Normalization penalty coefficient — penalises raw action sums far from 1.0
        # to resolve the scale ambiguity introduced by weight_sum normalisation.
        self._norm_penalty_lambda: np.float32 = np.float32(0.01)

        # Epsilon for weight-sum guard (avoids div-by-zero when all weights≈0)
        self._wsum_eps: np.float32 = np.float32(1e-8)

    # --------------------------------------------------------------------- #
    # MultiAgentEnv API
    # --------------------------------------------------------------------- #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Currently unused; reserved for future configuration.

        Returns:
            A tuple of (obs_dict, info_dict) where each maps agent IDs to
            per-agent observations and info dicts respectively.
        """
        # Manage RNG directly (MultiAgentEnv doesn't use gymnasium's seed protocol)
        self._rng = np.random.default_rng(seed)

        # Randomly designate malicious clients for this episode
        num_malicious = int(self.num_clients * self.malicious_fraction)
        malicious_indices = self._rng.choice(
            self.num_clients, size=num_malicious, replace=False
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
        self._prev_weights.fill(1.0 / self.num_clients)

        # Generate the first observation (writes into self._state in-place)
        self._generate_state(initial=True)

        # Build per-agent observation and info dicts using the pre-built id list
        # to avoid repeated f-string allocation in the hot path.
        obs_dict = {
            aid: self._state[i].copy()
            for i, aid in enumerate(self._agent_id_list)
        }
        info_dict = {
            aid: {"round": 0, "num_malicious": num_malicious}
            for aid in self._agent_id_list
        }
        return obs_dict, info_dict

    def step(
        self, action_dict: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Execute one FL aggregation round.

        Args:
            action_dict: Mapping of agent IDs to shape-(1,) arrays with per-agent
                         aggregation weights in [0, 1].

        Returns:
            A 5-tuple (obs_dict, rewards_dict, terminateds, truncateds, infos)
            where each is keyed by agent ID. terminateds and truncateds also
            include an "__all__" key.
        """
        # Parse per-agent actions into a single weight vector using the pre-built
        # id list to avoid repeated f-string formatting on every step call.
        weights = np.array(
            [float(action_dict[aid][0]) for aid in self._agent_id_list],
            dtype=np.float32,
        )

        # Clip action to valid range — astype with copy=False avoids a copy
        # when weights is already float32
        weights = np.clip(weights, 0.0, 1.0).astype(np.float32, copy=False)

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

        # Build per-agent return dicts using the pre-built id list to avoid
        # repeated f-string formatting — at 100 agents per step this matters.
        obs_dict = {
            aid: self._state[i].copy()
            for i, aid in enumerate(self._agent_id_list)
        }

        # Cooperative reward: every agent receives the same global reward
        rewards_dict = {aid: reward for aid in self._agent_id_list}

        terminateds: dict[str, bool] = {aid: terminated for aid in self._agent_id_list}
        terminateds["__all__"] = terminated

        truncateds: dict[str, bool] = {aid: False for aid in self._agent_id_list}
        truncateds["__all__"] = False

        # Build the shared info body once — then reference it per agent.
        # Each agent gets its own dict so callers can mutate without cross-contamination.
        _info_body = {
            "round": self._round,
            "cumulative_reward": self._cumulative_reward,
            **reward_info,
        }
        infos = {aid: dict(_info_body) for aid in self._agent_id_list}

        if self.render_mode == "human":
            self.render()

        return obs_dict, rewards_dict, terminateds, truncateds, infos

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
            f"  Malicious clients : {int(self._malicious_mask.sum())} / {self.num_clients}"
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

        Feature distributions decay exponentially with round progress t ∈ [0, 1]
        so the training distribution covers the full FL convergence trajectory
        (aggressive decay for _LOSS/_MAG; gentle decay for _ACC; _SIM unchanged).

        Args:
            initial: If True, set reputation column to 0.5 for all clients.
                     If False, carry forward the updated reputation column.
        """
        honest_mask = self._honest_mask
        malicious_mask = self._malicious_mask
        n_honest = int(honest_mask.sum())
        n_malicious = int(malicious_mask.sum())

        # Convergence progress t ∈ [0, 1] over the episode.
        # At t=0 (round 0) decay=1 → original distributions.
        # At t=1 (final round) decay→e^{-2}≈0.135, decay_soft→e^{-0.8}≈0.449.
        t          = np.float32(self._round / self.max_rounds)
        decay      = np.float32(np.exp(-2.0 * t))   # aggressive: _LOSS, _MAG
        decay_soft = np.float32(np.exp(-0.8 * t))   # gentle:     _ACC

        # Save reputation column before zeroing the whole buffer.
        # _update_reputation() writes directly into self._state[:, _REP]
        # and that data must survive the fill(0.0) below.
        if not initial:
            # Use _p_buffer as a temp slot to save reputation
            # (_p_buffer is only used in _compute_reward, not here)
            np.copyto(self._p_buffer, self._state[:, _REP])

        # Zero the entire state buffer in-place — one fast memset
        self._state.fill(0.0)

        # --- Honest clients — distributions decay as FL model converges ---
        self._state[honest_mask, _ACC]  = self._rng.normal(
            np.float32(0.7) * decay_soft,
            max(np.float32(0.04), np.float32(0.10) * decay_soft),
            size=n_honest,
        )
        # _SIM unchanged — gradient similarity stays high throughout convergence
        self._state[honest_mask, _SIM]  = self._rng.normal(0.8, 0.10, size=n_honest)
        self._state[honest_mask, _LOSS] = self._rng.normal(
            np.float32(0.6) * decay,
            max(np.float32(0.04), np.float32(0.15) * decay),
            size=n_honest,
        )
        self._state[honest_mask, _MAG]  = self._rng.normal(
            np.float32(0.5) * decay,
            max(np.float32(0.03), np.float32(0.10) * decay),
            size=n_honest,
        )

        # --- Malicious clients — _LOSS/_ACC decay relatively; _MAG stays elevated ---
        # (attacker injects large poisoned gradients even at convergence,
        #  keeping _MAG distinguishable from the decayed honest signal)
        self._state[malicious_mask, _ACC]  = self._rng.normal(
            np.float32(0.3) * decay_soft,
            max(np.float32(0.05), np.float32(0.15) * decay_soft),
            size=n_malicious,
        )
        self._state[malicious_mask, _SIM]  = self._rng.normal(0.2, 0.15, size=n_malicious)
        self._state[malicious_mask, _LOSS] = self._rng.normal(
            np.float32(0.2) * decay,
            max(np.float32(0.04), np.float32(0.10) * decay),
            size=n_malicious,
        )
        # Attacker _MAG: linear mild decay (0.90 → 0.75) so detection signal persists
        self._state[malicious_mask, _MAG]  = self._rng.normal(
            np.float32(0.9) - np.float32(0.15) * t,
            np.float32(0.20),
            size=n_malicious,
        )

        # --- Reputation column ---
        if initial:
            self._state[:, _REP] = 0.5
        else:
            # Restore reputation that was updated in-place by _update_reputation()
            # (was saved into _p_buffer above)
            np.copyto(self._state[:, _REP], self._p_buffer)

        # Clip to valid observation range in-place — no intermediate array
        np.clip(self._state, 0.0, 1.0, out=self._state)

        # --- Global context features (broadcast means of local features) ---
        # Computed AFTER clipping so the means reflect the valid [0, 1] range.
        mean_acc = self._state[:, _ACC].mean()
        mean_sim = self._state[:, _SIM].mean()
        mean_loss = self._state[:, _LOSS].mean()
        mean_mag = self._state[:, _MAG].mean()

        self._state[:, _GACC] = mean_acc
        self._state[:, _GSIM] = mean_sim
        self._state[:, _GLOSS] = mean_loss
        self._state[:, _GMAG] = mean_mag

        # Safety clip for global columns (values derive from clipped locals,
        # but guard against floating-point edge cases)
        np.clip(self._state[:, _GACC:_GMAG + 1], 0.0, 1.0,
                out=self._state[:, _GACC:_GMAG + 1])

    def _compute_reward(
        self, weights: np.ndarray
    ) -> tuple[float, dict[str, Any]]:
        """Compute the scalar reward for a given weight vector.

        Reward = alpha * weighted_accuracy - beta * attack_impact - lambda * |weight_sum - 1|

        All operations are vectorized numpy; no Python loops in the hot path.
        Pre-allocated buffers (_p_buffer) avoid per-call allocation.

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
                "norm_penalty": float(self._norm_penalty_lambda),
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

        # Normalization penalty — penalises raw weight sums far from 1.0.
        # Without this, any weight vector proportional to the optimal one
        # yields the same normalised result, making the value function
        # unable to distinguish scale, which slows policy convergence.
        norm_penalty = float(self._norm_penalty_lambda * abs(weight_sum - 1.0))

        reward = float(
            self.alpha * weighted_accuracy
            - self.beta * attack_impact
            - norm_penalty
        )

        info: dict[str, Any] = {
            "weighted_accuracy": weighted_accuracy,
            "attack_impact": float(attack_impact),
            "norm_penalty": norm_penalty,
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
