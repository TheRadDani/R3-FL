"""Custom Gymnasium environment for RL-based Federated Learning reputation management.

This environment simulates a Federated Learning system where an RL agent (PPO)
learns to assign aggregation weights to clients based on behavioral features.
The agent must learn to upweight honest clients and downweight malicious ones
to maintain global model accuracy while mitigating poisoning attacks.

Environment Spaces:
    Observation: Box(0, 1, shape=(NUM_CLIENTS, NUM_FEATURES)) — per-client feature matrix
    Action: Box(0, 1, shape=(NUM_CLIENTS,)) — aggregation weight per client
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
        self.alpha = alpha
        self.beta = beta

        # Episode configuration
        self.malicious_fraction = malicious_fraction
        self.max_rounds = max_rounds
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_CLIENTS, NUM_FEATURES),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_CLIENTS,),
            dtype=np.float32,
        )

        # Internal state (set properly in reset())
        self._rng: np.random.Generator = np.random.default_rng()
        self._malicious_mask: np.ndarray = np.zeros(NUM_CLIENTS, dtype=bool)
        self._state: np.ndarray = np.zeros(
            (NUM_CLIENTS, NUM_FEATURES), dtype=np.float32
        )
        self._round: int = 0
        self._prev_weights: np.ndarray = np.full(
            NUM_CLIENTS, 1.0 / NUM_CLIENTS, dtype=np.float32
        )
        self._cumulative_reward: float = 0.0

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
        self._malicious_mask = np.zeros(NUM_CLIENTS, dtype=bool)
        self._malicious_mask[malicious_indices] = True

        # Reset episode counters
        self._round = 0
        self._cumulative_reward = 0.0
        self._prev_weights = np.full(
            NUM_CLIENTS, 1.0 / NUM_CLIENTS, dtype=np.float32
        )

        # Generate the first observation
        self._state = self._generate_state(initial=True)

        info: dict[str, Any] = {
            "round": self._round,
            "num_malicious": int(num_malicious),
        }
        return self._state.copy(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one FL aggregation round.

        Args:
            action: Array of shape (NUM_CLIENTS,) with per-client weights in [0, 1].

        Returns:
            A 5-tuple (observation, reward, terminated, truncated, info).
        """
        # Clip action to valid range (defensive — handles slight numerical overflow)
        weights = np.clip(action, 0.0, 1.0).astype(np.float32)

        # Compute reward for this round
        reward, reward_info = self._compute_reward(weights)
        self._cumulative_reward += reward

        # Update historical reputation based on the weights the agent assigned
        self._update_reputation(weights)

        # Advance to next round
        self._round += 1
        self._prev_weights = weights.copy()

        # Generate next observation (blended with updated reputations)
        self._state = self._generate_state(initial=False)

        terminated = self._round >= self.max_rounds
        truncated = False

        info: dict[str, Any] = {
            "round": self._round,
            "cumulative_reward": self._cumulative_reward,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return self._state.copy(), reward, terminated, truncated, info

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

    def _generate_state(self, initial: bool = False) -> np.ndarray:
        """Generate a (NUM_CLIENTS, NUM_FEATURES) observation matrix.

        Honest clients exhibit high accuracy/similarity and moderate magnitude,
        while malicious clients show low accuracy/similarity and high magnitude.

        Args:
            initial: If True, reset reputations to 0.5 for all clients.
                     If False, carry forward the updated reputation column.

        Returns:
            State matrix clipped to [0, 1].
        """
        state = np.zeros((NUM_CLIENTS, NUM_FEATURES), dtype=np.float32)

        honest_mask = ~self._malicious_mask
        n_honest = int(honest_mask.sum())
        n_malicious = int(self._malicious_mask.sum())

        # --- Honest clients ---
        state[honest_mask, _ACC] = self._rng.normal(0.7, 0.10, size=n_honest)
        state[honest_mask, _SIM] = self._rng.normal(0.8, 0.10, size=n_honest)
        state[honest_mask, _LOSS] = self._rng.normal(0.6, 0.15, size=n_honest)
        state[honest_mask, _MAG] = self._rng.normal(0.5, 0.10, size=n_honest)

        # --- Malicious clients ---
        state[self._malicious_mask, _ACC] = self._rng.normal(
            0.3, 0.15, size=n_malicious
        )
        state[self._malicious_mask, _SIM] = self._rng.normal(
            0.2, 0.15, size=n_malicious
        )
        state[self._malicious_mask, _LOSS] = self._rng.normal(
            0.2, 0.10, size=n_malicious
        )
        state[self._malicious_mask, _MAG] = self._rng.normal(
            0.9, 0.20, size=n_malicious
        )

        # --- Reputation column ---
        if initial:
            state[:, _REP] = 0.5
        else:
            # Carry forward the existing reputation (already updated in step)
            state[:, _REP] = self._state[:, _REP]

        # Clip to valid observation range
        np.clip(state, 0.0, 1.0, out=state)

        return state

    def _compute_reward(
        self, weights: np.ndarray
    ) -> tuple[float, dict[str, Any]]:
        """Compute the scalar reward for a given weight vector.

        Reward = alpha * weighted_accuracy - beta * attack_impact

        A small entropy-based penalty is added when weights are nearly uniform,
        which would indicate the agent is not discriminating between clients.

        Args:
            weights: Per-client aggregation weights in [0, 1].

        Returns:
            (reward, info_dict) where info_dict contains reward component breakdown.
        """
        weight_sum = float(np.sum(weights))

        # Guard against all-zero weights (agent refuses to aggregate anything)
        if weight_sum < 1e-8:
            return -1.0, {
                "weighted_accuracy": 0.0,
                "attack_impact": 1.0,
                "entropy_penalty": 0.0,
                "reward": -1.0,
            }

        # Weighted accuracy: how good the aggregated model would be
        acc_contributions = self._state[:, _ACC]
        weighted_accuracy = float(
            np.sum(weights * acc_contributions) / weight_sum
        )

        # Attack impact: fraction of total weight allocated to malicious clients
        malicious_weight = float(np.sum(weights[self._malicious_mask]))
        attack_impact = malicious_weight / weight_sum

        # Entropy penalty: discourage near-uniform weighting (agent should
        # differentiate).  Normalise weights to a distribution and compute
        # entropy relative to maximum entropy (uniform distribution).
        p = weights / weight_sum
        # Avoid log(0) with a small epsilon
        log_p = np.log(p + 1e-10)
        entropy = -float(np.sum(p * log_p))
        max_entropy = float(np.log(NUM_CLIENTS))
        # Penalty is higher when entropy is close to max (uniform)
        normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        # Small penalty coefficient — we want gentle guidance, not domination
        entropy_penalty = 0.05 * normalised_entropy

        reward = (
            self.alpha * weighted_accuracy
            - self.beta * attack_impact
            - entropy_penalty
        )

        info: dict[str, Any] = {
            "weighted_accuracy": weighted_accuracy,
            "attack_impact": attack_impact,
            "entropy_penalty": entropy_penalty,
            "reward": reward,
        }
        return float(reward), info

    def _update_reputation(self, weights: np.ndarray) -> None:
        """Update the historical_reputation column via exponential moving average.

        Honest clients receiving high weights see their reputation increase;
        malicious clients receiving high weights see theirs decrease (simulating
        that over time the system should discover them).

        The reputation signal is derived from the *normalised* weight the agent
        assigned: honest clients with high weight → positive signal, malicious
        clients with high weight → negative signal (because giving weight to
        an attacker is bad, so their "true" reputation should drop).

        Args:
            weights: The aggregation weights the agent just assigned.
        """
        weight_sum = float(np.sum(weights))
        if weight_sum < 1e-8:
            return

        normalised_w = weights / weight_sum

        # For honest clients: higher weight → reputation goes up
        honest_signal = normalised_w[~self._malicious_mask]
        # For malicious clients: higher weight → reputation goes *down*
        # (Simulates the ground-truth feedback that their updates harmed the model)
        malicious_signal = 1.0 - normalised_w[self._malicious_mask]

        alpha = _REPUTATION_EMA_ALPHA

        self._state[~self._malicious_mask, _REP] = (
            (1 - alpha) * self._state[~self._malicious_mask, _REP]
            + alpha * honest_signal
        ).astype(np.float32)

        self._state[self._malicious_mask, _REP] = (
            (1 - alpha) * self._state[self._malicious_mask, _REP]
            + alpha * malicious_signal
        ).astype(np.float32)

        # Keep in valid range
        np.clip(self._state[:, _REP], 0.0, 1.0, out=self._state[:, _REP])
