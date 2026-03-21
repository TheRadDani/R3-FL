"""
Comprehensive pytest tests for the RL agent environment (src/rl_agent/env.py).

Covers:
  - Observation and action space definitions (shape, bounds, dtype)
  - Reset API compliance and reproducibility (MultiAgentEnv dict-based API)
  - Step API compliance (5-dict-tuple, termination, clipping)
  - Reward formula verification (weighted similarity, malicious weight penalty,
    low_sim_penalty, improvement bonus, norm penalty)
  - State generation (honest vs malicious feature distributions)
  - Reputation EMA updates (honest up, malicious down)
  - Episode lifecycle (termination at randomized max_rounds)
  - Constructor configuration (alpha, beta, delta, eta, malicious_fraction,
    max_rounds, min_rounds)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rl_agent.env import (
    NUM_CLIENTS,
    NUM_FEATURES,
    FEATURE_NAMES,
    FLReputationEnv,
    _REPUTATION_EMA_ALPHA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action_dict(env: FLReputationEnv, value: float = 0.5) -> dict[str, np.ndarray]:
    """Build an action dict mapping each agent ID to a scalar action."""
    return {
        f"agent_{i}": np.array([value], dtype=np.float32)
        for i in range(env.num_clients)
    }


def _first_agent(d: dict) -> str:
    """Return the first non-special agent key from a dict."""
    for k in d:
        if k != "__all__":
            return k
    raise KeyError("No agent key found")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> FLReputationEnv:
    """Default environment with standard parameters.

    Uses min_rounds=max_rounds to get deterministic episode length.
    """
    e = FLReputationEnv(
        alpha=0.5, beta=0.3, malicious_fraction=0.3,
        max_rounds=200, min_rounds=200,
    )
    e.reset(seed=42)
    return e


@pytest.fixture
def short_env() -> FLReputationEnv:
    """Short-episode environment for fast lifecycle tests.

    Uses min_rounds=max_rounds=10 for deterministic episode length.
    """
    e = FLReputationEnv(
        alpha=0.5, beta=0.3, malicious_fraction=0.3,
        max_rounds=10, min_rounds=10,
    )
    e.reset(seed=42)
    return e


@pytest.fixture
def seeded_env() -> FLReputationEnv:
    """Environment with fixed seed for reproducibility tests."""
    e = FLReputationEnv(
        alpha=0.5, beta=0.3, malicious_fraction=0.3,
        max_rounds=200, min_rounds=200,
    )
    return e


# =========================================================================
# 1. Space Definition Tests
# =========================================================================


class TestSpaceDefinitions:
    """Verify observation and action space geometry, bounds, and dtype."""

    def test_observation_space_shape(self, env: FLReputationEnv) -> None:
        """Per-agent observation space must be (NUM_FEATURES,) = (9,)."""
        assert env.observation_space.shape == (NUM_FEATURES,)

    def test_observation_space_bounds(self, env: FLReputationEnv) -> None:
        """Observation values bounded to [0, 1]."""
        np.testing.assert_array_equal(env.observation_space.low, 0.0)
        np.testing.assert_array_equal(env.observation_space.high, 1.0)

    def test_observation_space_dtype(self, env: FLReputationEnv) -> None:
        """Observation dtype must be float32."""
        assert env.observation_space.dtype == np.float32

    def test_action_space_shape(self, env: FLReputationEnv) -> None:
        """Per-agent action space must be (1,)."""
        assert env.action_space.shape == (1,)

    def test_action_space_bounds(self, env: FLReputationEnv) -> None:
        """Action values bounded to [0, 1]."""
        np.testing.assert_array_equal(env.action_space.low, 0.0)
        np.testing.assert_array_equal(env.action_space.high, 1.0)

    def test_action_space_dtype(self, env: FLReputationEnv) -> None:
        """Action dtype must be float32."""
        assert env.action_space.dtype == np.float32

    def test_feature_names_length(self) -> None:
        """FEATURE_NAMES list must match NUM_FEATURES."""
        assert len(FEATURE_NAMES) == NUM_FEATURES

    def test_num_clients_constant(self) -> None:
        """NUM_CLIENTS must be 100."""
        assert NUM_CLIENTS == 100

    def test_num_features_constant(self) -> None:
        """NUM_FEATURES must be 9 (5 local + 4 global context)."""
        assert NUM_FEATURES == 9


# =========================================================================
# 2. Reset Tests
# =========================================================================


class TestReset:
    """Verify reset returns valid per-agent observation and info dicts."""

    def test_reset_returns_obs_dict(self, env: FLReputationEnv) -> None:
        """Reset must return a dict mapping agent IDs to (NUM_FEATURES,) arrays."""
        obs_dict, _ = env.reset(seed=99)
        assert isinstance(obs_dict, dict)
        assert len(obs_dict) == env.num_clients
        for aid, obs in obs_dict.items():
            assert obs.shape == (NUM_FEATURES,)
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

    def test_reset_observation_dtype(self, env: FLReputationEnv) -> None:
        """Reset observation must be float32."""
        obs_dict, _ = env.reset(seed=99)
        agent = _first_agent(obs_dict)
        assert obs_dict[agent].dtype == np.float32

    def test_reset_returns_info_dict(self, env: FLReputationEnv) -> None:
        """Info dict per agent must contain 'round' key with value 0."""
        _, info_dict = env.reset(seed=99)
        agent = _first_agent(info_dict)
        assert "round" in info_dict[agent]
        assert info_dict[agent]["round"] == 0

    def test_reset_info_has_num_malicious(self, env: FLReputationEnv) -> None:
        """Info dict must report num_malicious as a positive integer."""
        _, info_dict = env.reset(seed=99)
        agent = _first_agent(info_dict)
        assert "num_malicious" in info_dict[agent]
        # Due to +-10% variation, check it is in a reasonable range
        assert info_dict[agent]["num_malicious"] >= 1
        assert info_dict[agent]["num_malicious"] <= env.num_clients

    def test_reset_with_seed_reproducible(
        self, seeded_env: FLReputationEnv
    ) -> None:
        """Two resets with the same seed must produce identical observations."""
        obs1, _ = seeded_env.reset(seed=123)
        obs2, _ = seeded_env.reset(seed=123)
        agent = _first_agent(obs1)
        np.testing.assert_array_equal(obs1[agent], obs2[agent])

    def test_reset_different_seeds_differ(
        self, seeded_env: FLReputationEnv
    ) -> None:
        """Different seeds must (almost surely) produce different observations."""
        obs1, _ = seeded_env.reset(seed=1)
        obs2, _ = seeded_env.reset(seed=2)
        agent = _first_agent(obs1)
        assert not np.array_equal(obs1[agent], obs2[agent])

    def test_reset_clears_round_counter(self, short_env: FLReputationEnv) -> None:
        """After stepping, reset should bring round back to 0."""
        action = _make_action_dict(short_env)
        short_env.step(action)
        short_env.step(action)
        _, info_dict = short_env.reset(seed=77)
        agent = _first_agent(info_dict)
        assert info_dict[agent]["round"] == 0

    def test_reset_observation_in_observation_space(
        self, env: FLReputationEnv
    ) -> None:
        """Reset observation must be contained in observation_space."""
        obs_dict, _ = env.reset(seed=55)
        agent = _first_agent(obs_dict)
        assert env.observation_space.contains(obs_dict[agent])


# =========================================================================
# 3. Step Tests
# =========================================================================


class TestStep:
    """Verify step API compliance (MultiAgentEnv dict-based API)."""

    def test_step_returns_five_tuple(self, env: FLReputationEnv) -> None:
        """Step must return 5 dicts: (obs, rewards, terminateds, truncateds, infos)."""
        result = env.step(_make_action_dict(env))
        assert len(result) == 5

    def test_step_observation_shape(self, env: FLReputationEnv) -> None:
        """Per-agent observation must have shape (NUM_FEATURES,) = (9,)."""
        obs_dict, _, _, _, _ = env.step(_make_action_dict(env))
        agent = _first_agent(obs_dict)
        assert obs_dict[agent].shape == (NUM_FEATURES,)

    def test_step_observation_in_bounds(self, env: FLReputationEnv) -> None:
        """All observation values must be in [0, 1]."""
        obs_dict, _, _, _, _ = env.step(_make_action_dict(env))
        for obs in obs_dict.values():
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

    def test_step_observation_in_observation_space(
        self, env: FLReputationEnv
    ) -> None:
        """Step observation must be contained in observation_space."""
        obs_dict, _, _, _, _ = env.step(_make_action_dict(env))
        agent = _first_agent(obs_dict)
        assert env.observation_space.contains(obs_dict[agent])

    def test_step_reward_is_float(self, env: FLReputationEnv) -> None:
        """Reward per agent must be a Python float."""
        _, rewards, _, _, _ = env.step(_make_action_dict(env))
        agent = _first_agent(rewards)
        assert isinstance(rewards[agent], float)

    def test_step_truncated_always_false(self, env: FLReputationEnv) -> None:
        """Truncated must always be False (no time limit truncation)."""
        _, _, _, truncateds, _ = env.step(_make_action_dict(env))
        assert truncateds["__all__"] is False

    def test_step_terminated_at_max_rounds(
        self, short_env: FLReputationEnv
    ) -> None:
        """After episode_max_rounds steps, terminated must be True."""
        action = _make_action_dict(short_env)
        # short_env has min_rounds=max_rounds=10, so episode is exactly 10 rounds
        for _ in range(10):
            _, _, terminateds, _, _ = short_env.step(action)
        assert terminateds["__all__"] is True

    def test_step_not_terminated_before_max(
        self, short_env: FLReputationEnv
    ) -> None:
        """Before episode_max_rounds, terminated must be False."""
        action = _make_action_dict(short_env)
        for i in range(9):
            _, _, terminateds, _, _ = short_env.step(action)
            assert terminateds["__all__"] is False, f"Terminated early at step {i}"

    def test_step_clips_action(self, env: FLReputationEnv) -> None:
        """Actions outside [0, 1] should be clipped without error."""
        action = {}
        for i in range(env.num_clients):
            val = -1.0 if i < 10 else 2.0
            action[f"agent_{i}"] = np.array([val], dtype=np.float32)
        obs_dict, rewards, terminateds, truncateds, infos = env.step(action)
        agent = _first_agent(obs_dict)
        assert obs_dict[agent].shape == (NUM_FEATURES,)
        assert isinstance(rewards[agent], float)

    def test_step_info_contains_round(self, env: FLReputationEnv) -> None:
        """Step info dict must contain the current round number."""
        _, _, _, _, infos = env.step(_make_action_dict(env))
        agent = _first_agent(infos)
        assert "round" in infos[agent]
        assert infos[agent]["round"] == 1

    def test_step_info_contains_reward_components(
        self, env: FLReputationEnv
    ) -> None:
        """Step info dict must contain reward breakdown with new keys."""
        _, _, _, _, infos = env.step(_make_action_dict(env))
        agent = _first_agent(infos)
        for key in ["weighted_similarity", "malicious_weight_penalty",
                     "low_sim_penalty", "improvement_bonus", "norm_penalty", "reward"]:
            assert key in infos[agent], f"Missing key '{key}' in step info"

    def test_step_all_zero_action(self, env: FLReputationEnv) -> None:
        """All-zero weights should return reward of -1.0 (guard clause)."""
        action = _make_action_dict(env, value=0.0)
        _, rewards, _, _, infos = env.step(action)
        agent = _first_agent(rewards)
        assert rewards[agent] == pytest.approx(-1.0)
        assert infos[agent]["malicious_weight_penalty"] == pytest.approx(1.0)


# =========================================================================
# 4. Reward Calculation Tests
# =========================================================================


class TestRewardCalculation:
    """Verify reward formula:
    R = alpha*weighted_sim - beta*malicious_penalty - delta*low_sim + eta*improvement - norm_penalty.
    """

    def test_reward_perfect_weights(self, env: FLReputationEnv) -> None:
        """Weights=1.0 for honest, 0.0 for malicious: low malicious penalty."""
        weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        weights[~env._malicious_mask] = 1.0

        reward, info = env._compute_reward(weights)

        assert info["malicious_weight_penalty"] == pytest.approx(0.0, abs=1e-6)
        # Honest gradient_similarity mean is ~0.8
        assert info["weighted_similarity"] > 0.4
        # Reward is better than worst-case (all-malicious) but may be negative
        # due to norm_penalty (weights sum far from 1.0)
        worst_weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        worst_weights[env._malicious_mask] = 1.0
        worst_reward, _ = env._compute_reward(worst_weights)
        assert reward > worst_reward

    def test_reward_worst_weights(self, env: FLReputationEnv) -> None:
        """Weights=0.0 for honest, 1.0 for malicious: high malicious penalty."""
        weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        weights[env._malicious_mask] = 1.0

        reward, info = env._compute_reward(weights)

        assert info["malicious_weight_penalty"] == pytest.approx(1.0, abs=1e-6)
        # Malicious gradient_similarity is low
        assert info["weighted_similarity"] < 0.5
        assert reward < 0.0

    def test_reward_uniform_weights(self, env: FLReputationEnv) -> None:
        """Equal weights for all: malicious_weight_penalty equals malicious_fraction."""
        weights = np.ones(NUM_CLIENTS, dtype=np.float32)

        reward, info = env._compute_reward(weights)

        # With uniform weights, fraction of malicious weight = malicious_fraction
        expected_penalty = env._malicious_mask.sum() / NUM_CLIENTS
        assert info["malicious_weight_penalty"] == pytest.approx(
            float(expected_penalty), abs=1e-6
        )

    def test_reward_manual_calculation(self) -> None:
        """Manually set state and verify reward matches hand-calculated value."""
        e = FLReputationEnv(
            alpha=0.5, beta=0.3, delta=0.3, eta=0.3,
            malicious_fraction=0.3, max_rounds=10, min_rounds=10,
        )
        e.reset(seed=42)

        # Set up a controlled state
        e._state = np.full((NUM_CLIENTS, NUM_FEATURES), 0.5, dtype=np.float32)
        # Set similarity column (col 1) to known values
        e._state[:, 1] = 0.8  # All gradient_similarity = 0.8

        # Set up known malicious mask: first 30 are malicious
        e._malicious_mask = np.zeros(NUM_CLIENTS, dtype=bool)
        e._malicious_mask[:30] = True

        # Reset prev_weighted_sim so improvement bonus is calculable
        e._prev_weighted_sim = 0.0

        # Weights: honest=1.0, malicious=0.5
        weights = np.ones(NUM_CLIENTS, dtype=np.float32)
        weights[:30] = 0.5  # malicious get 0.5

        weight_sum = 70 * 1.0 + 30 * 0.5  # = 85.0
        inv_ws = 1.0 / weight_sum

        # weighted_similarity = dot(weights, similarities) / sum(weights)
        # = (85 * 0.8) / 85 = 0.8
        expected_similarity = 0.8

        # malicious_weight_penalty = sum(weights[malicious]) / sum(weights)
        # = (30 * 0.5) / 85 = 15/85
        expected_penalty = 15.0 / 85.0

        # low_sim_penalty = dot(weights, 1-similarity) / sum(weights)
        # = dot(weights, 0.2) / 85 = (85*0.2)/85 = 0.2
        expected_low_sim = 0.2

        # improvement_bonus = max(0, weighted_sim - prev) = max(0, 0.8 - 0.0) = 0.8
        expected_improvement = 0.8

        # norm_penalty = 0.01 * |weight_sum - 1| = 0.01 * 84 = 0.84
        expected_norm_penalty = 0.01 * abs(weight_sum - 1.0)

        expected_reward = (
            0.5 * expected_similarity
            - 0.3 * expected_penalty
            - 0.3 * expected_low_sim
            + 0.3 * expected_improvement
            - expected_norm_penalty
        )

        reward, info = e._compute_reward(weights)

        assert info["weighted_similarity"] == pytest.approx(expected_similarity, abs=1e-5)
        assert info["malicious_weight_penalty"] == pytest.approx(expected_penalty, abs=1e-5)
        assert info["low_sim_penalty"] == pytest.approx(expected_low_sim, abs=1e-5)
        assert info["improvement_bonus"] == pytest.approx(expected_improvement, abs=1e-5)
        assert info["norm_penalty"] == pytest.approx(expected_norm_penalty, abs=1e-5)
        assert reward == pytest.approx(expected_reward, abs=1e-5)

    def test_reward_alpha_beta_sensitivity(self) -> None:
        """Higher alpha increases similarity contribution; higher beta increases malicious penalty."""
        high_alpha_env = FLReputationEnv(alpha=0.9, beta=0.1, max_rounds=10, min_rounds=10)
        high_beta_env = FLReputationEnv(alpha=0.1, beta=0.9, max_rounds=10, min_rounds=10)

        high_alpha_env.reset(seed=42)
        high_beta_env.reset(seed=42)

        # Copy state and malicious mask for consistency
        shared_state = high_alpha_env._state.copy()
        shared_mask = high_alpha_env._malicious_mask.copy()
        high_beta_env._state = shared_state.copy()
        high_beta_env._malicious_mask = shared_mask.copy()

        weights = np.ones(NUM_CLIENTS, dtype=np.float32)

        reward_high_alpha, info_ha = high_alpha_env._compute_reward(weights)
        reward_high_beta, info_hb = high_beta_env._compute_reward(weights)

        # High alpha emphasises similarity (positive), high beta emphasises penalty (negative)
        assert reward_high_alpha > reward_high_beta

    def test_reward_norm_penalty_uniform_vs_concentrated(
        self, env: FLReputationEnv
    ) -> None:
        """Weights summing to 1 should have lower norm_penalty than weights summing far from 1."""
        # Weights summing to 1
        unit_weights = np.full(NUM_CLIENTS, 1.0 / NUM_CLIENTS, dtype=np.float32)
        _, info_unit = env._compute_reward(unit_weights)

        # Weights summing to 100
        big_weights = np.ones(NUM_CLIENTS, dtype=np.float32)
        _, info_big = env._compute_reward(big_weights)

        assert info_unit["norm_penalty"] < info_big["norm_penalty"]

    def test_reward_zero_weights_guard(self) -> None:
        """All-zero weights should return -1.0 with malicious_weight_penalty=1.0."""
        e = FLReputationEnv(max_rounds=10, min_rounds=10)
        e.reset(seed=42)

        weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        reward, info = e._compute_reward(weights)

        assert reward == pytest.approx(-1.0)
        assert info["weighted_similarity"] == pytest.approx(0.0)
        assert info["malicious_weight_penalty"] == pytest.approx(1.0)
        assert info["improvement_bonus"] == pytest.approx(0.0)


# =========================================================================
# 5. State Generation Tests
# =========================================================================


class TestStateGeneration:
    """Verify generated state distributions and clipping."""

    def test_initial_state_reputation_column(self, env: FLReputationEnv) -> None:
        """On reset, column 2 (historical_reputation) should be 0.5 for all clients."""
        env.reset(seed=42)
        np.testing.assert_allclose(env._state[:, 2], 0.5, atol=1e-6)

    def test_honest_vs_malicious_accuracy_means(self) -> None:
        """Honest clients should have higher accuracy_contribution mean than malicious."""
        e = FLReputationEnv(max_rounds=10, min_rounds=10)
        honest_acc = []
        malicious_acc = []

        for seed in range(50):
            e.reset(seed=seed)
            honest_acc.append(e._state[~e._malicious_mask, 0].mean())
            malicious_acc.append(e._state[e._malicious_mask, 0].mean())

        mean_honest = np.mean(honest_acc)
        mean_malicious = np.mean(malicious_acc)
        assert mean_honest > mean_malicious, (
            f"Honest accuracy mean ({mean_honest:.3f}) should be > "
            f"malicious ({mean_malicious:.3f})"
        )

    def test_honest_vs_malicious_similarity_means(self) -> None:
        """Honest clients should have higher gradient_similarity mean than malicious."""
        e = FLReputationEnv(max_rounds=10, min_rounds=10)
        honest_sim = []
        malicious_sim = []

        for seed in range(50):
            e.reset(seed=seed)
            honest_sim.append(e._state[~e._malicious_mask, 1].mean())
            malicious_sim.append(e._state[e._malicious_mask, 1].mean())

        assert np.mean(honest_sim) > np.mean(malicious_sim)

    def test_malicious_higher_update_magnitude(self) -> None:
        """Malicious clients should have higher update_magnitude (col 4) mean."""
        e = FLReputationEnv(max_rounds=10, min_rounds=10)
        honest_mag = []
        malicious_mag = []

        for seed in range(50):
            e.reset(seed=seed)
            honest_mag.append(e._state[~e._malicious_mask, 4].mean())
            malicious_mag.append(e._state[e._malicious_mask, 4].mean())

        mean_honest = np.mean(honest_mag)
        mean_malicious = np.mean(malicious_mag)
        assert mean_malicious > mean_honest, (
            f"Malicious magnitude mean ({mean_malicious:.3f}) should be > "
            f"honest ({mean_honest:.3f})"
        )

    def test_state_values_clipped(self, env: FLReputationEnv) -> None:
        """All state values must be in [0, 1]."""
        env.reset(seed=42)
        assert np.all(env._state >= 0.0)
        assert np.all(env._state <= 1.0)

    def test_state_values_clipped_after_step(self, env: FLReputationEnv) -> None:
        """State values remain in [0, 1] after stepping."""
        action = _make_action_dict(env)
        for _ in range(5):
            env.step(action)
            assert np.all(env._state >= 0.0), "State has values below 0 after step"
            assert np.all(env._state <= 1.0), "State has values above 1 after step"


# =========================================================================
# 6. Reputation Update Tests
# =========================================================================


class TestReputationUpdate:
    """Verify EMA reputation updates for honest and malicious clients."""

    def test_reputation_update_honest_high_weight(self) -> None:
        """With few honest clients getting large weights, their individual reputation increases.

        The EMA signal for honest is w_i / sum(w). With 100 clients, normalised
        weight per client is small (~0.01 for uniform). To ensure a client's
        reputation *increases* from 0.5, we need w_i/sum(w) > 0.5, which requires
        concentrating weight on very few honest clients.
        """
        e = FLReputationEnv(max_rounds=200, min_rounds=200)
        e.reset(seed=42)

        # Set known initial reputation
        e._state[:, 2] = 0.0  # start from 0

        # Give weight only to honest clients, zero for malicious
        weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        weights[~e._malicious_mask] = 1.0

        e._update_reputation(weights)
        updated_rep_honest = e._state[~e._malicious_mask, 2]

        # Honest signal = w_i / sum(w) = 1/70 ~= 0.0143, which is > 0
        # EMA: 0.7 * 0.0 + 0.3 * 0.0143 > 0.0
        assert updated_rep_honest.mean() > 0.0, (
            f"Honest rep should increase from 0: {updated_rep_honest.mean():.4f}"
        )

    def test_reputation_update_malicious_high_weight(self) -> None:
        """After stepping with high weights for malicious clients, their reputation should decrease.

        The EMA formula for malicious uses signal = 1 - w_i/sum(w).
        When malicious clients get weight 1.0 and there are 30 of them with total
        sum ~30, signal = 1 - 1/30 ~= 0.967. Starting from reputation 1.0,
        EMA goes to 0.7*1.0 + 0.3*0.967 = 0.99 which is still < 1.0.
        We start from 1.0 so any signal < 1.0 causes a decrease.
        """
        e = FLReputationEnv(max_rounds=200, min_rounds=200)
        e.reset(seed=42)

        # Start reputation at 1.0 so any signal < 1.0 pulls it down
        e._state[:, 2] = 1.0
        initial_rep_malicious = e._state[e._malicious_mask, 2].copy()

        # Give weight only to malicious clients
        weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        weights[e._malicious_mask] = 1.0

        e._update_reputation(weights)
        updated_rep_malicious = e._state[e._malicious_mask, 2]

        assert updated_rep_malicious.mean() < initial_rep_malicious.mean(), (
            f"Malicious rep mean should decrease from 1.0: "
            f"{initial_rep_malicious.mean():.4f} -> {updated_rep_malicious.mean():.4f}"
        )

    def test_reputation_ema_alpha(self) -> None:
        """Verify the EMA uses the correct alpha (0.3) value."""
        e = FLReputationEnv(max_rounds=200, min_rounds=200)
        e.reset(seed=42)

        # Set reputation to a known value
        e._state[:, 2] = 0.5

        # Uniform weights
        weights = np.ones(NUM_CLIENTS, dtype=np.float32)
        weight_sum = weights.sum()
        normalised_w = weights / weight_sum  # = 0.01 for each

        alpha = _REPUTATION_EMA_ALPHA  # 0.3

        # Expected honest: (1 - 0.3) * 0.5 + 0.3 * 0.01 = 0.35 + 0.003 = 0.353
        expected_honest = (1 - alpha) * 0.5 + alpha * normalised_w[0]
        # Expected malicious: (1 - 0.3) * 0.5 + 0.3 * (1 - 0.01) = 0.35 + 0.297 = 0.647
        expected_malicious = (1 - alpha) * 0.5 + alpha * (1 - normalised_w[0])

        e._update_reputation(weights)

        honest_rep = e._state[~e._malicious_mask, 2]
        malicious_rep = e._state[e._malicious_mask, 2]

        np.testing.assert_allclose(honest_rep, expected_honest, atol=1e-5)
        np.testing.assert_allclose(malicious_rep, expected_malicious, atol=1e-5)

    def test_reputation_stays_in_bounds(self) -> None:
        """Reputation values must remain in [0, 1] after many updates."""
        e = FLReputationEnv(max_rounds=50, min_rounds=50)
        e.reset(seed=42)

        # Extreme weights for many rounds
        weights_extreme = np.zeros(NUM_CLIENTS, dtype=np.float32)
        weights_extreme[~e._malicious_mask] = 1.0

        for _ in range(50):
            e._update_reputation(weights_extreme)

        assert np.all(e._state[:, 2] >= 0.0)
        assert np.all(e._state[:, 2] <= 1.0)

    def test_reputation_zero_weights_no_update(self) -> None:
        """Zero weights should not change reputation (guard clause)."""
        e = FLReputationEnv(max_rounds=10, min_rounds=10)
        e.reset(seed=42)

        rep_before = e._state[:, 2].copy()
        e._update_reputation(np.zeros(NUM_CLIENTS, dtype=np.float32))
        rep_after = e._state[:, 2]

        np.testing.assert_array_equal(rep_before, rep_after)


# =========================================================================
# 7. Episode Lifecycle Tests
# =========================================================================


class TestEpisodeLifecycle:
    """Verify episode termination and length."""

    def test_full_episode_completes(self, short_env: FLReputationEnv) -> None:
        """Run episode_max_rounds steps; last step must have terminated=True."""
        action = _make_action_dict(short_env)
        terminateds = {"__all__": False}
        for _ in range(10):
            _, _, terminateds, _, _ = short_env.step(action)
        assert terminateds["__all__"] is True

    def test_episode_length_matches_episode_max_rounds(self) -> None:
        """Count steps until terminated; must equal _episode_max_rounds."""
        e = FLReputationEnv(max_rounds=15, min_rounds=15)
        e.reset(seed=42)
        action = _make_action_dict(e)

        steps = 0
        terminated = False
        while not terminated:
            _, _, terminateds, _, _ = e.step(action)
            terminated = terminateds["__all__"]
            steps += 1

        assert steps == 15

    def test_episode_length_varies_with_min_max_rounds(self) -> None:
        """With min_rounds < max_rounds, episode length should vary across seeds."""
        lengths = []
        for seed in range(20):
            e = FLReputationEnv(max_rounds=20, min_rounds=5)
            e.reset(seed=seed)
            lengths.append(e._episode_max_rounds)

        # Should have some variation
        assert min(lengths) >= 5
        assert max(lengths) <= 20
        # With 20 different seeds, we expect at least 2 distinct lengths
        assert len(set(lengths)) >= 2

    def test_gymnasium_check_env_skipped(self) -> None:
        """MultiAgentEnv is not compatible with gymnasium.utils.env_checker.
        Instead, verify the env conforms to MultiAgentEnv interface.
        """
        e = FLReputationEnv(max_rounds=10, min_rounds=10)
        obs_dict, info_dict = e.reset(seed=42)

        # Verify dict-based API
        assert isinstance(obs_dict, dict)
        assert isinstance(info_dict, dict)
        assert len(obs_dict) == e.num_clients

        action = _make_action_dict(e)
        obs_d, rew_d, term_d, trunc_d, info_d = e.step(action)
        assert "__all__" in term_d
        assert "__all__" in trunc_d

    def test_cumulative_reward_in_info(self, short_env: FLReputationEnv) -> None:
        """Step info should track cumulative_reward across the episode."""
        action = _make_action_dict(short_env)
        total = 0.0
        for _ in range(5):
            _, rewards, _, _, infos = short_env.step(action)
            agent = _first_agent(rewards)
            total += rewards[agent]
            assert "cumulative_reward" in infos[agent]
            assert infos[agent]["cumulative_reward"] == pytest.approx(total, abs=1e-5)

    def test_reset_after_episode(self) -> None:
        """Environment should be fully usable after reset following a complete episode."""
        e = FLReputationEnv(max_rounds=5, min_rounds=5)
        e.reset(seed=42)
        action = _make_action_dict(e)

        # Run full episode
        for _ in range(5):
            e.step(action)

        # Reset and run again
        obs_dict, info_dict = e.reset(seed=99)
        agent = _first_agent(obs_dict)
        assert obs_dict[agent].shape == (NUM_FEATURES,)
        assert info_dict[agent]["round"] == 0

        obs_d2, _, terminateds, _, _ = e.step(action)
        agent2 = _first_agent(obs_d2)
        assert obs_d2[agent2].shape == (NUM_FEATURES,)
        assert terminateds["__all__"] is False


# =========================================================================
# 8. Constructor Configuration Tests
# =========================================================================


class TestConstructorConfiguration:
    """Verify custom constructor parameters are respected."""

    def test_custom_alpha_beta(self) -> None:
        """Custom alpha/beta should affect reward calculation."""
        e1 = FLReputationEnv(alpha=0.9, beta=0.1, max_rounds=10, min_rounds=10)
        e2 = FLReputationEnv(alpha=0.1, beta=0.9, max_rounds=10, min_rounds=10)

        e1.reset(seed=42)
        e2.reset(seed=42)

        # Ensure identical state
        e2._state = e1._state.copy()
        e2._malicious_mask = e1._malicious_mask.copy()

        action = _make_action_dict(e1)
        _, rew1, _, _, _ = e1.step(action)
        _, rew2, _, _, _ = e2.step(action)

        agent = _first_agent(rew1)
        # Different alpha/beta should produce different rewards
        assert rew1[agent] != pytest.approx(rew2[agent], abs=1e-6)

    def test_custom_malicious_fraction(self) -> None:
        """malicious_fraction=0.5 should produce approximately 50 malicious clients.

        Due to +-10% variation and clipping to [0.05, 0.5], the exact count varies.
        """
        e = FLReputationEnv(malicious_fraction=0.5, max_rounds=10, min_rounds=10)
        _, info_dict = e.reset(seed=42)
        agent = _first_agent(info_dict)
        num_mal = info_dict[agent]["num_malicious"]
        # With malicious_fraction=0.5, effective is clipped to [0.05, 0.5] after +-0.1
        # So num_malicious is in [5, 50] approximately
        assert num_mal >= 1
        assert num_mal <= 50
        assert e._malicious_mask.sum() == num_mal

    def test_custom_max_rounds(self) -> None:
        """max_rounds=min_rounds=5 should terminate after exactly 5 steps."""
        e = FLReputationEnv(max_rounds=5, min_rounds=5)
        e.reset(seed=42)
        action = _make_action_dict(e)

        for i in range(4):
            _, _, term, _, _ = e.step(action)
            assert term["__all__"] is False
        _, _, term, _, _ = e.step(action)
        assert term["__all__"] is True

    def test_zero_malicious_fraction(self) -> None:
        """malicious_fraction=0.0: due to variation, at least 1 malicious, penalty > 0."""
        e = FLReputationEnv(malicious_fraction=0.0, max_rounds=10, min_rounds=10)
        _, info_dict = e.reset(seed=42)
        agent = _first_agent(info_dict)
        # clip(0.0 + uniform(-0.1, 0.1), 0.05, 0.5) >= 0.05
        # max(1, int(100 * 0.05)) >= 1
        assert info_dict[agent]["num_malicious"] >= 1

        weights = np.ones(NUM_CLIENTS, dtype=np.float32)
        _, reward_info = e._compute_reward(weights)
        # With at least 1 malicious, penalty > 0
        assert reward_info["malicious_weight_penalty"] > 0.0

    def test_full_malicious_fraction(self) -> None:
        """malicious_fraction=1.0: clipped to 0.5, so ~50 malicious clients."""
        e = FLReputationEnv(malicious_fraction=1.0, max_rounds=10, min_rounds=10)
        _, info_dict = e.reset(seed=42)
        agent = _first_agent(info_dict)
        num_mal = info_dict[agent]["num_malicious"]
        # clip(1.0 + uniform(-0.1, 0.1), 0.05, 0.5) = 0.5
        # int(100 * 0.5) = 50 (approximately, depending on random offset)
        assert num_mal >= 40
        assert num_mal <= 50

    def test_default_parameters(self) -> None:
        """Default constructor should use alpha=0.5, beta=0.3, fraction=0.3, max_rounds=200."""
        e = FLReputationEnv()
        assert e.alpha == pytest.approx(0.5)
        assert e.beta == pytest.approx(0.3)
        assert e.malicious_fraction == pytest.approx(0.3)
        assert e.max_rounds == 200
        assert e.min_rounds == 5
        assert e.eta == pytest.approx(0.3)
        assert e.delta == pytest.approx(0.3)

    def test_render_mode_stored(self) -> None:
        """render_mode should be stored as an attribute."""
        e = FLReputationEnv(render_mode="human")
        assert e.render_mode == "human"

    def test_render_mode_none_default(self) -> None:
        """Default render_mode should be None."""
        e = FLReputationEnv()
        assert e.render_mode is None


# =========================================================================
# Kernels Tests (reward normalization, Triton fallback, RunningMeanStd)
# =========================================================================

import torch

from src.rl_agent.kernels import (
    TRITON_AVAILABLE,
    RunningMeanStd,
    fused_reward_normalize,
    _torch_normalize,
)


class TestRunningMeanStd:
    """Unit tests for RunningMeanStd class (Welford online algorithm)."""

    def test_initialization(self) -> None:
        """RunningMeanStd should initialize with mean=0, var=epsilon, count=0."""
        rms = RunningMeanStd(epsilon=1e-4, device="cpu")
        assert float(rms.mean) == pytest.approx(0.0)
        assert float(rms.var) == pytest.approx(1e-4)
        assert int(rms.count) == 0

    def test_single_batch_update(self) -> None:
        """After one batch update, mean and var should match batch statistics."""
        rms = RunningMeanStd()
        batch = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        rms.update(batch)

        expected_mean = batch.mean().item()
        expected_var = batch.var(unbiased=False).item()

        assert float(rms.mean) == pytest.approx(expected_mean, rel=1e-5)
        assert float(rms.var) == pytest.approx(expected_var, rel=1e-5)
        assert int(rms.count) == 5

    def test_multiple_batch_updates_welford(self) -> None:
        """Multiple batches should correctly merge via Welford algorithm."""
        rms = RunningMeanStd()

        batch1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        batch2 = torch.tensor([4.0, 5.0], dtype=torch.float32)

        rms.update(batch1)
        rms.update(batch2)

        # Expected: mean=(1+2+3+4+5)/5=3.0, var of [1,2,3,4,5] unbiased
        all_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_mean = all_data.mean().item()
        expected_var = all_data.var(unbiased=False).item()

        assert float(rms.mean) == pytest.approx(expected_mean, rel=1e-5)
        assert float(rms.var) == pytest.approx(expected_var, rel=1e-5)
        assert int(rms.count) == 5

    def test_empty_batch_no_update(self) -> None:
        """Updating with an empty batch should not change statistics."""
        rms = RunningMeanStd()
        initial_mean = float(rms.mean)
        initial_var = float(rms.var)
        initial_count = int(rms.count)

        rms.update(torch.tensor([], dtype=torch.float32))

        assert float(rms.mean) == initial_mean
        assert float(rms.var) == initial_var
        assert int(rms.count) == initial_count

    def test_std_property_clamped(self) -> None:
        """std property should return sqrt(var), clamped to >= 1e-8."""
        rms = RunningMeanStd(epsilon=1e-4)
        rms.update(torch.tensor([1.0], dtype=torch.float32))

        std = rms.std
        assert float(std) >= 0.99e-8  # Allow small floating point tolerance
        assert float(std) == pytest.approx(torch.sqrt(rms.var).clamp(min=1e-8).item(), rel=1e-6)

    def test_large_batch_numerical_stability(self) -> None:
        """Large batches should not overflow or underflow."""
        rms = RunningMeanStd()
        large_batch = torch.linspace(1e6, 1e7, 1000, dtype=torch.float32)
        rms.update(large_batch)

        assert not torch.isnan(rms.mean)
        assert not torch.isnan(rms.var)
        assert float(rms.mean) > 0

    def test_negative_values_handled(self) -> None:
        """Negative batch values should be handled correctly."""
        rms = RunningMeanStd()
        batch = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=torch.float32)
        rms.update(batch)

        expected_mean = batch.mean().item()
        assert float(rms.mean) == pytest.approx(expected_mean, rel=1e-5)


class TestFusedRewardNormalize:
    """Unit tests for fused_reward_normalize function."""

    def test_returns_tensor_same_shape(self) -> None:
        """Normalized output should have same shape as input."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        normalized = fused_reward_normalize(rewards, mean=3.0, std=1.0)

        assert normalized.shape == rewards.shape
        assert normalized.dtype == torch.float32

    def test_zero_std_handled_with_epsilon(self) -> None:
        """Division by zero should be avoided via epsilon."""
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        normalized = fused_reward_normalize(rewards, mean=2.0, std=0.0, epsilon=1e-8)

        # Should not contain inf or nan
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_correct_normalization_computation(self) -> None:
        """Normalized output should be (rewards - mean) / (std + epsilon)."""
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        mean, std, epsilon = 2.0, 1.0, 1e-8

        normalized = fused_reward_normalize(rewards, mean=mean, std=std, epsilon=epsilon)

        expected = (rewards - mean) / (std + epsilon)
        assert torch.allclose(normalized, expected, atol=1e-6)

    def test_cpu_path_always_works(self) -> None:
        """PyTorch fallback should work on CPU."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        normalized = fused_reward_normalize(rewards, mean=3.0, std=1.4)

        # Should not error; should produce finite values
        assert normalized.shape == rewards.shape
        assert not torch.isnan(normalized).any()


class TestTorchNormalize:
    """Unit tests for the PyTorch fallback normalization (_torch_normalize)."""

    def test_torch_normalize_correctness(self) -> None:
        """_torch_normalize should match manual formula."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        mean_val, std_val, epsilon = 2.5, 1.1, 1e-8

        result = _torch_normalize(rewards, mean_val, std_val, epsilon)

        expected = (rewards - mean_val) / (std_val + epsilon)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_torch_normalize_zero_std(self) -> None:
        """_torch_normalize should handle zero std via epsilon."""
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = _torch_normalize(rewards, mean_val=2.0, std_val=0.0, epsilon=1e-8)

        assert not torch.isinf(result).any()
        assert not torch.isnan(result).any()


# =========================================================================
# Train Function Tests (configuration, GPU detection, PPO setup)
# =========================================================================

from src.rl_agent.train import (
    env_creator,
    _detect_gpu_resources,
    build_ppo_config,
)


class TestEnvCreator:
    """Unit tests for env_creator factory function."""

    def test_env_creator_default_config(self) -> None:
        """env_creator with empty config should use defaults."""
        env = env_creator({})

        assert isinstance(env, FLReputationEnv)
        assert env.alpha == pytest.approx(0.5)
        assert env.beta == pytest.approx(0.3)
        assert env.malicious_fraction == pytest.approx(0.3)

    def test_env_creator_custom_parameters(self) -> None:
        """env_creator should respect custom config parameters."""
        config = {
            "alpha": 0.7,
            "beta": 0.2,
            "malicious_fraction": 0.5,
            "num_clients": 50,
        }
        env = env_creator(config)

        assert env.alpha == pytest.approx(0.7)
        assert env.beta == pytest.approx(0.2)
        assert env.malicious_fraction == pytest.approx(0.5)
        assert env.num_clients == 50

    def test_env_creator_curriculum_phase_1(self) -> None:
        """Curriculum phase 1 should set easy difficulty."""
        config = {"curriculum_phase": 1}
        env = env_creator(config)

        assert env.malicious_fraction == pytest.approx(0.1)
        assert env.min_rounds == 20

    def test_env_creator_curriculum_phase_2(self) -> None:
        """Curriculum phase 2 should set medium difficulty."""
        config = {"curriculum_phase": 2}
        env = env_creator(config)

        assert env.malicious_fraction == pytest.approx(0.3)
        assert env.min_rounds == 10

    def test_env_creator_curriculum_phase_3(self) -> None:
        """Curriculum phase 3 should set hard difficulty."""
        config = {"curriculum_phase": 3}
        env = env_creator(config)

        assert env.malicious_fraction == pytest.approx(0.5)
        assert env.min_rounds == 5

    def test_env_creator_invalid_curriculum_phase(self) -> None:
        """Invalid curriculum phase should raise ValueError."""
        config = {"curriculum_phase": 4}
        with pytest.raises(ValueError, match="Invalid curriculum_phase"):
            env_creator(config)


class TestDetectGpuResources:
    """Unit tests for GPU resource detection."""

    def test_returns_tuple(self) -> None:
        """_detect_gpu_resources should return (int, float) tuple."""
        result = _detect_gpu_resources()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_gpu_allocation_sum(self) -> None:
        """When GPU available, allocation should be sensible."""
        num_gpus, worker_gpus = _detect_gpu_resources()

        if torch.cuda.is_available():
            # Learner gets all GPUs
            assert num_gpus == torch.cuda.device_count()
            # Workers should get 0 when GPUs available (CPU-only rollouts)
            assert worker_gpus == 0.0
        else:
            # No GPU: both should be 0
            assert num_gpus == 0
            assert worker_gpus == 0.0


class TestBuildPpoConfig:
    """Unit tests for build_ppo_config function."""

    def test_config_builds_without_error(self) -> None:
        """build_ppo_config should return valid PPOConfig."""
        config = build_ppo_config(num_workers=2)

        assert config is not None
        # PPOConfig should have build() method
        assert hasattr(config, 'build')

    def test_config_env_config_passed(self) -> None:
        """Config should accept and use env_config parameter."""
        env_cfg = {"malicious_fraction": 0.5, "num_clients": 50}
        config = build_ppo_config(num_workers=2, env_config=env_cfg)

        assert config is not None

    def test_config_with_empty_env_config(self) -> None:
        """Config should handle empty env_config dict."""
        config = build_ppo_config(num_workers=2, env_config={})

        assert config is not None
