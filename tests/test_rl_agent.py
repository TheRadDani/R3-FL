"""
Comprehensive pytest tests for the RL agent environment (src/rl_agent/env.py).

Covers:
  - Observation and action space definitions (shape, bounds, dtype)
  - Reset API compliance and reproducibility
  - Step API compliance (5-tuple, termination, clipping)
  - Reward formula verification (weighted accuracy, attack impact, entropy penalty)
  - State generation (honest vs malicious feature distributions)
  - Reputation EMA updates (honest up, malicious down)
  - Episode lifecycle (termination at max_rounds)
  - Constructor configuration (alpha, beta, malicious_fraction, max_rounds)
  - Gymnasium env_checker compliance
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> FLReputationEnv:
    """Default environment with standard parameters."""
    e = FLReputationEnv(alpha=0.6, beta=0.4, malicious_fraction=0.3, max_rounds=200)
    e.reset(seed=42)
    return e


@pytest.fixture
def short_env() -> FLReputationEnv:
    """Short-episode environment for fast lifecycle tests."""
    e = FLReputationEnv(alpha=0.6, beta=0.4, malicious_fraction=0.3, max_rounds=10)
    e.reset(seed=42)
    return e


@pytest.fixture
def seeded_env() -> FLReputationEnv:
    """Environment with fixed seed for reproducibility tests."""
    e = FLReputationEnv(alpha=0.6, beta=0.4, malicious_fraction=0.3, max_rounds=200)
    return e


# =========================================================================
# 1. Space Definition Tests
# =========================================================================


class TestSpaceDefinitions:
    """Verify observation and action space geometry, bounds, and dtype."""

    def test_observation_space_shape(self, env: FLReputationEnv) -> None:
        """Observation space must be (NUM_CLIENTS, NUM_FEATURES) = (100, 5)."""
        assert env.observation_space.shape == (NUM_CLIENTS, NUM_FEATURES)

    def test_observation_space_bounds(self, env: FLReputationEnv) -> None:
        """Observation values bounded to [0, 1]."""
        np.testing.assert_array_equal(env.observation_space.low, 0.0)
        np.testing.assert_array_equal(env.observation_space.high, 1.0)

    def test_observation_space_dtype(self, env: FLReputationEnv) -> None:
        """Observation dtype must be float32."""
        assert env.observation_space.dtype == np.float32

    def test_action_space_shape(self, env: FLReputationEnv) -> None:
        """Action space must be (NUM_CLIENTS,) = (100,)."""
        assert env.action_space.shape == (NUM_CLIENTS,)

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
        """NUM_FEATURES must be 5."""
        assert NUM_FEATURES == 5


# =========================================================================
# 2. Reset Tests
# =========================================================================


class TestReset:
    """Verify reset returns valid observation and info dict."""

    def test_reset_returns_valid_observation(self, env: FLReputationEnv) -> None:
        """Reset observation must have shape (100, 5) with values in [0, 1]."""
        obs, _ = env.reset(seed=99)
        assert obs.shape == (NUM_CLIENTS, NUM_FEATURES)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_reset_observation_dtype(self, env: FLReputationEnv) -> None:
        """Reset observation must be float32."""
        obs, _ = env.reset(seed=99)
        assert obs.dtype == np.float32

    def test_reset_returns_info_dict(self, env: FLReputationEnv) -> None:
        """Info dict must contain 'round' key with value 0."""
        _, info = env.reset(seed=99)
        assert "round" in info
        assert info["round"] == 0

    def test_reset_info_has_num_malicious(self, env: FLReputationEnv) -> None:
        """Info dict must report num_malicious = int(0.3 * 100) = 30."""
        _, info = env.reset(seed=99)
        assert "num_malicious" in info
        assert info["num_malicious"] == 30

    def test_reset_with_seed_reproducible(
        self, seeded_env: FLReputationEnv
    ) -> None:
        """Two resets with the same seed must produce identical observations."""
        obs1, _ = seeded_env.reset(seed=123)
        obs2, _ = seeded_env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_different_seeds_differ(
        self, seeded_env: FLReputationEnv
    ) -> None:
        """Different seeds must (almost surely) produce different observations."""
        obs1, _ = seeded_env.reset(seed=1)
        obs2, _ = seeded_env.reset(seed=2)
        assert not np.array_equal(obs1, obs2)

    def test_reset_clears_round_counter(self, short_env: FLReputationEnv) -> None:
        """After stepping, reset should bring round back to 0."""
        action = np.full(NUM_CLIENTS, 0.5, dtype=np.float32)
        short_env.step(action)
        short_env.step(action)
        _, info = short_env.reset(seed=77)
        assert info["round"] == 0

    def test_reset_observation_in_observation_space(
        self, env: FLReputationEnv
    ) -> None:
        """Reset observation must be contained in observation_space."""
        obs, _ = env.reset(seed=55)
        assert env.observation_space.contains(obs)


# =========================================================================
# 3. Step Tests
# =========================================================================


class TestStep:
    """Verify step API compliance."""

    def _uniform_action(self) -> np.ndarray:
        return np.full(NUM_CLIENTS, 0.5, dtype=np.float32)

    def test_step_returns_five_tuple(self, env: FLReputationEnv) -> None:
        """Step must return (obs, reward, terminated, truncated, info)."""
        result = env.step(self._uniform_action())
        assert len(result) == 5

    def test_step_observation_shape(self, env: FLReputationEnv) -> None:
        """Next observation must have shape (100, 5)."""
        obs, _, _, _, _ = env.step(self._uniform_action())
        assert obs.shape == (NUM_CLIENTS, NUM_FEATURES)

    def test_step_observation_in_bounds(self, env: FLReputationEnv) -> None:
        """All observation values must be in [0, 1]."""
        obs, _, _, _, _ = env.step(self._uniform_action())
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_step_observation_in_observation_space(
        self, env: FLReputationEnv
    ) -> None:
        """Step observation must be contained in observation_space."""
        obs, _, _, _, _ = env.step(self._uniform_action())
        assert env.observation_space.contains(obs)

    def test_step_reward_is_float(self, env: FLReputationEnv) -> None:
        """Reward must be a Python float."""
        _, reward, _, _, _ = env.step(self._uniform_action())
        assert isinstance(reward, float)

    def test_step_truncated_always_false(self, env: FLReputationEnv) -> None:
        """Truncated must always be False (no time limit truncation)."""
        _, _, _, truncated, _ = env.step(self._uniform_action())
        assert truncated is False

    def test_step_terminated_at_max_rounds(
        self, short_env: FLReputationEnv
    ) -> None:
        """After max_rounds steps, terminated must be True."""
        action = self._uniform_action()
        for _ in range(10):
            _, _, terminated, _, _ = short_env.step(action)
        assert terminated is True

    def test_step_not_terminated_before_max(
        self, short_env: FLReputationEnv
    ) -> None:
        """Before max_rounds, terminated must be False."""
        action = self._uniform_action()
        for i in range(9):
            _, _, terminated, _, _ = short_env.step(action)
            assert terminated is False, f"Terminated early at step {i}"

    def test_step_clips_action(self, env: FLReputationEnv) -> None:
        """Actions outside [0, 1] should be clipped without error."""
        action = np.full(NUM_CLIENTS, 2.0, dtype=np.float32)
        action[:10] = -1.0
        obs, reward, terminated, truncated, info = env.step(action)
        # Should complete without error
        assert obs.shape == (NUM_CLIENTS, NUM_FEATURES)
        assert isinstance(reward, float)

    def test_step_info_contains_round(self, env: FLReputationEnv) -> None:
        """Step info dict must contain the current round number."""
        _, _, _, _, info = env.step(self._uniform_action())
        assert "round" in info
        assert info["round"] == 1

    def test_step_info_contains_reward_components(
        self, env: FLReputationEnv
    ) -> None:
        """Step info dict must contain reward breakdown."""
        _, _, _, _, info = env.step(self._uniform_action())
        for key in ["weighted_accuracy", "attack_impact", "entropy_penalty", "reward"]:
            assert key in info, f"Missing key '{key}' in step info"

    def test_step_all_zero_action(self, env: FLReputationEnv) -> None:
        """All-zero weights should return reward of -1.0 (guard clause)."""
        action = np.zeros(NUM_CLIENTS, dtype=np.float32)
        _, reward, _, _, info = env.step(action)
        assert reward == pytest.approx(-1.0)
        assert info["attack_impact"] == pytest.approx(1.0)


# =========================================================================
# 4. Reward Calculation Tests
# =========================================================================


class TestRewardCalculation:
    """Verify reward formula: R = alpha*weighted_acc - beta*attack_impact - entropy_penalty."""

    def test_reward_perfect_weights(self, env: FLReputationEnv) -> None:
        """Weights=1.0 for honest, 0.0 for malicious: high reward, low attack impact."""
        weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        weights[~env._malicious_mask] = 1.0

        reward, info = env._compute_reward(weights)

        assert info["attack_impact"] == pytest.approx(0.0, abs=1e-6)
        # Honest accuracy_contribution mean is ~0.7
        assert info["weighted_accuracy"] > 0.4
        assert reward > 0.0

    def test_reward_worst_weights(self, env: FLReputationEnv) -> None:
        """Weights=0.0 for honest, 1.0 for malicious: low reward, high attack impact."""
        weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        weights[env._malicious_mask] = 1.0

        reward, info = env._compute_reward(weights)

        assert info["attack_impact"] == pytest.approx(1.0, abs=1e-6)
        # Malicious accuracy_contribution mean is ~0.3
        assert info["weighted_accuracy"] < 0.5
        assert reward < 0.0

    def test_reward_uniform_weights(self, env: FLReputationEnv) -> None:
        """Equal weights for all: attack_impact equals malicious_fraction."""
        weights = np.ones(NUM_CLIENTS, dtype=np.float32)

        reward, info = env._compute_reward(weights)

        # With uniform weights, fraction of malicious weight = malicious_fraction
        expected_attack = env._malicious_mask.sum() / NUM_CLIENTS
        assert info["attack_impact"] == pytest.approx(
            float(expected_attack), abs=1e-6
        )

    def test_reward_manual_calculation(self) -> None:
        """Manually set state and verify reward matches hand-calculated value."""
        e = FLReputationEnv(alpha=0.6, beta=0.4, malicious_fraction=0.3, max_rounds=10)
        e.reset(seed=42)

        # Set up a controlled state
        e._state = np.full((NUM_CLIENTS, NUM_FEATURES), 0.5, dtype=np.float32)
        # Set accuracy column to known values
        e._state[:, 0] = 0.8  # All accuracy = 0.8

        # Set up known malicious mask: first 30 are malicious
        e._malicious_mask = np.zeros(NUM_CLIENTS, dtype=bool)
        e._malicious_mask[:30] = True

        # Weights: honest=1.0, malicious=0.5
        weights = np.ones(NUM_CLIENTS, dtype=np.float32)
        weights[:30] = 0.5  # malicious get 0.5

        weight_sum = 70 * 1.0 + 30 * 0.5  # = 85.0
        expected_accuracy = (NUM_CLIENTS * 0.8) / weight_sum  # all acc=0.8
        # Actually: sum(weights * acc) / sum(weights) = (85 * 0.8) / 85 = 0.8
        # Wait: sum(weights * 0.8) = 0.8 * sum(weights) = 0.8 * 85 = 68
        # weighted_accuracy = 68 / 85 = 0.8
        expected_accuracy = 0.8
        expected_attack = (30 * 0.5) / 85.0  # = 15/85 ~= 0.17647

        # Entropy calculation
        p = weights / weight_sum
        log_p = np.log(p + 1e-10)
        entropy = -float(np.sum(p * log_p))
        max_entropy = float(np.log(NUM_CLIENTS))
        expected_entropy_penalty = 0.05 * (entropy / max_entropy)

        expected_reward = (
            0.6 * expected_accuracy - 0.4 * expected_attack - expected_entropy_penalty
        )

        reward, info = e._compute_reward(weights)

        assert info["weighted_accuracy"] == pytest.approx(expected_accuracy, abs=1e-5)
        assert info["attack_impact"] == pytest.approx(expected_attack, abs=1e-5)
        assert info["entropy_penalty"] == pytest.approx(
            expected_entropy_penalty, abs=1e-5
        )
        assert reward == pytest.approx(expected_reward, abs=1e-5)

    def test_reward_alpha_beta_sensitivity(self) -> None:
        """Higher alpha increases accuracy contribution; higher beta increases attack penalty."""
        high_alpha_env = FLReputationEnv(alpha=0.9, beta=0.1, max_rounds=10)
        high_beta_env = FLReputationEnv(alpha=0.1, beta=0.9, max_rounds=10)

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

        # With uniform weights and ~30% malicious, attack_impact ~0.3
        # High alpha emphasises accuracy (positive), high beta emphasises penalty (negative)
        # So high_alpha reward > high_beta reward
        assert reward_high_alpha > reward_high_beta

    def test_reward_entropy_penalty_uniform_vs_discriminating(
        self, env: FLReputationEnv
    ) -> None:
        """Uniform weights should have higher entropy penalty than discriminating weights."""
        # Uniform weights
        uniform = np.ones(NUM_CLIENTS, dtype=np.float32)
        _, info_uniform = env._compute_reward(uniform)

        # Discriminating weights (only honest get weight)
        discrim = np.zeros(NUM_CLIENTS, dtype=np.float32)
        discrim[~env._malicious_mask] = 1.0
        _, info_discrim = env._compute_reward(discrim)

        assert info_uniform["entropy_penalty"] > info_discrim["entropy_penalty"]

    def test_reward_zero_weights_guard(self) -> None:
        """All-zero weights should return -1.0 with attack_impact=1.0."""
        e = FLReputationEnv(max_rounds=10)
        e.reset(seed=42)

        weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        reward, info = e._compute_reward(weights)

        assert reward == pytest.approx(-1.0)
        assert info["weighted_accuracy"] == pytest.approx(0.0)
        assert info["attack_impact"] == pytest.approx(1.0)
        assert info["entropy_penalty"] == pytest.approx(0.0)


# =========================================================================
# 5. State Generation Tests
# =========================================================================


class TestStateGeneration:
    """Verify generated state distributions and clipping."""

    def test_initial_state_reputation_column(self, env: FLReputationEnv) -> None:
        """On reset, column 2 (historical_reputation) should be 0.5 for all clients."""
        obs, _ = env.reset(seed=42)
        np.testing.assert_allclose(obs[:, 2], 0.5, atol=1e-6)

    def test_honest_vs_malicious_accuracy_means(self) -> None:
        """Honest clients should have higher accuracy_contribution mean than malicious."""
        # Collect multiple resets for statistical significance
        e = FLReputationEnv(max_rounds=10)
        honest_acc = []
        malicious_acc = []

        for seed in range(50):
            obs, _ = e.reset(seed=seed)
            honest_acc.append(obs[~e._malicious_mask, 0].mean())
            malicious_acc.append(obs[e._malicious_mask, 0].mean())

        mean_honest = np.mean(honest_acc)
        mean_malicious = np.mean(malicious_acc)
        assert mean_honest > mean_malicious, (
            f"Honest accuracy mean ({mean_honest:.3f}) should be > "
            f"malicious ({mean_malicious:.3f})"
        )

    def test_honest_vs_malicious_similarity_means(self) -> None:
        """Honest clients should have higher gradient_similarity mean than malicious."""
        e = FLReputationEnv(max_rounds=10)
        honest_sim = []
        malicious_sim = []

        for seed in range(50):
            obs, _ = e.reset(seed=seed)
            honest_sim.append(obs[~e._malicious_mask, 1].mean())
            malicious_sim.append(obs[e._malicious_mask, 1].mean())

        assert np.mean(honest_sim) > np.mean(malicious_sim)

    def test_malicious_higher_update_magnitude(self) -> None:
        """Malicious clients should have higher update_magnitude (col 4) mean."""
        e = FLReputationEnv(max_rounds=10)
        honest_mag = []
        malicious_mag = []

        for seed in range(50):
            obs, _ = e.reset(seed=seed)
            honest_mag.append(obs[~e._malicious_mask, 4].mean())
            malicious_mag.append(obs[e._malicious_mask, 4].mean())

        mean_honest = np.mean(honest_mag)
        mean_malicious = np.mean(malicious_mag)
        assert mean_malicious > mean_honest, (
            f"Malicious magnitude mean ({mean_malicious:.3f}) should be > "
            f"honest ({mean_honest:.3f})"
        )

    def test_state_values_clipped(self, env: FLReputationEnv) -> None:
        """All state values must be in [0, 1]."""
        obs, _ = env.reset(seed=42)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_state_values_clipped_after_step(self, env: FLReputationEnv) -> None:
        """State values remain in [0, 1] after stepping."""
        action = np.full(NUM_CLIENTS, 0.5, dtype=np.float32)
        for _ in range(5):
            obs, _, _, _, _ = env.step(action)
            assert np.all(obs >= 0.0), "State has values below 0 after step"
            assert np.all(obs <= 1.0), "State has values above 1 after step"


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
        e = FLReputationEnv(max_rounds=200)
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
        e = FLReputationEnv(max_rounds=200)
        e.reset(seed=42)

        # Start reputation at 1.0 so any signal < 1.0 pulls it down
        e._state[:, 2] = 1.0
        initial_rep_malicious = e._state[e._malicious_mask, 2].copy()

        # Give weight only to malicious clients
        weights = np.zeros(NUM_CLIENTS, dtype=np.float32)
        weights[e._malicious_mask] = 1.0

        e._update_reputation(weights)
        updated_rep_malicious = e._state[e._malicious_mask, 2]

        # Malicious signal = 1 - 1/30 ~= 0.967, so EMA = 0.7*1.0 + 0.3*0.967 < 1.0
        assert updated_rep_malicious.mean() < initial_rep_malicious.mean(), (
            f"Malicious rep mean should decrease from 1.0: "
            f"{initial_rep_malicious.mean():.4f} -> {updated_rep_malicious.mean():.4f}"
        )

    def test_reputation_ema_alpha(self) -> None:
        """Verify the EMA uses the correct alpha (0.3) value."""
        e = FLReputationEnv(max_rounds=200)
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
        e = FLReputationEnv(max_rounds=50)
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
        e = FLReputationEnv(max_rounds=10)
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
        """Run max_rounds steps; last step must have terminated=True."""
        action = np.full(NUM_CLIENTS, 0.5, dtype=np.float32)
        terminated = False
        for _ in range(10):
            _, _, terminated, _, _ = short_env.step(action)
        assert terminated is True

    def test_episode_length_matches_max_rounds(self) -> None:
        """Count steps until terminated; must equal max_rounds."""
        max_r = 15
        e = FLReputationEnv(max_rounds=max_r)
        e.reset(seed=42)
        action = np.full(NUM_CLIENTS, 0.5, dtype=np.float32)

        steps = 0
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = e.step(action)
            steps += 1

        assert steps == max_r

    def test_gymnasium_check_env(self) -> None:
        """Gymnasium env_checker should not raise any errors."""
        from gymnasium.utils.env_checker import check_env

        e = FLReputationEnv(max_rounds=10)
        # check_env may emit warnings but should not raise
        check_env(e, skip_render_check=True)

    def test_cumulative_reward_in_info(self, short_env: FLReputationEnv) -> None:
        """Step info should track cumulative_reward across the episode."""
        action = np.full(NUM_CLIENTS, 0.5, dtype=np.float32)
        total = 0.0
        for _ in range(5):
            _, reward, _, _, info = short_env.step(action)
            total += reward
            assert "cumulative_reward" in info
            assert info["cumulative_reward"] == pytest.approx(total, abs=1e-5)

    def test_reset_after_episode(self) -> None:
        """Environment should be fully usable after reset following a complete episode."""
        e = FLReputationEnv(max_rounds=5)
        e.reset(seed=42)
        action = np.full(NUM_CLIENTS, 0.5, dtype=np.float32)

        # Run full episode
        for _ in range(5):
            e.step(action)

        # Reset and run again
        obs, info = e.reset(seed=99)
        assert obs.shape == (NUM_CLIENTS, NUM_FEATURES)
        assert info["round"] == 0

        obs2, _, terminated, _, _ = e.step(action)
        assert obs2.shape == (NUM_CLIENTS, NUM_FEATURES)
        assert terminated is False


# =========================================================================
# 8. Constructor Configuration Tests
# =========================================================================


class TestConstructorConfiguration:
    """Verify custom constructor parameters are respected."""

    def test_custom_alpha_beta(self) -> None:
        """Custom alpha/beta should affect reward calculation."""
        e1 = FLReputationEnv(alpha=0.9, beta=0.1, max_rounds=10)
        e2 = FLReputationEnv(alpha=0.1, beta=0.9, max_rounds=10)

        e1.reset(seed=42)
        e2.reset(seed=42)

        # Ensure identical state
        e2._state = e1._state.copy()
        e2._malicious_mask = e1._malicious_mask.copy()

        action = np.full(NUM_CLIENTS, 0.5, dtype=np.float32)
        _, r1, _, _, _ = e1.step(action)
        _, r2, _, _, _ = e2.step(action)

        # Different alpha/beta should produce different rewards
        assert r1 != pytest.approx(r2, abs=1e-6)

    def test_custom_malicious_fraction(self) -> None:
        """malicious_fraction=0.5 should produce 50 malicious clients."""
        e = FLReputationEnv(malicious_fraction=0.5, max_rounds=10)
        _, info = e.reset(seed=42)
        assert info["num_malicious"] == 50
        assert e._malicious_mask.sum() == 50

    def test_custom_max_rounds(self) -> None:
        """max_rounds=5 should terminate after exactly 5 steps."""
        e = FLReputationEnv(max_rounds=5)
        e.reset(seed=42)
        action = np.full(NUM_CLIENTS, 0.5, dtype=np.float32)

        for i in range(4):
            _, _, term, _, _ = e.step(action)
            assert term is False
        _, _, term, _, _ = e.step(action)
        assert term is True

    def test_zero_malicious_fraction(self) -> None:
        """malicious_fraction=0.0 should have 0 malicious clients and attack_impact=0."""
        e = FLReputationEnv(malicious_fraction=0.0, max_rounds=10)
        _, info = e.reset(seed=42)
        assert info["num_malicious"] == 0
        assert e._malicious_mask.sum() == 0

        weights = np.ones(NUM_CLIENTS, dtype=np.float32)
        _, reward_info = e._compute_reward(weights)
        assert reward_info["attack_impact"] == pytest.approx(0.0, abs=1e-8)

    def test_full_malicious_fraction(self) -> None:
        """malicious_fraction=1.0 should have all 100 clients malicious."""
        e = FLReputationEnv(malicious_fraction=1.0, max_rounds=10)
        _, info = e.reset(seed=42)
        assert info["num_malicious"] == 100
        assert e._malicious_mask.sum() == 100

    def test_default_parameters(self) -> None:
        """Default constructor should use alpha=0.6, beta=0.4, fraction=0.3, max_rounds=200."""
        e = FLReputationEnv()
        assert e.alpha == pytest.approx(0.6)
        assert e.beta == pytest.approx(0.4)
        assert e.malicious_fraction == pytest.approx(0.3)
        assert e.max_rounds == 200

    def test_render_mode_stored(self) -> None:
        """render_mode should be stored as an attribute."""
        e = FLReputationEnv(render_mode="human")
        assert e.render_mode == "human"

    def test_render_mode_none_default(self) -> None:
        """Default render_mode should be None."""
        e = FLReputationEnv()
        assert e.render_mode is None
