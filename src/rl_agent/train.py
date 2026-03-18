"""Ray RLlib PPO training script for the FLReputationEnv.

Trains a PPO agent to learn optimal aggregation weights for a Federated
Learning system with adversarial clients.  The agent learns to upweight
honest clients and downweight malicious ones.

Usage:
    python -m src.rl_agent.train
    python -m src.rl_agent.train --iterations 100 --num-workers 4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Import the custom environment
from src.rl_agent.env import FLReputationEnv, NUM_CLIENTS, NUM_FEATURES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fl_reputation_train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIR: str = "checkpoints/fl_reputation_ppo"
"""Directory where training checkpoints are saved."""

ENV_NAME: str = "FLReputationEnv-v0"
"""Registered environment name for RLlib."""


def env_creator(env_config: dict) -> FLReputationEnv:
    """Factory function for RLlib environment registration.

    Args:
        env_config: Configuration dict passed from RLlib's ``env_config``.

    Returns:
        A new FLReputationEnv instance.
    """
    return FLReputationEnv(
        alpha=env_config.get("alpha", 0.6),
        beta=env_config.get("beta", 0.4),
        malicious_fraction=env_config.get("malicious_fraction", 0.3),
        max_rounds=env_config.get("max_rounds", 200),
    )


def build_ppo_config(
    num_workers: int = 2,
    env_config: dict | None = None,
) -> PPOConfig:
    """Build a PPOConfig for training on FLReputationEnv.

    Args:
        num_workers: Number of parallel rollout workers.
        env_config: Optional environment configuration overrides.

    Returns:
        A configured PPOConfig instance.
    """
    if env_config is None:
        env_config = {}

    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
            env_config=env_config,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=num_workers,
        )
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            lr=3e-4,
            lambda_=0.95,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=10.0,
            vf_loss_coeff=1.0,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=5,
        )
    )

    return config


def train(iterations: int = 50, num_workers: int = 2) -> None:
    """Run the PPO training loop.

    Args:
        iterations: Number of training iterations.
        num_workers: Number of parallel rollout workers.
    """
    # ------------------------------------------------------------------
    # Ray initialisation
    # ------------------------------------------------------------------
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialised.")

    # Register the custom environment with RLlib
    register_env(ENV_NAME, env_creator)
    logger.info("Registered environment: %s", ENV_NAME)

    # ------------------------------------------------------------------
    # Build algorithm
    # ------------------------------------------------------------------
    config = build_ppo_config(num_workers=num_workers)
    algo = config.build()
    logger.info("PPO algorithm built successfully.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    checkpoint_path = Path(CHECKPOINT_DIR)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_reward: float = float("-inf")
    best_checkpoint: str | None = None

    logger.info(
        "Starting training for %d iterations (workers=%d).",
        iterations,
        num_workers,
    )

    try:
        for i in range(1, iterations + 1):
            result = algo.train()

            # Extract metrics — handle both old and new RLlib result formats
            ep_reward_mean = (
                result.get("env_runners", {}).get("episode_reward_mean")
                or result.get("episode_reward_mean", float("nan"))
            )
            ep_len_mean = (
                result.get("env_runners", {}).get("episode_len_mean")
                or result.get("episode_len_mean", float("nan"))
            )

            logger.info(
                "Iter %3d/%d | reward_mean=%.4f | ep_len_mean=%.1f",
                i,
                iterations,
                ep_reward_mean,
                ep_len_mean,
            )

            # Track best reward
            if ep_reward_mean > best_reward:
                best_reward = ep_reward_mean

            # Periodic checkpoint
            if i % 10 == 0:
                save_result = algo.save(str(checkpoint_path))
                ckpt_dir = (
                    save_result.checkpoint.path
                    if hasattr(save_result, "checkpoint")
                    else str(save_result)
                )
                logger.info("Checkpoint saved: %s", ckpt_dir)
                best_checkpoint = ckpt_dir

        # Final checkpoint
        save_result = algo.save(str(checkpoint_path))
        ckpt_dir = (
            save_result.checkpoint.path
            if hasattr(save_result, "checkpoint")
            else str(save_result)
        )
        logger.info("Final checkpoint saved: %s", ckpt_dir)
        best_checkpoint = ckpt_dir

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    finally:
        logger.info("Best reward achieved: %.4f", best_reward)
        if best_checkpoint:
            logger.info("Best checkpoint: %s", best_checkpoint)

        algo.stop()
        logger.info("Algorithm stopped.")

        ray.shutdown()
        logger.info("Ray shut down. Training complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train PPO on FLReputationEnv for FL client reputation learning."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of PPO training iterations (default: 50).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel rollout workers (default: 2).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(iterations=args.iterations, num_workers=args.num_workers)
