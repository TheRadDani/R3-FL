"""Ray RLlib MAPPO training script for the FLReputationEnv.

Trains a Multi-Agent PPO (MAPPO) system with parameter sharing to learn
optimal aggregation weights for a Federated Learning system with adversarial
clients.  Each agent corresponds to one FL client and shares a single policy
network, learning to upweight honest clients and downweight malicious ones.

Performance features:
    - Multi-GPU: num_gpus = torch.cuda.device_count() (graceful CPU fallback)
    - Worker scaling: num_env_runners = min(max(1, os.cpu_count() - 2), 8)
    - Workers collect rollouts on CPU only; full GPU is reserved for the learner
    - Mixed-precision hints via torch.set_default_dtype(torch.float32)
    - CUDA timing around policy inference (exposed via --time-inference flag)
    - Triton reward normalization kernel imported with graceful fallback
    - Batch sizes tuned for throughput (train_batch=4000, minibatch=256)
    - batch_mode=truncate_episodes + rollout_fragment_length=100 drops the per-worker
      observation buffer to 100 × 100 agents = 10,000 obs per fragment, just below the 250K RLlib warning threshold

Usage:
    python -m src.rl_agent.train
    python -m src.rl_agent.train --iterations 100 --num-workers 4
    python -m src.rl_agent.train --iterations 200 --time-inference
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import ray
import torch
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

# Import the custom environment
from src.rl_agent.env import FLReputationEnv, NUM_CLIENTS, NUM_FEATURES

# Import Triton reward normalization kernel (graceful fallback to PyTorch)
from src.rl_agent.kernels import (
    TRITON_AVAILABLE,
    RunningMeanStd,
    fused_reward_normalize,
)

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
# Force float32 globally — prevents silent float64 promotion in PyTorch ops
# which wastes VRAM and slows CUDA kernels.  Set before Ray/RLlib initializes.
# ---------------------------------------------------------------------------
torch.set_default_dtype(torch.float32)

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
        num_clients=env_config.get("num_clients", NUM_CLIENTS),
    )


def _detect_gpu_resources() -> tuple[int, float]:
    """Detect available GPU resources and return training allocation.

    Allocates all visible GPUs to the learner.  Rollout workers get a fractional
    share (0.5 GPU each) so they can batch-infer on GPU without wasting a full
    device per worker.  Falls back gracefully to CPU when no GPU is present.

    Returns:
        (num_gpus_learner, num_gpus_per_worker) tuple.
    """
    if not torch.cuda.is_available():
        logger.info("No CUDA GPUs detected — training on CPU.")
        return 0, 0.0

    n_gpus = torch.cuda.device_count()
    logger.info(
        "Detected %d CUDA GPU(s): %s",
        n_gpus,
        [torch.cuda.get_device_name(i) for i in range(n_gpus)],
    )

    # Give all GPUs to the learner; workers run rollouts on CPU only.
    # Allocating GPU fractions to workers on a single-GPU machine causes Ray to
    # demand more GPUs than exist (e.g. 30 workers × 0.5 = 15 GPUs required).
    # Workers do not need GPU for environment stepping; keeping them CPU-only
    # leaves the full GPU budget for the learner's policy-update step.
    return n_gpus, 0.0


def build_ppo_config(
    num_workers: int | None = None,
    env_config: dict | None = None,
) -> PPOConfig:
    """Build a PPOConfig for training on FLReputationEnv.

    Args:
        num_workers: Number of parallel rollout workers.  Defaults to
            max(4, os.cpu_count() - 2) for CPU-saturating rollouts.
        env_config: Optional environment configuration overrides.

    Returns:
        A configured PPOConfig instance.
    """
    if env_config is None:
        env_config = {}

    # ------------------------------------------------------------------ #
    # Worker count: saturate available CPUs while leaving 2 cores for the
    # driver/learner process.  At least 4 workers for meaningful parallelism.
    # ------------------------------------------------------------------ #
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        # Leave 2 cores for the driver/learner; cap at 8 for laptop-class machines
        # to avoid spawning more Ray workers than the OS can schedule efficiently.
        num_workers = min(max(1, cpu_count - 2), 8)
    logger.info("Configuring %d rollout workers.", num_workers)

    # ------------------------------------------------------------------ #
    # GPU allocation
    # ------------------------------------------------------------------ #
    num_gpus_learner, num_gpus_per_worker = _detect_gpu_resources()

    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
            env_config=env_config,
        )
        # Explicit framework="torch" — required for AMP and custom CUDA kernels.
        # RLlib defaults may vary by version; always set explicitly.
        .framework("torch")
        .resources(
            # num_gpus allocates GPUs to the learner (policy update) process
            num_gpus=num_gpus_learner,
        )
        .env_runners(
            # Scale workers to CPU count for maximum rollout throughput
            num_env_runners=num_workers,
            # rollout_fragment_length=100: explicit value instead of "auto".
            #
            # Key insight: fragment_length (100) < max_rounds (200), so with
            # batch_mode="truncate_episodes" each episode is cut at step 100
            # before it can complete.  The per-episode observation buffer never
            # grows past 100 env-steps × 100 agents = 10,000 agent-obs, just
            # below the 250K RLlib warning threshold.
            #
            # Collection speed: 8 workers × 100 env-steps = 800 env-steps per
            # round.  To fill train_batch_size=4000 env-steps only 5 collection
            # rounds are needed (4000 / 800 = 5), keeping worker round-trips low
            # and restoring ~1-2 min/iter throughput.  By contrast, a fragment
            # length of 5 required 100 collection rounds (4000 / (8×5) = 100),
            # which multiplied worker communication overhead ~20× (~6:30/iter).
            rollout_fragment_length=100,
            # truncate_episodes: release rollout fragments at rollout_fragment_length
            # boundaries instead of waiting for full episode completion.
            # With 100 agents × 200 steps, complete_episodes forced buffering
            # 20,000 agent-observations before any update — causing the
            # "More than 20000 observations buffered" warning and ~1:48/iter.
            # truncate_episodes is safe here because reward is per-step (not
            # episode-terminal), so the PPO value target is valid at any truncation.
            batch_mode="truncate_episodes",
            # num_envs_per_env_runner=1: explicit document of the implicit default.
            # Prevents future RLlib version defaults from silently changing behaviour.
            num_envs_per_env_runner=1,
            # num_gpus_per_env_runner replaces the deprecated num_gpus_per_worker
            # parameter (previously passed via .resources()) for rollout workers.
            num_gpus_per_env_runner=num_gpus_per_worker,
        )
        .training(
            # train_batch_size=4000: kept from original config.
            # With truncate_episodes and rollout_fragment_length=100, each of
            # the 8 workers delivers 100 env steps (100 × 100 agents = 10,000
            # agent-obs).  5 collection rounds × 8 workers = 4,000 env-steps
            # total per batch.  SGD update size unchanged; only collection is fast.
            train_batch_size=4000,
            # sgd_minibatch_size=256 balances gradient noise vs. throughput
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            lr=5e-4,
            lambda_=0.95,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff=0.005,
            vf_clip_param=50.0,
            vf_loss_coeff=1.0,
            grad_clip=40.0,
            model={
                # fcnet_hiddens sized to per-agent observation space: NUM_FEATURES = 9 inputs per agent (parameter-shared MAPPO)
                # Two 64-wide hidden layers provide sufficient capacity without over-parameterizing
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
                # Enable LSTM so the agent retains memory of client behaviour across FL rounds.
                # max_seq_len=20 means the LSTM unrolls over 20 consecutive rounds per training
                # sequence, giving the policy a wide enough window to detect slow-drift poisoning
                # attacks (e.g., clients that gradually increase update magnitudes over many rounds).
                "use_lstm": True,
                "max_seq_len": 20,
                # vf_share_layers=False uses separate networks for policy and value heads.
                # MAPPO benefits from decoupled actor/critic: the value function sees
                # global context and needs different representations than the policy head.
                "vf_share_layers": False,
            },
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=5,
        )
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    observation_space=spaces.Box(0.0, 1.0, shape=(NUM_FEATURES,), dtype=np.float32),
                    action_space=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
    )

    return config


def train(
    iterations: int = 50,
    num_workers: int | None = None,
    time_inference: bool = False,
) -> None:
    """Run the PPO training loop.

    Args:
        iterations:      Number of training iterations.
        num_workers:     Number of parallel rollout workers (None = auto from CPU count).
        time_inference:  If True, wrap algo.train() with CUDA timing to expose
                         per-iteration GPU bottlenecks.  Adds ~1 ms overhead per iter.
    """
    # ------------------------------------------------------------------
    # Log Triton status so operator knows which reward normalization path is active
    # ------------------------------------------------------------------
    logger.info(
        "Triton kernel available: %s (reward normalization path: %s)",
        TRITON_AVAILABLE,
        "triton" if TRITON_AVAILABLE else "pytorch-fallback",
    )

    # ------------------------------------------------------------------
    # Running reward normalizer — tracks mean/std across iterations for the
    # fused_reward_normalize() call below
    # ------------------------------------------------------------------
    reward_rms = RunningMeanStd(epsilon=1e-4, device="cpu")

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
        "Starting training for %d iterations.",
        iterations,
    )

    try:
        for i in range(1, iterations + 1):
            # ----------------------------------------------------------
            # CUDA timing gate — expose inference bottleneck when requested.
            # torch.cuda.synchronize() ensures all preceding CUDA ops are
            # flushed before starting the clock (avoids measuring async lag).
            # ----------------------------------------------------------
            if time_inference and torch.cuda.is_available():
                torch.cuda.synchronize()
                t_start = time.perf_counter()

            result = algo.train()

            if time_inference and torch.cuda.is_available():
                torch.cuda.synchronize()  # flush async CUDA ops before reading clock
                t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                logger.info("  [CUDA timing] iter %d wall time: %.2f ms", i, t_elapsed_ms)

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

            # ----------------------------------------------------------
            # Reward normalization via fused kernel (or PyTorch fallback).
            # We normalize the running reward for logging/monitoring purposes;
            # the normalizer's mean/std can also be used to pre-condition
            # rewards before policy updates in a custom training callback.
            # ----------------------------------------------------------
            if not (ep_reward_mean != ep_reward_mean):  # nan-safe check
                reward_batch = torch.tensor([ep_reward_mean], dtype=torch.float32)
                reward_rms.update(reward_batch)
                normalized_reward = fused_reward_normalize(
                    reward_batch, reward_rms.mean, reward_rms.std
                )
                logger.debug(
                    "  Normalized reward (z-score): %.4f",
                    normalized_reward.item(),
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
        default=None,
        help="Number of parallel rollout workers (default: auto from CPU count).",
    )
    parser.add_argument(
        "--time-inference",
        action="store_true",
        default=False,
        help="Wrap each training iteration with CUDA synchronize timing to expose GPU bottlenecks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        iterations=args.iterations,
        num_workers=args.num_workers,
        time_inference=args.time_inference,
    )
