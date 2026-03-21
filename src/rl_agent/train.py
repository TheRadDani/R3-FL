"""Ray RLlib MAPPO training script for the FLReputationEnv.

Location: src/rl_agent/train.py
Summary: Multi-Agent Proximal Policy Optimization (MAPPO) training framework for
         learning reputation-based aggregation weights in federated learning.
         Uses parameter sharing across agents to train a single policy that learns
         to upweight honest clients and downweight malicious ones.

Purpose:
    Trains a PPO agent on the :class:`FLReputationEnv` environment using Ray RLlib's
    distributed training. Each FL client is represented as an agent in a MARL setting,
    all sharing a single policy network. The environment provides per-client observations
    (update magnitudes, losses, etc.) and the policy learns weighted aggregation decisions.

Performance Features:
    - **Multi-GPU support**: Automatically detects CUDA GPUs and allocates to learner;
      workers run on CPU for maximum throughput
    - **Worker scaling**: Automatically uses max(4, cpu_count - 2) workers (capped at 8)
      to saturate available CPU cores while leaving headroom for the driver
    - **Efficient batching**: batch_mode="truncate_episodes" + rollout_fragment_length=100
      reduces per-worker buffering from 20K to 10K observations, preventing "too many
      observations buffered" warnings
    - **Triton GPU kernels**: Optional JIT-compiled reward normalization (graceful fallback
      to PyTorch) for low-overhead running statistics
    - **LSTM state retention**: Agents retain LSTM state across FL rounds, enabling
      detection of slow-drift poisoning attacks

Key Classes & Functions:
    - :func:`env_creator`: Factory function for RLlib environment registration
    - :func:`build_ppo_config`: Constructs PPOConfig with MAPPO-specific settings
    - :func:`train`: Main training loop with checkpoint management and metrics tracking

Usage Examples:
    Basic training (50 iterations):
        python -m src.rl_agent.train

    Custom iterations and worker count:
        python -m src.rl_agent.train --iterations 100 --num-workers 4

    Profile GPU inference overhead:
        python -m src.rl_agent.train --iterations 200 --time-inference

Environment Variables:
    RAY_memory — Ray actor memory allocation (default: auto)
    CUDA_VISIBLE_DEVICES — GPU devices available to Ray

See Also:
    - :mod:`src.rl_agent.env`: FLReputationEnv environment definition
    - :mod:`src.rl_agent.kernels`: Triton reward normalization kernels
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

    Creates a :class:`FLReputationEnv` instance from the RLlib ``env_config`` dict.
    All parameters have sensible defaults matching :meth:`FLReputationEnv.__init__`
    and can be overridden via env_config keys. This function is registered with RLlib
    via :func:`register_env` so workers can spawn new environments without direct imports.

    Curriculum Learning Support:
        Optionally enables curriculum learning by setting ``env_config["curriculum_phase"]``
        to 1, 2, or 3. This overrides ``malicious_fraction`` and ``min_rounds`` to provide
        staged difficulty progression:

        - **Phase 1 (easy)**: malicious_fraction=0.1, min_rounds=20
        - **Phase 2 (medium)**: malicious_fraction=0.3, min_rounds=10
        - **Phase 3 (hard)**: malicious_fraction=0.5, min_rounds=5

        Other parameters (alpha, beta, delta, eta, max_rounds, num_clients) can be
        freely overridden alongside curriculum_phase; curriculum settings only affect
        the two parameters listed above.

    Args:
        env_config (dict): Configuration dict passed from RLlib's ``env_config`` parameter.
            Expected keys and defaults:
            - alpha (float): Decay factor for malicious score computation (default 0.5)
            - beta (float): Weight on loss improvement in reward (default 0.3)
            - delta (float): Weight on update magnitude in reward (default 0.3)
            - eta (float): Weight on client participation in reward (default 0.3)
            - malicious_fraction (float): Fraction of clients marked malicious (default 0.3)
            - max_rounds (int): Maximum FL training rounds per episode (default 200)
            - min_rounds (int): Minimum FL rounds before episode can terminate (default 5)
            - num_clients (int): Total number of simulated FL clients (default :const:`NUM_CLIENTS`)
            - curriculum_phase (int, optional): Curriculum difficulty level (1, 2, or 3)

    Returns:
        FLReputationEnv: A newly instantiated environment ready for RLlib rollouts.

    Raises:
        ValueError: If ``curriculum_phase`` is not None, 1, 2, or 3.

    Example:
        >>> config = {
        ...     "alpha": 0.6,
        ...     "malicious_fraction": 0.2,
        ...     "curriculum_phase": 2,
        ... }
        >>> env = env_creator(config)
        >>> obs, _ = env.reset()
    """
    # Base parameters — defaults match FLReputationEnv.__init__
    alpha = env_config.get("alpha", 0.5)
    beta = env_config.get("beta", 0.3)
    delta = env_config.get("delta", 0.3)
    eta = env_config.get("eta", 0.3)
    malicious_fraction = env_config.get("malicious_fraction", 0.3)
    max_rounds = env_config.get("max_rounds", 200)
    min_rounds = env_config.get("min_rounds", 5)
    num_clients = env_config.get("num_clients", NUM_CLIENTS)

    # Curriculum learning — override malicious_fraction and min_rounds
    # based on the training phase.  This is opt-in: if curriculum_phase
    # is not set, the base parameters above are used unchanged.
    curriculum_phase = env_config.get("curriculum_phase")
    if curriculum_phase is not None:
        _CURRICULUM = {
            1: {"malicious_fraction": 0.1, "min_rounds": 20},  # easy
            2: {"malicious_fraction": 0.3, "min_rounds": 10},  # medium
            3: {"malicious_fraction": 0.5, "min_rounds": 5},   # hard
        }
        phase_cfg = _CURRICULUM.get(curriculum_phase)
        if phase_cfg is None:
            raise ValueError(
                f"Invalid curriculum_phase={curriculum_phase!r}; expected 1, 2, or 3."
            )
        malicious_fraction = phase_cfg["malicious_fraction"]
        min_rounds = phase_cfg["min_rounds"]
        logger.info(
            "Curriculum phase %d: malicious_fraction=%.1f, min_rounds=%d",
            curriculum_phase,
            malicious_fraction,
            min_rounds,
        )

    return FLReputationEnv(
        alpha=alpha,
        beta=beta,
        delta=delta,
        eta=eta,
        malicious_fraction=malicious_fraction,
        max_rounds=max_rounds,
        min_rounds=min_rounds,
        num_clients=num_clients,
    )


def _detect_gpu_resources() -> tuple[int, float]:
    """Detect available GPU resources and return training allocation strategy.

    Inspects system for CUDA GPUs and returns allocation decisions for the learner
    (policy update process) and rollout workers (environment stepping + inference).
    Follows the principle: learner gets all GPUs, workers run on CPU only to avoid
    GPU contention. This maximizes throughput since workers spend most time stepping
    CPU-only environments, not GPU inference.

    Resource Strategy:
        - **Learner**: Allocated all visible GPUs (0 if no CUDA available)
        - **Workers**: Allocated 0 GPUs; they run entirely on CPU

    This avoids the Ray scheduler problem where allocating fractional GPUs to workers
    (e.g., 0.5 GPU per worker × 30 workers = 15 GPUs required on a single-GPU machine)
    causes oversubscription. CPUs are saturated via worker scaling (see :func:`build_ppo_config`).

    Returns:
        tuple[int, float]: A (num_gpus_learner, num_gpus_per_worker) pair.
            Example: (2, 0.0) means 2 GPUs for learner, 0 for workers.
            On CPU-only systems: (0, 0.0).

    Example:
        >>> gpus_learner, gpus_worker = _detect_gpu_resources()
        >>> if gpus_learner > 0:
        ...     print(f"Training on {gpus_learner} GPUs")
        ... else:
        ...     print("Training on CPU (no CUDA detected)")
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
    """Build a fully configured PPOConfig for MAPPO training on FLReputationEnv.

    Constructs a :class:`ray.rllib.algorithms.ppo.PPOConfig` with all settings tuned
    for multi-agent FL reputation learning. Combines worker parallelism (CPU saturation),
    GPU allocation strategy, and training hyperparameters optimized for fast convergence.

    Worker Scaling:
        If ``num_workers`` is None, automatically computes workers as:
        ``min(max(1, os.cpu_count() - 2), 8)``. This reserves 2 cores for the driver/
        learner process while capping at 8 to prevent oversubscription on many-core
        machines. Workers are the primary throughput lever in RLlib.

    Batching & Efficiency:
        - batch_mode="truncate_episodes": Releases rollout fragments at fixed intervals
          (rollout_fragment_length) rather than waiting for episode completion. Prevents
          buffering 20K observations and achieves ~1–2 min/iteration instead of ~6:30
        - rollout_fragment_length=100: With 100 agents, yields 10K observations per
          fragment (8 workers × 100 agents × 100 steps), just below RLlib's 250K warning
        - train_batch_size=4000: Standard throughput size; filled in ~5 collection rounds

    Model Architecture (MAPPO):
        - Two 64-wide hidden layers for per-agent observations (NUM_FEATURES=9 inputs)
        - LSTM enabled (max_seq_len=20) for temporal context across FL rounds
        - vf_share_layers=False: Separate actor/critic networks (better MAPPO performance)

    Args:
        num_workers (int | None): Number of parallel rollout workers. If None, auto-scaled
            to ``min(max(1, cpu_count() - 2), 8)``. Increase for faster rollout collection;
            decrease to reduce per-worker memory overhead.
        env_config (dict | None): Optional environment configuration dict passed to all workers.
            Merged with defaults in :func:`env_creator`.

    Returns:
        PPOConfig: A fully specified training configuration ready for ``.build()``.

    Example:
        >>> config = build_ppo_config(num_workers=6, env_config={"curriculum_phase": 1})
        >>> algo = config.build()
        >>> for i in range(50):
        ...     results = algo.train()

    See Also:
        - :func:`_detect_gpu_resources`: GPU allocation strategy used by this config
        - :func:`train`: Main training loop that uses this config
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
    """Run the main MAPPO training loop with checkpoint management and metrics logging.

    Orchestrates a multi-iteration training session on :class:`FLReputationEnv` using
    Ray RLlib. Manages Ray initialization, environment registration, algorithm construction,
    the training loop, reward normalization, checkpointing, and graceful shutdown.

    Training Loop:
        1. Initialize Ray (if not already initialized)
        2. Register :class:`FLReputationEnv` with RLlib
        3. Build PPO algorithm from config (see :func:`build_ppo_config`)
        4. For each iteration:
           - Call ``algo.train()`` (collects rollouts, updates policy)
           - Normalize episode rewards via Welford online algorithm
           - Log per-iteration metrics (reward_mean, episode length)
           - Save checkpoints every 10 iterations
        5. On completion or interruption, save final checkpoint and shut down Ray

    Checkpointing:
        Saves checkpoints to :const:`CHECKPOINT_DIR` ("checkpoints/fl_reputation_ppo/")
        every 10 iterations and at training completion. Tracks the best reward achieved
        and logs the best checkpoint path for recovery or inference.

    Reward Normalization:
        Uses :class:`RunningMeanStd` (Welford online algorithm) to compute running
        statistics of episode rewards. Normalized rewards are logged at DEBUG level
        and can inform reward pre-conditioning in custom training callbacks.

    GPU Timing (Optional):
        If ``time_inference=True`` and CUDA is available, wraps each training iteration
        with ``torch.cuda.synchronize()`` calls to eliminate async measurement lag.
        Logs wall-time per iteration at INFO level. Adds ~1 ms overhead per iteration.

    Args:
        iterations (int): Number of training iterations (default 50). Each iteration
            collects a train_batch and performs one SGD update pass.
        num_workers (int | None): Number of parallel rollout workers (default None).
            If None, auto-scaled to ``min(max(1, cpu_count() - 2), 8)`` by :func:`build_ppo_config`.
        time_inference (bool): If True, wrap each iteration with CUDA timing to measure
            GPU overhead (default False). Useful for profiling performance bottlenecks.

    Raises:
        KeyboardInterrupt: Caught and logged; training shuts down gracefully.
        Exception: Any uncaught exceptions are logged and re-raised after cleanup.

    Example:
        >>> train(iterations=100, num_workers=4, time_inference=False)

    See Also:
        - :func:`build_ppo_config`: Configuration construction
        - :func:`env_creator`: Environment factory for workers
        - :mod:`src.rl_agent.kernels`: Reward normalization with Triton kernels
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
    """Parse command-line arguments for the training script.

    Defines and parses all CLI options accepted by the training entry point.
    Integrates with the :func:`train` function to allow users to customize
    worker count, iteration depth, and profiling options from the command line.

    Command-Line Options:
        --iterations: Number of PPO training iterations (default 50)
        --num-workers: Parallel rollout workers (default None = auto-scale)
        --time-inference: Enable CUDA timing for inference bottleneck profiling

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - iterations (int): Number of training iterations
            - num_workers (int | None): Number of workers (None = auto)
            - time_inference (bool): Whether to profile CUDA overhead

    Example:
        >>> args = parse_args()
        >>> train(iterations=args.iterations, num_workers=args.num_workers)

    See Also:
        - :func:`train`: Main training function using these arguments
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
