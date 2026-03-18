"""Triton and PyTorch kernels for RL reward normalization.

Provides a fused mean-std normalization kernel for running reward statistics.
When Triton is available, uses a JIT-compiled GPU kernel for minimal overhead.
Falls back to pure PyTorch when Triton is unavailable (CPU or CUDA without Triton).

Usage:
    from src.rl_agent.kernels import fused_reward_normalize, TRITON_AVAILABLE

    normalized = fused_reward_normalize(rewards_tensor, running_mean, running_std)
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Triton availability gate
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE: bool = True
except ImportError:
    TRITON_AVAILABLE = False

# ---------------------------------------------------------------------------
# Triton kernel (only compiled when Triton is present)
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[misc]
    def fused_reward_normalize_kernel(
        rewards_ptr,          # *float32  — input rewards
        out_ptr,              # *float32  — output (normalized)
        mean_val,             # float32   — running mean scalar
        std_val,              # float32   — running std scalar
        epsilon,              # float32   — numerical stability floor
        n_elements,           # int32     — total number of elements
        BLOCK_SIZE: tl.constexpr,  # compile-time block width
    ) -> None:
        """Fused element-wise reward normalization: out = (x - mean) / (std + eps).

        Each program instance handles a contiguous BLOCK_SIZE slice of the
        rewards buffer.  The mean and std are broadcast scalars (pre-computed
        outside the kernel on CPU) so no inter-block reduction is needed —
        the kernel is embarrassingly parallel.

        Args:
            rewards_ptr: Pointer to input reward tensor (float32).
            out_ptr:     Pointer to output tensor (float32, same shape).
            mean_val:    Running mean of rewards (scalar).
            std_val:     Running standard deviation of rewards (scalar).
            epsilon:     Small constant added to std to prevent division by zero.
            n_elements:  Total number of elements in the tensor.
            BLOCK_SIZE:  Compile-time constant block width (power of 2).
        """
        # Each kernel program handles one block of the array
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # Mask out-of-bounds lanes (last block may be smaller than BLOCK_SIZE)
        mask = offsets < n_elements

        # Load input values; out-of-bounds lanes read 0.0 (benign with mask)
        x = tl.load(rewards_ptr + offsets, mask=mask, other=0.0)

        # Fused normalization: (x - mean) / (std + epsilon)
        denom = std_val + epsilon
        out = (x - mean_val) / denom

        # Store result (masked to avoid writing out-of-bounds)
        tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper — dispatches to Triton kernel or PyTorch fallback
# ---------------------------------------------------------------------------


def fused_reward_normalize(
    rewards: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize a batch of rewards using a running mean and std.

    Dispatches to a Triton JIT kernel when available (GPU path) or falls back
    to pure PyTorch arithmetic (CPU or no-Triton path).

    The normalization is: ``out = (rewards - mean) / (std + epsilon)``

    Args:
        rewards: 1-D float32 tensor of reward values (any device).
        mean:    Running mean scalar (float or 0-d tensor).
        std:     Running std scalar (float or 0-d tensor).
        epsilon: Numerical stability floor added to std (default 1e-8).

    Returns:
        Normalized rewards tensor, same shape and device as input.
    """
    # Coerce scalar inputs for uniform downstream handling
    if isinstance(mean, torch.Tensor):
        mean_val = float(mean.item())
    else:
        mean_val = float(mean)

    if isinstance(std, torch.Tensor):
        std_val = float(std.item())
    else:
        std_val = float(std)

    if TRITON_AVAILABLE and rewards.is_cuda:
        return _triton_normalize(rewards, mean_val, std_val, epsilon)
    else:
        # Pure-PyTorch fallback — still vectorized, just not Triton-compiled
        return _torch_normalize(rewards, mean_val, std_val, epsilon)


def _triton_normalize(
    rewards: torch.Tensor,
    mean_val: float,
    std_val: float,
    epsilon: float,
) -> torch.Tensor:
    """Dispatch to the Triton fused kernel.

    Args:
        rewards:   1-D CUDA float32 tensor.
        mean_val:  Running mean as Python float.
        std_val:   Running std as Python float.
        epsilon:   Stability floor.

    Returns:
        Normalized tensor on the same CUDA device.
    """
    n = rewards.numel()
    out = torch.empty_like(rewards)  # pre-allocated output buffer — avoids GC pressure

    # Use 1024-wide blocks; Triton auto-tunes but this is a good default
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_reward_normalize_kernel[grid](
        rewards,
        out,
        mean_val,
        std_val,
        epsilon,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def _torch_normalize(
    rewards: torch.Tensor,
    mean_val: float,
    std_val: float,
    epsilon: float,
) -> torch.Tensor:
    """Fallback normalization using pure PyTorch.

    Equivalent result to the Triton kernel; runs on any device (CPU or CUDA).

    Args:
        rewards:   Reward tensor (any device/dtype compatible with float ops).
        mean_val:  Running mean as Python float.
        std_val:   Running std as Python float.
        epsilon:   Stability floor.

    Returns:
        Normalized tensor on the same device as input.
    """
    # fused sub + div in one expression — PyTorch will fuse these on CUDA
    return (rewards - mean_val) / (std_val + epsilon)


# ---------------------------------------------------------------------------
# Running statistics tracker — maintains mean and std across training steps
# ---------------------------------------------------------------------------


class RunningMeanStd:
    """Welford online algorithm for running mean and variance.

    Maintains exact mean and variance across an arbitrary number of batched
    updates without storing all values.  Used to feed stable mean/std into
    ``fused_reward_normalize``.

    Args:
        epsilon: Initial variance to avoid division-by-zero at startup.
        device:  Torch device to keep statistics on (default CPU; stats are
                 scalars so CPU is fine even when rewards are on GPU).
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        device: torch.device | str = "cpu",
    ) -> None:
        self.mean = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.var = torch.tensor(epsilon, device=device, dtype=torch.float32)
        self.count = torch.tensor(0, device=device, dtype=torch.long)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Update running statistics with a new batch of values.

        Uses Welford's online algorithm for numerical stability.

        Args:
            batch: 1-D tensor of new values (any device; stats stay on CPU).
        """
        # Move to CPU for scalar update — stats are scalars, no GPU needed
        b = batch.detach().float().cpu()
        batch_mean = b.mean()
        batch_var = b.var(unbiased=False)
        batch_count = torch.tensor(b.numel(), dtype=torch.long)

        if batch_count == 0:
            return

        # Welford parallel update
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self) -> torch.Tensor:
        """Current running standard deviation (clamped to avoid zero)."""
        return torch.sqrt(self.var).clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Quick self-test (run via: python -m src.rl_agent.kernels)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Triton available: {TRITON_AVAILABLE}")

    # Test with a simple batch on CPU (fallback path)
    rewards_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    rms = RunningMeanStd()
    rms.update(rewards_cpu)
    normalized = fused_reward_normalize(rewards_cpu, rms.mean, rms.std)
    print(f"CPU normalized rewards: {normalized.tolist()}")

    if torch.cuda.is_available():
        rewards_cuda = rewards_cpu.cuda()
        normalized_cuda = fused_reward_normalize(rewards_cuda, rms.mean, rms.std)
        print(f"CUDA normalized rewards: {normalized_cuda.cpu().tolist()}")
        print(f"Triton path used: {TRITON_AVAILABLE and rewards_cuda.is_cuda}")
