"""
Location: src/fl_core/server.py
Summary: Flower simulation server for federated learning with malicious clients.
         Configures a FedAvg strategy, assigns 30% of 100 clients as malicious
         (15 label flippers + 15 noise injectors), and runs 10 rounds of training.

Used by:
  - Entry point: ``python -m src.fl_core.server`` or ``python src/fl_core/server.py``

Dependencies:
  - dataset.py: Dataset loading, partitioning, and CNN model definition.
  - client.py: FlowerClient with malicious behavior support.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import torch

import flwr as fl
from flwr.client import Client
from flwr.common import Context
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

try:
    from .dataset import (
        FemnistCNN,
        create_client_dataloaders,
        load_femnist,
        partition_dataset_dirichlet,
    )
    from .client import FlowerClient, MaliciousType
except ImportError:
    # Allow running as `python src/fl_core/server.py` from project root
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.fl_core.dataset import (
        FemnistCNN,
        create_client_dataloaders,
        load_femnist,
        partition_dataset_dirichlet,
    )
    from src.fl_core.client import FlowerClient, MaliciousType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

NUM_CLIENTS: int = 100
NUM_ROUNDS: int = 10
FRACTION_FIT: float = 0.1        # 10% of clients sampled per round
FRACTION_EVALUATE: float = 0.05  # 5% of clients sampled for evaluation
MIN_FIT_CLIENTS: int = 10
MIN_EVALUATE_CLIENTS: int = 5
MIN_AVAILABLE_CLIENTS: int = NUM_CLIENTS

# Malicious client allocation (30% total: 15 label flippers + 15 noise injectors)
NUM_LABEL_FLIPPERS: int = 15
NUM_NOISE_INJECTORS: int = 15

# Data parameters
DATA_DIR: str = "./data"
DIRICHLET_ALPHA: float = 0.5
BATCH_SIZE: int = 32
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# GPU resource detection
# ---------------------------------------------------------------------------

def _get_client_resources() -> dict:
    """Return Ray client resources based on GPU availability.

    Each simulated client gets 1 CPU. If a CUDA GPU is available, clients
    share it — Ray allocates ``1 / num_gpus`` fraction per client so all 100
    can run concurrently on a single GPU using time-slicing.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_fraction = 1.0 / num_gpus  # one full GPU shared across clients
        logger.info(
            "GPU detected: %d device(s). Allocating %.4f GPU fraction per client.",
            num_gpus,
            gpu_fraction,
        )
        return {"num_cpus": 1, "num_gpus": gpu_fraction}
    logger.info("No GPU detected — running on CPU.")
    return {"num_cpus": 1, "num_gpus": 0.0}


# ---------------------------------------------------------------------------
# Malicious type assignment
# ---------------------------------------------------------------------------

def get_malicious_type(client_id: int) -> MaliciousType:
    """Determine the malicious behavior type for a given client.

    Client ID assignment:
      - 0  .. 14  → label_flipper   (15 clients)
      - 15 .. 29  → noise_injector  (15 clients)
      - 30 .. 99  → none / honest   (70 clients)

    Parameters
    ----------
    client_id : int
        Zero-based client identifier.

    Returns
    -------
    MaliciousType
        One of 'label_flipper', 'noise_injector', or 'none'.
    """
    if client_id < NUM_LABEL_FLIPPERS:
        return "label_flipper"
    elif client_id < NUM_LABEL_FLIPPERS + NUM_NOISE_INJECTORS:
        return "noise_injector"
    return "none"


# ---------------------------------------------------------------------------
# Server-side metric aggregation callback
# ---------------------------------------------------------------------------

def weighted_average_metrics(
    metrics: list[tuple[int, dict]],
) -> dict:
    """Aggregate evaluation metrics from multiple clients using weighted average.

    Parameters
    ----------
    metrics : list[tuple[int, dict]]
        List of (num_samples, metrics_dict) tuples from each client.

    Returns
    -------
    dict
        Aggregated metrics with weighted accuracy.
    """
    if not metrics:
        return {}

    total_samples = sum(num for num, _ in metrics)
    if total_samples == 0:
        return {"accuracy": 0.0}

    weighted_accuracy = sum(
        num * m.get("accuracy", 0.0) for num, m in metrics
    )
    return {"accuracy": weighted_accuracy / total_samples}


# ---------------------------------------------------------------------------
# Client factory for Flower simulation
# ---------------------------------------------------------------------------

def make_client_fn(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    partition_indices: list[list[int]],
    batch_size: int = BATCH_SIZE,
):
    """Create a ``client_fn`` closure for the Flower simulation.

    The returned function has the signature ``client_fn(context: Context) -> Client``
    as required by ``flwr.simulation.start_simulation`` (flwr >= 1.13).

    Parameters
    ----------
    train_dataset : Dataset
        Full training dataset.
    test_dataset : Dataset
        Full test dataset (shared across clients).
    partition_indices : list[list[int]]
        Per-client index lists from Dirichlet partitioning.
    batch_size : int
        Mini-batch size for client DataLoaders.

    Returns
    -------
    Callable[[Context], Client]
        A factory function that Flower calls to instantiate each client.
    """

    def client_fn(context: Context) -> Client:
        """Instantiate a FlowerClient for the given simulation context."""
        # Extract partition ID from context (set by Flower simulation runtime)
        partition_id = int(context.node_config["partition-id"])

        # Validate partition ID
        if partition_id < 0 or partition_id >= len(partition_indices):
            raise ValueError(
                f"partition_id {partition_id} out of range "
                f"[0, {len(partition_indices)})"
            )

        malicious_type = get_malicious_type(partition_id)
        model = FemnistCNN()

        train_loader, test_loader = create_client_dataloaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            partition_indices=partition_indices[partition_id],
            batch_size=batch_size,
        )

        flower_client = FlowerClient(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            malicious_type=malicious_type,
            client_id=partition_id,
        )

        # Convert NumPyClient to Client for the simulation API
        return flower_client.to_client()

    return client_fn


# ---------------------------------------------------------------------------
# Simulation entry point
# ---------------------------------------------------------------------------

def run_simulation(
    num_clients: int = NUM_CLIENTS,
    num_rounds: int = NUM_ROUNDS,
    data_dir: str = DATA_DIR,
    alpha: float = DIRICHLET_ALPHA,
    seed: int = RANDOM_SEED,
    gpu_fraction: float | None = None,
) -> fl.server.history.History:
    """Run the full Flower federated learning simulation.

    Parameters
    ----------
    num_clients : int
        Total number of simulated clients.
    num_rounds : int
        Number of federated training rounds.
    data_dir : str
        Path to dataset storage directory.
    alpha : float
        Dirichlet concentration parameter for non-IID partitioning.
    seed : int
        Random seed for reproducibility.
    gpu_fraction : float | None
        GPU fraction to allocate per client in the Ray simulation. When
        ``None`` (default), :func:`_get_client_resources` auto-detects GPU
        availability. Pass an explicit value (e.g. ``0.1``) to override.

    Returns
    -------
    flwr.server.history.History
        Object containing per-round metrics (losses, accuracies).
    """
    logger.info("=" * 60)
    logger.info("Federated Learning Simulation — FEMNIST + FedAvg")
    logger.info("=" * 60)
    logger.info(
        "Clients: %d total (%d label_flippers, %d noise_injectors, %d honest)",
        num_clients,
        NUM_LABEL_FLIPPERS,
        NUM_NOISE_INJECTORS,
        num_clients - NUM_LABEL_FLIPPERS - NUM_NOISE_INJECTORS,
    )
    logger.info("Rounds: %d, Dirichlet alpha: %.2f", num_rounds, alpha)

    # ---- 1. Load datasets ------------------------------------------------
    logger.info("Loading FEMNIST datasets...")
    train_dataset = load_femnist(data_dir=data_dir, train=True)
    test_dataset = load_femnist(data_dir=data_dir, train=False)

    # ---- 2. Partition training data (non-IID) ----------------------------
    logger.info("Partitioning training data across %d clients...", num_clients)
    partition_indices = partition_dataset_dirichlet(
        dataset=train_dataset,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    # ---- 3. Create client factory ----------------------------------------
    client_fn = make_client_fn(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        partition_indices=partition_indices,
        batch_size=BATCH_SIZE,
    )

    # ---- 4. Configure FedAvg strategy ------------------------------------
    strategy = FedAvg(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )

    # ---- 5. Run simulation -----------------------------------------------
    logger.info("Starting Flower simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=(
            {"num_cpus": 1, "num_gpus": gpu_fraction}
            if gpu_fraction is not None
            else _get_client_resources()
        ),
    )

    # ---- 6. Log results --------------------------------------------------
    _log_history(history)

    return history


def _log_history(history: fl.server.history.History) -> None:
    """Log per-round metrics from the simulation history.

    Parameters
    ----------
    history : History
        Flower History object containing losses and metrics.
    """
    logger.info("=" * 60)
    logger.info("Simulation Results")
    logger.info("=" * 60)

    # Centralized losses (from server-side evaluation if configured)
    if history.losses_centralized:
        logger.info("Centralized losses per round:")
        for rnd, loss in history.losses_centralized:
            logger.info("  Round %3d: loss = %.4f", rnd, loss)

    # Distributed losses (aggregated from client evaluations)
    if history.losses_distributed:
        logger.info("Distributed losses per round:")
        for rnd, loss in history.losses_distributed:
            logger.info("  Round %3d: loss = %.4f", rnd, loss)

    # Distributed accuracy
    if history.metrics_distributed:
        accuracy_key = "accuracy"
        if accuracy_key in history.metrics_distributed:
            logger.info("Distributed accuracy per round:")
            for rnd, acc in history.metrics_distributed[accuracy_key]:
                logger.info("  Round %3d: accuracy = %.4f", rnd, acc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configure logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    logger.info("Starting FL simulation (smoke test)...")
    try:
        history = run_simulation()
        logger.info("Simulation completed successfully.")
    except Exception:
        logger.exception("Simulation failed with an error")
        sys.exit(1)
