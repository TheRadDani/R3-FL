"""Flower simulation server for federated learning with malicious clients.

Location: src/fl_core/server.py
Summary: Orchestrates a multi-round FL simulation on FEMNIST dataset using FedAvg
         aggregation. Designates 30% of 100 clients as malicious (15 label flippers +
         15 noise injectors) to evaluate robustness against data poisoning attacks.

Purpose:
    Demonstrates end-to-end federated learning with adversarial robustness evaluation.
    Clients download global model parameters, train locally on their data partitions
    (honest or malicious), and send updates to the server for aggregation. The server
    combines updates via FedAvg and distributes the new global model for the next round.

Simulation Configuration:
    - **Clients**: 100 total (70 honest, 15 label flippers, 15 noise injectors)
    - **Rounds**: 10 (configurable via run_simulation() parameter)
    - **Data**: FEMNIST handwritten characters (non-IID partitioned via Dirichlet)
    - **Strategy**: FedAvg with 10% sample rate per round
    - **Device**: Auto-detects GPU and allocates fractional shares; falls back to CPU

Client Malicious Behavior:
    - **Label Flipper** (clients 0–14): Inverts training labels (y' = 9 - y)
    - **Noise Injector** (clients 15–29): Adds Gaussian noise to gradients
    - **Honest** (clients 30–99): Standard SGD training

Metrics:
    Per-round metrics logged:
    - Distributed losses (aggregated from client evaluations)
    - Distributed accuracy (weighted average across sampled clients)

Key Classes & Functions:
    - :class:`FlowerClient`: Clients with optional malicious behavior
    - :func:`run_simulation`: Main entry point; orchestrates all rounds
    - :func:`make_client_fn`: Factory for Flower's client_fn interface
    - :func:`_get_client_resources`: GPU/CPU allocation strategy
    - :func:`get_malicious_type`: Maps client ID to behavior type

Entry Points:
    CLI: ``python -m src.fl_core.server`` (from project root)
    Direct: ``from src.fl_core.server import run_simulation; history = run_simulation()``

Used by:
    - ``src/integration/strategy.py``: Reputation-based aggregation wrapper
    - ``tests/test_integration.py``: End-to-end integration tests

Dependencies:
    - :mod:`src.fl_core.dataset`: Dataset loading, partitioning, CNN model
    - :mod:`src.fl_core.client`: FlowerClient with malicious behavior support
    - ``flwr``: Flower framework for federated learning

Environment Variables:
    None required; optional Ray settings via RAY_* environment variables.

See Also:
    - :mod:`src.integration.strategy`: Wraps this server with reputation aggregation
    - :mod:`src.fl_core.client`: Client-side training logic
    - :mod:`src.fl_core.dataset`: Dataset and model definitions
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

def _get_client_resources() -> dict[str, float]:
    """Compute Ray resource allocations for simulated FL clients.

    Determines CPU and GPU resource budgets for the Flower simulation. Each
    simulated client is scheduled as a Ray actor and gets an allocation computed here.

    Resource Strategy:
        - **CPU**: Always 1 core per client (100 clients × 1 CPU = 100 CPUs needed).
          Ray maps these to physical cores with time-sharing on oversubscription.
        - **GPU**: If CUDA available, compute ``gpu_fraction = 1.0 / num_gpus``.
          Example: 1 GPU available → each client gets 0.5 GPU share; Ray handles
          scheduling 100 actors onto 1 GPU via time-slicing. For multi-GPU systems,
          fractions are computed to distribute load evenly.

    GPU Availability:
        - Detects CUDA GPUs via :func:`torch.cuda.is_available` and
          :func:`torch.cuda.device_count`
        - Logs available GPU device names for debugging

    Returns:
        dict[str, float]: Ray resource dict with keys:
            - ``"num_cpus"``: Always 1
            - ``"num_gpus"``: Fraction of GPU(s) per client (0.5–1.0 on 1 GPU; scales
              with gpu_count for multi-GPU). 0.0 if no CUDA available.

    Example:
        >>> resources = _get_client_resources()
        >>> if resources["num_gpus"] > 0:
        ...     print(f"GPU available: {resources['num_gpus']} per client")
        ... else:
        ...     print("Training on CPU")

    Note:
        The 1.0 (full GPU) allocation per client when only 1 GPU available ensures
        clients can batch-infer efficiently. Ray's scheduler handles serialization.
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
    """Determine the malicious behavior type for a given client ID.

    Maps client IDs to adversarial behavior types for evaluation of FL robustness.
    This deterministic assignment ensures reproducible scenarios; each client's
    role is fixed throughout the simulation.

    Client ID Ranges:
        - **0–14** (15 clients): ``label_flipper`` — inverts training labels
        - **15–29** (15 clients): ``noise_injector`` — adds Gaussian noise to gradients
        - **30–99** (70 clients): ``none`` (honest) — standard SGD training

    Behavior Details:
        - **Label Flipper**: Flips labels during local training (y' = num_classes - 1 - y)
          Creates data poison that gradually degrades global model accuracy
        - **Noise Injector**: Adds scaled Gaussian noise to computed gradients before upload
          Introduces gradient noise that disrupts convergence
        - **Honest**: Trains normally without manipulation

    Parameters
    ----------
    client_id : int
        Zero-based client identifier (0–99 for NUM_CLIENTS=100). Must be in valid range.

    Returns
    -------
    MaliciousType
        One of "label_flipper", "noise_injector", or "none" (literal type union).

    Raises:
        No explicit validation; returns "none" for any ID >= 30. Malicious clients
        outside [0, 29] are impossible; invalid IDs default to honest.

    Example:
        >>> get_malicious_type(5)
        'label_flipper'
        >>> get_malicious_type(20)
        'noise_injector'
        >>> get_malicious_type(50)
        'none'

    See Also:
        - :class:`FlowerClient`: Uses this to apply malicious behavior in training
        - :func:`make_client_fn`: Calls this for each worker
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
) -> dict[str, float]:
    """Aggregate client evaluation metrics using weighted sample-count averaging.

    Called by the Flower server after each evaluation round to combine per-client
    metrics into global metrics. Computes weighted averages where client weights
    are proportional to their local dataset sizes (num_samples).

    Aggregation Strategy:
        For each client metric (e.g., accuracy), compute:
        ``aggregated = sum(num_samples_i * metric_i) / sum(num_samples_i)``
        This weights larger clients more heavily, reflecting their influence on
        the global model performance.

    Parameters
    ----------
    metrics : list[tuple[int, dict]]
        List of (num_samples, metrics_dict) tuples from clients sampled in the round.
        - num_samples (int): Number of local training samples for this client
        - metrics_dict (dict): Client-reported metrics (e.g., {"accuracy": 0.92})
          Keys expected: "accuracy" (float in [0, 1])

    Returns
    -------
    dict[str, float]
        Aggregated metrics dict with keys:
        - "accuracy": Weighted-average accuracy across all clients
        Returns {"accuracy": 0.0} if metrics list is empty or total_samples is 0.

    Example:
        >>> metrics_from_clients = [
        ...     (1000, {"accuracy": 0.90}),
        ...     (500,  {"accuracy": 0.85}),
        ... ]
        >>> agg = weighted_average_metrics(metrics_from_clients)
        >>> print(agg["accuracy"])
        0.8833  # (1000*0.90 + 500*0.85) / 1500

    Note:
        Edge case handling:
        - Empty metrics list: Returns empty dict
        - Zero total samples: Returns {"accuracy": 0.0}
        - Missing "accuracy" key: Treated as 0.0 for that client
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
) -> callable:
    """Create a Flower client factory function for the simulation.

    Produces a closure that Flower's simulation engine calls to instantiate each
    simulated client at the start of training. The closure captures the full datasets
    and partition indices, and the returned ``client_fn`` function receives a Flower
    ``Context`` object that specifies which client (partition) to instantiate.

    Client Instantiation:
        For each client, this factory:
        1. Extracts partition ID from the Flower context
        2. Looks up malicious behavior type (label_flipper, noise_injector, or none)
        3. Creates a CNN model (:class:`FemnistCNN`)
        4. Builds DataLoaders for that client's data partition
        5. Wraps in :class:`FlowerClient` with malicious behavior support
        6. Converts to Flower :class:`Client` interface via ``.to_client()``

    Parameters
    ----------
    train_dataset : torch.utils.data.Dataset
        Full FEMNIST training dataset (shared read-only across clients).
        Individual clients receive subsets via partition_indices.
    test_dataset : torch.utils.data.Dataset
        Full FEMNIST test dataset (shared; clients evaluate on their own test data).
    partition_indices : list[list[int]]
        Per-client Dirichlet-partitioned index lists. Length = NUM_CLIENTS (100).
        Each element is a list of indices into train_dataset for that client.
    batch_size : int
        Mini-batch size for client DataLoaders (default 32, set by :const:`BATCH_SIZE`).

    Returns
    -------
    Callable[[flwr.common.Context], flwr.client.Client]
        A factory function with signature ``client_fn(context: Context) -> Client``.
        Flower calls this function once per simulated client, passing a context object
        with metadata (partition-id, node_config, etc.).

    Raises:
        ValueError: If context partition-id is out of range [0, NUM_CLIENTS).

    Example:
        >>> client_fn = make_client_fn(train_ds, test_ds, partitions)
        >>> # Flower calls this internally during simulation:
        >>> context = Context(node_config={"partition-id": "5"})
        >>> client = client_fn(context)
        >>> # client is now a FlowerClient for partition 5

    Note:
        The returned function is designed for Flower >= 1.13 (new simulation API).
        Earlier versions use NumPyClient; this factory uses the newer Client interface.

    See Also:
        - :func:`run_simulation`: Passes this factory to ``start_simulation``
        - :class:`FlowerClient`: The client implementation returned by the factory
        - :func:`get_malicious_type`: Determines client behavior type
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
    """Run a complete Flower federated learning simulation with malicious clients.

    Orchestrates a multi-round FL training session on FEMNIST. Manages data loading,
    client partitioning, strategy configuration, simulation execution, and results logging.

    Simulation Pipeline:
        1. **Data Loading**: Download and load FEMNIST (train/test splits)
        2. **Partitioning**: Dirichlet-partition training data across clients (non-IID)
        3. **Client Factory**: Create Flower client_fn that assigns malicious types
        4. **Strategy**: Configure FedAvg with 10% sampling, 5% evaluation sampling
        5. **Ray Simulation**: Launch distributed simulation with client_resources
        6. **Results**: Log and return per-round metrics (losses, accuracies)

    Data & Partitioning:
        - **Dataset**: FEMNIST (handwritten characters; ~60K train, ~10K test)
        - **Non-IID Distribution**: Dirichlet(alpha) partitioning across NUM_CLIENTS
          Higher alpha (closer to 1.0) = more homogeneous; lower = more heterogeneous
          Default alpha=0.5 creates significant data heterogeneity (realistic FL scenario)
        - **Seed**: Fixed seed ensures reproducible partitions across runs

    Malicious Clients (30% of total):
        - **Label Flippers** (15): Invert training labels (y' = 9 - y for FEMNIST)
        - **Noise Injectors** (15): Add Gaussian noise to gradients
        - **Honest** (70): Standard SGD training
        Assignment via :func:`get_malicious_type` (deterministic per client ID)

    Aggregation Strategy:
        - **Algorithm**: FedAvg (unweighted parameter averaging)
        - **Sampling**: 10% (10 clients) per round for training, 5% (5 clients) for evaluation
        - **Min clients**: 10 for fit, 5 for evaluate (required before proceeding)
        - **Metrics Aggregation**: Weighted accuracy via :func:`weighted_average_metrics`

    Parameters
    ----------
    num_clients : int
        Total number of simulated clients to create (default 100).
        Clients 0–14: label_flippers; 15–29: noise_injectors; 30–99: honest
    num_rounds : int
        Number of training rounds to execute (default 10).
        Each round: sample clients → local training → aggregation
    data_dir : str
        Directory for downloading/caching FEMNIST data (default "./data").
        If directory exists, data is not re-downloaded.
    alpha : float
        Dirichlet concentration parameter for non-IID partitioning (default 0.5).
        Controls heterogeneity: higher alpha = less heterogeneous, lower = more
    seed : int
        Random seed for partitioning and client initialization (default 42).
        Ensures reproducibility across runs.
    gpu_fraction : float | None
        Override GPU fraction per client in Ray simulation. If None (default),
        auto-detected via :func:`_get_client_resources`. Example: pass 0.1 to
        allocate 10% GPU to each client; pass 0.0 to force CPU-only.

    Returns
    -------
    flwr.server.history.History
        Flower History object with per-round results:
        - ``losses_centralized``: Server-side loss (if configured)
        - ``losses_distributed``: Aggregated client losses per round
        - ``metrics_distributed``: Aggregated accuracy per round
        Access via: ``history.metrics_distributed["accuracy"]`` (list of (round, acc) tuples)

    Raises:
        FileNotFoundError: If FEMNIST data cannot be downloaded or found

    Example:
        >>> history = run_simulation(num_rounds=10, alpha=0.5)
        >>> for rnd, acc in history.metrics_distributed.get("accuracy", []):
        ...     print(f"Round {rnd}: accuracy = {acc:.4f}")

    Simulation Logs:
        INFO-level logs for:
        - Detected GPU resources
        - Data loading and partitioning progress
        - Per-round metrics (losses, accuracies)
        - Overall simulation completion

    See Also:
        - :func:`make_client_fn`: Client factory function
        - :func:`get_malicious_type`: Malicious behavior assignment
        - :class:`FlowerClient`: Client implementation with malicious behavior
        - :func:`weighted_average_metrics`: Metrics aggregation callback
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
    """Log per-round metrics from the simulation history to INFO level.

    Formats and displays all available metrics from the Flower History object,
    including centralized and distributed losses, and distributed accuracy.
    Intended for human-readable post-simulation reporting.

    Metrics Displayed:
        - **Centralized Losses**: Server-side loss (if configured)
        - **Distributed Losses**: Aggregated client loss per round
        - **Distributed Accuracy**: Weighted client accuracy per round

    Parameters
    ----------
    history : fl.server.history.History
        Flower History object containing:
        - losses_centralized: List of (round, loss) tuples (or None)
        - losses_distributed: List of (round, loss) tuples
        - metrics_distributed: Dict[str, List[(round, value)]] (e.g., {"accuracy": [...]})

    Returns:
        None: Logs output via logger.info() at INFO level

    Example:
        >>> history = run_simulation()
        >>> _log_history(history)
        # Prints formatted losses and accuracies per round

    Note:
        Empty metrics are skipped (no logging for missing data).
        Formatted with 3 decimal places for loss, 4 for accuracy.
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
