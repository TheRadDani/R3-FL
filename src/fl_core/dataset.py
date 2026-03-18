"""
Location: src/fl_core/dataset.py
Summary: FEMNIST dataset loading, non-IID partitioning via Dirichlet distribution,
         and CNN model definition for federated learning experiments.

Used by:
  - client.py: Uses partitioned DataLoaders and FemnistCNN model for local training.
  - server.py: Calls partitioning functions to create per-client datasets before
    launching the Flower simulation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 62  # EMNIST byclass: 10 digits + 26 upper + 26 lower
INPUT_CHANNELS: int = 1
IMAGE_SIZE: int = 28
DEFAULT_NUM_CLIENTS: int = 100
DEFAULT_DIRICHLET_ALPHA: float = 0.5
DEFAULT_BATCH_SIZE: int = 32


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_femnist_transforms() -> transforms.Compose:
    """Standard transforms for EMNIST byclass images (1x28x28 grayscale)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def load_femnist(
    data_dir: str = "./data",
    train: bool = True,
    download: bool = True,
) -> datasets.EMNIST:
    """Load the EMNIST byclass split (federated EMNIST / FEMNIST).

    Parameters
    ----------
    data_dir : str
        Root directory for dataset storage.
    train : bool
        If True, load the training split; otherwise the test split.
    download : bool
        If True, download the dataset when it is not found locally.

    Returns
    -------
    datasets.EMNIST
        The loaded EMNIST dataset with standard transforms applied.
    """
    transform = get_femnist_transforms()
    dataset = datasets.EMNIST(
        root=data_dir,
        split="byclass",
        train=train,
        download=download,
        transform=transform,
    )
    logger.info(
        "Loaded EMNIST byclass (%s): %d samples, %d classes",
        "train" if train else "test",
        len(dataset),
        NUM_CLASSES,
    )
    return dataset


# ---------------------------------------------------------------------------
# Non-IID Dirichlet partitioning
# ---------------------------------------------------------------------------

def partition_dataset_dirichlet(
    dataset: Dataset,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
    seed: int = 42,
) -> list[list[int]]:
    """Partition a dataset into *num_clients* shards using a Dirichlet distribution.

    For each class, a Dirichlet draw determines what fraction of that class's
    samples are assigned to each client.  Lower *alpha* yields more heterogeneous
    (non-IID) splits.

    Parameters
    ----------
    dataset : Dataset
        PyTorch dataset. Must expose a ``targets`` attribute (tensor or list).
    num_clients : int
        Number of client shards to create.
    alpha : float
        Dirichlet concentration parameter.  Values < 1.0 create more skewed
        distributions; alpha -> infinity approaches IID.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[list[int]]
        A list of length *num_clients*, where each element is a list of sample
        indices assigned to that client.

    Raises
    ------
    ValueError
        If *num_clients* < 1 or *alpha* <= 0.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)  # works for tensors and lists
    num_samples = len(targets)
    num_actual_classes = int(targets.max()) + 1

    # Pre-allocate index buckets per client
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in range(num_actual_classes):
        class_mask = np.where(targets == class_id)[0]
        if len(class_mask) == 0:
            continue

        # Dirichlet draw: proportions[j] = fraction of this class going to client j
        proportions = rng.dirichlet(np.full(num_clients, alpha))

        # Convert proportions to integer counts (ensure they sum correctly)
        counts = (proportions * len(class_mask)).astype(int)
        remainder = len(class_mask) - counts.sum()
        # Distribute remainder to clients with highest fractional parts
        fractional_parts = (proportions * len(class_mask)) - counts
        top_indices = np.argsort(fractional_parts)[-remainder:]
        counts[top_indices] += 1

        # Shuffle and assign
        rng.shuffle(class_mask)
        offset = 0
        for client_id in range(num_clients):
            end = offset + counts[client_id]
            client_indices[client_id].extend(class_mask[offset:end].tolist())
            offset = end

    # Log partition statistics
    sizes = [len(idx) for idx in client_indices]
    logger.info(
        "Dirichlet partition (alpha=%.2f): %d clients, "
        "samples per client min=%d, max=%d, mean=%.1f",
        alpha,
        num_clients,
        min(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
        np.mean(sizes) if sizes else 0.0,
    )

    return client_indices


def create_client_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    partition_indices: list[int],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for a single client.

    The test set is shared across all clients (standard FL evaluation), but the
    train set is restricted to the indices in *partition_indices*.

    Parameters
    ----------
    train_dataset : Dataset
        Full training dataset.
    test_dataset : Dataset
        Full test dataset (shared across clients).
    partition_indices : list[int]
        Indices into *train_dataset* for this client's local data.
    batch_size : int
        Mini-batch size for both loaders.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, test_loader) pair.
    """
    if not partition_indices:
        logger.warning("Creating DataLoader with empty partition — client has no data")

    train_subset = Subset(train_dataset, partition_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Label-flipped dataset wrapper (used by malicious clients)
# ---------------------------------------------------------------------------

class LabelFlippedDataset(Dataset):
    """Wraps a dataset and shifts all labels by +1 (mod num_classes).

    This is used by label-flipper malicious clients to poison the federated
    learning process transparently during data loading.

    Parameters
    ----------
    base_dataset : Dataset
        The underlying dataset whose labels will be flipped.
    num_classes : int
        Total number of classes (for modular arithmetic).
    """

    def __init__(self, base_dataset: Dataset, num_classes: int = NUM_CLASSES) -> None:
        self.base_dataset = base_dataset
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        flipped_label = (label + 1) % self.num_classes
        return image, flipped_label


# ---------------------------------------------------------------------------
# CNN Model for FEMNIST
# ---------------------------------------------------------------------------

class FemnistCNN(nn.Module):
    """Simple CNN for FEMNIST classification.

    Architecture:
        Conv2d(1, 32, 3) -> ReLU -> MaxPool(2)
        Conv2d(32, 64, 3) -> ReLU -> MaxPool(2)
        Flatten -> Linear(1600, 128) -> ReLU -> Dropout(0.25)
        Linear(128, 62)

    Input:  1x28x28 grayscale images
    Output: 62-class logits (EMNIST byclass)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # After 2x pool: 28 -> 14 -> 7, so feature map is 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images with shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Logits with shape (B, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (B, 64,  7,  7)
        x = x.view(x.size(0), -1)              # -> (B, 3136)
        x = self.dropout(F.relu(self.fc1(x)))   # -> (B, 128)
        x = self.fc2(x)                         # -> (B, 62)
        return x
