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
from typing import Iterator, Optional

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

# Number of background workers for DataLoader I/O
_DATALOADER_NUM_WORKERS: int = 2
# How many batches to prefetch ahead per worker
_DATALOADER_PREFETCH_FACTOR: int = 2


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

    _cuda_available = torch.cuda.is_available()

    train_subset = Subset(train_dataset, partition_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        # pin_memory enables async CPU→GPU DMA transfer via page-locked host memory
        pin_memory=_cuda_available,
        num_workers=_DATALOADER_NUM_WORKERS,
        # prefetch_factor: each worker pre-fetches this many batches ahead of consumption
        prefetch_factor=_DATALOADER_PREFETCH_FACTOR if _DATALOADER_NUM_WORKERS > 0 else None,
        # persistent_workers keeps worker processes alive between rounds, avoiding
        # the fork/join overhead that dominates multi-worker startup time
        persistent_workers=_DATALOADER_NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        # pin_memory enables async CPU→GPU DMA transfer via page-locked host memory
        pin_memory=_cuda_available,
        num_workers=_DATALOADER_NUM_WORKERS,
        prefetch_factor=_DATALOADER_PREFETCH_FACTOR if _DATALOADER_NUM_WORKERS > 0 else None,
        # persistent_workers avoids worker respawn between evaluation calls
        persistent_workers=_DATALOADER_NUM_WORKERS > 0,
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
# PrefetchLoader: overlaps CPU→GPU transfer with GPU computation
# ---------------------------------------------------------------------------


class PrefetchLoader:
    """Wraps a DataLoader to overlap CPU→GPU memory transfer with GPU compute.

    On CUDA devices the next batch is transferred to GPU memory asynchronously
    (via a non-blocking copy on a dedicated CUDA stream) while the current
    batch is being processed.  This eliminates PCIe transfer latency from the
    critical path.

    On CPU-only machines this class is a transparent passthrough with no
    overhead.

    Parameters
    ----------
    loader : DataLoader
        The underlying DataLoader to wrap.
    device : torch.device
        Target device.  When ``device.type != 'cuda'`` the wrapper is a no-op.

    Example
    -------
    >>> loader = DataLoader(dataset, batch_size=32, pin_memory=True)
    >>> fast_loader = PrefetchLoader(loader, device=torch.device("cuda"))
    >>> for images, labels in fast_loader:
    ...     # images and labels are already on GPU
    ...     pass
    """

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        # Only meaningful on CUDA; CPU path is a passthrough
        self._use_prefetch = device.type == "cuda"

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if not self._use_prefetch:
            # CPU path: yield batches directly, move to device synchronously
            for batch in self.loader:
                images, labels = batch
                yield images.to(self.device), labels.to(self.device)
            return

        # CUDA path: dedicated stream for async H2D transfers
        stream = torch.cuda.Stream()
        loader_iter = iter(self.loader)

        # Seed the pipeline with the first batch
        try:
            images, labels = next(loader_iter)
        except StopIteration:
            return

        # Pre-fetch first batch onto GPU asynchronously
        with torch.cuda.stream(stream):
            # non_blocking=True: returns immediately; transfer happens in background
            next_images = images.to(self.device, non_blocking=True)
            next_labels = labels.to(self.device, non_blocking=True)

        for batch in loader_iter:
            # Yield the already-transferred batch (compute can overlap next H2D)
            torch.cuda.current_stream().wait_stream(stream)
            current_images, current_labels = next_images, next_labels

            images, labels = batch
            with torch.cuda.stream(stream):
                next_images = images.to(self.device, non_blocking=True)
                next_labels = labels.to(self.device, non_blocking=True)

            yield current_images, current_labels

        # Yield the final prefetched batch
        torch.cuda.current_stream().wait_stream(stream)
        yield next_images, next_labels


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
