"""
Location: src/fl_core/client.py
Summary: Flower federated learning client implementation with support for honest
         and malicious (label-flipper, noise-injector) client behaviors.

Used by:
  - server.py: The ``client_fn`` factory instantiates FlowerClient via this module
    inside the Flower simulation loop.

Dependencies:
  - dataset.py: Uses FemnistCNN model, LabelFlippedDataset wrapper, and DataLoaders.
"""

from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from typing import Literal

import numpy as np
import torch
import torch.amp
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import DataLoader

from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from .dataset import FemnistCNN, LabelFlippedDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for malicious behaviors
# ---------------------------------------------------------------------------

MaliciousType = Literal["none", "label_flipper", "noise_injector"]

# ---------------------------------------------------------------------------
# Training hyper-parameters (local to each client)
# ---------------------------------------------------------------------------

LOCAL_EPOCHS: int = 1
LEARNING_RATE: float = 0.01
NOISE_SCALE: float = 10.0  # Gaussian noise std for noise_injector clients

# cudnn.benchmark lets cuDNN auto-select the fastest convolution algorithm for
# the fixed input sizes used in FEMNIST (28x28).  Set once at module load.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# FlowerClient
# ---------------------------------------------------------------------------

class FlowerClient(NumPyClient):
    """Federated learning client compatible with Flower's NumPyClient interface.

    Supports three behavioral modes:
      - ``'none'``: Honest client — trains and reports faithfully.
      - ``'label_flipper'``: Shifts all training labels by +1 (mod num_classes)
        via a transparent dataset wrapper applied at construction time.
      - ``'noise_injector'``: Trains normally but adds massive Gaussian noise
        (std=10.0) to model parameters *after* local training, before returning
        them to the server.

    Parameters
    ----------
    model : FemnistCNN
        The CNN model instance (architecture must match server-side model).
    train_loader : DataLoader
        DataLoader for this client's local training data.
    test_loader : DataLoader
        DataLoader for evaluation data (typically shared test set).
    malicious_type : MaliciousType
        One of ``'none'``, ``'label_flipper'``, or ``'noise_injector'``.
    client_id : int
        Numeric identifier for this client (used in logging).
    device : torch.device | None
        Device to train on. Defaults to CUDA if available, else CPU.
    use_gradient_checkpointing : bool
        When True, applies ``torch.utils.checkpoint.checkpoint_sequential``
        to the convolutional feature extractor during training.  This trades
        ~30% more compute for a significant reduction in activation memory,
        which is valuable when simulating many clients concurrently.
    """

    def __init__(
        self,
        model: FemnistCNN,
        train_loader: DataLoader,
        test_loader: DataLoader,
        malicious_type: MaliciousType = "none",
        client_id: int = 0,
        device: torch.device | None = None,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.test_loader = test_loader
        self.malicious_type = malicious_type
        self.client_id = client_id
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Mixed-precision training support (CUDA only)
        self._use_amp = self.device.type == "cuda"
        self._grad_scaler = torch.amp.GradScaler("cuda") if self._use_amp else None

        # Gradient checkpointing: recomputes activations on backward pass instead
        # of storing them, trading ~30% extra compute for significantly lower
        # peak activation memory — useful when 50-100 clients run concurrently.
        self._use_gradient_checkpointing = use_gradient_checkpointing

        # For label flippers, wrap the train dataset transparently
        if malicious_type == "label_flipper":
            flipped_dataset = LabelFlippedDataset(train_loader.dataset)
            self.train_loader = DataLoader(
                flipped_dataset,
                batch_size=train_loader.batch_size or 32,
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                num_workers=2,
                prefetch_factor=2,
            )
            logger.info("Client %d: label_flipper — labels shifted by +1", client_id)
        else:
            self.train_loader = train_loader

        if malicious_type == "noise_injector":
            logger.info(
                "Client %d: noise_injector — will add Gaussian noise (std=%.1f) "
                "to parameters after training",
                client_id,
                NOISE_SCALE,
            )

    # ------------------------------------------------------------------
    # NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return current model parameters as a list of NumPy arrays.

        Uses ``detach().cpu()`` before ``.numpy()`` to avoid holding a reference
        to the computation graph and to ensure the tensor is on CPU memory
        before the zero-copy numpy view is created.
        """
        return [
            # detach() drops grad_fn before moving to CPU, preventing accidental
            # graph retention; contiguous() ensures the numpy view is valid
            val.detach().cpu().contiguous().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Load parameters received from the server into the local model."""
        state_keys = list(self.model.state_dict().keys())
        if len(parameters) != len(state_keys):
            raise ValueError(
                f"Parameter count mismatch: received {len(parameters)}, "
                f"model expects {len(state_keys)}"
            )
        state_dict = OrderedDict(
            {k: torch.as_tensor(v, device=self.device) for k, v in zip(state_keys, parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train the model on local data and return updated parameters.

        For noise_injector clients, Gaussian noise is added to ALL model
        parameters after training completes.

        Parameters
        ----------
        parameters : NDArrays
            Global model parameters from the server.
        config : dict[str, Scalar]
            Configuration dict (may contain 'local_epochs', 'lr').

        Returns
        -------
        tuple[NDArrays, int, dict[str, Scalar]]
            (updated_parameters, num_training_samples, metrics_dict).
        """
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        lr = float(config.get("lr", LEARNING_RATE))

        num_samples = self._train(local_epochs, lr)

        # Noise injection: add massive Gaussian noise to parameters AFTER training
        if self.malicious_type == "noise_injector":
            self._inject_noise()

        updated_params = self.get_parameters(config={})

        # Reclaim fragmented CUDA memory after each FL round.  With 50-100
        # concurrent clients each holding optimizer states and activations,
        # accumulated fragmentation causes OOM before physical memory is full.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        metrics: dict[str, Scalar] = {
            "client_id": self.client_id,
            "malicious_type": self.malicious_type,
        }
        return updated_params, num_samples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the model on the test set.

        Parameters
        ----------
        parameters : NDArrays
            Global model parameters from the server.
        config : dict[str, Scalar]
            Configuration dict (unused).

        Returns
        -------
        tuple[float, int, dict[str, Scalar]]
            (loss, num_test_samples, {"accuracy": float}).
        """
        self.set_parameters(parameters)
        loss, accuracy, num_samples = self._evaluate()
        return loss, num_samples, {"accuracy": accuracy}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train(self, epochs: int, lr: float) -> int:
        """Run local SGD training and return the number of training samples seen.

        Parameters
        ----------
        epochs : int
            Number of local epochs.
        lr : float
            Learning rate for SGD optimizer.

        Returns
        -------
        int
            Total number of training samples processed.
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        total_samples = 0

        for epoch in range(epochs):
            running_loss = torch.tensor(0.0, device=self.device)
            batch_count = 0
            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                if self._use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        if self._use_gradient_checkpointing:
                            # checkpoint_sequential recomputes activations during
                            # backward instead of storing them, reducing peak
                            # activation memory at ~30% extra FLOPs cost.
                            # Segment count=2 splits conv1→pool and conv2→pool.
                            conv_seq = nn.Sequential(
                                self.model.conv1,
                                nn.ReLU(),
                                self.model.pool,
                                self.model.conv2,
                                nn.ReLU(),
                                self.model.pool,
                            )
                            features = torch.utils.checkpoint.checkpoint_sequential(
                                conv_seq, segments=2, input=images,
                                use_reentrant=False,
                            )
                            features = features.view(features.size(0), -1)
                            features = self.model.dropout(
                                torch.nn.functional.relu(self.model.fc1(features))
                            )
                            outputs = self.model.fc2(features)
                        else:
                            outputs = self.model(images)
                        loss = criterion(outputs, labels)
                    self._grad_scaler.scale(loss).backward()
                    self._grad_scaler.step(optimizer)
                    self._grad_scaler.update()
                else:
                    if self._use_gradient_checkpointing:
                        conv_seq = nn.Sequential(
                            self.model.conv1,
                            nn.ReLU(),
                            self.model.pool,
                            self.model.conv2,
                            nn.ReLU(),
                            self.model.pool,
                        )
                        features = torch.utils.checkpoint.checkpoint_sequential(
                            conv_seq, segments=2, input=images,
                            use_reentrant=False,
                        )
                        features = features.view(features.size(0), -1)
                        features = self.model.dropout(
                            torch.nn.functional.relu(self.model.fc1(features))
                        )
                        outputs = self.model.fc2(features)
                    else:
                        outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.detach()
                batch_count += 1
                total_samples += len(images)

            if batch_count > 0:
                # Single GPU->CPU sync per epoch instead of per batch
                epoch_loss = running_loss.item() / batch_count
                logger.debug(
                    "Client %d epoch %d/%d — avg loss: %.4f",
                    self.client_id,
                    epoch + 1,
                    epochs,
                    epoch_loss,
                )

        return total_samples

    def _inject_noise(self) -> None:
        """Add massive Gaussian noise to all model parameters (noise_injector attack)."""
        with torch.no_grad():
            for param in self.model.parameters():
                param.data.add_(torch.randn_like(param.data), alpha=NOISE_SCALE)
        logger.debug(
            "Client %d: injected Gaussian noise (std=%.1f) into %d parameter tensors",
            self.client_id,
            NOISE_SCALE,
            sum(1 for _ in self.model.parameters()),
        )

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float, int]:
        """Evaluate the model on the test set.

        Returns
        -------
        tuple[float, float, int]
            (average_loss, accuracy, num_samples).
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.test_loader:
            # non_blocking=True overlaps H2D transfer with prior GPU work
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            total_loss += criterion(outputs, labels).item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

        if total == 0:
            logger.warning("Client %d: evaluation on empty test set", self.client_id)
            return 0.0, 0.0, 0

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy, total
