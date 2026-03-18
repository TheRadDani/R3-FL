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

import logging
from collections import OrderedDict
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
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
    """

    def __init__(
        self,
        model: FemnistCNN,
        train_loader: DataLoader,
        test_loader: DataLoader,
        malicious_type: MaliciousType = "none",
        client_id: int = 0,
        device: torch.device | None = None,
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

        # For label flippers, wrap the train dataset transparently
        if malicious_type == "label_flipper":
            flipped_dataset = LabelFlippedDataset(train_loader.dataset)
            self.train_loader = DataLoader(
                flipped_dataset,
                batch_size=train_loader.batch_size or 32,
                shuffle=True,
                drop_last=False,
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
        """Return current model parameters as a list of NumPy arrays."""
        return [
            val.cpu().numpy()
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
            {k: torch.tensor(v) for k, v in zip(state_keys, parameters)}
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
            epoch_loss = 0.0
            batch_count = 0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                total_samples += len(images)

            if batch_count > 0:
                logger.debug(
                    "Client %d epoch %d/%d — avg loss: %.4f",
                    self.client_id,
                    epoch + 1,
                    epochs,
                    epoch_loss / batch_count,
                )

        return total_samples

    def _inject_noise(self) -> None:
        """Add massive Gaussian noise to all model parameters (noise_injector attack)."""
        with torch.no_grad():
            for param in self.model.parameters():
                param.data += torch.randn_like(param.data) * NOISE_SCALE
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
            images = images.to(self.device)
            labels = labels.to(self.device)

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
