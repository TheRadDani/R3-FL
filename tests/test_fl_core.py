"""
Comprehensive pytest tests for the FL core module (src/fl_core/).

Covers:
  - FemnistCNN model architecture (shape, params, gradient flow)
  - Dirichlet dataset partitioning (correctness, non-IID, determinism)
  - LabelFlippedDataset (label shift, data preservation)
  - FlowerClient (honest, label_flipper, noise_injector behaviors)

All tests use small synthetic datasets -- no network downloads required.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl_core.dataset import (
    NUM_CLASSES,
    FemnistCNN,
    LabelFlippedDataset,
    create_client_dataloaders,
    partition_dataset_dirichlet,
)
from src.fl_core.client import (
    NOISE_SCALE,
    FlowerClient,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture()
def model():
    """Fresh FemnistCNN model instance with deterministic init."""
    torch.manual_seed(42)
    return FemnistCNN(num_classes=NUM_CLASSES)


@pytest.fixture()
def synthetic_train_dataset():
    """Small synthetic training dataset: 200 random 1x28x28 images, labels 0-61."""
    torch.manual_seed(42)
    images = torch.randn(200, 1, 28, 28)
    labels = torch.randint(0, NUM_CLASSES, (200,))
    return TensorDataset(images, labels)


@pytest.fixture()
def synthetic_test_dataset():
    """Small synthetic test dataset: 50 random 1x28x28 images, labels 0-61."""
    torch.manual_seed(99)
    images = torch.randn(50, 1, 28, 28)
    labels = torch.randint(0, NUM_CLASSES, (50,))
    return TensorDataset(images, labels)


@pytest.fixture()
def train_loader(synthetic_train_dataset):
    """DataLoader wrapping the synthetic training dataset."""
    return DataLoader(synthetic_train_dataset, batch_size=32, shuffle=False)


@pytest.fixture()
def test_loader(synthetic_test_dataset):
    """DataLoader wrapping the synthetic test dataset."""
    return DataLoader(synthetic_test_dataset, batch_size=32, shuffle=False)


@pytest.fixture()
def honest_client(model, train_loader, test_loader):
    """Honest FlowerClient (malicious_type='none')."""
    return FlowerClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        malicious_type="none",
        client_id=0,
        device=torch.device("cpu"),
    )


@pytest.fixture()
def label_flipper_client(train_loader, test_loader):
    """Label-flipper FlowerClient."""
    torch.manual_seed(42)
    m = FemnistCNN(num_classes=NUM_CLASSES)
    return FlowerClient(
        model=m,
        train_loader=train_loader,
        test_loader=test_loader,
        malicious_type="label_flipper",
        client_id=1,
        device=torch.device("cpu"),
    )


@pytest.fixture()
def noise_injector_client(train_loader, test_loader):
    """Noise-injector FlowerClient."""
    torch.manual_seed(42)
    m = FemnistCNN(num_classes=NUM_CLASSES)
    return FlowerClient(
        model=m,
        train_loader=train_loader,
        test_loader=test_loader,
        malicious_type="noise_injector",
        client_id=2,
        device=torch.device("cpu"),
    )


@pytest.fixture()
def partition_dataset():
    """Synthetic dataset with 1000 samples and a .targets attribute for partitioning."""
    torch.manual_seed(42)
    images = torch.randn(1000, 1, 28, 28)
    labels = torch.randint(0, 10, (1000,))
    ds = TensorDataset(images, labels)
    # partition_dataset_dirichlet expects a .targets attribute
    ds.targets = labels
    return ds


# =====================================================================
# 1. FemnistCNN Model Tests
# =====================================================================


class TestFemnistCNN:
    """Tests for the FemnistCNN architecture."""

    def test_model_output_shape(self, model):
        """Forward pass with (1, 1, 28, 28) input produces (1, 62) output."""
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES), (
            f"Expected output shape (1, {NUM_CLASSES}), got {out.shape}"
        )

    def test_model_output_shape_batch(self, model):
        """Forward pass with a batch of 8 images produces (8, 62) output."""
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, NUM_CLASSES)

    def test_model_parameter_count(self, model):
        """Model has the expected number of trainable parameters.

        Architecture breakdown:
          conv1: (1*32*3*3) + 32          = 320
          conv2: (32*64*3*3) + 64         = 18496
          fc1:   (3136*128) + 128         = 401536
          fc2:   (128*62) + 62            = 7998
          Total:                          = 428350
        """
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params == 428350, (
            f"Expected 428350 trainable parameters, got {total_params}"
        )

    def test_model_forward_backward(self, model):
        """Verify gradient flow: loss.backward() produces non-None gradients on all params."""
        x = torch.randn(4, 1, 28, 28)
        labels = torch.randint(0, NUM_CLASSES, (4,))
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, labels)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for parameter: {name}"

    def test_model_different_num_classes(self):
        """Model can be instantiated with a different number of output classes."""
        m = FemnistCNN(num_classes=10)
        x = torch.randn(1, 1, 28, 28)
        out = m(x)
        assert out.shape == (1, 10)


# =====================================================================
# 2. Dataset Partitioning Tests
# =====================================================================


class TestPartitionDatasetDirichlet:
    """Tests for the Dirichlet-based non-IID dataset partitioning."""

    def test_partition_returns_correct_client_count(self, partition_dataset):
        """partition_dataset_dirichlet returns exactly num_clients lists."""
        num_clients = 10
        partitions = partition_dataset_dirichlet(
            partition_dataset, num_clients=num_clients, alpha=0.5, seed=42
        )
        assert len(partitions) == num_clients

    def test_partition_covers_all_indices(self, partition_dataset):
        """Union of all partitions covers the entire dataset (no missing indices)."""
        partitions = partition_dataset_dirichlet(
            partition_dataset, num_clients=10, alpha=0.5, seed=42
        )
        all_indices = set()
        for client_indices in partitions:
            all_indices.update(client_indices)
        expected = set(range(len(partition_dataset)))
        assert all_indices == expected, (
            f"Missing indices: {expected - all_indices}, "
            f"Extra indices: {all_indices - expected}"
        )

    def test_partition_no_overlap(self, partition_dataset):
        """No index appears in multiple client partitions."""
        partitions = partition_dataset_dirichlet(
            partition_dataset, num_clients=10, alpha=0.5, seed=42
        )
        all_indices = []
        for client_indices in partitions:
            all_indices.extend(client_indices)
        assert len(all_indices) == len(set(all_indices)), "Duplicate indices found across clients"

    def test_partition_non_iid_property(self, partition_dataset):
        """With low alpha (0.1), partitions should be non-IID.

        Measures label distribution variance across clients: with very low alpha,
        at least some clients should have highly skewed class distributions
        (i.e., dominated by a small number of classes).
        """
        partitions = partition_dataset_dirichlet(
            partition_dataset, num_clients=10, alpha=0.1, seed=42
        )
        targets = np.array(partition_dataset.targets)
        num_classes = int(targets.max()) + 1

        # For each client, compute the fraction of each class
        distributions = []
        for client_indices in partitions:
            if len(client_indices) == 0:
                continue
            client_labels = targets[client_indices]
            class_counts = np.bincount(client_labels, minlength=num_classes)
            dist = class_counts / class_counts.sum()
            distributions.append(dist)

        # Measure non-IIDness: max class fraction across clients
        # In a non-IID setting (low alpha), at least one client should have
        # a dominant class fraction > 0.3 (vs ~0.1 for IID with 10 classes)
        max_fractions = [d.max() for d in distributions]
        assert max(max_fractions) > 0.3, (
            f"Expected non-IID partitioning with alpha=0.1: max class fraction "
            f"across clients was only {max(max_fractions):.3f}"
        )

    def test_partition_deterministic(self, partition_dataset):
        """Same seed produces identical partitions."""
        p1 = partition_dataset_dirichlet(partition_dataset, num_clients=10, alpha=0.5, seed=42)
        p2 = partition_dataset_dirichlet(partition_dataset, num_clients=10, alpha=0.5, seed=42)
        for i, (a, b) in enumerate(zip(p1, p2)):
            assert a == b, f"Client {i} partitions differ with same seed"

    def test_partition_different_seeds(self, partition_dataset):
        """Different seeds produce different partitions."""
        p1 = partition_dataset_dirichlet(partition_dataset, num_clients=10, alpha=0.5, seed=42)
        p2 = partition_dataset_dirichlet(partition_dataset, num_clients=10, alpha=0.5, seed=123)
        # At least some clients should have different index sets
        differences = sum(
            1 for a, b in zip(p1, p2) if set(a) != set(b)
        )
        assert differences > 0, "Different seeds produced identical partitions"

    def test_partition_invalid_num_clients(self, partition_dataset):
        """ValueError raised for num_clients < 1."""
        with pytest.raises(ValueError, match="num_clients must be >= 1"):
            partition_dataset_dirichlet(partition_dataset, num_clients=0)

    def test_partition_invalid_alpha(self, partition_dataset):
        """ValueError raised for alpha <= 0."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            partition_dataset_dirichlet(partition_dataset, num_clients=10, alpha=0.0)

    def test_partition_single_client(self, partition_dataset):
        """With num_clients=1, the single client gets all indices."""
        partitions = partition_dataset_dirichlet(
            partition_dataset, num_clients=1, alpha=0.5, seed=42
        )
        assert len(partitions) == 1
        assert set(partitions[0]) == set(range(len(partition_dataset)))


# =====================================================================
# 3. LabelFlippedDataset Tests
# =====================================================================


class TestLabelFlippedDataset:
    """Tests for the LabelFlippedDataset wrapper."""

    def test_label_flipping_shifts_labels(self):
        """Original label 0 -> 1, label 61 -> 0 (wraps around mod 62)."""
        images = torch.randn(5, 1, 28, 28)
        labels = torch.tensor([0, 1, 30, 60, 61])
        base = TensorDataset(images, labels)
        flipped = LabelFlippedDataset(base, num_classes=NUM_CLASSES)

        expected = [1, 2, 31, 61, 0]
        for i, exp in enumerate(expected):
            _, flipped_label = flipped[i]
            assert flipped_label == exp, (
                f"Index {i}: expected label {exp}, got {flipped_label}"
            )

    def test_label_flipping_preserves_data(self):
        """Image data is unchanged; only labels differ."""
        images = torch.randn(10, 1, 28, 28)
        labels = torch.randint(0, NUM_CLASSES, (10,))
        base = TensorDataset(images, labels)
        flipped = LabelFlippedDataset(base, num_classes=NUM_CLASSES)

        for i in range(len(base)):
            orig_img, orig_label = base[i]
            flip_img, flip_label = flipped[i]
            assert torch.equal(orig_img, flip_img), f"Image data changed at index {i}"
            assert flip_label != orig_label or orig_label == NUM_CLASSES - 1, (
                f"Label not flipped at index {i}: orig={orig_label}, flipped={flip_label}"
            )

    def test_label_flipping_preserves_length(self):
        """Flipped dataset has same length as base dataset."""
        images = torch.randn(20, 1, 28, 28)
        labels = torch.randint(0, NUM_CLASSES, (20,))
        base = TensorDataset(images, labels)
        flipped = LabelFlippedDataset(base, num_classes=NUM_CLASSES)
        assert len(flipped) == len(base)

    def test_label_flipping_custom_num_classes(self):
        """Label flipping works with custom num_classes (e.g., 10)."""
        images = torch.randn(5, 1, 28, 28)
        labels = torch.tensor([0, 5, 9, 3, 8])
        base = TensorDataset(images, labels)
        flipped = LabelFlippedDataset(base, num_classes=10)

        expected = [1, 6, 0, 4, 9]
        for i, exp in enumerate(expected):
            _, flipped_label = flipped[i]
            assert flipped_label == exp


# =====================================================================
# 4. create_client_dataloaders Tests
# =====================================================================


class TestCreateClientDataloaders:
    """Tests for create_client_dataloaders helper."""

    def test_returns_two_dataloaders(self, synthetic_train_dataset, synthetic_test_dataset):
        """Returns a (train_loader, test_loader) tuple."""
        indices = list(range(50))
        train_dl, test_dl = create_client_dataloaders(
            synthetic_train_dataset, synthetic_test_dataset, indices, batch_size=16
        )
        assert isinstance(train_dl, DataLoader)
        assert isinstance(test_dl, DataLoader)

    def test_train_loader_subset_size(self, synthetic_train_dataset, synthetic_test_dataset):
        """Train loader contains only the samples specified by partition_indices."""
        indices = list(range(50))
        train_dl, _ = create_client_dataloaders(
            synthetic_train_dataset, synthetic_test_dataset, indices, batch_size=16
        )
        total = sum(len(batch[0]) for batch in train_dl)
        assert total == 50

    def test_test_loader_uses_full_test_set(self, synthetic_train_dataset, synthetic_test_dataset):
        """Test loader uses the full test dataset (shared across clients)."""
        indices = list(range(50))
        _, test_dl = create_client_dataloaders(
            synthetic_train_dataset, synthetic_test_dataset, indices, batch_size=16
        )
        total = sum(len(batch[0]) for batch in test_dl)
        assert total == len(synthetic_test_dataset)


# =====================================================================
# 5. FlowerClient Tests
# =====================================================================


class TestFlowerClient:
    """Tests for FlowerClient with honest, label_flipper, and noise_injector modes."""

    def test_get_parameters_returns_ndarrays(self, honest_client):
        """get_parameters returns a list of numpy arrays matching model architecture."""
        params = honest_client.get_parameters(config={})
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)
        # FemnistCNN has: conv1.weight, conv1.bias, conv2.weight, conv2.bias,
        #                 fc1.weight, fc1.bias, fc2.weight, fc2.bias = 8 tensors
        assert len(params) == 8, f"Expected 8 parameter arrays, got {len(params)}"

    def test_get_parameters_shapes(self, honest_client):
        """Parameter shapes match the FemnistCNN architecture."""
        params = honest_client.get_parameters(config={})
        expected_shapes = [
            (32, 1, 3, 3),    # conv1.weight
            (32,),             # conv1.bias
            (64, 32, 3, 3),   # conv2.weight
            (64,),             # conv2.bias
            (128, 3136),       # fc1.weight
            (128,),            # fc1.bias
            (62, 128),         # fc2.weight
            (62,),             # fc2.bias
        ]
        for i, (param, expected) in enumerate(zip(params, expected_shapes)):
            assert param.shape == expected, (
                f"Param {i}: expected shape {expected}, got {param.shape}"
            )

    def test_set_parameters_updates_model(self, honest_client):
        """After set_parameters, get_parameters returns the same values."""
        # Create known parameters (all zeros)
        original_params = honest_client.get_parameters(config={})
        zero_params = [np.zeros_like(p) for p in original_params]
        honest_client.set_parameters(zero_params)
        retrieved = honest_client.get_parameters(config={})
        for i, (z, r) in enumerate(zip(zero_params, retrieved)):
            np.testing.assert_array_equal(z, r, err_msg=f"Param {i} mismatch after set_parameters")

    def test_set_parameters_count_mismatch(self, honest_client):
        """set_parameters raises ValueError when parameter count doesn't match."""
        with pytest.raises(ValueError, match="Parameter count mismatch"):
            honest_client.set_parameters([np.zeros(10)])

    def test_honest_client_fit(self, honest_client):
        """Honest client fit returns updated parameters, num_samples, and metrics dict."""
        initial_params = honest_client.get_parameters(config={})
        updated_params, num_samples, metrics = honest_client.fit(
            parameters=initial_params, config={}
        )

        # Returns list of numpy arrays
        assert isinstance(updated_params, list)
        assert all(isinstance(p, np.ndarray) for p in updated_params)
        # Same number of parameter arrays
        assert len(updated_params) == len(initial_params)
        # num_samples should be positive (we have 200 training samples)
        assert num_samples > 0
        # Metrics dict should contain client_id and malicious_type
        assert "client_id" in metrics
        assert "malicious_type" in metrics
        assert metrics["malicious_type"] == "none"

    def test_honest_client_fit_changes_parameters(self, honest_client):
        """After fit, at least some parameters should have changed (training happened)."""
        initial_params = honest_client.get_parameters(config={})
        initial_copies = [p.copy() for p in initial_params]
        updated_params, _, _ = honest_client.fit(parameters=initial_params, config={})

        changed = any(
            not np.array_equal(orig, upd)
            for orig, upd in zip(initial_copies, updated_params)
        )
        assert changed, "Fit did not change any parameters"

    def test_honest_client_evaluate(self, honest_client):
        """Evaluate returns (loss, num_samples, {'accuracy': float})."""
        params = honest_client.get_parameters(config={})
        loss, num_samples, metrics = honest_client.evaluate(parameters=params, config={})

        assert isinstance(loss, float)
        assert num_samples > 0
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_label_flipper_changes_predictions(
        self, honest_client, label_flipper_client
    ):
        """A label_flipper client produces different model weights than an honest client.

        Both start from the same initial parameters but train on differently-labeled
        data, so their outputs must diverge.
        """
        # Get shared initial parameters
        initial_params = honest_client.get_parameters(config={})

        # Train both from the same starting point
        honest_result, _, _ = honest_client.fit(parameters=initial_params, config={})
        flipper_result, _, _ = label_flipper_client.fit(
            parameters=initial_params, config={}
        )

        # Compare L2 distance between resulting parameters
        l2_diff = sum(
            np.sum((h - f) ** 2) for h, f in zip(honest_result, flipper_result)
        )
        assert l2_diff > 0, (
            "Label flipper produced identical parameters to honest client"
        )

    def test_noise_injector_adds_variance(self, train_loader, test_loader):
        """After fit, noise_injector params have significantly higher L2 norm than honest.

        The noise_injector adds N(0, NOISE_SCALE=10.0) to ALL parameters after training,
        so the L2 norm of the difference should be substantial.
        """
        torch.manual_seed(42)
        honest_model = FemnistCNN()
        honest = FlowerClient(
            model=honest_model,
            train_loader=train_loader,
            test_loader=test_loader,
            malicious_type="none",
            client_id=10,
            device=torch.device("cpu"),
        )

        torch.manual_seed(42)
        noisy_model = FemnistCNN()
        noisy = FlowerClient(
            model=noisy_model,
            train_loader=train_loader,
            test_loader=test_loader,
            malicious_type="noise_injector",
            client_id=11,
            device=torch.device("cpu"),
        )

        initial_params = honest.get_parameters(config={})

        honest_params, _, _ = honest.fit(parameters=initial_params, config={})
        noisy_params, _, _ = noisy.fit(parameters=initial_params, config={})

        # Compute L2 norms
        honest_l2 = sum(np.sum(p ** 2) for p in honest_params)
        noisy_l2 = sum(np.sum(p ** 2) for p in noisy_params)

        # The noisy params should have much larger L2 norm due to NOISE_SCALE=10.0
        # added to 428350 parameters -> expected added norm ~ sqrt(428350) * 10 ~ 6500
        assert noisy_l2 > honest_l2 * 2, (
            f"Noise injector L2 norm ({noisy_l2:.1f}) not significantly larger "
            f"than honest ({honest_l2:.1f})"
        )

    def test_noise_injector_scale(self, train_loader, test_loader):
        """The L2 norm difference between noised and clean params is proportional to NOISE_SCALE.

        We verify by computing the per-element std of the noise (difference between
        noisy and honest params). It should be approximately NOISE_SCALE.
        """
        torch.manual_seed(42)
        honest_model = FemnistCNN()
        honest = FlowerClient(
            model=honest_model,
            train_loader=train_loader,
            test_loader=test_loader,
            malicious_type="none",
            client_id=20,
            device=torch.device("cpu"),
        )

        torch.manual_seed(42)
        noisy_model = FemnistCNN()
        noisy = FlowerClient(
            model=noisy_model,
            train_loader=train_loader,
            test_loader=test_loader,
            malicious_type="noise_injector",
            client_id=21,
            device=torch.device("cpu"),
        )

        initial_params = honest.get_parameters(config={})

        honest_params, _, _ = honest.fit(parameters=initial_params, config={})
        noisy_params, _, _ = noisy.fit(parameters=initial_params, config={})

        # Compute per-element differences and their std
        all_diffs = np.concatenate([
            (n - h).flatten() for n, h in zip(noisy_params, honest_params)
        ])
        empirical_std = np.std(all_diffs)

        # Should be approximately NOISE_SCALE (10.0), allow generous tolerance
        assert NOISE_SCALE * 0.5 < empirical_std < NOISE_SCALE * 2.0, (
            f"Empirical noise std ({empirical_std:.2f}) not proportional to "
            f"NOISE_SCALE ({NOISE_SCALE})"
        )

    def test_noise_injector_metrics(self, noise_injector_client):
        """Noise injector client reports correct malicious_type in metrics."""
        params = noise_injector_client.get_parameters(config={})
        _, _, metrics = noise_injector_client.fit(parameters=params, config={})
        assert metrics["malicious_type"] == "noise_injector"
        assert metrics["client_id"] == 2

    def test_label_flipper_metrics(self, label_flipper_client):
        """Label flipper client reports correct malicious_type in metrics."""
        params = label_flipper_client.get_parameters(config={})
        _, _, metrics = label_flipper_client.fit(parameters=params, config={})
        assert metrics["malicious_type"] == "label_flipper"
        assert metrics["client_id"] == 1
