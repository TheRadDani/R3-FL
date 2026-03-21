"""
tests/test_blockchain.py
~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive pytest tests for the blockchain module:
  - storage_utils: Redis-backed tensor upload/download/delete (using fakeredis)
  - web3_utils: Contract interaction helpers (fully mocked, no Hardhat needed)
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
import torch

# ---------------------------------------------------------------------------
# We must reset the module-level singleton before each import-based test run,
# so we import the modules and patch them per-fixture.
# ---------------------------------------------------------------------------
import src.blockchain.storage_utils as storage_mod
import src.blockchain.web3_utils as web3_mod


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(autouse=True)
def _reset_storage_singleton():
    """Reset the module-level _redis_client singleton between tests."""
    storage_mod._redis_client = None
    yield
    storage_mod._redis_client = None


@pytest.fixture()
def fake_redis(monkeypatch):
    """Patch get_redis_client to return a fresh fakeredis instance."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server)

    monkeypatch.setattr(storage_mod, "get_redis_client", lambda *a, **kw: client)
    return client


@pytest.fixture()
def sample_tensors():
    """Return a list of tensors with various shapes for testing."""
    torch.manual_seed(42)
    return [
        torch.randn(3, 3),
        torch.randn(10),
        torch.randn(2, 4, 5),
    ]


@pytest.fixture()
def single_tensor():
    """Return a single simple tensor."""
    torch.manual_seed(42)
    return [torch.randn(4, 4)]


@pytest.fixture()
def mixed_dtype_tensors():
    """Return tensors of different dtypes."""
    torch.manual_seed(42)
    return [
        torch.randn(3, 3, dtype=torch.float32),
        torch.randn(3, 3, dtype=torch.float64),
        torch.tensor([1, 2, 3], dtype=torch.int64),
    ]


# -- Web3 fixtures --------------------------------------------------------

@pytest.fixture()
def mock_web3():
    """Return a MagicMock standing in for a Web3 instance."""
    w3 = MagicMock()
    w3.is_connected.return_value = True
    w3.eth.accounts = [
        "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
    ]
    return w3


@pytest.fixture()
def mock_contract():
    """Return a MagicMock standing in for a web3 Contract instance."""
    contract = MagicMock()
    contract.address = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    return contract


@pytest.fixture(autouse=True)
def _reset_web3_singletons():
    """Reset web3_utils module-level singletons between tests."""
    web3_mod._w3 = None
    web3_mod._contract_instance = None
    web3_mod._contract_address = None
    yield
    web3_mod._w3 = None
    web3_mod._contract_instance = None
    web3_mod._contract_address = None


# ===================================================================
# 1. Storage Utils Tests (fakeredis)
# ===================================================================


class TestUploadTensorToRedis:
    """Tests for upload_tensor_to_redis."""

    def test_upload_returns_uuid_string(self, fake_redis, single_tensor):
        """upload_tensor_to_redis should return a valid UUID4 string."""
        key = storage_mod.upload_tensor_to_redis(single_tensor)

        # Should be a valid UUID4 string
        parsed = uuid.UUID(key, version=4)
        assert str(parsed) == key

    def test_upload_download_roundtrip_single_tensor(self, fake_redis, single_tensor):
        """A single tensor should survive upload -> download unchanged."""
        key = storage_mod.upload_tensor_to_redis(single_tensor)
        result = storage_mod.download_tensor_from_redis(key)

        assert len(result) == 1
        assert torch.equal(result[0], single_tensor[0])

    def test_upload_download_roundtrip_multiple_tensors(self, fake_redis, sample_tensors):
        """Multiple tensors with different shapes should roundtrip correctly."""
        key = storage_mod.upload_tensor_to_redis(sample_tensors)
        result = storage_mod.download_tensor_from_redis(key)

        assert len(result) == len(sample_tensors)
        for original, recovered in zip(sample_tensors, result):
            assert original.shape == recovered.shape
            assert torch.equal(original, recovered)

    def test_upload_download_preserves_dtypes(self, fake_redis, mixed_dtype_tensors):
        """Tensors of different dtypes (float32, float64, int64) should
        preserve their dtype through a roundtrip."""
        key = storage_mod.upload_tensor_to_redis(mixed_dtype_tensors)
        result = storage_mod.download_tensor_from_redis(key)

        assert len(result) == len(mixed_dtype_tensors)
        for original, recovered in zip(mixed_dtype_tensors, result):
            assert original.dtype == recovered.dtype, (
                f"Expected dtype {original.dtype}, got {recovered.dtype}"
            )
            assert torch.equal(original, recovered)

    def test_upload_download_preserves_gradients(self, fake_redis):
        """Tensors with requires_grad=True should have matching data after
        roundtrip (grad metadata may not persist, but values must)."""
        torch.manual_seed(42)
        t = torch.randn(3, 3, requires_grad=True)
        tensor_list = [t]

        key = storage_mod.upload_tensor_to_redis(tensor_list)
        result = storage_mod.download_tensor_from_redis(key)

        assert len(result) == 1
        # Data values must match
        assert torch.equal(result[0].data, t.data)

    def test_upload_empty_list_raises(self, fake_redis):
        """Passing an empty list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            storage_mod.upload_tensor_to_redis([])

    def test_upload_with_ttl(self, fake_redis, single_tensor):
        """Upload with ttl_seconds should succeed (key is stored)."""
        key = storage_mod.upload_tensor_to_redis(single_tensor, ttl_seconds=300)

        # Key should exist and be downloadable
        result = storage_mod.download_tensor_from_redis(key)
        assert len(result) == 1
        assert torch.equal(result[0], single_tensor[0])

    def test_large_tensor_roundtrip(self, fake_redis):
        """A large tensor (1000x1000) should serialize and deserialize
        correctly without data corruption."""
        torch.manual_seed(42)
        large = torch.randn(1000, 1000)

        key = storage_mod.upload_tensor_to_redis([large])
        result = storage_mod.download_tensor_from_redis(key)

        assert len(result) == 1
        assert result[0].shape == (1000, 1000)
        assert torch.equal(result[0], large)


class TestDownloadTensorFromRedis:
    """Tests for download_tensor_from_redis."""

    def test_download_missing_key_raises(self, fake_redis):
        """Downloading a nonexistent key should raise KeyError."""
        with pytest.raises(KeyError, match="Key not found"):
            storage_mod.download_tensor_from_redis("nonexistent-key-12345")


class TestDeleteFromRedis:
    """Tests for delete_from_redis."""

    def test_delete_existing_key(self, fake_redis, single_tensor):
        """Deleting an existing key returns True and removes the data."""
        key = storage_mod.upload_tensor_to_redis(single_tensor)

        assert storage_mod.delete_from_redis(key) is True

        # Key should now be gone
        with pytest.raises(KeyError):
            storage_mod.download_tensor_from_redis(key)

    def test_delete_nonexistent_key(self, fake_redis):
        """Deleting a key that does not exist returns False."""
        assert storage_mod.delete_from_redis("nonexistent-key-999") is False


class TestSerialisationHelpers:
    """Tests for the internal _serialise / _deserialise helpers."""

    def test_serialise_deserialise_roundtrip(self, sample_tensors):
        """Internal helpers should produce identical tensors after roundtrip."""
        data = storage_mod._serialise(sample_tensors)
        assert isinstance(data, bytes)
        assert len(data) > 0

        recovered = storage_mod._deserialise(data)
        assert len(recovered) == len(sample_tensors)
        for orig, rec in zip(sample_tensors, recovered):
            assert torch.equal(orig, rec)


# ===================================================================
# 2. Web3 Utils Tests (fully mocked)
# ===================================================================


class TestUpdateClientScore:
    """Tests for update_client_score."""

    def test_update_client_score_calls_contract(self, mock_web3, mock_contract):
        """update_client_score should call updateClient with the correct args."""
        client_addr = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
        score = 85
        cid = "some-uuid-key"
        loss = 100
        magnitude = 200

        # Wire up the mocks
        mock_tx_hash = MagicMock()
        mock_tx_hash.hex.return_value = "0xabc123"
        mock_contract.functions.updateClient.return_value.transact.return_value = mock_tx_hash
        mock_web3.eth.wait_for_transaction_receipt.return_value = MagicMock()

        with patch.object(web3_mod, "get_web3", return_value=mock_web3), \
             patch.object(web3_mod, "get_contract", return_value=mock_contract):
            result = web3_mod.update_client_score(
                client_address=client_addr,
                score=score,
                cid=cid,
                loss=loss,
                magnitude=magnitude,
            )

        # Verify updateClient was called with checksummed address and correct args
        mock_contract.functions.updateClient.assert_called_once()
        call_args = mock_contract.functions.updateClient.call_args[0]
        assert call_args[1] == score
        assert call_args[2] == cid
        assert call_args[3] == loss
        assert call_args[4] == magnitude

        assert result == "0xabc123"

    def test_update_client_score_negative_score(self, mock_web3, mock_contract):
        """Negative scores should be accepted (int256 supports negative)."""
        client_addr = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

        mock_tx_hash = MagicMock()
        mock_tx_hash.hex.return_value = "0xdef456"
        mock_contract.functions.updateClient.return_value.transact.return_value = mock_tx_hash
        mock_web3.eth.wait_for_transaction_receipt.return_value = MagicMock()

        with patch.object(web3_mod, "get_web3", return_value=mock_web3), \
             patch.object(web3_mod, "get_contract", return_value=mock_contract):
            result = web3_mod.update_client_score(
                client_address=client_addr,
                score=-50,
                cid="neg-score-key",
            )

        call_args = mock_contract.functions.updateClient.call_args[0]
        assert call_args[1] == -50
        assert result == "0xdef456"


class TestGetClientScore:
    """Tests for get_client_score."""

    def test_get_client_score_returns_dict(self, mock_web3, mock_contract):
        """get_client_score should convert the contract tuple to a dict."""
        client_addr = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

        # Simulate contract returning a tuple (matching struct field order)
        mock_contract.functions.getClient.return_value.call.return_value = (
            85,              # reputationScore
            "uuid-key-123",  # gradientCidHash
            100,             # loss
            200,             # magnitude
            1710000000,      # lastUpdated
        )

        with patch.object(web3_mod, "get_web3", return_value=mock_web3), \
             patch.object(web3_mod, "get_contract", return_value=mock_contract):
            result = web3_mod.get_client_score(client_address=client_addr)

        assert isinstance(result, dict)
        assert result["reputationScore"] == 85
        assert result["gradientCidHash"] == "uuid-key-123"
        assert result["loss"] == 100
        assert result["magnitude"] == 200
        assert result["lastUpdated"] == 1710000000

    def test_get_client_score_dict_keys(self, mock_web3, mock_contract):
        """Returned dict must have exactly the expected keys."""
        client_addr = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

        mock_contract.functions.getClient.return_value.call.return_value = (
            0, "", 0, 0, 0,
        )

        with patch.object(web3_mod, "get_web3", return_value=mock_web3), \
             patch.object(web3_mod, "get_contract", return_value=mock_contract):
            result = web3_mod.get_client_score(client_address=client_addr)

        expected_keys = {
            "reputationScore",
            "gradientCidHash",
            "loss",
            "magnitude",
            "lastUpdated",
        }
        assert set(result.keys()) == expected_keys


class TestBatchUpdateClients:
    """Tests for batch_update_clients."""

    def test_batch_update_clients_calls_contract(self, mock_web3, mock_contract):
        """batch_update_clients should call batchUpdateClients with lists."""
        addresses = [
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
        ]
        scores = [80, 90]
        cids = ["key-a", "key-b"]
        losses = [10, 20]
        magnitudes = [30, 40]

        mock_tx_hash = MagicMock()
        mock_tx_hash.hex.return_value = "0xbatch789"
        mock_contract.functions.batchUpdateClients.return_value.transact.return_value = mock_tx_hash
        mock_web3.eth.wait_for_transaction_receipt.return_value = MagicMock()

        with patch.object(web3_mod, "get_web3", return_value=mock_web3), \
             patch.object(web3_mod, "get_contract", return_value=mock_contract):
            result = web3_mod.batch_update_clients(
                addresses=addresses,
                scores=scores,
                cids=cids,
                losses=losses,
                magnitudes=magnitudes,
            )

        mock_contract.functions.batchUpdateClients.assert_called_once()
        call_args = mock_contract.functions.batchUpdateClients.call_args[0]
        # Addresses should be checksummed
        assert len(call_args[0]) == 2
        assert call_args[1] == scores
        assert call_args[2] == cids
        assert call_args[3] == losses
        assert call_args[4] == magnitudes
        assert result == "0xbatch789"

    def test_batch_update_validates_list_lengths(self, mock_web3, mock_contract):
        """Mismatched list lengths should raise ValueError."""
        with patch.object(web3_mod, "get_web3", return_value=mock_web3), \
             patch.object(web3_mod, "get_contract", return_value=mock_contract):
            with pytest.raises(ValueError, match="same length"):
                web3_mod.batch_update_clients(
                    addresses=["0x70997970C51812dc3A010C7d01b50e0d17dc79C8"],
                    scores=[80, 90],  # length mismatch
                    cids=["key-a"],
                    losses=[10],
                    magnitudes=[30],
                )


# =========================================================================
# Additional web3_utils Tests (coverage expansion)
# =========================================================================

class TestGetWeb3Simple:
    """Simple unit tests for Web3 connection management."""

    def test_get_web3_returns_web3_instance(self, mock_web3):
        """get_web3 should return a Web3 instance."""
        with patch.object(web3_mod, "_w3", None), \
             patch("src.blockchain.web3_utils.Web3") as MockWeb3:
            MockWeb3.return_value = mock_web3
            result = web3_mod.get_web3()

            assert result is not None


class TestUpdateClientScoreAdditional:
    """Additional tests for update_client_score."""

    def test_update_client_score_negative_score(self, mock_web3, mock_contract):
        """update_client_score should handle negative scores."""
        mock_tx_hash = MagicMock()
        mock_tx_hash.hex.return_value = "0xupdate123"
        mock_contract.functions.updateClient.return_value.transact.return_value = mock_tx_hash
        mock_web3.eth.wait_for_transaction_receipt.return_value = MagicMock()

        with patch.object(web3_mod, "get_web3", return_value=mock_web3), \
             patch.object(web3_mod, "get_contract", return_value=mock_contract):
            result = web3_mod.update_client_score(
                client_address="0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                score=-50,
                cid="ipfs-key",
            )

            assert result == "0xupdate123"


class TestGetClientScoreAdditional:
    """Additional tests for get_client_score."""

    def test_get_client_score_returns_dict_structure(self, mock_web3, mock_contract):
        """get_client_score should return dict with expected keys."""
        mock_record = (100, "ipfs-key", 50, 75, 12345)
        mock_contract.functions.getClient.return_value.call.return_value = mock_record

        with patch.object(web3_mod, "get_web3", return_value=mock_web3), \
             patch.object(web3_mod, "get_contract", return_value=mock_contract):
            result = web3_mod.get_client_score(
                "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
            )

            assert isinstance(result, dict)
            assert "reputationScore" in result
            assert "gradientCidHash" in result
