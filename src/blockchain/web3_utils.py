"""
web3_utils.py
~~~~~~~~~~~~~
web3.py helpers for interacting with the on-chain ReputationManager contract.

Connects to a local Hardhat node at http://127.0.0.1:8545 by default.
The compiled contract ABI is loaded from the standard Hardhat artifact path:
    src/blockchain/artifacts/contracts/ReputationManager.sol/ReputationManager.json

Typical usage (local development)::

    from src.blockchain.web3_utils import deploy_contract, update_client_score, get_client_score

    contract_address = deploy_contract()
    update_client_score(contract_address, client_addr, score=85, cid="uuid-key")
    record = get_client_score(contract_address, client_addr)

Environment variables:
    HARDHAT_RPC_URL  — override the default RPC endpoint
    CONTRACT_ADDRESS — pre-deployed contract address (skips deployment)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RPC = "http://127.0.0.1:8545"

# Hardhat artifact relative to project root (two levels up from this file).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_PATH = (
    _PROJECT_ROOT
    / "src"
    / "blockchain"
    / "artifacts"
    / "contracts"
    / "ReputationManager.sol"
    / "ReputationManager.json"
)

# ---------------------------------------------------------------------------
# Internal state  (module-level singleton to avoid redundant connections)
# ---------------------------------------------------------------------------

_w3: Optional[Web3] = None
_contract_instance: Optional[Contract] = None
_contract_address: Optional[str] = None


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_web3() -> Web3:
    """Return a connected Web3 instance (lazy singleton).

    Returns
    -------
    Web3
        Connected to the Hardhat node.

    Raises
    ------
    ConnectionError
        If the node is not reachable.
    """
    global _w3
    if _w3 is None:
        rpc_url = os.environ.get("HARDHAT_RPC_URL", _DEFAULT_RPC)
        _w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not _w3.is_connected():
            raise ConnectionError(
                f"Cannot connect to Hardhat node at {rpc_url}. "
                "Make sure `npx hardhat node` is running."
            )
        logger.info("Connected to Hardhat node at %s", rpc_url)
    return _w3


def _load_artifact() -> Dict[str, Any]:
    """Load the Hardhat-compiled contract artifact.

    Returns
    -------
    dict
        Parsed JSON artifact (contains ``abi`` and ``bytecode`` keys).

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist — run ``npx hardhat compile`` first.
    """
    if not _ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Contract artifact not found at {_ARTIFACT_PATH}. "
            "Run `npx hardhat compile` inside the blockchain Hardhat project first."
        )
    with _ARTIFACT_PATH.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

def deploy_contract(account_index: int = 0) -> str:
    """Deploy the ReputationManager contract and return its address.

    The deploying account is used as the initial admin.  Hardhat's default
    test accounts (index 0) are used by default — **never use real private
    keys in production**.

    Parameters
    ----------
    account_index:
        Index into ``w3.eth.accounts`` (Hardhat unlocked accounts).

    Returns
    -------
    str
        Checksummed address of the deployed contract.
    """
    global _contract_instance, _contract_address

    w3 = get_web3()
    artifact = _load_artifact()
    abi      = artifact["abi"]
    bytecode = artifact["bytecode"]

    deployer = w3.eth.accounts[account_index]
    factory  = w3.eth.contract(abi=abi, bytecode=bytecode)

    logger.info("Deploying ReputationManager from account %s …", deployer)
    tx_hash    = factory.constructor().transact({"from": deployer})
    receipt    = w3.eth.wait_for_transaction_receipt(tx_hash)
    address    = receipt.contractAddress

    _contract_address  = address
    _contract_instance = w3.eth.contract(address=address, abi=abi)

    logger.info("ReputationManager deployed at %s (gas used: %s)", address, receipt.gasUsed)
    return address


def get_contract(address: Optional[str] = None) -> Contract:
    """Return a Contract instance, deploying if necessary.

    Parameters
    ----------
    address:
        Explicit contract address.  If omitted, falls back to the module
        singleton set by :func:`deploy_contract`, then to the
        ``CONTRACT_ADDRESS`` environment variable.

    Returns
    -------
    web3.contract.Contract
    """
    global _contract_instance, _contract_address

    if address is None:
        address = _contract_address or os.environ.get("CONTRACT_ADDRESS")

    if address and _contract_instance and _contract_instance.address == Web3.to_checksum_address(address):
        return _contract_instance

    if address is None:
        raise ValueError(
            "No contract address provided.  Call deploy_contract() first or "
            "set the CONTRACT_ADDRESS environment variable."
        )

    w3       = get_web3()
    artifact = _load_artifact()
    addr_cs  = Web3.to_checksum_address(address)
    _contract_instance = w3.eth.contract(address=addr_cs, abi=artifact["abi"])
    _contract_address  = addr_cs
    return _contract_instance


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def update_client_score(
    client_address: str,
    score: int,
    cid: str,
    loss: int = 0,
    magnitude: int = 0,
    contract_address: Optional[str] = None,
    account_index: int = 0,
) -> str:
    """Call ``updateClient`` on the contract.

    Parameters
    ----------
    client_address:
        Ethereum address of the FL client being updated.
    score:
        Reputation score (integer; may be negative).
    cid:
        IPFS CID or Redis UUID key pointing to the client's gradients.
    loss:
        Loss-improvement metric scaled by 1e18.
    magnitude:
        Gradient L2-norm scaled by 1e18.
    contract_address:
        Override the cached contract address.
    account_index:
        Hardhat account index used as ``from`` (must be the admin).

    Returns
    -------
    str
        Transaction hash (hex).

    Raises
    ------
    ContractLogicError
        If the contract reverts (e.g. caller is not admin).
    """
    w3       = get_web3()
    contract = get_contract(contract_address)
    sender   = w3.eth.accounts[account_index]
    client_cs = Web3.to_checksum_address(client_address)

    try:
        tx_hash = contract.functions.updateClient(
            client_cs, score, cid, loss, magnitude
        ).transact({"from": sender})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        logger.info(
            "updateClient(%s, score=%d, cid=%s) — tx %s",
            client_cs, score, cid, tx_hash.hex(),
        )
        return tx_hash.hex()
    except ContractLogicError as exc:
        logger.error("updateClient reverted for %s: %s", client_cs, exc)
        raise


def batch_update_clients(
    addresses: List[str],
    scores: List[int],
    cids: List[str],
    losses: List[int],
    magnitudes: List[int],
    contract_address: Optional[str] = None,
    account_index: int = 0,
) -> str:
    """Call ``batchUpdateClients`` on the contract.

    All lists must be the same length.

    Parameters
    ----------
    addresses:
        FL client Ethereum addresses.
    scores:
        Reputation scores (corresponding to *addresses*).
    cids:
        Gradient CID / Redis keys.
    losses:
        Loss-improvement metrics (scaled by 1e18).
    magnitudes:
        Update-magnitude metrics (scaled by 1e18).
    contract_address:
        Override the cached contract address.
    account_index:
        Hardhat account index used as ``from``.

    Returns
    -------
    str
        Transaction hash (hex).
    """
    if not (len(addresses) == len(scores) == len(cids) == len(losses) == len(magnitudes)):
        raise ValueError("All input lists must have the same length.")

    w3        = get_web3()
    contract  = get_contract(contract_address)
    sender    = w3.eth.accounts[account_index]
    addr_cs   = [Web3.to_checksum_address(a) for a in addresses]

    try:
        tx_hash = contract.functions.batchUpdateClients(
            addr_cs, scores, cids, losses, magnitudes
        ).transact({"from": sender})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        logger.info(
            "batchUpdateClients(%d clients) — tx %s",
            len(addresses), tx_hash.hex(),
        )
        return tx_hash.hex()
    except ContractLogicError as exc:
        logger.error("batchUpdateClients reverted: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_client_score(
    client_address: str,
    contract_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Call ``getClient`` and return the record as a Python dict.

    Parameters
    ----------
    client_address:
        Ethereum address to query.
    contract_address:
        Override the cached contract address.

    Returns
    -------
    dict
        Keys: ``reputationScore``, ``gradientCidHash``, ``loss``,
        ``magnitude``, ``lastUpdated``.
    """
    contract  = get_contract(contract_address)
    client_cs = Web3.to_checksum_address(client_address)

    try:
        record = contract.functions.getClient(client_cs).call()
    except ContractLogicError as exc:
        logger.error("getClient reverted for %s: %s", client_cs, exc)
        raise

    # The ABI tuple is returned as a list in order of the struct definition.
    result = {
        "reputationScore":  record[0],
        "gradientCidHash":  record[1],
        "loss":             record[2],
        "magnitude":        record[3],
        "lastUpdated":      record[4],
    }
    logger.debug("getClient(%s) → %s", client_cs, result)
    return result
