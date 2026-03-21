"""web3.py client library for on-chain ReputationManager contract interaction.

Location: src/blockchain/web3_utils.py
Summary: Python bindings for Web3 interaction with the ReputationManager smart contract.
         Provides deployment, read, and write operations for FL client reputation tracking
         on a local Hardhat Ethereum node.

Purpose:
    Abstracts low-level Web3 contract interaction into high-level Python functions.
    Manages connection state (lazy singleton), loads compiled contract artifacts,
    and provides intuitive APIs for deploying contracts and managing client records
    on the blockchain.

Contract & Network Configuration:
    - **Network**: Local Hardhat Ethereum node (HTTP RPC)
    - **Default RPC**: http://127.0.0.1:8545
    - **Artifact Path**: src/blockchain/artifacts/contracts/ReputationManager.sol/ReputationManager.json
      (compiled by ``npx hardhat compile``)
    - **Accounts**: Hardhat unlocked test accounts (no private key management needed locally)

On-Chain Client Records:
    Each FL client has an on-chain struct (via ReputationManager.getClient):
    - reputationScore: Integer reputation value (may be negative for Byzantine)
    - gradientCidHash: IPFS CID or Redis UUID key pointing to client updates
    - loss: Loss-improvement metric (scaled by 1e18)
    - magnitude: Gradient L2-norm (scaled by 1e18)
    - lastUpdated: Block timestamp of last update

Key Functions:
    - :func:`deploy_contract`: Deploy a new ReputationManager instance
    - :func:`get_contract`: Load existing contract by address
    - :func:`update_client_score`: Single-client update via ``updateClient()``
    - :func:`batch_update_clients`: Multi-client update via ``batchUpdateClients()``
    - :func:`get_client_score`: Query client record via ``getClient()``
    - :func:`get_web3`: Lazy-initialized Web3 connection

Connection Management:
    Web3 and contract instances are stored as module-level singletons to avoid
    redundant connections. First access to get_web3() connects to the RPC node;
    subsequent calls return the cached instance.

Typical Usage (Local Development):
    Deploy and interact with the contract:

    .. code-block:: python

        from src.blockchain.web3_utils import (
            deploy_contract, update_client_score, get_client_score
        )

        # Deploy
        contract_address = deploy_contract()
        print(f"Deployed at: {contract_address}")

        # Update client reputation
        client_addr = "0x70997970C51812e339D9B73b0245ad59ba0A0714"  # Hardhat test account
        tx_hash = update_client_score(
            client_address=client_addr,
            score=85,
            cid="QmXxxx...",  # IPFS CID
            loss=1000000000000000000,  # 1e18
            magnitude=500000000000000000  # 0.5e18
        )
        print(f"Transaction: {tx_hash}")

        # Query reputation
        record = get_client_score(client_addr)
        print(f"Reputation score: {record['reputationScore']}")

Integration:
    - :mod:`src.integration.strategy`: Reads/writes reputation on-chain during FL
    - Tests: ``tests/test_blockchain.py`` (if integration with FL is tested)

Environment Variables:
    - HARDHAT_RPC_URL: Override default RPC endpoint (e.g., "http://localhost:8545")
    - CONTRACT_ADDRESS: Pre-deployed contract address (skips deployment; load existing)

Prerequisites:
    1. Solidity contract compiled: ``npx hardhat compile`` in src/blockchain/
    2. Hardhat node running: ``npx hardhat node`` (listens on 127.0.0.1:8545)
    3. Python web3 installed: ``pip install web3``

Error Handling:
    - ConnectionError: Raised if Hardhat node is not reachable
    - FileNotFoundError: Raised if contract artifact (compiled ABI/bytecode) is missing
    - ContractLogicError: Raised if contract reverts (e.g., caller not admin)
    - ValueError: Raised if no contract address available

See Also:
    - :mod:`src.integration.strategy`: Uses this module for on-chain reputation
    - ReputationManager.sol: Smart contract source (src/blockchain/contracts/)
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
    """Return a connected Web3 instance (module-level lazy singleton).

    Manages a single persistent Web3 connection to the Hardhat node. First call
    connects to the RPC endpoint; subsequent calls return the cached instance.
    This minimizes overhead and maintains consistent node interaction across
    the module.

    Connection Strategy:
        1. On first call: Create HTTPProvider to RPC_URL
        2. Verify connection with ``is_connected()`` call
        3. Cache in module variable ``_w3``
        4. On subsequent calls: Return cached instance

    RPC Endpoint Selection:
        - Default: http://127.0.0.1:8545 (standard Hardhat local node)
        - Override: Set ``HARDHAT_RPC_URL`` environment variable

    Returns
    -------
    Web3
        Connected Web3 instance. Ready for contract interactions (deploy, call, etc).

    Raises
    ------
    ConnectionError
        If the Hardhat node is not reachable at the configured RPC endpoint.
        Error message includes the RPC URL and instructions to start ``npx hardhat node``.

    Example:
        >>> w3 = get_web3()
        >>> print(w3.eth.chain_id)  # Chain ID (typically 31337 for Hardhat)
        31337
        >>> print(len(w3.eth.accounts))  # Unlocked test accounts
        20

    Note:
        Caching is module-level; restarting the process or reloading the module
        reconnects on next access. To force reconnection, delete ``src.blockchain.web3_utils._w3``.

    See Also:
        - :func:`_load_artifact`: Loads contract ABI (pairs with this for deployment)
        - :func:`deploy_contract`: Uses this connection to deploy contracts
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
    """Load the Hardhat-compiled ReputationManager contract artifact.

    Reads and parses the JSON artifact file generated by ``npx hardhat compile``.
    The artifact contains the contract ABI (function signatures) and bytecode
    (machine-readable contract code) needed for deployment and interaction.

    Artifact Location:
        Hardhat generates artifacts at:
        ``src/blockchain/artifacts/contracts/ReputationManager.sol/ReputationManager.json``
        (relative to project root)

    Artifact Structure:
        The JSON file contains (among other fields):
        - ``abi`` (list): Array of contract function/event definitions
        - ``bytecode`` (str): Hex-encoded EVM bytecode for deployment
        - Other fields: constructor, metadata, deployment info (not used here)

    Returns
    -------
    dict
        Parsed JSON artifact with keys:
        - ``abi``: List of function definitions (used for contract instantiation)
        - ``bytecode``: Hex string of deployment bytecode
        Other keys may be present but are not typically used by this module.

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist at the expected path.
        Error message includes the artifact path and instructions to compile.

    Example:
        >>> artifact = _load_artifact()
        >>> abi = artifact["abi"]
        >>> bytecode = artifact["bytecode"]
        >>> # Use for: w3.eth.contract(abi=abi, bytecode=bytecode)

    Note:
        This is an internal function; users should not call directly.
        :func:`deploy_contract` and :func:`get_contract` handle artifact loading.

    See Also:
        - :func:`deploy_contract`: Uses bytecode and ABI to deploy
        - :func:`get_contract`: Uses ABI to bind to existing contract
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
    """Deploy a new ReputationManager contract to the blockchain.

    Creates and deploys a fresh ReputationManager instance on the Hardhat node.
    The deploying account becomes the contract admin and is the only account
    authorized to call updateClient/batchUpdateClients.

    Deployment Process:
        1. Load contract ABI and bytecode from artifact via :func:`_load_artifact`
        2. Get Web3 connection via :func:`get_web3`
        3. Select deployer account from ``w3.eth.accounts[account_index]``
        4. Create contract factory from ABI and bytecode
        5. Send constructor transaction (no arguments for ReputationManager)
        6. Wait for receipt; extract deployed address
        7. Cache contract instance in module variable ``_contract_instance``
        8. Return checksummed address

    Admin Authorization:
        The deployer (account at account_index) becomes the contract admin.
        Only the admin can call write functions (updateClient, batchUpdateClients).
        Read functions (getClient) are public.

    Parameters
    ----------
    account_index : int
        Index into Hardhat's unlocked test accounts (0–19 by default).
        Default 0 is the first test account (typically 0x70997970C51812e339D9B73b0245ad59ba0A0714).
        **For production: never use test accounts; use proper key management.**

    Returns
    -------
    str
        Checksummed Ethereum address of the deployed contract.
        Example: "0x5FbDB2315678afccB33F3...".

    Raises:
        FileNotFoundError: If contract artifact is missing (run ``npx hardhat compile``)
        ConnectionError: If Hardhat node is not reachable
        IndexError: If account_index is out of range

    Example:
        >>> address = deploy_contract(account_index=0)
        >>> print(f"Deployed to: {address}")
        Deployed to: 0x5FbDB2315678afccB33F3...

    Side Effects:
        - Sets module-level variables ``_contract_instance`` and ``_contract_address``
        - Logs deployment address and gas usage to logger.info()

    See Also:
        - :func:`get_contract`: Load an already-deployed contract by address
        - :func:`update_client_score`: Write to the deployed contract
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
    """Return a Contract instance bound to the ReputationManager contract.

    Locates and binds to an existing ReputationManager contract at the specified
    (or cached, or environment-configured) address. Supports multiple fallback
    strategies for address resolution, and caches the contract instance to avoid
    redundant binding.

    Address Resolution Strategy (in priority order):
        1. **Explicit address parameter**: If ``address`` is provided, use it
        2. **Module singleton**: Check ``_contract_address`` (set by prior :func:`deploy_contract`)
        3. **Environment variable**: Check ``CONTRACT_ADDRESS`` env var
        4. **Deployment file**: Check ``deployment.json`` in project root (if exists)
        5. **Fail**: Raise ValueError if no address available

    Caching:
        If a contract instance is already cached at ``_contract_instance`` and
        its address matches the target address, the cached instance is returned.
        This avoids redundant Web3 binding operations.

    Parameters
    ----------
    address : Optional[str]
        Explicit contract address (checksummed or unchecksummed).
        If None, falls back to module singleton → env var → deployment.json.

    Returns
    -------
    web3.contract.Contract
        A web3 contract instance bound to the ReputationManager ABI at the specified address.

    Raises:
        ValueError: If no address is provided and cannot be resolved from any fallback source.
        FileNotFoundError: If contract artifact (ABI) is missing.
        ConnectionError: If Hardhat node is not reachable.
        KeyError: If deployment.json exists but lacks the "address" key.

    Example:
        >>> # Explicit address
        >>> contract = get_contract("0x5FbDB2315678afccB33F3...")

        >>> # Or use cached address from deploy_contract()
        >>> address = deploy_contract()
        >>> contract = get_contract()  # Uses cached address

        >>> # Or set CONTRACT_ADDRESS env var and omit parameter
        >>> # export CONTRACT_ADDRESS="0x5FbDB2315678afccB33F3..."
        >>> contract = get_contract()

    Side Effects:
        - Sets module-level variables ``_contract_instance`` and ``_contract_address``
        - Logs warnings if address lookup falls back to unusual sources

    See Also:
        - :func:`deploy_contract`: Deploys a fresh contract and caches its address
        - :func:`update_client_score`: Uses get_contract() internally
        - :func:`get_client_score`: Uses get_contract() internally
    """
    global _contract_instance, _contract_address

    if address is None:
        address = _contract_address or os.environ.get("CONTRACT_ADDRESS")

    if address is None:
        _deployment_json = _PROJECT_ROOT / "deployment.json"
        if _deployment_json.exists():
            with _deployment_json.open() as _fh:
                _deployment_data = json.load(_fh)
            if "address" not in _deployment_data:
                raise KeyError(
                    f"deployment.json found at {_deployment_json} but does not "
                    "contain the expected 'address' key."
                )
            address = _deployment_data["address"]

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
    """Write a single FL client's reputation record to the contract.

    Calls the ``updateClient`` function on ReputationManager to record or update
    a client's reputation metrics. Only the contract admin can call this function.
    Emits an ``ClientUpdated`` event for off-chain monitoring.

    Parameters
    ----------
    client_address : str
        Ethereum address of the FL client being updated (checksummed or not).
        Will be converted to checksum format for the contract call.
    score : int
        Reputation score for this client (integer, may be negative for Byzantine).
        Stored directly on-chain; interpretation is application-specific.
    cid : str
        Content identifier (IPFS CID or Redis UUID key) pointing to the client's
        gradient/model updates. Example: "QmXxxx..." or "uuid-12345".
    loss : int
        Loss-improvement metric relative to global model (scaled by 1e18 for precision).
        Example: 100e18 (1e18 uint256) for a 1 unit improvement.
        Default: 0 (no loss improvement recorded).
    magnitude : int
        Gradient L2-norm or update magnitude metric (scaled by 1e18).
        Example: 5e17 (0.5e18 uint256) for 0.5 units magnitude.
        Default: 0 (no magnitude recorded).
    contract_address : Optional[str]
        Override the cached contract address. If None, uses :func:`get_contract`.
    account_index : int
        Hardhat account index to use as transaction sender (must be admin).
        Default: 0 (first test account). Must match the deployer account.

    Returns
    -------
    str
        Transaction hash (hex string, e.g., "0xabc123...").

    Raises
    ------
    ContractLogicError
        If the contract reverts. Common reasons:
        - Caller (account_index) is not the contract admin
        - Invalid client_address format
        - Other contract-side validation failures
    ConnectionError
        If Hardhat node is unreachable.
    ValueError
        If no contract address is available.

    Example:
        >>> client = "0x70997970C51812e339D9B73b0245ad59ba0A0714"
        >>> tx = update_client_score(
        ...     client_address=client,
        ...     score=85,
        ...     cid="QmXxxx...",
        ...     loss=int(1e18),     # 1.0 loss improvement
        ...     magnitude=int(0.5e18)  # 0.5 units magnitude
        ... )
        >>> print(f"Transaction: {tx}")

    Side Effects:
        - Emits ``ClientUpdated`` event on-chain
        - Increments client's ``lastUpdated`` timestamp
        - Awaits transaction receipt before returning

    See Also:
        - :func:`batch_update_clients`: Update multiple clients in one transaction
        - :func:`get_client_score`: Query a client's current reputation record
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
    """Batch-write multiple FL clients' reputation records in a single transaction.

    Calls the ``batchUpdateClients`` function on ReputationManager to efficiently
    update many clients at once. More gas-efficient than repeated single-client calls.
    Only the contract admin can call this function.

    Batch Processing:
        All input lists must be the same length (number of clients to update).
        The function zips them together and sends a single transaction with all updates.
        Reverts atomically if any update fails (all-or-nothing semantics).

    Parameters
    ----------
    addresses : List[str]
        Ethereum addresses of FL clients (length N). Will be converted to checksum format.
    scores : List[int]
        Reputation scores (length N, corresponding to addresses).
    cids : List[str]
        IPFS CIDs or Redis keys (length N, corresponding to addresses).
    losses : List[int]
        Loss-improvement metrics (length N, scaled by 1e18).
    magnitudes : List[int]
        Gradient magnitude metrics (length N, scaled by 1e18).
    contract_address : Optional[str]
        Override the cached contract address. If None, uses :func:`get_contract`.
    account_index : int
        Hardhat account index to use as transaction sender (must be admin).
        Default: 0 (first test account).

    Returns
    -------
    str
        Transaction hash (hex string).

    Raises
    ------
    ValueError
        If input lists are not all the same length.
    ContractLogicError
        If the contract reverts (e.g., caller is not admin, or a single client
        update fails; the entire batch is rolled back).
    ConnectionError
        If Hardhat node is unreachable.

    Example:
        >>> addresses = ["0x70997970C51812e339D9B73b0245ad59ba0A0714", "0x..."]
        >>> scores = [85, 70]
        >>> cids = ["QmXxx...", "QmYyy..."]
        >>> losses = [int(1e18), int(0.5e18)]
        >>> magnitudes = [int(0.5e18), int(0.3e18)]
        >>> tx = batch_update_clients(
        ...     addresses=addresses,
        ...     scores=scores,
        ...     cids=cids,
        ...     losses=losses,
        ...     magnitudes=magnitudes
        ... )
        >>> print(f"Batch transaction: {tx}")

    Side Effects:
        - Emits ``ClientUpdated`` event for each updated client
        - Atomically updates all clients' records and timestamps
        - Awaits transaction receipt before returning

    Performance Note:
        Gas cost is significantly lower per client compared to repeated calls
        to :func:`update_client_score`. Recommended when updating 3+ clients.

    See Also:
        - :func:`update_client_score`: Single-client update
        - :func:`get_client_score`: Query a client's current reputation record
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
    """Query an FL client's current reputation record from the contract.

    Calls the ``getClient`` read-only function on ReputationManager to retrieve
    a client's stored reputation struct. Returns the record as a Python dict with
    readable field names.

    Return Value Structure:
        The contract stores each client as a struct with 5 fields (in order):
        - reputationScore: Integer reputation value (may be negative)
        - gradientCidHash: Content identifier (IPFS CID or Redis key)
        - loss: Loss-improvement metric (scaled by 1e18)
        - magnitude: Gradient magnitude metric (scaled by 1e18)
        - lastUpdated: Block timestamp of the last update (uint256)

    Parameters
    ----------
    client_address : str
        Ethereum address of the client to query (checksummed or not).
        Will be converted to checksum format for the contract call.
    contract_address : Optional[str]
        Override the cached contract address. If None, uses :func:`get_contract`.

    Returns
    -------
    dict[str, Any]
        Client reputation record with keys:
        - "reputationScore" (int): Reputation value for this client
        - "gradientCidHash" (str): IPFS CID or Redis key identifier
        - "loss" (int): Loss improvement value (as uint256, 1e18-scaled)
        - "magnitude" (int): Gradient magnitude value (as uint256, 1e18-scaled)
        - "lastUpdated" (int): Block timestamp of most recent update

    Raises
    ------
    ContractLogicError
        If the contract reverts (e.g., client not found, though not typical
        since the contract returns a default struct for uninitialized clients).
    ConnectionError
        If Hardhat node is unreachable.
    ValueError
        If no contract address is available.

    Example:
        >>> client = "0x70997970C51812e339D9B73b0245ad59ba0A0714"
        >>> record = get_client_score(client)
        >>> print(f"Score: {record['reputationScore']}")
        >>> print(f"CID: {record['gradientCidHash']}")
        >>> print(f"Last updated: {record['lastUpdated']}")

    Example Output:
        >>> record
        {
            'reputationScore': 85,
            'gradientCidHash': 'QmXxxx...',
            'loss': 1000000000000000000,  # 1e18
            'magnitude': 500000000000000000,  # 0.5e18
            'lastUpdated': 1234567890
        }

    Note:
        This is a read-only (``call``) operation; it does not consume gas.
        Uninitialized clients return a struct with all fields at default values
        (score 0, empty CID, loss/magnitude/lastUpdated all 0).

    See Also:
        - :func:`update_client_score`: Write a single client's record
        - :func:`batch_update_clients`: Write multiple clients' records
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
