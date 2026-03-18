// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ReputationManager
 * @notice On-chain reputation registry for Federated Learning clients.
 *         Stores reputation scores and off-chain gradient CID hashes.
 *         Only the admin (FL server) may write reputation data.
 * @dev Designed for local Hardhat development. Gradient payloads are kept
 *      off-chain (IPFS / Redis); only the content-addressed hash is stored
 *      here to minimise gas costs.
 */
contract ReputationManager {
    // -------------------------------------------------------------------------
    // Types
    // -------------------------------------------------------------------------

    /// @notice Full record for a single FL client.
    struct ClientRecord {
        int256  reputationScore;  // RL-assigned reputation (scaled; may be negative)
        string  gradientCidHash;  // IPFS CID or Redis UUID key for latest gradients
        uint256 loss;             // Loss-improvement metric (scaled by 1e18)
        uint256 magnitude;        // L2 norm of gradient update (scaled by 1e18)
        uint256 lastUpdated;      // block.timestamp of last write
    }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    /// @notice Address of the FL server / contract admin.
    address public admin;

    /// @notice Mapping from client Ethereum address to their reputation record.
    mapping(address => ClientRecord) private _records;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    /// @notice Emitted each time a client record is written.
    event ClientUpdated(
        address indexed client,
        int256          score,
        string          cidHash
    );

    /// @notice Emitted when admin role is transferred.
    event AdminTransferred(address indexed previousAdmin, address indexed newAdmin);

    // -------------------------------------------------------------------------
    // Modifiers
    // -------------------------------------------------------------------------

    /// @dev Reverts if the caller is not the current admin.
    modifier onlyAdmin() {
        require(msg.sender == admin, "ReputationManager: caller is not admin");
        _;
    }

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    /**
     * @notice Deploy the registry.  The deploying account becomes the admin.
     */
    constructor() {
        admin = msg.sender;
        emit AdminTransferred(address(0), msg.sender);
    }

    // -------------------------------------------------------------------------
    // Write functions (admin only)
    // -------------------------------------------------------------------------

    /**
     * @notice Update a single client's reputation record.
     * @param client    Ethereum address of the FL client.
     * @param score     New reputation score (may be negative for penalised clients).
     * @param cidHash   IPFS CID or Redis key pointing to the client's gradients.
     * @param loss      Loss-improvement metric, scaled by 1e18.
     * @param magnitude L2 norm of the gradient update, scaled by 1e18.
     */
    function updateClient(
        address        client,
        int256         score,
        string calldata cidHash,
        uint256        loss,
        uint256        magnitude
    ) external onlyAdmin {
        require(client != address(0), "ReputationManager: zero address");
        require(bytes(cidHash).length > 0, "ReputationManager: empty cidHash");

        ClientRecord storage rec = _records[client];
        rec.reputationScore = score;
        rec.gradientCidHash = cidHash;
        rec.loss            = loss;
        rec.magnitude       = magnitude;
        rec.lastUpdated     = block.timestamp;

        emit ClientUpdated(client, score, cidHash);
    }

    /**
     * @notice Update multiple clients in a single transaction to save gas.
     * @dev All arrays must be the same length; reverts the entire batch on any
     *      validation failure so state is never partially updated.
     * @param clients    Array of FL client addresses.
     * @param scores     Corresponding reputation scores.
     * @param cidHashes  Corresponding gradient CID / Redis keys.
     * @param losses     Corresponding loss-improvement metrics (scaled by 1e18).
     * @param magnitudes Corresponding update-magnitude metrics (scaled by 1e18).
     */
    function batchUpdateClients(
        address[]  calldata clients,
        int256[]   calldata scores,
        string[]   calldata cidHashes,
        uint256[]  calldata losses,
        uint256[]  calldata magnitudes
    ) external onlyAdmin {
        uint256 n = clients.length;
        require(n > 0, "ReputationManager: empty batch");
        require(
            scores.length     == n &&
            cidHashes.length  == n &&
            losses.length     == n &&
            magnitudes.length == n,
            "ReputationManager: array length mismatch"
        );

        for (uint256 i = 0; i < n; ++i) {
            require(clients[i] != address(0),          "ReputationManager: zero address in batch");
            require(bytes(cidHashes[i]).length > 0,    "ReputationManager: empty cidHash in batch");

            ClientRecord storage rec = _records[clients[i]];
            rec.reputationScore = scores[i];
            rec.gradientCidHash = cidHashes[i];
            rec.loss            = losses[i];
            rec.magnitude       = magnitudes[i];
            rec.lastUpdated     = block.timestamp;

            emit ClientUpdated(clients[i], scores[i], cidHashes[i]);
        }
    }

    // -------------------------------------------------------------------------
    // Read functions
    // -------------------------------------------------------------------------

    /**
     * @notice Retrieve the full reputation record for a client.
     * @param client Ethereum address to query.
     * @return The client's `ClientRecord` struct (zero-valued if never written).
     */
    function getClient(address client)
        external
        view
        returns (ClientRecord memory)
    {
        return _records[client];
    }

    // -------------------------------------------------------------------------
    // Admin management
    // -------------------------------------------------------------------------

    /**
     * @notice Transfer the admin role to a new address.
     * @param newAdmin The address that will become the new admin.
     */
    function transferAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "ReputationManager: zero address for admin");
        emit AdminTransferred(admin, newAdmin);
        admin = newAdmin;
    }
}
