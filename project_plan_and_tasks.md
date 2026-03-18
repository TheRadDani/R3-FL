# Multi-Terminal Development Strategy & Claude Prompts

This document outlines the **Development Skeleton** for the codebase and provides exactly formatted prompts you can copy into four separate Claude consoles to begin the automated development concurrently. 

## 1. Project Skeleton (Directory Structure)

I have created this physical folder structure in your workspace (`/home/daniel/R3-FL/`):

```text
R3-FL/
│
├── src/
│   ├── fl_core/          # Neural Networks, Flower Clients and Servers
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── client.py
│   │   └── server.py
│   │
│   ├── blockchain/       # Web3, Smart Contracts, Storage wrappers
│   │   ├── __init__.py
│   │   ├── contracts/
│   │   │   └── ReputationManager.sol
│   │   ├── scripts/
│   │   │   └── deploy.py
│   │   ├── storage_utils.py
│   │   └── web3_utils.py
│   │
│   ├── rl_agent/         # PPO Environment and Training logic
│   │   ├── __init__.py
│   │   ├── env.py
│   │   └── train.py
│   │
│   └── integration/      # Custom Flower Strategy tying it all together
│       ├── __init__.py
│       └── strategy.py
│
├── data/                 # Raw dataset downloads (FEMNIST)
├── scripts/              # Bash scripts to start nodes and simulations
├── tests/                # Unit tests
└── CLAUDE.md             # Project context and rules
```

## 2. Task Division & Claude Prompts

To build this without confusing the AI, open **Four Separate Claude Consoles (Chats)**. Copy and paste the corresponding prompt into each respective console.

---

### Console 1: The Federated Learning Base
**Focus:** `src/fl_core/`
This console is strictly responsible for PyTorch models, data partitioning, and the baseline Flower setup with attackers. It does not know about Blockchain or RL.

**Copy this prompt:**
```text
I am developing the PyTorch and Flower (flwr) foundation of my research project in the `src/fl_core/` directory. We need to train a standard CNN on the FEMNIST dataset across 100 clients. 30% of these clients will act maliciously.

Please write the following three production-ready Python files:

1. `src/fl_core/dataset.py`: Functions to load FEMNIST (via Torchvision), partition it non-IID into 100 shards, and define the PyTorch CNN module.
2. `src/fl_core/client.py`: A custom `FlowerClient` class. It must accept a `malicious_type` string ('none', 'label_flipper', 'noise_injector'). If 'label_flipper', shift the labels by +1 during data loading. If 'noise_injector', add massive Gaussian noise (`torch.randn_like * 10.0`) to the model parameters at the end of the `fit()` method.
3. `src/fl_core/server.py`: A basic execution script using Flower's default `FedAvg` strategy that starts a server and simulates 100 clients (where 30% are malicious) across 10 rounds to establish our baseline failure.
```

---

### Console 2: The Blockchain Trust Layer
**Focus:** `src/blockchain/`
This console builds the immutable ledger for reputation tracking and off-chain caching for model sizes.

**Copy this prompt:**
```text
I am developing the blockchain and off-chain storage layer of my project in the `src/blockchain/` directory. We need to track client reputations on an Ethereum testnet but store PyTorch gradients off-chain in Redis (to avoid massive gas costs).

Please write the following four files:

1. `src/blockchain/contracts/ReputationManager.sol`: A Hardhat/Solidity contract mapping a client's Ethereum address to a struct containing `int reputationScore`, `string gradientCidHash`, and recent update `metadata` (loss, magnitude). Only an 'Admin' address can update the scores.
2. `src/blockchain/scripts/deploy.py`: A Python script using `web3.py` (or brownie/forge) to compile and deploy `ReputationManager.sol` to the local node, outputting the deployed contract address and ABI to a JSON file.
3. `src/blockchain/storage_utils.py`: A Python script containing two functions. `upload_tensor_to_redis(tensor_list)` that serializes PyTorch parameters to bytes, saves it in a local Redis dictionary with a unique UUID key, and returns the key. And `download_tensor_from_redis(key)` to retrieve it.
4. `src/blockchain/web3_utils.py`: A Web3.py wrapper that connects to a local Hardhat node (http://127.0.0.1:8545), loads the deployed `ReputationManager` ABI, and provides Python functions like `update_client_score(address, score, cid)` and `get_client_score(address)`.
```

---

### Console 3: The Reinforcement Learning Agent
**Focus:** `src/rl_agent/`
This console focuses entirely on Ray RLlib and the math for the trust algorithm. It assumes the state data will magically appear from the Integration later.

**Copy this prompt:**
```text
I am developing the Reinforcement Learning agent for my project in the `src/rl_agent/` directory using Ray RLlib (PPO) and Gymnasium. The agent acts as an autonomous aggregator, learning to weight 100 FL clients based on their behavior data.

Please write the following two files:

1. `src/rl_agent/env.py`: Define a Custom `Gymnasium` environment called `FLReputationEnv`. 
- The Observation Space is a 100x5 Continuous Matrix (features: accuracy_contribution, gradient_similarity, historical_reputation, loss_improvement, update_magnitude).
- The Action Space is a continuous vector of 100 values between [0,1] representing the aggregation weight $w_i$ for each client.
- The `step()` method must calculate a theoretical reward: $R = (\alpha * Accuracy) - (\beta * AttackImpact)$. Include a helper method inside `step()` to randomly generate dummy State matrices to allow local testing.
2. `src/rl_agent/train.py`: A Python script configuring Ray RLlib to register this custom environment, initialize a PPO model, and run a training loop for 50 iterations, saving checkpoints locally.
```

---

### Console 4: Integration (Execute exactly after Consoles 1, 2, and 3 are finished)
**Focus:** `src/integration/`
This represents the custom server logic that ties the raw model updates, the blockchain reputation scores, and the RL model inferences together.

**Copy this prompt:**
*(Wait until you have the code from Consoles 1, 2, and 3 before pasting this)*
```text
I am building the final integration layer of my project in `src/integration/`. I have a working `fl_core` (PyTorch models), `blockchain` (web3 logic), and `rl_agent` (PPO model).

I need to write a custom Flower strategy. Please write `src/integration/strategy.py`:

1. Define a class `RLReputationStrategy` that inherits from Flower's `Strategy`.
2. Override `aggregate_fit()`. In this method:
   - For every client update received, fetch their `gradientCidHash` and their historical reputation from `blockchain/web3_utils.py`.
   - Download the actual model parameters via `blockchain/storage_utils.py` using that CID.
   - Construct the (100x5) observation state matrix.
   - Load the compiled RLlib PPO checkpoint (from `src/rl_agent/train.py`) and call `compute_single_action(state)` to obtain the 100 float weights $w_i$.
   - Perform a weighted average calculation of the downloaded client parameters using PyTorch and the $w_i$ vector.
   - Return the newly aggregated global model parameters.
```

---

### Console 5: Testing & Validation
**Focus:** `tests/`
This console is responsible for writing unit tests to validate the individual modules before and after they are integrated.

**Copy this prompt:**
*(You can use this prompt after the basic modules are implemented)*
```text
I need to add comprehensive unit tests for the FL, Blockchain, and RL components of my project in the `tests/` directory.

Please write the following test files using `pytest`:

1. `tests/test_fl_core.py`: Tests for the FL dataset partitioning (ensure non-IID properties and correct client counts) and the `FlowerClient` behavior (verify that label flippers actually change labels, and noise injectors add variance to weights).
2. `tests/test_blockchain.py`: Tests for the off-chain storage utilities (verify `upload_tensor_to_redis` and `download_tensor_from_redis` accurately serialize and deserialize tensors) and mocked `web3_utils` functions (verify reputation score updates).
3. `tests/test_rl_agent.py`: Tests for the custom `FLReputationEnv` (verify the observation space shape is 100x5, action space is 100 values in [0,1], and the reward calculation matches the expected math formula).
```

---

### Console 6: Documentation & Generation
**Focus:** `docs/` and Codebase Docstrings
This console is responsible for documenting the code using Sphinx and ensuring all functions have proper docstrings.

**Copy this prompt:**
*(You should use this prompt after the basic modules are implemented and unit tests are written)*
```text
I need to document my complete Federated Learning Blockchain project using Sphinx. It is located in the `src/` directory, broken down into `fl_core/`, `blockchain/`, `rl_agent/`, and `integration/`.

Please assist me with the following tasks:

1. Write a script `scripts/generate_docstrings.py` (or give me instructions) that will scan my Python files and ensure comprehensive RST-style (Sphinx) docstrings are added to every class and method.
2. Provide the terminal commands to initialize a Sphinx project inside the `docs/` directory (`sphinx-quickstart`).
3. Write a correctly configured `docs/conf.py` that includes the `sphinx.ext.autodoc` and `sphinx.ext.napoleon` extensions, and points the `sys.path` to my `/src` folder so it can auto-generate API documentation for my entire codebase.
4. Write the `docs/index.rst` and the corresponding `.rst` files to cleanly map out the architecture of the 4 main modules.
```
