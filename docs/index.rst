.. R3-FL documentation master file, created by
   sphinx-quickstart on Wed Mar 18 00:09:55 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

R3-FL: Reinforcement Learning-based Reputation System for Robust Federated Learning
=================================================================================================

Welcome to the documentation for **R3-FL**!

**R3-FL** is a novel research architecture bridging **Federated Learning (FL)**, **Deep Reinforcement Learning (DRL)**, and **Blockchain**. 
It uses Proximal Policy Optimization (PPO) to autonomously learn and assign dynamic trust/reputation scores to edge clients. To prevent spoofing and ensure an immutable history of behavior, reputation matrices and gradient hashes are committed to an Ethereum Smart Contract, while heavy neural computations and model weights are managed by off-chain storage.

-------------------
System Architecture
-------------------

The R3-FL architecture separates compute-heavy AI elements from the storage constraint limits of the blockchain logic:

1. **Edge Clients (FL)**: Clients train models locally using PyTorch. 20-40% of clients simulate adversarial scenarios such as Label Flipping, Gaussian Noise Injection, or Sybil attacks.
2. **Off-chain Storage**: Massive gradient arrays are saved to a local Redis instance (or IPFS), which returns a lightweight CID Hash.
3. **Smart Contract**: The client submits the CID Hash and metadata to the Blockchain immutable ledger.
4. **RL Aggregator**: The central server pulls the immutable history and constructs a 5-feature State Vector (Accuracy Contribution, Gradient Similarity, Historical Reputation, Loss Improvement, Update Magnitude).
5. **PPO Agent**: The Ray RLlib agent observes the State Vector and outputs continuous weight assignments to maximize global accuracy while punishing attackers.

-------------------
API Documentation
-------------------

Explore the codebase modules below:

.. toctree::
   :maxdepth: 2
   :caption: Code Documentation:

   modules
