"""
Location: scripts/run_benchmarks.py
Summary: Benchmark FL aggregation strategies (FedAvg, Krum, Median, Trimmed Mean,
         FLTrust, RL Reputation) under identical attack conditions.

Usage:
    python scripts/run_benchmarks.py --strategies fedavg krum median
    python scripts/run_benchmarks.py --strategies all --num-rounds 50

Dependencies: src/fl_core/{dataset,client}.py, src/integration/strategy.py
"""
from __future__ import annotations

import argparse, gc, json, logging, os, random, sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import (
    EvaluateRes, FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.fl_core.dataset import (
    FemnistCNN, create_client_dataloaders, load_femnist,
    partition_dataset_dirichlet,
)
from src.fl_core.client import FlowerClient, MaliciousType

logger = logging.getLogger(__name__)

AVAILABLE_STRATEGIES = [
    "fedavg", "krum", "median", "trimmed_mean", "fltrust", "rl_reputation",
]
DEFAULTS = dict(
    num_rounds=5, num_clients=100, malicious_fraction=0.3,
    fraction_fit=0.1, target_accuracy=0.7, seed=42,
    trimmed_beta=0.1, batch_size=32, dirichlet_alpha=0.5,
)


def _benchmark_client_resources(num_concurrent_clients: int = 10) -> Dict[str, float]:
    """Return per-client resource allocation for Ray-based Flower simulation.

    Allocates a fractional GPU share so that many virtual clients can run
    concurrently on the same physical GPU(s).  With 10 concurrent clients and
    1 GPU, each client gets 0.1 GPU — preventing Ray resource exhaustion.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_fraction = num_gpus / max(num_concurrent_clients, 1)
        return {"num_cpus": 1, "num_gpus": gpu_fraction}
    return {"num_cpus": 1, "num_gpus": 0.0}


# ---------------------------------------------------------------------------
# Centralized evaluation factory
# ---------------------------------------------------------------------------

def make_evaluate_fn(test_dataset, device):
    """Return a server-side eval function: (round, params, config) -> (loss, metrics)."""
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    def evaluate_fn(server_round, parameters, config):
        model = FemnistCNN().to(device)
        ndarrays = parameters_to_ndarrays(parameters)
        keys = list(model.state_dict().keys())
        model.load_state_dict(
            OrderedDict({k: torch.tensor(v) for k, v in zip(keys, ndarrays)}),
            strict=True,
        )
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                total_loss += criterion(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()
                total += len(labels)
        if total == 0:
            return 0.0, {"accuracy": 0.0}
        return total_loss / total, {"accuracy": correct / total}

    return evaluate_fn


# ============================================================================
# Base robust strategy with shared boilerplate
# ============================================================================

class _BaseRobustStrategy(fl.server.strategy.Strategy):
    """Base for custom robust aggregation strategies."""

    def __init__(self, fraction_fit=0.1, fraction_evaluate=0.05,
                 min_fit_clients=10, min_evaluate_clients=5,
                 min_available_clients=100, evaluate_fn=None,
                 initial_parameters=None):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self._evaluate_fn = evaluate_fn
        self.initial_parameters = initial_parameters
        self._current_parameters: Optional[Parameters] = None
        self.last_round_weights: List[float] = []

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
        n = client_manager.num_available()
        size = max(self.min_fit_clients, int(n * self.fraction_fit))
        clients = client_manager.sample(num_clients=size, min_num_clients=self.min_fit_clients)
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        ins = fl.common.EvaluateIns(parameters, {"server_round": server_round})
        n = client_manager.num_available()
        if n < self.min_evaluate_clients:
            return []
        size = max(self.min_evaluate_clients, int(n * self.fraction_evaluate))
        clients = client_manager.sample(num_clients=size, min_num_clients=self.min_evaluate_clients)
        return [(c, ins) for c in clients]

    def evaluate(self, server_round, parameters):
        if self._evaluate_fn:
            return self._evaluate_fn(server_round, parameters, {})
        return None

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        total = sum(r.num_examples for _, r in results)
        if total == 0:
            return None, {}
        w_loss = sum(r.num_examples * r.loss for _, r in results)
        w_acc = sum(r.num_examples * r.metrics.get("accuracy", 0.0) for _, r in results)
        return w_loss / total, {"accuracy": float(w_acc / total)}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            # All clients failed — return previous parameters as a no-op round
            # instead of None, which would crash Flower's unpacking logic.
            logger.warning(
                "Round %d: aggregate_fit received 0 results and %d failures; "
                "returning previous parameters (no-op round).",
                server_round, len(failures),
            )
            fallback = self._current_parameters or self.initial_parameters
            return fallback, {}
        all_arrays = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        num_samples = [fr.num_examples for _, fr in results]
        aggregated, weights = self._robust_aggregate(all_arrays, num_samples)
        self.last_round_weights = weights
        aggregated_params = ndarrays_to_parameters(aggregated)
        self._current_parameters = aggregated_params
        return aggregated_params, {"num_clients": len(results)}

    def _robust_aggregate(self, client_arrays, num_samples):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Krum: select update closest to others by Euclidean distance
# ---------------------------------------------------------------------------

class KrumStrategy(_BaseRobustStrategy):
    def __init__(self, num_byzantine=3, **kw):
        super().__init__(**kw)
        self.num_byzantine = num_byzantine

    def _robust_aggregate(self, client_arrays, num_samples):
        n = len(client_arrays)
        flat = np.array([np.concatenate([a.flatten() for a in arrs]) for arrs in client_arrays])
        num_closest = max(1, n - self.num_byzantine - 2)
        scores = np.zeros(n)
        for i in range(n):
            dists = np.sum((flat - flat[i]) ** 2, axis=1)
            scores[i] = np.sum(np.sort(dists)[1:num_closest + 1])
        sel = int(np.argmin(scores))
        weights = [0.0] * n
        weights[sel] = 1.0
        logger.info("Krum: selected client %d (score=%.4f)", sel, scores[sel])
        return client_arrays[sel], weights


# ---------------------------------------------------------------------------
# Coordinate-wise Median
# ---------------------------------------------------------------------------

class MedianStrategy(_BaseRobustStrategy):
    def _robust_aggregate(self, client_arrays, num_samples):
        n = len(client_arrays)
        aggregated = []
        for j in range(len(client_arrays[0])):
            stacked = np.stack([client_arrays[i][j] for i in range(n)])
            aggregated.append(np.median(stacked, axis=0).astype(np.float32))
        return aggregated, [1.0 / n] * n


# ---------------------------------------------------------------------------
# Trimmed Mean: trim top/bottom beta fraction, average the rest
# ---------------------------------------------------------------------------

class TrimmedMeanStrategy(_BaseRobustStrategy):
    def __init__(self, beta=0.1, **kw):
        super().__init__(**kw)
        if not 0.0 <= beta < 0.5:
            raise ValueError(f"beta must be in [0, 0.5), got {beta}")
        self.beta = beta

    def _robust_aggregate(self, client_arrays, num_samples):
        n = len(client_arrays)
        trim = int(n * self.beta)
        aggregated = []
        for j in range(len(client_arrays[0])):
            stacked = np.sort(np.stack([client_arrays[i][j] for i in range(n)]), axis=0)
            trimmed = stacked[trim:n - trim] if trim > 0 else stacked
            aggregated.append(np.mean(trimmed, axis=0).astype(np.float32))
        return aggregated, [1.0 / n] * n


# ---------------------------------------------------------------------------
# FLTrust: server-side trust dataset, cosine-similarity scoring (Cao et al.)
# ---------------------------------------------------------------------------

class FLTrustStrategy(_BaseRobustStrategy):
    def __init__(self, trust_dataset, device, **kw):
        super().__init__(**kw)
        self.trust_loader = torch.utils.data.DataLoader(trust_dataset, batch_size=32, shuffle=True)
        self.device = device
        self._server_params: Optional[List[np.ndarray]] = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            # All clients failed — return previous parameters as a no-op round
            logger.warning(
                "Round %d (FLTrust): aggregate_fit received 0 results and %d "
                "failures; returning previous parameters (no-op round).",
                server_round, len(failures),
            )
            fallback = self._current_parameters or self.initial_parameters
            return fallback, {}
        all_arrays = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        num_samples = [fr.num_examples for _, fr in results]
        server_update = self._compute_server_update()
        if server_update is None:
            # First round: use simple weighted average as fallback
            aggregated, weights = self._weighted_avg(all_arrays, num_samples)
        else:
            aggregated, weights = self._fltrust_aggregate(all_arrays, server_update)
        self.last_round_weights = weights
        self._server_params = aggregated
        aggregated_params = ndarrays_to_parameters(aggregated)
        self._current_parameters = aggregated_params
        return aggregated_params, {"num_clients": len(results)}

    def _compute_server_update(self):
        """Train one step on clean data, return updated params (None on first round)."""
        if self._server_params is None:
            return None
        model = FemnistCNN().to(self.device)
        keys = list(model.state_dict().keys())
        model.load_state_dict(
            OrderedDict({k: torch.tensor(v).to(self.device)
                         for k, v in zip(keys, self._server_params)}),
            strict=True,
        )
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        for imgs, labels in self.trust_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            opt.zero_grad()
            crit(model(imgs), labels).backward()
            opt.step()
        return [v.cpu().detach().numpy() for v in model.state_dict().values()]

    def _fltrust_aggregate(self, client_arrays, server_update):
        """Cosine-similarity trust scoring with magnitude re-scaling."""
        n = len(client_arrays)
        srv_flat = np.concatenate([a.flatten() for a in server_update])
        srv_norm = np.linalg.norm(srv_flat)
        if srv_norm < 1e-10:
            return self._weighted_avg(client_arrays, [1] * n)

        # Compute ReLU(cosine_similarity) as trust scores
        trust = np.zeros(n, dtype=np.float32)
        c_flats = []
        for i, arrs in enumerate(client_arrays):
            cf = np.concatenate([a.flatten() for a in arrs])
            c_flats.append(cf)
            cn = np.linalg.norm(cf)
            if cn > 1e-10:
                trust[i] = max(0.0, float(np.dot(srv_flat, cf) / (srv_norm * cn)))

        t_sum = trust.sum()
        if t_sum < 1e-10:
            logger.warning("FLTrust: no trusted clients, using server update.")
            return list(server_update), [0.0] * n
        trust /= t_sum

        nl = len(server_update)
        agg = [np.zeros_like(server_update[j]) for j in range(nl)]
        for i in range(n):
            cn = np.linalg.norm(c_flats[i])
            if cn < 1e-10:
                continue
            scale = srv_norm / cn
            for j in range(nl):
                agg[j] += float(trust[i]) * scale * client_arrays[i][j]
        return agg, trust.tolist()

    @staticmethod
    def _weighted_avg(client_arrays, num_samples):
        n = len(client_arrays)
        total = max(sum(num_samples), 1)
        w = [s / total for s in num_samples]
        nl = len(client_arrays[0])
        agg = [np.zeros_like(client_arrays[0][j]) for j in range(nl)]
        for i in range(n):
            for j in range(nl):
                agg[j] += w[i] * client_arrays[i][j]
        return agg, w


# ============================================================================
# Strategy factory
# ============================================================================

def create_strategy(name, num_clients, malicious_fraction, evaluate_fn,
                    initial_parameters, trust_dataset=None, device=None):
    """Instantiate a named aggregation strategy."""
    frac = DEFAULTS["fraction_fit"]
    min_fit = max(2, int(num_clients * frac))
    kw = dict(fraction_fit=frac, fraction_evaluate=0.05, min_fit_clients=min_fit,
              min_evaluate_clients=5, min_available_clients=num_clients,
              evaluate_fn=evaluate_fn, initial_parameters=initial_parameters)

    if name == "fedavg":
        return FedAvg(**kw)
    elif name == "krum":
        return KrumStrategy(num_byzantine=max(1, int(min_fit * malicious_fraction)), **kw)
    elif name == "median":
        return MedianStrategy(**kw)
    elif name == "trimmed_mean":
        return TrimmedMeanStrategy(beta=DEFAULTS["trimmed_beta"], **kw)
    elif name == "fltrust":
        if trust_dataset is None or device is None:
            raise ValueError("FLTrust requires trust_dataset and device.")
        return FLTrustStrategy(trust_dataset=trust_dataset, device=device, **kw)
    elif name == "rl_reputation":
        return _create_rl_strategy(num_clients, evaluate_fn, initial_parameters, frac, min_fit)
    raise ValueError(f"Unknown strategy '{name}'. Choose from: {AVAILABLE_STRATEGIES}")


def _create_rl_strategy(num_clients, evaluate_fn, initial_params, frac, min_fit):
    """Load RLReputationStrategy; raise RuntimeError if unavailable."""
    try:
        from src.integration.strategy import RLReputationStrategy
    except ImportError as exc:
        raise RuntimeError("RLReputationStrategy deps missing.") from exc
    candidates = ["checkpoints/ppo_latest", "checkpoints/ppo",
                   os.environ.get("PPO_CHECKPOINT_PATH", "")]
    path = next((c for c in candidates if c and os.path.exists(c)), None)
    if path is None:
        raise RuntimeError("No PPO checkpoint found. Set PPO_CHECKPOINT_PATH.")
    return RLReputationStrategy(
        ppo_checkpoint_path=path, num_clients=num_clients,
        fraction_fit=frac, fraction_evaluate=0.05, min_fit_clients=min_fit,
        min_evaluate_clients=5, min_available_clients=num_clients,
        initial_parameters=initial_params,
    )


# ============================================================================
# Client factory for benchmarks
# ============================================================================

def _malicious_type(cid, num_clients, mal_frac, attack_type):
    """Assign malicious type: first mal_frac*num_clients clients are malicious."""
    if cid < int(num_clients * mal_frac):
        return "label_flipper" if attack_type == "label_flip" else "noise_injector"
    return "none"


def make_benchmark_client_fn(train_ds, test_ds, part_idx, num_clients,
                             mal_frac, attack_type, batch_size=32):
    """Create a Flower client_fn for benchmark simulations."""
    from flwr.common import Context

    def client_fn(context: Context):
        pid = int(context.node_config["partition-id"])
        if pid < 0 or pid >= len(part_idx):
            raise ValueError(f"partition_id {pid} out of range [0, {len(part_idx)})")
        mt = _malicious_type(pid, num_clients, mal_frac, attack_type)
        model = FemnistCNN()
        trl, tel = create_client_dataloaders(train_ds, test_ds, part_idx[pid], batch_size)
        return FlowerClient(model=model, train_loader=trl, test_loader=tel,
                            malicious_type=mt, client_id=pid).to_client()
    return client_fn


# ============================================================================
# Benchmark runner
# ============================================================================

def run_single_benchmark(strategy_name, num_clients, num_rounds,
                         malicious_fraction, attack_type, target_accuracy,
                         seed, output_dir):
    """Run one FL simulation with the given strategy, save results to JSON."""
    logger.info("=" * 60)
    logger.info("Benchmark: %s | attack=%s | malicious=%.0f%%",
                strategy_name, attack_type, malicious_fraction * 100)
    logger.info("=" * 60)
    _set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading and partitioning (identical across strategies via seed)
    train_ds = load_femnist(data_dir="./data", train=True)
    test_ds = load_femnist(data_dir="./data", train=False)
    part_idx = partition_dataset_dirichlet(train_ds, num_clients,
                                           alpha=DEFAULTS["dirichlet_alpha"], seed=seed)
    client_fn = make_benchmark_client_fn(
        train_ds, test_ds, part_idx, num_clients, malicious_fraction, attack_type)

    # Initial model parameters (same for all strategies)
    init_model = FemnistCNN()
    init_params = ndarrays_to_parameters(
        [v.cpu().numpy() for v in init_model.state_dict().values()])
    eval_fn = make_evaluate_fn(test_ds, device)

    # Trust dataset for FLTrust (1% of training data)
    trust_ds = None
    if strategy_name == "fltrust":
        rng = np.random.default_rng(seed)
        t_size = max(100, len(train_ds) // 100)
        trust_ds = torch.utils.data.Subset(
            train_ds, rng.choice(len(train_ds), size=t_size, replace=False).tolist())
        logger.info("FLTrust: %d clean trust samples.", t_size)

    try:
        strategy = create_strategy(strategy_name, num_clients, malicious_fraction,
                                   eval_fn, init_params, trust_ds, device)
    except (RuntimeError, ValueError) as exc:
        logger.error("Cannot create strategy '%s': %s", strategy_name, exc)
        return {}

    # Compute per-client GPU fraction based on how many clients run concurrently.
    # fraction_fit * num_clients gives the number of concurrent virtual clients.
    concurrent = max(2, int(num_clients * DEFAULTS["fraction_fit"]))
    client_res = _benchmark_client_resources(num_concurrent_clients=concurrent)
    logger.info("Client resources: %s (concurrent=%d)", client_res, concurrent)

    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn, num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy, client_resources=client_res,
        )
    except Exception:
        logger.exception(
            "start_simulation crashed for strategy '%s'; returning empty results.",
            strategy_name,
        )
        return {}

    results = _collect_results(history, strategy_name, attack_type,
                               malicious_fraction, num_clients, num_rounds,
                               target_accuracy, strategy)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{strategy_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)
    return results


def _collect_results(history, strategy_name, attack_type, mal_frac,
                     num_clients, num_rounds, target_acc, strategy):
    """Extract per-round metrics from Flower History into result dict."""
    rounds, accuracy, loss = [], [], []
    if history.losses_centralized:
        for rnd, l in history.losses_centralized:
            rounds.append(rnd)
            loss.append(float(l))
    if history.metrics_centralized and "accuracy" in history.metrics_centralized:
        for rnd, a in history.metrics_centralized["accuracy"]:
            if rnd not in rounds:
                rounds.append(rnd)
            accuracy.append(float(a))

    # Align list lengths
    mx = max(len(rounds), len(accuracy), len(loss), 1)
    while len(rounds) < mx:
        rounds.append(len(rounds))
    while len(accuracy) < mx:
        accuracy.append(0.0)
    while len(loss) < mx:
        loss.append(0.0)

    conv = None
    for i, a in enumerate(accuracy):
        if a >= target_acc:
            conv = rounds[i] if i < len(rounds) else i
            break

    client_weights = []
    if hasattr(strategy, "last_round_weights") and strategy.last_round_weights:
        client_weights = [strategy.last_round_weights]

    return {
        "strategy": strategy_name, "attack_type": attack_type,
        "malicious_fraction": mal_frac, "num_clients": num_clients,
        "num_rounds": num_rounds, "rounds": rounds, "accuracy": accuracy,
        "loss": loss, "attack_success_rate": [0.0] * len(rounds),
        "client_weights": client_weights,
        "convergence_round": conv, "target_accuracy": target_acc,
    }


# ============================================================================
# Utilities
# ============================================================================

def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark FL aggregation strategies under attack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--strategies", nargs="+", default=["all"],
                   help=f"Strategies to run. 'all' = {AVAILABLE_STRATEGIES}")
    p.add_argument("--num-rounds", type=int, default=DEFAULTS["num_rounds"])
    p.add_argument("--num-clients", type=int, default=DEFAULTS["num_clients"])
    p.add_argument("--malicious-fraction", type=float, default=DEFAULTS["malicious_fraction"])
    p.add_argument("--attack-type", type=str, default="label_flip",
                   choices=["label_flip", "noise_inject"])
    p.add_argument("--target-accuracy", type=float, default=DEFAULTS["target_accuracy"])
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    strategies = list(AVAILABLE_STRATEGIES) if "all" in args.strategies else [
        s.lower() for s in args.strategies]
    for s in strategies:
        if s not in AVAILABLE_STRATEGIES:
            logger.error("Unknown strategy '%s'. Available: %s", s, AVAILABLE_STRATEGIES)
            sys.exit(1)

    logger.info("Strategies: %s | rounds=%d clients=%d malicious=%.0f%% attack=%s seed=%d",
                strategies, args.num_rounds, args.num_clients,
                args.malicious_fraction * 100, args.attack_type, args.seed)

    all_results = {}
    for sn in strategies:
        try:
            r = run_single_benchmark(sn, args.num_clients, args.num_rounds,
                                     args.malicious_fraction, args.attack_type,
                                     args.target_accuracy, args.seed, args.output_dir)
            if r:
                all_results[sn] = r
                fa = r["accuracy"][-1] if r.get("accuracy") else 0.0
                logger.info("%s: final_acc=%.4f convergence=%s",
                            sn, fa, r.get("convergence_round", "N/A"))
        except Exception:
            logger.exception("Strategy '%s' failed, skipping.", sn)

        # Free Python objects and clear GPU VRAM between strategy runs to
        # prevent memory fragmentation when benchmarking sequentially.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    for name, r in all_results.items():
        fa = r["accuracy"][-1] if r.get("accuracy") else 0.0
        logger.info("  %-15s | acc=%.4f | conv=%s",
                    name, fa, r.get("convergence_round", "N/A"))
    logger.info("Results in: %s/", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    main()
