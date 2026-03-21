#!/usr/bin/env bash
# Kill remnant Ray processes and run the FL benchmark script.
# Usage: ./run_benchmark.sh [args...]
# Example: ./run_benchmark.sh --strategies rl_reputation
#          ./run_benchmark.sh --strategies fedavg krum median

set -euo pipefail

echo "Stopping lingering Ray processes..."
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
sleep 1

echo "Running benchmark..."
python scripts/run_benchmarks.py "$@"
