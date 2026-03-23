# =============================================================================
# R3-FL — Federated Learning with RL Reputation & Blockchain
# Multi-stage Dockerfile: builder installs deps; runtime is lean & non-root.
#
# Build (CPU, default):
#   docker build -t r3-fl:latest .
#   docker build --build-arg PYTHON_VERSION=3.11 -t r3-fl:latest .
#
# Build (GPU — CUDA 12.1 + cuDNN 8):
#   docker build \
#     --build-arg TORCH_DEVICE=gpu \
#     --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 \
#     -t r3-fl:latest-gpu .
#
# Run (FL simulation, CPU-only):
#   docker run --rm \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     -e REDIS_HOST=redis \
#     r3-fl:latest
#
# Run (GPU):
#   docker run --rm --gpus all \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     -e REDIS_HOST=redis \
#     r3-fl:latest-gpu
#
# Build args:
#   PYTHON_VERSION   Python version for the CPU base image (default: 3.11)
#   TORCH_DEVICE     cpu | gpu  — selects PyTorch index URL (default: cpu)
#   BASE_IMAGE       Runtime base image. Override to nvidia/cuda:* for GPU.
#                    (default: python:3.11-slim)
#
# Environment variables (all optional — sensible defaults supplied):
#   REDIS_HOST        Redis hostname            (default: localhost)
#   REDIS_PORT        Redis port                (default: 6379)
#   REDIS_DB          Redis DB index            (default: 0)
#   HARDHAT_RPC_URL   Ethereum RPC endpoint     (default: http://localhost:8545)
#   CONTRACT_ADDRESS  Pre-deployed contract     (skips auto-deploy when set)
#   FL_NUM_ROUNDS     Number of FL rounds       (default: 10)
#   FL_NUM_CLIENTS    Number of FL clients      (default: 100)
#   DATA_DIR          Dataset root path         (default: /app/data)
#   LOG_LEVEL         Python log level          (default: INFO)
# =============================================================================

ARG PYTHON_VERSION=3.11
# TORCH_DEVICE selects the PyTorch wheel index: "cpu" or "gpu"
ARG TORCH_DEVICE=cpu
# BASE_IMAGE drives the runtime stage. Override to the CUDA image for GPU builds.
ARG BASE_IMAGE=python:3.11-slim

# -----------------------------------------------------------------------------
# Stage 1 — builder: compile wheels for heavy C-extension packages
# Always uses python:slim regardless of the runtime base image so that the
# build toolchain stays consistent and the heavy compile layer is cacheable.
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

# Re-declare ARGs after FROM (Docker scoping rule)
ARG PYTHON_VERSION=3.11
ARG TORCH_DEVICE=cpu

# System packages needed to compile C/C++ extensions (numpy, torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Use a virtual environment inside the builder so we can copy it cleanly
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip/setuptools first for reliable wheel builds
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only dependency descriptors first — maximises Docker layer cache.
# The heavy pip install layer is invalidated only when pyproject.toml changes.
WORKDIR /build
COPY pyproject.toml ./

# Install PyTorch — CPU path uses the lightweight whl index; GPU path uses
# the CUDA 12.1 index. Both are conditional so only one RUN executes.
# (Shell-form RUN is required for the if/else conditional.)
RUN if [ "$TORCH_DEVICE" = "cpu" ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi
RUN if [ "$TORCH_DEVICE" = "gpu" ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121; \
    fi

RUN pip install --no-cache-dir "ray[rllib]"
RUN pip install --no-cache-dir flwr numpy gymnasium redis web3 matplotlib

# Install the project package itself (editable in the builder, copied to runtime)
COPY . /build/
RUN pip install --no-cache-dir --no-deps -e .

# -----------------------------------------------------------------------------
# Stage 2 — runtime: minimal image, non-root user, no build tools
#
# BASE_IMAGE controls which image this stage uses:
#   CPU (default): python:3.11-slim         — slim Debian, Python pre-installed
#   GPU:           nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
#                  Python 3.11 is installed via apt on the GPU path below.
# -----------------------------------------------------------------------------
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS runtime

ARG PYTHON_VERSION=3.11
ARG TORCH_DEVICE=cpu
ARG BASE_IMAGE=python:3.11-slim

LABEL org.opencontainers.image.title="r3-fl" \
      org.opencontainers.image.description="RL-based Reputation Federated Learning over Blockchain" \
      org.opencontainers.image.source="https://github.com/TheRadDani/r3-fl" \
      org.opencontainers.image.licenses="MIT"

# On the GPU (CUDA Ubuntu) base image, Python is not pre-installed.
# We install python3.11 + pip via deadsnakes PPA. On the CPU (python:slim)
# base image this block is a no-op because python3.11 is already present.
RUN if [ "$TORCH_DEVICE" = "gpu" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            software-properties-common curl && \
        add-apt-repository -y ppa:deadsnakes/ppa && \
        apt-get update && apt-get install -y --no-install-recommends \
            python3.11 python3.11-venv python3.11-distutils && \
        curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Runtime-only system libraries (no build toolchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (UID 1000) with a home directory
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Application directory
WORKDIR /app

# Copy source code (only what the app needs at runtime)
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup deployment.json ./deployment.json
COPY --chown=appuser:appgroup pyproject.toml ./pyproject.toml

# Install the package (non-editable) using already-built deps
RUN pip install --no-cache-dir --no-deps .

# Persistent data directories — owned by appuser so the process can write
RUN mkdir -p /app/data /app/checkpoints /app/results && \
    chown -R appuser:appgroup /app/data /app/checkpoints /app/results

# Volume declarations tell Docker (and compose) where persistent data lives
VOLUME ["/app/data", "/app/checkpoints", "/app/results"]

# Flower simulation runs gRPC on 8080; expose for compose/k8s networking
EXPOSE 8080

# Default environment — all overridable at runtime via -e or compose env_file
ENV REDIS_HOST="localhost" \
    REDIS_PORT="6379" \
    REDIS_DB="0" \
    HARDHAT_RPC_URL="http://localhost:8545" \
    FL_NUM_ROUNDS="10" \
    FL_NUM_CLIENTS="100" \
    DATA_DIR="/app/data" \
    LOG_LEVEL="INFO" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1"

# Drop privileges before the process starts
USER appuser

# Health check: verify Python can import the core module
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src.fl_core.server; print('ok')" || exit 1

# Default command: run the FL simulation
# Override with: docker run ... python -m src.integration.strategy
CMD ["python", "-m", "src.fl_core.server"]
