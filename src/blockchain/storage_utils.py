"""
storage_utils.py
~~~~~~~~~~~~~~~~
Off-chain gradient storage helpers using a local Redis instance.

Gradient tensors are serialised to bytes with ``torch.save`` (using a BytesIO
buffer) and stored under a UUID key.  The key is used as the ``gradientCidHash``
field in the on-chain ReputationManager contract, acting as a content pointer
without paying gas for large tensor payloads.

Redis connection defaults:
    host : localhost
    port : 6379
    db   : 0

Override these via environment variables:
    REDIS_HOST, REDIS_PORT, REDIS_DB
"""

from __future__ import annotations

import io
import logging
import os
import uuid
from typing import List, Optional

import redis
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

_redis_client: Optional[redis.Redis] = None


def get_redis_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None,
) -> redis.Redis:
    """Return a cached Redis client, creating it on first call.

    Parameters
    ----------
    host:
        Redis host.  Falls back to the ``REDIS_HOST`` env-var, then
        ``"localhost"``.
    port:
        Redis port.  Falls back to ``REDIS_PORT`` env-var, then ``6379``.
    db:
        Redis logical database index.  Falls back to ``REDIS_DB`` env-var,
        then ``0``.

    Returns
    -------
    redis.Redis
        A connected Redis client instance (connection is lazy — no network
        call is made until the first command).
    """
    global _redis_client
    if _redis_client is None:
        _host = host or os.environ.get("REDIS_HOST", "localhost")
        _port = int(port or os.environ.get("REDIS_PORT", 6379))
        _db   = int(db   or os.environ.get("REDIS_DB",   0))
        _redis_client = redis.Redis(host=_host, port=_port, db=_db)
        logger.debug("Redis client initialised: %s:%s db=%s", _host, _port, _db)
    return _redis_client


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise(tensors: List[torch.Tensor]) -> bytes:
    """Serialise a list of tensors to raw bytes using ``torch.save``."""
    buf = io.BytesIO()
    torch.save(tensors, buf)
    return buf.getvalue()


def _deserialise(data: bytes) -> List[torch.Tensor]:
    """Deserialise bytes produced by :func:`_serialise` back to tensors."""
    buf = io.BytesIO(data)
    # weights_only=False is intentional: we need to reconstruct full tensors.
    return torch.load(buf, weights_only=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_tensor_to_redis(
    tensor_list: List[torch.Tensor],
    ttl_seconds: Optional[int] = None,
) -> str:
    """Serialise and store a list of tensors in Redis.

    Parameters
    ----------
    tensor_list:
        The model parameters / gradients to store.  Can be obtained from
        ``list(model.parameters())`` or ``list(model.state_dict().values())``.
    ttl_seconds:
        Optional time-to-live in seconds.  If ``None`` the key persists
        indefinitely (appropriate for research experiments).

    Returns
    -------
    str
        A UUID string that serves as the storage key and as the
        ``gradientCidHash`` committed on-chain.

    Raises
    ------
    redis.RedisError
        Propagated (after logging) if the Redis write fails.
    """
    if not tensor_list:
        raise ValueError("tensor_list must not be empty")

    key = str(uuid.uuid4())
    payload = _serialise(tensor_list)

    client = get_redis_client()
    try:
        if ttl_seconds is not None:
            client.set(key, payload, ex=ttl_seconds)
        else:
            client.set(key, payload)
        logger.info(
            "Uploaded %d tensor(s) (%d bytes) to Redis under key %s",
            len(tensor_list),
            len(payload),
            key,
        )
    except redis.RedisError as exc:
        logger.error("Redis write failed for key %s: %s", key, exc)
        raise

    return key


def download_tensor_from_redis(key: str) -> List[torch.Tensor]:
    """Retrieve and deserialise tensors stored under *key*.

    Parameters
    ----------
    key:
        The UUID string returned by :func:`upload_tensor_to_redis`.

    Returns
    -------
    list[torch.Tensor]
        Reconstructed list of tensors.

    Raises
    ------
    KeyError
        If *key* does not exist in Redis.
    redis.RedisError
        Propagated (after logging) on connection / read errors.
    """
    client = get_redis_client()
    try:
        data = client.get(key)
    except redis.RedisError as exc:
        logger.error("Redis read failed for key %s: %s", key, exc)
        raise

    if data is None:
        raise KeyError(f"Key not found in Redis: {key!r}")

    tensors = _deserialise(data)
    logger.info(
        "Downloaded %d tensor(s) from Redis key %s",
        len(tensors),
        key,
    )
    return tensors


def delete_from_redis(key: str) -> bool:
    """Delete a gradient record from Redis.

    Parameters
    ----------
    key:
        The UUID key to delete.

    Returns
    -------
    bool
        ``True`` if the key existed and was deleted, ``False`` otherwise.
    """
    client = get_redis_client()
    try:
        deleted = client.delete(key)
        logger.debug("Deleted Redis key %s (existed=%s)", key, bool(deleted))
        return bool(deleted)
    except redis.RedisError as exc:
        logger.error("Redis delete failed for key %s: %s", key, exc)
        raise
