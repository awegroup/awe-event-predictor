"""
On-disk cache for per-file feature matrices.

A cache entry is keyed on the SHA-256 of the input CSV and of the
preprocessing parameters, so editing either the file or the parameters
forces recomputation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, Optional


def _hash_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Return a short SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _hash_dict(d: Dict[str, Any]) -> str:
    """Return a short SHA-256 hex digest of a JSON-serialisable dict."""
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_features_from_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    """Load a cached payload, or return None if missing or unreadable."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Ignoring unreadable cache file {cache_path}: {e}")
        return None


def save_features_to_cache(cache_path: str, payload: Dict[str, Any]) -> None:
    """Write a payload to the cache, creating the folder if needed."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
