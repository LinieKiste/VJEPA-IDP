"""Shared on-disk locations for the pouring probe.

All frame/feature caches and cached predictions live under one root on a fast local disk
(an SSD: the caches are tens of GB and are rewritten often). Override with $POUR_CACHE.
Individual scripts still honour their own per-directory env vars (POUR_FRAMES288_DIR, ...).
"""
import os
from pathlib import Path

CACHE_ROOT = Path(os.environ.get("POUR_CACHE", Path.home() / ".cache" / "pour_probe"))
