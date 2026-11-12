"""Loaders for the Phase 2 outputs (data/processed)."""
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")


def load_split(name: str, proc: Path = PROC):
    """Return (X memmap float32, meta DataFrame). Row i of X is row i of meta."""
    X = np.load(proc / f"X_{name}.npy", mmap_mode="r")
    meta = pd.read_pickle(proc / f"meta_{name}.pkl")
    assert len(X) == len(meta)
    return X, meta


def sample_rows(X, pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Random sample of n rows (without replacement) from the index pool, returned in RAM."""
    rng = np.random.default_rng(seed)
    idx = pool if n >= len(pool) else rng.choice(pool, n, replace=False)
    return np.asarray(X[np.sort(idx)], dtype=np.float32)
