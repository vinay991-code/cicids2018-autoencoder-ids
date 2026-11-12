"""How many of the 68 numeric features of an ordinary held-out test flow fall outside the training range [0, 1] (after scaling)?

Live and PCAP flows are compared with this reference: a flow with many more out-of-range features than the test flows looks unlike
anything the models were trained on, so its scores mean little. Writes results/ood_reference.json.
Usage (from the project folder): python tools/ood_reference.py
"""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
X = np.load(root / "data/processed/X_test.npy", mmap_mode="r")
meta_n = len(X)
idx = np.sort(np.random.default_rng(0).choice(meta_n, 200_000, replace=False))
Z = np.asarray(X[idx])[:, :68]
TOL = 1e-6                                   # same tolerance as app/service.py (float32 rounding of the stored minimum)
ood = ((Z < -TOL) | (Z > 1 + TOL)).sum(axis=1)
ref = {"n_rows_sampled": int(len(idx)), "seed": 0, "numeric_features": 68,
       "median": float(np.median(ood)), "p90": float(np.percentile(ood, 90)), "p99": float(np.percentile(ood, 99)),
       "share_with_any_out_of_range": float((ood > 0).mean())}
(root / "results/ood_reference.json").write_text(json.dumps(ref, indent=2))
print(json.dumps(ref, indent=2))
