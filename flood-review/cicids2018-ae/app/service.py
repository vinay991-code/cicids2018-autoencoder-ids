"""Scoring service: loads the saved autoencoders and applies exactly the preprocessing of Phase 2.

Nothing here is trained or tuned. Models, scaler and thresholds are read from the files produced by the experiments:
  data/processed/scaler.json        signed log1p + min-max (fitted on the training partition), protocol one-hot
  models/ae_*.pt                    the reference run (seed 42)
  results/phase3_results.json       thresholds of the two reconstruction models (fixed in advance, no test labels)
  results/phase4_results.json       supervised models use the fixed probability threshold 0.5
Attack labels are never an input. Where the replay shows a label it is for display and evaluation only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .npmodels import load_model   # numpy-only inference: no torch needed to serve (see npmodels.py)

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

MODEL_FILES = {"unsup_mixed": "ae_unsup_mixed.pt", "normal_only": "ae_normal_only.pt",
               "supervised": "ae_supervised.pt", "supervised_recon": "ae_supervised_recon.pt"}
LABELS = {"unsup_mixed": "Unsupervised AE", "normal_only": "One-class AE", "supervised": "Supervised AE",
          "supervised_recon": "Supervised AE + decoder", "hybrid": "Supervised OR unsupervised"}


OOD_TOLERANCE = 1e-6


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Detector:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        sc = json.loads((self.root / "data/processed/scaler.json").read_text())
        self.columns = sc["model_columns"]                       # 68 numeric flow features (CICFlowMeter names, lower case)
        self.lmin = np.asarray(sc["log_min"], dtype=np.float64)
        lmax = np.asarray(sc["log_max"], dtype=np.float64)
        self.span = np.where(lmax > self.lmin, lmax - self.lmin, 1.0)
        self.protocols = list(sc["protocol_categories"])         # [0, 6, 17] then "other"
        self.dim = len(self.columns) + len(self.protocols) + 1
        p3 = json.loads((self.root / "results/phase3_results.json").read_text())["variants"]
        self.thresholds = {"unsup_mixed": float(p3["unsup_mixed"]["primary"]["threshold"]),
                           "normal_only": float(p3["normal_only"]["primary"]["threshold"]),
                           "supervised": 0.5, "supervised_recon": 0.5}
        self.threshold_rules = {"unsup_mixed": "95th percentile of all validation errors (no labels)",
                                "normal_only": "99th percentile of benign validation errors (uses benign validation labels)",
                                "supervised": "probability >= 0.5 (fixed)", "supervised_recon": "probability >= 0.5 (fixed)"}
        self.models = {}
        for name, f in MODEL_FILES.items():
            self.models[name] = load_model(self.root / "models" / f, supervised=name in ("supervised", "supervised_recon"))

    # ---- preprocessing --------------------------------------------------------------------------------------------
    def transform(self, raw: np.ndarray, protocol: np.ndarray) -> np.ndarray:
        """raw: (n, 68) values in self.columns order (unscaled); protocol: (n,) IP protocol numbers -> (n, 72) float32."""
        raw = np.asarray(raw, dtype=np.float64)
        z = ((slog(raw) - self.lmin) / self.span).astype(np.float32)
        proto = np.asarray(protocol)
        onehot = np.stack([(proto == p) for p in self.protocols] + [~np.isin(proto, self.protocols)], axis=1).astype(np.float32)
        return np.concatenate([z, onehot], axis=1)

    def rows_to_arrays(self, rows: list[dict]):
        missing = sorted({c for r in rows for c in self.columns + ["protocol"] if c not in r})
        if missing:
            raise ValueError(f"missing fields: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        raw = np.array([[float(r[c]) for c in self.columns] for r in rows], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("non-finite value in input (NaN or infinity)")
        return raw, np.array([int(r["protocol"]) for r in rows])

    def out_of_range(self, z: np.ndarray) -> np.ndarray:
        """Number of the 68 numeric features of each row outside the training range [0, 1] after scaling. A tolerance of 1e-6 absorbs the float32
        rounding of the stored minimum (a flow with exactly one packet would otherwise count as 'below range' by -2e-10)."""
        zz = z[:, :len(self.columns)]
        return ((zz < -OOD_TOLERANCE) | (zz > 1 + OOD_TOLERANCE)).sum(axis=1)

    # ---- scoring --------------------------------------------------------------------------------------------------
    def score(self, X: np.ndarray) -> dict:
        """X: (n, 72) processed rows. Returns per-model score arrays and boolean alert arrays."""
        xb = np.ascontiguousarray(X, dtype=np.float32)
        scores = {}
        for name in ("unsup_mixed", "normal_only"):
            scores[name] = ((self.models[name](xb) - xb) ** 2).mean(axis=1)
        for name in ("supervised", "supervised_recon"):
            scores[name] = 1.0 / (1.0 + np.exp(-np.clip(self.models[name](xb)[0], -80, 80)))
        alerts = {k: scores[k] >= self.thresholds[k] for k in scores}
        alerts["hybrid"] = alerts["supervised"] | alerts["unsup_mixed"]      # the Table 6 rule, fixed in advance
        return {"scores": scores, "alerts": alerts}

    def info(self) -> dict:
        return {"input_dim": self.dim, "n_numeric_features": len(self.columns), "protocols": self.protocols,
                "models": [{"key": k, "label": LABELS[k], "threshold": self.thresholds[k], "rule": self.threshold_rules[k]} for k in MODEL_FILES]
                + [{"key": "hybrid", "label": LABELS["hybrid"], "threshold": None, "rule": "alert if the supervised OR the unsupervised AE alerts"}],
                "weights": "reference run (seed 42); static, never updated while serving"}
