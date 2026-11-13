"""Independent check of the Phase 14 (DAE-LR) numbers.

Recomputes ROC-AUC, PR-AUC and the F1/recall/FPR at the stored threshold of every DAE-LR model from its saved test scores and the test labels
(scikit-learn and plain counting, not the project's evaluation code), checks that each stored threshold is the stated percentile of the validation
errors recomputed from the saved model, and recomputes the means of results_daelr/daelr_summary.json from the per-seed files.
Usage (from the project folder): python src/verify_daelr.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from daelr import LatentRegulatedAE
from data import load_split
from models import recon_error

D = Path("results_daelr")
M = Path("models_daelr")
_, meta = load_split("test")
y = meta.label_binary.to_numpy()
Xva, mva = load_split("val")
Xva = np.asarray(Xva)
yva = mva.label_binary.to_numpy()
ok = bad = 0


def check(name, a, b, tol=1e-9):
    global ok, bad
    if abs(a - b) <= tol:
        ok += 1
    else:
        bad += 1
        print("MISMATCH", name, a, b)


summary = json.load(open(D / "daelr_summary.json"))
for s in summary["seeds"]:
    r = json.load(open(D / f"phase14_seed{s}.json"))
    for tag, m in r["models"].items():
        sc = np.load(D / f"scores_daelr_{tag}_s{s}_test.npy")
        assert len(sc) == len(y)
        check(f"{tag} s{s} roc_auc", roc_auc_score(y, sc), m["ranking"]["roc_auc"])
        check(f"{tag} s{s} pr_auc", average_precision_score(y, sc), m["ranking"]["pr_auc"])
        for k, t in m["thresholds"].items():
            pred = sc > t                                          # strict, as in evaluation.threshold_metrics (records exactly at the threshold are not flagged)
            tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
            tn = int((~pred & (y == 0)).sum())
            a = m["at_threshold"][k]
            check(f"{tag} s{s} f1 {k}", 2 * tp / max(2 * tp + fp + fn, 1), a["f1"])
            check(f"{tag} s{s} recall {k}", tp / max(tp + fn, 1), a["recall"])
            check(f"{tag} s{s} fpr {k}", fp / max(fp + tn, 1), a["fpr"])
        # the stored threshold is the stated percentile of the validation errors of the saved model
        net = LatentRegulatedAE(Xva.shape[1])
        net.load_state_dict(torch.load(M / f"daelr_{tag}_s{s}.pt", map_location="cpu", weights_only=True))
        net.eval()
        sv = recon_error(net, Xva)
        thr = float(np.percentile(sv, 95)) if "L1" in m["thresholds"] else float(np.percentile(sv[yva == 0], 99))
        check(f"{tag} s{s} threshold", thr, list(m["thresholds"].values())[0], 1e-7)
        # the saved scores are the saved model's scores (first 5,000 test rows)
        Xte, _ = load_split("test")
        check(f"{tag} s{s} scores reproduce", float(np.abs(recon_error(net, np.asarray(Xte[:5000])) - sc[:5000]).max()), 0.0, 1e-6)
for method, d in summary["methods"].items():
    if not method.startswith("DAE-LR"):
        continue
    variant = "mixed" if "mixed" in method else "clean"
    lam = method.split("lambda=")[1]
    vals = [json.load(open(D / f"phase14_seed{s}.json"))["models"][f"{variant}_lam{lam}"]["ranking"]["roc_auc"] for s in summary["seeds"]]
    check(f"{method} mean roc_auc", float(np.mean(vals)), d["ROC-AUC"]["mean"])
print(f"{ok} of {ok + bad} checks passed")
raise SystemExit(0 if bad == 0 else 1)
