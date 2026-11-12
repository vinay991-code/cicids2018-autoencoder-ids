"""Independent check of the Phase 13 (trimmed-loss) numbers.

Recomputes ROC-AUC and PR-AUC of every trimmed model from its saved test scores and the test labels with scikit-learn directly, compares
them with phase13_seed<seed>.json, and recomputes the means of results_trim/trimmed_summary.json from the per-seed files.
Usage (from the project folder): python src/verify_trimmed.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from data import load_split

TRIM = Path("results_trim")
_, meta = load_split("test")
y = meta.label_binary.to_numpy()
ok = bad = 0


def check(name, a, b, tol=1e-9):
    global ok, bad
    if abs(a - b) <= tol:
        ok += 1
    else:
        bad += 1
        print("MISMATCH", name, a, b)


summary = json.load(open(TRIM / "trimmed_summary.json"))
for s in summary["seeds"]:
    r = json.load(open(TRIM / f"phase13_seed{s}.json"))
    for tag, m in r["models"].items():
        sc = np.load(TRIM / f"scores_{tag}_s{s}_test.npy")
        assert len(sc) == len(y)
        check(f"{tag} s{s} roc_auc", roc_auc_score(y, sc), m["ranking"]["roc_auc"])
        check(f"{tag} s{s} pr_auc", average_precision_score(y, sc), m["ranking"]["pr_auc"])
        for k, t in m["thresholds"].items():                      # recompute F1 at each label-free threshold
            pred = sc >= t
            tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            check(f"{tag} s{s} f1 {k}", f1, m["at_threshold"][k]["f1"], 1e-9)
for method, d in summary["methods"].items():
    if not method.startswith("trimmed"):
        continue
    q = int(method.split()[1].rstrip("%"))
    vals = [json.load(open(TRIM / f"phase13_seed{s}.json"))["models"][f"trim{q:02d}"]["ranking"]["roc_auc"] for s in summary["seeds"]]
    check(f"{method} mean roc_auc", float(np.mean(vals)), d["ROC-AUC"]["mean"])
print(f"{ok} of {ok + bad} checks passed")
raise SystemExit(0 if bad == 0 else 1)
