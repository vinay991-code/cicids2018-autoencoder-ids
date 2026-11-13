"""Aggregate Phase 15 (F1 at a labelled threshold) over seeds and check it independently.

For every model: mean over the 20 random splits of each seed, then mean over the 3 seeds with the 95% t-interval of the mean (n = 3, wide) and the
smallest and largest seed. Two split types: rows (the protocol as described in Nkashama et al.) and blocks (no shared 60 s block between the part that
chooses the threshold and the part that is evaluated). REPORTING ONLY: the threshold uses test labels.

Independent check (runs with the aggregation): for 3 random (model, seed, split) combinations per split type the selecting part is rebuilt from the saved
split rule, the best-F1 threshold is recomputed with scikit-learn's precision_recall_curve (a different code path), and F1 on the evaluation part is
recomputed with plain counting; both must agree with the stored values.

Usage: python src/aggregate_labelled.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_curve

from data import load_split
from phase15_labelled_threshold import score_files, split_masks

D = Path("results_labelled")
R = json.load(open(D / "phase15_labelled_threshold.json"))
SEEDS = R["config"]["seeds"]


def ci(v):
    v = np.asarray(v, float)
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return {"mean": float(m), "ci95": [float(m - h), float(m + h)], "min": float(v.min()), "max": float(v.max()), "values": [float(x) for x in v]}


summary = {"seeds": SEEDS, "models": {}}
for model, modes in R["results"].items():
    summary["models"][model] = {}
    for mode, per_seed in modes.items():
        summary["models"][model][mode] = {k: ci([np.mean([x[k] for x in per_seed[str(s)]]) for s in SEEDS]) for k in ("f1", "precision", "recall", "fpr", "flagged_share")}
        summary["models"][model][mode]["split_sd_f1"] = float(np.mean([np.std([x["f1"] for x in per_seed[str(s)]], ddof=1) for s in SEEDS]))

_, meta = load_split("test")
y = meta.label_binary.to_numpy().astype(bool)
pi = float(y.mean())
summary["test_attack_share"] = pi
summary["flag_everything_f1"] = 2 * pi / (1 + pi)
json.dump(summary, open(D / "labelled_summary.json", "w"), indent=1)

# ---- independent check
blocks = meta.block.to_numpy()
rng = np.random.default_rng(123)
ok = bad = 0
models = list(R["results"])
for mode in ("rows", "blocks"):
    for _ in range(3):
        model, seed, k = models[rng.integers(len(models))], int(SEEDS[rng.integers(len(SEEDS))]), int(rng.integers(R["config"]["splits"]))
        sc = np.load(score_files(seed)[model]).astype(np.float64)
        sel = split_masks(mode, len(y), blocks, k)
        rec = R["results"][model][mode][str(seed)][k]
        p, r, thr = precision_recall_curve(y[sel], sc[sel])                 # points correspond to the rule `score >= thr`
        f1 = 2 * p * r / np.maximum(p + r, 1e-300)
        best = float(np.nanmax(f1[:-1]))
        ok_sel = abs(best - rec["f1_selecting_part"]) < 1e-9
        pred = sc[~sel] > rec["threshold"]
        yy = y[~sel]
        tp, fp, fn = int((pred & yy).sum()), int((pred & ~yy).sum()), int((~pred & yy).sum())
        ok_eval = abs(2 * tp / max(2 * tp + fp + fn, 1) - rec["f1"]) < 1e-9
        for good in (ok_sel, ok_eval):
            ok += good
            bad += not good
        if not (ok_sel and ok_eval):
            print("MISMATCH", mode, model, seed, k, best, rec["f1_selecting_part"])
print(f"independent check: {ok} of {ok + bad} passed")

# ---- readable table
f = lambda d: f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
L = ["# F1 at a labelled threshold (Phase 15, reporting only)\n",
     f"Test attack share {pi:.4f}; flag-everything F1 = {summary['flag_everything_f1']:.3f}. Mean over 20 random splits per seed, then over seeds {SEEDS}; 95% t-interval of the mean (n = 3, wide). "
     "The threshold maximises F1 on a labelled 20% and is applied to the other 80%.\n",
     "| Model | F1 (rows) | F1 (blocks) | Recall (blocks) | Precision (blocks) | False-positive rate (blocks) |", "|---|---|---|---|---|---|"]
for m, d in summary["models"].items():
    L.append(f"| {m} | {f(d['rows']['f1'])} | {f(d['blocks']['f1'])} | {d['blocks']['recall']['mean']:.3f} | {d['blocks']['precision']['mean']:.3f} | {d['blocks']['fpr']['mean']:.3f} |")
(D / "labelled_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
