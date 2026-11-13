"""Phase 15: F1 at a LABELLED threshold, the protocol of Nkashama et al. (arXiv:2407.08838), applied to the saved test scores. REPORTING ONLY.

Why: the main experiments of this project use thresholds that need no attack labels, and a ranking metric. The paper that proposes DAE-LR instead chooses
the decision threshold on a labelled 20% of the test set (the threshold that gives the best F1) and evaluates on the remaining 80%. To compare with it on
its own terms, the same is done here for every model whose test scores were saved. No model is trained or changed; attack labels of the test set are used
to choose the threshold, so these numbers are NOT label-free results and must never be mixed with the label-free ones.

Two ways of drawing the labelled 20% (20 random draws each, like the 20 runs of the paper):
  rows    a random 20% of the test rows (the paper's description). Rows of the same 60 s block are near-duplicates, so rows of the selecting part and of the
          evaluation part are neighbours in time: this is optimistic.
  blocks  a random 20% of the 60 s blocks (all rows of a block go to the same part): the selecting and the evaluation part share no block.
The threshold is the value that maximises the F1 of the selecting part exactly (all distinct score values are considered); a record is flagged when its
score is strictly greater than the threshold, as everywhere in this project. Metrics are computed on the evaluation part (80%).

Usage (from the project folder): python src/phase15_labelled_threshold.py --out-dir results_labelled
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data import load_split

SEEDS = [42, 1, 2]
LAMS = ["0.01", "0.1", "1", "10"]


def score_files(seed):
    """model name -> path of the saved test scores of this seed."""
    base = Path("results") if seed == 42 else Path("seeds") / f"seed_{seed}" / "results"
    f = {"unsup_mixed (ordinary loss)": base / "scores_unsup_mixed_test.npy", "normal_only (benign-only, ordinary loss)": base / "scores_normal_only_test.npy"}
    for q in (10, 20, 30):
        f[f"trimmed {q}%"] = Path("results_trim") / f"scores_trim{q}_s{seed}_test.npy"
    for l in LAMS:
        f[f"DAE-LR mixed, lambda={l}"] = Path("results_daelr") / f"scores_daelr_mixed_lam{l}_s{seed}_test.npy"
    for l in LAMS:
        f[f"DAE-LR benign-only, lambda={l}"] = Path("results_daelr") / f"scores_daelr_clean_lam{l}_s{seed}_test.npy"
    return f


def best_f1_threshold(score, y):
    """Exact F1-maximising threshold t for the rule `score > t` on the selecting part (score float array, y bool). Returns (t, f1)."""
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(~yy)
    P = int(yy.sum())
    f1 = 2 * tp / np.maximum(tp + fp + P, 1)                       # 2tp / (2tp + fp + fn), with fn = P - tp
    last = np.r_[s[1:] != s[:-1], True]                             # cut only after the last of equal scores
    f1 = np.where(last, f1, -1.0)
    k = int(np.argmax(f1))
    t = float(s[k + 1]) if k + 1 < len(s) else float(s[k]) - 1.0     # flag everything with a score >= s[k]
    return t, float(f1[k])


def metrics(score, y, t):
    pred = score > t
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * tp / max(2 * tp + fp + fn, 1), "fpr": fp / max(fp + tn, 1), "flagged_share": (tp + fp) / len(y)}


def split_masks(mode, n, block_ids, split, frac=0.2):
    rng = np.random.default_rng(10_000 + split)
    if mode == "rows":
        sel = np.zeros(n, dtype=bool)
        sel[rng.choice(n, int(round(frac * n)), replace=False)] = True
        return sel
    ub, inv = np.unique(block_ids, return_inverse=True)
    chosen = np.zeros(len(ub), dtype=bool)
    chosen[rng.choice(len(ub), int(round(frac * len(ub))), replace=False)] = True
    return chosen[inv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("results_labelled"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, meta = load_split("test")
    y = meta.label_binary.to_numpy().astype(bool)
    blocks = meta.block.to_numpy()
    n = len(y)
    print(f"test rows {n:,}, attack share {100 * y.mean():.2f}%, blocks {len(np.unique(blocks)):,}", flush=True)
    out = {"config": {"splits": args.splits, "frac": 0.2, "seeds": SEEDS, "rule": "score > threshold"}, "results": {}}
    masks = {mode: [split_masks(mode, n, blocks, k) for k in range(args.splits)] for mode in ("rows", "blocks")}
    for seed in SEEDS:
        for model, path in score_files(seed).items():
            sc = np.load(path).astype(np.float64)
            assert len(sc) == n, path
            for mode in ("rows", "blocks"):
                recs = []
                for k, sel in enumerate(masks[mode]):
                    t, f1_sel = best_f1_threshold(sc[sel], y[sel])
                    m = metrics(sc[~sel], y[~sel], t)
                    recs.append({"split": k, "threshold": t, "f1_selecting_part": f1_sel, **m})
                out["results"].setdefault(model, {}).setdefault(mode, {})[str(seed)] = recs
            r = out["results"][model]
            print(f"seed {seed:2d} {model:42s} F1 rows {np.mean([x['f1'] for x in r['rows'][str(seed)]]):.4f}  blocks {np.mean([x['f1'] for x in r['blocks'][str(seed)]]):.4f}", flush=True)
        json.dump(out, open(args.out_dir / "phase15_labelled_threshold.json", "w"))
    print("Phase 15 done")


if __name__ == "__main__":
    main()
