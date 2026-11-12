"""Phase 11 / Experiment G: hybrid detector = supervised classifier OR reconstruction-based novelty detector.

Uses ONLY scores already saved by Phases 3, 4 and 6 (no retraining, no torch needed) and the thresholds those phases fixed
beforehand from validation data. The combination rule was written down before it was evaluated and nothing is tuned:

    flag a record if  P(attack) > 0.5   OR   reconstruction error > the novelty model's primary threshold.

Two novelty partners are tried (strictly unsupervised AE, one-class AE), in two settings:
  seen    Phase 3/4 models; all attack types were in training; full test partition.
  unseen  Phase 6 models; Bot, DDoS-HOIC, SSH-BruteForce, DoS-Hulk withheld from every fitted step.
Test labels are used only to count outcomes. 95% intervals resample test blocks (not training randomness).

Usage: python src/phase11_hybrid.py
Writes results/phase11_hybrid.json and reports/phase11_hybrid.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R, P, REP = Path("results"), Path("data/processed"), Path("reports")
BOOT, SEED = 500, 0


def rates(flag, y):
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, fpr=fp / (fp + tn))


def block_ci(flag, groups, blocks):
    """95% block-bootstrap interval of the flagged share inside each boolean group mask."""
    ub, bi = np.unique(blocks, return_inverse=True)
    nb = len(ub)
    num = {g: np.bincount(bi, weights=(flag & m), minlength=nb) for g, m in groups.items()}
    den = {g: np.bincount(bi, weights=m, minlength=nb) for g, m in groups.items()}
    rng = np.random.default_rng(SEED)
    draws = {g: [] for g in groups}
    for _ in range(BOOT):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        for g in groups:
            d = w @ den[g]
            draws[g].append((w @ num[g]) / d if d else np.nan)
    return {g: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for g, v in draws.items()}


def main():
    mte = pd.read_pickle(P / "meta_test.pkl")
    y, at, blk = mte.label_binary.to_numpy(), mte.attack_type.to_numpy(), mte.block.to_numpy()
    benign = at == "Benign"
    out = {"rule": "supervised P(attack) > 0.5 OR novelty reconstruction error > primary threshold", "test_rows": int(len(y)), "seen": {}, "unseen": {}}

    # ---------------- seen setting (Phases 3-5)
    p3 = json.load(open(R / "phase3_results.json"))
    sup = np.load(R / "scores_supervised_test.npy") > 0.5
    comp = {"supervised": sup}
    for m in ("unsup_mixed", "normal_only"):
        comp[m] = np.load(R / f"scores_{m}_test.npy") > p3["variants"][m]["primary"]["threshold"]
    flags = {"supervised": sup, "unsup_mixed": comp["unsup_mixed"], "normal_only": comp["normal_only"],
             "sup OR unsup_mixed": sup | comp["unsup_mixed"], "sup OR normal_only": sup | comp["normal_only"]}
    groups = {"benign": benign, "attacks": y == 1}
    for name, fl in flags.items():
        out["seen"][name] = rates(fl, y) | {"ci": block_ci(fl, groups, blk)}
        out["seen"][name]["per_type_flagged"] = {t: float(fl[at == t].mean()) for t in sorted(set(at) - {"Benign"})}

    # ---------------- unseen setting (Phase 6)
    p6 = json.load(open(R / "phase6_results.json"))
    held = p6["withheld"]
    unseen, known = np.isin(at, held), (y == 1) & ~np.isin(at, held)
    s6 = {m: np.load(R / f"scores_unseen_{m}_test.npy") > p6["models"][m]["thresholds"]["primary"] for m in ("unsup_mixed", "normal_only", "supervised")}
    sup6 = s6["supervised"]
    flags6 = {"supervised": sup6, "unsup_mixed": s6["unsup_mixed"], "normal_only": s6["normal_only"],
              "sup OR unsup_mixed": sup6 | s6["unsup_mixed"], "sup OR normal_only": sup6 | s6["normal_only"]}
    groups6 = {"benign": benign, "known_attacks": known, "unseen_attacks": unseen}
    for name, fl in flags6.items():
        mk, mu = benign | known, benign | unseen
        out["unseen"][name] = {"known_recall": float(fl[known].mean()), "unseen_recall": float(fl[unseen].mean()), "benign_fpr": float(fl[benign].mean()),
                               "unseen_set": rates(fl[mu], y[mu]), "known_set": rates(fl[mk], y[mk]),
                               "per_type_flagged": {h: float(fl[at == h].mean()) for h in held}, "ci": block_ci(fl, groups6, blk)}

    json.dump(out, open(R / "phase11_hybrid.json", "w"), indent=2)

    # ---------------- report
    f = lambda v, n=3: f"{v:.{n}f}"
    ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
    L = ["# Phase 11 - hybrid detector (supervised OR novelty), computed from saved scores\n",
         "Rule fixed beforehand, nothing tuned: flag a record if the supervised probability is above 0.5 OR the novelty model's reconstruction error is above its primary threshold "
         "(95th percentile of validation errors for the unsupervised AE; 99th percentile of benign validation errors for the one-class AE). No retraining. 95% intervals are block bootstraps over test blocks.\n",
         "## Table 6a - seen attack types (Phase 3-5 models, full test partition, 2,163,852 records)\n",
         "| Detector | Precision | Recall [95% CI] | F1 | False-positive rate [95% CI] |", "|---|---:|---:|---:|---:|"]
    for n, r in out["seen"].items():
        L.append(f"| {n} | {f(r['precision'])} | {f(r['recall'])} {ci(r['ci']['attacks'])} | {f(r['f1'])} | {f(r['fpr'], 4)} {ci(r['ci']['benign'])} |")
    L += ["", "## Table 6b - unseen attack types (Phase 6 models; Bot, DDoS-HOIC, SSH-BruteForce, DoS-Hulk withheld)\n",
          "| Detector | Known-attack recall | Unseen-attack recall [95% CI] | Benign false-positive rate | Unseen-set F1 | Bot | DDoS-HOIC | SSH-BruteForce | DoS-Hulk |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, r in out["unseen"].items():
        pt = r["per_type_flagged"]
        L.append(f"| {n} | {f(r['known_recall'])} | {f(r['unseen_recall'])} {ci(r['ci']['unseen_attacks'])} | {f(r['benign_fpr'], 4)} | {f(r['unseen_set']['f1'])} | "
                 f"{f(pt['Bot'])} | {f(pt['DDoS-HOIC'])} | {f(pt['SSH-BruteForce'])} | {f(pt['DoS-Hulk'])} |")
    L += ["", "Per-type columns are the share of that withheld type flagged. A hybrid's false-positive rate can never be lower than its supervised part, and is at most the sum of the two parts."]
    (REP / "phase11_hybrid.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
