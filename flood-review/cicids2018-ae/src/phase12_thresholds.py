"""Phase 12 / Experiment H: label-free threshold selection for reconstruction-error detectors, at controlled attack prevalence.

Question: a reconstruction-error score can rank attacks well (ROC-AUC 0.89-0.93) yet flag few of them at a fixed percentile threshold.
Can a threshold be chosen WITHOUT target labels that comes close to the best achievable one, and does it help to know the attack prevalence?

Uses only scores saved by Phases 3 and 6 (no retraining, no torch). Methods were fixed before any result was looked at.
Scores are log10 of the per-record reconstruction error; a record is flagged when its score is above the threshold.

Label-free methods (the target set's labels are never used to choose the threshold):
  F-primary     the fixed rule of Phases 3/6 (95th pct of unlabeled validation errors for the unsupervised AE; 99th pct of benign validation errors for the one-class AE)
  F-ben-p95/p99/p99.9   quantiles of CLEAN BENIGN validation scores (needs a clean reference sample; seen setting only)
  B-1% / B-5% / B-10%   alert budget: flag the top q of the TARGET scores
  P-gmm         estimate the attack share with a 2-component Gaussian mixture on the target scores, then flag that share (top pi_hat)
  P-tail        estimate the attack share by a simple tail-ratio estimator against the clean reference, then flag the top pi_hat
                (pi_hat = 1 - min_t [target tail share / reference tail share]; reference tails restricted to >= 500 points; seen setting only)
  T-flag-all    trivial baseline that flags every record (a detector must beat this F1 to be of any use)
Reference rows that DO use labels, only to show what is achievable (never a candidate for deployment):
  O-prev        flag the top (true attack share) of the target: knowing the prevalence exactly
  O-F1          the threshold with the best F1 on the target itself
  V-F1          the Phase 3/6 threshold tuned for F1 on VALIDATION labels

Settings: seen (Phase 3 models, full test partition; benign vs all attacks) and unseen (Phase 6 models; benign vs the four withheld types).
Prevalence is controlled by resampling the TARGET set (attacks or benign subsampled without replacement, 5 seeds); the natural prevalence is run once.
Intervals in this file describe resampling of the evaluation set only; each model was trained once.

Usage: python src/phase12_thresholds.py
Writes results/phase12_thresholds.json and reports/phase12_thresholds.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.mixture import GaussianMixture

R, P, REP = Path("results"), Path("data/processed"), Path("reports")
PIS, SEEDS, MODELS = [0.01, 0.05, "natural", 0.30, 0.50], [0, 1, 2, 3, 4], ["unsup_mixed", "normal_only"]
SHORT = {"unsup_mixed": "unsupervised AE", "normal_only": "one-class AE"}


def log10s(s):
    return np.log10(np.maximum(np.asarray(s, dtype=np.float64), 1e-12))


def rates(flag, y):
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    return dict(precision=prec, recall=rec, f1=(2 * prec * rec / (prec + rec) if tp else 0.0), fpr=fp / (fp + tn), alert_rate=(tp + fp) / len(y))


def resample(y, pi, rng):
    """Indices of a target set with attack share pi (benign or attack rows subsampled without replacement)."""
    ib, ia = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if pi == "natural":
        return np.arange(len(y)), len(ia) / len(y)
    if pi <= len(ia) / len(y):
        ia = rng.choice(ia, int(round(pi / (1 - pi) * len(ib))), replace=False)
    else:
        ib = rng.choice(ib, int(round((1 - pi) / pi * len(ia))), replace=False)
    assert abs(len(ia) / (len(ia) + len(ib)) - pi) < 1e-3, "resampled prevalence is off target"
    return np.sort(np.concatenate([ib, ia])), pi


def top_share_threshold(s, q):
    return float(np.quantile(s, 1.0 - min(max(q, 1e-6), 1.0)))


def gmm_pi(s, seed=0):
    x = s if len(s) <= 100_000 else np.random.default_rng(seed).choice(s, 100_000, replace=False)
    g = GaussianMixture(2, random_state=seed).fit(x.reshape(-1, 1))
    return float(g.weights_[int(np.argmax(g.means_.ravel()))])


def tail_pi(s, ref):
    """1 - min_t (tail share of target above t) / (tail share of clean reference above t), t over reference quantiles with >= 500 reference points above."""
    ref = np.sort(ref)
    n = len(ref)
    grid = np.linspace(0.5, 1 - 500 / n, 200)
    t = ref[np.minimum((grid * n).astype(int), n - 1)]
    tail_ref = 1.0 - (np.searchsorted(ref, t, side="right") / n)
    ss = np.sort(s)
    tail_t = 1.0 - (np.searchsorted(ss, t, side="right") / len(ss))
    return float(np.clip(1.0 - np.min(tail_t / tail_ref), 0.0, 1.0))


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def evaluate(s, y, fixed, ref_benign):
    """All methods on one target set. fixed: dict name -> threshold (log10 scale). ref_benign: clean benign reference scores or None."""
    pi_true = float(y.mean())
    out, pi_hat = {}, {}
    for name, thr in fixed.items():
        out[name] = rates(s > thr, y)
    for q in (0.01, 0.05, 0.10):
        out[f"B-{int(q * 100)}%"] = rates(s > top_share_threshold(s, q), y)
    pi_hat["P-gmm"] = gmm_pi(s)
    out["P-gmm"] = rates(s > top_share_threshold(s, pi_hat["P-gmm"]), y)
    if ref_benign is not None:
        for nm, q in (("F-ben-p95", 95), ("F-ben-p99", 99), ("F-ben-p99.9", 99.9)):
            out[nm] = rates(s > float(np.percentile(ref_benign, q)), y)
        pi_hat["P-tail"] = tail_pi(s, ref_benign)
        out["P-tail"] = rates(s > top_share_threshold(s, pi_hat["P-tail"]), y)
    out["T-flag-all"] = rates(np.ones(len(y), dtype=bool), y)        # trivial baseline: no detector at all
    out["O-prev"] = rates(s > top_share_threshold(s, pi_true), y)
    out["O-F1"] = rates(s > best_f1_threshold(y, s), y)
    return out, pi_hat, pi_true


def run_setting(name, s_all, y_all, fixed, ref_benign):
    res = {}
    for pi in PIS:
        runs, hats, trues = [], [], []
        for seed in ([0] if pi == "natural" else SEEDS):
            idx, pt = resample(y_all, pi, np.random.default_rng(seed))
            o, h, t = evaluate(s_all[idx], y_all[idx], fixed, ref_benign)
            runs.append(o); hats.append(h); trues.append(t)
        methods = {}
        for m in runs[0]:
            methods[m] = {k: [float(np.mean([r[m][k] for r in runs])), float(np.std([r[m][k] for r in runs]))] for k in runs[0][m]}
        res[str(pi)] = {"true_prevalence": float(np.mean(trues)), "n_runs": len(runs), "methods": methods,
                        "pi_hat": {m: [float(np.mean([h[m] for h in hats])), float(np.std([h[m] for h in hats]))] for m in hats[0]}}
        print(f"  {name} pi={pi}: done ({len(runs)} run(s), {len(y_all[idx]):,} records)", flush=True)
    return res


def main():
    mva = pd.read_pickle(P / "meta_val.pkl")
    mte = pd.read_pickle(P / "meta_test.pkl")
    yva, yte, at = mva.label_binary.to_numpy(), mte.label_binary.to_numpy(), mte.attack_type.to_numpy()
    p3, p6 = json.load(open(R / "phase3_results.json")), json.load(open(R / "phase6_results.json"))
    held = p6["withheld"]
    out = {"pis": [str(p) for p in PIS], "seeds": SEEDS, "seen": {}, "unseen": {}}

    for m in MODELS:
        # ---- seen setting: Phase 3 models, benign vs all attacks
        s_te, s_va = log10s(np.load(R / f"scores_{m}_test.npy")), log10s(np.load(R / f"scores_{m}_val.npy"))
        fixed = {"F-primary": float(np.log10(p3["variants"][m]["primary"]["threshold"])),
                 "V-F1": float(np.log10(p3["variants"][m]["val_f1_tuned"]["threshold"]))}
        print(f"seen / {m}", flush=True)
        out["seen"][m] = run_setting(f"seen/{m}", s_te, yte, fixed, s_va[yva == 0])
        # ---- unseen setting: Phase 6 models, benign vs the four withheld types (known attacks removed from the target)
        s6 = log10s(np.load(R / f"scores_unseen_{m}_test.npy"))
        keep = (at == "Benign") | np.isin(at, held)
        fixed6 = {"F-primary": float(np.log10(p6["models"][m]["thresholds"]["primary"])),
                  "V-F1": float(np.log10(p6["models"][m]["thresholds"]["val_f1_tuned"]))}
        print(f"unseen / {m}", flush=True)
        out["unseen"][m] = run_setting(f"unseen/{m}", s6[keep], np.isin(at[keep], held).astype(int), fixed6, None)

    json.dump(out, open(R / "phase12_thresholds.json", "w"), indent=2)

    # ---------------- report
    names = ["F-primary", "F-ben-p95", "F-ben-p99", "F-ben-p99.9", "B-1%", "B-5%", "B-10%", "P-gmm", "P-tail", "T-flag-all", "O-prev", "O-F1", "V-F1"]
    label = {"T-flag-all": "T-flag-all (trivial: flag everything)","F-primary": "F-primary (fixed rule of Phase 3/6)", "O-prev": "O-prev (needs true prevalence)", "O-F1": "O-F1 (needs target labels)", "V-F1": "V-F1 (needs validation labels)"}
    L = ["# Phase 12 - label-free threshold selection at controlled attack prevalence (computed from saved scores)\n",
         "Methods and prevalence grid were fixed before any result was seen (see the header of `src/phase12_thresholds.py`). Values are means over 5 resamplings of the evaluation set "
         "(natural prevalence: one run; each model was trained once, so intervals do not include training randomness). `O-*` and `V-F1` rows use labels and only show what is achievable.\n"]
    for setting, desc in (("seen", "seen attack types (Phase 3 models, test partition, benign vs all attacks)"),
                          ("unseen", "unseen attack types (Phase 6 models; benign vs the four withheld types Bot, DDoS-HOIC, SSH-BruteForce, DoS-Hulk)")):
        for m in MODELS:
            res = out[setting][m]
            cols = [f"{float(res[str(p)]['true_prevalence']):.3f}" for p in PIS]
            L += [f"\n## {SHORT[m]}, {desc}\n"]
            for metric, title, nd in (("f1", "F1", 3), ("recall", "Recall", 3), ("fpr", "False-positive rate", 4)):
                L += [f"**{title}** by attack prevalence of the target set\n", "| Method | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
                for nm in names:
                    if nm not in res[str(PIS[0])]["methods"]:
                        continue
                    row = []
                    for p in PIS:
                        mu, sd = res[str(p)]["methods"][nm][metric]
                        row.append(f"{mu:.{nd}f}" + (f" ±{sd:.{nd}f}" if res[str(p)]["n_runs"] > 1 else ""))
                    L.append(f"| {label.get(nm, nm)} | " + " | ".join(row) + " |")
                L.append("")
            L += ["**Estimated attack share (pi_hat) against the true share**\n", "| Estimator | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
            for nm in ("P-gmm", "P-tail"):
                if nm in res[str(PIS[0])]["pi_hat"]:
                    L.append(f"| {nm} | " + " | ".join(f"{res[str(p)]['pi_hat'][nm][0]:.3f}" for p in PIS) + " |")
            L.append("")
    (REP / "phase12_thresholds.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/phase12_thresholds.md")


if __name__ == "__main__":
    main()
