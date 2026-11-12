"""Aggregate Phase 13 (trimmed-loss autoencoder) over seeds and compare with the Phase 3 baselines of the same seeds.

Baselines: `unsup_mixed` (same data, ordinary loss) and `normal_only` (benign-only training; its threshold rule uses benign validation labels).
Seed 42 baselines come from results/phase3_results.json (the reference laptop run), other seeds from seeds/seed_<n>/results/.

Usage: python src/aggregate_trimmed.py --trim-dir results_trim --seeds 42 1 2
Writes <trim-dir>/trimmed_summary.json and .md
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def summarise(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    mean, sd = float(v.mean()), (float(v.std(ddof=1)) if n > 1 else float("nan"))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci95": [max(0.0, mean - half), min(1.0, mean + half)], "min": float(v.min()), "max": float(v.max()), "values": [float(x) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim-dir", type=Path, default=Path("results_trim"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--seeds-dir", type=Path, default=Path("seeds"))
    args = ap.parse_args()
    rows = {}   # method -> metric -> list over seeds

    def add(method, metric, v):
        rows.setdefault(method, {}).setdefault(metric, []).append(float(v))

    used = []
    for s in args.seeds:
        tj = args.trim_dir / f"phase13_seed{s}.json"
        bj = (Path("results") if s == 42 else args.seeds_dir / f"seed_{s}" / "results") / "phase3_results.json"
        if not tj.exists() or not bj.exists():
            print(f"seed {s}: missing {tj if not tj.exists() else bj}, skipped")
            continue
        used.append(s)
        t, b = json.load(open(tj)), json.load(open(bj))
        for m, label in (("unsup_mixed", "unsup_mixed (ordinary loss, 0% trimmed)"), ("normal_only", "normal_only (benign-only; threshold uses benign labels)")):
            v = b["variants"][m]
            add(label, "ROC-AUC", v["ranking"]["roc_auc"]); add(label, "PR-AUC", v["ranking"]["pr_auc"])
            add(label, "F1 at its own rule", v["primary"]["f1"]); add(label, "Recall at its own rule", v["primary"]["recall"]); add(label, "FPR at its own rule", v["primary"]["fpr"])
        for tag, r in t["models"].items():
            label = f"trimmed {int(tag[4:])}%"
            add(label, "ROC-AUC", r["ranking"]["roc_auc"]); add(label, "PR-AUC", r["ranking"]["pr_auc"])
            for k in ("L1", "L2", "L3"):
                add(label, f"F1 at {k}", r["at_threshold"][k]["f1"])
            add(label, "Recall at L2", r["at_threshold"]["L2"]["recall"]); add(label, "FPR at L2", r["at_threshold"]["L2"]["fpr"])
            add(label, "(analysis) attack share kept in training", r["diagnostic_attack_share_kept"])
            add(label, "(analysis) share of attacks removed from training", r["diagnostic_attacks_removed_share"])
    if not used:
        raise SystemExit("no complete seeds found")
    summary = {"seeds": used, "methods": {m: {k: summarise(v) for k, v in d.items()} for m, d in rows.items()}}
    json.dump(summary, open(args.trim_dir / "trimmed_summary.json", "w"), indent=2)
    f = lambda x: "n/a" if x != x else f"{x:.3f}"
    L = ["# Trimmed-loss autoencoder over seeds (Phase 13)\n",
         f"Seeds: {', '.join(map(str, used))} (n = {len(used)}). Trained on the same unlabeled rows as `unsup_mixed`; only training randomness differs. Intervals: mean ± t(0.975, n-1)·sd/√n, cut to [0, 1]; "
         "with n this small they are wide. L1, L2 and L3 are label-free thresholds defined in `src/phase13_trimmed.py`. Rows marked (analysis) use labels for reporting only.\n",
         "| Method | Metric | Mean | sd | 95% CI of mean | min–max |", "|---|---|---:|---:|---:|---:|"]
    for m, d in summary["methods"].items():
        for k, s in d.items():
            L.append(f"| {m} | {k} | {f(s['mean'])} | {f(s['sd'])} | [{f(s['ci95'][0])}, {f(s['ci95'][1])}] | {f(s['min'])}–{f(s['max'])} |")
    (args.trim_dir / "trimmed_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
