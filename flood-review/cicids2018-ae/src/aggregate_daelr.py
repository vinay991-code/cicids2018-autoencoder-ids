"""Aggregate Phase 14 (DAE-LR) over seeds and compare with the Phase 3 baselines of the same seeds.

Baselines: `unsup_mixed` (same data, ordinary loss, no latent term) and `normal_only` (benign-only training; its threshold uses benign validation labels).
Seed 42 baselines come from results/phase3_results.json (the reference laptop run), other seeds from seeds/seed_<n>/results/.

Usage: python src/aggregate_daelr.py --dir results_daelr --seeds 42 1 2
Writes <dir>/daelr_summary.json and .md
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
    ap.add_argument("--dir", type=Path, default=Path("results_daelr"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--seeds-dir", type=Path, default=Path("seeds"))
    args = ap.parse_args()
    rows = {}

    def add(method, metric, v):
        rows.setdefault(method, {}).setdefault(metric, []).append(float(v))

    used = []
    for s in args.seeds:
        tj = args.dir / f"phase14_seed{s}.json"
        bj = (Path("results") if s == 42 else args.seeds_dir / f"seed_{s}" / "results") / "phase3_results.json"
        if not tj.exists() or not bj.exists():
            print(f"seed {s}: missing {tj if not tj.exists() else bj}, skipped")
            continue
        t, b = json.load(open(tj)), json.load(open(bj))
        if not all(v in t["models"] for v in ("mixed_lam0.01", "clean_lam10")):
            print(f"seed {s}: not finished, skipped")
            continue
        used.append(s)
        for m, label in (("unsup_mixed", "unsup_mixed (ordinary loss)"), ("normal_only", "normal_only (benign-only, ordinary loss)")):
            v = b["variants"][m]
            add(label, "ROC-AUC", v["ranking"]["roc_auc"]); add(label, "PR-AUC", v["ranking"]["pr_auc"])
            add(label, "F1 at its own rule", v["primary"]["f1"]); add(label, "Recall at its own rule", v["primary"]["recall"]); add(label, "FPR at its own rule", v["primary"]["fpr"])
        for tag, r in t["models"].items():
            variant, lam = tag.split("_lam")
            label = ("DAE-LR trained on unlabeled mixed data" if variant == "mixed" else "DAE-LR trained on benign rows only") + f", lambda={lam}"
            k = "L1" if variant == "mixed" else "B99"
            add(label, "ROC-AUC", r["ranking"]["roc_auc"]); add(label, "PR-AUC", r["ranking"]["pr_auc"])
            add(label, f"F1 at {k}", r["at_threshold"][k]["f1"]); add(label, f"Recall at {k}", r["at_threshold"][k]["recall"]); add(label, f"FPR at {k}", r["at_threshold"][k]["fpr"])
            add(label, "(analysis) validation reconstruction error (sum) at best epoch", r["final_val_recon_sum"])
            add(label, "(analysis) latent term ||z-c||^2 at best epoch", r["final_val_latent_sum"])
    if not used:
        raise SystemExit("no complete seeds found")
    summary = {"seeds": used, "methods": {m: {k: summarise(v) for k, v in d.items()} for m, d in rows.items()}}
    json.dump(summary, open(args.dir / "daelr_summary.json", "w"), indent=2)
    f = lambda x: "n/a" if x != x else f"{x:.3f}"
    L = ["# DAE-LR over seeds (Phase 14)\n",
         f"Seeds: {', '.join(map(str, used))} (n = {len(used)}). Trained on the same rows as `unsup_mixed` (mixed) or `normal_only` (benign only); only the loss differs. Intervals: mean ± t(0.975, n-1)·sd/√n, "
         "cut to [0, 1]; with n this small they are wide. L1 = 95th percentile of all unlabeled validation errors (no attack labels); B99 = 99th percentile of benign validation errors (uses validation labels). "
         "Rows marked (analysis) are diagnostics.\n",
         "| Method | Metric | Mean | sd | 95% CI of mean | min–max |", "|---|---|---:|---:|---:|---:|"]
    for m, d in summary["methods"].items():
        for k, s in d.items():
            L.append(f"| {m} | {k} | {f(s['mean'])} | {f(s['sd'])} | [{f(s['ci95'][0])}, {f(s['ci95'][1])}] | {f(s['min'])}–{f(s['max'])} |")
    (args.dir / "daelr_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
