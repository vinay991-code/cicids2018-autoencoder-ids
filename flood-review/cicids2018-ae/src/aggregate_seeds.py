"""Aggregate the results of several training runs (seeds) into means, standard deviations and 95% intervals.

Reads <seeds-dir>/seed_<n>/results/{phase3,phase4,phase6}_results.json written by run_seeds.py, plus any extra run given as
--extra <seed>=<results dir> (for example the reference seed 42 run). Each run is one observation per metric.

Reported per metric: number of runs n, mean, standard deviation (ddof=1), a 95% t-interval for the MEAN over training seeds
(mean +/- t(0.975, n-1) * sd / sqrt(n)), and the minimum and maximum. The sd and min-max describe how much a single run can differ
from another; the interval of the mean describes how well the average is known. Only training randomness varies: the train/validation/test
split, and therefore the test set, is the same in every run. With few seeds the intervals are wide; that is the honest answer.

Usage: python src/aggregate_seeds.py --seeds-dir seeds [--extra 42=results] [--out seeds]
Writes <out>/seeds_summary.json and <out>/seeds_summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEN = [("unsup_mixed", "phase3", "primary", "Unsupervised AE"), ("normal_only", "phase3", "primary", "One-class AE"),
        ("supervised", "phase4", "threshold_0.5", "Supervised AE"), ("supervised_recon", "phase4", "threshold_0.5", "Supervised AE + decoder")]
UNSEEN = [("unsup_mixed", "Unsupervised AE"), ("normal_only", "One-class AE"), ("supervised", "Supervised AE")]


def load_run(res):
    """One run's metrics as {key: value}."""
    p = {k: json.load(open(res / f"{k}_results.json")) for k in ("phase3", "phase4", "phase6")}
    out = {}
    for m, ph, key, _ in SEEN:
        v = p[ph]["variants"][m]
        out[f"seen|{m}|ROC-AUC"] = v["ranking"]["roc_auc"]
        out[f"seen|{m}|PR-AUC"] = v["ranking"]["pr_auc"]
        for k, nm in (("f1", "F1"), ("recall", "Recall"), ("precision", "Precision"), ("fpr", "False-positive rate")):
            out[f"seen|{m}|{nm}"] = v[key][k]
    held = p["phase6"]["withheld"]
    for m, _ in UNSEEN:
        r = p["phase6"]["models"][m]
        out[f"unseen|{m}|Known attacks: recall"] = r["known"]["primary"]["recall"]
        out[f"unseen|{m}|Benign: false-positive rate"] = r["known"]["primary"]["fpr"]
        out[f"unseen|{m}|Unseen attacks: recall"] = r["unseen"]["primary"]["recall"]
        out[f"unseen|{m}|Known set ROC-AUC"] = r["known_ranking"]["roc_auc"]
        out[f"unseen|{m}|Unseen set ROC-AUC"] = r["unseen_ranking"]["roc_auc"]
        for h in held:
            out[f"unseen|{m}|{h}: share flagged"] = r["unseen"]["primary"]["per_type"][h]["flagged_share"]
            out[f"unseen|{m}|{h}: ROC-AUC vs benign"] = r["unseen"]["primary"]["per_type"][h]["roc_auc_vs_benign"]
    return out


def summarise(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci95": [max(0.0, mean - half), min(1.0, mean + half)], "min": float(v.min()), "max": float(v.max()), "values": [float(x) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", type=Path, default=Path("seeds"))
    ap.add_argument("--extra", action="append", default=[], help="extra run as <seed>=<results dir>, e.g. 42=results")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out or args.seeds_dir
    runs, skipped = {}, []
    for d in sorted(args.seeds_dir.glob("seed_*")):
        try:
            runs[d.name.replace("seed_", "")] = load_run(d / "results")
        except (FileNotFoundError, KeyError) as e:
            skipped.append((d.name, repr(e)))
    for e in args.extra:
        s, path = e.split("=", 1)
        runs[s] = load_run(Path(path))
    if not runs:
        raise SystemExit("no completed runs found")
    keys = list(next(iter(runs.values())))
    summary = {"seeds": sorted(runs, key=lambda s: int(s)), "skipped_incomplete": skipped,
               "metrics": {k: summarise([runs[s][k] for s in runs]) for k in keys}}
    json.dump(summary, open(out_dir / "seeds_summary.json", "w"), indent=2)

    n = len(runs)
    f = lambda x, nd=3: "n/a" if x != x else f"{x:.{nd}f}"
    L = ["# Results over several training seeds\n",
         f"Runs (seeds): {', '.join(summary['seeds'])}  (n = {n}). Only training randomness differs between runs (initialisation, batch order, the 1,000,000 training rows); the data split and test set are identical. "
         "`95% CI of mean` is mean ± t(0.975, n-1)·sd/√n; `sd` and `min`–`max` show how much single runs differ. All metrics are proportions, so the interval is cut to [0, 1]. With n = " + str(n) + " the intervals are wide.\n"]
    if skipped:
        L.append(f"Skipped incomplete seed folders: {skipped}\n")
    for setting, title in (("seen", "Table 1 over seeds: attack types seen in training (test partition, 2,163,852 records)"),
                           ("unseen", "Table 5 over seeds: four attack types withheld from training")):
        L += [f"\n## {title}\n", "| Model | Metric | Mean | sd | 95% CI of mean | min–max |", "|---|---|---:|---:|---:|---:|"]
        names = {m: nm for m, *_, nm in SEEN} if setting == "seen" else {m: nm for m, nm in UNSEEN}
        for k, s in summary["metrics"].items():
            st, m, metric = k.split("|")
            if st != setting:
                continue
            ci = f"[{f(s['ci95'][0])}, {f(s['ci95'][1])}]"
            L.append(f"| {names[m]} | {metric} | {f(s['mean'])} | {f(s['sd'])} | {ci} | {f(s['min'])}–{f(s['max'])} |")
    L.append("\nValues per run are in `seeds_summary.json`.")
    (out_dir / "seeds_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
