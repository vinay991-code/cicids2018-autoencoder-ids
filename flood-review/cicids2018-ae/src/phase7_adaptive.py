"""Phase 7 / Experiment D: adaptive baseline.

Arms (all start from the same warm-up model and the same initial threshold):
  A_static              fixed model, fixed threshold (p99 of warm-up benign validation errors).
  B_adaptive_threshold  same fixed model; threshold re-estimated every interval as median + k*MAD of the recent
                        log10 scores of ALL records (robust, label-free); k is calibrated so the starting
                        threshold equals A's.
  C_adaptive_baseline   B's threshold rule + periodic fine-tuning of the autoencoder on a rolling TRUSTED buffer.
Trusted buffer: per interval a uniform random sample of records; only records the arm itself scored as normal
(score <= its threshold) enter. Contamination guards: per-interval sample cap, 25% replay of clean warm-up
records, and an update is rejected if the error on a fixed clean anchor set exceeds `anchor_tol` x its initial value.
Attack labels never control any update; they are used only to count results and to report buffer contamination.

Usage: python src/phase7_adaptive.py --tag main
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from adaptive import ArmCfg, run_stream
from stream_setup import get_baseline, interval_ids

ARMS = [ArmCfg("A_static"), ArmCfg("B_adaptive_threshold", adapt_threshold=True),
        ArmCfg("C_adaptive_baseline", adapt_threshold=True, retrain=True)]


def metrics(tp, fp, fn, tn):
    div = lambda a, b: a / b if b else float("nan")
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"n": tp + fp + fn + tn, "attacks": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if (tp and p == p and r == r) else (0.0 if tp + fn else float("nan")),
            "fpr": div(fp, fp + tn), "flagged_share": div(tp + fp, tp + fp + fn + tn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--interval-records", type=int, default=50_000, help="update interval, in records (stream order)")
    ap.add_argument("--sample", type=int, default=10_000)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--retrain-every", type=int, default=3)
    ap.add_argument("--buffer-intervals", type=int, default=12)
    ap.add_argument("--replay-frac", type=float, default=0.25)
    ap.add_argument("--ft-epochs", type=int, default=3)
    ap.add_argument("--ft-lr", type=float, default=5e-4)
    ap.add_argument("--ft-batch", type=int, default=1024)
    ap.add_argument("--anchor-tol", type=float, default=3.0)
    ap.add_argument("--min-trusted", type=int, default=5000)
    ap.add_argument("--warm-days", type=int, default=2)
    ap.add_argument("--init-train-rows", type=int, default=500_000)
    ap.add_argument("--init-epochs", type=int, default=50)
    ap.add_argument("--max-intervals", type=int, default=None, help="smoke test")
    ap.add_argument("--baseline-tag", default="main")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    B = get_baseline(args.warm_days, args.init_train_rows, args.init_epochs, args.seed, args.baseline_tag)
    print(f"baseline: thr0 {B['thr0']:.3e}, k {B['k']:.3f}, warm-up {B['info']['warm_records']:,} records, "
          f"eval {B['info']['eval_records']:,} records, days {B['info']['eval_days']}  [{time.time()-t0:.0f}s]", flush=True)
    m = B["meta"].iloc[B["e0"]:].reset_index(drop=True)
    X = B["X"][B["e0"]:]
    iid = interval_ids(m, args.interval_records)
    codes, types = pd.factorize(m.attack_type)
    if args.max_intervals:
        keep = np.isin(iid, np.unique(iid)[:args.max_intervals])
        n = int(keep.sum())
        assert keep[:n].all(), "smoke-test intervals must be a prefix of the stream"
        m, X, iid, codes = m.iloc[:n], X[:n], iid[:n], codes[:n]
    y = m.label_binary.to_numpy()
    cfg = dict(sample=args.sample, window=args.window, retrain_every=args.retrain_every, buffer_intervals=args.buffer_intervals,
               replay_frac=args.replay_frac, ft_epochs=args.ft_epochs, ft_lr=args.ft_lr, ft_batch=args.ft_batch,
               anchor_tol=args.anchor_tol, min_trusted=args.min_trusted, seed=args.seed, n_types=len(types))
    res = run_stream(X, y, codes, m.day.to_numpy(), iid, B["model"], B["thr0"], B["k"], ARMS, cfg, B["anchor"], B["replay"])
    print(f"stream finished [{time.time()-t0:.0f}s]", flush=True)

    ids = [r[0] for r in res["intervals"]]
    days = [r[1] for r in res["intervals"]]
    out = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "baseline": B["info"], "types": list(types), "arms": {}}
    rows = []
    for name, arm in res["arms"].items():
        cnt = np.array(arm.counts)
        for i in range(len(ids)):
            rows.append({"arm": name, "interval": ids[i], "day": days[i], "n": int(cnt[i].sum()), "tp": cnt[i][0], "fp": cnt[i][1],
                         "fn": cnt[i][2], "tn": cnt[i][3], "threshold": arm.thrs[i]})
        overall = metrics(*[int(v) for v in cnt.sum(0)])
        per_day = {}
        for d in sorted(set(days)):
            mk = np.array([x == d for x in days])
            per_day[d] = metrics(*[int(v) for v in cnt[mk].sum(0)])
            per_day[d]["median_threshold"] = float(np.median(np.array(arm.thrs)[mk]))
        flagged_t = np.sum(arm.by_type_flagged, axis=0)
        tot_t = np.bincount(codes, minlength=len(types))
        per_type = {str(t): {"n": int(tot_t[i]), "flagged_share": (float(flagged_t[i] / tot_t[i]) if tot_t[i] else None)} for i, t in enumerate(types)}
        out["arms"][name] = {"overall": overall, "per_day": per_day, "per_type": per_type, "retrain_log": arm.retrain_log,
                             "threshold_trace": [float(v) for v in arm.thrs]}
    # threshold-free quality of the two distinct score streams, per day
    sc = res["scores"]
    out["auc_per_day"] = {}
    for key, s in sc.items():
        out["auc_per_day"][key] = {}
        for d in sorted(set(days)):
            mk = (m.day.to_numpy() == d)
            if y[mk].sum() >= 100 and (1 - y[mk]).sum() >= 100:
                out["auc_per_day"][key][d] = {"roc_auc": float(roc_auc_score(y[mk], s[mk])), "pr_auc": float(average_precision_score(y[mk], s[mk]))}
    out["intervals"] = {"ids": ids, "days": days, "n_per_interval": [r[2] for r in res["intervals"]]}
    json.dump(out, open(args.out_dir / f"phase7_{args.tag}.json", "w"), indent=2, default=float)
    pd.DataFrame(rows).to_csv(args.out_dir / f"phase7_{args.tag}_intervals.csv", index=False)
    for name, a in out["arms"].items():
        o = a["overall"]
        print(f"{name:22s} overall: precision {o['precision']:.4f} recall {o['recall']:.4f} F1 {o['f1']:.4f} FPR {o['fpr']:.4f} flagged {o['flagged_share']:.4f}", flush=True)
    print("Phase 7 run done")


if __name__ == "__main__":
    main()
