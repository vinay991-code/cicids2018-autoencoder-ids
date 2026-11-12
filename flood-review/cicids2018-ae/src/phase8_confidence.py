"""Phase 8 / Experiment E: confidence-aware (gated) vs unrestricted baseline adaptation.

All arms start from the Phase 7 warm-up model; every arm's threshold is the q-quantile (default 0.95) of the errors of a
fixed clean anchor set under the arm's CURRENT model, recomputed after each update (replaces Phase 7's unstable
median/MAD rule). Reconstruction error is an ordinal anomaly score, not a calibrated probability.
  S_static          initial model and threshold, never updated (reference).
  U_unrestricted    unrestricted adaptation: every sampled record enters the retraining buffer.
  M_margin_only     ablation: only records with score <= margin * threshold are admitted.
  G_gated           confidence-gated: margin + temporal consistency (a deviating interval is held until the deviation
                    persists for `persist` intervals) + quarantine of records scored above the threshold.
Labels never control any update; they are used for result counts and contamination diagnostics only.

Usage: python src/phase8_confidence.py --tag main
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from adaptive import ArmCfg, run_stream
from phase7_adaptive import metrics
from stream_setup import get_baseline, interval_ids

ARMS = [ArmCfg("S_static", thr_mode="anchor_q"),
        ArmCfg("U_unrestricted", thr_mode="anchor_q", retrain=True, admission="all"),
        ArmCfg("M_margin_only", thr_mode="anchor_q", retrain=True, admission="margin"),
        ArmCfg("G_gated", thr_mode="anchor_q", retrain=True, admission="gated")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--baseline-tag", default="main")
    ap.add_argument("--interval-records", type=int, default=50_000)
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
    ap.add_argument("--anchor-q", type=float, default=0.95)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--level-tol", type=float, default=0.5, help="decades of log10 median score")
    ap.add_argument("--susp-max", type=float, default=0.15)
    ap.add_argument("--persist", type=int, default=3)
    ap.add_argument("--ewma", type=float, default=0.1)
    ap.add_argument("--warm-days", type=int, default=2)
    ap.add_argument("--max-intervals", type=int, default=None, help="smoke test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    B = get_baseline(args.warm_days, tag=args.baseline_tag, seed=args.seed)
    m = B["meta"].iloc[B["e0"]:].reset_index(drop=True)
    X = B["X"][B["e0"]:]
    iid = interval_ids(m, args.interval_records)
    codes, types = pd.factorize(m.attack_type)
    if args.max_intervals:
        n = int(np.isin(iid, np.unique(iid)[:args.max_intervals]).sum())
        m, X, iid, codes = m.iloc[:n], X[:n], iid[:n], codes[:n]
    y = m.label_binary.to_numpy()
    cfg = dict(sample=args.sample, window=args.window, retrain_every=args.retrain_every, buffer_intervals=args.buffer_intervals,
               replay_frac=args.replay_frac, ft_epochs=args.ft_epochs, ft_lr=args.ft_lr, ft_batch=args.ft_batch,
               anchor_tol=args.anchor_tol, min_trusted=args.min_trusted, seed=args.seed, n_types=len(types), anchor_q=args.anchor_q,
               gate=dict(margin=args.margin, level_tol=args.level_tol, susp_max=args.susp_max, persist=args.persist, ewma=args.ewma))
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
        per_day = {}
        for d in sorted(set(days)):
            mk = np.array([x == d for x in days])
            per_day[d] = metrics(*[int(v) for v in cnt[mk].sum(0)])
        flagged_t = np.sum(arm.by_type_flagged, axis=0)
        tot_t = np.bincount(codes, minlength=len(types))
        out["arms"][name] = {"overall": metrics(*[int(v) for v in cnt.sum(0)]), "per_day": per_day,
                             "per_type": {str(t): {"n": int(tot_t[i]), "flagged_share": (float(flagged_t[i] / tot_t[i]) if tot_t[i] else None)}
                                          for i, t in enumerate(types)},
                             "retrain_log": arm.retrain_log, "gate_log": arm.gate_log, "threshold_trace": [float(v) for v in arm.thrs]}
    out["auc_per_day"] = {}
    for key, s in res["scores"].items():
        out["auc_per_day"][key] = {}
        for d in sorted(set(days)):
            mk = (m.day.to_numpy() == d)
            if y[mk].sum() >= 100 and (1 - y[mk]).sum() >= 100:
                out["auc_per_day"][key][d] = {"roc_auc": float(roc_auc_score(y[mk], s[mk])), "pr_auc": float(average_precision_score(y[mk], s[mk]))}
    out["intervals"] = {"ids": ids, "days": days, "n_per_interval": [r[2] for r in res["intervals"]]}
    json.dump(out, open(args.out_dir / f"phase8_{args.tag}.json", "w"), indent=2, default=float)
    pd.DataFrame(rows).to_csv(args.out_dir / f"phase8_{args.tag}_intervals.csv", index=False)
    for name, a in out["arms"].items():
        o = a["overall"]
        rl = pd.DataFrame(a["retrain_log"])
        cont = f"buffer attack share mean {rl.trusted_attack_share.mean():.3f}" if len(rl) else "no retraining"
        print(f"{name:16s} precision {o['precision']:.4f} recall {o['recall']:.4f} F1 {o['f1']:.4f} FPR {o['fpr']:.4f} | {cont}", flush=True)
    print("Phase 8 run done")


if __name__ == "__main__":
    main()
