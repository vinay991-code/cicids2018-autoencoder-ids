"""Phase 9 / Experiment F: concept-drift monitoring.

A label-free DriftMonitor (ADWIN detectors on the fixed sentinel model's error distribution, flagged share, five traffic
features and the UDP share; see drift.py) watches the stream. Arms (all from the Phase 7 warm-up model; threshold = 95th
percentile of the clean anchor set under the arm's current model, as in Phase 8):
  A_no_detection     no drift detection and no updates (static).
  A2_periodic        reference without detection: unrestricted retraining on a fixed schedule (every 3 intervals).
  B_drift_adaptive   drift detection + adaptive baseline: after an alarm retrain on the recent samples for `response` rounds.
  C_drift_confirmed  drift detection + confidence-aware updates: an alarm is acted on only if the deviation persists in the
                     next interval, and the retraining data exclude the highest-scoring 20% of the recent samples.
Test labels never control the detector or any update; they are used for evaluation only.

Usage: python src/phase9_drift.py --tag main
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from adaptive import ArmCfg, run_stream
from drift import DriftMonitor
from models import recon_error
from phase7_adaptive import metrics
from stream_setup import get_baseline, interval_ids

ARMS = [ArmCfg("A_no_detection", thr_mode="anchor_q"),
        ArmCfg("A2_periodic", thr_mode="anchor_q", retrain=True, admission="all", trigger="periodic"),
        ArmCfg("B_drift_adaptive", thr_mode="anchor_q", retrain=True, admission="all", trigger="drift"),
        ArmCfg("C_drift_confirmed", thr_mode="anchor_q", retrain=True, admission="relative", trigger="drift")]


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
    ap.add_argument("--level-tol", type=float, default=0.5)
    ap.add_argument("--susp-max", type=float, default=0.15)
    ap.add_argument("--persist", type=int, default=3)
    ap.add_argument("--ewma", type=float, default=0.1)
    ap.add_argument("--response", type=int, default=3, help="retraining rounds after an accepted alarm")
    ap.add_argument("--recent-intervals", type=int, default=3)
    ap.add_argument("--q-rel", type=float, default=0.8)
    ap.add_argument("--adwin-delta", type=float, default=0.002)
    ap.add_argument("--monitor-batch", type=int, default=5000)
    ap.add_argument("--ref-records", type=int, default=300_000, help="end of the warm-up period pre-fed to the detectors")
    ap.add_argument("--warm-days", type=int, default=2)
    ap.add_argument("--max-intervals", type=int, default=None, help="smoke test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    B = get_baseline(args.warm_days, tag=args.baseline_tag, seed=args.seed)
    e0 = B["e0"]
    m = B["meta"].iloc[e0:].reset_index(drop=True)
    X = B["X"][e0:]
    iid = interval_ids(m, args.interval_records)
    codes, types = pd.factorize(m.attack_type)
    if args.max_intervals:
        n = int(np.isin(iid, np.unique(iid)[:args.max_intervals]).sum())
        m, X, iid, codes = m.iloc[:n], X[:n], iid[:n], codes[:n]
    y = m.label_binary.to_numpy()
    thr_s = float(np.quantile(recon_error(B["model"], B["anchor"]), args.anchor_q))
    mon = DriftMonitor(B["model"], thr_s, batch=args.monitor_batch, delta=args.adwin_delta)
    mon.prefeed(np.asarray(B["X"][e0 - args.ref_records:e0]))
    cfg = dict(sample=args.sample, window=args.window, retrain_every=args.retrain_every, buffer_intervals=args.buffer_intervals,
               replay_frac=args.replay_frac, ft_epochs=args.ft_epochs, ft_lr=args.ft_lr, ft_batch=args.ft_batch,
               anchor_tol=args.anchor_tol, min_trusted=args.min_trusted, seed=args.seed, n_types=len(types), anchor_q=args.anchor_q,
               gate=dict(margin=args.margin, level_tol=args.level_tol, susp_max=args.susp_max, persist=args.persist, ewma=args.ewma),
               response=args.response, recent_intervals=args.recent_intervals, q_rel=args.q_rel)
    res = run_stream(X, y, codes, m.day.to_numpy(), iid, B["model"], B["thr0"], B["k"], ARMS, cfg, B["anchor"], B["replay"], monitor=mon)
    print(f"stream finished [{time.time()-t0:.0f}s]; alarms: {len(res['alarms'])}", flush=True)

    ids = [r[0] for r in res["intervals"]]
    days = [r[1] for r in res["intervals"]]
    out = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "baseline": B["info"], "types": list(types), "arms": {},
           "alarms": res["alarms"], "monitor_names": mon.names}
    rows = []
    for name, arm in res["arms"].items():
        cnt = np.array(arm.counts)
        for i in range(len(ids)):
            rows.append({"arm": name, "interval": ids[i], "day": days[i], "n": int(cnt[i].sum()), "tp": cnt[i][0], "fp": cnt[i][1],
                         "fn": cnt[i][2], "tn": cnt[i][3], "threshold": arm.thrs[i]})
        per_day = {d: metrics(*[int(v) for v in cnt[np.array([x == d for x in days])].sum(0)]) for d in sorted(set(days))}
        flagged_t = np.sum(arm.by_type_flagged, axis=0)
        tot_t = np.bincount(codes, minlength=len(types))
        out["arms"][name] = {"overall": metrics(*[int(v) for v in cnt.sum(0)]), "per_day": per_day,
                             "per_type": {str(t): {"n": int(tot_t[i]), "flagged_share": (float(flagged_t[i] / tot_t[i]) if tot_t[i] else None)}
                                          for i, t in enumerate(types)},
                             "retrain_log": arm.retrain_log, "drift_log": arm.drift_log}
    out["auc_per_day"] = {}
    for key, s in res["scores"].items():
        out["auc_per_day"][key] = {}
        for d in sorted(set(days)):
            mk = (m.day.to_numpy() == d)
            if y[mk].sum() >= 100 and (1 - y[mk]).sum() >= 100:
                out["auc_per_day"][key][d] = {"roc_auc": float(roc_auc_score(y[mk], s[mk])), "pr_auc": float(average_precision_score(y[mk], s[mk]))}
    # attack share of each monitoring batch, for classifying alarms in the evaluation (labels: evaluation only)
    out["alarm_attack_share"] = []
    for al in res["alarms"]:
        end = al["record"]
        lo = max(0, end - args.monitor_batch)
        out["alarm_attack_share"].append(float(y[lo:end].mean()))
    out["intervals"] = {"ids": ids, "days": days, "n_per_interval": [r[2] for r in res["intervals"]]}
    json.dump(out, open(args.out_dir / f"phase9_{args.tag}.json", "w"), indent=2, default=float)
    pd.DataFrame(rows).to_csv(args.out_dir / f"phase9_{args.tag}_intervals.csv", index=False)
    for name, a in out["arms"].items():
        o = a["overall"]
        print(f"{name:18s} precision {o['precision']:.4f} recall {o['recall']:.4f} F1 {o['f1']:.4f} FPR {o['fpr']:.4f} | retrainings {len(a['retrain_log'])}", flush=True)
    print("Phase 9 run done")


if __name__ == "__main__":
    main()
