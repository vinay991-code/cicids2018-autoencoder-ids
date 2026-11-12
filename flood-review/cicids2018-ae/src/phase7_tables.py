"""Phase 7 tables and figure from results/phase7_*.json and the per-interval counts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plots import plot_phase7

R = Path("results")
ARMS = ["A_static", "B_adaptive_threshold", "C_adaptive_baseline"]
SHORT = {"A_static": "A static", "B_adaptive_threshold": "B adaptive threshold", "C_adaptive_baseline": "C adaptive baseline"}


def f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def rates(c):
    tp, fp, fn, tn = c
    div = lambda a, b: a / b if b else np.nan
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if tp else 0.0, "fpr": div(fp, fp + tn)}


def block_boot(iv, n_boot=1000, block=8, seed=0):
    """Moving-block bootstrap over update intervals (keeps neighbouring intervals together)."""
    rng = np.random.default_rng(seed)
    cnt = {a: iv[iv.arm == a][["tp", "fp", "fn", "tn"]].to_numpy() for a in ARMS}
    n = len(cnt[ARMS[0]])
    nb = -(-n // block)
    out = {a: [] for a in ARMS}
    for _ in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        for a in ARMS:
            out[a].append(rates(cnt[a][idx].sum(0)))
    return out


def main():
    o = json.load(open(R / "phase7_main.json"))
    iv = pd.read_csv(R / "phase7_main_intervals.csv")
    days = sorted(o["arms"][ARMS[0]]["per_day"])
    tot = o["arms"][ARMS[0]]["overall"]
    o["stream_attack_share"] = tot["attacks"] / tot["n"]
    L = []
    a = L.append
    B = o["baseline"]
    cfg = o["config"]
    a(f"Warm-up (initial baseline): days {', '.join(B['warm_days'])}; one-class AE trained on {B['init_train_rows']:,} benign records, threshold = 99th percentile of "
      f"{B['init_val_rows']:,} later warm-up benign records ({B['thr0']:.4g}). Evaluation stream: {B['eval_records']:,} records on {len(days)} days "
      f"({tot['attacks']:,} attacks, {100*o['stream_attack_share']:.2f}%). Update interval {cfg['interval_records']:,} records; threshold window {cfg['window']} intervals; "
      f"retrain every {cfg['retrain_every']} intervals on the last {cfg['buffer_intervals']} intervals' trusted samples ({cfg['sample']:,} sampled per interval) "
      f"with {int(100*cfg['replay_frac'])}% clean replay, {cfg['ft_epochs']} epochs at lr {cfg['ft_lr']}; anchor tolerance {cfg['anchor_tol']}x. One run, seed {cfg['seed']}.\n")
    a("## Table 2a - overall, all evaluation days\n\n| Metric | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|" + "---:|" * 3)
    for k, lab in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("fpr", "False-positive rate"), ("flagged_share", "Share of records flagged")):
        a(f"| {lab} | " + " | ".join(f(o["arms"][x]["overall"][k]) for x in ARMS) + " |")
    boot = block_boot(iv)
    bd = {}
    a("\nMoving-block bootstrap over update intervals (1,000 resamples, blocks of 8 intervals), 95% interval of the difference to the static arm A:\n")
    a("| Metric | B - A | C - A |\n|---|---|---|")
    for k in ("f1", "recall", "fpr"):
        row = []
        for x in ARMS[1:]:
            d = np.array([bx[k] - ba[k] for bx, ba in zip(boot[x], boot["A_static"])])
            row.append(f"{np.nanmean(d):+.4f} [{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}]")
            bd.setdefault(k, {})[f"{x} - A_static"] = [float(np.nanmean(d)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        a(f"| {k} | {row[0]} | {row[1]} |")
    a("\nThe intervals describe sampling variation across the stream only (one training run, one seed).\n")
    a("## Table 2b - per day (attack share varies: days 02-22 and 02-23 have 319 and 535 attacks, so recall/F1 there are not meaningful)\n")
    a("| Day | Records | Attacks | " + " | ".join(f"{SHORT[x]}: F1 / recall / FPR" for x in ARMS) + " |\n|---|---:|---:|" + "---|" * 3)
    for d in days:
        n = o["arms"][ARMS[0]]["per_day"][d]
        a(f"| {d} | {n['n']:,} | {n['attacks']:,} | " + " | ".join(
            f"{f(o['arms'][x]['per_day'][d]['f1'], 3)} / {f(o['arms'][x]['per_day'][d]['recall'], 3)} / {f(o['arms'][x]['per_day'][d]['fpr'], 3)}" for x in ARMS) + " |")
    a("\n## Table 2c - ranking quality per day (ROC-AUC / PR-AUC), static model vs retrained model\n\n| Day | Attack types (day) | Static model (A, B) | Adaptive baseline model (C) |\n|---|---|---|---|")
    types_by_day = {"2018-02-16": "Hulk, SlowHTTPTest", "2018-02-20": "LOIC-HTTP", "2018-02-21": "HOIC, LOIC-UDP", "2018-02-22": "web attacks (few)",
                    "2018-02-23": "web attacks (few)", "2018-02-28": "Infiltration", "2018-03-01": "Infiltration", "2018-03-02": "Bot"}
    for d in days:
        s, c = o["auc_per_day"]["static"].get(d), o["auc_per_day"]["C_adaptive_baseline"].get(d)
        if s:
            a(f"| {d} | {types_by_day.get(d, '')} | {s['roc_auc']:.3f} / {s['pr_auc']:.3f} | {c['roc_auc']:.3f} / {c['pr_auc']:.3f} |")
    a("\n## Table 2d - threshold behaviour\n\nThe largest reconstruction error in the Phase 3 test scores was 0.072, so a threshold above that flags essentially nothing "
      "(for arm C's retrained models this is an approximation).\n\n| Arm | Minimum | Median | Maximum | Intervals with threshold > 0.072 |\n|---|---:|---:|---:|---:|")
    for x in ARMS:
        t = np.array(o["arms"][x]["threshold_trace"])
        a(f"| {SHORT[x]} | {t.min():.3g} | {np.median(t):.3g} | {t.max():.3g} | {100*(t > 0.072).mean():.1f}% |")
    rl = pd.DataFrame(o["arms"]["C_adaptive_baseline"]["retrain_log"])
    a(f"\n## Table 2e - retraining and contamination (arm C)\n\n- Retrainings: {len(rl)}; accepted by the anchor check: {int(rl.accepted.sum())}.\n"
      f"- Share of attack records in the trusted buffer used for retraining (diagnostic from labels, never used for control): mean {rl.trusted_attack_share.mean():.3f}, maximum {rl.trusted_attack_share.max():.3f}; "
      f"attack share of the whole stream {o['stream_attack_share']:.3f}.\n"
      f"- Error on the clean anchor set after retraining relative to the initial model: median {rl.anchor_error_ratio.median():.2f}, maximum {rl.anchor_error_ratio.max():.2f} (rejection threshold {cfg['anchor_tol']}).\n")
    a("## Table 2f - attack types: share flagged over the whole stream\n\n| Type | Records | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|---:|---:|---:|---:|")
    for t in o["types"]:
        pt = o["arms"][ARMS[0]]["per_type"][t]
        a(f"| {t} | {pt['n']:,} | " + " | ".join(f(o["arms"][x]["per_type"][t]["flagged_share"], 3) for x in ARMS) + " |")
    sens = []
    for tag in ("retrain1", "main", "retrain6"):
        p = R / f"phase7_{tag}.json"
        if p.exists():
            s = json.load(open(p))
            sens.append((s["config"]["retrain_every"], s))
    if len(sens) > 1:
        a("\n## Table 2g - sensitivity to the retraining interval (arm C; other settings unchanged)\n\n| Retrain every (intervals of 50,000 records) | Retrainings | Mean trusted-buffer attack share | Overall F1 | Recall | FPR | Per-day ROC-AUC (02-16 / 02-20 / 02-21 / 03-02) |\n|---:|---:|---:|---:|---:|---:|---|")
        for n_, s in sorted(sens, key=lambda z: z[0]):
            c = s["arms"]["C_adaptive_baseline"]
            auc = s["auc_per_day"]["C_adaptive_baseline"]
            a(f"| {n_} | {len(c['retrain_log'])} | {np.nanmean([r['trusted_attack_share'] for r in c['retrain_log']]):.3f} | {c['overall']['f1']:.4f} | {c['overall']['recall']:.4f} | {c['overall']['fpr']:.4f} | "
              + " / ".join(f"{auc[d]['roc_auc']:.3f}" for d in ("2018-02-16", "2018-02-20", "2018-02-21", "2018-03-02")) + " |")
    (Path("reports") / "phase7_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    plot_phase7(o, R / "phase7_adaptive_overview.png")
    json.dump({"bootstrap_difference_mean_lo_hi": bd}, open(R / "phase7_bootstrap.json", "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
