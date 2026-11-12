"""Phase 9 tables (Table 4) and figure from results/phase9_*.json and the per-interval counts.

Evaluation-only ground truth: a day counts as 'shifted' if the static arm's false-positive rate on its benign records exceeds 0.5
(the benign traffic is then entirely above the warm-up model's threshold). True benign change points are the starts of days
whose shifted status differs from the previous day's (the day before the stream is the warm-up reference, not shifted).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plots import plot_phase9

R = Path("results")
ARMS = ["A_no_detection", "A2_periodic", "B_drift_adaptive", "C_drift_confirmed"]
SHORT = {"A_no_detection": "A no detection", "A2_periodic": "A2 periodic (ref.)", "B_drift_adaptive": "B drift-triggered", "C_drift_confirmed": "C drift-triggered + confirmation"}
RECOVER_FPR, F1_FRAC, ATTACK_BATCH = 0.10, 0.8, 0.25


def f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def rates(c):
    tp, fp, fn, tn = c
    div = lambda a, b: a / b if b else np.nan
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if tp else 0.0, "fpr": div(fp, fp + tn)}


def per_interval(iv_arm):
    d = iv_arm.reset_index(drop=True)
    fpr = (d.fp / (d.fp + d.tn).replace(0, np.nan)).to_numpy()
    rec = (d.tp / (d.tp + d.fn).replace(0, np.nan)).to_numpy()
    prec = (d.tp / (d.tp + d.fp).replace(0, np.nan)).to_numpy()
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    f1[np.isnan(prec) | np.isnan(rec)] = np.nan
    return fpr, rec, f1


def roll3(v):
    return pd.Series(v).rolling(3, min_periods=1).mean().to_numpy()


def block_boot(iv, arms, n_boot=1000, block=8, seed=0):
    rng = np.random.default_rng(seed)
    cnt = {a: iv[iv.arm == a][["tp", "fp", "fn", "tn"]].to_numpy() for a in arms}
    n = len(cnt[arms[0]])
    nb = -(-n // block)
    out = {a: [] for a in arms}
    for _ in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        for a in arms:
            out[a].append(rates(cnt[a][idx].sum(0)))
    return out


def main():
    o = json.load(open(R / "phase9_main.json"))
    iv = pd.read_csv(R / "phase9_main_intervals.csv")
    cfg = o["config"]
    days_list = o["intervals"]["days"]
    n_iv = o["intervals"]["n_per_interval"]
    days = sorted(set(days_list))
    tot = o["arms"][ARMS[0]]["overall"]
    # ---- evaluation ground truth
    shifted = {d: (o["arms"]["A_no_detection"]["per_day"][d]["fpr"] > 0.5) for d in days}
    prev, change = False, []
    for d in days:
        if shifted[d] != prev:
            i0 = days_list.index(d)
            change.append({"day": d, "interval": i0, "record": int(sum(n_iv[:i0])), "to": "shifted" if shifted[d] else "back to reference"})
        prev = shifted[d]
    day_len = {d: int(sum(n for dd, n in zip(days_list, n_iv) if dd == d)) for d in days}
    tol = {c["day"]: min(day_len[c["day"]], 12 * cfg["interval_records"]) for c in change}
    alarms = o["alarms"]
    for al, sh in zip(alarms, o["alarm_attack_share"]):
        al["attack_share"] = sh
    L = []
    a = L.append
    a(f"Evaluation stream: {tot['n']:,} records, {tot['attacks']:,} attacks, 8 days, update interval {cfg['interval_records']:,} records. Drift monitor: {len(o['monitor_names'])} ADWIN detectors "
      f"({', '.join(o['monitor_names'])}) on batches of {cfg['monitor_batch']:,} records scored by the fixed warm-up model, delta {cfg['adwin_delta']} split over the detectors (Bonferroni), "
      f"pre-fed with the last {cfg['ref_records']:,} warm-up records. Drift-triggered arms retrain for {cfg['response']} rounds after an alarm on the last {cfg['recent_intervals']} intervals' samples with 25% clean replay; "
      f"arm C acts only if the deviation persists one more interval (median score more than {cfg['level_tol']} decades from the reference, or more than {cfg['susp_max']:.0%} flagged) and excludes the highest-scoring {1-cfg['q_rel']:.0%} of the samples. "
      "Thresholds: 95th percentile of the clean anchor set under each arm's current model. One run, seed 42. Reconstruction error is an ordinal score, not a calibrated probability.\n")
    a("## Table 4a - overall\n\n| Metric | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|" + "---:|" * len(ARMS))
    for k, lab in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("fpr", "False-positive rate"), ("flagged_share", "Share flagged")):
        a(f"| {lab} | " + " | ".join(f(o["arms"][x]["overall"][k]) for x in ARMS) + " |")
    rows = []
    for x in ARMS:
        rl = pd.DataFrame(o["arms"][x]["retrain_log"])
        rows.append((len(rl), (rl.trusted_attack_share.mean() if len(rl) else np.nan), (rl.trusted_attack_share.max() if len(rl) else np.nan)))
    a("| Retrainings | " + " | ".join(str(r[0]) for r in rows) + " |")
    a("| Mean attack share in the retraining data (stream: %.3f) | " % (tot["attacks"] / tot["n"]) + " | ".join(f(r[1], 3) for r in rows) + " |")
    a("| Maximum attack share in the retraining data | " + " | ".join(f(r[2], 3) for r in rows) + " |")
    a("\n**Read with care:** on days where the static arm flags nearly every record (02-16, 02-21) its overall recall and F1 are inflated by flagging everything; compare false-positive rates and the per-day tables.\n")
    boot = block_boot(iv, ARMS)
    bd = {}
    a("Moving-block bootstrap over intervals (1,000 resamples, blocks of 8), 95% interval of the difference:\n\n| Metric | B - A2 (drift vs periodic) | C - B (confirmation) | B - A (drift vs none) |\n|---|---|---|---|")
    for k in ("f1", "recall", "fpr"):
        row = []
        for x, yv in (("B_drift_adaptive", "A2_periodic"), ("C_drift_confirmed", "B_drift_adaptive"), ("B_drift_adaptive", "A_no_detection")):
            d = np.array([bx[k] - by[k] for bx, by in zip(boot[x], boot[yv])])
            row.append(f"{np.nanmean(d):+.4f} [{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}]")
            bd.setdefault(k, {})[f"{x} - {yv}"] = [float(np.nanmean(d)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        a(f"| {k} | {row[0]} | {row[1]} | {row[2]} |")
    a("\n## Table 4b - per day: F1 / recall / false-positive rate\n\n| Day | Attack share | Static FPR>0.5 (shifted) | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|---:|---|" + "---|" * len(ARMS))
    for d in days:
        n = o["arms"][ARMS[0]]["per_day"][d]
        a(f"| {d} | {n['attacks']/n['n']:.3f} | {'yes' if shifted[d] else 'no'} | " + " | ".join(
            f"{f(o['arms'][x]['per_day'][d]['f1'], 3)} / {f(o['arms'][x]['per_day'][d]['recall'], 3)} / {f(o['arms'][x]['per_day'][d]['fpr'], 3)}" for x in ARMS) + " |")
    # ---- drift detection quality
    a("\n## Table 4c - drift detection (shared by arms B and C)\n\nTrue benign change points (evaluation only): " + "; ".join(f"{c['day']} start ({c['to']})" for c in change) +
      f". An alarm counts as a detection if it falls between the change point and the end of that day (at most {12*cfg['interval_records']:,} records later); alarms elsewhere are false alarms.\n")
    a("| True change point | Record index | First alarm (records after the change) | Delay in monitoring batches | Monitors that fired first | Alarms in the window |\n|---|---:|---:|---:|---|---:|")
    used = set()
    n_det, delays = 0, []
    for c in change:
        win = [(i, al) for i, al in enumerate(alarms) if c["record"] <= al["record"] < c["record"] + tol[c["day"]]]
        if win:
            i, al = win[0]
            used.update(j for j, _ in win)
            d_rec = al["record"] - c["record"]
            n_det += 1
            delays.append(d_rec)
            a(f"| {c['day']} ({c['to']}) | {c['record']:,} | {d_rec:,} | {d_rec/cfg['monitor_batch']:.0f} | {', '.join(al['monitors'])} | {len(win)} |")
        else:
            a(f"| {c['day']} ({c['to']}) | {c['record']:,} | missed | - | - | 0 |")
    false = [(i, al) for i, al in enumerate(alarms) if i not in used]
    a(f"\nDetected {n_det} of {len(change)} change points; median delay {np.median(delays) if delays else float('nan'):,.0f} records. "
      f"False alarms: {len(false)} of {len(alarms)} alarms ({1e6*len(false)/tot['n']:.2f} per million records); "
      f"{sum(1 for _, al in false if al['attack_share'] >= ATTACK_BATCH)} of them fall in batches that are at least {ATTACK_BATCH:.0%} attacks (attack onsets), the rest in benign-dominated batches.\n")
    a("| False alarm | Interval | Day | Record | Monitors | Attack share of the batch |\n|---|---:|---|---:|---|---:|")
    for n_, (i, al) in enumerate(false):
        a(f"| {n_+1} | {al['interval']} | {days_list[al['interval']]} | {al['record']:,} | {', '.join(al['monitors'])} | {al['attack_share']:.2f} |")
    # ---- performance around change points
    a("\n## Table 4d - performance around the true change points\n\nBefore = last 3 intervals of the previous day (not available for the first change); FPR recovery = intervals after the change until the trailing 3-interval FPR is at most "
      f"{RECOVER_FPR:.0%}; F1 recovery = intervals until the trailing 3-interval F1 is at least {F1_FRAC:.0%} of the before-change F1 (n/a if no before value or it is zero). 'never' = not within the day.\n")
    a("| Change point | Arm | FPR before | FPR first 3 intervals after | FPR recovery | F1 before | F1 lowest after | F1 recovery |\n|---|---|---:|---:|---:|---:|---:|---|")
    for c in change:
        d_iv = [i for i, d in enumerate(days_list) if d == c["day"]]
        for x in ARMS:
            fpr, rec, f1 = per_interval(iv[iv.arm == x])
            before = slice(max(0, c["interval"] - 3), c["interval"]) if c["interval"] > 0 else None
            fb = np.nanmean(fpr[before]) if before is not None and c["interval"] > 0 else np.nan
            f1b = np.nanmean(f1[before]) if before is not None and c["interval"] > 0 and not np.all(np.isnan(f1[before])) else np.nan
            after = d_iv
            fa = np.nanmean(fpr[after[:3]])
            r3 = roll3(np.nan_to_num(fpr[after], nan=0.0))
            ok = np.flatnonzero(r3 <= RECOVER_FPR)
            rec_fpr = "never" if len(ok) == 0 else str(int(ok[0]))
            f1a = pd.Series(f1[after]).rolling(3, min_periods=1).mean().to_numpy()
            lowest = np.nanmin(f1a) if not np.all(np.isnan(f1a)) else np.nan
            if f1b == f1b and f1b > 0:
                okf = np.flatnonzero(f1a >= F1_FRAC * f1b)
                rec_f1 = "never" if len(okf) == 0 else str(int(okf[0]))
            else:
                rec_f1 = "n/a"
            a(f"| {c['day']} ({c['to']}) | {SHORT[x]} | {f(fb, 3)} | {f(fa, 3)} | {rec_fpr} | {f(f1b, 3)} | {f(lowest, 3)} | {rec_f1} |")
    a("\n## Table 4e - ranking quality per day (ROC-AUC)\n\n| Day | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|" + "---:|" * len(ARMS))
    key = {"A_no_detection": "static"}
    for d in days:
        if d in o["auc_per_day"]["static"]:
            a(f"| {d} | " + " | ".join(f"{o['auc_per_day'][key.get(x, x)][d]['roc_auc']:.3f}" for x in ARMS) + " |")
    a("\n## Table 4f - share of each attack type flagged over the whole stream\n\n| Type | Records | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|---:|" + "---:|" * len(ARMS))
    for t in o["types"]:
        a(f"| {t} | {o['arms'][ARMS[0]]['per_type'][t]['n']:,} | " + " | ".join(f(o["arms"][x]["per_type"][t]["flagged_share"], 3) for x in ARMS) + " |")
    dl = pd.DataFrame(o["arms"]["C_drift_confirmed"]["drift_log"])
    if len(dl):
        a(f"\n## Table 4g - arm C: alarms confirmed or cancelled\n\nAlarms acted on or cancelled by the persistence check: {int((dl.event=='confirmed').sum())} confirmed, {int((dl.event=='cancelled').sum())} cancelled "
          f"({int((dl.event=='alarm').sum())} alarms reached the arm).")
    rec_out = {}
    for x in ARMS:
        rec_out[x] = {}
        for d in ("2018-02-16", "2018-02-21"):
            fpr_, _, _ = per_interval(iv[iv.arm == x])
            idx = [i for i, dd in enumerate(days_list) if dd == d]
            ok = np.flatnonzero(roll3(np.nan_to_num(fpr_[idx], nan=0.0)) <= RECOVER_FPR)
            rec_out[x][d] = int(ok[0]) if len(ok) else None
    json.dump({"change_points": change, "n_detected": n_det, "delays_records": delays, "alarms": len(alarms), "false_alarms": len(false),
               "false_alarms_in_attack_batches": sum(1 for _, al in false if al["attack_share"] >= ATTACK_BATCH),
               "false_alarms_per_million_records": 1e6 * len(false) / tot["n"], "recovery_intervals": rec_out,
               "mean_retrain_attack_share": {x: (float(pd.DataFrame(o["arms"][x]["retrain_log"]).trusted_attack_share.mean()) if o["arms"][x]["retrain_log"] else None) for x in ARMS},
               "retrainings": {x: len(o["arms"][x]["retrain_log"]) for x in ARMS}, "bootstrap_difference_mean_lo_hi": bd},
              open(R / "phase9_detection.json", "w"), indent=2, default=float)
    (Path("reports") / "phase9_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    series = {}
    for x in ARMS:
        fpr, rec, f1 = per_interval(iv[iv.arm == x])
        series[x] = (roll3(np.nan_to_num(fpr, nan=0.0)), pd.Series(rec).rolling(3, min_periods=1).mean().to_numpy())
    contam = {x: pd.DataFrame(o["arms"][x]["retrain_log"]).trusted_attack_share.to_numpy() for x in ARMS[1:]}
    auc = {x: {d: o["auc_per_day"][key.get(x, x)][d]["roc_auc"] for d in o["auc_per_day"]["static"]} for x in ARMS}
    plot_phase9(series, [(al["interval"], al["attack_share"] >= ATTACK_BATCH) for al in alarms], [c["interval"] for c in change], contam, auc, R / "phase9_drift_overview.png")
    print("\n".join(L))


if __name__ == "__main__":
    main()
