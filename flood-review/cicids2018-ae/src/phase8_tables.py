"""Phase 8 tables (Table 3) and figure from results/phase8_*.json and the per-interval counts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plots import plot_phase8

R = Path("results")
ARMS = ["S_static", "U_unrestricted", "M_margin_only", "G_gated"]
SHORT = {"S_static": "S static", "U_unrestricted": "U unrestricted", "M_margin_only": "M margin only", "G_gated": "G confidence-gated"}
TARGET_FPR, RECOVER_FPR = 0.05, 0.10


def f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def rates(c):
    tp, fp, fn, tn = c
    div = lambda a, b: a / b if b else np.nan
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if tp else 0.0, "fpr": div(fp, fp + tn)}


def roll3(v):
    v = pd.Series(v)
    return v.rolling(3, min_periods=1).mean().to_numpy()


def recovery(iv_arm, day):
    """Intervals from the start of `day` until the trailing 3-interval mean FPR is <= RECOVER_FPR; None if never within the day."""
    d = iv_arm[iv_arm.day == day]
    fpr = (d.fp / (d.fp + d.tn).replace(0, np.nan)).to_numpy()
    r = roll3(np.nan_to_num(fpr, nan=0.0))
    ok = np.flatnonzero(r <= RECOVER_FPR)
    return int(ok[0]) if len(ok) else None


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
    o = json.load(open(R / "phase8_main.json"))
    iv = pd.read_csv(R / "phase8_main_intervals.csv")
    days = sorted(o["arms"][ARMS[0]]["per_day"])
    iv_days = o["intervals"]["days"]
    cfg = o["config"]
    tot = o["arms"][ARMS[0]]["overall"]
    L = []
    a = L.append
    a(f"Same warm-up model and evaluation stream as Phase 7 ({tot['n']:,} records, {tot['attacks']:,} attacks, 8 days). Every arm's threshold is the {cfg['anchor_q']:.2f}-quantile of the errors "
      f"of a fixed clean anchor set under the arm's current model (recomputed after each update). Update interval {cfg['interval_records']:,} records; retraining every {cfg['retrain_every']} intervals; "
      f"gate: margin {cfg['margin']}, level tolerance {cfg['level_tol']} decades, flagged-share limit {cfg['susp_max']}, persistence {cfg['persist']} intervals. One run, seed {cfg['seed']}. "
      "Reconstruction error is treated as an ordinal anomaly score, not a calibrated probability.\n")
    a("## Table 3a - overall\n\n| Metric | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|" + "---:|" * len(ARMS))
    for k, lab in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("fpr", "False-positive rate"), ("flagged_share", "Share of records flagged")):
        a(f"| {lab} | " + " | ".join(f(o["arms"][x]["overall"][k]) for x in ARMS) + " |")
    a("\n**Read with care:** on 02-16 and 02-21 the benign traffic is shifted entirely above the static threshold, so arms S and M flag 90-94% of all records there. "
      "That 'flag everything' behaviour inflates their overall recall and F1 (flagging everything gives F1 = 2p/(1+p) for attack share p) while their false-positive rate is 99.6% and 99.9% on those days. Judge by the per-day table and the false-positive rate.\n")
    boot = block_boot(iv, ARMS)
    bd = {}
    a("Moving-block bootstrap over intervals (1,000 resamples, blocks of 8), 95% interval of the difference, G minus U and M minus U:\n\n| Metric | G - U | M - U |\n|---|---|---|")
    for k in ("f1", "recall", "fpr"):
        row = []
        for x in ("G_gated", "M_margin_only"):
            d = np.array([bx[k] - bu[k] for bx, bu in zip(boot[x], boot["U_unrestricted"])])
            row.append(f"{np.nanmean(d):+.4f} [{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}]")
            bd.setdefault(k, {})[f"{x} - U_unrestricted"] = [float(np.nanmean(d)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        a(f"| {k} | {row[0]} | {row[1]} |")
    a("\n## Table 3b - per day: F1 / recall / false-positive rate\n\n| Day | Attack share | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|---:|" + "---|" * len(ARMS))
    for d in days:
        n = o["arms"][ARMS[0]]["per_day"][d]
        a(f"| {d} | {n['attacks']/n['n']:.3f} | " + " | ".join(
            f"{f(o['arms'][x]['per_day'][d]['f1'], 3)} / {f(o['arms'][x]['per_day'][d]['recall'], 3)} / {f(o['arms'][x]['per_day'][d]['fpr'], 3)}" for x in ARMS) + " |")
    a("\n(02-22 and 02-23 have 319 and 535 attacks, so recall/F1 there are not meaningful; their false-positive rates are.)\n")
    a("## Table 3c - ranking quality per day (ROC-AUC; S and M differ only slightly because M also retrains)\n\n| Day | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|" + "---:|" * len(ARMS))
    key = {"S_static": "static"}
    for d in days:
        if d in o["auc_per_day"]["static"]:
            a(f"| {d} | " + " | ".join(f"{o['auc_per_day'][key.get(x, x)][d]['roc_auc']:.3f}" for x in ARMS) + " |")
    a("\n## Table 3d - contamination of the retraining buffer (share of attack records among the records used for retraining; diagnostic from labels, never used for control)\n\n| Arm | Retrainings (accepted) | Mean | Maximum | " + " | ".join(d[5:] for d in days) + " |\n|---|---:|---:|---:|" + "---:|" * len(days))
    contam = {}
    for x in ARMS[1:]:
        rl = pd.DataFrame(o["arms"][x]["retrain_log"])
        rl["day"] = [iv_days[i] for i in rl.interval]
        contam[x] = rl.trusted_attack_share.to_numpy()
        by = rl.groupby("day").trusted_attack_share.mean()
        a(f"| {SHORT[x]} | {len(rl)} ({int(rl.accepted.sum())}) | {rl.trusted_attack_share.mean():.3f} | {rl.trusted_attack_share.max():.3f} | " + " | ".join(f(by.get(d), 3) for d in days) + " |")
    a(f"\nAttack share of the whole stream: {tot['attacks']/tot['n']:.3f}.\n")
    a("## Table 3e - recovery after a change of traffic\n\n"
      f"Intervals (of {cfg['interval_records']:,} records) from the start of a day until the trailing 3-interval mean false-positive rate is at most {RECOVER_FPR:.0%} (twice the {TARGET_FPR:.0%} target); "
      "'never' = not reached before the day ends; 0 = no degradation at the start of the day. Days 02-16 and 02-21 are the two with a large shift of benign traffic.\n")
    a("| Day (intervals in day) | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|" + "---:|" * len(ARMS))
    for d in days:
        n_iv = int((iv[(iv.arm == ARMS[0]) & (iv.day == d)]).shape[0])
        cells = []
        for x in ARMS:
            r = recovery(iv[iv.arm == x], d)
            cells.append("never" if r is None else str(r))
        a(f"| {d} ({n_iv}) | " + " | ".join(cells) + " |")
    a("\n## Table 3f - share of each attack type flagged over the whole stream\n\n| Type | Records | " + " | ".join(SHORT[x] for x in ARMS) + " |\n|---|---:|" + "---:|" * len(ARMS))
    for t in o["types"]:
        a(f"| {t} | {o['arms'][ARMS[0]]['per_type'][t]['n']:,} | " + " | ".join(f(o["arms"][x]["per_type"][t]["flagged_share"], 3) for x in ARMS) + " |")
    g = pd.DataFrame(o["arms"]["G_gated"]["gate_log"])
    g["day"] = [iv_days[i] for i in g.interval]
    gd = g.groupby("day").agg(deviant=("deviant", "sum"), released=("released", "sum"), n=("interval", "size"))
    a(f"\n## Table 3g - gate behaviour (arm G)\n\nIntervals judged inconsistent: {int(g.deviant.sum())} of {len(g)}; releases of held intervals into the buffer (accepted as a lasting change): {int(g.released.sum())}.\n\n| Day | Intervals | Inconsistent | Releases |\n|---|---:|---:|---:|")
    for d, r in gd.iterrows():
        a(f"| {d} | {int(r.n)} | {int(r.deviant)} | {int(r.released)} |")
    sens = []
    for tag in ("main", "persist6", "persist12"):
        p = R / f"phase8_{tag}.json"
        if p.exists():
            s = json.load(open(p))
            ivs = pd.read_csv(R / f"phase8_{tag}_intervals.csv")
            sens.append((s["config"]["persist"], s, ivs))
    if len(sens) > 1:
        a("\n## Table 3h - sensitivity of the gated arm G to the persistence setting (others unchanged)\n\n"
          "| Persistence (intervals) | Overall F1 | Recall | FPR | Mean buffer attack share | 02-16 FPR / recovery | 02-21 FPR / recovery | 02-20 recall | LOIC-HTTP flagged | HOIC flagged | Bot flagged |\n|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|")
        for pz, s, ivs in sorted(sens, key=lambda z: z[0]):
            ga = s["arms"]["G_gated"]
            rl = pd.DataFrame(ga["retrain_log"])
            rec = lambda d: recovery(ivs[ivs.arm == "G_gated"], d)
            rc = lambda d: "never" if rec(d) is None else str(rec(d))
            a(f"| {pz} | {ga['overall']['f1']:.4f} | {ga['overall']['recall']:.4f} | {ga['overall']['fpr']:.4f} | {rl.trusted_attack_share.mean():.3f} | "
              f"{ga['per_day']['2018-02-16']['fpr']:.3f} / {rc('2018-02-16')} | {ga['per_day']['2018-02-21']['fpr']:.3f} / {rc('2018-02-21')} | {ga['per_day']['2018-02-20']['recall']:.3f} | "
              f"{ga['per_type']['DDoS-LOIC-HTTP']['flagged_share']:.3f} | {ga['per_type']['DDoS-HOIC']['flagged_share']:.3f} | {ga['per_type']['Bot']['flagged_share']:.3f} |")
    (Path("reports") / "phase8_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    series = {}
    for x in ARMS:
        d = iv[iv.arm == x]
        series[x] = (roll3(np.nan_to_num((d.fp / (d.fp + d.tn).replace(0, np.nan)).to_numpy(), nan=0.0)),
                     pd.Series((d.tp / (d.tp + d.fn).replace(0, np.nan)).to_numpy()).rolling(3, min_periods=1).mean().to_numpy())
    marks = [i for i in range(1, len(iv_days)) if iv_days[i] != iv_days[i - 1]]
    auc = {x: {d: o["auc_per_day"][{"S_static": "static"}.get(x, x)][d]["roc_auc"] for d in o["auc_per_day"]["static"]} for x in ARMS}
    plot_phase8(series, contam, auc, marks, R / "phase8_confidence_overview.png")
    sens_out = {}
    for pz, s, ivs in sens:
        ga = s["arms"]["G_gated"]
        sens_out[str(pz)] = {"overall": ga["overall"], "mean_buffer_attack_share": float(pd.DataFrame(ga["retrain_log"]).trusted_attack_share.mean()),
                             "fpr_0216": ga["per_day"]["2018-02-16"]["fpr"], "fpr_0221": ga["per_day"]["2018-02-21"]["fpr"],
                             "recovery_0216": recovery(ivs[ivs.arm == "G_gated"], "2018-02-16"), "recovery_0221": recovery(ivs[ivs.arm == "G_gated"], "2018-02-21"),
                             "loic_flagged": ga["per_type"]["DDoS-LOIC-HTTP"]["flagged_share"]}
    rec_out = {x: {d: recovery(iv[iv.arm == x], d) for d in ("2018-02-16", "2018-02-21")} for x in ARMS}
    json.dump({"bootstrap_difference_mean_lo_hi": bd, "recovery_intervals": rec_out, "persistence_sensitivity_G": sens_out,
               "mean_buffer_attack_share": {x: float(c.mean()) for x, c in contam.items()}, "max_buffer_attack_share": {x: float(c.max()) for x, c in contam.items()}},
              open(R / "phase8_summary.json", "w"), indent=2, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
