"""Tables and figure for Phase 16 from results_threshold/phase16_adaptive_threshold.json (+ phase16_intervals.csv).
Writes results_threshold/threshold_summary.md and results_threshold/phase16_overview.png. Phase 7 B and C are quoted from
results/phase7_main.json (not re-run)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results_threshold")
o = json.load(open(R / "phase16_adaptive_threshold.json"))
iv = pd.read_csv(R / "phase16_intervals.csv")
p7 = o["phase7_reference"]
ARMS = ["S_static", "D_adaptive", "D1_no_rate_limit", "D2_median_mad_limited"]
NICE = {"S_static": "S static", "D_adaptive": "D adaptive threshold (lower quartile, rate-limited)", "D1_no_rate_limit": "D1 = D without rate limit",
        "D2_median_mad_limited": "D2 = Phase 7 B statistic + rate limit"}


def f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def fnr(m):
    return m["fn"] / (m["fn"] + m["tp"]) if (m["fn"] + m["tp"]) else float("nan")


L = []
a = L.append
a("# Phase 16 - adaptive threshold on a frozen autoencoder (generated from results_threshold/phase16_adaptive_threshold.json)\n")
a("Only the threshold changes; the model, the scores and therefore ROC-AUC / PR-AUC are identical in every arm. "
  "Precision, recall, F1, false-positive rate (FPR) and false-negative rate (FNR = 1 - recall), counted over all 8 evaluation days "
  "(9,504,036 records, 194 intervals of 50,000).\n")
a("## Table 1 - overall (all 8 days)\n")
a("| Arm | Precision | Recall | F1 | FPR | FNR | MCC |\n|---|---:|---:|---:|---:|---:|---:|")
for k in ARMS:
    g = o["arms"][k]["groups"]["all"]
    a(f"| {NICE[k]} | {f(g['precision'])} | {f(g['recall'])} | {f(g['f1'])} | {f(g['fpr'])} | {f(g['fnr'])} | {f(g['mcc'])} |")
for k, lab in (("B_adaptive_threshold", "Phase 7 B (all-record median + k MAD, no rate limit; quoted)"), ("C_adaptive_baseline", "Phase 7 C (adaptive baseline, retrained model; quoted)")):
    g = p7[k]["overall"]
    a(f"| {lab} | {f(g['precision'])} | {f(g['recall'])} | {f(g['f1'])} | {f(g['fpr'])} | {f(1 - g['recall'])} | - |")
a("\n## Table 2 - bootstrap difference to the static arm (moving-block bootstrap over intervals, 1,000 resamples, blocks of 8; mean [95% interval])\n")
a("| Arm - S | Precision | Recall | F1 | FPR | FNR |\n|---|---:|---:|---:|---:|---:|")
for k in ARMS[1:]:
    b = o["bootstrap_difference_to_static_mean_lo_hi"][k]
    a(f"| {NICE[k]} | " + " | ".join(f"{b[m][0]:+.4f} [{b[m][1]:+.4f}, {b[m][2]:+.4f}]" for m in ("precision", "recall", "f1", "fpr", "fnr")) + " |")
a("\n## Table 3 - per day (S static / D adaptive / D2). Frozen-score ROC-AUC is the same for all arms.\n")
a("| Day | Attacks | Benign median score vs warm-up (x) | ROC-AUC (frozen) | FPR S / D / D2 | Recall S / D / D2 | Precision S / D / D2 | Median threshold S / D / D2 |\n|---|---:|---:|---:|---|---|---|---|")
for d in sorted(o["arms"]["S_static"]["per_day"]):
    r = {k: o["arms"][k]["per_day"][d] for k in ("S_static", "D_adaptive", "D2_median_mad_limited")}
    auc = o["frozen_score_auc_per_day"].get(d, {}).get("roc_auc")
    a(f"| {d} | {r['S_static']['attacks']:,} | {o['posthoc']['benign_median_ratio_to_warmup'][d]:.0f} | {f(auc, 3)} | " +
      " / ".join(f(r[k]["fpr"]) for k in r) + " | " + " / ".join(f(r[k]["recall"]) for k in r) + " | " +
      " / ".join(f(r[k]["precision"], 3) for k in r) + " | " + " / ".join(f"{r[k]['median_threshold']:.3g}" for k in r) + " |")
a("\n## Table 4 - shifted periods\n")
a(f"Rule declared in advance (static FPR > 0.5 on the day): matches **{len(o['declared_shifted_day_rule_matches'])} days** of this stream, so there is no shifted period under it.\n")
a(f"Post-hoc, descriptive grouping (labels used, added after seeing that; benign median score >= 10 x warm-up median): bulk-shifted days {', '.join(o['posthoc']['bulk_shifted_days'])}; other days {', '.join(o['posthoc']['other_days'])}.\n")
a("| Arm | Group | Precision | Recall | F1 | FPR | FNR |\n|---|---|---:|---:|---:|---:|---:|")
for k in ARMS:
    for grp, lab in (("bulk_shifted", "bulk-shifted days"), ("other_days", "other days")):
        g = o["posthoc"]["arms"][k][grp]
        a(f"| {NICE[k]} | {lab} | {f(g['precision'])} | {f(g['recall'])} | {f(g['f1'])} | {f(g['fpr'])} | {f(g['fnr'])} |")
a("\n## Table 5 - threshold behaviour and contamination of the statistic\n")
orc = np.array(o["diagnostics_use_labels"]["oracle_benign_p99_per_interval"], float)
days = np.array(o["interval_days"])
a("| Arm | Intervals with no alarm at all | Median |log10(threshold / oracle benign p99)| (decades) | Largest threshold | Smallest threshold |\n|---|---:|---:|---:|---:|")
for k in ARMS:
    s = iv[iv.arm == k]
    thr = s.threshold.to_numpy()
    zero = int(((s.tp + s.fp) == 0).sum())
    err = np.nanmedian(np.abs(np.log10(thr) - np.log10(orc)))
    a(f"| {NICE[k]} | {zero} of {len(s)} | {err:.2f} | {thr.max():.3g} | {thr.min():.3g} |")
a(f"\nOracle benign p99 = 99th percentile of the benign scores of the interval (uses labels, diagnostic only). "
  f"Attack share among the records scoring below the interval median (the part that a lower-quartile statistic is built from): mean {o['diagnostics_use_labels']['mean_attack_share_lower_half']:.3f}, "
  f"maximum {o['diagnostics_use_labels']['max_attack_share_lower_half']:.3f}.\n")
a("## Reproduction and structure checks\n")
for c in o["checks"]:
    a(f"- {'PASS' if c['ok'] else 'FAIL'} {c['check']}: {c['detail']}")
(R / "threshold_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(12, 5.2))
x = np.arange(len(days))
col = {"S_static": "#7F7F7F", "D_adaptive": "#0072B2", "D2_median_mad_limited": "#D55E00"}
for k in ("S_static", "D_adaptive", "D2_median_mad_limited"):
    ax.step(x, iv[iv.arm == k].threshold.to_numpy(), where="post", color=col[k], lw=2.2 if k == "D_adaptive" else 1.4, label=NICE[k].split(" (")[0])
ax.plot(x, orc, ".", color="black", ms=3, label="oracle: benign p99 of the interval (uses labels, diagnostic only)")
ax.set_yscale("log"); ax.set_ylim(1e-3, 60)
ax.set_ylabel("alarm threshold (reconstruction error)"); ax.set_xlabel("interval of 50,000 records, in stream order")
ax.legend(loc="upper right", fontsize=8, ncol=2)
edges = [0] + [i for i in range(1, len(days)) if days[i] != days[i - 1]] + [len(days)]
for a_, b_ in zip(edges[:-1], edges[1:]):
    ax.axvline(a_, color="#BBBBBB", lw=0.7)
    ax.text((a_ + b_) / 2, 1.3e-3, days[a_][5:], ha="center", fontsize=8, color="#444444", rotation=(90 if b_ - a_ < 10 else 0))
ax.grid(alpha=0.25)
ax.set_title("Phase 16: frozen autoencoder, static vs adaptive threshold (only the threshold changes; days labelled at the bottom)", fontsize=10)
fig.tight_layout()
fig.savefig(R / "phase16_overview.png", dpi=140)
print("written", R / "threshold_summary.md", R / "phase16_overview.png")
