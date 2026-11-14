"""Tables and figure for Phase 17 from results_drift/phase17_drift_recovery.json.
Writes results_drift/drift_summary.md and results_drift/phase17_overview.png. Phase 7 arm C is quoted from results/phase7_main.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results_drift")
o = json.load(open(R / "phase17_drift_recovery.json"))
days_iv = np.array(json.load(open("results_threshold/phase16_adaptive_threshold.json"))["interval_days"])
ARMS = list(o["arms"])
NICE = {"S_frozen": "S frozen", "N_adaptive": "N adaptive, no supervised gate", "Gm_gated_main": "Gm gated by the Phase 4 supervised model",
        "Gu_gated_unseen": "Gu gated by the Phase 6 model that never saw Bot, HOIC, Hulk"}
f = lambda v, nd=4: "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"
L, a = [], None
a = L.append
a("# Phase 17 - concept-drift recovery with a supervised normality gate (generated from results_drift/phase17_drift_recovery.json)\n")
a("NumPy fine-tuning (src/nptrain.py) of the Phase 7 warm-up autoencoder; same stream, intervals and starting threshold as Phases 7 and 16. "
  f"Bulk-shifted days (post-hoc descriptive grouping, benign median score at least {o['cfg']['bulk_ratio']:g} x warm-up): {', '.join(o['bulk_shifted_days'])}. "
  f"Unseen types of the Phase 6 gate present in the stream: {', '.join(o['unseen_types_in_stream'])}; known: {', '.join(o['known_types_in_stream'])}.\n")
a("## Table 1 - overall at each arm's own threshold (all 8 days)\n")
a("| Arm | Precision | Recall | F1 | FPR | FNR | MCC | Accepted updates |\n|---|---:|---:|---:|---:|---:|---:|---:|")
for k in ARMS:
    g = o["arms"][k]["groups"]["all"]
    a(f"| {NICE[k]} | {f(g['precision'])} | {f(g['recall'])} | {f(g['f1'])} | {f(g['fpr'])} | {f(g['fnr'])} | {f(g['mcc'])} | {o['arms'][k]['accepted_updates']} of {o['arms'][k]['update_attempts']} |")
c = o["phase7_reference"]["C_adaptive_baseline"]
a(f"| Phase 7 arm C (PyTorch, quoted) | {f(c['precision'])} | {f(c['recall'])} | {f(c['f1'])} | {f(c['fpr'])} | {f(1 - c['recall'])} | - | 64 of 64 |")
a("\n## Table 2 - difference to the frozen model (moving-block bootstrap over intervals, 1,000 resamples, blocks of 8; mean [95% interval])\n")
a("| Arm - S | Precision | Recall | F1 | FPR | FNR |\n|---|---:|---:|---:|---:|---:|")
for k, d in o["bootstrap_difference_to_frozen_mean_lo_hi"].items():
    a(f"| {NICE[k]} | " + " | ".join(f"{d[m][0]:+.4f} [{d[m][1]:+.4f}, {d[m][2]:+.4f}]" for m in ("precision", "recall", "f1", "fpr", "fnr")) + " |")
a("\n## Table 3 - within-day ROC-AUC of the scores each arm produced (prequential). All = benign + all attacks of the day; known / unseen = benign + the attacks of that group\n")
a("| Day | Attack types | S all | N all | Gm all | Gu all | S unseen | N unseen | Gm unseen | Gu unseen |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for d in sorted(o["arms"]["S_frozen"]["per_day"]):
    pdv = {k: o["arms"][k]["per_day"][d] for k in ARMS}
    a(f"| {d} | {', '.join(pdv['S_frozen']['attack_types']) or '-'} | " + " | ".join(f(pdv[k]["roc_auc_all"], 3) for k in ARMS) + " | " + " | ".join(f(pdv[k]["roc_auc_benign_vs_unseen"], 3) for k in ARMS) + " |")
a("\n## Table 4 - bulk-shifted days against other days at each arm's threshold\n")
a("| Arm | Group | Precision | Recall | F1 | FPR | FNR |\n|---|---|---:|---:|---:|---:|---:|")
for k in ARMS:
    for grp, lab in (("bulk_shifted_days", "bulk-shifted days"), ("other_days", "other days")):
        g = o["arms"][k]["groups"][grp]
        a(f"| {NICE[k]} | {lab} | {f(g['precision'])} | {f(g['recall'])} | {f(g['f1'])} | {f(g['fpr'])} | {f(g['fnr'])} |")
a("\n## Table 5 - detection of known and unseen attacks and false alarms on shifted benign traffic (share of records flagged)\n")
a("| Arm | Known attacks | Unseen attacks | Benign on bulk-shifted days | Benign on other days |\n|---|---:|---:|---:|---:|")
for k in ARMS:
    r = o["arms"][k]
    a(f"| {NICE[k]} | {f(r['group_recall']['known_attacks']['flagged_share'])} | {f(r['group_recall']['unseen_attacks']['flagged_share'])} | {f(r['benign_fpr']['bulk_shifted_days'])} | {f(r['benign_fpr']['other_days'])} |")
n21 = sum(v is not None for v in (o["arms"]["S_frozen"]["interval_auc"][i] for i in np.flatnonzero(days_iv == "2018-02-21")))
a("\n## Table 6 - recovery time on the shifted days\n")
a(f"Rule (declared in advance): intervals from the start of the day until the interval-level ROC-AUC is at least {o['cfg']['recovery_auc']} for {o['cfg']['recovery_run']} consecutive intervals; an interval needs at least {o['cfg']['recovery_min_class']:,} benign and {o['cfg']['recovery_min_class']:,} attack records. "
  f"'none' means the rule was not met; on 02-21 only {n21} of 12 intervals meet the minimum-class condition, so the rule cannot be met there for any arm (not measurable).\n")
a("| Arm | 02-16 | 02-21 | 02-16 interval AUC, intervals 1 to 12 |\n|---|---:|---:|---|")
i16 = np.flatnonzero(days_iv == "2018-02-16")
for k in ARMS:
    rt = o["arms"][k]["recovery_intervals"]
    ser = " ".join("nan" if o["arms"][k]["interval_auc"][i] is None else f"{o['arms'][k]['interval_auc'][i]:.2f}" for i in i16)
    a(f"| {NICE[k]} | {rt.get('2018-02-16') or 'none'} | {rt.get('2018-02-21') or 'none (not measurable)'} | {ser} |")
a("\n## Table 7 - retraining data and the supervised gates (labels used, diagnostics only)\n")
a("| Arm | Mean admitted records per update | Attack share in retraining data, mean / max | Anchor-error ratio range |\n|---|---:|---:|---:|")
for k in ARMS[1:]:
    ups = [u for u in o["arms"][k]["updates"] if u.get("attack_share_in_buffer_diagnostic") is not None]
    sh = np.array([u["attack_share_in_buffer_diagnostic"] for u in ups]); adm = np.array([u["admitted_records"] for u in ups])
    rr = [u["anchor_error_ratio"] for u in ups]
    a(f"| {NICE[k]} | {adm.mean():,.0f} | {sh.mean():.3f} / {sh.max():.3f} | {min(rr):.2f} to {max(rr):.2f} |")
a("\n| Gate | Benign blocked | Attacks blocked, all records | Attacks blocked, test partition only (never seen) | Attacks blocked, training partition |\n|---|---:|---:|---:|---:|")
for g, nm in (("main", "Phase 4 supervised (all types seen)"), ("unseen", "Phase 6 supervised (Bot, HOIC, SSH-BruteForce, Hulk withheld)")):
    d = o["gate_diagnostics_use_labels"][g]
    a(f"| {nm} | {f(d['benign_blocked_share'])} | {f(d['attack_blocked_share_all_records'])} | {f(d['attack_blocked_share_test_partition_only'])} | {f(d['attack_blocked_share_train_partition_only'])} |")
types = sorted(o["gate_diagnostics_use_labels"]["main"]["attack_blocked_by_type"])
a("\n| Attack type | " + " | ".join(types) + " |\n|---|" + "---:|" * len(types))
for g in ("main", "unseen"):
    a(f"| blocked by {g} gate | " + " | ".join(f(o["gate_diagnostics_use_labels"][g]["attack_blocked_by_type"][t], 3) for t in types) + " |")
a("\n## Table 8 - success criteria declared in advance\n")
a("| Arm | (a) AUC >= 0.5 and above frozen on shifted days | (b) AUC within 0.05 of frozen on 02-20 and 03-02 | (c) overall F1 not lower than frozen | All three |\n|---|:---:|:---:|:---:|:---:|")
for k, v in o["criteria"].items():
    a(f"| {NICE[k]} | {'yes' if v['a_recovers_on_shifted_days'] else 'no'} | {'yes' if v['b_keeps_ranking_on_attack_days'] else 'no'} | {'yes' if v['c_overall_f1_not_lower'] else 'no'} | {'yes' if v['all_three'] else 'no'} |")
a("\n## Validation checks\n")
for c_ in o["checks"]:
    a(f"- {'PASS' if c_['ok'] else 'FAIL'} {c_['check']}: {c_['detail']}")
(R / "drift_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
col = {"S_frozen": "#7F7F7F", "N_adaptive": "#D55E00", "Gm_gated_main": "#0072B2", "Gu_gated_unseen": "#009E73"}
for ax, d, title in zip(axs.ravel(), ("2018-02-16", "2018-02-21", "2018-02-20", "2018-03-02"),
                        ("02-16: benign shift, Hulk (unseen to Gu)", "02-21: benign shift, HOIC (unseen to Gu)", "02-20: LOIC-HTTP campaign (known to both gates)", "03-02: Bot (unseen to Gu)")):
    idx = np.flatnonzero(days_iv == d)
    for k in ARMS:
        v = np.array([np.nan if o["arms"][k]["interval_auc"][i] is None else o["arms"][k]["interval_auc"][i] for i in idx])
        ax.plot(np.arange(1, len(idx) + 1), v, "-o" if len(idx) < 20 else "-", ms=3, color=col[k], lw=1.6, label=NICE[k].split(" by ")[0] if False else k)
    ax.axhline(0.5, color="#BBBBBB", lw=0.8, ls="--")
    ax.set_title(title, fontsize=9); ax.set_xlabel("interval of the day"); ax.grid(alpha=0.25)
axs[0, 0].set_ylabel("interval ROC-AUC"); axs[1, 0].set_ylabel("interval ROC-AUC")
axs[0, 0].legend(fontsize=8, loc="center right")
fig.suptitle("Phase 17: interval-level ROC-AUC of each arm's scores (gaps: fewer than 1,000 benign or attack records in the interval)", fontsize=10)
fig.tight_layout()
fig.savefig(R / "phase17_overview.png", dpi=140)
print("written", R / "drift_summary.md", R / "phase17_overview.png")
