"""Phase 10: final report. Builds reports/FINAL_REPORT.md; every number in the tables is read from the saved result files
(results/*.json, reports/phase1_inspection.json, data/processed/phase2_summary.json), none is typed by hand.
Run after verify_results.py (which cross-checks the reported numbers).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_split

R, P, REP = Path("results"), Path("data/processed"), Path("reports")
J = lambda p: json.load(open(p))


def f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def ci(v, nd=3):
    return f"{v[0]:+.{nd}f} [{v[1]:+.{nd}f}, {v[2]:+.{nd}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    p1, p2 = J(REP / "phase1_inspection.json"), J(P / "phase2_summary.json")
    p3, p4, p5, p6 = (J(R / n) for n in ("phase3_results.json", "phase4_results.json", "phase5_comparison.json", "phase6_results.json"))
    p7, p7b = J(R / "phase7_main.json"), J(R / "phase7_bootstrap.json")
    p8, p8s = J(R / "phase8_main.json"), J(R / "phase8_summary.json")
    p9, p9d = J(R / "phase9_main.json"), J(R / "phase9_detection.json")
    M5 = p5["models"]
    L = []
    a = L.append
    tot = p1["total_rows"]
    cl = p2["split"]
    parts = p2["scale"]["partitions"]
    a("# Comparative evaluation of supervised and unsupervised autoencoders for network anomaly detection on CSE-CIC-IDS2018\n")
    a("**Final results report.** All experiments below were executed; every number in the tables is read from the saved result files and was re-derived independently by `src/verify_results.py` "
      "(`reports/verification.md`). All models were trained once (seed 42), so differences between close results should not be over-read.\n")

    # ------------------------------------------------------------------ 1. summary
    a("## 1. Summary: which approach performs better?\n")
    us, sp, oc, sr = M5["unsup_mixed"], M5["supervised"], M5["normal_only"], M5["supervised_recon"]
    u6, o6, s6 = (p6["models"][m] for m in ("unsup_mixed", "normal_only", "supervised"))
    a(f"The answer depends on what is asked of the detector, and no single number settles it (accuracy is 0.82-0.98 for every model because {100*(1-p5['attack_share']):.0f}% of the test records are benign).\n")
    a("1. **Attack types seen in training: the supervised autoencoder is clearly better.** On the identical test partition (%s records) its ROC-AUC is %.3f (supervised + decoder %.3f), against %.3f for the autoencoder trained on benign data only and %.3f for the strictly unsupervised one; F1 at the pre-fixed thresholds is %.3f vs %.3f (one-class: %.3f)." % (
        f"{p5['test_rows']:,}", sp["ranking"]["roc_auc"], sr["ranking"]["roc_auc"], oc["ranking"]["roc_auc"], us["ranking"]["roc_auc"], sp["primary"]["f1"], us["primary"]["f1"], oc["primary"]["f1"]))
    a("2. **Attack types never seen in training: the picture reverses.** With four attack types withheld from every fitted or tuned step, the supervised model flags %.1f%% of them (ROC-AUC %.3f) while the unsupervised autoencoder flags %.1f%% (ROC-AUC %.3f), at a benign false-positive rate of %.1f%%. The supervised model keeps recall %.2f on known attacks; the unsupervised one %.2f." % (
        100 * s6["unseen"]["primary"]["recall"], s6["unseen_ranking"]["roc_auc"], 100 * u6["unseen"]["primary"]["recall"], u6["unseen_ranking"]["roc_auc"],
        100 * u6["known"]["primary"]["fpr"], s6["known"]["primary"]["recall"], u6["known"]["primary"]["recall"]))
    a("3. **Why the unsupervised model fails on known attacks:** it reconstructs the attack traffic it is trained on. The same attack types that it detects poorly when present in its training data (ROC-AUC against benign 0.42-0.68) are detected well when withheld (0.91-0.95).")
    a("4. **Reconstruction error is a weak anomaly score on its own:** even with benign-only training the error ranks attacks well (ROC-AUC %.3f) but recall at 1%% false positives is only %.3f, because a few percent of ordinary benign flows score higher than most attacks." % (
        oc["ranking"]["roc_auc"], oc["tpr_at_fpr_1pct"]))
    a("5. **Adapting the baseline did not reliably help.** Static, adaptive-threshold and adaptive-baseline detectors were compared on a %s-record stream (Tables 2-4): the adaptive threshold rule was unstable, retraining helped where benign traffic shifted (2 of 8 days) and hurt where large attacks dominated, and a trusted buffer filtered on the model's own score did not keep attacks out (%.1f%% attacks in the buffer against %.1f%% in the stream)." % (
        f"{p7['arms']['A_static']['overall']['n']:,}", 100 * p8s["mean_buffer_attack_share"]["U_unrestricted"], 100 * p8["arms"]["S_static"]["overall"]["attacks"] / p8["arms"]["S_static"]["overall"]["n"]))
    a("6. **Confidence-aware updates (margin, temporal consistency, quarantine) gave no clear advantage over unrestricted adaptation;** a margin-only filter protected the buffer but could not adapt to a real shift, and a persistence rule cannot tell a lasting benign change from a long attack campaign.")
    a("7. **Drift monitoring (ADWIN) detected all %d of %d benign change points** (median delay %s records) with %d false alarms among %d alarms, and drift-triggered retraining beat scheduled retraining at the same false-positive rate in this run, which the evidence attributes mostly to timing (it did not retrain during the Bot campaign), not to detecting drift on that day." % (
        p9d["n_detected"], len(p9d["change_points"]), f"{int(np.median(p9d['delays_records'])):,}", p9d["false_alarms"], p9d["alarms"]))
    a("8. **A hybrid is suggested, not tested:** a supervised classifier for known attacks plus a novelty detector trained only on verified-clean data would combine the two strengths seen above. No experiment here evaluated that combination.\n")

    # ------------------------------------------------------------------ 2. dataset
    a("## 2. Dataset inspection (CSE-CIC-IDS2018, processed CICFlowMeter CSVs)\n")
    a(f"Source: official bucket listed on https://www.unb.ca/cic/datasets/ids-2018.html (10 daily CSVs, 6.41 GB, all verified by byte size). Full report: `reports/phase1_inspection.md`, `reports/phase1_findings.md`.\n")
    a(f"- **Records:** {tot:,} (after dropping {p1['embedded_header_rows_dropped']} header rows repeated inside three files). **Features:** 80 columns in 9 files, 84 in one (adds Flow ID, Src/Dst IP, Src Port); 78 feature columns are common to all files. **Benign:** {p1['benign_rows']:,} ({100*p1['benign_rows']/tot:.2f}%); **attacks:** {tot-p1['benign_rows']:,} in 14 categories.")
    a(f"- **Missing / infinite values:** {', '.join(f'`{k}` ({v['nan']:,} NaN, {v['inf']:,} infinite)' for k, v in p1['columns_with_nan_inf_nonnumeric'].items())}. **Constant columns:** {len(p1['constant_columns'])}. **Exact duplicate rows:** {p1['duplicates']['exact_row_duplicates']['duplicates']:,}; **duplicate feature vectors:** {p1['duplicates']['feature_vector_duplicates']['duplicates']:,}.")
    a("- **Potential leakage / data-quality issues found:** (i) **7 of the 10 files contain exactly 1,048,575 data lines (one of them 1,048,574 records plus a repeated header line), the spreadsheet row limit minus the header, which strongly suggests truncation** (what is missing cannot be seen); (ii) **timestamps are on a 12-hour clock without AM/PM** (found in Phase 7: afternoon events appear 12 hours early; verified against the official attack schedule, e.g. DoS-Hulk scheduled 13:45 is stored as 01:45), so time-of-day cannot be used directly and the corrected time was used only for the streaming experiments; (iii) rows within a file are not in time order; (iv) 14 rows carry January-1970 timestamps; (v) identifiers (Flow ID, IPs, Src Port) exist only in the 20-Feb file and were excluded; (vi) `Dst Port` alone almost identifies several attack types (e.g. 100% of FTP-BruteForce on port 21, 0.00% of benign) and was excluded from the model input.\n")
    labs = p1["label_counts"]
    a(table(["Label", "Records", "Share"], [[k, f"{v:,}", f"{100*v/tot:.3f}%"] for k, v in labs.items()]))
    a("")

    # ------------------------------------------------------------------ 3. preprocessing
    a("## 3. Preprocessing and split\n")
    drops = {}
    for fl in p2["clean"]:
        for k, v in fl["drops"].items():
            drops[k.split("|")[0]] = drops.get(k.split("|")[0], 0) + v
    for k, v in cl["dedup_drops"].items():
        drops[k.split("|")[0]] = drops.get(k.split("|")[0], 0) + v
    a(f"- **Cleaning and de-duplication:** {sum(fl['rows_in'] for fl in p2['clean']):,} records read -> {cl['rows_after_clean']:,} after cleaning (removed: " + ", ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in drops.items() if k not in ('duplicate_vector', 'label_conflict_vector')) +
      f") -> **{cl['rows_after_dedup']:,}** after removing identical feature vectors ({drops.get('duplicate_vector', 0):,}) and vectors seen with both a benign and an attack label ({drops.get('label_conflict_vector', 0):,}). De-duplication shrinks some attack types to a few dozen unique flows (FTP-BruteForce 193,360 -> 41; DoS-SlowHTTPTest 139,890 -> 55).")
    a("- **Model input:** 72 columns: 68 numeric features after a signed log1p transform and min-max scaling to [0, 1] fitted on the training partition only, plus a one-hot protocol (4 columns: 0, 6, 17, other); constant columns (8), `Dst Port`, timestamp, label and identifiers are excluded. Original labels are kept separately; the binary label is 0 = benign, 1 = attack. No clipping of validation/test values.")
    a("- **Split (scenario-aware by time block):** every capture day is cut into 60-second blocks on the stored clock; whole blocks go to train / validation / test (70/10/20% of blocks) stratified by the attack types a block contains, so no block straddles partitions and the partitions share no identical feature vector. The in-distribution test is therefore optimistic (neighbouring blocks of one attack are alike).\n")
    a(table(["Partition", "Records", "Benign", "Attacks", "Attack share"], [[s, f"{parts[s]['rows']:,}", f"{parts[s]['benign']:,}", f"{parts[s]['attack']:,}", f"{100*parts[s]['attack']/parts[s]['rows']:.2f}%"] for s in ("train", "val", "test")]))
    a("")

    # ------------------------------------------------------------------ 4. tables
    a("## 4. Results\n")
    a("### Table 1 - Supervised vs unsupervised autoencoder (test partition, identical for all models; thresholds fixed beforehand)\n")
    a(table(["Metric", "Unsupervised AE", "Supervised AE"], [
        ["Accuracy", f(us["primary"]["accuracy"]), f(sp["primary"]["accuracy"])], ["Precision", f(us["primary"]["precision"]), f(sp["primary"]["precision"])],
        ["Recall", f(us["primary"]["recall"]), f(sp["primary"]["recall"])], ["F1-score", f(us["primary"]["f1"]), f(sp["primary"]["f1"])],
        ["Specificity", f(us["primary"]["specificity"]), f(sp["primary"]["specificity"])], ["False-positive rate", f(us["primary"]["fpr"]), f(sp["primary"]["fpr"])],
        ["False-negative rate", f(us["primary"]["fnr"]), f(sp["primary"]["fnr"])], ["ROC-AUC", f(us["ranking"]["roc_auc"]), f(sp["ranking"]["roc_auc"])],
        ["PR-AUC (chance %.4f)" % p5["attack_share"], f(us["ranking"]["pr_auc"]), f(sp["ranking"]["pr_auc"])],
        ["Epochs / parameters", f"{us['cost']['epochs_run']} / {us['cost']['parameters']:,}", f"{sp['cost']['epochs_run']} / {sp['cost']['parameters']:,}"]]))
    a("\nUnsupervised AE: trained on unlabeled records, threshold = 95th percentile of validation errors (no labels). Supervised AE: class-weighted binary cross-entropy, threshold 0.5. "
      "Training and inference times were measured but are single runs on a shared CPU and vary about 3x between identical runs, so they are not used to rank models (inference was 0.7-1.9 ms per 1000 records for all four; all models are small, 20,225-39,945 parameters). Peak memory could not be separated per model.\n")
    a("**All four trained models** (same test set; thresholds fixed before seeing results; 95% intervals are block bootstraps over test blocks and do not include training randomness):\n")
    names = {"unsup_mixed": "Unsupervised (unlabeled train)", "normal_only": "One-class (benign-only train)", "supervised": "Supervised", "supervised_recon": "Supervised + decoder"}
    rows = []
    for k, lab in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("fpr", "False-positive rate")):
        rows.append([lab] + [f(M5[m]["primary"][k]) for m in names])
    rows.append(["ROC-AUC [95% CI]"] + [f"{M5[m]['ranking']['roc_auc']:.3f} [{M5[m]['ci_ranking']['roc_auc'][0]:.3f}, {M5[m]['ci_ranking']['roc_auc'][1]:.3f}]" for m in names])
    rows.append(["PR-AUC [95% CI]"] + [f"{M5[m]['ranking']['pr_auc']:.3f} [{M5[m]['ci_ranking']['pr_auc'][0]:.3f}, {M5[m]['ci_ranking']['pr_auc'][1]:.3f}]" for m in names])
    rows.append(["Recall at 1% FPR [95% CI]"] + [f"{M5[m]['tpr_at_fpr_1pct']:.3f} [{M5[m]['ci_ranking']['tpr1'][0]:.3f}, {M5[m]['ci_ranking']['tpr1'][1]:.3f}]" for m in names])
    rows.append(["F1 with validation-tuned threshold"] + [("degenerate (flags 99.99%)" if m == "unsup_mixed" else f(M5[m]["val_tuned"]["f1"])) for m in names])
    a(table(["Metric"] + list(names.values()), rows))
    pd5 = p5["paired_diff"]
    a("\nPaired differences (same resamples): supervised - one-class ROC-AUC " + ci([pd5['supervised - normal_only | roc_auc']['mean']] + pd5['supervised - normal_only | roc_auc']['ci95']) +
      "; supervised + decoder - supervised ROC-AUC " + ci([pd5['supervised_recon - supervised | roc_auc']['mean']] + pd5['supervised_recon - supervised | roc_auc']['ci95']) + " (not distinguishable); one-class - unsupervised ROC-AUC " +
      ci([pd5['normal_only - unsup_mixed | roc_auc']['mean']] + pd5['normal_only - unsup_mixed | roc_auc']['ci95']) + ".\n")
    a("Figures: `results/phase5_roc_pr_comparison.png`, `phase5_confusion_unsup_vs_sup.png`, `phase5_confusion_all_models.png`, `phase5_loss_curves_all.png`. Details: `reports/phase3_findings.md`, `phase4_findings.md`, `phase5_findings.md`, `phase5_tables.md`.\n")

    # Table 2
    a("### Table 2 - Adaptive experiments: static vs adaptive threshold vs adaptive baseline (Experiment D, Phase 7)\n")
    s7 = p7["arms"]
    arms7 = ["A_static", "B_adaptive_threshold", "C_adaptive_baseline"]
    a(f"Time-ordered stream of {s7['A_static']['overall']['n']:,} records ({s7['A_static']['overall']['attacks']:,} attacks, 8 days) after a warm-up on the first two days; update interval 50,000 records; threshold = 99th percentile of warm-up benign errors; "
      "B re-estimates the threshold each interval with a robust median + k x MAD rule; C adds fine-tuning on a rolling trusted buffer (records it scored as normal) with replay and an anchor check. Attack labels never controlled an update.\n")
    tt = {x: np.array(s7[x]["threshold_trace"]) for x in arms7}
    a(table(["Metric", "A static", "B adaptive threshold", "C adaptive baseline"], [
        ["Precision"] + [f(s7[x]["overall"]["precision"]) for x in arms7], ["Recall"] + [f(s7[x]["overall"]["recall"]) for x in arms7],
        ["F1-score"] + [f(s7[x]["overall"]["f1"]) for x in arms7], ["False-positive rate"] + [f(s7[x]["overall"]["fpr"]) for x in arms7],
        ["Threshold minimum / maximum"] + [f"{tt[x].min():.3g} / {tt[x].max():.3g}" for x in arms7],
        ["Intervals with threshold above 0.072 (above any error seen in Phase 3)"] + [f"{100*(tt[x] > 0.072).mean():.1f}%" for x in arms7],
        ["Retrainings (accepted)"] + ["0", "0"] + [f"{len(s7['C_adaptive_baseline']['retrain_log'])} ({sum(r['accepted'] for r in s7['C_adaptive_baseline']['retrain_log'])})"],
        ["Difference to A in F1 [95% CI]", "-", ci(p7b["bootstrap_difference_mean_lo_hi"]["f1"]["B_adaptive_threshold - A_static"], 4), ci(p7b["bootstrap_difference_mean_lo_hi"]["f1"]["C_adaptive_baseline - A_static"], 4)]]))
    rl7 = pd.DataFrame(s7["C_adaptive_baseline"]["retrain_log"])
    a(f"\nResult: adaptation did not improve detection. The static detector is already weak (the threshold sits at the benign tail); the adaptive-threshold rule is unstable (maximum {tt['B_adaptive_threshold'].max():.3g}); "
      f"retraining raised per-day ranking quality on the two days whose benign traffic shifted (02-16 ROC-AUC {p7['auc_per_day']['static']['2018-02-16']['roc_auc']:.3f} -> {p7['auc_per_day']['C_adaptive_baseline']['2018-02-16']['roc_auc']:.3f}; 02-21 {p7['auc_per_day']['static']['2018-02-21']['roc_auc']:.3f} -> {p7['auc_per_day']['C_adaptive_baseline']['2018-02-21']['roc_auc']:.3f}) "
      f"and lowered it on days dominated by large attacks (02-20 {p7['auc_per_day']['static']['2018-02-20']['roc_auc']:.3f} -> {p7['auc_per_day']['C_adaptive_baseline']['2018-02-20']['roc_auc']:.3f}); the trusted buffer held {100*rl7.trusted_attack_share.mean():.1f}% attacks on average (stream {100*s7['A_static']['overall']['attacks']/s7['A_static']['overall']['n']:.1f}%). Figure: `results/phase7_adaptive_overview.png`; details `reports/phase7_findings.md`.\n")

    # Table 3
    a("### Table 3 - Confidence-aware experiments: unrestricted vs confidence-gated adaptation (Experiment E, Phase 8)\n")
    s8 = p8["arms"]
    arms8 = ["S_static", "U_unrestricted", "M_margin_only", "G_gated"]
    a("Same stream and warm-up; each arm's threshold = 95th percentile of a fixed clean anchor set under the arm's current model (replaces Phase 7's unstable rule). U: every sampled record enters the retraining buffer. G: margin (score <= 0.5 x threshold) + quarantine of records above the threshold + temporal consistency "
      "(a deviating interval is held and released only if the deviation persists for 3 intervals). M: margin only (ablation). S: static reference. Reconstruction error is treated as an ordinal score, not a calibrated probability.\n")
    sr8 = p8s["recovery_intervals"]
    a(table(["Metric", "S static", "U unrestricted", "M margin only", "G confidence-gated"], [
        ["False-positive rate"] + [f(s8[x]["overall"]["fpr"]) for x in arms8], ["Recall"] + [f(s8[x]["overall"]["recall"]) for x in arms8], ["F1-score"] + [f(s8[x]["overall"]["f1"]) for x in arms8],
        ["Mean attack share in the retraining buffer (stream %.3f)" % (s8["S_static"]["overall"]["attacks"] / s8["S_static"]["overall"]["n"])] + ["-"] + [f(p8s["mean_buffer_attack_share"][x], 3) for x in arms8[1:]],
        ["Maximum attack share in a buffer"] + ["-"] + [f(p8s["max_buffer_attack_share"][x], 3) for x in arms8[1:]],
        ["Recovery after the 02-16 / 02-21 shift (intervals to FPR <= 10%)"] + [" / ".join("never" if sr8[x][d] is None else str(sr8[x][d]) for d in ("2018-02-16", "2018-02-21")) for x in arms8],
        ["02-20 (LOIC-HTTP day) F1 / recall / FPR"] + [f"{s8[x]['per_day']['2018-02-20']['f1']:.3f} / {s8[x]['per_day']['2018-02-20']['recall']:.3f} / {s8[x]['per_day']['2018-02-20']['fpr']:.3f}" for x in arms8],
        ["LOIC-HTTP flagged over the stream"] + [f(s8[x]["per_type"]["DDoS-LOIC-HTTP"]["flagged_share"], 3) for x in arms8]]))
    b8 = p8s["bootstrap_difference_mean_lo_hi"]
    a(f"\nOverall figures for S and M are inflated: on 02-16 and 02-21 the benign traffic is shifted entirely above their threshold, so they flag 90-94% of all records there (false-positive rate 99.6% and 99.9%). "
      f"G minus U (block bootstrap over intervals): F1 {ci(b8['f1']['G_gated - U_unrestricted'], 3)}, recall {ci(b8['recall']['G_gated - U_unrestricted'], 3)}, false-positive rate {ci(b8['fpr']['G_gated - U_unrestricted'], 3)}. "
      "Result: the gated design gave no clear advantage and did not lower contamination; only margin-only protected the buffer, and only by never adapting. Persistence sensitivity of G (3 / 6 / 12 intervals): LOIC-HTTP flagged " +
      " / ".join(f"{p8s['persistence_sensitivity_G'][k]['loic_flagged']:.3f}" for k in ("3", "6", "12")) + "; 02-16 false-positive rate " + " / ".join(f"{p8s['persistence_sensitivity_G'][k]['fpr_0216']:.3f}" for k in ("3", "6", "12")) +
      ". Figure: `results/phase8_confidence_overview.png`; details `reports/phase8_findings.md`.\n")

    # Table 4
    a("### Table 4 - Concept-drift experiments: drift detection approaches and recovery (Experiment F, Phase 9)\n")
    s9 = p9["arms"]
    arms9 = ["A_no_detection", "A2_periodic", "B_drift_adaptive", "C_drift_confirmed"]
    a("Drift monitor: 8 ADWIN detectors (river) on a fixed sentinel model's error distribution and flagged share, five traffic features and the UDP share, batches of 5,000 records, pre-fed with warm-up data. A: no detection, no updates. A2: scheduled retraining (reference). "
      "B: drift detection + adaptive baseline. C: drift detection + confidence-aware updates (persistence check + exclusion of the highest-scoring 20% of the recent samples). Test labels controlled neither the detector nor any update.\n")
    rc9 = p9d["recovery_intervals"]
    a(table(["Metric", "A no detection", "A2 periodic (ref.)", "B drift-triggered", "C drift + confirmation"], [
        ["False-positive rate"] + [f(s9[x]["overall"]["fpr"]) for x in arms9], ["Recall"] + [f(s9[x]["overall"]["recall"]) for x in arms9], ["F1-score"] + [f(s9[x]["overall"]["f1"]) for x in arms9],
        ["Retrainings"] + [str(p9d["retrainings"][x]) for x in arms9],
        ["Mean attack share in the retraining data"] + ["-"] + [f(p9d["mean_retrain_attack_share"][x], 3) for x in arms9[1:]],
        ["Recovery after the 02-16 / 02-21 shift (intervals to FPR <= 10%)"] + [" / ".join("never" if rc9[x][d] is None else str(rc9[x][d]) for d in ("2018-02-16", "2018-02-21")) for x in arms9],
        ["03-02 (Bot day) F1 / recall"] + [f"{s9[x]['per_day']['2018-03-02']['f1']:.3f} / {s9[x]['per_day']['2018-03-02']['recall']:.3f}" for x in arms9]]))
    bd9 = p9d["bootstrap_difference_mean_lo_hi"]
    dl = p9d["delays_records"]
    a(f"\n**Drift detection (B and C):** {p9d['n_detected']} of {len(p9d['change_points'])} true benign change points detected; delay after the change {', '.join(f'{d:,}' for d in dl)} records (median {int(np.median(dl)):,}); "
      f"**{p9d['false_alarms']} false alarms among {p9d['alarms']} alarms** ({p9d['false_alarms_per_million_records']:.2f} per million records; {p9d['false_alarms_in_attack_batches']} in attack-dominated batches). "
      f"B minus A2: F1 {ci(bd9['f1']['B_drift_adaptive - A2_periodic'], 3)}, false-positive rate {ci(bd9['fpr']['B_drift_adaptive - A2_periodic'], 3)}; C minus B: F1 {ci(bd9['f1']['C_drift_confirmed - B_drift_adaptive'], 3)}, false-positive rate {ci(bd9['fpr']['C_drift_confirmed - B_drift_adaptive'], 3)}. "
      "The true change points are four jumps between days defined from the static model's behaviour and the labels (evaluation only), so detection delay and false-alarm figures rest on four events; F1 degradation and recovery could not be assessed meaningfully (the before-change F1 is missing or an artefact of flagging everything), so recovery is reported through the false-positive rate. "
      "Figure: `results/phase9_drift_overview.png`; details `reports/phase9_findings.md`.\n")

    # Table 5
    a("### Table 5 - Unknown attack experiments: known vs unseen attack categories (Experiment C, Phase 6)\n")
    held = p6["withheld"]
    mte = load_split("test")[1]
    y, at = mte.label_binary.to_numpy(), mte.attack_type.to_numpy()
    unseen_m, benign_m = np.isin(at, held), at == "Benign"
    S6 = {m: np.load(R / f"scores_unseen_{m}_test.npy") for m in ("unsup_mixed", "normal_only", "supervised")}
    a(f"Withheld attack types: **{', '.join(held)}** (rule fixed beforehand: at least 10,000 test rows, distinct families, excluding Infiltration and LOIC-HTTP; I had seen earlier per-type results, so the choice is not blind). They were removed from the scaler fit, training, early stopping and the validation rows used for thresholds ({p6['refit']['train_rows_removed']:,} training attack rows removed). "
      f"Test: {int(benign_m.sum()):,} benign, {int(((y == 1) & ~unseen_m).sum()):,} known-attack, {int(unseen_m.sum()):,} withheld-attack records.\n")
    mm = ("unsup_mixed", "normal_only", "supervised")
    rows = []
    unseen_prec = {}
    for m in mm:
        pred = S6[m] > p6["models"][m]["thresholds"]["primary"]
        tp = int(pred[unseen_m].sum())
        fp = int(pred[benign_m].sum())
        unseen_prec[m] = tp / (tp + fp)
    rows.append(["Known attacks: recall"] + [f(p6["models"][m]["known"]["primary"]["recall"]) for m in mm])
    rows.append(["Known set: F1"] + [f(p6["models"][m]["known"]["primary"]["f1"]) for m in mm])
    rows.append(["Benign: false-positive rate"] + [f(p6["models"][m]["known"]["primary"]["fpr"]) for m in mm])
    rows.append(["**Unseen attacks: recall**"] + [f(p6["models"][m]["unseen"]["primary"]["recall"]) for m in mm])
    rows.append(["Unseen set: precision"] + [f(unseen_prec[m]) for m in mm])
    for h in held:
        rows.append([f"Unseen {h}: flagged share"] + [f(p6["models"][m]["unseen"]["primary"]["per_type"][h]["flagged_share"], 3) for m in mm])
    rows.append(["Known set ROC-AUC"] + [f(p6["models"][m]["known_ranking"]["roc_auc"]) for m in mm])
    rows.append(["Unseen set ROC-AUC (benign vs withheld)"] + [f(p6["models"][m]["unseen_ranking"]["roc_auc"]) for m in mm])
    rows.append(["Unseen set PR-AUC (chance %.4f)" % p6["models"]["supervised"]["unseen_base_rate"]] + [f(p6["models"][m]["unseen_ranking"]["pr_auc"]) for m in mm])
    a(table(["Metric", "Unsupervised AE", "One-class AE", "Supervised AE"], rows))
    a("\nThe same attack types when they were in the training data (Phases 3-5) versus withheld (ROC-AUC against benign; unsupervised / one-class / supervised):\n")
    rows = []
    for h in held:
        rows.append([h] + [f"{M5[m]['per_type_auc'][h]['auc_vs_benign']:.3f} -> {p6['models'][m]['unseen']['primary']['per_type'][h]['roc_auc_vs_benign']:.3f}" for m in mm])
    a(table(["Type (seen -> withheld)", "Unsupervised AE", "One-class AE", "Supervised AE"], rows))
    a("\nThe seen-versus-withheld models also differ in training size (45% fewer attack rows), and each withheld type is a single attack episode in few one-minute test blocks (Bot 68, SSH 18, HOIC 5, Hulk 1), so per-type numbers describe one episode. "
      "Limitations of reconstruction-error detection shown here: training-data contamination (any attack present in unlabeled training data is learned), error measures unfamiliarity rather than maliciousness (a few percent of benign flows score above most attacks, flooring the false-alarm rate), strong dependence on the threshold rule (the same one-class scores give 1.4% or 23% unseen recall), and attacks that look like ordinary traffic reconstruct well. "
      "Figure: `results/phase6_unseen_detection.png`; details `reports/phase6_findings.md`, `phase6_tables.md`.\n")

    # ------------------------------------------------------------------ 5 figures
    a("## 5. Figures\n")
    figs = [("phase3_loss_curves.png", "Experiment A loss curves (invalid-run fix noted in section 8)"), ("phase3_confusion_matrices.png", "Experiment A confusion matrices"), ("phase3_score_distributions.png", "Reconstruction-error distributions by class"),
            ("phase4_loss_curves.png", "Experiment B loss curves"), ("phase4_confusion_matrices.png", "Experiment B confusion matrices"), ("phase4_probability_distributions.png", "Supervised predicted probabilities"),
            ("phase5_roc_pr_comparison.png", "ROC and precision-recall curves, all four models"), ("phase5_confusion_unsup_vs_sup.png", "Confusion matrices, unsupervised vs supervised"), ("phase5_loss_curves_all.png", "Training and validation loss, all models"),
            ("phase6_unseen_detection.png", "Unknown-attack detection"), ("phase7_adaptive_overview.png", "Adaptive baseline overview"), ("phase8_confidence_overview.png", "Confidence-aware updates overview"), ("phase9_drift_overview.png", "Drift monitoring overview")]
    for fn, cap in figs:
        a(f"- `results/{fn}` - {cap}")
    a("")

    # ------------------------------------------------------------------ 6 observations
    a("## 6. Research observations\n")
    a("- **Supervised and unsupervised learning solve different problems.** A supervised autoencoder classifier learns the boundary between the classes it is shown, which is why it dominates when the test attacks resemble training attacks and fails (recall under 1%) on new ones. A reconstruction-based detector has no notion of the attack boundary; it measures unfamiliarity, which is what makes it able to flag unseen attacks and also what makes it flag unusual benign traffic.")
    a("- **Training-data purity is the dominant factor for the unsupervised family.** Removing 12% attacks from the training data changes the same architecture from near-chance (ROC-AUC 0.468) to 0.889; withholding specific attack types raises their detectability from 0.42-0.68 to 0.91-0.95. In the streaming experiments the same mechanism appears as buffer contamination, and in the one Phase 9 run the arm that retrained only on attack-free periods did best on the Bot day (a timing effect observed once, not a general result).")
    a("- **Operating point matters as much as ranking quality.** Several models rank well (high AUC) but flag few attacks at a threshold chosen from benign statistics, because the benign error distribution has a heavy tail. Validation-tuned thresholds change results by an order of magnitude and one of them is degenerate.")
    a("- **Label-free safeguards based on the model's own score do not separate attacks from benign change.** A margin filter, a persistence rule, a relative cut and a drift detector were each tried; none distinguishes a lasting benign shift from a long attack campaign, and the relative cut increased contamination.")
    a("- **Distribution shift between days was large for two days** (benign errors under the warm-up model one to two orders of magnitude higher), which the static detector cannot handle (it flagged 99.6-99.9% of benign records), and which retraining repaired in 5-7 intervals.\n")

    # ------------------------------------------------------------------ 7 limitations
    a("## 7. Limitations and threats to validity\n")
    for s in ["**One training run per model (seed 42).** Intervals shown describe sampling variation across test blocks or the stream, not training randomness; close results (for example the two supervised variants) are not distinguishable.",
              "**Data quality of the source files:** 7 of 10 files appear truncated at the spreadsheet row limit; timestamps lack AM/PM; de-duplication removed 33% of the cleaned records and reduced some attack types to a few dozen unique flows (per-type results for FTP-BruteForce, DoS-SlowHTTPTest, SQL injection, XSS and web brute force rest on under 110 test rows each).",
              "**In-distribution split is optimistic:** blocks of the same attack campaign are similar, and some flows span block boundaries. Only the unknown-attack experiment tests generalisation, with four withheld types, each a single episode.",
              "**Operating points:** thresholds were fixed in advance and perform poorly for the unsupervised family. Phases 3 and 6 used the 95th percentile of validation errors (unsupervised model) and the 99th percentile of benign validation errors (one-class model); Phase 7 the 99th percentile of warm-up benign errors; Phases 8-9 the 95th percentile of a clean reference set under the current model. These choices change recall strongly (Phase 7's 99th-percentile threshold sat at the benign tail and gave recall near 1%). Validation-tuned thresholds use validation labels.",
              "**Streaming experiments (Phases 7-9) rest on a stream with day-level changes only:** two days are single-hour snapshots, one day (02-20) is 57% of the records and one attack campaign, two days have almost no attacks, and the 'drift' events are four jumps between days defined from the static model's behaviour and the labels (evaluation only). Gradual drift was not tested. Design choices (gate parameters, response length, ADWIN settings, replay share) were fixed in advance and not tuned, so other designs may behave differently.",
              "**Preprocessing statistics:** the scaler was fitted on the Phase 2 training partition, which contains records from all days, so in the streaming experiments the min/max statistics (not the model weights) saw later days.",
              "**Cost measurements are unreliable:** wall-clock times vary about 3x between identical runs on this shared CPU and process-wide memory cannot be attributed to a model; only parameter counts and epochs are dependable.",
              "**Overall F1/recall of arms that flag nearly everything on shifted days are inflated** (Tables 3 and 4); false-positive rate and per-day tables should be read alongside.",
              "**No real-time deployment, packet capture, dashboard or live evaluation was built or tested,** as scoped."]:
        a(f"- {s}")
    a("")

    # ------------------------------------------------------------------ 8 corrections
    a("## 8. Corrections and disclosures made during the project\n")
    for s in ["**Phase 3 first run was invalid and discarded:** an argument-order bug set the early-stopping tolerance to 42, so epoch-1 weights were restored; the training functions now take keyword-only arguments and all Phase 3 numbers come from the corrected rerun.",
              "**Timestamps (found in Phase 7):** the Phase 1 'unexplained' hours 01-12 are a 12-hour clock without AM/PM; notes were added to the Phase 1 and Phase 2 reports. Phases 3-6 are unaffected (blocks are never split across partitions); the stream for Phases 7-9 uses corrected time.",
              "**Phase 5 timing caveat:** training times differed 3x between identical runs (313 s in Phase 4 vs 86 s in Phase 6), so the cost comparison was withdrawn as a ranking criterion.",
              "**Phase 7 threshold rule failed** (unstable median + k x MAD); Phase 8 and 9 use an anchor-based threshold chosen in response, which also changes the operating point from the 99th to the 95th percentile of clean errors. Phase 7's result is reported as run.",
              "**Phase 6:** the scaler refit without the withheld types left every min/max unchanged, so it did not influence results.",
              "**Hold-out selection (Phase 6)** was made after seeing earlier per-type results (stated in the Phase 6 report)."]:
        a(f"- {s}")
    a("")

    # ------------------------------------------------------------------ 9 reproduction
    a("## 9. Reproduction\n")
    a("See `README.md` for the environment and the exact commands for each phase. `reports/verification.md` lists the 118 independent cross-checks of the numbers in this report (the hybrid detector of Phase 11 is reported separately in `reports/phase11_hybrid.md`).\n")
    (REP / "FINAL_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/FINAL_REPORT.md,", len(L), "blocks")


if __name__ == "__main__":
    main()
