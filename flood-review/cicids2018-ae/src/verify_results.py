"""Independent cross-check of the reported numbers.

Recomputes the headline results from the raw saved artefacts (test scores, partition metadata, per-interval counts) with
scikit-learn / numpy directly (not the project's evaluation helpers) and compares them with the numbers each phase wrote
to its JSON. Writes reports/verification.md; exits non-zero if any check fails.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score

R, P = Path("results"), Path("data/processed")
checks = []


def check(name, got, expect, tol=1e-9):
    ok = bool(np.all(np.abs(np.asarray(got, float) - np.asarray(expect, float)) <= tol))
    checks.append((name, ok, float(np.max(np.abs(np.asarray(got, float) - np.asarray(expect, float))))))


def confusion_rates(y, pred):
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred), recall=recall_score(y, pred), f1=f1_score(y, pred),
                fpr=float(((pred == 1) & (y == 0)).sum() / (y == 0).sum()))


def main():
    mte = pd.read_pickle(P / "meta_test.pkl")
    y, at = mte.label_binary.to_numpy(), mte.attack_type.to_numpy()
    # ---- partitions and de-duplication
    p2 = json.load(open(P / "phase2_summary.json"))
    sizes = {s: len(pd.read_pickle(P / f"meta_{s}.pkl")) for s in ("train", "val", "test")}
    for s in sizes:
        check(f"Phase 2: {s} rows", sizes[s], p2["scale"]["partitions"][s]["rows"], 0)
    check("Phase 2: partitions sum to the de-duplicated total", sum(sizes.values()), p2["split"]["rows_after_dedup"], 0)
    check("Phase 2: no identical vectors across partitions", sum(p2["split"]["hash_overlap_between_partitions"].values()), 0, 0)
    p1 = json.load(open(R.parent / "reports" / "phase1_inspection.json"))
    check("Phase 1 vs Phase 2: records read", p1["total_rows"], sum(f["rows_in"] for f in p2["clean"]), 0)
    # ---- Table 1 (Phase 5): primary thresholds, recomputed from the saved test scores
    p3, p4, p5 = (json.load(open(R / f)) for f in ("phase3_results.json", "phase4_results.json", "phase5_comparison.json"))
    check("Phase 4: test label hash", int(hashlib.md5(y.astype(np.int8).tobytes()).hexdigest(), 16) % 10**12,
          int(p4["test_label_md5"], 16) % 10**12, 0)
    thr = {"unsup_mixed": p3["variants"]["unsup_mixed"]["primary"]["threshold"], "normal_only": p3["variants"]["normal_only"]["primary"]["threshold"],
           "supervised": 0.5, "supervised_recon": 0.5}
    for m, t in thr.items():
        s = np.load(R / f"scores_{m}_test.npy")
        rep = p5["models"][m]
        r = confusion_rates(y, (s > t).astype(int))
        for k, v in r.items():
            check(f"Table 1 {m}: {k}", v, rep["primary"][k], 1e-9)
        check(f"Table 1 {m}: ROC-AUC", roc_auc_score(y, s), rep["ranking"]["roc_auc"], 1e-9)
        check(f"Table 1 {m}: PR-AUC", average_precision_score(y, s), rep["ranking"]["pr_auc"], 1e-9)
    # ---- Table 5 (Phase 6)
    p6 = json.load(open(R / "phase6_results.json"))
    held = p6["withheld"]
    unseen, benign = np.isin(at, held), at == "Benign"
    for m in ("unsup_mixed", "normal_only", "supervised"):
        s = np.load(R / f"scores_unseen_{m}_test.npy")
        t = p6["models"][m]["thresholds"]["primary"]
        pred = s > t
        check(f"Table 5 {m}: known-attack recall", pred[(y == 1) & ~unseen].mean(), p6["models"][m]["known"]["primary"]["recall"], 1e-9)
        check(f"Table 5 {m}: unseen recall", pred[unseen].mean(), p6["models"][m]["unseen"]["primary"]["recall"], 1e-9)
        check(f"Table 5 {m}: benign FPR", pred[benign].mean(), p6["models"][m]["known"]["primary"]["fpr"], 1e-9)
        mk = benign | unseen
        check(f"Table 5 {m}: unseen ROC-AUC", roc_auc_score(unseen[mk].astype(int), s[mk]), p6["models"][m]["unseen_ranking"]["roc_auc"], 1e-9)
        mk = benign | ((y == 1) & ~unseen)
        check(f"Table 5 {m}: known ROC-AUC", roc_auc_score(y[mk], s[mk]), p6["models"][m]["known_ranking"]["roc_auc"], 1e-9)
    # ---- Phase 11 hybrid (supervised OR novelty), recomputed with sklearn's confusion matrix
    from sklearn.metrics import confusion_matrix
    p11 = json.load(open(R / "phase11_hybrid.json"))
    sup_s = np.load(R / "scores_supervised_test.npy") > 0.5
    for nm, m in (("sup OR unsup_mixed", "unsup_mixed"), ("sup OR normal_only", "normal_only")):
        fl = sup_s | (np.load(R / f"scores_{m}_test.npy") > thr[m])
        tn_, fp_, fn_, tp_ = confusion_matrix(y, fl.astype(int)).ravel()
        check(f"Phase 11 seen {nm}: tp,fp,fn,tn", [tp_, fp_, fn_, tn_], [p11["seen"][nm][k] for k in ("tp", "fp", "fn", "tn")], 0)
        fl6 = (np.load(R / "scores_unseen_supervised_test.npy") > 0.5) | (np.load(R / f"scores_unseen_{m}_test.npy") > p6["models"][m]["thresholds"]["primary"])
        check(f"Phase 11 unseen {nm}: unseen recall", fl6[unseen].mean(), p11["unseen"][nm]["unseen_recall"], 1e-12)
        check(f"Phase 11 unseen {nm}: known recall", fl6[(y == 1) & ~unseen].mean(), p11["unseen"][nm]["known_recall"], 1e-12)
        check(f"Phase 11 unseen {nm}: benign FPR", fl6[benign].mean(), p11["unseen"][nm]["benign_fpr"], 1e-12)
    # ---- Phase 12 threshold experiment at natural prevalence: raw-score thresholds and sklearn's f1_score (the script itself works on log10 scores)
    p12 = json.load(open(R / "phase12_thresholds.json"))
    for m in ("unsup_mixed", "normal_only"):
        for setting, raw, yy, thr_f in (("seen", np.load(R / f"scores_{m}_test.npy"), y, p3["variants"][m]["primary"]["threshold"]),
                                        ("unseen", np.load(R / f"scores_unseen_{m}_test.npy")[benign | unseen], unseen[benign | unseen].astype(int),
                                         p6["models"][m]["thresholds"]["primary"])):
            nat = p12[setting][m]["natural"]["methods"]
            pi_true = yy.mean()
            check(f"Phase 12 {setting} {m}: natural prevalence", pi_true, p12[setting][m]["natural"]["true_prevalence"], 1e-12)
            check(f"Phase 12 {setting} {m}: F-primary F1", f1_score(yy, (raw > thr_f).astype(int)), nat["F-primary"]["f1"][0], 1e-9)
            check(f"Phase 12 {setting} {m}: B-10% F1", f1_score(yy, (raw > np.percentile(raw, 90)).astype(int)), nat["B-10%"]["f1"][0], 1e-9)
            check(f"Phase 12 {setting} {m}: O-prev F1", f1_score(yy, (raw > np.percentile(raw, 100 * (1 - pi_true))).astype(int)), nat["O-prev"]["f1"][0], 1e-9)
            check(f"Phase 12 {setting} {m}: flag-all F1", 2 * pi_true / (1 + pi_true), nat["T-flag-all"]["f1"][0], 1e-9)
    # ---- Tables 2-4: per-interval counts vs reported overall metrics, and stream integrity
    ms = pd.read_pickle(P / "meta_stream.pkl")
    check("Stream: all records present once", len(ms), sum(sizes.values()), 0)
    check("Stream: sorted by corrected time", int(ms.time_true.is_monotonic_increasing), 1, 0)
    ev_n = len(ms) - p2_warm(ms)
    for tag, f, arms in (("Phase 7", "phase7_main", ["A_static", "B_adaptive_threshold", "C_adaptive_baseline"]),
                         ("Phase 8", "phase8_main", ["S_static", "U_unrestricted", "M_margin_only", "G_gated"]),
                         ("Phase 9", "phase9_main", ["A_no_detection", "A2_periodic", "B_drift_adaptive", "C_drift_confirmed"])):
        iv = pd.read_csv(R / f"{f}_intervals.csv")
        rep = json.load(open(R / f"{f}.json"))
        tot_att = None
        for a in arms:
            d = iv[iv.arm == a]
            tp, fp, fn, tn = (int(d[c].sum()) for c in ("tp", "fp", "fn", "tn"))
            ov = rep["arms"][a]["overall"]
            check(f"{tag} {a}: records in the stream", tp + fp + fn + tn, ev_n, 0)
            check(f"{tag} {a}: precision", tp / (tp + fp), ov["precision"], 1e-12)
            check(f"{tag} {a}: recall", tp / (tp + fn), ov["recall"], 1e-12)
            check(f"{tag} {a}: FPR", fp / (fp + tn), ov["fpr"], 1e-12)
            tot_att = tot_att if tot_att is not None else tp + fn
            check(f"{tag} {a}: attack records equal across arms", tp + fn, tot_att, 0)
        check(f"{tag}: attack records equal the stream's", tot_att, int(ms.iloc[len(ms) - ev_n:].label_binary.sum()), 0)
    L = ["# Verification of reported numbers\n", "Recomputed from the raw saved artefacts with scikit-learn / numpy and compared with what each phase reported.\n",
         "| Check | Result | Largest absolute difference |", "|---|---|---:|"]
    for n, ok, d in checks:
        L.append(f"| {n} | {'PASS' if ok else '**FAIL**'} | {d:.2e} |")
    fails = [c for c in checks if not c[1]]
    L.append(f"\n**{len(checks) - len(fails)} of {len(checks)} checks passed.**")
    (R.parent / "reports" / "verification.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(checks) - len(fails)} of {len(checks)} checks passed")
    for n, ok, d in fails:
        print("FAILED:", n, d)
    sys.exit(1 if fails else 0)


def p2_warm(ms):
    """Number of records on the two warm-up days (the first two days of the stream)."""
    days = sorted(ms.day.unique())
    return int(ms.day.isin(days[:2]).sum())


if __name__ == "__main__":
    main()
