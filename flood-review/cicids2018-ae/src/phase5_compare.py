"""Phase 5: compare the supervised and unsupervised autoencoders on the identical test partition.

Re-scores the saved test scores (no retraining), writes the result tables (reports/phase5_tables.md), figures
(results/phase5_*.png) and results/phase5_comparison.json. Also gives 95% block-bootstrap intervals: whole
one-minute test blocks are resampled, so the intervals reflect which time blocks happen to be in the test
partition. They do NOT include training randomness (one seed per model).

Usage: python src/phase5_compare.py --phase3-log <log> --phase4-log <log>
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from data import load_split
from evaluation import per_type_auc, per_type_recall, ranking_metrics, threshold_metrics
from models import Autoencoder, SupervisedAE
from plots import plot_confusion, plot_curves, plot_loss_curves

MODELS = ["unsup_mixed", "normal_only", "supervised", "supervised_recon"]
NAMES = {"unsup_mixed": "Unsupervised AE (unlabeled train)", "normal_only": "One-class AE (benign-only train)",
         "supervised": "Supervised AE", "supervised_recon": "Supervised AE + decoder"}
SHORT = {"unsup_mixed": "Unsupervised", "normal_only": "One-class", "supervised": "Supervised", "supervised_recon": "Sup.+decoder"}
SMALL_N = 110


def parse_log(path: Path):
    hist, cur = {}, None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\s*=== (\w+) ===", line)
        if m:
            cur = m.group(1)
            hist[cur] = {"train": [], "val": []}
            continue
        m = re.match(r"\s*epoch\s+(\d+)\s+train ([\d.]+)\s+val ([\d.]+)", line)
        if m and cur:
            hist[cur]["train"].append(float(m.group(2)))
            hist[cur]["val"].append(float(m.group(3)))
    return hist


def binned_counts(score, y, binv, nb, nbins=2000):
    edges = np.unique(np.quantile(score, np.linspace(0, 1, nbins + 1)[1:-1]))
    b = np.searchsorted(edges, score, side="right")
    k = len(edges) + 1
    H = [np.bincount(binv[y == c] * k + b[y == c], minlength=nb * k).reshape(nb, k) for c in (0, 1)]
    return H[1], H[0]                                   # positives, negatives (block x score-bin)


def ranking_from_hist(pos, neg, fpr_level=0.01):
    P, N = pos.sum(), neg.sum()
    below = np.cumsum(neg) - neg
    auc = float((pos * (below + 0.5 * neg)).sum() / (P * N))
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    prec, rec = tp / np.maximum(tp + fp, 1), tp / P
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    ok = fp / N <= fpr_level
    return auc, ap, float(rec[ok].max()) if ok.any() else 0.0


def metrics_from_counts(c):
    tp, fp, fn, tn = c
    div = lambda a, b: a / b if b else float("nan")
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"accuracy": div(tp + tn, tp + tn + fp + fn), "precision": p, "recall": r,
            "f1": div(2 * p * r, p + r) if tp else 0.0, "fpr": div(fp, fp + tn)}


def tpr_at_fpr(y, s, level=0.01):
    f, t, _ = roc_curve(y, s)
    return float(t[f <= level].max())


def thin(a, b, n=3000):
    idx = np.unique(np.linspace(0, len(a) - 1, n).astype(int))
    return a[idx], b[idx]


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci_str(lo_hi, nd=3):
    return f"[{lo_hi[0]:.{nd}f}, {lo_hi[1]:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--phase3-log", type=Path, required=True)
    ap.add_argument("--phase4-log", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    R = args.results_dir
    p3, p4 = json.load(open(R / "phase3_results.json")), json.load(open(R / "phase4_results.json"))
    _, mte = load_split("test")
    y = mte.label_binary.to_numpy()
    at = mte.attack_type.to_numpy()
    assert len(y) == p3["test_rows"] == p4["test_rows"], "test partitions differ"
    S = {m: np.load(R / f"scores_{m}_test.npy") for m in MODELS}
    assert all(len(s) == len(y) for s in S.values())

    # ---- thresholds: fixed in Phases 3/4 (primary) and validation-tuned
    prim, tuned = {}, {}
    for m in ("unsup_mixed", "normal_only"):
        prim[m] = p3["variants"][m]["primary"]["threshold"]
        tuned[m] = p3["variants"][m]["val_f1_tuned"]["threshold"]
    for m in ("supervised", "supervised_recon"):
        prim[m] = 0.5
        tuned[m] = p4["variants"][m]["threshold_val_f1"]["threshold"]
    saved = {**{m: p3["variants"][m] for m in ("unsup_mixed", "normal_only")}, **{m: p4["variants"][m] for m in ("supervised", "supervised_recon")}}

    out = {"test_rows": int(len(y)), "test_attack_rows": int(y.sum()), "attack_share": float(y.mean()), "models": {}}
    for m in MODELS:
        s = S[m]
        tm_p, tm_t = threshold_metrics(y, s, prim[m]), threshold_metrics(y, s, tuned[m])
        # consistency with the numbers saved when the models were run
        key_p = "primary" if m in ("unsup_mixed", "normal_only") else "threshold_0.5"
        key_t = "val_f1_tuned" if m in ("unsup_mixed", "normal_only") else "threshold_val_f1"
        for a, b in ((tm_p, saved[m][key_p]), (tm_t, saved[m][key_t])):
            assert all(a[k] == b[k] for k in ("tp", "fp", "fn", "tn")), f"{m}: recomputed confusion differs from saved"
        rk = ranking_metrics(y, s)
        assert abs(rk["roc_auc"] - saved[m]["ranking"]["roc_auc"]) < 1e-9
        out["models"][m] = {"primary": tm_p, "val_tuned": tm_t, "ranking": rk, "tpr_at_fpr_1pct": tpr_at_fpr(y, s, 0.01),
                            "per_type_flag_rate_primary": per_type_recall(at, s, prim[m]),
                            "per_type_auc": per_type_auc(at, s)}
    print("recomputed metrics match the saved Phase 3/4 results", flush=True)

    # ---- block bootstrap
    ub, binv = np.unique(mte.block.to_numpy(), return_inverse=True)
    nb = len(ub)
    H, C = {}, {}
    for m in MODELS:
        H[m] = binned_counts(S[m], y, binv, nb)
        for tag, thr in (("p", prim[m]), ("t", tuned[m])):
            pred = S[m] > thr
            C[m, tag] = np.stack([np.bincount(binv, weights=(pred & (y == 1)), minlength=nb), np.bincount(binv, weights=(pred & (y == 0)), minlength=nb),
                                  np.bincount(binv, weights=(~pred & (y == 1)), minlength=nb), np.bincount(binv, weights=(~pred & (y == 0)), minlength=nb)], 1)
    point_binned = {m: ranking_from_hist(H[m][0].sum(0), H[m][1].sum(0)) for m in MODELS}
    for m in MODELS:   # the histogram approximation must agree with sklearn's exact values
        pr_exact = out["models"][m]["ranking"]
        assert abs(point_binned[m][0] - pr_exact["roc_auc"]) < 2e-3 and abs(point_binned[m][1] - pr_exact["pr_auc"]) < 5e-3, (m, point_binned[m], pr_exact)
    rng = np.random.default_rng(args.seed)
    draws = {m: {"roc_auc": [], "pr_auc": [], "tpr1": []} for m in MODELS}
    tdraws = {(m, t): {k: [] for k in ("accuracy", "precision", "recall", "f1", "fpr")} for m in MODELS for t in ("p", "t")}
    for _ in range(args.boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        for m in MODELS:
            a, b, c = ranking_from_hist(w @ H[m][0], w @ H[m][1])
            draws[m]["roc_auc"].append(a), draws[m]["pr_auc"].append(b), draws[m]["tpr1"].append(c)
            for t in ("p", "t"):
                mm = metrics_from_counts(w @ C[m, t])
                for k, v in mm.items():
                    tdraws[m, t][k].append(v)
    q = lambda arr: [float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))]
    for m in MODELS:
        out["models"][m]["ci_ranking"] = {k: q(v) for k, v in draws[m].items()}
        out["models"][m]["ci_primary"] = {k: q(v) for k, v in tdraws[m, "p"].items()}
        out["models"][m]["ci_val_tuned"] = {k: q(v) for k, v in tdraws[m, "t"].items()}
    pairs = [("supervised", "normal_only"), ("supervised_recon", "supervised"), ("normal_only", "unsup_mixed")]
    out["paired_diff"] = {}
    for a, b in pairs:
        for k in ("roc_auc", "pr_auc"):
            d = np.array(draws[a][k]) - np.array(draws[b][k])
            out["paired_diff"][f"{a} - {b} | {k}"] = {"mean": float(d.mean()), "ci95": q(d), "share_draws_above_0": float((d > 0).mean())}
    out["bootstrap"] = {"resamples": args.boot, "blocks": int(nb), "unit": "one-minute test blocks"}

    # ---- cost
    for m in MODELS:
        mod = Autoencoder(72) if m in ("unsup_mixed", "normal_only") else SupervisedAE(72, decoder=(m == "supervised_recon"))
        mod.load_state_dict(torch.load(args.model_dir / f"ae_{m}.pt"))
        pth = args.model_dir / f"ae_{m}.pt"
        out["models"][m]["cost"] = {"parameters": int(sum(p.numel() for p in mod.parameters())), "model_file_kb": pth.stat().st_size / 1024,
                                    "epochs_run": saved[m]["epochs_run"], "train_seconds": saved[m]["train_seconds"],
                                    "test_inference_seconds": saved[m]["test_inference_seconds"],
                                    "ms_per_1000_records": saved[m]["test_inference_ms_per_1000_records"]}
    out["val_fpr_at_primary"] = {m: (p3["variants"][m]["val_metrics_at_primary"]["fpr"] if m in ("unsup_mixed", "normal_only")
                                     else p4["variants"][m]["val_metrics_at_0.5"]["fpr"]) for m in MODELS}

    # ---- loss histories recovered from the logged training output (checked against saved best-val values)
    hist = {**parse_log(args.phase3_log), **parse_log(args.phase4_log)}
    for m in MODELS:
        assert abs(min(hist[m]["val"]) - saved[m]["best_val_loss"]) < 2e-6, f"{m}: logged val loss does not match saved best"
        assert len(hist[m]["val"]) == saved[m]["epochs_run"]
    json.dump(hist, open(R / "loss_histories.json", "w"), indent=2)
    out["loss_history_source"] = "parsed from the training logs of the final Phase 3 and Phase 4 runs; verified against saved best-val loss and epoch counts"

    # ---- figures
    roc, prc = {}, {}
    for m in MODELS:
        f, t, _ = roc_curve(y, S[m])
        roc[m] = thin(f, t)
        p, r, _ = precision_recall_curve(y, S[m])
        prc[m] = thin(r[::-1], p[::-1])
    plot_curves(roc, prc, float(y.mean()), R / "phase5_roc_pr_comparison.png")
    plot_confusion({"Unsupervised AE, label-free p95": {k: out["models"]["unsup_mixed"]["primary"][k] for k in ("tn", "fp", "fn", "tp")},
                    "Supervised AE, threshold 0.5": {k: out["models"]["supervised"]["primary"][k] for k in ("tn", "fp", "fn", "tp")}},
                   R / "phase5_confusion_unsup_vs_sup.png", "Test partition (2,163,852 records), primary thresholds")
    plot_confusion({f"{NAMES[m]}": {k: out["models"][m]["primary"][k] for k in ("tn", "fp", "fn", "tp")} for m in MODELS},
                   R / "phase5_confusion_all_models.png", "Test partition, primary thresholds")
    plot_loss_curves({m: hist[m] for m in MODELS}, R / "phase5_loss_curves_all.png", ncols=2,
                     ylabels={"unsup_mixed": "MSE reconstruction loss", "normal_only": "MSE reconstruction loss",
                              "supervised": "weighted BCE loss", "supervised_recon": "weighted BCE + reconstruction"})
    json.dump(out, open(R / "phase5_comparison.json", "w"), indent=2)

    # ---- tables
    M = out["models"]
    L = []
    a = L.append
    row = lambda label, fn: a(f"| {label} | " + " | ".join(fn(m) for m in MODELS) + " |")
    hdr = "| Metric | " + " | ".join(SHORT[m] for m in MODELS) + " |\n|---|" + "---:|" * len(MODELS)
    a(f"Test partition: {out['test_rows']:,} records, {out['test_attack_rows']:,} attacks ({100*out['attack_share']:.2f}%), identical for all models.\n")
    a("## Table 1a - primary thresholds (rules fixed before seeing results)\n")
    a(hdr)
    rules = {"unsup_mixed": "95th pct. of val errors (no labels)", "normal_only": "99th pct. of benign val errors (val labels)",
             "supervised": "probability 0.5", "supervised_recon": "probability 0.5"}
    row("Threshold rule", lambda m: rules[m])
    for k, lab in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("specificity", "Specificity"), ("fpr", "False-positive rate"), ("fnr", "False-negative rate")):
        row(lab, lambda m, k=k: fmt(M[m]["primary"][k]))
    row("FPR on validation, same threshold", lambda m: fmt(out["val_fpr_at_primary"][m]))
    row("ROC-AUC", lambda m: fmt(M[m]["ranking"]["roc_auc"]))
    row("PR-AUC (chance = %.4f)" % out["attack_share"], lambda m: fmt(M[m]["ranking"]["pr_auc"]))
    a("\n## Table 1b - validation-tuned thresholds (F1-optimal on validation; uses validation labels)\n")
    a(hdr)
    row("Threshold", lambda m: f"{tuned[m]:.6g}")
    for k, lab in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("fpr", "False-positive rate"), ("fnr", "False-negative rate")):
        row(lab, lambda m, k=k: fmt(M[m]["val_tuned"][k]))
    a("\nUnsupervised column: the validation optimum is degenerate (flags 99.99% of records; F1 equals the trivial flag-everything value), so it is not a usable operating point.\n")
    a("## Table 1c - threshold-free metrics with 95% block-bootstrap intervals\n")
    a(hdr)
    row("ROC-AUC", lambda m: f"{fmt(M[m]['ranking']['roc_auc'], 3)} {ci_str(M[m]['ci_ranking']['roc_auc'])}")
    row("PR-AUC", lambda m: f"{fmt(M[m]['ranking']['pr_auc'], 3)} {ci_str(M[m]['ci_ranking']['pr_auc'])}")
    row("Recall at 1% FPR", lambda m: f"{fmt(M[m]['tpr_at_fpr_1pct'], 3)} {ci_str(M[m]['ci_ranking']['tpr1'])}")
    a("\n## Table 1d - 95% block-bootstrap intervals at the primary thresholds\n")
    a(hdr)
    for k, lab in (("f1", "F1-score"), ("recall", "Recall"), ("precision", "Precision"), ("fpr", "False-positive rate")):
        row(lab, lambda m, k=k: f"{fmt(M[m]['primary'][k], 3)} {ci_str(M[m]['ci_primary'][k])}")
    a("\n## Table 1e - paired differences (same bootstrap resamples)\n\n| Comparison | Metric | Mean difference | 95% interval | Share of resamples > 0 |\n|---|---|---:|---|---:|")
    for k, v in out["paired_diff"].items():
        cmp_, met = k.split(" | ")
        a(f"| {cmp_} | {met} | {v['mean']:+.4f} | {ci_str(v['ci95'], 4)} | {v['share_draws_above_0']:.2f} |")
    a("\nIntervals cover which test blocks are drawn, not training randomness: every model was trained once (seed 42).\n")
    a("## Table 1f - cost\n")
    a(hdr)
    row("Epochs run", lambda m: str(M[m]["cost"]["epochs_run"]))
    row("Training time (s, CPU)", lambda m: f"{M[m]['cost']['train_seconds']:.0f}")
    row("Inference, whole test set (s)", lambda m: f"{M[m]['cost']['test_inference_seconds']:.1f}")
    row("Inference (ms per 1000 records)", lambda m: f"{M[m]['cost']['ms_per_1000_records']:.2f}")
    row("Parameters", lambda m: f"{M[m]['cost']['parameters']:,}")
    row("Model file (KB)", lambda m: f"{M[m]['cost']['model_file_kb']:.0f}")
    a("\nProcess-wide peak memory was 5.15 GB (Phase 3) and 6.36 GB (Phase 4) including memory-mapped data; it cannot be attributed to a single model, so model size is given by parameters and file size.\n"
      "Training and inference times are single wall-clock measurements on a shared CPU and are not reliable for ranking models: the plain supervised AE needed 313 s (27 epochs) in Phase 4 but 86 s (24 epochs) when retrained in Phase 6 on the same amount of data. Epochs run and parameter counts are the dependable cost figures.\n")
    a("## Table 1g - detection by attack type on the test partition\n\nFlagged share at the primary threshold (benign row = false-positive rate), and ROC-AUC of the type against benign. Types with fewer than %d test rows are marked (*) and are not reliable.\n" % SMALL_N)
    a("| Type | Test rows | " + " | ".join(f"{SHORT[m]} flagged" for m in MODELS) + " | " + " | ".join(f"{SHORT[m]} AUC" for m in MODELS) + " |\n|---|---:|" + "---:|" * (2 * len(MODELS)))
    for t in ["Benign"] + sorted(set(at) - {"Benign"}):
        n = int((at == t).sum())
        fl = [fmt(M[m]["per_type_flag_rate_primary"][t]["rate"], 3) for m in MODELS]
        au = ["-" if t == "Benign" else fmt(M[m]["per_type_auc"][t]["auc_vs_benign"], 3) for m in MODELS]
        a(f"| {t}{' (*)' if (t != 'Benign' and n < SMALL_N) else ''} | {n:,} | " + " | ".join(fl) + " | " + " | ".join(au) + " |")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "phase5_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Phase 5 done")


if __name__ == "__main__":
    main()
