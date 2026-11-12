"""Phase 3 / Experiment A: unsupervised autoencoder anomaly detection.

Two variants share the architecture 72 -> 128 -> 64 -> 32 -> 64 -> 128 -> 72 (ReLU, sigmoid output, Adam, MSE):
  unsup_mixed  strictly unsupervised: trained on a random sample of ALL training rows, labels never read
               for training, early stopping or the primary threshold.
  normal_only  one-class / semi-supervised: trained only on benign training rows (selected by label BEFORE
               the labels are discarded). Clearly a different, label-assisted setting.

Thresholds come from VALIDATION data only:
  unsup_mixed  primary = the P-th percentile (default 95) of validation reconstruction error, no labels.
  normal_only  primary = the Q-th percentile (default 99) of validation error of benign rows (uses val labels).
  both         secondary = validation-F1-optimal threshold (uses val labels), reported separately.
Test labels are read only after the test predictions exist.

Usage: python src/phase3_unsupervised.py --out-dir results
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from data import load_split, sample_rows
from evaluation import best_f1_threshold, per_type_recall, ranking_metrics, threshold_metrics
from models import Autoencoder, recon_error
from plots import plot_confusion, plot_loss_curves, plot_score_hist
from training import fit_autoencoder


def peak_mb():
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2**20


def subsample(X, idx, n, seed):
    return X[np.sort(idx if n >= len(idx) else np.random.default_rng(seed).choice(idx, n, replace=False))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--train-rows", type=int, default=1_000_000)
    ap.add_argument("--es-val-rows", type=int, default=300_000, help="unlabeled val rows used for early stopping")
    ap.add_argument("--eval-rows", type=int, default=None, help="smoke test: subsample val and test")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--benign-percentile", type=float, default=99.0)
    ap.add_argument("--variants", default="unsup_mixed,normal_only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    Xtr, mtr = load_split("train")
    Xva, mva = load_split("val")
    Xte, mte = load_split("test")
    d = Xtr.shape[1]
    rng = np.random.default_rng(args.seed)
    if args.eval_rows:
        iv = np.sort(rng.choice(len(mva), min(args.eval_rows, len(mva)), replace=False))
        it = np.sort(rng.choice(len(mte), min(args.eval_rows, len(mte)), replace=False))
    else:
        iv, it = np.arange(len(mva)), np.arange(len(mte))
    Xva_e, Xte_e = np.asarray(Xva[iv]), np.asarray(Xte[it])
    yva, yte = mva.label_binary.to_numpy()[iv], mte.label_binary.to_numpy()[it]
    tte = mte.attack_type.to_numpy()[it]
    es_idx = np.sort(rng.choice(len(Xva_e), min(args.es_val_rows, len(Xva_e)), replace=False))
    Xes = Xva_e[es_idx]                       # unlabeled rows used only for early stopping

    results = {"config": vars(args) | {"out_dir": str(args.out_dir), "model_dir": str(args.model_dir)},
               "input_dim": d, "val_rows": int(len(iv)), "test_rows": int(len(it)), "variants": {}}
    hists, cms, scores_te, thr_primary = {}, {}, {}, {}

    for v in args.variants.split(","):
        print(f"\n=== {v} ===", flush=True)
        if v == "unsup_mixed":
            pool = np.arange(len(mtr))                     # every training row; labels not consulted
        elif v == "normal_only":
            pool = np.flatnonzero(mtr.label_binary.to_numpy() == 0)   # benign selected by label, then labels dropped
        else:
            raise SystemExit(f"unknown variant {v}")
        Xt = sample_rows(Xtr, pool, args.train_rows, args.seed)
        print(f"  training rows: {len(Xt):,} of {len(pool):,} eligible", flush=True)
        model = Autoencoder(d)
        model, hist, t_train = fit_autoencoder(
            model, Xt, Xes, epochs=args.epochs, batch=args.batch, lr=args.lr, patience=args.patience, seed=args.seed)
        torch.save(model.state_dict(), args.model_dir / f"ae_{v}.pt")
        del Xt

        t0 = time.time()
        s_val = recon_error(model, Xva_e)
        t_val = time.time() - t0
        t0 = time.time()
        s_te = recon_error(model, Xte_e)
        t_inf = time.time() - t0
        np.save(args.out_dir / f"scores_{v}_val.npy", s_val)
        np.save(args.out_dir / f"scores_{v}_test.npy", s_te)

        # ---- thresholds from validation only
        if v == "unsup_mixed":
            thr_main = float(np.percentile(s_val, args.percentile))
            rule = f"{args.percentile:g}th percentile of all validation errors (no labels)"
        else:
            thr_main = float(np.percentile(s_val[yva == 0], args.benign_percentile))
            rule = f"{args.benign_percentile:g}th percentile of benign validation errors (uses val labels)"
        thr_f1 = best_f1_threshold(yva, s_val)

        # ---- evaluation on test (labels used only now)
        rank = ranking_metrics(yte, s_te)
        out = {"threshold_rule_primary": rule, "train_rows": int(len(pool) if args.train_rows >= len(pool) else args.train_rows),
               "epochs_run": len(hist["train"]), "best_val_loss": float(min(hist["val"])),
               "train_seconds": t_train, "val_inference_seconds": t_val, "test_inference_seconds": t_inf,
               "test_inference_ms_per_1000_records": 1000 * 1000 * t_inf / len(s_te),
               "peak_process_memory_mb": peak_mb(), "ranking": rank,
               "primary": threshold_metrics(yte, s_te, thr_main) | per_type_summary(tte, s_te, thr_main),
               "val_f1_tuned": threshold_metrics(yte, s_te, thr_f1) | per_type_summary(tte, s_te, thr_f1),
               "val_f1_tuned_uses_val_labels": True,
               "val_metrics_at_primary": threshold_metrics(yva, s_val, thr_main)}
        results["variants"][v] = out
        hists[v], scores_te[v], thr_primary[v] = hist, s_te, thr_main
        cms[f"{v}: " + ("label-free p%g" % args.percentile if v == "unsup_mixed" else "benign-val p%g" % args.benign_percentile)] = {
            k: out["primary"][k] for k in ("tn", "fp", "fn", "tp")}
        p = out["primary"]
        print(f"  TEST  ROC-AUC {rank['roc_auc']:.4f}  PR-AUC {rank['pr_auc']:.4f} | primary thr {thr_main:.6f}: "
              f"acc {p['accuracy']:.4f} prec {p['precision']:.4f} rec {p['recall']:.4f} f1 {p['f1']:.4f} fpr {p['fpr']:.4f}", flush=True)
        q = out["val_f1_tuned"]
        print(f"        val-F1-tuned thr {thr_f1:.6f}: acc {q['accuracy']:.4f} prec {q['precision']:.4f} rec {q['recall']:.4f} "
              f"f1 {q['f1']:.4f} fpr {q['fpr']:.4f}", flush=True)

    json.dump(results, open(args.out_dir / "phase3_results.json", "w"), indent=2)
    plot_loss_curves(hists, args.out_dir / "phase3_loss_curves.png")
    plot_confusion(cms, args.out_dir / "phase3_confusion_matrices.png", "Test partition, primary threshold rule")
    plot_score_hist(scores_te, yte, args.out_dir / "phase3_score_distributions.png", thr_primary)
    print("\nPhase 3 done")


def per_type_summary(tte, s, thr):
    return {"per_type": per_type_recall(tte, s, thr)}


if __name__ == "__main__":
    main()
