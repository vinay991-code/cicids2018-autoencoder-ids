"""Phase 4 / Experiment B: supervised autoencoder classifier.

Encoder 72 -> 128 -> 64 -> 32 (latent) -> classification head (16 -> 1 logit). Loss: class-weighted binary
cross-entropy (pos_weight = n_benign / n_attack of the training rows), Adam, early stopping on the same loss over
labeled validation rows. Variant `supervised_recon` adds the decoder and a reconstruction term
(loss = weighted BCE + recon_weight * per-row summed squared error).

Trained on exactly the same 1,000,000 training rows as the Phase 3 unsupervised variant (same sampling seed),
and evaluated on exactly the same full test partition. Test labels are read only after test predictions exist.
Thresholds (both from validation data only): fixed 0.5, and the F1-optimal validation threshold.

Usage: python src/phase4_supervised.py --out-dir results
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from data import load_split, sample_rows
from evaluation import best_f1_threshold, per_type_auc, per_type_recall, ranking_metrics, threshold_metrics
from models import SupervisedAE, predict_proba
from plots import plot_confusion, plot_loss_curves, plot_prob_hist
from training import fit_supervised


def peak_mb():
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--train-rows", type=int, default=1_000_000)
    ap.add_argument("--es-val-rows", type=int, default=300_000, help="labeled val rows used for early stopping")
    ap.add_argument("--eval-rows", type=int, default=None, help="smoke test: subsample val and test")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--recon-weight", type=float, default=1.0)
    ap.add_argument("--variants", default="supervised,supervised_recon")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    Xtr, mtr = load_split("train")
    Xva, mva = load_split("val")
    Xte, mte = load_split("test")
    d = Xtr.shape[1]
    ytr_all = mtr.label_binary.to_numpy()
    # same sampling call as Phase 3 (pool = every training row, same seed) -> identical training rows
    rng = np.random.default_rng(args.seed)
    n_tr = min(args.train_rows, len(mtr))
    tr_idx = np.sort(np.random.default_rng(args.seed).choice(np.arange(len(mtr)), n_tr, replace=False)) if n_tr < len(mtr) else np.arange(len(mtr))
    Xt, yt = np.asarray(Xtr[tr_idx], dtype=np.float32), ytr_all[tr_idx]
    if args.eval_rows:
        iv = np.sort(rng.choice(len(mva), min(args.eval_rows, len(mva)), replace=False))
        it = np.sort(rng.choice(len(mte), min(args.eval_rows, len(mte)), replace=False))
    else:
        iv, it = np.arange(len(mva)), np.arange(len(mte))
    Xva_e, Xte_e = np.asarray(Xva[iv]), np.asarray(Xte[it])
    yva, yte = mva.label_binary.to_numpy()[iv], mte.label_binary.to_numpy()[it]
    tte = mte.attack_type.to_numpy()[it]
    es_idx = np.sort(rng.choice(len(Xva_e), min(args.es_val_rows, len(Xva_e)), replace=False))
    Xes, yes = Xva_e[es_idx], yva[es_idx]
    print(f"train rows {len(Xt):,} (attack share {100*yt.mean():.2f}%), val {len(Xva_e):,}, test {len(Xte_e):,}", flush=True)

    results = {"config": vars(args) | {"out_dir": str(args.out_dir), "model_dir": str(args.model_dir)},
               "input_dim": d, "train_rows": int(len(Xt)), "train_attack_rows": int(yt.sum()),
               "val_rows": int(len(iv)), "test_rows": int(len(it)),
               "test_label_md5": hashlib.md5(yte.astype(np.int8).tobytes()).hexdigest(), "variants": {}}
    hists, cms, probs_te, thr_all = {}, {}, {}, {}

    for v in args.variants.split(","):
        print(f"\n=== {v} ===", flush=True)
        if v not in ("supervised", "supervised_recon"):
            raise SystemExit(f"unknown variant {v}")
        use_dec = v == "supervised_recon"
        torch.manual_seed(args.seed)
        model = SupervisedAE(d, decoder=use_dec)
        model, hist, t_train = fit_supervised(
            model, Xt, yt, Xes, yes, epochs=args.epochs, batch=args.batch, lr=args.lr, patience=args.patience,
            seed=args.seed, recon_weight=args.recon_weight if use_dec else 0.0)
        torch.save(model.state_dict(), args.model_dir / f"ae_{v}.pt")

        t0 = time.time()
        p_val = predict_proba(model, Xva_e)
        t_val = time.time() - t0
        t0 = time.time()
        p_te = predict_proba(model, Xte_e)
        t_inf = time.time() - t0
        np.save(args.out_dir / f"scores_{v}_val.npy", p_val)
        np.save(args.out_dir / f"scores_{v}_test.npy", p_te)

        thr_f1 = best_f1_threshold(yva, p_val)          # validation labels only
        rank = ranking_metrics(yte, p_te)
        m05 = threshold_metrics(yte, p_te, 0.5)
        mf1 = threshold_metrics(yte, p_te, thr_f1)
        out = {"epochs_run": len(hist["train"]), "best_val_loss": float(min(hist["val"])), "pos_weight": hist["pos_weight"],
               "train_seconds": t_train, "val_inference_seconds": t_val, "test_inference_seconds": t_inf,
               "test_inference_ms_per_1000_records": 1000 * 1000 * t_inf / len(p_te),
               "peak_process_memory_mb": peak_mb(), "ranking": rank,
               "threshold_0.5": m05 | {"per_type": per_type_recall(tte, p_te, 0.5)},
               "threshold_val_f1": mf1 | {"per_type": per_type_recall(tte, p_te, thr_f1)},
               "per_type_auc_vs_benign": per_type_auc(tte, p_te),
               "val_metrics_at_0.5": threshold_metrics(yva, p_val, 0.5)}
        results["variants"][v] = out
        hists[v], probs_te[v], thr_all[v] = hist, p_te, {"0.5": 0.5, "val-F1": thr_f1}
        cms[f"{v}: threshold 0.5"] = {k: m05[k] for k in ("tn", "fp", "fn", "tp")}
        print(f"  TEST  ROC-AUC {rank['roc_auc']:.4f}  PR-AUC {rank['pr_auc']:.4f} | thr 0.5: acc {m05['accuracy']:.4f} "
              f"prec {m05['precision']:.4f} rec {m05['recall']:.4f} f1 {m05['f1']:.4f} fpr {m05['fpr']:.4f}", flush=True)
        print(f"        val-F1 thr {thr_f1:.4f}: acc {mf1['accuracy']:.4f} prec {mf1['precision']:.4f} rec {mf1['recall']:.4f} "
              f"f1 {mf1['f1']:.4f} fpr {mf1['fpr']:.4f}", flush=True)

    json.dump(results, open(args.out_dir / "phase4_results.json", "w"), indent=2)
    plot_loss_curves(hists, args.out_dir / "phase4_loss_curves.png", ylabel="weighted BCE loss (+ recon term)")
    plot_confusion(cms, args.out_dir / "phase4_confusion_matrices.png", "Test partition, threshold 0.5")
    plot_prob_hist(probs_te, yte, args.out_dir / "phase4_probability_distributions.png", thr_all)
    print("\nPhase 4 done")


if __name__ == "__main__":
    main()
