"""Phase 13 / Experiment I: does label-free robust training close the contamination gap? (evaluation of an EXISTING idea on this dataset)

Background: Phases 3-5 showed that the autoencoder trained on unlabeled traffic that contains about 12% attacks (`unsup_mixed`) ranks attacks
no better than chance (ROC-AUC 0.47-0.53) while the same network trained on benign rows only (`normal_only`) reaches 0.88-0.89. The literature
already contains robust training methods that need no labels (trimmed loss, RCA, iterative training-set cleaning), so this experiment makes
no claim of a new method; it measures how much of that gap a trimmed-loss autoencoder closes on CSE-CIC-IDS2018.

Design, fixed before any result was seen:
  data     the same 1,000,000 unlabeled training rows and 300,000 unlabeled early-stopping rows as `unsup_mixed` of Phase 3 for the same seed
  models   trimmed-loss autoencoder (src/robust.py) for trimmed shares q in {0.10, 0.20, 0.30} (the true attack share, about 0.123, is NOT used to choose q)
  seeds    42, 1, 2 (baselines of the same seeds come from Phase 3)
  scores   per-record reconstruction error
  label-free thresholds (validation errors of the unlabeled validation rows only):
     L1  95th percentile of all validation errors (the rule used for `unsup_mixed`)
     L2  95th percentile of the validation errors below the (1-q) quantile (a pseudo-clean reference selected without labels)
     L3  99th percentile of the same pseudo-clean reference
  metrics  test ROC-AUC and PR-AUC; precision/recall/F1/false-positive rate at L1-L3
Analysis-only diagnostic (labels used for reporting, never for training or thresholds): the attack share among the training rows that the final model
keeps (error below its (1-q) quantile), against the share in all training rows.

Usage: python src/phase13_trimmed.py --seed 42 --out-dir results_trim --model-dir models_trim
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load_split
from evaluation import per_type_auc, ranking_metrics, threshold_metrics
from models import Autoencoder, recon_error
from robust import fit_trimmed_autoencoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qs", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    ap.add_argument("--train-rows", type=int, default=1_000_000)
    ap.add_argument("--es-val-rows", type=int, default=300_000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--ramp", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=Path("results_trim"))
    ap.add_argument("--model-dir", type=Path, default=Path("models_trim"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    Xtr, mtr = load_split("train")
    Xva, mva = load_split("val")
    Xte, mte = load_split("test")
    d = Xtr.shape[1]
    Xva_e, Xte_e = np.asarray(Xva), np.asarray(Xte)
    yte, tte = mte.label_binary.to_numpy(), mte.attack_type.to_numpy()
    # same draws as Phase 3 (unsup_mixed): early-stopping rows first from rng(seed), training rows from a fresh rng(seed) over every training row
    rng = np.random.default_rng(args.seed)
    es_idx = np.sort(rng.choice(len(Xva_e), min(args.es_val_rows, len(Xva_e)), replace=False))
    Xes = Xva_e[es_idx]
    n = min(args.train_rows, len(mtr))
    tr_idx = np.sort(np.random.default_rng(args.seed).choice(np.arange(len(mtr)), n, replace=False)) if n < len(mtr) else np.arange(len(mtr))
    Xt = np.asarray(Xtr[tr_idx], dtype=np.float32)
    ytr = mtr.label_binary.to_numpy()[tr_idx]          # analysis-only diagnostic below
    print(f"seed {args.seed}: training rows {len(Xt):,} (attack share {100 * ytr.mean():.2f}%, not used), test {len(Xte_e):,}", flush=True)

    out = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "train_attack_share": float(ytr.mean()), "models": {}}
    for q in args.qs:
        print(f"\n=== trimmed q={q} ===", flush=True)
        torch.manual_seed(args.seed)
        model = Autoencoder(d)
        model, hist, secs = fit_trimmed_autoencoder(model, Xt, Xes, q, ramp=args.ramp, epochs=args.epochs, batch=args.batch, lr=args.lr,
                                                    patience=args.patience, seed=args.seed)
        tag = f"trim{int(round(q * 100)):02d}"
        torch.save(model.state_dict(), args.model_dir / f"ae_{tag}_s{args.seed}.pt")
        s_val, s_te, s_tr = recon_error(model, Xva_e), recon_error(model, Xte_e), recon_error(model, Xt)
        np.save(args.out_dir / f"scores_{tag}_s{args.seed}_test.npy", s_te)
        ref = s_val[s_val <= np.quantile(s_val, 1.0 - q)]
        thr = {"L1": float(np.percentile(s_val, 95)), "L2": float(np.percentile(ref, 95)), "L3": float(np.percentile(ref, 99))}
        keep_mask = s_tr <= np.quantile(s_tr, 1.0 - q)
        res = {"epochs_run": len(hist["train"]), "train_seconds": secs, "ranking": ranking_metrics(yte, s_te), "per_type_auc": per_type_auc(tte, s_te),
               "thresholds": thr, "at_threshold": {k: threshold_metrics(yte, s_te, t) for k, t in thr.items()},
               "diagnostic_attack_share_kept": float(ytr[keep_mask].mean()), "diagnostic_attack_share_all": float(ytr.mean()),
               "diagnostic_attacks_removed_share": float(1.0 - ytr[keep_mask].sum() / max(ytr.sum(), 1))}
        out["models"][tag] = res
        r, a = res["ranking"], res["at_threshold"]
        print(f"  TEST ROC-AUC {r['roc_auc']:.4f} PR-AUC {r['pr_auc']:.4f} | " + " | ".join(
            f"{k}: f1 {a[k]['f1']:.4f} rec {a[k]['recall']:.4f} fpr {a[k]['fpr']:.4f}" for k in a), flush=True)
        print(f"  diagnostic: attack share kept {100 * res['diagnostic_attack_share_kept']:.2f}% vs all {100 * res['diagnostic_attack_share_all']:.2f}%; "
              f"{100 * res['diagnostic_attacks_removed_share']:.1f}% of the attacks removed", flush=True)
    json.dump(out, open(args.out_dir / f"phase13_seed{args.seed}.json", "w"), indent=2)
    print("\nPhase 13 done")


if __name__ == "__main__":
    main()
