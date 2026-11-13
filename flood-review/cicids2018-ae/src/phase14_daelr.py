"""Phase 14 / Experiment I (report Table D): the latent-regulated autoencoder DAE-LR of Nkashama et al. (arXiv:2407.08838) under this project's protocol.

Question: Phase 13 showed that trimmed-loss training does not close the gap between the autoencoder trained on unlabeled traffic that contains
about 12% attacks (`unsup_mixed`, ROC-AUC 0.47-0.53) and the same network trained on benign rows only (`normal_only`, 0.88-0.90). DAE-LR is the
mitigation proposed in the closest prior work on contamination of CSE-CIC-IDS2018 detectors. Does it close the gap here? (An evaluation of an
existing method, not a new one.)

Design, fixed before any result was seen:
  method   loss ||x - x_hat||^2 + lambda ||z - c||^2 with a learnable centre c (src/daelr.py). Score = reconstruction error (mean squared error per record).
  data     the same 1,000,000 unlabeled training rows and 300,000 unlabeled early-stopping rows as `unsup_mixed` of Phase 3 and trimmed-loss of
           Phase 13 for the same seed; the full validation and test partitions for thresholds and evaluation.
  lambda   {0.01, 0.1, 1, 10}. The paper text read for this project does not give lambda; it is NOT chosen with labels: every value is reported.
  network  the same 72-128-64-32-64-128-72 autoencoder, Adam lr 1e-3, batch 1024, up to 50 epochs, patience 5 (as Phases 3 and 13). c starts at zero.
  seeds    42, 1, 2 (reference baselines of the same seeds come from Phase 3).
  threshold  L1: the 95th percentile of all unlabeled validation errors (the rule of `unsup_mixed`; no attack labels).
  control  the same method trained on benign rows only (rows selected by label before training, like `normal_only`), threshold = 99th percentile of the
           benign validation errors (uses validation labels, like `normal_only`). It shows whether the latent regulariser changes the clean-data ceiling.
  metrics  test ROC-AUC and PR-AUC; precision/recall/F1/false-positive rate at the threshold; per-type ROC-AUC.
No attack label is read for training, early stopping or the L1 threshold of the contaminated models; the benign-only control reads labels only to choose its rows
and its threshold, exactly as `normal_only` does.

Usage: python src/phase14_daelr.py --seed 42 --out-dir results_daelr --model-dir models_daelr
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from daelr import LatentRegulatedAE, fit_daelr
from data import load_split, sample_rows
from evaluation import per_type_auc, ranking_metrics, threshold_metrics
from models import recon_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--variants", default="mixed,clean")
    ap.add_argument("--train-rows", type=int, default=1_000_000)
    ap.add_argument("--es-val-rows", type=int, default=300_000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--out-dir", type=Path, default=Path("results_daelr"))
    ap.add_argument("--model-dir", type=Path, default=Path("models_daelr"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    Xtr, mtr = load_split("train")
    Xva, mva = load_split("val")
    Xte, mte = load_split("test")
    d = Xtr.shape[1]
    Xva_e, Xte_e = np.asarray(Xva), np.asarray(Xte)
    yva, yte, tte = mva.label_binary.to_numpy(), mte.label_binary.to_numpy(), mte.attack_type.to_numpy()
    # the same draws as Phase 3 (unsup_mixed) and Phase 13 for this seed
    rng = np.random.default_rng(args.seed)
    es_idx = np.sort(rng.choice(len(Xva_e), min(args.es_val_rows, len(Xva_e)), replace=False))
    Xes = Xva_e[es_idx]
    pools = {"mixed": np.arange(len(mtr)), "clean": np.flatnonzero(mtr.label_binary.to_numpy() == 0)}
    out = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "models": {}}
    result_file = args.out_dir / f"phase14_seed{args.seed}.json"
    if result_file.exists():                                    # resume after an interruption: keep the models that finished, redo the rest
        out["models"] = json.load(open(result_file))["models"]
        print(f"resuming: {len(out['models'])} models already finished: {sorted(out['models'])}", flush=True)
    for variant in args.variants.split(","):
        pool = pools[variant]
        Xt = sample_rows(Xtr, pool, args.train_rows, args.seed)
        idx = pool if args.train_rows >= len(pool) else np.random.default_rng(args.seed).choice(pool, args.train_rows, replace=False)   # same draw as sample_rows
        ytr_share = float(mtr.label_binary.to_numpy()[np.sort(idx)].mean())                                                           # diagnostic only
        assert len(idx) == len(Xt)
        print(f"\nseed {args.seed} variant {variant}: {len(Xt):,} training rows (attack share {100 * ytr_share:.2f}%, diagnostic only)", flush=True)
        for lam in args.lams:
            tag = f"{variant}_lam{lam:g}"
            if tag in out["models"] and (args.out_dir / f"scores_daelr_{tag}_s{args.seed}_test.npy").exists() and (args.model_dir / f"daelr_{tag}_s{args.seed}.pt").exists():
                print(f"\n=== DAE-LR {tag}: already finished, skipped ===", flush=True)
                continue
            print(f"\n=== DAE-LR {tag} ===", flush=True)
            torch.manual_seed(args.seed)
            model = LatentRegulatedAE(d)
            model, hist, secs = fit_daelr(model, Xt, Xes, lam, epochs=args.epochs, batch=args.batch, lr=args.lr, patience=args.patience, seed=args.seed)
            torch.save(model.state_dict(), args.model_dir / f"daelr_{tag}_s{args.seed}.pt")
            s_val, s_te = recon_error(model, Xva_e), recon_error(model, Xte_e)
            np.save(args.out_dir / f"scores_daelr_{tag}_s{args.seed}_test.npy", s_te)
            if variant == "mixed":
                thr = {"L1": float(np.percentile(s_val, 95))}
            else:
                thr = {"B99": float(np.percentile(s_val[yva == 0], 99))}
            res = {"variant": variant, "lambda": lam, "epochs_run": len(hist["train"]), "train_seconds": secs, "train_attack_share_diagnostic": ytr_share,
                   "final_val_recon_sum": hist["val_recon"][int(np.argmin(hist["val"]))], "final_val_latent_sum": hist["val_latent"][int(np.argmin(hist["val"]))],
                   "centre_norm": float(model.c.detach().norm()), "ranking": ranking_metrics(yte, s_te), "per_type_auc": per_type_auc(tte, s_te),
                   "thresholds": thr, "at_threshold": {k: threshold_metrics(yte, s_te, t) for k, t in thr.items()}}
            out["models"][tag] = res
            r, a = res["ranking"], res["at_threshold"]
            print(f"  TEST ROC-AUC {r['roc_auc']:.4f} PR-AUC {r['pr_auc']:.4f} | " + " | ".join(
                f"{k}: f1 {a[k]['f1']:.4f} rec {a[k]['recall']:.4f} fpr {a[k]['fpr']:.4f}" for k in a), flush=True)
            tmp = result_file.with_suffix(".tmp")                  # saved after every model; written to a temp file first so a power-off cannot leave a half-written result
            json.dump(out, open(tmp, "w"), indent=2)
            os.replace(tmp, result_file)
    print("\nPhase 14 done")


if __name__ == "__main__":
    main()
