"""Phase 6 / Experiment C: detection of attack types withheld from training.

The withheld types are removed from EVERYTHING that is fitted or tuned: the scaler min/max (refit on the
remaining training rows by inverting the Phase 2 scaling), the training rows, early stopping, and the
validation rows used for thresholds. Three models are then trained on the remaining data:
  unsup_mixed  strictly unsupervised AE on unlabeled remaining training rows (the withheld rows are not in it).
  normal_only  one-class AE on benign training rows.
  supervised   classifier AE on the remaining labeled rows (benign vs the KNOWN attack types).
Thresholds use the same pre-fixed rules as Phases 3/4 but computed on the filtered validation rows.
Evaluation is on the full Phase 2 test partition, split into benign, known attacks and withheld attacks.

Usage: python src/phase6_unseen.py --out-dir results
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

from data import load_split
from evaluation import best_f1_threshold, ranking_metrics, threshold_metrics
from models import Autoencoder, SupervisedAE, predict_proba, recon_error
from plots import plot_unseen
from training import fit_autoencoder, fit_supervised

MODELS = ["unsup_mixed", "normal_only", "supervised"]


def block_ci(flag, groups, blocks, boot, seed):
    """95% block-bootstrap interval of flagged share for each boolean group mask (resampling test blocks)."""
    ub, bi = np.unique(blocks, return_inverse=True)
    nb = len(ub)
    num = {g: np.bincount(bi, weights=(flag & m), minlength=nb) for g, m in groups.items()}
    den = {g: np.bincount(bi, weights=m, minlength=nb) for g, m in groups.items()}
    rng = np.random.default_rng(seed)
    draws = {g: [] for g in groups}
    for _ in range(boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        for g in groups:
            d = w @ den[g]
            draws[g].append((w @ num[g]) / d if d else np.nan)
    return {g: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for g, v in draws.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--holdout", default="Bot,DDoS-HOIC,SSH-BruteForce,DoS-Hulk")
    ap.add_argument("--train-rows", type=int, default=1_000_000)
    ap.add_argument("--es-val-rows", type=int, default=300_000)
    ap.add_argument("--eval-rows", type=int, default=None, help="smoke test: subsample filtered val and test")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--benign-percentile", type=float, default=99.0)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    held = [h.strip() for h in args.holdout.split(",")]

    sc = json.load(open("data/processed/scaler.json"))
    nnum = len(sc["model_columns"])
    lmin0, lmax0 = np.array(sc["log_min"]), np.array(sc["log_max"])
    span0 = np.where(lmax0 > lmin0, lmax0 - lmin0, 1.0)
    Xtr, mtr = load_split("train")
    Xva, mva = load_split("val")
    Xte, mte = load_split("test")
    d = Xtr.shape[1]
    assert set(held) <= set(mtr.attack_type.unique()), "unknown hold-out type"

    # ---- everything fitted below sees only non-withheld training rows
    att_tr = mtr.attack_type.to_numpy()
    keep_tr = np.flatnonzero(~np.isin(att_tr, held))
    lo, hi, proto_rows = np.full(nnum, np.inf), np.full(nnum, -np.inf), np.zeros(d - nnum)
    for a in range(0, len(keep_tr), 500_000):
        ch = np.asarray(Xtr[keep_tr[a:a + 500_000]], dtype=np.float64)
        L = ch[:, :nnum] * span0 + lmin0                       # back to the log space (Phase 2 scaling inverted)
        lo, hi = np.minimum(lo, L.min(0)), np.maximum(hi, L.max(0))
        proto_rows += ch[:, nnum:].sum(0)
    span1 = np.where(hi > lo, hi - lo, 1.0)
    cols = sc["output_columns"]
    refit = {"withheld_types": held, "train_rows_kept": int(len(keep_tr)), "train_rows_removed": int(len(att_tr) - len(keep_tr)),
             "removed_by_type": {h: int((att_tr == h).sum()) for h in held},
             "columns_constant_after_refit": [cols[i] for i in range(nnum) if hi[i] == lo[i]],
             "protocol_columns_empty_after_refit": [cols[nnum + i] for i in range(d - nnum) if proto_rows[i] == 0],
             "max_abs_change_in_numeric_min": float(np.abs(lo - lmin0).max()), "max_abs_change_in_numeric_max": float(np.abs(hi - lmax0).max())}
    print("scaler refit:", json.dumps(refit), flush=True)

    def rescale(X):
        out = np.array(X, dtype=np.float32)
        out[:, :nnum] = (((np.asarray(X[:, :nnum], dtype=np.float64) * span0 + lmin0) - lo) / span1).astype(np.float32)
        return out

    # ---- data views
    rng = np.random.default_rng(args.seed)
    yva_all = mva.label_binary.to_numpy()
    keep_va = np.flatnonzero(~np.isin(mva.attack_type.to_numpy(), held))
    if args.eval_rows:
        keep_va = np.sort(rng.choice(keep_va, min(args.eval_rows, len(keep_va)), replace=False))
    it = np.arange(len(mte))
    if args.eval_rows:
        it = np.sort(rng.choice(len(mte), min(args.eval_rows, len(mte)), replace=False))
    Xva_f, yva = rescale(Xva[keep_va]), yva_all[keep_va]
    Xte_e = rescale(Xte[it])
    yte, tte, blk = mte.label_binary.to_numpy()[it], mte.attack_type.to_numpy()[it], mte.block.to_numpy()[it]
    unseen = np.isin(tte, held)
    benign = tte == "Benign"
    known_att = (yte == 1) & ~unseen
    es_idx = np.sort(rng.choice(len(Xva_f), min(args.es_val_rows, len(Xva_f)), replace=False))
    Xes, yes = Xva_f[es_idx], yva[es_idx]
    ytr_all = mtr.label_binary.to_numpy()

    def sample(pool):
        n = min(args.train_rows, len(pool))
        idx = np.sort(np.random.default_rng(args.seed).choice(pool, n, replace=False)) if n < len(pool) else pool
        return rescale(np.asarray(Xtr[idx])), ytr_all[idx]

    results = {"config": vars(args) | {"out_dir": str(args.out_dir), "model_dir": str(args.model_dir)}, "withheld": held, "refit": refit,
               "test_rows": int(len(yte)), "test_counts": {t: int((tte == t).sum()) for t in np.unique(tte)},
               "val_rows_after_filter": int(len(keep_va)), "models": {}}
    flags = {}
    for v in args.models.split(","):
        print(f"\n=== {v} (withheld: {', '.join(held)}) ===", flush=True)
        torch.manual_seed(args.seed)
        if v == "unsup_mixed":
            Xt, yt = sample(keep_tr)
            model = Autoencoder(d)
            model, hist, t_train = fit_autoencoder(model, Xt, Xes, epochs=args.epochs, batch=args.batch, lr=args.lr, patience=args.patience, seed=args.seed)
            s_val, s_te = recon_error(model, Xva_f), recon_error(model, Xte_e)
            thr = {"primary": float(np.percentile(s_val, args.percentile))}
            rule = f"{args.percentile:g}th percentile of filtered validation errors (no labels)"
        elif v == "normal_only":
            Xt, yt = sample(keep_tr[ytr_all[keep_tr] == 0])
            model = Autoencoder(d)
            model, hist, t_train = fit_autoencoder(model, Xt, Xes, epochs=args.epochs, batch=args.batch, lr=args.lr, patience=args.patience, seed=args.seed)
            s_val, s_te = recon_error(model, Xva_f), recon_error(model, Xte_e)
            thr = {"primary": float(np.percentile(s_val[yva == 0], args.benign_percentile))}
            rule = f"{args.benign_percentile:g}th percentile of benign filtered validation errors (uses benign val labels)"
        elif v == "supervised":
            Xt, yt = sample(keep_tr)
            model = SupervisedAE(d)
            model, hist, t_train = fit_supervised(model, Xt, yt, Xes, yes, epochs=args.epochs, batch=args.batch, lr=args.lr,
                                                  patience=args.patience, seed=args.seed)
            s_val, s_te = predict_proba(model, Xva_f), predict_proba(model, Xte_e)
            thr = {"primary": 0.5}
            rule = "probability 0.5"
        else:
            raise SystemExit(f"unknown model {v}")
        thr["val_f1_tuned"] = best_f1_threshold(yva, s_val)
        torch.save(model.state_dict(), args.model_dir / f"ae_unseen_{v}.pt")
        np.save(args.out_dir / f"scores_unseen_{v}_test.npy", s_te)
        out = {"threshold_rule": rule, "thresholds": thr, "train_rows_used": int(len(Xt)), "train_attack_share": float(yt.mean()),
               "epochs_run": len(hist["train"]), "train_seconds": t_train, "best_val_loss": float(min(hist["val"])), "history": hist,
               "known": {}, "unseen": {}}
        for tag, th in thr.items():
            m_known = benign | known_att                       # benign + known attacks
            km = threshold_metrics(yte[m_known], s_te[m_known], th)
            m_unseen = benign | unseen                         # benign + withheld attacks
            um = threshold_metrics(yte[m_unseen], s_te[m_unseen], th)
            per = {}
            for h in held:
                mh = benign | (tte == h)
                per[h] = {"n": int((tte == h).sum()), "flagged_share": float((s_te[tte == h] > th).mean()),
                          "roc_auc_vs_benign": ranking_metrics((tte[mh] == h).astype(int), s_te[mh])["roc_auc"]}
            out["known"][tag] = km | {"n_known_attacks": int(known_att.sum()), "per_type_flagged": {
                t: float((s_te[tte == t] > th).mean()) for t in sorted(set(tte) - {"Benign"} - set(held))}}
            out["unseen"][tag] = {"recall": um["recall"], "n_unseen": int(unseen.sum()), "fpr": um["fpr"], "per_type": per}
            if tag == "primary":
                groups = {"benign": benign, "known_attacks": known_att, "unseen_all": unseen, **{h: tte == h for h in held}}
                out["ci_flagged_share_primary"] = block_ci(s_te > th, groups, blk, args.boot, args.seed)
        out["known_ranking"] = ranking_metrics(yte[benign | known_att], s_te[benign | known_att])
        out["unseen_ranking"] = ranking_metrics(yte[benign | unseen], s_te[benign | unseen])
        out["unseen_base_rate"] = float(unseen.sum() / (benign.sum() + unseen.sum()))
        f_, t_, _ = roc_curve(yte[benign | unseen], s_te[benign | unseen])
        idx = np.unique(np.linspace(0, len(f_) - 1, 3000).astype(int))
        out["_roc_unseen"] = [f_[idx].tolist(), t_[idx].tolist()]
        results["models"][v] = out
        flags[v] = out
        p, u = out["known"]["primary"], out["unseen"]["primary"]
        print(f"  KNOWN  thr {thr['primary']:.6g}: acc {p['accuracy']:.4f} prec {p['precision']:.4f} rec {p['recall']:.4f} f1 {p['f1']:.4f} fpr {p['fpr']:.4f} | "
              f"AUC {out['known_ranking']['roc_auc']:.4f} PR {out['known_ranking']['pr_auc']:.4f}", flush=True)
        print(f"  UNSEEN recall {u['recall']:.4f} | AUC {out['unseen_ranking']['roc_auc']:.4f} PR {out['unseen_ranking']['pr_auc']:.4f} | per type " +
              ", ".join(f"{h} {x['flagged_share']:.3f}" for h, x in u["per_type"].items()), flush=True)

    json.dump(results, open(args.out_dir / "phase6_results.json", "w"), indent=2)
    plot_unseen(results, held, args.out_dir / "phase6_unseen_detection.png")
    print("\nPhase 6 done")


if __name__ == "__main__":
    main()
