"""Warm-up baseline for the streaming experiments (Phases 7-9).

Initial baseline = one-class autoencoder trained on the BENIGN records of the first `warm_days` days of the stream
(first 80% by time for training, last 20% for the threshold). Labels are used only here, to build this clean warm-up
set (the same declared label use as the one-class variant of Experiment A); all later updates are label-free.
The evaluation stream is everything after the warm-up days.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adaptive import calibrate_k
from models import Autoencoder, recon_error
from stream import load_stream
from training import fit_autoencoder

CACHE = Path("data/interim")


def interval_ids(meta, records):
    """Update intervals = consecutive chunks of `records` records inside one day, in corrected-time order
    (some days cover a single hour, so a clock-time interval would be uneven). Ids are non-decreasing."""
    day_idx = pd.factorize(meta.day)[0].astype(np.int64)
    pos = meta.groupby("day").cumcount().to_numpy().astype(np.int64)
    return day_idx * 100_000 + pos // records


def get_baseline(warm_days=2, init_train_rows=500_000, init_epochs=50, seed=42, tag="main", force=False, log=print):
    X, m = load_stream()
    days = sorted(m.day.unique())
    warm = np.isin(m.day.to_numpy(), days[:warm_days])
    e0 = int(np.flatnonzero(~warm)[0])
    assert warm[:e0].all() and not warm[e0:].any(), "warm-up days must be the first days of the stream"
    y = m.label_binary.to_numpy()
    ben = np.flatnonzero(warm & (y == 0))
    n80 = int(0.8 * len(ben))
    rng = np.random.default_rng(seed)
    tr = np.sort(rng.choice(ben[:n80], min(init_train_rows, n80), replace=False))
    va = ben[n80:]
    paths = {k: CACHE / f"stream_{tag}_{k}" for k in ("model.pt", "info.json", "anchor.npy", "replay.npy")}
    if all(p.exists() for p in paths.values()) and not force:
        info = json.load(open(paths["info.json"]))
        model = Autoencoder(X.shape[1])
        model.load_state_dict(torch.load(paths["model.pt"]))
        model.eval()
        return dict(X=X, meta=m, e0=e0, model=model, thr0=info["thr0"], k=info["k"], anchor=np.load(paths["anchor.npy"]),
                    replay=np.load(paths["replay.npy"]), info=info, days=days)
    Xt, Xv = np.asarray(X[tr]), np.asarray(X[va])
    torch.manual_seed(seed)
    model = Autoencoder(X.shape[1])
    es = np.sort(rng.choice(len(Xv), min(100_000, len(Xv)), replace=False))
    model, hist, secs = fit_autoencoder(model, Xt, Xv[es], epochs=init_epochs, batch=1024, lr=1e-3, patience=5, seed=seed, log=log)
    sv = recon_error(model, Xv)
    thr0 = float(np.percentile(sv, 99))
    k = calibrate_k(sv, thr0)
    anchor = Xv[np.sort(rng.choice(len(Xv), min(50_000, len(Xv)), replace=False))]
    replay = np.asarray(X[np.sort(rng.choice(ben[:n80], min(60_000, n80), replace=False))])
    info = {"warm_days": days[:warm_days], "eval_days": days[warm_days:], "warm_records": int(e0), "warm_benign": int(len(ben)),
            "init_train_rows": int(len(tr)), "init_val_rows": int(len(va)), "epochs": len(hist["train"]), "train_seconds": secs,
            "thr0": thr0, "k": k, "val_median_error": float(np.median(sv)), "eval_records": int(len(m) - e0)}
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), paths["model.pt"])
    json.dump(info, open(paths["info.json"], "w"), indent=2)
    np.save(paths["anchor.npy"], anchor)
    np.save(paths["replay.npy"], replay)
    model.eval()
    return dict(X=X, meta=m, e0=e0, model=model, thr0=thr0, k=k, anchor=anchor, replay=replay, info=info, days=days)
