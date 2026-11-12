"""Trimmed-loss autoencoder training (a known idea: ignore the highest-error records of each batch so that attacks hidden in
unlabeled training data are not learned; compare Shen and Sanghavi 2019 'iterative trimmed loss minimization' and the sample
selection of RCA, Liu et al. IJCAI 2021). No labels are read anywhere in this file."""
import copy
import time

import numpy as np
import torch

from models import recon_error


def trimmed_mean(err, q):
    """Mean of the lowest (1 - q) share of the errors (numpy)."""
    k = max(1, int(round((1.0 - q) * len(err))))
    return float(np.sort(err)[:k].mean())


def fit_trimmed_autoencoder(model, Xtr, Xval, q, *, ramp=10, epochs=50, batch=1024, lr=1e-3, patience=5, min_delta=1e-6, seed=42, log=print):
    """Adam on the per-batch trimmed mean squared error.

    keep share in epoch e (0-based) = 1 - q * min(1, e / ramp): all records at first, then linearly down to 1 - q after `ramp` epochs
    (so the network can first learn the bulk of the data). Early stopping monitors the trimmed (lowest 1 - q share) validation error
    of the UNLABELED rows Xval, which is stationary over epochs. Returns (best model, history, seconds)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(Xtr)
    n = len(Xt)
    hist = {"train": [], "val": [], "keep": []}
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        keep = 1.0 - q * min(1.0, ep / ramp)
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch):
            xb = Xt[perm[i:i + batch]]
            opt.zero_grad()
            err = ((model(xb) - xb) ** 2).mean(dim=1)
            k = max(1, int(round(keep * len(err))))
            loss = torch.topk(err, k, largest=False).values.mean()
            loss.backward()
            opt.step()
            tot += loss.item() * k
            cnt += k
        val = trimmed_mean(recon_error(model, Xval), q)
        hist["train"].append(tot / cnt)
        hist["val"].append(val)
        hist["keep"].append(keep)
        log(f"  epoch {ep+1:3d}  keep {keep:.2f}  train {tot/cnt:.6f}  val(trimmed) {val:.6f}  [{time.time()-t0:5.0f}s]")
        if val < best - min_delta:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop at epoch {ep+1} (best val {best:.6f})")
                break
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0
