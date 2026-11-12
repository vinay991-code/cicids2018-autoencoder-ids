"""Reconstruction training loop with early stopping, reused by the later experiments."""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import recon_error


def _sup_loss(model, xb, yb, pos_weight, recon_weight, reduction="mean"):
    """Class-weighted BCE (+ recon_weight * per-row summed squared reconstruction error if a decoder exists)."""
    logit, rec = model(xb)
    loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight, reduction=reduction)
    if rec is not None and recon_weight > 0:
        sq = ((rec - xb) ** 2).sum(dim=1)
        loss = loss + recon_weight * (sq.sum() if reduction == "sum" else sq.mean())
    return loss


@torch.no_grad()
def _sup_eval(model, X, y, pos_weight, recon_weight, batch=65536):
    model.eval()
    tot = 0.0
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch])
        yb = torch.from_numpy(y[i:i + batch])
        tot += _sup_loss(model, xb, yb, pos_weight, recon_weight, reduction="sum").item()
    return tot / len(X)


def fit_supervised(model, Xtr, ytr, Xval, yval, *, epochs=50, batch=1024, lr=1e-3, patience=5, min_delta=1e-6,
                   seed=42, recon_weight=0.0, log=print):
    """Adam + class-weighted BCE (pos_weight = n_benign / n_attack of the training rows) with optional
    reconstruction term. Early stopping on the same loss over (Xval, yval). Returns (model, history, seconds)."""
    torch.manual_seed(seed)
    ytr, yval = ytr.astype(np.float32), yval.astype(np.float32)
    pos_weight = torch.tensor(float((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    n = len(Xt)
    hist = {"train": [], "val": [], "pos_weight": float(pos_weight)}
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = _sup_loss(model, Xt[idx], yt[idx], pos_weight, recon_weight)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        val = _sup_eval(model, Xval, yval, pos_weight, recon_weight)
        hist["train"].append(tot / n)
        hist["val"].append(val)
        log(f"  epoch {ep+1:3d}  train {tot/n:.6f}  val {val:.6f}  [{time.time()-t0:5.0f}s]")
        if val < best - min_delta:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop at epoch {ep+1} (best val {best:.6f})")
                break
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0


def fit_autoencoder(model, Xtr, Xval, *, epochs=50, batch=1024, lr=1e-3, patience=5, min_delta=1e-6, seed=42, log=print):
    """Train with Adam + MSE. Xtr, Xval: float32 numpy arrays. Early stopping monitors validation MSE
    (computed on Xval only, no labels). Returns (best model, history dict, train_seconds)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(Xtr)
    n = len(Xt)
    hist = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            xb = Xt[perm[i:i + batch]]
            opt.zero_grad()
            loss = ((model(xb) - xb) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        val = float(recon_error(model, Xval).mean())
        hist["train"].append(tot / n)
        hist["val"].append(val)
        log(f"  epoch {ep+1:3d}  train {tot/n:.6f}  val {val:.6f}  [{time.time()-t0:5.0f}s]")
        if val < best - min_delta:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop at epoch {ep+1} (best val {best:.6f})")
                break
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0
