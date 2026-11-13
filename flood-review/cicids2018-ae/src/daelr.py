"""Latent-regulated autoencoder (DAE-LR), re-implemented from the description in Nkashama et al., arXiv:2407.08838 (2024), Section 3.1.

Loss per record (Eq. 3 of that paper):  ||x - x_hat||^2 + lambda * ||z - c||^2
  z = encoder(x), x_hat = decoder(z), the centre c is a LEARNABLE parameter of the model, lambda > 0 is a fixed regularising factor.
  Both terms are squared L2 norms (sums over features), as in the paper's notation; the batch loss is the mean over records.
The anomaly score at test time is the reconstruction error only (the paper: "the same reconstruction error also serves as the anomaly scoring function").

What the paper text read for this project does NOT say, and was therefore fixed by this project before any result was seen (see phase14_daelr.py):
  the network (the same 72-128-64-32-64-128-72 autoencoder as every other model here), the optimiser and schedule (those of Phase 3),
  the initial value of c (zeros), and lambda (a grid, not tuned on labels). No labels are read anywhere in this file.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from models import Autoencoder, recon_error


class LatentRegulatedAE(nn.Module):
    def __init__(self, in_dim: int, latent: int = 32):
        super().__init__()
        self.net = Autoencoder(in_dim, latent=latent)
        self.c = nn.Parameter(torch.zeros(latent))                 # learnable centre of the latent codes

    def forward(self, x):                                           # the reconstruction, so recon_error() works unchanged
        return self.net(x)

    def terms(self, x):
        z = self.net.encoder(x)
        xh = self.net.decoder(z)
        return ((x - xh) ** 2).sum(dim=1), ((z - self.c) ** 2).sum(dim=1)


@torch.no_grad()
def mean_loss(model, X, lam, batch=65536):
    """Mean DAE-LR loss and its two terms over unlabeled rows X (numpy)."""
    model.eval()
    tot = np.zeros(2)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.array(X[i:i + batch], dtype=np.float32))
        r, l = model.terms(xb)
        tot += np.array([r.sum().item(), l.sum().item()])
    tot /= len(X)
    return float(tot[0] + lam * tot[1]), float(tot[0]), float(tot[1])


def fit_daelr(model, Xtr, Xval, lam, *, epochs=50, batch=1024, lr=1e-3, patience=5, min_delta=1e-6, seed=42, log=print):
    """Adam on the mean DAE-LR loss (parameters of the network AND the centre c). Early stopping monitors the same loss on the
    unlabeled early-stopping rows Xval. Returns (best model, history, seconds)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(Xtr)
    n = len(Xt)
    hist = {"train": [], "val": [], "val_recon": [], "val_latent": []}
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch):
            xb = Xt[perm[i:i + batch]]
            opt.zero_grad()
            r, l = model.terms(xb)
            loss = (r + lam * l).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
        val, vr, vl = mean_loss(model, Xval, lam)
        hist["train"].append(tot / cnt)
        hist["val"].append(val)
        hist["val_recon"].append(vr)
        hist["val_latent"].append(vl)
        log(f"  epoch {ep+1:3d}  train {tot/cnt:.6f}  val loss {val:.6f} (recon {vr:.6f}, latent {vl:.6f})  [{time.time()-t0:5.0f}s]")
        if val < best - min_delta:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop at epoch {ep+1} (best val {best:.6f})")
                break
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0
