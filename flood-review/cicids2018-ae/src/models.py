"""Autoencoder models (PyTorch)."""
import numpy as np
import torch
from torch import nn


def mlp(dims, last_activation=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    if last_activation is not None:
        layers.append(last_activation)
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Input -> 128 -> 64 -> 32 (latent) -> 64 -> 128 -> Input. ReLU hidden layers, sigmoid output
    (inputs are min-max scaled to [0, 1] using the training partition)."""

    def __init__(self, in_dim: int, hidden=(128, 64), latent: int = 32):
        super().__init__()
        self.encoder = mlp([in_dim, *hidden, latent])
        self.decoder = mlp([latent, *reversed(hidden), in_dim], last_activation=nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SupervisedAE(nn.Module):
    """Input -> encoder (128 -> 64 -> 32 latent) -> classification head -> 1 logit (attack).
    With decoder=True the same latent also feeds a decoder (32 -> 64 -> 128 -> input, sigmoid) so training can
    combine classification and reconstruction losses. forward returns (logit, reconstruction or None)."""

    def __init__(self, in_dim: int, hidden=(128, 64), latent: int = 32, head: int = 16, decoder: bool = False):
        super().__init__()
        self.encoder = mlp([in_dim, *hidden, latent])
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(latent, head), nn.ReLU(), nn.Linear(head, 1))
        self.decoder = mlp([latent, *reversed(hidden), in_dim], last_activation=nn.Sigmoid()) if decoder else None

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).squeeze(1), (self.decoder(z) if self.decoder is not None else None)


@torch.no_grad()
def predict_proba(model, X, batch: int = 65536):
    """Attack probability (sigmoid of the logit) for a SupervisedAE. numpy (n,) float32."""
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(np.array(X[i:i + batch], dtype=np.float32)))[0]).numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


@torch.no_grad()
def recon_error(model, X, batch: int = 65536):
    """Per-row mean squared reconstruction error. X: numpy array / memmap (n, d) -> numpy float32 (n,)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.array(X[i:i + batch], dtype=np.float32))
        out.append(((model(xb) - xb) ** 2).mean(dim=1).numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)
