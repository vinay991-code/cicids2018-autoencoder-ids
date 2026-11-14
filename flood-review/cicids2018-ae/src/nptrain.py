"""NumPy re-implementation of the autoencoder forward pass, backward pass and Adam fine-tuning used in the streaming
experiments (needed because PyTorch cannot be loaded on this machine: Windows Smart App Control blocks its DLLs).

It mirrors src/models.py and src/adaptive.py::finetune:
  Autoencoder = encoder mlp([d,128,64,32]) + decoder mlp([32,64,128,d], Sigmoid); every Linear except the last of the
  encoder and the last of the decoder is followed by ReLU; the decoder ends with a sigmoid.
  loss      ((model(x) - x) ** 2).mean()   (mean over the batch AND the features)
  optimiser torch.optim.Adam(lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) re-created for every fine-tuning call
            (state starts at zero), step formula of torch: denom = sqrt(v)/sqrt(1-b2^t) + eps; p -= lr/(1-b1^t) * m/denom
  batches   consecutive slices of a fresh random permutation per epoch (the last partial batch is kept)
Not identical to torch: the sequence of random permutations (numpy generator instead of torch.Generator) and float32
summation order. Validation against the saved PyTorch models and gradient checks: tests/test_nptrain.py and
src/phase17_drift_recovery.py (checks block).
"""
import numpy as np

ACTS = ("relu", "relu", "none", "relu", "relu", "sigmoid")      # activation after each of the 6 Linear layers


def from_state_dict(sd, dtype=np.float32):
    """Layers [(W, b)] of the Autoencoder in forward order from a state dict (keys encoder.{0,2,4}.*, decoder.{0,2,4}.*)."""
    out = []
    for prefix in ("encoder", "decoder"):
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith(prefix + ".") and k.endswith(".weight")})
        out += [(np.array(sd[f"{prefix}.{i}.weight"], dtype=dtype), np.array(sd[f"{prefix}.{i}.bias"], dtype=dtype)) for i in idx]
    assert len(out) == len(ACTS)
    return out


def copy_params(params):
    return [(w.copy(), b.copy()) for w, b in params]


def forward(params, x, cache=False):
    """Returns the reconstruction (and, if cache, the layer inputs and pre-activations needed by backward)."""
    h, ins, pre = x, [], []
    for (w, b), act in zip(params, ACTS):
        ins.append(h)
        z = h @ w.T + b
        pre.append(z)
        if act == "relu":
            h = np.maximum(z, 0)
        elif act == "sigmoid":
            with np.errstate(over="ignore"):
                h = (1.0 / (1.0 + np.exp(-z))).astype(z.dtype)
        else:
            h = z
    return (h, ins, pre) if cache else h


def loss_and_grads(params, x):
    """MSE loss ((out - x)**2).mean() and its gradients [(dW, db)] by manual backpropagation."""
    out, ins, pre = forward(params, x, cache=True)
    n = out.size
    diff = out - x
    loss = float((diff.astype(np.float64) ** 2).sum() / n)
    g = (2.0 / n) * diff                                     # d loss / d out
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        act = ACTS[i]
        if act == "sigmoid":
            s = out if i == len(params) - 1 else None
            g = g * s * (1.0 - s)
        elif act == "relu":
            g = g * (pre[i] > 0)
        w = params[i][0]
        grads[i] = (g.T @ ins[i], g.sum(axis=0))
        if i > 0:
            g = g @ w
    return loss, grads


class Adam:
    """torch.optim.Adam without weight decay / amsgrad."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, betas[0], betas[1], eps, 0
        self.m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
        self.v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]

    def step(self, params, grads):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            for j in range(2):
                m, v = self.m[i][j], self.v[i][j]
                m *= self.b1; m += (1 - self.b1) * g[j]
                v *= self.b2; v += (1 - self.b2) * g[j] * g[j]
                denom = np.sqrt(v) / np.sqrt(bc2) + self.eps
                p[j][...] -= (self.lr / bc1) * m / denom


def finetune(params, X, epochs, lr, batch, seed):
    """Continue training a COPY of `params` on X (float32, (n, d)) with Adam + MSE; returns (new params, loss before, loss after).
    Loss before/after: mean reconstruction MSE on X with the old / new parameters (diagnostic)."""
    p = copy_params(params)
    opt = Adam(p, lr=lr)
    rng = np.random.default_rng(seed)
    before = float(((forward(params, X) - X) ** 2).mean())
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            xb = X[perm[i:i + batch]]
            _, grads = loss_and_grads(p, xb)
            opt.step(p, grads)
    after = float(((forward(p, X) - X) ** 2).mean())
    return p, before, after


def recon_error(params, X, chunk=131_072):
    """Per-row mean squared reconstruction error (float32), same definition as models.recon_error."""
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), chunk):
        xb = np.ascontiguousarray(X[i:i + chunk], dtype=np.float32)
        out[i:i + chunk] = ((forward(params, xb) - xb) ** 2).mean(axis=1)
    return out
