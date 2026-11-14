"""Checks of the NumPy autoencoder trainer (src/nptrain.py) against independent references.
Tolerances (justified): gradients vs central finite differences in float64 with step 1e-6: |numeric - analytic| <= 1e-9 + 1e-6 *
max(|numeric|, |analytic|). The absolute part covers the rounding noise of the finite difference itself (loss ~1e-1 with
eps 1e-6 gives about 1e-11 to 1e-10; observed maximum 1.4e-11 over 346 parameters), which makes a purely relative test fail
on parameters whose true gradient is ~1e-7 although numeric and analytic values agree to 5 digits; Adam vs the transcription of the torch.optim.Adam algorithm and against
scikit-learn's independent Adam, absolute difference < 1e-9 / relative 1e-6 (they differ only in where eps enters);
forward pass vs saved PyTorch scores < 1e-6 absolute on the reconstruction error (float32 summation order)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import nptrain as nt  # noqa: E402


def _spec(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def random_params(rng, dims=(10, 8, 6, 4, 6, 8, 10), dtype=np.float64):
    return [(rng.normal(0, 0.5, (dims[i + 1], dims[i])).astype(dtype), rng.normal(0, 0.1, dims[i + 1]).astype(dtype)) for i in range(6)]


def grad_check(params, x, n_probe=30, eps=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    _, grads = nt.loss_and_grads(params, x)
    worst = 0.0
    for i in range(len(params)):
        for j in range(2):
            p = params[i][j]
            flat_idx = rng.choice(p.size, min(n_probe, p.size), replace=False)
            for k in flat_idx:
                idx = np.unravel_index(k, p.shape)
                old = p[idx]
                p[idx] = old + eps; lp, _ = nt.loss_and_grads(params, x)
                p[idx] = old - eps; lm, _ = nt.loss_and_grads(params, x)
                p[idx] = old
                num, ana = (lp - lm) / (2 * eps), grads[i][j][idx]
                worst = max(worst, abs(num - ana) / (1e-9 + 1e-6 * max(abs(num), abs(ana))))
    return worst           # < 1 means every probed parameter is within the tolerance


def test_gradients_random_network():
    rng = np.random.default_rng(1)
    params, x = random_params(rng), rng.uniform(0, 1, (16, 10))
    assert grad_check(params, x) < 1


def test_gradients_real_weights_and_data():
    p = ROOT / "data" / "interim" / "stream_main_model.pt"
    xs = ROOT / "data" / "processed" / "X_stream.npy"
    if not (p.exists() and xs.exists()):
        pytest.skip("stream data or baseline model not present")
    npm = _spec("npmodels", ROOT / "app" / "npmodels.py")
    params = nt.from_state_dict(npm.load_state_dict(p), dtype=np.float64)
    x = np.asarray(np.load(xs, mmap_mode="r")[2_000_000:2_000_012], dtype=np.float64)
    assert grad_check(params, x, n_probe=15) < 1


def test_adam_matches_torch_formula_and_sklearn():
    rng = np.random.default_rng(2)
    params = [(rng.normal(size=(3, 2)), rng.normal(size=3))]
    ref = [(w.copy(), b.copy()) for w, b in params]
    m = [np.zeros_like(a) for pair in params for a in pair]
    v = [np.zeros_like(a) for pair in params for a in pair]
    lr, b1, b2, eps = 5e-4, 0.9, 0.999, 1e-8
    opt = nt.Adam(params, lr=lr)
    for t in range(1, 6):
        grads = [(rng.normal(size=(3, 2)), rng.normal(size=3))]
        opt.step(params, grads)
        flat_ref, flat_g = [a for pair in ref for a in pair], [a for pair in grads for a in pair]
        for k in range(2):           # literal transcription of torch.optim.Adam (single-tensor path)
            m[k] = b1 * m[k] + (1 - b1) * flat_g[k]
            v[k] = b2 * v[k] + (1 - b2) * flat_g[k] ** 2
            denom = np.sqrt(v[k]) / np.sqrt(1 - b2 ** t) + eps
            flat_ref[k] -= (lr / (1 - b1 ** t)) * m[k] / denom
        for a, b in zip([a for pair in params for a in pair], flat_ref):
            assert np.max(np.abs(a - b)) < 1e-12
    try:
        from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
    except Exception:
        pytest.skip("scikit-learn Adam not importable")
    rng = np.random.default_rng(3)
    p0 = [rng.normal(size=(3, 2)), rng.normal(size=3)]
    mine = [(p0[0].copy(), p0[1].copy())]
    sk = [p0[0].copy(), p0[1].copy()]
    skopt = AdamOptimizer(sk, learning_rate_init=lr, beta_1=b1, beta_2=b2, epsilon=eps)
    opt = nt.Adam(mine, lr=lr)
    for _ in range(5):
        g = [rng.normal(size=(3, 2)), rng.normal(size=3)]
        opt.step(mine, [(g[0], g[1])])
        skopt.update_params(sk, [a.copy() for a in g])
    for a, b in zip(mine[0], sk):
        assert np.max(np.abs(a - b)) < 1e-9


def test_finetune_reduces_loss_and_is_deterministic():
    rng = np.random.default_rng(4)
    params = [(w.astype(np.float32), b.astype(np.float32)) for w, b in random_params(rng)]
    x = rng.uniform(0, 1, (512, 10)).astype(np.float32)
    p1, before, after = nt.finetune(params, x, epochs=5, lr=5e-3, batch=64, seed=7)
    p2, _, after2 = nt.finetune(params, x, epochs=5, lr=5e-3, batch=64, seed=7)
    assert after < before and after == after2
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(p1, p2))
    assert not np.array_equal(params[0][0], p1[0][0]), "the input parameters must not be modified in place... copy returned"


def test_forward_matches_saved_pytorch_scores():
    p = ROOT / "models" / "ae_unsup_mixed.pt"
    xt, st = ROOT / "data" / "processed" / "X_test.npy", ROOT / "results" / "scores_unsup_mixed_test.npy"
    if not (p.exists() and xt.exists() and st.exists()):
        pytest.skip("model, test data or saved scores not present")
    npm = _spec("npmodels", ROOT / "app" / "npmodels.py")
    params = nt.from_state_dict(npm.load_state_dict(p))
    x = np.load(xt, mmap_mode="r")[:100_000]
    ref = np.load(st, mmap_mode="r")[:100_000]
    assert np.max(np.abs(nt.recon_error(params, x) - ref)) < 1e-6
