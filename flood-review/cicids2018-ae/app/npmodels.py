"""Torch-free inference for the saved autoencoders (numpy only).

Why this exists: on some Windows machines Smart App Control blocks PyTorch's unsigned DLLs (WinError 4551), so
`import torch` fails although the models are tiny feed-forward networks. Serving them needs only matrix products, so the
dashboard reads the weights straight out of the `.pt` files and runs the forward pass in numpy. Nothing is trained here.

A `.pt` file written by `torch.save(state_dict)` is a zip archive holding a pickle (`data.pkl`) and one raw binary file per
tensor. The pickle is read with a restricted unpickler that accepts only the few names a plain state dict uses; it never
imports torch and refuses everything else, so a tampered checkpoint cannot run code.
"""
from __future__ import annotations

import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    size, stride = tuple(size), tuple(stride)
    n = int(np.prod(size)) if size else 1
    flat = storage[offset:offset + n]
    expected = tuple(int(np.prod(size[i + 1:])) for i in range(len(size)))
    if stride != expected:
        raise ValueError(f"non-contiguous tensor (stride {stride}) is not supported")
    return flat.reshape(size)


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, archive: zipfile.ZipFile, prefix: str):
        super().__init__(fh)
        self._zip, self._prefix = archive, prefix

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refused to load {module}.{name} from a model file")

    def persistent_load(self, pid):
        kind, stype, key, _location, _numel = pid
        if kind != "storage" or not isinstance(stype, _StorageType):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid[:2]}")
        raw = self._zip.read(f"{self._prefix}/data/{key}")
        return np.frombuffer(raw, dtype=_DTYPES[stype.name])


def load_state_dict(path) -> "OrderedDict[str, np.ndarray]":
    """Read a torch state dict (.pt) into numpy arrays without torch."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl = next(n for n in names if n.endswith("/data.pkl") or n == "data.pkl")
        prefix = pkl.rsplit("/", 1)[0] if "/" in pkl else ""
        order = z.read(f"{prefix}/byteorder").decode() if f"{prefix}/byteorder" in names else "little"
        if order != "little":
            raise ValueError("big-endian checkpoint is not supported")
        with z.open(pkl) as fh:
            sd = _Unpickler(fh, z, prefix).load()
    if not isinstance(sd, dict):
        raise ValueError("model file does not hold a state dict")
    return OrderedDict((k, np.array(v, dtype=np.float32)) for k, v in sd.items())


def _sigmoid(x):
    with np.errstate(over="ignore"):
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _mlp(layers, h, final_sigmoid=False):
    """ReLU after every Linear except the last (the layout of src/models.py: mlp())."""
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return _sigmoid(h) if final_sigmoid else h


def _linears(sd, prefix):
    """(weight, bias) pairs of the Linear layers under `prefix`, in layer order."""
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith(prefix + ".") and k.endswith(".weight")})
    return [(sd[f"{prefix}.{i}.weight"], sd[f"{prefix}.{i}.bias"]) for i in idx]


class NumpyAutoencoder:
    """Input -> 128 -> 64 -> 32 -> 64 -> 128 -> input, sigmoid output. Same layout as models.Autoencoder."""
    kind = "reconstruction"

    def __init__(self, sd):
        self.enc, self.dec = _linears(sd, "encoder"), _linears(sd, "decoder")

    def __call__(self, x):
        return _mlp(self.dec, _mlp(self.enc, x), final_sigmoid=True)


class NumpySupervisedAE:
    """Encoder -> ReLU -> 16 -> ReLU -> 1 logit; optional decoder. Same layout as models.SupervisedAE.
    __call__ returns (logit, reconstruction or None), like the torch module."""
    kind = "classifier"

    def __init__(self, sd):
        self.enc, self.head = _linears(sd, "encoder"), _linears(sd, "head")
        self.dec = _linears(sd, "decoder") if any(k.startswith("decoder.") for k in sd) else None

    def __call__(self, x):
        z = _mlp(self.enc, x)
        h = np.maximum(z, 0.0)                       # head = ReLU, Linear, ReLU, Linear
        (w1, b1), (w2, b2) = self.head
        logit = (np.maximum(h @ w1.T + b1, 0.0) @ w2.T + b2)[:, 0]
        return logit, (_mlp(self.dec, z, final_sigmoid=True) if self.dec is not None else None)


def load_model(path, supervised: bool):
    sd = load_state_dict(Path(path))
    return NumpySupervisedAE(sd) if supervised else NumpyAutoencoder(sd)
