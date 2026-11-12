"""Replay source: the held-out test partition in time order, scored by the saved models as if it were arriving live.

The records are real CSE-CIC-IDS2018 flows that none of the models saw in training. The true label and attack type are attached to
each tick ONLY so the dashboard can show how the alerts compare with the truth; nothing here (or in the service) feeds labels back
into a model, a threshold or an alert decision.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .service import ROOT, Detector

_lock = threading.Lock()
_state: dict = {}


def _load(root: Path = ROOT):
    with _lock:
        if _state:
            return _state
        proc = root / "data/processed"
        meta = pd.read_pickle(proc / "meta_test.pkl")
        ts = meta["timestamp"].to_numpy()
        order = np.argsort(ts, kind="stable")
        _state.update(X=np.load(proc / "X_test.npy", mmap_mode="r"), ts=ts, order=order,
                      y=meta["label_binary"].to_numpy(), atype=meta["attack_type"].to_numpy(), port=meta["dst_port"].to_numpy())
        day = pd.to_datetime(pd.Series(ts[order])).dt.strftime("%Y-%m-%d").to_numpy()
        _state["day"] = day
        return _state


def available(root: Path = ROOT) -> bool:
    return (root / "data/processed/X_test.npy").exists() and (root / "data/processed/meta_test.pkl").exists()


def days(root: Path = ROOT) -> list[dict]:
    s = _load(root)
    y = s["y"][s["order"]]
    out, start = [], 0
    day = s["day"]
    for i in range(1, len(day) + 1):
        if i == len(day) or day[i] != day[start]:
            out.append({"day": str(day[start]), "start": start, "records": i - start, "attack_share": float(y[start:i].mean())})
            start = i
    return out


def ticks(det: Detector, start: int, batch: int, max_ticks: int, root: Path = ROOT):
    """Yield one dict per batch of `batch` consecutive records, beginning at position `start` of the time-ordered test set."""
    s = _load(root)
    n_total = len(s["order"])
    pos = max(0, min(int(start), n_total - 1))
    for k in range(max_ticks):
        if pos >= n_total:
            break
        sel = s["order"][pos:pos + batch]
        idx = np.sort(sel)                                  # memmap reads are faster in file order; order inside a batch is irrelevant
        out = det.score(np.asarray(s["X"][idx]))
        y = s["y"][idx].astype(bool)
        flagged_any = np.zeros(len(idx), dtype=bool)
        per = {}
        for m, a in out["alerts"].items():
            per[m] = {"flagged": int(a.sum()), "tp": int((a & y).sum()), "fp": int((a & ~y).sum()),
                      "fn": int((~a & y).sum()), "tn": int((~a & ~y).sum())}
            flagged_any |= a
        order_by_score = np.flatnonzero(flagged_any)
        order_by_score = order_by_score[np.argsort(-out["scores"]["supervised"][order_by_score])][:6]
        sample = [{"time": str(s["ts"][idx[j]])[:19].replace("T", " "), "dst_port": int(s["port"][idx[j]]),
                   "scores": {m: float(out["scores"][m][j]) for m in out["scores"]},
                   "alerts": [m for m, a in out["alerts"].items() if a[j]],
                   "truth": str(s["atype"][idx[j]])} for j in order_by_score]
        yield {"pos": pos, "n": int(len(idx)), "time": str(s["ts"][idx[-1]])[:19].replace("T", " "), "true_attacks": int(y.sum()), "models": per,
               "mean_log10_unsup_error": float(np.log10(np.maximum(out["scores"]["unsup_mixed"], 1e-12)).mean()), "sample": sample}
        pos += batch
