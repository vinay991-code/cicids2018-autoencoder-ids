"""Time-ordered view of the whole de-duplicated dataset (train + val + test partitions merged), for the
streaming experiments (adaptive baseline, confidence-aware updates, concept drift).

build_stream() writes data/processed/X_stream.npy (float32, same 72 scaled columns as the partitions) and
meta_stream.pkl, both sorted by timestamp. Every record appears once; the Phase 2 partition tag is kept in `split`.
Labels in the metadata are for evaluation only; experiment control logic must not read them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")


def build_stream(proc: Path = PROC, chunk: int = 500_000):
    metas = []
    for k, s in enumerate(("train", "val", "test")):
        m = pd.read_pickle(proc / f"meta_{s}.pkl")
        m["split"], m["row"] = np.int8(k), np.arange(len(m))
        metas.append(m[["timestamp", "attack_type", "label_binary", "split", "row", "source_file"]])
    m = pd.concat(metas, ignore_index=True)
    # The CSV timestamps are on a 12-hour clock whose AM/PM marker was lost: afternoon events appear 12 hours early
    # (checked against the official attack schedule: Hulk 13:45 shows as 01:45, SSH 14:01 as 02:01, ...). Hours 1-7 are
    # therefore afternoon (+12 h); hours 8-12 are morning/noon as stored.
    hour = m.timestamp.dt.hour.to_numpy()
    m["time_true"] = m.timestamp + pd.to_timedelta(np.where((hour >= 1) & (hour <= 7), 12, 0), unit="h")
    order = np.argsort(m.time_true.to_numpy(), kind="stable")
    m = m.iloc[order].reset_index(drop=True)
    X = [np.load(proc / f"X_{s}.npy", mmap_mode="r") for s in ("train", "val", "test")]
    out = np.lib.format.open_memmap(proc / "X_stream.npy", mode="w+", dtype=np.float32, shape=(len(m), X[0].shape[1]))
    sp, rw = m.split.to_numpy(), m.row.to_numpy()
    for a in range(0, len(m), chunk):
        s_, r_ = sp[a:a + chunk], rw[a:a + chunk]
        buf = np.empty((len(s_), X[0].shape[1]), dtype=np.float32)
        for k in range(3):
            mk = s_ == k
            if mk.any():
                idx = r_[mk]
                o = np.argsort(idx)
                tmp = np.asarray(X[k][idx[o]])
                part = np.empty_like(tmp)
                part[o] = tmp
                buf[mk] = part
        out[a:a + len(s_)] = buf
    out.flush()
    m["day"] = m.timestamp.dt.strftime("%Y-%m-%d")
    m.to_pickle(proc / "meta_stream.pkl")
    return m


def load_stream(proc: Path = PROC):
    return np.load(proc / "X_stream.npy", mmap_mode="r"), pd.read_pickle(proc / "meta_stream.pkl")


if __name__ == "__main__":
    m = build_stream()
    print(f"stream built: {len(m):,} records, {m.day.nunique()} days, "
          f"{m.time_true.min()} -> {m.time_true.max()}; sorted by corrected time: {m.time_true.is_monotonic_increasing}")
