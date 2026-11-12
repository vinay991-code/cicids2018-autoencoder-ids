"""Phase 2: clean, de-duplicate, split and scale CSE-CIC-IDS2018.

Stages (each caches its output, re-run with --force):
  1. clean   per-file streaming clean -> data/interim/<file>.f32 (+ .npz metadata)
  2. split   global de-duplication and block-wise scenario-aware split
  3. scale   fit scaler on TRAIN only, write data/processed/X_{train,val,test}.npy + meta

Split strategy (see reports/phase2_report.md): every capture day is cut into BLOCK_SEC-second time
blocks; whole blocks go to train / val / test (70 / 10 / 20 % of blocks), stratified by the set of
attack types a block contains so every attack type reaches every partition where it has enough blocks.
Rows of one block never straddle partitions. Labels are used ONLY to stratify blocks and are never
model inputs.

Usage:  python src/phase2_preprocess.py --data-dir data/raw --out-dir data
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_inspect import norm

CANON = {  # lower-cased raw label -> canonical attack type
    "benign": "Benign",
    "bot": "Bot",
    "ftp-bruteforce": "FTP-BruteForce",
    "ssh-bruteforce": "SSH-BruteForce",
    "dos attacks-goldeneye": "DoS-GoldenEye",
    "dos attacks-slowloris": "DoS-Slowloris",
    "dos attacks-hulk": "DoS-Hulk",
    "dos attacks-slowhttptest": "DoS-SlowHTTPTest",
    "ddos attacks-loic-http": "DDoS-LOIC-HTTP",
    "ddos attack-loic-udp": "DDoS-LOIC-UDP",
    "ddos attack-hoic": "DDoS-HOIC",
    "infilteration": "Infiltration",
    "brute force -web": "WebBruteForce",
    "brute force -xss": "WebXSS",
    "sql injection": "SQLInjection",
}
ATYPES = list(dict.fromkeys(CANON.values()))      # code 0 = Benign
ACODE = {k: ATYPES.index(v) for k, v in CANON.items()}
NO_NEG_CHECK = {"init_fwd_win_byts", "init_bwd_win_byts"}  # -1 is a sentinel there
SPLITS = ["train", "val", "test"]


# ----------------------------------------------------------------------------- stage 1
def common_features(files):
    heads = {f: pd.read_csv(f, nrows=0, encoding="latin-1").columns.tolist() for f in files}
    common = set.intersection(*[{norm(c) for c in h} for h in heads.values()])
    first = [norm(c) for c in heads[files[0]]]
    feats = [c for c in first if c in common and c not in ("timestamp", "label")]
    return heads, feats


def clean_file(path: Path, idx: int, head, feats, interim: Path, max_rows, chunk=500_000):
    rawmap = {c: norm(c) for c in head}
    keep_raw = [c for c in head if rawmap[c] in set(feats) | {"timestamp", "label"}]
    neg_cols = [i for i, c in enumerate(feats) if c not in NO_NEG_CHECK]
    hash_cols = [c for c in feats if c != "dst_port"]       # dedup key = model-input features
    dport_i = feats.index("dst_port")
    out_bin = interim / f"{path.stem}.f32"
    meta = {k: [] for k in ("ts", "y", "atype", "dst_port", "hash")}
    drops, n_in, n_out = Counter(), 0, 0
    hdr_rows = 0
    with open(out_bin, "wb") as fb:
        for ch in pd.read_csv(path, usecols=keep_raw, chunksize=min(chunk, max_rows or chunk),
                              encoding="latin-1", low_memory=False):
            ch.columns = [rawmap[c] for c in ch.columns]
            lab = ch["label"].astype(str).str.strip().str.lower()
            is_hdr = lab == "label"
            hdr_rows += int(is_hdr.sum())
            ch, lab = ch[~is_hdr], lab[~is_hdr]
            if max_rows and n_in >= max_rows:
                break
            for c in feats:
                if ch[c].dtype == object:
                    ch[c] = pd.to_numeric(ch[c], errors="coerce")
            unknown = set(lab.unique()) - set(ACODE)
            if unknown:
                raise SystemExit(f"Unknown labels in {path.name}: {unknown}")
            atype = lab.map(ACODE).to_numpy(dtype=np.int8)
            ts = pd.to_datetime(ch["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            X = ch[feats].to_numpy(dtype=np.float64)
            n_in += len(ch)
            bad_ts = (ts.isna() | (ts.dt.year != 2018)).to_numpy()
            nonfin = ~np.isfinite(X).all(axis=1)
            Xn = np.where(np.isfinite(X), X, 0.0)
            neg = (Xn[:, neg_cols] < 0).any(axis=1)
            reason = np.where(bad_ts, 1, np.where(nonfin, 2, np.where(neg, 3, 0)))
            for r, name in ((1, "bad_or_non2018_timestamp"), (2, "nan_or_inf"), (3, "negative_value")):
                for a, n in zip(*np.unique(atype[reason == r], return_counts=True)):
                    drops[(name, ATYPES[a])] += int(n)
            ok = reason == 0
            Xk, tsk = X[ok], ts[ok]
            h = pd.util.hash_pandas_object(pd.DataFrame(Xk, columns=feats)[hash_cols], index=False).to_numpy()
            fb.write(Xk.astype(np.float32).tobytes())
            meta["ts"].append(tsk.astype("datetime64[ns]").astype("int64").to_numpy())
            meta["atype"].append(atype[ok])
            meta["y"].append((atype[ok] > 0).astype(np.int8))
            meta["dst_port"].append(Xk[:, dport_i].astype(np.int32))
            meta["hash"].append(h)
            n_out += int(ok.sum())
    arrs = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in meta.items()}
    np.savez(interim / f"{path.stem}.npz", **arrs)
    return {"file": path.name, "rows_in": n_in, "rows_out": n_out, "embedded_headers": hdr_rows,
            "drops": {f"{r}|{a}": n for (r, a), n in drops.items()}}


def stage_clean(args, files):
    interim = args.out_dir / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    heads, feats = common_features(files)
    stats_path = interim / "clean_stats.json"
    if stats_path.exists() and not args.force:
        print("stage clean: cached")
        return json.loads(stats_path.read_text())
    stats = {"features": feats, "files": []}
    t0 = time.time()
    for i, f in enumerate(files):
        s = clean_file(f, i, heads[f], feats, interim, args.max_rows)
        stats["files"].append(s)
        print(f"[{time.time()-t0:6.0f}s] {f.name}: in={s['rows_in']:,} kept={s['rows_out']:,}", flush=True)
    stats["files_order"] = [f.stem for f in files]
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


# ----------------------------------------------------------------------------- stage 2
def load_meta(interim, stems):
    parts = [dict(np.load(interim / f"{s}.npz")) for s in stems]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cat["file"] = np.concatenate([np.full(len(p["y"]), i, dtype=np.int16) for i, p in enumerate(parts)])
    cat["row"] = np.concatenate([np.arange(len(p["y"]), dtype=np.int64) for p in parts])
    return cat


def dedup(m):
    """Drop identical model-input vectors. Vectors seen with both benign and attack labels are removed
    entirely (ambiguous); otherwise the first occurrence is kept."""
    h, y = m["hash"], m["y"]
    order = np.argsort(h, kind="stable")
    hs, ys = h[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(hs)) + 1]
    gsize = np.diff(np.r_[starts, len(hs)])
    conflict = np.minimum.reduceat(ys, starts) != np.maximum.reduceat(ys, starts)
    keep_sorted = np.zeros(len(hs), dtype=bool)
    keep_sorted[starts[~conflict]] = True                 # first occurrence of each clean group
    reason = np.zeros(len(hs), dtype=np.int8)             # 0 keep, 1 duplicate, 2 label conflict
    reason[~keep_sorted] = 1
    reason[np.repeat(conflict, gsize)] = 2
    out = np.zeros(len(hs), dtype=np.int8)
    out[order] = reason
    return out


def assign_blocks(m, keep, block_sec, fracs, seed):
    sec = m["ts"][keep] // 1_000_000_000
    block_id = (sec // 86400) * 100_000 + (sec % 86400) // block_sec
    at = m["atype"][keep].astype(np.int64)
    order = np.argsort(block_id, kind="stable")
    b = block_id[order]
    starts = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
    ub = b[starts]
    bits = np.where(at[order] > 0, np.left_shift(1, at[order]), 0)
    bmask = np.bitwise_or.reduceat(bits, starts)
    rng = np.random.default_rng(seed)
    bsplit = np.zeros(len(ub), dtype=np.int8)
    for mval in np.unique(bmask):
        idx = np.flatnonzero(bmask == mval)
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            nt, nv = 0, 0
        elif n == 2:
            nt, nv = 1, 0
        else:
            nt, nv = max(1, round(fracs[2] * n)), max(1, round(fracs[1] * n))
        bsplit[idx[:nt]] = 2
        bsplit[idx[nt:nt + nv]] = 1
    return bsplit[np.searchsorted(ub, block_id)], block_id


def stage_split(args, stats):
    interim = args.out_dir / "interim"
    out = interim / "split.npz"
    if out.exists() and not args.force:
        print("stage split: cached")
        return json.loads((interim / "split_stats.json").read_text())
    m = load_meta(interim, stats["files_order"])
    n = len(m["y"])
    dres = dedup(m)
    keep = dres == 0
    drops = {}
    for code, name in ((1, "duplicate_vector"), (2, "label_conflict_vector")):
        for a, c in zip(*np.unique(m["atype"][dres == code], return_counts=True)):
            drops[f"{name}|{ATYPES[a]}"] = int(c)
    split_k, block_k = assign_blocks(m, keep, args.block_sec, (0.7, 0.1, 0.2), args.seed)
    split = np.full(n, -1, dtype=np.int8)
    block = np.zeros(n, dtype=np.int64)
    split[keep], block[keep] = split_k, block_k
    # sanity: no identical model-input vector across partitions
    hp = [m["hash"][split == s] for s in range(3)]
    overlap = {"train&val": len(np.intersect1d(hp[0], hp[1])), "train&test": len(np.intersect1d(hp[0], hp[2])),
               "val&test": len(np.intersect1d(hp[1], hp[2]))}
    assert not any(overlap.values()), overlap
    np.savez(out, split=split, block=block, fidx=m["file"], row=m["row"])
    ss = {"rows_after_clean": int(n), "rows_after_dedup": int(keep.sum()), "dedup_drops": drops,
          "hash_overlap_between_partitions": overlap}
    (interim / "split_stats.json").write_text(json.dumps(ss, indent=2))
    return ss


# ----------------------------------------------------------------------------- stage 3
def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def stage_scale(args, stats):
    interim, proc = args.out_dir / "interim", args.out_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    feats = stats["features"]
    d_all = len(feats)
    sp = np.load(interim / "split.npz")
    split, fidx, rows = sp["split"], sp["fidx"], sp["row"]
    stems = stats["files_order"]
    mm = {i: np.memmap(interim / f"{s}.f32", dtype=np.float32, mode="r").reshape(-1, d_all)
          for i, s in enumerate(stems)}
    meta = load_meta(interim, stems)

    # ---- fit on TRAIN only: constant columns, protocol categories, log-space min/max
    rmin, rmax = np.full(d_all, np.inf), np.full(d_all, -np.inf)
    proto_i = feats.index("protocol")
    protos = set()
    for i in mm:
        sel = rows[(fidx == i) & (split == 0)]
        for a in range(0, len(sel), 500_000):
            X = np.asarray(mm[i][sel[a:a + 500_000]])
            rmin, rmax = np.minimum(rmin, X.min(0)), np.maximum(rmax, X.max(0))
            protos.update(np.unique(X[:, proto_i]).tolist())
    constant = [c for c, lo, hi in zip(feats, rmin, rmax) if lo == hi]
    model_cols = [c for c in feats if c not in constant and c not in ("dst_port", "protocol")]
    mi = [feats.index(c) for c in model_cols]
    protos = sorted(protos)
    lmin, lmax = np.full(len(mi), np.inf), np.full(len(mi), -np.inf)
    for i in mm:
        sel = rows[(fidx == i) & (split == 0)]
        for a in range(0, len(sel), 500_000):
            L = slog(np.asarray(mm[i][sel[a:a + 500_000]])[:, mi])
            lmin, lmax = np.minimum(lmin, L.min(0)), np.maximum(lmax, L.max(0))
    span = np.where(lmax > lmin, lmax - lmin, 1.0)
    out_cols = model_cols + [f"protocol_{int(p)}" for p in protos] + ["protocol_other"]
    scaler = {"fit_on": "train partition only", "transform": "signed log1p then min-max to [0,1], no clipping",
              "model_columns": model_cols, "log_min": lmin.tolist(), "log_max": lmax.tolist(),
              "protocol_categories": [int(p) for p in protos], "constant_columns_dropped": constant,
              "excluded_identifiers": ["dst_port (kept in meta for analysis only)", "timestamp", "label"],
              "output_columns": out_cols}
    (proc / "scaler.json").write_text(json.dumps(scaler, indent=2))

    # ---- transform and write each partition
    summary = {}
    for s, name in enumerate(SPLITS):
        mask = split == s
        n = int(mask.sum())
        X_out = np.lib.format.open_memmap(proc / f"X_{name}.npy", mode="w+", dtype=np.float32, shape=(n, len(out_cols)))
        pos = 0
        gl = np.flatnonzero(mask)                     # global indices, file-major order
        for i in mm:
            sel_global = gl[fidx[gl] == i]
            sel = rows[sel_global]
            for a in range(0, len(sel), 500_000):
                raw = np.asarray(mm[i][sel[a:a + 500_000]])
                Z = ((slog(raw[:, mi]) - lmin) / span).astype(np.float32)
                P = np.zeros((len(raw), len(protos) + 1), dtype=np.float32)
                for k, p in enumerate(protos):
                    P[:, k] = raw[:, proto_i] == p
                P[:, -1] = ~np.isin(raw[:, proto_i], protos)
                X_out[pos:pos + len(raw)] = np.hstack([Z, P])
                pos += len(raw)
        X_out.flush()
        assert pos == n
        md = pd.DataFrame({
            "source_file": np.array(stems)[fidx[gl]], "row_in_file": rows[gl],
            "timestamp": pd.to_datetime(meta["ts"][gl]), "block": sp["block"][gl],
            "label_binary": meta["y"][gl], "attack_type": np.array(ATYPES)[meta["atype"][gl]],
            "dst_port": meta["dst_port"][gl]})
        md.to_pickle(proc / f"meta_{name}.pkl")
        summary[name] = {"rows": n, "benign": int((md.label_binary == 0).sum()),
                         "attack": int((md.label_binary == 1).sum()),
                         "by_type": md.attack_type.value_counts().to_dict(),
                         "blocks": int(md.block.nunique()),
                         "days": sorted(md.timestamp.dt.strftime("%Y-%m-%d").unique().tolist())}
        # sanity on the written array
        for a in range(0, n, 1_000_000):
            assert np.isfinite(np.asarray(X_out[a:a + 1_000_000])).all(), f"non-finite values in X_{name}"
        print(f"wrote X_{name}.npy {X_out.shape}", flush=True)

    # ---- port shortcut analysis on TRAIN
    mtr = pd.read_pickle(proc / "meta_train.pkl")
    ben_port = mtr[mtr.label_binary == 0].dst_port.value_counts(normalize=True)
    shortcut = {}
    for t, g in mtr[mtr.label_binary == 1].groupby("attack_type"):
        vc = g.dst_port.value_counts(normalize=True)
        shortcut[t] = {"top_port": int(vc.index[0]), "share_of_type": round(float(vc.iloc[0]), 4),
                       "share_of_benign_on_that_port": round(float(ben_port.get(vc.index[0], 0.0)), 4)}
    return {"model_input_dim": len(out_cols), "constant_columns_dropped": constant, "protocols": scaler["protocol_categories"],
            "partitions": summary, "dst_port_shortcut_train": shortcut}


def write_report(args, stats, ss, ssc):
    L = ["# Phase 2 - preprocessing and split report\n"]
    a = L.append
    tot_in = sum(f["rows_in"] for f in stats["files"])
    a(f"Records read (header rows removed): **{tot_in:,}**; after cleaning: **{ss['rows_after_clean']:,}**; "
      f"after de-duplication: **{ss['rows_after_dedup']:,}**.\n")
    a("## Rows removed, by reason and attack type\n\n| Reason | Attack type | Rows |\n|---|---|---:|")
    allr = Counter()
    for f in stats["files"]:
        for k, v in f["drops"].items():
            allr[k] += v
    for k, v in ss["dedup_drops"].items():
        allr[k] += v
    for k, v in sorted(allr.items()):
        r, t = k.split("|")
        a(f"| {r} | {t} | {v:,} |")
    a("\n## Partitions\n\n| Partition | Rows | Benign | Attack | Attack share | Blocks | Days |\n|---|---:|---:|---:|---:|---:|---|")
    for n, p in ssc["partitions"].items():
        a(f"| {n} | {p['rows']:,} | {p['benign']:,} | {p['attack']:,} | {100*p['attack']/p['rows']:.2f}% | {p['blocks']:,} | {len(p['days'])} days |")
    a("\n### Attack types per partition\n\n| Type | Train | Val | Test |\n|---|---:|---:|---:|")
    for t in ATYPES:
        a(f"| {t} | " + " | ".join(f"{ssc['partitions'][n]['by_type'].get(t, 0):,}" for n in SPLITS) + " |")
    a(f"\nIdentical model-input vectors shared between partitions: {ss['hash_overlap_between_partitions']} (must be 0).\n")
    a(f"## Model input\n\n- Dimension: **{ssc['model_input_dim']}** (numeric features after signed log1p + min-max fit on train, plus one-hot protocol {ssc['protocols']} and 'other')\n"
      f"- Constant (in train) columns dropped: {', '.join(ssc['constant_columns_dropped'])}\n"
      "- Excluded from model input: timestamp, label, `dst_port` (kept in meta for analysis only)\n"
      "- No clipping of val/test values to [0,1]: out-of-range values are kept on purpose.\n")
    a("## `dst_port` shortcut check (train partition)\n\n| Attack type | Most common dst_port | Share of type on it | Share of benign on it |\n|---|---:|---:|---:|")
    for t, v in ssc["dst_port_shortcut_train"].items():
        a(f"| {t} | {v['top_port']} | {v['share_of_type']:.2%} | {v['share_of_benign_on_that_port']:.2%} |")
    a("\n## Split strategy\n\n"
      "**Note (added in Phase 7):** blocks are formed on the timestamp as stored in the CSVs, a 12-hour clock without AM/PM "
      "(afternoon events appear 12 hours early), so one block can hold records from two real periods of a day. This creates no leakage "
      "(blocks are never split across partitions) but the blocks are not strictly consecutive real-time minutes.\n\n"
      f"Each capture day is cut into {args.block_sec}-second time blocks. Whole blocks are assigned to train/val/test "
      "(70/10/20% of blocks) at random with a fixed seed, stratified by the set of attack types present in the block. "
      "Rows of one block never appear in two partitions. Labels are used only for this stratification. "
      "Scaler, constant-column list and protocol categories are fitted on the training partition only. "
      "Limitations: neighbouring blocks of the same attack are similar, and flows can span a block boundary, so the "
      "in-distribution test is optimistic; Experiment C (unseen attacks) is the stronger generalisation test.\n")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "phase2_report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--block-sec", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke test: rows per file")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    files = sorted(args.data_dir.glob("*.csv"))
    stats = stage_clean(args, files)
    ss = stage_split(args, stats)
    ssc = stage_scale(args, stats)
    (args.out_dir / "processed" / "phase2_summary.json").write_text(json.dumps({"clean": stats["files"], "split": ss, "scale": ssc}, indent=2))
    write_report(args, stats, ss, ssc)
    print("Phase 2 done")


if __name__ == "__main__":
    main()
