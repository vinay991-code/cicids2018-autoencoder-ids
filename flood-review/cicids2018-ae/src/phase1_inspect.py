"""Phase 1: dataset inspection for CSE-CIC-IDS2018 (also works on any CIC-style per-day CSVs).

Streams every CSV in chunks (the full dataset is too big to hold as strings) and reports only
what is measured from the files: rows, columns, labels, NaN / inf / non-numeric cells,
duplicates, timestamps, constant columns, cross-file schema compatibility and leakage risks.

Usage:
    python -I src/phase1_inspect.py --data-dir <folder with CSVs> --out-dir reports
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = {"flow_id", "src_ip", "source_ip", "dst_ip", "destination_ip",
           "src_port", "source_port", "timestamp"}
MISSING_TOKENS = {"", "nan"}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def read_headers(files):
    headers = {}
    for f in files:
        raw = pd.read_csv(f, nrows=0, encoding="latin-1").columns.tolist()
        headers[f.name] = {"raw": raw, "norm": [norm(c) for c in raw]}
    return headers


def dup_stats(hashes: list) -> dict:
    h = np.concatenate(hashes) if hashes else np.empty(0, dtype=np.uint64)
    n = len(h)
    u = len(np.unique(h))
    return {"rows": int(n), "unique": int(u), "duplicates": int(n - u)}


def inspect(data_dir: Path, chunksize: int, max_files):
    files = sorted(data_dir.glob("*.csv"))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir}")
    headers = read_headers(files)

    # ---- schema comparison -------------------------------------------------
    col_sets = {k: set(v["norm"]) for k, v in headers.items()}
    union = set().union(*col_sets.values())
    common = set.intersection(*col_sets.values())
    schema = {
        "n_files": len(files),
        "columns_per_file": {k: len(v["norm"]) for k, v in headers.items()},
        "common_columns": len(common),
        "union_columns": len(union),
        "columns_missing_per_file": {k: sorted(union - s) for k, s in col_sets.items() if union - s},
        "dup_normalized_names_in_file": {k: [c for c, n in Counter(v["norm"]).items() if n > 1]
                                         for k, v in headers.items()
                                         if len(set(v["norm"])) != len(v["norm"])},
    }
    label_col = "label"
    if label_col not in common:
        raise SystemExit("No common 'label' column across files; inspect headers manually.")
    feat_cols = sorted(c for c in common if c != label_col and c not in ID_COLS)

    # ---- streaming pass ----------------------------------------------------
    per_file = {}
    label_total = Counter()
    label_files = defaultdict(set)
    label_ts = {}                      # label -> [min, max]
    colstat = {c: {"min": np.inf, "max": -np.inf, "nan": 0, "inf": 0, "nonnum": 0, "neg": 0}
               for c in feat_cols}
    h_full, h_feat, h_featlab = [], [], []
    total_rows = embedded_headers = 0
    ts_fail = 0
    ts_nonmono = 0
    t0 = time.time()

    for f in files:
        fs = {"rows": 0, "embedded_header_rows": 0, "labels": Counter(),
              "ts_min": None, "ts_max": None, "ts_fail": 0, "ts_non_monotonic_steps": 0}
        prev_last = None
        for chunk in pd.read_csv(f, dtype=str, chunksize=chunksize, encoding="latin-1",
                                 keep_default_na=False, na_filter=False):
            chunk.columns = [norm(c) for c in chunk.columns]
            # some CIC files repeat the header row mid-file
            is_hdr = chunk[label_col].str.strip().str.lower() == "label"
            n_hdr = int(is_hdr.sum())
            if n_hdr:
                fs["embedded_header_rows"] += n_hdr
                chunk = chunk[~is_hdr]
            if chunk.empty:
                continue
            chunk = chunk.reset_index(drop=True)
            labels = chunk[label_col].str.strip()
            fs["rows"] += len(chunk)
            fs["labels"].update(labels.value_counts().to_dict())

            # timestamps (day-first, mixed formats tolerated)
            if "timestamp" in chunk.columns:
                ts = pd.to_datetime(chunk["timestamp"], dayfirst=True, errors="coerce", format="mixed")
                bad = int(ts.isna().sum())
                fs["ts_fail"] += bad
                ok = ts.dropna()
                if len(ok):
                    fs["ts_min"] = ok.min() if fs["ts_min"] is None else min(fs["ts_min"], ok.min())
                    fs["ts_max"] = ok.max() if fs["ts_max"] is None else max(fs["ts_max"], ok.max())
                    fs["ts_non_monotonic_steps"] += int((ok.diff().dt.total_seconds() < 0).sum())
                    if prev_last is not None and ok.iloc[0] < prev_last:
                        fs["ts_non_monotonic_steps"] += 1
                    prev_last = ok.iloc[-1]
                    g = pd.DataFrame({"l": labels[ts.notna()].values, "t": ok.values}).groupby("l")["t"].agg(["min", "max"])
                    for lab, r in g.iterrows():
                        cur = label_ts.get(lab)
                        label_ts[lab] = [r["min"], r["max"]] if cur is None else [min(cur[0], r["min"]), max(cur[1], r["max"])]

            # numeric quality per feature column
            num = pd.DataFrame(index=chunk.index)
            for c in feat_cols:
                raw = chunk[c]
                v = pd.to_numeric(raw, errors="coerce")
                st = colstat[c]
                isinf = np.isinf(v.to_numpy(dtype="float64", na_value=np.nan))
                isnan = v.isna().to_numpy()
                if isnan.any():
                    nonnum = isnan & ~raw.str.strip().str.lower().isin(MISSING_TOKENS).to_numpy()
                else:
                    nonnum = isnan
                st["inf"] += int(isinf.sum())
                st["nonnum"] += int(nonnum.sum())
                st["nan"] += int(isnan.sum() - nonnum.sum())
                fin = v[~isnan & ~isinf]
                if len(fin):
                    st["min"] = min(st["min"], float(fin.min()))
                    st["max"] = max(st["max"], float(fin.max()))
                    st["neg"] += int((fin < 0).sum())
                num[c] = v
            # duplicate hashing
            raw_all = chunk.drop(columns=[c for c in chunk.columns if c not in common])
            h_full.append(pd.util.hash_pandas_object(raw_all, index=False).to_numpy())
            hf = pd.util.hash_pandas_object(chunk[feat_cols], index=False).to_numpy()
            h_feat.append(hf)
            hl = pd.util.hash_pandas_object(pd.DataFrame({"h": hf, "l": labels}), index=False).to_numpy()
            h_featlab.append(hl)

        embedded_headers += fs["embedded_header_rows"]
        total_rows += fs["rows"]
        ts_fail += fs["ts_fail"]
        ts_nonmono += fs["ts_non_monotonic_steps"]
        for lab, n in fs["labels"].items():
            label_total[lab] += n
            label_files[lab].add(f.name)
        per_file[f.name] = fs
        print(f"[{time.time()-t0:6.0f}s] {f.name}: {fs['rows']:,} rows", flush=True)

    # ---- duplicates --------------------------------------------------------
    full = dup_stats(h_full)
    feat = dup_stats(h_feat)
    featlab = dup_stats(h_featlab)
    conflicting = feat["unique"] and (featlab["unique"] - feat["unique"])  # feature vectors seen with >1 label
    dups = {"exact_row_duplicates": full, "feature_vector_duplicates": feat,
            "feature_vectors_with_conflicting_labels": int(conflicting)}

    # ---- constant / unusable columns --------------------------------------
    constant = [c for c, s in colstat.items() if s["min"] == s["max"] and s["min"] != np.inf]
    all_nan = [c for c, s in colstat.items() if s["min"] == np.inf]
    col_quality = {c: s for c, s in colstat.items() if s["nan"] or s["inf"] or s["nonnum"]}

    # ---- leakage checks ----------------------------------------------------
    present_ids = sorted(ID_COLS & common)
    label_ts_out = {k: [str(v[0]), str(v[1])] for k, v in label_ts.items()}
    leakage = {
        "identifier_columns_present_in_all_files": present_ids,
        "identifier_columns_present_in_some_files": sorted((ID_COLS & union) - common),
        "label_time_windows": label_ts_out,
        "label_only_in_one_file": sorted(l for l, fs_ in label_files.items()
                                      if len(fs_) == 1 and l.strip().lower() != "benign"),
    }

    return {
        "data_dir": str(data_dir), "n_files": len(files), "total_rows": int(total_rows),
        "embedded_header_rows_dropped": int(embedded_headers),
        "n_feature_columns_common_excl_ids_label": len(feat_cols),
        "label_counts": dict(label_total.most_common()),
        "benign_rows": int(sum(n for l, n in label_total.items() if l.strip().lower() == "benign")),
        "timestamps": {"unparseable": int(ts_fail), "non_monotonic_steps": int(ts_nonmono)},
        "schema": schema, "duplicates": dups,
        "constant_columns": constant, "all_nan_columns": all_nan,
        "columns_with_nan_inf_nonnumeric": col_quality,
        "columns_with_negative_values": {c: s["neg"] for c, s in colstat.items() if s["neg"]},
        "leakage": leakage,
        "per_file": {k: {**{kk: (str(vv) if kk.startswith("ts_m") else vv) for kk, vv in v.items() if kk != "labels"},
                         "labels": dict(v["labels"])} for k, v in per_file.items()},
    }


def to_markdown(r: dict) -> str:
    L = []
    a = L.append
    tot = r["total_rows"]
    a("# Phase 1 - Dataset inspection report\n")
    a(f"- Folder: `{r['data_dir']}`\n- CSV files: **{r['n_files']}**\n- Total records (after dropping {r['embedded_header_rows_dropped']:,} embedded header rows): **{tot:,}**")
    a(f"- Common feature columns (excluding label and identifiers): **{r['n_feature_columns_common_excl_ids_label']}**")
    a(f"- Columns across files: common={r['schema']['common_columns']}, union={r['schema']['union_columns']}\n")
    a("## Class distribution\n\n| Label | Records | Share |\n|---|---:|---:|")
    for l, n in r["label_counts"].items():
        a(f"| {l} | {n:,} | {100*n/tot:.3f}% |")
    a(f"\nBenign: **{r['benign_rows']:,}** ({100*r['benign_rows']/tot:.2f}%); attack: **{tot-r['benign_rows']:,}**\n")
    a("## Per-file summary\n\n| File | Columns | Rows | First timestamp | Last timestamp | Labels |\n|---|---:|---:|---|---|---|")
    for k, v in r["per_file"].items():
        labs = ", ".join(f"{l}: {n:,}" for l, n in v["labels"].items())
        a(f"| {k} | {r['schema']['columns_per_file'][k]} | {v['rows']:,} | {v['ts_min']} | {v['ts_max']} | {labs} |")
    a("\n## Data quality\n")
    d = r["duplicates"]
    a(f"- Exact duplicate rows: **{d['exact_row_duplicates']['duplicates']:,}**")
    a(f"- Duplicate feature vectors (ignoring ids/timestamp/label): **{d['feature_vector_duplicates']['duplicates']:,}**")
    a(f"- Extra (feature vector, label) pairs beyond one label per vector, i.e. label conflicts among identical feature vectors: **{d['feature_vectors_with_conflicting_labels']:,}**")
    a(f"- Unparseable timestamps: **{r['timestamps']['unparseable']:,}**; out-of-order timestamp steps: **{r['timestamps']['non_monotonic_steps']:,}**")
    a(f"- Constant columns ({len(r['constant_columns'])}): {', '.join(r['constant_columns']) or 'none'}")
    a(f"- All-NaN columns: {', '.join(r['all_nan_columns']) or 'none'}")
    a("\n### Columns with NaN / inf / non-numeric cells\n")
    if r["columns_with_nan_inf_nonnumeric"]:
        a("| Column | NaN/empty | +-inf | Non-numeric text |\n|---|---:|---:|---:|")
        for c, s in r["columns_with_nan_inf_nonnumeric"].items():
            a(f"| {c} | {s['nan']:,} | {s['inf']:,} | {s['nonnum']:,} |")
    else:
        a("None found.")
    a("\n### Columns containing negative values\n")
    a(", ".join(f"{c} ({n:,})" for c, n in r["columns_with_negative_values"].items()) or "none")
    a("\n## Feature compatibility across files\n")
    s = r["schema"]
    if s["columns_missing_per_file"]:
        for k, miss in s["columns_missing_per_file"].items():
            a(f"- `{k}` lacks: {', '.join(miss)}")
    else:
        a("All files share the same normalized column set.")
    if s["dup_normalized_names_in_file"]:
        a(f"\nDuplicate normalized column names: {s['dup_normalized_names_in_file']}")
    a("\n## Potential leakage\n")
    lk = r["leakage"]
    a(f"- Identifier/time columns present in all files: {', '.join(lk['identifier_columns_present_in_all_files']) or 'none'}")
    a(f"- Identifier/time columns present in only some files: {', '.join(lk['identifier_columns_present_in_some_files']) or 'none'}")
    a(f"- Labels that occur in exactly one file (each such attack is confined to a single capture day): {', '.join(lk['label_only_in_one_file']) or 'none'}")
    a("\n| Label | First seen | Last seen |\n|---|---|---|")
    for l, (lo, hi) in lk["label_time_windows"].items():
        a(f"| {l} | {lo} | {hi} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("reports"), type=Path)
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--max-files", type=int, default=None, help="smoke-test on the first N files only")
    ap.add_argument("--tag", default="phase1_inspection")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rep = inspect(args.data_dir, args.chunksize, args.max_files)
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(rep, indent=2, default=str))
    (args.out_dir / f"{args.tag}.md").write_text(to_markdown(rep), encoding="utf-8")
    print(f"Saved {args.out_dir / (args.tag + '.md')}")


if __name__ == "__main__":
    main()
