"""Phase 16 / Experiment G: label-free, rate-limited ADAPTIVE THRESHOLD on a FROZEN autoencoder.

What this is, and what it is not
  * Only the alarm threshold changes over the stream. The autoencoder weights are never updated, so the score of every
    record is identical in all arms, ROC-AUC / PR-AUC are identical in all arms, and the ranking of attacks versus
    benign records cannot improve. This is threshold adaptation, NOT model adaptation (Phases 7C, 8, 9) and NOT
    concept-drift recovery: a model whose ranking is inverted on a shifted day stays inverted.
  * The threshold policy sees ONLY the reconstruction scores of past and current intervals. Attack labels are read
    after the thresholds are fixed (policy_trace takes no labels), to count results and for diagnostics.

Data: exactly the Phase 7 stream and warm-up baseline (data/interim/stream_main_*): one-class autoencoder trained on the
benign records of days 02-14 and 02-15, initial threshold theta0 = 99th percentile of later benign warm-up errors,
evaluation stream = the 9,504,036 records of the 8 following days in consecutive 50,000-record intervals.
Each interval is flagged with the threshold produced BEFORE it, then used to update the threshold for the next one.

Arms (settings fixed before the first result was seen; this is the first and only run reported):
  S   static: theta0 for the whole stream.
  D   adaptive threshold (full): window = last 6 intervals (all their records, 300,000 scores);
        1. robust statistic: the lower quartile Q25 of the window's log10 scores. Records with high scores, benign or
           attack, are not in it: it moves only if more than about a quarter of the records are replaced by anomalies,
           and even then only to a lower-than-median benign quantile;
        2. target log10(theta*) = Q25(window) + d, with d = log10(theta0) - log10(Q25 of the warm-up benign anchor scores),
           so the rule reproduces theta0 on warm-up-like traffic (same 99th-percentile false-alarm target);
        3. rate limit: log10(theta) moves toward the target by at most 0.25 decades (factor 1.78) per interval;
        4. clamp: log10(theta) stays within [log10(theta0) - 1, log10(theta0) + 3].
  D1  D without the rate limit (ablation, declared in advance).
  D2  the Phase 7 B statistic (median + k x 1.4826 MAD of the window's log10 scores, k = 4.286 from warm-up) with the rate
      limit and the clamp of D (ablation: is the lower quartile what matters?).
Shifted periods: the rule declared in advance (a day is shifted if the static arm flags more than half of its benign records)
matches NO day of this stream (the Phase 8/9 stream used a different baseline, on which 02-16 and 02-21 were flagged almost
entirely). After seeing this, a DESCRIPTIVE post-hoc grouping was added and is labelled as such everywhere: a day is
"bulk-shifted" if the median frozen score of its benign records is at least 10 times the warm-up benign median
(2.43e-5). It uses labels and describes the data only; it was not chosen by looking at any arm's result.
Phase 7 results (A static, B adaptive threshold from all records without rate limit, C adaptive baseline) are kept as they
are and quoted from results/phase7_main.json; B and C are not re-run.

Usage: python src/phase16_adaptive_threshold.py --check-only   (scores + reproduction checks, no arm results)
       python src/phase16_adaptive_threshold.py                (full run)
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream import load_stream  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "interim"
SETTINGS = dict(interval_records=50_000, window=6, q_bulk=0.25, delta_decades=0.25, floor_decades=-1.0, ceil_decades=3.0,
                warm_days=2, tag="main", mad_k_source="stream_main_info.json", shifted_day_rule="static FPR > 0.5 on the day (labels, evaluation only)")


def _load_npmodels():
    spec = importlib.util.spec_from_file_location("npmodels", ROOT / "app" / "npmodels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def interval_ids(meta, records):
    """Same definition as stream_setup.interval_ids (copied to avoid importing torch)."""
    day_idx = pd.factorize(meta.day)[0].astype(np.int64)
    pos = meta.groupby("day").cumcount().to_numpy().astype(np.int64)
    return day_idx * 100_000 + pos // records


def score_rows(model, X, chunk=262_144):
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), chunk):
        xb = np.ascontiguousarray(X[i:i + chunk], dtype=np.float32)
        out[i:i + chunk] = ((model(xb) - xb) ** 2).mean(axis=1)
    return out


# ------------------------------------------------------------------ the threshold policies (labels never enter)
def policy_trace(scores, iid, theta0, d_offset, k_mad, spec, st=SETTINGS):
    """Returns the threshold used for each interval (before seeing it). `scores`: per-record reconstruction errors."""
    ids = np.unique(iid)
    starts, ends = np.searchsorted(iid, ids, "left"), np.searchsorted(iid, ids, "right")
    lt0 = float(np.log10(theta0))
    lo, hi = lt0 + st["floor_decades"], lt0 + st["ceil_decades"]
    lt, hist, thr = lt0, [], []
    for a, b in zip(starts, ends):
        thr.append(10 ** lt)
        hist = (hist + [np.log10(np.maximum(scores[a:b], 1e-12))])[-st["window"]:]
        if spec["static"]:
            continue
        pooled = np.concatenate(hist)
        if spec["stat"] == "q25":
            target = float(np.quantile(pooled, st["q_bulk"])) + d_offset
        else:
            med = float(np.median(pooled))
            mad = 1.4826 * float(np.median(np.abs(pooled - med)))
            target = med + k_mad * max(mad, 1e-3)
        step = target - lt
        if spec["rate"]:
            step = float(np.clip(step, -st["delta_decades"], st["delta_decades"]))
        lt = float(np.clip(lt + step, lo, hi))
    return np.array(thr)


ARMS = {"S_static": dict(static=True), "D_adaptive": dict(static=False, stat="q25", rate=True),
        "D1_no_rate_limit": dict(static=False, stat="q25", rate=False), "D2_median_mad_limited": dict(static=False, stat="mad", rate=True)}


# ------------------------------------------------------------------ evaluation helpers
def metrics(tp, fp, fn, tn):
    div = lambda a, b: a / b if b else float("nan")
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    f1 = (2 * p * r / (p + r)) if (tp and p == p and r == r) else (0.0 if tp + fn else float("nan"))
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"n": int(tp + fp + fn + tn), "attacks": int(tp + fn), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": p, "recall": r, "f1": f1, "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp),
            "mcc": float((tp * tn - fp * fn) / mcc_den) if mcc_den else float("nan"), "flagged_share": div(tp + fp, tp + fp + fn + tn)}


def interval_counts(scores, y, iid, thr):
    ids = np.unique(iid)
    starts, ends = np.searchsorted(iid, ids, "left"), np.searchsorted(iid, ids, "right")
    out = np.zeros((len(ids), 4), np.int64)
    for i, (a, b) in enumerate(zip(starts, ends)):
        flag, att = scores[a:b] > thr[i], y[a:b].astype(bool)
        out[i] = ((flag & att).sum(), (flag & ~att).sum(), (~flag & att).sum(), (~flag & ~att).sum())
    return ids, out


def block_boot(cnt, base="S_static", n_boot=1000, block=8, seed=0):
    """Moving-block bootstrap over update intervals (same scheme as phase7_tables.py): 95% interval of arm - static."""
    rng = np.random.default_rng(seed)
    n = len(cnt[base])
    nb = -(-n // block)
    keys = ("precision", "recall", "f1", "fpr", "fnr")
    res = {a: {k: [] for k in keys} for a in cnt if a != base}
    for _ in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        mb = metrics(*cnt[base][idx].sum(0))
        for a in res:
            ma = metrics(*cnt[a][idx].sum(0))
            for k in keys:
                res[a][k].append(ma[k] - mb[k])
    return {a: {k: [float(np.nanmean(v)), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in d.items()} for a, d in res.items()}


def get_scores(args, B):
    path = CACHE / "stream_main_frozen_scores.npy"
    if path.exists() and not args.rescore:
        return np.load(path)
    s = score_rows(B["model"], B["X"][B["e0"]:])
    np.save(path, s)
    return s


def load_baseline(st=SETTINGS):
    npm = _load_npmodels()
    X, m = load_stream()
    days = sorted(m.day.unique())
    warm = np.isin(m.day.to_numpy(), days[:st["warm_days"]])
    e0 = int(np.flatnonzero(~warm)[0])
    assert warm[:e0].all() and not warm[e0:].any()
    info = json.load(open(CACHE / "stream_main_info.json"))
    model = npm.load_model(CACHE / "stream_main_model.pt", supervised=False)
    return dict(X=X, meta=m, e0=e0, warm=warm, model=model, info=info, anchor=np.load(CACHE / "stream_main_anchor.npy"))


def reproduction_checks(B, scores, iid, m, y, p7):
    """Checks that this script scores and cuts the stream exactly like Phase 7. Returns (list of dicts, ok)."""
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + ": " + detail, flush=True)

    y_all = B["meta"].label_binary.to_numpy()
    ben = np.flatnonzero(B["warm"] & (y_all == 0))
    va = ben[int(0.8 * len(ben)):]
    sv = score_rows(B["model"], B["X"][va])
    thr_re = float(np.percentile(sv, 99))
    rel = abs(thr_re - B["info"]["thr0"]) / B["info"]["thr0"]
    add("V1 numpy model reproduces the stored warm-up threshold (p99 of validation errors)", rel < 1e-5,
        f"recomputed {thr_re:.9e}, stored {B['info']['thr0']:.9e}, relative difference {rel:.2e}")
    ids, cnt = interval_counts(scores, y, iid, np.full(len(np.unique(iid)), B["info"]["thr0"]))
    iv = pd.read_csv(ROOT / "results" / "phase7_main_intervals.csv")
    a = iv[iv.arm == "A_static"].reset_index(drop=True)
    add("V2 same intervals as Phase 7 (ids, days, sizes)", len(a) == len(ids) and (a.interval.to_numpy() == ids).all()
        and (a.n.to_numpy() == cnt.sum(1)).all(), f"{len(ids)} intervals, 50,000 records each except day ends")
    ref = a[["tp", "fp", "fn", "tn"]].to_numpy()
    diff = np.abs(ref - cnt).sum()
    add("V3 static counts equal Phase 7 arm A (tp/fp/fn/tn per interval)", diff <= 20,
        f"sum of absolute differences over {ref.size:,} count cells = {int(diff)} (records within float rounding of the threshold)")
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results_threshold")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    B = load_baseline()
    m = B["meta"].iloc[B["e0"]:].reset_index(drop=True)
    iid = interval_ids(m, SETTINGS["interval_records"])
    y = m.label_binary.to_numpy()
    scores = get_scores(args, B)
    assert len(scores) == len(m)
    print(f"scored {len(scores):,} evaluation records with the frozen warm-up model", flush=True)
    p7 = json.load(open(ROOT / "results" / "phase7_main.json"))
    checks = reproduction_checks(B, scores, iid, m, y, p7)
    if args.check_only:
        print("check-only run done; no arm was evaluated")
        return
    assert all(c["ok"] for c in checks), "reproduction checks failed; not evaluating"

    theta0, k_mad = B["info"]["thr0"], B["info"]["k"]
    a_sc = score_rows(B["model"], B["anchor"])
    d_offset = float(np.log10(theta0) - np.log10(np.quantile(a_sc, SETTINGS["q_bulk"])))
    print(f"theta0 {theta0:.5e}, offset d {d_offset:.4f} decades, k_mad {k_mad:.3f}", flush=True)

    thr, cnt = {}, {}
    for name, spec in ARMS.items():
        thr[name] = policy_trace(scores, iid, theta0, d_offset, k_mad, spec)
        ids, cnt[name] = interval_counts(scores, y, iid, thr[name])
        assert (cnt[name].sum(1) == cnt["S_static"].sum(1)).all() if "S_static" in cnt else True
    days = np.array([m.day.to_numpy()[np.searchsorted(iid, i, "left")] for i in ids])
    day_list = sorted(set(days))

    # ---- structural checks on the policies
    lt0 = np.log10(theta0)
    for name in ("D_adaptive", "D2_median_mad_limited"):
        lt = np.log10(thr[name])
        step = np.abs(np.diff(lt))
        ok = step.max() <= SETTINGS["delta_decades"] + 1e-9
        checks.append({"check": f"V4 rate limit respected ({name})", "ok": bool(ok), "detail": f"largest step {step.max():.4f} decades per interval, limit {SETTINGS['delta_decades']}"})
        print(("PASS " if ok else "FAIL ") + checks[-1]["check"] + ": " + checks[-1]["detail"])
    for name in ARMS:
        lt = np.log10(thr[name])
        ok = lt.min() >= lt0 + SETTINGS["floor_decades"] - 1e-9 and lt.max() <= lt0 + SETTINGS["ceil_decades"] + 1e-9
        checks.append({"check": f"V5 clamp respected ({name})", "ok": bool(ok), "detail": f"log10 range {lt.min():.3f} .. {lt.max():.3f} (allowed {lt0 + SETTINGS['floor_decades']:.3f} .. {lt0 + SETTINGS['ceil_decades']:.3f})"})
        print(("PASS " if ok else "FAIL ") + checks[-1]["check"] + ": " + checks[-1]["detail"])
    ok = (np.log10(thr["S_static"]) == lt0).all() or np.allclose(thr["S_static"], theta0)
    checks.append({"check": "V6 static arm threshold constant", "ok": bool(ok), "detail": "all intervals use theta0"})
    print(("PASS " if ok else "FAIL ") + "V6 static arm threshold constant")
    # label-independence is structural: policy_trace has no label argument; a recomputation must reproduce the trace exactly
    perm_trace = policy_trace(scores, iid, theta0, d_offset, k_mad, ARMS["D_adaptive"])
    ok = np.array_equal(perm_trace, thr["D_adaptive"])
    checks.append({"check": "V7 policy trace is a pure function of the scores", "ok": bool(ok), "detail": "recomputed trace identical (the function has no label argument)"})
    print(("PASS " if ok else "FAIL ") + "V7 policy trace deterministic function of scores")

    # ---- results
    out = {"settings": SETTINGS | {"theta0": theta0, "offset_d_decades": d_offset, "k_mad": k_mad}, "arms": {}, "checks": checks,
           "interval_ids": [int(i) for i in ids], "interval_days": list(days)}
    static_overall = metrics(*[int(v) for v in cnt["S_static"].sum(0)])
    per_day_static = {d: metrics(*[int(v) for v in cnt["S_static"][days == d].sum(0)]) for d in day_list}
    shifted = [d for d in day_list if per_day_static[d]["fpr"] > 0.5]
    stable = [d for d in day_list if d not in shifted]
    out["shifted_days"], out["stable_days"] = shifted, stable
    out["declared_shifted_day_rule_matches"] = shifted
    warm_med = B["info"]["val_median_error"]
    ratio = {d: float(np.median(scores[(m.day.to_numpy() == d) & (y == 0)]) / warm_med) for d in day_list}
    bulk = [d for d in day_list if ratio[d] >= 10]
    other = [d for d in day_list if d not in bulk]
    out["posthoc"] = {"note": "descriptive grouping added after the declared rule matched no day; uses labels; not tuned to any arm", "benign_median_ratio_to_warmup": ratio,
                      "bulk_shifted_days": bulk, "other_days": other, "arms": {}}
    rows = []
    for name in ARMS:
        c = cnt[name]
        per_day = {d: metrics(*[int(v) for v in c[days == d].sum(0)]) | {"median_threshold": float(np.median(thr[name][days == d]))} for d in day_list}
        grp = {"all": metrics(*[int(v) for v in c.sum(0)]),
               "shifted_days": metrics(*[int(v) for v in c[np.isin(days, shifted)].sum(0)]),
               "stable_days": metrics(*[int(v) for v in c[np.isin(days, stable)].sum(0)])}
        rec = {}
        for d in shifted:                                     # intervals (from the start of the day) until FPR <= 10%
            idx = np.flatnonzero(days == d)
            fprs = [metrics(*c[i])["fpr"] for i in idx]
            first = next((j + 1 for j, v in enumerate(fprs) if v <= 0.10), None)
            rec[d] = {"intervals_in_day": len(idx), "first_interval_fpr_le_10pct": first, "fpr_by_interval": [float(v) for v in fprs]}
        out["posthoc"]["arms"][name] = {"bulk_shifted": metrics(*[int(v) for v in c[np.isin(days, bulk)].sum(0)]),
                                        "other_days": metrics(*[int(v) for v in c[np.isin(days, other)].sum(0)])}
        out["arms"][name] = {"groups": grp, "per_day": per_day, "recovery_on_shifted_days": rec, "threshold_trace": [float(v) for v in thr[name]]}
        for i in range(len(ids)):
            rows.append({"arm": name, "interval": int(ids[i]), "day": days[i], "n": int(c[i].sum()), "tp": int(c[i][0]), "fp": int(c[i][1]),
                         "fn": int(c[i][2]), "tn": int(c[i][3]), "threshold": float(thr[name][i])})
    out["bootstrap_difference_to_static_mean_lo_hi"] = block_boot(cnt)
    # ---- ranking quality of the frozen scores (identical for every arm)
    from sklearn.metrics import average_precision_score, roc_auc_score
    out["frozen_score_auc_per_day"] = {}
    for d in day_list:
        mk = m.day.to_numpy() == d
        if y[mk].sum() >= 100 and (1 - y[mk]).sum() >= 100:
            out["frozen_score_auc_per_day"][d] = {"roc_auc": float(roc_auc_score(y[mk], scores[mk])), "pr_auc": float(average_precision_score(y[mk], scores[mk]))}
    ok = all(abs(out["frozen_score_auc_per_day"][d]["roc_auc"] - p7["auc_per_day"]["static"][d]["roc_auc"]) < 1e-3 for d in out["frozen_score_auc_per_day"] if d in p7["auc_per_day"]["static"])
    checks.append({"check": "V8 per-day ROC-AUC equals Phase 7 static", "ok": bool(ok), "detail": "tolerance 1e-3"})
    print(("PASS " if ok else "FAIL ") + "V8 per-day ROC-AUC equals Phase 7 static")
    # ---- diagnostics that read labels (reporting only)
    oracle, bulk_attack = [], []
    starts, ends = np.searchsorted(iid, ids, "left"), np.searchsorted(iid, ids, "right")
    for a, b in zip(starts, ends):
        s_i, y_i = scores[a:b], y[a:b].astype(bool)
        oracle.append(float(np.quantile(s_i[~y_i], 0.99)) if (~y_i).sum() > 1000 else float("nan"))
        cut = np.quantile(s_i, 0.5)
        bulk_attack.append(float(y_i[s_i <= cut].mean()))
    out["diagnostics_use_labels"] = {"oracle_benign_p99_per_interval": oracle, "attack_share_in_lower_half_of_scores": bulk_attack,
                                     "mean_attack_share_lower_half": float(np.mean(bulk_attack)), "max_attack_share_lower_half": float(np.max(bulk_attack))}
    # ---- Phase 7 reference results (not re-run)
    out["phase7_reference"] = {arm: {"overall": p7["arms"][arm]["overall"], "per_day": p7["arms"][arm]["per_day"]} for arm in p7["arms"]}
    out["checks"] = checks
    json.dump(out, open(args.out_dir / "phase16_adaptive_threshold.json", "w"), indent=2, default=float)
    pd.DataFrame(rows).to_csv(args.out_dir / "phase16_intervals.csv", index=False)
    np.save(args.out_dir / "phase16_thresholds.npy", np.array([thr[a] for a in ARMS]))
    for name in ARMS:
        g = out["arms"][name]["groups"]
        g = {"all": g["all"]} | out["posthoc"]["arms"][name]
        for k, v in g.items():
            print(f"{name:24s} {k:13s} P {v['precision']:.4f} R {v['recall']:.4f} F1 {v['f1']:.4f} FPR {v['fpr']:.4f} FNR {v['fnr']:.4f}", flush=True)
    print("failed checks:", [c["check"] for c in checks if not c["ok"]] or "none")


if __name__ == "__main__":
    main()
