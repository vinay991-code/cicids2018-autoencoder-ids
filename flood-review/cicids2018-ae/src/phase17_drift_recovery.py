"""Phase 17 / Experiment H: concept-drift recovery with a supervised normality gate (NumPy training, frozen starting model).

Question: can the unsupervised autoencoder be updated after a change of benign traffic WITHOUT learning the attacks, if a
second, offline-trained supervised model must also call a record normal before it may enter the retraining buffer?

Same stream, warm-up baseline, intervals and starting threshold as Phase 7/16 (data/interim/stream_main_*): one-class
autoencoder trained on the benign records of days 02-14 and 02-15, theta0 = 99th percentile of later benign warm-up errors,
194 intervals of at most 50,000 records over 8 days. Each interval is scored and flagged with the model and threshold that
exist BEFORE it (prequential), then used for the update. PyTorch cannot be loaded on this machine, so the fine-tuning is the
NumPy implementation in src/nptrain.py (checked against the saved PyTorch models and by gradient checks first).

Arms (all settings below fixed before the first arm was run; this is the first and only evaluated run):
  S   frozen autoencoder, theta0 (identical to Phase 16 arm S).
  N   adaptive autoencoder, no supervised gate: a sampled record may enter the retraining buffer if the arm itself scores it
      <= its current threshold (the admission rule of Phase 7 arm C).
  Gm  N plus the gate: the offline-trained supervised autoencoder of Phase 4 (models/ae_supervised.pt) must also give
      probability < 0.5 (its fixed rule). It was trained on the Phase 2 training partition, which contains records of every
      day of the stream: it already knows these attack campaigns. This is the requested gate, but it is an OPTIMISTIC one
      (see the findings); the diagnostics report how well it blocks attacks on the test partition, which it never saw.
  Gu  N plus the gate of the Phase 6 supervised model that never saw Bot, DDoS-HOIC, SSH-BruteForce and DoS-Hulk (its scaler is
      refit without them, exactly as in src/phase6_unseen.py): these three types present in the stream (Hulk 02-16, HOIC
      02-21, Bot 03-02) are the "unseen attacks" of this experiment, all others "known".
Update rules (same as Phase 7 arm C, all arms): every 3 intervals, 3 epochs of Adam (lr 5e-4, batch 1,024) on the admitted
records of the last 12 intervals (at most 10,000 sampled per interval, so at most 120,000) plus 25% replay of clean warm-up
records; at least 5,000 admitted records are required; an update is rejected if the mean error on a fixed clean anchor set
(50,000 benign warm-up records) exceeds 3 times its initial value. Threshold of the adaptive arms: theta = c x (99th percentile
of the anchor errors under the current model), c fixed so that the starting threshold equals theta0 (Phase 8 and 9 used the
95th percentile; Phase 7 and 16 used theta0, which is kept here so that S equals Phase 7's static arm).
No stream label controls a decision: the gate probabilities depend on features only, admission uses scores and gate
probabilities, and labels are used after the fact for counts and for diagnostics (a check reruns the arms with all labels
replaced by zeros and requires identical models, thresholds and flags).
Success criteria, declared in advance, per adaptive arm: (a) within-day ROC-AUC on the bulk-shifted days (02-16, 02-21) is at
least 0.5 and higher than the frozen model's; (b) on 02-20 and 03-02 it is not more than 0.05 below the frozen model's;
(c) overall F1 at the arm's own threshold is not lower than the frozen model's.
Recovery time: on each bulk-shifted day, the number of intervals from the start of the day until the interval-level ROC-AUC is
at least 0.5 for three consecutive intervals (intervals need at least 1,000 benign and 1,000 attack records).
Phase 7/8/9 results are kept as they are and are not modified; Phase 7 arm C is quoted for reference.

Usage: python src/phase17_drift_recovery.py --check-only   (validation checks and a smoke test; no arm is evaluated)
       python src/phase17_drift_recovery.py                (full run)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nptrain as nt  # noqa: E402
from phase16_adaptive_threshold import (_load_npmodels, block_boot, interval_counts, interval_ids, load_baseline, metrics)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "interim"
HELD = ["Bot", "DDoS-HOIC", "SSH-BruteForce", "DoS-Hulk"]                    # Phase 6 hold-out
CFG = dict(interval_records=50_000, sample=10_000, retrain_every=3, buffer_intervals=12, replay_frac=0.25, ft_epochs=3, ft_lr=5e-4,
           ft_batch=1024, anchor_tol=3.0, min_trusted=5000, anchor_q=0.99, gate_tau=0.5, seed=42, recovery_auc=0.5, recovery_run=3,
           recovery_min_class=1000, bulk_ratio=10.0)
ARMS = {"S_frozen": dict(update=False, gate=None), "N_adaptive": dict(update=True, gate=None),
        "Gm_gated_main": dict(update=True, gate="main"), "Gu_gated_unseen": dict(update=True, gate="unseen")}


# ------------------------------------------------------------------ gate models (features only, labels never involved)
def unseen_refit():
    path = CACHE / "unseen_refit.npz"
    if path.exists():
        z = np.load(path)
        return z["lo"], z["span1"]
    sc = json.load(open(ROOT / "data/processed/scaler.json"))
    nnum = len(sc["model_columns"])
    lmin0, lmax0 = np.array(sc["log_min"]), np.array(sc["log_max"])
    span0 = np.where(lmax0 > lmin0, lmax0 - lmin0, 1.0)
    Xtr = np.load(ROOT / "data/processed/X_train.npy", mmap_mode="r")
    att = pd.read_pickle(ROOT / "data/processed/meta_train.pkl").attack_type.to_numpy()
    keep = np.flatnonzero(~np.isin(att, HELD))
    lo, hi = np.full(nnum, np.inf), np.full(nnum, -np.inf)
    for a in range(0, len(keep), 500_000):          # identical to src/phase6_unseen.py
        ch = np.asarray(Xtr[keep[a:a + 500_000]], dtype=np.float64)
        L = ch[:, :nnum] * span0 + lmin0
        lo, hi = np.minimum(lo, L.min(0)), np.maximum(hi, L.max(0))
    span1 = np.where(hi > lo, hi - lo, 1.0)
    np.savez(path, lo=lo, span1=span1)
    return lo, span1


def rescale_unseen(X, lo, span1):
    sc = json.load(open(ROOT / "data/processed/scaler.json"))
    nnum = len(sc["model_columns"])
    lmin0, lmax0 = np.array(sc["log_min"]), np.array(sc["log_max"])
    span0 = np.where(lmax0 > lmin0, lmax0 - lmin0, 1.0)
    out = np.array(X, dtype=np.float32)
    out[:, :nnum] = (((np.asarray(X[:, :nnum], dtype=np.float64) * span0 + lmin0) - lo) / span1).astype(np.float32)
    return out


def gate_probs(model, X, rescale=None, chunk=262_144):
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), chunk):
        xb = np.ascontiguousarray(X[i:i + chunk], dtype=np.float32)
        if rescale is not None:
            xb = rescale(xb)
        logit = model(xb)[0]
        with np.errstate(over="ignore"):
            out[i:i + chunk] = 1.0 / (1.0 + np.exp(-np.clip(logit, -80, 80)))
    return out


# ------------------------------------------------------------------ one arm over the stream
def run_arm(spec, X, y_diag, iid, theta0, params0, anchor, replay, p_gate, cfg=CFG, log=None):
    """Returns dict with per-record scores and per-interval thresholds, plus the update log. `y_diag` is read ONLY for the
    contamination diagnostic (attack share of admitted records); no decision depends on it."""
    ids = np.unique(iid)
    starts, ends = np.searchsorted(iid, ids, "left"), np.searchsorted(iid, ids, "right")
    params = nt.copy_params(params0)
    a0 = nt.recon_error(params0, anchor)
    anchor_ref = float(a0.mean())
    c = theta0 / float(np.quantile(a0, cfg["anchor_q"]))
    thr = theta0
    N = int(ends[-1])
    scores = np.empty(N, np.float32)
    thr_trace, buffer, updates = [], [], []
    for t, (a, b) in enumerate(zip(starts, ends)):
        Xi = np.asarray(X[a:b], dtype=np.float32)
        s = nt.recon_error(params, Xi)
        scores[a:b] = s
        thr_trace.append(thr)
        if not spec["update"]:
            continue
        samp = np.sort(np.random.default_rng([cfg["seed"], t]).choice(b - a, min(cfg["sample"], b - a), replace=False))
        xs, ss = Xi[samp], s[samp]
        admit = ss <= thr                                              # the arm's own normality call
        if p_gate is not None:
            admit &= p_gate[a:b][samp] < cfg["gate_tau"]               # and the supervised model's
        buffer.append((xs[admit], y_diag[a:b][samp][admit], int(len(samp)), int(admit.sum())))
        buffer = buffer[-cfg["buffer_intervals"]:]
        if (t + 1) % cfg["retrain_every"] == 0:
            trusted = np.concatenate([e[0] for e in buffer]) if buffer else np.empty((0, Xi.shape[1]), np.float32)
            att = np.concatenate([e[1] for e in buffer]) if buffer else np.empty(0)
            ev = {"interval_index": t, "admitted_records": int(len(trusted)), "sampled_records": int(sum(e[2] for e in buffer)),
                  "attack_share_in_buffer_diagnostic": float(att.mean()) if len(att) else None, "attack_records_in_buffer_diagnostic": int(att.sum())}
            if len(trusted) >= cfg["min_trusted"]:
                rp_idx = np.random.default_rng([cfg["seed"], t, 1]).choice(len(replay), min(int(cfg["replay_frac"] * len(trusted)), len(replay)), replace=False)
                new, before, after = nt.finetune(params, np.concatenate([trusted, replay[rp_idx]]), cfg["ft_epochs"], cfg["ft_lr"], cfg["ft_batch"], cfg["seed"] + t)
                an = nt.recon_error(new, anchor)
                ratio = float(an.mean() / anchor_ref)
                ev.update(train_loss_before=before, train_loss_after=after, anchor_error_ratio=ratio, accepted=bool(ratio <= cfg["anchor_tol"]))
                if ev["accepted"]:
                    params, thr = new, c * float(np.quantile(an, cfg["anchor_q"]))
            else:
                ev.update(accepted=False, skipped="fewer than min_trusted admitted records")
            updates.append(ev)
        if log and (t + 1) % 24 == 0:
            log(f"    interval {t+1}/{len(ids)}  thr {thr:.3e}  updates {len(updates)} (accepted {sum(u.get('accepted', False) for u in updates)})")
    return {"scores": scores, "thr": np.array(thr_trace), "updates": updates, "params": params}


def params_hash(params):
    h = hashlib.sha256()
    for w, b in params:
        h.update(np.ascontiguousarray(w).tobytes()); h.update(np.ascontiguousarray(b).tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ evaluation helpers
def auc_safe(y, s, min_class=100):
    from sklearn.metrics import roc_auc_score
    n1 = int(y.sum())
    return float(roc_auc_score(y, s)) if n1 >= min_class and len(y) - n1 >= min_class else float("nan")


def interval_aucs(scores, y, iid, min_class):
    ids = np.unique(iid)
    st, en = np.searchsorted(iid, ids, "left"), np.searchsorted(iid, ids, "right")
    return np.array([auc_safe(y[a:b], scores[a:b], min_class) for a, b in zip(st, en)])


def recovery_time(aucs, days, day, cfg=CFG):
    idx = np.flatnonzero(days == day)
    v = aucs[idx]
    for k in range(len(v) - cfg["recovery_run"] + 1):
        w = v[k:k + cfg["recovery_run"]]
        if np.all(~np.isnan(w)) and np.all(w >= cfg["recovery_auc"]):
            return k + 1                       # 1-based: intervals from the start of the day until the run begins
    return None


def validation_checks(B, scores_frozen, log):
    """Compares the NumPy models with the saved PyTorch outputs. Returns list of dicts."""
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        log(("PASS " if ok else "FAIL ") + name + ": " + detail)

    npm = _load_npmodels()
    n = 200_000
    Xt = np.load(ROOT / "data/processed/X_test.npy", mmap_mode="r")[:n]
    # C1 main supervised model vs saved torch probabilities (Phase 4)
    sup = npm.load_model(ROOT / "models" / "ae_supervised.pt", supervised=True)
    ref = np.load(ROOT / "results" / "scores_supervised_test.npy", mmap_mode="r")[:n]
    d = float(np.max(np.abs(gate_probs(sup, Xt) - ref)))
    add("C1 numpy supervised gate reproduces saved PyTorch probabilities (Phase 4, first 200,000 test rows)", d < 1e-5, f"max abs difference {d:.2e} (tolerance 1e-5)")
    # C2 unseen supervised model (with the refit scaler) vs saved torch probabilities (Phase 6)
    lo, span1 = unseen_refit()
    sup_u = npm.load_model(ROOT / "models" / "ae_unseen_supervised.pt", supervised=True)
    ref_u = np.load(ROOT / "results" / "scores_unseen_supervised_test.npy", mmap_mode="r")
    same_len = len(ref_u) == len(np.load(ROOT / "data/processed/X_test.npy", mmap_mode="r"))
    d = float(np.max(np.abs(gate_probs(sup_u, Xt, rescale=lambda x: rescale_unseen(x, lo, span1)) - ref_u[:n]))) if same_len else float("inf")
    add("C2 numpy unseen-supervised gate with refit scaler reproduces saved PyTorch probabilities (Phase 6, first 200,000 test rows)", d < 1e-5,
        f"max abs difference {d:.2e} (tolerance 1e-5); saved scores cover the full test partition: {same_len}")
    # C3 baseline autoencoder vs Phase 7 (warm-up threshold exactly, static counts)
    ae = nt.from_state_dict(npm.load_state_dict(CACHE / "stream_main_model.pt"))
    anchor = np.load(CACHE / "stream_main_anchor.npy")
    old = npm.load_model(CACHE / "stream_main_model.pt", supervised=False)
    a_new = nt.recon_error(ae, anchor)
    a_old = ((old(anchor) - anchor) ** 2).mean(axis=1)
    d = float(np.max(np.abs(a_new - a_old)))
    add("C3 nptrain forward equals the verified npmodels forward on the anchor set", d < 1e-6, f"max abs difference {d:.2e}")
    return checks, ae, anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results_drift")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--smoke-intervals", type=int, default=12)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    t0 = time.time()

    B = load_baseline()
    m = B["meta"].iloc[B["e0"]:].reset_index(drop=True)
    X = B["X"][B["e0"]:]
    iid = interval_ids(m, CFG["interval_records"])
    y = m.label_binary.to_numpy()
    theta0 = B["info"]["thr0"]
    frozen = np.load(CACHE / "stream_main_frozen_scores.npy")             # Phase 16 cache: frozen model scores of the evaluation stream
    checks, ae0, anchor = validation_checks(B, frozen, log)
    replay = np.load(CACHE / "stream_main_replay.npy")

    # gate probabilities for the whole evaluation stream (features only)
    npm = _load_npmodels()
    gp = {}
    lo, span1 = unseen_refit()
    for name, path, rs in (("main", "ae_supervised.pt", None), ("unseen", "ae_unseen_supervised.pt", lambda x: rescale_unseen(x, lo, span1))):
        cp = CACHE / f"stream_main_gate_{name}.npy"
        if cp.exists() and not args.rescore:
            gp[name] = np.load(cp)
        else:
            gp[name] = gate_probs(npm.load_model(ROOT / "models" / path, supervised=True), X, rescale=rs)
            np.save(cp, gp[name])
    log(f"gate probabilities ready for {len(gp['main']):,} records  [{time.time()-t0:.0f}s]")

    # C4 frozen scores of this module equal the Phase 16 cache; C5 smoke: determinism and label independence
    n_smoke = int(np.searchsorted(iid, np.unique(iid)[args.smoke_intervals - 1], "right"))
    sm = lambda spec, gate, yy: run_arm(spec, X[:n_smoke], yy[:n_smoke], iid[:n_smoke], theta0, ae0, anchor, replay, gp[gate][:n_smoke] if gate else None)
    spec_g = ARMS["Gm_gated_main"]
    r1, r2 = sm(spec_g, "main", y), sm(spec_g, "main", y)
    r3 = sm(spec_g, "main", np.zeros_like(y))
    ok = params_hash(r1["params"]) == params_hash(r2["params"]) and np.array_equal(r1["scores"], r2["scores"])
    checks.append({"check": "C4 determinism: two runs give identical parameters and scores", "ok": ok, "detail": f"{args.smoke_intervals} intervals, parameter hash {params_hash(r1['params'])}"})
    log(("PASS " if ok else "FAIL ") + checks[-1]["check"] + ": " + checks[-1]["detail"])
    ok = params_hash(r1["params"]) == params_hash(r3["params"]) and np.array_equal(r1["scores"], r3["scores"]) and np.array_equal(r1["thr"], r3["thr"])
    checks.append({"check": "C5 label independence: all labels replaced by zeros gives identical parameters, scores and thresholds", "ok": ok,
                   "detail": f"{args.smoke_intervals} intervals, accepted updates {sum(u.get('accepted', False) for u in r1['updates'])}"})
    log(("PASS " if ok else "FAIL ") + checks[-1]["check"] + ": " + checks[-1]["detail"])
    s_chk = run_arm(ARMS["S_frozen"], X[:n_smoke], y[:n_smoke], iid[:n_smoke], theta0, ae0, anchor, replay, None)
    d = float(np.max(np.abs(s_chk["scores"] - frozen[:n_smoke])))
    ok = d < 1e-6
    checks.append({"check": "C6 frozen arm scores equal the Phase 16 cached scores", "ok": ok, "detail": f"max abs difference {d:.2e} on {n_smoke:,} records"})
    log(("PASS " if ok else "FAIL ") + checks[-1]["check"] + ": " + checks[-1]["detail"])
    ups = r1["updates"]
    ok = len(ups) > 0 and all(u.get("train_loss_after", 0) < u.get("train_loss_before", 1) for u in ups if "train_loss_after" in u)
    checks.append({"check": "C7 every update lowered the reconstruction loss on its own training batch", "ok": ok, "detail": f"{len(ups)} update attempts in the smoke run"})
    log(("PASS " if ok else "FAIL ") + checks[-1]["check"] + ": " + checks[-1]["detail"])
    if not all(c["ok"] for c in checks):
        json.dump({"checks": checks}, open(args.out_dir / "phase17_checks_FAILED.json", "w"), indent=2)
        raise SystemExit("validation failed: numpy and PyTorch / determinism checks did not pass; the experiment was NOT run")
    if args.check_only:
        log(f"check-only run done [{time.time()-t0:.0f}s]; no arm was evaluated")
        return

    # ---------------------------------------------------------------- full run
    results, rec = {}, {}
    for name, spec in ARMS.items():
        t1 = time.time()
        log(f"arm {name} ...")
        rec[name] = run_arm(spec, X, y, iid, theta0, ae0, anchor, replay, gp[spec["gate"]] if spec["gate"] else None, log=log)
        log(f"  done [{time.time()-t1:.0f}s]; accepted updates {sum(u.get('accepted', False) for u in rec[name]['updates'])} of {len(rec[name]['updates'])}")
    ids = np.unique(iid)
    days_iv = np.array([m.day.to_numpy()[np.searchsorted(iid, i, "left")] for i in ids])
    day_list = sorted(set(days_iv))
    day_arr, atype = m.day.to_numpy(), m.attack_type.to_numpy()
    warm_med = B["info"]["val_median_error"]
    ratio = {d: float(np.median(rec["S_frozen"]["scores"][(day_arr == d) & (y == 0)]) / warm_med) for d in day_list}
    bulk = [d for d in day_list if ratio[d] >= CFG["bulk_ratio"]]
    other = [d for d in day_list if d not in bulk]
    unseen_stream = sorted(set(atype) & set(HELD))
    known_stream = sorted(set(atype) - set(HELD) - {"Benign"})
    sizes = np.diff(np.r_[np.searchsorted(iid, ids, "left"), len(iid)])
    out = {"cfg": CFG, "arms_spec": ARMS, "checks": checks, "bulk_shifted_days": bulk, "other_days": other, "benign_median_ratio_to_warmup": ratio,
           "unseen_types_in_stream": unseen_stream, "known_types_in_stream": known_stream, "arms": {}}
    cnt = {}
    for name, r in rec.items():
        ids_, cnt[name] = interval_counts(r["scores"], y, iid, r["thr"])
        flags = r["scores"] > np.repeat(r["thr"], sizes)
        aucs = interval_aucs(r["scores"], y, iid, CFG["recovery_min_class"])
        grp = {"all": metrics(*[int(v) for v in cnt[name].sum(0)]),
               "bulk_shifted_days": metrics(*[int(v) for v in cnt[name][np.isin(days_iv, bulk)].sum(0)]),
               "other_days": metrics(*[int(v) for v in cnt[name][np.isin(days_iv, other)].sum(0)])}
        per_day = {}
        for d in day_list:
            mk = day_arr == d
            kn = mk & ((y == 0) | np.isin(atype, known_stream))
            un = mk & ((y == 0) | np.isin(atype, unseen_stream))
            per_day[d] = metrics(*[int(v) for v in cnt[name][days_iv == d].sum(0)]) | {
                "roc_auc_all": auc_safe(y[mk], r["scores"][mk]), "roc_auc_benign_vs_known": auc_safe((y[kn]), r["scores"][kn]),
                "roc_auc_benign_vs_unseen": auc_safe(y[un] if un.any() else y[:0], r["scores"][un]) if un.any() else float("nan"),
                "median_threshold": float(np.median(r["thr"][days_iv == d])), "attack_types": sorted(set(atype[mk & (y == 1)]))}
        att_groups = {"known_attacks": (y == 1) & np.isin(atype, known_stream), "unseen_attacks": (y == 1) & np.isin(atype, unseen_stream)}
        group_recall = {g: {"n": int(mk.sum()), "flagged_share": float(flags[mk].mean())} for g, mk in att_groups.items()}
        per_type = {t: {"n": int((atype == t).sum()), "flagged_share": float(flags[atype == t].mean())} for t in sorted(set(atype) - {"Benign"})}
        benign_fpr = {"bulk_shifted_days": float(flags[(y == 0) & np.isin(day_arr, bulk)].mean()), "other_days": float(flags[(y == 0) & np.isin(day_arr, other)].mean())}
        out["arms"][name] = {"groups": grp, "per_day": per_day, "group_recall": group_recall, "per_type_flagged_share": per_type, "benign_fpr": benign_fpr,
                             "interval_auc": [None if np.isnan(v) else float(v) for v in aucs],
                             "recovery_intervals": {d: recovery_time(aucs, days_iv, d) for d in bulk},
                             "threshold_trace": [float(v) for v in r["thr"]], "updates": r["updates"],
                             "accepted_updates": int(sum(u.get("accepted", False) for u in r["updates"])), "update_attempts": len(r["updates"])}
    out["bootstrap_difference_to_frozen_mean_lo_hi"] = block_boot(cnt, base="S_frozen")
    # success criteria
    fz = out["arms"]["S_frozen"]["per_day"]
    out["criteria"] = {}
    for name in ("N_adaptive", "Gm_gated_main", "Gu_gated_unseen"):
        pdv = out["arms"][name]["per_day"]
        a = all(pdv[d]["roc_auc_all"] >= 0.5 and pdv[d]["roc_auc_all"] > fz[d]["roc_auc_all"] for d in bulk)
        b = all(pdv[d]["roc_auc_all"] >= fz[d]["roc_auc_all"] - 0.05 for d in ("2018-02-20", "2018-03-02"))
        c_ = out["arms"][name]["groups"]["all"]["f1"] >= out["arms"]["S_frozen"]["groups"]["all"]["f1"]
        out["criteria"][name] = {"a_recovers_on_shifted_days": bool(a), "b_keeps_ranking_on_attack_days": bool(b), "c_overall_f1_not_lower": bool(c_),
                                 "all_three": bool(a and b and c_)}
    # gate diagnostics (labels used, reporting only)
    split = m.split.to_numpy()
    out["gate_diagnostics_use_labels"] = {}
    for g, p in gp.items():
        blocked = p >= CFG["gate_tau"]
        d_ = {"benign_blocked_share": float(blocked[y == 0].mean()), "attack_blocked_share_all_records": float(blocked[y == 1].mean()),
              "attack_blocked_share_test_partition_only": float(blocked[(y == 1) & (split == 2)].mean()),
              "attack_blocked_share_train_partition_only": float(blocked[(y == 1) & (split == 0)].mean()),
              "attack_blocked_by_type": {t: float(blocked[atype == t].mean()) for t in sorted(set(atype) - {"Benign"})}}
        out["gate_diagnostics_use_labels"][g] = d_
    pq = json.load(open(ROOT / "results" / "phase7_main.json"))
    out["phase7_reference"] = {"C_adaptive_baseline": pq["arms"]["C_adaptive_baseline"]["overall"], "A_static": pq["arms"]["A_static"]["overall"]}
    out["checks"] = checks
    json.dump(out, open(args.out_dir / "phase17_drift_recovery.json", "w"), indent=2, default=float)
    for name in ARMS:
        g = out["arms"][name]["groups"]["all"]
        log(f"{name:18s} P {g['precision']:.4f} R {g['recall']:.4f} F1 {g['f1']:.4f} FPR {g['fpr']:.4f} FNR {g['fnr']:.4f} | accepted {out['arms'][name]['accepted_updates']}/{out['arms'][name]['update_attempts']}")
    log("criteria: " + json.dumps(out["criteria"]))
    np.savez_compressed(args.out_dir / "phase17_thresholds.npz", **{k: v["thr"] for k, v in rec.items()})
    log(f"finished [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
