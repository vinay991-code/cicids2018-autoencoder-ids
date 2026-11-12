"""Streaming core shared by the adaptive-baseline experiments (Phases 7-9).

The stream is processed in fixed intervals. For every interval, each arm first scores and flags the interval with the
model and threshold it already has (nothing from the interval is used before predictions are made), then updates its
state from that interval. Attack labels are read only to compute evaluation counts and contamination diagnostics; the
update logic never touches them (labels enter only through `_count` and through the `yy` diagnostic arrays).

Threshold modes: "fixed"; "robust" (median + k*MAD of recent log10 scores, Phase 7); "anchor_q" (the q-quantile of the
reconstruction errors of a fixed clean anchor set under the current model, recomputed after every model update).
Admission modes for the trusted retraining buffer:
  "all"       unrestricted adaptation: every sampled record is admitted.
  "normal"    records the arm scored as normal (score <= threshold).
  "margin"    only records clearly below the threshold (score <= margin * threshold).
  "gated"     margin + temporal consistency + quarantine (see `_gate`).
  "relative"  (Phase 9, drift-triggered arms) at retraining time the highest-scoring `1 - q_rel` of the recent samples are
              quarantined, i.e. only records typical for the CURRENT regime are used.
Update triggers: "periodic" (every `retrain_every` intervals, plus on a gate release) or "drift" (after an alarm of the
label-free DriftMonitor; arms with admission "relative" first confirm that the deviation persists in the next interval).
Reconstruction error is used only as an ordinal anomaly score, not as a calibrated probability.
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch

from models import recon_error


def robust_log_threshold(scores, k, floor=1e-12):
    """Threshold = median + k * (1.4826 * MAD) of the log10 scores."""
    l = np.log10(np.maximum(scores, floor))
    med = np.median(l)
    mad = 1.4826 * np.median(np.abs(l - med))
    return float(10 ** (med + k * max(mad, 1e-3)))


def calibrate_k(benign_scores, thr, floor=1e-12):
    l = np.log10(np.maximum(benign_scores, floor))
    med = np.median(l)
    mad = 1.4826 * np.median(np.abs(l - med))
    return float((np.log10(thr) - med) / mad)


def median_log(scores, floor=1e-12):
    return float(np.median(np.log10(np.maximum(scores, floor))))


def finetune(model, X, epochs, lr, batch, seed):
    """Continue training a copy of `model` on X (float32) with Adam + MSE for a fixed number of epochs."""
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.from_numpy(X)
    g = torch.Generator().manual_seed(seed)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            xb = Xt[perm[i:i + batch]]
            opt.zero_grad()
            ((m(xb) - xb) ** 2).mean().backward()
            opt.step()
    m.eval()
    return m


@dataclass
class ArmCfg:
    name: str
    adapt_threshold: bool = False      # Phase 7 shorthand for thr_mode="robust"
    retrain: bool = False
    thr_mode: str = ""                 # "", "fixed", "robust", "anchor_q"
    admission: str = "normal"          # "all" | "normal" | "margin" | "gated" | "relative"
    trigger: str = "periodic"          # "periodic" | "drift"

    def mode(self):
        return self.thr_mode or ("robust" if self.adapt_threshold else "fixed")


class Arm:
    def __init__(self, cfg: ArmCfg, model, thr, n_types):
        self.cfg, self.model, self.thr = cfg, model, thr
        self.window, self.buffer = [], []      # buffer entries: (features, admit mask, attack flags [diagnostic], scores)
        self.counts, self.thrs, self.by_type_flagged = [], [], []
        self.retrain_log, self.gate_log, self.drift_log = [], [], []
        self.n_types = n_types
        self.level_ref, self.persist, self.pending = 0.0, 0, []
        self.response_left, self.pending_confirm = 0, False


def _gate(arm, t, xs, ss, fl, yy, g):
    """Confidence-aware admission for one interval sample (Phase 8). Returns True if a regime change was accepted.
    1. Margin: a record is a trusted candidate only if score <= g.margin * threshold ("clearly normal").
    2. Quarantine: records scored above the threshold are never admitted while the interval looks consistent.
    3. Temporal consistency: the interval is consistent if its median log10 score is within g.level_tol decades of
       the running reference level of accepted intervals and its share of flagged records is at most g.susp_max.
       An inconsistent interval is held in quarantine; if inconsistency persists for g.persist consecutive intervals it
       is treated as a lasting change of the traffic: all held records are released into the buffer and the reference
       level is reset."""
    lvl = median_log(ss)
    dev = abs(lvl - arm.level_ref) > g["level_tol"] or float(fl.mean()) > g["susp_max"]
    ev = {"interval": t, "level": lvl, "reference": arm.level_ref, "flagged_share": float(fl.mean()), "deviant": bool(dev),
          "released": False, "quarantined_suspicious": int(fl.sum()) if not dev else int(len(xs))}
    released = False
    if not dev:
        arm.buffer.append((xs, ss <= g["margin"] * arm.thr, yy, ss))
        arm.level_ref = (1 - g["ewma"]) * arm.level_ref + g["ewma"] * lvl
        arm.persist, arm.pending = 0, []
    else:
        arm.persist += 1
        arm.pending.append((xs, yy, ss))
        if arm.persist >= g["persist"]:
            for px, py, ps in arm.pending:
                arm.buffer.append((px, np.ones(len(px), bool), py, ps))
            arm.pending, arm.persist, arm.level_ref = [], 0, lvl
            ev["released"] = released = True
    arm.gate_log.append(ev)
    return released


def run_stream(X, y, atype, day, interval_id, model0, thr0, k, arms, cfg, anchor, replay, monitor=None, log=print):
    """X: stream features (memmap ok), already restricted to the evaluation part; y/atype/day/interval_id aligned arrays.
    cfg keys: sample, window, retrain_every, buffer_intervals, replay_frac, ft_epochs, ft_lr, ft_batch, anchor_tol, seed,
    min_trusted, n_types, anchor_q, gate (margin, level_tol, susp_max, persist, ewma), and for drift-triggered arms:
    response (retraining rounds after an alarm), recent_intervals, q_rel. `monitor`: a DriftMonitor (fed with the
    unmodified initial model's scores of every interval)."""
    rng = np.random.default_rng(cfg["seed"])
    n_types = int(cfg.get("n_types", int(atype.max()) + 1))
    ids = np.unique(interval_id)
    starts = np.searchsorted(interval_id, ids, side="left")
    ends = np.searchsorted(interval_id, ids, side="right")
    anchor_s0 = recon_error(model0, anchor)
    anchor_ref = float(anchor_s0.mean())
    A = {}
    for c in arms:
        arm = Arm(c, copy.deepcopy(model0) if c.retrain else model0, thr0, n_types)
        if c.mode() == "anchor_q":
            arm.thr = float(np.quantile(anchor_s0, cfg["anchor_q"]))
        arm.level_ref = median_log(anchor_s0)
        A[c.name] = arm
    score_store = {}
    N = len(interval_id)
    meta_rows, alarms = [], []
    for t, (a, b) in enumerate(zip(starts, ends)):
        Xi, yi, ti = np.asarray(X[a:b]), y[a:b], atype[a:b]
        cache = {}
        if monitor is not None:                                  # label-free drift monitoring on the fixed sentinel
            cache["static"] = recon_error(model0, Xi)
            score_store.setdefault("static", np.full(N, np.nan, dtype=np.float32))[a:b] = cache["static"]
            new_alarms = monitor.feed(Xi, cache["static"], record_offset=int(a), interval=t)
            alarms.extend(new_alarms)
        else:
            new_alarms = []
        alarm_now = len(new_alarms) > 0
        samp_idx = np.sort(rng.choice(b - a, min(cfg["sample"], b - a), replace=False))
        for arm in A.values():
            key = arm.cfg.name if arm.cfg.retrain else "static"      # arms that never retrain share the initial model's scores
            if key not in cache:
                cache[key] = recon_error(arm.model, Xi)
                score_store.setdefault(key, np.full(N, np.nan, dtype=np.float32))[a:b] = cache[key]
            s = cache[key]
            flag = s > arm.thr
            arm.counts.append(_count(flag, yi))
            arm.thrs.append(arm.thr)
            arm.by_type_flagged.append(np.bincount(ti[flag], minlength=n_types))
            # ---- state update: uses only features, scores and the arm's own flags (never labels)
            xs, ss, fl, yy = Xi[samp_idx], s[samp_idx], flag[samp_idx], yi[samp_idx]   # yy: contamination diagnostic only
            arm.window.append(xs)
            arm.window = arm.window[-cfg["window"]:]
            released = False
            adm = arm.cfg.admission
            if adm in ("all", "relative"):
                arm.buffer.append((xs, np.ones(len(xs), bool), yy, ss))
            elif adm == "normal":
                arm.buffer.append((xs, ~fl, yy, ss))
            elif adm == "margin":
                arm.buffer.append((xs, ss <= cfg["gate"]["margin"] * arm.thr, yy, ss))
            elif adm == "gated":
                released = _gate(arm, t, xs, ss, fl, yy, cfg["gate"])
            arm.buffer = arm.buffer[-cfg["buffer_intervals"]:]
            do_retrain, reason = False, ""
            if arm.cfg.retrain and arm.cfg.trigger == "periodic":
                do_retrain, reason = ((t + 1) % cfg["retrain_every"] == 0 or released), "periodic"
            elif arm.cfg.retrain and arm.cfg.trigger == "drift":
                if arm.cfg.admission == "relative":
                    # temporal consistency: an alarm is acted on only if the deviation is still present one interval later
                    if arm.pending_confirm:
                        lvl = median_log(ss)
                        confirmed = abs(lvl - arm.level_ref) > cfg["gate"]["level_tol"] or float(fl.mean()) > cfg["gate"]["susp_max"]
                        arm.drift_log.append({"interval": t, "event": "confirmed" if confirmed else "cancelled", "level": lvl,
                                              "reference": arm.level_ref, "flagged_share": float(fl.mean())})
                        arm.pending_confirm = False
                        if confirmed:
                            arm.response_left = cfg["response"]
                    if alarm_now and arm.response_left == 0:
                        arm.pending_confirm = True
                        arm.drift_log.append({"interval": t, "event": "alarm", "monitors": sorted({m for al in new_alarms for m in al["monitors"]})})
                elif alarm_now and arm.response_left == 0:
                    arm.response_left = cfg["response"]
                    arm.drift_log.append({"interval": t, "event": "alarm", "monitors": sorted({m for al in new_alarms for m in al["monitors"]})})
                if arm.response_left > 0:
                    do_retrain, reason = True, "drift"
                    arm.response_left -= 1
            if do_retrain:
                entries = arm.buffer[-cfg["recent_intervals"]:] if arm.cfg.trigger == "drift" else arm.buffer
                if adm == "relative":
                    pooled = np.concatenate([e[3] for e in entries])
                    cut = np.quantile(pooled, cfg["q_rel"])
                    masks = [e[3] <= cut for e in entries]
                else:
                    masks = [e[1] for e in entries]
                trusted = np.concatenate([e[0][m_] for e, m_ in zip(entries, masks)]) if entries else np.empty((0, X.shape[1]), np.float32)
                att = np.concatenate([e[2][m_] for e, m_ in zip(entries, masks)]) if entries else np.empty(0)
                ev = {"interval": t, "trigger": reason, "trusted_records": int(len(trusted)),
                      "trusted_attack_share": float(att.mean()) if len(att) else float("nan"), "attack_records_in_buffer": int(att.sum()),
                      "triggered_by_release": bool(released)}
                if len(trusted) >= cfg["min_trusted"]:
                    rp = replay[rng.choice(len(replay), min(int(cfg["replay_frac"] * len(trusted)), len(replay)), replace=False)]
                    new = finetune(arm.model, np.concatenate([trusted, rp]), cfg["ft_epochs"], cfg["ft_lr"], cfg["ft_batch"], cfg["seed"] + t)
                    ratio = float(recon_error(new, anchor).mean() / anchor_ref)
                    ev.update(anchor_error_ratio=ratio, accepted=bool(ratio <= cfg["anchor_tol"]))
                    if ev["accepted"]:
                        arm.model = new
                        if arm.cfg.mode() == "anchor_q":
                            arm.thr = float(np.quantile(recon_error(new, anchor), cfg["anchor_q"]))
                        if adm in ("gated", "relative"):
                            arm.level_ref = median_log(recon_error(new, np.concatenate(arm.window)))
                else:
                    ev.update(anchor_error_ratio=None, accepted=False, skipped="too few trusted records")
                arm.retrain_log.append(ev)
            if arm.cfg.mode() == "robust":
                arm.thr = robust_log_threshold(recon_error(arm.model, np.concatenate(arm.window)), k)
        meta_rows.append((int(ids[t]), str(day[a]), int(b - a)))
        if (t + 1) % 12 == 0 or t == len(ids) - 1:
            log(f"  interval {t+1}/{len(ids)} ({day[a]}) alarms so far {len(alarms)} | " + " | ".join(
                f"{n}: thr {arm.thr:.2e}, flagged {int(arm.counts[-1][0] + arm.counts[-1][1]):,}/{b-a:,}" for n, arm in A.items()))
    return {"arms": A, "intervals": meta_rows, "scores": score_store, "anchor_ref": anchor_ref, "alarms": alarms}


def _count(flag, y):
    f, a = flag.astype(bool), y.astype(bool)
    return (int((f & a).sum()), int((f & ~a).sum()), int((~f & a).sum()), int((~f & ~a).sum()))   # tp, fp, fn, tn
