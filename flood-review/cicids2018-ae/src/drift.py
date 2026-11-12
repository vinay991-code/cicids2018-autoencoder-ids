"""Label-free drift monitoring for the streaming experiments (Phase 9).

A fixed sentinel model (the warm-up autoencoder, never updated) scores every record, so adaptation of any arm cannot
feed back into detection. The stream is cut into small batches; each batch yields a few statistics mapped to [0, 1]:
  log_error      mean log10 reconstruction error (distribution of the sentinel's errors, location)
  flagged_share  share of records whose sentinel error exceeds the sentinel threshold (distribution tail)
  <feature>      mean of selected traffic features (scaled values)
  udp_share      mean of the protocol_17 indicator (protocol mix)
Each statistic feeds its own ADWIN detector (river). The per-detector delta is delta / number_of_monitors (Bonferroni),
and the detectors are pre-fed with batches from the end of the warm-up period so the first evaluation batch is already
compared against the reference regime. An alarm is raised when any detector reports a change.
"""
import json

import numpy as np
from river.drift import ADWIN

from models import recon_error

FEATURES = ["flow_duration", "tot_fwd_pkts", "totlen_fwd_pkts", "flow_byts_s", "pkt_len_mean"]


def feature_columns(scaler_path="data/processed/scaler.json"):
    cols = json.load(open(scaler_path))["output_columns"]
    return {n: cols.index(n) for n in FEATURES + ["protocol_17"]}


class DriftMonitor:
    def __init__(self, sentinel, thr, batch=5000, delta=0.002, min_batch=1000):
        self.sentinel, self.thr, self.batch, self.min_batch = sentinel, thr, batch, min_batch
        self.cols = feature_columns()
        self.names = ["log_error", "flagged_share"] + FEATURES + ["udp_share"]
        self.det = {n: ADWIN(delta=delta / len(self.names), clock=1) for n in self.names}
        self.n_batches = 0
        self.alarms = []

    def _stats(self, Xb, sb):
        v = [float(np.clip((np.log10(np.maximum(sb, 1e-12)).mean() + 7.0) / 7.0, 0, 1)), float((sb > self.thr).mean())]
        v += [float(np.clip(Xb[:, self.cols[n]].mean(), 0, 1)) for n in FEATURES]
        v.append(float(Xb[:, self.cols["protocol_17"]].mean()))
        return v

    def prefeed(self, Xref):
        s = recon_error(self.sentinel, Xref)
        for i in range(0, len(Xref) - self.min_batch + 1, self.batch):
            for n, x in zip(self.names, self._stats(Xref[i:i + self.batch], s[i:i + self.batch])):
                self.det[n].update(x)
        self.n_batches = 0

    def feed(self, X, s, record_offset=0, interval=None):
        """X: features of one interval, s: sentinel scores. Returns the alarms raised inside this interval."""
        out = []
        for i in range(0, len(X), self.batch):
            if len(X) - i < self.min_batch:
                break
            fired = []
            for n, x in zip(self.names, self._stats(X[i:i + self.batch], s[i:i + self.batch])):
                self.det[n].update(x)
                if self.det[n].drift_detected:
                    fired.append(n)
            self.n_batches += 1
            if fired:
                out.append({"interval": interval, "batch": self.n_batches, "record": int(record_offset + i + min(self.batch, len(X) - i)), "monitors": fired})
        self.alarms.extend(out)
        return out
