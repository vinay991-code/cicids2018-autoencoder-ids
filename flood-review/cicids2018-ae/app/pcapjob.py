"""Analyse a Wireshark/tshark capture (pcap or pcapng, file or stream): packets -> flows -> saved models -> flagged flows with Wireshark filters.

Wireshark integration, in practice:
  * Save a capture in Wireshark (File > Save As, pcap or pcapng) and upload it on the dashboard, or run tools/pcap_score.py on it.
  * Or pipe a live capture from Wireshark's command-line tool:  tshark -i <interface> -F pcap -w - | python tools/pcap_score.py -
  * Every flagged flow comes with a Wireshark display filter; paste it into Wireshark's filter bar to see the packets.
Limits (also shown in the dashboard): only IPv4 packets are used (TCP, UDP, and other IP protocols as protocol 0, like the training data);
the flow features are a port of CICFlowMeter's flow logic that was checked against the tool and against the training data (reports/FLOW_VALIDATION.md),
but a flag is still a lead to inspect, not a verdict; the models were trained on a 2018 lab network. The 'out_of_range' column counts
how many of the 68 numeric features fall outside the training range: held-out test flows have 0 in all but 0.0005% of cases, so a flow with
a higher number looks unlike the training data.
Addresses appear in these results because the analyst needs them to find the flow in Wireshark; they are kept in memory only, never sent
to any other service, and the uploaded file is deleted as soon as it has been read.
"""
from __future__ import annotations

import csv
import io
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np

from .capture import FlowTable, packet_fields
from .service import Detector

MAX_STORED = 100_000          # flows kept per job; beyond this only flagged flows are stored
BATCH = 1000
EXPIRE_EVERY = 2000           # packets between timeout sweeps


def wireshark_filter(m: dict) -> str:
    """Display filter that selects the packets of one flow (both directions, within its time span)."""
    lo, hi = m["start_us"] / 1e6 - 0.000001, m["last_us"] / 1e6 + 0.000001
    span = f"frame.time_epoch >= {lo:.6f} && frame.time_epoch <= {hi:.6f}"
    if m["proto"] not in (6, 17):                                  # protocol 0: every IPv4 packet that is neither TCP nor UDP (ICMP, IGMP, ...)
        return f"ip.addr == {m['src']} && ip.addr == {m['dst']} && !tcp && !udp && {span}"
    p = "tcp" if m["proto"] == 6 else "udp"
    if m["src"] == m["dst"]:                                        # both addresses equal (loopback): the tool builds one flow per direction
        return f"ip.src == {m['src']} && {p}.srcport == {m['sport']} && {p}.dstport == {m['dport']} && {span}"
    return f"ip.addr == {m['src']} && ip.addr == {m['dst']} && {p}.port == {m['sport']} && {p}.port == {m['dport']} && {span}"


def combined_filter(flows: list[dict], limit: int = 100) -> str:
    return " || ".join(f"({f['filter']})" for f in flows[:limit])


class Analysis:
    """Incremental analysis, usable for files and for streams."""

    def __init__(self, det: Detector, on_flow=None, batch: int = BATCH, store: bool = True):
        self.det, self.on_flow, self.batch, self.store = det, on_flow, batch, store
        self.table = FlowTable()
        self.pending: list[tuple[dict, dict]] = []
        self.flows: list[dict] = []
        self.packets = self.used = 0
        self.skipped = 0
        self.n_flows = 0
        self.flagged_counts: dict[str, int] = {}
        self.first_t = self.last_t = None
        self._n_in_batch = 0

    def packet(self, pkt):
        self.packets += 1
        f = packet_fields(pkt)
        if f is None:
            self.skipped += 1
            return
        self.used += 1
        src, dst, sp, dp, proto, plen, hlen, flags, win = f
        t = int(pkt.time * 1_000_000)                      # exact microseconds (scapy keeps pcap times as decimals)
        self.first_t = t if self.first_t is None else min(self.first_t, t)
        self.last_t = t if self.last_t is None else max(self.last_t, t)
        self.pending += self.table.packet(src, dst, sp, dp, proto, t, plen, hlen, flags, win)
        if self.packets % EXPIRE_EVERY == 0:
            self.pending += self.table.expire(t)
        if len(self.pending) >= self.batch:
            self.score_pending()

    def score_pending(self):
        if not self.pending:
            return
        feats, metas = zip(*self.pending)
        self.pending = []
        raw, proto = self.det.rows_to_arrays(list(feats))
        z = self.det.transform(raw, proto)
        out = self.det.score(z)
        ood = self.det.out_of_range(z)
        for i, m in enumerate(metas):
            alerts = [k for k, a in out["alerts"].items() if a[i]]
            self.n_flows += 1
            for k in alerts:
                self.flagged_counts[k] = self.flagged_counts.get(k, 0) + 1
            if not self.store and not self.on_flow:
                continue
            if self.store and len(self.flows) >= MAX_STORED and not alerts:
                continue
            rec = {"id": self.n_flows, "src": m["src"], "sport": m["sport"], "dst": m["dst"], "dport": m["dport"],
                   "proto": {6: "TCP", 17: "UDP"}.get(m["proto"], "other IP"), "start": m["start_us"] / 1e6, "duration_s": (m["last_us"] - m["start_us"]) / 1e6,
                   "fwd_pkts": int(feats[i]["tot_fwd_pkts"]), "bwd_pkts": int(feats[i]["tot_bwd_pkts"]),
                   "bytes": int(feats[i]["totlen_fwd_pkts"] + feats[i]["totlen_bwd_pkts"]), "out_of_range": int(ood[i]),
                   "scores": {k: float(out["scores"][k][i]) for k in out["scores"]}, "alerts": alerts, "filter": wireshark_filter(m)}
            if self.store:
                self.flows.append(rec)
            if self.on_flow:
                self.on_flow(rec)

    def finish(self):
        self.pending += self.table.flush()
        self.score_pending()

    def summary(self) -> dict:
        return {"packets_read": self.packets, "packets_used_ipv4": self.used, "packets_ignored": self.skipped, "flows": self.n_flows,
                "flows_stored": len(self.flows), "flows_flagged_by": self.flagged_counts,
                "capture_span_s": ((self.last_t - self.first_t) / 1e6) if self.first_t is not None else 0.0,
                "flows_with_out_of_range_features": sum(1 for f in self.flows if f["out_of_range"] > 0)}


def analyse_source(det: Detector, source, on_progress=None, on_flow=None, stop=None) -> Analysis:
    """source: path or binary file-like (e.g. sys.stdin.buffer). pcap and pcapng are detected from the file header."""
    from scapy.utils import PcapReader
    a = Analysis(det, on_flow)
    rd = PcapReader(source)
    try:
        for pkt in rd:
            a.packet(pkt)
            if on_progress and a.packets % 5000 == 0:
                on_progress(a.packets)
            if stop and stop():
                break
    finally:
        rd.close()
    a.finish()
    return a


def to_csv(flows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    model_keys = ["unsup_mixed", "normal_only", "supervised", "supervised_recon"]
    w.writerow(["id", "start_epoch", "duration_s", "proto", "src", "sport", "dst", "dport", "fwd_pkts", "bwd_pkts", "bytes", "out_of_range_features"]
               + [f"score_{k}" for k in model_keys] + ["alerts", "wireshark_filter"])
    for f in flows:
        w.writerow([f["id"], f"{f['start']:.6f}", f"{f['duration_s']:.6f}", f["proto"], f["src"], f["sport"], f["dst"], f["dport"], f["fwd_pkts"],
                    f["bwd_pkts"], f["bytes"], f["out_of_range"]] + [f"{f['scores'][k]:.8g}" for k in model_keys] + [";".join(f["alerts"]), f["filter"]])
    return buf.getvalue()


# ------------------------------------------------------------------------- background jobs ----------------------------------------------
class Jobs:
    def __init__(self, keep: int = 5):
        self.keep, self.items, self.lock = keep, OrderedDict(), threading.Lock()

    def start(self, det: Detector, path: str, filename: str, cleanup) -> str:
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "filename": filename, "status": "running", "packets": 0, "error": None, "created": time.time(), "flows": [], "summary": None}
        with self.lock:
            self.items[jid] = job
            while len(self.items) > self.keep:
                self.items.popitem(last=False)

        def run():
            try:
                a = analyse_source(det, path, on_progress=lambda n: job.__setitem__("packets", n))
                order = sorted(a.flows, key=lambda f: (-len(f["alerts"]), -f["scores"]["supervised"]))
                job["flows"], job["summary"] = order, a.summary()
                job["status"] = "done"
            except Exception as e:             # unreadable or unsupported capture file
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
            finally:
                cleanup()                      # the uploaded file is deleted as soon as it has been read
        threading.Thread(target=run, daemon=True).start()
        return jid

    def get(self, jid: str):
        return self.items.get(jid)

    def delete(self, jid: str) -> bool:
        with self.lock:
            return self.items.pop(jid, None) is not None
