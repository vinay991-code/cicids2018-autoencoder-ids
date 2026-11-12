"""Packets -> flow features, built to reproduce the flow records of the training data (used by the PCAP analysis and the live capture).

What this module is: a port of the flow logic of CICFlowMeter (the Java tool that produced the CSE-CIC-IDS2018 files), including its known
quirks, because the models were trained on records that contain those quirks. It was written from the tool's source (the February 2018 version,
commit eaa853d, of BasicFlow.java / FlowGenerator.java, github.com/ahlashkari/CICFlowMeter) and then checked in two independent ways (details: reports/FLOW_VALIDATION.md):
  1. Against the training data itself: every definition below was tested on 1.5 million training rows (tools/validate_flow_definitions.py).
     Each rule marked DATA below was first measured on the data, and the February 2018 source agrees with every one of them.
  2. Against the CICFlowMeter code itself, run on the same packets (tools/cicflowmeter_ref/): feature-by-feature comparison on pcap files.

Rules reproduced (all measured on the training data unless marked SOURCE):
  * A flow is identified by IPs, ports and protocol, both directions together. Packets that are neither TCP nor UDP (ICMP, IGMP, ...) form flows with
    protocol 0, ports 0 and no payload; IPv6 and non-IP frames are ignored (the training data has only protocols 0, 6 and 17).
  * A flow ends when a packet arrives more than 120 s after the flow started (the packet starts a new flow), or when a packet with the FIN flag
    arrives (that packet is the last one of the flow; the rest of the TCP closing handshake forms a new, short flow). RST does not end a flow. DATA
  * A flow with a single packet is never reported. DATA
  * Flow length statistics (Packet Length Mean/Std/Var, Packet Size Avg) count the payload of the first packet twice. DATA
  * Flag counts are taken from the first packet of the flow only, and the eight columns hold the flags in the order of a Java HashMap
    (so the column called FIN holds RST, SYN holds PSH, RST holds ECE, PSH holds SYN, ACK holds ACK, URG holds FIN, CWE holds URG, ECE holds CWR). DATA
  * "Header length" is the TCP header (or 8 for UDP), not IP + TCP. Payload is taken from the IP length (Ethernet padding is not counted). DATA
  * Init Win Bytes Forward is the window of the first packet; Init Win Bytes Backward is the window of the LAST backward packet (a quirk of the
    tool); both are -1 when there is no such TCP packet. DATA
  * Active Data Packets does not count the first packet; Active and Idle statistics use a 5 s gap, and the last active period is never recorded. DATA
  * Subflow columns equal the flow totals. DATA (in the source the subflow detector runs only for the first packet, so the divisor stays 1)

Not reproduced because they are constant in the training data and were dropped: bulk features, Bwd PSH/URG flags.

What can still differ from the training data: the 2018 tool read packets with jnetpcap; the byte-level parsing here uses scapy (checked on
ordinary traffic, see the validation report). The training network was an AWS lab; other networks look different (see the out-of-range count).
Flow features contain no payload bytes. The flow records shown to the analyst carry addresses and ports (needed to find the flow in Wireshark);
they stay in memory on this computer. Use live capture only on a network you own or may monitor.
"""
from __future__ import annotations

import socket
import threading
from functools import lru_cache

import numpy as np

# Importing these registers the link-layer types (Ethernet etc.) that the pcap reader needs; without them every packet is read as raw bytes.
import scapy.layers.inet  # noqa: F401
import scapy.layers.inet6  # noqa: F401
import scapy.layers.l2  # noqa: F401

FLOW_TIMEOUT_US = 120_000_000          # CICFlowMeter default flow timeout (120 s, measured from the first packet of the flow)
ACTIVE_TIMEOUT_US = 5_000_000          # an idle gap longer than this ends an active period

# column name in the training data  <-  flag whose first-packet value it holds (HashMap order of the tool, measured on the data)
FLAG_COLUMNS = {"fin_flag_cnt": "RST", "syn_flag_cnt": "PSH", "rst_flag_cnt": "ECE", "psh_flag_cnt": "SYN",
                "ack_flag_cnt": "ACK", "urg_flag_cnt": "FIN", "cwe_flag_count": "URG", "ece_flag_cnt": "CWR"}
_FLAG_BITS = {"FIN": 0x01, "SYN": 0x02, "RST": 0x04, "PSH": 0x08, "ACK": 0x10, "URG": 0x20, "ECE": 0x40, "CWR": 0x80}


@lru_cache(maxsize=65536)
def _ip_bytes(ip: str):
    """Signed bytes of an IPv4 address (Java bytes), used to order the two ends of a flow like the tool does."""
    return tuple(b - 256 if b > 127 else b for b in socket.inet_aton(ip))


def flow_key(src, dst, sport, dport, proto):
    """Direction-independent flow id, ordered by the first differing signed address byte (equal addresses keep the packet direction)."""
    a, b = _ip_bytes(src), _ip_bytes(dst)
    for x, y in zip(a, b):
        if x != y:
            return (src, dst, sport, dport, proto) if x < y else (dst, src, dport, sport, proto)
    return (src, dst, sport, dport, proto)


def _sd(a):
    """Sample standard deviation (n - 1), 0 for fewer than two values, like the Java statistics class."""
    return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def _summary(a):
    """(max, min, mean, std) of a list, zeros when empty."""
    if not len(a):
        return 0.0, 0.0, 0.0, 0.0
    x = np.asarray(a, dtype=np.float64)
    return float(x.max()), float(x.min()), float(x.mean()), _sd(x)


class Flow:
    """One flow. Direction 'forward' = packets whose source address equals the source address of the flow."""

    def __init__(self, pkt, orient=None):
        src, dst, sport, dport, proto, t, plen, hlen, flags, win = pkt
        self.start = self.last = t
        self.start_active = self.end_active = t
        self.src, self.sport, self.dst, self.dport, self.protocol = src, sport, dst, dport, proto
        # the first packet is always counted as forward, and (quirk) twice in the all-packet length statistics
        self.all_len = [plen, plen]
        self.fwd_len, self.bwd_len = [plen], []
        self.fwd_t, self.bwd_t = [t], []
        self.flow_iat, self.fwd_iat, self.bwd_iat = [], [], []
        self.fwd_bytes, self.bwd_bytes = plen, 0
        self.fwd_hdr, self.bwd_hdr = hlen, 0
        self.min_seg = hlen
        self.init_fwd, self.init_bwd = win, -1
        self.act_data = 0
        self.active, self.idle = [], []
        self.fwd_psh = 1 if flags & 0x08 else 0
        self.fwd_urg = 1 if flags & 0x20 else 0
        self.first_flags = flags                                    # flag counts come from the first packet only
        self.fwd_last = self.bwd_last = t
        if orient is not None:                                      # a flow restarted after the timeout keeps the old orientation
            self.src, self.sport, self.dst, self.dport = orient.src, orient.sport, orient.dst, orient.dport

    @property
    def n(self) -> int:
        return len(self.fwd_len) + len(self.bwd_len)

    def update_active_idle(self, t):
        if t - self.end_active > ACTIVE_TIMEOUT_US:
            if self.end_active - self.start_active > 0:
                self.active.append(self.end_active - self.start_active)
            self.idle.append(t - self.end_active)
            self.start_active = self.end_active = t
        else:
            self.end_active = t

    def add(self, pkt):
        src, dst, sport, dport, proto, t, plen, hlen, flags, win = pkt
        self.all_len.append(plen)
        if src == self.src:
            if plen >= 1:
                self.act_data += 1
            self.fwd_len.append(plen)
            self.fwd_hdr += hlen
            self.fwd_bytes += plen
            self.fwd_t.append(t)
            if len(self.fwd_len) > 1:
                self.fwd_iat.append(t - self.fwd_last)
            self.fwd_last = t
            self.min_seg = min(hlen, self.min_seg)
        else:
            self.bwd_len.append(plen)
            self.init_bwd = win                                     # overwritten by every backward packet (quirk of the tool)
            self.bwd_hdr += hlen
            self.bwd_bytes += plen
            self.bwd_t.append(t)
            if len(self.bwd_len) > 1:
                self.bwd_iat.append(t - self.bwd_last)
            self.bwd_last = t
        self.flow_iat.append(t - self.last)
        self.last = t

    def meta(self) -> dict:
        """Who talked to whom (kept only for local analysis of a capture file; never part of the model input)."""
        return {"src": self.src, "sport": self.sport, "dst": self.dst, "dport": self.dport, "proto": self.protocol,
                "start_us": self.start, "last_us": self.last}

    def features(self) -> dict:
        dur = float(self.last - self.start)
        nf, nb, n = len(self.fwd_len), len(self.bwd_len), self.n
        fmax, fmin, fmean, fstd = _summary(self.fwd_len)
        bmax, bmin, bmean, bstd = _summary(self.bwd_len)
        pmax, pmin, pmean, pstd = _summary(self.all_len)
        fi, bi = np.asarray(self.fwd_iat, dtype=np.float64), np.asarray(self.bwd_iat, dtype=np.float64)
        ii = _summary(self.flow_iat)
        use_f, use_b = nf > 1, nb > 1                                # the tool writes zeros unless the direction has more than one packet
        secs = dur / 1e6
        # zero-duration flows give an infinite rate in the tool; such rows were removed from the training data, here the rate is reported as 0
        feats = {
            "flow_duration": dur, "tot_fwd_pkts": nf, "tot_bwd_pkts": nb,
            "totlen_fwd_pkts": float(sum(self.fwd_len)), "totlen_bwd_pkts": float(sum(self.bwd_len)),
            "fwd_pkt_len_max": fmax, "fwd_pkt_len_min": fmin, "fwd_pkt_len_mean": fmean, "fwd_pkt_len_std": fstd,
            "bwd_pkt_len_max": bmax, "bwd_pkt_len_min": bmin, "bwd_pkt_len_mean": bmean, "bwd_pkt_len_std": bstd,
            "flow_byts_s": (self.fwd_bytes + self.bwd_bytes) / secs if secs > 0 else 0.0,
            "flow_pkts_s": n / secs if secs > 0 else 0.0,
            "flow_iat_mean": ii[2], "flow_iat_std": ii[3], "flow_iat_max": ii[0], "flow_iat_min": ii[1],
            "fwd_iat_tot": float(fi.sum()) if use_f else 0.0, "fwd_iat_mean": float(fi.mean()) if use_f else 0.0,
            "fwd_iat_std": _sd(fi) if use_f else 0.0, "fwd_iat_max": float(fi.max()) if use_f else 0.0, "fwd_iat_min": float(fi.min()) if use_f else 0.0,
            "bwd_iat_tot": float(bi.sum()) if use_b else 0.0, "bwd_iat_mean": float(bi.mean()) if use_b else 0.0,
            "bwd_iat_std": _sd(bi) if use_b else 0.0, "bwd_iat_max": float(bi.max()) if use_b else 0.0, "bwd_iat_min": float(bi.min()) if use_b else 0.0,
            "fwd_psh_flags": self.fwd_psh, "fwd_urg_flags": self.fwd_urg,
            "fwd_header_len": float(self.fwd_hdr), "bwd_header_len": float(self.bwd_hdr),
            "fwd_pkts_s": nf / secs if secs > 0 else 0.0, "bwd_pkts_s": nb / secs if secs > 0 else 0.0,
            "pkt_len_min": pmin, "pkt_len_max": pmax, "pkt_len_mean": pmean, "pkt_len_std": pstd, "pkt_len_var": pstd ** 2,
            "down_up_ratio": float(nb // nf) if nf else 0.0,
            "pkt_size_avg": float(sum(self.all_len)) / n,
            "fwd_seg_size_avg": fmean, "bwd_seg_size_avg": bmean,
            "subflow_fwd_pkts": nf, "subflow_fwd_byts": float(self.fwd_bytes), "subflow_bwd_pkts": nb, "subflow_bwd_byts": float(self.bwd_bytes),
            "init_fwd_win_byts": self.init_fwd, "init_bwd_win_byts": self.init_bwd,
            "fwd_act_data_pkts": self.act_data, "fwd_seg_size_min": float(self.min_seg),
            "active_mean": float(np.mean(self.active)) if self.active else 0.0, "active_std": _sd(self.active),
            "active_max": float(max(self.active)) if self.active else 0.0, "active_min": float(min(self.active)) if self.active else 0.0,
            "idle_mean": float(np.mean(self.idle)) if self.idle else 0.0, "idle_std": _sd(self.idle),
            "idle_max": float(max(self.idle)) if self.idle else 0.0, "idle_min": float(min(self.idle)) if self.idle else 0.0,
            "protocol": self.protocol,
        }
        for column, flag in FLAG_COLUMNS.items():
            feats[column] = 1 if self.first_flags & _FLAG_BITS[flag] else 0
        return feats


class FlowTable:
    """Assembles packets into flows with the rules of the module docstring. A flow is released when a FIN packet ends it, when a later packet
    of the same id arrives after the 120 s timeout, on expire() once it is older than the timeout (a live capture must not wait for a next
    packet; the features are the same as when the next packet releases it), or on flush() at the end of a capture file."""

    def __init__(self, timeout_us: int = FLOW_TIMEOUT_US):
        self.flows: dict = {}
        self.timeout = timeout_us
        self.lock = threading.Lock()
        self.single_packet_flows = 0                                # flows of one packet are never reported (as in the tool)

    @staticmethod
    def _normalise(src, dst, sport, dport, proto, t_us, payload_len, hdr_len, flags, win):
        if proto not in (6, 17):                                    # other IP protocols: protocol 0, no ports, no payload
            return src, dst, 0, 0, 0, t_us, 0, 0, 0, -1
        return src, dst, int(sport), int(dport), proto, t_us, int(payload_len), int(hdr_len), (int(flags) if proto == 6 else 0), (-1 if win is None else int(win))

    def _release(self, f, done):
        if f.n > 1:
            done.append((f.features(), f.meta()))
        else:
            self.single_packet_flows += 1

    def packet(self, src, dst, sport, dport, proto, t_us, payload_len, hdr_len, flags=0, win=None):
        """Add a packet; returns a list of (features dict, meta dict) for flows that just ended."""
        pkt = self._normalise(src, dst, sport, dport, proto, t_us, payload_len, hdr_len, flags, win)
        key = flow_key(src, dst, pkt[2], pkt[3], pkt[4])
        done = []
        with self.lock:
            f = self.flows.get(key)
            if f is None:
                self.flows[key] = Flow(pkt)
            elif t_us - f.start > self.timeout:
                self._release(f, done)
                self.flows[key] = Flow(pkt, orient=f)
            elif pkt[4] == 6 and pkt[8] & 0x01:                     # FIN: the packet is the last of the flow
                f.add(pkt)
                self._release(f, done)
                del self.flows[key]
            else:
                f.update_active_idle(t_us)
                f.add(pkt)
        return done

    def expire(self, now_us: int):
        out = []
        with self.lock:
            for key in [k for k, f in self.flows.items() if now_us - f.start > self.timeout]:
                self._release(self.flows.pop(key), out)
        return out

    def flush(self):
        """Release every open flow (end of a capture file); single-packet flows are dropped."""
        out = []
        with self.lock:
            for f in self.flows.values():
                self._release(f, out)
            self.flows.clear()
        return out


def packet_fields(pkt):
    """Extract (src, dst, sport, dport, proto, payload_len, header_len, flags, window) from a scapy IPv4 packet, else None.
    TCP and UDP give their own fields; every other IPv4 packet (ICMP, IGMP, non-first fragments, ...) is reported with proto 0."""
    from scapy.layers.inet import IP, TCP, UDP
    if IP not in pkt:
        return None
    ip = pkt[IP]
    if int(ip.frag) == 0 and TCP in pkt:
        t = pkt[TCP]
        hdr = t.dataofs * 4
        plen = max(0, int(ip.len) - ip.ihl * 4 - hdr) if ip.len else len(bytes(t.payload))      # from the IP length: right even if the capture was cut to headers
        return ip.src, ip.dst, int(t.sport), int(t.dport), 6, plen, hdr, int(t.flags), int(t.window)
    if int(ip.frag) == 0 and UDP in pkt:
        u = pkt[UDP]
        plen = max(0, int(u.len) - 8) if u.len else len(bytes(u.payload))
        return ip.src, ip.dst, int(u.sport), int(u.dport), 17, plen, 8, 0, None
    return ip.src, ip.dst, 0, 0, 0, 0, 0, 0, None
