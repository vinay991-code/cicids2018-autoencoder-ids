"""Test every flow-feature definition used by app/capture.py against the training data, and apply the same tests to the flows the port builds.

  python tools/validate_flow_definitions.py --training                      # tests on a random sample of the cleaned training rows
  python tools/validate_flow_definitions.py --pcap a.pcap b.pcap            # the same tests on flows built from captures
  python tools/validate_flow_definitions.py --training --pcap a.pcap --json results/flow_validation.json

Each test is a row-level property that follows from one definition of the original tool (for example "the packet-length mean counts the first
packet twice" implies that mean*(n+1) - total_payload is a non-negative integer, namely the first packet's payload). The output is the share of
rows satisfying each property. A share of 1.0 on the training data means the training data obeys that definition; the same share on the
port's flows means the port obeys it too. Properties that only hold statistically (marked ~) are reported but not required to be 1.0.
No labels are used. Training rows are read from data/interim/*.f32 (written by Phase 2) and are not modified.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def training_columns(per_file: int, seed: int = 0):
    st = json.loads((ROOT / "data/interim/clean_stats.json").read_text())
    feats = st["features"]
    rng = np.random.default_rng(seed)
    parts = []
    for f in sorted(glob.glob(str(ROOT / "data/interim/*.f32"))):
        m = np.memmap(f, dtype=np.float32, mode="r").reshape(-1, len(feats))
        idx = np.sort(rng.choice(len(m), min(per_file, len(m)), replace=False))
        parts.append(np.asarray(m[idx], dtype=np.float64))
    a = np.concatenate(parts)
    return {c: a[:, i] for i, c in enumerate(feats)}


def pcap_columns(paths):
    from scapy.utils import PcapReader
    from app.capture import FlowTable, packet_fields
    rows = []
    for p in paths:
        t = FlowTable()
        with PcapReader(str(p)) as rd:
            for pkt in rd:
                f = packet_fields(pkt)
                if f is not None:
                    s, d, sp, dp, pr, pl, hl, fl, w = f
                    rows += [x[0] for x in t.packet(s, d, sp, dp, pr, int(pkt.time * 1_000_000), pl, hl, fl, w)]
        rows += [x[0] for x in t.flush()]
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}, len(rows)


def close(a, b, tol=1e-3):
    return np.abs(a - b) <= tol * np.maximum(1.0, np.abs(b))


def properties(D: dict) -> dict:
    """name -> boolean array over rows (None where a property does not apply)."""
    tf, tb = D["tot_fwd_pkts"], D["tot_bwd_pkts"]
    n = tf + tb
    lf, lb = D["totlen_fwd_pkts"], D["totlen_bwd_pkts"]
    dur, proto = D["flow_duration"], D["protocol"]
    tcp, udp, other = proto == 6, proto == 17, proto == 0
    pos = dur > 0
    first = D["pkt_len_mean"] * (n + 1) - (lf + lb)                    # payload of the first packet, if it is counted twice
    flag_cols = ["fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt", "cwe_flag_count", "ece_flag_cnt"]
    active, idle = D["active_mean"] > 0, D["idle_mean"] > 0
    P = {
        "flow has at least 2 packets and at least 1 forward packet": (n >= 2) & (tf >= 1),
        "flow duration is at most 120 s": dur <= 120e6 + 1,
        "first packet counted twice: mean*(n+1) - total payload is an integer >= 0 and <= max packet length":
            (first >= -0.01) & (np.abs(first - np.round(first)) < 0.05) & (first <= D["pkt_len_max"] + 0.05),
        "Packet Size Avg = (total payload + first payload) / n": close(D["pkt_size_avg"] * n - (lf + lb), np.maximum(first, 0), 1e-2),
        "Packet Length Var = Packet Length Std squared": close(D["pkt_len_var"], D["pkt_len_std"] ** 2, 1e-2),
        "Packet Length Min/Max = min/max of the forward and backward extremes":
            close(D["pkt_len_max"], np.maximum(D["fwd_pkt_len_max"], D["bwd_pkt_len_max"])),
        "Down/Up Ratio = integer part of bwd/fwd packets": close(D["down_up_ratio"], np.floor(tb / np.maximum(tf, 1))),
        "subflow columns equal the flow totals": close(D["subflow_fwd_pkts"], tf) & close(D["subflow_bwd_pkts"], tb) & close(D["subflow_fwd_byts"], lf) & close(D["subflow_bwd_byts"], lb),
        "Fwd/Bwd segment size avg = mean forward/backward payload": close(D["fwd_seg_size_avg"], D["fwd_pkt_len_mean"]) & close(D["bwd_seg_size_avg"], D["bwd_pkt_len_mean"]),
        "forward mean * forward packets = total forward payload": close(D["fwd_pkt_len_mean"] * tf, lf),
        "Flow IAT mean * (n-1) = flow duration": close(D["flow_iat_mean"] * (n - 1), dur, 1e-2),
        "Flow Bytes/s = total payload / duration (duration > 0)": np.where(pos, close(D["flow_byts_s"], (lf + lb) / np.maximum(dur, 1) * 1e6, 1e-2), True),
        "Flow Packets/s = packets / duration (duration > 0)": np.where(pos, close(D["flow_pkts_s"], n / np.maximum(dur, 1) * 1e6, 1e-2), True),
        "the eight flag columns are 0 or 1": np.all([(D[c] == 0) | (D[c] == 1) for c in flag_cols], axis=0),
        "'SYN' column = Fwd PSH Flags (HashMap column order, first packet only)": D["syn_flag_cnt"] == D["fwd_psh_flags"],
        "'CWE' column = Fwd URG Flags": D["cwe_flag_count"] == D["fwd_urg_flags"],
        "'RST' column = 'ECE' column (ECN-setup SYN carries both)": np.where(tcp, np.abs(D["rst_flag_cnt"] - D["ece_flag_cnt"]) <= 0, True),
        "UDP and other-IP flows have all flag columns 0 and both init windows -1":
            np.where(~tcp, np.all([D[c] == 0 for c in flag_cols], axis=0) & (D["init_fwd_win_byts"] == -1) & (D["init_bwd_win_byts"] == -1), True),
        "TCP flows have an init forward window >= 0": np.where(tcp, D["init_fwd_win_byts"] >= 0, True),
        "active data packets <= forward packets - 1 (the first packet is not counted)": D["fwd_act_data_pkts"] <= tf - 1,
        "header length is the transport header: UDP forward header = 8 per packet": np.where(udp, close(D["fwd_header_len"], 8 * tf) & (D["fwd_seg_size_min"] == 8), True),
        "TCP header per forward packet is 20 to 60 bytes": np.where(tcp, (D["fwd_header_len"] / np.maximum(tf, 1) >= 20) & (D["fwd_header_len"] / np.maximum(tf, 1) <= 60), True),
        "TCP minimum forward header is 20 to 60 and a multiple of 4": np.where(tcp, (D["fwd_seg_size_min"] >= 20) & (D["fwd_seg_size_min"] <= 60) & (D["fwd_seg_size_min"] % 4 == 0), True),
        "protocol-0 flows have no payload, no header bytes, windows -1":
            np.where(other, (lf + lb == 0) & (D["fwd_header_len"] + D["bwd_header_len"] == 0) & (D["init_fwd_win_byts"] == -1), True),
        "an active period is recorded only together with an idle gap": ~(active & ~idle),
        "every idle gap is longer than 5 s": np.where(idle, D["idle_min"] > 5e6, True),
    }
    return P


def statistical(D: dict) -> dict:
    """Properties that hold for most rows but not all (reported, not required to be 1)."""
    tcp = D["protocol"] == 6
    out = {}
    if tcp.any():
        out["~ TCP flows whose first packet is SYN or ACK (the 'PSH' and 'ACK' columns add to 1)"] = float(np.mean(((D["psh_flag_cnt"] + D["ack_flag_cnt"]) == 1)[tcp]))
        out["~ TCP flows that start with a FIN (the 'URG' column), all of which have exactly 2 packets"] = float(np.mean(((D["urg_flag_cnt"] == 1) & ((D["tot_fwd_pkts"] + D["tot_bwd_pkts"]) == 2))[tcp & (D["urg_flag_cnt"] == 1)])) if (tcp & (D["urg_flag_cnt"] == 1)).any() else None
        out["~ share of TCP flows that start with a FIN"] = float(np.mean((D["urg_flag_cnt"] == 1)[tcp]))
    if tcp.any():
        out["~ TCP flows that contain a payload-free packet (Pkt Len Min = 0)"] = float(np.mean((D["pkt_len_min"] == 0)[tcp]))
    out["~ share of flows with protocol 0"] = float(np.mean(D["protocol"] == 0))
    return out


def run(D: dict) -> dict:
    n = len(D["protocol"])
    res = {name: float(np.mean(ok)) for name, ok in properties(D).items()}
    return {"rows": n, "properties": res, "statistical": statistical(D)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--per-file", type=int, default=150_000)
    ap.add_argument("--pcap", nargs="*", default=[])
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    if a.training:
        out["training_data"] = run(training_columns(a.per_file))
    if a.pcap:
        cols, _ = pcap_columns(a.pcap)
        out["port_on_" + "+".join(Path(p).name for p in a.pcap)] = run(cols)
    names = list(next(iter(out.values()))["properties"])
    heads = list(out)
    print(f"{'property':100s} " + " ".join(f"{h[:22]:>22s}" for h in heads))
    for nm in names:
        print(f"{nm[:100]:100s} " + " ".join(f"{out[h]['properties'][nm]:22.4f}" for h in heads))
    print("rows:" + " " * 95 + " ".join(f"{out[h]['rows']:22d}" for h in heads))
    for h in heads:
        for k, v in out[h]["statistical"].items():
            print(f"  [{h[:22]}] {k}: {v if v is None else round(v, 4)}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
