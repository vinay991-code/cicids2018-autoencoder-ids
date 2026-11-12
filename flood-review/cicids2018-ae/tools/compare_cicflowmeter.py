"""Compare this project's flow builder with the CICFlowMeter flow code on the same packets, feature by feature.

  python tools/compare_cicflowmeter.py capture.pcap [--json out.json]

The capture must be a classic pcap with Ethernet frames (convert pcapng with: editcap -F pcap in.pcapng out.pcap).
The reference is the real CICFlowMeter 2018 flow code driven by a small pcap reader (tools/cicflowmeter_ref/); it is built on first use
(needs git, a JDK and internet access). Flows are matched by (start time in microseconds, addresses, ports, protocol); then the 68 model
features are compared. Two numbers per feature: flows that differ, and the largest relative difference.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/cicflowmeter_ref"))

from app.capture import FlowTable, packet_fields  # noqa: E402
import build_reference  # noqa: E402

# position of each value in one line of the reference output (CICFlowMeter's own CSV row, after the six id columns and the timestamp)
JAVA_COLUMNS = (
    "flow_id src sport dst dport protocol timestamp flow_duration tot_fwd_pkts tot_bwd_pkts totlen_fwd_pkts totlen_bwd_pkts "
    "fwd_pkt_len_max fwd_pkt_len_min fwd_pkt_len_mean fwd_pkt_len_std bwd_pkt_len_max bwd_pkt_len_min bwd_pkt_len_mean bwd_pkt_len_std "
    "flow_byts_s flow_pkts_s flow_iat_mean flow_iat_std flow_iat_max flow_iat_min "
    "fwd_iat_tot fwd_iat_mean fwd_iat_std fwd_iat_max fwd_iat_min bwd_iat_tot bwd_iat_mean bwd_iat_std bwd_iat_max bwd_iat_min "
    "fwd_psh_flags bwd_psh_flags fwd_urg_flags bwd_urg_flags fwd_header_len bwd_header_len fwd_pkts_s bwd_pkts_s "
    "pkt_len_min pkt_len_max pkt_len_mean pkt_len_std pkt_len_var "
    "fin_flag_cnt syn_flag_cnt rst_flag_cnt psh_flag_cnt ack_flag_cnt urg_flag_cnt cwe_flag_count ece_flag_cnt "
    "down_up_ratio pkt_size_avg fwd_seg_size_avg bwd_seg_size_avg dup_fwd_header_len "
    "fwd_byts_b_avg fwd_pkts_b_avg fwd_blk_rate_avg bwd_byts_b_avg bwd_pkts_b_avg bwd_blk_rate_avg "
    "subflow_fwd_pkts subflow_fwd_byts subflow_bwd_pkts subflow_bwd_byts init_fwd_win_byts init_bwd_win_byts fwd_act_data_pkts fwd_seg_size_min "
    "active_mean active_std active_max active_min idle_mean idle_std idle_max idle_min label").split()


def python_flows(pcap: Path):
    """Flows of this project's builder for a pcap, using the same packet path as the dashboard."""
    from scapy.utils import PcapReader
    table, flows, n_pkts = FlowTable(), [], 0
    with PcapReader(str(pcap)) as rd:
        for pkt in rd:
            n_pkts += 1
            f = packet_fields(pkt)
            if f is None:
                continue
            src, dst, sp, dp, proto, plen, hlen, flags, win = f
            flows += table.packet(src, dst, sp, dp, proto, int(pkt.time * 1_000_000), plen, hlen, flags, win)
    flows += table.flush()
    return flows, n_pkts, table.single_packet_flows


def reference_flows(pcap: Path):
    out = Path(tempfile.mkdtemp()) / "reference.txt"
    log = build_reference.run_reference(pcap, out)
    rows = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        start, src, sport, dst, dport, proto, rest = line.split("|", 6)
        vals = rest.split(",")
        row = dict(zip(JAVA_COLUMNS, vals))
        row = {k: v for k, v in row.items() if k not in ("flow_id", "timestamp", "label")}
        rows[(int(start), src, int(sport), dst, int(dport), int(proto))] = {k: _num(v) for k, v in row.items()}
    return rows, log


def _num(v):
    try:
        return float(v)
    except ValueError:
        return float("nan")


def compare(pcap: Path, columns: list[str]) -> dict:
    ref, log = reference_flows(pcap)
    mine, n_pkts, single = python_flows(pcap)
    mine_by_key = {}
    for feats, meta in mine:
        key = (meta["start_us"], meta["src"], meta["sport"], meta["dst"], meta["dport"], meta["proto"])
        mine_by_key[key] = feats
    both = sorted(set(ref) & set(mine_by_key))
    only_ref, only_mine = sorted(set(ref) - set(mine_by_key)), sorted(set(mine_by_key) - set(ref))
    per_col = {c: {"differs": 0, "max_rel_diff": 0.0, "compared": 0} for c in columns}
    zero_duration = 0
    flows_all_equal = 0
    examples = []
    for key in both:
        r, m = ref[key], mine_by_key[key]
        all_ok = True
        for c in columns:
            a, b = float(m[c]), r[c]
            if c in ("flow_byts_s", "flow_pkts_s") and not math.isfinite(b):
                zero_duration += c == "flow_byts_s"          # the tool gives an infinite rate for a zero-duration flow; here it is 0 (excluded from the comparison)
                continue
            per_col[c]["compared"] += 1
            rel = abs(a - b) / max(1.0, abs(b))
            per_col[c]["max_rel_diff"] = max(per_col[c]["max_rel_diff"], rel)
            if rel > 1e-9:
                per_col[c]["differs"] += 1
                all_ok = False
                if len(examples) < 8:
                    examples.append({"flow": key, "feature": c, "this_project": a, "cicflowmeter": b})
        flows_all_equal += all_ok
    return {"pcap": pcap.name, "packets": n_pkts, "reference_log": log, "flows_reference": len(ref), "flows_this_project": len(mine_by_key),
            "flows_matched": len(both), "flows_only_in_reference": len(only_ref), "flows_only_in_this_project": len(only_mine),
            "flows_with_all_features_equal": flows_all_equal, "zero_duration_flows_rate_excluded": zero_duration,
            "single_packet_flows_dropped_by_this_project": single, "features_that_differ": {c: v for c, v in per_col.items() if v["differs"]},
            "features_compared": len(columns), "examples": examples}


def model_columns() -> list[str]:
    sc = json.loads((ROOT / "data/processed/scaler.json").read_text())
    return list(sc["model_columns"]) + ["protocol"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = compare(Path(a.pcap), model_columns())
    keys = ("pcap", "packets", "flows_reference", "flows_this_project", "flows_matched", "flows_only_in_reference", "flows_only_in_this_project",
            "flows_with_all_features_equal", "zero_duration_flows_rate_excluded", "single_packet_flows_dropped_by_this_project")
    for k in keys:
        print(f"{k:45s} {res[k]}")
    print(f"features compared per flow: {res['features_compared']}; features that differ:", res["features_that_differ"] or "none")
    for e in res["examples"]:
        print("  e.g.", e)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
