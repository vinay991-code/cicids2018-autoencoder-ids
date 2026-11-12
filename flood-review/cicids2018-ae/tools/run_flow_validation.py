"""Run the whole flow-builder validation and write results/flow_validation_parity.json (the dashboard and the report quote this file).

  python tools/run_flow_validation.py --ordinary ordinary1.pcap ordinary2.pcap [--stress-seeds 1 7 21]

For every capture: (1) the 69 model features of every flow against the real CICFlowMeter flow code (tools/compare_cicflowmeter.py);
(2) every IPv4 packet's parsed fields against Wireshark's tshark (tools/check_packet_fields.py). Stress captures are generated here
(tools/make_stress_pcap.py); the ordinary captures are your own (see tools/ordinary_traffic_check.py). Captures are not stored.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/cicflowmeter_ref"))

import build_reference  # noqa: E402
import check_packet_fields as chk  # noqa: E402
import compare_cicflowmeter as cmp  # noqa: E402
import make_stress_pcap  # noqa: E402
from scapy.utils import wrpcap  # noqa: E402


def one(p: Path, kind: str, cols):
    r = cmp.compare(p, cols)
    c = chk.check(str(p))
    return {"capture": p.name, "kind": kind, "packets": r["packets"], "flows": r["flows_matched"],
            "flows_only_in_reference": r["flows_only_in_reference"], "flows_only_in_this_project": r["flows_only_in_this_project"],
            "flows_with_all_69_features_equal": r["flows_with_all_features_equal"], "features_that_differ": r["features_that_differ"],
            "zero_duration_flows_rate_excluded": r["zero_duration_flows_rate_excluded"], "single_packet_flows_dropped": r["single_packet_flows_dropped_by_this_project"],
            "ipv4_packets_checked_vs_tshark": c["ipv4_packets_checked"], "packet_field_mismatches": c["mismatching_packets"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ordinary", nargs="*", default=[])
    ap.add_argument("--stress-seeds", nargs="*", type=int, default=[1, 7, 21])
    ap.add_argument("--sessions", type=int, default=800)
    ap.add_argument("--out", default=str(ROOT / "results/flow_validation_parity.json"))
    a = ap.parse_args()
    build_reference.build()
    cols = cmp.model_columns()
    rows = [one(Path(p), "ordinary traffic (real Wi-Fi capture)", cols) for p in a.ordinary]
    demo = ROOT / "samples/demo_synthetic.pcap"
    rows.append(one(demo, "synthetic demo capture", cols))
    tmp = Path(tempfile.mkdtemp())
    for s in a.stress_seeds:
        p = tmp / f"stress_seed{s}.pcap"
        wrpcap(str(p), make_stress_pcap.build(seed=s, sessions=a.sessions))
        rows.append(one(p, "randomised stress capture", cols))
    tot = lambda k: sum(r[k] for r in rows)  # noqa: E731
    res = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "reference": {"tool": "CICFlowMeter (github.com/ahlashkari/CICFlowMeter) flow code, commit eaa853d (2018-02-13), unmodified",
                         "driver": "tools/cicflowmeter_ref/RefDriver.java reads the pcap and fills the packet fields like PacketReader.java"},
           "captures": rows,
           "total": {"captures": len(rows), "packets": tot("packets"), "flows_compared": tot("flows"), "flows_with_all_69_features_equal": tot("flows_with_all_69_features_equal"),
                     "flows_missing_on_either_side": tot("flows_only_in_reference") + tot("flows_only_in_this_project"),
                     "ipv4_packets_checked_vs_tshark": tot("ipv4_packets_checked_vs_tshark"), "packet_field_mismatches": tot("packet_field_mismatches"),
                     "ordinary_traffic_flows": sum(r["flows"] for r in rows if r["kind"].startswith("ordinary"))}}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r['capture']:26s} {r['kind'][:34]:34s} packets {r['packets']:6d}  flows {r['flows']:5d}  all-equal {r['flows_with_all_69_features_equal']:5d}  tshark-checked {r['ipv4_packets_checked_vs_tshark']:6d}  mismatches {r['packet_field_mismatches']}")
    print("TOTAL", res["total"])


if __name__ == "__main__":
    main()
