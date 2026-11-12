"""Check the packet parsing of app/capture.py against Wireshark's own dissector (tshark), packet by packet.

  python tools/check_packet_fields.py capture.pcap

For every IPv4 packet: addresses, protocol, ports, TCP payload length (tcp.len), header length (tcp.hdr_len), TCP flags and window
(tcp.window_size_value), UDP payload (udp.length - 8) are taken from tshark and compared with packet_fields(). Needs tshark (installed with Wireshark).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.capture import packet_fields  # noqa: E402

FIELDS = ["frame.number", "ip.src", "ip.dst", "ip.proto", "ip.frag_offset", "tcp.srcport", "tcp.dstport", "tcp.len", "tcp.hdr_len", "tcp.flags",
          "tcp.window_size_value", "udp.srcport", "udp.dstport", "udp.length"]


def find_tshark() -> str:
    t = shutil.which("tshark") or r"C:\Program Files\Wireshark\tshark.exe"
    if not Path(t).exists() and not shutil.which("tshark"):
        raise SystemExit("tshark not found (it is installed with Wireshark)")
    return t


def tshark_rows(pcap: str):
    cmd = [find_tshark(), "-r", pcap, "-Y", "ip", "-T", "fields", "-E", "separator=|", "-E", "occurrence=f"]
    for f in FIELDS:
        cmd += ["-e", f]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()
    return [dict(zip(FIELDS, line.split("|"))) for line in out]


def check(pcap: str) -> dict:
    from scapy.utils import PcapReader
    ts = {int(r["frame.number"]): r for r in tshark_rows(pcap)}
    n = bad = 0
    kinds = {"tcp": 0, "udp": 0, "other": 0}
    problems = []
    with PcapReader(pcap) as rd:
        for i, pkt in enumerate(rd, start=1):
            f = packet_fields(pkt)
            r = ts.get(i)
            if f is None or r is None:
                if (f is None) != (r is None):
                    bad += 1
                    problems.append((i, "ip presence differs"))
                continue
            n += 1
            src, dst, sp, dp, proto, plen, hlen, flags, win = f
            exp = [r["ip.src"], r["ip.dst"]]
            got = [src, dst]
            if proto == 6:
                kinds["tcp"] += 1
                exp += [int(r["tcp.srcport"]), int(r["tcp.dstport"]), int(r["tcp.len"]), int(r["tcp.hdr_len"]), int(r["tcp.flags"], 16), int(r["tcp.window_size_value"])]
                got += [sp, dp, plen, hlen, flags, win]
            elif proto == 17:
                kinds["udp"] += 1
                exp += [int(r["udp.srcport"]), int(r["udp.dstport"]), max(0, int(r["udp.length"]) - 8)]
                got += [sp, dp, plen]
            else:
                kinds["other"] += 1
                if int(r["ip.proto"]) in (6, 17) and int(r["ip.frag_offset"] or 0) == 0:
                    bad += 1
                    problems.append((i, "tshark sees TCP/UDP, packet_fields does not"))
                continue
            if exp != got:
                bad += 1
                if len(problems) < 10:
                    problems.append((i, {"tshark": exp, "this_project": got}))
    return {"ipv4_packets_checked": n, "tcp": kinds["tcp"], "udp": kinds["udp"], "other_ip": kinds["other"], "mismatching_packets": bad, "examples": problems[:10]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap")
    res = check(ap.parse_args().pcap)
    for k, v in res.items():
        print(f"{k:24s} {v}")
