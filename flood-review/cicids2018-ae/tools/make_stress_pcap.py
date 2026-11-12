"""Write a randomised capture that exercises the awkward cases of flow building, for comparing flow builders (tools/compare_cicflowmeter.py).

  python tools/make_stress_pcap.py out.pcap [--seed 1] [--sessions 600]

Contains: TCP sessions that end with FIN (either side first), with RST, or just stop; sessions longer than the 120 s flow timeout; idle gaps of
more than 5 s; port reuse after a close; one-way UDP and request/response UDP; ICMP and other IP protocols; traffic whose two addresses are equal;
packets sharing one microsecond; retransmitted and out-of-order-looking sequences. All addresses are private or documentation ranges. Nothing here
is real traffic.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.utils import wrpcap

MAC_A, MAC_B = "02:00:00:00:00:01", "02:00:00:00:00:02"


def build(seed: int = 1, sessions: int = 600):
    rnd = random.Random(seed)
    pkts = []                                   # (time_us, packet)
    hosts = [f"10.{i // 250}.{i % 250}.{rnd.randint(2, 250)}" for i in range(40)]
    servers = [f"192.0.2.{i}" for i in range(1, 30)]

    def add(t, pkt):
        pkt.time = t / 1e6
        pkts.append((t, pkt))

    def eth(a_to_b):
        return Ether(src=MAC_A if a_to_b else MAC_B, dst=MAC_B if a_to_b else MAC_A)

    def tcp(src, dst, sp, dp, flags, payload, win, a_to_b, seq=1000, ack=1):
        return eth(a_to_b) / IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags=flags, window=win, seq=seq, ack=ack) / (b"x" * payload)

    start = 1_700_000_000_000_000
    for s in range(sessions):
        kind = rnd.choice(["tcp_fin", "tcp_fin", "tcp_rst", "tcp_open", "tcp_long", "tcp_gap", "udp_rr", "udp_oneway", "icmp", "samehost", "tcp_reuse", "tcp_burst"])
        t = start + rnd.randint(0, 900_000_000)
        c, sv = rnd.choice(hosts), rnd.choice(servers)
        sp, dp = rnd.randint(1024, 65000), rnd.choice([80, 443, 22, 8080, 53, 123, 3389])
        if kind.startswith("tcp"):
            if kind == "samehost":
                sv = c
            w1, w2 = rnd.choice([8192, 29200, 65535, 14600]), rnd.choice([8192, 26883, 65535, 5840])
            add(t, tcp(c, sv, sp, dp, "S", 0, w1, True))
            t += rnd.choice([0, 0, 200, 1500, 30000])
            add(t, tcp(sv, c, dp, sp, "SA", 0, w2, False))
            t += rnd.choice([0, 100, 2500])
            add(t, tcp(c, sv, sp, dp, "A", 0, w1, True))
            n_data = {"tcp_burst": rnd.randint(8, 30), "tcp_long": rnd.randint(4, 12)}.get(kind, rnd.randint(0, 10))
            for i in range(n_data):
                gap = rnd.choice([0, 50, 900, 20_000, 400_000, 1_500_000]) if kind != "tcp_long" else rnd.choice([5_000_000, 30_000_000, 60_000_000])
                if kind == "tcp_gap" and i == n_data // 2:
                    gap = rnd.choice([6_000_000, 9_000_000, 40_000_000])
                t += gap
                fwd = rnd.random() < 0.5
                fl = rnd.choice(["PA", "A", "A", "PA", "PAU"] if fwd else ["PA", "A", "A"])
                add(t, tcp(c if fwd else sv, sv if fwd else c, sp if fwd else dp, dp if fwd else sp, fl, rnd.choice([0, 0, 1, 40, 512, 1448, 1460]), rnd.choice([w1, w2, 211, 502, 0]), fwd))
            t += rnd.choice([0, 300, 10_000])
            if kind in ("tcp_fin", "tcp_reuse", "tcp_burst", "tcp_gap", "samehost") and rnd.random() < 0.9:
                first_client = rnd.random() < 0.7
                a, b, pa, pb = (c, sv, sp, dp) if first_client else (sv, c, dp, sp)
                add(t, tcp(a, b, pa, pb, "FA", 0, 1000, first_client)); t += rnd.choice([0, 100, 3000])
                add(t, tcp(b, a, pb, pa, "FA", 0, 1000, not first_client)); t += rnd.choice([0, 100, 3000])
                add(t, tcp(a, b, pa, pb, "A", 0, 1000, first_client))
                if kind == "tcp_reuse":                                # the same ports are used again after the close
                    t += rnd.choice([1_000_000, 70_000_000, 150_000_000])
                    add(t, tcp(c, sv, sp, dp, "S", 0, w1, True)); add(t + 500, tcp(sv, c, dp, sp, "SA", 0, w2, False)); add(t + 900, tcp(c, sv, sp, dp, "A", 0, w1, True))
            elif kind == "tcp_rst":
                add(t, tcp(c, sv, sp, dp, "R", 0, 0, True))
                if rnd.random() < 0.5:
                    add(t + 50, tcp(sv, c, dp, sp, "RA", 0, 0, False))
        elif kind.startswith("udp"):
            n = rnd.randint(1, 12)
            for i in range(n):
                t += rnd.choice([0, 20, 900, 5_000, 2_000_000, 7_000_000])
                fwd = kind == "udp_oneway" or i % 2 == 0
                a, b, pa, pb = (c, sv, sp, dp) if fwd else (sv, c, dp, sp)
                add(t, eth(fwd) / IP(src=a, dst=b) / UDP(sport=pa, dport=pb) / (b"u" * rnd.choice([0, 12, 48, 120, 512, 1200])))
        else:                                                         # icmp, other protocols, same-address traffic
            for i in range(rnd.randint(1, 6)):
                t += rnd.choice([0, 1000, 1_000_000])
                fwd = i % 2 == 0
                a, b = (c, sv) if fwd else (sv, c)
                if kind == "icmp":
                    add(t, eth(fwd) / IP(src=a, dst=b) / ICMP(type=8 if fwd else 0) / (b"p" * 32))
                else:
                    add(t, eth(fwd) / IP(src=a, dst=b, proto=2) / (b"igmp" * 3))
    pkts.sort(key=lambda x: x[0])
    return [p for _, p in pkts]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sessions", type=int, default=600)
    a = ap.parse_args()
    packets = build(a.seed, a.sessions)
    wrpcap(a.out, packets)
    print(f"wrote {len(packets)} packets to {a.out}")
