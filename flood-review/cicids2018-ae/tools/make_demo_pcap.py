"""Write samples/demo_synthetic.pcap: a small SYNTHETIC capture (25 TCP sessions, 5 UDP exchanges, one open session, an ARP and an IPv6 packet)
so the Wireshark/PCAP tab can be tried without real traffic. It is made-up traffic: do not read anything into its scores.
Usage: python tools/make_demo_pcap.py"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_pcap import make_pcap  # noqa: E402  (the same generator the tests use)

if __name__ == "__main__":
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    make_pcap(out / "demo_synthetic.pcap", n_tcp=25, seed=1)
    print("wrote", out / "demo_synthetic.pcap")
