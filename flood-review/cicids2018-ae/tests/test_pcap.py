"""Tests for the Wireshark/PCAP integration. tshark (from Wireshark) is used as an independent check when it is installed:
its display filters must select exactly the packets that this code assigns to each flow."""
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from scapy.all import ARP, IP, TCP, UDP, Ether, IPv6, wrpcap  # noqa: E402

from app.main import app, find_tshark  # noqa: E402
from app.pcapjob import analyse_source, to_csv  # noqa: E402
from app.service import Detector  # noqa: E402

client = TestClient(app)
DET = Detector()
TSHARK = find_tshark()
EDITCAP = shutil.which("editcap") or (str(Path(TSHARK).with_name("editcap.exe")) if TSHARK and Path(TSHARK).with_name("editcap.exe").exists() else None)
needs_tshark = pytest.mark.skipif(not TSHARK, reason="tshark not installed")


ETH = dict(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")     # fixed MACs: scapy must not try to resolve them


def tcp_flow(a, b, sport, dport, t0, rng, n_data=5, close=True):
    pk, t = [], t0
    def add(src, dst, sp, dp, flags, payload=b"", win=8192):
        nonlocal t
        p = Ether(**ETH) / IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags=flags, window=win) / payload
        p.time = t
        t += rng.uniform(0.001, 0.05)
        pk.append(p)
    add(a, b, sport, dport, "S")
    add(b, a, dport, sport, "SA", win=4096)
    add(a, b, sport, dport, "A")
    for _ in range(n_data):
        add(a, b, sport, dport, "PA", b"x" * rng.randint(1, 400))
        add(b, a, dport, sport, "PA", b"y" * rng.randint(1, 1200), win=4096)
    if close:
        add(a, b, sport, dport, "FA")
        add(b, a, dport, sport, "FA", win=4096)
        add(a, b, sport, dport, "A")
    return pk


def make_pcap(path, n_tcp=25, seed=1):
    rng = random.Random(seed)
    pk = []
    for i in range(n_tcp):
        pk += tcp_flow(f"10.0.{i % 4}.{i + 10}", "192.168.1.5", 40000 + i, 80 if i % 2 else 443, 1_700_000_000 + i * 0.7, rng, n_data=rng.randint(1, 8))
    for i in range(5):                                           # UDP request/response
        for k, (s, d, sp, dp) in enumerate([("10.0.9.1", "8.8.8.8", 5300 + i, 53), ("8.8.8.8", "10.0.9.1", 53, 5300 + i)]):
            p = Ether(**ETH) / IP(src=s, dst=d) / UDP(sport=sp, dport=dp) / (b"q" * (30 + 5 * i))
            p.time = 1_700_000_100 + i + 0.01 * k
            pk.append(p)
    pk += tcp_flow("10.0.8.8", "192.168.1.5", 50000, 22, 1_700_000_200, rng, n_data=2, close=False)     # stays open: released at the end
    arp = Ether(**ETH) / ARP(hwsrc=ETH["src"], psrc="10.0.0.1", pdst="10.0.0.2"); arp.time = 1_700_000_001; pk.append(arp)                                         # ignored
    v6 = Ether(**ETH) / IPv6(src="2001:db8::1", dst="2001:db8::2") / TCP(); v6.time = 1_700_000_002; pk.append(v6)                                  # ignored (IPv6)
    pk.sort(key=lambda p: p.time)
    wrpcap(str(path), pk)
    return pk


@pytest.fixture(scope="module")
def pcap(tmp_path_factory):
    p = tmp_path_factory.mktemp("pcap") / "t.pcap"
    packets = make_pcap(p)
    return p, packets


def test_analysis_counts(pcap):
    p, packets = pcap
    a = analyse_source(DET, str(p))
    s = a.summary()
    assert s["packets_read"] == len(packets)
    assert s["packets_ignored"] == 2                                  # the ARP and the IPv6 packet
    # the tool ends a flow at the first FIN and the rest of the closing handshake (FIN-ACK, ACK) forms a second flow: 25 sessions -> 50 flows;
    # plus 5 UDP exchanges and 1 session still open at the end. CICFlowMeter's own code gives the same 56 flows on this file (tools/compare_cicflowmeter.py).
    assert s["flows"] == 25 * 2 + 5 + 1
    for f in a.flows:
        assert set(f["scores"]) == {"unsup_mixed", "normal_only", "supervised", "supervised_recon"}
        assert f["fwd_pkts"] >= 1 and f["out_of_range"] >= 0 and f["filter"].startswith("ip.addr ==")


@needs_tshark
def test_wireshark_filters_select_exactly_the_flow_packets(pcap):
    """Independent check: Wireshark's own dissector, given this code's filter, must return exactly fwd+bwd packets of each flow."""
    p, _ = pcap
    a = analyse_source(DET, str(p))
    bad = []
    for f in a.flows:
        r = subprocess.run([TSHARK, "-r", str(p), "-Y", f["filter"], "-T", "fields", "-e", "frame.number"], capture_output=True, text=True, timeout=60)
        n = len([x for x in r.stdout.split() if x])
        if n != f["fwd_pkts"] + f["bwd_pkts"]:
            bad.append((f["id"], n, f["fwd_pkts"] + f["bwd_pkts"], r.stderr[:100]))
    assert not bad, bad


@needs_tshark
def test_tshark_conversation_counts_agree(pcap):
    p, _ = pcap
    a = analyse_source(DET, str(p))
    # independent total: number of IPv4 TCP packets according to tshark's own dissector
    r2 =subprocess.run([TSHARK, "-r", str(p), "-Y", "ip && tcp", "-T", "fields", "-e", "frame.number"], capture_output=True, text=True, timeout=60)
    tshark_tcp = len([x for x in r2.stdout.split() if x])
    mine = sum(f["fwd_pkts"] + f["bwd_pkts"] for f in a.flows if f["proto"] == "TCP")
    assert tshark_tcp == mine


def test_csv_export(pcap):
    p, _ = pcap
    a = analyse_source(DET, str(p))
    text = to_csv(a.flows)
    lines = text.strip().splitlines()
    assert len(lines) == len(a.flows) + 1 and lines[0].startswith("id,start_epoch")


def test_upload_api_roundtrip(pcap):
    p, _ = pcap
    with open(p, "rb") as fh:
        r = client.post("/api/pcap", files={"file": ("t.pcap", fh, "application/octet-stream")})
    assert r.status_code == 200, r.text
    jid = r.json()["job"]
    for _ in range(100):
        st = client.get(f"/api/pcap/{jid}").json()
        if st["status"] != "running":
            break
        time.sleep(0.2)
    assert st["status"] == "done", st
    assert st["summary"]["flows"] == 56 and len(st["flows"]) == 56
    assert client.get(f"/api/pcap/{jid}/csv").text.startswith("id,start_epoch")
    assert client.delete(f"/api/pcap/{jid}").json() == {"deleted": True}
    assert client.get(f"/api/pcap/{jid}").status_code == 404


def test_upload_rejects_non_pcap(tmp_path):
    f = tmp_path / "x.pcap"
    f.write_bytes(b"this is not a capture file")
    with open(f, "rb") as fh:
        assert client.post("/api/pcap", files={"file": ("x.pcap", fh)}).status_code == 415


def test_upload_size_limit(monkeypatch, tmp_path):
    import app.main as m
    monkeypatch.setattr(m, "MAX_UPLOAD", 1000)
    f = tmp_path / "big.pcap"
    f.write_bytes(b"\xd4\xc3\xb2\xa1" + b"0" * 5000)
    with open(f, "rb") as fh:
        assert client.post("/api/pcap", files={"file": ("big.pcap", fh)}).status_code == 413


@pytest.mark.skipif(not EDITCAP, reason="editcap not installed")
def test_pcapng_input(pcap, tmp_path):
    p, packets = pcap
    ng = tmp_path / "t.pcapng"
    subprocess.run([EDITCAP, "-F", "pcapng", str(p), str(ng)], check=True, capture_output=True)
    assert ng.read_bytes()[:4] == b"\x0a\x0d\x0d\x0a"
    a = analyse_source(DET, str(ng))
    assert a.summary()["flows"] == 56 and a.summary()["packets_read"] == len(packets)


def test_cli_reads_stream_from_stdin(pcap):
    p, _ = pcap
    with open(p, "rb") as fh:
        r = subprocess.run([sys.executable, "tools/pcap_score.py", "-"], stdin=fh, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert "'flows': 56" in r.stdout


def test_cli_file_and_csv(pcap, tmp_path):
    p, _ = pcap
    out = tmp_path / "f.csv"
    r = subprocess.run([sys.executable, "tools/pcap_score.py", str(p), "--csv", str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and out.exists() and len(out.read_text().splitlines()) == 57


def test_wireshark_info_endpoint():
    j = client.get("/api/wireshark/info").json()
    assert "live_pipe_command" in j and "warning" in j


@pytest.mark.skipif(not EDITCAP, reason="editcap not installed")
def test_headers_only_capture_gives_same_features(pcap, tmp_path):
    """A capture cut to 96 bytes per packet (like the live headers-only file) must give the same flows and byte counts."""
    p, _ = pcap
    cut = tmp_path / "cut.pcap"
    subprocess.run([EDITCAP, "-s", "96", str(p), str(cut)], check=True, capture_output=True)
    a, b = analyse_source(DET, str(p)), analyse_source(DET, str(cut))
    key = lambda f: (f["src"], f["sport"], f["dst"], f["dport"], f["start"])
    A, B = {key(f): f for f in a.flows}, {key(f): f for f in b.flows}
    assert A.keys() == B.keys()
    for k in A:
        assert (A[k]["fwd_pkts"], A[k]["bwd_pkts"], A[k]["bytes"]) == (B[k]["fwd_pkts"], B[k]["bwd_pkts"], B[k]["bytes"])
