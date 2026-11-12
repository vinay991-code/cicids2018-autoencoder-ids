"""Score a Wireshark capture with the saved autoencoders. Works on a file or on a live stream from tshark.

  python tools/pcap_score.py capture.pcapng --csv flows.csv          # a capture saved from Wireshark (pcap or pcapng)
  tshark -i 5 -F pcap -w - | python tools/pcap_score.py -             # live: tshark writes pcap to a pipe, this script reads it
  python tools/pcap_score.py - --tshark-live 5 --seconds 60           # same, but this script starts tshark for you

Each flow that any detector flags is printed with a Wireshark display filter; paste it into Wireshark's filter bar to see its packets.
NOTE: flow features follow CICFlowMeter's flow logic and were checked against it (reports/FLOW_VALIDATION.md); a flag is a lead to inspect, not a verdict.
Live capture needs Npcap and rights to capture; capture only on networks you own or may monitor.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from app.pcapjob import Analysis, analyse_source, to_csv  # noqa: E402
from app.service import Detector  # noqa: E402


def find_tshark():
    p = shutil.which("tshark")
    if p:
        return p
    for d in (os.environ.get("WIRESHARK_DIR"), r"C:\Program Files\Wireshark", r"C:\Program Files (x86)\Wireshark"):
        if d and (Path(d) / "tshark.exe").exists():
            return str(Path(d) / "tshark.exe")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="capture file (pcap or pcapng), or - to read a pcap stream from standard input")
    ap.add_argument("--csv", help="write every flow to this CSV file at the end")
    ap.add_argument("--detector", default="hybrid", choices=["unsup_mixed", "normal_only", "supervised", "supervised_recon", "hybrid"],
                    help="which detector's alerts to print (default: the supervised-OR-unsupervised rule)")
    ap.add_argument("--max-print", type=int, default=50, help="print at most this many flagged flows (all go to the CSV)")
    ap.add_argument("--tshark-live", metavar="IFACE", help="start tshark on this interface and read its output (needs Npcap)")
    ap.add_argument("--seconds", type=int, default=0, help="with --tshark-live: stop after this many seconds (0 = until Ctrl+C)")
    a = ap.parse_args()

    det = Detector()
    printed = [0]

    def show(rec):
        if a.detector in rec["alerts"] and printed[0] < a.max_print:
            printed[0] += 1
            print(f"[{time.strftime('%H:%M:%S', time.localtime(rec['start']))}] {rec['proto']} {rec['src']}:{rec['sport']} -> {rec['dst']}:{rec['dport']}  "
                  f"alerts={','.join(rec['alerts'])}  sup_p={rec['scores']['supervised']:.3f}  out_of_range={rec['out_of_range']}", flush=True)
            print(f"    wireshark filter: {rec['filter']}", flush=True)

    proc = None
    stop_at = time.time() + a.seconds if (a.tshark_live and a.seconds) else None
    try:
        if a.tshark_live:
            t = find_tshark()
            if not t:
                sys.exit("tshark not found (install Wireshark or set WIRESHARK_DIR)")
            proc = subprocess.Popen([t, "-i", a.tshark_live, "-F", "pcap", "-w", "-", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            src = proc.stdout
        elif a.source == "-":
            src = sys.stdin.buffer
        else:
            src = a.source
        an = analyse_source(det, src, on_flow=show, stop=(lambda: time.time() > stop_at) if stop_at else None)
    except KeyboardInterrupt:
        sys.exit("interrupted")
    finally:
        if proc:
            proc.terminate()
            err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
            if err and "Capturing on" not in err:
                print("tshark:", err[:500], file=sys.stderr)
    s = an.summary()
    print("\nSummary:", s)
    if a.csv:
        Path(a.csv).write_text(to_csv(an.flows), encoding="utf-8")
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
