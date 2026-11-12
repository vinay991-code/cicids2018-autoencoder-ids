"""Live capture from a network interface (Npcap on Windows) -> flows -> the saved models. The flow rules are those of CICFlowMeter, see app/capture.py.

While it runs, a HEADERS-ONLY pcap (the first 96 bytes of each packet: Ethernet, IP and TCP/UDP headers, almost never payload) can be written to
captures/ so the same flows can be opened in Wireshark and found with the display filters the dashboard shows. Turn that off to keep nothing on
disk. The file is capped (default 50 MB); after that only the live view continues.
"""
from __future__ import annotations

import struct
import threading
import time
from collections import deque
from pathlib import Path

from .pcapjob import Analysis
from .service import ROOT, Detector

SNAPLEN = 96


def list_interfaces() -> list[dict]:
    """Interfaces scapy/Npcap can capture on: name (what to pass back), description, IPv4 address."""
    from scapy.interfaces import get_working_ifaces
    out = []
    for i in get_working_ifaces():
        out.append({"name": i.network_name if hasattr(i, "network_name") and i.network_name else i.name, "label": i.name,
                    "description": getattr(i, "description", "") or "", "ip": getattr(i, "ip", "") or ""})
    return out


class HeaderPcap:
    """Minimal pcap writer that stores only the first SNAPLEN bytes of each packet (and records the true length)."""

    def __init__(self, path: Path, linktype: int, cap_bytes: int):
        self.f = open(path, "wb")
        self.f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, SNAPLEN, linktype))
        self.written, self.cap, self.path = 24, cap_bytes, path
        self.full = False

    def write(self, raw: bytes, t: float):
        if self.full:
            return
        part = raw[:SNAPLEN]
        if self.written + 16 + len(part) > self.cap:
            self.full = True
            return
        sec = int(t)
        self.f.write(struct.pack("<IIII", sec, int((t - sec) * 1e6), len(part), len(raw)) + part)
        self.written += 16 + len(part)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class LiveCapture:
    def __init__(self, det: Detector, iface: str | None, bpf: str = "ip", save: bool = True, max_save_mb: int = 50):
        self.det, self.iface, self.bpf, self.save, self.max_save = det, iface, bpf, save, max_save_mb * 1024 * 1024
        self.an = Analysis(det, on_flow=self._on_flow, batch=1, store=False)
        self.new: deque = deque(maxlen=2000)             # flow records not yet sent to the browser
        self.recent: deque = deque(maxlen=300)
        self.packets = self.flows = self.flagged = 0
        self.running, self.error, self.started = False, None, None
        self.save_path: Path | None = None
        self.save_full = False
        self._thread = None
        self._writer: HeaderPcap | None = None

    def _on_flow(self, rec):
        self.flows += 1
        if rec["alerts"]:
            self.flagged += 1
        self.new.append(rec)
        self.recent.append(rec)

    def _handle(self, pkt):
        self.packets += 1
        if self.save:
            try:
                if self._writer is None:
                    from scapy.layers.l2 import Ether, Loopback
                    lt = 1 if Ether in pkt else (0 if Loopback in pkt else 101)
                    d = ROOT / "captures"
                    d.mkdir(exist_ok=True)
                    self.save_path = d / f"live_{time.strftime('%Y%m%d_%H%M%S')}.pcap"
                    self._writer = HeaderPcap(self.save_path, lt, self.max_save)
                self._writer.write(bytes(pkt), float(pkt.time))
                self.save_full = self._writer.full
            except Exception as e:                      # a disk problem must not stop the live view
                self.save, self.error = False, f"saving stopped: {type(e).__name__}: {e}"
        self.an.packet(pkt)

    def _run(self):
        try:
            from scapy.all import sniff
            kw = {"iface": self.iface} if self.iface else {}
            while self.running:
                sniff(filter=self.bpf, prn=self._handle, store=False, timeout=1, **kw)
                self.an.pending += self.an.table.expire(int(time.time() * 1e6))
                self.an.score_pending()
        except Exception as e:                          # no Npcap, interface not usable, or the driver is restricted to administrators
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.running = False
            self.an.pending += self.an.table.flush()
            try:
                self.an.score_pending()
            except Exception:
                pass
            if self._writer:
                self._writer.close()

    def start(self):
        if self.running:
            return
        self.error, self.running, self.started = None, True, time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def status(self) -> dict:
        return {"running": self.running, "iface": self.iface, "packets": self.packets, "flows": self.flows, "flagged_flows": self.flagged,
                "ignored_packets": self.an.skipped, "error": self.error, "saving": bool(self._writer or self.save),
                "save_path": str(self.save_path) if self.save_path else None, "save_full": self.save_full,
                "seconds": int(time.time() - self.started) if self.started else 0}
