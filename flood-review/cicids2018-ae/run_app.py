"""Start the dashboard/backend:  python run_app.py [--host 127.0.0.1] [--port 8000] [--open] [--enable-capture]
Default host is 127.0.0.1 (this computer only). Do not bind to 0.0.0.0 on an untrusted network without setting APP_API_KEY.

If the port is already taken: when it is this application you are told so (and the page is opened with --open);
when it is another program the next free port is used and the new address is printed."""
import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

logging.getLogger("scapy.loading").setLevel(logging.ERROR)      # hides scapy's "No libpcap provider" notice; file analysis does not need it
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

import uvicorn  # noqa: E402


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def is_this_app(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=2) as r:
            j = json.loads(r.read().decode())
        return j.get("status") == "ok" and "replay_data_available" in j
    except Exception:
        return False


def open_when_ready(url: str, host: str, port: int):
    for _ in range(120):
        if is_this_app(host, port):
            webbrowser.open(url)
            return
        time.sleep(0.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--enable-capture", action="store_true", help="allow live packet capture (needs Npcap); same as setting ENABLE_CAPTURE=1")
    ap.add_argument("--open", action="store_true", help="open the dashboard in the default browser once it is ready")
    a = ap.parse_args()
    if a.enable_capture:
        os.environ["ENABLE_CAPTURE"] = "1"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())

    port = a.port
    if not port_free(a.host, port):
        if is_this_app(a.host, port):
            url = f"http://{a.host}:{port}"
            print(f"The dashboard is already running at {url} (another window or process started it). Nothing more to do.")
            if a.open:
                webbrowser.open(url)
            sys.exit(0)
        for p in range(port + 1, port + 21):
            if port_free(a.host, p):
                print(f"Port {port} is used by another program; starting on port {p} instead.")
                port = p
                break
        else:
            sys.exit(f"Ports {port} to {port + 20} are all in use. Close the program using port {port} or choose another with --port.")
    url = f"http://{a.host}:{port}"
    print(f"Starting the dashboard at {url}  (Ctrl+C to stop)")
    if a.open:
        threading.Thread(target=open_when_ready, args=(url, a.host, port), daemon=True).start()
    uvicorn.run("app.main:app", host=a.host, port=port, log_level="info")
