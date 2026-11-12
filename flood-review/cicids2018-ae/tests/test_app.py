"""Tests for the dashboard backend. Run from the project folder:  python -m pytest tests -q
Tests that need the large data files are skipped when those files are missing (for example inside the Docker image)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app import replay  # noqa: E402
from app.capture import FlowTable  # noqa: E402
from app.main import app  # noqa: E402
from app.service import Detector  # noqa: E402

HAVE_DATA = (ROOT / "data/processed/X_test.npy").exists() and (ROOT / "data/processed/meta_test.pkl").exists()
HAVE_RAW = (ROOT / "data/interim/clean_stats.json").exists() and any((ROOT / "data/interim").glob("*.f32"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="processed test data not present")
needs_raw = pytest.mark.skipif(not (HAVE_DATA and HAVE_RAW), reason="raw interim data not present")

client = TestClient(app)
DET = Detector()


def raw_row_from_data(i=0):
    import pandas as pd
    st = json.loads((ROOT / "data/interim/clean_stats.json").read_text())
    feats = st["features"]
    meta = pd.read_pickle(ROOT / "data/processed/meta_test.pkl")
    stem, r = meta.source_file.iloc[i], int(meta.row_in_file.iloc[i])
    row = np.asarray(np.memmap(ROOT / f"data/interim/{stem}.f32", dtype=np.float32, mode="r").reshape(-1, len(feats))[r])
    d = {c: float(row[feats.index(c)]) for c in DET.columns}
    d["protocol"] = int(row[feats.index("protocol")])
    return d


# ---------------------------------------------------------------- service ----------------------------------------------------
@needs_data
def test_scores_reproduce_saved_scores():
    X = np.load(ROOT / "data/processed/X_test.npy", mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(X), 3000, replace=False))
    out = DET.score(np.asarray(X[idx]))
    for k, f in [("unsup_mixed", "scores_unsup_mixed_test.npy"), ("normal_only", "scores_normal_only_test.npy"),
                 ("supervised", "scores_supervised_test.npy"), ("supervised_recon", "scores_supervised_recon_test.npy")]:
        ref = np.load(ROOT / "results" / f, mmap_mode="r")[idx]
        assert np.abs(out["scores"][k] - ref).max() < 1e-5, k


@needs_raw
def test_transform_matches_phase2_pipeline():
    import pandas as pd
    X = np.load(ROOT / "data/processed/X_test.npy", mmap_mode="r")
    st = json.loads((ROOT / "data/interim/clean_stats.json").read_text())
    feats = st["features"]
    meta = pd.read_pickle(ROOT / "data/processed/meta_test.pkl")
    for i in np.random.default_rng(1).choice(len(meta), 50, replace=False):
        stem, r = meta.source_file.iloc[i], int(meta.row_in_file.iloc[i])
        row = np.asarray(np.memmap(ROOT / f"data/interim/{stem}.f32", dtype=np.float32, mode="r").reshape(-1, len(feats))[r])
        z = DET.transform(row[[feats.index(c) for c in DET.columns]][None], np.array([int(row[feats.index("protocol")])]))[0]
        assert np.abs(z - np.asarray(X[i])).max() < 1e-5


def test_thresholds_are_the_fixed_rules():
    assert DET.thresholds["supervised"] == 0.5 and DET.thresholds["supervised_recon"] == 0.5
    p3 = json.loads((ROOT / "results/phase3_results.json").read_text())["variants"]
    assert DET.thresholds["unsup_mixed"] == p3["unsup_mixed"]["primary"]["threshold"]


def test_hybrid_is_or_of_supervised_and_unsup():
    X = np.random.default_rng(0).random((200, DET.dim)).astype(np.float32)
    out = DET.score(X)
    assert (out["alerts"]["hybrid"] == (out["alerts"]["supervised"] | out["alerts"]["unsup_mixed"])).all()


# ---------------------------------------------------------------- API ---------------------------------------------------------
def test_health_models_results():
    assert client.get("/api/health").json()["status"] == "ok"
    m = client.get("/api/models").json()
    assert m["input_dim"] == 72 and len(m["models"]) == 5
    r = client.get("/api/results")
    assert r.status_code == 200 and set(r.json()["table1_run_a"]) == {"unsup_mixed", "normal_only", "supervised", "supervised_recon"}


def test_results_numbers_equal_files():
    r = client.get("/api/results").json()["table1_run_a"]
    p4 = json.loads((ROOT / "results/phase4_results.json").read_text())["variants"]
    assert r["supervised"]["roc_auc"] == p4["supervised"]["ranking"]["roc_auc"]
    assert r["supervised"]["recall"] == p4["supervised"]["threshold_0.5"]["recall"]


@needs_raw
def test_score_endpoint_matches_direct_scoring():
    row = raw_row_from_data(5)
    resp = client.post("/api/score", json={"rows": [row]})
    assert resp.status_code == 200
    direct = DET.score(DET.transform(*DET.rows_to_arrays([row])))
    got = resp.json()["results"][0]["scores"]
    for k in direct["scores"]:
        assert abs(got[k] - float(direct["scores"][k][0])) < 1e-9


def test_score_rejects_bad_input():
    assert client.post("/api/score", json={"rows": [{"flow_duration": 1}]}).status_code == 422
    assert client.post("/api/score", json={"rows": []}).status_code == 422
    row = {c: 1.0 for c in DET.columns}
    row["protocol"] = 6
    row["flow_duration"] = float("nan")
    # NaN is not valid JSON, so send a string that float() cannot read
    row["flow_duration"] = "abc"
    assert client.post("/api/score", json={"rows": [row]}).status_code == 422
    big = {c: 1.0 for c in DET.columns}
    big["protocol"] = 6
    assert client.post("/api/score", json={"rows": [big] * 5001}).status_code == 413


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "secret")
    row = {c: 1.0 for c in DET.columns}
    row["protocol"] = 6
    assert client.post("/api/score", json={"rows": [row]}).status_code == 401
    assert client.post("/api/score", json={"rows": [row]}, headers={"X-API-Key": "secret"}).status_code == 200


def test_capture_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_CAPTURE", raising=False)
    assert client.post("/api/capture/start").status_code == 403
    assert client.get("/api/capture/stream").status_code == 403
    assert client.get("/api/capture/status").json()["enabled"] is False
    assert client.get("/api/capture/interfaces").status_code == 403


def test_figure_whitelist():
    assert client.get("/figures/../scaler.json").status_code in (404, 422)
    assert client.get("/figures/secret.png").status_code == 404


# ---------------------------------------------------------------- replay -----------------------------------------------------
@needs_data
def test_replay_counts_consistent():
    ticks = list(replay.ticks(DET, start=1000, batch=300, max_ticks=3))
    assert len(ticks) == 3
    for t in ticks:
        for m, c in t["models"].items():
            assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == t["n"] == 300
            assert c["flagged"] == c["tp"] + c["fp"]


@needs_data
def test_replay_stream_endpoint():
    with client.stream("GET", "/api/replay/stream?start=0&batch=100&interval=0.05&ticks=2") as r:
        text = "".join(r.iter_text())
    assert text.count("data: ") >= 3 and "event: end" in text


# ---------------------------------------------------------------- capture feature builder ---------------------------------------
def _flow(packets):
    t = FlowTable()
    done = []
    for p in packets:
        done += t.packet(*p)
    return t, done


def test_flow_features_hand_computed():
    """Expected values worked out by hand from the rules of app/capture.py (CICFlowMeter's flow logic), not from the code's output."""
    A, B = "10.0.0.1", "10.0.0.2"
    # (src, dst, sport, dport, proto, t_us, payload, header, flags, window)
    pk = [(A, B, 1234, 80, 6, 0, 0, 32, 0x02, 8192),               # SYN (first packet of the flow)
          (B, A, 80, 1234, 6, 1000, 0, 32, 0x12, 4096),           # SYN-ACK
          (A, B, 1234, 80, 6, 2000, 100, 20, 0x18, 8192),         # PSH-ACK, 100 bytes
          (B, A, 80, 1234, 6, 4000, 300, 20, 0x18, 5000),         # PSH-ACK, 300 bytes
          (A, B, 1234, 80, 6, 6000, 0, 20, 0x11, 8192),           # FIN-ACK: the FIN packet is the last packet of the flow
          (B, A, 80, 1234, 6, 7000, 0, 20, 0x11, 4096),           # the other FIN-ACK starts a NEW flow ...
          (A, B, 1234, 80, 6, 8000, 0, 20, 0x10, 8192)]           # ... which the final ACK joins
    tab, done = _flow(pk)
    assert len(done) == 1                                          # flow 1 ended at the FIN
    f = done[0][0]
    for c in DET.columns:
        assert c in f, c
    assert f["protocol"] == 6
    assert f["tot_fwd_pkts"] == 3 and f["tot_bwd_pkts"] == 2       # packets 1, 3, 5 forward; 2 and 4 backward
    assert f["totlen_fwd_pkts"] == 100 and f["totlen_bwd_pkts"] == 300
    assert f["flow_duration"] == 6000
    assert f["fwd_header_len"] == 32 + 20 + 20 and f["bwd_header_len"] == 32 + 20
    assert f["fwd_seg_size_min"] == 20
    # flags come from the first packet only (a SYN), in the column order of the original tool: the column called PSH holds SYN
    assert f["psh_flag_cnt"] == 1 and f["ack_flag_cnt"] == 0 and f["syn_flag_cnt"] == 0 and f["urg_flag_cnt"] == 0 and f["fwd_psh_flags"] == 0
    assert f["init_fwd_win_byts"] == 8192 and f["init_bwd_win_byts"] == 5000      # backward: the window of the LAST backward packet
    assert f["fwd_act_data_pkts"] == 1                              # forward packets with data, not counting the first packet
    # packet length statistics count the first packet (0 bytes) twice: values 0,0,0,100,300,0
    assert abs(f["pkt_len_mean"] - 400 / 6) < 1e-9 and f["pkt_len_max"] == 300 and f["pkt_len_min"] == 0
    assert abs(f["pkt_size_avg"] - 400 / 5) < 1e-9                  # total payload / packets
    assert f["flow_iat_max"] == 2000 and f["flow_iat_min"] == 1000 and f["flow_iat_mean"] == 1500
    assert f["fwd_iat_tot"] == 6000 and f["fwd_iat_max"] == 4000 and f["fwd_iat_min"] == 2000
    assert f["bwd_iat_tot"] == 3000
    assert abs(f["flow_pkts_s"] - 5 / 0.006) < 1e-6 and abs(f["flow_byts_s"] - 400 / 0.006) < 1e-6
    assert f["down_up_ratio"] == 0                                  # integer part of 2 / 3
    assert f["subflow_fwd_pkts"] == 3 and f["subflow_bwd_byts"] == 300
    assert f["active_mean"] == 0 and f["idle_mean"] == 0
    # the rest of the closing handshake is a second flow: starts with a FIN-ACK (the column called URG holds FIN), 2 packets
    rest = tab.flush()
    assert len(rest) == 1
    g = rest[0][0]
    assert g["tot_fwd_pkts"] == 1 and g["tot_bwd_pkts"] == 1 and g["urg_flag_cnt"] == 1 and g["ack_flag_cnt"] == 1
    # the feature dict scores without error
    out = DET.score(DET.transform(*DET.rows_to_arrays([f])))
    assert set(out["scores"]) == {"unsup_mixed", "normal_only", "supervised", "supervised_recon"}


def test_flow_timeout_single_packets_and_rst():
    t = FlowTable(timeout_us=1_000_000)
    assert t.packet("10.1.0.1", "10.1.0.2", 1, 2, 17, 0, 10, 8) == []
    assert t.packet("10.1.0.1", "10.1.0.2", 1, 2, 17, 500_000, 10, 8) == []
    done = t.packet("10.1.0.1", "10.1.0.2", 1, 2, 17, 1_500_000, 10, 8)       # more than the timeout after the FIRST packet: old flow released, a new one starts
    assert len(done) == 1 and done[0][0]["tot_fwd_pkts"] == 2 and done[0][0]["flow_duration"] == 500_000
    assert t.flush() == [] and t.single_packet_flows == 1        # a flow of one packet is never reported
    t = FlowTable()
    for p in [("10.2.0.1", "10.2.0.2", 3, 4, 6, 0, 0, 20, 0x02, 100), ("10.2.0.2", "10.2.0.1", 4, 3, 6, 10, 0, 20, 0x04, 0), ("10.2.0.1", "10.2.0.2", 3, 4, 6, 20, 0, 20, 0x10, 100)]:
        assert t.packet(*p) == []                                  # RST does not end a flow
    (f, _), = t.flush()
    assert f["tot_fwd_pkts"] == 2 and f["tot_bwd_pkts"] == 1


def test_other_ip_protocols_and_active_idle():
    t = FlowTable()
    t.packet("10.1.0.1", "10.1.0.2", 0, 0, 1, 0, 32, 8, 0)                       # ICMP echo request ...
    t.packet("10.1.0.2", "10.1.0.1", 0, 0, 1, 1000, 32, 8, 0)                    # ... and reply: one flow of protocol 0, no ports, no payload
    (f, m), = t.flush()
    assert f["protocol"] == 0 and m["sport"] == 0 and m["dport"] == 0 and f["totlen_fwd_pkts"] == 0 and f["fwd_header_len"] == 0
    assert f["init_fwd_win_byts"] == -1 and f["init_bwd_win_byts"] == -1 and f["tot_fwd_pkts"] == 1 and f["tot_bwd_pkts"] == 1
    t = FlowTable()
    for ts in (0, 1_000_000, 7_000_000, 7_500_000):                # an idle gap of 6 s (> 5 s) after 1 s of activity
        t.packet("10.1.0.1", "10.1.0.2", 5, 6, 17, ts, 10, 8)
    (f, _), = t.flush()
    assert f["active_mean"] == 1_000_000 and f["idle_mean"] == 6_000_000 and f["idle_min"] == f["idle_max"] == 6_000_000   # the last active period is not recorded
    assert f["init_fwd_win_byts"] == -1 and f["fwd_header_len"] == 32 and f["fwd_act_data_pkts"] == 3


def test_same_address_flows_are_separate_per_direction():
    """Quirk kept on purpose: when both addresses are equal (loopback) the original tool builds one flow per direction."""
    t = FlowTable()
    t.packet("1.1.1.1", "1.1.1.1", 10, 20, 17, 0, 5, 8)
    t.packet("1.1.1.1", "1.1.1.1", 20, 10, 17, 10, 5, 8)
    assert len(t.flows) == 2


def test_capture_columns_match_scaler():
    # every column the scaler expects must come out of the flow builder (no silent zero-filling)
    t = FlowTable()
    t.packet("10.1.0.1", "10.1.0.2", 1, 2, 17, 0, 10, 8)
    t.packet("10.1.0.2", "10.1.0.1", 2, 1, 17, 5, 10, 8)
    f = list(t.flows.values())[0].features()
    assert set(DET.columns) <= set(f)


# ---------------------------------------------------------------- launcher -----------------------------------------------------
def test_launcher_port_helpers():
    import socket

    import run_app
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        assert run_app.port_free("127.0.0.1", port) is False          # taken by this test socket
        assert run_app.is_this_app("127.0.0.1", port) is False        # and it is not the dashboard
    finally:
        s.close()
    assert run_app.port_free("127.0.0.1", port) is True


# ---------------------------------------------------------------- live capture (needs Npcap; skipped when it is absent) -----------
def _loopback_iface():
    try:
        from app.livecapture import list_interfaces
        for i in list_interfaces():
            if "loopback" in (i["description"] + i["label"]).lower():
                return i["name"]
    except Exception:
        pass
    return None


@pytest.mark.skipif(_loopback_iface() is None, reason="Npcap / loopback interface not available")
def test_live_capture_on_loopback_end_to_end(monkeypatch):
    import socket
    import subprocess
    import threading
    import time
    from app.livecapture import LiveCapture
    from app.pcapjob import analyse_source
    iface = _loopback_iface()
    cap = LiveCapture(DET, iface, save=True)
    cap.start()
    time.sleep(1.5)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]

    def serve():
        for _ in range(2):
            c, _a = srv.accept()
            c.recv(100)
            c.sendall(b"x" * 300)
            c.close()
    threading.Thread(target=serve, daemon=True).start()
    for _ in range(2):
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(b"hi")
        s.recv(1000)
        s.close()
        time.sleep(0.2)
    time.sleep(3)
    cap.stop()
    time.sleep(3)
    srv.close()
    ours = [r for r in cap.recent if port in (r["sport"], r["dport"])]
    assert len(ours) >= 2, cap.status()
    assert cap.save_path and cap.save_path.exists()
    a = analyse_source(DET, str(cap.save_path))
    assert a.summary()["flows"] >= 2
    if TSHARK_PATH:
        for r in ours:
            out = subprocess.run([TSHARK_PATH, "-r", str(cap.save_path), "-Y", r["filter"], "-T", "fields", "-e", "frame.number"], capture_output=True, text=True)
            assert len(out.stdout.split()) == r["fwd_pkts"] + r["bwd_pkts"]
    cap.save_path.unlink()


TSHARK_PATH = None
try:
    from app.main import find_tshark as _ft
    TSHARK_PATH = _ft()
except Exception:
    pass


# ---------------------------------------------------------------- torch-free model loader ------------------------------------
def test_model_loader_needs_no_torch_and_has_expected_shapes():
    from app.npmodels import load_state_dict
    sd = load_state_dict(ROOT / "models/ae_unsup_mixed.pt")
    assert sd["encoder.0.weight"].shape == (128, DET.dim) and sd["decoder.4.weight"].shape == (DET.dim, 128)
    assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in sd.values())


def test_model_loader_refuses_unexpected_pickle_content(tmp_path):
    """A tampered checkpoint must not be able to run code: only the names of a plain state dict are accepted."""
    import pickle
    import zipfile
    from app.npmodels import load_state_dict

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))
    bad = tmp_path / "bad.pt"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError):
        load_state_dict(bad)


def test_model_loader_matches_torch_when_torch_is_available():
    try:
        import torch
    except Exception:                                    # not installed, or blocked (Smart App Control, WinError 4551)
        pytest.skip("torch cannot be imported here")
    from app.npmodels import load_state_dict
    for f in ("ae_unsup_mixed.pt", "ae_supervised_recon.pt"):
        mine = load_state_dict(ROOT / "models" / f)
        ref = torch.load(ROOT / "models" / f, map_location="cpu", weights_only=True)
        assert list(mine) == list(ref)
        for k in ref:
            assert np.array_equal(mine[k], ref[k].numpy()), (f, k)
