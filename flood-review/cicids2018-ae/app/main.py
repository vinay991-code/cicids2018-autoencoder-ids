"""Backend (FastAPI) for the CICIDS2018 detector dashboard.

Run:  python run_app.py            (serves http://127.0.0.1:8000 )
Endpoints
  GET  /                      dashboard page
  GET  /api/health            liveness and what is available
  GET  /api/models            models, thresholds and the rule that set each threshold
  GET  /api/results           measured results read from the experiment files (nothing typed in by hand)
  POST /api/score             score flow records (JSON rows with the 68 feature names + protocol)
  GET  /api/replay/days       days of the held-out test partition
  GET  /api/replay/stream     server-sent events: the test partition replayed in time order through the saved models
  GET  /api/capture/status|start|stop|stream   live capture (needs ENABLE_CAPTURE=1 and Npcap; flows follow CICFlowMeter's rules, see /api/flow-validation)
  GET  /api/stream-experiments   saved results of the adaptive-baseline and concept-drift experiments (read-only; the served models are never adapted)
  GET  /api/flow-validation   how the flow builder was checked against CICFlowMeter, the training data and Wireshark, and false alarms on ordinary traffic
Security defaults: binds to 127.0.0.1; if APP_API_KEY is set, POST and capture endpoints need the header X-API-Key.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import replay
from .service import LABELS, ROOT, Detector

app = FastAPI(title="CICIDS2018 autoencoder detector", version="1.0")
STATIC = Path(__file__).parent / "static"
_det: Detector | None = None
_cap = None
FIGURES = ["phase5_roc_pr_comparison.png", "phase5_confusion_unsup_vs_sup.png", "phase3_score_distributions.png",
           "phase6_unseen_detection.png", "phase7_adaptive_overview.png", "phase8_confidence_overview.png", "phase9_drift_overview.png"]
MAX_ROWS = 5000


def det() -> Detector:
    global _det
    if _det is None:
        _det = Detector()
    return _det


def need_key(x_api_key: str | None = Header(default=None)):
    key = os.environ.get("APP_API_KEY")
    if key and x_api_key != key:
        raise HTTPException(401, "missing or wrong X-API-Key")


def capture_enabled() -> bool:
    return os.environ.get("ENABLE_CAPTURE") == "1"


class ScoreRequest(BaseModel):
    rows: list[dict] = Field(..., min_length=1, description="Flow records: the 68 feature names plus 'protocol', raw (unscaled) values.")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": _det is not None, "replay_data_available": replay.available(),
            "capture_enabled": capture_enabled(), "api_key_required": bool(os.environ.get("APP_API_KEY"))}


@app.get("/api/models")
def models():
    return det().info()


@app.post("/api/score", dependencies=[Depends(need_key)])
def score(req: ScoreRequest):
    if len(req.rows) > MAX_ROWS:
        raise HTTPException(413, f"at most {MAX_ROWS} rows per request")
    d = det()
    try:
        raw, proto = d.rows_to_arrays(req.rows)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))
    out = d.score(d.transform(raw, proto))
    return {"n": len(req.rows),
            "results": [{"scores": {m: float(out["scores"][m][i]) for m in out["scores"]},
                         "alerts": [m for m, a in out["alerts"].items() if a[i]]} for i in range(len(req.rows))],
            "note": "Scores are not calibrated probabilities for the reconstruction models; thresholds are the fixed rules of /api/models."}


def _f(x):
    return None if x is None else float(x)


@app.get("/api/results")
def results():
    """Everything shown on the Results tab, read from the experiment output files."""
    r = ROOT / "results"
    try:
        p3 = json.loads((r / "phase3_results.json").read_text())["variants"]
        p4 = json.loads((r / "phase4_results.json").read_text())["variants"]
    except FileNotFoundError as e:
        raise HTTPException(404, f"result file not found: {e.filename}")
    t1 = {}
    for name, blk in (("unsup_mixed", p3["unsup_mixed"]), ("normal_only", p3["normal_only"])):
        t1[name] = {**{k: _f(blk["primary"][k]) for k in ("accuracy", "precision", "recall", "f1", "fpr")}, **{k: _f(v) for k, v in blk["ranking"].items()}}
    for name in ("supervised", "supervised_recon"):
        blk = p4[name]
        t1[name] = {**{k: _f(blk["threshold_0.5"][k]) for k in ("accuracy", "precision", "recall", "f1", "fpr")}, **{k: _f(v) for k, v in blk["ranking"].items()}}
    seeds_f = ROOT / "seeds/seeds_summary.json"
    seeds = json.loads(seeds_f.read_text()) if seeds_f.exists() else None
    trim_f = ROOT / "results_trim/trimmed_summary.json"
    return {"labels": LABELS, "table1_run_a": t1,
            "seeds": ({"seeds": seeds["seeds"], "metrics": seeds["metrics"]} if seeds else None),
            "trimmed": json.loads(trim_f.read_text()) if trim_f.exists() else None,
            "figures": [f for f in FIGURES if (r / f).exists()]}


@app.get("/figures/{name}")
def figure(name: str):
    if name not in FIGURES or not (ROOT / "results" / name).exists():
        raise HTTPException(404)
    return FileResponse(ROOT / "results" / name, media_type="image/png")


@app.get("/api/replay/days")
async def replay_days():
    if not replay.available():
        raise HTTPException(503, "data/processed/X_test.npy and meta_test.pkl are needed for the replay (not included in the Docker image; mount them)")
    return await asyncio.to_thread(replay.days)


def _sse(obj, event=None):
    return (f"event: {event}\n" if event else "") + "data: " + json.dumps(obj) + "\n\n"


@app.get("/api/replay/stream")
async def replay_stream(start: int = Query(0, ge=0), batch: int = Query(500, ge=10, le=5000), interval: float = Query(0.5, ge=0.05, le=5.0),
                        ticks: int = Query(200, ge=1, le=5000)):
    if not replay.available():
        raise HTTPException(503, "replay data not available")
    d = det()
    gen = replay.ticks(d, start, batch, ticks)

    async def body():
        yield _sse({"batch": batch, "interval": interval, "note": "held-out test records replayed in time order; labels shown for evaluation only"}, "hello")
        while True:
            item = await asyncio.to_thread(next, gen, None)
            if item is None:
                yield _sse({}, "end")
                return
            yield _sse(item)
            await asyncio.sleep(interval)
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ------------------------------------------------------------------ Wireshark / PCAP analysis -------------------------------
import shutil
import tempfile

from fastapi import File, UploadFile

from .pcapjob import Jobs, combined_filter, to_csv

JOBS = Jobs()
MAX_UPLOAD = int(os.environ.get("MAX_PCAP_MB", "200")) * 1024 * 1024


def find_tshark() -> str | None:
    p = shutil.which("tshark")
    if p:
        return p
    for d in (os.environ.get("WIRESHARK_DIR"), r"C:\Program Files\Wireshark", r"C:\Program Files (x86)\Wireshark", "/usr/bin", "/usr/local/bin"):
        if d and (Path(d) / ("tshark.exe" if os.name == "nt" else "tshark")).exists():
            return str(Path(d) / ("tshark.exe" if os.name == "nt" else "tshark"))
    return None


def flow_validation():
    """Numbers behind the claim that the flow builder follows CICFlowMeter: read from the result files written by tools/run_flow_validation.py,
    tools/validate_flow_definitions.py and tools/ordinary_traffic_check.py (never typed here), plus the held-out false-alarm rates of the models."""
    def load(name):
        f = ROOT / "results" / name
        return json.loads(f.read_text()) if f.exists() else None
    parity, defs, ordinary = load("flow_validation_parity.json"), load("flow_validation_definitions.json"), load("ordinary_traffic_check.json")
    if not (parity and ordinary):
        return None
    out = {"parity": parity["total"], "reference": parity["reference"]["tool"], "ordinary": {"flows": ordinary["total"]["flows"], "flagged_share": ordinary["total"]["flagged_share"],
           "share_with_out_of_range_feature": ordinary["total"]["share_with_out_of_range_feature"]}}
    if defs:
        tr = defs.get("training_data")
        pk = next((v for k, v in defs.items() if k.startswith("port_on_")), None)
        if tr and pk:
            out["definitions"] = {"properties": len(tr["properties"]), "training_rows": tr["rows"], "training_lowest_share": min(tr["properties"].values()),
                                  "port_flows": pk["rows"], "port_lowest_share": min(pk["properties"].values())}
    try:
        p3 = json.loads((ROOT / "results/phase3_results.json").read_text())["variants"]
        p4 = json.loads((ROOT / "results/phase4_results.json").read_text())["variants"]
        out["held_out_false_alarm_rate"] = {"unsup_mixed": p3["unsup_mixed"]["primary"]["fpr"], "normal_only": p3["normal_only"]["primary"]["fpr"],
                                            "supervised": p4["supervised"]["threshold_0.5"]["fpr"], "supervised_recon": p4["supervised_recon"]["threshold_0.5"]["fpr"]}
    except (OSError, KeyError, ValueError):
        pass
    return out


@app.get("/api/stream-experiments")
def stream_experiments():
    """Saved results of the Phase 7 (adaptive baseline) and Phase 9 (concept drift) experiments, for the tab 'Drift and adaptation'. Read-only."""
    from .streamexp import load
    return {"data": load()}


@app.get("/api/flow-validation")
def flow_validation_endpoint():
    return {"validation": flow_validation()}


@app.get("/api/wireshark/info")
def wireshark_info():
    t = find_tshark()
    ref = ROOT / "results/ood_reference.json"
    return {"tshark_found": bool(t), "tshark_path": t,
            "live_pipe_command": (f'"{t}" -i <interface number or name> -F pcap -w - | python tools/pcap_score.py -' if t else
                                  "tshark -i <interface> -F pcap -w - | python tools/pcap_score.py -"),
            "file_command": "python tools/pcap_score.py capture.pcapng --csv flows.csv",
            "max_upload_mb": MAX_UPLOAD // (1024 * 1024),
            "out_of_range_reference": json.loads(ref.read_text()) if ref.exists() else None,
            "validation": flow_validation(),
            "warning": "A flag is a lead to inspect in Wireshark, not a verdict: detection of real attacks on other networks has not been measured."}


@app.post("/api/pcap", dependencies=[Depends(need_key)])
async def pcap_upload(file: UploadFile = File(...)):
    d = det()
    fd, path = tempfile.mkstemp(suffix=".pcap")
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, f"file larger than {MAX_UPLOAD // (1024 * 1024)} MB; use tools/pcap_score.py for big captures")
                out.write(chunk)
    except HTTPException:
        os.unlink(path)
        raise
    head = Path(path).read_bytes()[:4]
    if head not in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d", b"\x0a\x0d\x0d\x0a"):
        os.unlink(path)
        raise HTTPException(415, "not a pcap or pcapng file (save the capture from Wireshark as pcap or pcapng)")
    jid = JOBS.start(d, path, file.filename or "upload", cleanup=lambda: os.path.exists(path) and os.unlink(path))
    return {"job": jid, "bytes": size}


@app.post("/api/pcap-demo", dependencies=[Depends(need_key)])
def pcap_demo():
    """Analyse the bundled synthetic capture (samples/demo_synthetic.pcap). The file is kept."""
    p = ROOT / "samples/demo_synthetic.pcap"
    if not p.exists():
        raise HTTPException(404, "samples/demo_synthetic.pcap not found; run: python tools/make_demo_pcap.py")
    return {"job": JOBS.start(det(), str(p), "demo_synthetic.pcap (made-up traffic)", cleanup=lambda: None), "bytes": p.stat().st_size}


@app.get("/api/pcap/{jid}", dependencies=[Depends(need_key)])
def pcap_status(jid: str, limit: int = Query(200, ge=1, le=2000), flagged_only: bool = False):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    flows = j["flows"]
    if flagged_only:
        flows = [f for f in flows if f["alerts"]]
    flagged = [f for f in j["flows"] if f["alerts"]]
    return {"job": jid, "filename": j["filename"], "status": j["status"], "packets_read": j["packets"], "error": j["error"], "summary": j["summary"],
            "flows": flows[:limit], "shown": min(limit, len(flows)),
            "filter_all_flagged": combined_filter(flagged) if flagged else None, "filter_note": "first 100 flagged flows" if len(flagged) > 100 else None}


@app.get("/api/pcap/{jid}/csv", dependencies=[Depends(need_key)])
def pcap_csv(jid: str):
    from fastapi.responses import Response
    j = JOBS.get(jid)
    if not j or j["status"] != "done":
        raise HTTPException(404, "job not finished")
    return Response(to_csv(j["flows"]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="flows_{jid}.csv"'})


@app.delete("/api/pcap/{jid}", dependencies=[Depends(need_key)])
def pcap_delete(jid: str):
    return {"deleted": JOBS.delete(jid)}


# ------------------------------------------------------------------ live capture -------------------------------------------
def _cap_status():
    base = {"enabled": capture_enabled(), "running": False, "packets": 0, "flows": 0, "flagged_flows": 0, "error": None, "save_path": None,
            "warning": "A flag is a lead to inspect in Wireshark, not a verdict: detection of real attacks on other networks has not been measured."}
    if _cap:
        base.update(_cap.status())
    return base


@app.get("/api/capture/interfaces", dependencies=[Depends(need_key)])
def capture_interfaces():
    if not capture_enabled():
        raise HTTPException(403, "live capture is disabled; start the server with ENABLE_CAPTURE=1")
    try:
        from .livecapture import list_interfaces
        return {"interfaces": list_interfaces()}
    except Exception as e:
        raise HTTPException(500, f"could not list interfaces (is Npcap installed?): {type(e).__name__}: {e}")


@app.get("/api/capture/status")
def capture_status():
    return _cap_status()


@app.post("/api/capture/start", dependencies=[Depends(need_key)])
def capture_start(iface: str | None = None, save: bool = True):
    global _cap
    if not capture_enabled():
        raise HTTPException(403, "live capture is disabled; start the server with ENABLE_CAPTURE=1 (Npcap must be installed)")
    from .livecapture import LiveCapture, list_interfaces
    if _cap and _cap.running:
        raise HTTPException(409, "a capture is already running; stop it first")
    if iface:
        try:
            names = {i["name"] for i in list_interfaces()}
        except Exception as e:
            raise HTTPException(500, f"could not list interfaces: {e}")
        if iface not in names:
            raise HTTPException(422, "unknown interface; choose one from /api/capture/interfaces")
    _cap = LiveCapture(det(), iface or None, save=save)
    _cap.start()
    return {"started": True, "iface": iface}


@app.post("/api/capture/stop", dependencies=[Depends(need_key)])
def capture_stop():
    if _cap:
        _cap.stop()
    return {"stopped": True}


@app.get("/api/capture/stream", dependencies=[Depends(need_key)])
async def capture_stream():
    if not capture_enabled():
        raise HTTPException(403, "live capture is disabled")
    if _cap is None:
        raise HTTPException(409, "start a capture first")
    c = _cap

    async def body():
        yield _sse({"note": "live flows; addresses stay on this computer"}, "hello")
        while True:
            flows = []
            while c.new and len(flows) < 100:
                flows.append(c.new.popleft())
            yield _sse({"flows": flows, "status": c.status()})
            if not c.running and not c.new:
                yield _sse(c.status(), "end")
                return
            await asyncio.sleep(1.0)
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
