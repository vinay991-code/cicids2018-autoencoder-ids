# Dashboard, backend and live monitor

Added after the experiments, at the request of the project owner (the original task excluded a dashboard, a backend, live monitoring and deployment files). The experiments, tables and report do not depend on any of this.

## What is in it

| Part | File | What it does |
|---|---|---|
| Backend (FastAPI) | `app/main.py` | Serves the dashboard and a JSON API: `/api/results`, `/api/models`, `/api/score`, `/api/replay/*`, `/api/capture/*`, interactive docs at `/docs` |
| Scoring service | `app/service.py` | Loads the saved models, the Phase 2 scaler and the fixed thresholds. Nothing is trained here |
| Replay monitor | `app/replay.py` | Replays the held-out test records in time order through the saved models (server-sent events) |
| Dashboard (frontend) | `app/static/index.html` | One page, no external libraries: Results, Live replay, Live capture, Models and thresholds |
| Live capture | `app/livecapture.py`, `app/capture.py` | Interface to flows to models, with an interface picker and a headers-only pcap for Wireshark. Off unless started with `--enable-capture` |
| Wireshark / PCAP analysis | `app/pcapjob.py`, `tools/pcap_score.py` | Capture file or tshark stream to flows to scores to Wireshark filters and CSV |
| Tests | `tests/test_app.py`, `tests/test_pcap.py` | 27 tests (11 of them use Wireshark's `tshark` as an independent check) |
| Deployment files | `start_app.bat`, `deploy/windows_autostart.ps1`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements-app.txt` | One-click start, optional start at logon, container build (not yet built, see below) |

## Wireshark integration

Wireshark and the dashboard exchange data through capture files, the Wireshark command-line tool and display filters. Nothing is installed inside Wireshark.

1. **Saved capture (works without Npcap).** In Wireshark use File > Save As (pcap or pcapng), then on the dashboard open **Wireshark / PCAP** and choose the file (up to 200 MB; set `MAX_PCAP_MB` to change), or run `python tools/pcap_score.py capture.pcapng --csv flows.csv` for any size. Packets are grouped into flows, scored by every model, and the flagged flows come first.
2. **Back into Wireshark.** Each flow has a ready-made display filter (addresses, ports and time window). Click **Copy filter** and paste it into Wireshark's filter bar to see exactly that flow's packets. There is also one filter for all flagged flows (first 100).
3. **Live, through tshark.** `tshark -i <interface> -F pcap -w - | python tools/pcap_score.py -` (or `python tools/pcap_score.py - --tshark-live <interface> --seconds 60`). Needs Npcap (Wireshark's installer offers it). The dashboard's own **Live capture** tab does the same without tshark, see below.
4. **CSV.** Every flow with its scores, alerts, out-of-range count and Wireshark filter.

`samples/demo_synthetic.pcap` (made-up traffic, button "Try the demo capture") lets you try the page. Its 56 flows are made up; they were used for the first comparison with CICFlowMeter's code (all 56 equal). All of them are flagged and have out-of-range features, which is the expected result for invented traffic.

**Checked here:** with Wireshark 4.6.8's own `tshark`, the display filter of every flow in the test capture selects exactly that flow's packets (fwd + bwd count), the number of IPv4 TCP packets agrees with tshark's, pcap and pcapng files are both read, and the stream mode reads a pcap from standard input.

**Flow features, checked three ways** (full report: `reports/FLOW_VALIDATION.md`; one command, `python tools/run_flow_validation.py`):
1. *Against CICFlowMeter's own flow code.* The flow logic of CICFlowMeter (the unmodified February 2018 source) was run on the same packets as this project's flow builder. On 7 captures (102,551 packets, 4,701 flows, 1,149 of them real Wi-Fi traffic) every flow has the same 69 feature values.
2. *Against the training data.* 26 row-level properties that follow from the original tool's definitions (the first packet counted twice in the length statistics, flag counts taken from the first packet in a shuffled column order, the transport header as header length, and so on) hold for 1.5 million training rows (every one at 0.999 or better) and for every flow built here (all 1.0).
3. *Against Wireshark.* The parsed packet fields equal tshark's on 98,552 IPv4 packets.

**Not shown:** that flags are real attacks. The models were trained on one 2018 lab network; on 1,149 flows of ordinary web, DNS and ping traffic the supervised model flagged 2.3% (0.35% on lab test flows), the one-class model 4.4%, but the unsupervised model flagged 69% (it was already near chance in the lab). No attack traffic was captured on a live network. The 2018 tool used the jnetpcap library to read packets and this project uses scapy; the packet fields agree with Wireshark's, and the training data agrees with the byte counting used, but jnetpcap itself was not run. Addresses are kept in memory for the analyst and are not sent anywhere; the uploaded file is deleted once read.

## Run it on Windows

```bash
pip install -r requirements-app.txt
python run_app.py
```

Or double-click `start_app.bat`. To start it at every Windows logon, run `powershell -ExecutionPolicy Bypass -File deploy\windows_autostart.ps1` (a per-user scheduled task; `-Remove` undoes it; it has not been run). Then open http://127.0.0.1:8000. The server listens on this computer only. The Live replay tab needs `data/processed/X_test.npy` and `meta_test.pkl` (created by Phase 2).

Run the tests: `python -m pytest tests -q`.

## PyTorch is not needed (and a Windows error it can cause)

Earlier versions imported PyTorch and stopped with `OSError: [WinError 4551] An Application Control policy has blocked this file ... torch\lib\shm.dll` on a laptop where Windows Smart App Control (state `VerifiedAndReputablePolicyState` = 1) blocked PyTorch's unsigned DLLs. The dashboard no longer imports torch, so that error cannot occur here; it can still occur when re-training (`src/`), see `TEAMMATE_GUIDE.md`. Do not turn Smart App Control off for this project: Windows cannot switch it back on without a reset.

## If the port is already in use

`python run_app.py` checks port 8000 first. If the dashboard is already running (another window, `start_app.bat`, or the autostart task) it says so and, with `--open`, opens it. If a different program uses the port it starts on the next free port and prints the address. Earlier versions crashed with "error 10048 ... only one usage of each socket address" in that situation.

## What was checked

* The service reproduces the scores saved by the experiments (which were computed with PyTorch). Checked on 200,000 random held-out rows: the largest difference is 2e-6 and no alert changes for any of the four models. The service runs in numpy only (`app/npmodels.py` reads the saved `.pt` weights), so PyTorch is not needed to serve. A test in `tests/` compares the loader's weights with `torch.load` whenever torch can be imported. The service's preprocessing reproduces the Phase 2 matrix from the raw cleaned flow rows (largest difference 6e-8). The 3,000-row version of this check and the preprocessing check are automated tests.
* The Results tab reads the experiment files at load time; a test compares its values with those files.
* The replay's confusion counts add up to the batch size for every model and tick (tested).
* In a browser the Results and Live replay tabs were exercised (replay of 2018-02-16, 61 ticks).

## Rules that still hold

* No attack label controls a model, a threshold or an alert. The replay shows the true label next to each alert only so the alerts can be compared with the truth.
* The served models are static. The adaptation experiments of the report did not give a reliable gain, so none is served.
* Thresholds are the fixed rules of the experiments (`/api/models` lists each one). Reconstruction errors are not probabilities.

## Live capture from a network interface

1. Install Npcap (comes with Wireshark's installer). On the Npcap page you may tick "Restrict driver access to Administrators only"; then start the dashboard from an administrator terminal.
2. Start the dashboard with capture enabled: double-click `start_app_capture.bat`, or `python run_app.py --enable-capture` (same as `ENABLE_CAPTURE=1`). If the dashboard is already running without capture, stop it first; the setting is read at start-up.
3. Open the **Live capture** tab, choose the interface (Wi-Fi or Ethernet for real traffic; "Loopback" for a safe test), and click **Start capture**. Flows appear when a FIN packet ends them or when they are 2 minutes old, with their scores, an out-of-range count and a Wireshark filter.
4. While capturing, a headers-only pcap (first 96 bytes per packet, capped at 50 MB) is saved in `captures/`. Open it in Wireshark and paste a flow's filter. Untick "save headers-only pcap" to keep nothing on disk.

**Tested:** the dashboard's own live capture on the Wi-Fi adapter through the API calls that the tab makes (start, status, stop; about 70 s of ordinary browsing: 12,377 packets, 272 flows; the saved headers-only file, analysed offline, gives exactly the same 272 flows and all of them equal CICFlowMeter's code), two more real Wi-Fi captures made with Wireshark's `dumpcap` (see `reports/FLOW_VALIDATION.md`), and capture on the loopback interface through the dashboard and through the Python class, end to end: packets, flows, scores, the saved file opened by `tshark`, and the filter of every flow selecting exactly that flow's packets (an automated test). The saved headers-only file gives the same flows and byte counts as the full capture (also tested; byte counts come from the IP header length).

**Not tested:** clicking through the Live capture tab in a browser on the Wi-Fi adapter (the API calls were used), long captures, or busy networks (the packet handling is pure Python and will drop packets under heavy load; for heavy traffic capture with Wireshark and analyse the file).

**Cautions**
* **Flags are leads, not verdicts.** The flow features follow CICFlowMeter (see above), but the models' detection of real attacks on other networks was not measured. FIN ends a flow as in the original tool, so a normal TCP close produces a second short flow of 2 packets; those exist in the training data too (6% of its TCP flows start with a FIN).
* The models were trained on a 2018 lab network. The unsupervised model in particular flags most ordinary traffic. The "Out of range" column helps: ordinary test flows almost never have any, and only 0.4% of the ordinary Wi-Fi flows had one.
* Only IPv4 packets are scored (TCP, UDP, and other IP protocols as protocol 0, like the training data); IPv6 is ignored. Capture only on a network you own or are allowed to monitor. Addresses are shown in the page and stay on this computer.

## Security defaults

* Binds to 127.0.0.1. If you bind elsewhere, set `APP_API_KEY`; POST and capture endpoints then require the header `X-API-Key`.
* `/api/score` accepts at most 5,000 rows per request and rejects missing fields and non-numeric values.
* There is no login, user store or HTTPS. This is a project demo and not a hardened service.

## Docker

```bash
docker compose up --build
```

The `Dockerfile` and `docker-compose.yml` were written but **not built or run**: the Docker daemon was not running on the development machine. Expect to fix small problems on the first build. The image holds code, models, results and the scaler, not the large data files; the compose file mounts the two test files read-only to enable the replay. The container is published on 127.0.0.1 only.
