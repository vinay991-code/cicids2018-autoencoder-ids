# CICIDS2018: supervised vs unsupervised autoencoders (research experiments only)

Experiments, preprocessing, models and evaluation for "Comparative Evaluation of Supervised and Unsupervised Autoencoders for Network
Anomaly Detection Using CICIDS2018". No packet capture, dashboard or application is included.

**Contributor:** Vinay Kumar Kamisetty ([@vinay991-code](https://github.com/vinay991-code)).

**Start with `reports/FINAL_REPORT.md`** (Tables 1-5, figures, limitations). `reports/verification.md` lists 138 independent
cross-checks of the reported numbers. Per-phase details are in `reports/phase*_findings.md` and `reports/phase*_tables.md`.

**To re-run only the supervised vs unsupervised comparison (Tables 1 and 5):** see `TEAMMATE_GUIDE.md` and run `python run_sup_vs_unsup.py` (writes to `rerun/`, never overwrites `results/`).

**To get intervals over several training seeds (Tables 1 and 5):** `python run_seeds.py --seeds 1 2 3 4` (about 65 min per seed on Colab CPU; each seed writes to `seeds/seed_<n>/` and finished seeds are skipped on a re-run), then `python src/aggregate_seeds.py --seeds-dir seeds` for means, standard deviations and 95% intervals. The Colab notebook for this is `colab_seeds.ipynb`.

## Environment (Windows, Python 3.13 tested)
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```
(CPU only; on Windows, if the torch install stops with a long-path error, enable long paths or install into a short path.
On Google Colab use the same commands and set the paths below.) The full run used about 20 GB of disk (6.4 GB raw data plus about 13 GB of intermediate and processed arrays) and ran on a 15.6 GB RAM machine; the preprocessing streams the files in chunks.

## Data
Download the 10 "Processed Traffic Data for ML Algorithms" CSVs (6.41 GB) from the official bucket listed on
https://www.unb.ca/cic/datasets/ids-2018.html into `data/raw`. Expected byte sizes are in `data/expected_sizes.json`.

## Commands, in order (run from the project folder; times are from one Windows machine and vary)
| Phase | Command | Output | Time |
|---|---|---|---|
| 1 Dataset inspection | `python src/phase1_inspect.py --data-dir data/raw --out-dir reports` | `reports/phase1_inspection.*` | ~30 min |
| 2 Preprocess and split | `python src/phase2_preprocess.py --data-dir data/raw --out-dir data --report-dir reports` | `data/processed`, `reports/phase2_report.md` | ~8 min |
| 3 Unsupervised AE | `python src/phase3_unsupervised.py --out-dir results --model-dir models` | `results/phase3_*` | ~15 min |
| 4 Supervised AE | `python src/phase4_supervised.py --out-dir results --model-dir models` | `results/phase4_*` | ~15 min |
| 5 Comparison | `python src/phase5_compare.py --phase3-log <phase 3 console log> --phase4-log <phase 4 console log>` | `results/phase5_*`, `reports/phase5_tables.md` | ~2 min |
| 6 Unknown attacks | `python src/phase6_unseen.py` then `python src/phase6_tables.py` | `results/phase6_*`, `reports/phase6_tables.md` | ~25 min |
| 7 Adaptive baseline | `python src/stream.py` then `python src/phase7_adaptive.py --tag main` then `python src/phase7_tables.py` | `results/phase7_*` | ~15 min |
| 8 Confidence-aware | `python src/phase8_confidence.py --tag main` (plus `--tag persist6 --persist 6`, `--tag persist12 --persist 12`) then `python src/phase8_tables.py` | `results/phase8_*` | ~12 min per run |
| 9 Concept drift | `python src/phase9_drift.py --tag main` then `python src/phase9_tables.py` | `results/phase9_*` | ~8 min |
| 11 Hybrid (supervised OR unsupervised) | `python src/phase11_hybrid.py` (uses saved scores from Phases 3, 4 and 6; no torch needed) | `results/phase11_hybrid.json`, `reports/phase11_hybrid.md` | ~1 min |
| 12 Label-free thresholds (exploratory) | `python src/phase12_thresholds.py` (uses saved scores; no torch needed) | `results/phase12_thresholds.json`, `reports/phase12_thresholds.md`, `reports/phase12_findings.md` | ~3 min |
| 16 Adaptive threshold, frozen model | `python src/phase16_adaptive_threshold.py` then `python src/summarise_threshold.py` (numpy only, no torch; needs the Phase 7 baseline in `data/interim` and `data/processed/X_stream.npy`) | `results_threshold/`, `reports/phase16_findings.md` | ~5 min |
| 17 Drift recovery, supervised gate (numpy training) | `python -m pytest tests/test_nptrain.py` then `python src/phase17_drift_recovery.py --check-only` then `python src/phase17_drift_recovery.py` then `python src/summarise_drift.py` (numpy only, no torch; needs the Phase 7 baseline in `data/interim`, `data/processed/X_stream.npy` and the Phase 4 and 6 models) | `results_drift/`, `reports/phase17_findings.md` | ~30 min |
| 10 Verify and report | `python src/verify_results.py` then `python src/phase10_report.py` | `reports/verification.md`, `reports/FINAL_REPORT.md` | ~3 min |

Notes: the Phase 5 script needs the console logs of the Phase 3 and 4 runs because the loss histories were not written to a file during
training (the script checks them against the saved best-validation losses). All models use seed 42. A full re-run of Phases 3-6 on the original machine on 2026-11-11 (`python run_sup_vs_unsup.py`, 47.5 min) reproduced every saved test score bit for bit
and every Table 1 and Table 5 value; only the timing rows differed. On a different machine or thread count expect results close to, but not necessarily identical to, the reported ones. Phase 7 sensitivity runs: `--tag retrain1 --retrain-every 1` and `--tag retrain6 --retrain-every 6`.

## Trimmed-loss experiment (phase 13)

`python src/phase13_trimmed.py --seed 42 --out-dir results_trim --model-dir models_trim` (about 1.5 hours per seed on a laptop), then `python src/aggregate_trimmed.py --trim-dir results_trim --seeds 42 1 2` and `python src/verify_trimmed.py`. Result: no gain over the ordinary unsupervised model; see the report, Section 5.5.

## DAE-LR experiment (phase 14)

`python src/phase14_daelr.py --seed 42 --out-dir results_daelr --model-dir models_daelr` (8 models per seed, about 1 hour per seed; `resume_phase14.bat` restarts an interrupted run and skips finished models), then `python src/aggregate_daelr.py --dir results_daelr --seeds 42 1 2` and `python src/verify_daelr.py`. Result: no gain over the ordinary unsupervised model; see the report, Section 5.5 (Table 8b).

## Dashboard, backend and live monitor (added afterwards)

`python run_app.py` serves a results dashboard, a scoring API and a replay of the held-out test records through the saved models at http://127.0.0.1:8000. A packet-capture module (flows built with the rules of CICFlowMeter, the tool that made the training data) and Docker files are included. The flow builder was checked against CICFlowMeter's own flow code, the training data and Wireshark: see `reports/FLOW_VALIDATION.md`. Read `APP_README.md` first: it states what was tested and what was not (the Docker image was not built; detection of real attacks on live networks was not measured).

## Layout
`src/` code (`data.py`, `models.py`, `training.py`, `evaluation.py`, `plots.py` are shared; `adaptive.py`, `drift.py`, `stream.py`,
`stream_setup.py` are the streaming core), `data/` (raw, interim, processed; large), `results/` (JSON results, scores, figures),
`models/` (trained weights), `reports/` (all reports and tables).
