# CICIDS2018: supervised vs unsupervised autoencoders - review cheat sheet

Code: github.com/vinay991-code/cicids2018-autoencoder-ids   |   folder: flood-review/cicids2018-ae   |   paper draft: reports/DRAFT_REPORT.docx

## 30-second pitch

Two guards for network traffic, **same autoencoder brain**, same 1 million training flows, same 2.16 million test flows. One learns from labels (supervised), one from unlabelled data (unsupervised). **Supervised wins on known attacks, unsupervised wins on brand-new ones.** The unsupervised guard fails on known attacks because ~12% attacks hid in its training data and it learned them as normal. Automatic cleaning (trimmed loss, DAE-LR) did not fix that. After a change of benign traffic, an adaptive threshold lost detections, and retraining behind a supervised gate restored the ranking only for attacks the gate knew.

## How supervised vs unsupervised was done (recipe)

**1 Data.** CSE-CIC-IDS2018, 10 files, cleaned and de-duplicated to 10,775,819 flows (12% attacks). 72 inputs (68 numbers + protocol); log scaling fitted on training only; destination port dropped (it is a shortcut).

**2 Split.** 60-second blocks, 70/10/20 (train 7.44M, val 1.17M, test 2.16M). No block and no identical flow in two parts.

**3 Same brain.** Encoder 72-128-64-32, Adam 0.001, batch 1,024, early stopping 5, seed 42, same 1M training rows for all four models.

**4 Unsupervised AE.** Decoder rebuilds the flow; loss = rebuild error; labels never read; score = rebuild error; alarm above the 95th percentile of validation errors.

**5 One-class AE (control).** Same, but clean rows only; alarm at the 99th percentile of clean validation errors.

**6 Supervised AE.** Same encoder + head 32-16-1; cross-entropy with attacks weighted x7.14; alarm at probability 0.5. A variant adds the decoder.

**7 Testing.** Precision, recall, F1, false-alarm rate, ROC-AUC; 95% block-bootstrap intervals; test labels read only after predictions.

**8 Repeats.** Laptop and Colab runs, 4 more seeds; unseen test = 4 attack types removed from scaler, training, early stopping and thresholds.

**9 Adapting (Phases 16, 17).** Frozen AE with a label-free, rate-limited threshold (lower quartile of the last 6 intervals); or retrain only on records that BOTH the AE and a supervised model call normal. Fine-tuning written in NumPy (torch is blocked here) and checked against saved PyTorch outputs.

Scripts: `src/phase3_unsupervised.py, phase4_supervised.py, phase5_compare.py, phase6_unseen.py, phase16_adaptive_threshold.py, phase17_drift_recovery.py, nptrain.py`

## Demo order (5 minutes)

`python run_app.py` then http://127.0.0.1:8000 (or `start_app.bat`). 1 Results tab (Table 1). 2 Wireshark / PCAP > *Try the demo capture*. 3 Live replay > start. 4 Drift and adaptation (shows Phases 7 to 9 only). 5 Models tab. Live capture needs Npcap: `start_app_capture.bat`.

## Do not claim

Detection of real attacks on live networks (never measured) - Docker works (never built) - Colab notebooks were run by me (not run) - more than one dataset - busy-network capture (tested briefly on Wi-Fi only). Unseen test = 4 attack types, one episode each; seeds are few, so read threshold-based numbers as ranges. Never say drift recovery works in general: it needs a gate that knows the attacks, and F1 did not change. The adaptive threshold did not help.

## Numbers to remember (Run A / Run B)

| What | Value |
|---|---|
| Known attacks, ROC-AUC (2 runs) | Sup 0.944 / 0.966   Unsup 0.468 / 0.528 |
| Known attacks, F1 | Sup 0.912 / 0.923   Unsup 0.109 / 0.101 |
| Unseen types, ROC-AUC | Sup 0.648 / 0.615   Unsup 0.927 / 0.904 |
| Unseen types flagged | Sup 0.8% / 13.6%   Unsup 74.9% / 52.8% (7.7% benign flagged) |
| Clean training of unsup | ROC-AUC 0.468 -> 0.889 (Run B 0.528 -> 0.879) |
| OR of both guards | 87.9% known, 75.1% unseen caught; 7.9% false alarms |
| Trimmed loss / DAE-LR | 0.470-0.498 / 0.482-0.496; clean training 0.902 |
| Phases 8-9 adaptive / drift | Nothing beat the static guard (gate vs free updates F1 +0.059 [-0.013, +0.131]) |
| Adaptive threshold (frozen AE) | FPR 0.0122 -> 0.0006, recall 0.0136 -> 0.0031, F1 0.0247 -> 0.0061; 64 of 194 intervals no alarm |
| Drift recovery, within-day ROC-AUC | Gate = Phase 4 model: 02-16 0.079 -> 0.741, 02-21 0.013 -> 0.751 (0.742 without gate); keeps 02-20 0.947, 03-02 0.935 (ungated 0.430, 0.581). Recovery 4 intervals on 02-16 |
| Gate that never saw Hulk, HOIC, Bot | 02-16 stays 0.466, 03-02 falls to 0.556; it blocks those types 0.3%, 0.1%, 1.9% |
| Drift recovery, overall F1 | 0.0247 -> 0.0251, intervals include 0 (threshold stays near 0.0139) |
| Ordinary Wi-Fi, 1,149 flows | Unsup flags 69%, one-class 4.4%, supervised 2.3% |

## Proofs and where they are

| Proof | Where |
|---|---|
| 138 / 48 / 176 / 12 checks | src/verify_results.py, verify_trimmed.py, verify_daelr.py, aggregate_labelled.py; reports/verification.md |
| 51 tests pass, 1 skipped (needs torch) | python -m pytest tests -q (one PCAP test can time out when the machine is busy; it passes alone) |
| NumPy trainer = PyTorch | tests/test_nptrain.py (gradients, Adam, forward vs saved scores 7e-7); checks C1-C7 in results_drift/phase17_run.log |
| Phase 16 and 17 numbers | reports/phase16_findings.md, phase17_findings.md; report Tables 11, 12, 12b |
| Flow code = CICFlowMeter | reports/FLOW_VALIDATION.md: 4,701 flows equal on 69 features; 98,552 packets match Wireshark |
| Dashboard = saved numbers | tests/test_app.py (also: engine matches saved scores to 2e-6) |
| What is new, honestly | reports/NOVELTY_AUDIT.md (section 2) |

## Likely questions

- **Why not accuracy?** 88% of test flows are benign: saying 'no attack' scores 88%.
- **Leak-free?** Block split, train-only scaling, thresholds fixed before testing, de-duplication, port dropped.
- **Why is unsupervised bad on known attacks?** Its training data held 12% attacks, learned as normal; clean data gives 0.889.
- **What is new?** Loss-based cleaning fails: highest-loss rows are mostly benign (trimmed loss removed only 5.6/7.4/12.3% of attacks at 10/20/30% trim); DAE-LR also gives no gain, on ROC-AUC and on labelled-threshold F1. A candidate, partly known elsewhere.
- **Does drift recovery work?** Ranking recovers (0.079 -> 0.741 on 02-16) only with a gate that knows the attacks; the gate is optimistic (trained on all days); with a gate that never saw them it fails; F1 unchanged.
- **Why did the adaptive threshold fail?** It assumes the benign score shape is fixed: on 02-16 the bulk rose 253x but the tail 1.9x, so the threshold overshot to 6.39.
- **Real-time?** Live capture ran on Wi-Fi briefly; attack detection on live networks was not measured.
