# Phase 9 - Experiment F: concept-drift monitoring (measured results)

Code: `src/drift.py`, `src/adaptive.py` (extended), `src/phase9_drift.py`, `src/phase9_tables.py`. Full tables (Table 4): `reports/phase9_tables.md`; data: `results/phase9_main.json` (+ `_intervals.csv`); figure: `results/phase9_drift_overview.png`. One run, seed 42, settings fixed before the comparison. No labels control the detector or any update; labels are used only to count results and to build the evaluation ground truth.

## Setup
- **Monitor** (label-free): 8 ADWIN detectors (river 0.26.1; delta 0.002 split over the detectors) on batches of 5,000 records scored by a fixed sentinel model (the warm-up autoencoder, never updated, so adaptation cannot feed back into detection): mean log10 reconstruction error and flagged share (error distribution), the mean of five traffic features (flow duration, forward packets, forward bytes, flow bytes/s, packet length mean) and the UDP share. Detectors were pre-fed with the last 300,000 warm-up records, so the start of the stream is a change from that reference. An alarm is raised when any detector fires.
- **Arms** (same warm-up model, stream and anchor-based threshold as Phase 8): **A** no detection, no updates; **A2** reference without detection: unrestricted retraining every 3 intervals; **B** drift detection + adaptive baseline: after an alarm, retrain for 3 rounds on the last 3 intervals' samples (25% clean replay); **C** drift detection + confidence-aware updates: an alarm is acted on only if the deviation persists one more interval, and the retraining data exclude the highest-scoring 20% of the recent samples.
- **Ground truth for evaluation only**: a day is "shifted" if the static arm flags more than half of its benign records. True benign change points: 02-16 start (to shifted), 02-20 start (back), 02-21 start (shifted), 02-22 start (back). All four are jumps between days; two of those days (02-16, 02-21) are single-hour snapshots of 12 intervals.

## Table 4 - drift detection and its effect on recovery
| | A no detection | A2 periodic (ref.) | B drift-triggered | C drift + confirmation |
|---|---:|---:|---:|---:|
| False-positive rate, overall | 0.1602 | 0.0666 | 0.0694 | 0.0922 |
| Recall, overall | 0.7035 | 0.1410 | 0.3071 | 0.3830 |
| F1, overall | 0.4925 | 0.1741 | 0.3399 | 0.3742 |
| Retrainings | 0 | 64 | 45 | 15 |
| Mean attack share in the retraining data (stream: 0.122) | - | 0.123 | 0.150 | 0.321 |
| Recovery after the 02-16 / 02-21 shift (intervals to FPR <= 10%) | never / never | 5 / 7 | 6 / 6 | 7 / 7 |

The overall figures for A are inflated by flagging nearly every record on 02-16 and 02-21 (false-positive rate 99.6% and 99.9% there); compare false-positive rates and the per-day table.

Drift detection (shared by B and C): all **4 of 4** change points detected; first alarm after 190,000, 125,000, 55,000 and 110,000 records (38, 25, 11, 22 monitoring batches; median 117,500 records, about 2.4 update intervals). **False alarms: 10 of 20 alarms (1.05 per million records)**: 8 on 02-20 (3 of them in batches that are more than half attacks, the LOIC onset; 5 in benign-dominated batches), 2 on 02-23. Five further alarms inside the 02-21 window repeat the first detection.

## Findings
1. **The detector finds the large shifts but not quickly and not selectively.** Delays of 55,000-190,000 records follow from ADWIN's conservative bound, even though benign errors changed by two orders of magnitude. About half the alarms are outside the four known changes. Three coincide with the start of the LOIC attack: a distribution detector cannot separate an attack onset from a benign change.
2. **Drift-triggered retraining (B) beat scheduled retraining (A2) at the same false-positive rate** (B - A2, moving-block bootstrap over intervals: F1 +0.172 [+0.016, +0.345], recall +0.171 [+0.014, +0.400], false-positive rate +0.001 [-0.014, +0.012]). The advantage comes mostly from timing: B retrained only on 02-16, 02-20, 02-21, 02-22 and 02-23, mostly on attack-free days (attack share 0.000-0.001 on 02-22 and 02-23), and never during the Bot campaign on 03-02, so on that day it flags 91% of Bot (A2 and static: 5% and 4%) and its day-level ROC-AUC is 0.945 versus 0.540 for A2. This shows the benefit of retraining only on clean periods; it is not evidence that a drift occurred on 03-02, and I did not isolate the model-level mechanism behind the 91% figure.
3. **Recovery from the benign shifts is about as fast as with periodic retraining**: 6 and 6 intervals (B) vs 5 and 7 (A2); the static arm never recovers within the day. Detection delay is included in these numbers.
4. **Confirmation plus a relative margin (C) did not reduce contamination; it increased it.** Mean attack share in C's retraining data is 0.321 (0.57 on 02-21, maximum 1.000) against 0.150 for B. Dropping the highest-scoring 20% removes unusual-but-benign flows and leaves attacks that reconstruct well. C acted on 5 of 18 alarms (13 cancelled), and its false-positive rate is higher than B's (+0.022 [+0.012, +0.036]) with a recall gain (+0.079 [+0.013, +0.151]); the F1 difference is not significant (+0.036 [-0.016, +0.087]).
5. **On ordinary days no adaptation is needed and it costs detection.** On 02-20 the static arm has F1 0.743 (LOIC-HTTP flagged 94%); B and C reach 0.205 and 0.233 (LOIC 17% and 21%) because alarms at the attack onset triggered retraining on attack-heavy data (B: up to 0.51 attack share; C: 0.40).
6. **F1 degradation and recovery could not be assessed meaningfully.** The F1 "before" a change is missing or is the artifact of flagging everything, so Table 4d reports it where available and the recovery claims rest on the false-positive rate.

## Limitations
- Only four change points, all day-level jumps; two days are single-hour snapshots, so recovery cannot exceed 12 intervals; gradual drift was not tested.
- The ground truth was built from the static model's behaviour and labels (evaluation only); within-day changes on 02-20, a 5.4M-record day, are not in it, so part of the "false alarms" may be real small shifts. The false-alarm rate is an upper bound under this definition.
- One seed and one run; ADWIN settings, monitor set, response length, persistence rule and the 20% exclusion were fixed in advance and not tuned; bootstrap intervals describe sampling variation across the stream only.
- The overall F1/recall of the static arm is inflated on two days; Feb 20 is 57% of the records and one attack campaign.
