# Phase 8 - Experiment E: confidence-aware update vs unrestricted adaptation (measured results)

Code: `src/adaptive.py` (extended), `src/phase8_confidence.py`, `src/phase8_tables.py`. Full tables (Table 3): `reports/phase8_tables.md`; data: `results/phase8_main.json` (+ `_intervals.csv`, `persist6`, `persist12`); figure: `results/phase8_confidence_overview.png`. One run, seed 42; all settings fixed before the comparison.

## Setup
Same warm-up model and 9,504,036-record evaluation stream as Phase 7 (1,158,860 attacks, 8 days; update interval 50,000 records; retrain every 3 intervals on the last 12 intervals' buffer with 25% clean replay). Each arm's threshold is the 95th percentile of the errors of a fixed clean anchor set under the arm's *current* model, recomputed after every update (this replaces Phase 7's unstable median/MAD rule and gives a usable operating point: the static detector flags about 6% of benign traffic and 94% of the LOIC attack on normal days).
Arms: **S** static reference; **U** unrestricted adaptation (every sampled record enters the retraining buffer); **M** margin only (admit records with score <= 0.5 x threshold); **G** confidence-gated = margin 0.5 + quarantine of records above the threshold + temporal consistency (an interval whose median log10 score moves more than 0.5 decades from the running reference, or with more than 15% flagged records, is held; if that persists for 3 consecutive intervals it is accepted as a lasting change, released into the buffer, and retraining is triggered). Reconstruction error is used only as an ordinal anomaly score, not as a calibrated probability.

## Table 3 - U (unrestricted) vs G (confidence-gated), with M and S for reference
| Metric | S static | U unrestricted | M margin only | G gated |
|---|---:|---:|---:|---:|
| False-positive rate, overall | 0.1602 | 0.0665 | 0.1541 | 0.0835 |
| Recall, overall | 0.7035 | 0.1416 | 0.7017 | 0.2100 |
| F1, overall | 0.4925 | 0.1748 | 0.4991 | 0.2319 |
| Mean attack share in the retraining buffer (stream: 0.122) | - | 0.123 | 0.038 | 0.128 |
| Maximum attack share in a buffer | - | 0.490 | 0.461 | 0.513 |
| Recovery after the 02-16 / 02-21 change (intervals to FPR <= 10%) | never / never | 5 / 7 | never / never | 5 / 9 |
| 02-20 (LOIC-HTTP day): F1 / recall / FPR | 0.743 / 0.941 / 0.070 | 0.083 / 0.061 / 0.050 | 0.754 / 0.941 / 0.066 | 0.156 / 0.126 / 0.058 |

**Read the overall S and M rows with care**: on 02-16 and 02-21 the benign traffic is shifted entirely above the static threshold, so S and M flag 90-94% of all records there (false-positive rate 99.6% and 99.9%). Flagging everything inflates their overall recall and F1, so they are not better detectors on those days. Per-day results are in `phase8_tables.md` (Table 3b).

## Findings
1. **The confidence-gated design gave no clear advantage over unrestricted adaptation.** G minus U (moving-block bootstrap over intervals): F1 +0.059 [-0.013, +0.131], recall +0.074 [-0.008, +0.169], false-positive rate +0.018 [+0.004, +0.044]. Contamination of the buffer was not reduced (mean attack share 12.8% vs 12.3%; maximum 51.3% vs 49.0%). G kept somewhat more of the HOIC detection (59% flagged vs 37%) and of LOIC (12.6% vs 6.1%) but recovered more slowly on 02-21 (9 vs 7 intervals; false-positive rate that day 42% vs 15%).
2. **Both adapting arms learned the attacks.** LOIC-HTTP detection fell from 94% (static) to 6% (U) and 13% (G); DoS-Hulk from 60% to 24%; day-level ranking quality (ROC-AUC) on 02-20 fell from 0.948 to 0.415 (U) and 0.431 (G). A one-hour campaign looks like a lasting change of traffic: the gate judged it inconsistent and then, after 3 intervals, released it (2 releases on 02-20).
3. **Margin-only (M) is the only arm that protected the buffer** (mean attack share 3.8%; 1.2% on 02-20) and it kept LOIC detection at 94% and its false-positive rate near 7% on 02-20. Its cost is that it cannot adapt to a real change: on 02-16 and 02-21 almost no benign records pass the margin, so its model never updates and its false-positive rate stays at 99.6% / 99.9% ("never" recovers).
4. **Recovery from a change and protection from contamination pull in opposite directions in this setup.** Persistence sensitivity for G (Table 3h): persistence 3 / 6 / 12 intervals gives LOIC-HTTP flagged 0.126 / 0.246 / 0.485, but 02-16 false-positive rate 0.246 / 0.486 / 0.996 (recovery 5 / 8 / never intervals) and 02-21 0.422 / 0.285 / 0.999. No setting both adapts quickly to the benign shift and keeps the campaign out of the buffer. A rule based on how long a deviation lasts cannot tell a lasting benign change from a long attack, because both look like sustained deviations (interpretation supported by this sensitivity and by the 2 releases on 02-20, not a general result).
5. **Attacks that reconstruct like normal traffic are not filtered by any score-based gate**: Bot (flagged 4-6% by all arms) and Infiltration (11-16%) pass the gates, and the buffers of all adapting arms contain 15-18% attacks on the Bot day (03-02), where ranking quality fell from 0.931 (static) to 0.54-0.66.

## Limitations
- One training run and one stream; the intervals reflect sampling variation across the stream only.
- 02-16 and 02-21 are single-hour snapshots of 12 intervals each (row-limit truncation), so recovery cannot exceed 12 intervals, and the "shift" is a jump between days, not gradual drift.
- Gate settings (margin 0.5, level tolerance 0.5 decades, flagged-share limit 0.15, persistence 3) were fixed in advance and not tuned; other values or other gate signals (for example a second model or feature-based checks) were not tested, so this does not show that no gating scheme can work.
- Overall F1/recall are dominated by 02-20 (57% of records, one campaign) and distorted for S and M by the flag-everything days.
- Contamination is measured from labels as a diagnostic; it counts attack records in the training buffer, not how much of each attack the model actually learned.
