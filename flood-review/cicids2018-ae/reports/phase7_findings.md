# Phase 7 - Experiment D: adaptive baseline (measured results)

Code: `src/stream.py`, `src/stream_setup.py`, `src/adaptive.py`, `src/phase7_adaptive.py`, `src/phase7_tables.py`. Full tables: `reports/phase7_tables.md`; data: `results/phase7_main.json` (+ `_intervals.csv`, and `retrain1` / `retrain6` sensitivity runs); figure: `results/phase7_adaptive_overview.png`. One run, seed 42, protocol fixed before the comparison (settings below were not changed after seeing results).

## Setup
- **Stream**: all 10,775,819 de-duplicated records of the 10 capture days (train + val + test partitions merged), ordered by corrected time. The stored timestamps are a 12-hour clock without AM/PM (checked against the official attack schedule: Hulk scheduled 13:45 is stored as 01:45, etc.), so hours 1-7 were shifted by +12 h. Update intervals are consecutive 50,000-record chunks within a day (two days, 02-16 and 02-21, are almost single-hour snapshots because their files are truncated).
- **Warm-up (initial baseline)**: days 02-14 and 02-15. One-class autoencoder trained on 500,000 benign warm-up records (first 80% by time); threshold = 99th percentile of the later 225,288 benign warm-up records (0.01394). Labels are used only here, to select clean warm-up data. The evaluation stream is the remaining 9,504,036 records (1,158,860 attacks, 12.19%) on 8 days.
- **Arms**: **A** static model and threshold. **B** same model, threshold re-estimated every interval as median + k x MAD of the log10 scores of the last 6 intervals' sampled records (k = 4.29 calibrated so the starting threshold equals A's; no labels). **C** B's threshold rule plus fine-tuning every 3 intervals (3 epochs, lr 5e-4) on a rolling trusted buffer (the last 12 intervals, 10,000 sampled records per interval, only records the arm itself scored as normal) plus 25% replay of clean warm-up records; an update is rejected if the error on a clean anchor set exceeds 3x its initial value.
- Each interval is scored and flagged before any update uses it. Labels are used only for counting results and for the contamination diagnostic.

## Table 2 - static vs adaptive threshold vs adaptive baseline (all 8 evaluation days)

| Metric | A static | B adaptive threshold | C adaptive baseline |
|---|---:|---:|---:|
| Precision | 0.1347 | 0.0348 | 0.1002 |
| Recall | 0.0136 | 0.0041 | 0.0115 |
| F1-score | 0.0247 | 0.0074 | 0.0206 |
| False-positive rate | 0.0122 | 0.0159 | 0.0143 |

Differences to A (moving-block bootstrap over intervals, 95%): F1 B -0.0157 [-0.0399, -0.0011]; C -0.0031 [-0.0077, +0.0001]. Per-day results, per-type flag rates and the sensitivity table are in `phase7_tables.md`.

## Findings
1. **Adaptation did not improve detection overall.** The static detector is already weak (recall 1.4%) because its threshold sits at the high-error tail of benign traffic, as in Phase 3. B is significantly worse than A; C is statistically indistinguishable from A.
2. **The adaptive threshold rule is unstable.** Threshold values ranged from 0.0014 to 1,660; in 25.8% of intervals the threshold exceeded 0.072, the largest error seen in the Phase 3 test scores, so almost nothing could be flagged there, and on 02-21 it fell to 0.0028 and flagged 12% of records (FPR 18.3%, precision 2.5%). I did not diagnose the cause beyond observing that the spikes occur when the window's scores spread widely; the rule was not changed after seeing this.
3. **Retraining helps ranking where benign traffic shifted and hurts where large attacks dominate.** Within-day ROC-AUC of the static model is 0.079 on 02-16 and 0.013 on 02-21, below chance, because benign errors on those days are much higher than in the warm-up (benign median error on 02-16 is about 100 times higher in the earlier exploration); the retrained model raises these to 0.472 and 0.745. On 02-20 (LOIC-HTTP, 575,364 attacks) and 03-02 (Bot, 144,337) the static model ranks well (0.948, 0.931) and retraining lowers it to 0.409 and 0.556.
4. **The "trusted buffer" did not protect against contamination.** Attack records made up 12.3% of the buffers on average (the stream's base rate is 12.2%) and up to 49.5% at times (02-16 and 02-21). All 64 retrainings passed the anchor check (median error ratio 0.94, maximum 1.50 against a limit of 3). Attacks that reconstruct well score as normal, so filtering on the model's own score removes almost none of them. This matches the Phase 3 and 6 evidence that an autoencoder learns the attack traffic it is shown.
5. **Update frequency changes the details, not the conclusion** (arm C): retraining every 1 / 3 / 6 intervals gives overall F1 0.0270 / 0.0206 / 0.0165, attack share in the buffer 0.126 / 0.123 / 0.122, and 02-20 ROC-AUC 0.379 / 0.409 / 0.469. All are in the range of the static detector (F1 0.0247).

## Limitations
- One training run and one stream; the bootstrap intervals only describe sampling variation across the stream.
- The score is dominated by 02-20 (57% of the records, one attack episode); 02-22 and 02-23 have only 319 and 535 attacks, so their recall/F1 are not meaningful.
- The "drift" present is mainly a jump between days (and two single-hour snapshot days), not a gradual change; conclusions about slow drift cannot be drawn.
- Fixed design choices (threshold rule, buffer size, replay share, anchor tolerance) were not tuned; a different rule could behave differently.
- The scaler was fitted on the Phase 2 training partition, which includes records from all days (only min/max statistics, no model weights, saw later days).
- Interval-based (record count) updates, not wall-clock time.
