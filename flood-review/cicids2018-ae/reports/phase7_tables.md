Warm-up (initial baseline): days 2018-02-14, 2018-02-15; one-class AE trained on 500,000 benign records, threshold = 99th percentile of 225,288 later warm-up benign records (0.01394). Evaluation stream: 9,504,036 records on 8 days (1,158,860 attacks, 12.19%). Update interval 50,000 records; threshold window 6 intervals; retrain every 3 intervals on the last 12 intervals' trusted samples (10,000 sampled per interval) with 25% clean replay, 3 epochs at lr 0.0005; anchor tolerance 3.0x. One run, seed 42.

## Table 2a - overall, all evaluation days

| Metric | A static | B adaptive threshold | C adaptive baseline |
|---|---:|---:|---:|
| Precision | 0.1347 | 0.0348 | 0.1002 |
| Recall | 0.0136 | 0.0041 | 0.0115 |
| F1-score | 0.0247 | 0.0074 | 0.0206 |
| False-positive rate | 0.0122 | 0.0159 | 0.0143 |
| Share of records flagged | 0.0123 | 0.0145 | 0.0139 |

Moving-block bootstrap over update intervals (1,000 resamples, blocks of 8 intervals), 95% interval of the difference to the static arm A:

| Metric | B - A | C - A |
|---|---|---|
| f1 | -0.0157 [-0.0399, -0.0011] | -0.0031 [-0.0077, +0.0001] |
| recall | -0.0087 [-0.0231, -0.0005] | -0.0015 [-0.0044, +0.0003] |
| fpr | +0.0050 [-0.0067, +0.0228] | +0.0025 [-0.0030, +0.0083] |

The intervals describe sampling variation across the stream only (one training run, one seed).

## Table 2b - per day (attack share varies: days 02-22 and 02-23 have 319 and 535 attacks, so recall/F1 there are not meaningful)

| Day | Records | Attacks | A static: F1 / recall / FPR | B adaptive threshold: F1 / recall / FPR | C adaptive baseline: F1 / recall / FPR |
|---|---:|---:|---|---|---|
| 2018-02-16 | 591,870 | 145,254 | 0.126 / 0.077 / 0.046 | 0.020 / 0.010 / 0.005 | 0.116 / 0.068 / 0.033 |
| 2018-02-20 | 5,432,970 | 575,364 | 0.002 / 0.001 / 0.009 | 0.003 / 0.001 / 0.007 | 0.002 / 0.001 / 0.015 |
| 2018-02-21 | 561,354 | 200,590 | 0.017 / 0.009 / 0.000 | 0.013 / 0.009 / 0.183 | 0.017 / 0.009 / 0.001 |
| 2018-02-22 | 744,193 | 319 | 0.000 / 0.003 / 0.009 | 0.000 / 0.000 / 0.013 | 0.005 / 0.147 / 0.025 |
| 2018-02-23 | 828,043 | 535 | 0.001 / 0.007 / 0.013 | 0.000 / 0.007 / 0.022 | 0.000 / 0.000 / 0.003 |
| 2018-02-28 | 377,006 | 43,299 | 0.031 / 0.018 / 0.019 | 0.007 / 0.004 / 0.008 | 0.005 / 0.003 / 0.005 |
| 2018-03-01 | 211,994 | 49,162 | 0.049 / 0.027 / 0.016 | 0.024 / 0.012 / 0.000 | 0.025 / 0.013 / 0.001 |
| 2018-03-02 | 756,606 | 144,337 | 0.001 / 0.001 / 0.018 | 0.000 / 0.000 / 0.000 | 0.002 / 0.001 / 0.018 |

## Table 2c - ranking quality per day (ROC-AUC / PR-AUC), static model vs retrained model

| Day | Attack types (day) | Static model (A, B) | Adaptive baseline model (C) |
|---|---|---|---|
| 2018-02-16 | Hulk, SlowHTTPTest | 0.079 / 0.171 | 0.472 / 0.261 |
| 2018-02-20 | LOIC-HTTP | 0.948 / 0.609 | 0.409 / 0.089 |
| 2018-02-21 | HOIC, LOIC-UDP | 0.013 / 0.214 | 0.745 / 0.533 |
| 2018-02-22 | web attacks (few) | 0.866 / 0.003 | 0.881 / 0.095 |
| 2018-02-23 | web attacks (few) | 0.868 / 0.005 | 0.848 / 0.003 |
| 2018-02-28 | Infiltration | 0.493 / 0.115 | 0.478 / 0.113 |
| 2018-03-01 | Infiltration | 0.533 / 0.279 | 0.527 / 0.272 |
| 2018-03-02 | Bot | 0.931 / 0.567 | 0.556 / 0.231 |

## Table 2d - threshold behaviour

The largest reconstruction error in the Phase 3 test scores was 0.072, so a threshold above that flags essentially nothing (for arm C's retrained models this is an approximation).

| Arm | Minimum | Median | Maximum | Intervals with threshold > 0.072 |
|---|---:|---:|---:|---:|
| A static | 0.0139 | 0.0139 | 0.0139 | 0.0% |
| B adaptive threshold | 0.00141 | 0.0162 | 1.66e+03 | 25.8% |
| C adaptive baseline | 5.7e-05 | 0.0111 | 0.134 | 4.6% |

## Table 2e - retraining and contamination (arm C)

- Retrainings: 64; accepted by the anchor check: 64.
- Share of attack records in the trusted buffer used for retraining (diagnostic from labels, never used for control): mean 0.123, maximum 0.495; attack share of the whole stream 0.122.
- Error on the clean anchor set after retraining relative to the initial model: median 0.94, maximum 1.50 (rejection threshold 3.0).

## Table 2f - attack types: share flagged over the whole stream

| Type | Records | A static | B adaptive threshold | C adaptive baseline |
|---|---:|---:|---:|---:|
| Benign | 8,345,176 | 0.012 | 0.016 | 0.014 |
| DoS-SlowHTTPTest | 55 | 0.000 | 0.000 | 0.000 |
| DoS-Hulk | 145,199 | 0.077 | 0.010 | 0.068 |
| DDoS-LOIC-HTTP | 575,364 | 0.001 | 0.001 | 0.001 |
| DDoS-LOIC-UDP | 1,730 | 0.999 | 0.999 | 1.000 |
| DDoS-HOIC | 198,860 | 0.000 | 0.000 | 0.000 |
| WebBruteForce | 551 | 0.000 | 0.000 | 0.085 |
| SQLInjection | 78 | 0.064 | 0.051 | 0.000 |
| WebXSS | 225 | 0.000 | 0.000 | 0.000 |
| Infiltration | 92,461 | 0.023 | 0.008 | 0.008 |
| Bot | 144,337 | 0.001 | 0.000 | 0.001 |

## Table 2g - sensitivity to the retraining interval (arm C; other settings unchanged)

| Retrain every (intervals of 50,000 records) | Retrainings | Mean trusted-buffer attack share | Overall F1 | Recall | FPR | Per-day ROC-AUC (02-16 / 02-20 / 02-21 / 03-02) |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 194 | 0.126 | 0.0270 | 0.0157 | 0.0207 | 0.436 / 0.379 / 0.762 / 0.513 |
| 3 | 64 | 0.123 | 0.0206 | 0.0115 | 0.0143 | 0.472 / 0.409 / 0.745 / 0.556 |
| 6 | 32 | 0.122 | 0.0165 | 0.0091 | 0.0119 | 0.437 / 0.469 / 0.710 / 0.671 |
