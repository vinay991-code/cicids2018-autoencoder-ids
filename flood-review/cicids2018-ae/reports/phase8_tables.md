Same warm-up model and evaluation stream as Phase 7 (9,504,036 records, 1,158,860 attacks, 8 days). Every arm's threshold is the 0.95-quantile of the errors of a fixed clean anchor set under the arm's current model (recomputed after each update). Update interval 50,000 records; retraining every 3 intervals; gate: margin 0.5, level tolerance 0.5 decades, flagged-share limit 0.15, persistence 3 intervals. One run, seed 42. Reconstruction error is treated as an ordinal anomaly score, not a calibrated probability.

## Table 3a - overall

| Metric | S static | U unrestricted | M margin only | G confidence-gated |
|---|---:|---:|---:|---:|
| Precision | 0.3789 | 0.2284 | 0.3873 | 0.2589 |
| Recall | 0.7035 | 0.1416 | 0.7017 | 0.2100 |
| F1-score | 0.4925 | 0.1748 | 0.4991 | 0.2319 |
| False-positive rate | 0.1602 | 0.0665 | 0.1541 | 0.0835 |
| Share of records flagged | 0.2264 | 0.0756 | 0.2209 | 0.0989 |

**Read with care:** on 02-16 and 02-21 the benign traffic is shifted entirely above the static threshold, so arms S and M flag 90-94% of all records there. That 'flag everything' behaviour inflates their overall recall and F1 (flagging everything gives F1 = 2p/(1+p) for attack share p) while their false-positive rate is 99.6% and 99.9% on those days. Judge by the per-day table and the false-positive rate.

Moving-block bootstrap over intervals (1,000 resamples, blocks of 8), 95% interval of the difference, G minus U and M minus U:

| Metric | G - U | M - U |
|---|---|---|
| f1 | +0.0593 [-0.0129, +0.1311] | +0.3415 [+0.0910, +0.5840] |
| recall | +0.0738 [-0.0081, +0.1685] | +0.5794 [+0.2998, +0.8124] |
| fpr | +0.0183 [+0.0043, +0.0443] | +0.0881 [+0.0197, +0.1871] |

## Table 3b - per day: F1 / recall / false-positive rate

| Day | Attack share | S static | U unrestricted | M margin only | G confidence-gated |
|---|---:|---|---|---|---|
| 2018-02-16 | 0.245 | 0.257 / 0.600 / 0.996 | 0.238 / 0.237 / 0.246 | 0.257 / 0.600 / 0.996 | 0.238 / 0.237 / 0.246 |
| 2018-02-20 | 0.106 | 0.743 / 0.941 / 0.070 | 0.083 / 0.061 / 0.050 | 0.754 / 0.941 / 0.066 | 0.156 / 0.126 / 0.058 |
| 2018-02-21 | 0.357 | 0.455 / 0.825 / 0.999 | 0.455 / 0.373 / 0.148 | 0.455 / 0.825 / 0.999 | 0.506 / 0.597 / 0.422 |
| 2018-02-22 | 0.000 | 0.005 / 0.408 / 0.063 | 0.010 / 0.502 / 0.043 | 0.007 / 0.367 / 0.045 | 0.010 / 0.502 / 0.043 |
| 2018-02-23 | 0.001 | 0.011 / 0.544 / 0.060 | 0.009 / 0.346 / 0.047 | 0.013 / 0.529 / 0.051 | 0.014 / 0.581 / 0.051 |
| 2018-02-28 | 0.115 | 0.132 / 0.141 / 0.128 | 0.116 / 0.107 / 0.096 | 0.123 / 0.123 / 0.114 | 0.116 / 0.109 / 0.099 |
| 2018-03-01 | 0.232 | 0.242 / 0.182 / 0.097 | 0.171 / 0.113 / 0.064 | 0.229 / 0.165 / 0.085 | 0.186 / 0.126 / 0.069 |
| 2018-03-02 | 0.191 | 0.062 / 0.040 / 0.058 | 0.096 / 0.062 / 0.056 | 0.057 / 0.037 / 0.057 | 0.058 / 0.037 / 0.056 |

(02-22 and 02-23 have 319 and 535 attacks, so recall/F1 there are not meaningful; their false-positive rates are.)

## Table 3c - ranking quality per day (ROC-AUC; S and M differ only slightly because M also retrains)

| Day | S static | U unrestricted | M margin only | G confidence-gated |
|---|---:|---:|---:|---:|
| 2018-02-16 | 0.079 | 0.469 | 0.079 | 0.462 |
| 2018-02-20 | 0.948 | 0.415 | 0.935 | 0.431 |
| 2018-02-21 | 0.013 | 0.748 | 0.016 | 0.498 |
| 2018-02-22 | 0.866 | 0.882 | 0.885 | 0.881 |
| 2018-02-23 | 0.868 | 0.849 | 0.873 | 0.868 |
| 2018-02-28 | 0.493 | 0.489 | 0.494 | 0.501 |
| 2018-03-01 | 0.533 | 0.538 | 0.539 | 0.545 |
| 2018-03-02 | 0.931 | 0.537 | 0.626 | 0.662 |

## Table 3d - contamination of the retraining buffer (share of attack records among the records used for retraining; diagnostic from labels, never used for control)

| Arm | Retrainings (accepted) | Mean | Maximum | 02-16 | 02-20 | 02-21 | 02-22 | 02-23 | 02-28 | 03-01 | 03-02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U unrestricted | 64 (64) | 0.123 | 0.490 | 0.292 | 0.115 | 0.289 | 0.038 | 0.001 | 0.048 | 0.118 | 0.179 |
| M margin only | 64 (60) | 0.038 | 0.461 | 0.000 | 0.012 | 0.189 | 0.032 | 0.000 | 0.044 | 0.114 | 0.163 |
| G confidence-gated | 68 (68) | 0.128 | 0.513 | 0.289 | 0.118 | 0.274 | 0.074 | 0.000 | 0.047 | 0.115 | 0.151 |

Attack share of the whole stream: 0.122.

## Table 3e - recovery after a change of traffic

Intervals (of 50,000 records) from the start of a day until the trailing 3-interval mean false-positive rate is at most 10% (twice the 5% target); 'never' = not reached before the day ends; 0 = no degradation at the start of the day. Days 02-16 and 02-21 are the two with a large shift of benign traffic.

| Day (intervals in day) | S static | U unrestricted | M margin only | G confidence-gated |
|---|---:|---:|---:|---:|
| 2018-02-16 (12) | never | 5 | never | 5 |
| 2018-02-20 (109) | 1 | 1 | 1 | 3 |
| 2018-02-21 (12) | never | 7 | never | 9 |
| 2018-02-22 (15) | 0 | 0 | 0 | 0 |
| 2018-02-23 (17) | 0 | 0 | 0 | 0 |
| 2018-02-28 (8) | 0 | 0 | 0 | 0 |
| 2018-03-01 (5) | 0 | 0 | 0 | 0 |
| 2018-03-02 (16) | 1 | 1 | 1 | 1 |

## Table 3f - share of each attack type flagged over the whole stream

| Type | Records | S static | U unrestricted | M margin only | G confidence-gated |
|---|---:|---:|---:|---:|---:|
| Benign | 8,345,176 | 0.160 | 0.066 | 0.154 | 0.083 |
| DoS-SlowHTTPTest | 55 | 1.000 | 1.000 | 1.000 | 1.000 |
| DoS-Hulk | 145,199 | 0.600 | 0.237 | 0.600 | 0.237 |
| DDoS-LOIC-HTTP | 575,364 | 0.941 | 0.061 | 0.941 | 0.126 |
| DDoS-LOIC-UDP | 1,730 | 1.000 | 1.000 | 1.000 | 1.000 |
| DDoS-HOIC | 198,860 | 0.823 | 0.368 | 0.823 | 0.593 |
| WebBruteForce | 551 | 0.485 | 0.363 | 0.477 | 0.588 |
| SQLInjection | 78 | 0.513 | 0.462 | 0.295 | 0.436 |
| WebXSS | 225 | 0.507 | 0.484 | 0.507 | 0.502 |
| Infiltration | 92,461 | 0.163 | 0.110 | 0.146 | 0.118 |
| Bot | 144,337 | 0.040 | 0.062 | 0.037 | 0.037 |

## Table 3g - gate behaviour (arm G)

Intervals judged inconsistent: 22 of 194; releases of held intervals into the buffer (accepted as a lasting change): 7.

| Day | Intervals | Inconsistent | Releases |
|---|---:|---:|---:|
| 2018-02-16 | 12 | 3 | 1 |
| 2018-02-20 | 109 | 6 | 2 |
| 2018-02-21 | 12 | 6 | 2 |
| 2018-02-22 | 15 | 3 | 1 |
| 2018-02-23 | 17 | 0 | 0 |
| 2018-02-28 | 8 | 3 | 1 |
| 2018-03-01 | 5 | 0 | 0 |
| 2018-03-02 | 16 | 1 | 0 |

## Table 3h - sensitivity of the gated arm G to the persistence setting (others unchanged)

| Persistence (intervals) | Overall F1 | Recall | FPR | Mean buffer attack share | 02-16 FPR / recovery | 02-21 FPR / recovery | 02-20 recall | LOIC-HTTP flagged | HOIC flagged | Bot flagged |
|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| 3 | 0.2319 | 0.2100 | 0.0835 | 0.128 | 0.246 / 5 | 0.422 / 9 | 0.126 | 0.126 | 0.593 | 0.037 |
| 6 | 0.3275 | 0.3251 | 0.0917 | 0.120 | 0.486 / 8 | 0.285 / 8 | 0.246 | 0.246 | 0.823 | 0.037 |
| 12 | 0.3709 | 0.4754 | 0.1511 | 0.102 | 0.996 / never | 0.999 / never | 0.485 | 0.485 | 0.823 | 0.037 |
