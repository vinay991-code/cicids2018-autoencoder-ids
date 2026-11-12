Evaluation stream: 9,504,036 records, 1,158,860 attacks, 8 days, update interval 50,000 records. Drift monitor: 8 ADWIN detectors (log_error, flagged_share, flow_duration, tot_fwd_pkts, totlen_fwd_pkts, flow_byts_s, pkt_len_mean, udp_share) on batches of 5,000 records scored by the fixed warm-up model, delta 0.002 split over the detectors (Bonferroni), pre-fed with the last 300,000 warm-up records. Drift-triggered arms retrain for 3 rounds after an alarm on the last 3 intervals' samples with 25% clean replay; arm C acts only if the deviation persists one more interval (median score more than 0.5 decades from the reference, or more than 15% flagged) and excludes the highest-scoring 20% of the samples. Thresholds: 95th percentile of the clean anchor set under each arm's current model. One run, seed 42. Reconstruction error is an ordinal score, not a calibrated probability.

## Table 4a - overall

| Metric | A no detection | A2 periodic (ref.) | B drift-triggered | C drift-triggered + confirmation |
|---|---:|---:|---:|---:|
| Precision | 0.3789 | 0.2273 | 0.3806 | 0.3659 |
| Recall | 0.7035 | 0.1410 | 0.3071 | 0.3830 |
| F1-score | 0.4925 | 0.1741 | 0.3399 | 0.3742 |
| False-positive rate | 0.1602 | 0.0666 | 0.0694 | 0.0922 |
| Share flagged | 0.2264 | 0.0757 | 0.0984 | 0.1276 |
| Retrainings | 0 | 64 | 45 | 15 |
| Mean attack share in the retraining data (stream: 0.122) | n/a | 0.123 | 0.150 | 0.321 |
| Maximum attack share in the retraining data | n/a | 0.488 | 0.986 | 1.000 |

**Read with care:** on days where the static arm flags nearly every record (02-16, 02-21) its overall recall and F1 are inflated by flagging everything; compare false-positive rates and the per-day tables.

Moving-block bootstrap over intervals (1,000 resamples, blocks of 8), 95% interval of the difference:

| Metric | B - A2 (drift vs periodic) | C - B (confirmation) | B - A (drift vs none) |
|---|---|---|---|
| f1 | +0.1716 [+0.0160, +0.3448] | +0.0363 [-0.0159, +0.0868] | -0.1636 [-0.4659, +0.1570] |
| recall | +0.1705 [+0.0143, +0.3998] | +0.0791 [+0.0125, +0.1507] | -0.4116 [-0.7369, +0.0447] |
| fpr | +0.0006 [-0.0139, +0.0124] | +0.0220 [+0.0118, +0.0363] | -0.0935 [-0.1983, -0.0246] |

## Table 4b - per day: F1 / recall / false-positive rate

| Day | Attack share | Static FPR>0.5 (shifted) | A no detection | A2 periodic (ref.) | B drift-triggered | C drift-triggered + confirmation |
|---|---:|---|---|---|---|---|
| 2018-02-16 | 0.245 | yes | 0.257 / 0.600 / 0.996 | 0.238 / 0.237 / 0.246 | 0.236 / 0.267 / 0.323 | 0.241 / 0.307 / 0.403 |
| 2018-02-20 | 0.106 | no | 0.743 / 0.941 / 0.070 | 0.084 / 0.062 / 0.050 | 0.205 / 0.168 / 0.056 | 0.233 / 0.211 / 0.071 |
| 2018-02-21 | 0.357 | yes | 0.455 / 0.825 / 0.999 | 0.455 / 0.373 / 0.148 | 0.533 / 0.372 / 0.013 | 0.640 / 0.596 / 0.149 |
| 2018-02-22 | 0.000 | no | 0.005 / 0.408 / 0.063 | 0.010 / 0.502 / 0.044 | 0.007 / 0.382 / 0.046 | 0.007 / 0.498 / 0.065 |
| 2018-02-23 | 0.001 | no | 0.011 / 0.544 / 0.060 | 0.009 / 0.350 / 0.047 | 0.008 / 0.293 / 0.048 | 0.013 / 0.596 / 0.060 |
| 2018-02-28 | 0.115 | no | 0.132 / 0.141 / 0.128 | 0.116 / 0.108 / 0.096 | 0.129 / 0.130 / 0.114 | 0.131 / 0.137 / 0.122 |
| 2018-03-01 | 0.232 | no | 0.242 / 0.182 / 0.097 | 0.171 / 0.113 / 0.063 | 0.237 / 0.172 / 0.083 | 0.237 / 0.177 / 0.095 |
| 2018-03-02 | 0.191 | no | 0.062 / 0.040 / 0.058 | 0.083 / 0.054 / 0.056 | 0.848 / 0.912 / 0.056 | 0.889 / 0.992 / 0.056 |

## Table 4c - drift detection (shared by arms B and C)

True benign change points (evaluation only): 2018-02-16 start (shifted); 2018-02-20 start (back to reference); 2018-02-21 start (shifted); 2018-02-22 start (back to reference). An alarm counts as a detection if it falls between the change point and the end of that day (at most 600,000 records later); alarms elsewhere are false alarms.

| True change point | Record index | First alarm (records after the change) | Delay in monitoring batches | Monitors that fired first | Alarms in the window |
|---|---:|---:|---:|---|---:|
| 2018-02-16 (shifted) | 0 | 190,000 | 38 | flagged_share | 1 |
| 2018-02-20 (back to reference) | 591,870 | 125,000 | 25 | flagged_share | 1 |
| 2018-02-21 (shifted) | 6,024,840 | 55,000 | 11 | flow_duration | 6 |
| 2018-02-22 (back to reference) | 6,586,194 | 110,000 | 22 | flagged_share | 2 |

Detected 4 of 4 change points; median delay 117,500 records. False alarms: 10 of 20 alarms (1.05 per million records); 3 of them fall in batches that are at least 25% attacks (attack onsets), the rest in benign-dominated batches.

| False alarm | Interval | Day | Record | Monitors | Attack share of the batch |
|---|---:|---|---:|---|---:|
| 1 | 27 | 2018-02-20 | 1,356,870 | log_error | 0.00 |
| 2 | 35 | 2018-02-20 | 1,761,870 | flagged_share | 0.56 |
| 3 | 47 | 2018-02-20 | 2,386,870 | totlen_fwd_pkts | 0.56 |
| 4 | 49 | 2018-02-20 | 2,461,870 | pkt_len_mean | 0.53 |
| 5 | 58 | 2018-02-20 | 2,931,870 | flagged_share | 0.00 |
| 6 | 74 | 2018-02-20 | 3,721,870 | log_error | 0.00 |
| 7 | 75 | 2018-02-20 | 3,776,870 | udp_share | 0.01 |
| 8 | 93 | 2018-02-20 | 4,651,870 | flow_byts_s | 0.00 |
| 9 | 151 | 2018-02-23 | 7,510,387 | flow_byts_s | 0.00 |
| 10 | 158 | 2018-02-23 | 7,845,387 | udp_share | 0.00 |

## Table 4d - performance around the true change points

Before = last 3 intervals of the previous day (not available for the first change); FPR recovery = intervals after the change until the trailing 3-interval FPR is at most 10%; F1 recovery = intervals until the trailing 3-interval F1 is at least 80% of the before-change F1 (n/a if no before value or it is zero). 'never' = not within the day.

| Change point | Arm | FPR before | FPR first 3 intervals after | FPR recovery | F1 before | F1 lowest after | F1 recovery |
|---|---|---:|---:|---:|---:|---:|---|
| 2018-02-16 (shifted) | A no detection | n/a | 0.997 | never | n/a | 0.193 | n/a |
| 2018-02-16 (shifted) | A2 periodic (ref.) | n/a | 0.997 | 5 | n/a | 0.143 | n/a |
| 2018-02-16 (shifted) | B drift-triggered | n/a | 0.997 | 6 | n/a | 0.147 | n/a |
| 2018-02-16 (shifted) | C drift-triggered + confirmation | n/a | 0.997 | 7 | n/a | 0.160 | n/a |
| 2018-02-20 (back to reference) | A no detection | 0.995 | 0.106 | 1 | 0.261 | 0.041 | 19 |
| 2018-02-20 (back to reference) | A2 periodic (ref.) | 0.029 | 0.106 | 1 | 0.160 | 0.000 | 19 |
| 2018-02-20 (back to reference) | B drift-triggered | 0.029 | 0.107 | 3 | 0.160 | 0.000 | 19 |
| 2018-02-20 (back to reference) | C drift-triggered + confirmation | 0.029 | 0.108 | 3 | 0.160 | 0.001 | 19 |
| 2018-02-21 (shifted) | A no detection | 0.095 | 0.918 | never | n/a | 0.000 | n/a |
| 2018-02-21 (shifted) | A2 periodic (ref.) | 0.060 | 0.899 | 7 | n/a | 0.000 | n/a |
| 2018-02-21 (shifted) | B drift-triggered | 0.078 | 0.888 | 6 | n/a | 0.000 | n/a |
| 2018-02-21 (shifted) | C drift-triggered + confirmation | 0.105 | 0.915 | 7 | n/a | 0.000 | n/a |
| 2018-02-22 (back to reference) | A no detection | 1.000 | 0.066 | 0 | 0.000 | 0.010 | n/a |
| 2018-02-22 (back to reference) | A2 periodic (ref.) | 0.000 | 0.055 | 0 | 0.000 | 0.014 | n/a |
| 2018-02-22 (back to reference) | B drift-triggered | 0.001 | 0.073 | 0 | 0.000 | 0.011 | n/a |
| 2018-02-22 (back to reference) | C drift-triggered + confirmation | 0.001 | 0.068 | 0 | 0.000 | 0.009 | n/a |

## Table 4e - ranking quality per day (ROC-AUC)

| Day | A no detection | A2 periodic (ref.) | B drift-triggered | C drift-triggered + confirmation |
|---|---:|---:|---:|---:|
| 2018-02-16 | 0.079 | 0.473 | 0.480 | 0.461 |
| 2018-02-20 | 0.948 | 0.416 | 0.465 | 0.546 |
| 2018-02-21 | 0.013 | 0.729 | 0.698 | 0.612 |
| 2018-02-22 | 0.866 | 0.858 | 0.870 | 0.815 |
| 2018-02-23 | 0.868 | 0.834 | 0.868 | 0.829 |
| 2018-02-28 | 0.493 | 0.494 | 0.491 | 0.487 |
| 2018-03-01 | 0.533 | 0.538 | 0.540 | 0.538 |
| 2018-03-02 | 0.931 | 0.540 | 0.945 | 0.953 |

## Table 4f - share of each attack type flagged over the whole stream

| Type | Records | A no detection | A2 periodic (ref.) | B drift-triggered | C drift-triggered + confirmation |
|---|---:|---:|---:|---:|---:|
| Benign | 8,345,176 | 0.160 | 0.067 | 0.069 | 0.092 |
| DoS-SlowHTTPTest | 55 | 1.000 | 1.000 | 1.000 | 1.000 |
| DoS-Hulk | 145,199 | 0.600 | 0.237 | 0.267 | 0.306 |
| DDoS-LOIC-HTTP | 575,364 | 0.941 | 0.062 | 0.168 | 0.211 |
| DDoS-LOIC-UDP | 1,730 | 1.000 | 1.000 | 1.000 | 1.000 |
| DDoS-HOIC | 198,860 | 0.823 | 0.368 | 0.367 | 0.593 |
| WebBruteForce | 551 | 0.485 | 0.363 | 0.247 | 0.593 |
| SQLInjection | 78 | 0.513 | 0.487 | 0.436 | 0.487 |
| WebXSS | 225 | 0.507 | 0.484 | 0.484 | 0.502 |
| Infiltration | 92,461 | 0.163 | 0.111 | 0.152 | 0.158 |
| Bot | 144,337 | 0.040 | 0.054 | 0.912 | 0.992 |

## Table 4g - arm C: alarms confirmed or cancelled

Alarms acted on or cancelled by the persistence check: 5 confirmed, 13 cancelled (18 alarms reached the arm).
