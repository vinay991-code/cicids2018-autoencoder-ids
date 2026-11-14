# Phase 16 - adaptive threshold on a frozen autoencoder (generated from results_threshold/phase16_adaptive_threshold.json)

Only the threshold changes; the model, the scores and therefore ROC-AUC / PR-AUC are identical in every arm. Precision, recall, F1, false-positive rate (FPR) and false-negative rate (FNR = 1 - recall), counted over all 8 evaluation days (9,504,036 records, 194 intervals of 50,000).

## Table 1 - overall (all 8 days)

| Arm | Precision | Recall | F1 | FPR | FNR | MCC |
|---|---:|---:|---:|---:|---:|---:|
| S static | 0.1347 | 0.0136 | 0.0247 | 0.0122 | 0.9864 | 0.0043 |
| D adaptive threshold (lower quartile, rate-limited) | 0.4338 | 0.0031 | 0.0061 | 0.0006 | 0.9969 | 0.0280 |
| D1 = D without rate limit | 0.4348 | 0.0031 | 0.0061 | 0.0006 | 0.9969 | 0.0280 |
| D2 = Phase 7 B statistic + rate limit | 0.0701 | 0.0041 | 0.0077 | 0.0075 | 0.9959 | -0.0134 |
| Phase 7 B (all-record median + k MAD, no rate limit; quoted) | 0.0348 | 0.0041 | 0.0074 | 0.0159 | 0.9959 | - |
| Phase 7 C (adaptive baseline, retrained model; quoted) | 0.1002 | 0.0115 | 0.0206 | 0.0143 | 0.9885 | - |

## Table 2 - bootstrap difference to the static arm (moving-block bootstrap over intervals, 1,000 resamples, blocks of 8; mean [95% interval])

| Arm - S | Precision | Recall | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| D adaptive threshold (lower quartile, rate-limited) | +0.3017 [-0.0163, +0.5872] | -0.0098 [-0.0255, -0.0005] | -0.0168 [-0.0434, -0.0004] | -0.0112 [-0.0149, -0.0087] | +0.0098 [+0.0005, +0.0255] |
| D1 = D without rate limit | +0.2997 [-0.0163, +0.5838] | -0.0098 [-0.0255, -0.0005] | -0.0168 [-0.0435, -0.0004] | -0.0112 [-0.0149, -0.0087] | +0.0098 [+0.0005, +0.0255] |
| D2 = Phase 7 B statistic + rate limit | -0.0494 [-0.1180, +0.0131] | -0.0088 [-0.0232, -0.0005] | -0.0153 [-0.0392, -0.0008] | -0.0041 [-0.0091, +0.0003] | +0.0088 [+0.0005, +0.0232] |

## Table 3 - per day (S static / D adaptive / D2). Frozen-score ROC-AUC is the same for all arms.

| Day | Attacks | Benign median score vs warm-up (x) | ROC-AUC (frozen) | FPR S / D / D2 | Recall S / D / D2 | Precision S / D / D2 | Median threshold S / D / D2 |
|---|---:|---:|---:|---|---|---|---|
| 2018-02-16 | 145,254 | 253 | 0.079 | 0.0457 / 0.0008 / 0.0052 | 0.0766 / 0.0017 / 0.0109 | 0.353 / 0.425 / 0.405 | 0.0139 / 0.344 / 0.0282 |
| 2018-02-20 | 575,364 | 2 | 0.948 | 0.0090 / 0.0001 / 0.0052 | 0.0013 / 0.0012 / 0.0014 | 0.017 / 0.567 / 0.030 | 0.0139 / 0.0241 / 0.0173 |
| 2018-02-21 | 200,590 | 61 | 0.013 | 0.0001 / 0.0000 / 0.0003 | 0.0086 / 0.0085 / 0.0086 | 0.981 / 0.999 / 0.947 | 0.0139 / 0.158 / 0.0118 |
| 2018-02-22 | 319 | 1 | 0.866 | 0.0086 / 0.0001 / 0.0157 | 0.0031 / 0.0000 / 0.0031 | 0.000 / 0.000 / 0.000 | 0.0139 / 0.0623 / 0.00773 |
| 2018-02-23 | 535 | 1 | 0.868 | 0.0131 / 0.0013 / 0.0243 | 0.0075 / 0.0000 / 0.0075 | 0.000 / 0.000 / 0.000 | 0.0139 / 0.0162 / 0.00908 |
| 2018-02-28 | 43,299 | 2 | 0.493 | 0.0193 / 0.0034 / 0.0082 | 0.0179 / 0.0052 / 0.0035 | 0.108 / 0.164 / 0.053 | 0.0139 / 0.0168 / 0.0457 |
| 2018-03-01 | 49,162 | 2 | 0.533 | 0.0162 / 0.0008 / 0.0004 | 0.0266 / 0.0134 / 0.0095 | 0.331 / 0.828 / 0.884 | 0.0139 / 0.024 / 0.0789 |
| 2018-03-02 | 144,337 | 1 | 0.931 | 0.0179 / 0.0022 / 0.0002 | 0.0006 / 0.0000 / 0.0000 | 0.008 / 0.005 / 0.000 | 0.0139 / 0.0219 / 0.132 |

## Table 4 - shifted periods

Rule declared in advance (static FPR > 0.5 on the day): matches **0 days** of this stream, so there is no shifted period under it.

Post-hoc, descriptive grouping (labels used, added after seeing that; benign median score >= 10 x warm-up median): bulk-shifted days 2018-02-16, 2018-02-21; other days 2018-02-20, 2018-02-22, 2018-02-23, 2018-02-28, 2018-03-01, 2018-03-02.

| Arm | Group | Precision | Recall | F1 | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|
| S static | bulk-shifted days | 0.3857 | 0.0372 | 0.0678 | 0.0253 | 0.9628 |
| S static | other days | 0.0350 | 0.0036 | 0.0065 | 0.0107 | 0.9964 |
| D adaptive threshold (lower quartile, rate-limited) | bulk-shifted days | 0.8519 | 0.0057 | 0.0113 | 0.0004 | 0.9943 |
| D adaptive threshold (lower quartile, rate-limited) | other days | 0.2706 | 0.0020 | 0.0039 | 0.0006 | 0.9980 |
| D1 = D without rate limit | bulk-shifted days | 0.8873 | 0.0056 | 0.0112 | 0.0003 | 0.9944 |
| D1 = D without rate limit | other days | 0.2682 | 0.0020 | 0.0039 | 0.0006 | 0.9980 |
| D2 = Phase 7 B statistic + rate limit | bulk-shifted days | 0.5782 | 0.0096 | 0.0188 | 0.0030 | 0.9904 |
| D2 = Phase 7 B statistic + rate limit | other days | 0.0229 | 0.0017 | 0.0032 | 0.0080 | 0.9983 |

## Table 5 - threshold behaviour and contamination of the statistic

| Arm | Intervals with no alarm at all | Median |log10(threshold / oracle benign p99)| (decades) | Largest threshold | Smallest threshold |
|---|---:|---:|---:|---:|
| S static | 7 of 194 | 0.00 | 0.0139 | 0.0139 |
| D adaptive threshold (lower quartile, rate-limited) | 64 of 194 | 0.23 | 6.39 | 0.0139 |
| D1 = D without rate limit | 56 of 194 | 0.23 | 6.39 | 0.0139 |
| D2 = Phase 7 B statistic + rate limit | 64 of 194 | 0.26 | 13.9 | 0.00206 |

Oracle benign p99 = 99th percentile of the benign scores of the interval (uses labels, diagnostic only). Attack share among the records scoring below the interval median (the part that a lower-quartile statistic is built from): mean 0.066, maximum 1.000.

## Reproduction and structure checks

- PASS V1 numpy model reproduces the stored warm-up threshold (p99 of validation errors): recomputed 1.393706165e-02, stored 1.393706165e-02, relative difference 0.00e+00
- PASS V2 same intervals as Phase 7 (ids, days, sizes): 194 intervals, 50,000 records each except day ends
- PASS V3 static counts equal Phase 7 arm A (tp/fp/fn/tn per interval): sum of absolute differences over 776 count cells = 6 (records within float rounding of the threshold)
- PASS V4 rate limit respected (D_adaptive): largest step 0.2500 decades per interval, limit 0.25
- PASS V4 rate limit respected (D2_median_mad_limited): largest step 0.2500 decades per interval, limit 0.25
- PASS V5 clamp respected (S_static): log10 range -1.856 .. -1.856 (allowed -2.856 .. 1.144)
- PASS V5 clamp respected (D_adaptive): log10 range -1.856 .. 0.806 (allowed -2.856 .. 1.144)
- PASS V5 clamp respected (D1_no_rate_limit): log10 range -1.856 .. 0.806 (allowed -2.856 .. 1.144)
- PASS V5 clamp respected (D2_median_mad_limited): log10 range -2.686 .. 1.144 (allowed -2.856 .. 1.144)
- PASS V6 static arm threshold constant: all intervals use theta0
- PASS V7 policy trace is a pure function of the scores: recomputed trace identical (the function has no label argument)
- PASS V8 per-day ROC-AUC equals Phase 7 static: tolerance 1e-3
