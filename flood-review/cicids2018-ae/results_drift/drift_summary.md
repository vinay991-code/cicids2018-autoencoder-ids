# Phase 17 - concept-drift recovery with a supervised normality gate (generated from results_drift/phase17_drift_recovery.json)

NumPy fine-tuning (src/nptrain.py) of the Phase 7 warm-up autoencoder; same stream, intervals and starting threshold as Phases 7 and 16. Bulk-shifted days (post-hoc descriptive grouping, benign median score at least 10 x warm-up): 2018-02-16, 2018-02-21. Unseen types of the Phase 6 gate present in the stream: Bot, DDoS-HOIC, DoS-Hulk; known: DDoS-LOIC-HTTP, DDoS-LOIC-UDP, DoS-SlowHTTPTest, Infiltration, SQLInjection, WebBruteForce, WebXSS.

## Table 1 - overall at each arm's own threshold (all 8 days)

| Arm | Precision | Recall | F1 | FPR | FNR | MCC | Accepted updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| S frozen | 0.1347 | 0.0136 | 0.0247 | 0.0122 | 0.9864 | 0.0043 | 0 of 0 |
| N adaptive, no supervised gate | 0.1372 | 0.0136 | 0.0247 | 0.0119 | 0.9864 | 0.0052 | 64 of 64 |
| Gm gated by the Phase 4 supervised model | 0.1400 | 0.0138 | 0.0251 | 0.0118 | 0.9862 | 0.0061 | 64 of 64 |
| Gu gated by the Phase 6 model that never saw Bot, HOIC, Hulk | 0.1413 | 0.0137 | 0.0250 | 0.0116 | 0.9863 | 0.0065 | 64 of 64 |
| Phase 7 arm C (PyTorch, quoted) | 0.1002 | 0.0115 | 0.0206 | 0.0143 | 0.9885 | - | 64 of 64 |

## Table 2 - difference to the frozen model (moving-block bootstrap over intervals, 1,000 resamples, blocks of 8; mean [95% interval])

| Arm - S | Precision | Recall | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| N adaptive, no supervised gate | +0.0025 [-0.0030, +0.0119] | +0.0000 [-0.0002, +0.0003] | +0.0001 [-0.0003, +0.0006] | -0.0002 [-0.0011, +0.0005] | -0.0000 [-0.0003, +0.0002] |
| Gm gated by the Phase 4 supervised model | +0.0054 [-0.0009, +0.0151] | +0.0002 [-0.0002, +0.0005] | +0.0004 [-0.0002, +0.0010] | -0.0003 [-0.0013, +0.0004] | -0.0002 [-0.0005, +0.0002] |
| Gu gated by the Phase 6 model that never saw Bot, HOIC, Hulk | +0.0064 [-0.0007, +0.0178] | +0.0002 [-0.0001, +0.0007] | +0.0004 [-0.0001, +0.0013] | -0.0005 [-0.0016, +0.0003] | -0.0002 [-0.0007, +0.0001] |

## Table 3 - within-day ROC-AUC of the scores each arm produced (prequential). All = benign + all attacks of the day; known / unseen = benign + the attacks of that group

| Day | Attack types | S all | N all | Gm all | Gu all | S unseen | N unseen | Gm unseen | Gu unseen |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018-02-16 | DoS-Hulk, DoS-SlowHTTPTest | 0.079 | 0.466 | 0.741 | 0.466 | 0.079 | 0.466 | 0.741 | 0.466 |
| 2018-02-20 | DDoS-LOIC-HTTP | 0.948 | 0.430 | 0.947 | 0.948 | n/a | n/a | n/a | n/a |
| 2018-02-21 | DDoS-HOIC, DDoS-LOIC-UDP | 0.013 | 0.742 | 0.751 | 0.741 | 0.004 | 0.740 | 0.749 | 0.739 |
| 2018-02-22 | SQLInjection, WebBruteForce, WebXSS | 0.866 | 0.882 | 0.869 | 0.858 | n/a | n/a | n/a | n/a |
| 2018-02-23 | SQLInjection, WebBruteForce, WebXSS | 0.868 | 0.862 | 0.859 | 0.865 | n/a | n/a | n/a | n/a |
| 2018-02-28 | Infiltration | 0.493 | 0.494 | 0.500 | 0.501 | n/a | n/a | n/a | n/a |
| 2018-03-01 | Infiltration | 0.533 | 0.540 | 0.543 | 0.542 | n/a | n/a | n/a | n/a |
| 2018-03-02 | Bot | 0.931 | 0.581 | 0.935 | 0.556 | 0.931 | 0.581 | 0.935 | 0.556 |

## Table 4 - bulk-shifted days against other days at each arm's threshold

| Arm | Group | Precision | Recall | F1 | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|
| S frozen | bulk-shifted days | 0.3857 | 0.0372 | 0.0678 | 0.0253 | 0.9628 |
| S frozen | other days | 0.0350 | 0.0036 | 0.0065 | 0.0107 | 0.9964 |
| N adaptive, no supervised gate | bulk-shifted days | 0.4294 | 0.0371 | 0.0683 | 0.0211 | 0.9629 |
| N adaptive, no supervised gate | other days | 0.0343 | 0.0036 | 0.0065 | 0.0109 | 0.9964 |
| Gm gated by the Phase 4 supervised model | bulk-shifted days | 0.4295 | 0.0371 | 0.0684 | 0.0211 | 0.9629 |
| Gm gated by the Phase 4 supervised model | other days | 0.0371 | 0.0038 | 0.0070 | 0.0108 | 0.9962 |
| Gu gated by the Phase 6 model that never saw Bot, HOIC, Hulk | bulk-shifted days | 0.4295 | 0.0371 | 0.0684 | 0.0211 | 0.9629 |
| Gu gated by the Phase 6 model that never saw Bot, HOIC, Hulk | other days | 0.0370 | 0.0038 | 0.0068 | 0.0106 | 0.9962 |

## Table 5 - detection of known and unseen attacks and false alarms on shifted benign traffic (share of records flagged)

| Arm | Known attacks | Unseen attacks | Benign on bulk-shifted days | Benign on other days |
|---|---:|---:|---:|---:|
| S frozen | 0.0068 | 0.0230 | 0.0253 | 0.0107 |
| N adaptive, no supervised gate | 0.0069 | 0.0228 | 0.0211 | 0.0109 |
| Gm gated by the Phase 4 supervised model | 0.0072 | 0.0228 | 0.0211 | 0.0108 |
| Gu gated by the Phase 6 model that never saw Bot, HOIC, Hulk | 0.0071 | 0.0228 | 0.0211 | 0.0106 |

## Table 6 - recovery time on the shifted days

Rule (declared in advance): intervals from the start of the day until the interval-level ROC-AUC is at least 0.5 for 3 consecutive intervals; an interval needs at least 1,000 benign and 1,000 attack records. 'none' means the rule was not met; on 02-21 only 4 of 12 intervals meet the minimum-class condition, so the rule cannot be met there for any arm (not measurable).

| Arm | 02-16 | 02-21 | 02-16 interval AUC, intervals 1 to 12 |
|---|---:|---:|---|
| S frozen | none | none (not measurable) | 0.02 0.05 0.07 0.10 0.08 0.09 0.09 0.12 0.14 0.08 0.12 0.11 |
| N adaptive, no supervised gate | none | none (not measurable) | 0.02 0.05 0.07 0.25 0.29 0.33 0.30 0.31 0.34 0.46 0.49 0.48 |
| Gm gated by the Phase 4 supervised model | 4 | none (not measurable) | 0.02 0.05 0.07 0.76 0.80 0.84 0.93 0.94 0.95 0.97 0.95 0.95 |
| Gu gated by the Phase 6 model that never saw Bot, HOIC, Hulk | none | none (not measurable) | 0.02 0.05 0.07 0.25 0.28 0.32 0.32 0.34 0.38 0.44 0.49 0.47 |

## Table 7 - retraining data and the supervised gates (labels used, diagnostics only)

| Arm | Mean admitted records per update | Attack share in retraining data, mean / max | Anchor-error ratio range |
|---|---:|---:|---:|
| N adaptive, no supervised gate | 115,765 | 0.123 / 0.493 | 0.91 to 1.48 |
| Gm gated by the Phase 4 supervised model | 102,993 | 0.015 / 0.162 | 0.91 to 1.63 |
| Gu gated by the Phase 6 model that never saw Bot, HOIC, Hulk | 108,128 | 0.062 / 0.356 | 0.92 to 1.46 |

| Gate | Benign blocked | Attacks blocked, all records | Attacks blocked, test partition only (never seen) | Attacks blocked, training partition |
|---|---:|---:|---:|---:|
| Phase 4 supervised (all types seen) | 0.0041 | 0.9094 | 0.8422 | 0.9253 |
| Phase 6 supervised (Bot, HOIC, SSH-BruteForce, Hulk withheld) | 0.0036 | 0.5077 | 0.4765 | 0.4988 |

| Attack type | Bot | DDoS-HOIC | DDoS-LOIC-HTTP | DDoS-LOIC-UDP | DoS-Hulk | DoS-SlowHTTPTest | Infiltration | SQLInjection | WebBruteForce | WebXSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blocked by main gate | 0.994 | 0.998 | 0.999 | 1.000 | 0.877 | 0.000 | 0.085 | 0.308 | 0.394 | 0.533 |
| blocked by unseen gate | 0.019 | 0.001 | 0.999 | 1.000 | 0.003 | 0.018 | 0.089 | 0.321 | 0.436 | 0.689 |

## Table 8 - success criteria declared in advance

| Arm | (a) AUC >= 0.5 and above frozen on shifted days | (b) AUC within 0.05 of frozen on 02-20 and 03-02 | (c) overall F1 not lower than frozen | All three |
|---|:---:|:---:|:---:|:---:|
| N adaptive, no supervised gate | no | no | yes | no |
| Gm gated by the Phase 4 supervised model | yes | yes | yes | yes |
| Gu gated by the Phase 6 model that never saw Bot, HOIC, Hulk | no | no | yes | no |

## Validation checks

- PASS C1 numpy supervised gate reproduces saved PyTorch probabilities (Phase 4, first 200,000 test rows): max abs difference 7.75e-07 (tolerance 1e-5)
- PASS C2 numpy unseen-supervised gate with refit scaler reproduces saved PyTorch probabilities (Phase 6, first 200,000 test rows): max abs difference 6.85e-07 (tolerance 1e-5); saved scores cover the full test partition: True
- PASS C3 nptrain forward equals the verified npmodels forward on the anchor set: max abs difference 0.00e+00
- PASS C4 determinism: two runs give identical parameters and scores: 12 intervals, parameter hash 661b9155efdb2db9
- PASS C5 label independence: all labels replaced by zeros gives identical parameters, scores and thresholds: 12 intervals, accepted updates 4
- PASS C6 frozen arm scores equal the Phase 16 cached scores: max abs difference 6.71e-08 on 591,870 records
- PASS C7 every update lowered the reconstruction loss on its own training batch: 4 update attempts in the smoke run
