# Phase 11 - hybrid detector (supervised OR novelty), computed from saved scores

Rule fixed beforehand, nothing tuned: flag a record if the supervised probability is above 0.5 OR the novelty model's reconstruction error is above its primary threshold (95th percentile of validation errors for the unsupervised AE; 99th percentile of benign validation errors for the one-class AE). No retraining. 95% intervals are block bootstraps over test blocks.

## Table 6a - seen attack types (Phase 3-5 models, full test partition, 2,163,852 records)

| Detector | Precision | Recall [95% CI] | F1 | False-positive rate [95% CI] |
|---|---:|---:|---:|---:|
| supervised | 0.971 | 0.860 [0.771, 0.938] | 0.912 | 0.0035 [0.003, 0.004] |
| unsup_mixed | 0.137 | 0.090 [0.020, 0.172] | 0.109 | 0.0770 [0.049, 0.125] |
| normal_only | 0.035 | 0.008 [0.003, 0.013] | 0.013 | 0.0294 [0.010, 0.065] |
| sup OR unsup_mixed | 0.609 | 0.919 [0.879, 0.945] | 0.733 | 0.0798 [0.052, 0.127] |
| sup OR normal_only | 0.781 | 0.861 [0.772, 0.939] | 0.819 | 0.0327 [0.013, 0.068] |

## Table 6b - unseen attack types (Phase 6 models; Bot, DDoS-HOIC, SSH-BruteForce, DoS-Hulk withheld)

| Detector | Known-attack recall | Unseen-attack recall [95% CI] | Benign false-positive rate | Unseen-set F1 | Bot | DDoS-HOIC | SSH-BruteForce | DoS-Hulk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| supervised | 0.870 | 0.008 [0.005, 0.013] | 0.0031 | 0.015 | 0.019 | 0.000 | 0.000 | 0.008 |
| unsup_mixed | 0.046 | 0.749 [0.652, 0.879] | 0.0768 | 0.507 | 0.733 | 0.863 | 1.000 | 0.586 |
| normal_only | 0.007 | 0.013 [0.000, 0.025] | 0.0098 | 0.023 | 0.001 | 0.000 | 0.002 | 0.035 |
| sup OR unsup_mixed | 0.879 | 0.751 [0.654, 0.881] | 0.0790 | 0.502 | 0.742 | 0.863 | 1.000 | 0.586 |
| sup OR normal_only | 0.872 | 0.021 [0.005, 0.034] | 0.0127 | 0.035 | 0.019 | 0.000 | 0.002 | 0.043 |

Per-type columns are the share of that withheld type flagged. A hybrid's false-positive rate can never be lower than its supervised part, and is at most the sum of the two parts.
