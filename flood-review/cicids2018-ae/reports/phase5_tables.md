Test partition: 2,163,852 records, 258,132 attacks (11.93%), identical for all models.

## Table 1a - primary thresholds (rules fixed before seeing results)

| Metric | Unsupervised | One-class | Supervised | Sup.+decoder |
|---|---:|---:|---:|---:|
| Threshold rule | 95th pct. of val errors (no labels) | 99th pct. of benign val errors (val labels) | probability 0.5 | probability 0.5 |
| Accuracy | 0.8236 | 0.8558 | 0.9802 | 0.9736 |
| Precision | 0.1368 | 0.0348 | 0.9712 | 0.8623 |
| Recall | 0.0901 | 0.0078 | 0.8599 | 0.9265 |
| F1-score | 0.1086 | 0.0128 | 0.9122 | 0.8932 |
| Specificity | 0.9230 | 0.9706 | 0.9965 | 0.9800 |
| False-positive rate | 0.0770 | 0.0294 | 0.0035 | 0.0200 |
| False-negative rate | 0.9099 | 0.9922 | 0.1401 | 0.0735 |
| FPR on validation, same threshold | 0.0515 | 0.0100 | 0.0038 | 0.0043 |
| ROC-AUC | 0.4684 | 0.8894 | 0.9437 | 0.9796 |
| PR-AUC (chance = 0.1193) | 0.1090 | 0.4096 | 0.9139 | 0.9540 |

## Table 1b - validation-tuned thresholds (F1-optimal on validation; uses validation labels)

| Metric | Unsupervised | One-class | Supervised | Sup.+decoder |
|---|---:|---:|---:|---:|
| Threshold | 8.54637e-07 | 0.000207779 | 0.800144 | 0.845962 |
| Accuracy | 0.1194 | 0.9005 | 0.9801 | 0.9838 |
| Precision | 0.1193 | 0.5598 | 0.9868 | 0.9624 |
| Recall | 1.0000 | 0.7758 | 0.8448 | 0.8991 |
| F1-score | 0.2132 | 0.6504 | 0.9103 | 0.9297 |
| False-positive rate | 0.9999 | 0.0826 | 0.0015 | 0.0048 |
| False-negative rate | 0.0000 | 0.2242 | 0.1552 | 0.1009 |

Unsupervised column: the validation optimum is degenerate (flags 99.99% of records; F1 equals the trivial flag-everything value), so it is not a usable operating point.

## Table 1c - threshold-free metrics with 95% block-bootstrap intervals

| Metric | Unsupervised | One-class | Supervised | Sup.+decoder |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.468 [0.389, 0.528] | 0.889 [0.851, 0.916] | 0.944 [0.897, 0.984] | 0.980 [0.968, 0.986] |
| PR-AUC | 0.109 [0.067, 0.149] | 0.410 [0.326, 0.545] | 0.914 [0.855, 0.967] | 0.954 [0.917, 0.970] |
| Recall at 1% FPR | 0.008 [0.001, 0.016] | 0.005 [0.003, 0.008] | 0.869 [0.787, 0.941] | 0.910 [0.865, 0.942] |

## Table 1d - 95% block-bootstrap intervals at the primary thresholds

| Metric | Unsupervised | One-class | Supervised | Sup.+decoder |
|---|---:|---:|---:|---:|
| F1-score | 0.109 [0.028, 0.175] | 0.013 [0.006, 0.018] | 0.912 [0.859, 0.956] | 0.893 [0.830, 0.955] |
| Recall | 0.090 [0.020, 0.172] | 0.008 [0.003, 0.013] | 0.860 [0.771, 0.938] | 0.926 [0.877, 0.952] |
| Precision | 0.137 [0.048, 0.183] | 0.035 [0.029, 0.065] | 0.971 [0.953, 0.980] | 0.862 [0.749, 0.974] |
| False-positive rate | 0.077 [0.049, 0.125] | 0.029 [0.010, 0.065] | 0.003 [0.003, 0.004] | 0.020 [0.004, 0.051] |

## Table 1e - paired differences (same bootstrap resamples)

| Comparison | Metric | Mean difference | 95% interval | Share of resamples > 0 |
|---|---|---:|---|---:|
| supervised - normal_only | roc_auc | +0.0601 | [0.0368, 0.0800] | 1.00 |
| supervised - normal_only | pr_auc | +0.4923 | [0.4201, 0.5435] | 1.00 |
| supervised_recon - supervised | roc_auc | +0.0284 | [-0.0003, 0.0818] | 0.75 |
| supervised_recon - supervised | pr_auc | +0.0317 | [-0.0002, 0.0904] | 0.92 |
| normal_only - unsup_mixed | roc_auc | +0.4320 | [0.3433, 0.5216] | 1.00 |
| normal_only - unsup_mixed | pr_auc | +0.3219 | [0.2098, 0.4398] | 1.00 |

Intervals cover which test blocks are drawn, not training randomness: every model was trained once (seed 42).

## Table 1f - cost

| Metric | Unsupervised | One-class | Supervised | Sup.+decoder |
|---|---:|---:|---:|---:|
| Epochs run | 50 | 50 | 27 | 33 |
| Training time (s, CPU) | 442 | 376 | 313 | 750 |
| Inference, whole test set (s) | 1.7 | 1.6 | 1.9 | 4.0 |
| Inference (ms per 1000 records) | 0.77 | 0.74 | 0.89 | 1.85 |
| Parameters | 39,400 | 39,400 | 20,225 | 39,945 |
| Model file (KB) | 159 | 159 | 83 | 162 |

Process-wide peak memory was 5.15 GB (Phase 3) and 6.36 GB (Phase 4) including memory-mapped data; it cannot be attributed to a single model, so model size is given by parameters and file size.
Training and inference times are single wall-clock measurements on a shared CPU and are not reliable for ranking models: the plain supervised AE needed 313 s (27 epochs) in Phase 4 but 86 s (24 epochs) when retrained in Phase 6 on the same amount of data. Epochs run and parameter counts are the dependable cost figures.

## Table 1g - detection by attack type on the test partition

Flagged share at the primary threshold (benign row = false-positive rate), and ROC-AUC of the type against benign. Types with fewer than 110 test rows are marked (*) and are not reliable.

| Type | Test rows | Unsupervised flagged | One-class flagged | Supervised flagged | Sup.+decoder flagged | Unsupervised AUC | One-class AUC | Supervised AUC | Sup.+decoder AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 1,905,720 | 0.077 | 0.029 | 0.003 | 0.020 | - | - | - | - |
| Bot | 29,164 | 0.009 | 0.001 | 0.993 | 0.995 | 0.416 | 0.887 | 0.999 | 0.999 |
| DDoS-HOIC | 27,938 | 0.003 | 0.000 | 0.995 | 0.997 | 0.590 | 0.892 | 1.000 | 1.000 |
| DDoS-LOIC-HTTP | 105,838 | 0.001 | 0.002 | 0.999 | 0.999 | 0.286 | 0.924 | 1.000 | 1.000 |
| DDoS-LOIC-UDP | 315 | 0.108 | 0.968 | 1.000 | 1.000 | 0.898 | 0.998 | 1.000 | 1.000 |
| DoS-GoldenEye | 9,422 | 0.415 | 0.000 | 0.991 | 0.995 | 0.852 | 0.934 | 0.999 | 0.999 |
| DoS-Hulk | 45,509 | 0.347 | 0.022 | 0.607 | 0.980 | 0.678 | 0.926 | 0.777 | 0.992 |
| DoS-SlowHTTPTest (*) | 38 | 1.000 | 0.000 | 0.000 | 0.947 | 0.940 | 0.931 | 0.938 | 0.989 |
| DoS-Slowloris | 1,962 | 0.301 | 0.000 | 0.999 | 0.999 | 0.874 | 0.910 | 0.999 | 0.998 |
| FTP-BruteForce (*) | 25 | 1.000 | 0.000 | 0.000 | 1.000 | 0.938 | 0.930 | 0.953 | 0.990 |
| Infiltration | 19,087 | 0.118 | 0.027 | 0.081 | 0.080 | 0.539 | 0.535 | 0.772 | 0.748 |
| SQLInjection (*) | 19 | 0.263 | 0.000 | 0.368 | 0.316 | 0.806 | 0.811 | 0.964 | 0.946 |
| SSH-BruteForce | 18,660 | 0.002 | 0.000 | 0.996 | 0.998 | 0.573 | 0.942 | 1.000 | 1.000 |
| WebBruteForce (*) | 105 | 0.267 | 0.000 | 0.390 | 0.276 | 0.756 | 0.731 | 0.829 | 0.864 |
| WebXSS (*) | 50 | 0.520 | 0.000 | 0.540 | 0.620 | 0.802 | 0.818 | 0.969 | 0.948 |
