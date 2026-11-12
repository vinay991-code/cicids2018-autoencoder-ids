# Trimmed-loss autoencoder over seeds (Phase 13)

Seeds: 42, 1, 2 (n = 3). Trained on the same unlabeled rows as `unsup_mixed`; only training randomness differs. Intervals: mean ± t(0.975, n-1)·sd/√n, cut to [0, 1]; with n this small they are wide. L1, L2 and L3 are label-free thresholds defined in `src/phase13_trimmed.py`. Rows marked (analysis) use labels for reporting only.

| Method | Metric | Mean | sd | 95% CI of mean | min–max |
|---|---|---:|---:|---:|---:|
| unsup_mixed (ordinary loss, 0% trimmed) | ROC-AUC | 0.506 | 0.049 | [0.384, 0.629] | 0.468–0.562 |
| unsup_mixed (ordinary loss, 0% trimmed) | PR-AUC | 0.116 | 0.011 | [0.090, 0.142] | 0.109–0.128 |
| unsup_mixed (ordinary loss, 0% trimmed) | F1 at its own rule | 0.086 | 0.051 | [0.000, 0.213] | 0.028–0.122 |
| unsup_mixed (ordinary loss, 0% trimmed) | Recall at its own rule | 0.071 | 0.043 | [0.000, 0.177] | 0.023–0.102 |
| unsup_mixed (ordinary loss, 0% trimmed) | FPR at its own rule | 0.078 | 0.003 | [0.071, 0.086] | 0.076–0.082 |
| normal_only (benign-only; threshold uses benign labels) | ROC-AUC | 0.902 | 0.012 | [0.873, 0.932] | 0.889–0.913 |
| normal_only (benign-only; threshold uses benign labels) | PR-AUC | 0.482 | 0.063 | [0.326, 0.639] | 0.410–0.524 |
| normal_only (benign-only; threshold uses benign labels) | F1 at its own rule | 0.374 | 0.314 | [0.000, 1.000] | 0.013–0.573 |
| normal_only (benign-only; threshold uses benign labels) | Recall at its own rule | 0.327 | 0.277 | [0.000, 1.000] | 0.008–0.508 |
| normal_only (benign-only; threshold uses benign labels) | FPR at its own rule | 0.034 | 0.004 | [0.024, 0.043] | 0.029–0.036 |
| trimmed 10% | ROC-AUC | 0.498 | 0.026 | [0.432, 0.564] | 0.476–0.527 |
| trimmed 10% | PR-AUC | 0.113 | 0.002 | [0.109, 0.116] | 0.111–0.114 |
| trimmed 10% | F1 at L1 | 0.042 | 0.021 | [0.000, 0.093] | 0.024–0.064 |
| trimmed 10% | F1 at L2 | 0.104 | 0.002 | [0.098, 0.109] | 0.102–0.106 |
| trimmed 10% | F1 at L3 | 0.106 | 0.003 | [0.097, 0.114] | 0.102–0.108 |
| trimmed 10% | Recall at L2 | 0.127 | 0.002 | [0.121, 0.133] | 0.125–0.130 |
| trimmed 10% | FPR at L2 | 0.179 | 0.001 | [0.176, 0.183] | 0.178–0.181 |
| trimmed 10% | (analysis) attack share kept in training | 0.129 | 0.001 | [0.127, 0.130] | 0.128–0.129 |
| trimmed 10% | (analysis) share of attacks removed from training | 0.056 | 0.004 | [0.045, 0.066] | 0.051–0.058 |
| trimmed 20% | ROC-AUC | 0.473 | 0.018 | [0.428, 0.517] | 0.459–0.493 |
| trimmed 20% | PR-AUC | 0.107 | 0.005 | [0.093, 0.120] | 0.101–0.112 |
| trimmed 20% | F1 at L1 | 0.048 | 0.027 | [0.000, 0.114] | 0.021–0.074 |
| trimmed 20% | F1 at L2 | 0.105 | 0.006 | [0.089, 0.120] | 0.098–0.111 |
| trimmed 20% | F1 at L3 | 0.092 | 0.001 | [0.090, 0.094] | 0.091–0.092 |
| trimmed 20% | Recall at L2 | 0.171 | 0.009 | [0.149, 0.193] | 0.162–0.180 |
| trimmed 20% | FPR at L2 | 0.284 | 0.005 | [0.273, 0.295] | 0.279–0.287 |
| trimmed 20% | (analysis) attack share kept in training | 0.142 | 0.001 | [0.141, 0.143] | 0.141–0.142 |
| trimmed 20% | (analysis) share of attacks removed from training | 0.074 | 0.003 | [0.067, 0.082] | 0.072–0.078 |
| trimmed 30% | ROC-AUC | 0.470 | 0.016 | [0.431, 0.510] | 0.454–0.486 |
| trimmed 30% | PR-AUC | 0.104 | 0.005 | [0.091, 0.116] | 0.100–0.109 |
| trimmed 30% | F1 at L1 | 0.027 | 0.006 | [0.012, 0.042] | 0.021–0.033 |
| trimmed 30% | F1 at L2 | 0.139 | 0.012 | [0.110, 0.168] | 0.126–0.147 |
| trimmed 30% | F1 at L3 | 0.116 | 0.008 | [0.097, 0.135] | 0.109–0.124 |
| trimmed 30% | Recall at L2 | 0.284 | 0.027 | [0.218, 0.351] | 0.254–0.302 |
| trimmed 30% | FPR at L2 | 0.378 | 0.003 | [0.371, 0.386] | 0.376–0.381 |
| trimmed 30% | (analysis) attack share kept in training | 0.154 | 0.005 | [0.141, 0.166] | 0.148–0.157 |
| trimmed 30% | (analysis) share of attacks removed from training | 0.123 | 0.029 | [0.050, 0.196] | 0.102–0.157 |
