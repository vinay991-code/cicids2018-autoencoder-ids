# DAE-LR over seeds (Phase 14)

Seeds: 42, 1, 2 (n = 3). Trained on the same rows as `unsup_mixed` (mixed) or `normal_only` (benign only); only the loss differs. Intervals: mean ± t(0.975, n-1)·sd/√n, cut to [0, 1]; with n this small they are wide. L1 = 95th percentile of all unlabeled validation errors (no attack labels); B99 = 99th percentile of benign validation errors (uses validation labels). Rows marked (analysis) are diagnostics.

| Method | Metric | Mean | sd | 95% CI of mean | min–max |
|---|---|---:|---:|---:|---:|
| unsup_mixed (ordinary loss) | ROC-AUC | 0.506 | 0.049 | [0.384, 0.629] | 0.468–0.562 |
| unsup_mixed (ordinary loss) | PR-AUC | 0.116 | 0.011 | [0.090, 0.142] | 0.109–0.128 |
| unsup_mixed (ordinary loss) | F1 at its own rule | 0.086 | 0.051 | [0.000, 0.213] | 0.028–0.122 |
| unsup_mixed (ordinary loss) | Recall at its own rule | 0.071 | 0.043 | [0.000, 0.177] | 0.023–0.102 |
| unsup_mixed (ordinary loss) | FPR at its own rule | 0.078 | 0.003 | [0.071, 0.086] | 0.076–0.082 |
| normal_only (benign-only, ordinary loss) | ROC-AUC | 0.902 | 0.012 | [0.873, 0.932] | 0.889–0.913 |
| normal_only (benign-only, ordinary loss) | PR-AUC | 0.482 | 0.063 | [0.326, 0.639] | 0.410–0.524 |
| normal_only (benign-only, ordinary loss) | F1 at its own rule | 0.374 | 0.314 | [0.000, 1.000] | 0.013–0.573 |
| normal_only (benign-only, ordinary loss) | Recall at its own rule | 0.327 | 0.277 | [0.000, 1.000] | 0.008–0.508 |
| normal_only (benign-only, ordinary loss) | FPR at its own rule | 0.034 | 0.004 | [0.024, 0.043] | 0.029–0.036 |
| DAE-LR trained on unlabeled mixed data, lambda=0.01 | ROC-AUC | 0.482 | 0.029 | [0.410, 0.554] | 0.448–0.500 |
| DAE-LR trained on unlabeled mixed data, lambda=0.01 | PR-AUC | 0.112 | 0.004 | [0.102, 0.122] | 0.107–0.115 |
| DAE-LR trained on unlabeled mixed data, lambda=0.01 | F1 at L1 | 0.086 | 0.023 | [0.028, 0.144] | 0.072–0.113 |
| DAE-LR trained on unlabeled mixed data, lambda=0.01 | Recall at L1 | 0.070 | 0.020 | [0.021, 0.119] | 0.059–0.093 |
| DAE-LR trained on unlabeled mixed data, lambda=0.01 | FPR at L1 | 0.075 | 0.002 | [0.071, 0.079] | 0.073–0.077 |
| DAE-LR trained on unlabeled mixed data, lambda=0.01 | (analysis) validation reconstruction error (sum) at best epoch | 0.007 | 0.002 | [0.002, 0.012] | 0.005–0.009 |
| DAE-LR trained on unlabeled mixed data, lambda=0.01 | (analysis) latent term ||z-c||^2 at best epoch | 0.070 | 0.009 | [0.048, 0.091] | 0.061–0.079 |
| DAE-LR trained on unlabeled mixed data, lambda=0.1 | ROC-AUC | 0.489 | 0.008 | [0.470, 0.508] | 0.483–0.498 |
| DAE-LR trained on unlabeled mixed data, lambda=0.1 | PR-AUC | 0.112 | 0.001 | [0.108, 0.115] | 0.110–0.113 |
| DAE-LR trained on unlabeled mixed data, lambda=0.1 | F1 at L1 | 0.068 | 0.017 | [0.027, 0.109] | 0.057–0.087 |
| DAE-LR trained on unlabeled mixed data, lambda=0.1 | Recall at L1 | 0.055 | 0.014 | [0.020, 0.089] | 0.045–0.071 |
| DAE-LR trained on unlabeled mixed data, lambda=0.1 | FPR at L1 | 0.075 | 0.000 | [0.074, 0.076] | 0.074–0.075 |
| DAE-LR trained on unlabeled mixed data, lambda=0.1 | (analysis) validation reconstruction error (sum) at best epoch | 0.013 | 0.004 | [0.003, 0.022] | 0.010–0.017 |
| DAE-LR trained on unlabeled mixed data, lambda=0.1 | (analysis) latent term ||z-c||^2 at best epoch | 0.009 | 0.003 | [0.003, 0.016] | 0.007–0.012 |
| DAE-LR trained on unlabeled mixed data, lambda=1 | ROC-AUC | 0.496 | 0.014 | [0.461, 0.532] | 0.480–0.506 |
| DAE-LR trained on unlabeled mixed data, lambda=1 | PR-AUC | 0.116 | 0.003 | [0.108, 0.123] | 0.112–0.118 |
| DAE-LR trained on unlabeled mixed data, lambda=1 | F1 at L1 | 0.071 | 0.019 | [0.023, 0.120] | 0.059–0.094 |
| DAE-LR trained on unlabeled mixed data, lambda=1 | Recall at L1 | 0.058 | 0.016 | [0.017, 0.098] | 0.047–0.076 |
| DAE-LR trained on unlabeled mixed data, lambda=1 | FPR at L1 | 0.075 | 0.000 | [0.074, 0.076] | 0.075–0.075 |
| DAE-LR trained on unlabeled mixed data, lambda=1 | (analysis) validation reconstruction error (sum) at best epoch | 0.019 | 0.007 | [0.002, 0.037] | 0.014–0.027 |
| DAE-LR trained on unlabeled mixed data, lambda=1 | (analysis) latent term ||z-c||^2 at best epoch | 0.002 | 0.000 | [0.001, 0.003] | 0.002–0.003 |
| DAE-LR trained on unlabeled mixed data, lambda=10 | ROC-AUC | 0.491 | 0.019 | [0.445, 0.538] | 0.476–0.512 |
| DAE-LR trained on unlabeled mixed data, lambda=10 | PR-AUC | 0.110 | 0.004 | [0.100, 0.120] | 0.105–0.113 |
| DAE-LR trained on unlabeled mixed data, lambda=10 | F1 at L1 | 0.039 | 0.017 | [0.000, 0.081] | 0.028–0.058 |
| DAE-LR trained on unlabeled mixed data, lambda=10 | Recall at L1 | 0.031 | 0.014 | [0.000, 0.065] | 0.021–0.047 |
| DAE-LR trained on unlabeled mixed data, lambda=10 | FPR at L1 | 0.073 | 0.007 | [0.056, 0.091] | 0.066–0.079 |
| DAE-LR trained on unlabeled mixed data, lambda=10 | (analysis) validation reconstruction error (sum) at best epoch | 0.078 | 0.047 | [0.000, 0.195] | 0.024–0.112 |
| DAE-LR trained on unlabeled mixed data, lambda=10 | (analysis) latent term ||z-c||^2 at best epoch | 0.000 | 0.000 | [0.000, 0.001] | 0.000–0.001 |
| DAE-LR trained on benign rows only, lambda=0.01 | ROC-AUC | 0.904 | 0.007 | [0.887, 0.922] | 0.897–0.911 |
| DAE-LR trained on benign rows only, lambda=0.01 | PR-AUC | 0.486 | 0.021 | [0.435, 0.537] | 0.465–0.506 |
| DAE-LR trained on benign rows only, lambda=0.01 | F1 at B99 | 0.365 | 0.151 | [0.000, 0.739] | 0.226–0.525 |
| DAE-LR trained on benign rows only, lambda=0.01 | Recall at B99 | 0.290 | 0.146 | [0.000, 0.653] | 0.160–0.448 |
| DAE-LR trained on benign rows only, lambda=0.01 | FPR at B99 | 0.035 | 0.000 | [0.034, 0.035] | 0.035–0.035 |
| DAE-LR trained on benign rows only, lambda=0.01 | (analysis) validation reconstruction error (sum) at best epoch | 0.012 | 0.002 | [0.008, 0.016] | 0.010–0.013 |
| DAE-LR trained on benign rows only, lambda=0.01 | (analysis) latent term ||z-c||^2 at best epoch | 0.053 | 0.017 | [0.011, 0.095] | 0.043–0.072 |
| DAE-LR trained on benign rows only, lambda=0.1 | ROC-AUC | 0.884 | 0.013 | [0.852, 0.917] | 0.870–0.894 |
| DAE-LR trained on benign rows only, lambda=0.1 | PR-AUC | 0.426 | 0.046 | [0.311, 0.541] | 0.378–0.471 |
| DAE-LR trained on benign rows only, lambda=0.1 | F1 at B99 | 0.067 | 0.083 | [0.000, 0.274] | 0.015–0.163 |
| DAE-LR trained on benign rows only, lambda=0.1 | Recall at B99 | 0.044 | 0.058 | [0.000, 0.189] | 0.010–0.112 |
| DAE-LR trained on benign rows only, lambda=0.1 | FPR at B99 | 0.026 | 0.015 | [0.000, 0.063] | 0.009–0.035 |
| DAE-LR trained on benign rows only, lambda=0.1 | (analysis) validation reconstruction error (sum) at best epoch | 0.023 | 0.007 | [0.005, 0.041] | 0.015–0.029 |
| DAE-LR trained on benign rows only, lambda=0.1 | (analysis) latent term ||z-c||^2 at best epoch | 0.013 | 0.007 | [0.000, 0.030] | 0.008–0.021 |
| DAE-LR trained on benign rows only, lambda=1 | ROC-AUC | 0.878 | 0.011 | [0.851, 0.904] | 0.866–0.887 |
| DAE-LR trained on benign rows only, lambda=1 | PR-AUC | 0.408 | 0.024 | [0.349, 0.467] | 0.380–0.423 |
| DAE-LR trained on benign rows only, lambda=1 | F1 at B99 | 0.038 | 0.016 | [0.000, 0.079] | 0.021–0.054 |
| DAE-LR trained on benign rows only, lambda=1 | Recall at B99 | 0.024 | 0.012 | [0.000, 0.053] | 0.011–0.035 |
| DAE-LR trained on benign rows only, lambda=1 | FPR at B99 | 0.026 | 0.015 | [0.000, 0.064] | 0.009–0.035 |
| DAE-LR trained on benign rows only, lambda=1 | (analysis) validation reconstruction error (sum) at best epoch | 0.029 | 0.010 | [0.004, 0.053] | 0.022–0.040 |
| DAE-LR trained on benign rows only, lambda=1 | (analysis) latent term ||z-c||^2 at best epoch | 0.003 | 0.001 | [0.001, 0.005] | 0.002–0.004 |
| DAE-LR trained on benign rows only, lambda=10 | ROC-AUC | 0.851 | 0.006 | [0.837, 0.865] | 0.846–0.857 |
| DAE-LR trained on benign rows only, lambda=10 | PR-AUC | 0.355 | 0.017 | [0.314, 0.396] | 0.340–0.373 |
| DAE-LR trained on benign rows only, lambda=10 | F1 at B99 | 0.027 | 0.005 | [0.013, 0.040] | 0.020–0.030 |
| DAE-LR trained on benign rows only, lambda=10 | Recall at B99 | 0.017 | 0.004 | [0.008, 0.026] | 0.013–0.019 |
| DAE-LR trained on benign rows only, lambda=10 | FPR at B99 | 0.035 | 0.001 | [0.033, 0.037] | 0.034–0.035 |
| DAE-LR trained on benign rows only, lambda=10 | (analysis) validation reconstruction error (sum) at best epoch | 0.047 | 0.008 | [0.027, 0.067] | 0.039–0.055 |
| DAE-LR trained on benign rows only, lambda=10 | (analysis) latent term ||z-c||^2 at best epoch | 0.000 | 0.000 | [0.000, 0.001] | 0.000–0.001 |
