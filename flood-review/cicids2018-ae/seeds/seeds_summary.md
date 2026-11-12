# Results over several training seeds

Runs (seeds): 1, 2, 3, 42  (n = 4). Only training randomness differs between runs (initialisation, batch order, the 1,000,000 training rows); the data split and test set are identical. `95% CI of mean` is mean ± t(0.975, n-1)·sd/√n; `sd` and `min`–`max` show how much single runs differ. All metrics are proportions, so the interval is cut to [0, 1]. With n = 4 the intervals are wide.

Skipped incomplete seed folders: [('seed_4', "FileNotFoundError(2, 'No such file or directory')")]


## Table 1 over seeds: attack types seen in training (test partition, 2,163,852 records)

| Model | Metric | Mean | sd | 95% CI of mean | min–max |
|---|---|---:|---:|---:|---:|
| Unsupervised AE | ROC-AUC | 0.513 | 0.043 | [0.446, 0.581] | 0.468–0.562 |
| Unsupervised AE | PR-AUC | 0.118 | 0.009 | [0.103, 0.133] | 0.109–0.128 |
| Unsupervised AE | F1 | 0.094 | 0.044 | [0.023, 0.164] | 0.028–0.122 |
| Unsupervised AE | Recall | 0.078 | 0.037 | [0.019, 0.137] | 0.023–0.102 |
| Unsupervised AE | Precision | 0.118 | 0.055 | [0.031, 0.205] | 0.036–0.153 |
| Unsupervised AE | False-positive rate | 0.078 | 0.003 | [0.074, 0.082] | 0.076–0.082 |
| One-class AE | ROC-AUC | 0.899 | 0.011 | [0.882, 0.917] | 0.889–0.913 |
| One-class AE | PR-AUC | 0.467 | 0.060 | [0.372, 0.562] | 0.410–0.524 |
| One-class AE | F1 | 0.282 | 0.315 | [0.000, 0.784] | 0.006–0.573 |
| One-class AE | Recall | 0.246 | 0.278 | [0.000, 0.688] | 0.004–0.508 |
| One-class AE | Precision | 0.336 | 0.359 | [0.000, 0.907] | 0.015–0.658 |
| One-class AE | False-positive rate | 0.034 | 0.003 | [0.029, 0.039] | 0.029–0.036 |
| Supervised AE | ROC-AUC | 0.966 | 0.020 | [0.933, 0.998] | 0.944–0.983 |
| Supervised AE | PR-AUC | 0.938 | 0.024 | [0.899, 0.977] | 0.914–0.961 |
| Supervised AE | F1 | 0.917 | 0.026 | [0.875, 0.959] | 0.882–0.943 |
| Supervised AE | Recall | 0.895 | 0.030 | [0.847, 0.942] | 0.860–0.927 |
| Supervised AE | Precision | 0.941 | 0.040 | [0.878, 1.000] | 0.882–0.971 |
| Supervised AE | False-positive rate | 0.008 | 0.006 | [0.000, 0.017] | 0.003–0.016 |
| Supervised AE + decoder | ROC-AUC | 0.981 | 0.001 | [0.979, 0.983] | 0.980–0.983 |
| Supervised AE + decoder | PR-AUC | 0.955 | 0.004 | [0.949, 0.961] | 0.951–0.960 |
| Supervised AE + decoder | F1 | 0.912 | 0.035 | [0.857, 0.968] | 0.873–0.945 |
| Supervised AE + decoder | Recall | 0.922 | 0.010 | [0.907, 0.938] | 0.908–0.928 |
| Supervised AE + decoder | Precision | 0.906 | 0.073 | [0.789, 1.000] | 0.824–0.971 |
| Supervised AE + decoder | False-positive rate | 0.014 | 0.011 | [0.000, 0.032] | 0.004–0.027 |

## Table 5 over seeds: four attack types withheld from training

| Model | Metric | Mean | sd | 95% CI of mean | min–max |
|---|---|---:|---:|---:|---:|
| Unsupervised AE | Known attacks: recall | 0.074 | 0.022 | [0.038, 0.109] | 0.046–0.092 |
| Unsupervised AE | Benign: false-positive rate | 0.076 | 0.001 | [0.074, 0.077] | 0.075–0.077 |
| Unsupervised AE | Unseen attacks: recall | 0.848 | 0.096 | [0.695, 1.000] | 0.749–0.964 |
| Unsupervised AE | Known set ROC-AUC | 0.418 | 0.039 | [0.356, 0.481] | 0.393–0.476 |
| Unsupervised AE | Unseen set ROC-AUC | 0.932 | 0.003 | [0.926, 0.937] | 0.927–0.935 |
| Unsupervised AE | Bot: share flagged | 0.659 | 0.294 | [0.191, 1.000] | 0.273–0.982 |
| Unsupervised AE | Bot: ROC-AUC vs benign | 0.928 | 0.009 | [0.913, 0.942] | 0.920–0.941 |
| Unsupervised AE | DDoS-HOIC: share flagged | 0.863 | 0.000 | [0.863, 0.863] | 0.863–0.863 |
| Unsupervised AE | DDoS-HOIC: ROC-AUC vs benign | 0.901 | 0.011 | [0.883, 0.919] | 0.888–0.912 |
| Unsupervised AE | SSH-BruteForce: share flagged | 1.000 | 0.000 | [1.000, 1.000] | 1.000–1.000 |
| Unsupervised AE | SSH-BruteForce: ROC-AUC vs benign | 0.960 | 0.006 | [0.951, 0.970] | 0.953–0.966 |
| Unsupervised AE | DoS-Hulk: share flagged | 0.897 | 0.207 | [0.567, 1.000] | 0.586–1.000 |
| Unsupervised AE | DoS-Hulk: ROC-AUC vs benign | 0.941 | 0.007 | [0.930, 0.952] | 0.931–0.946 |
| One-class AE | Known attacks: recall | 0.177 | 0.264 | [0.000, 0.597] | 0.007–0.563 |
| One-class AE | Benign: false-positive rate | 0.030 | 0.013 | [0.009, 0.051] | 0.010–0.037 |
| One-class AE | Unseen attacks: recall | 0.052 | 0.080 | [0.000, 0.180] | 0.006–0.172 |
| One-class AE | Known set ROC-AUC | 0.877 | 0.009 | [0.862, 0.892] | 0.869–0.890 |
| One-class AE | Unseen set ROC-AUC | 0.920 | 0.013 | [0.900, 0.941] | 0.906–0.936 |
| One-class AE | Bot: share flagged | 0.008 | 0.011 | [0.000, 0.026] | 0.001–0.024 |
| One-class AE | Bot: ROC-AUC vs benign | 0.909 | 0.031 | [0.860, 0.957] | 0.869–0.941 |
| One-class AE | DDoS-HOIC: share flagged | 0.000 | 0.000 | [0.000, 0.000] | 0.000–0.000 |
| One-class AE | DDoS-HOIC: ROC-AUC vs benign | 0.877 | 0.007 | [0.866, 0.888] | 0.870–0.885 |
| One-class AE | SSH-BruteForce: share flagged | 0.251 | 0.499 | [0.000, 1.000] | 0.000–0.999 |
| One-class AE | SSH-BruteForce: ROC-AUC vs benign | 0.955 | 0.009 | [0.940, 0.970] | 0.944–0.966 |
| One-class AE | DoS-Hulk: share flagged | 0.029 | 0.011 | [0.012, 0.047] | 0.013–0.037 |
| One-class AE | DoS-Hulk: ROC-AUC vs benign | 0.940 | 0.011 | [0.922, 0.959] | 0.926–0.954 |
| Supervised AE | Known attacks: recall | 0.872 | 0.002 | [0.869, 0.875] | 0.870–0.874 |
| Supervised AE | Benign: false-positive rate | 0.004 | 0.002 | [0.002, 0.007] | 0.003–0.006 |
| Supervised AE | Unseen attacks: recall | 0.167 | 0.133 | [0.000, 0.378] | 0.008–0.333 |
| Supervised AE | Known set ROC-AUC | 0.969 | 0.000 | [0.969, 0.970] | 0.969–0.970 |
| Supervised AE | Unseen set ROC-AUC | 0.700 | 0.083 | [0.568, 0.832] | 0.648–0.824 |
| Supervised AE | Bot: share flagged | 0.188 | 0.204 | [0.000, 0.512] | 0.006–0.393 |
| Supervised AE | Bot: ROC-AUC vs benign | 0.745 | 0.160 | [0.490, 1.000] | 0.586–0.965 |
| Supervised AE | DDoS-HOIC: share flagged | 0.231 | 0.306 | [0.000, 0.718] | 0.000–0.646 |
| Supervised AE | DDoS-HOIC: ROC-AUC vs benign | 0.841 | 0.094 | [0.691, 0.991] | 0.770–0.979 |
| Supervised AE | SSH-BruteForce: share flagged | 0.000 | 0.000 | [0.000, 0.000] | 0.000–0.000 |
| Supervised AE | SSH-BruteForce: ROC-AUC vs benign | 0.244 | 0.064 | [0.142, 0.347] | 0.208–0.341 |
| Supervised AE | DoS-Hulk: share flagged | 0.182 | 0.116 | [0.000, 0.368] | 0.008–0.252 |
| Supervised AE | DoS-Hulk: ROC-AUC vs benign | 0.771 | 0.116 | [0.586, 0.955] | 0.613–0.886 |

Values per run are in `seeds_summary.json`.
