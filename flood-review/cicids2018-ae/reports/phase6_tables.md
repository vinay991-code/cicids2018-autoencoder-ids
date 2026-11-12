Withheld attack types (excluded from the scaler fit, training, early stopping and validation/threshold rows): **Bot, DDoS-HOIC, SSH-BruteForce, DoS-Hulk**. Test partition: 2,163,852 records = 1,905,720 benign + 136,861 known-attack + 121,271 withheld-attack. Training rows removed: 415,595; the scaler refit on the remaining rows left every min/max unchanged.

## Test blocks per withheld type (each type is one attack episode)

| Type | Test rows | Distinct one-minute test blocks |
|---|---:|---:|
| Bot | 29,164 | 68 |
| DDoS-HOIC | 27,938 | 5 |
| SSH-BruteForce | 18,660 | 18 |
| DoS-Hulk | 45,509 | 1 |

Block-bootstrap intervals for a single type are therefore not meaningful when it occupies one or two blocks (Hulk's interval collapses to a point).

## Table 5a - known vs unseen attacks, primary thresholds (rules fixed in advance)

| Metric | Unsupervised | One-class | Supervised |
|---|---:|---:|---:|
| Threshold | 0.000199903 | 0.013903 | 0.5 |
| Known set: accuracy | 0.8644 | 0.9243 | 0.9884 |
| Known set: precision | 0.0413 | 0.0467 | 0.9526 |
| Known attacks: recall | 0.0461 | 0.0067 | 0.8704 |
| Known set: F1 | 0.0436 | 0.0117 | 0.9096 |
| Benign: false-positive rate | 0.0768 | 0.0098 | 0.0031 |
| Unseen attacks: recall | 0.7489 | 0.0135 | 0.0077 |
| Unseen set: precision | 0.3829 | 0.0807 | 0.1357 |
| Unseen set: F1 | 0.5067 | 0.0231 | 0.0145 |
| Unseen Bot: flagged share | 0.733 | 0.001 | 0.019 |
| Unseen DDoS-HOIC: flagged share | 0.863 | 0.000 | 0.000 |
| Unseen SSH-BruteForce: flagged share | 1.000 | 0.002 | 0.000 |
| Unseen DoS-Hulk: flagged share | 0.586 | 0.035 | 0.008 |

## Table 5b - known vs unseen attacks, validation-F1-tuned thresholds (filtered validation labels)

| Metric | Unsupervised | One-class | Supervised |
|---|---:|---:|---:|
| Threshold | 2.19208e-06 | 0.000372341 | 0.933639 |
| Known set: accuracy | 0.1535 | 0.9306 | 0.9899 |
| Known set: precision | 0.0726 | 0.4883 | 0.9876 |
| Known attacks: recall | 0.9881 | 0.7450 | 0.8594 |
| Known set: F1 | 0.1353 | 0.5899 | 0.9191 |
| Benign: false-positive rate | 0.9065 | 0.0561 | 0.0008 |
| Unseen attacks: recall | 1.0000 | 0.2327 | 0.0022 |
| Unseen set: precision | 0.0656 | 0.2089 | 0.1542 |
| Unseen set: F1 | 0.1231 | 0.2202 | 0.0044 |
| Unseen Bot: flagged share | 1.000 | 0.025 | 0.000 |
| Unseen DDoS-HOIC: flagged share | 1.000 | 0.013 | 0.000 |
| Unseen SSH-BruteForce: flagged share | 1.000 | 1.000 | 0.000 |
| Unseen DoS-Hulk: flagged share | 1.000 | 0.186 | 0.006 |

Unsupervised column: the validation optimum is degenerate (flags most benign records, so recall of anything is trivially high); it is not a usable operating point.

## Table 5c - threshold-free metrics

| Metric | Unsupervised | One-class | Supervised |
|---|---:|---:|---:|
| Known set ROC-AUC (benign vs known attacks) | 0.4088 | 0.8788 | 0.9692 |
| Known set PR-AUC (chance 0.0670) | 0.0542 | 0.3191 | 0.9133 |
| Unseen set ROC-AUC (benign vs withheld attacks) | 0.9274 | 0.9162 | 0.6483 |
| Unseen set PR-AUC (chance 0.0598) | 0.2889 | 0.2820 | 0.1601 |
| ROC-AUC Bot vs benign | 0.923 | 0.902 | 0.739 |
| ROC-AUC DDoS-HOIC vs benign | 0.909 | 0.870 | 0.817 |
| ROC-AUC SSH-BruteForce vs benign | 0.953 | 0.951 | 0.341 |
| ROC-AUC DoS-Hulk vs benign | 0.931 | 0.939 | 0.613 |

## Table 5d - the same attack types when they were / were not in the training data

Seen = models of Phases 3-5 (trained with these types present); withheld = Phase 6 models. Flagged share at each model's primary threshold; ROC-AUC against benign. The two sets of models share the architecture and training procedure but differ in their training rows, so this contrast also includes the loss of 45% of the training attacks.

| Type | Model | Flagged, seen | Flagged, withheld | AUC, seen | AUC, withheld |
|---|---|---:|---:|---:|---:|
| Bot | Unsupervised | 0.009 | 0.733 | 0.416 | 0.923 |
| Bot | One-class | 0.001 | 0.001 | 0.887 | 0.902 |
| Bot | Supervised | 0.993 | 0.019 | 0.999 | 0.739 |
| DDoS-HOIC | Unsupervised | 0.003 | 0.863 | 0.590 | 0.909 |
| DDoS-HOIC | One-class | 0.000 | 0.000 | 0.892 | 0.870 |
| DDoS-HOIC | Supervised | 0.995 | 0.000 | 1.000 | 0.817 |
| SSH-BruteForce | Unsupervised | 0.002 | 1.000 | 0.573 | 0.953 |
| SSH-BruteForce | One-class | 0.000 | 0.002 | 0.942 | 0.951 |
| SSH-BruteForce | Supervised | 0.996 | 0.000 | 1.000 | 0.341 |
| DoS-Hulk | Unsupervised | 0.347 | 0.586 | 0.678 | 0.931 |
| DoS-Hulk | One-class | 0.022 | 0.035 | 0.926 | 0.939 |
| DoS-Hulk | Supervised | 0.607 | 0.008 | 0.777 | 0.613 |

## Table 5e - 95% block-bootstrap intervals of the flagged share (primary thresholds)

| Group | Unsupervised | One-class | Supervised |
|---|---:|---:|---:|
| benign | [0.050, 0.129] | [0.008, 0.012] | [0.003, 0.004] |
| known_attacks | [0.023, 0.085] | [0.004, 0.012] | [0.771, 0.919] |
| unseen_all | [0.639, 0.876] | [0.000, 0.026] | [0.004, 0.013] |
| Bot | [0.715, 0.748] | [0.000, 0.002] | [0.017, 0.021] |
| DDoS-HOIC | [0.022, 0.906] | [0.000, 0.000] | [0.000, 0.000] |
| SSH-BruteForce | [1.000, 1.000] | [0.001, 0.003] | [0.000, 0.000] |
| DoS-Hulk | [0.586, 0.586] | [0.035, 0.035] | [0.008, 0.008] |

Intervals resample one-minute test blocks only (not training randomness; one seed per model). For types occupying one or two blocks they are not informative.

