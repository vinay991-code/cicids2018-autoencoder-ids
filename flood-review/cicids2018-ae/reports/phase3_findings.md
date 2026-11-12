# Phase 3 - Experiment A: unsupervised autoencoder (measured results)

Run: `python src/phase3_unsupervised.py` (seed 42, one run each). Architecture 72-128-64-32-64-128-72, ReLU, sigmoid output, Adam lr 1e-3, batch 1024, MSE, max 50 epochs, early-stopping patience 5 on 300k unlabeled validation rows. 1,000,000 randomly sampled training rows per variant (of 7,438,408 all / 6,525,735 benign). Test partition: 2,163,852 rows, 258,132 attacks (11.93%).

An earlier run of this experiment was discarded: a keyword-argument bug set the early-stopping tolerance to 42, so the epoch-1 weights were restored. Fixed (arguments are now keyword-only) and rerun; all numbers below are from the corrected run.

| Metric (test) | A: unsupervised (unlabeled train), label-free p95 threshold | B: one-class (benign-only train), benign-val p99 threshold | B with validation-F1-tuned threshold (uses val labels) |
|---|---:|---:|---:|
| Accuracy | 0.8236 | 0.8558 | 0.9005 |
| Precision | 0.1368 | 0.0348 | 0.5598 |
| Recall | 0.0901 | 0.0078 | 0.7758 |
| F1 | 0.1086 | 0.0128 | 0.6504 |
| Specificity | 0.9230 | 0.9706 | 0.9174 |
| False-positive rate | 0.0770 | 0.0294 | 0.0826 |
| False-negative rate | 0.9099 | 0.9922 | 0.2242 |
| ROC-AUC (threshold-free) | 0.4684 | 0.8894 | 0.8894 |
| PR-AUC (threshold-free; chance = 0.1193) | 0.1090 | 0.4096 | 0.4096 |

Variant A with a validation-F1-tuned threshold degenerates: the optimum is 1e-6, which flags 99.99% of rows (recall 1.000, FPR 0.9999, F1 0.2132 = the trivial "flag everything" score), so it is not reported as a result.

Cost: training 442 s (A) / 376 s (B) for 50 / 50 epochs on CPU; inference on the whole test set 1.7 s / 1.6 s (0.77 / 0.74 ms per 1000 records); peak process working set 5.15 GB (includes memory-mapped data).

## Per-attack-type ROC-AUC against benign (test; diagnostic computed after the run, nothing was tuned on it)

High-volume types: A (unsupervised) LOIC-HTTP 0.286, Bot 0.416, HOIC 0.590, Hulk 0.678, SSH-BF 0.573, Infiltration 0.539. B (one-class) LOIC-HTTP 0.924, Bot 0.887, HOIC 0.892, Hulk 0.926, SSH-BF 0.942, Infiltration 0.535.
Full table: `results/phase3_per_type_auc.json`.

## Observations
- Strictly unsupervised training on contaminated data gives ROC-AUC 0.468 and PR-AUC below the 0.1193 chance level: on this test set, reconstruction error ranks attacks slightly lower than benign traffic. The types with the most training rows (LOIC-HTTP, Bot) have the lowest AUC; the types that were nearly removed by de-duplication (FTP-BruteForce 41 rows, SlowHTTPTest 55) have AUC about 0.94. This pattern is consistent with the autoencoder learning to reconstruct whatever is frequent in its training data, including attacks; it is an interpretation, not tested.
- Benign-only training separates most flood/DoS/brute-force types well (AUC 0.89-0.94) but Infiltration stays near chance (0.535), as expected from Phase 2, where 40% of its rows were identical to benign flows.
- The fixed percentile thresholds perform badly: the p99 benign threshold (0.0139) sits at a cluster of benign flows with very high error (histogram, `results/phase3_score_distributions.png`) above where most attacks lie (log10 error -4 to -2.5), giving recall 0.8%. Benign FPR on test is 2.94% against 1.00% on validation, so the benign error distribution differs between partitions.
- The loss curves show no overfitting (train and validation overlap) but a few sudden steps (epochs 14, 21-22 and 32) typical of a constant learning rate. Variant A had not fully converged at the 50-epoch cap.

## Limitations of this result
Single seed, single run, no hyper-parameter search, 1M-row training samples, no learning-rate schedule. Thresholds were fixed before seeing results. These are in-distribution numbers on a block-wise split (see Phase 2); the supervised comparison (Phase 4-5) uses the same test partition.
