# Phase 4 - Experiment B: supervised autoencoder (measured results)

Run: `python src/phase4_supervised.py` (seed 42, one run each). Encoder 72-128-64-32 (latent), head 32-16-1, Adam lr 1e-3, batch 1024, class-weighted BCE (pos_weight 7.14 = benign/attack in the training rows), early stopping (patience 5) on 300k labeled validation rows. Trained on exactly the same 1,000,000 training rows as Phase 3 variant A (12.28% attacks; sampling verified identical). Evaluated on the same full test partition (2,163,852 rows, 258,132 attacks; label hash stored in `results/phase4_results.json`).
`supervised_recon` adds a decoder and the term `1.0 x summed squared reconstruction error` (weight fixed beforehand, not tuned).
Early stopping behaved correctly: stopped at epoch 27 (best epoch 22) and epoch 33 (best epoch 28).

| Metric (test) | Supervised AE, thr 0.5 | Supervised AE, val-F1 thr | + decoder, thr 0.5 | + decoder, val-F1 thr |
|---|---:|---:|---:|---:|
| Threshold | 0.5000 | 0.8001 | 0.5000 | 0.8460 |
| Accuracy | 0.9802 | 0.9801 | 0.9736 | 0.9838 |
| Precision | 0.9712 | 0.9868 | 0.8623 | 0.9624 |
| Recall | 0.8599 | 0.8448 | 0.9265 | 0.8991 |
| F1 | 0.9122 | 0.9103 | 0.8932 | 0.9297 |
| Specificity | 0.9965 | 0.9985 | 0.9800 | 0.9952 |
| False-positive rate | 0.0035 | 0.0015 | 0.0200 | 0.0048 |
| False-negative rate | 0.1401 | 0.1552 | 0.0735 | 0.1009 |
| ROC-AUC (threshold-free) | 0.9437 | 0.9437 | 0.9796 | 0.9796 |
| PR-AUC (threshold-free; chance 0.1193) | 0.9139 | 0.9139 | 0.9540 | 0.9540 |

Cost: training 313 s (27 epochs) / 750 s (33 epochs); test inference 1.9 s / 4.0 s (0.89 / 1.86 ms per 1000 records). Peak working set 6.36 GB is a process-wide peak (both variants ran in one process, data memory-mapped), so it is not a per-model figure.

## Per-attack-type detection on test (flagged share at threshold 0.5; ROC-AUC against benign)
Large types (>= 18k rows): Bot 0.993/0.999, DDoS-HOIC 0.995/1.000, DDoS-LOIC-HTTP 0.999/1.000, SSH-BruteForce 0.996/1.000, DoS-GoldenEye 0.991/0.999 for the plain classifier; the decoder variant is within 0.004 of those. **DoS-Hulk (45,509 rows): 0.607 / AUC 0.777 for the plain classifier vs 0.980 / 0.992 with the decoder. Infiltration (19,087): 0.081 / 0.772 and 0.080 / 0.748 respectively.**
Types with fewer than 110 test rows (FTP-BruteForce 25, SlowHTTPTest 38, SQLInjection 19, WebXSS 50, WebBruteForce 105) swing between 0 and 1 and are not evidence of anything. Full table: `results/phase4_results.json`.

## Observations
- Both supervised variants far exceed the autoencoders trained without labels in Phase 3 on this test partition (ROC-AUC 0.944 / 0.980 vs 0.889 one-class and 0.468 unsupervised). This is an in-distribution test on a block-wise split; Experiment C will test unseen attacks.
- The plain classifier is best when judged by accuracy and false positives; the decoder variant has higher recall and AUC but at threshold 0.5 raises benign false positives from 0.35% to 2.00% (validation FPR is 0.43%, so the benign mix in the test blocks differs from validation). The val-F1 threshold repairs most of this (FPR 0.48%, F1 0.930). Which is "better" depends on the metric, so the winner is not declared on one number.
- The plain classifier misses most DoS-Hulk test traffic (recall 60.7%) although it detects it on validation, which shows how much a block-wise split can vary by attack type; single-seed results for such types are unstable.
- Infiltration remains almost undetected (about 8% flagged) by every model, consistent with its overlap with benign flows found in Phase 2.

## Limitations
Single seed and run, no hyper-parameter search, 1M-row sample (12.28% attack), threshold 0.5 and the reconstruction weight fixed in advance, in-distribution split. Differences of a point or two between the supervised variants should not be treated as significant.
