# Phase 5 - supervised vs unsupervised autoencoder: comparison

All numbers are recomputed from the saved test scores of the four models and checked against the Phase 3 / 4 results (confusion counts and ROC-AUC match exactly). Full tables: `reports/phase5_tables.md`; data: `results/phase5_comparison.json`. Test partition: 2,163,852 records, 258,132 attacks (11.93%), identical for every model. Each model was trained once (seed 42).

## Table 1 - the requested comparison (primary thresholds fixed before seeing results)

| Metric | Unsupervised AE (no labels) | Supervised AE |
|---|---:|---:|
| Accuracy | 0.8236 | 0.9802 |
| Precision | 0.1368 | 0.9712 |
| Recall | 0.0901 | 0.8599 |
| F1-score | 0.1086 | 0.9122 |
| False-positive rate | 0.0770 | 0.0035 |
| ROC-AUC | 0.4684 | 0.9437 |
| PR-AUC (chance 0.1193) | 0.1090 | 0.9139 |

The unsupervised column uses a label-free threshold (95th percentile of validation errors); the supervised column uses probability 0.5. Two further models complete the picture: the one-class AE (trained on benign rows only, ROC-AUC 0.889, PR-AUC 0.410) and the supervised AE with a decoder (ROC-AUC 0.980, PR-AUC 0.954); see `phase5_tables.md` Tables 1a-1g, which also give 95% block-bootstrap intervals, validation-tuned thresholds, cost and per-attack-type results.

Figures: `results/phase5_roc_pr_comparison.png`, `phase5_confusion_unsup_vs_sup.png`, `phase5_confusion_all_models.png`, `phase5_loss_curves_all.png`.

## Which model is better? It depends on the criterion

| Criterion | Result |
|---|---|
| Detection quality (ROC-AUC, PR-AUC, F1) | Both supervised models are far ahead of both autoencoders trained without attack labels. Paired difference supervised - one-class: ROC-AUC +0.060 [0.037, 0.080], PR-AUC +0.492 [0.420, 0.544]; one-class - unsupervised: ROC-AUC +0.432 [0.343, 0.522]. |
| Supervised vs supervised + decoder | **Not distinguishable.** ROC-AUC difference +0.028 [-0.0003, 0.082] and PR-AUC +0.032 [-0.0002, 0.090]; the decoder variant was ahead in 75% / 92% of resamples. It has higher recall (0.926 vs 0.860) but a higher false-positive rate (0.0200 vs 0.0035) at threshold 0.5. With validation-tuned thresholds the decoder variant has the higher F1 (0.930 vs 0.910). |
| False alarms | Plain supervised AE: FPR 0.35% at threshold 0.5 (interval 0.3-0.4%). The decoder variant's FPR varies with the test blocks (interval 0.4-5.1%). |
| Detection at low false-alarm rate (recall at 1% FPR) | Supervised 0.869, supervised + decoder 0.910, one-class 0.005, unsupervised 0.008. The one-class model's AUC of 0.889 does not translate to a usable operating point at low FPR (about 5% of benign flows score above nearly every attack; see the ROC panel). |
| Needs labels | Unsupervised: none. One-class: benign labels (and its threshold used benign validation labels). Supervised: all labels. |
| Cost (CPU) | Parameters 39,400 / 39,400 / 20,225 / 39,945 (unsupervised / one-class / supervised / supervised + decoder); epochs run 50 / 50 / 27 / 33. Measured wall-clock times (training 442 / 376 / 313 / 750 s; inference 0.74-0.89 ms per 1000 records, 1.85 ms for the decoder variant) are single runs on a shared CPU and **not reliable for ranking models**: the plain supervised AE needed 313 s in Phase 4 but 86 s in Phase 6 for a similar number of epochs. All models are small and fast enough for real-time scoring at this scale. |

No single metric decides this: accuracy is 0.82-0.98 for all four models mostly because 88% of records are benign, and the strictly unsupervised model scores 0.82 accuracy while detecting 9% of attacks.

## Why the unsupervised model fails here (explanations, with the evidence for each)
1. **Attacks in the training data get reconstructed.** Variants A and B use the same architecture, optimiser and the same sampling procedure; the only difference is that A's training rows include 12% attacks. ROC-AUC rises from 0.468 to 0.889 when they are excluded. In A, attack types with many training rows have the lowest AUC (LOIC-HTTP 0.286, Bot 0.416) and types reduced to a few dozen rows by de-duplication have the highest (about 0.94). This is evidence for contamination, though not a controlled test of the mechanism.
2. **Reconstruction error measures unfamiliarity, not maliciousness.** Even with benign-only training, about 5% of benign flows get higher error than almost every attack, so the ranking is capped (recall 0.5% at 1% FPR). The error distributions (`phase3_score_distributions.png`) show attacks in a mid-range band and a benign tail above it.
3. **Supervision uses the labels directly**, so the classifier can separate classes along directions that carry little reconstruction error.
The interval for the unsupervised ROC-AUC (0.389-0.528) includes 0.5: the data are consistent with no ranking ability, not proof that it is worse than chance.

## What is hard for every model
Infiltration: flagged 3-12% by all models, ROC-AUC 0.54-0.77; Phase 2 found 40% of its rows identical to benign flows. DoS-Hulk is detected well by the decoder variant (98%) but only 61% by the plain classifier, although the plain classifier did well on validation: results for one attack type depend on the blocks sampled. Types with fewer than 110 test rows (marked (*) in Table 1g) are noise.

## Limitations
- One seed per model; the bootstrap intervals only reflect which one-minute test blocks are included, not training randomness. Several intervals are wide (supervised ROC-AUC 0.897-0.984).
- In-distribution split (random blocks): neighbouring blocks of the same attack are similar, so supervised numbers are optimistic. Phase 6 tests attack types never seen in training.
- The thresholding rules for the unsupervised family were fixed in advance and perform poorly (recall <10%); validation-tuned thresholds help only for the one-class model (F1 0.650), not the unsupervised one (degenerate optimum).
- Only 1M of 7.4M training rows were used, with no hyper-parameter search and no learning-rate schedule.
- Peak memory could not be measured per model (process-wide peaks of 5.15 and 6.36 GB include memory-mapped data); model size is reported through parameter counts and file size instead.
- The loss histories were recovered from the logged training output (verified against saved best-validation losses and epoch counts), not from files written during training.
