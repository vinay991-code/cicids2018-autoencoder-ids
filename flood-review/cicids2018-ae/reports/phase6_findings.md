# Phase 6 - Experiment C: unknown attack detection (measured results)

Run: `python src/phase6_unseen.py`, then `python src/phase6_tables.py` (seed 42, one run per model). Full tables: `reports/phase6_tables.md`; data: `results/phase6_results.json`; figure: `results/phase6_unseen_detection.png`.

**Withheld types (Table 5 "unseen")**: Bot, DDoS-HOIC, SSH-BruteForce, DoS-Hulk. Rule, fixed before running: attack types with at least 10,000 test rows, covering distinct families, excluding Infiltration (overlaps benign traffic) and LOIC-HTTP (43% of training attacks). I had already seen the Phase 3-5 per-type results, so the choice is not blind; it rests on family structure and size. Two have a related type still in training (HOIC with LOIC-HTTP, Hulk with GoldenEye/Slowloris); Bot and SSH have none (FTP-BruteForce, SSH's only sibling, has 15 training rows).

**What the withheld types could not influence**: the scaler (refit on the remaining 7,022,813 training rows; min/max came out identical to Phase 2, so the refit changed nothing), the training rows (415,595 attack rows removed, 45.5% of training attacks), early stopping, and the validation rows used for thresholds (1,127,988 rows remain). Thresholds use the same rules as Phases 3/4 on the filtered validation set. Test partition unchanged: 1,905,720 benign + 136,861 known-attack + 121,271 withheld-attack records.

## Table 5 - known vs unseen attacks (primary thresholds)

| Metric | Unsupervised AE | One-class AE | Supervised AE |
|---|---:|---:|---:|
| Known attacks: recall | 0.0461 | 0.0067 | 0.8704 |
| Known set: F1 | 0.0436 | 0.0117 | 0.9096 |
| Benign: false-positive rate | 0.0768 | 0.0098 | 0.0031 |
| Unseen attacks: recall | 0.7489 | 0.0135 | 0.0077 |
| Unseen set: precision / F1 | 0.3829 / 0.5067 | 0.0807 / 0.0231 | 0.1357 / 0.0145 |
| Known set ROC-AUC | 0.4088 | 0.8788 | 0.9692 |
| Unseen set ROC-AUC (benign vs withheld) | 0.9274 | 0.9162 | 0.6483 |
| Unseen set PR-AUC (chance 0.0598) | 0.2889 | 0.2820 | 0.1601 |
| Unseen flagged share: Bot / HOIC / SSH / Hulk | 0.733 / 0.863 / 1.000 / 0.586 | 0.001 / 0.000 / 0.002 / 0.035 | 0.019 / 0.000 / 0.000 / 0.008 |

Primary thresholds: unsupervised = 95th percentile of filtered validation errors (no labels); one-class = 99th percentile of benign validation errors; supervised = probability 0.5. Validation-F1-tuned thresholds (Table 5b) change the one-class unseen recall to 0.233 (SSH 1.000, Bot 0.025, HOIC 0.013, Hulk 0.186) and leave the supervised model at 0.002; the unsupervised optimum is degenerate (flags 91% of benign records) and is not a usable operating point.

## Findings
1. **The supervised classifier does not generalise to attack types it has not seen**: unseen recall 0.8% and ROC-AUC 0.648, against recall 87% on the known attacks. For SSH-BruteForce its AUC is 0.341, below chance: it scores those flows as more benign than the average benign flow.
2. **The unsupervised autoencoder is the only model that flags most unseen attacks at its fixed threshold** (recall 74.9%, ROC-AUC 0.927), but at 7.7% benign false alarms (unseen-set precision 0.38 at a 6.0% attack share). On the same test set its known-attack AUC is 0.409.
3. **Seen versus withheld (Table 5d) is the clearest result.** For the same four types, the unsupervised AE's ROC-AUC against benign rises from 0.42-0.68 when the types were in its training data to 0.91-0.95 when withheld (Bot 0.416 to 0.923, HOIC 0.590 to 0.909, SSH 0.573 to 0.953, Hulk 0.678 to 0.931). The supervised AE goes the other way (0.78-1.00 to 0.34-0.82). The one-class AE barely changes (0.87-0.94 either way), as expected since it never saw attacks. This supports the Phase 5 explanation that an autoencoder trained on unlabeled data reconstructs attack traffic it is exposed to. It is one seed, and the withheld models also trained on 45% fewer attacks (7.1% vs 12.3% of the training sample), so the size of the effect is not isolated.
4. **A high AUC is not a usable detector**: the one-class AE has ROC-AUC 0.916 on unseen attacks but flags 1.4% of them at its p99 threshold, because the benign tail is higher than most attack errors (Phase 3). With the validation-F1 threshold it flags 23%, mainly SSH.

## Limitations of reconstruction-error detection (what this experiment shows)
- **Training-data contamination**: any attack present in the unlabeled training set is learned and then scores low; the strictly unsupervised model works on unseen types only because they were removed from its training data, which a real deployment cannot guarantee.
- **Error measures unfamiliarity, not maliciousness**: some benign traffic (about 1-5% of benign flows) is unfamiliar too, which floors the false-alarm rate; the one-class model reaches 95% unseen recall only near 8-10% FPR (right panel of the figure), i.e. on the order of 150,000-190,000 benign flows flagged in this test set.
- **Threshold dependence**: the same scores give 1.4% or 23% unseen recall depending on the rule, and the label-free p95 rule fixes the alarm rate regardless of how many attacks exist.
- **Attacks that look like ordinary traffic** reconstruct well (Infiltration, Phase 5).
- Reconstruction scoring is not a fix for supervised models' blind spot unless its false alarms are acceptable: no model here has both low false alarms and good unseen recall.

## Limitations of this experiment
One seed; one withheld set. Each withheld type is a single attack episode concentrated in few one-minute test blocks (Bot 68 blocks, SSH 18, HOIC 5, **Hulk 1**), so per-type numbers (especially Hulk and HOIC) describe one episode and their block-bootstrap intervals are not meaningful (Table 5e). Results may differ for other held-out types or seeds. The known-attack evaluation excludes the withheld types, so its numbers differ from Phase 5.
