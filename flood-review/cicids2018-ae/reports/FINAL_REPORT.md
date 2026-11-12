# Comparative evaluation of supervised and unsupervised autoencoders for network anomaly detection on CSE-CIC-IDS2018

**Final results report.** All experiments below were executed; every number in the tables is read from the saved result files and was re-derived independently by `src/verify_results.py` (`reports/verification.md`). All models were trained once (seed 42), so differences between close results should not be over-read.

## 1. Summary: which approach performs better?

The answer depends on what is asked of the detector, and no single number settles it (accuracy is 0.82-0.98 for every model because 88% of the test records are benign).

1. **Attack types seen in training: the supervised autoencoder is clearly better.** On the identical test partition (2,163,852 records) its ROC-AUC is 0.944 (supervised + decoder 0.980), against 0.889 for the autoencoder trained on benign data only and 0.468 for the strictly unsupervised one; F1 at the pre-fixed thresholds is 0.912 vs 0.109 (one-class: 0.013).
2. **Attack types never seen in training: the picture reverses.** With four attack types withheld from every fitted or tuned step, the supervised model flags 0.8% of them (ROC-AUC 0.648) while the unsupervised autoencoder flags 74.9% (ROC-AUC 0.927), at a benign false-positive rate of 7.7%. The supervised model keeps recall 0.87 on known attacks; the unsupervised one 0.05.
3. **Why the unsupervised model fails on known attacks:** it reconstructs the attack traffic it is trained on. The same attack types that it detects poorly when present in its training data (ROC-AUC against benign 0.42-0.68) are detected well when withheld (0.91-0.95).
4. **Reconstruction error is a weak anomaly score on its own:** even with benign-only training the error ranks attacks well (ROC-AUC 0.889) but recall at 1% false positives is only 0.005, because a few percent of ordinary benign flows score higher than most attacks.
5. **Adapting the baseline did not reliably help.** Static, adaptive-threshold and adaptive-baseline detectors were compared on a 9,504,036-record stream (Tables 2-4): the adaptive threshold rule was unstable, retraining helped where benign traffic shifted (2 of 8 days) and hurt where large attacks dominated, and a trusted buffer filtered on the model's own score did not keep attacks out (12.3% attacks in the buffer against 12.2% in the stream).
6. **Confidence-aware updates (margin, temporal consistency, quarantine) gave no clear advantage over unrestricted adaptation;** a margin-only filter protected the buffer but could not adapt to a real shift, and a persistence rule cannot tell a lasting benign change from a long attack campaign.
7. **Drift monitoring (ADWIN) detected all 4 of 4 benign change points** (median delay 117,500 records) with 10 false alarms among 20 alarms, and drift-triggered retraining beat scheduled retraining at the same false-positive rate in this run, which the evidence attributes mostly to timing (it did not retrain during the Bot campaign), not to detecting drift on that day.
8. **A hybrid is suggested, not tested:** a supervised classifier for known attacks plus a novelty detector trained only on verified-clean data would combine the two strengths seen above. No experiment here evaluated that combination.

## 2. Dataset inspection (CSE-CIC-IDS2018, processed CICFlowMeter CSVs)

Source: official bucket listed on https://www.unb.ca/cic/datasets/ids-2018.html (10 daily CSVs, 6.41 GB, all verified by byte size). Full report: `reports/phase1_inspection.md`, `reports/phase1_findings.md`.

- **Records:** 16,232,943 (after dropping 59 header rows repeated inside three files). **Features:** 80 columns in 9 files, 84 in one (adds Flow ID, Src/Dst IP, Src Port); 78 feature columns are common to all files. **Benign:** 13,484,708 (83.07%); **attacks:** 2,748,235 in 14 categories.
- **Missing / infinite values:** `flow_byts_s` (59,721 NaN, 36,039 infinite), `flow_pkts_s` (0 NaN, 95,760 infinite). **Constant columns:** 8. **Exact duplicate rows:** 433,193; **duplicate feature vectors:** 4,251,152.
- **Potential leakage / data-quality issues found:** (i) **7 of the 10 files contain exactly 1,048,575 data lines (one of them 1,048,574 records plus a repeated header line), the spreadsheet row limit minus the header, which strongly suggests truncation** (what is missing cannot be seen); (ii) **timestamps are on a 12-hour clock without AM/PM** (found in Phase 7: afternoon events appear 12 hours early; verified against the official attack schedule, e.g. DoS-Hulk scheduled 13:45 is stored as 01:45), so time-of-day cannot be used directly and the corrected time was used only for the streaming experiments; (iii) rows within a file are not in time order; (iv) 14 rows carry January-1970 timestamps; (v) identifiers (Flow ID, IPs, Src Port) exist only in the 20-Feb file and were excluded; (vi) `Dst Port` alone almost identifies several attack types (e.g. 100% of FTP-BruteForce on port 21, 0.00% of benign) and was excluded from the model input.

| Label | Records | Share |
|---|---|---|
| Benign | 13,484,708 | 83.070% |
| DDOS attack-HOIC | 686,012 | 4.226% |
| DDoS attacks-LOIC-HTTP | 576,191 | 3.550% |
| DoS attacks-Hulk | 461,912 | 2.846% |
| Bot | 286,191 | 1.763% |
| FTP-BruteForce | 193,360 | 1.191% |
| SSH-Bruteforce | 187,589 | 1.156% |
| Infilteration | 161,934 | 0.998% |
| DoS attacks-SlowHTTPTest | 139,890 | 0.862% |
| DoS attacks-GoldenEye | 41,508 | 0.256% |
| DoS attacks-Slowloris | 10,990 | 0.068% |
| DDOS attack-LOIC-UDP | 1,730 | 0.011% |
| Brute Force -Web | 611 | 0.004% |
| Brute Force -XSS | 230 | 0.001% |
| SQL Injection | 87 | 0.001% |

## 3. Preprocessing and split

- **Cleaning and de-duplication:** 16,232,943 records read -> 16,137,168 after cleaning (removed: nan or inf 95,760, negative value 1, bad or non2018 timestamp 14) -> **10,775,819** after removing identical feature vectors (4,944,797) and vectors seen with both a benign and an attack label (416,552). De-duplication shrinks some attack types to a few dozen unique flows (FTP-BruteForce 193,360 -> 41; DoS-SlowHTTPTest 139,890 -> 55).
- **Model input:** 72 columns: 68 numeric features after a signed log1p transform and min-max scaling to [0, 1] fitted on the training partition only, plus a one-hot protocol (4 columns: 0, 6, 17, other); constant columns (8), `Dst Port`, timestamp, label and identifiers are excluded. Original labels are kept separately; the binary label is 0 = benign, 1 = attack. No clipping of validation/test values.
- **Split (scenario-aware by time block):** every capture day is cut into 60-second blocks on the stored clock; whole blocks go to train / validation / test (70/10/20% of blocks) stratified by the attack types a block contains, so no block straddles partitions and the partitions share no identical feature vector. The in-distribution test is therefore optimistic (neighbouring blocks of one attack are alike).

| Partition | Records | Benign | Attacks | Attack share |
|---|---|---|---|---|
| train | 7,438,408 | 6,525,735 | 912,673 | 12.27% |
| val | 1,173,559 | 1,040,160 | 133,399 | 11.37% |
| test | 2,163,852 | 1,905,720 | 258,132 | 11.93% |

## 4. Results

### Table 1 - Supervised vs unsupervised autoencoder (test partition, identical for all models; thresholds fixed beforehand)

| Metric | Unsupervised AE | Supervised AE |
|---|---|---|
| Accuracy | 0.8236 | 0.9802 |
| Precision | 0.1368 | 0.9712 |
| Recall | 0.0901 | 0.8599 |
| F1-score | 0.1086 | 0.9122 |
| Specificity | 0.9230 | 0.9965 |
| False-positive rate | 0.0770 | 0.0035 |
| False-negative rate | 0.9099 | 0.1401 |
| ROC-AUC | 0.4684 | 0.9437 |
| PR-AUC (chance 0.1193) | 0.1090 | 0.9139 |
| Epochs / parameters | 50 / 39,400 | 27 / 20,225 |

Unsupervised AE: trained on unlabeled records, threshold = 95th percentile of validation errors (no labels). Supervised AE: class-weighted binary cross-entropy, threshold 0.5. Training and inference times were measured but are single runs on a shared CPU and vary about 3x between identical runs, so they are not used to rank models (inference was 0.7-1.9 ms per 1000 records for all four; all models are small, 20,225-39,945 parameters). Peak memory could not be separated per model.

**All four trained models** (same test set; thresholds fixed before seeing results; 95% intervals are block bootstraps over test blocks and do not include training randomness):

| Metric | Unsupervised (unlabeled train) | One-class (benign-only train) | Supervised | Supervised + decoder |
|---|---|---|---|---|
| Accuracy | 0.8236 | 0.8558 | 0.9802 | 0.9736 |
| Precision | 0.1368 | 0.0348 | 0.9712 | 0.8623 |
| Recall | 0.0901 | 0.0078 | 0.8599 | 0.9265 |
| F1 | 0.1086 | 0.0128 | 0.9122 | 0.8932 |
| False-positive rate | 0.0770 | 0.0294 | 0.0035 | 0.0200 |
| ROC-AUC [95% CI] | 0.468 [0.389, 0.528] | 0.889 [0.851, 0.916] | 0.944 [0.897, 0.984] | 0.980 [0.968, 0.986] |
| PR-AUC [95% CI] | 0.109 [0.067, 0.149] | 0.410 [0.326, 0.545] | 0.914 [0.855, 0.967] | 0.954 [0.917, 0.970] |
| Recall at 1% FPR [95% CI] | 0.008 [0.001, 0.016] | 0.005 [0.003, 0.008] | 0.869 [0.787, 0.941] | 0.910 [0.865, 0.942] |
| F1 with validation-tuned threshold | degenerate (flags 99.99%) | 0.6504 | 0.9103 | 0.9297 |

Paired differences (same resamples): supervised - one-class ROC-AUC +0.060 [+0.037, +0.080]; supervised + decoder - supervised ROC-AUC +0.028 [-0.000, +0.082] (not distinguishable); one-class - unsupervised ROC-AUC +0.432 [+0.343, +0.522].

Figures: `results/phase5_roc_pr_comparison.png`, `phase5_confusion_unsup_vs_sup.png`, `phase5_confusion_all_models.png`, `phase5_loss_curves_all.png`. Details: `reports/phase3_findings.md`, `phase4_findings.md`, `phase5_findings.md`, `phase5_tables.md`.

### Table 2 - Adaptive experiments: static vs adaptive threshold vs adaptive baseline (Experiment D, Phase 7)

Time-ordered stream of 9,504,036 records (1,158,860 attacks, 8 days) after a warm-up on the first two days; update interval 50,000 records; threshold = 99th percentile of warm-up benign errors; B re-estimates the threshold each interval with a robust median + k x MAD rule; C adds fine-tuning on a rolling trusted buffer (records it scored as normal) with replay and an anchor check. Attack labels never controlled an update.

| Metric | A static | B adaptive threshold | C adaptive baseline |
|---|---|---|---|
| Precision | 0.1347 | 0.0348 | 0.1002 |
| Recall | 0.0136 | 0.0041 | 0.0115 |
| F1-score | 0.0247 | 0.0074 | 0.0206 |
| False-positive rate | 0.0122 | 0.0159 | 0.0143 |
| Threshold minimum / maximum | 0.0139 / 0.0139 | 0.00141 / 1.66e+03 | 5.7e-05 / 0.134 |
| Intervals with threshold above 0.072 (above any error seen in Phase 3) | 0.0% | 25.8% | 4.6% |
| Retrainings (accepted) | 0 | 0 | 64 (64) |
| Difference to A in F1 [95% CI] | - | -0.0157 [-0.0399, -0.0011] | -0.0031 [-0.0077, +0.0001] |

Result: adaptation did not improve detection. The static detector is already weak (the threshold sits at the benign tail); the adaptive-threshold rule is unstable (maximum 1.66e+03); retraining raised per-day ranking quality on the two days whose benign traffic shifted (02-16 ROC-AUC 0.079 -> 0.472; 02-21 0.013 -> 0.745) and lowered it on days dominated by large attacks (02-20 0.948 -> 0.409); the trusted buffer held 12.3% attacks on average (stream 12.2%). Figure: `results/phase7_adaptive_overview.png`; details `reports/phase7_findings.md`.

### Table 3 - Confidence-aware experiments: unrestricted vs confidence-gated adaptation (Experiment E, Phase 8)

Same stream and warm-up; each arm's threshold = 95th percentile of a fixed clean anchor set under the arm's current model (replaces Phase 7's unstable rule). U: every sampled record enters the retraining buffer. G: margin (score <= 0.5 x threshold) + quarantine of records above the threshold + temporal consistency (a deviating interval is held and released only if the deviation persists for 3 intervals). M: margin only (ablation). S: static reference. Reconstruction error is treated as an ordinal score, not a calibrated probability.

| Metric | S static | U unrestricted | M margin only | G confidence-gated |
|---|---|---|---|---|
| False-positive rate | 0.1602 | 0.0665 | 0.1541 | 0.0835 |
| Recall | 0.7035 | 0.1416 | 0.7017 | 0.2100 |
| F1-score | 0.4925 | 0.1748 | 0.4991 | 0.2319 |
| Mean attack share in the retraining buffer (stream 0.122) | - | 0.123 | 0.038 | 0.128 |
| Maximum attack share in a buffer | - | 0.490 | 0.461 | 0.513 |
| Recovery after the 02-16 / 02-21 shift (intervals to FPR <= 10%) | never / never | 5 / 7 | never / never | 5 / 9 |
| 02-20 (LOIC-HTTP day) F1 / recall / FPR | 0.743 / 0.941 / 0.070 | 0.083 / 0.061 / 0.050 | 0.754 / 0.941 / 0.066 | 0.156 / 0.126 / 0.058 |
| LOIC-HTTP flagged over the stream | 0.941 | 0.061 | 0.941 | 0.126 |

Overall figures for S and M are inflated: on 02-16 and 02-21 the benign traffic is shifted entirely above their threshold, so they flag 90-94% of all records there (false-positive rate 99.6% and 99.9%). G minus U (block bootstrap over intervals): F1 +0.059 [-0.013, +0.131], recall +0.074 [-0.008, +0.169], false-positive rate +0.018 [+0.004, +0.044]. Result: the gated design gave no clear advantage and did not lower contamination; only margin-only protected the buffer, and only by never adapting. Persistence sensitivity of G (3 / 6 / 12 intervals): LOIC-HTTP flagged 0.126 / 0.246 / 0.485; 02-16 false-positive rate 0.246 / 0.486 / 0.996. Figure: `results/phase8_confidence_overview.png`; details `reports/phase8_findings.md`.

### Table 4 - Concept-drift experiments: drift detection approaches and recovery (Experiment F, Phase 9)

Drift monitor: 8 ADWIN detectors (river) on a fixed sentinel model's error distribution and flagged share, five traffic features and the UDP share, batches of 5,000 records, pre-fed with warm-up data. A: no detection, no updates. A2: scheduled retraining (reference). B: drift detection + adaptive baseline. C: drift detection + confidence-aware updates (persistence check + exclusion of the highest-scoring 20% of the recent samples). Test labels controlled neither the detector nor any update.

| Metric | A no detection | A2 periodic (ref.) | B drift-triggered | C drift + confirmation |
|---|---|---|---|---|
| False-positive rate | 0.1602 | 0.0666 | 0.0694 | 0.0922 |
| Recall | 0.7035 | 0.1410 | 0.3071 | 0.3830 |
| F1-score | 0.4925 | 0.1741 | 0.3399 | 0.3742 |
| Retrainings | 0 | 64 | 45 | 15 |
| Mean attack share in the retraining data | - | 0.123 | 0.150 | 0.321 |
| Recovery after the 02-16 / 02-21 shift (intervals to FPR <= 10%) | never / never | 5 / 7 | 6 / 6 | 7 / 7 |
| 03-02 (Bot day) F1 / recall | 0.062 / 0.040 | 0.083 / 0.054 | 0.848 / 0.912 | 0.889 / 0.992 |

**Drift detection (B and C):** 4 of 4 true benign change points detected; delay after the change 190,000, 125,000, 55,000, 110,000 records (median 117,500); **10 false alarms among 20 alarms** (1.05 per million records; 3 in attack-dominated batches). B minus A2: F1 +0.172 [+0.016, +0.345], false-positive rate +0.001 [-0.014, +0.012]; C minus B: F1 +0.036 [-0.016, +0.087], false-positive rate +0.022 [+0.012, +0.036]. The true change points are four jumps between days defined from the static model's behaviour and the labels (evaluation only), so detection delay and false-alarm figures rest on four events; F1 degradation and recovery could not be assessed meaningfully (the before-change F1 is missing or an artefact of flagging everything), so recovery is reported through the false-positive rate. Figure: `results/phase9_drift_overview.png`; details `reports/phase9_findings.md`.

### Table 5 - Unknown attack experiments: known vs unseen attack categories (Experiment C, Phase 6)

Withheld attack types: **Bot, DDoS-HOIC, SSH-BruteForce, DoS-Hulk** (rule fixed beforehand: at least 10,000 test rows, distinct families, excluding Infiltration and LOIC-HTTP; I had seen earlier per-type results, so the choice is not blind). They were removed from the scaler fit, training, early stopping and the validation rows used for thresholds (415,595 training attack rows removed). Test: 1,905,720 benign, 136,861 known-attack, 121,271 withheld-attack records.

| Metric | Unsupervised AE | One-class AE | Supervised AE |
|---|---|---|---|
| Known attacks: recall | 0.0461 | 0.0067 | 0.8704 |
| Known set: F1 | 0.0436 | 0.0117 | 0.9096 |
| Benign: false-positive rate | 0.0768 | 0.0098 | 0.0031 |
| **Unseen attacks: recall** | 0.7489 | 0.0135 | 0.0077 |
| Unseen set: precision | 0.3829 | 0.0807 | 0.1357 |
| Unseen Bot: flagged share | 0.733 | 0.001 | 0.019 |
| Unseen DDoS-HOIC: flagged share | 0.863 | 0.000 | 0.000 |
| Unseen SSH-BruteForce: flagged share | 1.000 | 0.002 | 0.000 |
| Unseen DoS-Hulk: flagged share | 0.586 | 0.035 | 0.008 |
| Known set ROC-AUC | 0.4088 | 0.8788 | 0.9692 |
| Unseen set ROC-AUC (benign vs withheld) | 0.9274 | 0.9162 | 0.6483 |
| Unseen set PR-AUC (chance 0.0598) | 0.2889 | 0.2820 | 0.1601 |

The same attack types when they were in the training data (Phases 3-5) versus withheld (ROC-AUC against benign; unsupervised / one-class / supervised):

| Type (seen -> withheld) | Unsupervised AE | One-class AE | Supervised AE |
|---|---|---|---|
| Bot | 0.416 -> 0.923 | 0.887 -> 0.902 | 0.999 -> 0.739 |
| DDoS-HOIC | 0.590 -> 0.909 | 0.892 -> 0.870 | 1.000 -> 0.817 |
| SSH-BruteForce | 0.573 -> 0.953 | 0.942 -> 0.951 | 1.000 -> 0.341 |
| DoS-Hulk | 0.678 -> 0.931 | 0.926 -> 0.939 | 0.777 -> 0.613 |

The seen-versus-withheld models also differ in training size (45% fewer attack rows), and each withheld type is a single attack episode in few one-minute test blocks (Bot 68, SSH 18, HOIC 5, Hulk 1), so per-type numbers describe one episode. Limitations of reconstruction-error detection shown here: training-data contamination (any attack present in unlabeled training data is learned), error measures unfamiliarity rather than maliciousness (a few percent of benign flows score above most attacks, flooring the false-alarm rate), strong dependence on the threshold rule (the same one-class scores give 1.4% or 23% unseen recall), and attacks that look like ordinary traffic reconstruct well. Figure: `results/phase6_unseen_detection.png`; details `reports/phase6_findings.md`, `phase6_tables.md`.

## 5. Figures

- `results/phase3_loss_curves.png` - Experiment A loss curves (invalid-run fix noted in section 8)
- `results/phase3_confusion_matrices.png` - Experiment A confusion matrices
- `results/phase3_score_distributions.png` - Reconstruction-error distributions by class
- `results/phase4_loss_curves.png` - Experiment B loss curves
- `results/phase4_confusion_matrices.png` - Experiment B confusion matrices
- `results/phase4_probability_distributions.png` - Supervised predicted probabilities
- `results/phase5_roc_pr_comparison.png` - ROC and precision-recall curves, all four models
- `results/phase5_confusion_unsup_vs_sup.png` - Confusion matrices, unsupervised vs supervised
- `results/phase5_loss_curves_all.png` - Training and validation loss, all models
- `results/phase6_unseen_detection.png` - Unknown-attack detection
- `results/phase7_adaptive_overview.png` - Adaptive baseline overview
- `results/phase8_confidence_overview.png` - Confidence-aware updates overview
- `results/phase9_drift_overview.png` - Drift monitoring overview

## 6. Research observations

- **Supervised and unsupervised learning solve different problems.** A supervised autoencoder classifier learns the boundary between the classes it is shown, which is why it dominates when the test attacks resemble training attacks and fails (recall under 1%) on new ones. A reconstruction-based detector has no notion of the attack boundary; it measures unfamiliarity, which is what makes it able to flag unseen attacks and also what makes it flag unusual benign traffic.
- **Training-data purity is the dominant factor for the unsupervised family.** Removing 12% attacks from the training data changes the same architecture from near-chance (ROC-AUC 0.468) to 0.889; withholding specific attack types raises their detectability from 0.42-0.68 to 0.91-0.95. In the streaming experiments the same mechanism appears as buffer contamination, and in the one Phase 9 run the arm that retrained only on attack-free periods did best on the Bot day (a timing effect observed once, not a general result).
- **Operating point matters as much as ranking quality.** Several models rank well (high AUC) but flag few attacks at a threshold chosen from benign statistics, because the benign error distribution has a heavy tail. Validation-tuned thresholds change results by an order of magnitude and one of them is degenerate.
- **Label-free safeguards based on the model's own score do not separate attacks from benign change.** A margin filter, a persistence rule, a relative cut and a drift detector were each tried; none distinguishes a lasting benign shift from a long attack campaign, and the relative cut increased contamination.
- **Distribution shift between days was large for two days** (benign errors under the warm-up model one to two orders of magnitude higher), which the static detector cannot handle (it flagged 99.6-99.9% of benign records), and which retraining repaired in 5-7 intervals.

## 7. Limitations and threats to validity

- **One training run per model (seed 42).** Intervals shown describe sampling variation across test blocks or the stream, not training randomness; close results (for example the two supervised variants) are not distinguishable.
- **Data quality of the source files:** 7 of 10 files appear truncated at the spreadsheet row limit; timestamps lack AM/PM; de-duplication removed 33% of the cleaned records and reduced some attack types to a few dozen unique flows (per-type results for FTP-BruteForce, DoS-SlowHTTPTest, SQL injection, XSS and web brute force rest on under 110 test rows each).
- **In-distribution split is optimistic:** blocks of the same attack campaign are similar, and some flows span block boundaries. Only the unknown-attack experiment tests generalisation, with four withheld types, each a single episode.
- **Operating points:** thresholds were fixed in advance and perform poorly for the unsupervised family. Phases 3 and 6 used the 95th percentile of validation errors (unsupervised model) and the 99th percentile of benign validation errors (one-class model); Phase 7 the 99th percentile of warm-up benign errors; Phases 8-9 the 95th percentile of a clean reference set under the current model. These choices change recall strongly (Phase 7's 99th-percentile threshold sat at the benign tail and gave recall near 1%). Validation-tuned thresholds use validation labels.
- **Streaming experiments (Phases 7-9) rest on a stream with day-level changes only:** two days are single-hour snapshots, one day (02-20) is 57% of the records and one attack campaign, two days have almost no attacks, and the 'drift' events are four jumps between days defined from the static model's behaviour and the labels (evaluation only). Gradual drift was not tested. Design choices (gate parameters, response length, ADWIN settings, replay share) were fixed in advance and not tuned, so other designs may behave differently.
- **Preprocessing statistics:** the scaler was fitted on the Phase 2 training partition, which contains records from all days, so in the streaming experiments the min/max statistics (not the model weights) saw later days.
- **Cost measurements are unreliable:** wall-clock times vary about 3x between identical runs on this shared CPU and process-wide memory cannot be attributed to a model; only parameter counts and epochs are dependable.
- **Overall F1/recall of arms that flag nearly everything on shifted days are inflated** (Tables 3 and 4); false-positive rate and per-day tables should be read alongside.
- **No real-time deployment, packet capture, dashboard or live evaluation was built or tested,** as scoped.

## 8. Corrections and disclosures made during the project

- **Phase 3 first run was invalid and discarded:** an argument-order bug set the early-stopping tolerance to 42, so epoch-1 weights were restored; the training functions now take keyword-only arguments and all Phase 3 numbers come from the corrected rerun.
- **Timestamps (found in Phase 7):** the Phase 1 'unexplained' hours 01-12 are a 12-hour clock without AM/PM; notes were added to the Phase 1 and Phase 2 reports. Phases 3-6 are unaffected (blocks are never split across partitions); the stream for Phases 7-9 uses corrected time.
- **Phase 5 timing caveat:** training times differed 3x between identical runs (313 s in Phase 4 vs 86 s in Phase 6), so the cost comparison was withdrawn as a ranking criterion.
- **Phase 7 threshold rule failed** (unstable median + k x MAD); Phase 8 and 9 use an anchor-based threshold chosen in response, which also changes the operating point from the 99th to the 95th percentile of clean errors. Phase 7's result is reported as run.
- **Phase 6:** the scaler refit without the withheld types left every min/max unchanged, so it did not influence results.
- **Hold-out selection (Phase 6)** was made after seeing earlier per-type results (stated in the Phase 6 report).

## 9. Reproduction

See `README.md` for the environment and the exact commands for each phase. `reports/verification.md` lists the 118 independent cross-checks of the numbers in this report (the hybrid detector of Phase 11 is reported separately in `reports/phase11_hybrid.md`).

