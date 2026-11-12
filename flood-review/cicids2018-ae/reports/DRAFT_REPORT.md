# Comparative Evaluation of Supervised and Unsupervised Autoencoders for Network Anomaly Detection Using CICIDS2018

Project report (draft). Student name: [to be filled]. Guide: [to be filled]. Department and institution: [to be filled]. Date: [to be filled].

## Abstract

A network intrusion detector can learn from traffic that is labelled as normal or attack (supervised learning), or it can learn only what normal traffic looks like and report whatever deviates from it (unsupervised learning). This project compares the two approaches under controlled conditions. Both use the same autoencoder encoder (72-128-64-32 units), the same one million training records and the same held-out test set of 2,163,852 flow records taken from the CSE-CIC-IDS2018 dataset. Four models were trained: a strictly unsupervised autoencoder trained on unlabelled traffic, a one-class autoencoder trained on benign traffic only, a supervised autoencoder classifier, and a supervised classifier with a decoder. A second experiment withholds four attack types (Bot, DDoS-HOIC, SSH-BruteForce and DoS-Hulk) from every training and tuning step to test detection of unseen attacks. Further experiments on a 9.5-million-record time-ordered stream test three ways of adapting the detector over time, and a final experiment combines the supervised and unsupervised detectors.

The training experiments were run twice with identical code, data and seed, on a laptop (Run A) and on Google Colab (Run B). On attack types present in training, the supervised autoencoder is clearly better in both runs (ROC-AUC 0.944 and 0.966, F1 0.912 and 0.923, against ROC-AUC 0.468 and 0.528 and F1 0.109 and 0.101 for the unsupervised autoencoder). On the four withheld attack types the result reverses: the supervised model ranks them poorly (ROC-AUC 0.648 and 0.615) and flags 0.8% and 13.6% of them, whereas the unsupervised model ranks them well (ROC-AUC 0.927 and 0.904) and flags 74.9% and 52.8%, at the cost of flagging 7.7% of benign traffic. The ordering of the models is the same in both runs, but threshold-dependent counts such as these vary considerably between runs. Training-data purity was the largest single factor for the unsupervised models: removing the 12% of attack records from the training data raised ROC-AUC from 0.468 to 0.889 (0.528 to 0.879 in Run B). In Run A, a simple OR combination of the supervised and unsupervised detectors kept both strengths (87.9% of known and 75.1% of unseen attacks flagged) but also kept the unsupervised model's 7.9% false-alarm rate. Adapting the detector over time did not give reliable gains. A label-free trimmed-loss training of the unsupervised model, a known idea tested over three seeds and three trimming levels, did not remove the contamination effect (ROC-AUC 0.470 to 0.498, against 0.902 for benign-only training), because the highest-error training records were mostly benign. Over four laptop seeds the unsupervised model's ROC-AUC on seen attacks was 0.513 (95% interval 0.446 to 0.581, containing chance) against 0.966 and 0.981 for the two supervised models, whose intervals overlap, and its ROC-AUC on withheld attacks was 0.932 against 0.700 for the supervised model. Threshold-dependent values varied widely between seeds, so close results should not be over-read.

## 1. Introduction

### 1.1 Background

Intrusion detection systems watch network traffic and raise an alert when a connection looks malicious. A detector built with supervised learning is shown examples of normal and attack traffic and learns the boundary between them. It can be very accurate on the kinds of attack it was shown, but it needs labelled attack data, and labels go out of date as new attacks appear. A detector built with unsupervised learning needs no attack labels. An autoencoder is a neural network trained to rebuild its own input through a narrow middle layer; when it is trained on normal traffic it rebuilds normal traffic well and unusual traffic badly, so the reconstruction error can be used as an anomaly score. The question this project addresses is which of the two approaches works better, and for what purpose.

### 1.2 Problem statement

Published results for supervised and unsupervised detectors are usually obtained with different models, different data splits and different thresholds, so they cannot be compared directly. This project holds everything fixed except the way the network is trained, so that the difference can be attributed to the learning method.

### 1.3 Objectives

1. Inspect and preprocess the CSE-CIC-IDS2018 data without leaking information between training and test sets.
2. Train supervised and unsupervised autoencoders with the same architecture and compare them on the same test set (Table 1).
3. Measure detection of attack types that were never seen in training (Table 5).
4. Test whether adapting the detector over time (adaptive baseline, confidence-aware updates, drift-triggered retraining) helps (Tables 2 to 4).
5. Test whether combining the two detectors keeps the strengths of both (Table 6).

### 1.4 Scope

The evaluated results cover data preprocessing, model implementation, experiments, evaluation and results. A dashboard, a Wireshark/PCAP analyser, a live-capture module and deployment files were built afterwards; they serve the saved models and are not part of the evaluated results (Appendix B).

### 1.5 Contributions

The following points describe what this study adds. No systematic literature search was made, so none of them is claimed to be new to the field; they are the parts of the work that were designed and measured here.

1. A controlled comparison. The four models share the encoder, the one million training rows, the test partition (2,163,852 records) and the rule for fixing thresholds before testing. Intervals are block bootstraps over test blocks. The training experiments were repeated on a second computer, which shows which results are stable and which are not.
2. A measurement of training-data contamination. The same autoencoder is trained on unlabelled traffic that contains 12% attacks and on benign traffic only; the ROC-AUC changes from 0.468 to 0.889 (0.528 to 0.879 in the repeat run).
3. A strict unseen-attack protocol. Four attack types are removed from the scaler fit, training, early stopping and the validation rows used for thresholds, and the same types are compared with and without being seen in training (Table 5).
4. A test of label-free safeguards for adapting a detector on a stream (Tables 2 to 4), reported including the negative results.
5. A test of a combined supervised and unsupervised detector, using thresholds fixed in advance (Table 6).
6. A data-quality audit of the processed CSV files (Section 3.2), including points not found in the sources checked for Chapter 2: seven files that appear truncated at the spreadsheet row limit and a 12-hour clock without AM/PM in the timestamps.
7. A test of a known label-free robust-training method (trimmed loss, Chapter 2) on the contamination problem, over three seeds and three trimming levels, reported with its negative result (Section 5.5). It is an evaluation of an existing idea, not a new method.
8. A dashboard and a Wireshark/PCAP and live-capture application whose flow builder was checked against CICFlowMeter's own flow code, the training data and Wireshark, with a measurement of false alarms on ordinary traffic (Appendix B). It serves the saved models; it is not part of the evaluated results.

### 1.6 Organisation of the report

Chapter 2 reviews related work. Chapter 3 describes the data and preprocessing. Chapter 4 describes the models and the experimental method. Chapter 5 gives the results and their discussion. Chapter 6 lists limitations and Chapter 7 concludes.

## 2. Related work

The CSE-CIC-IDS2018 dataset was produced by the Canadian Institute for Cybersecurity and the Communications Security Establishment. Its web page lists seven attack scenarios (brute force, Heartbleed, botnet, DoS, DDoS, web attacks and infiltration from inside) run between 14 February and 2 March 2018, and lists Sharafaldin, Habibi Lashkari and Ghorbani (2018) as the research paper describing the principles behind the dataset [1, 2].

The quality of this family of datasets has been questioned. Engelen, Rimmer and Joosen re-examined the CICIDS2017 dataset, found problems in traffic generation, flow construction, feature extraction and labelling, and reported that results improve on the corrected data [3]. Liu and co-authors extended the analysis to CIC-IDS-2017 and CSE-CIC-IDS-2018 and report previously undocumented errors in attack orchestration, feature generation, documentation and labelling [4]. These findings are the reason this project audits the data before using it (Section 3.2) and treats absolute accuracy figures with caution.

Autoencoder-based detectors trained on traffic assumed to be benign are sensitive to attack records hidden in that training data. Nkashama and co-authors evaluated six unsupervised deep learning detectors under such contamination, found significant degradation, and note that little attention had been paid to contamination in network intrusion detection [5]. Madani and Vlajic studied deep autoencoders for intrusion detection under adversarial contamination of the training data [6]. The present project measures the same effect on CSE-CIC-IDS2018 by comparing an autoencoder trained on unlabelled data (which contains attacks) with one trained on benign data only.

Nkashama and co-authors had already evaluated several unsupervised detectors on CSE-CIC-IDS2018 with contaminated training data [7]. Robust training methods that need no labels also exist. Trimmed-loss training ignores the records with the largest loss at each step and has been analysed for learning from corrupted training data by Shen and Sanghavi [8]. RCA trains collaborative autoencoders that keep, in every mini-batch, the samples with the lowest reconstruction error; it was evaluated on 18 benchmark datasets, none of them a network-intrusion dataset [9]. Beggel and co-authors refine the training set iteratively for images [10], and Kennedy and co-authors describe an iterative cleaning loop that uses an autoencoder as the learner [11]. For choosing a detection threshold without labels, Perini and co-authors estimate the contamination factor from unlabelled data and set the threshold from the estimate [12]. For these reasons a trimmed-loss autoencoder and a contamination-based threshold are not new methods. This report measures how far such a detector closes the gap between unlabelled and clean training on CSE-CIC-IDS2018 under a strict unseen-attack protocol (Chapter 5).

[Author to add: further sources on supervised intrusion detection and on zero-day or unseen-attack evaluation. Only sources that have been read should be added.]

## 3. Dataset and preprocessing

### 3.1 The data

The 10 daily files of the "Processed Traffic Data for ML Algorithms" release (6.41 GB) were used. They contain 16,232,943 flow records after 59 repeated header lines inside three files were dropped. Each record has 80 columns (84 in the file for 20 February, which adds Flow ID, source and destination IP and source port); 78 feature columns are common to all files. There are 13,484,708 benign records (83.07%) and 2,748,235 attack records in 14 categories.

**Table A.** Records per label in the raw data.

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

### 3.2 Data-quality findings

Inspection (script phase1_inspect.py) found the following. The columns flow_byts_s and flow_pkts_s contain 59,721 missing and 131,799 infinite values in total, and eight columns are constant. There are 433,193 exact duplicate rows and 4,251,152 duplicate feature vectors. Seven of the ten files contain exactly 1,048,575 data lines, which is the spreadsheet row limit minus the header line and strongly suggests that these files were truncated; what is missing cannot be seen. The timestamps use a 12-hour clock with no AM/PM marker, so afternoon events appear twelve hours early; this was confirmed against the attack schedule on the dataset page (for example, DoS-Hulk scheduled for 13:45 is stored as 01:45). Rows are not in time order, 14 rows carry timestamps in January 1970, and the identifier columns exist only in one file and were not used. The destination port alone almost identifies some attacks (all FTP-BruteForce records use port 21, against none of the benign records), so it was excluded to prevent the models from learning a shortcut.

### 3.3 Cleaning, de-duplication and scaling

Records with missing or infinite values (95,760), a negative value (1) or an invalid timestamp (14) were removed, leaving 16,137,168. Identical feature vectors were then removed (4,944,797), together with vectors that appear with both a benign and an attack label (416,552), leaving 10,775,819 records. De-duplication shrinks some attack types to a few dozen distinct flows (FTP-BruteForce falls from 193,360 to 41 records).

The model input has 72 columns: 68 numeric features and a one-hot protocol code (protocols 0, 6, 17 and other). Each numeric feature is transformed by sign(x)·log(1+|x|) and scaled to the range [0, 1] using the minimum and maximum of the training partition only. Validation and test values are not clipped.

### 3.4 Train, validation and test split

Neighbouring flows from the same attack are nearly identical, so a random split of records would put near-copies in both training and test sets. Each capture day was therefore cut into 60-second blocks and whole blocks were assigned to training, validation and test (70%, 10% and 20% of blocks), stratified by the attack types in each block. No block is divided between partitions and no identical feature vector appears in two partitions. This split is still optimistic, because adjacent blocks of one attack campaign resemble each other.

**Table B.** The three partitions.

| Partition | Records | Benign | Attacks | Attack share |
|---|---|---|---|---|
| Train | 7,438,408 | 6,525,735 | 912,673 | 12.27% |
| Validation | 1,173,559 | 1,040,160 | 133,399 | 11.37% |
| Test | 2,163,852 | 1,905,720 | 258,132 | 11.93% |

## 4. Methodology

### 4.1 Models

All models share the encoder 72 → 128 → 64 → 32, with ReLU activations. They were trained with the Adam optimiser (learning rate 0.001, batch size 1,024, early stopping with patience 5, seed 42) on the same one million rows drawn from the training partition, and tested on the same test partition.

**Table C.** The four models.

| Model | Training data | Loss | Anomaly score | Threshold, fixed before testing |
|---|---|---|---|---|
| Unsupervised AE | 1M unlabelled training rows (about 12% attacks); labels never read | mean squared reconstruction error | reconstruction error | 95th percentile of validation errors (no labels) |
| One-class AE | benign training rows only | mean squared reconstruction error | reconstruction error | 99th percentile of benign validation errors |
| Supervised AE | the same rows, with labels | class-weighted binary cross-entropy (weight 7.14 on attacks) | attack probability from a 32 → 16 → 1 head | 0.5 |
| Supervised AE + decoder | the same rows, with labels | weighted cross-entropy plus reconstruction error | attack probability | 0.5 |

For the autoencoders the decoder mirrors the encoder (32 → 64 → 128 → 72, sigmoid output). A record is reported as an attack when its score exceeds the threshold. Reconstruction error is used only to rank records, never as a probability. The one-class model is not strictly unsupervised because benign rows are selected using labels; it is included to separate the effect of training-data purity from the effect of learning method.

### 4.2 Evaluation

Accuracy is a poor measure here because 88% of test records are benign. The report therefore uses precision, recall, F1, false-positive rate, ROC-AUC (how well the score ranks attacks above benign records, independent of any threshold) and PR-AUC. Test labels were read only after predictions existed and were never used to set a threshold. Confidence intervals are 95% block bootstraps over test blocks; they measure sampling variation in the test set and do not include variation from training.

### 4.3 Experiments

The experiments, with the script that runs each one, are listed in Table D.

**Table D.** Experiments and scripts.

| Experiment | Question | Script |
|---|---|---|
| A and B | train the unsupervised and supervised models | phase3_unsupervised.py, phase4_supervised.py |
| Comparison | Table 1, intervals, figures | phase5_compare.py |
| C | detection of withheld attack types, Table 5 | phase6_unseen.py, phase6_tables.py |
| D | adaptive baseline, Table 2 | stream.py, phase7_adaptive.py |
| E | confidence-aware updates, Table 3 | phase8_confidence.py |
| F | concept drift, Table 4 | phase9_drift.py |
| G | combined supervised and unsupervised detector, Table 6 | phase11_hybrid.py |
| H | does label-free trimmed-loss training close the contamination gap? Table 8 | robust.py, phase13_trimmed.py, aggregate_trimmed.py |
| Check | independent recomputation of the Run A numbers (138 checks, all passed) and of the Table 8 numbers (48 checks, all passed) | verify_results.py, verify_trimmed.py |

In experiment C, Bot, DDoS-HOIC, SSH-BruteForce and DoS-Hulk are removed from the scaler fit, the training rows, early stopping and the validation rows used for thresholds, and all three model types are retrained. These types were chosen by a rule fixed in advance (at least 10,000 test rows, distinct families, excluding Infiltration and LOIC-HTTP), but earlier per-type results had been seen, so the choice is not blind.

Experiments D to F use the whole dataset as a time-ordered stream of 9,504,036 records over eight days (1,158,860 attacks), with the first two days as warm-up. Each block of 50,000 records is scored first and only then used to update the model. Attack labels never controlled an update, a drift detector or a threshold.

Repeat run. The training experiments A to C (Tables 1 and 5) were run twice with identical code, data and seed: first on a laptop (Run A, the reference run, used for every table unless stated otherwise) and then on Google Colab (Run B, a CPU session). Because the hardware differs, training does not follow exactly the same numerical path, so the two runs act as two repetitions of each model. Run B values were transcribed by hand from the output of the Colab notebook and are given in Tables 1b and 5b; they are not covered by the independent verification script, which checks Run A. To quantify run-to-run variation more directly, experiments A to C were also repeated on the laptop with three further training seeds (1, 2 and 3); Tables 1c and 5c summarise these together with Run A (seed 42) as mean, 95% t-interval and range over four seeds. The experiments of Tables 2 to 4 and Table 6 were run once, in Run A.

## 5. Results and discussion

### 5.1 Supervised against unsupervised on attack types seen in training

**Table 1.** Run A (laptop): results on the test partition (2,163,852 records, 11.93% attacks). Brackets give 95% block-bootstrap intervals.

| Metric | Unsupervised AE | One-class AE | Supervised AE | Supervised AE + decoder |
|---|---|---|---|---|
| Accuracy | 0.8236 | 0.8558 | 0.9802 | 0.9736 |
| Precision | 0.1368 | 0.0348 | 0.9712 | 0.8623 |
| Recall | 0.0901 | 0.0078 | 0.8599 | 0.9265 |
| F1-score | 0.1086 | 0.0128 | 0.9122 | 0.8932 |
| False-positive rate | 0.0770 | 0.0294 | 0.0035 | 0.0200 |
| ROC-AUC | 0.468 [0.389, 0.528] | 0.889 [0.851, 0.916] | 0.944 [0.897, 0.984] | 0.980 [0.968, 0.986] |
| PR-AUC (chance 0.119) | 0.109 [0.067, 0.149] | 0.410 [0.326, 0.545] | 0.914 [0.855, 0.967] | 0.954 [0.917, 0.970] |
| Recall at 1% false positives | 0.008 | 0.005 | 0.869 | 0.910 |

**Table 1b.** Run B (Google Colab): the same experiment repeated with identical code, data and seed. Intervals and recall at 1% false positives were not extracted for this run.

| Metric | Unsupervised AE | One-class AE | Supervised AE | Supervised AE + decoder |
|---|---|---|---|---|
| Accuracy | 0.8230 | 0.8726 | 0.9824 | 0.9837 |
| Precision | 0.1282 | 0.1154 | 0.9711 | 0.9620 |
| Recall | 0.0834 | 0.0101 | 0.8790 | 0.8988 |
| F1-score | 0.1011 | 0.0186 | 0.9227 | 0.9293 |
| False-positive rate | 0.0768 | 0.0105 | 0.0035 | 0.0048 |
| ROC-AUC | 0.528 | 0.879 | 0.966 | 0.979 |
| PR-AUC (chance 0.119) | 0.123 | 0.398 | 0.936 | 0.952 |

**Table 1c.** Four training seeds on the laptop (seeds 1, 2, 3 and 42; seed 42 is Run A). Mean with the 95% t-interval of the mean in brackets; the last two rows give the smallest and largest single-seed value. Only training randomness differs between seeds (initialisation, batch order, the one million training rows); the split and test set are identical. With n = 4 the intervals are wide and assume roughly normal variation, and an upper limit of 1.000 or a lower limit of 0.000 is the cut at the edge of the possible range.

| Metric | Unsupervised AE | One-class AE | Supervised AE | Supervised AE + decoder |
|---|---|---|---|---|
| ROC-AUC | 0.513 [0.446, 0.581] | 0.899 [0.882, 0.917] | 0.966 [0.933, 0.998] | 0.981 [0.979, 0.983] |
| PR-AUC (chance 0.119) | 0.118 [0.103, 0.133] | 0.467 [0.372, 0.562] | 0.938 [0.899, 0.977] | 0.955 [0.949, 0.961] |
| F1-score | 0.094 [0.023, 0.164] | 0.282 [0.000, 0.784] | 0.917 [0.875, 0.959] | 0.912 [0.857, 0.968] |
| Recall | 0.078 [0.019, 0.137] | 0.246 [0.000, 0.688] | 0.895 [0.847, 0.942] | 0.922 [0.907, 0.938] |
| Precision | 0.118 [0.031, 0.205] | 0.336 [0.000, 0.907] | 0.941 [0.878, 1.000] | 0.906 [0.789, 1.000] |
| False-positive rate | 0.078 [0.074, 0.082] | 0.034 [0.029, 0.039] | 0.008 [0.000, 0.017] | 0.014 [0.000, 0.032] |
| ROC-AUC, single seeds (min–max) | 0.468–0.562 | 0.889–0.913 | 0.944–0.983 | 0.980–0.983 |
| F1, single seeds (min–max) | 0.028–0.122 | 0.006–0.573 | 0.882–0.943 | 0.873–0.945 |

![Figure 1. Run A: ROC curves (left and centre) and precision-recall curves (right) of the four models on the test set.](results/phase5_roc_pr_comparison.png)

The supervised models are clearly better on attacks that resemble their training data, in both runs. In Run A the supervised autoencoder correctly flags 221,972 of the 258,132 attacks and raises 6,577 false alarms on 1,905,720 benign records, whereas the unsupervised autoencoder flags 23,253 attacks and raises 146,752 false alarms (Figure 2). Adding a decoder to the supervised model raised ROC-AUC from 0.944 to 0.980 in Run A (paired difference +0.028, interval -0.000 to +0.082, not distinguishable from zero) and from 0.966 to 0.979 in Run B. The ordering of the four models by ROC-AUC is the same in both runs: supervised with decoder, supervised, one-class, unsupervised. Other numbers differ between runs, for example the supervised-with-decoder false-positive rate (2.0% in Run A, 0.48% in Run B), which shows that single-run values of threshold-dependent metrics are imprecise.

The four laptop seeds (Table 1c) confirm the pattern and show which parts of it are stable. The interval of the unsupervised model's ROC-AUC (0.446 to 0.581) contains 0.5, so its ranking of attacks cannot be distinguished from chance. The one-class model is clearly above it (0.882 to 0.917) and the two supervised models are clearly above the one-class model. The two supervised models are not separated from each other: their intervals overlap for ROC-AUC (0.933 to 0.998 against 0.979 to 0.983), PR-AUC and F1, so with four seeds there is no evidence that the decoder helps, although its ROC-AUC varied much less between seeds (0.980 to 0.983 against 0.944 to 0.983). Threshold-dependent values are the least stable: the one-class F1 ranged from 0.006 to 0.573 across seeds which is consistent with the sensitivity of a 99th-percentile threshold to a small group of benign records with very large errors (Section 5.1), and the false-positive rates of the two supervised models varied five- to sevenfold (0.3% to 1.6% and 0.4% to 2.7%).

![Figure 2. Run A: confusion matrices on the test set at the fixed thresholds: unsupervised AE (left) and supervised AE (right).](results/phase5_confusion_unsup_vs_sup.png)

The unsupervised autoencoder ranks attacks little better than chance (ROC-AUC 0.468 in Run A, 0.528 in Run B). Its training data contained about 12% attacks, and the network learned to rebuild them as well as benign traffic, so attack and benign records receive similar errors (Figure 3, left). When the attacks are removed from its training data (one-class model), the attacks receive clearly larger errors (Figure 3, right) and ROC-AUC rises to 0.889 (0.879 in Run B). A further observation points the same way: the same attack types are detected much better by the unsupervised model when they are withheld from training (Table 5). The one-class model also uses labels to select its benign rows, and the withheld-type comparison changes the amount of attack data in training, so the explanation is best described as consistent with contamination rather than proved.

A high ROC-AUC does not guarantee a useful detector. The one-class model ranks attacks well (0.889 and 0.879) but flags only 0.8% of them at its threshold (1.0% in Run B). Its threshold is the 99th percentile of benign errors, and a small group of ordinary benign flows has very large errors (the benign bar at about 10^-1.9 in Figure 3, right), which pushes the threshold above the errors of most attacks. A threshold chosen to maximise F1 on validation data lifts its F1 to 0.650 (0.625 in Run B), but that rule uses validation labels and fails for the unsupervised model, which then flags 99.99% of all records in Run A and about 80% of benign records in Run B.

![Figure 3. Run A: distribution of reconstruction errors (log scale) for benign and attack records, with the fixed threshold. Left: unsupervised AE. Right: one-class AE.](results/phase3_score_distributions.png)

### 5.2 Detection of unseen attack types

**Table 5.** Run A (laptop): attack types withheld from training. The test set has 1,905,720 benign, 136,861 known-attack and 121,271 withheld-attack records.

| Metric | Unsupervised AE | One-class AE | Supervised AE |
|---|---|---|---|
| Known attacks: recall | 0.0461 | 0.0067 | 0.8704 |
| Benign: false-positive rate | 0.0768 | 0.0098 | 0.0031 |
| Unseen attacks: recall | 0.7489 | 0.0135 | 0.0077 |
| Unseen Bot: share flagged | 0.733 | 0.001 | 0.019 |
| Unseen DDoS-HOIC: share flagged | 0.863 | 0.000 | 0.000 |
| Unseen SSH-BruteForce: share flagged | 1.000 | 0.002 | 0.000 |
| Unseen DoS-Hulk: share flagged | 0.586 | 0.035 | 0.008 |
| Known set ROC-AUC | 0.4088 | 0.8788 | 0.9692 |
| Unseen set ROC-AUC | 0.9274 | 0.9162 | 0.6483 |

**Table 5b.** Run B (Google Colab): the same experiment repeated. Bracketed intervals were not extracted for this run.

| Metric | Unsupervised AE | One-class AE | Supervised AE |
|---|---|---|---|
| Known attacks: recall | 0.0437 | 0.0069 | 0.8713 |
| Benign: false-positive rate | 0.0767 | 0.0098 | 0.0033 |
| Unseen attacks: recall | 0.5281 | 0.0134 | 0.1364 |
| Unseen Bot: share flagged | 0.026 | 0.001 | 0.008 |
| Unseen DDoS-HOIC: share flagged | 0.863 | 0.000 | 0.081 |
| Unseen SSH-BruteForce: share flagged | 1.000 | 0.002 | 0.000 |
| Unseen DoS-Hulk: share flagged | 0.451 | 0.035 | 0.309 |
| Known set ROC-AUC | 0.4886 | 0.8821 | 0.9690 |
| Unseen set ROC-AUC | 0.9042 | 0.9198 | 0.6151 |

**Table 5c.** Four training seeds on the laptop (seeds 1, 2, 3 and 42), same layout as Table 1c. The last row gives the single-seed range of the withheld-set recall.

| Metric | Unsupervised AE | One-class AE | Supervised AE |
|---|---|---|---|
| Known attacks: recall | 0.074 [0.038, 0.109] | 0.177 [0.000, 0.597] | 0.872 [0.869, 0.875] |
| Benign: false-positive rate | 0.076 [0.074, 0.077] | 0.030 [0.009, 0.051] | 0.004 [0.002, 0.007] |
| Unseen attacks: recall | 0.848 [0.695, 1.000] | 0.052 [0.000, 0.180] | 0.167 [0.000, 0.378] |
| Known set ROC-AUC | 0.418 [0.356, 0.481] | 0.877 [0.862, 0.892] | 0.969 [0.969, 0.970] |
| Unseen set ROC-AUC | 0.932 [0.926, 0.937] | 0.920 [0.900, 0.941] | 0.700 [0.568, 0.832] |
| Unseen attacks: recall, single seeds (min–max) | 0.749–0.964 | 0.006–0.172 | 0.008–0.333 |

Across the four seeds the unsupervised model ranks the withheld attacks well and consistently (ROC-AUC 0.927 to 0.935), the one-class model similarly (0.906 to 0.936), and the supervised model poorly and unevenly (0.648 to 0.824). On the known attacks the unsupervised model stays near or below chance (interval 0.356 to 0.481). The recall values at fixed thresholds again vary much more than the rankings: the unsupervised model's recall of withheld attacks ranged from 74.9% to 96.4% over the laptop seeds, and the supervised model's from 0.8% to 33.3%. Its per-type shares flagged are unstable as well (unsupervised Bot 27% to 98%, supervised DDoS-HOIC 0% to 65%). The Colab Run B value for the unsupervised model (52.8%) lies below the laptop range, so the spread between runs is not fully covered by four laptop seeds, and it is a reason to read the recall values at fixed thresholds as indicative only.

![Figure 4. Run A: detection of withheld attack types at the fixed thresholds (left) and ROC curves for benign against all withheld attacks (right).](results/phase6_unseen_detection.png)

The supervised model flags few attacks it has not seen (0.8% in Run A, 13.6% in Run B), and its ranking of them is poor (ROC-AUC 0.648 and 0.615). The unsupervised family ranks them well (0.927 and 0.904 for the unsupervised model, 0.916 and 0.920 for the one-class model). The large difference in recall between the unsupervised model (74.9% and 52.8%) and the one-class model (1.35% and 1.34%) comes mostly from the threshold: the unsupervised threshold is looser and flags 7.7% of benign records in both runs, while the one-class threshold flags 0.98% in both. The evidence therefore supports the statement that reconstruction-based models can rank unseen attacks, but a usable operating point is hard to set; it does not show that the unsupervised model is better.

The two runs agree on the ranking results (ROC-AUC) but not on counts at a fixed threshold. The unsupervised model flagged 73.3% of unseen Bot records in Run A and 2.6% in Run B, although its ROC-AUC for Bot was high in both (0.923 and 0.871), which means Bot records score close to the threshold and small changes in the trained network move them across it. Per-type flagged shares and the recall values at fixed thresholds are therefore less reliable than the ROC-AUC values, and the latter are the stronger evidence in this report.

The same four attack types give a direct comparison. When they were in training, the unsupervised model's ROC-AUC against benign traffic was 0.42 to 0.68 in Run A (0.36 to 0.81 in Run B); when withheld it was 0.91 to 0.95 (0.87 to 0.95). It was higher when withheld for each of the four types in both runs. The supervised model went the other way, from 0.78 to 1.00 when seen to 0.34 to 0.82 when withheld in Run A (0.91 to 1.00 to 0.26 to 0.84 in Run B), lower for each type in both runs. The two sets of models also differ in training size (45% fewer attack rows in the withheld setting), and each withheld type is a single attack episode occupying few test blocks (Bot 68, SSH-BruteForce 18, DDoS-HOIC 5, DoS-Hulk 1), so the per-type numbers describe one episode each.

**Limitations of reconstruction-error detection, as observed here.** The experiments show six limits, each measured and not assumed. (1) Contamination: when the training data contains attacks (about 12%), the network rebuilds them as well as benign records, and the ranking falls to chance (ROC-AUC 0.513, interval 0.446 to 0.581). (2) The score measures unusualness, not maliciousness: on attack types seen in training, the unsupervised model's known-set ROC-AUC was 0.418 (interval 0.356 to 0.481), below chance, meaning many of these attacks were rebuilt better than the average benign record. (3) The operating point is fragile: the one-class model ranks attacks well (0.899) but at its 99th-percentile threshold its F1 ranged from 0.006 to 0.573 across seeds, because a small group of benign flows with very large errors sets the threshold, and the unsupervised model's recall on withheld attacks ranged from 74.9% to 96.4% at a fixed threshold. (4) Individual attack types near the threshold are unstable (unsupervised Bot flagged 27% to 98% across seeds). (5) Cost in false alarms: the setting that caught most withheld attacks also flagged 7.5% to 8.2% of benign records across seeds. (6) The model's own score cannot tell a harmless change in normal traffic from an attack, which is the likely reason the adaptation experiments of Section 5.4, which rely on that score to choose what to retrain on, gave no reliable gain (the retraining buffer held 12.3% attacks, the same share as the stream). These limits belong to this dataset, this architecture and the thresholds tested; they are not general properties of all reconstruction-based detectors.

### 5.3 Combining the two detectors

The results above suggest combining a supervised classifier (good on known attacks) with a reconstruction-based detector (able to flag unseen ones). This was tested with a rule written down before it was evaluated and nothing tuned: report an attack if the supervised probability exceeds 0.5 or the reconstruction error exceeds the novelty model's fixed threshold. It uses only scores already saved by the earlier experiments, with no retraining.

**Table 6.** Combined detector on the unseen-attack setting (models of experiment C). Brackets give 95% block-bootstrap intervals.

| Detector | Known-attack recall | Unseen-attack recall | Benign false-positive rate | Unseen-set F1 |
|---|---|---|---|---|
| Supervised | 0.870 | 0.008 [0.005, 0.013] | 0.0031 | 0.015 |
| Unsupervised | 0.046 | 0.749 [0.652, 0.879] | 0.0768 | 0.507 |
| Supervised OR unsupervised | 0.879 | 0.751 [0.654, 0.881] | 0.0790 | 0.502 |
| Supervised OR one-class | 0.872 | 0.021 [0.005, 0.034] | 0.0127 | 0.035 |

Combining the supervised and the unsupervised model is the only design tested that flags most known attacks (87.9%) and most unseen ones (75.1%) together. It does so at the unsupervised model's false-alarm rate: 7.9% of benign records, about 25 times the supervised model's 0.31%. On the seen-attack test set the combination reaches recall 0.919 but precision only 0.609 (F1 0.733, against 0.912 for the supervised model alone), so whether it is acceptable depends on how many false alarms a security team can handle. Combining with the one-class model changes little at the fixed threshold because that model flags almost nothing. The result is a single run (Run A) with fixed thresholds and is not an optimised system. The unsupervised component flagged 52.8% of unseen attacks in Run B instead of 74.9%, and since the union of two flagged sets is at least as large as the larger one, the combination's unseen-attack recall in Run B is at least 52.8% rather than the 75.1% above; the Run B scores were not analysed for this table.

### 5.4 Adapting the detector over time

Tables 2 to 4 report experiments on the time-ordered stream. Only the main findings are summarised here; full tables are in reports/FINAL_REPORT.md and the per-phase files.

**Table 2.** Adaptive baseline (experiment D): static detector (A), adaptive threshold (B), adaptive baseline with retraining on a trusted buffer (C).

| Metric | A static | B adaptive threshold | C adaptive baseline |
|---|---|---|---|
| Precision | 0.1347 | 0.0348 | 0.1002 |
| Recall | 0.0136 | 0.0041 | 0.0115 |
| F1-score | 0.0247 | 0.0074 | 0.0206 |
| False-positive rate | 0.0122 | 0.0159 | 0.0143 |
| Retrainings | 0 | 0 | 64 |

Adaptation did not improve detection. The adaptive-threshold rule was unstable (its threshold reached 1,660, far above any error seen before). Retraining improved ranking on the two days whose benign traffic changed (ROC-AUC 0.079 to 0.472 on 16 February and 0.013 to 0.745 on 21 February) and worsened it on a day dominated by a large attack (0.948 to 0.409 on 20 February). The "trusted" buffer, selected using the model's own score, held 12.3% attacks, the same as the stream (12.2%), so the filter kept no attacks out.

**Table 3.** Confidence-aware updates (experiment E): static (S), unrestricted updates (U), margin filter only (M), full confidence gate (G).

| Metric | S static | U unrestricted | M margin only | G gated |
|---|---|---|---|---|
| False-positive rate | 0.1602 | 0.0665 | 0.1541 | 0.0835 |
| Recall | 0.7035 | 0.1416 | 0.7017 | 0.2100 |
| F1-score | 0.4925 | 0.1748 | 0.4991 | 0.2319 |
| Attack share in the retraining buffer (stream 0.122) | n/a | 0.123 | 0.038 | 0.128 |

The confidence gate gave no clear advantage over unrestricted updates (difference in F1 +0.059, interval -0.013 to +0.131). Only the margin filter kept attacks out of the buffer (0.038), and only because it never adapted. The overall scores of S and M are inflated, because on two days the benign traffic shifted above their threshold and they flagged 90% to 94% of everything.

**Table 4.** Concept drift (experiment F): no detection (A), scheduled retraining (A2), drift-triggered retraining (B), drift-triggered with confirmation (C).

| Metric | A none | A2 periodic | B drift-triggered | C drift + confirmation |
|---|---|---|---|---|
| False-positive rate | 0.1602 | 0.0666 | 0.0694 | 0.0922 |
| Recall | 0.7035 | 0.1410 | 0.3071 | 0.3830 |
| F1-score | 0.4925 | 0.1741 | 0.3399 | 0.3742 |
| Retrainings | 0 | 64 | 45 | 15 |

The ADWIN drift detector found all four benign change points (median delay 117,500 records) but raised 10 false alarms among its 20 alarms. Drift-triggered retraining scored higher than scheduled retraining in this run, but the evidence attributes this mostly to timing: it did not retrain during the Bot attack campaign, whereas scheduled retraining ran throughout the stream. The four change points are jumps between days, so the drift results rest on four events.

### 5.5 Does label-free robust training close the contamination gap?

Section 5.1 suggests that the unsupervised model fails because about 12% of its training rows are attacks. Trimmed-loss training is a known label-free remedy (Chapter 2): at every step the highest-error records of the batch are left out of the loss, so that attacks hidden among normal records are not learned. This experiment (H) measures whether it helps on this dataset. It is an evaluation of an existing idea, not a new method.

Design, written before any result was seen: the same one million unlabelled training rows and 300,000 early-stopping rows as the unsupervised model of the same seed; trimmed shares q of 10%, 20% and 30% (the true attack share, about 12.3%, was not used to choose them); seeds 42, 1 and 2. The keep share falls linearly from 100% to 1 - q over the first ten epochs. Early stopping uses the trimmed error of unlabelled rows. Three label-free thresholds were defined in advance: L1, the 95th percentile of all validation errors (the rule of the unsupervised model); L2, the 95th percentile of the validation errors below the (1 - q) quantile; L3, the 99th percentile of the same reference. Labels are used only to report results and in the diagnostic columns, never for training or thresholds.

**Table 8.** Trimmed-loss autoencoder over three seeds (42, 1, 2). Mean with the 95% t-interval of the mean in brackets (n = 3, wide). The two reference rows are the Phase 3 models of the same seeds.

| Model | ROC-AUC | PR-AUC (chance 0.119) | F1 at L2 | Recall at L2 | False-positive rate at L2 | Attacks among the training rows kept | Attacks removed from training |
|---|---|---|---|---|---|---|---|
| Unsupervised, ordinary loss (reference) | 0.506 [0.384, 0.629] | 0.116 [0.090, 0.142] | n/a | n/a | n/a | 12.3% (all rows) | 0% |
| Trimmed 10% | 0.498 [0.432, 0.564] | 0.113 [0.109, 0.116] | 0.104 [0.098, 0.109] | 0.127 [0.121, 0.133] | 0.179 [0.176, 0.183] | 12.9% | 5.6% |
| Trimmed 20% | 0.473 [0.428, 0.517] | 0.107 [0.093, 0.120] | 0.105 [0.089, 0.120] | 0.171 [0.149, 0.193] | 0.284 [0.273, 0.295] | 14.2% | 7.4% |
| Trimmed 30% | 0.470 [0.431, 0.510] | 0.104 [0.091, 0.116] | 0.139 [0.110, 0.168] | 0.284 [0.218, 0.351] | 0.378 [0.371, 0.386] | 15.4% | 12.3% |
| One-class, benign rows only (reference) | 0.902 [0.873, 0.932] | 0.482 [0.326, 0.639] | n/a | n/a | n/a | 0% | 100% |

Trimming did not help. At every trimming level the interval of the ROC-AUC contains 0.5, so the ranking cannot be distinguished from chance, and it cannot be distinguished from the untrimmed model either, while it stays far below the 0.902 of the model trained on benign rows only. The label-free thresholds did not rescue it: at L2 the F1 is 0.10 to 0.14, and the share of benign records flagged rises with the trimmed share, from 17.9% to 37.8%.

The diagnostic columns show why. A trimming rule that picked attacks would remove a larger share of the attacks than of all rows (10%, 20% and 30%), and the training rows it keeps would contain a smaller attack share than 12.3%. The opposite happened: the share of attacks removed (5.6%, 7.4% and 12.3%) is below the trimmed share each time, and the training rows that remain contain a larger attack share (12.9%, 14.2% and 15.4%) than before. In this data the records with the largest reconstruction error are mostly benign, which fits the earlier observation that many seen attacks are rebuilt better than the average benign record (known-set ROC-AUC 0.418, Table 5c). This is an explanation consistent with the numbers and was not tested further. A method that ranks records by reconstruction error to find contamination inherits this property of the score.

The result is limited to this variant: one architecture, three trimming levels, a linear schedule, three seeds and one dataset. It does not show that other robust methods, such as the collaborative autoencoders of RCA or an explicit contamination estimate, would also fail. The numbers of Table 8 were recomputed independently from the saved scores (48 checks, all passed); the reference rows are the Phase 3 results of seeds 42, 1 and 2.

### 5.6 Which approach is better?

Neither approach is better overall; each is better at a different task.

**Table 7.** Summary of which approach fits which situation.

| Situation | Better fit | Evidence |
|---|---|---|
| Attack types are known and labelled data exists | Supervised | Tables 1 and 1b: F1 0.91 to 0.92, false-positive rate 0.35%; recall 0.87 at 1% false positives (Run A) |
| New or unlabelled attack types must be caught | Reconstruction-based, trained on clean traffic | Tables 5 and 5b: unseen ROC-AUC 0.90 to 0.93 against 0.62 to 0.65 for supervised (two runs) |
| Only unlabelled data containing attacks is available | Neither works well | Tables 1 and 1b: unsupervised ROC-AUC 0.47 to 0.53; trimmed-loss training did not help (Table 8) |
| False alarms must be very rare | Supervised | false-positive rate 0.35% against 1% to 8% |
| Both known and unseen attacks matter | A combination, at the cost of more false alarms | Table 6 (Run A only) |

The supervised autoencoder is the better detector of the attacks it was taught, and the reconstruction-based autoencoder is the only one that detects attacks it was not taught, provided its training data is clean and its threshold is chosen carefully. Time-adaptation methods based on the model's own score, and trimmed-loss training that drops the highest-error records, did not help in these experiments.

## 6. Limitations

Experiments A to C were trained with four seeds on the laptop (Tables 1c and 5c) and once on Google Colab (Run B), with the same code and data and no hyper-parameter search; four seeds give wide intervals, and the bootstrap intervals in Tables 1 and 5 do not include training variation. The seeds and the two runs agree on the ordering of the models and on the ROC-AUC results, but not on several threshold-dependent numbers; between Run A and Run B: the unseen-attack recall of the unsupervised model was 74.9% and 52.8%, that of the supervised model 0.8% and 13.6%, and the share of unseen Bot flagged by the unsupervised model 73.3% and 2.6%. Such numbers should be read as ranges, and close results (for example the two supervised variants) are not distinguishable. The Colab value for the unsupervised model's unseen-attack recall (52.8%) lies below the range of the four laptop seeds (74.9% to 96.4%), so even four seeds do not cover the variation between machines. Tables 2 to 4 and Table 6 come from one run only. Run B values were transcribed from the notebook output and were not independently verified. Seven of the ten source files appear truncated, the timestamps lack AM/PM, and de-duplication removed 33% of the cleaned records, leaving some attack types with fewer than 110 test rows. The in-distribution split is optimistic because neighbouring blocks of one attack are similar; only the unseen-attack experiment tests generalisation, with four types, each a single attack episode. The thresholds were fixed in advance and are poor for the unsupervised family, and the results change by an order of magnitude with the threshold rule. The streaming experiments rest on day-level changes only (four events, one day holding 57% of the records) and did not test gradual drift. The scaler was fitted on a training partition that contains all days. Training times and memory use varied about threefold between identical runs and were not used to compare models. The combined detector (Table 6) used fixed thresholds and was not tuned. The trimmed-loss experiment (Table 8) used three seeds, three trimming levels and one training schedule, so its negative result does not extend to other robust methods. A dashboard and a Wireshark and live-capture tool were built afterwards (APP_README.md); they are not part of the evaluated results. They serve the saved models. Their flow builder is a port of CICFlowMeter's flow logic that was checked against the tool's own flow code (4,701 flows, all 4,701 identical in all 69 feature values), against the training data and against Wireshark (Appendix B), but no detection result for attacks on a live network is claimed: on 1149 flows of ordinary Wi-Fi traffic the unsupervised model flagged 69%, and the other models 2.3% to 4.4%.

## 7. Conclusion and future work

Under identical architecture, training rows and test data, the supervised autoencoder clearly outperformed the unsupervised autoencoders on attack types seen in training, in both runs. On attack types never seen in training, the unsupervised autoencoder ranked them well (ROC-AUC 0.90 to 0.93) and flagged between half and three quarters of them (52.8% and 74.9% in the two runs), while the supervised autoencoder ranked them poorly (ROC-AUC 0.62 to 0.65) and flagged few (0.8% and 13.6%). The result depends strongly on training-data purity and on the threshold. In one run, a simple combination of the two detectors kept both strengths at the price of 7.9% false alarms on benign traffic. Adapting the detector with label-free safeguards did not give reliable improvement, and neither did trimmed-loss training (three seeds, three trimming levels): the records it removed were mostly benign, and the ranking stayed at chance. Future work should test other robust-training methods, repeat the stream experiments with several random seeds, withhold more attack types and episodes, tune a combined detector with a stated alert budget, use data with gradual drift, and use a corrected version of the dataset.

## References

[1] I. Sharafaldin, A. Habibi Lashkari and A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018. (Listed on the CSE-CIC-IDS2018 page [2]; pages and DOI to be added.)

[2] Canadian Institute for Cybersecurity, "CSE-CIC-IDS2018 dataset", https://www.unb.ca/cic/datasets/ids-2018.html (the page also directs users to the AWS open-data registry, https://registry.opendata.aws/cse-cic-ids2018). Accessed November 2026.

[3] G. Engelen, V. Rimmer and W. Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study", IEEE Security and Privacy Workshops (SPW), 2021, pp. 7-12, doi:10.1109/SPW53761.2021.00009.

[4] Liu, Engelen, Lynar, Essam and Joosen, "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018", IEEE Conference on Communications and Network Security (CNS), 2022. (First names, pages and DOI to be added.)

[5] D. K. Nkashama, J. Masakuna Félicien, A. Soltani, J.-C. Verdier, P.-M. Tardif, M. Frappier and F. Kabanza, "Deep Learning for Network Anomaly Detection under Data Contamination: Evaluating Robustness and Mitigating Performance Degradation", arXiv:2407.08838, 2024.

[6] P. Madani and N. Vlajic, "Robustness of Deep Autoencoder in Intrusion Detection under Adversarial Contamination", 2018. (Venue to be confirmed.)

[7] D. K. Nkashama et al., "Robustness Evaluation of Deep Unsupervised Learning Algorithms for Intrusion Detection Systems", arXiv:2207.03576, 2022. (Author list and venue to be added.)

[8] Shen and Sanghavi, work on iterative trimmed loss minimization for learning with bad training data, ICML 2019. (Exact title, first names and pages to be added.)

[9] B. Liu, D. Wang, K. Lin, P.-N. Tan and J. Zhou, "RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection", Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21), 2021, pp. 1505-1511, doi:10.24963/ijcai.2021/208.

[10] Beggel et al., "Robust Anomaly Detection in Images using Adversarial Autoencoders", arXiv:1901.06355, 2019. (Author list and venue to be added.)

[11] Kennedy et al., Journal of Big Data, 2023, article doi:10.1186/s40537-023-00750-3. (Title and author list to be added; the article could not be read.)

[12] Perini, Bürkner and Klami, "Estimating the Contamination Factor's Distribution in Unsupervised Anomaly Detection", arXiv:2210.10487. (Year and venue to be added.)

Note on the references: [2], [5] and [9] were read on the web (for [9] the publisher's page with the abstract and full text); [3], [4], [6], [7], [8], [10] and [12] were checked only through search-result summaries and library records; [11] was found in search results but its page could not be read; and [1] is cited only as listed on [2]. Read each source and complete the missing details before submission.

## Appendix A. Corrections made during the project

The first run of the unsupervised experiment was invalid: an argument-order error set the early-stopping tolerance to 42, so the weights of the first epoch were restored. All reported numbers come from the corrected rerun. The unexplained timestamp hours found in the first inspection were a lost AM/PM marker, identified in the streaming experiments; the earlier experiments are unaffected because blocks are never split across partitions. The training-time comparison was withdrawn because identical runs differed about threefold (313 s against 86 s). The adaptive-threshold rule of experiment D was unstable and was replaced by an anchor-based rule in experiments E and F; experiment D is reported as run. The withheld attack types were chosen after earlier per-type results had been seen. The scaler refit in experiment C left every minimum and maximum unchanged. In an earlier version of the project documents the numeric feature count was given as 64; the correct count is 68 numeric features plus 4 protocol columns, 72 in total. An earlier draft of this report presented the Run A values of Table 5 (for example 74.9% and 0.8% unseen-attack recall) as firm results from a single run. After the repeat on Google Colab (Run B) gave 52.8% and 13.6%, the text was changed to report both runs and to treat threshold-dependent counts as ranges. On the laptop, a repeat of Run A reproduced every saved test score exactly, but that holds only for the same machine; on a different computer the training diverges.

## Appendix B. Checking the flow builder of the application

The application turns packets into the 68 flow features that the models expect with a port of the flow logic of CICFlowMeter (the tool that produced the CSE-CIC-IDS2018 records), including its quirks: a flow ends at the first FIN or after 120 s, single-packet flows are not reported, the first packet is counted twice in the packet-length statistics, the eight flag columns hold first-packet flags in the order of a Java hash map, and so on. These rules were first measured on the training data and then built in. The port was compared with the unmodified February 2018 source of CICFlowMeter's flow classes, run on the same packets through a small packet reader (reports/FLOW_VALIDATION.md has the evidence for each rule, the version choice and the limits).

**Table 9.** Flow builder against CICFlowMeter's flow code, flow by flow (69 values per flow, relative tolerance 1e-9), and packet fields against Wireshark's tshark. Flows are matched by start time, addresses, ports and protocol; no flow existed on only one side.

| Capture kind | Capture | Packets | Flows | Flows with all 69 values equal | IPv4 packets whose fields equal tshark's |
|---|---|---|---|---|---|
| ordinary traffic (real Wi-Fi capture) | ordinary_wifi.pcap | 8,990 | 315 | 315 | 7,213 |
| ordinary traffic (real Wi-Fi capture) | ordinary_wifi2.pcap | 57,164 | 562 | 562 | 54,944 |
| ordinary traffic (real Wi-Fi capture) | ordinary_live.pcap | 12,377 | 272 | 272 | 12,377 |
| synthetic demo capture | demo_synthetic.pcap | 383 | 56 | 56 | 381 |
| randomised stress capture | stress_seed1.pcap | 7,857 | 1,163 | 1,163 | 7,857 |
| randomised stress capture | stress_seed7.pcap | 7,877 | 1,174 | 1,174 | 7,877 |
| randomised stress capture | stress_seed21.pcap | 7,903 | 1,159 | 1,159 | 7,903 |
| **Total** | | **102,551** | **4,701** | **4,701** | **98,552** |

Three ordinary captures were recorded on a Wi-Fi adapter (two with Wireshark's dumpcap, headers only, and one with the application's own Live capture tab) while loading public web pages, downloading files and running DNS lookups and pings. The stress captures are random mixtures of the awkward cases (timeouts, FIN and RST endings, port reuse, ICMP, equal addresses, packets in the same microsecond). In addition, 26 row-level properties that follow from the tool's definitions were tested on 1,500,000 training rows (lowest share satisfying a property 0.9994) and on the 1,149 flows of the ordinary captures (all 1.0000).

**Table 10.** Share of 1149 flows of ordinary traffic flagged by each model at its fixed threshold, and the share of benign held-out lab flows flagged by the same model. The ordinary flows are presumed benign, not labelled; one network, a few minutes.

| Model | Ordinary Wi-Fi traffic | Benign lab test flows |
|---|---|---|
| Supervised autoencoder | 27 of 1149 (2.3%) | 0.35% |
| Supervised autoencoder with decoder | 20 of 1149 (1.7%) | 2.00% |
| One-class autoencoder | 50 of 1149 (4.4%) | 2.94% |
| Unsupervised autoencoder | 795 of 1149 (69.2%) | 7.70% |
| Combined (supervised OR unsupervised) | 803 of 1149 (69.9%) | n/a |

Only 5 of the 1149 flows (0.4%) had a feature outside the training range. The first version of the flow builder, written from the tool's documentation before this check, gave 43.9% of the 964 flows of the first two ordinary captures an out-of-range feature and flagged 100% of them with the unsupervised model and 21.0% with the supervised model; the port removed that difference. The unsupervised model, which ranks flows at chance level in the laboratory (Table 1), does not transfer to this traffic, and the combined detector that contains it inherits its false alarms. Not shown: detection of real attacks on a live network, and agreement with the original packet-reading library (jnetpcap), which could not be run; the packet fields were checked against Wireshark instead.

