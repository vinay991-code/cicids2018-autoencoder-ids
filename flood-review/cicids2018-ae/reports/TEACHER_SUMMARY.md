# Project summary for review

**Title:** Comparative Evaluation of Supervised and Unsupervised Autoencoders for Network Anomaly Detection (CSE-CIC-IDS2018)

**What we did.** We trained four autoencoders with the same architecture, the same 1M training flows and the same 2.16M-flow test set. They were an unsupervised AE, a one-class AE, a supervised AE and a supervised AE with a decoder. Data was split by 60-second blocks, scaling was fitted on training data only, and thresholds were fixed before any test label was read.

**Main results (two runs)**
- **Known attacks:** supervised is clearly better. ROC-AUC is 0.944 and 0.966 against 0.468 and 0.528, and F1 is 0.91 to 0.92 against about 0.10.
- **Unseen attack types:** the result reverses. Unsupervised ROC-AUC is 0.927 and 0.904 against 0.648 and 0.615.
- **Cause:** about 12% of the unsupervised model's training data were attacks. With clean training data its ROC-AUC rises from 0.468 to 0.889.
- **Combining both detectors** catches 87.9% of known and 75.1% of unseen attacks, with 7.9% false alarms.

**Fixes tested.** Trimmed-loss training and DAE-LR did not help (ROC-AUC 0.47 to 0.50), because the highest-loss records were mostly benign.

**Adaptation on a 9.5M-record stream.**
- Adaptive baseline, confidence gate and drift detection: no gain.
- Adaptive threshold on a frozen model: fewer false alarms but lower recall, with F1 falling from 0.0247 to 0.0061.
- Retraining behind a supervised normality gate: the ranking recovers (ROC-AUC 0.079 to 0.741 on 16 February), but only when the gate already knows the attacks. Overall F1 is unchanged.

**System.** A dashboard handles PCAP upload and live capture. Its flow builder was checked against CICFlowMeter's own code (4,701 flows equal on 69 features) and against Wireshark.

**Limits, stated openly.** One dataset; the unseen test has only 4 attack types; few seeds; live attack detection was not measured; the drift gate was trained with labels from all days, so it is optimistic.

**Where to check.** The report is `reports/DRAFT_REPORT.docx` and the one-page summary is `reports/REVIEW_CHEAT_SHEET.pdf`. The slides are `reports/CICIDS2018_Presentation.pptx`. The Phase 16 and 17 write-ups are `reports/phase16_findings.md` and `reports/phase17_findings.md`. Everything is on GitHub at vinay991-code/cicids2018-autoencoder-ids.
