# Verification of reported numbers

Recomputed from the raw saved artefacts with scikit-learn / numpy and compared with what each phase reported.

| Check | Result | Largest absolute difference |
|---|---|---:|
| Phase 2: train rows | PASS | 0.00e+00 |
| Phase 2: val rows | PASS | 0.00e+00 |
| Phase 2: test rows | PASS | 0.00e+00 |
| Phase 2: partitions sum to the de-duplicated total | PASS | 0.00e+00 |
| Phase 2: no identical vectors across partitions | PASS | 0.00e+00 |
| Phase 1 vs Phase 2: records read | PASS | 0.00e+00 |
| Phase 4: test label hash | PASS | 0.00e+00 |
| Table 1 unsup_mixed: accuracy | PASS | 0.00e+00 |
| Table 1 unsup_mixed: precision | PASS | 0.00e+00 |
| Table 1 unsup_mixed: recall | PASS | 0.00e+00 |
| Table 1 unsup_mixed: f1 | PASS | 1.39e-17 |
| Table 1 unsup_mixed: fpr | PASS | 0.00e+00 |
| Table 1 unsup_mixed: ROC-AUC | PASS | 0.00e+00 |
| Table 1 unsup_mixed: PR-AUC | PASS | 0.00e+00 |
| Table 1 normal_only: accuracy | PASS | 0.00e+00 |
| Table 1 normal_only: precision | PASS | 0.00e+00 |
| Table 1 normal_only: recall | PASS | 0.00e+00 |
| Table 1 normal_only: f1 | PASS | 0.00e+00 |
| Table 1 normal_only: fpr | PASS | 0.00e+00 |
| Table 1 normal_only: ROC-AUC | PASS | 0.00e+00 |
| Table 1 normal_only: PR-AUC | PASS | 0.00e+00 |
| Table 1 supervised: accuracy | PASS | 0.00e+00 |
| Table 1 supervised: precision | PASS | 0.00e+00 |
| Table 1 supervised: recall | PASS | 0.00e+00 |
| Table 1 supervised: f1 | PASS | 1.11e-16 |
| Table 1 supervised: fpr | PASS | 0.00e+00 |
| Table 1 supervised: ROC-AUC | PASS | 0.00e+00 |
| Table 1 supervised: PR-AUC | PASS | 0.00e+00 |
| Table 1 supervised_recon: accuracy | PASS | 0.00e+00 |
| Table 1 supervised_recon: precision | PASS | 0.00e+00 |
| Table 1 supervised_recon: recall | PASS | 0.00e+00 |
| Table 1 supervised_recon: f1 | PASS | 1.11e-16 |
| Table 1 supervised_recon: fpr | PASS | 0.00e+00 |
| Table 1 supervised_recon: ROC-AUC | PASS | 0.00e+00 |
| Table 1 supervised_recon: PR-AUC | PASS | 0.00e+00 |
| Table 5 unsup_mixed: known-attack recall | PASS | 0.00e+00 |
| Table 5 unsup_mixed: unseen recall | PASS | 0.00e+00 |
| Table 5 unsup_mixed: benign FPR | PASS | 0.00e+00 |
| Table 5 unsup_mixed: unseen ROC-AUC | PASS | 0.00e+00 |
| Table 5 unsup_mixed: known ROC-AUC | PASS | 0.00e+00 |
| Table 5 normal_only: known-attack recall | PASS | 0.00e+00 |
| Table 5 normal_only: unseen recall | PASS | 0.00e+00 |
| Table 5 normal_only: benign FPR | PASS | 0.00e+00 |
| Table 5 normal_only: unseen ROC-AUC | PASS | 0.00e+00 |
| Table 5 normal_only: known ROC-AUC | PASS | 0.00e+00 |
| Table 5 supervised: known-attack recall | PASS | 0.00e+00 |
| Table 5 supervised: unseen recall | PASS | 0.00e+00 |
| Table 5 supervised: benign FPR | PASS | 0.00e+00 |
| Table 5 supervised: unseen ROC-AUC | PASS | 0.00e+00 |
| Table 5 supervised: known ROC-AUC | PASS | 0.00e+00 |
| Phase 11 seen sup OR unsup_mixed: tp,fp,fn,tn | PASS | 0.00e+00 |
| Phase 11 unseen sup OR unsup_mixed: unseen recall | PASS | 0.00e+00 |
| Phase 11 unseen sup OR unsup_mixed: known recall | PASS | 0.00e+00 |
| Phase 11 unseen sup OR unsup_mixed: benign FPR | PASS | 0.00e+00 |
| Phase 11 seen sup OR normal_only: tp,fp,fn,tn | PASS | 0.00e+00 |
| Phase 11 unseen sup OR normal_only: unseen recall | PASS | 0.00e+00 |
| Phase 11 unseen sup OR normal_only: known recall | PASS | 0.00e+00 |
| Phase 11 unseen sup OR normal_only: benign FPR | PASS | 0.00e+00 |
| Phase 12 seen unsup_mixed: natural prevalence | PASS | 0.00e+00 |
| Phase 12 seen unsup_mixed: F-primary F1 | PASS | 1.39e-17 |
| Phase 12 seen unsup_mixed: B-10% F1 | PASS | 1.39e-17 |
| Phase 12 seen unsup_mixed: O-prev F1 | PASS | 0.00e+00 |
| Phase 12 seen unsup_mixed: flag-all F1 | PASS | 0.00e+00 |
| Phase 12 unseen unsup_mixed: natural prevalence | PASS | 0.00e+00 |
| Phase 12 unseen unsup_mixed: F-primary F1 | PASS | 0.00e+00 |
| Phase 12 unseen unsup_mixed: B-10% F1 | PASS | 0.00e+00 |
| Phase 12 unseen unsup_mixed: O-prev F1 | PASS | 2.78e-17 |
| Phase 12 unseen unsup_mixed: flag-all F1 | PASS | 0.00e+00 |
| Phase 12 seen normal_only: natural prevalence | PASS | 0.00e+00 |
| Phase 12 seen normal_only: F-primary F1 | PASS | 0.00e+00 |
| Phase 12 seen normal_only: B-10% F1 | PASS | 5.55e-17 |
| Phase 12 seen normal_only: O-prev F1 | PASS | 0.00e+00 |
| Phase 12 seen normal_only: flag-all F1 | PASS | 0.00e+00 |
| Phase 12 unseen normal_only: natural prevalence | PASS | 0.00e+00 |
| Phase 12 unseen normal_only: F-primary F1 | PASS | 0.00e+00 |
| Phase 12 unseen normal_only: B-10% F1 | PASS | 0.00e+00 |
| Phase 12 unseen normal_only: O-prev F1 | PASS | 0.00e+00 |
| Phase 12 unseen normal_only: flag-all F1 | PASS | 0.00e+00 |
| Stream: all records present once | PASS | 0.00e+00 |
| Stream: sorted by corrected time | PASS | 0.00e+00 |
| Phase 7 A_static: records in the stream | PASS | 0.00e+00 |
| Phase 7 A_static: precision | PASS | 0.00e+00 |
| Phase 7 A_static: recall | PASS | 0.00e+00 |
| Phase 7 A_static: FPR | PASS | 0.00e+00 |
| Phase 7 A_static: attack records equal across arms | PASS | 0.00e+00 |
| Phase 7 B_adaptive_threshold: records in the stream | PASS | 0.00e+00 |
| Phase 7 B_adaptive_threshold: precision | PASS | 0.00e+00 |
| Phase 7 B_adaptive_threshold: recall | PASS | 0.00e+00 |
| Phase 7 B_adaptive_threshold: FPR | PASS | 0.00e+00 |
| Phase 7 B_adaptive_threshold: attack records equal across arms | PASS | 0.00e+00 |
| Phase 7 C_adaptive_baseline: records in the stream | PASS | 0.00e+00 |
| Phase 7 C_adaptive_baseline: precision | PASS | 0.00e+00 |
| Phase 7 C_adaptive_baseline: recall | PASS | 0.00e+00 |
| Phase 7 C_adaptive_baseline: FPR | PASS | 0.00e+00 |
| Phase 7 C_adaptive_baseline: attack records equal across arms | PASS | 0.00e+00 |
| Phase 7: attack records equal the stream's | PASS | 0.00e+00 |
| Phase 8 S_static: records in the stream | PASS | 0.00e+00 |
| Phase 8 S_static: precision | PASS | 0.00e+00 |
| Phase 8 S_static: recall | PASS | 0.00e+00 |
| Phase 8 S_static: FPR | PASS | 0.00e+00 |
| Phase 8 S_static: attack records equal across arms | PASS | 0.00e+00 |
| Phase 8 U_unrestricted: records in the stream | PASS | 0.00e+00 |
| Phase 8 U_unrestricted: precision | PASS | 0.00e+00 |
| Phase 8 U_unrestricted: recall | PASS | 0.00e+00 |
| Phase 8 U_unrestricted: FPR | PASS | 0.00e+00 |
| Phase 8 U_unrestricted: attack records equal across arms | PASS | 0.00e+00 |
| Phase 8 M_margin_only: records in the stream | PASS | 0.00e+00 |
| Phase 8 M_margin_only: precision | PASS | 0.00e+00 |
| Phase 8 M_margin_only: recall | PASS | 0.00e+00 |
| Phase 8 M_margin_only: FPR | PASS | 0.00e+00 |
| Phase 8 M_margin_only: attack records equal across arms | PASS | 0.00e+00 |
| Phase 8 G_gated: records in the stream | PASS | 0.00e+00 |
| Phase 8 G_gated: precision | PASS | 0.00e+00 |
| Phase 8 G_gated: recall | PASS | 0.00e+00 |
| Phase 8 G_gated: FPR | PASS | 0.00e+00 |
| Phase 8 G_gated: attack records equal across arms | PASS | 0.00e+00 |
| Phase 8: attack records equal the stream's | PASS | 0.00e+00 |
| Phase 9 A_no_detection: records in the stream | PASS | 0.00e+00 |
| Phase 9 A_no_detection: precision | PASS | 0.00e+00 |
| Phase 9 A_no_detection: recall | PASS | 0.00e+00 |
| Phase 9 A_no_detection: FPR | PASS | 0.00e+00 |
| Phase 9 A_no_detection: attack records equal across arms | PASS | 0.00e+00 |
| Phase 9 A2_periodic: records in the stream | PASS | 0.00e+00 |
| Phase 9 A2_periodic: precision | PASS | 0.00e+00 |
| Phase 9 A2_periodic: recall | PASS | 0.00e+00 |
| Phase 9 A2_periodic: FPR | PASS | 0.00e+00 |
| Phase 9 A2_periodic: attack records equal across arms | PASS | 0.00e+00 |
| Phase 9 B_drift_adaptive: records in the stream | PASS | 0.00e+00 |
| Phase 9 B_drift_adaptive: precision | PASS | 0.00e+00 |
| Phase 9 B_drift_adaptive: recall | PASS | 0.00e+00 |
| Phase 9 B_drift_adaptive: FPR | PASS | 0.00e+00 |
| Phase 9 B_drift_adaptive: attack records equal across arms | PASS | 0.00e+00 |
| Phase 9 C_drift_confirmed: records in the stream | PASS | 0.00e+00 |
| Phase 9 C_drift_confirmed: precision | PASS | 0.00e+00 |
| Phase 9 C_drift_confirmed: recall | PASS | 0.00e+00 |
| Phase 9 C_drift_confirmed: FPR | PASS | 0.00e+00 |
| Phase 9 C_drift_confirmed: attack records equal across arms | PASS | 0.00e+00 |
| Phase 9: attack records equal the stream's | PASS | 0.00e+00 |

**138 of 138 checks passed.**
