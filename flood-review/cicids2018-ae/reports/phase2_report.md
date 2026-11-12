# Phase 2 - preprocessing and split report

Records read (header rows removed): **16,232,943**; after cleaning: **16,137,168**; after de-duplication: **10,775,819**.

## Rows removed, by reason and attack type

| Reason | Attack type | Rows |
|---|---|---:|
| bad_or_non2018_timestamp | Benign | 14 |
| duplicate_vector | Benign | 3,575,396 |
| duplicate_vector | Bot | 134,360 |
| duplicate_vector | DDoS-HOIC | 487,150 |
| duplicate_vector | DDoS-LOIC-HTTP | 827 |
| duplicate_vector | DoS-GoldenEye | 102 |
| duplicate_vector | DoS-Hulk | 316,713 |
| duplicate_vector | DoS-SlowHTTPTest | 139,835 |
| duplicate_vector | DoS-Slowloris | 1,017 |
| duplicate_vector | FTP-BruteForce | 193,313 |
| duplicate_vector | Infiltration | 2,469 |
| duplicate_vector | SQLInjection | 8 |
| duplicate_vector | SSH-BruteForce | 93,548 |
| duplicate_vector | WebBruteForce | 57 |
| duplicate_vector | WebXSS | 2 |
| label_conflict_vector | Benign | 343,223 |
| label_conflict_vector | Bot | 7,494 |
| label_conflict_vector | DDoS-HOIC | 2 |
| label_conflict_vector | DoS-Slowloris | 117 |
| label_conflict_vector | Infiltration | 65,709 |
| label_conflict_vector | SQLInjection | 1 |
| label_conflict_vector | WebBruteForce | 3 |
| label_conflict_vector | WebXSS | 3 |
| nan_or_inf | Benign | 94,459 |
| nan_or_inf | FTP-BruteForce | 6 |
| nan_or_inf | Infiltration | 1,295 |
| negative_value | Benign | 1 |

## Partitions

| Partition | Rows | Benign | Attack | Attack share | Blocks | Days |
|---|---:|---:|---:|---:|---:|---|
| train | 7,438,408 | 6,525,735 | 912,673 | 12.27% | 3,459 | 10 days |
| val | 1,173,559 | 1,040,160 | 133,399 | 11.37% | 496 | 10 days |
| test | 2,163,852 | 1,905,720 | 258,132 | 11.93% | 991 | 10 days |

### Attack types per partition

| Type | Train | Val | Test |
|---|---:|---:|---:|
| Benign | 6,525,735 | 1,040,160 | 1,905,720 |
| Bot | 100,429 | 14,744 | 29,164 |
| FTP-BruteForce | 15 | 1 | 25 |
| SSH-BruteForce | 66,045 | 9,336 | 18,660 |
| DoS-GoldenEye | 27,849 | 4,135 | 9,422 |
| DoS-Slowloris | 6,853 | 1,041 | 1,962 |
| DoS-Hulk | 78,892 | 20,798 | 45,509 |
| DoS-SlowHTTPTest | 16 | 1 | 38 |
| DDoS-LOIC-HTTP | 395,986 | 73,540 | 105,838 |
| DDoS-LOIC-UDP | 1,270 | 145 | 315 |
| DDoS-HOIC | 170,229 | 693 | 27,938 |
| Infiltration | 64,501 | 8,873 | 19,087 |
| WebBruteForce | 385 | 61 | 105 |
| WebXSS | 153 | 22 | 50 |
| SQLInjection | 50 | 9 | 19 |

Identical model-input vectors shared between partitions: {'train&val': 0, 'train&test': 0, 'val&test': 0} (must be 0).

## Model input

- Dimension: **72** (numeric features after signed log1p + min-max fit on train, plus one-hot protocol [0, 6, 17] and 'other')
- Constant (in train) columns dropped: bwd_psh_flags, bwd_urg_flags, fwd_byts_b_avg, fwd_pkts_b_avg, fwd_blk_rate_avg, bwd_byts_b_avg, bwd_pkts_b_avg, bwd_blk_rate_avg
- Excluded from model input: timestamp, label, `dst_port` (kept in meta for analysis only)
- No clipping of val/test values to [0,1]: out-of-range values are kept on purpose.

## `dst_port` shortcut check (train partition)

| Attack type | Most common dst_port | Share of type on it | Share of benign on it |
|---|---:|---:|---:|
| Bot | 8080 | 96.60% | 0.00% |
| DDoS-HOIC | 80 | 100.00% | 13.35% |
| DDoS-LOIC-HTTP | 80 | 100.00% | 13.35% |
| DDoS-LOIC-UDP | 80 | 100.00% | 13.35% |
| DoS-GoldenEye | 80 | 100.00% | 13.35% |
| DoS-Hulk | 80 | 100.00% | 13.35% |
| DoS-SlowHTTPTest | 21 | 100.00% | 0.00% |
| DoS-Slowloris | 80 | 100.00% | 13.35% |
| FTP-BruteForce | 21 | 100.00% | 0.00% |
| Infiltration | 53 | 29.06% | 21.58% |
| SQLInjection | 80 | 100.00% | 13.35% |
| SSH-BruteForce | 22 | 100.00% | 0.09% |
| WebBruteForce | 80 | 74.81% | 13.35% |
| WebXSS | 80 | 98.69% | 13.35% |

## Split strategy

**Note added in Phase 7:** the time blocks below were formed on the timestamp as stored in the CSVs, which is a 12-hour clock without AM/PM (afternoon events appear 12 hours early). A "60-second block" is therefore one clock-minute that can contain records from two real periods of the same day (morning and afternoon, 12 hours apart). This does not create leakage (blocks are never split across partitions, and the partitions share no identical vectors), but it means the blocks are not strictly consecutive real-time minutes. The conclusions of Phases 3-6 are unaffected; their wording about time should be read this way.

Each capture day is cut into 60-second time blocks. Whole blocks are assigned to train/val/test (70/10/20% of blocks) at random with a fixed seed, stratified by the set of attack types present in the block. Rows of one block never appear in two partitions. Labels are used only for this stratification. Scaler, constant-column list and protocol categories are fitted on the training partition only. Limitations: neighbouring blocks of the same attack are similar, and flows can span a block boundary, so the in-distribution test is optimistic; Experiment C (unseen attacks) is the stronger generalisation test.
