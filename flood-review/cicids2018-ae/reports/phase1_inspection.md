# Phase 1 - Dataset inspection report

- Folder: `data\raw`
- CSV files: **10**
- Total records (after dropping 59 embedded header rows): **16,232,943**
- Common feature columns (excluding label and identifiers): **78**
- Columns across files: common=80, union=84

## Class distribution

| Label | Records | Share |
|---|---:|---:|
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

Benign: **13,484,708** (83.07%); attack: **2,748,235**

## Per-file summary

| File | Columns | Rows | First timestamp | Last timestamp | Labels |
|---|---:|---:|---|---|---|
| Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 80 | 1,048,575 | 2018-03-02 01:00:00 | 2018-03-02 12:59:59 | Bot: 286,191, Benign: 762,384 |
| Friday-16-02-2018_TrafficForML_CICFlowMeter.csv | 80 | 1,048,574 | 2018-02-16 01:00:32 | 2018-02-16 12:58:24 | DoS attacks-Hulk: 461,912, DoS attacks-SlowHTTPTest: 139,890, Benign: 446,772 |
| Friday-23-02-2018_TrafficForML_CICFlowMeter.csv | 80 | 1,048,575 | 2018-02-23 01:00:00 | 2018-02-23 12:59:59 | Benign: 1,048,009, Brute Force -Web: 362, Brute Force -XSS: 151, SQL Injection: 53 |
| Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv | 84 | 7,948,748 | 2018-02-20 01:00:00 | 2018-02-20 12:59:59 | DDoS attacks-LOIC-HTTP: 576,191, Benign: 7,372,557 |
| Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv | 80 | 331,100 | 2018-03-01 01:00:00 | 2018-03-01 12:59:59 | Benign: 238,037, Infilteration: 93,063 |
| Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv | 80 | 1,048,575 | 2018-02-15 01:00:00 | 2018-02-15 12:59:59 | Benign: 996,077, DoS attacks-GoldenEye: 41,508, DoS attacks-Slowloris: 10,990 |
| Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv | 80 | 1,048,575 | 1970-01-10 03:04:26 | 2018-02-22 12:59:59 | Benign: 1,048,213, Brute Force -Web: 249, Brute Force -XSS: 79, SQL Injection: 34 |
| Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv | 80 | 1,048,575 | 1970-01-05 03:01:17 | 2018-02-14 12:59:59 | FTP-BruteForce: 193,360, SSH-Bruteforce: 187,589, Benign: 667,626 |
| Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv | 80 | 1,048,575 | 2018-02-21 01:55:46 | 2018-02-21 10:43:21 | DDOS attack-HOIC: 686,012, Benign: 360,833, DDOS attack-LOIC-UDP: 1,730 |
| Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv | 80 | 613,071 | 2018-02-28 01:00:00 | 2018-02-28 12:59:59 | Benign: 544,200, Infilteration: 68,871 |

## Data quality

- Exact duplicate rows: **433,193**
- Duplicate feature vectors (ignoring ids/timestamp/label): **4,251,152**
- Extra (feature vector, label) pairs beyond one label per vector, i.e. label conflicts among identical feature vectors: **22,136**
- Unparseable timestamps: **0**; out-of-order timestamp steps: **5,249,787**
- Constant columns (8): bwd_blk_rate_avg, bwd_byts_b_avg, bwd_pkts_b_avg, bwd_psh_flags, bwd_urg_flags, fwd_blk_rate_avg, fwd_byts_b_avg, fwd_pkts_b_avg
- All-NaN columns: none

### Columns with NaN / inf / non-numeric cells

| Column | NaN/empty | +-inf | Non-numeric text |
|---|---:|---:|---:|
| flow_byts_s | 59,721 | 36,039 | 0 |
| flow_pkts_s | 0 | 95,760 | 0 |

### Columns containing negative values

flow_duration (14), flow_iat_max (3), flow_iat_mean (14), flow_iat_min (15), flow_pkts_s (14), fwd_iat_max (3), fwd_iat_mean (14), fwd_iat_min (15), fwd_iat_tot (14), init_bwd_win_byts (8,255,549), init_fwd_win_byts (4,432,608)

## Feature compatibility across files

- `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` lacks: dst_ip, flow_id, src_ip, src_port
- `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` lacks: dst_ip, flow_id, src_ip, src_port
- `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` lacks: dst_ip, flow_id, src_ip, src_port
- `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` lacks: dst_ip, flow_id, src_ip, src_port
- `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` lacks: dst_ip, flow_id, src_ip, src_port
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` lacks: dst_ip, flow_id, src_ip, src_port
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` lacks: dst_ip, flow_id, src_ip, src_port
- `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` lacks: dst_ip, flow_id, src_ip, src_port
- `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` lacks: dst_ip, flow_id, src_ip, src_port

## Potential leakage

- Identifier/time columns present in all files: timestamp
- Identifier/time columns present in only some files: dst_ip, flow_id, src_ip, src_port
- Labels that occur in exactly one file (each such attack is confined to a single capture day): Bot, DDOS attack-HOIC, DDOS attack-LOIC-UDP, DDoS attacks-LOIC-HTTP, DoS attacks-GoldenEye, DoS attacks-Hulk, DoS attacks-SlowHTTPTest, DoS attacks-Slowloris, FTP-BruteForce, SSH-Bruteforce

**Correction (found in Phase 7):** timestamps are on a 12-hour clock without AM/PM, so the times below are 12 hours early for afternoon events (see `phase1_findings.md`, item 4). Add 12 hours to stored hours 01-07.

| Label | First seen | Last seen |
|---|---|---|
| Benign | 1970-01-05 03:01:17 | 2018-03-02 12:59:59 |
| Bot | 2018-03-02 01:00:00 | 2018-03-02 12:59:59 |
| DoS attacks-Hulk | 2018-02-16 01:45:27 | 2018-02-16 01:48:44 |
| DoS attacks-SlowHTTPTest | 2018-02-16 10:12:14 | 2018-02-16 10:58:08 |
| Brute Force -Web | 2018-02-22 10:13:44 | 2018-02-23 11:02:58 |
| Brute Force -XSS | 2018-02-22 01:51:39 | 2018-02-23 02:10:05 |
| SQL Injection | 2018-02-22 01:52:33 | 2018-02-23 10:53:21 |
| DDoS attacks-LOIC-HTTP | 2018-02-20 01:14:17 | 2018-02-20 11:16:48 |
| Infilteration | 2018-02-28 01:42:00 | 2018-03-01 10:54:59 |
| DoS attacks-GoldenEye | 2018-02-15 09:27:42 | 2018-02-15 10:02:59 |
| DoS attacks-Slowloris | 2018-02-15 11:00:12 | 2018-02-15 11:42:01 |
| FTP-BruteForce | 2018-02-14 10:33:26 | 2018-02-14 12:10:31 |
| SSH-Bruteforce | 2018-02-14 02:01:21 | 2018-02-14 03:32:30 |
| DDOS attack-HOIC | 2018-02-21 02:11:08 | 2018-02-21 02:33:29 |
| DDOS attack-LOIC-UDP | 2018-02-21 10:08:51 | 2018-02-21 10:43:16 |
