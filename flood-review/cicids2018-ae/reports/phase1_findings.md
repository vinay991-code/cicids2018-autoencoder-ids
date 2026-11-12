# Phase 1 - findings and issues to handle in Phase 2

All numbers come from `reports/phase1_inspection.md` (full run over the 10 CSVs in `data/raw`, all matching the bucket's byte sizes) or from the follow-up checks noted below.

## Headline numbers (measured)
- 16,232,943 records, 10 files, 59 repeated header rows dropped (Friday-16: 1, Thursday-01-03: 25, Wednesday-28-02: 33).
- 80 columns in 9 files; `Thuesday-20-02` has 84 (adds `Flow ID`, `Src IP`, `Src Port`, `Dst IP`). 78 model-feature candidates are common to all files.
- Benign 13,484,708 (83.07%); attack 2,748,235 across 14 attack labels (87 SQL Injection ... 686,012 DDOS attack-HOIC).
- No unparseable timestamps. No all-NaN columns. 8 constant columns.
- NaN: `flow_byts_s` 59,721. Infinity: `flow_byts_s` 36,039, `flow_pkts_s` 95,760.
- Exact duplicate rows 433,193; duplicate feature vectors (ignoring timestamp and label) 4,251,152 (26.2% of rows); 22,136 extra (vector,label) pairs, i.e. identical features carrying different labels.

## Issues found (each must be handled in Phase 2)
1. **Probable row-limit truncation.** Seven files contain exactly 1,048,575 records (1,048,574 plus 1 header row in Friday-16), which equals the spreadsheet row limit (1,048,576) minus the header. This is a strong sign they were cut at that limit; I cannot see what is missing. Consequence: per-day attack counts for those days may be incomplete. The three other files (Thuesday-20-02: 7,948,748; Thursday-01-03: 331,100; Wednesday-28-02: 613,071) are not at the limit.
2. **One file is half the dataset.** `Thuesday-20-02` holds 7,948,748 records (49.0%), 7,372,557 benign and 576,191 LOIC-HTTP. It is the only file with identifier columns. These must be dropped, otherwise features differ between files.
3. **Rows are not in time order.** 5,249,787 backwards timestamp steps. Example, Wednesday-14: row 0 is 08:31:01, row 500,000 is 10:26:39, row 1,000,000 is 08:35:21. Sort by timestamp before any chronological split.
4. **Timestamps only cover hours 01:00-12:59** in every file (Wednesday-14 has hours 1-5 and 7-12; raw strings are `dd/mm/yyyy HH:MM:SS` with no AM/PM text). *Explained in Phase 7:* the stored time is a 12-hour clock whose AM/PM marker was lost, so afternoon events appear 12 hours early. Checked against the official attack schedule (Table 2 at https://www.unb.ca/cic/datasets/ids-2018.html): DoS-Hulk is scheduled 13:45-14:19 and appears at 01:45-01:48, SSH-BruteForce 14:01-15:31 appears at 02:01-03:32, HOIC 14:05-15:05 at 02:11-02:33, while morning events (FTP-BruteForce, GoldenEye, Slowloris, SlowHTTPTest) appear at their scheduled times. Adding 12 hours to stored hours 1-7 reproduces the schedule for every attack window (Infiltration, Bot, web attacks, SQL injection included) to within about two minutes. The true timeline is 08:00-17:59 per day. The attack-window table in `phase1_inspection.md` is shown on the stored clock and is therefore 12 hours early for afternoon windows. Two days are almost single-hour snapshots (Feb 16: 99.99% of records at 13:xx; Feb 21: 99.5% at 14:xx), consistent with the row-limit truncation in item 1.
5. **Corrupted timestamps.** Rows dated January 1970: 9 in Thursday-22, 5 in Wednesday-14, all Benign with `Dst Port` 0. Drop them or exclude from ordering.
6. **Attacks sit in narrow time windows**, e.g. DoS-Hulk 2018-02-16 01:45:27-01:48:44, DDOS-HOIC 02-21 02:11:08-02:33:29. A chronological split would put an entire attack type into a single partition. The split must be scenario-aware (see below).
7. **Label spelling is inconsistent**: `Benign`, `DDOS attack-HOIC` vs `DDoS attacks-LOIC-HTTP`, `Infilteration`, `Brute Force -Web`. Normalise case-insensitively; keep originals.
8. **Sentinel/negative values**: `init_fwd_win_byts` 4,432,608 and `init_bwd_win_byts` 8,255,549 negative cells; a few negative durations/IATs (<=15). Keep -1 sentinels; treat negative durations as invalid.
9. **Label conflicts**: 22,136 identical feature pairs with different labels. Deduplicate on features and decide how to resolve (drop or keep majority) before training.
10. **Not yet examined**: whether `Dst Port` / `Protocol` act as shortcuts for particular attacks; the full list of attack types per day against the dataset documentation.

## Proposed split (to confirm at start of Phase 2)
Scenario-aware by capture day, since attacks occupy short windows and chronological order within the files is unreliable: train/validation/test are formed from distinct days (benign from all days), with whole attack types held out for Experiment C. Exact assignment will be decided after seeing per-day counts after de-duplication.
