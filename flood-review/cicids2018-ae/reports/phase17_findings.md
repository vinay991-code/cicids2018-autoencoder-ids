# Phase 17 - Experiment H: concept-drift recovery with a supervised normality gate (measured results)

Code: `src/nptrain.py` (NumPy autoencoder fine-tuning), `src/phase17_drift_recovery.py` (experiment and checks), `src/summarise_drift.py` (tables and figure), `tests/test_nptrain.py`. Full tables: `results_drift/drift_summary.md`; data: `results_drift/phase17_drift_recovery.json`; figure: `results_drift/phase17_overview.png`; run log: `results_drift/phase17_run.log`. One run, seed 42, settings and success criteria fixed in the script header before the first arm was evaluated. The Phase 7 to 9 and Phase 16 results are untouched and quoted where used.

## Question and design
Can the unsupervised autoencoder be updated after a change of benign traffic without learning the attacks, if a second, offline-trained supervised model must also call a record normal before it may enter the retraining buffer?
- **Stream and baseline: those of Phase 7 and 16.** One-class autoencoder trained on the benign records of 02-14 and 02-15, starting threshold 0.013937, 194 intervals of at most 50,000 records, 8 evaluation days. Each interval is scored and flagged with the model and threshold that exist before it, then used for the update.
- **Arms.** **S** frozen model. **N** adaptive, no supervised gate (a sampled record enters the buffer if the arm itself scores it at or below its threshold, the Phase 7 arm C rule). **Gm** N plus the gate of the existing Phase 4 supervised autoencoder (probability below 0.5). **Gu** N plus the gate of the Phase 6 supervised autoencoder that never saw Bot, DDoS-HOIC, SSH-BruteForce and DoS-Hulk (its scaler refit without them, as in Phase 6). In the stream the unseen types are Bot (03-02), DDoS-HOIC (02-21) and DoS-Hulk (02-16); the known types are DDoS-LOIC-HTTP, DDoS-LOIC-UDP, DoS-SlowHTTPTest, Infiltration, SQLInjection, WebBruteForce and WebXSS.
- **Update rules, identical in all adaptive arms.** Every 3 intervals, 3 epochs of Adam (lr 5e-4, batch 1,024) on the admitted records of the last 12 intervals (at most 10,000 sampled per interval, so at most 120,000) plus 25% replay of clean warm-up records; at least 5,000 admitted records are required; an update is rejected if the mean error on a fixed clean anchor set exceeds 3 times its initial value. Threshold: 0.013937 at the start, then the 99th percentile of the anchor errors under the current model, scaled by a constant fixed at the start (Phase 8 and 9 used the 95th percentile; Phase 7 and 16 used 0.013937, kept here so that S equals Phase 7's static arm).
- **No stream label controls any decision.** The gate probabilities depend on features only; admission uses scores and gate probabilities. A check reruns an arm with all labels replaced by zeros and requires identical parameters, scores and thresholds (it passes).
- **Success criteria, declared in advance.** (a) within-day ROC-AUC on the bulk-shifted days (02-16 and 02-21, the same post-hoc descriptive grouping as in Phase 16) is at least 0.5 and above the frozen model's; (b) on 02-20 and 03-02 it is not more than 0.05 below the frozen model's; (c) overall F1 at the arm's threshold is not lower than the frozen model's. **Recovery time:** intervals from the start of the day until the interval-level ROC-AUC is at least 0.5 for three consecutive intervals.

## Validation before the run (all pass; if one had failed the experiment would not have run)
- NumPy models against the saved PyTorch outputs: Phase 4 supervised probabilities on 200,000 test rows, maximum difference 7.8e-07; Phase 6 unseen-supervised probabilities with the refit scaler, 6.9e-07; the baseline autoencoder forward pass equals the already verified numpy forward exactly; the frozen scores equal the Phase 16 cache to 6.7e-08.
- Gradients of the manual backpropagation against central finite differences in float64, on a random network and on the real baseline weights and stream rows: every probed parameter within |difference| <= 1e-9 + 1e-6 x |value| (largest absolute difference 1.4e-11). A purely relative tolerance fails only on parameters whose true gradient is about 1e-7, where the finite difference's own rounding noise dominates; the check was changed to the mixed tolerance after inspecting this, not to hide a mismatch.
- Adam against a literal transcription of the torch.optim.Adam update and against scikit-learn's independent Adam: identical to 1e-12 and 1e-9. Not identical to PyTorch: the random batch order (numpy generator) and float32 summation order.
- Determinism (two runs identical), label independence, and every update lowered the loss on its own training batch.
- **Independent cross-check:** arm N, which uses the Phase 7 arm C recipe, reproduces Phase 7 arm C's per-day ROC-AUC although the random sequences and the threshold rule differ: 02-16 0.466 (Phase 7: 0.472), 02-21 0.742 (0.745), 02-20 0.430 (0.409), 03-02 0.581 (0.556).

## Results

Table 1 - overall at each arm's own threshold (all 8 days)

| Arm | Precision | Recall | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| S frozen | 0.1347 | 0.0136 | 0.0247 | 0.0122 | 0.9864 |
| N adaptive, no gate | 0.1372 | 0.0136 | 0.0247 | 0.0119 | 0.9864 |
| Gm gate = Phase 4 supervised | 0.1400 | 0.0138 | 0.0251 | 0.0118 | 0.9862 |
| Gu gate = Phase 6 supervised (3 stream types unseen) | 0.1413 | 0.0137 | 0.0250 | 0.0116 | 0.9863 |

Differences to S (moving-block bootstrap over intervals): every interval includes 0; for example Gm F1 +0.0004 [-0.0002, +0.0010], recall +0.0002 [-0.0002, +0.0005], FPR -0.0003 [-0.0013, +0.0004].

Table 2 - within-day ROC-AUC of each arm's scores

| Day (attack types) | S | N | Gm | Gu |
|---|---:|---:|---:|---:|
| 02-16 (Hulk, SlowHTTPTest) | 0.079 | 0.466 | **0.741** | 0.466 |
| 02-21 (HOIC, LOIC-UDP) | 0.013 | 0.742 | 0.751 | 0.741 |
| 02-20 (LOIC-HTTP) | 0.948 | 0.430 | 0.947 | 0.948 |
| 03-02 (Bot) | 0.931 | 0.581 | 0.935 | 0.556 |

Recovery time on 02-16: **Gm 4 intervals** (the interval-level ROC-AUC is 0.76 or higher from the fourth interval, right after the first update, and 0.93 to 0.97 from the seventh); S, N and Gu never reach the rule within the day (N and Gu rise slowly and are at 0.48 and 0.47 in the twelfth interval). On 02-21 the rule cannot be evaluated for any arm: only 4 of the 12 intervals have 1,000 or more benign and attack records. Descriptively, the tenth interval has ROC-AUC 0.64 for Gm, 0.15 for Gu, 0.12 for N and 0.00 for S.

Success criteria: **Gm meets all three; N and Gu meet only (c).**

Table 3 - what the gates block (labels used, diagnostics only)

| Gate | Benign blocked | Attacks blocked (all / test partition only) | Bot | HOIC | Hulk | SlowHTTPTest | LOIC-HTTP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 4 supervised | 0.4% | 90.9% / 84.2% | 99.4% | 99.8% | 87.7% | 0.0% | 99.9% |
| Phase 6 supervised (Bot, HOIC, Hulk unseen) | 0.4% | 50.8% / 47.7% (47.65) | 1.9% | 0.1% | 0.3% | 1.8% | 99.9% |

Attack share in the retraining data (mean / maximum over updates): N 0.123 / 0.493; Gm 0.015 / 0.162; Gu 0.062 / 0.356. All 64 update attempts of every adaptive arm were accepted; the anchor-error ratio stayed between 0.91 and 1.63, so the rejection safeguard (limit 3) never fired in this stream.

## Findings
1. **With the requested gate, drift recovery worked as a ranking recovery, and the ungated version did not keep it.** Gm raised the within-day ROC-AUC on the shifted days from 0.079 and 0.013 to 0.741 and 0.751 and kept the ranking on the attack days (0.947 and 0.935 against 0.948 and 0.931). Ungated adaptation lost it there (0.430 and 0.581), as Phase 7 arm C did. On 02-16 the recovery was fast: one update was enough.
2. **The gate helps only for attack types it knows, and this gate knows too much.** The Phase 4 model was trained on the Phase 2 training partition, which holds labelled records of every day of the stream, so it already knows these campaigns; it blocks 84.2% of the attacks of the test partition, which it never saw, and 92.5% of those of the training partition. A realistic gate would only know past attacks. Gu is the cleaner test: its gate had never seen Hulk, HOIC and Bot and blocked them 0.3%, 0.1% and 1.9% of the time. On the days dominated by those types Gu behaves like the ungated arm (02-16: 0.466, no recovery; 03-02: 0.556, ranking lost) and fails the criteria; it still protects the known LOIC-HTTP campaign of 02-20 (0.948). Same-day knowledge of the attack is what makes Gm succeed; recovery from a benign shift while attacks are present, with attacks the gate has not seen, was not achieved.
3. **02-21 recovers without the gate.** N and Gu reach 0.742 and 0.741, Gm 0.751. The gate matters on 02-16 and for keeping the ranking on 02-20 and 03-02, not on 02-21.
4. **No gain in detection at the arms' thresholds.** Overall F1 is 0.0247 (S), 0.0247 (N), 0.0251 (Gm), 0.0250 (Gu); all intervals of the differences include 0. On the bulk-shifted days recall is 0.0372 (S) and 0.0371 (adaptive arms) and the false-positive rate on benign records falls from 0.0253 to 0.0211 in all three adaptive arms. The threshold stays at about 0.0139 and the detector still flags only about 1.4% of attacks, so a better ranking is not turned into more detections at this operating point. A more sensitive threshold was not tested, because choosing it would be tuning on this stream.
5. **The contamination safeguards were mostly not the limiting factor.** Bounded buffers, the 5,000-record minimum and the update spacing applied as designed, but the anchor check, which is meant to reject harmful updates, never rejected one; contamination was controlled (or not) by the gate alone.

## Limitations
- The requested gate (Phase 4 supervised model) is optimistic by construction (finding 2). A gate trained only on the warm-up days' labels, the realistic temporal case, was not run.
- One stream, one baseline model, one run (seed 42); the bootstrap intervals describe sampling variation over the stream only, and no interval was computed for the ROC-AUC differences, which are large (0.08 to 0.74 on 02-16) but come from a single run.
- The shifted-day grouping is the post-hoc descriptive one of Phase 16. 02-21 is a one-hour snapshot with few measurable intervals, so recovery time is not measurable there. The "drift" is a jump between days, not gradual drift.
- Threshold rule, buffer size, update spacing and tolerance were fixed in advance and not tuned; other settings could behave differently.
- The NumPy trainer matches the PyTorch one in algorithm, validated numerically as above, but not bit for bit (batch order); arm N matches Phase 7 arm C in per-day ROC-AUC to within 0.025.
