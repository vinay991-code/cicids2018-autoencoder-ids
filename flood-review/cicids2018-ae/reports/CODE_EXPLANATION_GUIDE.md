# Code explanation guide: supervised vs unsupervised autoencoders (for presenting the project)

This guide explains what each part of the code does and why, using the real code and the real results, so that you can explain the project yourself. All paths are inside the project folder. Every number comes from the saved result files. Code excerpts are shortened for reading; open the named file for the full version.

## 1. The idea in one minute

A network flow is one row of numbers (duration, packet counts, byte counts, and so on). An **autoencoder** is a small neural network that squeezes a row down to 32 numbers and then tries to rebuild the original row. The gap between the rebuilt row and the original is the **reconstruction error**.

There are two ways to turn this into an attack detector:

- **Unsupervised (reconstruction-based).** Train the network only to rebuild rows. Traffic it has seen a lot of is rebuilt well (small error). Strange traffic is rebuilt badly (large error). Report a row as an attack when the error is above a threshold. The network never sees the attack/benign label while learning.
- **Supervised.** Use the same squeezing part (the encoder) but add a small output layer that says "attack or benign", and train it with the labels. It learns the boundary between the two classes it is shown.

The project holds everything else equal (same encoder, same training rows, same test rows) so that any difference comes from the learning method.

## 2. How the data flows through the scripts

```
raw CSVs  ->  phase1_inspect.py   (look for problems)
          ->  phase2_preprocess.py (clean, remove duplicates, scale, split into train/val/test)
data/processed/X_*.npy  ->  phase3_unsupervised.py  (unsupervised + one-class)
                        ->  phase4_supervised.py    (supervised + supervised with decoder)
results/scores_*.npy    ->  phase5_compare.py       (Table 1, curves, intervals)
                        ->  phase6_unseen.py        (withhold 4 attack types, retrain, Table 5)
saved scores            ->  phase11_hybrid.py       (combine the detectors, Table 6)
                        ->  phase12_thresholds.py   (label-free threshold rules, exploratory)
time-ordered stream     ->  phase7/8/9              (adapt over time, Tables 2-4)
run_seeds.py            ->  phases 3, 4, 6 again for seeds 1, 2, 3 -> aggregate_seeds.py (Tables 1c, 5c)
robust.py               ->  phase13_trimmed.py      (trimmed-loss autoencoder) -> aggregate_trimmed.py
everything              ->  verify_results.py       (recompute the numbers independently)
```

## 3. Preprocessing (src/phase2_preprocess.py)

**Why it matters:** if the same attack appears in both training and test data, the score is inflated. Most of the code in this file exists to prevent that.

**Scaling.** Network features have huge ranges (a byte count can be 0 or 10^9). Each numeric feature is passed through a signed logarithm and then squeezed to [0, 1]:

```python
def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))
```

The minimum and maximum used for the [0, 1] scaling are taken from the **training rows only**, then applied unchanged to validation and test rows. Using test rows to compute them would leak information.

**Block-wise split.** Neighbouring flows of one attack are almost identical, so a random row-by-row split would put near-copies in both training and test. Instead, each day is cut into 60-second blocks and whole blocks go to train, validation or test (70%, 10%, 20%):

```python
block_id = (sec // 86400) * 100_000 + (sec % 86400) // block_sec
```

**Also done here:** records with missing or infinite values are removed; identical feature vectors are removed, and so are vectors that appear with both labels; `Dst Port` is excluded because it almost gives the answer for some attacks (shortcut). Result: 7,438,408 training, 1,173,559 validation and 2,163,852 test records.

## 4. The two networks (src/models.py)

```python
class Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), latent=32):
        self.encoder = mlp([in_dim, *hidden, latent])             # 72 -> 128 -> 64 -> 32
        self.decoder = mlp([latent, *reversed(hidden), in_dim],   # 32 -> 64 -> 128 -> 72
                           last_activation=nn.Sigmoid())
    def forward(self, x):
        return self.decoder(self.encoder(x))
```

The sigmoid at the end keeps the output in [0, 1], matching the scaled input.

```python
class SupervisedAE(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), latent=32, head=16, decoder=False):
        self.encoder = mlp([in_dim, *hidden, latent])             # the SAME encoder
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(latent, head), nn.ReLU(), nn.Linear(head, 1))
        self.decoder = mlp(...) if decoder else None              # optional, for "supervised + decoder"
```

The supervised network reuses the exact encoder and adds a 32 → 16 → 1 head that outputs one number (attack score). Same body, different task.

The anomaly score of the unsupervised models is the mean squared difference between a row and its rebuilt version:

```python
def recon_error(model, X, batch=65536):
    ...
    out.append(((model(xb) - xb) ** 2).mean(dim=1).numpy())
```

## 5. Training (src/training.py)

**Unsupervised loop.** The loss is simply the reconstruction error. Training stops when the validation error has not improved for 5 epochs (early stopping) and the best weights are restored:

```python
loss = ((model(xb) - xb) ** 2).mean()      # rebuild the input
loss.backward(); opt.step()                # Adam, learning rate 0.001
...
if val < best - min_delta: best_state = copy.deepcopy(model.state_dict())
else: bad += 1; if bad >= patience: break
```

No labels appear anywhere in this function.

**Supervised loss.** Attacks are only 12% of the rows, so attack mistakes are weighted more (weight = benign count / attack count = 7.14):

```python
loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
```

With the decoder variant, a reconstruction term is added to this loss.

## 6. Experiment A and B: the two training scripts

**src/phase3_unsupervised.py** trains two models. The only difference between them is which rows they get:

```python
if v == "unsup_mixed":
    pool = np.arange(len(mtr))                                  # every training row, labels not consulted
elif v == "normal_only":
    pool = np.flatnonzero(mtr.label_binary.to_numpy() == 0)     # benign rows only
Xt = sample_rows(Xtr, pool, args.train_rows, args.seed)         # 1,000,000 rows
```

Thresholds come from the **validation** set only:

```python
thr_main = float(np.percentile(s_val, args.percentile))                      # unsup: 95th percentile, no labels
thr_main = float(np.percentile(s_val[yva == 0], args.benign_percentile))     # one-class: 99th percentile of benign
```

**src/phase4_supervised.py** trains the supervised models on exactly the same 1,000,000 rows (same random seed and sampling call) and uses the fixed threshold 0.5 on the predicted probability. The test labels are read only after the test predictions exist.

## 7. Metrics (src/evaluation.py)

```python
pred = score > thr                       # 1 = attack
precision = tp / (tp + fp)               # of the alarms, how many were real
recall    = tp / (tp + fn)               # of the real attacks, how many were caught
fpr       = fp / (fp + tn)               # of the benign flows, how many raised a false alarm
roc_auc_score(y, score)                  # ranking quality, independent of any threshold
```

Worked example from the real confusion matrix of the supervised model on the test set: 221,972 attacks caught, 36,160 missed, 6,577 false alarms, 1,899,143 correct passes. Precision = 221,972 / (221,972 + 6,577) = 0.971; recall = 221,972 / 258,132 = 0.860; false-positive rate = 6,577 / 1,905,720 = 0.0035.

**Why not just accuracy?** 88% of test records are benign, so a detector that never raises an alarm gets 88% accuracy. That is why F1, recall, false-positive rate and ROC-AUC are reported.

**ROC-AUC in plain words:** pick one random attack row and one random benign row; ROC-AUC is the chance that the attack gets the higher score. 0.5 is a coin flip. The unsupervised model's 0.468 means it ranks attacks no better than chance.

## 8. Table 1 and the comparison (src/phase5_compare.py)

This script loads the saved test scores of all four models, recomputes every metric from scratch, checks that they match what phases 3 and 4 printed, and adds **block-bootstrap intervals**: it re-draws the test blocks with replacement 500 times and recomputes the metric each time, so the interval shows how much the result depends on which blocks happened to be in the test set. (It does not show variation from training the network again; each model was trained once.)

## 9. Experiment C: unseen attacks (src/phase6_unseen.py)

Four attack types are removed from everything that is fitted or tuned:

```python
held = ["Bot", "DDoS-HOIC", "SSH-BruteForce", "DoS-Hulk"]
keep_tr = np.flatnonzero(~np.isin(att_tr, held))                    # training rows without them
keep_va = np.flatnonzero(~np.isin(mva.attack_type.to_numpy(), held)) # validation rows without them
```

The scaler minimum/maximum is recomputed on the remaining rows, all three models are retrained, and the test set is split into benign, known attacks and withheld attacks. This answers: "what happens when the attack was never seen?"

## 10. Experiment G: the combination (src/phase11_hybrid.py)

```python
flags6["sup OR unsup_mixed"] = sup6 | s6["unsup_mixed"]      # attack if EITHER detector flags it
```

It uses only scores already saved, with the thresholds already fixed, so nothing is tuned on the test set.

## 11. Streaming experiments (phases 7 to 9), briefly

The data is put in time order (src/stream.py) and processed in chunks of 50,000 records. For each chunk the code **first scores it, then** updates the model, as a real system would. Three ideas were tried: retraining on a buffer of recent traffic that the model itself judged normal (Table 2), filtering what goes into that buffer by confidence (Table 3), and retraining only when a drift detector (ADWIN) signals a change (Table 4). None of them used attack labels to decide anything. None of them gave a reliable improvement. If asked why: the model's own score cannot tell a harmless change in normal traffic from a long attack, so the filters let attacks through (the buffer held 12.3% attacks, the same as the stream).

## 12. The independent check (src/verify_results.py)

It recomputes the headline numbers with scikit-learn directly from the raw saved scores and compares them with what each phase reported. 138 of 138 checks pass (they cover the laptop reference run, Run A; the Colab values and the extra seeds are not covered). It exists so that no table number rests on one script's word.

## 12b. Several seeds (run_seeds.py, src/aggregate_seeds.py)

`run_seeds.py --seeds 1 2 3` repeats phases 3, 4 and 6 with a different random seed (different initial weights, batch order and 1,000,000-row sample; the split and the test set stay the same) and stores each seed in `seeds/seed_<n>/`. Finished seeds are skipped, so an interrupted run can be restarted. `src/aggregate_seeds.py --seeds-dir seeds --extra 42=results` then computes, for every metric, the mean, the standard deviation, the range and a 95% t-interval of the mean over the four runs (1, 2, 3 and the reference seed 42). The interval is `mean ± t(0.975, n-1) · sd / sqrt(n)`; with n = 4 the t value is 3.18, so intervals are wide, and values cut at 0 or 1 are the edge of the possible range. This is training variation, which the bootstrap intervals in Tables 1 and 5 do not include.

## 12c. The trimmed-loss experiment (src/robust.py, src/phase13_trimmed.py, src/aggregate_trimmed.py)

Question: the unsupervised model fails because about 12% of its training rows are attacks. Can a label-free fix help? Trimmed-loss training is an existing idea (not ours): ignore the records with the largest error in each batch, so the model does not learn attacks.

```python
err  = ((model(xb) - xb) ** 2).mean(dim=1)            # error of every record in the batch
k    = int(round(keep * len(err)))                     # keep share falls from 100% to (1 - q) over 10 epochs
loss = torch.topk(err, k, largest=False).values.mean() # average only the k LOWEST errors
```

It uses the same 1,000,000 rows as the unsupervised model, trims q = 10%, 20% and 30% (the true attack share 12.3% is not used), seeds 42, 1 and 2. Labels are read only for the diagnostic columns. `src/verify_trimmed.py` recomputes the numbers independently (48 of 48 checks pass).

Result (report Table 8): no help. ROC-AUC 0.498, 0.473 and 0.470 (all intervals contain 0.5) against 0.902 for benign-only training. The diagnostic explains it: the records with the largest error are mostly benign, so trimming removed fewer attacks (5.6%, 7.4%, 12.3%) than a random trim of the same size would, and the attack share among the rows kept went up (12.9%, 14.2%, 15.4%). Say it as a negative result for an existing method, limited to this variant and dataset, and do not claim that all robust methods would fail.

## 13. What the results say, in your own words

The project was run twice with the same code and seed: on the laptop (Run A) and on Google Colab (Run B). Numbers below are Run A, with Run B in brackets. The multi-seed summary (four laptop seeds, mean with 95% interval) follows the list.

1. **Seen attacks:** supervised is far better (ROC-AUC 0.944 [0.966], F1 0.912 [0.923]) than unsupervised (0.468 [0.528], 0.109 [0.101]).
2. **Why the unsupervised model failed there:** its training data contained about 12% attacks, so it learned to rebuild attacks too. Remove the attacks from training and ROC-AUC jumps to 0.889 [0.879] (one-class). This is consistent with contamination, not proved by itself.
3. **Unseen attacks:** the supervised model ranks them poorly (ROC-AUC 0.648 [0.615]) and flags few: 0.8% [13.6%], because it only knows the boundary it was shown. The unsupervised model ranks them well (0.927 [0.904]) and flags 74.9% [52.8%], but also flags 7.7% of normal traffic. The ranking (ROC-AUC) agrees between runs; the flagged percentages do not, so quote them as ranges.
4. **Neither is better overall.** Supervised is better at known attacks and at keeping false alarms low. Reconstruction-based is the only one that catches new attacks, if its training data is clean and its threshold is chosen well.
5. **Fixing contamination without labels did not work:** trimmed-loss training stayed at chance (ROC-AUC 0.47 to 0.50) because the highest-error training rows were mostly benign.
6. **Combining them** (flag if either flags) catches 87.9% of known and 75.1% of unseen attacks, at 7.9% false alarms on normal traffic. That is a trade-off, not a free win.

**What four seeds add (laptop seeds 1, 2, 3 and 42; report Tables 1c and 5c):**

| Quantity | Result over four seeds | What it means |
|---|---|---|
| Unsupervised ROC-AUC, seen attacks | 0.513 [0.446, 0.581] | interval contains 0.5: no better than chance |
| One-class ROC-AUC | 0.899 [0.882, 0.917] | clearly above chance |
| Supervised ROC-AUC | 0.966 [0.933, 0.998] | clearly above one-class |
| Supervised + decoder ROC-AUC | 0.981 [0.979, 0.983] | overlaps supervised: no proof the decoder helps |
| Unseen-set ROC-AUC, unsupervised | 0.932 [0.926, 0.937] | stable and high |
| Unseen-set ROC-AUC, supervised | 0.700 [0.568, 0.832] | poor and unstable |
| Unseen-attack recall, unsupervised | 0.848, single seeds 0.749 to 0.964 | threshold-dependent, varies |
| One-class F1 | single seeds 0.006 to 0.573 | the threshold is the problem, not the ranking |

Colab (Run B) unsupervised unseen recall, 52.8%, is outside the laptop range, so quote fixed-threshold percentages as ranges.

## 14. Likely questions and short answers

**Why autoencoders?** They give a fair way to compare: the same encoder is used for the supervised and the unsupervised model.

**Why is the supervised result so much better? Is it too good?** On seen attack types it is expected. The split is block-wise and duplicates were removed, but neighbouring blocks of one attack are still similar, so these numbers are optimistic. That is why the unseen-attack experiment was added.

**Why did the unsupervised model do so badly?** Its training data included attacks, so attacks looked normal to it. With benign-only training the ROC-AUC is 0.889.

**Then why is the one-class model's recall only 0.8% if its ROC-AUC is 0.889?** Ranking and the operating point are different things. Its threshold (99th percentile of benign errors) lies above the errors of most attacks, because some ordinary benign flows have very large errors (Figure 3 in the report). Ranking is good; this particular threshold is too strict.

**Did you tune on the test set?** No. Thresholds came from validation data or were fixed (0.5). Test labels were read only after predictions existed.

**Why remove duplicates? It removed 33% of the data.** Identical rows in both training and test would inflate results. The cost is that some attack types shrink to a few dozen distinct flows (FTP-BruteForce: 193,360 to 41).

**Why was Dst Port removed?** It almost identifies some attacks (all FTP-BruteForce uses port 21), so a model could use it as a shortcut instead of learning behaviour.

**Why 60-second blocks?** So that near-identical neighbouring flows stay together in one partition.

**How do you know the numbers are right?** An independent script recomputes them (138 of 138 checks pass for the reference run, and 48 of 48 for the trimmed-loss numbers), and the first Phase 3 run, which had a bug, was discarded and rerun (disclosed in Appendix A of the report).

**How stable are the results?** Experiments A to C were run with four seeds on the laptop and once on Google Colab. Model ordering and ROC-AUC are stable across all of them, but threshold-dependent numbers moved a lot (unsupervised unseen-attack recall 74.9% to 96.4% over the laptop seeds, 52.8% on Colab; supervised 0.8% to 33.3% on the laptop, 13.6% on Colab). So quote ROC-AUC as the main evidence and give fixed-threshold percentages as ranges. Four seeds give wide intervals; close results (for example supervised with and without decoder, whose intervals overlap) cannot be separated. Tables 2 to 4 and 6 are from one run.

**Did you try to fix the contamination problem?** Yes: trimmed-loss training (an existing method) over three seeds and three trimming levels. It did not help; the rows it removed were mostly benign, and the attack share among the rows it kept rose from 12.3% to 12.9 to 15.4%. So the report does not claim a fix, and does not claim that other robust methods fail.

**Is the dataset reliable?** Seven of ten files appear cut at the spreadsheet row limit and the timestamps lack AM/PM. Other studies also report errors in this dataset family [3, 4 in the report], so absolute numbers should be read with caution.

**What is novel here?** No claim of novelty in the field is made, because no full literature search was done. What the project adds is a controlled comparison under identical conditions, a measurement of the contamination effect, a strict unseen-attack protocol, tested adaptation safeguards (including negative results) and the combined-detector test.

**Which would you deploy?** It depends on the goal. For known attacks with labelled data: supervised. To catch new attacks: a reconstruction-based detector trained on clean traffic. For both: the combination, if the team can handle about 8% false alarms on benign traffic. This is a conclusion from one dataset and one run, not a deployment test.

## 15. A 10-minute demonstration plan

1. Show the report (reports/DRAFT_REPORT.docx) Table 1 and Figure 1 (ROC curves): supervised curves are near the top-left, the unsupervised curve hugs the diagonal.
2. Show Figure 3 (error histograms): attacks and benign overlap for the unsupervised model, and separate for the one-class model.
3. Show Table 5 and Figure 4: the reversal on unseen attacks.
4. Show `src/models.py` (about 60 lines) and the two loss lines in `src/training.py`.
5. Run `python src/verify_results.py` (about 1 minute): it prints "138 of 138 checks passed".
6. Close with Table 6 and the decision table (Table 7).
