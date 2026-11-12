# How to run the supervised vs unsupervised autoencoder experiments (CICIDS2018)

This runs the code behind **Table 1** (supervised vs unsupervised autoencoder) and **Table 5** (known vs unseen attacks). Everything is written to a new folder `rerun/`, so the reference results in `results/` and `reports/` are never overwritten.

## 1. What you need

| Item | Detail |
|---|---|
| Python | 3.13 was used and tested; other 3.10+ versions should work but were not tested |
| RAM / disk | the reference run used a 15.6 GB RAM machine; I have not measured the minimum. About 4 GB of disk for the processed data, under 1 GB for outputs |
| Time | about 1 hour on a laptop CPU (times from the reference machine: Phase 3 ~15 min, Phase 4 ~15 min, Phase 5 ~2 min, Phase 6 ~25 min) |
| Data | the processed arrays in `data/processed` (next section) |

## 2. Get the data (not included in the code zip: it is too large)

**Option A (easiest): copy the processed data.** Ask the project owner for these files and put them in `data/processed/` inside the project folder (about 3.8 GB):
`X_train.npy`, `X_val.npy`, `X_test.npy`, `meta_train.pkl`, `meta_val.pkl`, `meta_test.pkl`, `scaler.json`, `phase2_summary.json`.

**Option B: rebuild from the raw files (about 40 minutes plus the download).**
1. Download the 10 "Processed Traffic Data for ML Algorithms" CSVs (6.41 GB) from the official bucket listed at https://www.unb.ca/cic/datasets/ids-2018.html into `data/raw/`. Expected byte sizes are in `data/expected_sizes.json`.
2. Run:
```bash
python src/phase1_inspect.py --data-dir data/raw --out-dir reports
python src/phase2_preprocess.py --data-dir data/raw --out-dir data --report-dir reports
```
Phase 2 needs about 13 GB of free disk for its intermediate and processed files, on top of the 6.4 GB of raw data.

## 3. Set up in VS Code

1. Install VS Code and the **Python** extension. Use **File > Open Folder** and open the project folder (the one that contains `src/` and `run_sup_vs_unsup.py`).
2. Open a terminal: **Terminal > New Terminal**.
3. Create an environment and install the packages:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```
(On macOS/Linux use `source .venv/bin/activate` instead of the second line.)
4. In VS Code press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and pick the `.venv` one.

## 4. Run

First check the commands without running anything:
```bash
python run_sup_vs_unsup.py --dry-run
```
Then a quick trial (a few minutes; the numbers are NOT valid results, it only proves everything works):
```bash
python run_sup_vs_unsup.py --quick --out rerun_quick
```
Then the full run:
```bash
python run_sup_vs_unsup.py
```
It stops with a message if something is missing or a step fails. Keep the terminal open until it prints "All steps finished".

## 5. What you get and how to read it

| Output (in `rerun/`) | Content |
|---|---|
| `reports/phase5_tables.md` | **Table 1**: all four models, same test set |
| `reports/phase6_tables.md` | **Table 5**: known vs unseen attacks |
| `results/*.png` | figures (ROC/PR curves, confusion matrices, loss curves, unseen-attack detection) |
| `results/*.json` | all metrics, as numbers |
| `logs/*.log` | the training logs of every step |

Compare with the reference run in `reports/phase5_tables.md` and `reports/phase6_tables.md`. All models use seed 42. On the project owner's laptop a full re-run reproduced the reference numbers exactly (all saved test scores identical), but on a different computer, Python or torch version, or Colab, expect numbers **close to, not necessarily identical to**, the reference (CPU arithmetic is not exactly repeatable across machines). The conclusions to look for:
- Table 1: the supervised models have the highest ROC-AUC (reference 0.944 and 0.980) and F1 (0.912, 0.893); the unsupervised one is near chance (reference ROC-AUC 0.468).
- Table 5: on the four withheld attack types the supervised model flags almost none (reference 0.8%), while the unsupervised one flags most (reference 74.9%).

If your numbers differ by a small amount, that is expected. If the ordering of the models changes, tell the project owner before using the numbers.

## 6. If it fails

- **`An Application Control policy has blocked this file` (WinError 4551) when importing torch.** Windows Smart App Control is blocking PyTorch's unsigned DLL. This happened on the project owner's laptop on 2026-11-06, although the same install ran the whole project on 2026-11-02; what changed was not investigated. Do not edit security settings casually: Smart App Control cannot be switched back on once it is off. Use a machine or account without that policy, WSL, or Google Colab: open `colab_sup_vs_unsup.ipynb` in Colab (https://colab.research.google.com, File > Upload notebook) and follow its first cell. It needs the code zip and the 8 `data/processed` files in a Google Drive folder `cicids2018-ae-share`. The notebook was syntax-checked but has not been run in Colab yet.
- **`Missing data/processed files`**: finish step 2.
- **Out of memory**: close other programs. The scripts read the big arrays from disk in pieces, but the training sample and test scoring still need several GB.
- **Anything else**: send the last 30 lines of the failing `rerun/logs/*.log` file.

## 7. Where the code is

| File | What it does |
|---|---|
| `run_sup_vs_unsup.py` | runs everything below in order |
| `src/models.py` | the autoencoder (`Autoencoder`) and the supervised autoencoder (`SupervisedAE`) |
| `src/training.py` | training loops with early stopping |
| `src/phase3_unsupervised.py` | unsupervised + one-class training and testing |
| `src/phase4_supervised.py` | supervised training and testing |
| `src/phase5_compare.py` | Table 1, confidence intervals, comparison figures |
| `src/phase6_unseen.py`, `src/phase6_tables.py` | unknown-attack experiment and Table 5 |
| `src/evaluation.py`, `src/data.py`, `src/plots.py` | shared helpers |
| `src/phase1_inspect.py`, `src/phase2_preprocess.py` | dataset inspection and preprocessing (Option B only) |
| `src/verify_results.py` | independent check of the reference numbers (needs the reference `results/` folder) |
| `reports/FINAL_REPORT.md`, `reports/DRAFT_REPORT.md` | reference results and the report draft |

The adaptive, confidence-aware and drift experiments (Phases 7 to 9) are not part of this runner; their commands are in `README.md`.

## The dashboard, Wireshark tools and tests (added later)

* `pip install -r requirements-app.txt` (and `pip install pytest httpx` for the tests), then `python run_app.py` and open http://127.0.0.1:8000. The Results, Wireshark / PCAP and Models tabs work from the package alone (the trained models, the scaler and the results are included).
* The **Live replay** tab needs `data/processed/X_test.npy` and `meta_test.pkl`, which are too large for this package: create them with the Phase 2 command in README.md, or copy them from the project folder.
* `python -m pytest tests -q` runs 44 tests; those that need the large data files, the CICFlowMeter reference build or a live capture driver (Npcap) are skipped without them.
* The flow builder is checked against CICFlowMeter's own flow code, the training data and Wireshark: read `reports/FLOW_VALIDATION.md`; to repeat it, `python tools/cicflowmeter_ref/build_reference.py` (git, a JDK, internet) and then `python -m pytest tests/test_flow_parity.py -q`.
* Read `APP_README.md` first (what was tested, what was not, and the cautions for live capture).
