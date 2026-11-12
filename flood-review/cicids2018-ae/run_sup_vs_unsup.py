"""One-command re-run of the supervised vs unsupervised autoencoder experiments (Tables 1 and 5).

Runs, in order: Phase 3 (unsupervised + one-class AE), Phase 4 (supervised AEs), Phase 5 (Table 1 comparison),
Phase 6 (unknown attacks) and the Table 5 report. Everything is written to a separate folder (default `rerun/`),
so the verified results in `results/` and `reports/` are never overwritten.

Needs data/processed (made by Phase 2; see TEAMMATE_GUIDE.md) and the packages in requirements.txt plus torch.

    python run_sup_vs_unsup.py              # full run, about 1 hour on a laptop CPU
    python run_sup_vs_unsup.py --quick      # trial run on 100k training rows / 3 epochs (numbers NOT comparable)
    python run_sup_vs_unsup.py --dry-run    # only print the commands
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEEDED = ["X_train.npy", "X_val.npy", "X_test.npy", "meta_train.pkl", "meta_val.pkl", "meta_test.pkl", "scaler.json"]


def preflight():
    missing = [f for f in NEEDED if not (ROOT / "data" / "processed" / f).exists()]
    if missing:
        sys.exit("Missing data/processed files: " + ", ".join(missing) +
                 "\nCopy data/processed from the project owner, or build it from the raw CSVs (Phase 1-2); see TEAMMATE_GUIDE.md.")
    probe = subprocess.run([sys.executable, "-c", "import torch, sklearn, pandas, numpy, matplotlib, psutil"],
                           capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, MPLBACKEND="Agg"))
    if probe.returncode != 0:
        msg = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else "import failed"
        sys.exit(f"Python packages are not usable: {msg}\n"
                 "Install them with:  pip install -r requirements.txt  and the CPU torch build (see README.md).\n"
                 "If the message says 'An Application Control policy has blocked this file' (WinError 4551), Windows Smart App Control\n"
                 "is blocking torch; see 'If it fails' in TEAMMATE_GUIDE.md. Do not work around it by editing security settings casually.")


def run(name, cmd, log):
    print(f"\n=== {name} ===\n$ {' '.join(cmd)}", flush=True)
    # MPLBACKEND=Agg: figures are only written to files; this also overrides the notebook-only backend that Colab/Jupyter set in the environment
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
        code = p.wait()
    print(f"--- {name}: exit {code}, {time.time() - t0:.0f} s, log {log.relative_to(ROOT)}", flush=True)
    if code != 0:
        sys.exit(f"{name} failed (exit {code}); see {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("rerun"), help="output folder (default rerun)")
    ap.add_argument("--quick", action="store_true", help="trial run: 100k training rows, 3 epochs, 50 bootstrap draws")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.resolve() in (ROOT / "results", ROOT / "reports", ROOT):
        sys.exit("Choose an output folder other than results/, reports/ or the project root.")
    res, mod, rep, logs = out / "results", out / "models", out / "reports", out / "logs"
    py = [sys.executable]
    quick3 = ["--train-rows", "100000", "--epochs", "3"] if args.quick else []
    boot = ["--boot", "50"] if args.quick else []
    steps = [
        ("Phase 3 unsupervised + one-class", py + ["src/phase3_unsupervised.py", "--out-dir", str(res), "--model-dir", str(mod)] + quick3, logs / "phase3.log"),
        ("Phase 4 supervised", py + ["src/phase4_supervised.py", "--out-dir", str(res), "--model-dir", str(mod)] + quick3, logs / "phase4.log"),
        ("Phase 5 Table 1 comparison", py + ["src/phase5_compare.py", "--results-dir", str(res), "--model-dir", str(mod), "--report-dir", str(rep),
                                             "--phase3-log", str(logs / "phase3.log"), "--phase4-log", str(logs / "phase4.log")] + boot, logs / "phase5.log"),
        ("Phase 6 unknown attacks", py + ["src/phase6_unseen.py", "--out-dir", str(res), "--model-dir", str(mod)] + quick3 + boot, logs / "phase6.log"),
        ("Phase 6 Table 5", py + ["src/phase6_tables.py", "--results-dir", str(res), "--report-dir", str(rep)], logs / "phase6_tables.log"),
    ]
    if args.dry_run:
        for name, cmd, _ in steps:
            print(f"{name}:\n  {' '.join(cmd)}")
        return
    preflight()
    for d in (res, mod, rep, logs):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name, cmd, log in steps:
        run(name, cmd, log)
    print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} min.\nTable 1: {rep / 'phase5_tables.md'}\nTable 5: {rep / 'phase6_tables.md'}\n"
          f"Figures and JSON: {res}\nCompare with the reference run in results/ and reports/ (numbers should be close, not identical).")
    if args.quick:
        print("QUICK MODE: these numbers come from a tiny training run and must not be reported.")


if __name__ == "__main__":
    main()
