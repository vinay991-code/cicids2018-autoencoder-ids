"""Repeat the supervised vs unsupervised experiments with several random seeds, to get real intervals over training randomness.

For every seed this runs Phase 3 (unsupervised + one-class), Phase 4 (supervised) and Phase 6 (withheld attack types) with that seed
(model initialisation, batch order and the 1,000,000 training rows all change; the train/validation/test split from Phase 2 does not).
Each seed writes to <out>/seed_<n>/ and is skipped if its result files already exist, so an interrupted run can simply be started again.
Afterwards run  python src/aggregate_seeds.py  to get means, standard deviations and 95% intervals.

    python run_seeds.py --seeds 1 2 3 4          # about 65 minutes per seed on Google Colab (CPU)
    python run_seeds.py --seeds 1 --quick        # trial: 100k training rows, 3 epochs (numbers are NOT valid results)
    python run_seeds.py --seeds 1 2 --dry-run    # only print the commands
"""
import argparse
import sys
import time
from pathlib import Path

from run_sup_vs_unsup import ROOT, preflight, run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="seeds to run, for example 1 2 3 4 (42 is the reference seed)")
    ap.add_argument("--out", type=Path, default=Path("seeds"), help="output folder (default seeds)")
    ap.add_argument("--quick", action="store_true", help="trial run: 100k training rows, 3 epochs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.resolve() in (ROOT / "results", ROOT / "reports", ROOT):
        sys.exit("Choose an output folder other than results/, reports/ or the project root.")
    quick = ["--train-rows", "100000", "--epochs", "3"] if args.quick else []
    plan = []
    for s in args.seeds:
        d = out / f"seed_{s}"
        res, mod, logs = d / "results", d / "models", d / "logs"
        py = [sys.executable]
        base = ["--seed", str(s), "--out-dir", str(res), "--model-dir", str(mod)]
        plan.append((s, res, mod, logs, [
            ("Phase 3", py + ["src/phase3_unsupervised.py"] + base + quick, "phase3_results.json"),
            ("Phase 4", py + ["src/phase4_supervised.py"] + base + quick, "phase4_results.json"),
            ("Phase 6", py + ["src/phase6_unseen.py"] + base + quick + ["--boot", "50"], "phase6_results.json"),
        ]))
    if args.dry_run:
        for s, res, _, _, steps in plan:
            for name, cmd, marker in steps:
                print(f"seed {s} {name} ({'done already' if (res / marker).exists() else 'to run'}):\n  {' '.join(cmd)}")
        return
    preflight()
    t0 = time.time()
    for s, res, mod, logs, steps in plan:
        for d in (res, mod, logs):
            d.mkdir(parents=True, exist_ok=True)
        for name, cmd, marker in steps:
            if (res / marker).exists():
                print(f"seed {s} {name}: result file exists, skipped", flush=True)
                continue
            run(f"seed {s} {name}", cmd, logs / f"{name.replace(' ', '').lower()}.log")
    print(f"\nFinished seeds {args.seeds} in {(time.time() - t0) / 60:.1f} min. Next: python src/aggregate_seeds.py --seeds-dir {args.out}")
    if args.quick:
        print("QUICK MODE: these numbers come from a tiny training run and must not be reported.")


if __name__ == "__main__":
    main()
