"""Phase 6 tables: Table 5 (known vs unseen attack categories) and a seen-vs-withheld contrast with Phase 5.
Reads results/phase6_results.json, results/phase5_comparison.json and the saved test scores; writes reports/phase6_tables.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data import load_split
from evaluation import threshold_metrics

R = Path("results")
MODELS = ["unsup_mixed", "normal_only", "supervised"]
SHORT = {"unsup_mixed": "Unsupervised", "normal_only": "One-class", "supervised": "Supervised"}


def f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    global R
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()
    R = args.results_dir
    p6, p5 = json.load(open(R / "phase6_results.json")), json.load(open(R / "phase5_comparison.json"))
    held = p6["withheld"]
    _, mte = load_split("test")
    tte, y, blk = mte.attack_type.to_numpy(), mte.label_binary.to_numpy(), mte.block.to_numpy()
    assert len(tte) == p6["test_rows"]
    unseen, benign = np.isin(tte, held), tte == "Benign"
    known_att = (y == 1) & ~unseen
    S = {m: np.load(R / f"scores_unseen_{m}_test.npy") for m in MODELS}
    M = p6["models"]
    L = []
    a = L.append
    hdr = "| Metric | " + " | ".join(SHORT[m] for m in MODELS) + " |\n|---|" + "---:|" * len(MODELS)
    row = lambda lab, fn: a(f"| {lab} | " + " | ".join(fn(m) for m in MODELS) + " |")
    a(f"Withheld attack types (excluded from the scaler fit, training, early stopping and validation/threshold rows): **{', '.join(held)}**. "
      f"Test partition: {p6['test_rows']:,} records = {int(benign.sum()):,} benign + {int(known_att.sum()):,} known-attack + {int(unseen.sum()):,} withheld-attack. "
      f"Training rows removed: {p6['refit']['train_rows_removed']:,}; the scaler refit on the remaining rows left every min/max unchanged.\n")
    a("## Test blocks per withheld type (each type is one attack episode)\n\n| Type | Test rows | Distinct one-minute test blocks |\n|---|---:|---:|")
    for h in held:
        a(f"| {h} | {int((tte == h).sum()):,} | {len(np.unique(blk[tte == h]))} |")
    a("\nBlock-bootstrap intervals for a single type are therefore not meaningful when it occupies one or two blocks (Hulk's interval collapses to a point).\n")

    for tag, title in (("primary", "primary thresholds (rules fixed in advance)"), ("val_f1_tuned", "validation-F1-tuned thresholds (filtered validation labels)")):
        a(f"## Table 5{'a' if tag == 'primary' else 'b'} - known vs unseen attacks, {title}\n")
        a(hdr)
        row("Threshold", lambda m: f"{M[m]['thresholds'][tag]:.6g}")
        for k, lab in (("accuracy", "Known set: accuracy"), ("precision", "Known set: precision"), ("recall", "Known attacks: recall"),
                       ("f1", "Known set: F1"), ("fpr", "Benign: false-positive rate")):
            row(lab, lambda m, k=k: f(M[m]["known"][tag][k]))
        un = {}
        for m in MODELS:
            th = M[m]["thresholds"][tag]
            mm = threshold_metrics(y[benign | unseen], S[m][benign | unseen], th)
            assert abs(mm["recall"] - M[m]["unseen"][tag]["recall"]) < 1e-9
            un[m] = mm
        row("Unseen attacks: recall", lambda m: f(un[m]["recall"]))
        row("Unseen set: precision", lambda m: f(un[m]["precision"]))
        row("Unseen set: F1", lambda m: f(un[m]["f1"]))
        for h in held:
            row(f"Unseen {h}: flagged share", lambda m, h=h: f(M[m]["unseen"][tag]["per_type"][h]["flagged_share"], 3))
        a("")
        if tag == "val_f1_tuned":
            a("Unsupervised column: the validation optimum is degenerate (flags most benign records, so recall of anything is trivially high); it is not a usable operating point.\n")
    a("## Table 5c - threshold-free metrics\n")
    a(hdr)
    row("Known set ROC-AUC (benign vs known attacks)", lambda m: f(M[m]["known_ranking"]["roc_auc"]))
    row("Known set PR-AUC (chance %.4f)" % (known_att.sum() / (benign.sum() + known_att.sum())), lambda m: f(M[m]["known_ranking"]["pr_auc"]))
    row("Unseen set ROC-AUC (benign vs withheld attacks)", lambda m: f(M[m]["unseen_ranking"]["roc_auc"]))
    row("Unseen set PR-AUC (chance %.4f)" % M["supervised"]["unseen_base_rate"], lambda m: f(M[m]["unseen_ranking"]["pr_auc"]))
    for h in held:
        row(f"ROC-AUC {h} vs benign", lambda m, h=h: f(M[m]["unseen"]["primary"]["per_type"][h]["roc_auc_vs_benign"], 3))
    a("\n## Table 5d - the same attack types when they were / were not in the training data\n\n"
      "Seen = models of Phases 3-5 (trained with these types present); withheld = Phase 6 models. Flagged share at each model's primary threshold; ROC-AUC against benign. "
      "The two sets of models share the architecture and training procedure but differ in their training rows, so this contrast also includes the loss of 45% of the training attacks.\n")
    a("| Type | Model | Flagged, seen | Flagged, withheld | AUC, seen | AUC, withheld |\n|---|---|---:|---:|---:|---:|")
    for h in held:
        for m in MODELS:
            ps = p5["models"][m]
            a(f"| {h} | {SHORT[m]} | {f(ps['per_type_flag_rate_primary'][h]['rate'], 3)} | {f(M[m]['unseen']['primary']['per_type'][h]['flagged_share'], 3)} | "
              f"{f(ps['per_type_auc'][h]['auc_vs_benign'], 3)} | {f(M[m]['unseen']['primary']['per_type'][h]['roc_auc_vs_benign'], 3)} |")
    a("\n## Table 5e - 95% block-bootstrap intervals of the flagged share (primary thresholds)\n")
    a("| Group | " + " | ".join(SHORT[m] for m in MODELS) + " |\n|---|" + "---:|" * len(MODELS))
    for g in ["benign", "known_attacks", "unseen_all"] + held:
        a(f"| {g} | " + " | ".join(f"[{M[m]['ci_flagged_share_primary'][g][0]:.3f}, {M[m]['ci_flagged_share_primary'][g][1]:.3f}]" for m in MODELS) + " |")
    a("\nIntervals resample one-minute test blocks only (not training randomness; one seed per model). For types occupying one or two blocks they are not informative.\n")
    (args.report_dir / "phase6_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
