"""Results of the stream experiments (Phase 7 adaptive baseline, Phase 9 concept-drift monitoring) for the dashboard tab 'Drift and adaptation'.

This module only READS the saved experiment files (results/phase7_main*.json|csv, results/phase9_main*.json|csv, results/phase9_detection.json).
Nothing is recomputed, simulated or adapted here: the models served by this application stay static. Every number shown on the tab comes from those files;
the per-interval curves are rebuilt from the interval counts (tp, fp, fn, tn) in the CSV files, which add up to the totals of the JSON files (tested).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .service import ROOT

RES = ROOT / "results"


def _series(csv_path: Path):
    """arm -> per-interval lists, plus the day of every interval."""
    arms: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    days: dict[int, str] = {}
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        tp, fp, fn, tn, n = (int(r[k]) for k in ("tp", "fp", "fn", "tn", "n"))
        a, i = arms[r["arm"]], int(r["interval"])
        days[i] = r["day"]
        a["flagged_share"].append((tp + fp) / n if n else None)
        a["fpr"].append(fp / (fp + tn) if fp + tn else None)
        a["recall"].append(tp / (tp + fn) if tp + fn else None)
        a["attack_share"].append((tp + fn) / n if n else None)
        a["threshold"].append(float(r["threshold"]))
        a["n"].append(n)
    order = sorted(days)
    return {k: dict(v) for k, v in arms.items()}, [days[i] for i in order]


def _overall(arm: dict) -> dict:
    o = arm["overall"]
    return {k: o[k] for k in ("n", "attacks", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "flagged_share")}


def load() -> dict | None:
    needed = ["phase7_main.json", "phase7_main_intervals.csv", "phase9_main.json", "phase9_main_intervals.csv", "phase9_detection.json"]
    if not all((RES / f).exists() for f in needed):
        return None
    d7, d9 = json.loads((RES / "phase7_main.json").read_text()), json.loads((RES / "phase9_main.json").read_text())
    det = json.loads((RES / "phase9_detection.json").read_text())
    s7, days7 = _series(RES / "phase7_main_intervals.csv")
    s9, days9 = _series(RES / "phase9_main_intervals.csv")
    c7, c9 = d7["config"], d9["config"]
    labels7 = {"A_static": "A. Static model and threshold",
               "B_adaptive_threshold": f"B. Adaptive threshold (median + k x MAD of the last {c7['window']} intervals)",
               "C_adaptive_baseline": f"C. B plus retraining on a trusted buffer every {c7['retrain_every']} intervals"}
    labels9 = {"A_no_detection": "A. Static model, no detection, no updates",
               "A2_periodic": f"A2. Scheduled retraining every {c9['retrain_every']} intervals, no detection",
               "B_drift_adaptive": f"B. Drift alarm, then retraining for {c9['response']} rounds",
               "C_drift_confirmed": f"C. Drift alarm acted on only if it persists, highest {round(100 * (1 - c9['q_rel']))}% of scores left out of the retraining data"}
    alarm_attack = det.get("alarm_attack_share") or d9.get("alarm_attack_share")
    return {
        "interval_records": c7["interval_records"],
        "phase7": {
            "config": {k: c7[k] for k in ("interval_records", "sample", "window", "retrain_every", "buffer_intervals", "replay_frac", "ft_epochs", "anchor_tol")},
            "warm_up_days": d7["baseline"]["warm_days"], "evaluation_days": d7["baseline"]["eval_days"], "initial_threshold": d7["baseline"]["thr0"],
            "arms": {k: {"label": labels7[k], "overall": _overall(d7["arms"][k]), "series": s7[k], "retrainings": len(d7["arms"][k]["retrain_log"]),
                         "retrainings_accepted": sum(1 for r in d7["arms"][k]["retrain_log"] if r.get("accepted")),
                         "mean_trusted_attack_share": (sum(r["trusted_attack_share"] for r in d7["arms"][k]["retrain_log"]) / len(d7["arms"][k]["retrain_log"])) if d7["arms"][k]["retrain_log"] else None}
                     for k in labels7},
            "days": days7, "roc_auc_per_day": {"A_static": d7["auc_per_day"]["static"], "C_adaptive_baseline": d7["auc_per_day"]["C_adaptive_baseline"]},
        },
        "phase9": {
            "config": {k: c9[k] for k in ("interval_records", "retrain_every", "response", "persist", "q_rel", "adwin_delta", "monitor_batch")},
            "monitors": d9["monitor_names"],
            "arms": {k: {"label": labels9[k], "overall": _overall(d9["arms"][k]), "series": s9[k], "retrainings": det["retrainings"][k],
                         "mean_retraining_attack_share": det["mean_retrain_attack_share"][k], "recovery_intervals": det["recovery_intervals"][k]} for k in labels9},
            "days": days9,
            "alarms": [{"interval": a["interval"], "record": a["record"], "monitors": a["monitors"]} for a in d9["alarms"]],
            "alarm_attack_share": alarm_attack,
            "change_points": det["change_points"],
            "detection": {"known_changes": len(det["change_points"]), "detected": det["n_detected"], "delays_records": det["delays_records"], "alarms": det["alarms"],
                          "alarms_outside_known_changes": det["false_alarms"], "alarms_in_attack_batches": det["false_alarms_in_attack_batches"],
                          "alarms_outside_known_changes_per_million_records": det["false_alarms_per_million_records"]},
            "roc_auc_per_day": {("A_no_detection" if k == "static" else k): v for k, v in d9["auc_per_day"].items()},
        },
        "source_files": needed + ["reports/phase7_findings.md", "reports/phase9_findings.md"],
    }
