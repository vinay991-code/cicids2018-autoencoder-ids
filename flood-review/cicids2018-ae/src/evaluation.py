"""Metrics shared by all experiments. Convention: 1 = attack/anomaly, score > threshold -> predicted 1."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def confusion(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return {"tn": int((~y & ~pred).sum()), "fp": int((~y & pred).sum()),
            "fn": int((y & ~pred).sum()), "tp": int((y & pred).sum())}


def threshold_metrics(y, score, thr):
    pred = score > thr
    c = confusion(y, pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    div = lambda a, b: float(a / b) if b else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {**c, "threshold": float(thr), "accuracy": div(tp + tn, tp + tn + fp + fn),
            "precision": prec, "recall": rec,
            "f1": div(2 * prec * rec, prec + rec) if tp else 0.0,
            "specificity": div(tn, tn + fp), "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp)}


def ranking_metrics(y, score):
    """Threshold-free metrics."""
    return {"roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score))}


def best_f1_threshold(y, score):
    """Threshold that maximises F1 on the given data (call on VALIDATION data only)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def per_type_auc(attack_type, score):
    """ROC-AUC of each attack type against benign rows (diagnostic; n is reported alongside)."""
    out = {}
    ben = attack_type == "Benign"
    for t in sorted(set(attack_type) - {"Benign"}):
        m = (attack_type == t) | ben
        out[str(t)] = {"n": int((attack_type == t).sum()), "auc_vs_benign": float(roc_auc_score(attack_type[m] == t, score[m]))}
    return out


def per_type_recall(attack_type, score, thr):
    """Share of each attack type scored above the threshold (benign row = false-positive rate)."""
    out = {}
    pred = score > thr
    for t in np.unique(attack_type):
        m = attack_type == t
        out[str(t)] = {"n": int(m.sum()), "flagged": int(pred[m].sum()), "rate": float(pred[m].mean())}
    return out
