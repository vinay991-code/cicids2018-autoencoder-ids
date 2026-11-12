"""Plot helpers. Fixed colour per model (validated default palette, light surface), thin lines,
legend whenever there are two or more series, text in ink colours rather than series colours."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLORS = {"unsup_mixed": "#2a78d6", "normal_only": "#1baf7a", "supervised": "#eb6834", "supervised_recon": "#6250d6"}
LABELS = {"unsup_mixed": "Unsupervised AE (unlabeled train)", "normal_only": "One-class AE (benign-only train)",
          "supervised": "Supervised AE (classifier)", "supervised_recon": "Supervised AE + decoder"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _fig(w, h, **kw):
    fig, axes = plt.subplots(figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, axes


def plot_loss_curves(histories: dict, path, ylabel="MSE reconstruction loss", ncols=None, ylabels=None):
    """histories: name -> {'train': [...], 'val': [...]}. One panel per model (loss scales differ)."""
    ncols = ncols or len(histories)
    nrows = -(-len(histories) // ncols)
    fig, axes = _fig(5.2 * ncols, 3.8 * nrows, ncols=ncols, nrows=nrows, squeeze=False)
    for ax in axes.flatten()[len(histories):]:
        ax.axis("off")
    for ax, (name, h) in zip(axes.flatten(), histories.items()):
        style(ax)
        ep = np.arange(1, len(h["train"]) + 1)
        ax.plot(ep, h["train"], color=COLORS.get(name, "#2a78d6"), linewidth=2, label="training")
        ax.plot(ep, h["val"], color=COLORS.get(name, "#2a78d6"), linewidth=2, linestyle="--", label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel((ylabels or {}).get(name, ylabel))
        ax.set_title(LABELS.get(name, name), fontsize=10, loc="left")
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


LINESTYLE = {"unsup_mixed": "--", "normal_only": "--", "supervised": "-", "supervised_recon": "-"}


def plot_curves(roc: dict, pr: dict, base_rate: float, path):
    """roc: name -> (fpr, tpr); pr: name -> (recall, precision). Panels: ROC, ROC with log FPR axis, precision-recall."""
    fig, axes = _fig(16, 4.8, ncols=3)
    for ax in axes:
        style(ax)
    for name, (f, t) in roc.items():
        kw = dict(color=COLORS[name], linestyle=LINESTYLE[name], linewidth=2, label=LABELS[name])
        axes[0].plot(f, t, **kw)
        axes[1].plot(np.maximum(f, 1e-6), t, **kw)
    for name, (r, p) in pr.items():
        axes[2].plot(r, p, color=COLORS[name], linestyle=LINESTYLE[name], linewidth=2, label=LABELS[name])
    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="chance")
    axes[2].axhline(base_rate, color=MUTED, linewidth=1, linestyle=":", label=f"chance ({base_rate:.3f})")
    axes[0].set(xlabel="false-positive rate", ylabel="true-positive rate (recall)")
    axes[1].set(xlabel="false-positive rate (log scale)", ylabel="true-positive rate (recall)", xscale="log", xlim=(1e-4, 1))
    axes[2].set(xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    for ax, title in zip(axes, ("ROC curve", "ROC curve, low-FPR region", "Precision-recall curve")):
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower right")
    axes[2].legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


ARM_COLORS = {"A_static": "#2a78d6", "B_adaptive_threshold": "#1baf7a", "C_adaptive_baseline": "#eb6834"}
ARM_LABELS = {"A_static": "A: static", "B_adaptive_threshold": "B: adaptive threshold", "C_adaptive_baseline": "C: adaptive baseline"}


def plot_phase7(o: dict, path):
    """2x2: threshold traces, per-day ROC-AUC (static vs adaptive baseline), per-day F1, contamination of retrain buffers."""
    arms = list(o["arms"])
    days = sorted(o["arms"][arms[0]]["per_day"])
    iv_days = o["intervals"]["days"]
    fig, axes = _fig(15, 9, ncols=2, nrows=2)
    ax = axes[0, 0]
    style(ax)
    x = np.arange(len(iv_days))
    for a in arms:
        ax.plot(x, np.maximum(o["arms"][a]["threshold_trace"], 1e-5), color=ARM_COLORS[a], linewidth=1.4, label=ARM_LABELS[a])
    for i in range(1, len(iv_days)):
        if iv_days[i] != iv_days[i - 1]:
            ax.axvline(i, color=MUTED, linewidth=0.6, linestyle=":")
    ax.axhline(0.072, color=INK2, linewidth=0.8, linestyle="--")
    ax.text(len(x) * 0.99, 0.09, "0.072 = largest error in the Phase 3 test scores", ha="right", fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("update interval (dotted lines = new day)")
    ax.set_ylabel("anomaly threshold (log scale)")
    ax.set_title("Threshold in force at each interval", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower left")
    ax = axes[0, 1]
    style(ax)
    keys = [k for k in ("static", "C_adaptive_baseline") if k in o["auc_per_day"]]
    dd = [d for d in days if d in o["auc_per_day"]["static"]]
    w = 0.38
    for i, k in enumerate(keys):
        vals = [o["auc_per_day"][k].get(d, {}).get("roc_auc", np.nan) for d in dd]
        ax.bar(np.arange(len(dd)) + (i - 0.5) * w, vals, w * 0.92, color=ARM_COLORS["A_static" if k == "static" else k],
               label="static model (A, B)" if k == "static" else "adaptive baseline model (C)")
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xticks(np.arange(len(dd)), [d[5:] for d in dd], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC within the day (0.5 = chance)")
    ax.set_title("Ranking quality per day (threshold-free)", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right")
    ax = axes[1, 0]
    style(ax)
    w = 0.27
    for i, a in enumerate(arms):
        vals = [(o["arms"][a]["per_day"][d]["f1"] if o["arms"][a]["per_day"][d]["f1"] == o["arms"][a]["per_day"][d]["f1"] else 0) for d in days]
        ax.bar(np.arange(len(days)) + (i - 1) * w, vals, w * 0.92, color=ARM_COLORS[a], label=ARM_LABELS[a])
    ax.set_xticks(np.arange(len(days)), [d[5:] for d in days], fontsize=8)
    ax.set_ylabel("F1-score at the arm's own thresholds")
    ax.set_title("F1 per day (days 02-22 and 02-23 have under 1,000 attacks)", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right")
    ax = axes[1, 1]
    style(ax)
    rl = o["arms"]["C_adaptive_baseline"]["retrain_log"]
    rx = np.arange(len(rl))
    ax.bar(rx, [r["trusted_attack_share"] for r in rl], color="#eb6834", width=0.8)
    ax.axhline(o["stream_attack_share"], color=INK, linewidth=1.2, linestyle="--", label=f"attack share of the whole stream ({o['stream_attack_share']:.3f})")
    ax.set_xlabel("retraining event")
    ax.set_ylabel("share of attacks in the trusted buffer")
    ax.set_title("Contamination of the trusted buffer at each retraining (all were accepted)", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


ARM8_COLORS = {"S_static": "#2a78d6", "U_unrestricted": "#eb6834", "M_margin_only": "#1baf7a", "G_gated": "#6250d6"}
ARM8_LABELS = {"S_static": "S: static", "U_unrestricted": "U: unrestricted", "M_margin_only": "M: margin only", "G_gated": "G: confidence-gated"}


def plot_phase8(series: dict, contam: dict, auc: dict, day_marks: list, path):
    """2x2: rolling false-positive rate, rolling recall, buffer contamination per retraining, per-day ROC-AUC.
    series[arm] = (fpr_roll, rec_roll) arrays per interval; contam[arm] = list of attack shares per retraining;
    auc[arm] = dict day -> roc_auc; day_marks = interval indices where a new day starts."""
    arms = list(series)
    fig, axes = _fig(15, 9, ncols=2, nrows=2)
    for ax, k, ttl, yl in ((axes[0, 0], 0, "False-positive rate (rolling mean of 3 intervals)", "benign records flagged"),
                           (axes[0, 1], 1, "Recall (rolling mean of 3 intervals; blank where no attacks)", "attack records flagged")):
        style(ax)
        for a in arms:
            ax.plot(np.arange(len(series[a][k])), series[a][k], color=ARM8_COLORS[a], linewidth=1.5,
                    linestyle="--" if a == "S_static" else "-", label=ARM8_LABELS[a])
        for d in day_marks:
            ax.axvline(d, color=MUTED, linewidth=0.6, linestyle=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("update interval (dotted lines = new day)")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10, loc="left", color=INK)
        ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right" if k == 0 else "lower right", ncol=2)
    ax = axes[1, 0]
    style(ax)
    for a, v in contam.items():
        ax.plot(np.arange(len(v)), v, color=ARM8_COLORS[a], linewidth=1.6, label=ARM8_LABELS[a])
    ax.set_xlabel("retraining event")
    ax.set_ylabel("share of attacks in the buffer used for retraining")
    ax.set_title("Contamination of the training buffer", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right")
    ax = axes[1, 1]
    style(ax)
    dd = sorted(next(iter(auc.values())))
    w = 0.8 / len(auc)
    for i, (a, v) in enumerate(auc.items()):
        ax.bar(np.arange(len(dd)) + (i - (len(auc) - 1) / 2) * w, [v.get(d, np.nan) for d in dd], w * 0.92, color=ARM8_COLORS[a], label=ARM8_LABELS[a])
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xticks(np.arange(len(dd)), [d[5:] for d in dd], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC within the day (0.5 = chance)")
    ax.set_title("Ranking quality per day (threshold-free)", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


ARM9_COLORS = {"A_no_detection": "#2a78d6", "A2_periodic": "#898781", "B_drift_adaptive": "#eb6834", "C_drift_confirmed": "#6250d6"}
ARM9_LABELS = {"A_no_detection": "A: no detection (static)", "A2_periodic": "A2: periodic retraining (reference)",
               "B_drift_adaptive": "B: drift-triggered", "C_drift_confirmed": "C: drift-triggered, confirmed"}


def plot_phase9(series: dict, alarms: list, change_pts: list, contam: dict, auc: dict, path):
    """2x2: rolling FPR and rolling recall per interval (true change points = solid grey lines, drift alarms = triangles),
    buffer contamination per retraining, per-day ROC-AUC. alarms: list of (interval, attack_related_bool)."""
    arms = list(series)
    fig, axes = _fig(15, 9, ncols=2, nrows=2)
    for ax, k, ttl, yl in ((axes[0, 0], 0, "False-positive rate (rolling mean of 3 intervals)", "benign records flagged"),
                           (axes[0, 1], 1, "Recall (rolling mean of 3 intervals; blank where no attacks)", "attack records flagged")):
        style(ax)
        for a in arms:
            ax.plot(np.arange(len(series[a][k])), series[a][k], color=ARM9_COLORS[a], linewidth=1.5,
                    linestyle="--" if a in ("A_no_detection", "A2_periodic") else "-", label=ARM9_LABELS[a])
        for c in change_pts:
            ax.axvline(c, color=INK2, linewidth=1.0, alpha=0.7)
        ax.scatter([i for i, att in alarms if not att], [1.0] * sum(1 for _, att in alarms if not att), marker="v", s=34, color=INK, zorder=5, label="drift alarm")
        ax.scatter([i for i, att in alarms if att], [1.0] * sum(1 for _, att in alarms if att), marker="v", s=34, facecolors="none", edgecolors=INK, zorder=5,
                   label="drift alarm during an attack")
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel("update interval (grey lines = true benign change points)")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10, loc="left", color=INK)
        if k == 0:
            ax.legend(frameon=False, labelcolor=INK2, fontsize=7, loc="center right")
    ax = axes[1, 0]
    style(ax)
    for a, v in contam.items():
        ax.plot(np.arange(len(v)), v, color=ARM9_COLORS[a], linewidth=1.6, label=ARM9_LABELS[a])
    ax.set_xlabel("retraining event")
    ax.set_ylabel("share of attacks in the retraining data")
    ax.set_title("Contamination of the retraining data", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right")
    ax = axes[1, 1]
    style(ax)
    dd = sorted(next(iter(auc.values())))
    w = 0.8 / len(auc)
    for i, (a, v) in enumerate(auc.items()):
        ax.bar(np.arange(len(dd)) + (i - (len(auc) - 1) / 2) * w, [v.get(d, np.nan) for d in dd], w * 0.92, color=ARM9_COLORS[a], label=ARM9_LABELS[a])
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xticks(np.arange(len(dd)), [d[5:] for d in dd], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC within the day (0.5 = chance)")
    ax.set_title("Ranking quality per day (threshold-free)", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_unseen(results: dict, held: list, path):
    """Left: share of each withheld attack type flagged at the primary threshold (plus benign = false-positive rate
    and known attacks), per model. Right: ROC curve of benign vs all withheld attacks."""
    models = [m for m in ("unsup_mixed", "normal_only", "supervised") if m in results["models"]]
    fig, axes = _fig(15.5, 4.8, ncols=2, gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    style(ax)
    groups = ["benign"] + ["known_attacks"] + list(held)
    names = ["Benign\n(false alarms)", "Known\nattacks"] + list(held)
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [ci_mid(results["models"][m], g, held) for g in groups]
        x = np.arange(len(groups)) + (i - (len(models) - 1) / 2) * w
        ax.bar(x, vals, w * 0.9, color=COLORS[m], label=LABELS[m])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(np.arange(len(groups)), names, fontsize=8)
    ax.set_ylabel("share of records flagged")
    ax.set_ylim(0, 1.12)
    ax.set_title("Flagged share at the primary threshold (withheld types were never seen in training)", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right")
    ax = axes[1]
    style(ax)
    for m in models:
        f, t = results["models"][m]["_roc_unseen"]
        ax.plot(f, t, color=COLORS[m], linestyle=LINESTYLE[m] if m in LINESTYLE else "-", linewidth=2, label=LABELS[m])
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="chance")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate (recall)")
    ax.set_title("ROC: benign vs all withheld attacks", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def ci_mid(model_res, group, held):
    if group == "benign":
        return model_res["known"]["primary"]["fpr"]
    if group == "known_attacks":
        return model_res["known"]["primary"]["recall"]
    return model_res["unseen"]["primary"]["per_type"][group]["flagged_share"]


def plot_prob_hist(probs: dict, y, path, thr=None):
    """Predicted attack probability by true class. probs: name -> array in [0, 1]; y-axis log."""
    fig, axes = _fig(5.2 * len(probs), 3.8, ncols=len(probs), squeeze=False)
    bins = np.linspace(0, 1, 51)
    for ax, (name, p) in zip(axes[0], probs.items()):
        style(ax)
        ax.hist(p[y == 0], bins=bins, color="#2a78d6", alpha=0.75, label="benign", linewidth=0)
        ax.hist(p[y == 1], bins=bins, color="#eb6834", alpha=0.75, label="attack", linewidth=0)
        ax.set_yscale("log")
        if thr and name in thr:
            for lab, t in thr[name].items():
                ax.axvline(t, color=INK, linewidth=1.2, linestyle="-" if lab == "0.5" else ":", label=f"threshold {lab}")
        ax.set_xlabel("predicted attack probability")
        ax.set_ylabel("records (log scale)")
        ax.set_title(LABELS.get(name, name), fontsize=10, loc="left")
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(cms: dict, path, subtitle=""):
    """cms: title -> {'tn','fp','fn','tp'}. Counts plus row-normalised share in each cell."""
    fig, axes = _fig(4.6 * len(cms), 4.3, ncols=len(cms), squeeze=False)
    for ax, (title, c) in zip(axes[0], cms.items()):
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
        rows = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(rows, cmap=SEQ, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(m[i, j]):,}\n{100*rows[i, j]:.2f}%", ha="center", va="center", fontsize=10,
                        color="#ffffff" if rows[i, j] > 0.5 else INK)
        ax.set_xticks([0, 1], ["Normal", "Anomaly"], color=INK2)
        ax.set_yticks([0, 1], ["Benign", "Attack"], color=INK2)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.xaxis.label.set_color(INK2)
        ax.yaxis.label.set_color(INK2)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    if subtitle:
        fig.suptitle(subtitle, fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_score_hist(scores: dict, y, path, thr=None, title=""):
    """Distribution of reconstruction error by true class (log x). scores: name -> array."""
    fig, axes = _fig(5.2 * len(scores), 3.8, ncols=len(scores), squeeze=False)
    for ax, (name, s) in zip(axes[0], scores.items()):
        style(ax)
        ls = np.log10(np.maximum(s, 1e-9))
        bins = np.linspace(np.percentile(ls, 0.1), np.percentile(ls, 99.9), 80)
        ax.hist(ls[y == 0], bins=bins, color="#2a78d6", alpha=0.75, label="benign", density=True, linewidth=0)
        ax.hist(ls[y == 1], bins=bins, color="#eb6834", alpha=0.75, label="attack", density=True, linewidth=0)
        if thr and name in thr:
            ax.axvline(np.log10(thr[name]), color=INK, linewidth=1.2, label="threshold")
        ax.set_xlabel("log10 reconstruction error")
        ax.set_ylabel("density")
        ax.set_title(LABELS.get(name, name), fontsize=10, loc="left")
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
