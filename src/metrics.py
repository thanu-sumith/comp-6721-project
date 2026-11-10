"""Evaluation metrics shared by every model: log loss, accuracy, macro F1, ECE, plots.

Label order everywhere: 0 = A wins, 1 = B wins, 2 = tie.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

CLASS_NAMES = ["A wins", "B wins", "tie"]
EPS = 1e-15


def normalize(p):
    """Clip and renormalise so every row is a valid probability vector."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def ece_top_label(y, p, n_bins=15):
    """Expected calibration error of the predicted (top) class, equal-width bins."""
    y, p = np.asarray(y), normalize(p)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece, rows = 0.0, []
    for b in range(n_bins):
        m = ids == b
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            rows.append((edges[b], edges[b + 1], int(m.sum()), conf[m].mean(), correct[m].mean()))
    return float(ece), rows


def ece_classwise(y, p, n_bins=15):
    """Mean over classes of the one-vs-rest ECE (also checks the tie probability)."""
    y, p = np.asarray(y), normalize(p)
    out = []
    for k in range(p.shape[1]):
        yk, pk = (y == k).astype(float), p[:, k]
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ids = np.clip(np.digitize(pk, edges[1:-1]), 0, n_bins - 1)
        e = 0.0
        for b in range(n_bins):
            m = ids == b
            if m.any():
                e += m.mean() * abs(yk[m].mean() - pk[m].mean())
        out.append(e)
    return [float(v) for v in out], float(np.mean(out))


def summarize(y, p, n_bins=15):
    """All scalar metrics for one set of predictions."""
    y, p = np.asarray(y), normalize(p)
    pred = p.argmax(axis=1)
    ece, _ = ece_top_label(y, p, n_bins)
    per_class, mean_cw = ece_classwise(y, p, n_bins)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(
            y, pred, labels=[0, 1, 2], average=None, zero_division=0)],
        "ece_top_label": ece,
        "ece_classwise_mean": mean_cw,
        "ece_classwise": per_class,
        "mean_confidence": float(p.max(axis=1).mean()),
        "predicted_class_share": [float((pred == k).mean()) for k in range(3)],
    }


def bootstrap_log_loss_ci(y, p, groups=None, n_boot=300, seed=0):
    """95% bootstrap interval for log loss, resampling prompt groups if given."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), normalize(p)
    nll = -np.log(p[np.arange(len(y)), y])
    if groups is None:
        draws = [nll[rng.integers(0, len(nll), len(nll))].mean() for _ in range(n_boot)]
    else:
        codes = np.unique(np.asarray(groups), return_inverse=True)[1]
        n_groups = codes.max() + 1
        sums = np.bincount(codes, weights=nll, minlength=n_groups)
        cnts = np.bincount(codes, minlength=n_groups)
        draws = []
        for _ in range(n_boot):
            pick = rng.integers(0, n_groups, n_groups)
            draws.append(sums[pick].sum() / cnts[pick].sum())
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def reliability_figure(y, probs_by_name, path, n_bins=10, title="Reliability (top-label)"):
    """Reliability diagram plus confidence histogram for one or more prediction sets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1.4]}, sharex=True)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, p in probs_by_name.items():
        ece, rows = ece_top_label(y, p, n_bins)
        if rows:
            conf = [r[3] for r in rows]
            acc = [r[4] for r in rows]
            ax.plot(conf, acc, marker="o", label=f"{name} (ECE {ece:.3f})")
        axh.hist(normalize(p).max(axis=1), bins=np.linspace(0, 1, 26), alpha=0.5, label=name)
    ax.set_ylabel("Accuracy in bin")
    ax.set_title(title)
    ax.legend(fontsize=8)
    axh.set_xlabel("Confidence (max predicted probability)")
    axh.set_ylabel("Examples")
    axh.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
