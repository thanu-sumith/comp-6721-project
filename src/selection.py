"""Protocol-consistent hyperparameter selection.

Selection must never look at the validation partition, otherwise validation scores stop
being comparable across models (a model whose C was picked on validation is optimistically
biased relative to one that used a fixed C). Everything here runs a prompt-grouped
cross-validation **inside the training partition only**.

The residual optimism is limited to TF-IDF vocabulary construction: the vectoriser is fit on
the whole training partition (never on validation), so a CV fold's held-out rows contributed
document frequencies to the vocabulary. No validation row, label, or prompt ever enters
feature fitting or selection.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .metrics import normalize
from .models import PreferenceModel

DEFAULT_GRIDS = {
    "numeric_lr": [0.01, 0.1, 1.0],
    "numeric_hgb": [1.0],          # no C; grid is a placeholder so the loop is uniform
    "tfidf_lr": [0.3, 1.0, 3.0],
}


def _fold_log_loss(m, feats, idx, y):
    p = normalize(m.predict_proba(feats.take(idx)))
    return float(-np.log(np.clip(p[np.arange(len(idx)), y[idx]], 1e-15, 1.0)).mean())


def select_C(kind, feats, y, groups, grid, seed=42, swap_aug=False, n_splits=3,
             log=print, tag=""):
    """Return (best_C, table). `feats` covers TRAIN rows only; `groups` are prompt groups."""
    y, groups = np.asarray(y), np.asarray(groups)
    if len(grid) <= 1:
        return grid[0], {"single_value": grid[0], "protocol": "no_selection"}
    skf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), y, groups))
    table, best, best_ll = {}, grid[0], np.inf
    for C in grid:
        lls = []
        for k, (fit_i, ev_i) in enumerate(folds):
            m = PreferenceModel(kind, seed, C, swap_aug).fit(feats.take(fit_i), y[fit_i])
            lls.append(_fold_log_loss(m, feats, ev_i, y))
        mean = float(np.mean(lls))
        table[str(C)] = {"cv_log_loss_mean": mean, "folds": [float(v) for v in lls]}
        log(f"    {tag} C={C:<5} train-CV log loss {mean:.5f}  (folds "
            + ", ".join(f"{v:.5f}" for v in lls) + ")")
        if mean < best_ll:
            best, best_ll = C, mean
    table["selected_C"] = best
    table["protocol"] = f"train_only_grouped_cv_{n_splits}fold"
    table["cv_seed"] = seed
    table["cv_groups"] = "parsed prompt"
    return best, table
