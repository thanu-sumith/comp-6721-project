"""Post-hoc temperature scaling, fitted on validation data only (cross-fitted by prompt group)."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedGroupKFold

from .metrics import normalize


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(p, T):
    return normalize(_softmax(np.log(normalize(p)) / T))


def fit_temperature(y, p, bounds=(0.05, 10.0)):
    """Single scalar T minimising negative log likelihood. T>1 softens, T<1 sharpens."""
    y, logits = np.asarray(y), np.log(normalize(p))

    def nll(T):
        q = _softmax(logits / T)
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-15, 1)).mean()

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)


def crossfit_temperature(y, p, groups, n_folds=2, seed=42):
    """Calibrate every validation row with a T learned on the OTHER prompt groups.

    Returns (calibrated_probs, per_fold_temperatures). This avoids scoring a
    calibrator on the same rows it was fitted on.
    """
    y, p = np.asarray(y), normalize(p)
    out, temps = np.zeros_like(p), []
    skf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
    for fit_idx, apply_idx in skf.split(p, y, groups):
        T = fit_temperature(y[fit_idx], p[fit_idx])
        temps.append(T)
        out[apply_idx] = apply_temperature(p[apply_idx], T)
    return out, temps
