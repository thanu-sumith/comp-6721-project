"""Baseline classifiers with optional swap augmentation and swap-averaged inference."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import normalize

SWAP_PERM = [1, 0, 2]  # swapping A and B exchanges classes 0 and 1; tie stays tie
KINDS = ["numeric_lr", "numeric_hgb", "tfidf_lr"]


def swap_labels(y):
    y = np.asarray(y)
    out = y.copy()
    out[y == 0], out[y == 1] = 1, 0
    return out


def class_prior_probs(y_train, n):
    priors = np.bincount(y_train, minlength=3) / len(y_train)
    return np.tile(priors, (n, 1))


class PreferenceModel:
    """kind in KINDS. swap_aug=True trains on both orderings and swap-averages at inference."""

    def __init__(self, kind, seed=42, C=1.0, swap_aug=False, hgb_iter=200):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind, self.seed, self.C, self.swap_aug, self.hgb_iter = kind, seed, C, swap_aug, hgb_iter

    def _estimator(self):
        if self.kind == "numeric_lr":
            return make_pipeline(StandardScaler(), LogisticRegression(
                C=self.C, max_iter=2000, random_state=self.seed))
        if self.kind == "numeric_hgb":
            return HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=self.hgb_iter, max_leaf_nodes=15,
                min_samples_leaf=50, l2_regularization=1.0, max_features=0.8,
                random_state=self.seed)
        return LogisticRegression(C=self.C, max_iter=300, random_state=self.seed)

    def fit(self, feats, y):
        from scipy.sparse import issparse, vstack
        y = np.asarray(y)
        X = feats.design(swap=False)
        if self.swap_aug:
            Xs = feats.design(swap=True)
            X = vstack([X, Xs], format="csr") if issparse(X) else np.vstack([X, Xs])
            y = np.concatenate([y, swap_labels(y)])
        self.est_ = self._estimator().fit(X, y)
        return self

    def _proba(self, X):
        p = self.est_.predict_proba(X)
        full = np.zeros((p.shape[0], 3))
        full[:, self.est_.classes_ if hasattr(self.est_, "classes_") else [0, 1, 2]] = p
        return full

    def predict_proba(self, feats):
        p = self._proba(feats.design(swap=False))
        if self.swap_aug:
            p = 0.5 * (p + self._proba(feats.design(swap=True))[:, SWAP_PERM])
        return normalize(p)
