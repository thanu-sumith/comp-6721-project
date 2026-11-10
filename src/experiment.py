"""Shared plumbing: load the saved split, cache features, save/load prediction files."""
import json
from pathlib import Path

import numpy as np

from .baseline import ROOT, load_data, saved_split
from .features import NumericFeatures, extract


def load_split(data=None, splits=None):
    """Return (df, fingerprint, split Series) using the team's fixed saved assignments."""
    df, fingerprint = load_data(data)
    split = saved_split(df, fingerprint, splits)
    return df, fingerprint, split


def numeric_features(df, fingerprint, cache_dir=None):
    """Extract (and cache) numeric features; cache is keyed by dataset fingerprint."""
    cache_dir = Path(cache_dir) if cache_dir else ROOT / "data/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"numeric_features_{fingerprint[:12]}.npz"
    if path.exists():
        z = np.load(path)
        return NumericFeatures(z["FA"], z["FB"], z["shared"])
    fa, fb, sh = extract(df)
    np.savez_compressed(path, FA=fa, FB=fb, shared=sh)
    return NumericFeatures(fa, fb, sh)


def pred_path(results, model, variant, seed, partition="validation"):
    d = Path(results) / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model}__{variant}__seed{seed}__{partition}.npz"


def save_predictions(path, ids, probs, y):
    np.savez_compressed(path, ids=np.asarray(ids).astype(str), probs=probs, y=np.asarray(y))


def load_predictions(path):
    z = np.load(path, allow_pickle=False)
    return z["ids"], z["probs"], z["y"]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")
