"""Run from the repository root with: python -m src.baseline."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
WINNERS = ["winner_model_a", "winner_model_b", "winner_tie"]


def parse_turns(value):
    """Require JSON lists; preserve turn boundaries in canonical grouping."""
    if not isinstance(value, str):
        raise ValueError("Missing/non-text prompt or response: audit the CSV.")
    turns = json.loads(value)
    if not isinstance(turns, list) or not all(
        item is None or isinstance(item, str) for item in turns
    ):
        raise ValueError("Expected a JSON list of strings/null turns.")
    return turns


def load_data(path=None):
    path = Path(path) if path else ROOT / "data/raw/train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Download the competition train.csv and place it at {path}. "
            "See README.md. No project results have been calculated."
        )
    df = pd.read_csv(path)
    required = ["id", "prompt", "response_a", "response_b"] + WINNERS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("IDs must be present and unique.")
    labels = df[WINNERS]
    if not labels.isin([0, 1]).all().all() or not labels.sum(axis=1).eq(1).all():
        raise ValueError("Each row must have exactly one binary winner flag.")
    df["label"] = labels.to_numpy().argmax(axis=1)
    for col in ["prompt", "response_a", "response_b"]:
        parsed = df[col].map(parse_turns)
        df[col + "_text"] = parsed.map(
            lambda turns: "\n".join(t or "" for t in turns)
        )
        if col == "prompt":
            df["prompt_group"] = parsed.map(
                lambda turns: hashlib.sha256(
                    json.dumps(turns, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
            )
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    return df, fingerprint


def saved_split(df, fingerprint, directory=None):
    directory = Path(directory) if directory else ROOT / "data/splits"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    ids_path = directory / "split_ids.csv"
    if manifest_path.exists() != ids_path.exists():
        raise ValueError("Incomplete split files: inspect them before continuing.")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["dataset_sha256"] != fingerprint:
            raise ValueError("Dataset changed. Review/version splits explicitly.")
        assignments = pd.read_csv(ids_path, dtype={"id": str})
        if assignments["id"].duplicated().any():
            raise ValueError("Duplicate IDs in saved splits.")
        if set(assignments["id"]) != set(df["id"].astype(str)):
            raise ValueError("Saved IDs do not match dataset IDs.")
        split = df["id"].astype(str).map(assignments.set_index("id")["split"])
    else:
        # Approximately 60/20/20; grouping takes precedence over exact ratios.
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        remaining, test = next(outer.split(df, df.label, df.prompt_group))
        subset = df.iloc[remaining]
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
        train_rel, val_rel = next(inner.split(
            subset, subset.label, subset.prompt_group
        ))
        split = pd.Series("", index=df.index)
        split.iloc[remaining[train_rel]] = "train"
        split.iloc[remaining[val_rel]] = "validation"
        split.iloc[test] = "test"

    if set(split.unique()) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, test assignments.")
    audit = pd.DataFrame({"group": df.prompt_group, "split": split})
    if audit.groupby("group")["split"].nunique().max() != 1:
        raise ValueError("A prompt appears across partitions.")
    for name in ["train", "validation", "test"]:
        if set(df.loc[split == name, "label"]) != {0, 1, 2}:
            raise ValueError(f"{name} lacks a class; review grouped splitting.")
    if not manifest_path.exists():
        pd.DataFrame({"id": df.id, "split": split}).to_csv(ids_path, index=False)
        manifest_path.write_text(json.dumps({
            "seed": SEED,
            "dataset_sha256": fingerprint,
            "method": "nested StratifiedGroupKFold by parsed prompt",
            "label_order": ["A wins", "B wins", "tie"],
            "sklearn_version": version("scikit-learn")
        }, indent=2) + "\n")
    return split


def prior_baseline(df, split, fingerprint):
    training_labels = df.loc[split == "train", "label"].to_numpy()
    validation_labels = df.loc[split == "validation", "label"].to_numpy()
    priors = np.bincount(training_labels, minlength=3) / len(training_labels)
    probabilities = np.tile(priors, (len(validation_labels), 1))
    predictions = probabilities.argmax(axis=1)
    return {
        "model": "training_class_prior",
        "evaluation_partition": "validation",
        "dataset_sha256": fingerprint,
        "seed": SEED,
        "class_order": ["A wins", "B wins", "tie"],
        "training_priors": priors.tolist(),
        "log_loss": float(log_loss(validation_labels, probabilities, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(validation_labels, predictions)),
        "macro_f1": float(f1_score(
            validation_labels, predictions, labels=[0, 1, 2],
            average="macro", zero_division=0
        ))
    }


def main():
    df, fingerprint = load_data()
    print("Dataset shape:", df.shape)
    print("Class counts (0=A, 1=B, 2=tie):\n", df.label.value_counts().sort_index())
    print("Missing values:\n", df.isna().sum())
    print("Repeated prompt rows:", df.prompt_group.duplicated().sum())
    split = saved_split(df, fingerprint)
    print("Partition counts:\n", pd.crosstab(split, df.label))
    metrics = prior_baseline(df, split, fingerprint)
    directory = ROOT / "results/metrics"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "class_prior_validation.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    versions = {pkg: version(pkg) for pkg in ["numpy", "pandas", "scikit-learn", "matplotlib"]}
    versions["python"] = platform.python_version()
    (directory / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print("Saved validation metrics. Final local test has not been scored.")


if __name__ == "__main__":
    main()
