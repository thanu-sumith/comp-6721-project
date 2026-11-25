"""Final fit + Kaggle submission for the late-submission protocol.

Trains the chosen baseline on TRAIN + VALIDATION (the local test partition is not used for
this, and never was), scores the Kaggle test.csv, and writes submission.csv in the exact
column order the competition expects:

    id, winner_model_a, winner_model_b, winner_tie

Hyperparameters are re-selected by train-only CV **inside train+validation** at submission
time, so nothing is tuned on the test file. Reported validation numbers stay the train-only
fit from `scripts/run_baselines.py`; this script only produces the final predictions.

Usage
  # 1. put the Kaggle test file next to the training file
  #    data/raw/test.csv  (columns: id, prompt, response_a, response_b)

  python scripts/make_submission.py --model numeric_hgb --variant swapaug
  python scripts/make_submission.py --model tfidf_lr --variant swapaug --seeds 42 43 44

Outputs
  submission.csv                 the file to upload to Kaggle
  results/metrics/submission_<model>__<variant>.json   provenance + class shares

Sanity checks before writing: row count equals the test file's, ids appear in the same order,
every probability row sums to 1, no NaN, and the predicted class mix is not degenerate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import load_data, parse_turns  # noqa: E402
from src.experiment import load_split, numeric_features, write_json  # noqa: E402
from src.features import build_tfidf_with_vectorizer, transform_tfidf  # noqa: E402
from src.metrics import normalize  # noqa: E402
from src.models import PreferenceModel, swap_labels  # noqa: E402
from src.selection import DEFAULT_GRIDS, select_C  # noqa: E402

CLASS_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]


def load_test(path):
    """Parse test.csv into the same derived columns load_data produces, without labels."""
    df = pd.read_csv(path)
    required = ["id", "prompt", "response_a", "response_b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"test.csv is missing columns: {missing}")
    if df["id"].duplicated().any():
        raise SystemExit("test.csv has duplicate ids.")
    for col in ["prompt", "response_a", "response_b"]:
        turns = df[col].map(parse_turns)
        df[col + "_text"] = turns.map(lambda t: "\n".join(x or "" for x in t))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="train.csv (default data/raw/train.csv)")
    ap.add_argument("--test", default=None, help="test.csv  (default data/raw/test.csv)")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--model", default="numeric_hgb",
                    choices=["numeric_hgb", "numeric_lr", "tfidf_lr"])
    ap.add_argument("--variant", default="swapaug", choices=["plain", "swapaug"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                    help="Average probabilities over these seeds.")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--cv-folds", type=int, default=3)
    args = ap.parse_args()
    results = Path(args.results)
    test_path = Path(args.test) if args.test else ROOT / "data/raw/test.csv"
    if not test_path.exists():
        raise SystemExit(
            f"test.csv not found at {test_path}.\n"
            "Download it from https://www.kaggle.com/competitions/"
            "llm-classification-finetuning/data (it is a separate file from train.csv) "
            "and place it at data/raw/test.csv.")

    # ---- training data: train + validation only ---------------------------------------
    df, fingerprint, split = load_split(args.data, args.splits)
    fit_mask = (split == "train").to_numpy() | (split == "validation").to_numpy()
    print(f"fitting on train+validation: {fit_mask.sum():,} rows "
          f"(local test partition excluded: {(split == 'test').sum():,} rows untouched)")
    y_fit = df.label.to_numpy()[fit_mask]
    groups_fit = df.prompt_group.to_numpy()[fit_mask]

    swap_aug = args.variant == "swapaug"
    vectorizer = None
    if args.model.startswith("numeric"):
        all_feats = numeric_features(df, fingerprint, args.cache)
    else:
        all_feats, vectorizer = build_tfidf_with_vectorizer(df, fit_mask)
    feats_fit = all_feats.take(np.where(fit_mask)[0])

    grid = DEFAULT_GRIDS[args.model]
    best_C, sel = select_C(args.model, feats_fit, y_fit, groups_fit, grid,
                           swap_aug=swap_aug, n_splits=args.cv_folds, tag="submission")
    print(f"selected C = {best_C} (train+validation CV, {args.cv_folds} folds)")

    # ---- test data -------------------------------------------------------------------
    df_test = load_test(test_path)
    print(f"test rows: {len(df_test):,}")
    if df_test["id"].astype(str).duplicated().any():
        raise SystemExit("test ids are not unique after casting to str.")
    if args.model.startswith("numeric"):
        # cache key differs from the training fingerprint, so the test matrix never collides
        feats_test = numeric_features(df_test, "test" + fingerprint[:8], args.cache)
    else:
        # vectoriser was fitted on train+validation only; the test rows are only transformed
        feats_test = transform_tfidf(vectorizer, df_test)

    # ---- fit, predict, average over seeds --------------------------------------------
    probs = np.zeros((len(df_test), 3))
    for seed in args.seeds:
        m = PreferenceModel(args.model, seed, best_C, swap_aug).fit(feats_fit, y_fit)
        probs += normalize(m.predict_proba(feats_test))
        print(f"  seed {seed}: fitted and predicted")
    probs /= len(args.seeds)
    probs = normalize(probs)

    # ---- checks ----------------------------------------------------------------------
    assert probs.shape == (len(df_test), 3), probs.shape
    assert np.isfinite(probs).all(), "non-finite probabilities"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), "rows do not sum to 1"
    share = probs.mean(axis=0)
    print(f"mean predicted probabilities A/B/tie: {share[0]:.4f} / {share[1]:.4f} / {share[2]:.4f}")
    print(f"argmax class shares: "
          f"{np.bincount(probs.argmax(axis=1), minlength=3) / len(probs)}")
    if share.max() > 0.9:
        print("WARNING: predictions are nearly constant — check the feature pipeline.")

    # ---- write submission ------------------------------------------------------------
    sub = pd.DataFrame({CLASS_COLS[0]: probs[:, 0],
                        CLASS_COLS[1]: probs[:, 1],
                        CLASS_COLS[2]: probs[:, 2]})
    sub.insert(0, "id", df_test["id"].to_numpy())
    out = Path(args.out)
    sub.to_csv(out, index=False)
    print(f"\nwrote {out} with {len(sub):,} rows and columns {list(sub.columns)}")

    write_json(results / "metrics" / f"submission_{args.model}__{args.variant}.json", {
        "model": args.model, "variant": args.variant, "seeds": args.seeds,
        "chosen_C": best_C, "selection": sel,
        "fit_rows": int(fit_mask.sum()), "test_rows": int(len(df_test)),
        "fit_partitions": ["train", "validation"],
        "local_test_partition_used": False,
        "dataset_sha256": fingerprint,
        "test_file": str(test_path),
        "mean_probabilities": [float(v) for v in share],
        "submission_file": str(out),
    })
    print("Recorded provenance in results/metrics/submission_*.json")
    print("\nUpload it at https://www.kaggle.com/competitions/llm-classification-finetuning/submit")


if __name__ == "__main__":
    main()
