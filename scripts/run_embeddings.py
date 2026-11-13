"""Frozen MiniLM + logistic regression on the team's existing split (validation only)."""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import load_data, saved_split
from src.embeddings import (ChunkedEncoder, EmbeddingFeatures, MODEL_NAME, MODEL_REVISION,
                            POLICY, cached_documents, validate_predictions)
from src.experiment import pred_path, save_predictions, write_json
from src.metrics import bootstrap_log_loss_ci, reliability_figure, summarize
from src.models import SWAP_PERM, class_prior_probs, swap_labels


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/raw/train.csv")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--cache", type=Path, default=ROOT / "data/cache/embeddings")
    ap.add_argument("--model-cache", type=Path, default=ROOT / "data/cache/huggingface")
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0, help="Fixed regularization; no validation grid search")
    ap.add_argument("--swap-aug", action="store_true", help="Train both answer orders and average predictions")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        ap.error("--revision must be an immutable 40-character model commit SHA")
    if args.C <= 0 or args.threads < 1:
        ap.error("C and threads must be positive")
    variant = "minilm_chunkmean_swapaug" if args.swap_aug else "minilm_chunkmean"
    output = pred_path(args.results, "embeddings", variant, args.seed)
    if output.exists():
        raise FileExistsError(f"Use a new --results directory to preserve the earlier run: {output}")
    # Never silently create new split assignments for this experiment.
    for name in ["manifest.json", "split_ids.csv"]:
        if not (args.splits / name).is_file():
            raise FileNotFoundError(f"Obtain the team's existing split file: {args.splits/name}")
    started = time.monotonic()
    df, fingerprint = load_data(args.data)
    split = saved_split(df, fingerprint, args.splits)
    train = (split == "train").to_numpy()
    validation = (split == "validation").to_numpy()
    # Local test rows are neither encoded nor fitted nor scored.
    active = df.loc[train | validation].reset_index(drop=True)
    is_train = train[train | validation]
    is_validation = validation[train | validation]
    y = active.label.to_numpy()
    versions = {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "torch",
                                       "sentence-transformers", "transformers", "tokenizers"]}
    versions["python"] = platform.python_version()
    configuration = {"model": MODEL_NAME, "revision": args.revision,
                     "max_tokens": args.max_tokens, "pooling": POLICY, "versions": versions}
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    encoder = ChunkedEncoder(args.revision, args.max_tokens, args.batch_size,
                             args.device, args.model_cache, args.local_files_only)
    configuration["dimension"] = encoder.dimension
    device = encoder.device
    texts = (active.prompt_text.tolist() + active.response_a_text.tolist()
             + active.response_b_text.tolist())
    print(f"Using {device}; train={is_train.sum():,}, validation={is_validation.sum():,}; local test excluded", flush=True)
    vectors, tokens, chunks, cache_path = cached_documents(texts, encoder, args.cache, configuration)
    n = len(active)
    feats = EmbeddingFeatures(vectors[:n], vectors[n:2*n], vectors[2*n:])
    del texts, encoder
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    feature_seconds = time.monotonic() - started
    X_train = feats.take(np.where(is_train)[0]).design()
    y_train = y[is_train]
    if args.swap_aug:
        X_train = np.vstack([X_train, feats.take(np.where(is_train)[0]).design(swap=True)])
        y_train = np.concatenate([y_train, swap_labels(y_train)])
    print(f"Fitting logistic regression: {X_train.shape}; C={args.C}", flush=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=args.C, max_iter=2000, random_state=args.seed, solver="lbfgs"))
    with threadpool_limits(limits=args.threads), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)
    del X_train
    val_feats = feats.take(np.where(is_validation)[0])
    p = model.predict_proba(val_feats.design())
    if args.swap_aug:
        p = .5 * (p + model.predict_proba(val_feats.design(swap=True))[:, SWAP_PERM])
    ids = active.id.to_numpy()[is_validation]
    labels = y[is_validation]
    expected = df.loc[validation]
    validate_predictions(ids, p, labels, expected.id.to_numpy(), expected.label.to_numpy())
    save_predictions(output, ids, p, labels)
    metrics = summarize(labels, p)
    metrics["log_loss_ci95"] = bootstrap_log_loss_ci(labels, p, expected.prompt_group.to_numpy())
    prior = summarize(labels, class_prior_probs(y[is_train], len(labels)))
    model_dir = args.results / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"embeddings__{variant}__seed{args.seed}.joblib"
    joblib.dump({"classifier": model, "encoder": configuration, "swap_aug": args.swap_aug}, model_path)
    figure_dir = args.results / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    reliability_figure(labels, {"class prior": class_prior_probs(y[is_train], len(labels)),
                               "embeddings": p}, figure_dir / f"embeddings_{variant}_reliability.png")
    report = {"model": "embeddings", "variant": variant, "partition": "validation",
              "dataset_sha256": fingerprint,
              "split_files_sha256": {name: file_sha(args.splits/name) for name in ["manifest.json", "split_ids.csv"]},
              "split_counts": {k: int(v) for k,v in split.value_counts().items()},
              "class_order": ["A wins", "B wins", "tie"], "seed": args.seed,
              "encoder": configuration, "encoder_frozen": True, "device": device,
              "classifier": {"kind": "StandardScaler + LogisticRegression", "C": args.C,
                             "hyperparameter_search": False, "iterations": model[-1].n_iter_.tolist(),
                             "feature_dimension": int(val_feats.design().shape[1])},
              "text_coverage": {"documents": len(tokens), "multi_chunk_documents": int((chunks>1).sum()),
                                "empty_documents": int((tokens==0).sum()), "max_content_tokens": int(tokens.max()),
                                "max_chunks": int(chunks.max()), "content_tokens_discarded": 0},
              "validation_metrics": metrics, "class_prior_metrics": prior,
              "log_loss_improvement_over_prior": prior["log_loss"]-metrics["log_loss"],
              "predictions": str(output.relative_to(args.results)), "predictions_sha256": file_sha(output),
              "classifier_artifact": str(model_path.relative_to(args.results)),
              "feature_seconds": feature_seconds, "elapsed_seconds": time.monotonic()-started,
              "limitations": ["Current v1 split retains the previously reported 27 cross-partition response-pair groups (196 rows).",
                              "English pretrained encoder; multilingual and code performance not separately established.",
                              "Chunk averaging loses interactions across chunks and explicit turn alignment.",
                              "Confidence interval is conditional on this fixed model and split; it is not training-seed variability.",
                              "Local test was not encoded, fitted or scored."]}
    write_json(args.results / "metrics" / f"embeddings_{variant}_validation.json", report)
    print(json.dumps({"validation": metrics, "prior_log_loss": prior["log_loss"],
                      "improvement": report["log_loss_improvement_over_prior"]}, indent=2), flush=True)
    print(f"Saved validated probabilities for Avani: {output}. Local test not scored.", flush=True)


if __name__ == "__main__":
    main()
