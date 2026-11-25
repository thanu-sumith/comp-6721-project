"""Week 5 baselines: class prior, numeric features (LR / gradient boosting), TF-IDF + LR.

Every model is trained on the TRAIN partition and scored on VALIDATION using the
team's saved split. The local test partition is never touched here.

Hyperparameter selection is train-only (see src/selection.py): the C grid is scored by a
prompt-grouped CV **inside train**, never on validation. This keeps every model in the
comparison on the same footing -- in particular it does not let these baselines tune on
the same partition where the frozen-embedding model reported a fixed C = 1.

Each (model, variant) pair writes its own metrics shard
`results/metrics/baselines__<model>__<variant>.json` and is skipped on a rerun when that
shard exists, so the study can be executed in stages or resumed after an interruption.
`results/metrics/baselines_validation.json` merges every shard present.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import (load_split, numeric_features, pred_path, save_predictions,  # noqa: E402
                            write_json)
from src.features import build_tfidf  # noqa: E402
from src.metrics import bootstrap_log_loss_ci, reliability_figure, summarize  # noqa: E402
from src.models import KINDS, PreferenceModel, class_prior_probs  # noqa: E402
from src.selection import DEFAULT_GRIDS, select_C  # noqa: E402

VARIANTS = {"plain": False, "swapaug": True}  # swapaug = swap augmentation + swap-averaging


def merge_shards(results, fingerprint, prior_ll):
    shards = sorted((results / "metrics").glob("baselines__*.json"))
    report = {"dataset_sha256": fingerprint, "partition": "validation",
              "selection_protocol": "train-only prompt-grouped CV (src/selection.py)",
              "models": {}}
    for path in shards:
        shard = json.loads(path.read_text())
        report["models"].setdefault(shard["model"], {})[shard["variant"]] = shard
    if shards:
        report["best_by_model"] = {
            m: min(v.items(), key=lambda kv: kv[1]["variants"][0]["log_loss_mean"])[0]
            for m, v in report["models"].items()}
        report["prior_log_loss"] = prior_ll
    write_json(results / "metrics" / "baselines_validation.json", report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--models", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--cv-folds", type=int, default=3)
    ap.add_argument("--cv-seed", type=int, default=42)
    ap.add_argument("--fixed-C", type=float, default=None,
                    help="Skip selection and use this C for every model (matches a fixed-C protocol).")
    ap.add_argument("--no-resume", action="store_true", help="Recompute shards that already exist.")
    args = ap.parse_args()
    results = Path(args.results)

    t0 = time.time()
    df, fingerprint, split = load_split(args.data, args.splits)
    y = df.label.to_numpy()
    tr, va = (split == "train").to_numpy(), (split == "validation").to_numpy()
    ids_va, y_va, groups_va = df.id.to_numpy()[va], y[va], df.prompt_group.to_numpy()[va]
    groups_tr = df.prompt_group.to_numpy()[tr]
    print(f"train {tr.sum():,}  validation {va.sum():,}  dataset {fingerprint[:12]}")

    # ---- class prior (free, always recomputed, keeps its original filename) --------------
    prior = class_prior_probs(y[tr], int(va.sum()))
    s = summarize(y_va, prior)
    s["log_loss_ci95"] = bootstrap_log_loss_ci(y_va, prior, groups_va)
    save_predictions(pred_path(results, "class_prior", "plain", 42), ids_va, prior, y_va)
    prior_ll = s["log_loss"]
    write_json(results / "metrics" / "class_prior_validation.json", {
        "model": "training_class_prior", "evaluation_partition": "validation",
        "dataset_sha256": fingerprint, "seed": 42,
        "class_order": ["A wins", "B wins", "tie"],
        "training_priors": [float(v) for v in np.bincount(y[tr], minlength=3) / tr.sum()],
        "log_loss": prior_ll, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
        "log_loss_ci95": s["log_loss_ci95"]})
    print(f"class_prior         log loss {prior_ll:.6f}   accuracy {s['accuracy']:.4f}")

    for kind in args.models:
        feats = (numeric_features(df, fingerprint, args.cache) if kind.startswith("numeric")
                 else build_tfidf(df, tr))
        f_tr, f_va = feats.take(np.where(tr)[0]), feats.take(np.where(va)[0])
        for variant in args.variants:
            shard_path = results / "metrics" / f"baselines__{kind}__{variant}.json"
            if shard_path.exists() and not args.no_resume:
                print(f"{kind:<12} {variant:<8} already done ({shard_path.name}); skipping")
                continue
            swap = VARIANTS[variant]
            grid = [args.fixed_C] if args.fixed_C is not None else DEFAULT_GRIDS[kind]
            best_C, sel = select_C(kind, f_tr, y[tr], groups_tr, grid, seed=args.cv_seed,
                                   swap_aug=swap, n_splits=args.cv_folds,
                                   tag=f"{kind}/{variant}")
            runs, ts = [], []
            for seed in args.seeds:
                t = time.time()
                m = PreferenceModel(kind, seed, best_C, swap).fit(f_tr, y[tr])
                p = m.predict_proba(f_va)
                save_predictions(pred_path(results, kind, variant, seed), ids_va, p, y_va)
                row = summarize(y_va, p)
                row["seed"] = seed
                row["log_loss_ci95"] = bootstrap_log_loss_ci(y_va, p, groups_va)
                runs.append(row)
                ts.append(time.time() - t)
                print(f"{kind:<12} {variant:<8} C={best_C:<5} seed {seed}  "
                      f"LL {row['log_loss']:.6f}  acc {row['accuracy']:.4f}  "
                      f"F1 {row['macro_f1']:.4f}  ({ts[-1]:.0f}s)")
            lls = np.array([r["log_loss"] for r in runs])
            shard = {
                "model": kind, "variant": variant, "partition": "validation",
                "dataset_sha256": fingerprint, "seeds": args.seeds,
                "selection": sel, "chosen_C": best_C,
                "variants": [{"chosen_C": best_C, "seeds": args.seeds,
                              "log_loss_mean": float(lls.mean()), "log_loss_std": float(lls.std()),
                              "log_loss_min": float(lls.min()), "log_loss_max": float(lls.max()),
                              "accuracy_mean": float(np.mean([r["accuracy"] for r in runs])),
                              "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
                              "ece_top_label_mean": float(np.mean([r["ece_top_label"] for r in runs])),
                              "improvement_over_prior": float(prior_ll - lls.mean()),
                              "per_seed": runs,
                              "mean_fit_seconds": float(np.mean(ts))}],
                "prior_log_loss": prior_ll,
            }
            write_json(shard_path, shard)
            if variant == "plain":
                (results / "figures").mkdir(parents=True, exist_ok=True)
                p0 = np.load(pred_path(results, kind, variant, args.seeds[0]))["probs"]
                reliability_figure(y_va, {f"{kind} seed{args.seeds[0]}": p0},
                                   results / "figures" / f"reliability_{kind}_{variant}.png",
                                   title=f"{kind}/{variant} (uncalibrated)")
            merge_shards(results, fingerprint, prior_ll)

    report = merge_shards(results, fingerprint, prior_ll)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. "
          f"{len(list((results / 'metrics').glob('baselines__*.json')))} shard(s).")
    print("Saved baselines_validation.json. Local test NOT scored.")


if __name__ == "__main__":
    main()
