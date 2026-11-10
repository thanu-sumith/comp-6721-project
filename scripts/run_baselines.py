"""Week 5 baselines: class prior, numeric features (LR / gradient boosting), TF-IDF + LR.

Every model is trained on the TRAIN partition and scored on VALIDATION using the
team's saved split. The local test partition is never touched here.

"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import (load_split, numeric_features, pred_path, save_predictions,  # noqa: E402
                            write_json)
from src.features import build_tfidf  # noqa: E402
from src.metrics import bootstrap_log_loss_ci, reliability_figure, summarize  # noqa: E402
from src.models import KINDS, PreferenceModel, class_prior_probs  # noqa: E402

C_GRID = {"numeric_lr": [0.01, 0.1, 1.0], "tfidf_lr": [0.3, 1.0, 3.0], "numeric_hgb": [None]}
VARIANTS = {"plain": False, "swapaug": True}  # swapaug = swap augmentation + swap-averaging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--models", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    results = Path(args.results)

    df, fingerprint, split = load_split(args.data, args.splits)
    y = df.label.to_numpy()
    tr, va = (split == "train").to_numpy(), (split == "validation").to_numpy()
    ids_va, y_va, groups_va = df.id.to_numpy()[va], y[va], df.prompt_group.to_numpy()[va]

    report = {"dataset_sha256": fingerprint, "partition": "validation", "models": {}}

    prior = class_prior_probs(y[tr], va.sum())
    s = summarize(y_va, prior)
    s["log_loss_ci95"] = bootstrap_log_loss_ci(y_va, prior, groups_va)
    report["models"]["class_prior"] = {"plain": {"summary": s}}
    save_predictions(pred_path(results, "class_prior", "plain", 42), ids_va, prior, y_va)
    prior_ll = s["log_loss"]
    print(f"class_prior        log loss {prior_ll:.6f}")

    for kind in args.models:
        if kind.startswith("numeric"):
            feats = numeric_features(df, fingerprint, args.cache)
        else:
            feats = build_tfidf(df, tr)
        f_tr, f_va = feats.take(np.where(tr)[0]), feats.take(np.where(va)[0])
        report["models"][kind] = {}
        for variant in args.variants:
            swap = VARIANTS[variant]
            # Select C on validation with the first seed (the grid is tiny; see report note).
            best_C, best_ll = None, np.inf
            for C in C_GRID[kind]:
                m = PreferenceModel(kind, args.seeds[0], C or 1.0, swap).fit(f_tr, y[tr])
                ll = summarize(y_va, m.predict_proba(f_va))["log_loss"]
                if ll < best_ll:
                    best_C, best_ll = C, ll
            runs = []
            for seed in args.seeds:
                m = PreferenceModel(kind, seed, best_C or 1.0, swap).fit(f_tr, y[tr])
                p = m.predict_proba(f_va)
                save_predictions(pred_path(results, kind, variant, seed), ids_va, p, y_va)
                runs.append(summarize(y_va, p))
            lls = np.array([r["log_loss"] for r in runs])
            entry = {
                "chosen_C": best_C, "seeds": args.seeds,
                "log_loss_mean": float(lls.mean()), "log_loss_std": float(lls.std()),
                "improvement_over_prior": float(prior_ll - lls.mean()),
                "accuracy_mean": float(np.mean([r["accuracy"] for r in runs])),
                "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
                "ece_mean": float(np.mean([r["ece_top_label"] for r in runs])),
                "per_seed": runs,
                "log_loss_ci95_seed0": bootstrap_log_loss_ci(
                    y_va, np.load(pred_path(results, kind, variant, args.seeds[0]))["probs"], groups_va),
            }
            report["models"][kind][variant] = entry
            print(f"{kind:<12} {variant:<8} log loss {entry['log_loss_mean']:.6f} "
                  f"± {entry['log_loss_std']:.6f}  acc {entry['accuracy_mean']:.4f}  "
                  f"ECE {entry['ece_mean']:.4f}  (C={best_C})")

    write_json(results / "metrics" / "baselines_validation.json", report)
    fig_probs = {"class_prior": prior}
    for kind in args.models:
        for variant in args.variants:
            fig_probs[f"{kind}/{variant}"] = np.load(
                pred_path(results, kind, variant, args.seeds[0]))["probs"]
    (results / "figures").mkdir(parents=True, exist_ok=True)
    reliability_figure(y_va, fig_probs, results / "figures" / "baselines_reliability.png",
                       title="Baselines: reliability on validation")
    print("Saved metrics, predictions and reliability figure. Local test NOT scored.")


if __name__ == "__main__":
    main()
