"""Error analysis for any model that saved a validation prediction file.

Slices required by the proposal: length gap, response type, confidence level, and
high-confidence errors. Adds the tie-class breakdown, the A/B asymmetry check and the
predicted-class confusion matrix.

Usage
  python scripts/run_error_analysis.py --models token_prior tfidf_lr numeric_hgb \
         --variants plain --seeds 42
  python scripts/run_error_analysis.py --pattern 'embeddings__*__validation.npz'

Outputs, under results/error_analysis/
  <model>__<variant>__seed<k>.json      every table plus headline numbers
  <model>__<variant>__seed<k>__confusion.csv
  <model>__<variant>__seed<k>__slices.csv
  <model>__<variant>__seed<k>__high_confidence_errors.csv
  figures/confusion_<model>_<variant>.png
  figures/confidence_profile_<model>_<variant>.png
and a cross-model summary at results/metrics/error_analysis_summary.json

The local test partition is never touched: only saved validation predictions are read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import load_predictions, load_split, write_json  # noqa: E402
from src.features import HEADER_RE, LIST_RE, REFUSAL_RE  # noqa: E402
from src.metrics import CLASS_NAMES, normalize, summarize  # noqa: E402

CODE_RE = None


def _has_code(t):
    return "```" in t


def response_flags(texts):
    return {
        "code": np.array([_has_code(t) for t in texts]),
        "markdown_list": np.array([bool(LIST_RE.search(t)) for t in texts]),
        "markdown_header": np.array([bool(HEADER_RE.search(t)) for t in texts]),
        "refusal": np.array([bool(REFUSAL_RE.search(t)) for t in texts]),
        "empty": np.array([len(t.strip()) == 0 for t in texts]),
        "very_long": np.array([len(t) > 4000 for t in texts]),
    }


def slice_table(y, p, slices):
    """One row per slice: n, log loss, accuracy, macro F1 on that subset."""
    rows = []
    pred = p.argmax(axis=1)
    for name, mask in slices.items():
        for val in (True, False):
            m = mask == val
            if m.sum() < 25:
                continue
            s = summarize(y[m], p[m])
            rows.append({"slice": name, "value": bool(val), "n": int(m.sum()),
                         "log_loss": s["log_loss"], "accuracy": s["accuracy"],
                         "macro_f1": s["macro_f1"],
                         "tie_recall": float((pred[m][y[m] == 2] == 2).mean())
                         if (y[m] == 2).any() else None})
    return pd.DataFrame(rows)


def analyse(y, p, df_va, stem):
    p = normalize(p)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    out = {"file": stem, "n": int(len(y)), "headline": summarize(y, p)}

    # confusion matrix, rows = true, cols = predicted
    cm = np.zeros((3, 3), dtype=int)
    for t, q in zip(y, pred):
        cm[t, q] += 1
    cmn = cm / cm.sum(axis=1, keepdims=True)
    out["confusion_counts"] = cm.tolist()
    out["confusion_row_normalised"] = np.round(cmn, 4).tolist()
    out["per_class_recall"] = [float(cmn[k, k]) for k in range(3)]
    out["predicted_distribution_when_true_tie"] = np.round(cmn[2], 4).tolist()
    out["true_distribution_when_predicted_tie"] = np.round(
        cm[:, 2] / max(cm[:, 2].sum(), 1), 4).tolist()

    # A/B asymmetry: does the model find one direction easier?
    out["accuracy_true_A"] = float((pred[y == 0] == 0).mean())
    out["accuracy_true_B"] = float((pred[y == 1] == 1).mean())
    out["accuracy_true_tie"] = float((pred[y == 2] == 2).mean())

    # confidence profile: 10 equal-mass bins
    order = np.argsort(-conf)
    bins = np.array_split(order, 10)
    prof = []
    for i, idx in enumerate(bins):
        prof.append({"decile": i + 1, "n": int(len(idx)),
                     "conf_min": float(conf[idx].min()), "conf_mean": float(conf[idx].mean()),
                     "accuracy": float((pred[idx] == y[idx]).mean()),
                     "log_loss": summarize(y[idx], p[idx])["log_loss"]})
    out["confidence_deciles"] = prof
    out["ece_by_decile_gap"] = float(np.mean([abs(r["accuracy"] - r["conf_mean"]) for r in prof]))

    # slices
    la = df_va["response_a_text"].to_numpy()
    lb = df_va["response_b_text"].to_numpy()
    len_ratio = np.log(np.array([len(t) + 1 for t in la])) - np.log(
        np.array([len(t) + 1 for t in lb]))
    fa, fb = response_flags(la), response_flags(lb)
    slices = {
        "abs_log_length_ratio>0.5": np.abs(len_ratio) > 0.5,
        "A_longer": len_ratio > 0,
        "A_has_code": fa["code"], "B_has_code": fb["code"],
        "A_has_list": fa["markdown_list"], "B_has_list": fb["markdown_list"],
        "A_has_header": fa["markdown_header"], "B_has_header": fb["markdown_header"],
        "A_refusal": fa["refusal"], "B_refusal": fb["refusal"],
        "A_empty": fa["empty"], "B_empty": fb["empty"],
        "A_very_long": fa["very_long"], "B_very_long": fb["very_long"],
        "prompt_len>median": (pd.Series([len(t) for t in df_va["prompt_text"]]).to_numpy()
                              > np.median([len(t) for t in df_va["prompt_text"]])),
    }
    st = slice_table(y, p, slices)
    worst = st.sort_values("log_loss", ascending=False).head(3).to_dict(orient="records")
    out["worst_slices"] = worst

    # high-confidence errors
    wrong = np.where(pred != y)[0]
    hc = wrong[np.argsort(-conf[wrong])][:100]
    hce = pd.DataFrame({
        "id": df_va["id"].to_numpy()[hc].astype(str),
        "top1_confidence": np.round(conf[hc], 4),
        "true_label": [CLASS_NAMES[v] for v in y[hc]],
        "predicted_label": [CLASS_NAMES[v] for v in pred[hc]],
        "abs_log_length_ratio": np.round(np.abs(len_ratio[hc]), 3),
        "prompt_chars": [len(t) for t in df_va["prompt_text"].to_numpy()[hc]],
        "prompt_head": [t[:120].replace("\n", " ") for t in df_va["prompt_text"].to_numpy()[hc]],
        "response_a_head": [t[:120].replace("\n", " ") for t in la[hc]],
        "response_b_head": [t[:120].replace("\n", " ") for t in lb[hc]],
    })
    out["high_confidence_errors"] = {
        "n_wrong": int(len(wrong)),
        "n_conf_above_0.6": int((conf[wrong] > 0.6).sum()),
        "n_conf_above_0.8": int((conf[wrong] > 0.8).sum()),
        "max_confidence_on_a_wrong_row": float(conf[wrong].max()) if len(wrong) else None,
        "examples": hce.head(10).to_dict(orient="records"),
    }
    return out, cm, cmn, st, hce, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--pattern", default=None,
                    help="Glob of prediction files; overrides --models/--variants/--seeds.")
    ap.add_argument("--models", nargs="+", default=["tfidf_lr", "numeric_lr", "numeric_hgb"])
    ap.add_argument("--variants", nargs="+", default=["plain"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--top-errors", type=int, default=100)
    args = ap.parse_args()
    results = Path(args.results)
    outdir = results / "error_analysis"
    figdir = results / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    df, fingerprint, split = load_split(args.data, args.splits)
    df_va = df.loc[split == "validation"].reset_index(drop=True)

    if args.pattern:
        paths = sorted((results / "predictions").glob(args.pattern))
    else:
        paths = [results / "predictions" / f"{m}__{v}__seed{s}__validation.npz"
                 for m in args.models for v in args.variants for s in args.seeds]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("No validation prediction files found. Run the models first.")

    summary = {"dataset_sha256": fingerprint, "partition": "validation",
               "local_test_scored": False, "files": {}}
    for path in paths:
        ids, p, y = load_predictions(path)
        lookup = df_va.set_index(df_va.id.astype(str))
        df_al = lookup.loc[list(ids)].reset_index(drop=True)
        out, cm, cmn, st, hce, prof = analyse(y, p, df_al, path.stem)
        write_json(outdir / f"{path.stem}.json", out)
        pd.DataFrame(cmn, index=[f"true_{c}" for c in CLASS_NAMES],
                     columns=[f"pred_{c}" for c in CLASS_NAMES]).to_csv(
            outdir / f"{path.stem}__confusion.csv")
        st.to_csv(outdir / f"{path.stem}__slices.csv", index=False)
        hce.head(args.top_errors).to_csv(
            outdir / f"{path.stem}__high_confidence_errors.csv", index=False)

        h = out["headline"]
        summary["files"][path.stem] = {
            "log_loss": h["log_loss"], "accuracy": h["accuracy"], "macro_f1": h["macro_f1"],
            "illegal_perm": "none",
            "accuracy_true_A": out["accuracy_true_A"], "accuracy_true_B": out["accuracy_true_B"],
            "accuracy_true_tie": out["accuracy_true_tie"],
            "tie_recall": out["per_class_recall"][2],
            "high_confidence_errors": out["high_confidence_errors"]["n_wrong"],
            "worst_slices": [w["slice"] for w in out["worst_slices"]],
        }
        print(f"{path.stem:<44} LL {h['log_loss']:.5f} acc {h['accuracy']:.4f} "
              f"tieRecall {out['per_class_recall'][2]:.3f} "
              f"highConfWrong {(out['high_confidence_errors']['n_wrong']):>4}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                        fontsize=9, color="black" if cmn[i, j] < 0.6 else "white")
        ax.set_xticks(range(3), CLASS_NAMES, rotation=20, ha="right")
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion — {path.stem}\n(row-normalised)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(figdir / f"confusion_{path.stem}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        d = pd.DataFrame(prof)
        ax.plot(d.conf_mean, d.accuracy, marker="o", label="accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.plot(d.conf_mean, d.conf_mean, color="grey", lw=0.8, label="mean confidence")
        ax.set_xlabel("Mean confidence in decile")
        ax.set_ylabel("Accuracy in decile")
        ax.set_title(f"Confidence profile — {path.stem}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / f"confidence_profile_{path.stem}.png", dpi=200)
        plt.close(fig)

    write_json(results / "metrics" / "error_analysis_summary.json", summary)
    best = min(summary["files"], key=lambda k: summary["files"][k]["log_loss"])
    print(f"\nBest raw log loss: {best} ({summary['files'][best]['log_loss']:.5f})")
    print("Saved error_analysis/ tables + figures and error_analysis_summary.json. "
          "Local test NOT scored.")


if __name__ == "__main__":
    main()
