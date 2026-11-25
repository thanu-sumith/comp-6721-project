"""Build the single experiment table the report needs, from whatever result files exist.

Reads every metrics artifact that is present and writes
  results/metrics/experiment_table.json
  reports/results_table.md

Design rules applied here (they are the point of the script):
  * the PRIMARY metric, multiclass log loss, is the first numeric column;
  * every model is compared against the same class-prior floor on the same partition;
  * an uncalibrated row and a temperature-scaled row are separate rows, never merged;
  * hyperparameter-selection protocol is printed, so a validation-tuned model cannot be
    silently compared with a fixed-hyperparameter one;
  * seed mean +- spread is shown when more than one seed exists.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import write_json  # noqa: E402

SELECTION_LABEL = {
    "train_only_grouped_cv_3fold": "train-CV (3-fold, prompt-grouped)",
    "train_only_grouped_cv_2fold": "train-CV (2-fold, prompt-grouped)",
    "no_selection": "none (single value)",
}


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=5):
    return "—" if x is None else f"{x:.{nd}f}"


def pct(x):
    return "—" if x is None else f"{x * 100:.2f}%"


def pm(mean, std, nd=5):
    if mean is None:
        return "—"
    return f"{mean:.{nd}f}" + ("" if not std else f" ± {std:.{nd}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None,
                    help="Defaults to <results>/results_table.md; a copy is also written to "
                         "reports/results_table.md only when the default results dir is used.")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows cp1252 cannot print Δ/+-/>
    R = Path(args.results)
    out = Path(args.out) if args.out else (ROOT / "reports" / "results_table.md"
                                           if R.resolve() == (ROOT / "results").resolve()
                                           else R / "results_table.md")

    prior = load(R / "metrics" / "class_prior_validation.json")
    prior_ll = prior["log_loss"] if prior else None
    cal = load(R / "metrics" / "calibration_study.json") or {}
    cal_sum = cal.get("summary_mean_over_seeds", {})
    rows = []

    if prior:
        rows.append({
            "model": "class prior", "variant": "—", "selection": "none",
            "log_loss": prior_ll, "log_loss_std": None,
            "accuracy": prior["accuracy"], "macro_f1": prior["macro_f1"],
            "log_loss_cal": None, "ece": None, "ece_cal": None,
            "note": "constant training priors; macro F1 collapses because B and tie are never predicted",
        })

    for shard_path in sorted((R / "metrics").glob("baselines__*.json")):
        sh = json.loads(shard_path.read_text())
        v = sh["variants"][0]
        key = f"{sh['model']}/{sh['variant']}"
        c = cal_sum.get(key, {})
        rows.append({
            "model": sh["model"], "variant": sh["variant"],
            "selection": SELECTION_LABEL.get(sh["selection"].get("protocol"),
                                             sh["selection"].get("protocol")),
            "chosen_C": sh.get("chosen_C"),
            "log_loss": v["log_loss_mean"], "log_loss_std": v.get("log_loss_std"),
            "accuracy": v["accuracy_mean"], "macro_f1": v["macro_f1_mean"],
            "log_loss_cal": c.get("log_loss_after"),
            "ece": c.get("ece_before", v.get("ece_top_label_mean")),
            "ece_cal": c.get("ece_after"),
            "seeds": v.get("seeds"), "note": "",
        })

    emb = load(R / "metrics" / "embeddings_minilm_chunkmean_validation.json")
    if emb:
        vm = emb["validation_metrics"]
        c = cal_sum.get("embeddings/minilm_chunkmean", {})
        rows.append({
            "model": "MiniLM + LR (Sumith)", "variant": "minilm_chunkmean",
            "selection": "none (fixed C = 1)",
            "log_loss": vm["log_loss"], "log_loss_std": None,
            "accuracy": vm["accuracy"], "macro_f1": vm["macro_f1"],
            "log_loss_cal": c.get("log_loss_after"),
            "ece": vm.get("ece_top_label"), "ece_cal": c.get("ece_after"),
            "note": "frozen encoder; accuracy gain is real, raw log loss is above the prior",
        })

    bias = load(R / "metrics" / "bias_analysis.json")
    if bias:
        lr = bias["length_shortcut"]["single_feature_lr"]
        rule = bias["length_shortcut"]["rule"]
        rows.append({"model": "length only (1 feature LR)", "variant": "—",
                     "selection": "none", "log_loss": lr["log_loss"], "accuracy": lr["accuracy"],
                     "macro_f1": lr["macro_f1"], "note": "E1 shortcut floor"})
        rows.append({"model": "length rule (longer wins)", "variant": "—",
                     "selection": "none", "log_loss": rule["log_loss"],
                     "accuracy": rule["accuracy"], "macro_f1": None,
                     "note": "E1 deterministic shortcut"})
        ap_ = bias.get("author_predictiveness", {})
        if "log_loss" in ap_:
            rows.append({"model": "author identity only", "variant": "—",
                         "selection": "none", "log_loss": ap_["log_loss"],
                         "accuracy": ap_["accuracy"], "macro_f1": ap_["macro_f1"],
                         "note": "E1; analysis only — author is absent from the hidden test set"})

    rows.sort(key=lambda r: (r["log_loss"] is None, r["log_loss"] if r["log_loss"] is not None else 9))

    L = ["# Experiment table (validation partition)", "",
         "Auto-generated by `scripts/summarize_results.py`. Primary metric first: "
         "**multiclass log loss, lower is better**. The local test partition is not scored.", "",
         "| Model | Variant | Selection | Log loss (primary) | Δ vs prior | Accuracy | Macro F1 "
         "| Log loss, temp-scaled | ECE raw → cal |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        d = ("—" if (r["log_loss"] is None or prior_ll is None) else f"{r['log_loss'] - prior_ll:+.5f}")
        ece = ("—" if r.get("ece") is None else
               f"{r['ece']:.4f}" + ("" if r.get("ece_cal") is None else f" → {r['ece_cal']:.4f}"))
        L.append(f"| {r['model']} | {r['variant']} | {r['selection']} | "
                 f"{pm(r['log_loss'], r.get('log_loss_std'))} | {d} | {pct(r['accuracy'])} | "
                 f"{fmt(r['macro_f1'], 4)} | {fmt(r.get('log_loss_cal'))} | {ece} |")
    L += ["", "Notes:", ""]
    for r in rows:
        if r.get("note"):
            L.append(f"- **{r['model']} / {r['variant']}** — {r['note']}")
    if prior_ll is not None:
        L += ["", f"Class-prior floor: log loss **{prior_ll:.6f}**, accuracy {pct(prior['accuracy'])}, "
                  f"macro F1 {prior['macro_f1']:.6f}. Uniform three-class predictor = "
                  f"ln 3 = {__import__('math').log(3):.6f}.", ""]
    de = load(R / "metrics" / "data_efficiency.json")
    if de:
        L += ["## E3 summary", "",
              "| E3 item | Result |", "|---|---|"]
        ln = de["label_noise"]["duplicate_conflicts_whole_dataset"]
        L.append(f"| Duplicate prompt/response-pair groups | {ln['groups_with_repeats']} |")
        rate = ln.get("conflict_rate_among_repeated_groups")
        L.append(f"| …with conflicting labels | {ln['groups_with_conflicting_labels']}"
                 + ("" if rate is None else f" ({rate:.1%} of repeated groups)") + " |")
        for rec in de["learning_curves"]:
            L.append(f"| {rec['model']} @ {rec['fraction']:.0%} of train groups "
                     f"({rec['n_train']:.0f} rows) | log loss {rec['log_loss_mean']:.5f} "
                     f"± {rec['log_loss_std'] or 0:.5f} |")
        L.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(R / "metrics" / "experiment_table.json", {"prior_log_loss": prior_ll, "rows": rows})
    print("\n".join(L[:8]))
    print(f"\nSaved {out} and experiment_table.json ({len(rows)} rows).")


if __name__ == "__main__":
    main()
