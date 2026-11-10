"""Extension E3: data-efficiency learning curves and a label-noise audit.

1. Learning curves: train on 10/25/50/100% of the TRAIN prompt groups (several seeds),
   score on the fixed VALIDATION partition.
2. Label-noise audit: (a) conflicting labels on identical prompt/response pairs,
   (b) out-of-fold probability of the given label, flagging the least-supported rows
   (within each class). Flagged rows are candidates for manual review, not proven errors.
3. Retrain on cleaned training data (validation labels are never edited) and compare.

"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import load_split, numeric_features, write_json  # noqa: E402
from src.features import build_tfidf  # noqa: E402
from src.metrics import summarize  # noqa: E402
from src.models import KINDS, PreferenceModel, class_prior_probs, swap_labels  # noqa: E402

DEFAULT_C = {"numeric_lr": 0.1, "tfidf_lr": 1.0, "numeric_hgb": 1.0}


def sample_groups(groups, y_rows, fraction, seed):
    """Row indices (into the train arrays) of a random `fraction` of prompt groups."""
    if fraction >= 1.0:
        return np.arange(len(groups))
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    sizes = pd.Series(groups).value_counts()
    target, total, chosen = fraction * len(groups), 0, []
    for g in uniq:
        chosen.append(g)
        total += sizes[g]
        if total >= target:
            break
    idx = np.where(np.isin(groups, chosen))[0]
    return idx


def conflicting_duplicates(df):
    """Identical (prompt, response pair) rows, treating A/B-swapped copies as the same pair."""
    swapped = (df.response_a_text > df.response_b_text).to_numpy()
    lo = np.where(swapped, df.response_b_text, df.response_a_text)
    hi = np.where(swapped, df.response_a_text, df.response_b_text)
    canon = np.where(swapped, swap_labels(df.label.to_numpy()), df.label.to_numpy())
    t = pd.DataFrame({"p": df.prompt_text.to_numpy(), "lo": lo, "hi": hi, "label": canon})
    g = t.groupby(["p", "lo", "hi"])["label"].agg(["size", "nunique"])
    dup = g[g["size"] > 1]
    return {
        "groups_with_repeats": int(len(dup)),
        "rows_in_repeated_groups": int(dup["size"].sum()),
        "groups_with_conflicting_labels": int((dup["nunique"] > 1).sum()),
        "conflict_rate_among_repeated_groups": float((dup["nunique"] > 1).mean()) if len(dup) else None,
    }


def fit_score(kind, f_tr, y_tr, f_va, y_va, seed, swap_aug):
    m = PreferenceModel(kind, seed, DEFAULT_C[kind], swap_aug).fit(f_tr, y_tr)
    return summarize(y_va, m.predict_proba(f_va))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--models", nargs="+", default=["numeric_hgb", "tfidf_lr"], choices=KINDS)
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--clean-fractions", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--audit-model", default="numeric_hgb", choices=KINDS)
    ap.add_argument("--no-swap-aug", action="store_true")
    args = ap.parse_args()
    results, swap_aug = Path(args.results), not args.no_swap_aug

    df, fingerprint, split = load_split(args.data, args.splits)
    y = df.label.to_numpy()
    tr_idx, va_idx = np.where(split == "train")[0], np.where(split == "validation")[0]
    y_tr, y_va = y[tr_idx], y[va_idx]
    groups_tr = df.prompt_group.to_numpy()[tr_idx]
    prior_ll = summarize(y_va, class_prior_probs(y_tr, len(y_va)))["log_loss"]

    feats = {}
    for kind in set(args.models) | {args.audit_model}:
        feats[kind] = (numeric_features(df, fingerprint, args.cache) if kind.startswith("numeric")
                       else build_tfidf(df, (split == "train").to_numpy()))
    out = {"dataset_sha256": fingerprint, "swap_aug": swap_aug, "prior_log_loss": prior_ll,
           "learning_curves": {}, "label_noise": {}}

    # ---- 1. learning curves -------------------------------------------------------------
    rows = []
    for kind in args.models:
        f_tr, f_va = feats[kind].take(tr_idx), feats[kind].take(va_idx)
        for frac in args.fractions:
            for seed in args.seeds:
                sub = sample_groups(groups_tr, y_tr, frac, seed)
                s = fit_score(kind, f_tr.take(sub), y_tr[sub], f_va, y_va, seed, swap_aug)
                rows.append({"model": kind, "fraction": frac, "seed": seed, "n_train": int(len(sub)),
                             "log_loss": s["log_loss"], "accuracy": s["accuracy"],
                             "macro_f1": s["macro_f1"], "ece": s["ece_top_label"]})
                print(f"[curve] {kind:<12} {frac:>4.0%}  seed {seed}  n={len(sub):>6}  "
                      f"LL {s['log_loss']:.5f}  acc {s['accuracy']:.4f}")
    curve = pd.DataFrame(rows)
    (results / "metrics").mkdir(parents=True, exist_ok=True)
    curve.to_csv(results / "metrics" / "learning_curves.csv", index=False)
    agg = curve.groupby(["model", "fraction"]).agg(
        n_train=("n_train", "mean"), log_loss_mean=("log_loss", "mean"),
        log_loss_std=("log_loss", "std"), accuracy_mean=("accuracy", "mean")).reset_index()
    out["learning_curves"] = agg.to_dict(orient="records")

    # ---- 2. label-noise audit -----------------------------------------------------------
    out["label_noise"]["duplicate_conflicts_whole_dataset"] = conflicting_duplicates(df)
    print("[audit] duplicate-pair label conflicts:", out["label_noise"]["duplicate_conflicts_whole_dataset"])

    kind = args.audit_model
    f_tr, f_va = feats[kind].take(tr_idx), feats[kind].take(va_idx)
    p_true = np.zeros(len(tr_idx))
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for fit_i, oof_i in skf.split(np.zeros(len(tr_idx)), y_tr, groups_tr):
        m = PreferenceModel(kind, 42, DEFAULT_C[kind], swap_aug).fit(f_tr.take(fit_i), y_tr[fit_i])
        p_true[oof_i] = m.predict_proba(f_tr.take(oof_i))[np.arange(len(oof_i)), y_tr[oof_i]]
    order = np.argsort(p_true)  # least-supported labels first
    out["label_noise"]["oof_mean_p_true"] = float(p_true.mean())

    # ---- 3. retrain on cleaned training data -------------------------------------------
    clean_rows = []
    for q in args.clean_fractions:
        # Drop the lowest-support q% WITHIN each class, so ties (the hardest class) are not
        # removed wholesale; otherwise "cleaning" would just delete the tie class.
        drop = np.concatenate([
            (lambda o: o[: int(round(q * len(o)))])(order[y_tr[order] == k]) for k in range(3)])
        keep = np.setdiff1d(np.arange(len(order)), drop)
        flagged_classes = np.bincount(y_tr[drop], minlength=3).tolist() if len(drop) else [0, 0, 0]
        for seed in args.seeds:
            s = fit_score(kind, f_tr.take(keep), y_tr[keep], f_va, y_va, seed, swap_aug)
            clean_rows.append({"drop_fraction": q, "seed": seed, "n_train": int(len(keep)),
                               "log_loss": s["log_loss"], "accuracy": s["accuracy"],
                               "ece": s["ece_top_label"], "dropped_by_class": flagged_classes})
        sub = pd.DataFrame([r for r in clean_rows if r["drop_fraction"] == q])
        print(f"[clean] drop {q:>5.0%}  n={int(sub.n_train.iloc[0])}  LL {sub.log_loss.mean():.5f}  "
              f"acc {sub.accuracy.mean():.4f}  dropped by class {flagged_classes}")
    out["label_noise"]["retrain_on_cleaned"] = clean_rows

    qmax = max(args.clean_fractions)
    flagged = np.concatenate([
        (lambda o: o[: int(round(qmax * len(o)))])(order[y_tr[order] == k]) for k in range(3)])
    pd.DataFrame({"id": df.id.to_numpy()[tr_idx][flagged], "oof_p_true": p_true[flagged],
                  "label": y_tr[flagged]}).sort_values("oof_p_true").to_csv(
        results / "metrics" / "noise_audit_flagged_ids.csv", index=False)
    write_json(results / "metrics" / "data_efficiency.json", out)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (results / "figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for kind_, g in agg.groupby("model"):
        ax.errorbar(g.n_train, g.log_loss_mean, yerr=g.log_loss_std.fillna(0), marker="o",
                    capsize=3, label=kind_)
    ax.axhline(prior_ll, color="k", ls="--", lw=1, label="class prior")
    ax.set_xscale("log")
    ax.set_xlabel("Training rows (10/25/50/100% of train prompt groups)")
    ax.set_ylabel("Validation log loss")
    ax.set_title("Learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "figures" / "learning_curves.png", dpi=200)
    plt.close(fig)
    cr = pd.DataFrame(clean_rows).groupby("drop_fraction").log_loss.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(cr.drop_fraction * 100, cr["mean"], yerr=cr["std"].fillna(0), marker="o", capsize=3)
    ax.axhline(prior_ll, color="k", ls="--", lw=1, label="class prior")
    ax.set_xlabel("% of lowest-support training labels removed")
    ax.set_ylabel("Validation log loss (original labels)")
    ax.set_title(f"Retraining on cleaned data ({args.audit_model})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "figures" / "label_cleaning.png", dpi=200)
    plt.close(fig)
    print("Saved data_efficiency.json, learning_curves.csv, flagged ids, figures. Local test NOT scored.")


if __name__ == "__main__":
    main()
