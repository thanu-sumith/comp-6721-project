"""Extension E1: bias and shortcut analysis.

Four questions, all answered on the fixed VALIDATION partition with everything fitted on
TRAIN only. Nothing here scores the local test partition.

1. Length shortcut. How much of the signal is just "the longer answer wins"?
   - a deterministic length-rule baseline (longer response wins, near-ties -> tie);
   - a single-feature logistic regression on log length ratio;
   - accuracy and log loss by |log length ratio| quartile for each real model.
   If a bag-of-words model barely beats the length rule, length is doing the work.

2. Position bias. Does the model prefer response A?
   - share of A-wins predictions vs the true share;
   - accuracy broken down by true label (a position prior shows up as high accuracy on
     A-wins and near-zero on B-wins);
   - swap flip rate: predict with (A,B) and with (B,A); a position-unbiased model keeps the
     same judged winner. Reported as flip rate and as mean P(A wins) - P(B wins) under swap.

3. Author / model-style predictiveness. `model_a` / `model_b` exist in the Kaggle training
   data but never in the hidden test set, so they are never model features. Here they are
   used only to measure how much of the label is explained by *who wrote the answer*:
   - a one-hot author-pair logistic regression scored on validation;
   - per-author win rates with counts (train partition).
   A high score here is not usable signal, it is a warning: it means part of what a text
   model learns may be author style rather than answer quality.

4. Numerical-feature attribution. Standardised logistic-regression coefficients, so the
   report can name the features the baseline actually leans on.

Outputs
  results/metrics/bias_analysis.json
  results/metrics/author_win_rates.csv
  results/metrics/numeric_lr_coefficients.csv
  results/metrics/bias_by_length_quartile.csv
  results/figures/bias_position_flip.png
  results/figures/bias_length_quartiles.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import load_split, numeric_features, pred_path, save_predictions, write_json  # noqa: E402
from src.features import build_tfidf  # noqa: E402
from src.metrics import normalize, summarize  # noqa: E402
from src.models import PreferenceModel, class_prior_probs, swap_labels  # noqa: E402

SWAP_PERM = np.array([1, 0, 2])


# --------------------------------------------------------------------------- length
def length_rule(y_va, la, lb, tie_band=0.02):
    """Deterministic rule: longer response wins; log-length ratio inside +-tie_band -> tie."""
    r = np.log(la + 1.0) - np.log(lb + 1.0)
    pred = np.where(np.abs(r) <= tie_band, 2, np.where(r > 0, 0, 1))
    acc = float((pred == y_va).mean())
    # a hard rule has no probabilities; smooth it into a softmax over three fixed scores so a
    # log loss can be quoted for it. Only the ordering matters, the scale is arbitrary.
    scores = np.stack([np.clip(r, 0, None), np.clip(-r, 0, None),
                       np.full_like(r, tie_band)], axis=1) * 3.0
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    return pred, p, {"rule": "longer wins, |log ratio|<=%.2f -> tie" % tie_band,
                     "accuracy": acc, "log_loss": summarize(y_va, p)["log_loss"],
                     "predicted_share": [float((pred == k).mean()) for k in range(3)]}


def length_only_lr(f_tr, y_tr, f_va, y_va):
    """One feature (log length A - log length B) through a multinomial LR."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    r_tr = (f_tr.FA[:, 0] - f_tr.FB[:, 0]).reshape(-1, 1)
    r_va = (f_va.FA[:, 0] - f_va.FB[:, 0]).reshape(-1, 1)
    est = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=2000, random_state=42)).fit(r_tr, y_tr)
    p = np.zeros((len(r_va), 3))
    p[:, est.classes_] = est.predict_proba(r_va)
    s = summarize(y_va, p)
    return p, {"features": ["log1p(len A) - log1p(len B)"],
               "log_loss": s["log_loss"], "accuracy": s["accuracy"],
               "macro_f1": s["macro_f1"],
               "coefficients": {str(k): float(c) for k, c in
                                zip(est.classes_, est[-1].coef_[:, 0])}}


def by_length_quartile(y_va, la, lb, models):
    ratio = np.abs(np.log(la + 1.0) - np.log(lb + 1.0))
    q = pd.qcut(ratio, 4, labels=False, duplicates="drop")
    rows = []
    for name, p in models.items():
        pred = p.argmax(axis=1)
        for b in range(q.max() + 1):
            m = q == b
            if not m.any():
                continue
            rows.append({"model": name, "quartile": int(b) + 1,
                         "abs_log_length_ratio_mean": float(ratio[m].mean()),
                         "n": int(m.sum()),
                         "log_loss": summarize(y_va[m], p[m])["log_loss"],
                         "accuracy": float((pred[m] == y_va[m]).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- position
def position_stats(y_va, p, name):
    pred = p.argmax(axis=1)
    out = {
        "model": name,
        "predicted_share": [float((pred == k).mean()) for k in range(3)],
        "true_share": [float((y_va == k).mean()) for k in range(3)],
        "accuracy_when_true_A": float((pred[y_va == 0] == 0).mean()),
        "accuracy_when_true_B": float((pred[y_va == 1] == 1).mean()),
        "accuracy_when_true_tie": float((pred[y_va == 2] == 2).mean()),
        "mean_pA_minus_pB": float((p[:, 0] - p[:, 1]).mean()),
        "flip_rate": None, "flip_rate_label_matched": None,
    }
    return out


def swap_flip(kind, f_tr, y_tr, f_va, y_va, seed=42, C=1.0):
    """Train once, then judge each validation pair twice: (A,B) and (B,A)."""
    m = PreferenceModel(kind, seed, C, False).fit(f_tr, y_tr)
    p = m.predict_proba(f_va)            # original ordering
    q = m.predict_proba_swapped(f_va)    # judged with A and B exchanged, mapped back
    pred, back = p.argmax(axis=1), q.argmax(axis=1)
    stats = position_stats(y_va, p, kind)
    stats["flip_rate"] = float((pred != back).mean())
    stats["accuracy_after_swap"] = float((back == y_va).mean())
    stats["log_loss_after_swap"] = summarize(y_va, q)["log_loss"]
    stats["log_loss_swap_averaged"] = summarize(y_va, normalize(0.5 * (p + q)))["log_loss"]
    stats["mean_pA_minus_pB_after_swap"] = float((q[:, 0] - q[:, 1]).mean())
    stats["swap_consistency"] = ("flip_rate = share of validation pairs whose judged winner "
                                 "changes when A and B are exchanged")
    return stats, p, q


# --------------------------------------------------------------------------- author
def author_analysis(df, tr, va, y, top_k=40):
    """One-hot (model_a, model_b) LR. Train-only vocabulary. Analysis only, never a feature."""
    tr_models = pd.concat([df.loc[tr, 'model_a'], df.loc[tr, 'model_b']])
    counts = tr_models.value_counts()
    keep = set(counts.head(top_k).index)
    win_a = (df.loc[tr].assign(w=df.loc[tr].label).groupby('model_a').w.apply(
        lambda s: float((s == 0).mean()))).to_dict()
    win_b = (df.loc[tr].assign(w=df.loc[tr].label).groupby('model_b').w.apply(
        lambda s: float((s == 1).mean()))).to_dict()
    rows = []
    for mname, n in counts.items():
        rows.append({"model": mname, "train_appearances": int(n),
                     "win_rate_as_A": win_a.get(mname),
                     "win_rate_as_B": win_b.get(mname)})
    rates = pd.DataFrame(rows).sort_values("train_appearances", ascending=False)

    def design(sub):
        cat = pd.Categorical(pd.concat([sub.model_a, sub.model_b]),
                             categories=sorted(keep))
        codes = np.asarray(cat.codes, dtype=int)
        n, k = len(sub), len(keep)
        ca, cb = codes[:n], codes[n:]
        cols = []
        for c, other in ((ca, cb), (cb, ca)):
            M = np.zeros((n, k + 1), dtype=np.float32)
            ok = c >= 0
            M[np.where(ok)[0], c[ok]] = 1.0
            M[~ok, k] = 1.0  # unseen author -> "other"
            cols.append(M)
        # interaction-ish: also mark same-author pairs
        same = (ca == cb).astype(np.float32).reshape(-1, 1)
        return np.hstack([cols[0], cols[1], same])

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    Xtr, Xva = design(df.loc[tr]), design(df.loc[va])
    m = make_pipeline(StandardScaler(with_mean=False),
                      LogisticRegression(C=1.0, max_iter=2000, random_state=42)).fit(Xtr, y[tr])
    p = normalize(m.predict_proba(Xva))
    s = summarize(y[va], p)
    return rates, p, {
        "features": f"one-hot model_a ({len(keep)+1} incl. 'other') + one-hot model_b + same-author flag",
        "train_vocabulary_size": int(len(keep)),
        "log_loss": s["log_loss"], "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
        "note": ("Author identity is present in the training data and absent from the hidden "
                 "test set. A low log loss here measures how much of the label is explained by "
                 "who wrote the answer; it is a leakage/shortcut warning, not usable signal."),
    }


# --------------------------------------------------------------------- coefficients
def numeric_coefficients(f_tr, y_tr, names):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = f_tr.design(swap=False)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
                                                          random_state=42)).fit(X, y_tr)
    coef = m[-1].coef_  # (3, n_features)
    rows = []
    for j, name in enumerate(names):
        rows.append({"feature": name,
                     "coef_A_wins": float(coef[0, j]),
                     "coef_B_wins": float(coef[1, j]),
                     "coef_tie": float(coef[2, j]),
                     "max_abs_coef": float(np.abs(coef[:, j]).max())})
    return pd.DataFrame(rows).sort_values("max_abs_coef", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--swap-models", nargs="+", default=["numeric_lr", "tfidf_lr"])
    ap.add_argument("--author-top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    results = Path(args.results)

    df, fingerprint, split = load_split(args.data, args.splits)
    y = df.label.to_numpy()
    tr, va = (split == "train").to_numpy(), (split == "validation").to_numpy()
    tr_idx, va_idx = np.where(tr)[0], np.where(va)[0]
    y_tr, y_va = y[tr], y[va]
    la, lb = df.response_a_text.str.len().to_numpy()[va], df.response_b_text.str.len().to_numpy()[va]
    out = {"dataset_sha256": fingerprint, "partition": "validation", "seed": args.seed,
           "local_test_scored": False}

    # ---- 1. length -------------------------------------------------------------------
    feats = numeric_features(df, fingerprint, args.cache)
    f_tr, f_va = feats.take(tr_idx), feats.take(va_idx)
    rule_pred, rule_p, rule = length_rule(y_va, la, lb)
    lr_p, lr = length_only_lr(f_tr, y_tr, f_va, y_va)
    out["length_shortcut"] = {
        "rule": rule, "single_feature_lr": lr,
        "prior_log_loss": summarize(y_va, class_prior_probs(y_tr, len(y_va)))["log_loss"],
        "note": ("Compare a real model's validation log loss against these two floors. If it is "
                 "close, length explains most of the signal."),
    }
    save_predictions(pred_path(results, "length_rule", "plain", args.seed),
                     df.id.to_numpy()[va], rule_p, y_va)
    save_predictions(pred_path(results, "length_lr", "plain", args.seed),
                     df.id.to_numpy()[va], lr_p, y_va)
    print(f"[length] rule acc {rule['accuracy']:.4f}  LL {rule['log_loss']:.5f} | "
          f"1-feature LR acc {lr['accuracy']:.4f}  LL {lr['log_loss']:.5f}")

    # ---- 2. position -----------------------------------------------------------------
    models = {"length_lr": lr_p, "length_rule": rule_p}
    pos = []
    for kind in args.swap_models:
        if kind.startswith("numeric"):
            fk_tr, fk_va = f_tr, f_va
        else:
            fk = build_tfidf(df, tr)
            fk_tr, fk_va = fk.take(tr_idx), fk.take(va_idx)
        stats, p, q = swap_flip(kind, fk_tr, y_tr, fk_va, y_va, seed=args.seed)
        pos.append(stats)
        models[kind] = p
        models[kind + "_swapaveraged"] = normalize(0.5 * (p + q))
        print(f"[position] {kind:<12} A-share {stats['predicted_share'][0]:.3f} vs true "
              f"{stats['true_share'][0]:.3f} | acc(A) {stats['accuracy_when_true_A']:.3f} "
              f"acc(B) {stats['accuracy_when_true_B']:.3f} | flip {stats['flip_rate']:.4f}")
    for name, p in models.items():
        if not any(name == s["model"] for s in pos):
            pos.append(position_stats(y_va, p, name))
    out["position_bias"] = pos

    # ---- 3. author -------------------------------------------------------------------
    if "model_a" in df.columns and "model_b" in df.columns:
        rates, ap_, astats = author_analysis(df, tr, va, y, args.author_top_k)
        rates.to_csv(results / "metrics" / "author_win_rates.csv", index=False)
        out["author_predictiveness"] = astats
        models["author_only"] = ap_
        print(f"[author] one-hot author-pair LR  LL {astats['log_loss']:.5f}  "
              f"acc {astats['accuracy']:.4f}  (analysis only, never a feature)")
    else:
        out["author_predictiveness"] = {"skipped": "model_a/model_b columns absent"}
        print("[author] skipped: model_a/model_b not in the CSV")

    # ---- 4. coefficients -------------------------------------------------------------
    coef = numeric_coefficients(f_tr, y_tr, feats.names())
    coef.to_csv(results / "metrics" / "numeric_lr_coefficients.csv", index=False)
    out["top_numeric_features"] = coef.head(12).to_dict(orient="records")
    print("[coef] top numeric features by |coef|:",
          ", ".join(coef.feature.head(6)))

    # ---- 1b. length quartile table ---------------------------------------------------
    tq = by_length_quartile(y_va, la, lb, models)
    tq.to_csv(results / "metrics" / "bias_by_length_quartile.csv", index=False)

    # ---- figures ---------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (results / "figures").mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    names = list(models)
    flips = [next((s["flip_rate"] for s in pos if s["model"] == n), np.nan) for n in names]
    ok = ~np.isnan(np.array(flips, dtype=float))
    if ok.any():
        ax.bar(np.array(names)[ok], np.array(flips)[ok], color="#4472c4")
        ax.set_ylabel("Share of pairs whose judged winner flips")
        ax.set_title("Position bias: A/B swap flip rate (lower is better)")
        ax.set_ylim(0, max(0.05, float(np.nanmax(np.array(flips, dtype=float))) * 1.25))
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(results / "figures" / "bias_position_flip.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, g in tq.groupby("model"):
        if name == "length_rule":
            continue
        ax.plot(g.quartile, g.log_loss, marker="o", label=name)
    ax.axhline(out["length_shortcut"]["prior_log_loss"], color="k", ls="--", lw=1,
               label="class prior")
    ax.set_xlabel("Quartile of |log length ratio| (1 = similar lengths, 4 = very different)")
    ax.set_ylabel("Validation log loss")
    ax.set_title("Does the model only work when one answer is longer?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "figures" / "bias_length_quartiles.png", dpi=200)
    plt.close(fig)

    write_json(results / "metrics" / "bias_analysis.json", out)
    print("Saved bias_analysis.json, author_win_rates.csv, numeric_lr_coefficients.csv, "
          "bias_by_length_quartile.csv and two figures. Local test NOT scored.")


if __name__ == "__main__":
    main()
