# Baselines, calibration studies and error analysis — methods section

Owner: Avani Dipakbhai Gajjar (40345876). Covers proposal §4 (baselines), §5 (evaluation and
validation), §6 weeks 5 / 10–11 / 11–12, and extensions **E1** and **E3**.

Every number quoted in this section is produced by the scripts listed at the end. Nothing here
scores the local test partition; the single late Kaggle submission is the only test evaluation.

---

## 1. Evaluation protocol

**Metrics.** Primary: multiclass log loss over three classes in the fixed order
`0 = A wins, 1 = B wins, 2 = tie`. Secondary: accuracy and macro F1. Calibration: expected
calibration error (ECE, top-label, 15 equal-width bins, plus a class-wise variant because the
tie probability matters on its own) and reliability diagrams with confidence histograms.

**Floor.** A training-class-prior predictor is reported alongside every model. The proposal
quotes ln 3 ≈ 1.0986 as the uniform floor; the *measured* prior floor is **1.097224**, which is
lower because the classes are imbalanced (34.91 % / 34.19 % / 30.90 %). We report the measured
value, not the theoretical one, so every improvement is stated against the real floor.

**Split.** The team's frozen v1 assignment (`data/splits/split_ids.csv`, seed 42, nested
`StratifiedGroupKFold` grouped by parsed prompt, dataset SHA-256 `0692154a…f5be2`) is reused
unchanged: 34,487 train / 11,496 validation / 11,494 local test. No prompt group crosses a
partition, and every model in this report uses those exact assignments.

**Hyperparameter selection is train-only.** This is a deliberate methodological change from the
first version of `run_baselines.py`, which scored its `C` grid *on the validation partition* and
then reported validation metrics for the winning `C`. Two problems followed: the reported numbers
were optimistically biased, and the baselines would have been tuned on validation while the
frozen-embedding model reported a fixed `C = 1`, making the comparison unequal. Selection now
runs a prompt-grouped cross-validation **inside train** (`src/selection.py`), so validation is
used only for reporting. The chosen `C`, the per-fold CV losses and the protocol name are saved
in every result file. The one residual optimism is that the TF-IDF vocabulary is fitted on the
whole training partition (never on validation), so a CV fold's held-out rows contributed document
frequencies to the vocabulary; no validation row, label or prompt enters feature fitting or
selection.

**Calibration parameters are cross-fitted.** A single scalar temperature is fitted with
`scipy.optimize.minimize_scalar` on the negative log likelihood of one validation fold and applied
to the other, then the roles are reversed (2 folds, grouped by prompt). No validation row is
scored by a calibrator that saw it. The temperature fitted on the **full** validation partition is
saved separately and is the one to reuse for the final test submission; it is never used to score
the rows it was fitted on.

---

## 2. Baselines

**(a) Class prior.** Training class frequencies repeated for every validation row. No text
features. Accuracy is 34.91 % because the hard prediction is always "A wins"; macro F1 is 0.1725
because B wins and tie are never predicted. This is the majority-class floor, not a model.

**(b) Numeric features + logistic regression / gradient boosting.** 35 features per row, computed
for A and B separately, then stacked as `[A, B, A−B, shared]` (model names and IDs are never
features):

| Group | Features |
|---|---|
| Length / shape | log1p chars, words, lines, list items, markdown headers, bold spans, code blocks, `?`, `!`, URLs |
| Content | refusal-phrase hits (capped at 3), empty flag, average word length, type–token ratio |
| Prompt relation | Jaccard overlap with the prompt, prompt coverage |
| Shared | log1p prompt chars, log1p prompt words, A/B Jaccard |

Counts are `log1p`-transformed so a difference becomes a log-ratio. The LR variant is
`StandardScaler` + multinomial logistic regression; the boosting variant is
`HistGradientBoostingClassifier` (learning rate 0.05, 200 iterations, 15 leaves, min 50 samples
per leaf, L2 = 1.0).

**(c) TF-IDF + logistic regression.** Word 1–2-grams, `min_df = 5`, 50,000 features, sublinear
TF, responses truncated at 5,000 characters, vocabulary fitted on **training responses only**.
The design matrix is `[tfidf(A), tfidf(B), |tfidf(A) − tfidf(B)|]`.

**Swap augmentation and swap-averaging.** Both are implemented for every baseline and reported as
a separate variant. Training on `(A,B)` and `(B,A)` with labels `0↔1` exchanged, then averaging
the two orderings' probabilities at inference, is a symmetry prior: it removes the model's ability
to use response *position* as a feature. The `plain` and `swapaug` variants are therefore also the
cleanest available measurement of position bias (§4.2).

Each model × variant × seed writes its own metrics shard and its validation probabilities as
`results/predictions/<model>__<variant>__seed<k>__validation.npz`, so the calibration study and
the error analysis pick them up automatically and never recompute anything.

---

## 3. Calibration study

For every validation prediction file: raw metrics, cross-fitted temperature-scaled metrics, both
fold temperatures, the full-validation temperature, and the log-loss gain and ECE change.
Reliability diagrams show accuracy against mean confidence per bin with the identity line, plus a
confidence histogram, before and after scaling.

Interpretation rules applied in the report:

- Temperature scaling is a **positive scalar**, so it cannot change any row's argmax. Accuracy and
  macro F1 are therefore identical before and after by construction, and any difference would be a
  bug. Only log loss, ECE and mean confidence can move.
- A calibrated row is a **separate row in the results table**, never merged into the raw row. The
  raw log loss is the model's honest out-of-the-box score; the calibrated figure is a second model
  that used validation labels.
- The prior baseline's near-zero ECE (1.0 × 10⁻⁵) is not evidence of good calibration in any useful
  sense: its probabilities are nearly constant across rows and happen to match the class
  proportions. ECE alone cannot distinguish "well calibrated from ignorance" from "well calibrated
  and informative", which is why accuracy, macro F1 and the class-wise ECE are reported together.

---

## 4. Extension E1 — bias and shortcut analysis

### 4.1 Length shortcut
Three measurements, all on validation:

1. A **deterministic rule**: the longer response wins, and a |log length ratio| ≤ 0.02 counts as a
   tie. Its accuracy is directly comparable to a model's.
2. A **single-feature logistic regression** on `log1p(len A) − log1p(len B)`, which gives a
   probabilistic log-loss floor attributable to length alone.
3. **Log loss and accuracy by quartile of |log length ratio|** for every model. If a model's
   advantage disappears in quartile 1 (similar-length pairs), it is mostly reading length; if it
   holds up there, it is reading content.

### 4.2 Position bias
`position_bias` reports, per model: the share of A-wins predictions against the true share
(34.91 %), accuracy broken down by true label, and the **swap flip rate** — the share of validation
pairs whose judged winner changes when A and B are exchanged, computed by evaluating the same
fitted model on the swapped feature view and mapping the result back. A position-unbiased model
has a flip rate near zero; a model that simply prefers A has a flip rate near the share of pairs
where the two responses differ enough to matter. Swap-averaged log loss is reported as well, since
swap-averaging is itself a cheap debiasing technique.

### 4.3 Author / model-style predictiveness
`model_a` and `model_b` are present in the Kaggle training data and absent from the hidden test
set, so they are never model features. They are used **only for analysis**: a one-hot
`(model_a, model_b)` logistic regression plus a same-author flag, fitted on train and scored on
validation, measures how much of the label is explained by *who wrote the answer* rather than by
answer quality. A low log loss here is a warning, not usable signal — it means part of what a text
model learns may be author style. Per-author win rates with training counts are saved to
`author_win_rates.csv`.

### 4.4 Feature attribution
Standardised logistic-regression coefficients for the numeric baseline, sorted by maximum absolute
coefficient across the three classes, so the report can name the features the simplest model
actually leans on instead of guessing.

---

## 5. Extension E3 — data efficiency and label noise

**Learning curves.** The numeric and TF-IDF baselines are trained on a random 10 / 25 / 50 / 100 %
of **training prompt groups** (whole groups, so no prompt is split), at three seeds each, and
scored on the fixed validation partition. Sampling whole groups rather than rows matters: sampling
rows would let one prompt's duplicates appear on both sides of the fit.

**Label-noise audit.** Two independent signals:
1. *Duplicate conflicts.* Identical `(prompt, response pair)` rows, treating A/B-swapped copies as
   the same pair with the label swapped, are counted per group; groups with `nunique(label) > 1`
   are directly contradictory labels.
2. *Out-of-fold support.* A 5-fold prompt-grouped CV inside train produces, for each training row,
   the out-of-fold probability the model assigns to that row's **given** label. Low values flag
   rows whose label the model (and its features) cannot support. These are candidates for manual
   review, not proven errors.

**Retraining on cleaned data.** The lowest-support 0 / 5 / 10 / 20 % of training labels are dropped
**within each class**, so the tie class is not deleted wholesale, and the model is retrained and
rescored on the unchanged validation labels. If validation log loss improves, the flagged rows
were behaving like label noise; if it degrades, the flags are mostly hard-but-correct examples and
should be kept. Either outcome is reportable — the point is that the claim is tested, not assumed.

---

## 6. Error analysis

Per model: row-normalised confusion matrix (counts and percentages), per-class recall, the
predicted-class distribution conditional on a true tie and vice versa, accuracy split by true label
(the A/B asymmetry check), ten equal-mass confidence deciles with accuracy and log loss in each,
and the 100 highest-confidence errors with IDs, label pair, prompt/response heads and length gap.

Slices, each reported with `n`, log loss, accuracy, macro F1 and tie recall: absolute log length
ratio > 0.5, A longer, code block in A / in B, markdown list in A / in B, markdown header in A / in
B, refusal phrase in A / in B, empty A / empty B, very long (> 4,000 characters) A / B, and
above-median prompt length.

---

## 7. Threats to validity

- **Response-pair overlap.** The audit found 27 unordered response-pair groups spanning partitions
  (196 rows). Those overlaps remain in the split; their effect is bounded and reported rather than
  removed, because changing the split invalidates every completed run.
- **Validation reuse.** Validation is used for reporting, for cross-fitted calibration and for the
  E3 cleaning comparison. Selection no longer uses it, but it is not a pristine holdout, and the
  final claim must come from the single late Kaggle submission.
- **Truncation.** TF-IDF reads only the first 5,000 characters of each response; the numeric
  features read the full text. A TF-IDF/embedding comparison therefore differs in input coverage
  as well as in representation.
- **No seed variance for the frozen-embedding run.** It reports one seed; the baselines report
  three. Differences smaller than the seed spread should not be interpreted.
- **Author identity is unmeasurable at test time**, so the author-only result is diagnostic only.
- **No user/session/document identifiers** exist in the CSV, so underlying-source separation cannot
  be verified.

---

## 8. Scripts

```bash
python tests/smoke_avani.py                      # synthetic end-to-end check, no real data needed
python scripts/run_all_avani.py                  # every stage, resumable
python scripts/run_all_avani.py --stage-list     # show which stages are done
python scripts/run_baselines.py --models tfidf_lr --variants plain swapaug --seeds 42 43 44
python scripts/run_calibration.py                # all prediction files, cross-fitted temperature
python scripts/run_bias_analysis.py              # E1
python scripts/run_data_efficiency.py            # E3
python scripts/run_error_analysis.py             # confusion, slices, confidence, high-conf errors
python scripts/summarize_results.py              # reports/results_table.md
```

All scripts accept `--data`, `--splits`, `--results` and write JSON/CSV/PNG rather than printing
numbers that exist only in a terminal. None of them scores the local test partition.
