# Results and discussion — baselines, calibration, bias, data efficiency and error analysis

Avani Dipakbhai Gajjar (40345876). All numbers are on the fixed validation partition
(11,496 rows) of the team's frozen v1 split; the local test partition was never scored.
Reported values come from `results/metrics/*.json`, produced by the scripts in `scripts/`.

---

## 1. Headline

A **35-feature gradient-boosted model reaches a validation log loss of 1.015843** (accuracy
48.48 %, macro F1 0.4750), a **7.42 % reduction** against the class-prior floor of 1.097224.

Three findings matter more than the raw score:

1. **The cheap feature baseline outperforms the frozen-embedding model.** MiniLM + logistic
   regression reports 1.106242 log loss and 43.07 % accuracy. The numeric baseline is better by
   **0.0904 log loss and 5.41 percentage points of accuracy**, and the two 95 % prompt-group
   bootstrap intervals do not overlap (`[1.0087, 1.0222]` versus `[1.0946, 1.1161]`), so the
   difference is not seed or sampling noise.
2. **One feature beats it.** A logistic regression on `log length(A) − log length(B)` alone scores
   **1.077582** — better than the embedding model on the primary metric. Length is not a
   curiosity here; it is a dominant part of the signal.
3. **Author identity is highly predictive and unusable.** A one-hot `(model_a, model_b)` model
   scores **1.050225** (accuracy 44.46 %) without reading a single token of the responses.

Implication for the project plan: the embedding pipeline needs revisiting before GPU time is
spent on DeBERTa. A frozen encoder whose linear probe loses to a length feature is not a
representation-quality result, it is a pipeline result — the 2,307-dimensional feature block is
almost certainly mis-scaled for a single fixed `C = 1`, which is consistent with its log loss
landing *above* the class prior.

---

## 2. Experiment table

| Model | Variant | Selection | Log loss (primary) | Δ vs prior | Accuracy | Macro F1 | Log loss, temp-scaled | ECE raw → cal |
|---|---|---|---:|---:|---:|---:|---:|---:|
| numeric_hgb | swapaug | none (no C) | **1.01584 ± 0.00018** | **−0.08138** | **48.48 %** | **0.4750** | 1.01534 | 0.0154 → 0.0091 |
| numeric_hgb | plain | none (no C) | 1.01913 ± 0.00011 | −0.07809 | 48.29 % | 0.4729 | 1.01871 | 0.0142 → 0.0096 |
| numeric_lr | swapaug | train-CV (3-fold) | 1.03510 | −0.06212 | 47.56 % | 0.4596 | 1.03507 | 0.0202 → 0.0141 |
| numeric_lr | plain | train-CV (3-fold) | 1.03518 | −0.06204 | 47.43 % | 0.4580 | 1.03524 | 0.0193 → 0.0133 |
| tfidf_lr | swapaug | train-CV (3-fold) | 1.04690 | −0.05032 | 46.62 % | 0.4621 | 1.04291 | 0.0193 → 0.0144 |
| *author identity only* (E1) | — | none | 1.05022 | −0.04700 | 44.46 % | 0.3965 | — | — |
| tfidf_lr | plain | train-CV (3-fold) | 1.05483 | −0.04239 | 44.17 % | 0.4370 | 1.05212 | 0.0307 → 0.0132 |
| *length only, 1 feature* (E1) | — | none | 1.07758 | −0.01964 | 42.61 % | 0.3358 | — | — |
| MiniLM + LR (Sumith) | minilm_chunkmean | **none (fixed C = 1)** | 1.10624 | +0.00902 | 43.07 % | 0.4285 | 1.06322 | 0.1023 → 0.0104 |
| class prior | — | none | 1.09722 | 0.00000 | 34.91 % | 0.1725 | — | — |
| *length rule, longer wins* (E1) | — | none | 1.56048 | +0.46326 | 43.42 % | — | — | — |

Rows in italics are diagnostic constructs, not candidates for the final model. The length rule is
a hard classifier, so its log loss is not comparable to a probabilistic model; its **accuracy** is
the meaningful number.

### 2.1 Selection protocol (why this table is internally comparable)

The first version of `run_baselines.py` selected `C` **on the validation partition** and then
reported validation scores for that same partition. That is optimistically biased, and it would
have placed these baselines in a privileged position relative to the embedding run, which used a
fixed `C = 1`. Selection now runs a prompt-grouped 3-fold cross-validation **inside train**
(`src/selection.py`), and the protocol is recorded in every result file.

Selected values and how flat the selection surface is:

| Model | CV log loss at C = 0.01 | C = 0.1 | C = 1.0 | Chosen |
|---|---:|---:|---:|---|
| numeric_lr | 1.046376 | 1.047029 | 1.047416 | 0.01 |

The spread across the whole grid is 0.001, so the choice of `C` is immaterial for this model —
worth stating explicitly, because it means the numeric-LR result is not an artefact of tuning.
`numeric_hgb` has no `C`, so no selection was needed.

### 2.2 Swap augmentation

Swap augmentation (train on both `(A,B)` and `(B,A)` with labels `0↔1`, then average the two
orderings at inference) helps every model, and helps the text model most:

| Model | Δ log loss from swapaug | Δ accuracy |
|---|---:|---:|
| numeric_hgb | −0.00329 | +0.19 pp |
| numeric_lr | −0.00008 | +0.13 pp |
| tfidf_lr | **−0.00793** | **+2.45 pp** |

The pattern is diagnostic: TF-IDF gains most because it is the model most exposed to position
(§4.2).

### 2.3 Seed variability

`numeric_lr` and `tfidf_lr` produce **identical** scores across seeds 42/43/44 — both are convex
problems, so the seed only affects fold shuffling, not the fitted model. Only `numeric_hgb` shows
real seed spread, and it is small: standard deviation **0.00018** on log loss and 0.03 pp on
accuracy across three seeds. Any difference below ~0.0005 log loss between the boosting variants
should not be interpreted.

---

## 3. Calibration

Temperature scaling was cross-fitted by prompt group inside validation (2 folds), so no row is
scored by a calibrator fitted on it. Because temperature is a positive scalar it cannot change any
argmax — accuracy and macro F1 are identical before and after by construction, and they are.

| Model | Log loss raw → calibrated | Gain | ECE raw → calibrated | Temperature |
|---|---|---:|---:|---:|
| numeric_hgb / swapaug | 1.015843 → 1.015342 | +0.00050 | 0.0154 → 0.0091 | 0.915 |
| numeric_hgb / plain | 1.019129 → 1.018708 | +0.00042 | 0.0142 → 0.0096 | 0.917 |
| numeric_lr / swapaug | 1.035104 → 1.035069 | +0.00004 | 0.0202 → 0.0141 | 0.939 |
| numeric_lr / plain | 1.035181 → 1.035241 | **−0.00006** | 0.0193 → 0.0133 | 0.939 |
| tfidf_lr / swapaug | 1.046904 → 1.042911 | **+0.00399** | 0.0193 → 0.0144 | 1.298 |
| tfidf_lr / plain | 1.054834 → 1.052122 | **+0.00271** | 0.0307 → 0.0132 | 1.261 |
| MiniLM + LR | 1.106242 → 1.063222 | +0.04302 | 0.1023 → 0.0104 | 2.209 |

Three observations:

1. **The direction of the temperature tells you which way each model is wrong.** The numeric
   models have `T < 1` (0.92–0.94): they are *under*-confident and benefit from sharpening. TF-IDF
   has `T > 1` (1.26–1.30): it is *over*-confident, and softening it recovers 0.004 log loss. The
   embedding model's `T = 2.209` is the extreme case, which is consistent with 2,307 unscaled
   features feeding a fixed `C = 1`.
2. **A calibrated model is a different row, never a merged one.** The raw column is the honest
   out-of-the-box score; the calibrated column used validation labels and must be reported as a
   separate model. This matters most for the embedding row, where the entire apparent win over the
   prior comes from calibration (1.106 → 1.063).
3. **One gain is negative, and it is reported as such.** `numeric_lr / plain` gets 0.00006 worse.
   ECE still improves. That is the expected behaviour of a two-fold cross-fit on a model that was
   already close to calibrated, and it is not hidden.

The prior baseline's near-zero ECE (1.0 × 10⁻⁵) is not evidence of good calibration: its
probabilities are nearly constant across rows and happen to equal the class proportions. ECE alone
cannot separate "well calibrated and ignorant" from "well calibrated and informative", which is
why it is always reported next to accuracy and macro F1.

---

## 4. Extension E1 — bias and shortcut analysis

### 4.1 Length shortcut

| Construct | Log loss | Accuracy |
|---|---:|---:|
| Class prior | 1.097224 | 34.91 % |
| Longer-wins rule (|log ratio| ≤ 0.02 → tie) | 1.560481 | 43.42 % |
| Single-feature LR on `log len(A) − log len(B)` | 1.077582 | 42.61 % |
| numeric_hgb / swapaug | 1.015843 | 48.48 % |
| MiniLM + LR | 1.106242 | 43.07 % |

- **The deterministic length rule alone reaches 43.42 % accuracy** — 8.51 pp above the prior, and
  *higher than the embedding model's 43.07 %*. Its log loss is poor because a hard rule has no
  meaningful probabilities; the accuracy is the number that counts.
- **One feature reaches 1.077582**, i.e. better than the class prior *and* better than the
  2,307-feature embedding model.
- Log loss by quartile of `|log length ratio|` (figure `bias_length_quartiles.png`) shows whether a
  model survives on similar-length pairs. This is the fair test of whether a model is reading
  content or length.

The numeric model's coefficients confirm the same story. The eight largest standardised
coefficients are all length or shape features:

| Feature | coef A wins | coef B wins | coef tie |
|---|---:|---:|---:|
| `B_log_chars` | +0.240 | +0.034 | −0.275 |
| `B_log_words` | −0.237 | +0.053 | +0.185 |
| `diff_log_words` | +0.229 | −0.175 | −0.054 |
| `diff_log_lines` | +0.149 | −0.152 | +0.004 |
| `A_log_chars` | +0.145 | +0.095 | −0.240 |
| `diff_log_chars` | −0.114 | +0.074 | +0.039 |

Longer responses push probability away from "tie" and toward whichever side is longer. The
remaining signal is prompt–response overlap (`response_jaccard`, `prompt_coverage`).

### 4.2 Position bias

Each model judges every validation pair twice — once as `(A, B)`, once as `(B, A)` — and the second
result is mapped back to the original A/B/tie coordinates.

| Model | Predicted A / B / tie | True share | Flip rate | Accuracy after swap |
|---|---|---:|---:|---:|
| numeric_lr | 0.434 / 0.389 / 0.178 | 0.349 / 0.342 / 0.309 | **0.0829** | — |
| tfidf_lr | 0.371 / 0.360 / 0.268 | 0.349 / 0.342 / 0.309 | **0.5287** | — |
| numeric_lr, swap-averaged | 0.414 / 0.407 / 0.178 | 0.349 / 0.342 / 0.309 | — | — |
| tfidf_lr, swap-averaged | 0.355 / 0.356 / 0.289 | 0.349 / 0.342 / 0.309 | — | — |

**TF-IDF's judged winner changes on 52.9 % of pairs when A and B are exchanged.** The feature
block `[tfidf(A), tfidf(B), |tfidf(A) − tfidf(B)|]` is symmetric in construction, yet the fitted
multinomial coefficients are not, so the model is nearly coin-flipping with respect to which
response is shown first. That explains both its swap-augmentation gain (§2.2) and why swap-averaging
it moves its predicted class shares from 0.371/0.360 to 0.355/0.356, close to the true 0.349/0.342.

The numeric model is far more stable (flip rate 0.083) because its features are already mostly
differences. Both models predict *tie* less often than the true 30.9 % share, and both are better at
recognising true A wins than true B wins (numeric_hgb: 56.29 % vs 54.41 %), a residual A-position
advantage that swap-augmentation reduces.

### 4.3 Author / model-style predictiveness

`model_a` and `model_b` exist in the Kaggle training data and never in the hidden test set, so they
are never model features. A one-hot `(model_a, model_b)` logistic regression with a same-author flag,
fitted on train and scored on validation:

| | Value |
|---|---|
| Log loss | **1.050225** |
| Accuracy | 44.46 % |
| Macro F1 | 0.3965 |

Knowing *who wrote the answers* — with no response text at all — beats the class prior by 0.047 and
beats the embedding model by 0.056 log loss. Two conclusions, both required by the proposal:

1. It is a **shortcut warning**: part of what any text model learns on this data may be author style
   rather than answer quality, and the hidden test set removes the identifier without removing the
   style.
2. It is **not usable signal** and was never added as a feature. Per-author win rates with training
   counts are saved to `author_win_rates.csv` for the report appendix.

---

## 5. Extension E3 — data efficiency and label noise

### 5.1 Learning curves

Training on 10 / 25 / 50 / 100 % of **training prompt groups** (whole groups, so no prompt is split
across the fit), three seeds, scored on the fixed validation partition:

| Train fraction | Rows | numeric_hgb log loss | tfidf_lr log loss |
|---|---:|---:|---:|
| 10 % | 3,449 | 1.05483 ± 0.00362 | 1.08111 ± 0.00064 |
| 25 % | 8,622 | 1.02208 ± 0.00174 | 1.08572 ± 0.00330 |
| 50 % | 17,244 | 1.01965 ± 0.00149 | 1.10229 ± 0.00177 |
| 100 % | 34,487 | **1.01614 ± 0.00045** | 1.11590 ± 0.00000 |

- **`numeric_hgb` saturates early.** Two thirds of the total gain is obtained by 25 % of the data
  (1.0548 → 1.0221), and 50 % is within 0.0035 of the full-data score. Practically: one GPU-free
  experiment at half the data would have answered the question to within noise, and for the DeBERTa
  work this curve is the argument for a data-subset pilot before a full fine-tune.
- **`tfidf_lr` gets worse with more data** (1.0811 → 1.1159). This is not a bug, it is a
  hyperparameter effect that must be disclosed: the learning-curve script trains at a fixed
  `C = 1.0` for every fraction, whereas the main table uses the CV-selected `C = 0.3`. At `C = 1.0`
  the model is over-flexible and more data lets it fit more spurious n-grams. The curve is
  therefore valid *as a curve at fixed C*, but its absolute level is not comparable to §2. It also
  predicts that the TF-IDF model would benefit from stronger regularisation — consistent with its
  calibration temperature of 1.26 showing over-confidence.

### 5.2 Label-noise audit

| Item | Count |
|---|---|
| Groups with an identical `(prompt, response pair)` — A/B-swapped copies treated as the same pair | 39 |
| Rows in those groups | 123 |
| Groups with **conflicting labels** | 6 (15.4 % of repeated groups) |
| Mean out-of-fold probability of the given label | 0.382 |

Only 6 repeated groups actually contradict themselves, i.e. at most a few dozen rows out of 57,477
(**~0.02 % of the dataset**) are provably mislabelled. The out-of-fold support score flags many more rows, but the retraining experiment
says those flags are not noise:

| Lowest-support labels dropped (within each class) | Rows kept | Validation log loss | Accuracy |
|---|---:|---:|---:|
| 0 % | 34,487 | **1.01652** | 48.36 % |
| 5 % | 32,762 | 1.03278 | 48.58 % |
| 10 % | 31,038 | 1.06813 | 48.35 % |
| 20 % | 27,590 | 1.17479 | 48.51 % |

Log loss degrades monotonically while accuracy stays flat or rises slightly. That combination means
the flagged rows are **hard-but-correct**, not mislabelled: removing them makes the model more
decisive (slightly higher accuracy) but destroys its probability estimates (much worse log loss).
**We therefore keep the original labels** and report the flagged IDs
(`noise_audit_flagged_ids.csv`) as candidates for manual review only, which is exactly what the
proposal's wording requires ("candidates for manual review, not proven errors").

---

## 6. Error analysis

| Model | Log loss | Accuracy | Macro F1 | Tie recall | Acc (true A) | Acc (true B) | Wrong rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| numeric_hgb | 1.01920 | 48.18 % | 0.4718 | 0.321 | 56.29 % | 54.41 % | 5,957 |
| numeric_lr | 1.03518 | 47.43 % | 0.4580 | 0.267 | 58.76 % | 54.59 % | 6,044 |
| tfidf_lr | 1.05483 | 44.17 % | 0.4370 | **0.344** | 50.69 % | 46.32 % | 6,418 |

- **Tie is the hard class.** Best tie recall is 34.4 % (TF-IDF); the boosting model manages 32.1 %,
  against a 30.9 % base rate — barely above chance for that class. Every model under-predicts tie
  (17.8–28.9 % of rows versus a true 30.9 %), which is the single largest source of lost macro F1.
- **Worst slices, consistently `B_has_code` and `A_has_code`** — pairs containing code blocks are
  errors for all three models, followed by markdown headers and very long responses. Code and long
  formatted answers are where the preference judgement is least explained by length and style.
- **A-position advantage persists** even after all corrections (numeric_hgb: 56.29 % on true A wins
  vs 54.41 % on true B wins), which is the residual form of the position bias measured in §4.2.
- High-confidence errors are extracted with IDs, label pair, confidence, length gap and text heads
  (`*__high_confidence_errors.csv`), giving the report concrete qualitative examples.

---

## 7. Threats to validity

1. **The frozen-embedding comparison is not like-for-like on hyperparameters.** MiniLM used a fixed
   `C = 1` with no search; the baselines used train-only CV. This asymmetry *favours* the baselines
   and is stated wherever the comparison is made. It does not explain a 0.09 log-loss gap, and a
   length-only feature still beats it.
2. **Response-pair overlap remains in the split.** The audit found 27 unordered response-pair groups
   spanning partitions (196 rows). It was not removed, because changing the split would invalidate
   every completed run. Its effect is bounded and reported, not assumed away.
3. **Validation is not pristine.** It is used for reporting, for cross-fitted calibration and for the
   cleaning comparison. Selection no longer touches it. The final claim must come from the single
   late Kaggle submission.
4. **The `tfidf_lr` learning curve runs at a different `C`** than the main table (§5.1) and must be
   read as a fixed-`C` curve.
5. **Truncation differs between models.** TF-IDF reads the first 5,000 characters; numeric features
   read the full text; the frozen encoder kept every token via chunk averaging. Part of the gap
   between the feature baseline and the encoder is input coverage, not representation quality.
6. **No seed variance exists for the linear models** (convex problems), so "mean ± std over seeds" is
   only meaningful for `numeric_hgb` and the embedding run, which reports a single seed.
7. **No user/session/document identifiers** exist in the CSV, so underlying-source separation cannot
   be verified.

---

## 8. Figure placement

| Figure | File | Where it goes |
|---|---|---|
| Learning curves | `learning_curves.png` | E3 section |
| Label cleaning curve | `label_cleaning.png` | E3 section |
| Length-quartile log loss | `bias_length_quartiles.png` | E1 §4.1 |
| Swap flip rate | `bias_position_flip.png` | E1 §4.2 |
| Confusion matrices | `confusion_*.png` (3) | Error analysis, §6 |
| Confidence profiles | `confidence_profile_*.png` (3) | Error analysis, §6 |
| Reliability before/after temperature | `calibration_*.png` (12) | Calibration, §3 — pick one per model |
| Uncalibrated reliability | `reliability_*.png` | Calibration, §3 |

---

## 9. Recommended next actions

1. **Revisit the embedding pipeline before DeBERTa.** Scale the 2,307 features properly (per-block
   standardisation, or `C` selected by train-only CV exactly as the baselines do) and re-run. If a
   length feature beats it, the conclusion is about the pipeline, not about MiniLM.
2. **Always use swap-augmentation + swap-averaging for text models.** It is free, it helps every
   model, and it removes most of the 52.9 % flip rate.
3. **Pilot DeBERTa on 25–50 % of the training groups** (8.6k–17.2k rows): the boosting curve says
   that is where nearly all the available signal is, and it costs a fraction of a full fine-tune.
4. **Report tie separately.** Tie recall just above its 30.9 % base rate is the project's weakest
   point and deserves its own paragraph rather than a footnote in macro F1.
5. Optionally re-run `run_data_efficiency.py` with the CV-selected `C` for TF-IDF, so the E3 curve
   sits on the same hyperparameter as the main table.
