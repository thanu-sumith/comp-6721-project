# Frozen embedding experiment — November 12, 2026

The first frozen MiniLM + logistic-regression experiment is complete on the
team's existing v1 split. It improves validation accuracy and macro F1 over the
class-prior baseline, but its uncalibrated log loss is worse. Applying the team's
existing prompt-grouped cross-fitted temperature calibration improves log loss
below the prior baseline. The local test partition has not been encoded or scored.

## Method

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, pinned revision
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; all weights frozen.
- Inputs: joined parsed prompt, response A, and response B; no model names or IDs
  are classifier features.
- Long text: consecutive chunks of 254 content tokens plus CLS/SEP; normalized
  chunk vectors are averaged by content-token count and normalized again.
  Empty text maps to zero. No content tokens are deliberately dropped.
- Features: prompt/A/B vectors, absolute A/B difference, prompt-answer products,
  and three cosine similarities; 2,307 features in total.
- Classifier: training-only StandardScaler followed by multinomial logistic
  regression, C=1, LBFGS, maximum 2,000 iterations, seed 42. It converged in 209
  iterations. No hyperparameter search and no swap augmentation in this run.
- Training: 34,487 examples; validation: 11,496; final local test: 11,494.
- The saved split and dataset hashes match the original team artifacts.

## Measured validation results

| Metric | Class prior | Embeddings, raw | Embeddings, cross-fitted calibration |
|---|---:|---:|---:|
| Log loss (lower is better; primary) | 1.097224 | 1.106242 | **1.063222** |
| Accuracy | 34.9078% | **43.0672%** | **43.0672%** |
| Macro F1 | 0.172502 | **0.428520** | **0.428520** |
| Top-label ECE (15 bins) | 0.000010 | 0.102334 | 0.010378 |

The classifier predicted 4,951 of 11,496 validation labels correctly. Accuracy
improved by 8.1594 percentage points. Predictions include all three classes:
4,214 A wins, 4,037 B wins, and 3,245 ties. Raw per-class F1 is 0.454115 for A,
0.439508 for B, and 0.391938 for tie.

The baseline's tiny ECE reflects its nearly constant probabilities matching the
class proportions; it does not imply that it discriminates between examples.

The raw log-loss 95% prompt-group bootstrap interval is [1.094559, 1.116089].
This is conditional on the fitted model and fixed split, not an estimate of
training-seed variability or an interval for the calibrated model.

## Calibration and interpretation

The unchanged `scripts/run_calibration.py` fits temperature on one validation
group fold and evaluates the other, then reverses roles. Fold temperatures were
2.165732 and 2.253752. The full-validation temperature saved for a later final
evaluation is 2.208852; it is not the value used to score those same validation
rows in the cross-fitted result.

Calibration reduces overconfidence: mean top-class confidence changes from
0.533006 to 0.428174 while accuracy stays 0.430672. A positive scalar temperature
preserves each row's most likely class, explaining the unchanged accuracy and F1.
The calibrated validation result supports a useful first embedding experiment;
it does not establish final test performance.

## Coverage, timing, and checks

Train plus validation supplied 137,949 text fields, or 131,525 unique exact texts.
46,735 text fields needed multiple chunks; 54 were empty. The longest contained
15,672 content tokens and used 62 chunks. Chunk averaging compresses information
and loses explicit cross-chunk interactions despite retaining all content tokens.

Encoding and setup took about 11.1 minutes on the Mac GPU. The complete
uncalibrated run took about 11.5 minutes. Six focused software tests passed,
including exact token coverage, empty vectors, cache deduplication and resumption,
swap feature consistency, and prediction export validation. A separate real
encoder check processed a 2,000-token document in eight chunks.

Post-run verification checked all 11,496 prediction IDs and labels against the
saved validation assignments, finite normalized probabilities, reproduced metrics,
unchanged split-file hashes, a scaler fitted on exactly 34,487 training rows, and
matching predictions after reloading the saved classifier. The team's calibration
script successfully consumed the prediction file.

## Artifacts and reproduction

- `results/metrics/embeddings_minilm_chunkmean_validation.json`: raw metrics,
  dataset/split fingerprints, encoder revision, coverage and environment versions.
- `results/metrics/calibration_study.json`: cross-fitted calibration results.
- `results/metrics/embedding_verification.json`: verification checks and source hashes.
- `results/predictions/embeddings__minilm_chunkmean__seed42__validation.npz`:
  complete validation probabilities, aligned IDs and labels; share privately.
- `results/models/embeddings__minilm_chunkmean__seed42.joblib`: classifier and
  encoder configuration; local only.
- `results/figures/embeddings_minilm_chunkmean_reliability.png` and
  `results/figures/calibration_embeddings_minilm_chunkmean.png`: reliability plots.

```bash
python -m pip install -r requirements-embeddings.txt
python -m unittest discover -s tests -p test_embeddings.py -v
python scripts/run_embeddings.py --device mps --results results/reproduction
python scripts/run_calibration.py --results results/reproduction --pattern 'embeddings__*__validation.npz'
```

The exact original v1 split files and matching train.csv are prerequisites. Use
`--device cpu` on a machine without Apple GPU support. Model/cache directories
are local; a first run may need to download the pinned encoder. Existing prediction
files are not silently overwritten. Environment versions are in the metrics JSON;
a complete package freeze is retained locally at `data/cache/embedding_environment.txt`.

## Limits and next work

The prior audit found 27 response-pair groups spanning the current partitions,
involving 196 rows. This overlap remains unresolved and may affect text-model
comparisons. Model selection must preserve the agreed split policy. Inspect the
overlap, compare with real numerical/TF-IDF results, and consider training-only
grouped cross-validation for classifier regularization. This run uses an English
encoder and does not separately establish multilingual or code performance.
DeBERTa is still a separate future experiment. No final local test evaluation has
been performed.

The encoder design and input limits are described in its
[model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
