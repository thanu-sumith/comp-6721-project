# COMP 6721 — Lab 04 course-project report

**Project:** NLP-3: LLM / chatbot response-preference classification

**Team:** Varshain Gopichandar Sreedevi; Thanugundla Sumith Reddy; Avani Dipakbhai Gajjar

**Report date:** October 21, 2026

**Assignment reference:** Lab 04_New, slide 18, “Course-project activity (45 minutes),” supplied photograph.

This report answers the six questions on the slide using the current project code, local dataset, saved split assignments, and a freshly recomputed validation baseline. The experiment plan is in [project_plan.md](project_plan.md). Earlier draft documents retain some pending-status wording; the measured findings below are supported by [the report evidence](../results/metrics/lab04_report_evidence.json).

## 1. Is the target clearly defined?

Yes. Given a user prompt and two chatbot responses, A and B, predict the human preference among three mutually exclusive classes:

| Label | Meaning | Dataset target column | Examples | Percentage |
|---|---|---|---:|---:|
| 0 | A wins | `winner_model_a` | 20,064 | 34.91% |
| 1 | B wins | `winner_model_b` | 19,652 | 34.19% |
| 2 | Tie | `winner_tie` | 17,761 | 30.90% |

The local training CSV contains **57,477 rows and 9 original columns**. Parsing text and deriving labels/group identifiers produces 14 columns; these are not 14 raw input fields. Each row has exactly one winner flag. The loader validates the labels, unique IDs, and JSON turn structure; all rows parsed successfully and the raw CSV has no missing cells. JSON null turns are allowed by the loader and become empty text, so this missing-cell check does not establish that every response contains text.

The desired output is a probability vector `[P(A wins), P(B wins), P(tie)]` summing to one. The primary metric is multiclass log loss; accuracy and macro F1 are secondary metrics. The existing class-prior baseline uses only training labels; later text models will use the prompt and response text. Model names and row IDs are excluded from predictive features.

## 2. Can source, patient or object identifiers leak across partitions?

Yes, related examples can leak across a naive row-level split. Here the relevant object is a prompt or response pair, rather than a patient. There are **5,743 repeated prompt rows after the first occurrence** and **71 duplicate joined prompt/response-A/response-B triplets after the first occurrence**. The latter check ignores labels and model names and should not be interpreted as 71 identical complete CSV rows.

The current split groups by the SHA-256 hash of the parsed prompt-turn list. Verification found **zero prompt groups spanning partitions**. The row ID is used to recover assignments, not as an input feature. Model identities may repeat across partitions; excluding their columns prevents direct identity features, but does not remove textual style or shared-source effects.

An additional audit found **27 unordered response-pair groups spanning partitions, involving 196 rows**. This comparison uses joined parsed response text and treats A/B-swapped pairs as equivalent. Repeated pairs can include generic replies or empty text; this result establishes textual overlap, not that every overlap causes inflated performance. It does mean the current split cannot be described as fully free of response overlap.

Before comparing text-based classifiers, inspect these overlapping pairs and decide whether to remove duplicates or group connected prompts and response pairs into a newly versioned split. Every method must then be rerun on that same version. Preserve the existing assignments rather than silently replacing them. The CSV does not provide user/session/source-document identifiers, so separation by those underlying sources remains unverified. The current prior baseline does not consume response text, but later text models may exploit such overlap.

## 3. Is the positive class stated?

This project has **three classes and no single positive class**. The fixed label order is `0 = A wins`, `1 = B wins`, `2 = tie`; the word “positive” from a binary-classification exercise does not designate A as globally positive here.

If we later compute one-vs-rest precision, recall, F1, or ROC curves, the selected class is positive and the other two classes are negative for that calculation. Results must identify which class was selected. Macro F1 averages the three class-specific F1 scores equally.

## 4. Would majority-class accuracy overstate success?

Accuracy alone would give an incomplete assessment. The classes are fairly close in size, so this is not an extreme-imbalance dataset. However, always predicting A wins achieves **34.91% validation accuracy** without reading any prompt or response and never predicts B wins or tie.

The freshly recomputed class-prior baseline estimates probabilities only from the training partition: `[0.349088, 0.341897, 0.309015]`. Its hard prediction is therefore always A wins.

| Validation metric | Measured result |
|---|---:|
| Multiclass log loss | 1.097224 |
| Accuracy | 34.9078% |
| Macro F1 | 0.172502 |

A uniform three-class probability predictor has log loss `ln(3) ≈ 1.098612`, so the prior baseline improves log loss only slightly. The low macro F1 exposes the failure on B wins and tie. Future classifiers must improve on this measured reference, using the identical split, and include per-class results. No final local test model scores were calculated for this report.

## 5. Can other classifiers reuse precisely the same split?

Yes. [src/baseline.py](../src/baseline.py) provides `saved_split`, which reuses the existing ID-to-partition assignments in `data/splits/split_ids.csv` and checks the dataset fingerprint against `data/splits/manifest.json`. The manifest records seed 42, label order, split method, and scikit-learn version. Saving assignments is stronger than relying on the same random seed alone.

| Partition | A wins | B wins | Tie | Total |
|---|---:|---:|---:|---:|
| Train | 12,039 | 11,791 | 10,657 | 34,487 |
| Validation | 4,013 | 3,931 | 3,552 | 11,496 |
| Local test | 4,012 | 3,930 | 3,552 | 11,494 |

The nested `StratifiedGroupKFold` procedure creates approximately 60/20/20 partitions, prioritizing prompt grouping. Reuse these assignments for numerical-feature, TF-IDF, embedding, and fine-tuned model comparisons, subject to the overlap review in question 2. Fit vocabulary, feature transformations, class priors, and model parameters using training data only. Use validation data for model selection; reserve a separate validation subset or a nested procedure for calibration. Keep the local test set for final evaluation after model choices are fixed. Reporting partition composition here is not a test performance evaluation.

The raw dataset and split files are Git-ignored. Teammates must obtain authorized dataset access and share the exact split files privately; cloning the repository alone does not supply those assignments. The current dataset SHA-256 is `0692154aaf20fc6649090c3f49b6b5dd1e693a765ecb5f0856ae0da5946f5be2`.

## 6. Where will figures and intermediate results be stored?

| Artifact | Project location | Current status |
|---|---|---|
| Raw training CSV | `data/raw/train.csv` | Available locally; excluded from Git |
| Fixed row assignments and fingerprint | `data/splits/split_ids.csv`, `data/splits/manifest.json` | Available locally; excluded from Git |
| Exploration notebook | `notebooks/01_data_exploration.ipynb` | Local execution outputs present |
| Class-distribution chart | `results/figures/class_distribution.png` | Generated locally |
| Text-length chart | `results/figures/text_lengths.png` | Generated locally |
| Original validation baseline output | `results/metrics/class_prior_validation.json` | Available locally; matches recomputation |
| Original environment record | `results/metrics/versions.json` | Available locally |
| Report audit and recomputed baseline | `results/metrics/lab04_report_evidence.json` | Included with this report |
| Audit reproduction script | `scripts/lab04_report_evidence.py` | Included with this report |
| Plan and reports | `reports/` | Plan, earlier draft, and this report |

Use `results/figures/` for future confusion matrices, calibration plots, and comparison charts, and `results/metrics/` for scores, experiment settings, and environment versions. Use distinct filenames for each model/split version. Future serialized models need an explicitly managed artifact location because `.joblib` and `.pkl` files are currently ignored.

## Evidence and next experiment

The report audit reran the current loader and `saved_split` checks and reproduced all three stored baseline metrics exactly. It also verified counts, missing cells, duplicates, partition composition, prompt separation, and response-pair overlap. The recorded environment is Python 3.11.4, NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, and Matplotlib 3.11.2.

To reproduce the report evidence from the project root after obtaining the same dataset and saved split files:

```bash
source .venv/bin/activate
python scripts/lab04_report_evidence.py
```

The script requires existing split files, writes the report evidence JSON, and does not fit or score a test model. The immediate next step is to review the cross-partition response pairs, agree on a split policy, then train numerical-feature and TF-IDF logistic-regression baselines on the agreed assignments. Embeddings, DeBERTa, calibration comparisons, and final test results remain future work.
