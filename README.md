# COMP 6721 Project — NLP-3

Predict human preferences between two chatbot responses: **A wins**, **B wins**, or **tie**. Inputs are the prompt and both responses. The primary metric is multiclass log loss; secondary metrics are accuracy and macro F1.

## Start here

Run these commands from this repository folder. Use a separate virtual environment for this project (Python 3.11+ recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Download `train.csv` yourself from the [LLM Classification Finetuning competition](https://www.kaggle.com/competitions/llm-classification-finetuning/data), after accepting its access terms, and put it in `data/raw/train.csv`. Do not commit the dataset or Kaggle credentials.

Open `notebooks/01_data_exploration.ipynb` and run its cells in order. It inspects the data, saves fixed prompt-grouped split IDs, and evaluates a class-prior baseline on validation data. It stops with a helpful message if the dataset is missing. Data exploration and the class-prior validation baseline have been completed. See [the first-draft report](reports/first_draft_report.md) for measured findings and the remaining response-overlap review.

## Files

- `notebooks/01_data_exploration.ipynb`: exploration, split creation, validation baseline.
- `src/baseline.py`: reusable parsing, checks, grouped split, and baseline code.
- `data/raw/`: local downloaded data; ignored by Git.
- `data/splits/`: fixed row-ID assignments and data fingerprint; ignored by Git. Share these files privately with teammates so every model uses exactly the same split.
- `results/figures/`: saved charts.
- `results/metrics/`: validation scores and package versions.
- `reports/project_plan.md`: concise Task 8 experiment plan.
- `reports/first_draft_report.md`: initial report with pending results clearly marked.

Alternatively, run the starter from the terminal:

```bash
python -m src.baseline
```

## Baselines, calibration and data-efficiency (Avani's tasks)

Run from the repo root after `data/raw/train.csv` and the team's shared `data/splits/` files are in place:

```bash
python tests/smoke_test.py                 # optional: checks the code runs (synthetic data only)
python scripts/run_baselines.py            # prior, numeric LR / gradient boosting, TF-IDF LR (validation only)
python scripts/run_calibration.py          # temperature scaling, ECE, reliability diagrams
python scripts/run_data_efficiency.py      # E3: learning curves + label-noise audit
```

Code lives in `src/metrics.py`, `src/features.py`, `src/models.py`, `src/calibration.py`, `src/experiment.py`. Every model saves validation probabilities to `results/predictions/<model>__<variant>__seed<k>__validation.npz`; any later model (embeddings, DeBERTa, LoRA) that saves the same format is picked up by `run_calibration.py` automatically. None of these scripts score the local test partition.

## Frozen embedding experiment (Sumith)

The first real-data run is complete. Validation accuracy is **43.0672%** versus
34.9078% for the class prior. Raw log loss is 1.106242 (worse than the prior's
1.097224); the team's cross-fitted calibration improves it to **1.063222**.
See [the experiment report](reports/embedding_experiment.md) for results, checks,
reproduction and the unresolved v1 response-overlap limitation. DeBERTa is not
implemented by this experiment; final local test performance remains unmeasured.

The first embedding model uses the pinned `all-MiniLM-L6-v2` encoder with a
training-only `StandardScaler` and multinomial logistic regression. Encoder
weights stay frozen. It requires the team's existing split files; it will not
create replacement assignments. Only train and validation text is encoded.

```bash
python -m pip install -r requirements-embeddings.txt
python -m unittest discover -s tests -p test_embeddings.py -v
python scripts/run_embeddings.py
```

`--device mps` selects Apple GPU acceleration; `--device cpu` runs on CPU.
The encoder's default revision is pinned in `src/embeddings.py`. Model downloads
and resumable document embeddings are cached under the ignored `data/cache/`.
`--local-files-only` supports a pre-populated local model cache.

Long text is split into consecutive chunks of at most 254 content tokens plus
special tokens. Normalized chunk embeddings are averaged by content-token count,
then normalized again. No content tokens are deliberately truncated. Empty text
maps to a zero vector. Features combine prompt/A/B embeddings, absolute A/B
differences, prompt-answer products, and three cosine similarities. With the
384-dimensional encoder this gives 2,307 classifier features. This pooling loses
cross-chunk interactions and explicit conversation-turn alignment.

The initial classifier uses fixed C=1 and seed 42; there is no validation grid
search. Optional `--swap-aug` trains both orderings and averages swapped
probabilities. Any scaler is fitted on training rows only. Reports include
dataset and split hashes, model revision, environment versions, token coverage,
class-prior comparison, and a prompt-group bootstrap log-loss interval. The
interval does not measure training-seed variability. Current v1 response-pair
overlap remains a limitation. The local test partition is not encoded or scored.

Outputs:

- `results/metrics/embeddings_minilm_chunkmean_validation.json`: measured scores and provenance.
- `results/predictions/embeddings__minilm_chunkmean__seed42__validation.npz`: complete aligned validation `ids`, `probs`, `y` in A/B/tie order.
- `results/models/embeddings__minilm_chunkmean__seed42.joblib`: classifier and encoder configuration (local only).
- `results/figures/embeddings_minilm_chunkmean_reliability.png`: validation reliability plot.

Prediction files are Git-ignored because their IDs reveal validation membership.
Share the prediction file privately with Avani, then run
`python scripts/run_calibration.py --pattern 'embeddings__*__validation.npz'`.
Each report lists whether log loss improved; improvement is not assumed.
Use a fresh `--results` directory to preserve a previous experiment rather than
overwriting its predictions.

## Evaluation rules

Label order: `0 = A wins`, `1 = B wins`, `2 = tie`. There is no single positive class in this multiclass task. Group by parsed prompt; never let a prompt cross partitions. Fit features, class priors, and models on training data only. Tune and calibrate using validation data (use a separate calibration subset or nested procedure for later calibration comparisons). Reserve the local test partition for final evaluation. Hidden Kaggle labels cannot support local metrics.

Use seed 42 for the initial saved split; keep that split fixed for model comparisons. Later repeat training with several seeds and report variability. Never overwrite split files silently. Prompt grouping addresses repeated prompts; additional source or duplicate-response leakage must still be audited before final reporting.

## Next milestones

1. Review the completed data audit, including empty responses and cross-partition response pairs.
2. Agree on the split policy and privately share the exact saved assignments.
3. Preserve the measured class-prior validation baseline; rerun it if the split changes.
4. Add numerical-feature and TF-IDF logistic-regression baselines using the same IDs.
5. Study embeddings, fine-tuning, calibration, and bias according to the proposal and available compute.

## Team and source

Varshain Gopichandar Sreedevi; Thanugundla Sumith Reddy; Avani Dipakbhai Gajjar.

Scope and responsibilities follow the supplied Week 3 revised NLP-3 proposal. This repository does not establish instructor approval or completed experiments. The dataset is not redistributed.
