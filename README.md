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
