# Course Project Experiment Plan — NLP-3

**Course:** COMP 6721 Applied Artificial Intelligence  
**Project:** LLM / Chatbot Response-Preference Classification  
**Team:** Varshain Gopichandar Sreedevi, Thanugundla Sumith Reddy, Avani Dipakbhai Gajjar

**Goal and target.** Given a prompt and responses A and B, predict probabilities for three mutually exclusive labels: A wins (0), B wins (1), and tie (2). This is multiclass classification with no single positive class.

**Data and preparation.** Use Kaggle's LLM Classification Finetuning data, described in the revised proposal as approximately 57,000 labeled examples. Parse multi-turn JSON text and inspect missingness, label counts, duplicate prompts, duplicate examples, and text lengths. Exclude model names and identifiers from predictive features. Exploration verified 57,477 labeled rows; measured counts and findings are recorded in the first-draft report.

**Leakage and reusable splits.** Use seed 42 and prompt-grouped, approximately label-stratified training/validation/test partitions. Save row IDs and a dataset fingerprint so every classifier reuses exactly the same split. Audit duplicate response pairs and source identifiers for additional overlap. Learn features and class priors on training data only. Use validation data for selection and a designated validation subset for calibration; keep the local test set untouched until final evaluation.

**Experiments and metrics.** Start with training-class-prior probabilities, then numerical-feature classifiers and TF-IDF with logistic regression. Later compare frozen embeddings and DeBERTa; attempt LoRA/QLoRA subject to compute. Primary metric: multiclass log loss (lower is better). Secondary metrics: accuracy and macro F1. Assess calibration with ECE and reliability diagrams. Report a majority baseline so class imbalance cannot be hidden by accuracy. Measure the class-prior baseline rather than assuming it equals ln(3). Repeat later training runs across fixed seeds.

**Immediate responsibilities.** The proposal assigns setup, parsing, exploration, and grouped splitting to Varshain; Week 5 baselines and evaluation to Sumith and Avani; Weeks 6–7 embeddings and DeBERTa to Sumith. Confirm completed work together before recording it as a result.

**Outputs and current status.** Store figures in `results/figures/`, scores and versions in `results/metrics/`, split IDs in `data/splits/`, and reports in `reports/`. Data exploration and the class-prior validation baseline are completed. The draft report records measured findings. Review empty responses and 27 response-pair groups crossing the current prompt-grouped partitions before locking the split policy for text-model comparisons. Long inputs, noisy labels, and GPU limits motivate starting with inexpensive methods. Review extensions at the Week 8 checkpoint.
