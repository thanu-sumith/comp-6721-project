# First-draft report — NLP-3

## Problem and scope

Predict a human judge's preference between responses A and B to the same prompt, with a third label for a tie. Return three probabilities. Follow the supplied revised proposal; this draft records planned experiments, not completed findings.

## Current evidence

The repository contains setup instructions, a data-exploration notebook, grouped-split and class-prior starter code, and the experiment plan. No project dataset or empirical model results were available when this draft was prepared. Software checks on synthetic fixtures are not dataset results.

## Data audit — pending execution

| Item | Result |
|---|---|
| Training CSV rows and columns | Pending |
| A wins / B wins / tie counts | Pending |
| Missing fields and invalid JSON | Pending |
| Repeated prompts and duplicate examples | Pending |
| Prompt and response length distributions | Pending |
| Additional source/response leakage audit | Pending |

## Evaluation design

Group by parsed prompt and save fixed row IDs using seed 42. Check that prompts do not overlap and all classes exist in each partition. Model names and IDs are not model features. Training data determines model parameters and prior probabilities. Validation supports development. Local test data remains untouched; later calibration needs a designated validation subset or nested procedure.

## Experiment table

| Method | Validation log loss | Accuracy | Macro F1 | Status |
|---|---|---|---|---|
| Class-prior probabilities | — | — | — | Ready to run after data access |
| Numerical features + classifier | — | — | — | Planned |
| TF-IDF + logistic regression | — | — | — | Planned |
| Frozen embeddings + classifier | — | — | — | Planned |
| DeBERTa | — | — | — | Planned |

Lower log loss is better. A uniform three-class predictor has log loss ln(3), approximately 1.099; a learned class-prior predictor need not have that value. Report measured baselines. Accuracy can obscure minority-class failures, so include per-class analysis and macro F1.

## Planned analysis and deliverables

Save class and length charts, baseline validation metrics, package versions, and split fingerprints. Later examine A/B position bias, response-length effects, calibration, and high-confidence errors. Compare models on identical partitions and report variation across training seeds. Make a final local test evaluation only after model choices are fixed.

## Next work

Obtain data; run the exploration notebook; review leakage checks; lock and share split IDs privately; record validation baseline results; implement feature and TF-IDF baselines; replace pending entries with actual evidence. Dataset access and compute availability remain unverified.
