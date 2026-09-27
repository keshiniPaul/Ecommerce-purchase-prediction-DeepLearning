# Ecommerce-purchase-prediction-DeepLearning
SE4050 Deep Learning project: Comparing four supervised deep-learning architectures for online purchase prediction using the RetailRocket e-commerce dataset.


File Structure
```text
Ecommerce-purchase-prediction-DeepLearning/
│
├── README.md                          ← this file
├── requirements.txt                   ← pinned Python dependencies
├── .gitignore                         ← excludes data/raw/, large checkpoints, etc.
│
├── data/
│   ├── README.md                      ← how to download raw data
│   ├── raw/                           ← LOCAL ONLY (not committed)
│   │   └── events.csv                 ← RetailRocket events (from Kaggle)
│   └── processed/                     ← shared frozen outputs
│       ├── model_events.parquet       ← (or .csv) view/addtocart before purchase
│       ├── labels.csv                 ← session_id, target (0/1)
│       ├── split_ids.json             ← chronological train/val/test session IDs
│       ├── session_features.csv       ← optional aggregates for MLP
│       └── README.md                  ← how processed files were built
│
├── common/                            ← SHARED package (import from all models)
│   ├── __init__.py
│   ├── config.py                      ← SEED, 30-min gap, paths, split ratios
│   ├── cleaning.py                    ← load events, drop duplicates, datetime, sort
│   ├── sessionization.py              ← 30-minute inactivity → session_id
│   ├── labeling.py                    ← target + leakage-free model inputs
│   ├── split.py                       ← chronological 70/15/15 → split_ids.json
│   ├── utils.py                       ← set_seed, Timer, count_params
│   ├── metrics.py                     ← accuracy, precision, recall, F1, ROC-AUC, CM
│   └── evaluation.py                  ← learning curves, ROC, CM, comparison table
│
├── notebooks/
│   ├── 01_common_pipeline.ipynb       ← run once → writes data/processed/
│   ├── 03_evaluation_demo.ipynb       ← demo of common.metrics / plots
│   ├── Transformer_RetailRocket_Keshini.ipynb
│   ├── MLP_....ipynb                  ← Anushika
│   ├── CNN_....ipynb                  ← Hansani
│   └── LSTM_....ipynb                 ← Sheran
│
├── models/                            ← optional modular .py code per model
│   ├── transformer/                   ← Keshini
│   ├── mlp/                           ← Anushika
│   ├── cnn/                           ← Hansani
│   └── lstm/                          ← Sheran
│       ├── sequence_dataset.py
│       ├── lstm_model.py
│       ├── train_lstm.py
│       ├── hyperparameters.yaml
│       ├── checkpoints/
│       └── outputs/
│
├── src/                               ← optional (e.g. Keshini’s transformer helpers)
│   └── transformer/
│
└── results/
    ├── figures/                       ← EDA / report figures
    ├── metrics/                       ← dataset summaries, per-model CSVs
    └── final_comparison.csv           ← all four models side by side
```


## SE4050 - Deep Learning Group Project

This project develops and compares four deep learning architectures for predicting whether an e-commerce browsing session results in a purchase.

The project uses the RetailRocket e-commerce dataset and treats the problem as supervised binary classification.

## Problem Definition

The objective is to predict whether an e-commerce browsing session will result in a purchase.

Target:

- `0` - Session does not result in a purchase
- `1` - Session results in a purchase

The prediction unit is a browsing session.

## Dataset

The project uses the RetailRocket e-commerce dataset.

The main dataset used for modelling is:

`events.csv`

Important columns:

- `timestamp`
- `visitorid`
- `event`
- `itemid`
- `transactionid`

The raw dataset is not stored in this GitHub repository.

See `data/README.md` for dataset setup instructions.

## Deep Learning Models

| Model | Member |
|---|---|
| Transformer Encoder | Keshini |
| Deep MLP | Anushika |
| 1D CNN | Hansani |
| LSTM | Sheran |

## Common Methodology

All models use the same:

- Dataset
- Data cleaning rules
- 30-minute inactivity session definition
- Session labels
- Leakage prevention rules
- Chronological train/validation/test split
- Final test set
- Evaluation metrics

Dataset split:

- 70% Training
- 15% Validation
- 15% Testing

Random seed:

`42`

## Leakage Prevention

Transaction events are used to construct the target but are never provided as model inputs.

For purchase sessions, only behavioural events occurring before the first transaction are used as model inputs.

## Evaluation Metrics

All models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Additional comparisons include:

- Training time
- Inference time
- Number of trainable parameters
- Training/validation performance

## Project Structure

- `common/` - Shared preprocessing, splitting, utilities and evaluation code.
- `data/` - Dataset instructions and processed data.
- `notebooks/` - Common and model-specific notebooks.
- `models/` - Model-specific implementation files.
- `results/` - Final evaluation results and figures.

## Development Environment

The project can be developed using:

- Visual Studio Code
- Google Colab

VS Code is used for Git/repository management and shared code development.

Google Colab can be used for preprocessing and model training with GPU support.

## Reproducibility

All models must use the shared preprocessing pipeline, dataset split and evaluation methodology.

Common random seed:

`42`

Any preprocessing operation that learns parameters from the data must be fitted using training data only.

The test set must remain unseen until final evaluation.