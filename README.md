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