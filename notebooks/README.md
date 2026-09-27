# Project Notebooks

This directory contains the Jupyter and Google Colab notebooks used in the project.

## Common Notebooks

### 01_common_pipeline.ipynb

Shared preprocessing pipeline.

The notebook will:

1. Load the RetailRocket events dataset
2. Clean the dataset
3. Create browsing sessions using the 30-minute inactivity rule
4. Create purchase labels
5. Remove target leakage
6. Create the chronological train/validation/test split
7. Save the shared processed datasets

### 03_evaluation_demo.ipynb

Demonstrates the shared evaluation functions and visualizations.

## Model Notebooks

Each team member creates their model notebook on their own Git branch.

Expected model notebooks:

- `Transformer_RetailRocket_Keshini.ipynb`
- `MLP_RetailRocket_Anushika.ipynb`
- `CNN_RetailRocket_Hansani.ipynb`
- `LSTM_RetailRocket_Sheran.ipynb`

All model notebooks must use the common dataset preparation, split and evaluation methodology.