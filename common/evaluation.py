"""
Shared plotting and final model comparison utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)


def plot_learning_curves(
    history,
    model_name,
    save_dir
):
    """
    Plot and save training/validation loss and
    performance curves.

    Accepts either a Keras History object or dictionary.
    """

    Path(save_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    if hasattr(history, "history"):
        history = history.history

    # -----------------------------
    # Loss
    # -----------------------------

    plt.figure(figsize=(7, 5))

    if "loss" in history:
        plt.plot(
            history["loss"],
            label="Training Loss"
        )

    if "val_loss" in history:
        plt.plot(
            history["val_loss"],
            label="Validation Loss"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        Path(save_dir)
        / f"{model_name}_loss_curve.png",
        dpi=150
    )

    plt.close()

    # -----------------------------
    # Accuracy or AUC
    # -----------------------------

    metric_name = None

    if "accuracy" in history:
        metric_name = "accuracy"

    elif "auc" in history:
        metric_name = "auc"

    if metric_name is not None:

        plt.figure(figsize=(7, 5))

        plt.plot(
            history[metric_name],
            label=f"Training {metric_name.upper()}"
        )

        validation_key = f"val_{metric_name}"

        if validation_key in history:
            plt.plot(
                history[validation_key],
                label=f"Validation {metric_name.upper()}"
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric_name.upper())
        plt.title(
            f"{model_name} - {metric_name.upper()}"
        )
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            Path(save_dir)
            / f"{model_name}_{metric_name}_curve.png",
            dpi=150
        )

        plt.close()


def plot_roc_curve(
    y_true,
    y_prob,
    model_name,
    save_dir
):
    """
    Plot and save ROC curve.
    """

    Path(save_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        Path(save_dir)
        / f"{model_name}_roc_curve.png",
        dpi=150
    )

    plt.close()


def plot_confusion_matrix(
    cm,
    model_name,
    save_dir
):
    """
    Plot and save confusion matrix.
    """

    Path(save_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Purchase",
            "Purchase"
        ]
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        Path(save_dir)
        / f"{model_name}_confusion_matrix.png",
        dpi=150
    )

    plt.close()


def build_comparison_table(
    list_of_dataframes,
    output_path="results/final_comparison.csv"
):
    """
    Combine the metric rows from all four models into
    the final comparison table.
    """

    if not list_of_dataframes:
        raise ValueError(
            "No model metric DataFrames were provided."
        )

    comparison = pd.concat(
        list_of_dataframes,
        ignore_index=True
    )

    expected_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "params",
        "train_time_sec",
        "infer_time_sec",
    ]

    for column in expected_columns:

        if column not in comparison.columns:
            comparison[column] = None

    comparison = comparison[
        expected_columns
    ]

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        output_path,
        index=False
    )

    return comparison