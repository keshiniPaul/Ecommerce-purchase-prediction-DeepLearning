"""
Common classification metrics used by all four models.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_all_metrics(
    y_true,
    y_pred,
    y_prob=None
):
    """
    Calculate common classification metrics.

    Parameters
    ----------
    y_true :
        Ground-truth binary labels.

    y_pred :
        Predicted binary labels.

    y_prob :
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Common evaluation metrics.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
    }

    if y_prob is not None:

        y_prob = np.asarray(y_prob).reshape(-1)

        # ROC-AUC requires both classes
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    y_prob
                )
            )
        else:
            metrics["roc_auc"] = np.nan

    else:
        metrics["roc_auc"] = np.nan

    metrics["confusion_matrix"] = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )
    )

    return metrics


def metrics_to_dataframe(
    metrics_dict,
    model_name
):
    """
    Convert metric dictionary into a one-row DataFrame.
    """

    row = {
        key: value
        for key, value in metrics_dict.items()
        if key != "confusion_matrix"
    }

    row["model"] = model_name

    return pd.DataFrame([row])