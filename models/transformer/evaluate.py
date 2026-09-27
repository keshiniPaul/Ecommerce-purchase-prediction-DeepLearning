"""Final evaluation utilities for the Transformer model."""

import time
import numpy as np

from common.metrics import compute_all_metrics


def evaluate_transformer(
    model,
    X_event,
    X_item,
    y_true,
    threshold=0.5,
):
    """
    Evaluate the Transformer on final unseen data.
    """

    start_time = time.perf_counter()

    y_prob = model.predict(
        {
            "event_input": X_event,
            "item_input": X_item,
        },
        verbose=1,
    ).reshape(-1)

    inference_time = (
        time.perf_counter() - start_time
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    metrics["inference_time_sec"] = float(
        inference_time
    )

    metrics["params"] = int(
        model.count_params()
    )

    return metrics, y_pred, y_prob