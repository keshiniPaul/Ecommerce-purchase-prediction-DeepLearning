"""Training utilities for the Transformer Encoder."""

from pathlib import Path
import time

import tensorflow as tf

from models.transformer.config import (
    BATCH_SIZE,
    EPOCHS,
)


def train_transformer(
    model,
    X_train_event,
    X_train_item,
    y_train,
    X_val_event,
    X_val_item,
    y_val,
    class_weights=None,
    checkpoint_path=None,
):
    """
    Train the Transformer using training and validation data.
    """

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if checkpoint_path is not None:
        checkpoint_path = Path(
            checkpoint_path
        )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )

    start_time = time.perf_counter()

    history = model.fit(
        {
            "event_input": X_train_event,
            "item_input": X_train_item,
        },
        y_train,
        validation_data=(
            {
                "event_input": X_val_event,
                "item_input": X_val_item,
            },
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    training_time = (
        time.perf_counter() - start_time
    )

    return history, training_time