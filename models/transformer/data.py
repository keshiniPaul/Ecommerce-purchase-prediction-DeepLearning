"""Sequence preprocessing for the RetailRocket Transformer.

All learned preprocessing is fitted using training data only.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.transformer.config import (
    MAX_SEQUENCE_LENGTH,
    PAD_TOKEN,
    OOV_TOKEN,
)


# Event encoding
EVENT_TO_ID = {
    "view": 1,
    "addtocart": 2,
}


def split_events_by_session_ids(model_events, split_ids):
    """
    Split events using the shared chronological session split.
    """

    train_ids = set(map(str, split_ids["train"]))
    val_ids = set(map(str, split_ids["val"]))
    test_ids = set(map(str, split_ids["test"]))

    events = model_events.copy()
    events["session_id"] = events["session_id"].astype(str)

    train_events = events[
        events["session_id"].isin(train_ids)
    ].copy()

    val_events = events[
        events["session_id"].isin(val_ids)
    ].copy()

    test_events = events[
        events["session_id"].isin(test_ids)
    ].copy()

    return train_events, val_events, test_events


def build_item_vocabulary(train_events):
    """
    Build item vocabulary using training data only.

    0 = padding
    1 = out-of-vocabulary
    2+ = known item IDs
    """

    unique_items = (
        train_events["itemid"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    item_vocab = {
        item_id: index + 2
        for index, item_id in enumerate(unique_items)
    }

    return item_vocab


def encode_item(item_id, item_vocab):
    """
    Convert an item ID to its vocabulary index.

    Unseen items become OOV.
    """

    return item_vocab.get(item_id, OOV_TOKEN)


def create_session_sequences(
    events,
    labels,
    item_vocab,
):
    """
    Convert session events into ordered event/item sequences.
    """

    events = events.copy()
    labels = labels.copy()

    events["session_id"] = events["session_id"].astype(str)
    labels["session_id"] = labels["session_id"].astype(str)

    # Ensure chronological event order
    events = events.sort_values(
        ["session_id", "datetime"]
    )

    label_map = dict(
        zip(
            labels["session_id"],
            labels["target"],
        )
    )

    sessions = []

    for session_id, group in events.groupby(
        "session_id",
        sort=False,
    ):

        if session_id not in label_map:
            continue

        event_sequence = [
            EVENT_TO_ID.get(event, PAD_TOKEN)
            for event in group["event"]
        ]

        item_sequence = [
            encode_item(item, item_vocab)
            for item in group["itemid"]
        ]

        sessions.append(
            {
                "session_id": session_id,
                "event_sequence": event_sequence,
                "item_sequence": item_sequence,
                "target": int(label_map[session_id]),
            }
        )

    return pd.DataFrame(sessions)


def pad_session_sequences(session_df):
    """
    Pad/truncate event and item sequences to the configured
    maximum sequence length.

    Post-padding and post-truncation are used.
    """

    event_sequences = session_df[
        "event_sequence"
    ].tolist()

    item_sequences = session_df[
        "item_sequence"
    ].tolist()

    X_event = pad_sequences(
        event_sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
        value=PAD_TOKEN,
        dtype="int32",
    )

    X_item = pad_sequences(
        item_sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
        value=PAD_TOKEN,
        dtype="int32",
    )

    y = session_df["target"].to_numpy(
        dtype=np.float32
    )

    return X_event, X_item, y