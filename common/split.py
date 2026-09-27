"""
Shared chronological train/validation/test split.
"""

import json
from pathlib import Path

from common.config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    PROCESSED_DIR,
)


def chronological_split(model_events, labels, output_dir=None):
    """
    Create one chronological session-level split shared
    by every deep learning model.

    Split:
        70% train
        15% validation
        15% test

    Parameters
    ----------
    model_events : pandas.DataFrame
        Leakage-free model events.

    labels : pandas.DataFrame
        Session-level binary labels.

    output_dir : str or pathlib.Path, optional
        Directory where split_ids.json will be saved.
        If not provided, PROCESSED_DIR from config is used.

    Returns
    -------
    dict
        Session IDs assigned to train, validation and test.
    """

    # --------------------------------------------------
    # 1. Validate split ratios
    # --------------------------------------------------

    total_ratio = (
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
    )

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1."
        )

    # --------------------------------------------------
    # 2. Determine the start time of each session
    # --------------------------------------------------

    session_meta = (
        model_events
        .groupby("session_id")["datetime"]
        .min()
        .rename("session_start")
        .reset_index()
    )

    # --------------------------------------------------
    # 3. Keep only sessions that have labels
    # --------------------------------------------------

    valid_session_ids = set(
        labels["session_id"].astype(str)
    )

    session_meta["session_id"] = (
        session_meta["session_id"].astype(str)
    )

    session_meta = session_meta[
        session_meta["session_id"].isin(
            valid_session_ids
        )
    ].copy()

    # --------------------------------------------------
    # 4. Sort sessions chronologically
    # --------------------------------------------------

    session_meta = session_meta.sort_values(
        ["session_start", "session_id"]
    ).reset_index(drop=True)

    n_sessions = len(session_meta)

    if n_sessions == 0:
        raise ValueError(
            "No valid sessions available for splitting."
        )

    # --------------------------------------------------
    # 5. Calculate split boundaries
    # --------------------------------------------------

    train_end = int(
        n_sessions * TRAIN_RATIO
    )

    val_end = train_end + int(
        n_sessions * VAL_RATIO
    )

    # --------------------------------------------------
    # 6. Create train / validation / test splits
    # --------------------------------------------------

    split = {
        "train": (
            session_meta
            .iloc[:train_end]["session_id"]
            .tolist()
        ),
        "val": (
            session_meta
            .iloc[train_end:val_end]["session_id"]
            .tolist()
        ),
        "test": (
            session_meta
            .iloc[val_end:]["session_id"]
            .tolist()
        ),
    }

    # --------------------------------------------------
    # 7. Determine output directory
    # --------------------------------------------------

    if output_dir is None:
        output_dir = PROCESSED_DIR

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    split_path = (
        output_dir / "split_ids.json"
    )

    # --------------------------------------------------
    # 8. Save split efficiently
    # --------------------------------------------------
    #
    # Do NOT use indent=2 here.
    # The split contains more than one million session
    # IDs, so pretty-printing creates a much larger and
    # slower JSON file.
    # --------------------------------------------------

    with open(
        split_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            split,
            file,
            separators=(",", ":")
        )

    print(
        f"Split saved to: {split_path}"
    )

    return split