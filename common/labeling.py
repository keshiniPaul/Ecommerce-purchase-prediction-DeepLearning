"""
Create session-level purchase labels and leakage-free model events.
"""

from common.config import ALLOWED_INPUT_EVENTS


def create_labels_and_model_events(events_with_sessions):
    """
    Create binary session labels and leakage-free model input events.

    Target:
        0 = session contains no transaction
        1 = session contains a transaction

    Transaction events are never used as model input.

    For positive sessions, only view/addtocart events occurring
    before the first transaction are retained.

    Sessions with no valid behavioural events remaining after
    leakage prevention are removed.

    Parameters
    ----------
    events_with_sessions : pandas.DataFrame
        Sessionized event dataframe.

    Returns
    -------
    model_events : pandas.DataFrame
        Leakage-free behavioural events.

    labels : pandas.DataFrame
        Session IDs and binary targets.
    """

    df = events_with_sessions.copy()

    # --------------------------------------------------
    # Create session-level target
    # --------------------------------------------------

    labels = (
        df.groupby("session_id")["event"]
        .apply(
            lambda events: int(
                (events == "transaction").any()
            )
        )
        .rename("target")
        .reset_index()
    )

    df = df.merge(
        labels,
        on="session_id",
        how="left"
    )

    # --------------------------------------------------
    # Find first transaction in every positive session
    # --------------------------------------------------

    first_transaction_times = (
        df[df["event"] == "transaction"]
        .groupby("session_id")["datetime"]
        .min()
        .rename("first_transaction_time")
    )

    df = df.merge(
        first_transaction_times,
        on="session_id",
        how="left"
    )

    # --------------------------------------------------
    # Keep only events occurring before purchase
    # --------------------------------------------------

    before_transaction = (
        df["first_transaction_time"].isna()
        | (
            df["datetime"]
            < df["first_transaction_time"]
        )
    )

    # --------------------------------------------------
    # Remove transaction events from model input
    # --------------------------------------------------

    model_events = df[
        before_transaction
        & df["event"].isin(ALLOWED_INPUT_EVENTS)
    ].copy()

    model_events = model_events[
        [
            "session_id",
            "visitorid",
            "event",
            "itemid",
            "datetime",
            "timestamp",
            "target",
        ]
    ]

    # --------------------------------------------------
    # Remove sessions that have no usable input events
    # --------------------------------------------------

    valid_session_ids = (
        model_events["session_id"]
        .drop_duplicates()
    )

    model_events = model_events[
        model_events["session_id"].isin(
            valid_session_ids
        )
    ].reset_index(drop=True)

    labels = labels[
        labels["session_id"].isin(
            valid_session_ids
        )
    ].reset_index(drop=True)

    return model_events, labels