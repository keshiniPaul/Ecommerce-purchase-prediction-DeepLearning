"""
Sessionization functions.

A new browsing session begins when the inactivity gap
between two consecutive events from the same visitor
is greater than 30 minutes.
"""

from common.config import SESSION_GAP_SECONDS


def create_sessions(events_clean):
    """
    Create browsing sessions using the common inactivity rule.

    Parameters
    ----------
    events_clean : pandas.DataFrame
        Cleaned event dataframe sorted by visitor and datetime.

    Returns
    -------
    pandas.DataFrame
        Event dataframe with session information.
    """

    df = events_clean.copy()

    # Previous event time for each visitor
    df["previous_datetime"] = (
        df.groupby("visitorid")["datetime"].shift(1)
    )

    # Time difference between consecutive events
    df["time_gap_seconds"] = (
        df["datetime"] - df["previous_datetime"]
    ).dt.total_seconds()

    # First event or gap > 30 minutes = new session
    df["new_session"] = (
        df["previous_datetime"].isna()
        | (df["time_gap_seconds"] > SESSION_GAP_SECONDS)
    )

    # Number sessions independently for each visitor
    df["session_number"] = (
        df.groupby("visitorid")["new_session"]
        .cumsum()
        .astype(int)
    )

    # Globally usable session identifier
    df["session_id"] = (
        df["visitorid"].astype(str)
        + "_"
        + df["session_number"].astype(str)
    )

    return df