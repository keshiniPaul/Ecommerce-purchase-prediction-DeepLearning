"""
Functions for loading and cleaning the RetailRocket events dataset.
"""

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "visitorid",
    "event",
    "itemid",
    "transactionid",
}


def load_and_clean_events(path):
    """
    Load and clean RetailRocket events.csv.

    Steps:
    1. Load CSV.
    2. Validate required columns.
    3. Remove exact duplicate rows.
    4. Convert timestamp from milliseconds to datetime.
    5. Remove rows missing essential event information.
    6. Keep relevant RetailRocket event types.
    7. Sort events chronologically for each visitor.

    Parameters
    ----------
    path : str
        Path to events.csv.

    Returns
    -------
    pandas.DataFrame
        Cleaned event table.
    """

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Remove exact duplicate records
    df = df.drop_duplicates().copy()

    # Convert Unix timestamp in milliseconds to datetime
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        errors="coerce"
    )

    # Remove records missing information required for modelling
    df = df.dropna(
        subset=[
            "datetime",
            "visitorid",
            "event",
            "itemid",
        ]
    )

    # Keep the event types relevant to this project
    df = df[
        df["event"].isin(
            ["view", "addtocart", "transaction"]
        )
    ].copy()

    # Sort chronologically within visitor
    df = df.sort_values(
        ["visitorid", "datetime"]
    ).reset_index(drop=True)

    return df[
        [
            "timestamp",
            "datetime",
            "visitorid",
            "event",
            "itemid",
            "transactionid",
        ]
    ]