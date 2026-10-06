import pandas as pd


def add_date_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract useful features from the Date column."""
    dataframe = dataframe.copy()

    dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    dataframe["Month"] = dataframe["Date"].dt.month

    dataframe = dataframe.drop(columns=["Date"])

    return dataframe