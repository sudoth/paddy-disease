from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Split:
    """
    Split class
    """

    train_df: pd.DataFrame
    val_df: pd.DataFrame


def make_split(
    df: pd.DataFrame,
    val_size: float = 0.05,
    seed: int = 42,
) -> Split:
    """
    Function that applies split

    :param df: Description
    :type df: pd.DataFrame
    :param val_size: Description
    :type val_size: float
    :param seed: Description
    :type seed: int
    :return: Description
    :rtype: Split
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["label"],
    )
    # сброс индекса для аккуратности
    return Split(train_df=train_df.reset_index(drop=True), val_df=val_df.reset_index(drop=True))
