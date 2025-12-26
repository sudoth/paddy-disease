from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from paddy_disease.config import SplitConfig


@dataclass(frozen=True)
class Split:
    train_df: pd.DataFrame
    val_df: pd.DataFrame


def make_split(df: pd.DataFrame, cfg: SplitConfig) -> Split:
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=df["label"],
    )
    return Split(train_df=train_df.reset_index(drop=True), val_df=val_df.reset_index(drop=True))
