from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LabelMapping:
    """
    Class for label mapping
    """

    label_to_idx: dict[str, int]
    idx_to_label: list[str]


def read_train_csv(train_csv: Path) -> pd.DataFrame:
    """
    Function to read .csv

    :param train_csv: Description
    :type train_csv: Path
    :return: Description
    :rtype: DataFrame
    """
    df = pd.read_csv(train_csv)
    required = {"image_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")
    return df


def build_label_mapping(df: pd.DataFrame) -> LabelMapping:
    """
    Function that returns LabelMapping

    :param df: Description
    :type df: pd.DataFrame
    :return: Description
    :rtype: LabelMapping
    """
    classes = sorted(df["label"].unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(classes)}
    return LabelMapping(label_to_idx=label_to_idx, idx_to_label=classes)
