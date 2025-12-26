from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from paddy_disease.data.dataset import PaddyTrainDataset
from paddy_disease.data.labels import build_label_mapping, read_train_csv
from paddy_disease.data.split import make_split
from paddy_disease.data.transforms import build_test_transform, build_train_transform


@dataclass(frozen=True)
class Loaders:
    """
    Loaders class
    """

    train_loader: DataLoader
    val_loader: DataLoader
    label_to_idx: dict[str, int]
    idx_to_label: list[str]


def build_train_val_loaders(
    raw_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    val_size: float = 0.05,
    image_size: int = 224,
) -> Loaders:
    """
    Function that returns Loaders

    :param raw_dir: Description
    :type raw_dir: Path
    :param batch_size: Description
    :type batch_size: int
    :param num_workers: Description
    :type num_workers: int
    :param seed: Description
    :type seed: int
    :param val_size: Description
    :type val_size: float
    :param image_size: Description
    :type image_size: int
    :return: Description
    :rtype: Loaders
    """
    df = read_train_csv(raw_dir / "train.csv")
    mapping = build_label_mapping(df)
    split = make_split(df, val_size=val_size, seed=seed)

    train_ds = PaddyTrainDataset(
        raw_dir=raw_dir,
        df=split.train_df,
        label_to_idx=mapping.label_to_idx,
        transform=build_train_transform(image_size=image_size),
    )
    val_ds = PaddyTrainDataset(
        raw_dir=raw_dir,
        df=split.val_df,
        label_to_idx=mapping.label_to_idx,
        transform=build_test_transform(image_size=image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return Loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        label_to_idx=mapping.label_to_idx,
        idx_to_label=mapping.idx_to_label,
    )
