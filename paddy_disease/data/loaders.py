from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from paddy_disease.config import DataConfig
from paddy_disease.data.dataset import PaddyTrainDataset
from paddy_disease.data.labels import build_label_mapping, read_train_csv
from paddy_disease.data.split import make_split
from paddy_disease.data.transforms import build_test_transform, build_train_transform


@dataclass(frozen=True)
class Loaders:
    train_loader: DataLoader
    val_loader: DataLoader
    label_to_idx: dict[str, int]
    idx_to_label: list[str]


def build_train_val_loaders(cfg: DataConfig) -> Loaders:
    raw_dir = Path(cfg.raw_dir)

    df = read_train_csv(raw_dir / "train.csv")
    mapping = build_label_mapping(df)
    split = make_split(df, cfg.split)

    train_ds = PaddyTrainDataset(
        raw_dir=raw_dir,
        df=split.train_df,
        label_to_idx=mapping.label_to_idx,
        transform=build_train_transform(cfg.transforms),
    )
    val_ds = PaddyTrainDataset(
        raw_dir=raw_dir,
        df=split.val_df,
        label_to_idx=mapping.label_to_idx,
        transform=build_test_transform(cfg.transforms),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle_train,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )

    return Loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        label_to_idx=mapping.label_to_idx,
        idx_to_label=mapping.idx_to_label,
    )
