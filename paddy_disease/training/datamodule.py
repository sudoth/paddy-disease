from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from paddy_disease.data.dataset import PaddyTrainDataset
from paddy_disease.data.labels import build_label_mapping, read_train_csv
from paddy_disease.data.split import make_split
from paddy_disease.data.transforms import build_test_transform, build_train_transform


@dataclass(frozen=True)
class DataConfig:
    """
    Temporary config
    """

    raw_dir: Path
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    val_size: float = 0.05
    image_size: int = 224


class PaddyDataModule(pl.LightningDataModule):
    """
    DataModule for lightning
    """

    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pin_memory = torch.cuda.is_available()

        self.label_to_idx: dict[str, int] | None = None
        self.idx_to_label: list[str] | None = None

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: str | None = None) -> None:
        df = read_train_csv(self.cfg.raw_dir / "train.csv")
        mapping = build_label_mapping(df)
        split = make_split(df, val_size=self.cfg.val_size, seed=self.cfg.seed)

        self.label_to_idx = mapping.label_to_idx
        self.idx_to_label = mapping.idx_to_label

        self.train_ds = PaddyTrainDataset(
            raw_dir=self.cfg.raw_dir,
            df=split.train_df,
            label_to_idx=mapping.label_to_idx,
            transform=build_train_transform(image_size=self.cfg.image_size),
        )
        self.val_ds = PaddyTrainDataset(
            raw_dir=self.cfg.raw_dir,
            df=split.val_df,
            label_to_idx=mapping.label_to_idx,
            transform=build_test_transform(image_size=self.cfg.image_size),
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
        )
