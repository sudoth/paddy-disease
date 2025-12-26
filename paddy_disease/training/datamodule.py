from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from paddy_disease.config import DataConfig
from paddy_disease.data.dataset import PaddyTrainDataset
from paddy_disease.data.labels import build_label_mapping, read_train_csv
from paddy_disease.data.split import make_split
from paddy_disease.data.transforms import build_test_transform, build_train_transform


class PaddyDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: DataConfig, seed: int) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.seed = seed
        self.pin_memory = torch.cuda.is_available()

        self.label_to_idx: dict[str, int] | None = None
        self.idx_to_label: list[str] | None = None

        self.train_ds = None
        self.val_ds = None

    @property
    def raw_dir(self) -> Path:
        # data_cfg.raw_dir может прийти как Path (в идеале) или как str — приведём к Path
        return Path(self.data_cfg.raw_dir)

    def setup(self, stage: str | None = None) -> None:
        df = read_train_csv(self.raw_dir / "train.csv")
        mapping = build_label_mapping(df)
        split = make_split(df, val_size=self.data_cfg.val_size, seed=self.seed)

        self.label_to_idx = mapping.label_to_idx
        self.idx_to_label = mapping.idx_to_label

        self.train_ds = PaddyTrainDataset(
            raw_dir=self.raw_dir,
            df=split.train_df,
            label_to_idx=mapping.label_to_idx,
            transform=build_train_transform(image_size=self.data_cfg.image_size),
        )
        self.val_ds = PaddyTrainDataset(
            raw_dir=self.raw_dir,
            df=split.val_df,
            label_to_idx=mapping.label_to_idx,
            transform=build_test_transform(image_size=self.data_cfg.image_size),
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.data_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.pin_memory,
        )
