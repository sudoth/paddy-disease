from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from paddy_disease.config import DataConfig
from paddy_disease.data.loaders import Loaders, build_train_val_loaders


class PaddyDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: DataConfig) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self._loaders: Optional[Loaders] = None

    def setup(self, stage: str | None = None) -> None:
        self._loaders = build_train_val_loaders(self.data_cfg)

    def train_dataloader(self) -> DataLoader:
        assert self._loaders is not None
        return self._loaders.train_loader

    def val_dataloader(self) -> DataLoader:
        assert self._loaders is not None
        return self._loaders.val_loader

    @property
    def idx_to_label(self) -> list[str]:
        assert self._loaders is not None
        return self._loaders.idx_to_label

    @property
    def label_to_idx(self) -> dict[str, int]:
        assert self._loaders is not None
        return self._loaders.label_to_idx
