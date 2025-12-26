from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrainItem:
    image_id: str
    label: str
    variety: str | None
    age: int | None
    path: Path


class PaddyTrainDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """
    Train/val dataset.
    Returns: (image_tensor, target_idx, image_id)
    """

    def __init__(
        self,
        raw_dir: Path,
        df: pd.DataFrame,
        label_to_idx: dict[str, int],
        transform: Callable[[Image.Image], Any],
    ) -> None:
        self.raw_dir = raw_dir
        self.train_images_dir = raw_dir / "train_images"
        self.label_to_idx = label_to_idx
        self.transform = transform

        items: list[TrainItem] = []
        for row in df.itertuples(index=False):
            image_id = getattr(row, "image_id")
            label = getattr(row, "label")
            variety = getattr(row, "variety", None)
            age = getattr(row, "age", None)
            path = self.train_images_dir / label / image_id
            items.append(
                TrainItem(
                    image_id=str(image_id),
                    label=str(label),
                    variety=None if pd.isna(variety) else str(variety),
                    age=None if pd.isna(age) else int(age),
                    path=path,
                )
            )
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        if not item.path.exists():
            raise FileNotFoundError(f"Image not found: {item.path}")

        image = Image.open(item.path).convert("RGB")
        x = self.transform(image)
        y = self.label_to_idx[item.label]
        return x, y, item.image_id


class PaddyTestDataset(Dataset[tuple[torch.Tensor, str]]):
    """
    Test dataset (for inference/submission).
    Returns: (image_tensor, image_id)
    """

    def __init__(
        self,
        raw_dir: Path,
        image_ids: list[str],
        transform: Callable[[Image.Image], Any],
    ) -> None:
        self.raw_dir = raw_dir
        self.test_images_dir = raw_dir / "test_images"
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_id = self.image_ids[idx]
        path = self.test_images_dir / image_id
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        return x, image_id
