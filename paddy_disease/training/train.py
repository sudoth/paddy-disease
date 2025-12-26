from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl

from paddy_disease.training.datamodule import DataConfig, PaddyDataModule
from paddy_disease.training.lightning_module import ModelConfig, OptimConfig, PaddyLightningModule
from paddy_disease.utils.data import ensure_data


@dataclass(frozen=True)
class TrainConfig:
    """
    Temporary config
    """

    raw_dir: Path = Path("data/raw")
    epochs: int = 2
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    val_size: float = 0.05
    image_size: int = 224
    lr: float = 1e-3
    momentum: float = 0.9
    dropout: float = 0.1
    pretrained: bool = True


def train_main(cfg: TrainConfig) -> None:
    """
    Main training function

    :param cfg: Description
    :type cfg: TrainConfig
    """
    # скачивание данных встроено в train/infer
    ensure_data()

    pl.seed_everything(cfg.seed, workers=True)

    data_cfg = DataConfig(
        raw_dir=cfg.raw_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        val_size=cfg.val_size,
        image_size=cfg.image_size,
    )
    dm = PaddyDataModule(data_cfg)

    model_cfg = ModelConfig(num_classes=10, dropout=cfg.dropout, pretrained=cfg.pretrained)
    optim_cfg = OptimConfig(lr=cfg.lr, momentum=cfg.momentum)
    model = PaddyLightningModule(model_cfg=model_cfg, optim_cfg=optim_cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=20,
    )
    trainer.fit(model, datamodule=dm)
