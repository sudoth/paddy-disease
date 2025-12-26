from __future__ import annotations

import pytorch_lightning as pl

from paddy_disease.config import AppConfig
from paddy_disease.training.datamodule import PaddyDataModule
from paddy_disease.training.lightning_module import PaddyLightningModule
from paddy_disease.utils.data import ensure_data


def train_main(cfg: AppConfig) -> None:
    ensure_data()
    pl.seed_everything(cfg.seed, workers=True)

    dm = PaddyDataModule(data_cfg=cfg.data, seed=cfg.seed)
    model = PaddyLightningModule(model_cfg=cfg.model, optim_cfg=cfg.optim)

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=cfg.train.log_every_n_steps,
    )
    trainer.fit(model, datamodule=dm)
