from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from paddy_disease.config import AppConfig
from paddy_disease.training.datamodule import PaddyDataModule
from paddy_disease.training.lightning_module import PaddyLightningModule
from paddy_disease.training.plot_callback import SavePlotsCallback
from paddy_disease.utils.data import ensure_data
from paddy_disease.utils.git import get_git_commit


def train_main(cfg: AppConfig) -> None:
    ensure_data()
    pl.seed_everything(cfg.seed, workers=True)

    dm = PaddyDataModule(data_cfg=cfg.data, seed=cfg.seed)
    model = PaddyLightningModule(model_cfg=cfg.model, optim_cfg=cfg.optim)

    repo_root = Path(__file__).parents[2]

    mlf_logger = MLFlowLogger(
        tracking_uri=cfg.logging.tracking_uri,
        experiment_name=cfg.logging.experiment_name,
        run_name=cfg.logging.run_name,
    )

    # log git commit id
    commit = get_git_commit(repo_root)
    if commit is not None:
        mlf_logger.log_hyperparams({"git_commit": commit})

    plots_dir = repo_root / "plots"
    callbacks = [SavePlotsCallback(plots_dir)]

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=cfg.train.log_every_n_steps,
        callbacks=callbacks,
        logger=mlf_logger,
    )
    trainer.fit(model, datamodule=dm)
