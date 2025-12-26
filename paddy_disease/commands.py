from dataclasses import dataclass
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from paddy_disease.config import (
    AppConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from paddy_disease.training.train import train_main


@dataclass
class Commands:
    config_name: str = "config"

    def _config_dir_abs(self) -> str:
        repo_root = Path(__file__).resolve().parents[1]
        return str(repo_root / "configs")

    def _load_cfg(self, overrides: list[str] | None = None) -> AppConfig:
        overrides = overrides or []
        config_dir = self._config_dir_abs()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=self.config_name, overrides=overrides)

        data = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(data, dict)

        data_cfg = DataConfig(**data["data"])  # type: ignore[arg-type]
        model_cfg = ModelConfig(**data["model"])
        optim_cfg = OptimConfig(**data["optim"])
        train_cfg = TrainConfig(**data["train"])
        logging_cfg = LoggingConfig(**data["logging"])

        return AppConfig(
            seed=int(data["seed"]),
            data=data_cfg,
            model=model_cfg,
            optim=optim_cfg,
            train=train_cfg,
            logging=logging_cfg,
        )

    def train(self, *overrides: str) -> None:
        cfg = self._load_cfg(list(overrides))
        train_main(cfg)


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
