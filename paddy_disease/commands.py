from dataclasses import dataclass
from pathlib import Path

import fire
import uvicorn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from paddy_disease.config import (
    AppConfig,
    CheckpointConfig,
    DataConfig,
    ExportOnnxConfig,
    ExportTensorRTConfig,
    LoaderConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    SplitConfig,
    TrainConfig,
    TransformsConfig,
)
from paddy_disease.export.onnx_export import export_onnx_main
from paddy_disease.export.tensorrt_export import export_tensorrt_main
from paddy_disease.inference.labels_export import export_labels
from paddy_disease.training.train import train_main
from paddy_disease.web.app import app


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

        data_dict = data["data"]
        data_cfg = DataConfig(
            raw_dir=data_dict["raw_dir"],
            split=SplitConfig(**data_dict["split"]),
            loader=LoaderConfig(**data_dict["loader"]),
            transforms=TransformsConfig(**data_dict["transforms"]),
        )
        model_cfg = ModelConfig(**data["model"])
        optim_cfg = OptimConfig(**data["optim"])
        train_cfg = TrainConfig(**data["train"])
        logging_cfg = LoggingConfig(**data["logging"])
        ckpt_cfg = CheckpointConfig(**data["checkpoint"])

        return AppConfig(
            seed=int(data["seed"]),
            data=data_cfg,
            model=model_cfg,
            optim=optim_cfg,
            train=train_cfg,
            logging=logging_cfg,
            checkpoint=ckpt_cfg,
        )

    def _load_export_onnx_cfg(self, overrides: list[str] | None = None) -> ExportOnnxConfig:
        overrides = overrides or []
        config_dir = self._config_dir_abs()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="export/onnx", overrides=overrides)

        data = OmegaConf.to_container(cfg, resolve=True)

        if "export" in data and isinstance(data["export"], dict):
            data = data["export"]

        return ExportOnnxConfig(**data)

    def _load_export_tensorrt_cfg(self, overrides: list[str] | None = None) -> ExportTensorRTConfig:
        overrides = overrides or []
        config_dir = self._config_dir_abs()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="export/tensorrt", overrides=overrides)

        data = OmegaConf.to_container(cfg, resolve=True)

        if "export" in data and isinstance(data["export"], dict):
            data = data["export"]

        return ExportTensorRTConfig(**data)

    def export_onnx(self, *overrides: str) -> None:
        app_cfg = self._load_cfg(list(overrides))
        export_cfg = self._load_export_onnx_cfg(list(overrides))
        export_onnx_main(app_cfg.model, app_cfg.optim, export_cfg)

    def export_tensorrt(self, *overrides: str) -> None:
        export_cfg = self._load_export_tensorrt_cfg(list(overrides))
        export_tensorrt_main(export_cfg)

    def export_labels(self) -> None:
        export_labels()

    def train(self, *overrides: str) -> None:
        cfg = self._load_cfg(list(overrides))
        train_main(cfg)

    def web(self, host: str = "0.0.0.0", port: int = 8081) -> None:
        uvicorn.run(app, host=host, port=port)


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
