from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    raw_dir: Path
    batch_size: int
    num_workers: int
    val_size: float
    image_size: int


@dataclass
class ModelConfig:
    num_classes: int
    dropout: float
    pretrained: bool


@dataclass
class OptimConfig:
    lr: float
    momentum: float
    weight_decay: float


@dataclass
class TrainConfig:
    epochs: int
    log_every_n_steps: int


@dataclass
class LoggingConfig:
    tracking_uri: str
    experiment_name: str
    run_name: str | None
    log_model: bool


@dataclass
class CheckpointConfig:
    dirpath: str
    monitor: str
    mode: str
    save_top_k: int
    save_last: bool
    filename: str


@dataclass
class AppConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
