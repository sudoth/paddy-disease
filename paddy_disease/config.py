from dataclasses import dataclass, field


@dataclass
class SplitConfig:
    val_size: float = 0.05
    seed: int = 42


@dataclass
class LoaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True


@dataclass
class TransformsConfig:
    image_size: int = 224
    use_augmentations: bool = True


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    split: SplitConfig = field(default_factory=SplitConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)


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
class ExportOnnxConfig:
    ckpt_path: str
    onnx_path: str
    opset: int = 17
    image_size: int = 224
    dynamic_batch: bool = True


@dataclass
class ExportTensorRTConfig:
    onnx_path: str
    engine_path: str
    fp16: bool = True
    max_batch: int = 8
    image_size: int = 224
    workspace_mb: int = 2048


@dataclass
class AppConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
