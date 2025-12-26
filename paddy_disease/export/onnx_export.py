from dataclasses import dataclass
from pathlib import Path

import onnx
import torch

from paddy_disease.config import ModelConfig, OptimConfig
from paddy_disease.training.lightning_module import PaddyLightningModule


@dataclass
class ExportOnnxConfig:
    ckpt_path: str
    onnx_path: str
    opset: int = 17
    image_size: int = 224
    dynamic_batch: bool = True


def export_onnx_main(
    model_cfg: ModelConfig, optim_cfg: OptimConfig, export_cfg: ExportOnnxConfig
) -> None:
    ckpt_path = Path(export_cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = Path(export_cfg.onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    module = PaddyLightningModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
    )
    module.eval()

    model = module.model
    model.eval()

    dummy = torch.randn(1, 3, export_cfg.image_size, export_cfg.image_size)

    input_names = ["input"]
    output_names = ["logits"]

    dynamic_axes = None
    if export_cfg.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=int(export_cfg.opset),
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    model = onnx.load(str(out_path))
    onnx.checker.check_model(model)
    print(f"Exported ONNX in {out_path}")
