import subprocess
from pathlib import Path

from paddy_disease.config import ExportTensorRTConfig


def export_tensorrt_main(cfg: ExportTensorRTConfig) -> None:
    onnx_path = Path(cfg.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    engine_path = Path(cfg.engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "tensorrt.sh"

    if subprocess.run(["bash", "-lc", "command -v trtexec"], capture_output=True).returncode != 0:
        raise RuntimeError("trtexec not found in PATH")

    subprocess.run(
        [
            str(script),
            str(onnx_path),
            str(engine_path),
            str(cfg.max_batch),
            str(cfg.image_size),
            str(cfg.workspace_mb),
            "true" if cfg.fp16 else "false",
        ],
        check=True,
    )

    print(f"Exported TensorRT engine: {engine_path}")
