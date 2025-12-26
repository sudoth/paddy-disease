import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class TritonCfg:
    url: str = "localhost:8000"
    model_name: str = "paddy_resnet34"
    image_size: int = 224
    top_k: int = 3
    labels_path: Path = Path("models/labels.json")


def _preprocess(img: Image.Image, image_size: int) -> np.ndarray:
    img = img.convert("RGB").resize((image_size, image_size))
    x = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, axis=0)  # NCHW
    return x.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _load_labels(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_image(img: Image.Image, cfg: TritonCfg) -> list[dict]:
    labels = _load_labels(cfg.labels_path)

    x = _preprocess(img, cfg.image_size)

    client = httpclient.InferenceServerClient(url=cfg.url, verbose=False)

    inputs = [httpclient.InferInput("input", x.shape, "FP32")]
    inputs[0].set_data_from_numpy(x)

    outputs = [httpclient.InferRequestedOutput("logits")]

    resp = client.infer(model_name=cfg.model_name, inputs=inputs, outputs=outputs)
    logits = resp.as_numpy("logits")  # (1, 10)
    probs = _softmax(logits)[0]

    top_idx = np.argsort(-probs)[: cfg.top_k]
    result = []
    for i in top_idx:
        result.append({"label": labels[int(i)], "prob": float(probs[int(i)])})
    return result
