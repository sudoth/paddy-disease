# flake8: noqa

import base64
import json
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from paddy_disease.inference.triton_client import TritonCfg, predict_image

app = FastAPI(title="Paddy Disease классификатор")

_LABELS_PATH = Path("models/labels.json")


def _load_labels() -> list[str]:
    if not _LABELS_PATH.exists():
        return []
    return json.loads(_LABELS_PATH.read_text(encoding="utf-8"))


def _render_page(
    *,
    image_b64: str | None = None,
    filename: str | None = None,
    pred_label: str | None = None,
    pred_conf: float | None = None,
    topk: list[tuple[str, float]] | None = None,
    error: str | None = None,
) -> str:
    img_block = ""
    if image_b64:
        img_block = f"""
        <div style="margin-top:16px;">
          <h3>Загруженное изображение</h3>
          <img src="data:image/jpeg;base64,{image_b64}"
               alt="{filename or 'uploaded'}"
               style="max-width:420px; width:100%; border-radius:12px; border:1px solid #ddd;" />
          <div style="margin-top:8px; color:#666;">{filename or ""}</div>
        </div>
        """

    pred_block = ""
    if pred_label:
        conf_txt = f" ({pred_conf:.3f})" if pred_conf is not None else ""
        pred_block = f"""
        <div style="margin-top:16px; padding:12px; border-radius:12px; border:1px solid #ddd;">
          <h3 style="margin:0 0 8px 0;">Вердикт</h3>
          <div style="font-size:18px;">
            <b>{pred_label}</b>{conf_txt}
          </div>
        </div>
        """

    topk_block = ""
    if topk:
        rows = "\n".join(
            f"<tr><td style='padding:6px 10px;border-bottom:1px solid #eee;'>{lbl}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:right;'>{score:.4f}</td></tr>"
            for lbl, score in topk
        )
        topk_block = f"""
        <div style="margin-top:16px;">
          <h3>Top-k</h3>
          <table style="border-collapse:collapse; width:420px; max-width:100%;">
            <thead>
              <tr>
                <th style="text-align:left;padding:6px 10px;border-bottom:2px solid #ddd;">Класс</th>
                <th style="text-align:right;padding:6px 10px;border-bottom:2px solid #ddd;">Score</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """

    err_block = ""
    if error:
        err_block = f"""
        <div style="margin-top:16px; padding:12px; border-radius:12px; border:1px solid #f2b8b5; background:#fff3f2;">
          <b>Ошибка:</b> {error}
        </div>
        """

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Paddy Disease</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:0; padding:0;">
  <div style="max-width:900px; margin:0 auto; padding:24px;">
    <h1 style="margin-top:0;">Paddy Disease классификатор</h1>

    <form action="/predict" method="post" enctype="multipart/form-data"
          style="padding:12px; border:1px solid #ddd; border-radius:12px;">
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
        <input type="file" name="file" accept="image/*" required />
        <button type="submit"
                style="padding:8px 14px; border-radius:10px; border:1px solid #111; background:#111; color:#fff;">
          Predict
        </button>
      </div>
      <div style="margin-top:8px; color:#666;">
        Загрузите фото листа риса — получите предсказание класса болезни.
      </div>
    </form>

    {err_block}
    {img_block}
    {pred_block}
    {topk_block}

    <div style="margin-top:28px; color:#777; font-size:13px;">
      Inference: Triton HTTP. Model: ResNet34 (ONNX).
    </div>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _render_page()


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)) -> str:
    image_bytes = await file.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    img = Image.open(BytesIO(image_bytes))
    cfg = TritonCfg(
        url="localhost:8000",
        model_name="paddy_resnet34",
        top_k=3,
        labels_path=Path("models/labels.json"),
    )
    topk = predict_image(img, cfg)

    verdict = topk[0]["label"]
    conf = float(topk[0]["prob"])

    return _render_page(
        image_b64=image_b64,
        filename=file.filename,
        pred_label=verdict,
        pred_conf=conf,
        topk=[(x["label"], float(x["prob"])) for x in topk],
    )
