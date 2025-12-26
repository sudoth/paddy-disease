from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from paddy_disease.inference.triton_client import TritonCfg, predict_image

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

TRITON_CFG = TritonCfg()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    img = Image.open(file.file)
    result = predict_image(img, TRITON_CFG)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
