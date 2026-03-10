from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import uuid

from src.inference.predict import predict
from src.model.load_model import load_model


MODEL_PATH = "weights/mobilenet_transfer.pth"
DEVICE = "cpu"  # можно "cuda"

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


app = FastAPI(
    title="Landmark Recognition API",
    description="API для распознавания мировых достопримечательностей",
    version="1.0"
)


# Загружаем модель при старте сервера
model = load_model(
    model_type="mobilenet",
    weights_path=MODEL_PATH,
    num_classes=10,
    device=DEVICE
)


@app.get("/")
def root():
    return {
        "service": "Landmark Recognition API",
        "status": "running"
    }


@app.post("/predict")
async def predict_landmark(file: UploadFile = File(...)):

    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_id = str(uuid.uuid4())

    file_path = os.path.join(TEMP_DIR, f"{file_id}.jpg")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:

        result = predict(
            model,
            file_path,
            DEVICE
        )

    finally:
        os.remove(file_path)

    return {
        "landmark": result["name"],
        "probability": result["probability"],
        "description": result["description"]
    }