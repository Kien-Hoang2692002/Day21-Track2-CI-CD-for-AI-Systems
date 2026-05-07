from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Đọc tên bucket từ biến môi trường (được đặt trong systemd service)
GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

LABEL_MAP = {0: "thấp", 1: "trung_bình", 2: "cao"}


def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    blob.download_to_filename(MODEL_PATH)
    print(f"Model đã được tải xuống từ gs://{GCS_BUCKET}/{GCS_MODEL_KEY}")


# Gọi hàm này khi module được import (chạy khi server khởi động)
download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiểm tra sức khỏe server.
    GitHub Actions dùng endpoint này để xác nhận deploy thành công.
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận chính.

    Đầu vào : JSON {"features": [f1, f2, ..., f12]}
    Đầu ra  : JSON {"prediction": <0|1|2>, "label": <"thấp"|"trung_bình"|"cao">}

    Thứ tự 12 đặc trưng:
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 12 features (wine quality), got {len(req.features)}",
        )

    pred = int(model.predict([req.features])[0])
    return {"prediction": pred, "label": LABEL_MAP[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
