from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

# Doc ten bucket tu bien moi truong
S3_BUCKET = os.environ.get("CLOUD_BUCKET")
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("models/model.pkl")

def download_model():
    """Tai file model.pkl tu S3 ve may khi server khoi dong."""
    if not S3_BUCKET:
        print("Warning: CLOUD_BUCKET environment variable not set.")
        return

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        s3 = boto3.client("s3")
        print(f"Downloading model from s3://{S3_BUCKET}/{S3_MODEL_KEY}...")
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Tai model khi khoi dong
if not os.path.exists(MODEL_PATH):
    download_model()

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan.
    Dau vao: JSON {"features": [f1, f2, ..., f12]}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    prediction = model.predict([req.features])[0]
    
    # Anh xa nhan
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = labels.get(int(prediction), "unknown")

    return {"prediction": int(prediction), "label": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
