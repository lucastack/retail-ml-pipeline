import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

app = FastAPI()


class Instance(BaseModel):
    features: List[float]


class PredictRequest(BaseModel):
    instances: List[Instance]


model = None


@app.on_event("startup")
def load_model():
    global model
    model_path = os.environ.get("AIP_STORAGE_URI", "/model")
    logging.info("Loading model from:", model_path)
    model = joblib.load(os.path.join(model_path, "model.joblib"))


@app.post("/predict")
def predict(request: PredictRequest):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    instances = [instance.features for instance in request.instances]

    preds = model.predict(instances)

    return {
        "predictions": preds.tolist() if hasattr(preds, "tolist") else preds
    }