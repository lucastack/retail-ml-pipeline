import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import torch
import logging
import pandas as pd
import numpy as np


app = FastAPI()


class Instance(BaseModel):
    brand: str
    description: str
    product_category: str


class PredictRequest(BaseModel):
    instances: List[Instance]


session = None


class InferenceSession:
    def __init__(self, mlp, brands_encoder, embeddings_model):
        self.mlp = mlp
        self.brands_encoder = brands_encoder
        self.embeddings_model = embeddings_model


@app.on_event("startup")
def load_model():
    global session
    model_path = os.environ.get("AIP_STORAGE_URI", "model")
    logging.info("Loading model artifacts from:", model_path)
    logging.info(f"Artifacts: {os.listdir(model_path)}")
    encoder_path = os.path.join(model_path, "brands_encoder.pkl")
    brands_encoder = joblib.load(encoder_path)
    linguerie_model_path = os.path.join(model_path, "model.pt")
    model = torch.load(linguerie_model_path)
    model.eval()
    embeddings_model = SentenceTransformer(
        "sentence-transformers/paraphrase-albert-small-v2"
    )
    session = InferenceSession(
        mlp=model,
        brands_encoder=brands_encoder,
        embeddings_model=embeddings_model,
    )


@app.post("/predict")
def predict(request: PredictRequest):
    global session
    if session is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    brands = [instance.brand for instance in request.instances]
    descriptions = [instance.description for instance in request.instances]
    products_categories = [
        instance.product_category for instance in request.instances
    ]
    brands_df = pd.DataFrame({"brand_name": brands})
    brands_embeddings = session.brands_encoder.transform(brands_df)
    descriptions_embeddings = session.embeddings_model.encode(descriptions)
    products_categories_embeddings = session.embeddings_model.encode(
        products_categories
    )
    x = np.hstack(
        [
            descriptions_embeddings,
            products_categories_embeddings,
            brands_embeddings,
        ]
    )
    x = torch.tensor(x).to(torch.float32)
    y = session.mlp(x).detach().numpy()
    preds = y * session.mlp.std + session.mlp.mean
    return {"predictions": preds.tolist()}
