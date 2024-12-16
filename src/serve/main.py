import os
import joblib
from google.cloud import storage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import torch
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


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def setup_artifacts(artifacts_uri):
    path_parts = artifacts_uri.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    artifacts_path = "/".join(path_parts[1:])

    files_to_download = ["brands_encoder.pkl", "model.pt"]
    for file_name in files_to_download:
        local_file_path = f"model/{file_name}"
        blob_name = f"{artifacts_path}/{file_name}"
        download_blob(bucket_name, blob_name, local_file_path)
    return


@app.on_event("startup")
def load_model():
    global session
    model_path = os.environ.get("AIP_STORAGE_URI", "model")
    if "gs://" in model_path:
        setup_artifacts(model_path)
    print("Loading model artifacts from:", model_path)
    encoder_path = os.path.join("model", "brands_encoder.pkl")
    brands_encoder = joblib.load(encoder_path)
    linguerie_model_path = os.path.join("model", "model.pt")
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


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
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
