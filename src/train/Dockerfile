FROM python:3.11.7-slim

COPY src/train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/train/train.py /app/train.py
COPY src/train/linguerie_model.py /app/linguerie_model.py

WORKDIR /app
