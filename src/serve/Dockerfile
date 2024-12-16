FROM python:3.11.7-slim

COPY src/serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
RUN mkdir /app/model

COPY src/serve/main.py /app/main.py


EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
