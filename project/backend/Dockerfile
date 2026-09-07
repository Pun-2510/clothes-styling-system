FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend

COPY src ./src

COPY data/processed/products.csv ./data/processed/products.csv

COPY embeddings ./embeddings

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0"]