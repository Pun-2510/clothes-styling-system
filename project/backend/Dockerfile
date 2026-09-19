FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

# Compose currently serves on CPU. Install the CPU wheel explicitly so the
# backend image does not pull CUDA and NVIDIA runtime packages from PyPI.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 && \
    pip install -r requirements.txt

COPY backend ./backend

COPY src ./src

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
