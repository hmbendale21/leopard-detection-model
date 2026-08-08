FROM python:3.11-slim

# System deps required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000
EXPOSE 5000

# 2 workers x 1 thread keeps memory bounded — this app holds the whole
# YOLO + MobileNetV3 model in memory per worker.
CMD ["sh", "-c", "gunicorn -w 1 --threads 4 -b 0.0.0.0:${PORT} --timeout 120 app:app"]
