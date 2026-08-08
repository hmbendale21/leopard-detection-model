FROM python:3.11-slim

WORKDIR /app

# System deps needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 4 --timeout 120 -b 0.0.0.0:${PORT} app:app"]
