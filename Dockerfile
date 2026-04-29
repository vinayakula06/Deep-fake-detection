# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System dependencies:
#   tesseract-ocr  – required by pytesseract
#   libgl1         – required by OpenCV (headless build still needs libGL)
#   libglib2.0-0   – required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying the rest of the source so that
# the layer is cached as long as requirements.txt does not change.
COPY dfdc/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY dfdc/ ./

# Create the uploads directory that app.py expects
RUN mkdir -p uploads

EXPOSE 8080

CMD ["python", "app.py"]
