FROM python:3.10-slim

# Install system dependencies required for OpenCV + wget for model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .

# Install CPU-only PyTorch wheels to dramatically reduce image footprint
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application files
COPY main.py .

# Download model weights from HuggingFace Hub (URL injected via MODEL_URL build arg or env var)
# If MODEL_URL is not set the app falls back to adaptive-threshold inference automatically.
ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then \
      echo "Downloading model from $MODEL_URL" && \
      wget -q -O /app/best_model.pth "$MODEL_URL"; \
    else \
      echo "MODEL_URL not set — skipping model download (fallback inference will be used)"; \
    fi

EXPOSE 8000

ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
