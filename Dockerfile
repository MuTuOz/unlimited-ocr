# torch 2.5.1 + CUDA 12.4 already baked in, so pip never resolves torch itself
# (that resolution is what dragged in the CUDA 13 wheels and 500 MB of churn).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Model weights live on a mounted volume, never inside the image.
    HF_HOME=/models \
    # Fixes the "reserved but unallocated" fragmentation in the OOM message.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Remote code is required by Unlimited-OCR; acknowledge it once here.
    TRUST_REMOTE_CODE=1

# libgl1/libglib2.0-0: OpenCV pulled in by the OCR remote code needs these.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# /data = your PDFs in, OCR text + FAISS index out
VOLUME ["/models", "/data"]

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
