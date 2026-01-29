# syntax=docker/dockerfile:1
# PaddleOCR v5 ML Backend for Label Studio
#
# Build:
#   docker build -t paddleocr-label-studio-backend .
#
# Run:
#   docker run -p 9090:9090 paddleocr-label-studio-backend

ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim AS python-base
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8

# Install system dependencies required by PaddleOCR and OpenCV
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install --no-install-recommends -y \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        curl; \
    apt-get autoremove -y

# Install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# Install custom requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Install test requirements if needed
COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

# Copy application code
COPY . .

# Environment variables with defaults
ENV IMAGE_DATA_KEY=image \
    IMAGE_TO_NAME=image \
    BBOX_FROM_NAME=bbox \
    LABEL_FROM_NAME=label \
    TEXT_FROM_NAME=transcription \
    PREDICTED_LABEL=Text \
    PADDLEOCR_VERSION=PP-OCRv5 \
    PADDLEOCR_MODEL_TYPE=server \
    PADDLEOCR_LANG=ch \
    PADDLEOCR_USE_GPU=false \
    PADDLEOCR_USE_DOC_ORIENTATION=false \
    PADDLEOCR_USE_DOC_UNWARPING=false \
    PADDLEOCR_USE_TEXTLINE_ORIENTATION=true \
    PADDLEOCR_SCORE_THRESHOLD=0.5

EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
