# =============================================================================
# VEGA BACKEND - Multi-stage Production Dockerfile
# =============================================================================
# Stages:
# 1. base - Common dependencies
# 2. builder - Compile/install packages
# 3. production - Final minimal image
# =============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: BASE
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} vega && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash vega

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    portaudio19-dev \
    ffmpeg \
    libsndfile1 \
    libpq-dev \
    redis-tools \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# STAGE 2: BUILDER
# -----------------------------------------------------------------------------
FROM base AS builder

# Install additional build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA (will use CPU if no GPU)
ARG CUDA_VERSION=cu118
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install additional ML packages
RUN pip install --no-cache-dir \
    ultralytics \
    deepface \
    insightface \
    mediapipe \
    onnxruntime-gpu || onnxruntime

# Pre-download models (optional - can be done at runtime)
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
#     python -c "import whisper; whisper.load_model('base')" && \
#     python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"

# -----------------------------------------------------------------------------
# STAGE 3: PRODUCTION
# -----------------------------------------------------------------------------
FROM base AS production

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/.vega && \
    chown -R vega:vega /app

# Copy application code
COPY --chown=vega:vega . .

# Switch to non-root user
USER vega

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "main.py"]

# -----------------------------------------------------------------------------
# STAGE 4: DEVELOPMENT (optional)
# -----------------------------------------------------------------------------
FROM base AS development

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    pre-commit \
    ipython \
    jupyter

# Install PyTorch CPU for development
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY --chown=vega:vega . .

USER vega

# Override command for development
CMD ["python", "main.py"]