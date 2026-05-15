# syntax=docker/dockerfile:1.7
FROM python:3.10-slim

WORKDIR /app

ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Устанавливаем torch через параметризуемый индекс.
# По умолчанию используется CPU-колесо, GPU-режим включается через docker-compose.gpu.yml.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

# Устанавливаем легкие зависимости сначала
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    numpy==1.24.3 \
    qdrant-client==1.16.2

# Устанавливаем зависимости sentence-transformers вручную из обычного PyPI
# Используем совместимые версии
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    transformers==4.35.2 \
    sentencepiece==0.1.99 \
    scikit-learn==1.3.2 \
    huggingface-hub==0.20.0 \
    nltk==3.8.1 \
    scipy==1.11.4 \
    Pillow==10.4.0

# Ставим sentence-transformers без dependency-resolution: все ключевые зависимости уже закреплены выше.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install sentence-transformers==2.2.2 --no-deps

# Копирование кода
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
