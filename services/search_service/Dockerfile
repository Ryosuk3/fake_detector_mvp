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
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
# Обновляем pip для лучшей производительности
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
    httpx==0.25.2 \
    beautifulsoup4==4.12.2 \
    lxml==4.9.3 \
    numpy==1.24.3 \
    minio==7.2.0 \
    python-dateutil==2.8.2 \
    simhash==2.1.2 \
    feedparser==6.0.10 \
    qdrant-client==1.16.2

# Устанавливаем тяжелые зависимости отдельно
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install trafilatura==1.6.3 newspaper3k==0.2.8

# Устанавливаем зависимости sentence-transformers вручную из обычного PyPI,
# чтобы не переустанавливать torch из другого индекса.
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

# Копирование конфигурации
COPY config/ /app/config/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
