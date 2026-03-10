FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
# Обновляем pip для лучшей производительности
RUN pip install --upgrade pip setuptools wheel

# Устанавливаем torch CPU-only версию с явным указанием индекса
# Это предотвратит установку CUDA версии
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Устанавливаем легкие зависимости сначала
RUN pip install --no-cache-dir \
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
    qdrant-client==1.7.0

# Устанавливаем тяжелые зависимости отдельно
RUN pip install --no-cache-dir trafilatura==1.6.3 newspaper3k==0.2.8

# Устанавливаем зависимости sentence-transformers вручную из обычного PyPI
# Это предотвратит переустановку torch с CUDA
RUN pip install --no-cache-dir \
    transformers \
    sentencepiece \
    scikit-learn

# Устанавливаем sentence-transformers с указанием что torch уже установлен
# Используем --upgrade-strategy only-if-needed чтобы не переустанавливать torch
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed sentence-transformers==2.2.2

# Копирование кода
COPY . .

# Копирование конфигурации
COPY config/ /app/config/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

