FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
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
    numpy==1.24.3 \
    qdrant-client==1.7.0

# Устанавливаем зависимости sentence-transformers вручную из обычного PyPI
# Используем совместимые версии
RUN pip install --no-cache-dir \
    transformers==4.35.2 \
    sentencepiece==0.1.99 \
    scikit-learn==1.3.2 \
    huggingface-hub==0.20.0

# Устанавливаем sentence-transformers с указанием что torch уже установлен
# Используем --upgrade-strategy only-if-needed чтобы не переустанавливать torch
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed sentence-transformers==2.2.2

# Копирование кода
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]

