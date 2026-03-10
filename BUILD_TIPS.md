# Советы по ускорению сборки

## Проблема: Долгая сборка (20+ минут)

`sentence-transformers` - очень тяжелый пакет, который:
- Компилирует много зависимостей из исходников
- Устанавливает transformers, torch и другие большие пакеты
- Может занимать 15-20 минут только на установку

## Решения

### 1. Использовать BuildKit (рекомендуется)

```bash
# Windows PowerShell
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1
docker-compose build

# Linux/Mac
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
docker-compose build
```

BuildKit:
- Параллельная сборка сервисов
- Умное кэширование слоев
- Ускорение на 20-30%

### 2. Сборка по отдельности

Если один сервис собирается долго, можно собрать остальные параллельно:

```bash
# Собрать только инфраструктуру (быстро)
docker-compose build postgres redis qdrant minio

# Собрать сервисы по отдельности
docker-compose build backend
docker-compose build frontend
docker-compose build nlp_service

# Тяжелые сервисы собирать в последнюю очередь
docker-compose build search_service verifier_service
```

### 3. Использовать готовые образы (для разработки)

Можно использовать предварительно собранные образы или разделить на базовый образ и сервисный:

```dockerfile
# Базовый образ с зависимостями (собирается редко)
FROM python:3.10-slim as base
RUN pip install torch sentence-transformers ...

# Сервисный образ (собирается часто)
FROM base
COPY . /app
```

### 4. Мониторинг процесса

Следите за прогрессом:

```bash
# В другом терминале
docker-compose logs -f

# Или проверяйте размер слоев
docker system df
```

## Ожидаемое время

- **Backend**: 1-2 минуты
- **Frontend**: 2-3 минуты  
- **NLP Service**: 2-3 минуты
- **Search Service**: 15-20 минут (sentence-transformers)
- **Verifier Service**: 15-20 минут (sentence-transformers)

**Итого: 35-45 минут** при первой сборке

При повторной сборке (с кэшем): **5-10 минут**

## Если сборка зависла

Если сборка не прогрессирует более 30 минут:

1. Проверьте интернет-соединение
2. Проверьте логи: `docker-compose logs`
3. Попробуйте собрать по одному сервису
4. Очистите кэш: `docker builder prune`

## Альтернатива: Использовать готовые образы

Для продакшена рекомендуется использовать CI/CD для предварительной сборки образов и загрузки их в registry.

