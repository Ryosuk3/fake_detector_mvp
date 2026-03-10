# Система проверки достоверности информации (Fake Detector MVP)

Система для автоматизированной проверки достоверности информации в новостных текстах с использованием ML и векторного поиска.

## Архитектура

Система состоит из следующих компонентов:

- **Frontend** - веб-интерфейс для ввода текста и отображения результатов
- **Backend API** - основной API сервис для оркестрации пайплайна
- **NLP Service** - извлечение утверждений (claims), NER, предобработка
- **Search Service** - поиск новостей в интернете, парсинг, индексация
- **Verifier Service** - верификация фактов через NLI/Entailment
- **PostgreSQL** - хранение метаданных, запросов, результатов
- **Qdrant** - векторное хранилище для семантического поиска
- **Redis** - очереди задач
- **MinIO** - хранение объектов (HTML, тексты)

## Быстрый старт

### Требования

- Docker и Docker Compose
- Python 3.10+ (для локальной разработки)

### Запуск

**💡 Совет:** Для ускорения сборки включите BuildKit:
```bash
# Windows PowerShell
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

# Linux/Mac
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

#### Вариант 1: Используя Makefile (рекомендуется)

```bash
# Собрать все образы
make build

# Запустить все сервисы
make up

# Просмотр логов
make logs

# Остановить сервисы
make down

# Очистить все (включая данные)
make clean
```

#### Вариант 2: Используя Docker Compose напрямую

```bash
# Собрать и запустить все сервисы
docker-compose up -d --build

# Проверить статус
docker-compose ps

# Просмотр логов
docker-compose logs -f

# Остановить сервисы
docker-compose down

# С удалением данных
docker-compose down -v
```

**Примечание:** 
- Сборка Docker образов должна занимать **15-20 минут** (оптимизировано с ~30+ минут)
- При первом запуске загрузка ML моделей может занять 2-5 минут (модели загружаются при первом запросе, а не при сборке)
- Если сборка занимает более 30 минут, проверьте интернет-соединение и логи: `docker-compose logs`
- Подробнее об оптимизациях: см. [BUILD_OPTIMIZATION.md](BUILD_OPTIMIZATION.md)

Сервисы будут доступны по адресам:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API документация: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

### Остановка

```bash
# Используя Makefile
make down

# Или напрямую
docker-compose down

# С удалением всех данных
docker-compose down -v
```

## API

### POST /verify

Проверка достоверности текста.

**Request:**
```json
{
  "text": "Текст новости для проверки..."
}
```

**Response:**
```json
{
  "status": "PARTIALLY_CONFIRMED",
  "confidence": 0.72,
  "claims": [
    {
      "claim": "Утверждение из текста",
      "label": "ENTAILS",
      "evidence": [
        {
          "url": "https://...",
          "title": "Заголовок статьи",
          "date": "2025-12-26",
          "snippet": "Фрагмент подтверждающего текста"
        }
      ]
    }
  ],
  "sources": [
    {
      "url": "https://...",
      "domain": "trusted-source.ru",
      "date": "2025-12-26",
      "trust_level": 0.9
    }
  ],
  "warnings": [
    "Недостаточно данных по части утверждений"
  ]
}
```

## Статусы результата

- `CONFIRMED` - большинство утверждений подтверждено доверенными источниками
- `REFUTED` - есть противоречащие доверенные источники
- `PARTIALLY_CONFIRMED` - часть утверждений подтверждена, часть неизвестна
- `INSUFFICIENT_DATA` - недостаточно данных для проверки

## Конфигурация

### Доверенные источники

Список доверенных доменов настраивается в файлах:
- `backend/config/trusted_domains.json` - для backend
- `services/search_service/config/trusted_domains.json` - для search service

Формат файла:
```json
{
  "domains": ["ria.ru", "tass.ru", ...],
  "domain_weights": {
    "ria.ru": 1.0,
    "tass.ru": 1.0,
    ...
  }
}
```

### Модели

Система использует предобученные модели для русского языка:
- **paraphrase-multilingual-MiniLM-L12-v2** - для генерации эмбеддингов (sentence-transformers)
- **ru_core_news_sm** - spaCy модель для NER и обработки русского текста
- Семантическое сходство для приблизительной NLI верификации

**Примечание:** В продакшене рекомендуется использовать специализированные NLI модели для более точной верификации.

### Переменные окружения

Основные переменные окружения (настраиваются в `docker-compose.yml`):
- `DATABASE_URL` - URL подключения к PostgreSQL
- `QDRANT_URL` - URL Qdrant сервера
- `REDIS_URL` - URL Redis сервера
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` - настройки MinIO

## Разработка

### Структура проекта

```
fake_detector_mvp/
├── backend/                    # Backend API
│   ├── main.py                 # Основной файл приложения
│   ├── database.py              # Модели БД
│   ├── models.py                # Pydantic модели
│   ├── alembic/                 # Миграции БД
│   └── config/                  # Конфигурационные файлы
├── frontend/                    # Frontend приложение (React)
│   ├── src/
│   │   ├── App.js              # Главный компонент
│   │   └── components/         # React компоненты
│   └── public/
├── services/
│   ├── nlp_service/            # NLP сервис (claim extraction, NER)
│   ├── search_service/         # Сервис поиска и парсинга
│   └── verifier_service/       # Сервис верификации (NLI)
├── docker-compose.yml           # Docker Compose конфигурация
├── Makefile                     # Удобные команды
└── README.md
```

### Локальная разработка

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Запуск с hot-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm start
```

#### Запуск зависимостей (БД, Redis, Qdrant)

Для локальной разработки можно запустить только инфраструктурные сервисы:

```bash
docker-compose up -d postgres redis qdrant minio
```

### Миграции БД

```bash
# Создать новую миграцию
docker-compose exec backend alembic revision --autogenerate -m "описание изменений"

# Применить миграции
docker-compose exec backend alembic upgrade head

# Или используя Makefile
make migrate
```

## Известные ограничения MVP

1. **Поиск в интернете**: В текущей версии поиск новых статей в интернете не реализован. Система работает только с уже проиндексированными статьями в Qdrant.

2. **NLI модель**: Используется упрощенная верификация на основе семантического сходства. Для продакшена рекомендуется использовать специализированные NLI модели.

3. **Парсинг**: Парсинг статей работает только для статей, которые уже есть в системе или переданы напрямую.

## Планы развития

- [ ] Интеграция с поисковыми API (Google Custom Search, Bing, Yandex)
- [ ] Использование специализированных NLI моделей для русского языка
- [ ] Улучшение claim extraction с использованием более продвинутых моделей
- [ ] Кэширование результатов проверки
- [ ] Асинхронная обработка через Celery/RQ
- [ ] Метрики и мониторинг
- [ ] Аутентификация и авторизация пользователей

## Лицензия

MIT

