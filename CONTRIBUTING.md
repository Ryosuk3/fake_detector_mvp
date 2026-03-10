# Руководство по разработке

## Настройка окружения разработки

### Предварительные требования

- Docker и Docker Compose
- Python 3.10+
- Node.js 18+ (для frontend)

### Первый запуск

1. Клонируйте репозиторий
2. Запустите инфраструктурные сервисы:
   ```bash
   docker-compose up -d postgres redis qdrant minio
   ```
3. Настройте локальное окружение для backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Примените миграции:
   ```bash
   alembic upgrade head
   ```

## Структура кода

### Backend

- `main.py` - основной файл FastAPI приложения
- `database.py` - модели SQLAlchemy и настройка БД
- `models.py` - Pydantic модели для валидации
- `tasks.py` - Celery задачи (для будущей реализации)

### Сервисы

Каждый сервис является независимым FastAPI приложением:
- `nlp_service` - обработка текста, извлечение claims и NER
- `search_service` - поиск и парсинг статей, индексация в Qdrant
- `verifier_service` - верификация фактов через NLI

## Тестирование

### Запуск тестов

```bash
# Backend тесты
cd backend
pytest

# Frontend тесты
cd frontend
npm test
```

### Тестирование API

Используйте Swagger UI: http://localhost:8000/docs

## Стиль кода

- Python: следуйте PEP 8
- JavaScript: используйте ESLint конфигурацию из package.json
- Используйте type hints в Python коде
- Документируйте функции и классы

## Pull Request процесс

1. Создайте ветку от `main`
2. Внесите изменения
3. Убедитесь, что все тесты проходят
4. Создайте Pull Request с описанием изменений

