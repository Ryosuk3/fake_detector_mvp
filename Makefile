.PHONY: help build up down restart logs clean migrate

help:
	@echo "Доступные команды:"
	@echo "  make build      - Собрать все Docker образы"
	@echo "  make up         - Запустить все сервисы"
	@echo "  make down       - Остановить все сервисы"
	@echo "  make restart   - Перезапустить все сервисы"
	@echo "  make logs       - Показать логи всех сервисов"
	@echo "  make clean      - Остановить и удалить все контейнеры и volumes"
	@echo "  make migrate    - Запустить миграции БД"

build:
	@echo "Сборка с BuildKit для ускорения..."
	@export DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1; docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -f

migrate:
	docker-compose exec backend alembic upgrade head

