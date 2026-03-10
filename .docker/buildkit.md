# Использование BuildKit для ускорения сборки

Для ускорения сборки используйте BuildKit:

```bash
# Включить BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Или в PowerShell
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

# Затем собрать
docker-compose build
```

BuildKit использует параллельную сборку и кэширование слоев.

