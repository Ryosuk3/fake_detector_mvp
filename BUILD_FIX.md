# Исправление ошибки сборки

## Проблема

При сборке `search_service` и `verifier_service` возникала ошибка:
```
ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
```

### Причина

1. **Двойная установка torch**: `sentence-transformers` требует `torch>=1.6.0`, но pip не видит уже установленную CPU-only версию и пытается установить полную версию с CUDA (~900MB)

2. **CUDA зависимости**: pip пытается установить ненужные CUDA библиотеки:
   - nvidia-cuda-runtime-cu12 (954 KB)
   - nvidia-cudnn-cu12 (706.8 MB)
   - nvidia-cusparse-cu12 (288.2 MB)
   - nvidia-nvshmem-cu12 (124.7 MB)
   - torchvision (8.0 MB)
   - torch с CUDA (899.8 MB)

3. **Таймаут**: Загрузка 900MB файла прерывается по таймауту сети

## Решение

### Изменения в Dockerfile

1. **Явное указание CPU-only индекса для torch**:
   ```dockerfile
   RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
   ```

2. **Установка зависимостей sentence-transformers вручную**:
   ```dockerfile
   RUN pip install --no-cache-dir transformers sentencepiece scikit-learn
   ```

3. **Установка sentence-transformers без зависимостей**:
   ```dockerfile
   RUN pip install --no-cache-dir --no-deps sentence-transformers==2.2.2
   ```

Это предотвращает:
- Переустановку torch
- Установку CUDA зависимостей
- Таймауты при загрузке больших файлов

## Результат

- ✅ Только CPU-only torch (без CUDA)
- ✅ Нет дублирования установки
- ✅ Быстрее сборка (меньше файлов для загрузки)
- ✅ Меньше вероятность таймаутов

## Если все еще возникают таймауты

1. **Увеличить таймаут pip**:
   ```dockerfile
   RUN pip install --default-timeout=1000 --no-cache-dir ...
   ```

2. **Использовать зеркало PyPI**:
   ```dockerfile
   RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple ...
   ```

3. **Собрать в несколько этапов**:
   - Сначала установить все легкие пакеты
   - Затем тяжелые по одному

