# 🚀 Деплой на Render

## Быстрый деплой

### 1. Подготовка GitHub

```bash
# Инициализация Git (если еще не сделано)
git init
git add .
git commit -m "Initial commit"

# Создание репозитория на GitHub
# 1. Идите на https://github.com/new
# 2. Создайте репозиторий с именем "neural-network-chat"
# 3. Не добавляйте README, .gitignore или лицензию (уже есть)

# Подключение к GitHub
git remote add origin https://github.com/YOUR_USERNAME/neural-network-chat.git
git branch -M main
git push -u origin main
```

### 2. Деплой на Render

#### Вариант A: Автоматический деплой (рекомендуется)

1. **Идите на [Render.com](https://render.com)**
2. **Нажмите "New +" → "Web Service"**
3. **Подключите GitHub репозиторий**
4. **Настройте сервис**:
   ```
   Name: neural-network-chat
   Environment: Docker
   Dockerfile Path: ./Dockerfile.web
   Docker Context: .
   Branch: main
   Region: Oregon (US West)
   Plan: Starter (Free)
   ```

5. **Добавьте переменные окружения**:
   ```
   DOMAIN = neural-network-chat.onrender.com
   EMAIL = admin@neural-network-chat.onrender.com
   PYTHONUNBUFFERED = 1
   GIN_MODE = release
   ```

6. **Нажмите "Create Web Service"**

#### Вариант B: Используя render.yaml

1. **Идите на [Render.com](https://render.com)**
2. **Нажмите "New +" → "Blueprint"**
3. **Подключите GitHub репозиторий**
4. **Render автоматически найдет render.yaml**
5. **Нажмите "Apply"**

### 3. Настройка базы данных

1. **Создайте PostgreSQL**:
   - New + → PostgreSQL
   - Name: neural-network-postgres
   - Plan: Starter (Free)
   - Region: Oregon (US West)

2. **Создайте Redis**:
   - New + → Redis
   - Name: neural-network-redis
   - Plan: Starter (Free)
   - Region: Oregon (US West)

3. **Подключите к Web Service**:
   - В настройках Web Service
   - Environment → Add Environment Variable
   ```
   DATABASE_URL = <PostgreSQL connection string>
   REDIS_URL = <Redis connection string>
   ```

## 🔧 Конфигурация

### Переменные окружения

```bash
# Основные
DOMAIN=neural-network-chat.onrender.com
EMAIL=admin@neural-network-chat.onrender.com
PORT=8080

# Python
PYTHONUNBUFFERED=1
PYTHONPATH=/app

# Go
GIN_MODE=release

# Базы данных
DATABASE_URL=postgresql://user:password@host:port/database
REDIS_URL=redis://host:port

# Нейросеть
MODEL_PATH=/app/models
LOG_LEVEL=INFO
```

### Лимиты Render (Free план)

- **RAM**: 512 MB
- **CPU**: 0.1 CPU
- **Disk**: 1 GB
- **Bandwidth**: 100 GB/месяц
- **Sleep**: После 15 минут неактивности

## 📊 Мониторинг

### Render Dashboard
- **URL**: https://dashboard.render.com
- **Логи**: Real-time logs в Dashboard
- **Метрики**: CPU, Memory, Response time

### Health Check
- **URL**: `https://your-app.onrender.com/api/health`
- **Проверка**: Каждые 30 секунд

## 🚀 Оптимизация для Render

### 1. Уменьшение размера образа

```dockerfile
# Используйте multi-stage build
FROM python:3.10-slim as builder
# ... build stage ...

FROM python:3.10-slim as production
# ... production stage ...
```

### 2. Оптимизация Python

```python
# В main.py
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Ограничить потоки
os.environ['MKL_NUM_THREADS'] = '1'
```

### 3. Кеширование

```python
# Используйте Redis для кеша
import redis
redis_client = redis.from_url(os.getenv('REDIS_URL'))
```

## 🔧 Troubleshooting

### Проблемы с памятью

```bash
# Уменьшите размер модели
model_config = ModelConfig(
    vocab_size=2000,  # Вместо 4000
    n_embd=384,       # Вместо 768
    n_head=6,         # Вместо 12
    n_layer=6,        # Вместо 12
)
```

### Медленный старт

```python
# Добавьте в api_server.py
@app.on_event("startup")
async def startup_event():
    # Предзагрузите модель
    model.eval()
    # Сделайте тестовый запрос
    test_input = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        _ = model(test_input)
```

### Таймауты

```python
# Увеличьте таймауты в Render
# Settings → Environment → Add
REQUEST_TIMEOUT = 60
GENERATION_TIMEOUT = 30
```

## 📈 Масштабирование

### Upgrade до Paid плана

1. **Starter** ($7/месяц):
   - 512 MB RAM
   - 0.1 CPU
   - Нет sleep

2. **Standard** ($25/месяц):
   - 2 GB RAM
   - 0.5 CPU
   - Автоскейлинг

3. **Pro** ($85/месяц):
   - 8 GB RAM
   - 2 CPU
   - Выделенные ресурсы

### Горизонтальное масштабирование

```yaml
# В render.yaml
services:
  - type: web
    name: neural-network-web-1
    # ... конфигурация ...
  - type: web
    name: neural-network-web-2
    # ... конфигурация ...
```

## 🔒 Безопасность

### SSL/TLS
- **Автоматически** включен на Render
- **HTTPS** всегда доступен
- **HTTP** автоматически редиректится на HTTPS

### CORS
```python
# В api_server.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📝 Полезные команды

### Локальное тестирование

```bash
# Тест с Render переменными
export DATABASE_URL="postgresql://..."
export REDIS_URL="redis://..."
python api_server.py
```

### Проверка деплоя

```bash
# Health check
curl https://your-app.onrender.com/api/health

# Тест генерации
curl -X POST https://your-app.onrender.com/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

### Логи

```bash
# В Render Dashboard
# Services → Your Service → Logs
# Или используйте Render CLI
render logs --service your-service-name
```

## 🎯 Результат

После успешного деплоя у вас будет:

- ✅ **Живой сайт**: `https://your-app.onrender.com`
- ✅ **API**: `https://your-app.onrender.com/api/v1/generate`
- ✅ **Мониторинг**: Render Dashboard
- ✅ **Автодеплой**: При каждом push в GitHub
- ✅ **SSL**: Автоматические сертификаты
- ✅ **Масштабирование**: По требованию

**Поздравляем! Ваша нейросеть работает в облаке!** 🎉
