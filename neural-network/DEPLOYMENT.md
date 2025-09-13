# 🚀 Neural Network Deployment Guide

## Быстрый старт

### Windows (PowerShell)
```powershell
# Клонируйте репозиторий
git clone <your-repo-url>
cd neural-network

# Запустите деплой
.\deploy.ps1

# Или с кастомным доменом
.\deploy.ps1 -Domain "your-domain.com" -Email "admin@your-domain.com"
```

### Linux/macOS (Bash)
```bash
# Клонируйте репозиторий
git clone <your-repo-url>
cd neural-network

# Сделайте скрипт исполняемым
chmod +x deploy.sh

# Запустите деплой
./deploy.sh

# Или с кастомным доменом
./deploy.sh your-domain.com admin@your-domain.com
```

## 🏗️ Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │────│   Go Gateway    │────│ Python Neural   │
│   (Port 80/443) │    │   (Port 8090)   │    │   (Port 8080)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Static Files  │    │   Redis Cache   │    │  PostgreSQL DB  │
│   (Web UI)      │    │   (Port 6379)   │    │   (Port 5432)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Компоненты

### 1. **Nginx** - Веб-сервер и прокси
- Обслуживает статические файлы (HTML, CSS, JS)
- Проксирует API запросы к Go сервису
- SSL терминация
- Rate limiting
- CORS настройки

### 2. **Go API Gateway** - Маршрутизация
- Принимает HTTP запросы
- Проксирует к Python нейросети
- Обработка ошибок
- Логирование

### 3. **Python Neural Network** - ИИ модель
- Трансформерная архитектура
- BPE токенизация
- Авторегрессивная генерация
- CUDA поддержка (если доступна)

### 4. **Redis** - Кеширование
- Кеш для быстрого доступа
- Сессии пользователей
- Временные данные

### 5. **PostgreSQL** - База данных
- История диалогов
- Метаданные модели
- Конфигурация

### 6. **Prometheus + Grafana** - Мониторинг
- Метрики производительности
- Дашборды
- Алерты

## 🌐 Доступные эндпоинты

| Сервис | URL | Описание |
|--------|-----|----------|
| **Веб-интерфейс** | `http://localhost/` | Чат с нейросетью |
| **API Health** | `http://localhost/api/health` | Статус сервисов |
| **Chat API** | `http://localhost/api/v1/generate` | Генерация текста |
| **Grafana** | `http://localhost/grafana/` | Мониторинг (admin/admin123) |
| **Prometheus** | `http://localhost/prometheus/` | Метрики |

## 📊 Мониторинг

### Grafana Dashboard
- **URL**: `http://localhost/grafana/`
- **Логин**: `admin`
- **Пароль**: `admin123`

### Ключевые метрики:
- Время ответа API
- Количество запросов
- Использование памяти
- Загрузка CPU
- Статус сервисов

## 🔐 Безопасность

### SSL/TLS
- Автоматическая генерация самоподписанных сертификатов
- Для продакшена используйте Let's Encrypt

### Rate Limiting
- API: 10 запросов/сек
- Chat: 5 запросов/сек

### CORS
- Настроен для всех доменов
- Настраивается в nginx.conf

## 🚀 Масштабирование

### Горизонтальное масштабирование
```bash
# Увеличить количество Python сервисов
docker-compose -f docker-compose.prod.yml up -d --scale neural-network-python=3

# Увеличить количество Go сервисов
docker-compose -f docker-compose.prod.yml up -d --scale neural-network-go=2
```

### Вертикальное масштабирование
```yaml
# В docker-compose.prod.yml
services:
  neural-network-python:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## 🛠️ Управление

### Полезные команды

```bash
# Просмотр логов
docker-compose -f docker-compose.prod.yml logs -f

# Остановка всех сервисов
docker-compose -f docker-compose.prod.yml down

# Перезапуск сервиса
docker-compose -f docker-compose.prod.yml restart neural-network-python

# Обновление образа
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# Просмотр статуса
docker-compose -f docker-compose.prod.yml ps

# Выполнение команд в контейнере
docker-compose -f docker-compose.prod.yml exec neural-network-python python --version
```

## 🔧 Настройка

### Переменные окружения
```bash
# Go сервис
GIN_MODE=release
PORT=8090
PYTHON_API_URL=http://neural-network-python:8080

# Python сервис
PYTHONUNBUFFERED=1
PYTHONPATH=/app
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://neural:password@postgres:5432/neural_db

# Grafana
GF_SECURITY_ADMIN_PASSWORD=admin123
```

### Конфигурация Nginx
- Файл: `nginx/nginx.conf`
- SSL сертификаты: `nginx/ssl/`
- Логи: `logs/`

## 🐛 Отладка

### Проверка здоровья сервисов
```bash
# Проверка всех сервисов
curl http://localhost/api/health

# Проверка конкретного сервиса
curl http://localhost:8090/health  # Go
curl http://localhost:8080/health  # Python
```

### Логи
```bash
# Все логи
docker-compose -f docker-compose.prod.yml logs

# Конкретный сервис
docker-compose -f docker-compose.prod.yml logs neural-network-python

# Следить за логами в реальном времени
docker-compose -f docker-compose.prod.yml logs -f neural-network-python
```

### Мониторинг ресурсов
```bash
# Использование ресурсов
docker stats

# Информация о контейнерах
docker-compose -f docker-compose.prod.yml ps
```

## 📝 Обновление

### Обновление кода
```bash
# 1. Остановить сервисы
docker-compose -f docker-compose.prod.yml down

# 2. Обновить код
git pull

# 3. Пересобрать и запустить
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d
```

### Обновление зависимостей
```bash
# Обновить Python пакеты
docker-compose -f docker-compose.prod.yml exec neural-network-python pip install --upgrade -r requirements.txt

# Обновить Go модули
docker-compose -f docker-compose.prod.yml exec neural-network-go go mod tidy
```

## 🆘 Поддержка

### Частые проблемы

1. **Порт занят**
   ```bash
   # Проверить какие процессы используют порт
   netstat -tulpn | grep :80
   
   # Остановить процесс
   sudo kill -9 <PID>
   ```

2. **Нехватка памяти**
   ```bash
   # Увеличить лимиты Docker
   # В Docker Desktop: Settings -> Resources -> Memory
   ```

3. **SSL ошибки**
   ```bash
   # Перегенерировать сертификаты
   rm nginx/ssl/*
   ./deploy.sh
   ```

### Контакты
- GitHub Issues: [Создать issue]
- Email: admin@example.com
- Telegram: @your_username
