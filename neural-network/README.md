# 🧠 Neural Network Chat Application

Полнофункциональная нейросеть с трансформерной архитектурой, BPE токенизацией и веб-интерфейсом для диалогов.

## 🚀 Демо

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## ✨ Особенности

- 🧠 **Настоящая нейросеть** - трансформерная архитектура с 88M параметров
- 🔤 **BPE токенизация** - динамическое разбиение текста на подслова
- 🌍 **Многоязычность** - поддержка русского и английского языков
- ⚡ **Авторегрессивная генерация** - токен за токеном
- 🎯 **Вероятностный sampling** - temperature, top-k, top-p
- 🐳 **Docker контейнеризация** - легкий деплой
- 📊 **Мониторинг** - Prometheus + Grafana
- 🔒 **Безопасность** - SSL, rate limiting, CORS

## 🏗️ Архитектура

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Nginx    │────│ Go Gateway  │────│Python Neural│
│  (Port 80)  │    │ (Port 8090) │    │ (Port 8080) │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Static Files│    │ Redis Cache │    │ PostgreSQL  │
│   (Web UI)  │    │ (Port 6379) │    │ (Port 5432) │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 🛠️ Технологии

### Backend
- **Python** - нейросеть (PyTorch, Transformers)
- **Go** - API Gateway (Gin, HTTP)
- **Rust** - высокопроизводительные операции
- **C++/CUDA** - ускорение матричных операций

### Frontend
- **HTML/CSS/JavaScript** - веб-интерфейс
- **TypeScript** - типизация
- **Vite** - сборка

### Infrastructure
- **Docker** - контейнеризация
- **Nginx** - веб-сервер
- **Redis** - кеширование
- **PostgreSQL** - база данных
- **Prometheus** - метрики
- **Grafana** - дашборды

## 🚀 Быстрый старт

### Локальный запуск

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/neural-network-chat.git
cd neural-network-chat

# Запустить все сервисы
docker-compose up -d

# Открыть в браузере
open http://localhost:8090
```

### Деплой на Render

1. **Fork репозиторий** на GitHub
2. **Подключить к Render**:
   - Создать новый Web Service
   - Подключить GitHub репозиторий
   - Выбрать `docker-compose.prod.yml`
3. **Настроить переменные окружения**:
   ```
   DOMAIN=your-app.onrender.com
   EMAIL=admin@your-domain.com
   ```
4. **Задеплоить** - Render автоматически соберет и запустит

## 📁 Структура проекта

```
neural-network/
├── 🐍 main.py                 # Python нейросеть
├── 🐍 api_server.py           # FastAPI сервер
├── 🔧 go-service/             # Go API Gateway
├── 🦀 rust-service/           # Rust сервис
├── 🌐 web-interface/          # Веб-интерфейс
├── 🐳 docker/                 # Docker конфигурации
├── 📊 nginx/                  # Nginx конфигурация
├── 📈 monitoring/             # Prometheus/Grafana
├── 🗄️ sql/                   # База данных
├── 📋 docker-compose.yml      # Локальная разработка
├── 🚀 docker-compose.prod.yml # Продакшен
├── 🔧 deploy.sh              # Скрипт деплоя (Linux/macOS)
├── 🔧 deploy.ps1             # Скрипт деплоя (Windows)
└── 📖 DEPLOYMENT.md          # Подробная инструкция
```

## 🧠 Нейросеть

### Архитектура
- **Тип**: Transformer
- **Параметры**: 88,886,784
- **Слои**: 12 transformer блоков
- **Внимание**: Multi-head self-attention
- **Эмбеддинги**: 768 измерений

### Токенизация
- **Тип**: Byte Pair Encoding (BPE)
- **Словарь**: 4000 токенов
- **Языки**: Русский, английский
- **Специальные токены**: `<|startoftext|>`, `<|endoftext|>`

### Генерация
- **Метод**: Авторегрессивная (токен за токеном)
- **Sampling**: Temperature, top-k, top-p
- **Повторы**: Repetition penalty
- **Контекст**: До 1024 токенов

## 🌐 API

### Эндпоинты

| Метод | URL | Описание |
|-------|-----|----------|
| `GET` | `/api/health` | Статус сервисов |
| `POST` | `/api/v1/generate` | Генерация текста |
| `GET` | `/metrics` | Метрики Prometheus |

### Пример запроса

```bash
curl -X POST http://localhost:8090/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Привет, как дела?",
    "max_tokens": 50,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1
  }'
```

### Пример ответа

```json
{
  "text": "Привет! У меня все хорошо, спасибо за вопрос. Как дела у тебя?",
  "tokens_generated": 15,
  "generation_time_ms": 1234,
  "model_used": "transformer-neural-network",
  "device": "CPU"
}
```

## 📊 Мониторинг

### Grafana Dashboard
- **URL**: `http://your-app.onrender.com/grafana/`
- **Логин**: `admin`
- **Пароль**: `admin123`

### Ключевые метрики
- Время ответа API
- Количество запросов
- Использование памяти
- Загрузка CPU
- Статус сервисов

## 🔧 Разработка

### Требования
- Docker & Docker Compose
- Python 3.10+
- Go 1.19+
- Rust 1.70+
- Node.js 18+

### Локальная разработка

```bash
# Запуск всех сервисов
docker-compose up -d

# Только Python нейросеть
docker-compose up neural-network-python

# Только Go API
docker-compose up neural-network-go

# Просмотр логов
docker-compose logs -f neural-network-python
```

### Тестирование

```bash
# Тест API
curl http://localhost:8090/api/health

# Тест нейросети
curl -X POST http://localhost:8090/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

## 🤝 Вклад в проект

1. **Fork** репозиторий
2. **Создать** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** изменения (`git commit -m 'Add amazing feature'`)
4. **Push** в branch (`git push origin feature/amazing-feature`)
5. **Создать** Pull Request

## 📝 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## 🙏 Благодарности

- [PyTorch](https://pytorch.org/) - фреймворк для машинного обучения
- [Transformers](https://huggingface.co/transformers/) - архитектура нейросетей
- [Docker](https://www.docker.com/) - контейнеризация
- [Render](https://render.com/) - хостинг

## 📞 Поддержка

- 🐛 **Баг-репорты**: [GitHub Issues](https://github.com/your-username/neural-network-chat/issues)
- 💬 **Обсуждения**: [GitHub Discussions](https://github.com/your-username/neural-network-chat/discussions)
- 📧 **Email**: admin@example.com

---

⭐ **Поставьте звезду, если проект понравился!**