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

## 🚀 Быстрый деплой на Render

### Автоматический деплой

1. **Fork этот репозиторий** на GitHub
2. **Идите на [Render.com](https://render.com)**
3. **New + → Blueprint**
4. **Подключите GitHub репозиторий**
5. **Render автоматически найдет `render.yaml`**
6. **Нажмите "Apply"**

### Ручной деплой

1. **New + → Web Service**
2. **Подключите GitHub репозиторий**
3. **Настройки**:
   ```
   Name: neural-network-chat
   Environment: Docker
   Dockerfile Path: ./neural-network/Dockerfile.web
   Docker Context: ./neural-network
   Branch: main
   Plan: Starter (Free)
   ```
4. **Переменные окружения**:
   ```
   DOMAIN = neural-network-chat.onrender.com
   EMAIL = admin@neural-network-chat.onrender.com
   PYTHONUNBUFFERED = 1
   GIN_MODE = release
   ```

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

## 📁 Структура проекта

```
├── 🐍 neural-network/         # Основная папка с нейросетью
│   ├── main.py               # Python нейросеть
│   ├── api_server.py         # FastAPI сервер
│   ├── go-service/           # Go API Gateway
│   ├── rust-service/         # Rust сервис
│   ├── web-interface/        # Веб-интерфейс
│   ├── docker/               # Docker конфигурации
│   ├── nginx/                # Nginx конфигурация
│   ├── monitoring/           # Prometheus/Grafana
│   ├── sql/                  # База данных
│   ├── Dockerfile.web        # Docker для Render Web Service
│   ├── Dockerfile.worker     # Docker для Render Worker
│   └── requirements.txt      # Python зависимости
├── 📋 render.yaml            # Конфигурация Render (Blueprint)
├── 📖 README.md              # Этот файл
└── 🔧 .gitignore             # Git игнорирование
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
curl -X POST https://your-app.onrender.com/api/v1/generate \
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

## 🔧 Локальная разработка

### Требования
- Docker & Docker Compose
- Python 3.10+
- Go 1.19+
- Rust 1.70+

### Запуск

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/neural-network-chat.git
cd neural-network-chat

# Запустить все сервисы
cd neural-network
docker-compose up -d

# Открыть в браузере
open http://localhost:8090
```

## 🤝 Вклад в проект

1. **Fork** репозиторий
2. **Создать** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** изменения (`git commit -m 'Add amazing feature'`)
4. **Push** в branch (`git push origin feature/amazing-feature`)
5. **Создать** Pull Request

## 📝 Лицензия

Этот проект лицензирован под MIT License.

## 🙏 Благодарности

- [PyTorch](https://pytorch.org/) - фреймворк для машинного обучения
- [Transformers](https://huggingface.co/transformers/) - архитектура нейросетей
- [Docker](https://www.docker.com/) - контейнеризация
- [Render](https://render.com/) - хостинг

---

⭐ **Поставьте звезду, если проект понравился!**
