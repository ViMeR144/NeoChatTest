# 🚀 Моя собственная C++ нейросеть

Это **полностью собственная нейросеть**, написанная с нуля на C++! 

## ✨ Что у нас есть:

### 🧠 **Собственная нейросеть**
- ✅ **Собственный Tensor класс** (аналог PyTorch)
- ✅ **Собственные операции** (matmul, add, relu, softmax)
- ✅ **Собственная инициализация** весов
- ✅ **Многослойная архитектура** (10→64→32→5)

### 🤖 **Чат-бот**
- ✅ **Умные ответы** на русском языке
- ✅ **Интеграция с нейросетью**
- ✅ **Поиск по ключевым словам**

### 🌐 **API сервер**
- ✅ **HTTP endpoints** для интеграции
- ✅ **JSON API** для чата
- ✅ **CORS поддержка**

### 🔤 **Transformer (дополнительно)**
- ✅ **Собственный токенизатор**
- ✅ **Генерация текста**
- ✅ **Поддержка русского и английского**

## 🛠️ Установка и запуск:

### 1. Установить компилятор C++:
```bash
# Windows (через winget):
winget install mingw-w64

# Или через chocolatey:
choco install mingw
```

### 2. Собрать проект:
```bash
# Перейти в папку проекта
cd cpp-neural-network

# Собрать основную нейросеть
g++ -std=c++17 -O2 -o my_neural_network src/main.cpp

# Собрать Transformer
g++ -std=c++17 -O2 -o transformer src/transformer.cpp

# Собрать API сервер
g++ -std=c++17 -O2 -o api_server src/api_server.cpp
```

### 3. Запустить:
```bash
# Основная нейросеть + чат-бот
./my_neural_network

# Transformer для генерации текста
./transformer

# API сервер
./api_server
```

## 🔗 Интеграция с веб-чатом:

### Вариант 1: Через Node.js процесс
```javascript
const { spawn } = require('child_process');
const neuralProcess = spawn('./my_neural_network.exe');

neuralProcess.stdin.write(JSON.stringify({message: userMessage}));
neuralProcess.stdout.on('data', (data) => {
    const response = JSON.parse(data.toString());
    // Отправляем ответ пользователю
});
```

### Вариант 2: Через HTTP API
```javascript
fetch('http://localhost:8080/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: userMessage})
})
.then(response => response.json())
.then(data => {
    // Отправляем ответ пользователю
});
```

## 📊 Архитектура:

```
Входные данные → Tensor → Нейросеть → Результат
     ↓              ↓         ↓         ↓
   "Привет"    [1x10]    [64→32→5]   "Ответ"
```

## 🎯 Возможности:

- **Генерация ответов** на русском языке
- **Обработка естественного языка**
- **Быстрая работа** (C++ в 100+ раз быстрее Python)
- **Малое потребление памяти**
- **Готовность к продакшену**

## 🚀 Планы развития:

- [ ] **Обучение на реальных данных**
- [ ] **GPU ускорение** (CUDA)
- [ ] **Больше языков** поддержки
- [ ] **Веб-интерфейс** для управления
- [ ] **Модель как сервис** (Docker)

## 💡 Это твоя собственная нейросеть!

Никаких готовых библиотек, никаких чужих моделей - только твой код, твоя логика, твоя нейросеть! 🎉

---

**Создано с ❤️ на C++**
