# Makefile для собственной C++ нейросети

# Компилятор
CXX = g++

# Флаги компиляции
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# Папки
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Создаем папки если не существуют
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Основные цели
.PHONY: all clean neural transformer api server help

all: neural transformer api server

# Основная нейросеть + чат-бот
neural: $(BIN_DIR) $(SRC_DIR)/main.cpp
	@echo "🧠 Собираем основную нейросеть..."
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/my_neural_network $(SRC_DIR)/main.cpp
	@echo "✅ Основная нейросеть собрана!"

# Transformer для генерации текста
transformer: $(BIN_DIR) $(SRC_DIR)/transformer.cpp
	@echo "🔤 Собираем Transformer..."
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/transformer $(SRC_DIR)/transformer.cpp
	@echo "✅ Transformer собран!"

# API сервер
api: $(BIN_DIR) $(SRC_DIR)/api_server.cpp
	@echo "🌐 Собираем API сервер..."
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/api_server $(SRC_DIR)/api_server.cpp
	@echo "✅ API сервер собран!"

# Запуск основного сервера
server: neural
	@echo "🚀 Запускаем нейросеть..."
	./$(BIN_DIR)/my_neural_network

# Запуск Transformer
run-transformer: transformer
	@echo "🔤 Запускаем Transformer..."
	./$(BIN_DIR)/transformer

# Запуск API сервера
run-api: api
	@echo "🌐 Запускаем API сервер..."
	./$(BIN_DIR)/api_server

# Тестирование всех компонентов
test: all
	@echo "🧪 Тестируем все компоненты..."
	@echo "\n=== Тест основной нейросети ==="
	./$(BIN_DIR)/my_neural_network
	@echo "\n=== Тест Transformer ==="
	./$(BIN_DIR)/transformer
	@echo "\n=== Тест API сервера ==="
	./$(BIN_DIR)/api_server

# Очистка
clean:
	@echo "🧹 Очищаем проект..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "✅ Проект очищен!"

# Помощь
help:
	@echo "🚀 Собственная C++ нейросеть - Makefile"
	@echo ""
	@echo "Доступные команды:"
	@echo "  make all          - Собрать все компоненты"
	@echo "  make neural       - Собрать основную нейросеть"
	@echo "  make transformer  - Собрать Transformer"
	@echo "  make api          - Собрать API сервер"
	@echo "  make server       - Запустить основную нейросеть"
	@echo "  make run-transformer - Запустить Transformer"
	@echo "  make run-api      - Запустить API сервер"
	@echo "  make test         - Тестировать все компоненты"
	@echo "  make clean        - Очистить проект"
	@echo "  make help         - Показать эту справку"
	@echo ""
	@echo "Примеры:"
	@echo "  make all && make test    # Собрать и протестировать все"
	@echo "  make neural && make server  # Собрать и запустить нейросеть"

# Установка зависимостей (для Windows)
install-deps:
	@echo "📦 Установка зависимостей..."
	@echo "Для Windows:"
	@echo "  winget install mingw-w64"
	@echo "Или через chocolatey:"
	@echo "  choco install mingw"
	@echo "Для Linux:"
	@echo "  sudo apt-get install g++"
	@echo "Для macOS:"
	@echo "  brew install gcc"

# Проверка компилятора
check-compiler:
	@echo "🔍 Проверяем компилятор..."
	@$(CXX) --version || echo "❌ Компилятор не найден! Выполните: make install-deps"



