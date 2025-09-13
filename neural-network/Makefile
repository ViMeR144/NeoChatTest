# Advanced Neural Network - Build System
# Multi-language neural network with Python, C++, CUDA, Rust, Go, and TypeScript

.PHONY: all clean install build test run help

# Default target
all: install build

# Installation targets
install: install-python install-rust install-go install-node

install-python:
	@echo "🐍 Installing Python dependencies..."
	pip install -r requirements.txt

install-rust:
	@echo "🦀 Installing Rust dependencies..."
	cd rust-service && cargo build

install-go:
	@echo "🐹 Installing Go dependencies..."
	cd go-service && go mod tidy

install-node:
	@echo "📦 Installing Node.js dependencies..."
	cd web-interface && npm install

# Build targets
build: build-cuda build-python build-rust build-go build-web

build-cuda:
	@echo "🚀 Building CUDA kernels..."
	nvcc -o cuda_kernels.so --shared cuda_kernels.cu -lcublas -lcudnn

build-python:
	@echo "🐍 Building Python neural network..."
	python -m py_compile main.py

build-rust:
	@echo "🦀 Building Rust service..."
	cd rust-service && cargo build --release

build-go:
	@echo "🐹 Building Go service..."
	cd go-service && go build -o neural-network-go .

build-web:
	@echo "🌐 Building web interface..."
	cd web-interface && npm run build

# Test targets
test: test-python test-rust test-go test-web

test-python:
	@echo "🧪 Testing Python neural network..."
	python -m pytest tests/ -v

test-rust:
	@echo "🧪 Testing Rust service..."
	cd rust-service && cargo test

test-go:
	@echo "🧪 Testing Go service..."
	cd go-service && go test ./...

test-web:
	@echo "🧪 Testing web interface..."
	cd web-interface && npm test

# Run targets
run: run-all

run-python:
	@echo "🐍 Starting Python neural network..."
	python main.py

run-rust:
	@echo "🦀 Starting Rust service..."
	cd rust-service && cargo run

run-go:
	@echo "🐹 Starting Go service..."
	cd go-service && ./neural-network-go

run-web:
	@echo "🌐 Starting web interface..."
	cd web-interface && npm run dev

run-all:
	@echo "🚀 Starting all services..."
	@echo "Starting Rust service on port 8080..."
	cd rust-service && cargo run &
	@echo "Starting Go service on port 8090..."
	cd go-service && go run . &
	@echo "Starting Python neural network..."
	python main.py &
	@echo "Starting web interface on port 3000..."
	cd web-interface && npm run dev &
	@echo "All services started! Check the logs above for connection details."

# Development targets
dev: dev-setup dev-run

dev-setup:
	@echo "🛠️  Setting up development environment..."
	@echo "Installing development tools..."
	pip install -r requirements.txt
	cd rust-service && cargo build
	cd go-service && go mod tidy
	cd web-interface && npm install
	@echo "Development environment ready!"

dev-run:
	@echo "🚀 Starting development servers..."
	@echo "Rust service: http://localhost:8080"
	@echo "Go service: http://localhost:8090"
	@echo "Web interface: http://localhost:3000"
	@echo "Python neural network will run on demand"
	make run-all

# Docker targets
docker: docker-build docker-run

docker-build:
	@echo "🐳 Building Docker containers..."
	docker-compose build

docker-run:
	@echo "🐳 Starting Docker containers..."
	docker-compose up -d

docker-stop:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down

# Performance testing
benchmark: benchmark-python benchmark-rust benchmark-go

benchmark-python:
	@echo "⚡ Benchmarking Python neural network..."
	python -m pytest tests/benchmarks.py -v

benchmark-rust:
	@echo "⚡ Benchmarking Rust service..."
	cd rust-service && cargo bench

benchmark-go:
	@echo "⚡ Benchmarking Go service..."
	cd go-service && go test -bench=. -benchmem

# Monitoring and logging
logs:
	@echo "📊 Showing service logs..."
	docker-compose logs -f

monitor:
	@echo "📈 Starting monitoring dashboard..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3001"
	docker-compose -f docker-compose.monitoring.yml up -d

# Security and quality
security: security-audit security-scan

security-audit:
	@echo "🔒 Running security audit..."
	npm audit
	pip check
	cargo audit
	go list -json -deps ./... | nancy sleuth

security-scan:
	@echo "🔍 Running security scan..."
	bandit -r . -f json -o security-report.json
	semgrep --config=auto .

# Code quality
quality: lint format type-check

lint:
	@echo "🔍 Running linters..."
	cd web-interface && npm run lint
	cd rust-service && cargo clippy
	cd go-service && golangci-lint run
	flake8 .
	black --check .

format:
	@echo "🎨 Formatting code..."
	cd web-interface && npm run lint:fix
	cd rust-service && cargo fmt
	cd go-service && go fmt ./...
	black .
	isort .

type-check:
	@echo "🔍 Type checking..."
	cd web-interface && npm run type-check
	mypy .

# Documentation
docs: docs-build docs-serve

docs-build:
	@echo "📚 Building documentation..."
	pdoc --html . --output-dir docs
	cd web-interface && npm run build

docs-serve:
	@echo "📚 Serving documentation..."
	cd docs && python -m http.server 8080

# Cleanup
clean: clean-python clean-rust clean-go clean-web clean-docker

clean-python:
	@echo "🧹 Cleaning Python artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

clean-rust:
	@echo "🧹 Cleaning Rust artifacts..."
	cd rust-service && cargo clean

clean-go:
	@echo "🧹 Cleaning Go artifacts..."
	cd go-service && go clean

clean-web:
	@echo "🧹 Cleaning web artifacts..."
	cd web-interface && rm -rf node_modules dist

clean-docker:
	@echo "🧹 Cleaning Docker artifacts..."
	docker-compose down -v
	docker system prune -f

# Help
help:
	@echo "🧠 Advanced Neural Network - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install all dependencies"
	@echo "  build       - Build all components"
	@echo "  test        - Run all tests"
	@echo "  run         - Run all services"
	@echo "  dev         - Set up and run development environment"
	@echo "  docker      - Build and run Docker containers"
	@echo "  benchmark   - Run performance benchmarks"
	@echo "  security    - Run security audits and scans"
	@echo "  quality     - Run code quality checks"
	@echo "  docs        - Build and serve documentation"
	@echo "  clean       - Clean all build artifacts"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Individual service targets:"
	@echo "  install-python, install-rust, install-go, install-node"
	@echo "  build-python, build-rust, build-go, build-web"
	@echo "  test-python, test-rust, test-go, test-web"
	@echo "  run-python, run-rust, run-go, run-web"
	@echo ""
	@echo "Examples:"
	@echo "  make dev           # Set up development environment"
	@echo "  make run-all       # Start all services"
	@echo "  make benchmark     # Run performance tests"
	@echo "  make security      # Check security vulnerabilities"

