#!/bin/bash

# Advanced Neural Network - Startup Script
# This script starts all services in the correct order

set -e

echo "🧠 Starting Advanced Neural Network System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if required tools are installed
check_requirements() {
    print_status "Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Rust
    if ! command -v cargo &> /dev/null; then
        print_error "Rust is not installed"
        exit 1
    fi
    
    # Check Go
    if ! command -v go &> /dev/null; then
        print_error "Go is not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        exit 1
    fi
    
    # Check CUDA (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_success "CUDA is available"
        export CUDA_AVAILABLE=true
    else
        print_warning "CUDA is not available - will run on CPU"
        export CUDA_AVAILABLE=false
    fi
    
    print_success "All requirements satisfied"
}

# Build all services
build_services() {
    print_status "Building all services..."
    
    # Build Python neural network
    print_status "Building Python neural network..."
    cd python-service
    pip install -r requirements.txt
    cd ..
    
    # Build Rust service
    print_status "Building Rust service..."
    cd rust-service
    cargo build --release
    cd ..
    
    # Build Go service
    print_status "Building Go service..."
    cd go-service
    go build -o neural-network-go .
    cd ..
    
    # Build web interface
    print_status "Building web interface..."
    cd web-interface
    npm install
    npm run build
    cd ..
    
    print_success "All services built successfully"
}

# Start infrastructure services
start_infrastructure() {
    print_status "Starting infrastructure services..."
    
    # Start Redis
    print_status "Starting Redis..."
    docker run -d --name neural-network-redis \
        -p 6379:6379 \
        redis:7-alpine \
        redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    
    # Start PostgreSQL
    print_status "Starting PostgreSQL..."
    docker run -d --name neural-network-postgres \
        -p 5432:5432 \
        -e POSTGRES_DB=neural_db \
        -e POSTGRES_USER=neural \
        -e POSTGRES_PASSWORD=password \
        -v $(pwd)/sql/init.sql:/docker-entrypoint-initdb.d/init.sql \
        postgres:15-alpine
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    print_success "Infrastructure services started"
}

# Start neural network services
start_neural_services() {
    print_status "Starting neural network services..."
    
    # Start Python neural network
    print_status "Starting Python neural network..."
    cd python-service
    python main.py &
    PYTHON_PID=$!
    cd ..
    
    # Start Rust service
    print_status "Starting Rust service..."
    cd rust-service
    ./target/release/neural-network-rust &
    RUST_PID=$!
    cd ..
    
    # Start Go service
    print_status "Starting Go service..."
    cd go-service
    ./neural-network-go &
    GO_PID=$!
    cd ..
    
    # Start web interface
    print_status "Starting web interface..."
    cd web-interface
    npm run dev &
    WEB_PID=$!
    cd ..
    
    print_success "All neural network services started"
}

# Start monitoring services
start_monitoring() {
    print_status "Starting monitoring services..."
    
    # Start Prometheus
    print_status "Starting Prometheus..."
    docker run -d --name neural-network-prometheus \
        -p 9090:9090 \
        -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
        -v $(pwd)/monitoring/rules:/etc/prometheus/rules \
        prom/prometheus:latest \
        --config.file=/etc/prometheus/prometheus.yml \
        --storage.tsdb.path=/prometheus \
        --web.console.libraries=/etc/prometheus/console_libraries \
        --web.console.templates=/etc/prometheus/consoles \
        --web.enable-lifecycle
    
    # Start Grafana
    print_status "Starting Grafana..."
    docker run -d --name neural-network-grafana \
        -p 3001:3000 \
        -e GF_SECURITY_ADMIN_PASSWORD=admin \
        -e GF_USERS_ALLOW_SIGN_UP=false \
        -v $(pwd)/monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards \
        -v $(pwd)/monitoring/grafana/datasources:/etc/grafana/provisioning/datasources \
        grafana/grafana:latest
    
    print_success "Monitoring services started"
}

# Display service information
show_services() {
    print_success "All services started successfully!"
    echo ""
    echo "🌐 Web Interface: http://localhost:3000"
    echo "🔧 Go API Gateway: http://localhost:8090"
    echo "🦀 Rust Service: http://localhost:8080"
    echo "🐍 Python Neural Network: http://localhost:8081"
    echo "📊 Prometheus: http://localhost:9090"
    echo "📈 Grafana: http://localhost:3001 (admin/admin)"
    echo "🗄️  Redis: localhost:6379"
    echo "🐘 PostgreSQL: localhost:5432"
    echo ""
    echo "📚 API Documentation:"
    echo "  - POST /api/v1/generate - Generate text"
    echo "  - GET /api/v1/status - Service status"
    echo "  - GET /api/v1/health - Health check"
    echo "  - WebSocket /ws - Real-time streaming"
    echo ""
    echo "🛑 To stop all services: ./stop.sh"
    echo "📋 To view logs: docker logs <container-name>"
}

# Cleanup function
cleanup() {
    print_status "Stopping services..."
    
    # Kill background processes
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null || true
    fi
    if [ ! -z "$RUST_PID" ]; then
        kill $RUST_PID 2>/dev/null || true
    fi
    if [ ! -z "$GO_PID" ]; then
        kill $GO_PID 2>/dev/null || true
    fi
    if [ ! -z "$WEB_PID" ]; then
        kill $WEB_PID 2>/dev/null || true
    fi
    
    # Stop Docker containers
    docker stop neural-network-redis neural-network-postgres neural-network-prometheus neural-network-grafana 2>/dev/null || true
    docker rm neural-network-redis neural-network-postgres neural-network-prometheus neural-network-grafana 2>/dev/null || true
    
    print_success "All services stopped"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    print_status "Advanced Neural Network Startup Script"
    echo ""
    
    # Check requirements
    check_docker
    check_requirements
    
    # Build services
    build_services
    
    # Start services
    start_infrastructure
    start_neural_services
    start_monitoring
    
    # Show service information
    show_services
    
    # Keep script running
    print_status "Press Ctrl+C to stop all services..."
    wait
}

# Run main function
main "$@"

