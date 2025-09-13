#!/bin/bash

# Advanced Neural Network - Stop Script
# This script stops all services and cleans up resources

set -e

echo "🛑 Stopping Advanced Neural Network System..."

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

# Stop all Docker containers
stop_docker_containers() {
    print_status "Stopping Docker containers..."
    
    # Stop and remove containers
    docker stop neural-network-redis neural-network-postgres neural-network-prometheus neural-network-grafana 2>/dev/null || true
    docker rm neural-network-redis neural-network-postgres neural-network-prometheus neural-network-grafana 2>/dev/null || true
    
    print_success "Docker containers stopped"
}

# Stop all background processes
stop_background_processes() {
    print_status "Stopping background processes..."
    
    # Kill processes by name
    pkill -f "python main.py" 2>/dev/null || true
    pkill -f "neural-network-rust" 2>/dev/null || true
    pkill -f "neural-network-go" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    
    print_success "Background processes stopped"
}

# Clean up temporary files
cleanup_temp_files() {
    print_status "Cleaning up temporary files..."
    
    # Remove temporary files
    rm -f /tmp/neural-network-*.log 2>/dev/null || true
    rm -f /tmp/neural-network-*.pid 2>/dev/null || true
    
    print_success "Temporary files cleaned up"
}

# Main execution
main() {
    print_status "Advanced Neural Network Stop Script"
    echo ""
    
    # Stop services
    stop_docker_containers
    stop_background_processes
    cleanup_temp_files
    
    print_success "All services stopped successfully!"
    echo ""
    echo "🧹 Cleanup completed"
    echo "🔄 To start again: ./start.sh"
}

# Run main function
main "$@"

