#!/bin/bash

# Neural Network Deployment Script
echo "🚀 Deploying Neural Network to Production..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN=${1:-"localhost"}
EMAIL=${2:-"admin@example.com"}

echo -e "${BLUE}📋 Configuration:${NC}"
echo -e "  Domain: $DOMAIN"
echo -e "  Email: $EMAIL"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose is not installed. Please install docker-compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and docker-compose are available${NC}"

# Stop existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose -f docker-compose.prod.yml down

# Pull latest images
echo -e "${YELLOW}📥 Pulling latest images...${NC}"
docker-compose -f docker-compose.prod.yml pull

# Build custom images
echo -e "${YELLOW}🔨 Building custom images...${NC}"
docker-compose -f docker-compose.prod.yml build --no-cache

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p nginx/ssl
mkdir -p monitoring
mkdir -p logs

# Generate SSL certificates (self-signed for development)
echo -e "${YELLOW}🔐 Generating SSL certificates...${NC}"
if [ ! -f nginx/ssl/cert.pem ] || [ ! -f nginx/ssl/key.pem ]; then
    openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem -days 365 -nodes \
        -subj "/C=RU/ST=Moscow/L=Moscow/O=NeuralNetwork/OU=IT/CN=$DOMAIN"
    echo -e "${GREEN}✅ SSL certificates generated${NC}"
else
    echo -e "${GREEN}✅ SSL certificates already exist${NC}"
fi

# Create Prometheus configuration
echo -e "${YELLOW}📊 Creating Prometheus configuration...${NC}"
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'neural-network-go'
    static_configs:
      - targets: ['neural-network-go:8090']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'neural-network-python'
    static_configs:
      - targets: ['neural-network-python:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    metrics_path: '/nginx_status'
    scrape_interval: 5s
EOF

# Start services
echo -e "${YELLOW}🚀 Starting services...${NC}"
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 30

# Health checks
echo -e "${YELLOW}🏥 Performing health checks...${NC}"

# Check Go API
if curl -f http://localhost/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Go API is healthy${NC}"
else
    echo -e "${RED}❌ Go API is not responding${NC}"
fi

# Check Python API
if curl -f http://localhost/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Python API is healthy${NC}"
else
    echo -e "${RED}❌ Python API is not responding${NC}"
fi

# Check Nginx
if curl -f http://localhost/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Nginx is serving content${NC}"
else
    echo -e "${RED}❌ Nginx is not responding${NC}"
fi

# Show running containers
echo -e "${BLUE}📋 Running containers:${NC}"
docker-compose -f docker-compose.prod.yml ps

# Show logs
echo -e "${BLUE}📝 Recent logs:${NC}"
docker-compose -f docker-compose.prod.yml logs --tail=10

echo -e "${GREEN}🎉 Deployment completed!${NC}"
echo -e "${BLUE}🌐 Access your Neural Network at:${NC}"
echo -e "  HTTP:  http://$DOMAIN"
echo -e "  HTTPS: https://$DOMAIN"
echo -e "  Grafana: http://$DOMAIN/grafana (admin/admin123)"
echo -e "  Prometheus: http://$DOMAIN/prometheus"

echo -e "${YELLOW}📊 Useful commands:${NC}"
echo -e "  View logs: docker-compose -f docker-compose.prod.yml logs -f"
echo -e "  Stop services: docker-compose -f docker-compose.prod.yml down"
echo -e "  Restart: docker-compose -f docker-compose.prod.yml restart"
echo -e "  Scale: docker-compose -f docker-compose.prod.yml up -d --scale neural-network-python=2"
