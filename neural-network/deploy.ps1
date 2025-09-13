# Neural Network Deployment Script for Windows
param(
    [string]$Domain = "localhost",
    [string]$Email = "admin@example.com"
)

Write-Host "🚀 Deploying Neural Network to Production..." -ForegroundColor Blue

# Configuration
Write-Host "📋 Configuration:" -ForegroundColor Blue
Write-Host "  Domain: $Domain" -ForegroundColor White
Write-Host "  Email: $Email" -ForegroundColor White

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is available
try {
    docker-compose --version | Out-Null
    Write-Host "✅ docker-compose is available" -ForegroundColor Green
} catch {
    Write-Host "❌ docker-compose is not installed. Please install docker-compose first." -ForegroundColor Red
    exit 1
}

# Stop existing containers
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.prod.yml down

# Build custom images
Write-Host "🔨 Building custom images..." -ForegroundColor Yellow
docker-compose -f docker-compose.prod.yml build --no-cache

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "nginx\ssl" | Out-Null
New-Item -ItemType Directory -Force -Path "monitoring" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Generate SSL certificates (self-signed for development)
Write-Host "🔐 Generating SSL certificates..." -ForegroundColor Yellow
if (-not (Test-Path "nginx\ssl\cert.pem") -or -not (Test-Path "nginx\ssl\key.pem")) {
    # Use OpenSSL if available, otherwise create dummy certificates
    try {
        openssl req -x509 -newkey rsa:4096 -keyout nginx\ssl\key.pem -out nginx\ssl\cert.pem -days 365 -nodes -subj "/C=RU/ST=Moscow/L=Moscow/O=NeuralNetwork/OU=IT/CN=$Domain"
        Write-Host "✅ SSL certificates generated" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ OpenSSL not found, creating dummy certificates..." -ForegroundColor Yellow
        # Create dummy certificates for development
        echo "-----BEGIN CERTIFICATE-----" > nginx\ssl\cert.pem
        echo "DUMMY CERTIFICATE FOR DEVELOPMENT" >> nginx\ssl\cert.pem
        echo "-----END CERTIFICATE-----" >> nginx\ssl\cert.pem
        echo "-----BEGIN PRIVATE KEY-----" > nginx\ssl\key.pem
        echo "DUMMY PRIVATE KEY FOR DEVELOPMENT" >> nginx\ssl\key.pem
        echo "-----END PRIVATE KEY-----" >> nginx\ssl\key.pem
    }
} else {
    Write-Host "✅ SSL certificates already exist" -ForegroundColor Green
}

# Create Prometheus configuration
Write-Host "📊 Creating Prometheus configuration..." -ForegroundColor Yellow
@"
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
"@ | Out-File -FilePath "monitoring\prometheus.yml" -Encoding UTF8

# Start services
Write-Host "🚀 Starting services..." -ForegroundColor Yellow
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Health checks
Write-Host "🏥 Performing health checks..." -ForegroundColor Yellow

# Check Go API
try {
    $response = Invoke-RestMethod -Uri "http://localhost/api/health" -Method Get -TimeoutSec 10
    Write-Host "✅ Go API is healthy" -ForegroundColor Green
} catch {
    Write-Host "❌ Go API is not responding" -ForegroundColor Red
}

# Check Python API
try {
    $response = Invoke-RestMethod -Uri "http://localhost/api/health" -Method Get -TimeoutSec 10
    Write-Host "✅ Python API is healthy" -ForegroundColor Green
} catch {
    Write-Host "❌ Python API is not responding" -ForegroundColor Red
}

# Check Nginx
try {
    $response = Invoke-WebRequest -Uri "http://localhost/" -Method Get -TimeoutSec 10
    Write-Host "✅ Nginx is serving content" -ForegroundColor Green
} catch {
    Write-Host "❌ Nginx is not responding" -ForegroundColor Red
}

# Show running containers
Write-Host "📋 Running containers:" -ForegroundColor Blue
docker-compose -f docker-compose.prod.yml ps

# Show logs
Write-Host "📝 Recent logs:" -ForegroundColor Blue
docker-compose -f docker-compose.prod.yml logs --tail=10

Write-Host "🎉 Deployment completed!" -ForegroundColor Green
Write-Host "🌐 Access your Neural Network at:" -ForegroundColor Blue
Write-Host "  HTTP:  http://$Domain" -ForegroundColor White
Write-Host "  HTTPS: https://$Domain" -ForegroundColor White
Write-Host "  Grafana: http://$Domain/grafana (admin/admin123)" -ForegroundColor White
Write-Host "  Prometheus: http://$Domain/prometheus" -ForegroundColor White

Write-Host "📊 Useful commands:" -ForegroundColor Yellow
Write-Host "  View logs: docker-compose -f docker-compose.prod.yml logs -f" -ForegroundColor White
Write-Host "  Stop services: docker-compose -f docker-compose.prod.yml down" -ForegroundColor White
Write-Host "  Restart: docker-compose -f docker-compose.prod.yml restart" -ForegroundColor White
Write-Host "  Scale: docker-compose -f docker-compose.prod.yml up -d --scale neural-network-python=2" -ForegroundColor White
