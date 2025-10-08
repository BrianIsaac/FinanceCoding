# Production Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the GNN Portfolio Optimization System in production environments. The deployment process is designed for institutional requirements with emphasis on security, reliability, and scalability.

## Prerequisites

### Hardware Requirements
- **GPU**: RTX GeForce 5070Ti with 12GB VRAM (minimum)
- **RAM**: 32GB system memory (64GB recommended)
- **Storage**: 1TB NVMe SSD with high IOPS
- **CPU**: 16-core processor (Intel Xeon or AMD EPYC series)
- **Network**: Stable internet connection for data feeds

### Software Requirements
- **Operating System**: Ubuntu 22.04 LTS or RHEL 8+
- **Python**: 3.12+
- **CUDA**: 12.8+ drivers
- **Docker**: 24.0+ (optional for containerized deployment)

## Installation Methods

### Method 1: Direct Installation (Recommended)

#### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    build-essential \
    curl \
    git \
    htop \
    nvidia-driver-550 \
    python3-dev \
    software-properties-common \
    wget

# Install CUDA Toolkit 12.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8

# Verify CUDA installation
nvidia-smi
nvcc --version
```

#### 2. Python Environment Setup

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Verify uv installation
uv --version

# Create production user and directories
sudo useradd -m -s /bin/bash gnn-prod
sudo mkdir -p /opt/gnn-portfolio
sudo chown gnn-prod:gnn-prod /opt/gnn-portfolio

# Switch to production user
sudo su - gnn-prod
cd /opt/gnn-portfolio
```

#### 3. Project Installation

```bash
# Clone repository
git clone <repository-url> gnn-portfolio-system
cd gnn-portfolio-system

# Create Python environment
uv python install 3.12
uv venv --python 3.12 production-env

# Activate environment
source production-env/bin/activate

# Install production dependencies
uv pip install -e ".[data,graph,requests,logging]"

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print('PyTorch Geometric installed successfully')"
```

#### 4. GPU Optimization Setup

```bash
# Create GPU memory configuration
cat > configs/gpu_config.yaml << EOF
gpu:
  device_id: 0
  memory_limit_gb: 11.0  # Conservative limit for 12GB card
  mixed_precision: true
  gradient_checkpointing: true
  memory_monitoring: true
  
optimization:
  batch_size_strategy: "adaptive"
  gradient_accumulation: 4
  memory_cleanup_frequency: 10
  oom_recovery: true

monitoring:
  memory_threshold: 0.9
  temperature_threshold: 85
  alert_on_threshold: true
EOF
```

#### 5. Production Configuration

```bash
# Create production configuration directory
mkdir -p configs/production

# Production system configuration
cat > configs/production/system_config.yaml << EOF
environment: production
debug: false
log_level: INFO

data:
  universe: "sp_midcap_400" 
  data_path: "/opt/gnn-portfolio/data"
  cache_path: "/opt/gnn-portfolio/cache"
  backup_path: "/opt/gnn-portfolio/backups"
  refresh_schedule: "0 6 * * 1-5"  # 6 AM weekdays

models:
  checkpoint_path: "/opt/gnn-portfolio/models"
  auto_reload: true
  backup_frequency: "daily"
  
monitoring:
  metrics_retention_days: 365
  alert_email: "ops@yourcompany.com"
  dashboard_port: 8080
  
security:
  api_key_rotation_days: 90
  audit_log_retention_days: 2555  # 7 years
  encryption_at_rest: true
EOF

# Model-specific configurations
cat > configs/production/model_config.yaml << EOF
hrp:
  lookback_days: 756
  clustering_config:
    linkage_method: "ward"
    distance_metric: "correlation"
    min_observations: 252
  rebalance_frequency: "monthly"

lstm:
  sequence_length: 60
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  learning_rate: 0.001
  early_stopping_patience: 10

gat:
  attention_heads: 4
  hidden_dim: 128
  dropout: 0.3
  graph_construction: "k_nn"
  k_neighbors: 10
  
constraints:
  long_only: true
  top_k_positions: 50
  max_position_weight: 0.10
  max_monthly_turnover: 0.20
  transaction_cost_bps: 10.0
EOF
```

### Method 2: Docker Containerized Deployment

#### 1. Docker Setup

```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM nvidia/cuda:12.8-devel-ubuntu22.04

# System dependencies
RUN apt update && apt install -y \\
    python3.12 \\
    python3.12-dev \\
    curl \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN uv pip install -e ".[data,graph,requests,logging]"

# Create non-root user
RUN useradd -m -u 1000 gnn-prod
USER gnn-prod

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start command
CMD ["python", "src/main.py", "--config", "configs/production/system_config.yaml"]
EOF

# Create docker-compose for production
cat > docker-compose.prod.yaml << EOF
version: '3.8'

services:
  gnn-portfolio:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "8000:8000"  # API
      - "8080:8080"  # Dashboard
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./configs:/app/configs
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
EOF
```

#### 2. Container Deployment

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yaml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yaml ps
docker logs gnn-portfolio_gnn-portfolio_1

# Check GPU access in container
docker exec -it gnn-portfolio_gnn-portfolio_1 nvidia-smi
```

## Environment Configuration

### Environment Variables

```bash
# Create environment configuration
cat > .env.production << EOF
# Environment
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://gnn_user:secure_password@localhost:5432/gnn_portfolio

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_SECRET_KEY=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 32)
API_RATE_LIMIT=1000

# Data Sources
YAHOO_FINANCE_RATE_LIMIT=200
DATA_CACHE_TTL=3600

# Monitoring
PROMETHEUS_METRICS=true
GRAFANA_DASHBOARD=true
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Security
ALLOWED_HOSTS=api.gnn-portfolio.com,localhost
CORS_ORIGINS=https://dashboard.gnn-portfolio.com
SSL_REQUIRED=true

# Performance
MAX_WORKERS=8
ASYNC_POOL_SIZE=20
GPU_MEMORY_LIMIT=11.0
EOF

# Set secure permissions
chmod 600 .env.production
```

### SSL/TLS Configuration

```bash
# Generate SSL certificates (for production, use proper CA-signed certificates)
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -keyout ssl/private.key -out ssl/certificate.crt -days 365 -nodes

# Create nginx configuration for reverse proxy
cat > nginx.conf << EOF
upstream gnn_api {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl http2;
    server_name api.gnn-portfolio.com;

    ssl_certificate /opt/gnn-portfolio/ssl/certificate.crt;
    ssl_certificate_key /opt/gnn-portfolio/ssl/private.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    location / {
        proxy_pass http://gnn_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }
}
EOF
```

## Database Setup

### PostgreSQL Installation and Configuration

```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE gnn_portfolio;
CREATE USER gnn_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE gnn_portfolio TO gnn_user;
ALTER USER gnn_user CREATEDB;
\\q
EOF

# Initialize database schema
python src/database/init_db.py --config configs/production/system_config.yaml

# Create database backup script
cat > scripts/backup_database.sh << EOF
#!/bin/bash
BACKUP_DIR="/opt/gnn-portfolio/backups"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
pg_dump gnn_portfolio > "\$BACKUP_DIR/gnn_portfolio_\$TIMESTAMP.sql"
# Keep only last 30 days of backups
find \$BACKUP_DIR -name "gnn_portfolio_*.sql" -mtime +30 -delete
EOF

chmod +x scripts/backup_database.sh
```

## Monitoring and Logging

### System Monitoring Setup

```bash
# Create monitoring configuration
mkdir -p monitoring

cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'gnn-portfolio'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'gpu-metrics'
    static_configs:
      - targets: ['localhost:9400']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
EOF

# Install Node Exporter for system metrics
wget https://github.com/prometheus/node_exporter/releases/download/v1.8.2/node_exporter-1.8.2.linux-amd64.tar.gz
tar xvfz node_exporter-1.8.2.linux-amd64.tar.gz
sudo cp node_exporter-1.8.2.linux-amd64/node_exporter /usr/local/bin/
rm -rf node_exporter-1.8.2.*

# Create systemd service for Node Exporter
sudo tee /etc/systemd/system/node_exporter.service << EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=gnn-prod
Group=gnn-prod
Type=simple
ExecStart=/usr/local/bin/node_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter
```

### Application Logging

```bash
# Create logging configuration
mkdir -p logs

cat > configs/logging_config.yaml << EOF
version: 1
disable_existing_loggers: False

formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'
  
  json:
    format: '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: detailed
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: logs/gnn_portfolio.log
    maxBytes: 104857600  # 100MB
    backupCount: 10
    
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: json
    filename: logs/gnn_portfolio_errors.log
    maxBytes: 104857600
    backupCount: 5

loggers:
  src:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: False
    
  torch:
    level: WARNING
    handlers: [console, file]
    
  urllib3:
    level: WARNING
    handlers: [console, file]

root:
  level: INFO
  handlers: [console, file]
EOF
```

## Service Management

### Systemd Service Setup

```bash
# Create systemd service file
sudo tee /etc/systemd/system/gnn-portfolio.service << EOF
[Unit]
Description=GNN Portfolio Optimization System
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=gnn-prod
Group=gnn-prod
WorkingDirectory=/opt/gnn-portfolio/gnn-portfolio-system
Environment=PATH=/opt/gnn-portfolio/gnn-portfolio-system/production-env/bin
ExecStart=/opt/gnn-portfolio/gnn-portfolio-system/production-env/bin/python src/main.py --config configs/production/system_config.yaml
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gnn-portfolio

# Resource limits
LimitNOFILE=65536
MemoryMax=32G

# Security settings
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/gnn-portfolio
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable gnn-portfolio.service
sudo systemctl start gnn-portfolio.service

# Check service status
sudo systemctl status gnn-portfolio.service
sudo journalctl -u gnn-portfolio.service -f
```

### Process Management with Supervisor

```bash
# Install supervisor (alternative to systemd)
sudo apt install -y supervisor

# Create supervisor configuration
sudo tee /etc/supervisor/conf.d/gnn-portfolio.conf << EOF
[program:gnn-portfolio]
command=/opt/gnn-portfolio/gnn-portfolio-system/production-env/bin/python src/main.py --config configs/production/system_config.yaml
directory=/opt/gnn-portfolio/gnn-portfolio-system
user=gnn-prod
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/gnn-portfolio/logs/supervisor.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=CUDA_VISIBLE_DEVICES=0

[program:gnn-worker]
command=/opt/gnn-portfolio/gnn-portfolio-system/production-env/bin/python src/worker.py --config configs/production/system_config.yaml
directory=/opt/gnn-portfolio/gnn-portfolio-system
user=gnn-prod
autostart=true
autorestart=true
numprocs=4
process_name=%(program_name)s_%(process_num)02d
redirect_stderr=true
stdout_logfile=/opt/gnn-portfolio/logs/worker_%(process_num)02d.log
EOF

# Reload supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status
```

## Security Hardening

### Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow specific ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 443/tcp   # HTTPS API
sudo ufw allow 8080/tcp  # Dashboard (optional, can be restricted to internal network)

# Restrict SSH access to specific IPs (recommended)
sudo ufw delete allow 22/tcp
sudo ufw allow from 192.168.1.0/24 to any port 22

# Check firewall status
sudo ufw status numbered
```

### Application Security

```bash
# Create security configuration
cat > configs/security_config.yaml << EOF
authentication:
  jwt_expiry_hours: 24
  refresh_token_days: 30
  max_login_attempts: 5
  lockout_duration_minutes: 15

api_security:
  rate_limiting: true
  cors_enabled: true
  allowed_origins:
    - "https://dashboard.gnn-portfolio.com"
  content_security_policy: true
  
encryption:
  at_rest: true
  in_transit: true
  key_rotation_days: 90
  
audit:
  log_all_requests: true
  log_authentication: true
  log_portfolio_access: true
  retention_days: 2555  # 7 years

compliance:
  data_residency: "US"
  encryption_standard: "AES-256"
  audit_trail: true
  access_logging: true
EOF

# Set up SSL certificate renewal (if using Let's Encrypt)
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.gnn-portfolio.com
```

## Backup and Disaster Recovery

### Automated Backup Setup

```bash
# Create comprehensive backup script
cat > scripts/full_backup.sh << EOF
#!/bin/bash

BACKUP_DIR="/opt/gnn-portfolio/backups"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="gnn_portfolio_full_\$TIMESTAMP"

echo "Starting full backup at \$(date)"

# Create backup directory
mkdir -p "\$BACKUP_DIR/\$BACKUP_NAME"

# Database backup
pg_dump gnn_portfolio > "\$BACKUP_DIR/\$BACKUP_NAME/database.sql"

# Model checkpoints
rsync -av --progress /opt/gnn-portfolio/models/ "\$BACKUP_DIR/\$BACKUP_NAME/models/"

# Configuration files
rsync -av --progress /opt/gnn-portfolio/gnn-portfolio-system/configs/ "\$BACKUP_DIR/\$BACKUP_NAME/configs/"

# Data cache
rsync -av --progress /opt/gnn-portfolio/cache/ "\$BACKUP_DIR/\$BACKUP_NAME/cache/"

# Logs (last 7 days)
find /opt/gnn-portfolio/logs -mtime -7 -type f -exec cp {} "\$BACKUP_DIR/\$BACKUP_NAME/logs/" \\;

# Create archive
cd "\$BACKUP_DIR"
tar -czf "\$BACKUP_NAME.tar.gz" "\$BACKUP_NAME"
rm -rf "\$BACKUP_NAME"

# Upload to remote storage (configure as needed)
# aws s3 cp "\$BACKUP_NAME.tar.gz" s3://your-backup-bucket/

# Cleanup old backups (keep 30 days)
find "\$BACKUP_DIR" -name "gnn_portfolio_full_*.tar.gz" -mtime +30 -delete

echo "Backup completed at \$(date)"
EOF

chmod +x scripts/full_backup.sh

# Set up daily backup cron job
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/gnn-portfolio/gnn-portfolio-system/scripts/full_backup.sh") | crontab -
```

### Disaster Recovery Plan

```bash
# Create disaster recovery script
cat > scripts/disaster_recovery.sh << EOF
#!/bin/bash

BACKUP_FILE=\$1
RECOVERY_DIR="/opt/gnn-portfolio-recovery"

if [[ -z "\$BACKUP_FILE" ]]; then
    echo "Usage: \$0 <backup_file.tar.gz>"
    exit 1
fi

echo "Starting disaster recovery from \$BACKUP_FILE"

# Create recovery directory
mkdir -p "\$RECOVERY_DIR"
cd "\$RECOVERY_DIR"

# Extract backup
tar -xzf "\$BACKUP_FILE"
BACKUP_NAME=\$(basename "\$BACKUP_FILE" .tar.gz)

# Restore database
echo "Restoring database..."
sudo -u postgres dropdb gnn_portfolio || true
sudo -u postgres createdb gnn_portfolio
sudo -u postgres psql gnn_portfolio < "\$BACKUP_NAME/database.sql"

# Restore models
echo "Restoring models..."
rsync -av "\$BACKUP_NAME/models/" /opt/gnn-portfolio/models/

# Restore configurations
echo "Restoring configurations..."
rsync -av "\$BACKUP_NAME/configs/" /opt/gnn-portfolio/gnn-portfolio-system/configs/

# Restart services
echo "Restarting services..."
sudo systemctl restart gnn-portfolio.service
sudo systemctl restart postgresql

echo "Disaster recovery completed"
EOF

chmod +x scripts/disaster_recovery.sh
```

## Deployment Verification

### Health Check Script

```bash
# Create comprehensive health check
cat > scripts/health_check.sh << EOF
#!/bin/bash

echo "=== GNN Portfolio System Health Check ==="

# Check service status
echo "1. Service Status:"
systemctl is-active gnn-portfolio.service
if systemctl is-active --quiet gnn-portfolio.service; then
    echo "✓ GNN Portfolio service is running"
else
    echo "✗ GNN Portfolio service is not running"
fi

# Check GPU
echo -e "\\n2. GPU Status:"
if nvidia-smi &>/dev/null; then
    echo "✓ GPU is accessible"
    nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
else
    echo "✗ GPU is not accessible"
fi

# Check API endpoints
echo -e "\\n3. API Health:"
if curl -f -s http://localhost:8000/health >/dev/null; then
    echo "✓ API is responding"
    curl -s http://localhost:8000/health | jq .
else
    echo "✗ API is not responding"
fi

# Check database connection
echo -e "\\n4. Database Connection:"
if PGPASSWORD=secure_password psql -h localhost -U gnn_user -d gnn_portfolio -c "SELECT 1;" >/dev/null 2>&1; then
    echo "✓ Database is accessible"
else
    echo "✗ Database is not accessible"
fi

# Check disk space
echo -e "\\n5. Disk Space:"
df -h /opt/gnn-portfolio | tail -1
DISK_USAGE=\$(df /opt/gnn-portfolio | tail -1 | awk '{print \$5}' | sed 's/%//')
if [[ \$DISK_USAGE -lt 80 ]]; then
    echo "✓ Disk usage is acceptable (\$DISK_USAGE%)"
else
    echo "⚠ Disk usage is high (\$DISK_USAGE%)"
fi

# Check recent logs for errors
echo -e "\\n6. Recent Errors:"
ERROR_COUNT=\$(tail -100 /opt/gnn-portfolio/logs/gnn_portfolio_errors.log 2>/dev/null | wc -l)
if [[ \$ERROR_COUNT -eq 0 ]]; then
    echo "✓ No recent errors"
else
    echo "⚠ \$ERROR_COUNT recent errors found"
fi

echo -e "\\n=== Health Check Complete ==="
EOF

chmod +x scripts/health_check.sh

# Run initial health check
./scripts/health_check.sh
```

## Performance Optimization

### Production Performance Tuning

```bash
# Create performance optimization script
cat > scripts/performance_tune.sh << EOF
#!/bin/bash

echo "Applying production performance optimizations..."

# GPU performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 400  # Set power limit to 400W

# System swappiness for high-memory workloads  
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# File system optimizations
echo 'fs.file-max = 2097152' | sudo tee -a /etc/sysctl.conf

# Network optimizations
echo 'net.core.somaxconn = 65535' | sudo tee -a /etc/sysctl.conf
echo 'net.core.netdev_max_backlog = 5000' | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p

# CPU governor for performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

echo "Performance optimizations applied"
EOF

chmod +x scripts/performance_tune.sh
./scripts/performance_tune.sh
```

This comprehensive deployment guide provides institutional-grade deployment procedures with emphasis on security, monitoring, and reliability for production environments.