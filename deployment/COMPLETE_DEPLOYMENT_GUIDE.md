
# 🚀 Complete Deployment Guide - Credit Default Prediction

## Table of Contents
1. [Docker Containerization](#docker-containerization)
2. [GitHub Actions CI/CD](#github-actions-cicd)
3. [AWS Deployment (S3, ECR, EC2)](#aws-deployment)
4. [Production Setup](#production-setup)

---

## 1. Docker Containerization

### 1.1 Create Dockerfile

```dockerfile
# Multi-stage build for Credit Default Prediction
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "app.py"]
```

### 1.2 Create Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  credit-api:
    build: .
    container_name: credit-default-api
    ports:
      - "8000:8000"
    volumes:
      - ./artifacts:/app/artifacts
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
      - MODEL_PATH=/app/artifacts/model_trainer/model.pkl
      - PREPROCESSOR_PATH=/app/artifacts/data_transformation/preprocessor.pkl
    command: ["python", "app.py", "--mode", "api"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  credit-dashboard:
    build: .
    container_name: credit-default-dashboard
    ports:
      - "8501:8501"
    volumes:
      - ./artifacts:/app/artifacts
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
      - API_URL=http://credit-api:8000
    command: ["streamlit", "run", "streamlit_dashboard_with_shap.py", "--server.port=8501", "--server.address=0.0.0.0"]
    depends_on:
      - credit-api
    restart: unless-stopped

  mlflow:
    image: python:3.10-slim
    container_name: mlflow-server
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow_data:/mlflow
    command: >
      bash -c "pip install mlflow &&
               mlflow server --backend-store-uri file:///mlflow --default-artifact-root file:///mlflow --host 0.0.0.0 --port 5000"
    restart: unless-stopped

volumes:
  mlflow_data:

networks:
  default:
    name: credit-network
```

### 1.3 Docker Commands

```bash
# Build and run with Docker Compose
docker-compose up --build -d

# View logs
docker-compose logs -f credit-api
docker-compose logs -f credit-dashboard

# Stop services
docker-compose down

# Rebuild specific service
docker-compose up --build credit-api

# Scale services
docker-compose up --scale credit-api=2
```

---

## 2. GitHub Actions CI/CD

### 2.1 Create GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Credit Default Prediction CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/credit-default-prediction

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Run tests
      run: |
        python -m pytest test_credit_default_fixed.py -v --cov=./ --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build-and-push:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix=sha-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging

    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add staging deployment commands here

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Deploy to AWS ECS
      run: |
        echo "Deploying to production environment"
        # Add production deployment commands here

  model-monitoring:
    needs: deploy-production
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Run model performance checks
      run: |
        echo "Checking model performance metrics"
        # Add model monitoring commands here

  notify:
    needs: [deploy-staging, deploy-production]
    runs-on: ubuntu-latest
    if: always()

    steps:
    - name: Notify deployment status
      run: |
        echo "Deployment completed with status: ${{ job.status }}"
        # Add notification logic (Slack, email, etc.)
```

### 2.2 GitHub Secrets Configuration

Required secrets in GitHub repository settings:

```
AWS_ACCESS_KEY_ID: Your AWS access key
AWS_SECRET_ACCESS_KEY: Your AWS secret key
AWS_REGION: us-east-1 (or your preferred region)
ECR_REPOSITORY: credit-default-prediction
ECS_CLUSTER: credit-default-cluster
ECS_SERVICE: credit-default-service
```

---

## 3. AWS Deployment (S3, ECR, EC2)

### 3.1 AWS Setup Scripts

#### Create S3 Bucket for Artifacts

```bash
#!/bin/bash
# setup-s3.sh

AWS_REGION="us-east-1"
BUCKET_NAME="credit-default-artifacts-$(date +%s)"

# Create S3 bucket
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled

# Set bucket policy
cat > bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowModelAccess",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::ACCOUNT-ID:role/EC2-CreditDefault-Role"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::$BUCKET_NAME/*"
        }
    ]
}
EOF

aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file://bucket-policy.json

echo "S3 bucket created: $BUCKET_NAME"
```

#### Setup ECR Repository

```bash
#!/bin/bash
# setup-ecr.sh

AWS_REGION="us-east-1"
REPO_NAME="credit-default-prediction"

# Create ECR repository
aws ecr create-repository \
    --repository-name $REPO_NAME \
    --region $AWS_REGION

# Get login token and login to ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com

echo "ECR repository created: $REPO_NAME"
```

#### EC2 Instance Setup

```bash
#!/bin/bash
# setup-ec2.sh

# User data script for EC2 instance
cat > user-data.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y docker

# Start Docker service
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /opt/credit-default
cd /opt/credit-default

# Download docker-compose.yml from S3 or GitHub
# aws s3 cp s3://your-bucket/docker-compose.yml .

# Login to ECR and pull image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com

# Start services
docker-compose up -d
EOF

# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --count 1 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --user-data file://user-data.sh \
    --iam-instance-profile Name=EC2-CreditDefault-Role \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=CreditDefault-API}]'
```

### 3.2 IAM Roles and Policies

#### EC2 Instance Role

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::credit-default-artifacts-*/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

### 3.3 Deployment Scripts

#### Build and Deploy Script

```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPOSITORY="credit-default-prediction"
IMAGE_TAG="latest"
S3_BUCKET="your-credit-default-bucket"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY"

echo "🚀 Starting deployment process..."

# Step 1: Build Docker image
echo "📦 Building Docker image..."
docker build -t $ECR_REPOSITORY:$IMAGE_TAG .

# Step 2: Tag for ECR
echo "🏷️  Tagging image for ECR..."
docker tag $ECR_REPOSITORY:$IMAGE_TAG $ECR_URI:$IMAGE_TAG

# Step 3: Login to ECR
echo "🔑 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_URI

# Step 4: Push to ECR
echo "⬆️  Pushing image to ECR..."
docker push $ECR_URI:$IMAGE_TAG

# Step 5: Upload artifacts to S3
echo "☁️  Uploading artifacts to S3..."
if [ -d "artifacts" ]; then
    aws s3 sync artifacts/ s3://$S3_BUCKET/artifacts/ --delete
fi

# Step 6: Update ECS service (if using ECS)
if [ "$1" == "ecs" ]; then
    echo "🔄 Updating ECS service..."
    aws ecs update-service \
        --cluster credit-default-cluster \
        --service credit-default-service \
        --force-new-deployment \
        --region $AWS_REGION
fi

# Step 7: Update EC2 instance (if using EC2)
if [ "$1" == "ec2" ]; then
    echo "🖥️  Updating EC2 deployment..."

    # Get instance IP (assuming single instance with specific tag)
    INSTANCE_IP=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=CreditDefault-API" "Name=instance-state-name,Values=running" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text --region $AWS_REGION)

    if [ "$INSTANCE_IP" != "None" ]; then
        echo "📡 Connecting to EC2 instance: $INSTANCE_IP"

        # SSH and update (you'll need to setup SSH keys)
        ssh -i your-key.pem ec2-user@$INSTANCE_IP << EOF
            cd /opt/credit-default
            aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
            docker-compose pull
            docker-compose up -d
            docker system prune -f
EOF
    else
        echo "❌ No running EC2 instance found"
        exit 1
    fi
fi

echo "✅ Deployment completed successfully!"
```

### 3.4 Monitoring and Health Checks

#### CloudWatch Setup

```bash
#!/bin/bash
# setup-monitoring.sh

# Create CloudWatch Log Group
aws logs create-log-group \
    --log-group-name /aws/ec2/credit-default \
    --region us-east-1

# Create CloudWatch alarms
aws cloudwatch put-metric-alarm \
    --alarm-name "CreditDefault-HighCPU" \
    --alarm-description "Credit Default API High CPU" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:credit-default-alerts

# Create custom metrics for API
aws cloudwatch put-metric-alarm \
    --alarm-name "CreditDefault-API-Errors" \
    --alarm-description "Credit Default API Error Rate" \
    --metric-name ErrorRate \
    --namespace CreditDefault/API \
    --statistic Average \
    --period 300 \
    --threshold 5 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1
```

---

## 4. Production Setup

### 4.1 Environment Configuration

#### Production Environment Variables

```bash
# production.env
ENVIRONMENT=production
LOG_LEVEL=INFO
MODEL_PATH=/app/artifacts/model_trainer/model.pkl
PREPROCESSOR_PATH=/app/artifacts/data_transformation/preprocessor.pkl
EXPLAINER_PATH=/app/artifacts/model_explainer/explainer.pkl

# Database (if using)
DATABASE_URL=postgresql://user:pass@db:5432/creditdb

# Redis (for caching)
REDIS_URL=redis://redis:6379/0

# Monitoring
SENTRY_DSN=your-sentry-dsn
NEW_RELIC_LICENSE_KEY=your-newrelic-key

# AWS
AWS_REGION=us-east-1
S3_BUCKET=credit-default-artifacts
```

### 4.2 Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  credit-api:
    image: ${ECR_URI}:${IMAGE_TAG}
    container_name: credit-default-api-prod
    ports:
      - "80:8000"
    volumes:
      - ./artifacts:/app/artifacts:ro
      - ./logs:/app/logs
    env_file:
      - production.env
    command: ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app:app"]
    restart: always
    logging:
      driver: awslogs
      options:
        awslogs-group: /aws/ec2/credit-default
        awslogs-region: us-east-1
        awslogs-stream: api

  credit-dashboard:
    image: ${ECR_URI}:${IMAGE_TAG}
    container_name: credit-default-dashboard-prod
    ports:
      - "8501:8501"
    volumes:
      - ./artifacts:/app/artifacts:ro
    env_file:
      - production.env
    environment:
      - API_URL=http://credit-api:8000
    command: ["streamlit", "run", "streamlit_dashboard_with_shap.py", "--server.port=8501", "--server.address=0.0.0.0"]
    depends_on:
      - credit-api
    restart: always
    logging:
      driver: awslogs
      options:
        awslogs-group: /aws/ec2/credit-default
        awslogs-region: us-east-1
        awslogs-stream: dashboard

  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - credit-api
      - credit-dashboard
    restart: always

  redis:
    image: redis:alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: always

volumes:
  redis_data:

networks:
  default:
    name: credit-network-prod
```

### 4.3 Load Balancer Configuration (Nginx)

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server credit-api:8000;
    }

    upstream dashboard_backend {
        server credit-dashboard:8501;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # API routes
        location /api/ {
            proxy_pass http://api_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Dashboard routes
        location / {
            proxy_pass http://dashboard_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Health check
        location /health {
            proxy_pass http://api_backend/health;
        }
    }
}
```

### 4.4 Backup and Recovery

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups"
S3_BACKUP_BUCKET="credit-default-backups"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup artifacts
tar -czf $BACKUP_DIR/artifacts_$DATE.tar.gz artifacts/

# Backup database (if using)
# pg_dump $DATABASE_URL > $BACKUP_DIR/database_$DATE.sql

# Upload to S3
aws s3 cp $BACKUP_DIR/artifacts_$DATE.tar.gz s3://$S3_BACKUP_BUCKET/artifacts/

# Clean old local backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: artifacts_$DATE.tar.gz"
```

### 4.5 Startup Scripts

```bash
#!/bin/bash
# start-production.sh

set -e

echo "🚀 Starting Credit Default Prediction in Production Mode"

# Check if required files exist
if [ ! -f "docker-compose.prod.yml" ]; then
    echo "❌ docker-compose.prod.yml not found"
    exit 1
fi

if [ ! -f "production.env" ]; then
    echo "❌ production.env not found"
    exit 1
fi

# Export environment variables
export ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/credit-default-prediction"
export IMAGE_TAG="latest"

# Pull latest images
echo "📥 Pulling latest images..."
docker-compose -f docker-compose.prod.yml pull

# Start services
echo "🏃 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🔍 Performing health checks..."
curl -f http://localhost/health || exit 1

echo "✅ Production deployment successful!"
echo "🌐 API available at: http://localhost/api/"
echo "📊 Dashboard available at: http://localhost/"
```

---

## Summary

This comprehensive deployment guide provides:

1. **Docker Containerization**: Multi-stage builds, compose files, and production configurations
2. **GitHub Actions CI/CD**: Complete pipeline with testing, security scanning, and automated deployment
3. **AWS Deployment**: S3 for artifacts, ECR for images, EC2 for hosting
4. **Production Setup**: Load balancing, monitoring, backup, and recovery procedures

### Next Steps:

1. Replace placeholder values (account IDs, bucket names, etc.) with your actual values
2. Set up AWS credentials and GitHub secrets
3. Configure domain names and SSL certificates
4. Test the deployment in a staging environment first
5. Monitor logs and metrics after production deployment

### Security Considerations:

- Use IAM roles with minimal required permissions
- Enable encryption for S3 buckets and EBS volumes
- Implement proper network security groups
- Regular security updates and vulnerability scanning
- Monitor access logs and set up alerts
