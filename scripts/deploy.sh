#!/bin/bash
set -euo pipefail

# Production Deployment Script executed on EC2 via AWS Systems Manager
echo "=================================================="
echo "Starting Zero-Trust Container Deployment via SSM"
echo "Timestamp: $(date -u)"
echo "=================================================="

APP_DIR="/home/ubuntu/banana-mlops"
mkdir -p "${APP_DIR}"
cd "${APP_DIR}"

# Validate parameters
ECR_REGISTRY="${1:-${ECR_REGISTRY:-}}"
AWS_REGION="${2:-${AWS_REGION:-us-east-1}}"
IMAGE_TAG="${3:-${IMAGE_TAG:-latest}}"
S3_BUCKET="${4:-${S3_BUCKET:-}}"

if [ -z "${ECR_REGISTRY}" ]; then
    echo "Error: ECR_REGISTRY parameter is required."
    exit 1
fi

export ECR_REGISTRY
export AWS_REGION
export IMAGE_TAG

echo "Target Registry: ${ECR_REGISTRY}"
echo "Target Tag:      ${IMAGE_TAG}"
echo "AWS Region:      ${AWS_REGION}"

# 1. Authenticate Docker with Amazon ECR via IAM Instance Profile
echo "--> Authenticating Docker with Amazon ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# 2. Sync models and artifacts from S3 if configured
if [ -n "${S3_BUCKET}" ]; then
    echo "--> Syncing model artifacts from S3 bucket: ${S3_BUCKET}..."
    mkdir -p "${APP_DIR}/models"
    aws s3 sync "s3://${S3_BUCKET}/models" "${APP_DIR}/models" --no-progress || echo "Notice: S3 model sync skipped or no existing remote models found."
fi

# 3. Pull latest container images
echo "--> Pulling updated container images from ECR..."
docker compose -f docker-compose.prod.yml pull

# 4. Deploy updated service stack
echo "--> Launching updated service containers..."
docker compose -f docker-compose.prod.yml up -d --remove-orphans

# 5. Health Check Verification Loop
echo "--> Verifying container health..."
MAX_ATTEMPTS=12
ATTEMPT=0
HEALTHY=false

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "Health probe attempt ${ATTEMPT}/${MAX_ATTEMPTS}..."
    if curl -s -f "http://localhost:8000/health" > /dev/null 2>&1; then
        echo "✅ FastAPI Backend Health Check Passed (200 OK)!"
        HEALTHY=true
        break
    fi
    sleep 5
done

if [ "${HEALTHY}" != "true" ]; then
    echo "❌ Deployment Failed: Container failed to become healthy within timeout."
    docker compose -f docker-compose.prod.yml logs --tail 50
    exit 1
fi

# 6. Housekeeping: Remove dangling images to preserve disk space
echo "--> Pruning unused Docker images..."
docker image prune -f

echo "=================================================="
echo "Zero-Trust Deployment Succeeded!"
echo "=================================================="
