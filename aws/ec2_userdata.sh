#!/bin/bash
set -euxo pipefail

# Log all user data output
exec > >(tee -a /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "========================================="
echo "Starting Banana MLOps Zero-Trust Bootstrap"
echo "========================================="

# 1. Setup 2GB Swap File (Crucial for t3.small 2GB RAM budget)
if [ ! -f /swapfile ]; then
    echo "Creating 2GB swap file..."
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "Swap created successfully."
fi

# 2. Update and Install Prerequisites
apt-get update -y
apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    unzip \
    jq

# 3. Install AWS CLI v2
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI v2..."
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install
    rm -rf aws awscliv2.zip
fi

# 4. Install Docker Engine and Docker Compose Plugin
echo "Installing Docker Engine..."
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# 5. Ensure AWS SSM Agent is Running
echo "Configuring AWS SSM Agent..."
if snap list amazon-ssm-agent &> /dev/null; then
    systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service
    systemctl restart snap.amazon-ssm-agent.amazon-ssm-agent.service
else
    snap install amazon-ssm-agent --classic
    systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service
    systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service
fi

# 6. Initialize Application Directory
echo "Setting up application directory..."
mkdir -p /home/ubuntu/banana-mlops/{models,data/production_logs/images,reports}
chown -R ubuntu:ubuntu /home/ubuntu/banana-mlops

echo "========================================="
echo "Banana MLOps Zero-Trust Bootstrap Complete"
echo "========================================="
