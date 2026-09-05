# 🍌 FreshScore AWS Zero-Trust Production Setup Guide

This guide details the step-by-step process for deploying the FreshScore Banana Ripeness MLOps system to AWS using a **Zero-Trust Enterprise Architecture**.

---

## 🛡️ Key Architectural Highlights
* **Zero Open Inbound Ports for Admin:** Port 22 (SSH) is **completely closed**. Deployment commands are executed via encrypted AWS Systems Manager (SSM) agent.
* **Keyless CI/CD:** GitHub Actions uses AWS IAM OpenID Connect (OIDC) federation. No long-lived AWS Access Keys are stored in GitHub Secrets.
* **AWS Free Tier Friendly:** Uses `t3.small` with 2GB swap, Amazon ECR (500MB free), and Amazon S3.

---

## 📋 Step-by-Step Implementation Checkpoints

### Checkpoint 1: Create Amazon S3 Bucket
1. Go to [AWS S3 Console](https://s3.console.aws.amazon.com/s3/).
2. Click **Create bucket**.
3. **Bucket name:** `banana-freshscore-storage-<your-suffix>` (e.g., `banana-freshscore-storage-harsh`). Must be globally unique.
4. **AWS Region:** Select your region (e.g., `us-east-1`).
5. **Block Public Access:** Keep **Block all public access** enabled (checked).
6. Click **Create bucket**. Save your bucket name.

---

### Checkpoint 2: Create Amazon ECR Private Repositories
1. Go to [Amazon ECR Console](https://console.aws.amazon.com/ecr/).
2. Click **Create repository**:
   - Repository name: `banana-fastapi`
   - Visibility: **Private**
   - Click **Create repository**.
3. Click **Create repository** again:
   - Repository name: `banana-streamlit`
   - Visibility: **Private**
   - Click **Create repository**.
4. Note your **12-digit AWS Account ID** (shown in top-right corner of AWS console).

---

### Checkpoint 3: Setup GitHub Actions IAM OIDC Identity Provider & Role

#### 3.1 Register GitHub OIDC Identity Provider (One-time per AWS account)
1. Go to [IAM Console $\rightarrow$ Identity providers](https://console.aws.amazon.com/iamv2/home#/identity_providers).
2. Click **Add provider**:
   - Provider type: **OpenID Connect**
   - Provider URL: `https://token.actions.githubusercontent.com`
   - Audience: `sts.amazonaws.com`
3. Click **Get thumbprint**, then click **Add provider**.

#### 3.2 Create the GitHub Actions IAM Role
1. Go to [IAM Console $\rightarrow$ Roles $\rightarrow$ Create role](https://console.aws.amazon.com/iamv2/home#/roles/create).
2. Select **Custom trust policy**.
3. Open `aws/iam_oidc_trust_policy.json` from this repository:
   - Replace `<ACCOUNT_ID>` with your 12-digit AWS Account ID.
   - Paste the JSON into the trust policy editor.
   - Click **Next**.
4. On the Permissions page:
   - Click **Create policy** (opens in a new tab).
   - Switch to the **JSON** tab.
   - Paste the contents of `aws/github_actions_role_policy.json`.
   - Click **Next**. Name the policy `BananaGitHubActionsPolicy` and click **Create policy**.
5. Return to the Role creation tab, click refresh, select `BananaGitHubActionsPolicy`, and click **Next**.
6. Role name: `GitHubActionsBananaMLOpsRole`.
7. Click **Create role**.
8. Copy the **Role ARN** (format: `arn:aws:iam::<ACCOUNT_ID>:role/GitHubActionsBananaMLOpsRole`).

---

### Checkpoint 4: Create EC2 Zero-Trust IAM Instance Profile
1. Go to [IAM Console $\rightarrow$ Roles $\rightarrow$ Create role](https://console.aws.amazon.com/iamv2/home#/roles/create).
2. Select **AWS service**, use case: **EC2**. Click **Next**.
3. Search and select these two AWS Managed Policies:
   - `AmazonSSMManagedInstanceCore`
   - `AmazonEC2ContainerRegistryReadOnly`
4. Click **Next**.
5. Role name: `BananaEC2SSMInstanceProfileRole`.
6. Click **Create role**.

---

### Checkpoint 5: Launch EC2 Instance (Zero SSH, Port 22 Closed)
1. Go to [Amazon EC2 Console $\rightarrow$ Launch instance](https://console.aws.amazon.com/ec2/home#LaunchInstances:).
2. **Name:** `banana-mlops-production`.
3. **Application and OS Images:** Select **Ubuntu Server 22.04 LTS (HVM)**, SSD Volume Type (64-bit x86).
4. **Instance type:** `t3.small` (2 vCPU, 2 GiB Memory).
5. **Key pair (login):** Select **Proceed without a key pair** (we use zero-trust SSM; SSH keys are not required!).
6. **Network settings $\rightarrow$ Firewall (security groups):**
   - Click **Create security group**: Name: `banana-mlops-sg`.
   - **DELETE or REMOVE the SSH (Port 22) rule** if present.
   - Click **Add security group rule**:
     - Type: **Custom TCP** | Port range: `8000` | Source type: **Anywhere (0.0.0.0/0)** (FastAPI Serving API)
   - Click **Add security group rule**:
     - Type: **Custom TCP** | Port range: `8501` | Source type: **Anywhere (0.0.0.0/0)** (Streamlit UI)
7. **Configure storage:** Change root size to `20 GiB` gp3.
8. **Advanced details:**
   - **IAM instance profile:** Select `BananaEC2SSMInstanceProfileRole`.
   - **User data:** Copy and paste the entire contents of `aws/ec2_userdata.sh`.
9. Click **Launch instance**.
10. Once launched, record:
    - **Instance ID** (e.g., `i-0123456789abcdef0`)
    - **Public IPv4 address** (e.g., `3.85.120.45`)

---

### Checkpoint 6: Configure GitHub Repository Secrets
1. In your GitHub repository: go to **Settings $\rightarrow$ Secrets and variables $\rightarrow$ Actions**.
2. Click **New repository secret** for each of the following:

| Secret Name | Description / Value | Example |
|---|---|---|
| `AWS_ROLE_TO_ASSUME` | The ARN of the IAM role created in Checkpoint 3.2 | `arn:aws:iam::123456789012:role/GitHubActionsBananaMLOpsRole` |
| `AWS_REGION` | The AWS region of your resources | `us-east-1` |
| `AWS_ACCOUNT_ID` | Your 12-digit AWS Account ID | `123456789012` |
| `EC2_INSTANCE_ID` | The EC2 instance ID from Checkpoint 5 | `i-0123456789abcdef0` |
| `EC2_PUBLIC_IP` | The public IPv4 address of the EC2 instance | `3.85.120.45` |
| `S3_BUCKET` | The S3 bucket name created in Checkpoint 1 | `banana-freshscore-storage-harsh` |

---

### Checkpoint 7: Trigger Zero-Trust Automated Deployment
1. With all secrets configured, push any commit to the `main` branch, or navigate to **Actions $\rightarrow$ FreshScore Zero-Trust Production Deployment $\rightarrow$ Run workflow**.
2. Watch the automated pipeline execute:
   - Automated linting & unit tests run.
   - Keyless OIDC authentication to AWS.
   - Multi-stage Docker images build and push to AWS ECR.
   - Command dispatched via AWS Systems Manager.
   - EC2 instance pulls images and starts container stack over encrypted SSM.
   - External health probe validates both ports 8000 and 8501.
3. Access your live applications:
   - 🍌 **Streamlit Dashboard:** `http://<EC2_PUBLIC_IP>:8501`
   - 🚀 **FastAPI Docs:** `http://<EC2_PUBLIC_IP>:8000/docs`
