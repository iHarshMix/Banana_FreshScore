# Banana Ripeness MLOps Regression Pipeline — Full Implementation Plan

---

## 1. Project Objective & Portfolio Narrative

### 1.1 What This Project Proves

Build an **MLOps-complete regression pipeline** predicting a continuous banana spoilage/ripeness score from an image, wrapped in a full production lifecycle demo. The core story is **not** modeling novelty — it is proving **MLOps maturity**: detecting model decay in production and responding to it automatically.

This project carries the **"I understand what happens after deployment"** story in the portfolio, complementing PackVote's **"I can build a sophisticated system"** story.

### 1.2 Headline Result

A **before / after / recovered MAE table** demonstrating:

| Stage | MAE | Status |
|---|---|---|
| Baseline Model (clean data) | ~0.04 | ✅ Healthy |
| After Drift (perturbed data) | ~0.12 | ⚠️ Degraded |
| After Auto-Retrain (recovered) | ~0.05 | ✅ Recovered |

### 1.3 Interview Defense Points

- **"Why regression instead of classification?"** → A continuous score enables granular shelf-life estimation (e.g., 3.6 days vs. 1.2 days), which is more valuable for warehouse inventory management than a static "Ripe" label.
- **"Isn't the drift synthetic?"** → Framed as a plausible deployment scenario: _"A new warehouse camera has a cooler color temperature and lower resolution than the original training camera."_
- **"Why not a tabular dataset?"** → This project demonstrates CNN-based MLOps and computer vision serving, while the Elec2 project (separate) covers tabular/gradient-boosted drift detection. Together they prove versatility.

---

## 2. Dataset Specification

### 2.1 Source Dataset

| Field | Value |
|---|---|
| **Dataset Title** | Banana Ripeness Classification Dataset |
| **Author** | S.M. Shahriar (`@shahriar26s`) |
| **Kaggle API Slug** | `shahriar26s/banana-ripeness-classification-dataset` |
| **URL** | https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset |
| **License** | Apache 2.0 (open for commercial and portfolio use) |
| **Total Images** | **13,478 RGB images** |
| **Compressed Size** | 231.6 MB |
| **Native Resolution** | 416 × 416 pixels (uniformly stretched) |

### 2.2 Dataset Split Breakdown

```
data/raw/banana-ripeness-classification/
├── train/ (11,793 images | 87%)
│   ├── unripe/
│   ├── ripe/
│   ├── overripe/
│   └── rotten/
├── valid/ (1,123 images | 8%)
│   ├── unripe/
│   ├── ripe/
│   ├── overripe/
│   └── rotten/
└── test/  (562 images | 4%)
    ├── unripe/
    ├── ripe/
    ├── overripe/
    └── rotten/
```

### 2.3 Built-In Augmentations (Train Split Only)

- **Multiplier:** 3 augmented variations per original training image
- **Spatial:** Horizontal/Vertical flips, 90°/180°/270° rotations, random rotations (±15°), random zoom crop (0–20%)
- **Photometric:** Hue (±10°), Saturation (±10%), Brightness (±10%), Exposure (±10%), Gaussian Blur (up to 1px)

### 2.4 Continuous Target Synthesis Strategy

The 4 discrete class folders are mapped to continuous regression targets $y \in [0.0, 1.0]$:

```python
TARGET_MAPPING = {
    "unripe":   (0.00, 0.25),   # y ~ Uniform(0.00, 0.25)
    "ripe":     (0.25, 0.50),   # y ~ Uniform(0.25, 0.50)
    "overripe": (0.50, 0.75),   # y ~ Uniform(0.50, 0.75)
    "rotten":   (0.75, 1.00),   # y ~ Uniform(0.75, 1.00)
}
```

Each image receives a random continuous score drawn uniformly within its class band, converting the discrete classification dataset into a continuous regression target.

### 2.5 Automated Ingestion

```bash
# Kaggle CLI download
kaggle datasets download -d shahriar26s/banana-ripeness-classification-dataset --unzip -p ./data/raw
```

```python
# Programmatic download (src/data/download.py)
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    'shahriar26s/banana-ripeness-classification-dataset',
    path='data/raw',
    unzip=True
)
```

---

## 3. Model Architecture

### 3.1 Architecture Specification

| Parameter | Value |
|---|---|
| **Backbone** | EfficientNet-B0 or ResNet18 (transfer learning, ImageNet pretrained) |
| **Input Resolution** | 224 × 224 × 3 (RGB, resized from 416 × 416) |
| **Head** | Custom regression head: `AdaptiveAvgPool2d → Dropout(0.3) → Linear(512, 1) → Sigmoid` |
| **Output** | Single continuous scalar $y \in [0.0, 1.0]$ |
| **Loss Function** | Huber Loss (robust to label noise from synthetic continuous mapping) |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max=epochs) |
| **Epochs** | 15–20 (baseline training) / 5 (fine-tuning after drift) |
| **Batch Size** | 32 |

### 3.2 Transfer Learning Strategy

- **Baseline Training:** Backbone layers frozen for first 5 epochs (feature extraction only), then unfrozen with reduced lr (1e-5) for end-to-end fine-tuning.
- **Drift Fine-Tuning:** Only regression head layers unfrozen. Backbone remains frozen. Learning rate = 1e-4. Runs in under 3 minutes on CPU.

### 3.3 Inference Optimization

- Model exported to **TorchScript (`.pt`)** via `torch.jit.trace()` for CPU-optimized inference on EC2 `t3.small`.
- Target inference latency: $< 100\text{ms}$ per image on CPU.

### 3.4 PyTorch Preprocessing Pipeline

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 4. Multi-Tier Evaluation Metrics Strategy

### 4.1 Primary Regression Metrics (MLflow Tracked)

| Metric | Formula | Purpose |
|---|---|---|
| **MAE (Headline)** | $\frac{1}{N}\sum_{i=1}^{N} \lvert y_i - \hat{y}_i \rvert$ | Primary model quality gate. Target: $< 0.05$ |
| **Huber Loss** | Smooth L1 loss (training objective) | Robust against label noise outliers |
| **RMSE** | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$ | Penalizes large prediction errors |

### 4.2 Derived Business Metrics (MLflow Tracked)

| Metric | Method | Purpose |
|---|---|---|
| **Binned Confusion Matrix** | Map continuous $\hat{y}$ back to 4 ordinal bins, compare against true bin | Verifies errors stay between adjacent stages |
| **Adjacent-Stage F1-Score** | Macro F1 after binning | Business accuracy metric |

### 4.3 Drift & Monitoring Metrics (Evidently AI)

| Metric | Statistical Test | Drift Alert Threshold |
|---|---|---|
| **Prediction Score Drift** | Kolmogorov-Smirnov (KS) Test | $p\text{-value} < 0.05$ |
| **Model Decay (MAE)** | MAE on production batch | $\text{MAE}_{\text{prod}} > 0.08$ (or $> 20\%$ decay) |
| **Image Feature Drift** | HSL Hue / Brightness distribution shift | Drift score $> 0.25$ |

### 4.4 Derived Shelf-Life Business Metric

The continuous ML score is transformed to a user-facing shelf-life estimate in FastAPI:

$$S_{\text{remaining}} = 12 \times (1.0 - y) \text{ days}$$

> [!IMPORTANT]
> Shelf-life is strictly a **business-layer post-processing transformation** inside FastAPI/Streamlit. The ML model's native target remains $y \in [0.0, 1.0]$ trained with Huber Loss. The model never directly predicts days.

| Continuous Score Range | Qualitative Category | Shelf-Life Estimate | Recommended Action |
|---|---|---|---|
| `0.00 – 0.25` | **Unripe (Green)** | 9–12 days | Store in warehouse |
| `0.26 – 0.50` | **Slightly Ripe** | 6–9 days | Ready for retail distribution |
| `0.51 – 0.75` | **Ripe** | 3–6 days | Place on store shelves immediately |
| `0.76 – 1.00` | **Overripe / Rotten** | 0–3 days | Discount / discard |

---

## 5. Finalized Tech Stack (AWS Free Tier Compliant)

| Subsystem | Technology | Configuration |
|---|---|---|
| **Model Framework** | PyTorch / PyTorch Lightning | EfficientNet-B0 / ResNet18, Huber Loss, MAE |
| **Data Versioning** | DVC + AWS S3 | Remote backend on S3 bucket |
| **Experimentation & Registry** | MLflow | Tracking server in Docker, artifacts to S3, metadata to SQLite |
| **Orchestration** | Apache Airflow | LocalExecutor, scheduled monitoring & retrain DAGs |
| **Drift Monitoring** | Evidently AI | KS-test, HSL drift, HTML report generation |
| **Serving API** | FastAPI + Uvicorn + Pydantic | `multipart/form-data` with input guardrails |
| **Frontend** | Streamlit | 2-tab UI: Prediction + MLOps Telemetry Dashboard |
| **Container Registry** | AWS ECR | 500 MB free storage for Docker images |
| **Compute Host** | AWS EC2 `t3.small` | 2 vCPU, 2 GB RAM, Ubuntu 22.04, 1 GB swap file |
| **Object Storage** | AWS S3 | DVC datasets + MLflow artifacts |
| **CI/CD** | GitHub Actions | OIDC keyless auth, build/push ECR, deploy to EC2 |
| **Linting** | Ruff | Replaces flake8 + isort + black |
| **Testing** | Pytest + httpx.AsyncClient | Async API tests, mocked inference |
| **Packaging** | pyproject.toml + Hatchling | PEP 517/518 modern build system |

---

## 6. Project Directory Structure

```
banana-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Lint + test on every PR
│       └── deploy.yml                # Build → ECR → EC2 deployment
├── aws/                              # AWS infrastructure docs & deployment journals
├── config/
│   └── settings.py                   # Pydantic Settings (12-Factor App)
├── dags/
│   ├── drift_monitoring_dag.py       # DAG 1: Scheduled Evidently AI drift check
│   └── retrain_model_dag.py          # DAG 2: Auto-retrain triggered by drift
├── data/                             # DVC-tracked (gitignored)
│   ├── raw/                          # Original Kaggle download
│   ├── processed/
│   │   ├── baseline_v1/              # Clean reference dataset
│   │   └── perturbed_v2/             # Synthetic drift dataset
│   ├── replay_buffer/                # 80:20 mix for fine-tuning
│   └── production_logs/              # Logged inference images & predictions
│       ├── images/
│       └── predictions.csv
├── models/                           # Serialized model artifacts (.pt)
├── reports/                          # Evidently AI HTML drift reports
├── src/
│   └── banana_mlops/                 # Installable Python package
│       ├── __init__.py
│       ├── data/
│       │   ├── download.py           # Kaggle API dataset ingestion
│       │   ├── make_dataset.py       # Continuous target synthesis
│       │   ├── drift_generator.py    # Synthetic perturbation engine
│       │   └── build_replay_buffer.py
│       ├── models/
│       │   ├── architecture.py       # ResNet18/EfficientNet backbone + head
│       │   ├── train.py              # Training loop with MLflow logging
│       │   └── evaluate_gate.py      # Quality gate: candidate MAE < prod MAE
│       ├── serving/
│       │   ├── app.py                # FastAPI application & /predict endpoint
│       │   ├── guardrails.py         # MIME, size, PIL integrity validation
│       │   └── schemas.py            # Pydantic request/response models
│       └── utils/
│           ├── seed.py               # seed_everything(42) utility
│           └── logger.py             # Structured logging setup
├── frontend/
│   ├── app.py                        # Streamlit 2-tab UI
│   ├── Dockerfile                    # Streamlit container build
│   └── requirements.txt
├── backend/
│   └── Dockerfile                    # FastAPI container build
├── tests/
│   ├── test_guardrails.py            # Input validation unit tests
│   ├── test_prediction.py            # API endpoint integration tests
│   ├── test_label_synthesis.py       # Continuous target mapping tests
│   └── conftest.py                   # Shared fixtures & async client setup
├── scripts/
│   └── deploy.sh                     # EC2 runtime deployment script
├── docker-compose.yml                # Full local stack orchestration
├── pyproject.toml                    # PEP 517/518 build config (Hatchling)
├── .env.example                      # Environment variable template
├── .gitignore                        # Protects venv, .env, data/, models/, caches
├── .dvc/                             # DVC configuration
├── Dvcfile                           # DVC pipeline definition
└── README.md
```

---

## 7. Architectural Components (Detailed Design)

---

### Component 1: Data & Feature Pipeline

```
[ Kaggle API ] → [ download.py ] → [ data/raw/ ] → [ make_dataset.py ] → [ data/processed/baseline_v1/ ]
                                                                                     │
                                                      ┌─────────────────────────────┘
                                                      ▼
                                            [ drift_generator.py ] → [ data/processed/perturbed_v2/ ]
                                                      │
                                                      ▼
                                            [ build_replay_buffer.py ] → [ data/replay_buffer/ ]
                                                                                     │
                                                                                     ▼
                                                                              [ DVC push → AWS S3 ]
```

**Drift Simulation Engine (Synthetic Perturbations):**

| Perturbation Type | Transformation | Simulated Real-World Cause |
|---|---|---|
| Color Temperature Shift | HSL Hue adjustment (ΔH = +15°) | New camera sensor / different white balance |
| Lighting Decay | Brightness scale (0.7×) | Dim lighting in warehouse sorting facility |
| Compression & Blur | Gaussian Blur (σ=1.5) + JPEG compression | Low-bandwidth video stream transmission |

**Replay Buffer Construction:**
- 800 newly labeled images sampled from perturbed production distribution (80%)
- 200 clean images sampled from `baseline_v1` (20%)
- Combined into `data/replay_buffer/retrain_vX/` to prevent catastrophic forgetting

**DVC Tracking:**
```bash
dvc remote add -d s3store s3://banana-mlops-bucket/dvc-store
dvc add data/processed/baseline_v1
dvc push
```

---

### Component 2: FastAPI Serving & Input Guardrail Architecture

```
[ Client Request ] → [ FastAPI POST /predict ] → [ Input Guardrails ] → [ TorchScript Inference ] → [ Response ]
                                                                                                          │
                                                                                                          ▼
                                                                                              [ Async Background Logger ]
                                                                                              (Save to production_logs/)
```

**API Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | `POST` | Image inference with guardrails (multipart/form-data) |
| `/health` | `GET` | Container health probe for Docker / EC2 |
| `/metrics` | `GET` | Prometheus-formatted telemetry |

**Input Guardrails (`src/banana_mlops/serving/guardrails.py`):**
1. **MIME Type Check:** Only `image/jpeg`, `image/png`, `image/webp` accepted
2. **File Size Bounds:** 5 KB minimum, 10 MB maximum
3. **PIL Header Integrity:** `Image.open().verify()` to detect corrupted payloads
4. **RGB Normalization:** Grayscale/RGBA automatically converted to 3-channel RGB

**Response Schema:**
```json
{
    "spoilage_score": 0.3812,
    "category": "Slightly Ripe",
    "shelf_life_days": 7.4,
    "model_version": "v1.0.0",
    "latency_ms": 45.2
}
```

**Production Logging:** Every prediction is asynchronously logged (image + score + timestamp) to `data/production_logs/` via FastAPI `BackgroundTasks`. This log store feeds the Airflow monitoring DAG.

---

### Component 3: Airflow Orchestration & Evidently AI Drift Engine

```
[ Airflow Scheduler ] → [ DAG 1: Drift Monitoring (Daily) ] → [ Evidently AI Drift Check ]
                                                                        │
                                                           Drift Detected?
                                                          ├── NO → Log Healthy & End
                                                          └── YES → [ DAG 2: Auto-Retrain ]
                                                                           │
                                                                           ▼
                                                              [ Build Replay Buffer (80:20) ]
                                                                           │
                                                                           ▼
                                                              [ Fine-Tune PyTorch + Log MLflow ]
                                                                           │
                                                                           ▼
                                                              [ Quality Gate: Candidate MAE < Prod MAE? ]
                                                                           │
                                                                ├── FAIL → Reject Candidate
                                                                └── PASS → Promote to MLflow Production
```

**DAG 1: Scheduled Drift Monitoring (`dags/drift_monitoring_dag.py`):**
- Runs `@daily` via Airflow scheduler
- Uses `BranchPythonOperator` to evaluate Evidently AI drift report
- Compares latest 500 production predictions against reference baseline
- Generates HTML report saved to `reports/evidently_drift_report.html` for Streamlit dashboard
- Triggers DAG 2 via `TriggerDagRunOperator` if drift threshold breached

**DAG 2: Automated Retraining (`dags/retrain_model_dag.py`):**
- `schedule_interval=None` (only triggered programmatically by DAG 1)
- Step 1: Build replay buffer (80% perturbed + 20% baseline)
- Step 2: Fine-tune model (5 epochs, lr=1e-4, backbone frozen, head unfrozen)
- Step 3: Evaluate quality gate and promote/reject via MLflow Registry

**Airflow Configuration for EC2 `t3.small`:**
- Executor: `LocalExecutor` or `SequentialExecutor` (lightweight, single-node)
- Heavy background plugins disabled to conserve 2 GB RAM budget

---

### Component 4: MLflow Model Registry & CI/CD Deployment Loop

```
[ Retrain Complete ] → [ MLflow Log Metrics ] → [ Quality Gate ] → [ MLflow Registry Promotion ]
                                                                              │
                                                                              ▼
                                                               [ GitHub Actions CI/CD Pipeline ]
                                                                              │
                                                         ┌────────────────────┴────────────────────┐
                                                         ▼                                         ▼
                                                [ Build & Push to ECR ]                   [ Deploy to EC2 ]
                                                                                                   │
                                                                                                   ▼
                                                                                          [ Health Check Probe ]
                                                                                          ├── 200 OK → Done!
                                                                                          └── Fail → Rollback
```

**MLflow Model Registry Lifecycle:**

```
[ New Model Logged ] → Staging → [ Quality Gate: val_mae < prod_mae ] → Production → [ Old Model ] → Archived
```

**Quality Gate Logic (`src/banana_mlops/models/evaluate_gate.py`):**
- Fetches latest `Staging` model from MLflow Registry
- Compares candidate `val_mae` against active `Production` model `val_mae`
- If candidate wins: transition candidate → `Production`, old model → `Archived`
- If candidate loses: reject candidate, log alert, keep existing production model

**Immutable Docker Image Tagging:**
```bash
docker build -t $ECR_REGISTRY/banana-fastapi:${{ github.sha }} -t $ECR_REGISTRY/banana-fastapi:latest ./backend
```

---

### Component 5: Streamlit Frontend Architecture

```
┌─────────────────────────┐     HTTP POST      ┌─────────────────────────┐
│   Streamlit Frontend    │ ──────────────────► │   FastAPI Backend       │
│   (Port 8501)           │ ◄────────────────── │   (Port 8000)           │
│                         │     JSON Response   │                         │
└─────────────────────────┘                     └─────────────────────────┘
```

> [!IMPORTANT]
> Streamlit acts **only** as an HTTP client. Model inference (`torch.jit.load`, `model.forward()`) is **never** executed inside Streamlit code. This preserves microservice separation of concerns.

**Tab 1: Banana Ripeness & Shelf-Life Predictor**
- Input: `st.file_uploader("Upload Banana Image")` or `st.camera_input("Snap Photo")`
- Output: `st.metric("Ripeness Score", ...)`, `st.metric("Est. Shelf-Life", ...)`, `st.metric("Recommended Action", ...)`
- Visual: Color progress bar (HSL gradient gauge)
- Footer: Model version + inference latency

**Tab 2: MLOps Telemetry & Drift Monitor**
- Evidently AI embedded HTML report: `st.components.v1.html(evidently_report_html)`
- MLflow active model metadata (model tag, validation MAE)
- Airflow DAG run status and trigger logs

---

## 8. Three-Phase Progressive Deployment Roadmap

---

### Phase 1: Local-First Prototype & MLOps Loop Validation (100% Local)

**Environment:** Local machine (Python virtualenv + Docker Compose)

**Goal:** Verify the complete MLOps lifecycle loop works end-to-end before touching any cloud service.

**Deliverables:**
- [ ] Kaggle dataset download & continuous target synthesis
- [ ] PyTorch ResNet18/EfficientNet-B0 baseline training with MLflow local tracking
- [ ] Synthetic drift generation (color shift, blur, compression)
- [ ] Local Evidently AI drift detection report
- [ ] Local Airflow DAGs: scheduled monitoring + auto-retrain
- [ ] MAE degradation → auto-retrain → MAE recovery (headline result table)
- [ ] FastAPI serving container with TorchScript inference + input guardrails
- [ ] Streamlit frontend container (2-tab UI)
- [ ] Full `docker-compose.yml` running all services locally
- [ ] DVC tracking with local remote (before S3 migration)
- [ ] Pytest test suite passing (guardrails, label synthesis, API endpoints)
- [ ] Ruff linting clean

**Local Docker Compose Stack:**
```yaml
services:
  fastapi-backend:
    build: ./backend
    ports: ["8000:8000"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  streamlit-frontend:
    build: ./frontend
    ports: ["8501:8501"]
    environment:
      - FASTAPI_URL=http://fastapi-backend:8000
    depends_on:
      fastapi-backend:
        condition: service_healthy

  mlflow-server:
    image: python:3.10-slim
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
    ports: ["5000:5000"]
```

---

### Phase 2: Cloud Migration & OIDC + SSH CI/CD

**Environment:** AWS (S3, ECR, EC2 `t3.small`) + GitHub Actions

**Goal:** Migrate data/artifacts to AWS and automate deployments with enterprise-grade keyless authentication, while keeping SSH as the deployment transport.

**Deliverables:**
- [ ] Migrate DVC remote from local to AWS S3 bucket
- [ ] Migrate MLflow artifact store to AWS S3
- [ ] Setup AWS IAM OIDC Identity Provider for GitHub Actions
- [ ] Create least-privilege IAM Role (`GitHubActionsBananaMLOpsRole`) with policies scoped to ECR push, S3 access, and EC2 SSH
- [ ] Store EC2 SSH private key in AWS Secrets Manager (fetched dynamically in CI/CD)
- [ ] GitHub Actions workflow: lint → test → build → push ECR → SSH deploy to EC2
- [ ] Immutable Docker image tagging with Git SHA + `latest`
- [ ] `.env.example` committed; production secrets fetched from AWS Secrets Manager via `deploy.sh`
- [ ] EC2 Security Group: Port 22 open (restricted to GitHub Actions IP ranges), Port 8000 + 8501 open
- [ ] Health check probe after deployment with rollback on failure

**GitHub Actions Workflow (Phase 2):**
```yaml
permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    steps:
      - name: Configure AWS Credentials via OIDC
        uses: aws-actions/configure-aws-credentials@v3
        with:
          role-to-assume: arn:aws:iam::${{ secrets.AWS_ACCOUNT_ID }}:role/GitHubActionsBananaMLOpsRole
          aws-region: us-east-1

      - name: Fetch SSH Key from AWS Secrets Manager
        run: |
          aws secretsmanager get-secret-value --secret-id banana-mlops/ec2-ssh-key \
            --query SecretString --output text > /tmp/ec2_key.pem
          chmod 600 /tmp/ec2_key.pem

      - name: Deploy via SSH
        run: |
          ssh -i /tmp/ec2_key.pem -o StrictHostKeyChecking=no ubuntu@${{ secrets.EC2_PUBLIC_IP }} \
            'cd /home/ubuntu/banana-mlops && bash deploy.sh'
```

---

### Phase 3: Zero-Trust Enterprise Security & Production Hardening

**Environment:** AWS (EC2 + SSM Agent + IAM Instance Profile)

**Goal:** Eliminate all SSH exposure. Achieve a fully hardened, zero-trust production deployment.

**Deliverables:**
- [ ] Install and configure AWS SSM Agent on EC2 instance
- [ ] Attach IAM Instance Profile to EC2 with SSM permissions + ECR pull + S3 access
- [ ] **CLOSE PORT 22 COMPLETELY** on EC2 Security Group
- [ ] Replace SSH deployment step with `aws ssm send-command` in GitHub Actions
- [ ] Runtime secrets injected via IAM Instance Profile (EC2 fetches from Secrets Manager directly)
- [ ] AWS CloudTrail audit logging for every OIDC session and SSM command invocation
- [ ] Final observability dashboards (optional: Prometheus + Grafana)

**GitHub Actions Workflow (Phase 3 — Zero SSH):**
```yaml
      - name: Deploy via AWS SSM (Port 22 Closed)
        run: |
          COMMAND_ID=$(aws ssm send-command \
            --instance-ids "${{ secrets.EC2_INSTANCE_ID }}" \
            --document-name "AWS-RunShellScript" \
            --comment "Deploy banana-mlops containers" \
            --parameters 'commands=[
              "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ECR_REGISTRY>",
              "cd /home/ubuntu/banana-mlops",
              "docker compose pull",
              "docker compose up -d --remove-orphans",
              "docker image prune -f"
            ]' \
            --query "Command.CommandId" --output text)

          aws ssm wait command-executed \
            --command-id "$COMMAND_ID" \
            --instance-id "${{ secrets.EC2_INSTANCE_ID }}"
```

**Security Comparison Across Phases:**

| Security Dimension | Phase 1 (Local) | Phase 2 (OIDC + SSH) | Phase 3 (Zero-Trust SSM) |
|---|---|---|---|
| AWS Auth | N/A | OIDC short-lived tokens | OIDC short-lived tokens |
| Server Access | localhost | SSH (Port 22 open, key in Secrets Manager) | **SSM (Port 22 CLOSED)** |
| Secrets Storage | `.env` local file | AWS Secrets Manager | IAM Instance Profile |
| Audit Trail | Git log | CloudTrail (partial) | **Full CloudTrail + SSM logs** |

---

## 9. AWS Infrastructure Specification

| Resource | Configuration | Free Tier / Cost |
|---|---|---|
| **EC2 Instance** | `t3.small` (2 vCPU, 2 GB RAM, 20 GB EBS SSD gp3) | ~$0.02/hr (start/stop as needed) |
| **EC2 OS** | Ubuntu 22.04 LTS + Docker Engine + 1 GB Swap | — |
| **ECR** | 2 repositories (`banana-fastapi`, `banana-streamlit`) | 500 MB free storage |
| **S3 Bucket** | `banana-mlops-bucket` (DVC datasets + MLflow artifacts) | 5 GB free storage |
| **IAM** | OIDC Provider + `GitHubActionsBananaMLOpsRole` + EC2 Instance Profile | Free |
| **Secrets Manager** | SSH key (Phase 2), runtime env vars | $0.40/secret/month |
| **SSM (Phase 3)** | Systems Manager Agent on EC2 | Free |

---

## 10. Standard Engineering Practices (Inherited from PackVote)

### 10.1 Modern Packaging (`pyproject.toml` + Hatchling)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "banana-mlops"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "torchvision",
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "Pillow",
    "mlflow",
    "evidently",
    "pydantic>=2.0",
    "pydantic-settings",
    "python-dotenv",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx", "ruff"]
```

### 10.2 Configuration via `pydantic-settings` (12-Factor App)

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "models/production_model.pt"
    mlflow_tracking_uri: str = "http://localhost:5000"
    s3_bucket: str = "banana-mlops-bucket"
    fastapi_port: int = 8000
    max_shelf_life_days: float = 12.0

    class Config:
        env_file = ".env"
```

### 10.3 `.env.example` (Committed Template)

```env
# Application
MODEL_PATH=models/production_model.pt
MLFLOW_TRACKING_URI=http://localhost:5000
S3_BUCKET=banana-mlops-bucket
FASTAPI_PORT=8000

# AWS (production only — fetched from Secrets Manager)
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=
```

### 10.4 Lazy Imports for Fast Test Discovery

```python
# Heavy imports loaded lazily inside functions, not at module top-level
def load_model(path: str):
    import torch  # Lazy import — avoids 10s pytest discovery hang
    return torch.jit.load(path, map_location="cpu")
```

### 10.5 Deterministic Reproducibility

```python
# src/banana_mlops/utils/seed.py
import random, os, numpy as np, torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 10.6 Static Analysis & Linting (`ruff`)

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I", "W"]
```

### 10.7 Async Testing (`pytest` + `httpx.AsyncClient`)

```python
# tests/conftest.py
import pytest
from httpx import AsyncClient, ASGITransport
from banana_mlops.serving.app import app

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

### 10.8 Container Health Probes & Dependency Ordering

All `docker-compose.yml` services use health checks with `depends_on: condition: service_healthy` to prevent startup race conditions.

### 10.9 Immutable Git SHA Image Tags

Every Docker image is dual-tagged (`${{ github.sha }}` + `latest`) enabling instant rollback to any specific commit.

### 10.10 Comprehensive `.gitignore`

Protects: `env/`, `.env`, `data/`, `models/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, `mlartifacts/`, `*.pt`, `*.onnx`

---

## 11. Verification Plan

### 11.1 Automated Tests (CI/CD Gate)

```bash
# Run full test suite
pytest tests/ --doctest-modules -v

# Run linting
ruff check src/ tests/
```

| Test File | What It Validates |
|---|---|
| `test_guardrails.py` | MIME rejection, file size bounds, corrupted image handling, RGBA→RGB conversion |
| `test_prediction.py` | `/predict` returns valid JSON with `spoilage_score`, `category`, `shelf_life_days` |
| `test_label_synthesis.py` | Continuous target mapping stays within class bounds |
| `test_health.py` | `/health` returns `{"status": "healthy", "model_loaded": true}` |

### 11.2 Manual / Visual Verification

- [ ] Upload banana images of each ripeness stage via Streamlit → verify continuous scores and categories are intuitive
- [ ] Inject synthetic drift images → verify Evidently AI report shows drift detected
- [ ] Trigger retraining DAG → verify MAE recovers in MLflow tracking dashboard
- [ ] Push to `main` branch → verify GitHub Actions pipeline builds, pushes to ECR, and deploys to EC2
- [ ] Verify Streamlit Tab 2 (MLOps Dashboard) displays Evidently report and MLflow model metadata
- [ ] (Phase 3) Verify Port 22 is closed and SSM deployment succeeds

### 11.3 Headline Result Validation

The final portfolio deliverable is the **before / after / recovered MAE table** demonstrating the complete drift → detect → retrain → recover lifecycle.

---

## 12. Open Questions

> [!IMPORTANT]
> **Airflow on EC2 `t3.small`:** Airflow is memory-heavy (~700 MB). With FastAPI + Streamlit + MLflow also running, the 2 GB budget is tight. If memory pressure becomes a problem during development, we may need to:
> - Run Airflow DAGs as standalone Python scripts triggered by cron (simpler, lighter)
> - Upgrade to `t3.medium` (4 GB RAM, ~$0.04/hr)
>
> **Decision:** Start with `t3.small` + 1 GB swap. Upgrade only if OOM issues arise.

> [!NOTE]
> **Prometheus + Grafana (Phase 3 optional):** Listed in the tech stack for operational telemetry. These are optional polish items — the core MLOps story is complete without them. Add only if time permits after the 3-phase roadmap is finished.
