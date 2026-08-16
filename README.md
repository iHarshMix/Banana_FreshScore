# 🍌 Banana Ripeness MLOps Regression Pipeline (FreshScore)

[![CI](https://github.com/iHarshMix/Banana_FreshScore/actions/workflows/ci.yml/badge.svg)](https://github.com/iHarshMix/Banana_FreshScore/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An end-to-end MLOps-complete computer vision regression pipeline predicting continuous banana spoilage scores from RGB imagery, featuring automated drift detection, replay-buffer retraining, continuous quality gates, and zero-trust cloud deployment.

---

## 🎯 Portfolio Narrative & Headline Result

This project proves **MLOps maturity after deployment**: detecting model decay in production and recovering automatically without human intervention.

| Stage | MAE | Status | Description |
|---|---|---|---|
| **Baseline Model** | ~0.04 | ✅ Healthy | Clean reference camera distribution |
| **After Drift** | ~0.12 | ⚠️ Degraded | Cooler color temp ($\Delta H = +15^\circ$), lighting drop ($0.7\times$), blur |
| **After Auto-Retrain** | ~0.05 | ✅ Recovered | Automated Airflow + Evidently AI + 80:20 Replay Buffer |

---

## 🏗️ Architecture Overview

```
[ Kaggle Ingestion ] ──> [ Target Synthesis ] ──> [ PyTorch ResNet18/EfficientNet ] ──> [ TorchScript (.pt) ]
                                                                                              │
                                                                                              ▼
[ Streamlit UI ] ──────HTTP POST──────> [ FastAPI Serving + Guardrails ] ─────────> [ Production Logger ]
                                                                                              │
                                                                                              ▼
[ MLflow Registry ] <── [ Quality Gate ] <── [ Retrain DAG ] <── [ Drift DAG + Evidently AI ]
```

---

## 🚀 Quickstart (Local Development)

### 1. Environment Setup
```bash
git clone https://github.com/iHarshMix/Banana_FreshScore.git
cd Banana_FreshScore
conda create -p ./.venv python=3.10 pip -y
conda activate ./.venv
pip install -e ".[dev]"
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env to add Kaggle credentials
```

### 3. Run Quality Checks
```bash
ruff check src/ tests/
pytest tests/ -v
```

---

## 📁 Repository Structure

```
├── .github/workflows/       # CI and CD workflows (OIDC + AWS SSM)
├── config/                  # 12-factor Pydantic settings
├── dags/                    # Apache Airflow drift monitoring & retrain DAGs
├── data/                    # DVC tracked dataset tiers (raw, processed, replay)
├── frontend/                # Streamlit 2-tab interactive dashboard
├── backend/                 # FastAPI serving application & Dockerfile
├── models/                  # Serialized TorchScript model artifacts
├── reports/                 # Evidently AI HTML drift reports
├── src/banana_mlops/        # Core installable Python package
│   ├── data/                # Ingestion, synthesis, synthetic drift engine
│   ├── models/              # Architecture, training loop, quality gates
│   ├── serving/             # Endpoints, guardrails, schemas
│   └── utils/               # Seed, structured logger
└── tests/                   # Pytest unit and integration test suite
```
