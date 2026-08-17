"""FastAPI serving application for Banana Ripeness continuous regression."""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config.settings import get_settings
from src.banana_mlops.data.make_dataset import get_transforms
from src.banana_mlops.serving.guardrails import (
    compute_shelf_life_and_category,
    validate_image_payload,
)
from src.banana_mlops.serving.schemas import HealthResponse, PredictionResponse
from src.banana_mlops.utils.logger import setup_logger

logger = setup_logger("banana_mlops.serving.app")

# Global serving state
state: dict = {
    "model": None,
    "transform": None,
    "settings": None,
    "model_loaded": False,
    "total_predictions": 0,
}


def log_production_prediction(
    image: Image.Image,
    score: float,
    category: str,
    latency_ms: float,
) -> None:
    """Asynchronously log production inference image and metadata for drift tracking."""
    try:
        log_dir = Path("data/production_logs/images")
        log_dir.mkdir(parents=True, exist_ok=True)

        req_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        img_filename = f"{timestamp}_{req_id}.jpg"
        img_path = log_dir / img_filename
        image.save(img_path, format="JPEG", quality=85)

        csv_path = Path("data/production_logs/predictions.csv")
        file_exists = csv_path.exists()

        with open(csv_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write(
                    "timestamp,image_path,spoilage_score,category,latency_ms\n"
                )
            f.write(
                f"{timestamp},{img_path.resolve()},{score:.4f},{category},{latency_ms:.2f}\n"
            )
    except Exception as e:
        logger.error(f"Failed to log production prediction asynchronously: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager loading model and warm-up."""
    settings = get_settings()
    state["settings"] = settings
    state["transform"] = get_transforms("test")

    model_path = Path(settings.model_path)
    if model_path.exists():
        try:
            logger.info(
                f"Loading production TorchScript model from {model_path} on CPU..."
            )
            model = torch.jit.load(str(model_path), map_location="cpu")
            model.eval()
            state["model"] = model
            state["model_loaded"] = True

            # Warmup inference
            dummy = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = model(dummy)
            logger.info("TorchScript model loaded and warmed up successfully.")
        except Exception as e:
            logger.error(f"Error loading TorchScript model: {e}")
            state["model_loaded"] = False
    else:
        logger.warning(
            f"Production model not found at {model_path}. Serving will operate in degraded mode."
        )
        state["model_loaded"] = False

    yield
    logger.info("Shutting down FastAPI serving application.")


app = FastAPI(
    title="Banana Ripeness & Spoilage Regression API",
    description=(
        "Production MLOps continuous regression service with input guardrails "
        "and telemetry logging."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check probe endpoint for Docker and cloud orchestration."""
    settings = get_settings()
    return HealthResponse(
        status="healthy" if state.get("model_loaded") else "degraded",
        model_loaded=bool(state.get("model_loaded")),
        model_path=settings.model_path,
        device="cpu",
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Service telemetry metrics endpoint."""
    return {
        "status": "online",
        "total_predictions": state.get("total_predictions", 0),
        "model_loaded": state.get("model_loaded", False),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_spoilage(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Predict continuous banana spoilage score and remaining shelf life with input guardrails."""
    start_time = time.perf_counter()

    # Read binary payload
    payload = await file.read()

    # Input guardrail validation
    verified_img = validate_image_payload(
        payload, content_type=file.content_type
    )

    # Transform image for TorchScript inference
    transform = state.get("transform") or get_transforms("test")
    input_tensor = transform(verified_img).unsqueeze(0)

    # Perform model inference
    model = state.get("model")
    if model is None:
        model_path = Path(get_settings().model_path)
        if model_path.exists():
            model = torch.jit.load(str(model_path), map_location="cpu")
            model.eval()
            state["model"] = model
            state["model_loaded"] = True

    if model is not None:
        with torch.no_grad():
            raw_score = float(model(input_tensor).item())
    else:
        raw_score = 0.50  # Fallback default

    # Business post-processing (Shelf-life days & qualitative category)
    settings = get_settings()
    category, shelf_life, action = compute_shelf_life_and_category(
        score=raw_score,
        max_days=settings.max_shelf_life_days,
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    state["total_predictions"] = state.get("total_predictions", 0) + 1

    # Asynchronous production inference logging
    background_tasks.add_task(
        log_production_prediction,
        image=verified_img,
        score=raw_score,
        category=category,
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        spoilage_score=round(raw_score, 4),
        category=category,
        shelf_life_days=shelf_life,
        recommended_action=action,
        model_version="v1.0.0",
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.banana_mlops.serving.app:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=True,
    )
