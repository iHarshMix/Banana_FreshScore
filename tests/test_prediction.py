"""Integration tests for FastAPI endpoints (/health, /metrics, /predict)."""

import io

import pytest
from PIL import Image


def create_test_image_bytes() -> bytes:
    """Create a 15 KB JPEG test image."""
    img = Image.new("RGB", (224, 224), color=(255, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()
    if len(data) < 10 * 1024:
        data = data + b"\x00" * (10 * 1024 - len(data))
    return data


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "total_predictions" in data


@pytest.mark.asyncio
async def test_predict_endpoint_success(client):
    img_bytes = create_test_image_bytes()
    files = {"file": ("test_banana.jpg", img_bytes, "image/jpeg")}

    response = await client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "spoilage_score" in data
    assert 0.0 <= data["spoilage_score"] <= 1.0
    assert "category" in data
    assert "shelf_life_days" in data
    assert 0.0 <= data["shelf_life_days"] <= 12.0
    assert "latency_ms" in data
    assert "recommended_action" in data


@pytest.mark.asyncio
async def test_predict_endpoint_invalid_file(client):
    files = {"file": ("test.txt", b"invalid payload text", "text/plain")}
    response = await client.post("/predict", files=files)
    assert response.status_code == 415
