"""Unit tests for FastAPI input guardrails and business transformations."""

import io

import pytest
from fastapi import HTTPException
from PIL import Image

from src.banana_mlops.serving.guardrails import (
    compute_shelf_life_and_category,
    validate_image_payload,
)


def create_sample_image_bytes(
    size_kb: int = 10, fmt: str = "JPEG", mode: str = "RGB"
) -> bytes:
    """Helper creating synthetic image payload of a given size."""
    img = Image.new(mode, (200, 200), color=(255, 255, 0))
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    data = buffer.getvalue()
    if len(data) < size_kb * 1024:
        # Pad with dummy exif/metadata
        data = data + b"\x00" * ((size_kb * 1024) - len(data))
    return data


def test_valid_image_passes_guardrails():
    payload = create_sample_image_bytes(size_kb=15, fmt="JPEG")
    verified = validate_image_payload(payload, content_type="image/jpeg")
    assert isinstance(verified, Image.Image)
    assert verified.mode == "RGB"


def test_unsupported_mime_type_rejected():
    payload = create_sample_image_bytes(size_kb=15, fmt="JPEG")
    with pytest.raises(HTTPException) as exc_info:
        validate_image_payload(payload, content_type="application/pdf")
    assert exc_info.value.status_code == 415


def test_payload_too_small_rejected():
    tiny_payload = b"tiny"
    with pytest.raises(HTTPException) as exc_info:
        validate_image_payload(tiny_payload, content_type="image/jpeg")
    assert exc_info.value.status_code == 400


def test_payload_too_large_rejected():
    huge_payload = b"\x00" * (11 * 1024 * 1024)  # 11 MB
    with pytest.raises(HTTPException) as exc_info:
        validate_image_payload(huge_payload, content_type="image/jpeg")
    assert exc_info.value.status_code == 413


def test_corrupted_image_rejected():
    corrupted = b"RIFF" + b"\x00" * 6000  # Fake header with 6 KB zeroes
    with pytest.raises(HTTPException) as exc_info:
        validate_image_payload(corrupted, content_type="image/jpeg")
    assert exc_info.value.status_code == 400


def test_rgba_converted_to_rgb():
    payload = create_sample_image_bytes(size_kb=15, fmt="PNG", mode="RGBA")
    verified = validate_image_payload(payload, content_type="image/png")
    assert verified.mode == "RGB"


def test_shelf_life_categories():
    # Unripe
    cat, days, act = compute_shelf_life_and_category(0.10)
    assert cat == "Unripe (Green)"
    assert days >= 9.0

    # Slightly Ripe
    cat, days, act = compute_shelf_life_and_category(0.35)
    assert cat == "Slightly Ripe"
    assert 6.0 <= days <= 9.0

    # Ripe
    cat, days, act = compute_shelf_life_and_category(0.65)
    assert cat == "Ripe"
    assert 3.0 <= days <= 6.0

    # Overripe
    cat, days, act = compute_shelf_life_and_category(0.90)
    assert cat == "Overripe / Rotten"
    assert days <= 3.0
