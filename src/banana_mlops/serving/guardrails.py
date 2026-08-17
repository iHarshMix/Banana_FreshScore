"""Input validation guardrails and business logic transformations."""

import io
from typing import Optional, Set, Tuple

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from src.banana_mlops.utils.logger import setup_logger

logger = setup_logger("banana_mlops.serving.guardrails")

ALLOWED_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
MIN_FILE_SIZE_BYTES: int = 5 * 1024  # 5 KB
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB


def validate_image_payload(
    file_bytes: bytes,
    content_type: Optional[str] = None,
) -> Image.Image:
    """Validate uploaded image against MIME, size bounds, and header integrity guardrails.

    Args:
        file_bytes: Raw binary image payload.
        content_type: Uploaded MIME type header.

    Returns:
        Verified and RGB-normalized PIL Image.

    Raises:
        HTTPException: If payload fails any input guardrail check.
    """
    # 1. MIME Type Validation
    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported media type '{content_type}'. "
                f"Allowed formats: {list(ALLOWED_MIME_TYPES)}"
            ),
        )

    # 2. File Size Bounds Check
    payload_size = len(file_bytes)
    if payload_size < MIN_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File size too small ({payload_size} bytes). "
                f"Minimum acceptable payload is {MIN_FILE_SIZE_BYTES} bytes (5 KB)."
            ),
        )
    if payload_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size too large ({payload_size} bytes). "
                f"Maximum acceptable payload is {MAX_FILE_SIZE_BYTES} bytes (10 MB)."
            ),
        )

    # 3. PIL Header Integrity Check
    try:
        verify_buffer = io.BytesIO(file_bytes)
        img_verify = Image.open(verify_buffer)
        img_verify.verify()
    except (UnidentifiedImageError, Exception) as e:
        logger.warning(f"Rejected corrupted image payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image payload: {e}",
        )

    # 4. RGB Normalization (re-open fresh buffer after verify())
    load_buffer = io.BytesIO(file_bytes)
    img = Image.open(load_buffer).convert("RGB")
    return img


def compute_shelf_life_and_category(
    score: float,
    max_days: float = 12.0,
) -> Tuple[str, float, str]:
    """Transform continuous regression score into qualitative ripeness category and shelf-life.

    Args:
        score: Continuous spoilage score y in [0.0, 1.0].
        max_days: Maximum shelf-life constant (default 12.0 days).

    Returns:
        Tuple of (category: str, shelf_life_days: float, recommended_action: str).
    """
    clamped_score = max(0.0, min(1.0, float(score)))
    shelf_life_days = round(max_days * (1.0 - clamped_score), 2)

    if clamped_score <= 0.25:
        category = "Unripe (Green)"
        action = "Store in warehouse"
    elif clamped_score <= 0.50:
        category = "Slightly Ripe"
        action = "Ready for retail distribution"
    elif clamped_score <= 0.75:
        category = "Ripe"
        action = "Place on store shelves immediately"
    else:
        category = "Overripe / Rotten"
        action = "Discount / discard"

    return category, shelf_life_days, action
