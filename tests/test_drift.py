"""Unit tests for synthetic drift generator."""

import numpy as np
from PIL import Image

from src.banana_mlops.data.drift_generator import apply_synthetic_drift


def test_apply_synthetic_drift():
    # Create solid yellow test image
    img = Image.new("RGB", (100, 100), color=(255, 255, 0))
    drifted = apply_synthetic_drift(
        img,
        hue_shift_deg=25.0,
        brightness_factor=0.55,
        blur_radius=2.0,
        jpeg_quality=40,
    )

    assert drifted.size == (100, 100)
    assert drifted.mode == "RGB"

    orig_arr = np.array(img, dtype=float)
    drift_arr = np.array(drifted, dtype=float)

    # Verify brightness is dimmed
    assert drift_arr.mean() < orig_arr.mean()
