"""Unit tests for drift monitor and telemetry extraction."""

import tempfile
from pathlib import Path

from PIL import Image

from src.banana_mlops.data.drift_monitor import extract_image_color_stats


def test_extract_image_color_stats():
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_img_path = Path(tmp_dir) / "test_img.jpg"
        img = Image.new("RGB", (50, 50), color=(255, 200, 0))
        img.save(test_img_path)

        hue, sat, val = extract_image_color_stats(str(test_img_path))
        assert hue >= 0
        assert sat >= 0
        assert val >= 0
