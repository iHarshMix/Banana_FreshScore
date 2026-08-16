"""Synthetic Drift Generation Engine simulating warehouse camera sensor shifts."""

from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter

from src.banana_mlops.data.make_dataset import get_transforms
from src.banana_mlops.models.train import calculate_metrics
from src.banana_mlops.utils.logger import setup_logger
from src.banana_mlops.utils.seed import seed_everything

logger = setup_logger("banana_mlops.data.drift_generator")


def apply_synthetic_drift(
    image: Image.Image,
    hue_shift_deg: float = 25.0,
    brightness_factor: float = 0.55,
    blur_radius: float = 2.0,
    jpeg_quality: int = 40,
) -> Image.Image:
    """Apply realistic warehouse sensor drift transformations to an image.

    Transformations:
    1. Cooler color temperature (Hue shift)
    2. Dim warehouse lighting (Brightness decay)
    3. Low-bandwidth sensor stream (Gaussian blur + JPEG recompression)

    Args:
        image: Original PIL RGB Image.
        hue_shift_deg: Hue rotation in degrees.
        brightness_factor: Brightness multiplier (0.7 = 30% reduction).
        blur_radius: Gaussian blur sigma radius.
        jpeg_quality: JPEG compression quality (1-95).

    Returns:
        Perturbed PIL Image exhibiting synthetic distribution drift.
    """
    img = image.convert("RGB")

    # 1. Color Temperature Shift via HSV
    hsv = np.array(img.convert("HSV"), dtype=np.int16)
    # Hue in PIL is [0, 255] mapping to [0, 360 degrees]
    hue_offset = int((hue_shift_deg / 360.0) * 255.0)
    hsv[..., 0] = (hsv[..., 0] + hue_offset) % 256
    img = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")

    # 2. Lighting Decay (Dim Lighting)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    # 3. Sensor Blur
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 4. Transmission Compression
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    img = Image.open(buffer).convert("RGB")

    return img


def generate_drifted_dataset(
    source_metadata_csv: str = "data/processed/baseline_v1/metadata.csv",
    output_dir: str = "data/processed/perturbed_v2",
    sample_size: Optional[int] = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a drifted dataset simulating new warehouse camera distribution.

    Args:
        source_metadata_csv: Source metadata containing baseline images.
        output_dir: Target directory for perturbed images and metadata.
        sample_size: Number of images to perturb (None for all).
        seed: Random seed for sampling.

    Returns:
        DataFrame of drifted dataset metadata.
    """
    seed_everything(seed)
    df = pd.read_csv(source_metadata_csv)

    if sample_size and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=seed).reset_index(
            drop=True
        )
    else:
        df_sample = df.copy().reset_index(drop=True)

    out_path = Path(output_dir)
    out_images_dir = out_path / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    logger.info(
        f"Generating {len(df_sample)} perturbed images into {out_images_dir}..."
    )

    for idx, row in df_sample.iterrows():
        src_path = Path(row["absolute_path"])
        if not src_path.exists():
            continue

        orig_img = Image.open(src_path)
        drifted_img = apply_synthetic_drift(orig_img)

        dest_filename = f"drift_{idx:05d}_{src_path.name}"
        dest_path = out_images_dir / dest_filename
        drifted_img.save(dest_path, "JPEG", quality=90)

        records.append(
            {
                "split": row["split"],
                "relative_path": str(dest_path.relative_to(out_path)),
                "absolute_path": str(dest_path.resolve()),
                "original_label": row["original_label"],
                "continuous_score": row["continuous_score"],
                "is_drifted": True,
            }
        )

    drift_df = pd.DataFrame(records)
    metadata_csv = out_path / "metadata.csv"
    drift_df.to_csv(metadata_csv, index=False)
    logger.info(f"Drift dataset generated successfully: {metadata_csv}")
    return drift_df


def evaluate_drift_degradation(
    model_path: str = "models/production_model.pt",
    drift_metadata_csv: str = "data/processed/perturbed_v2/metadata.csv",
    batch_size: int = 64,
) -> Tuple[float, float]:
    """Evaluate baseline production model on drifted dataset to quantify decay.

    Returns:
        Tuple of (drift_mae, drift_rmse).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    df = pd.read_csv(drift_metadata_csv)
    transform = get_transforms("test")

    all_preds = []
    all_targets = []

    logger.info(f"Evaluating {len(df)} drifted images against {model_path}...")
    for idx in range(0, len(df), batch_size):
        batch_rows = df.iloc[idx : idx + batch_size]
        tensors = []
        for _, row in batch_rows.iterrows():
            img = Image.open(row["absolute_path"]).convert("RGB")
            tensors.append(transform(img))

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            preds = model(batch_tensor).cpu().numpy().tolist()

        all_preds.extend(preds)
        all_targets.extend(batch_rows["continuous_score"].tolist())

    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    logger.warning(
        f"🚨 DRIFT DEGRADATION DETECTED: "
        f"MAE = {metrics['mae']:.4f} (Degraded from baseline ~0.067), "
        f"RMSE = {metrics['rmse']:.4f}"
    )
    return metrics["mae"], metrics["rmse"]


if __name__ == "__main__":
    generate_drifted_dataset()
    evaluate_drift_degradation()
