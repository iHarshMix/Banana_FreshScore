"""Dataset processing and continuous target synthesis module."""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.banana_mlops.utils.logger import setup_logger
from src.banana_mlops.utils.seed import seed_everything

logger = setup_logger("banana_mlops.data.make_dataset")

# Continuous score mapping bands for discrete stages
CLASS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "unripe": (0.00, 0.25),
    "ripe": (0.25, 0.50),
    "overripe": (0.50, 0.75),
    "rotten": (0.75, 1.00),
}


def synthesize_continuous_target(
    label: str,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Sample a continuous ripeness target score within the class's band.

    Args:
        label: Qualitative class string ('unripe', 'ripe', 'overripe', 'rotten').
        rng: Optional NumPy random Generator for reproducible sampling.

    Returns:
        Continuous scalar y in [0.0, 1.0].
    """
    normalized_label = label.lower().strip()
    if normalized_label not in CLASS_BOUNDS:
        raise ValueError(
            f"Unknown label '{label}'. Must be one of: {list(CLASS_BOUNDS.keys())}"
        )

    low, high = CLASS_BOUNDS[normalized_label]
    if rng is not None:
        score = float(rng.uniform(low, high))
    else:
        score = float(np.random.uniform(low, high))

    return round(score, 4)


def process_raw_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed/baseline_v1",
    seed: int = 42,
) -> pd.DataFrame:
    """Scan raw downloaded dataset splits and synthesize metadata with continuous scores.

    Args:
        raw_dir: Directory containing train/, valid/, test/ folders.
        output_dir: Directory where processed metadata will be saved.
        seed: Random seed for deterministic score generation.

    Returns:
        DataFrame containing columns: [split, relative_path, absolute_path,
        original_label, continuous_score].
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Some Kaggle downloads extract directly to raw_dir or into a subfolder
    subdirs = [d for d in raw_path.iterdir() if d.is_dir() and d.name != ".gitkeep"]
    if len(subdirs) == 1 and (subdirs[0] / "train").exists():
        search_root = subdirs[0]
    else:
        search_root = raw_path

    records = []
    splits = ["train", "valid", "test"]

    for split in splits:
        split_dir = search_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist.")
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name not in CLASS_BOUNDS:
                continue

            class_name = class_dir.name
            for img_file in class_dir.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    score = synthesize_continuous_target(class_name, rng=rng)
                    records.append(
                        {
                            "split": split,
                            "relative_path": str(img_file.relative_to(search_root)),
                            "absolute_path": str(img_file.resolve()),
                            "original_label": class_name,
                            "continuous_score": score,
                        }
                    )

    df = pd.DataFrame(records)
    metadata_csv = out_path / "metadata.csv"
    df.to_csv(metadata_csv, index=False)
    logger.info(
        f"Processed {len(df)} images across splits into {metadata_csv}. "
        f"Split breakdown:\n{df['split'].value_counts()}"
    )
    return df


def get_transforms(split: str = "train", img_size: int = 224) -> transforms.Compose:
    """Return PyTorch torchvision transforms for training or evaluation.

    Args:
        split: 'train' for data augmentations, 'val' or 'test' for standard resizing.
        img_size: Target square image dimension (default 224).

    Returns:
        torchvision transforms.Compose pipeline.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )


class BananaRipenessDataset(Dataset):
    """PyTorch Dataset for Banana Ripeness continuous regression."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["absolute_path"]
        target = float(row["continuous_score"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.float32)
