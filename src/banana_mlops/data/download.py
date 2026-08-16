"""Kaggle dataset ingestion module for Banana Ripeness dataset."""

import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from config.settings import get_settings
from src.banana_mlops.utils.logger import setup_logger

logger = setup_logger("banana_mlops.data.download")

DATASET_SLUG = "shahriar26s/banana-ripeness-classification-dataset"


def download_dataset(destination_dir: str = "data/raw", force: bool = False) -> Path:
    """Download and unzip the Kaggle dataset into destination directory.

    Args:
        destination_dir: Path to directory where raw dataset will be stored.
        force: If True, re-download even if files already exist.

    Returns:
        Path to raw data directory.
    """
    settings = get_settings()
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Set Kaggle environment variables if configured in settings/.env
    if settings.kaggle_username and settings.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key

    # Check if data already exists
    train_dir = dest_path / "train"
    if not force and train_dir.exists() and any(train_dir.iterdir()):
        logger.info(f"Dataset already present in {dest_path}. Skipping download.")
        return dest_path

    logger.info(f"Authenticating with Kaggle API and downloading '{DATASET_SLUG}'...")
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            DATASET_SLUG,
            path=str(dest_path),
            unzip=True,
        )
        logger.info(f"Dataset successfully downloaded and extracted to {dest_path}.")
    except Exception as e:
        logger.error(
            f"Failed to download dataset from Kaggle: {e}. "
            "Please ensure Kaggle API credentials are in ~/.kaggle/kaggle.json "
            "or set KAGGLE_USERNAME & KAGGLE_KEY."
        )
        raise

    return dest_path


if __name__ == "__main__":
    download_dataset()
