"""Replay Buffer builder mixing drifted and baseline data to prevent catastrophic forgetting."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.banana_mlops.utils.logger import setup_logger
from src.banana_mlops.utils.seed import seed_everything

logger = setup_logger("banana_mlops.data.build_replay_buffer")


def build_replay_buffer(
    drift_metadata_csv: str = "data/processed/perturbed_v2/metadata.csv",
    baseline_metadata_csv: str = "data/processed/baseline_v1/metadata.csv",
    output_dir: str = "data/replay_buffer/retrain_v1",
    total_samples: int = 1000,
    drift_ratio: float = 0.80,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str]:
    """Sample an 80:20 mix of drifted images and clean baseline images.

    Args:
        drift_metadata_csv: Path to drifted dataset metadata.
        baseline_metadata_csv: Path to reference clean baseline metadata.
        output_dir: Destination folder for replay buffer metadata.
        total_samples: Total number of training samples for retraining.
        drift_ratio: Ratio of drifted samples (0.80 = 80%).
        seed: Random seed for deterministic sampling.

    Returns:
        Tuple of (DataFrame, metadata_csv_path).
    """
    seed_everything(seed)
    drift_df = pd.read_csv(drift_metadata_csv)
    base_df = pd.read_csv(baseline_metadata_csv)

    n_drift = int(total_samples * drift_ratio)
    n_base = total_samples - n_drift

    n_drift_actual = min(n_drift, len(drift_df))
    n_base_actual = min(n_base, len(base_df))

    drift_sample = drift_df.sample(
        n=n_drift_actual, random_state=seed
    ).reset_index(drop=True)
    base_sample = base_df.sample(
        n=n_base_actual, random_state=seed
    ).reset_index(drop=True)

    # Combine into 80:20 buffer
    combined_df = (
        pd.concat([drift_sample, base_sample], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    # Assign 80% train, 10% valid, 10% test
    n_train = int(len(combined_df) * 0.80)
    n_val = int(len(combined_df) * 0.10)

    combined_df.loc[:n_train, "split"] = "train"
    combined_df.loc[n_train : n_train + n_val, "split"] = "valid"
    combined_df.loc[n_train + n_val :, "split"] = "test"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metadata_csv = out_path / "metadata.csv"
    combined_df.to_csv(metadata_csv, index=False)

    drift_pct = int(drift_ratio * 100)
    base_pct = int((1 - drift_ratio) * 100)
    logger.info(
        f"Built replay buffer ({drift_pct}% drifted / {base_pct}% baseline) "
        f"with {len(combined_df)} samples at {metadata_csv}."
    )
    return combined_df, str(metadata_csv)


if __name__ == "__main__":
    build_replay_buffer()
