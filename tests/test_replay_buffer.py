"""Unit tests for replay buffer builder."""

import tempfile
from pathlib import Path

import pandas as pd

from src.banana_mlops.data.build_replay_buffer import build_replay_buffer


def test_build_replay_buffer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy baseline and drift metadata CSVs
        base_df = pd.DataFrame(
            {
                "split": ["train"] * 10,
                "relative_path": [f"img_{i}.jpg" for i in range(10)],
                "absolute_path": [f"/tmp/img_{i}.jpg" for i in range(10)],
                "original_label": ["ripe"] * 10,
                "continuous_score": [0.4] * 10,
            }
        )
        drift_df = pd.DataFrame(
            {
                "split": ["train"] * 10,
                "relative_path": [f"drift_{i}.jpg" for i in range(10)],
                "absolute_path": [f"/tmp/drift_{i}.jpg" for i in range(10)],
                "original_label": ["ripe"] * 10,
                "continuous_score": [0.4] * 10,
            }
        )

        base_csv = Path(tmp_dir) / "baseline.csv"
        drift_csv = Path(tmp_dir) / "drift.csv"
        base_df.to_csv(base_csv, index=False)
        drift_df.to_csv(drift_csv, index=False)

        out_dir = Path(tmp_dir) / "replay"
        combined_df, metadata_path = build_replay_buffer(
            drift_metadata_csv=str(drift_csv),
            baseline_metadata_csv=str(base_csv),
            output_dir=str(out_dir),
            total_samples=10,
            drift_ratio=0.80,
        )

        assert len(combined_df) == 10
        assert Path(metadata_path).exists()
        assert "train" in combined_df["split"].values
