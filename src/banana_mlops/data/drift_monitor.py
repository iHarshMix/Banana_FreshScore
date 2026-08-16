"""Evidently AI Data & Prediction Drift Monitoring Engine."""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import ks_2samp

from src.banana_mlops.data.make_dataset import get_transforms
from src.banana_mlops.utils.logger import setup_logger

logger = setup_logger("banana_mlops.data.drift_monitor")


def extract_image_color_stats(img_path: str) -> Tuple[float, float, float]:
    """Compute average Hue, Saturation, and Value (Brightness) for an image."""
    img = Image.open(img_path).convert("HSV")
    arr = np.array(img, dtype=float)
    mean_hue = float(arr[..., 0].mean())
    mean_sat = float(arr[..., 1].mean())
    mean_val = float(arr[..., 2].mean())
    return mean_hue, mean_sat, mean_val


def build_telemetry_dataframe(
    metadata_csv: str,
    model_path: str = "models/production_model.pt",
    max_samples: int = 500,
) -> pd.DataFrame:
    """Build telemetry dataframe containing prediction scores and HSV image features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    df = pd.read_csv(metadata_csv)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    transform = get_transforms("test")
    records = []

    for _, row in df.iterrows():
        img_path = row["absolute_path"]
        try:
            hue, sat, val = extract_image_color_stats(img_path)
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tensor).item()

            records.append(
                {
                    "prediction": round(pred, 4),
                    "mean_hue": round(hue, 2),
                    "mean_saturation": round(sat, 2),
                    "mean_brightness": round(val, 2),
                    "true_target": round(float(row["continuous_score"]), 4),
                }
            )
        except Exception as e:
            logger.warning(f"Error extracting features from {img_path}: {e}")

    return pd.DataFrame(records)


def run_drift_analysis(
    reference_csv: str = "data/processed/baseline_v1/metadata.csv",
    current_csv: str = "data/processed/perturbed_v2/metadata.csv",
    model_path: str = "models/production_model.pt",
    report_output_path: str = "reports/evidently_drift_report.html",
    ks_p_value_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Execute Evidently AI drift analysis and generate visual HTML report.

    Args:
        reference_csv: Metadata of clean baseline reference distribution.
        current_csv: Metadata of production / perturbed batch.
        model_path: TorchScript model used for predictions.
        report_output_path: Output path for HTML report.
        ks_p_value_threshold: P-value threshold below which drift is flagged.

    Returns:
        Dictionary containing drift verdict, KS statistics, and report path.
    """
    logger.info("Building reference telemetry dataset...")
    ref_df = build_telemetry_dataframe(reference_csv, model_path=model_path)

    logger.info("Building current production telemetry dataset...")
    curr_df = build_telemetry_dataframe(current_csv, model_path=model_path)

    # 1. Prediction Drift via Kolmogorov-Smirnov test
    ks_stat, ks_p_val = ks_2samp(ref_df["prediction"], curr_df["prediction"])

    # 2. Image Feature Drift (Hue & Brightness shift)
    hue_stat, hue_p_val = ks_2samp(ref_df["mean_hue"], curr_df["mean_hue"])
    bright_stat, bright_p_val = ks_2samp(
        ref_df["mean_brightness"], curr_df["mean_brightness"]
    )

    drift_detected = bool(ks_p_val < ks_p_value_threshold or hue_p_val < 0.01)

    # 3. Generate Evidently HTML Report
    out_file = Path(report_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=curr_df)
        report.save_html(str(out_file))
        logger.info(f"Evidently AI HTML drift report generated: {out_file}")
    except Exception as e:
        logger.warning(
            f"Evidently Report rendering fallback (using HTML generator): {e}"
        )
        # Fallback interactive HTML generator if evidently preset API differs
        generate_custom_drift_html(
            ref_df,
            curr_df,
            ks_stat,
            ks_p_val,
            drift_detected,
            save_path=str(out_file),
        )

    summary = {
        "drift_detected": drift_detected,
        "prediction_ks_stat": round(float(ks_stat), 4),
        "prediction_p_value": float(ks_p_val),
        "hue_p_value": float(hue_p_val),
        "brightness_p_value": float(bright_p_val),
        "reference_samples": len(ref_df),
        "current_samples": len(curr_df),
        "report_path": str(out_file),
    }

    status_icon = "🚨 DRIFT DETECTED" if drift_detected else "✅ NO DRIFT"
    logger.info(
        f"Drift Analysis Completed [{status_icon}]: "
        f"KS p-value = {ks_p_val:.4e} (threshold < {ks_p_value_threshold})"
    )

    return summary


def generate_custom_drift_html(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    ks_stat: float,
    ks_p_val: float,
    drift_detected: bool,
    save_path: str = "reports/evidently_drift_report.html",
) -> None:
    """Generate self-contained interactive HTML drift telemetry report."""
    status_class = "danger" if drift_detected else "success"
    status_text = (
        "CRITICAL DRIFT DETECTED — Retraining Trigger Recommended"
        if drift_detected
        else "HEALTHY — Data Distribution In-Bounds"
    )

    mean_hue_diff = abs(curr_df["mean_hue"].mean() - ref_df["mean_hue"].mean())
    mean_bright_diff = abs(
        curr_df["mean_brightness"].mean() - ref_df["mean_brightness"].mean()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Evidently AI Drift Telemetry Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0e1117; color: #ffffff; padding: 24px;
        }}
        .card {{
            background: #1a1f2c; border-radius: 12px; padding: 20px;
            margin-bottom: 20px; border: 1px solid #2d3748;
        }}
        .header {{ display: flex; justify-content: space-between; align-items: center; }}
        .badge {{ padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px; }}
        .badge.danger {{ background: #ff4b4b; color: white; }}
        .badge.success {{ background: #00c04b; color: white; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #2d3748; }}
        th {{ color: #a0aec0; font-size: 13px; text-transform: uppercase; }}
        .metric-val {{ font-size: 20px; font-weight: 600; color: #00d4b2; }}
    </style>
</head>
<body>
    <div class="card header">
        <div>
            <h2>🍌 Banana Ripeness Drift Monitoring Report</h2>
            <p style="color: #a0aec0; margin: 4px 0 0 0;">
                Kolmogorov-Smirnov & Sensor Color Distribution Drift Check
            </p>
        </div>
        <span class="badge {status_class}">{status_text}</span>
    </div>

    <div class="card">
        <h3>📊 Summary Statistics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Reference Baseline</th>
                <th>Current Production Batch</th>
                <th>Drift Indicator</th>
            </tr>
            <tr>
                <td>Prediction Mean ± Std</td>
                <td>{ref_df['prediction'].mean():.4f} ± {ref_df['prediction'].std():.4f}</td>
                <td>{curr_df['prediction'].mean():.4f} ± {curr_df['prediction'].std():.4f}</td>
                <td class="metric-val">KS p-val: {ks_p_val:.4e}</td>
            </tr>
            <tr>
                <td>Image Hue Mean (HSV)</td>
                <td>{ref_df['mean_hue'].mean():.1f}°</td>
                <td>{curr_df['mean_hue'].mean():.1f}°</td>
                <td>Δ = {mean_hue_diff:.1f}° shift</td>
            </tr>
            <tr>
                <td>Image Brightness (0-255)</td>
                <td>{ref_df['mean_brightness'].mean():.1f}</td>
                <td>{curr_df['mean_brightness'].mean():.1f}</td>
                <td>Δ = {mean_bright_diff:.1f} drop</td>
            </tr>
        </table>
    </div>
</body>
</html>"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
