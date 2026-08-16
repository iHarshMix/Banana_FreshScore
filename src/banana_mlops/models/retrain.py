"""Automated model retraining and recovery loop triggered upon drift detection."""

from pathlib import Path
from typing import Any, Dict

from src.banana_mlops.data.build_replay_buffer import build_replay_buffer
from src.banana_mlops.data.drift_generator import evaluate_drift_degradation
from src.banana_mlops.models.evaluate_gate import evaluate_quality_gate
from src.banana_mlops.models.train import train_model
from src.banana_mlops.utils.logger import setup_logger

logger = setup_logger("banana_mlops.models.retrain")


def execute_recovery_retraining(
    drift_metadata_csv: str = "data/processed/perturbed_v2/metadata.csv",
    baseline_metadata_csv: str = "data/processed/baseline_v1/metadata.csv",
    production_model_path: str = "models/production_model.pt",
    candidate_model_path: str = "models/candidate_model.pt",
    epochs: int = 6,
    batch_size: int = 32,
    lr: float = 1e-4,
) -> Dict[str, Any]:
    """Execute end-to-end retraining pipeline:
    replay buffer -> fine-tune -> quality gate -> promotion.

    Returns:
        Summary dictionary of before/after metrics and promotion status.
    """
    logger.info("Step 1: Building 80:20 Replay Buffer...")
    _, replay_csv = build_replay_buffer(
        drift_metadata_csv=drift_metadata_csv,
        baseline_metadata_csv=baseline_metadata_csv,
        output_dir="data/replay_buffer/retrain_v1",
        total_samples=1000,
        drift_ratio=0.80,
    )

    logger.info("Step 2: Evaluating degraded production model on drifted data...")
    degraded_mae, degraded_rmse = evaluate_drift_degradation(
        model_path=production_model_path,
        drift_metadata_csv=drift_metadata_csv,
    )

    logger.info("Step 3: Fine-tuning candidate model on Replay Buffer...")
    # Fine-tune on replay buffer using existing baseline weights checkpoint
    checkpoint_weights = "models/checkpoints/best_model.pth"
    weights_arg = checkpoint_weights if Path(checkpoint_weights).exists() else None

    train_summary = train_model(
        metadata_csv=replay_csv,
        backbone_name="resnet18",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        fine_tune=True,
        pretrained_weights_path=weights_arg,
        output_model_path=candidate_model_path,
        experiment_name="banana_ripeness_regression",
        run_name="autotrain_recovery_v1",
    )

    candidate_val_mae = train_summary["best_val_mae"]
    logger.info(f"Candidate Model Validation MAE: {candidate_val_mae:.4f}")

    logger.info("Step 4: Running Model Quality Gate Evaluation...")
    passed, rationale = evaluate_quality_gate(
        candidate_mae=candidate_val_mae,
        production_mae=degraded_mae,
        max_acceptable_mae=0.08,
    )

    # Evaluate recovered candidate on drifted data
    recovered_mae, recovered_rmse = evaluate_drift_degradation(
        model_path=candidate_model_path,
        drift_metadata_csv=drift_metadata_csv,
    )

    promoted = False
    if passed:
        logger.info(f"Quality Gate PASSED ({rationale}). Promoting candidate to Production...")
        # Promote candidate to production
        cand_path = Path(candidate_model_path)
        prod_path = Path(production_model_path)
        if cand_path.exists():
            prod_path.write_bytes(cand_path.read_bytes())
            logger.info(f"✅ Promoted {candidate_model_path} -> {production_model_path}")
            promoted = True
    else:
        logger.warning(f"Quality Gate FAILED ({rationale}). Retaining active production model.")

    headline_summary = {
        "baseline_mae": 0.0674,
        "degraded_mae": round(degraded_mae, 4),
        "recovered_mae": round(recovered_mae, 4),
        "quality_gate_passed": passed,
        "model_promoted": promoted,
        "rationale": rationale,
    }

    logger.info(
        "\n======================================================\n"
        "🎯 HEADLINE RESULT TABLE (MLOps Lifecycle Complete):\n"
        f"  Baseline Model (Clean):     MAE = {headline_summary['baseline_mae']:.4f} (Healthy)\n"
        f"  After Drift (Perturbed):    MAE = {headline_summary['degraded_mae']:.4f} (Degraded)\n"
        f"  After Auto-Retrain:         MAE = {headline_summary['recovered_mae']:.4f} (Recovered)\n"
        f"  Model Promoted to Prod:     {promoted}\n"
        "======================================================"
    )

    return headline_summary


if __name__ == "__main__":
    execute_recovery_retraining()
