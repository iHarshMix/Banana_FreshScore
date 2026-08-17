"""Apache Airflow autonomous retraining DAG triggered upon distribution drift."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

# Airflow imports with graceful fallback if imported in standalone environment
try:
    from airflow import DAG
    from airflow.operators.python import (
        BranchPythonOperator,
        PythonOperator,
        ShortCircuitOperator,
    )
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = object
    PythonOperator = object
    BranchPythonOperator = object
    ShortCircuitOperator = object

from src.banana_mlops.data.build_replay_buffer import build_replay_buffer
from src.banana_mlops.data.drift_monitor import run_drift_analysis
from src.banana_mlops.models.retrain import execute_recovery_retraining
from src.banana_mlops.utils.logger import setup_logger

logger = setup_logger("banana_mlops.dags.retrain")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_ingest_telemetry(**kwargs) -> str:
    """Ingest production prediction logs and reference baseline datasets."""
    logger.info("Task 1: Ingesting production telemetry and reference data...")
    ref_csv = "data/processed/baseline_v1/metadata.csv"
    curr_csv = "data/processed/perturbed_v2/metadata.csv"
    if not Path(ref_csv).exists() or not Path(curr_csv).exists():
        raise FileNotFoundError("Reference or production metadata missing.")
    return "Telemetry ingestion completed."


def task_evaluate_drift(**kwargs) -> str:
    """Run Evidently AI drift detection and branch execution."""
    logger.info("Task 2: Running Evidently AI drift detection...")
    result = run_drift_analysis(
        reference_csv="data/processed/baseline_v1/metadata.csv",
        current_csv="data/processed/perturbed_v2/metadata.csv",
        model_path="models/production_model.pt",
        report_output_path="reports/evidently_drift_report.html",
        ks_p_value_threshold=0.05,
    )

    ti = kwargs.get("ti")
    if ti:
        ti.xcom_push(key="drift_result", value=result)

    if result.get("drift_detected", False):
        logger.warning("🚨 Drift detected! Routing to retraining branch.")
        return "build_replay_buffer_task"
    else:
        logger.info("✅ No drift detected. Routing to skip task.")
        return "skip_retraining_task"


def task_build_replay_buffer(**kwargs) -> str:
    """Construct 80:20 replay buffer dataset."""
    logger.info("Task 3: Building 80:20 Replay Buffer...")
    _, replay_csv = build_replay_buffer(
        drift_metadata_csv="data/processed/perturbed_v2/metadata.csv",
        baseline_metadata_csv="data/processed/baseline_v1/metadata.csv",
        output_dir="data/replay_buffer/retrain_v1",
        total_samples=1000,
        drift_ratio=0.80,
    )
    return replay_csv


def task_execute_retraining_and_promotion(**kwargs) -> Dict:
    """Fine-tune candidate model, evaluate quality gate, and promote to production."""
    logger.info("Task 4: Fine-tuning candidate model & running Quality Gate...")
    summary = execute_recovery_retraining(
        drift_metadata_csv="data/processed/perturbed_v2/metadata.csv",
        baseline_metadata_csv="data/processed/baseline_v1/metadata.csv",
        production_model_path="models/production_model.pt",
        candidate_model_path="models/candidate_model.pt",
        epochs=5,
        batch_size=32,
        lr=1e-4,
    )
    return summary


def task_skip_retraining(**kwargs) -> str:
    """Terminal task when no drift is detected."""
    logger.info("Model performance remains healthy. Retraining skipped.")
    return "Skipped retraining."


# Define Airflow DAG if running in Airflow runtime
if AIRFLOW_AVAILABLE:
    dag = DAG(
        "banana_spoilage_autonomous_retraining",
        default_args=default_args,
        description="Autonomous drift detection and recovery retraining pipeline",
        schedule_interval=timedelta(days=1),
        catchup=False,
        tags=["mlops", "freshscore", "drift-recovery"],
    )

    t1_ingest = PythonOperator(
        task_id="ingest_telemetry_task",
        python_callable=task_ingest_telemetry,
        dag=dag,
    )

    t2_drift = BranchPythonOperator(
        task_id="check_sensor_drift_task",
        python_callable=task_evaluate_drift,
        dag=dag,
    )

    t3_buffer = PythonOperator(
        task_id="build_replay_buffer_task",
        python_callable=task_build_replay_buffer,
        dag=dag,
    )

    t4_retrain = PythonOperator(
        task_id="retrain_and_promote_task",
        python_callable=task_execute_retraining_and_promotion,
        dag=dag,
    )

    t5_skip = PythonOperator(
        task_id="skip_retraining_task",
        python_callable=task_skip_retraining,
        dag=dag,
    )

    t1_ingest >> t2_drift
    t2_drift >> t3_buffer >> t4_retrain
    t2_drift >> t5_skip


if __name__ == "__main__":
    print("Executing standalone Airflow task sequence simulation...")
    task_ingest_telemetry()
    next_task = task_evaluate_drift()
    print(f"Branch Decision: {next_task}")
    if next_task == "build_replay_buffer_task":
        task_build_replay_buffer()
        res = task_execute_retraining_and_promotion()
        print("DAG Execution Result:", res)
    else:
        task_skip_retraining()
