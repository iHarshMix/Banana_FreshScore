"""Unit tests for Airflow DAG tasks and branching logic."""

from unittest.mock import patch

from dags.drift_retrain_pipeline import (
    task_evaluate_drift,
    task_ingest_telemetry,
    task_skip_retraining,
)


def test_task_ingest_telemetry():
    res = task_ingest_telemetry()
    assert "completed" in res


@patch("dags.drift_retrain_pipeline.run_drift_analysis")
def test_task_evaluate_drift_branching(mock_drift):
    # Test Branch 1: Drift detected -> route to retraining
    mock_drift.return_value = {
        "drift_detected": True,
        "prediction_drift_p_value": 0.001,
        "feature_drift_detected": True,
    }
    next_task = task_evaluate_drift()
    assert next_task == "build_replay_buffer_task"

    # Test Branch 2: No drift -> route to skip
    mock_drift.return_value = {
        "drift_detected": False,
        "prediction_drift_p_value": 0.45,
        "feature_drift_detected": False,
    }
    next_task = task_evaluate_drift()
    assert next_task == "skip_retraining_task"


def test_task_skip_retraining():
    res = task_skip_retraining()
    assert "Skipped" in res

