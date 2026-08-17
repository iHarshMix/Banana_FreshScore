"""Unit tests for Airflow DAG tasks and branching logic."""

from dags.drift_retrain_pipeline import (
    task_evaluate_drift,
    task_ingest_telemetry,
    task_skip_retraining,
)


def test_task_ingest_telemetry():
    res = task_ingest_telemetry()
    assert "completed" in res


def test_task_evaluate_drift_branching():
    next_task = task_evaluate_drift()
    assert next_task in ["build_replay_buffer_task", "skip_retraining_task"]


def test_task_skip_retraining():
    res = task_skip_retraining()
    assert "Skipped" in res
