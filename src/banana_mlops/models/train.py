"""Model training and evaluation pipeline with MLflow tracking."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.settings import get_settings
from src.banana_mlops.data.make_dataset import (
    BananaRipenessDataset,
    get_transforms,
)
from src.banana_mlops.models.architecture import (
    BananaRipenessRegressor,
    export_torchscript,
)
from src.banana_mlops.utils.logger import setup_logger
from src.banana_mlops.utils.seed import seed_everything

logger = setup_logger("banana_mlops.models.train")


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute regression evaluation metrics."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    max_error = float(np.max(np.abs(y_true - y_pred)))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "max_error": round(max_error, 4),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on a DataLoader and return loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * len(targets)
            all_preds.extend(outputs.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    return avg_loss, metrics


def train_model(
    metadata_csv: str = "data/processed/baseline_v1/metadata.csv",
    backbone_name: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: Optional[str] = None,
    output_model_path: str = "models/production_model.pt",
    experiment_name: str = "banana_ripeness_regression",
    run_name: Optional[str] = None,
    fine_tune: bool = False,
    pretrained_weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train or fine-tune Banana Ripeness Regressor with MLflow tracking.

    Args:
        metadata_csv: Path to metadata dataframe containing splits and targets.
        backbone_name: CNN architecture ('resnet18' or 'efficientnet_b0').
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoaders.
        lr: Optimizer learning rate.
        weight_decay: L2 regularization penalty.
        device_name: 'cuda' or 'cpu' (auto-selected if None).
        output_model_path: Destination path for TorchScript model artifact.
        experiment_name: MLflow experiment name.
        run_name: MLflow run identifier.
        fine_tune: If True, only train regression head layers.
        pretrained_weights_path: Path to existing model weights for fine-tuning.

    Returns:
        Dictionary containing training summary metrics and artifact paths.
    """
    settings = get_settings()
    seed_everything(42)

    # Resolve compute device
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Training on compute device: {device}")

    # Load dataset splits
    df = pd.read_csv(metadata_csv)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "valid"]
    test_df = df[df["split"] == "test"]

    logger.info(
        f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    train_dataset = BananaRipenessDataset(
        train_df, transform=get_transforms("train")
    )
    val_dataset = BananaRipenessDataset(val_df, transform=get_transforms("val"))
    test_dataset = BananaRipenessDataset(
        test_df, transform=get_transforms("test")
    )

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize model
    model = BananaRipenessRegressor(
        backbone_name=backbone_name,
        pretrained=True,
        freeze_backbone=fine_tune,
    )

    if pretrained_weights_path and Path(pretrained_weights_path).exists():
        logger.info(
            f"Loading initial model weights from {pretrained_weights_path}"
        )
        state_dict = torch.load(pretrained_weights_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.HuberLoss(delta=0.1)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # MLflow Setup
    tracking_uri = settings.mlflow_tracking_uri
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(
            f"Unable to connect to MLflow at '{tracking_uri}' ({e}). "
            "Falling back to local SQLite 'sqlite:///mlflow.db'."
        )
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)

    run_title = run_name or (
        f"{'finetune' if fine_tune else 'baseline'}_{backbone_name}"
    )

    best_val_mae = float("inf")
    best_weights_path = Path("models/checkpoints/best_model.pth")
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_title):
        mlflow.log_params(
            {
                "backbone": backbone_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "fine_tune": fine_tune,
                "loss_function": "HuberLoss(delta=0.1)",
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
            }
        )

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(targets)
                train_preds.extend(outputs.detach().cpu().numpy().tolist())
                train_targets.extend(targets.detach().cpu().numpy().tolist())

            scheduler.step()

            avg_train_loss = train_loss / len(train_dataset)
            train_metrics = calculate_metrics(
                np.array(train_targets), np.array(train_preds)
            )
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device
            )

            mlflow.log_metrics(
                {
                    "train_loss": avg_train_loss,
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "val_loss": val_loss,
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                },
                step=epoch,
            )

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {avg_train_loss:.4f}, Train MAE: {train_metrics['mae']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val MAE: {val_metrics['mae']:.4f}"
            )

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                torch.save(model.state_dict(), best_weights_path)
                logger.info(f"[*] New best validation MAE: {best_val_mae:.4f}")

        # Load best weights for final test evaluation
        model.load_state_dict(
            torch.load(best_weights_path, map_location=device, weights_only=True)
        )
        test_loss, test_metrics = evaluate(
            model, test_loader, criterion, device
        )
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            }
        )

        logger.info(
            f"=== Final Test Evaluation: MAE={test_metrics['mae']:.4f}, "
            f"RMSE={test_metrics['rmse']:.4f} ==="
        )

        # Export to optimized TorchScript (.pt) on CPU for production serving
        exported_pt = export_torchscript(
            model, save_path=output_model_path, device="cpu"
        )
        mlflow.log_artifact(str(exported_pt))

        return {
            "best_val_mae": best_val_mae,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "model_path": str(exported_pt),
        }


if __name__ == "__main__":
    train_model()
