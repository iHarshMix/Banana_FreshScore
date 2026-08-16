"""Unit tests for model architecture, TorchScript export, and quality gating."""

import tempfile
from pathlib import Path
import torch
from src.banana_mlops.models.architecture import (
    BananaRipenessRegressor,
    export_torchscript,
)
from src.banana_mlops.models.evaluate_gate import evaluate_quality_gate


def test_model_forward_shape():
    model = BananaRipenessRegressor(backbone_name="resnet18", pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (4,)
    assert (output >= 0.0).all() and (output <= 1.0).all()


def test_torchscript_export():
    model = BananaRipenessRegressor(backbone_name="resnet18", pretrained=False)
    with tempfile.TemporaryDirectory() as tmp_dir:
        export_path = Path(tmp_dir) / "test_model.pt"
        exported = export_torchscript(model, save_path=str(export_path), device="cpu")
        assert exported.exists()

        # Load TorchScript model and verify inference
        loaded_model = torch.jit.load(str(exported), map_location="cpu")
        dummy_input = torch.randn(2, 3, 224, 224)
        output = loaded_model(dummy_input)
        assert output.shape == (2,)


def test_quality_gate_promotion():
    # Better MAE promotes
    passed, rationale = evaluate_quality_gate(candidate_mae=0.042, production_mae=0.055)
    assert passed is True
    assert "PROMOTE" in rationale

    # Worse MAE rejects
    passed, rationale = evaluate_quality_gate(candidate_mae=0.095, production_mae=0.040)
    assert passed is False
    assert "REJECT" in rationale
