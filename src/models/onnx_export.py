import torch
import torch.nn as nn
from pathlib import Path
from src.models.acoustic_mlp import AcousticMLP
from src.models.baseline import VADPauseMLP
from src.models.hybrid import HybridTurnModel

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def export_acoustic_mlp_onnx(model: AcousticMLP, output_path: str = None):
    if output_path is None:
        output_path = str(MODEL_DIR / "acoustic_mlp.onnx")

    model.eval()
    dummy_input = torch.randn(1, 32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["acoustic_features"],
        output_names=["end_probability"],
        dynamic_axes={"acoustic_features": {0: "batch_size"}, "end_probability": {0: "batch_size"}}
    )
    print(f"Exported Acoustic MLP to ONNX: {output_path}")
    return output_path


def export_hybrid_onnx(model: HybridTurnModel, output_path: str = None):
    if output_path is None:
        output_path = str(MODEL_DIR / "hybrid_turn_model.onnx")

    model.eval()
    dummy_audio = torch.randn(1, 100, 384)
    dummy_pause = torch.randn(1, 6)

    torch.onnx.export(
        model,
        (dummy_audio, dummy_pause),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["encoder_features", "pause_features"],
        output_names=["end_probability"],
        dynamic_axes={
            "encoder_features": {0: "batch_size", 1: "seq_len"},
            "pause_features": {0: "batch_size"},
            "end_probability": {0: "batch_size"}
        }
    )
    print(f"Exported Hybrid Model to ONNX: {output_path}")
    return output_path


if __name__ == "__main__":
    model = AcousticMLP()
    export_acoustic_mlp_onnx(model)
