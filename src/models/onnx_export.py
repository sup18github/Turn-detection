import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.acoustic_mlp import AcousticMLP
from src.models.baseline import VADPauseMLP
from src.models.hybrid import HybridTurnModel

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def export_acoustic_mlp_onnx(model: AcousticMLP = None, output_path: str = None):
    if model is None:
        model = AcousticMLP(in_features=32, hidden_dims=[128, 64, 32])
        ckpt = MODEL_DIR / "exp_002_acoustic.pth"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"Loaded weights from {ckpt}")

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

    # Quantize to INT8
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = str(MODEL_DIR / "acoustic_mlp_int8.onnx")
        quantize_dynamic(output_path, int8_path, weight_type=QuantType.QInt8)
        print(f"Exported Quantized INT8 Acoustic MLP: {int8_path}")
    except Exception as e:
        print(f"Note: INT8 quantization skipped ({e})")

    return output_path


def export_hybrid_onnx(model: HybridTurnModel = None, output_path: str = None):
    if model is None:
        model = HybridTurnModel(audio_feature_dim=384, pause_feature_dim=6,
                                temporal_type="gru", gru_hidden_dim=64,
                                fusion_hidden_dims=[128, 64], dropout=0.2)
        ckpt = MODEL_DIR / "exp_005_hybrid.pth"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"Loaded weights from {ckpt}")

    if output_path is None:
        output_path = str(MODEL_DIR / "hybrid_turn_model.onnx")

    model.eval()
    dummy_audio = torch.randn(1, 50, 384)
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

    # Quantize to INT8
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = str(MODEL_DIR / "hybrid_turn_model_int8.onnx")
        quantize_dynamic(output_path, int8_path, weight_type=QuantType.QInt8)
        print(f"Exported Quantized INT8 Hybrid Model: {int8_path}")
    except Exception as e:
        print(f"Note: INT8 quantization skipped ({e})")

    return output_path


def validate_onnx_models():
    import onnxruntime as ort

    print("\n--- Validating ONNX Runtime Sessions ---")
    acoust_path = str(MODEL_DIR / "acoustic_mlp.onnx")
    if os.path.exists(acoust_path):
        sess = ort.InferenceSession(acoust_path)
        dummy_in = np.random.randn(1, 32).astype(np.float32)
        out = sess.run(None, {"acoustic_features": dummy_in})
        print(f"Acoustic ONNX output shape: {out[0].shape}, P(END)={float(out[0][0]):.4f}")

    hybrid_path = str(MODEL_DIR / "hybrid_turn_model.onnx")
    if os.path.exists(hybrid_path):
        sess = ort.InferenceSession(hybrid_path)
        dummy_enc = np.random.randn(1, 50, 384).astype(np.float32)
        dummy_pause = np.random.randn(1, 6).astype(np.float32)
        out = sess.run(None, {"encoder_features": dummy_enc, "pause_features": dummy_pause})
        print(f"Hybrid ONNX output shape: {out[0].shape}, P(END)={float(out[0][0]):.4f}")


if __name__ == "__main__":
    export_acoustic_mlp_onnx()
    export_hybrid_onnx()
    validate_onnx_models()

