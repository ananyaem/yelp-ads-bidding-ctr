"""Shared demo checkpoint + Platt + ONNX for CLI and Streamlit when prod models are absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.calibration import PlattScaler
from src.models.deepfm import DeepFM
from src.models.export_onnx import export_deepfm_onnx


def write_demo_artifacts(root: Path) -> tuple[Path, Path, Path]:
    """Create a tiny ad-style checkpoint, Platt scaler, and ONNX in *root*."""
    feature_config = {
        "cuisine": {"name": "cuisine", "type": "sparse", "vocab_size": 12, "embedding_dim": 8},
        "city": {"name": "city", "type": "sparse", "vocab_size": 8, "embedding_dim": 8},
        "ad_position": {
            "name": "ad_position",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
        "bid_amount": {
            "name": "bid_amount",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
        "norm_rating": {
            "name": "norm_rating",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
    }
    model = DeepFM(feature_config, dnn_layers=[32, 16], dropout=0.1)
    torch.manual_seed(0)
    ckpt_path = root / "demo_deepfm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_config": feature_config,
            "best_val_auc": 0.5,
            "best_epoch": 1,
        },
        ckpt_path,
    )
    platt = PlattScaler()
    platt.fit(np.array([0.0, 1.0]), np.array([0.2, 0.8]))
    platt_path = root / "demo_platt.pkl"
    platt.save(platt_path)
    onnx_path = root / "demo_deepfm.onnx"
    export_deepfm_onnx(ckpt_path, onnx_path)
    return ckpt_path, platt_path, onnx_path
