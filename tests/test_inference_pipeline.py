"""Tests for InferencePipeline, ONNX export parity, and CLI helpers."""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.inference.pipeline import InferencePipeline
from src.models.calibration import PlattScaler
from src.models.deepfm import DeepFM
from src.models.export_onnx import export_deepfm_onnx, verify_onnx_matches_pytorch


def _tiny_ad_feature_config() -> dict:
    return {
        "cuisine": {"name": "cuisine", "type": "sparse", "vocab_size": 10, "embedding_dim": 8},
        "city": {"name": "city", "type": "sparse", "vocab_size": 8, "embedding_dim": 8},
        "ad_position": {"name": "ad_position", "type": "dense", "vocab_size": None, "embedding_dim": 0},
        "bid_amount": {"name": "bid_amount", "type": "dense", "vocab_size": None, "embedding_dim": 0},
        "norm_rating": {"name": "norm_rating", "type": "dense", "vocab_size": None, "embedding_dim": 0},
    }


def _save_demo_bundle(root: Path) -> tuple[Path, Path, Path]:
    fc = _tiny_ad_feature_config()
    model = DeepFM(fc, dnn_layers=[32, 16], dropout=0.1)
    torch.manual_seed(1)
    ckpt = root / "m.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "feature_config": fc, "best_val_auc": 0.5, "best_epoch": 1},
        ckpt,
    )
    platt = PlattScaler()
    platt.fit(np.array([0.0, 1.0]), np.array([0.25, 0.75]))
    pp = root / "platt.pkl"
    platt.save(pp)
    onnx_p = root / "m.onnx"
    export_deepfm_onnx(ckpt, onnx_p)
    return ckpt, pp, onnx_p


def test_onnx_matches_pytorch_atol():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ckpt, _, onnx_p = _save_demo_bundle(root)
        verify_onnx_matches_pytorch(ckpt, onnx_p, batch_size=8, atol=1e-5)


def test_pipeline_missing_model_file():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        missing = root / "nope.pt"
        with pytest.raises(FileNotFoundError) as exc:
            InferencePipeline(model_path=missing, engineer_path=None)
        assert str(missing.resolve()) in str(exc.value)


def test_pipeline_missing_platt_file():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ckpt, _, _ = _save_demo_bundle(root)
        bad_platt = root / "missing.pkl"
        with pytest.raises(FileNotFoundError) as exc:
            InferencePipeline(model_path=ckpt, engineer_path=None, platt_path=bad_platt)
        assert str(bad_platt.resolve()) in str(exc.value)


def test_pipeline_empty_candidates():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ckpt, pp, _ = _save_demo_bundle(root)
        pipe = InferencePipeline(model_path=ckpt, engineer_path=None, platt_path=pp)
        out = pipe.get_sponsored_listings(
            {"user_id": "u1"}, {"timestamp": "2024-01-01"}, []
        )
        assert out == []


def test_pipeline_all_zero_ctr_returns_empty_with_warning(caplog):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ckpt, pp, _ = _save_demo_bundle(root)
        pipe = InferencePipeline(model_path=ckpt, engineer_path=None, platt_path=pp)
        cands = [
            {
                "restaurant_id": "r1",
                "business_id": "b1",
                "campaign_id": "c1",
                "bid": 1.0,
                "cuisine": 1,
                "city": 1,
                "norm_rating": 0.5,
            }
        ]
        with caplog.at_level("WARNING"):
            with patch.object(InferencePipeline, "_predict_pytorch", return_value=np.array([0.0])):
                out = pipe.get_sponsored_listings(
                    {"user_id": "u1"}, {"timestamp": "2024-01-01"}, cands
                )
        assert out == []
        assert "non-positive" in caplog.text.lower() or "ctr" in caplog.text.lower()


def test_pipeline_runs_gsp_ranking():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ckpt, pp, _ = _save_demo_bundle(root)
        pipe = InferencePipeline(model_path=ckpt, engineer_path=None, platt_path=pp)
        cands = [
            {
                "restaurant_id": "r1",
                "business_id": "b1",
                "campaign_id": "c1",
                "bid": 5.0,
                "cuisine": 2,
                "city": 1,
                "norm_rating": 0.9,
            },
            {
                "restaurant_id": "r2",
                "business_id": "b2",
                "campaign_id": "c2",
                "bid": 2.0,
                "cuisine": 3,
                "city": 2,
                "norm_rating": 0.5,
            },
        ]
        out = pipe.get_sponsored_listings({"user_id": "u1"}, {"timestamp": "2024-01-01 12:00:00"}, cands)
        assert len(out) == 2
        assert {o["position"] for o in out} == {1, 2}
        for o in out:
            assert "restaurant" in o and "predicted_ctr" in o and "price" in o
            assert o["price"] <= 5.0 + 1e-6


def test_pipeline_onnx_path_matches_torch():
    pytest.importorskip("onnxruntime")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ckpt, pp, onnx_p = _save_demo_bundle(root)
        cands = [
            {
                "restaurant_id": "r1",
                "business_id": "b1",
                "campaign_id": "c1",
                "bid": 2.0,
                "cuisine": 1,
                "city": 1,
                "norm_rating": 0.7,
            }
        ]
        pt_pipe = InferencePipeline(model_path=ckpt, engineer_path=None, platt_path=pp, use_onnx=False)
        ort_pipe = InferencePipeline(
            model_path=ckpt,
            engineer_path=None,
            platt_path=pp,
            onnx_path=onnx_p,
            use_onnx=True,
        )
        ctx = {"timestamp": "2024-01-01"}
        u = {"user_id": "u1"}
        a = pt_pipe.get_sponsored_listings(u, ctx, cands)
        b = ort_pipe.get_sponsored_listings(u, ctx, cands)
        assert [x["restaurant"] for x in a] == [x["restaurant"] for x in b]
        assert np.allclose(
            [x["predicted_ctr"] for x in a],
            [x["predicted_ctr"] for x in b],
            atol=1e-4,
        )


def test_run_inference_cli_demo():
    from src.inference import run_inference

    buf = io.StringIO()
    old_argv = sys.argv[:]
    old_stdout = sys.stdout
    try:
        sys.argv = ["run_inference", "--demo"]
        sys.stdout = buf
        run_inference.main()
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
    out = buf.getvalue()
    assert "r_sushi" in out or "position" in out.lower()
