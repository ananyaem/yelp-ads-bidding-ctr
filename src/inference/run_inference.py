"""CLI: load InferencePipeline and print sponsored listings for sample inputs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.config import MODELS_DIR
from src.inference.pipeline import InferencePipeline, default_model_paths
from src.models.calibration import PlattScaler
from src.models.deepfm import DeepFM


def _write_demo_artifacts(root: Path) -> tuple[Path, Path, Path]:
    """Create a tiny ad-style checkpoint + Platt scaler (no FeatureEngineer)."""
    feature_config = {
        "cuisine": {"name": "cuisine", "type": "sparse", "vocab_size": 12, "embedding_dim": 8},
        "city": {"name": "city", "type": "sparse", "vocab_size": 8, "embedding_dim": 8},
        "ad_position": {"name": "ad_position", "type": "dense", "vocab_size": None, "embedding_dim": 0},
        "bid_amount": {"name": "bid_amount", "type": "dense", "vocab_size": None, "embedding_dim": 0},
        "norm_rating": {"name": "norm_rating", "type": "dense", "vocab_size": None, "embedding_dim": 0},
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
    from src.models.export_onnx import export_deepfm_onnx

    export_deepfm_onnx(ckpt_path, onnx_path)
    return ckpt_path, platt_path, onnx_path


def _demo_candidates() -> list[dict]:
    return [
        {
            "restaurant_id": "r_sushi_1",
            "business_id": "b1",
            "campaign_id": "c_a",
            "bid": 2.5,
            "cuisine": 3,
            "city": 1,
            "norm_rating": 0.75,
        },
        {
            "restaurant_id": "r_pizza_2",
            "business_id": "b2",
            "campaign_id": "c_b",
            "bid": 3.0,
            "cuisine": 7,
            "city": 2,
            "norm_rating": 0.60,
        },
        {
            "restaurant_id": "r_taco_3",
            "business_id": "b3",
            "campaign_id": "c_c",
            "bid": 1.8,
            "cuisine": 5,
            "city": 1,
            "norm_rating": 0.55,
        },
    ]


def _format_listings(rows: list[dict]) -> str:
    lines = ["position  restaurant          ctr        price"]
    lines.append("-" * 52)
    for r in rows:
        lines.append(
            f"{r['position']:<9} {str(r['restaurant']):<20} {r['predicted_ctr']:<10.6f} ${r['price']:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sponsored listing inference pipeline.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic checkpoint in a temp dir")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--engineer", type=Path, default=None)
    parser.add_argument("--platt", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--use-onnx", action="store_true", help="Run inference with ONNX Runtime")
    parser.add_argument(
        "--candidates-json",
        type=Path,
        default=None,
        help="Optional JSON file: list of candidate dicts (else demo fixtures)",
    )
    args = parser.parse_args()

    if args.demo:
        tmp = Path(tempfile.mkdtemp(prefix="yelp_infer_demo_"))
        model_p, platt_p, onnx_p = _write_demo_artifacts(tmp)
        pipe = InferencePipeline(
            model_path=model_p,
            engineer_path=None,
            platt_path=platt_p,
            onnx_path=onnx_p,
            use_onnx=args.use_onnx,
        )
        user_features = {"user_id": "demo_user"}
        context = {"timestamp": "2025-06-15 18:30:00"}
        candidates = _demo_candidates()
    else:
        paths = default_model_paths()
        model_path = args.model or paths["model"]
        engineer_path = args.engineer or paths["engineer"]
        platt_path = args.platt or paths["platt"]
        onnx_path = args.onnx or paths["onnx"]

        pipe = InferencePipeline(
            model_path=model_path,
            engineer_path=engineer_path if engineer_path.is_file() else None,
            platt_path=platt_path if platt_path.is_file() else None,
            onnx_path=onnx_path if onnx_path.is_file() else None,
            use_onnx=args.use_onnx and onnx_path.is_file(),
        )
        user_features = {"user_id": "u_sample"}
        context = {"timestamp": "2025-06-15 12:00:00"}
        if args.candidates_json and args.candidates_json.is_file():
            candidates = json.loads(args.candidates_json.read_text(encoding="utf-8"))
        else:
            candidates = _demo_candidates()

    listings = pipe.get_sponsored_listings(user_features, context, candidates)
    print(_format_listings(listings))
    print()
    print(json.dumps(listings, indent=2))


if __name__ == "__main__":
    main()
