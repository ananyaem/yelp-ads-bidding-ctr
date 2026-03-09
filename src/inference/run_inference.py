"""CLI: load InferencePipeline and print sponsored listings for sample inputs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from src.inference.demo_assets import write_demo_artifacts
from src.inference.pipeline import InferencePipeline, default_model_paths


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
    parser.add_argument(
        "--demo", action="store_true", help="Use synthetic checkpoint in a temp dir"
    )
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
        model_p, platt_p, onnx_p = write_demo_artifacts(tmp)
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
