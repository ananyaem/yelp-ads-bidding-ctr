#!/usr/bin/env python3
"""Build a minimal Space bundle and push to Hugging Face from your machine.

Requires: pip install "huggingface_hub>=0.23"
Auth:       huggingface-cli login   OR   export HF_TOKEN=hf_...

Usage:
  export HF_SPACE_REPO=your-username/yelp-ads-bidding-ctr
  PYTHONPATH=. .venv/bin/python3 scripts/push_hf_space.py

  # Create a private Space if missing, then upload (uses Docker SDK + Dockerfile):
  PYTHONPATH=. .venv/bin/python3 scripts/push_hf_space.py --repo-id your-username/yelp-ads-bidding-ctr --create --private

Trained weights (not in Git): if ``models/best_deepfm.pt`` exists locally, it is copied into
the bundle so the Space loads your checkpoint instead of synthesizing a demo model on startup.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

_IgnoreFn = Callable[[str, list[str]], set[str]]


def _copy_tree(src: Path, dst: Path, ignore: _IgnoreFn | None = None) -> None:
    if not src.is_dir():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def _ignore_pycache(dirpath: str, names: list[str]) -> set[str]:
    skip = {".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__", ".DS_Store"}
    return {n for n in names if n in skip or n.endswith(".pyc")}


# Files ``InferencePipeline`` / ``default_model_paths`` can use (same names as local ``models/``).
_MODEL_ARTIFACTS: tuple[str, ...] = (
    "best_deepfm.pt",
    "platt_scaler.pkl",
    "deepfm.onnx",
    "feature_engineer.pkl",
)


def build_staging(
    root: Path,
    staging: Path,
    *,
    require_trained_model: bool = False,
    include_model_weights: bool = True,
) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    docker_src = root / "Dockerfile.huggingface"
    if not docker_src.is_file():
        raise FileNotFoundError(f"Missing required file: {docker_src}")
    shutil.copy2(docker_src, staging / "Dockerfile")

    files = [
        (root / "app.py", staging / "app.py"),
        (root / "requirements-huggingface.txt", staging / "requirements.txt"),
        (root / "README_HUGGINGFACE.md", staging / "README.md"),
    ]
    for src, dst in files:
        if not src.is_file():
            raise FileNotFoundError(f"Missing required file: {src}")
        shutil.copy2(src, dst)

    _copy_tree(root / "app", staging / "app", ignore=_ignore_pycache)
    _copy_tree(root / "src", staging / "src", ignore=_ignore_pycache)
    if (root / ".streamlit").is_dir():
        _copy_tree(root / ".streamlit", staging / ".streamlit", ignore=_ignore_pycache)

    models_dir = root / "models"
    staging_models = staging / "models"
    staging_models.mkdir(exist_ok=True)

    if models_dir.is_dir():
        for name in ("resume_metrics_summary.json", "feature_config.json", "training_history.json"):
            p = models_dir / name
            if p.is_file():
                shutil.copy2(p, staging_models / name)

    ckpt = models_dir / "best_deepfm.pt"
    if not include_model_weights:
        print("Skipping model weights (.pt/.pkl/.onnx) per --no-model-weights.", file=sys.stderr)
    elif include_model_weights and models_dir.is_dir():
        copied: list[str] = []
        for name in _MODEL_ARTIFACTS:
            p = models_dir / name
            if p.is_file():
                shutil.copy2(p, staging_models / name)
                copied.append(name)

        if not ckpt.is_file():
            if require_trained_model:
                raise FileNotFoundError(
                    f"Missing trained checkpoint (required): {ckpt}. "
                    "Train locally or drop --require-trained-model."
                )
            print(
                "Warning: models/best_deepfm.pt not found; Space will build a temporary demo model on first visit.",
                file=sys.stderr,
            )
        else:
            print(f"Including trained weights in upload: {', '.join(copied)}", file=sys.stderr)
    elif include_model_weights:
        if require_trained_model:
            raise FileNotFoundError(
                f"Missing models/ directory or checkpoint at {ckpt}. "
                "Train locally or drop --require-trained-model."
            )
        print(
            "Warning: no models/ directory; Space will use a temporary demo model on first visit.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Push Streamlit app to a Hugging Face Space.")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Space id: ORG/NAME (default: env HF_SPACE_REPO)",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the Space if it does not exist (Docker SDK; runs Streamlit in container).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="When used with --create, create a private Space.",
    )
    parser.add_argument(
        "--keep-staging",
        type=Path,
        default=None,
        help="Copy the bundle here instead of a temp dir (for inspection).",
    )
    parser.add_argument(
        "--require-trained-model",
        action="store_true",
        help="Fail if models/best_deepfm.pt is missing (avoid shipping demo-only Space).",
    )
    parser.add_argument(
        "--no-model-weights",
        action="store_true",
        help="Do not copy .pt/.pkl/.onnx from models/ (smaller upload; app uses demo checkpoint).",
    )
    args = parser.parse_args()
    if args.require_trained_model and args.no_model_weights:
        print("Cannot combine --require-trained-model with --no-model-weights.", file=sys.stderr)
        return 2

    repo_id = args.repo_id or __import__("os").environ.get("HF_SPACE_REPO", "").strip()
    if not repo_id:
        print("Set HF_SPACE_REPO or pass --repo-id USER/SPACE", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface_hub: pip install 'huggingface_hub>=0.23'", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parents[1]
    if args.keep_staging:
        staging = args.keep_staging.resolve()
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
    else:
        staging = Path(tempfile.mkdtemp(prefix="hf_space_"))

    try:
        build_staging(
            root,
            staging,
            require_trained_model=args.require_trained_model,
            include_model_weights=not args.no_model_weights,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    api = HfApi()
    if args.create:
        # Hub API create endpoint only accepts gradio | docker | static (not streamlit).
        # We ship a Dockerfile that runs Streamlit on port 7860.
        api.create_repo(
            repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )

    print(f"Uploading {staging} -> https://huggingface.co/spaces/{repo_id}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy Streamlit app from local export",
    )
    print(f"Done. Open https://huggingface.co/spaces/{repo_id}")
    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
