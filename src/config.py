"""Project configuration, paths, and reproducibility helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Project root is the parent directory of `src/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory paths.
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
APP_DIR = PROJECT_ROOT / "app"
TESTS_DIR = PROJECT_ROOT / "tests"

# Yelp source files.
BUSINESS_JSON_PATH = RAW_DATA_DIR / "business.json"
REVIEW_JSON_PATH = RAW_DATA_DIR / "review.json"
USER_JSON_PATH = RAW_DATA_DIR / "user.json"

# Processed output files.
BUSINESS_PARQUET_PATH = PROCESSED_DATA_DIR / "business_restaurants.parquet"
REVIEW_PARQUET_PATH = PROCESSED_DATA_DIR / "review_restaurants.parquet"
USER_PARQUET_PATH = PROCESSED_DATA_DIR / "user_restaurants.parquet"


@dataclass(frozen=True)
class HyperParams:
    """Default model/training hyperparameters."""

    embedding_dim: int = 8
    dnn_layers: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 2048
    epochs: int = 20
    patience: int = 3
    seed: int = 42


DEFAULT_HPARAMS = HyperParams()


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        return
    except Exception:
        # Some binary wheels can fail during import (e.g., ABI mismatches).
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Use deterministic kernels where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some environments / operations do not support strict deterministic mode.
        pass
