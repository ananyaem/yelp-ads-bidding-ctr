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

# Simulated ad-impression outputs.
AD_IMPRESSIONS_PARQUET_PATH = PROCESSED_DATA_DIR / "ad_impressions.parquet"
AD_TRAIN_PARQUET_PATH = PROCESSED_DATA_DIR / "ad_impressions_train.parquet"
AD_VAL_PARQUET_PATH = PROCESSED_DATA_DIR / "ad_impressions_val.parquet"
AD_TEST_PARQUET_PATH = PROCESSED_DATA_DIR / "ad_impressions_test.parquet"
AD_SCHEMA_PATH = PROCESSED_DATA_DIR / "ad_impressions_schema.json"

# Ground-truth coefficients used in synthetic click simulation.
AD_SIM_TRUE_COEFFICIENTS = {
    "intercept": None,  # Calibrated at runtime to hit target CTR.
    "cuisine_match": 2.0,
    "norm_rating": 1.2,
    "inv_position": 3.0,
    "price_distance": -0.8,
    "is_evening": 0.5,
    "city_match": 1.0,
    "log_review_count": 0.6,
    "user_activity_bias_std": 0.3,
    "cuisine_evening_interaction": 0.8,
    "noise_std": 0.5,
}


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
