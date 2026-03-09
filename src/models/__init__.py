"""Model package exports.

Torch-dependent classes (DeepFM, PositionDebiaser) are imported lazily to
avoid crashing in environments where torch is broken or absent.
"""

from typing import Any

from src.models.calibration import BinReliability, PlattScaler, compute_ece
from src.models.evaluation import (
    RevenueResult,
    SegmentCalibration,
    dcg_at_k,
    full_evaluation_report,
    grouped_ndcg,
    ndcg_at_k,
    per_segment_auc,
    per_segment_ece,
    simulate_revenue,
)


def __getattr__(name: str) -> Any:
    _torch_names = {
        "EmbeddingLayer",
        "FMLayer",
        "DNNLayer",
        "DeepFM",
        "PositionDebiaser",
    }
    if name in _torch_names:
        if name == "PositionDebiaser":
            from src.models.calibration import PositionDebiaser

            return PositionDebiaser
        from src.models import deepfm

        return getattr(deepfm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EmbeddingLayer",
    "FMLayer",
    "DNNLayer",
    "DeepFM",
    "PlattScaler",
    "PositionDebiaser",
    "BinReliability",
    "compute_ece",
    "dcg_at_k",
    "ndcg_at_k",
    "grouped_ndcg",
    "per_segment_auc",
    "per_segment_ece",
    "SegmentCalibration",
    "simulate_revenue",
    "RevenueResult",
    "full_evaluation_report",
]
