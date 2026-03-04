"""Post-hoc calibration and position debiasing for CTR models."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


@dataclass
class BinReliability:
    """Per-bin reliability data for plotting calibration diagrams."""

    ece: float
    bin_edges: np.ndarray
    bin_mean_pred: np.ndarray
    bin_mean_true: np.ndarray
    bin_count: np.ndarray


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> BinReliability:
    """Compute Expected Calibration Error with per-bin reliability data."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    n = len(y_true)
    bin_mean_pred = np.full(n_bins, np.nan)
    bin_mean_true = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    ece = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        bin_count[b] = count
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        mean_true = float(y_true[mask].mean())
        bin_mean_pred[b] = mean_pred
        bin_mean_true[b] = mean_true
        ece += (count / n) * abs(mean_true - mean_pred)

    return BinReliability(
        ece=float(ece),
        bin_edges=bin_edges,
        bin_mean_pred=bin_mean_pred,
        bin_mean_true=bin_mean_true,
        bin_count=bin_count,
    )


class PlattScaler:
    """Platt scaling: fits a logistic regression on (raw_model_output, true_label).

    Transforms raw sigmoid outputs into calibrated probabilities.
    """

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None
        self.a: float = 1.0
        self.b: float = 0.0
        self._fitted = False

    def fit(self, y_true: np.ndarray, raw_probs: np.ndarray) -> PlattScaler:
        y_true = np.asarray(y_true, dtype=float).ravel()
        raw_probs = np.asarray(raw_probs, dtype=float).ravel()

        eps = 1e-7
        clipped = np.clip(raw_probs, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
        lr.fit(logits, y_true)
        self._lr = lr
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        self._fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        raw_probs = np.asarray(raw_probs, dtype=float).ravel()
        if not self._fitted:
            return raw_probs

        eps = 1e-7
        clipped = np.clip(raw_probs, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped))
        cal = 1.0 / (1.0 + np.exp(-(self.a * logits + self.b)))
        return np.clip(cal, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"a": self.a, "b": self.b, "fitted": self._fitted}, f)

    @classmethod
    def load(cls, path: str | Path) -> PlattScaler:
        with Path(path).open("rb") as f:
            data = pickle.load(f)  # noqa: S301
        scaler = cls()
        scaler.a = data["a"]
        scaler.b = data["b"]
        scaler._fitted = data["fitted"]
        return scaler


class PositionDebiaser:
    """Wraps a trained DeepFM to override position to a neutral value at inference."""

    def __init__(
        self,
        model: Any,
        feature_config: dict[str, dict[str, Any]],
        scaler: PlattScaler | None = None,
        neutral_position: float = 1.0,
        device: Any = None,
    ) -> None:
        import torch
        self.model = model
        self.feature_config = feature_config
        self.scaler = scaler
        self.neutral_position = neutral_position
        self.device = (
            torch.device(device) if device else
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        dense_features = [k for k, v in feature_config.items() if v.get("type") == "dense"]
        self._position_idx: int | None = None
        if "ad_position" in dense_features:
            self._position_idx = dense_features.index("ad_position")

    def predict(
        self,
        feature_df: pd.DataFrame,
        calibrate: bool = True,
        batch_size: int = 2048,
    ) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader
        from src.training.trainer import AdClickDataset, _collate_batch

        infer_df = feature_df.copy()
        infer_df["ad_position"] = self.neutral_position
        if "click" not in infer_df.columns:
            infer_df["click"] = 0

        ds = AdClickDataset(infer_df, self.feature_config)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)

        self.model.eval()
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for raw_batch in loader:
                sparse = {k: v.to(self.device) for k, v in raw_batch["sparse"].items()}
                dense = raw_batch["dense"].to(self.device)
                out = self.model(sparse, dense)
                pred = out["prediction"]
                all_probs.append(np.asarray(pred.detach().cpu().view(-1).tolist(), dtype=float))

        probs = np.concatenate(all_probs)
        if calibrate and self.scaler is not None:
            probs = self.scaler.calibrate(probs)
        return probs
