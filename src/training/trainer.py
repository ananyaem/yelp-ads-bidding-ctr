"""Training utilities for DeepFM CTR model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset

from src.config import MODELS_DIR
from src.models.calibration import PlattScaler
from src.models.deepfm import DeepFM


class AdClickDataset(Dataset):
    """Dataset that separates sparse/dense features from parquet-backed data."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_config: dict[str, dict[str, Any]],
        label_col: str = "click",
        position_feature_name: str = "ad_position",
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.feature_config = feature_config
        self.label_col = label_col
        self.position_feature_name = position_feature_name

        self.sparse_features = [k for k, v in feature_config.items() if v.get("type") == "sparse"]
        self.dense_features = [k for k, v in feature_config.items() if v.get("type") == "dense"]

        required = set(self.sparse_features + self.dense_features + [self.label_col])
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(f"Missing required columns for dataset: {missing}")

        self.sparse_data = {}
        for name in self.sparse_features:
            self.sparse_data[name] = torch.as_tensor(
                pd.to_numeric(self.frame[name], errors="coerce").fillna(0).astype(np.int64).to_numpy(),
                dtype=torch.long,
            )

        self.dense_matrix = torch.as_tensor(
            self.frame[self.dense_features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32).to_numpy(),
            dtype=torch.float32,
        )
        self.labels = torch.as_tensor(
            pd.to_numeric(self.frame[self.label_col], errors="coerce").fillna(0).astype(np.float32).to_numpy(),
            dtype=torch.float32,
        ).unsqueeze(-1)

        if self.position_feature_name in self.frame.columns:
            self.position = torch.as_tensor(
                pd.to_numeric(self.frame[self.position_feature_name], errors="coerce").fillna(1).astype(np.float32).to_numpy(),
                dtype=torch.float32,
            ).unsqueeze(-1)
        else:
            self.position = torch.ones((len(self.frame), 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sparse = {name: tensor[idx] for name, tensor in self.sparse_data.items()}
        dense = self.dense_matrix[idx]
        return {
            "sparse": sparse,
            "dense": dense,
            "position": self.position[idx],
            "label": self.labels[idx],
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    sparse_keys = list(batch[0]["sparse"].keys())
    sparse = {k: torch.stack([item["sparse"][k] for item in batch], dim=0) for k in sparse_keys}
    dense = torch.stack([item["dense"] for item in batch], dim=0)
    position = torch.stack([item["position"] for item in batch], dim=0)
    label = torch.stack([item["label"] for item in batch], dim=0)
    return {"sparse": sparse, "dense": dense, "position": position, "label": label}


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    batch_size: int = 2048
    epochs: int = 20
    patience: int = 3
    weight_decay: float = 0.0
    num_workers: int = 0


class Trainer:
    """Trainer wrapper with early stopping, logging, evaluation, and debiased predict."""

    def __init__(
        self,
        model: DeepFM,
        feature_config: dict[str, dict[str, Any]],
        config: TrainerConfig | None = None,
        device: str | torch.device | None = None,
        checkpoint_path: str | Path = MODELS_DIR / "best_deepfm.pt",
        history_path: str | Path = MODELS_DIR / "training_history.json",
    ) -> None:
        self.model = model
        self.feature_config = feature_config
        self.config = config or TrainerConfig()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.history_path = Path(history_path)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )
        self.criterion = torch.nn.BCELoss()

        self.history: list[dict[str, float]] = []
        self.best_val_auc = -np.inf
        self.best_epoch = -1
        self._calibrator_a: float = 1.0
        self._calibrator_b: float = 0.0
        self.platt_scaler = PlattScaler()
        self._position_dense_idx = self._find_dense_position_idx()

    def _find_dense_position_idx(self) -> int | None:
        dense_features = [k for k, v in self.feature_config.items() if v.get("type") == "dense"]
        try:
            return dense_features.index("ad_position")
        except ValueError:
            return None

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        sparse = {k: v.to(self.device) for k, v in batch["sparse"].items()}
        dense = batch["dense"].to(self.device)
        label = batch["label"].to(self.device)
        position = batch["position"].to(self.device)
        return {"sparse": sparse, "dense": dense, "label": label, "position": position}

    @staticmethod
    def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins) - 1
        ece = 0.0
        n = len(y_true)
        for b in range(n_bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            conf = y_prob[mask].mean()
            acc = y_true[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)
        return float(ece)

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))

    def _fit_calibrator(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        # Platt-like calibration via simple linear fit in logit space.
        eps = 1e-6
        clipped = np.clip(y_prob, eps, 1.0 - eps)
        logit = np.log(clipped / (1.0 - clipped))
        # Closed-form least squares for y ~ a*logit + b, then pass through sigmoid at predict.
        x = np.vstack([logit, np.ones_like(logit)]).T
        target = y_true.astype(float)
        coef, _, _, _ = np.linalg.lstsq(x, target, rcond=None)
        self._calibrator_a = float(coef[0])
        self._calibrator_b = float(coef[1])

    def _apply_calibrator(self, y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-6
        clipped = np.clip(y_prob, eps, 1.0 - eps)
        logit = np.log(clipped / (1.0 - clipped))
        cal = 1.0 / (1.0 + np.exp(-(self._calibrator_a * logit + self._calibrator_b)))
        return np.clip(cal, 0.0, 1.0)

    def _epoch_pass(self, loader: DataLoader, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        losses: list[float] = []
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for raw_batch in loader:
            batch = self._to_device(raw_batch)
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                out = self.model(batch["sparse"], batch["dense"])
                pred = out["prediction"]
                loss = self.criterion(pred, batch["label"])
                if train:
                    loss.backward()
                    self.optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            all_probs.append(np.asarray(pred.detach().cpu().view(-1).tolist(), dtype=float))
            all_labels.append(np.asarray(batch["label"].detach().cpu().view(-1).tolist(), dtype=float))

        return float(np.mean(losses)), np.concatenate(all_labels), np.concatenate(all_probs)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[dict[str, float]]:
        train_ds = AdClickDataset(train_df, self.feature_config)
        val_ds = AdClickDataset(val_df, self.feature_config)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=_collate_batch,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate_batch,
            pin_memory=(self.device.type == "cuda"),
        )

        no_improve_epochs = 0
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            train_loss, _, _ = self._epoch_pass(train_loader, train=True)
            val_loss, val_y, val_prob = self._epoch_pass(val_loader, train=False)
            val_auc = self._safe_auc(val_y, val_prob)
            val_ll = float(log_loss(val_y, np.clip(val_prob, 1e-6, 1.0 - 1e-6)))

            self.scheduler.step(val_loss)
            lr = float(self.optimizer.param_groups[0]["lr"])

            row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "val_logloss": float(val_ll),
                "lr": float(lr),
            }
            self.history.append(row)
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"val_auc={val_auc:.5f} val_logloss={val_ll:.5f} lr={lr:.6f}"
            )

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                no_improve_epochs = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "feature_config": self.feature_config,
                        "best_val_auc": self.best_val_auc,
                        "best_epoch": self.best_epoch,
                    },
                    self.checkpoint_path,
                )
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= self.config.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        # Restore best checkpoint and fit calibrators on validation set.
        self.load_checkpoint(self.checkpoint_path)
        _, val_y_best, val_prob_best = self._epoch_pass(val_loader, train=False)
        self._fit_calibrator(val_y_best, val_prob_best)
        self.platt_scaler.fit(val_y_best, val_prob_best)

        scaler_path = self.checkpoint_path.parent / "platt_scaler.pkl"
        self.platt_scaler.save(scaler_path)

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return self.history

    def evaluate(self, eval_df: pd.DataFrame, bucket_features: list[str] | None = None) -> dict[str, Any]:
        ds = AdClickDataset(eval_df, self.feature_config)
        loader = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate_batch,
            pin_memory=(self.device.type == "cuda"),
        )
        eval_loss, y_true, y_prob_raw = self._epoch_pass(loader, train=False)
        y_prob = self._apply_calibrator(y_prob_raw)

        metrics: dict[str, Any] = {
            "loss": float(eval_loss),
            "auc": self._safe_auc(y_true, y_prob),
            "logloss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6))),
            "ece_10": self._ece(y_true, y_prob, n_bins=10),
        }

        # Per-feature-bucket AUC.
        bucket_features = bucket_features or []
        bucket_auc: dict[str, dict[str, float]] = {}
        frame = eval_df.reset_index(drop=True).copy()
        frame["_y_true"] = y_true
        frame["_y_prob"] = y_prob
        for feat in bucket_features:
            if feat not in frame.columns:
                continue
            grouped = {}
            series = frame[feat]
            # Dense numeric -> quartile buckets. Categorical -> per value (top 20).
            if pd.api.types.is_numeric_dtype(series):
                buckets = pd.qcut(series.rank(method="first"), q=4, duplicates="drop")
                tmp = frame.groupby(buckets)
                for bucket, sub in tmp:
                    grouped[str(bucket)] = self._safe_auc(sub["_y_true"].to_numpy(), sub["_y_prob"].to_numpy())
            else:
                top_vals = series.astype(str).value_counts().head(20).index
                for v in top_vals:
                    sub = frame[series.astype(str) == v]
                    grouped[str(v)] = self._safe_auc(sub["_y_true"].to_numpy(), sub["_y_prob"].to_numpy())
            bucket_auc[feat] = grouped
        metrics["bucket_auc"] = bucket_auc
        return metrics

    def predict(self, feature_df: pd.DataFrame, calibrate: bool = True) -> np.ndarray:
        """Predict debiased probabilities with position forced to 1."""
        infer_df = feature_df.copy()
        if self._position_dense_idx is not None and "ad_position" in infer_df.columns:
            infer_df["ad_position"] = 1.0

        # AdClickDataset requires label, add a dummy one for inference.
        if "click" not in infer_df.columns:
            infer_df["click"] = 0

        ds = AdClickDataset(infer_df, self.feature_config)
        loader = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate_batch,
            pin_memory=(self.device.type == "cuda"),
        )
        _, _, y_prob = self._epoch_pass(loader, train=False)
        if calibrate:
            return self._apply_calibrator(y_prob)
        return y_prob

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_val_auc = float(ckpt.get("best_val_auc", self.best_val_auc))
        self.best_epoch = int(ckpt.get("best_epoch", self.best_epoch))

