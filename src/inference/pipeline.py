"""End-to-end sponsored listing inference: DeepFM / ONNX, calibration, GSP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.auction.gsp import GSPAuction
from src.config import DEFAULT_HPARAMS, MODELS_DIR
from src.features.engineer import FeatureEngineer
from src.models.calibration import PlattScaler
from src.models.deepfm import DeepFM
from src.training.trainer import AdClickDataset, _collate_batch

logger = logging.getLogger(__name__)


def _infer_dnn_layer_sizes(state_dict: dict[str, torch.Tensor]) -> list[int]:
    """Recover DNN hidden sizes from DeepFM checkpoint weights."""
    sizes: list[int] = []
    i = 0
    prefix = "dnn_layer.network."
    while True:
        key = f"{prefix}{i}.weight"
        if key not in state_dict:
            break
        w = state_dict[key]
        if w.dim() == 2:
            sizes.append(int(w.shape[0]))
        i += 4  # Linear, BatchNorm, ReLU, Dropout per block
        if i > 200:
            break
    return sizes


class InferencePipeline:
    """Load DeepFM (PyTorch or ONNX), FeatureEngineer, Platt scaling, and GSP ranking."""

    def __init__(
        self,
        model_path: str | Path,
        engineer_path: str | Path | None = None,
        platt_path: str | Path | None = None,
        onnx_path: str | Path | None = None,
        use_onnx: bool = False,
        auction: GSPAuction | None = None,
        device: str | torch.device | None = None,
        dnn_layers: list[int] | None = None,
        dropout: float | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.engineer_path = Path(engineer_path).resolve() if engineer_path else None
        self.platt_path = Path(platt_path).resolve() if platt_path is not None else None
        self.onnx_path = Path(onnx_path).resolve() if onnx_path is not None else None
        self.use_onnx = bool(use_onnx)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.auction = auction or GSPAuction()
        self._dnn_override = dnn_layers
        self._dropout = dropout if dropout is not None else DEFAULT_HPARAMS.dropout

        self.engineer: FeatureEngineer | None = None
        self.platt: PlattScaler | None = None
        self.model: DeepFM | None = None
        self.feature_config: dict[str, dict[str, Any]] = {}
        self._ort_session: Any = None
        self._sparse_keys: list[str] = []

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at expected path: {self.model_path}"
            )

        if self.engineer_path is not None:
            if not self.engineer_path.is_file():
                raise FileNotFoundError(
                    f"FeatureEngineer pickle not found at expected path: {self.engineer_path}"
                )
            self.engineer = FeatureEngineer.load(self.engineer_path)

        try:
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(self.model_path, map_location="cpu")
        self.feature_config = ckpt["feature_config"]
        inferred = _infer_dnn_layer_sizes(ckpt["model_state_dict"])
        layers = self._dnn_override if self._dnn_override is not None else (
            inferred if inferred else list(DEFAULT_HPARAMS.dnn_layers)
        )
        self.model = DeepFM(self.feature_config, dnn_layers=layers, dropout=self._dropout)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._sparse_keys = [k for k, v in self.feature_config.items() if v.get("type") == "sparse"]

        if "primary_cuisine" in self.feature_config and self.engineer is None:
            raise ValueError(
                "Checkpoint uses review-style features (e.g. primary_cuisine). "
                "Pass engineer_path to a fitted FeatureEngineer pickle from FeatureEngineer.save()."
            )

        if self.platt_path is not None:
            if not self.platt_path.is_file():
                raise FileNotFoundError(
                    f"Platt scaler not found at expected path: {self.platt_path}"
                )
            self.platt = PlattScaler.load(self.platt_path)

        if self.use_onnx:
            if self.onnx_path is None or not self.onnx_path.is_file():
                raise FileNotFoundError(
                    f"use_onnx=True but ONNX file not found at expected path: {self.onnx_path}"
                )
            try:
                import onnxruntime as ort  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for ONNX inference. Install with: pip install onnxruntime"
                ) from exc
            self._ort_session = ort.InferenceSession(
                str(self.onnx_path), providers=["CPUExecutionProvider"]
            )

    def _needs_engineer_transform(self) -> bool:
        return self.engineer is not None and "primary_cuisine" in self.feature_config

    def _build_feature_frame(
        self,
        user_features: dict[str, Any],
        context: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> pd.DataFrame:
        sparse_names = [k for k, v in self.feature_config.items() if v.get("type") == "sparse"]
        dense_names = [k for k, v in self.feature_config.items() if v.get("type") == "dense"]

        ts = context.get("timestamp", context.get("date"))
        if ts is None:
            raise ValueError("context must include 'timestamp' or 'date'")
        ts = pd.to_datetime(ts)

        user_id = user_features.get("user_id")
        if user_id is None:
            raise ValueError("user_features must include 'user_id'")

        if self._needs_engineer_transform():
            assert self.engineer is not None
            inter = pd.DataFrame(
                {
                    "user_id": [str(user_id)] * len(candidates),
                    "business_id": [str(c["business_id"]) for c in candidates],
                    "date": [ts] * len(candidates),
                }
            )
            base = self.engineer.transform(inter)
        else:
            base = pd.DataFrame(index=range(len(candidates)))

        for col in sparse_names + dense_names:
            if col in base.columns:
                continue
            if col == "ad_position":
                base[col] = float(context.get("ad_position", 1.0))
            elif col == "bid_amount":
                base[col] = [float(c.get("bid", c.get("bid_amount", 0.0))) for c in candidates]
            else:
                vals = []
                for c in candidates:
                    if col in c:
                        vals.append(c[col])
                    elif col in context:
                        vals.append(context[col])
                    elif col in user_features:
                        v = user_features[col]
                        vals.append(v if not isinstance(v, (list, np.ndarray)) else v)
                    else:
                        vals.append(0)
                base[col] = vals

        for col in sparse_names:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0).astype(np.int64)
        for col in dense_names:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0).astype(np.float64)

        ordered = sparse_names + dense_names
        return base[ordered]

    def _predict_pytorch(self, feature_df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        infer_df = feature_df.copy()
        if "click" not in infer_df.columns:
            infer_df["click"] = 0
        dense_feats = [k for k, v in self.feature_config.items() if v.get("type") == "dense"]
        if "ad_position" in infer_df.columns and "ad_position" in dense_feats:
            infer_df["ad_position"] = 1.0

        ds = AdClickDataset(infer_df, self.feature_config)
        loader = DataLoader(ds, batch_size=min(512, len(ds)), shuffle=False, collate_fn=_collate_batch)
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                sparse = {k: v.to(self.device) for k, v in batch["sparse"].items()}
                dense = batch["dense"].to(self.device)
                out = self.model(sparse, dense)
                p = np.asarray(
                    out["prediction"].detach().cpu().view(-1).tolist(),
                    dtype=np.float64,
                )
                probs.append(p)
        raw = np.concatenate(probs) if probs else np.array([], dtype=np.float64)
        if self.platt is not None:
            return np.asarray(self.platt.calibrate(raw), dtype=np.float64)
        return raw

    def _predict_onnx(self, feature_df: pd.DataFrame) -> np.ndarray:
        if self._ort_session is None:
            raise RuntimeError("ONNX session not initialized")
        infer_df = feature_df.copy()
        sparse_mat = np.stack(
            [infer_df[k].to_numpy(dtype=np.int64) for k in self._sparse_keys],
            axis=1,
        )
        dense_names = [k for k, v in self.feature_config.items() if v.get("type") == "dense"]
        dense_mat = infer_df[dense_names].to_numpy(dtype=np.float32)

        input_names = [inp.name for inp in self._ort_session.get_inputs()]
        feeds = {input_names[0]: sparse_mat, input_names[1]: dense_mat}
        out = self._ort_session.run(None, feeds)[0]
        raw = np.asarray(out, dtype=np.float64).reshape(-1)
        if self.platt is not None:
            return np.asarray(self.platt.calibrate(raw), dtype=np.float64)
        return raw

    def get_sponsored_listings(
        self,
        user_features: dict[str, Any],
        context: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank candidates by quality score, price with GSP, return listing dicts."""
        if not candidates:
            return []

        if self._needs_engineer_transform() and self.engineer is None:
            raise RuntimeError("This checkpoint expects a fitted FeatureEngineer.")

        for c in candidates:
            if "business_id" not in c:
                raise ValueError("Each candidate must include 'business_id'")
            if "bid" not in c and "bid_amount" not in c:
                raise ValueError("Each candidate must include 'bid' or 'bid_amount'")
            if "campaign_id" not in c:
                raise ValueError("Each candidate must include 'campaign_id'")
            if "restaurant_id" not in c:
                raise ValueError("Each candidate must include 'restaurant_id'")

        feature_df = self._build_feature_frame(user_features, context, candidates)

        if self.use_onnx and self._ort_session is not None:
            ctr = self._predict_onnx(feature_df)
        else:
            ctr = self._predict_pytorch(feature_df)

        if np.all(ctr <= 0):
            logger.warning(
                "All predicted CTRs are non-positive after calibration; returning empty listings."
            )
            return []

        gsp_candidates: list[dict[str, Any]] = []
        for i, c in enumerate(candidates):
            bid = float(c.get("bid", c.get("bid_amount", 0.0)))
            gsp_candidates.append(
                {
                    "restaurant_id": str(c["restaurant_id"]),
                    "bid": bid,
                    "predicted_ctr": float(max(ctr[i], 1e-12)),
                    "campaign_id": str(c["campaign_id"]),
                }
            )

        ranked = self.auction.rank_ads(gsp_candidates)
        prices = self.auction.compute_prices(ranked)

        out: list[dict[str, Any]] = []
        for pos, (ad, price) in enumerate(zip(ranked, prices), start=1):
            out.append(
                {
                    "restaurant": ad["restaurant_id"],
                    "predicted_ctr": float(ad["predicted_ctr"]),
                    "price": float(price),
                    "position": pos,
                }
            )
        return out


def default_model_paths() -> dict[str, Path]:
    """Convention paths under ``models/`` (may not exist on a fresh clone)."""
    return {
        "model": MODELS_DIR / "best_deepfm.pt",
        "engineer": MODELS_DIR / "feature_engineer.pkl",
        "platt": MODELS_DIR / "platt_scaler.pkl",
        "onnx": MODELS_DIR / "deepfm.onnx",
    }
