"""Unit tests for Trainer: fit, evaluate, predict, checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.models.deepfm import DeepFM
from src.training.trainer import Trainer, TrainerConfig


def _fc() -> dict:
    return {
        "restaurant_cuisine": {
            "name": "restaurant_cuisine",
            "type": "sparse",
            "vocab_size": 12,
            "embedding_dim": 8,
        },
        "campaign_id": {
            "name": "campaign_id",
            "type": "sparse",
            "vocab_size": 6,
            "embedding_dim": 8,
        },
        "ad_position": {
            "name": "ad_position",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
        "bid_amount": {
            "name": "bid_amount",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
        "norm_rating": {
            "name": "norm_rating",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
    }


def _df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "restaurant_cuisine": rng.integers(0, 12, n),
            "campaign_id": rng.integers(0, 6, n),
            "ad_position": rng.uniform(1, 5, n).astype(float),
            "bid_amount": rng.lognormal(0.5, 0.4, n),
            "norm_rating": rng.uniform(0.2, 0.95, n),
            "click": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def tmp_ckpt(tmp_path: Path) -> Path:
    return tmp_path / "ckpt.pt"


def test_trainer_fit_evaluate_predict(tmp_ckpt: Path) -> None:
    fc = _fc()
    model = DeepFM(fc, dnn_layers=[24, 12], dropout=0.1)
    cfg = TrainerConfig(lr=1e-2, batch_size=64, epochs=3, patience=10, num_workers=0)
    trainer = Trainer(
        model=model,
        feature_config=fc,
        config=cfg,
        device="cpu",
        checkpoint_path=tmp_ckpt,
        history_path=tmp_ckpt.parent / "hist.json",
    )
    train_df = _df(400, 1)
    val_df = _df(120, 2)
    hist = trainer.fit(train_df, val_df)
    assert len(hist) >= 1
    assert trainer.best_epoch >= 1

    metrics = trainer.evaluate(
        val_df,
        bucket_features=["restaurant_cuisine", "ad_position"],
    )
    assert "auc" in metrics
    assert "ece_10" in metrics
    assert "bucket_auc" in metrics

    sub = val_df.head(20).copy()
    p_cal = trainer.predict(sub, calibrate=True)
    trainer.predict(sub, calibrate=False)
    assert len(p_cal) == 20
    assert np.all((p_cal >= 0) & (p_cal <= 1))


def test_trainer_load_checkpoint_roundtrip(tmp_ckpt: Path) -> None:
    fc = _fc()
    model = DeepFM(fc, dnn_layers=[16, 8], dropout=0.0)
    trainer = Trainer(
        model=model,
        feature_config=fc,
        config=TrainerConfig(epochs=2, batch_size=32, patience=5, num_workers=0),
        device="cpu",
        checkpoint_path=tmp_ckpt,
        history_path=tmp_ckpt.parent / "h.json",
    )
    trainer.fit(_df(200, 3), _df(80, 4))

    model2 = DeepFM(fc, dnn_layers=[16, 8], dropout=0.0)
    t2 = Trainer(
        model=model2,
        feature_config=fc,
        device="cpu",
        checkpoint_path=tmp_ckpt,
    )
    t2.load_checkpoint(tmp_ckpt)
    assert t2.best_val_auc == trainer.best_val_auc


def test_trainer_safe_auc_single_class() -> None:
    fc = _fc()
    model = DeepFM(fc, dnn_layers=[8], dropout=0.0)
    trainer = Trainer(
        model=model, feature_config=fc, device="cpu", checkpoint_path=Path("/tmp/x.pt")
    )
    y = np.zeros(50)
    p = np.ones(50) * 0.5
    assert np.isnan(trainer._safe_auc(y, p))
