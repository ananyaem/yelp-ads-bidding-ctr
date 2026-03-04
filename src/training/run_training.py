"""CLI entry point for DeepFM training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import AD_IMPRESSIONS_PARQUET_PATH, DEFAULT_HPARAMS, MODELS_DIR, set_seed
from src.models.deepfm import DeepFM
from src.training.trainer import Trainer, TrainerConfig


def infer_feature_config(frame: pd.DataFrame, embedding_dim: int) -> dict[str, dict[str, Any]]:
    sparse_candidates = [
        "restaurant_city",
        "restaurant_cuisine",
        "campaign_id",
        "campaign_city",
        "campaign_cuisine",
        "time_of_day",
        "day_of_week",
    ]
    dense_candidates = [
        "ad_position",
        "bid_amount",
        "norm_rating",
        "price_distance",
        "cuisine_match",
        "is_evening",
        "is_weekend",
    ]

    config: dict[str, dict[str, Any]] = {}

    for col in sparse_candidates:
        if col not in frame.columns:
            continue
        vals = pd.Categorical(frame[col].astype(str))
        frame[col] = vals.codes.astype(np.int64)
        vocab_size = int(frame[col].max()) + 1 if len(frame) else 1
        config[col] = {
            "name": col,
            "type": "sparse",
            "vocab_size": vocab_size,
            "embedding_dim": int(embedding_dim),
        }

    for col in dense_candidates:
        if col not in frame.columns:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(np.float32)
        config[col] = {
            "name": col,
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        }

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepFM on ad impression data.")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_HPARAMS.embedding_dim)
    parser.add_argument("--lr", type=float, default=DEFAULT_HPARAMS.lr)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_HPARAMS.batch_size)
    parser.add_argument("--epochs", type=int, default=DEFAULT_HPARAMS.epochs)
    parser.add_argument("--seed", type=int, default=DEFAULT_HPARAMS.seed)
    parser.add_argument("--data-path", type=str, default=str(AD_IMPRESSIONS_PARQUET_PATH))
    args = parser.parse_args()

    set_seed(args.seed)
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing training data parquet: {data_path}")

    df = pd.read_parquet(data_path).copy()
    if "split" not in df.columns:
        raise ValueError("Expected split column with values train/val/test.")

    feature_config = infer_feature_config(df, embedding_dim=args.embedding_dim)
    if not feature_config:
        raise ValueError("Could not infer any features for training.")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Expected non-empty train/val/test splits.")

    model = DeepFM(
        feature_config=feature_config,
        dnn_layers=DEFAULT_HPARAMS.dnn_layers,
        dropout=DEFAULT_HPARAMS.dropout,
    )
    trainer = Trainer(
        model=model,
        feature_config=feature_config,
        config=TrainerConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=DEFAULT_HPARAMS.patience,
        ),
        checkpoint_path=MODELS_DIR / "best_deepfm.pt",
        history_path=MODELS_DIR / "training_history.json",
    )

    history = trainer.fit(train_df, val_df)
    print(f"Training complete. Best epoch={trainer.best_epoch}, best val AUC={trainer.best_val_auc:.5f}")
    print(f"History saved to {MODELS_DIR / 'training_history.json'}")
    print(f"Checkpoint saved to {MODELS_DIR / 'best_deepfm.pt'}")

    # Save feature config used for this run.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (MODELS_DIR / "feature_config.json").open("w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)

    metrics = trainer.evaluate(
        test_df,
        bucket_features=["restaurant_cuisine", "ad_position", "campaign_city"],
    )
    with (MODELS_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Test metrics:", metrics)

    # Save debiased predictions (position fixed to 1).
    pred = trainer.predict(test_df, calibrate=True)
    out = test_df[["timestamp", "user_id", "business_id", "campaign_id", "ad_position", "click"]].copy()
    out["debiased_pred_proba"] = pred
    out.to_parquet(MODELS_DIR / "test_predictions_debiased.parquet", index=False)

    # Keep history variable used to avoid lint complaining in notebooks-oriented environments.
    _ = history


if __name__ == "__main__":
    main()
