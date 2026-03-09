"""Tests for training CLI helpers and error paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from src.training.run_training import infer_feature_config, main


def test_infer_feature_config_builds_sparse_and_dense() -> None:
    dfc = pd.DataFrame(
        {
            "restaurant_city": ["a", "b", "a"],
            "restaurant_cuisine": ["x", "y", "x"],
            "campaign_id": ["c1", "c1", "c2"],
            "time_of_day": [0, 1, 0],
            "day_of_week": [1, 2, 3],
            "ad_position": [1.0, 2.0, 1.0],
            "bid_amount": [1.0, 2.0, 3.0],
            "norm_rating": [0.5, 0.6, 0.7],
        }
    )
    out = infer_feature_config(dfc, embedding_dim=4)
    assert "restaurant_city" in out and out["restaurant_city"]["type"] == "sparse"
    assert "ad_position" in out and out["ad_position"]["type"] == "dense"
    assert pd.api.types.is_integer_dtype(dfc["restaurant_city"])


def test_main_raises_without_parquet(tmp_path: Path) -> None:
    missing = tmp_path / "nope.parquet"
    with patch("sys.argv", ["run_training", "--data-path", str(missing)]):
        with pytest.raises(FileNotFoundError, match="Missing training data parquet"):
            main()


def test_main_raises_without_split(tmp_path: Path) -> None:
    p = tmp_path / "d.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(p)
    with patch("sys.argv", ["run_training", "--data-path", str(p)]):
        with pytest.raises(ValueError, match="split column"):
            main()
