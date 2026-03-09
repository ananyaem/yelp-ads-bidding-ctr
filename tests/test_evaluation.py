"""Tests for the extended evaluation metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.models.evaluation import (
    dcg_at_k,
    full_evaluation_report,
    grouped_ndcg,
    ndcg_at_k,
    per_segment_auc,
    per_segment_ece,
    simulate_revenue,
)


def _make_eval_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clicks = rng.binomial(1, 0.06, size=n)
    probs = np.clip(clicks * 0.5 + rng.normal(0.06, 0.03, size=n), 0.001, 0.999)
    return pd.DataFrame(
        {
            "impression_id": [f"imp_{i}" for i in range(n)],
            "click": clicks,
            "y_prob": probs,
            "bid_amount": rng.lognormal(0.7, 0.5, size=n),
            "restaurant_cuisine": rng.choice(["Italian", "Japanese", "Mexican", "Other"], size=n),
            "campaign_id": rng.integers(0, 5, size=n),
            "ad_position": rng.integers(1, 11, size=n),
        }
    )


def test_dcg_at_k_perfect_ranking() -> None:
    relevance = np.array([1.0, 1.0, 0.0, 0.0])
    dcg = dcg_at_k(relevance, k=4)
    assert dcg > 0


def test_ndcg_perfect_score() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2])
    score = ndcg_at_k(y_true, y_score, k=4)
    assert score == 1.0


def test_ndcg_all_zeros_returns_zero() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    assert ndcg_at_k(y_true, y_score, k=4) == 0.0


def test_grouped_ndcg_has_mean() -> None:
    df = _make_eval_df(200)
    result = grouped_ndcg(df, "campaign_id", k=5)
    assert "__mean__" in result
    assert 0.0 <= result["__mean__"] <= 1.0


def test_per_segment_auc_returns_dict() -> None:
    df = _make_eval_df(500)
    result = per_segment_auc(df, "restaurant_cuisine")
    assert isinstance(result, dict)
    assert len(result) > 0
    for auc_val in result.values():
        assert 0.0 <= auc_val <= 1.0 or np.isnan(auc_val)


def test_per_segment_ece_returns_list() -> None:
    df = _make_eval_df(500)
    result = per_segment_ece(df, "restaurant_cuisine")
    assert isinstance(result, list)
    for seg in result:
        assert seg.ece >= 0.0
        assert seg.count > 0


def test_simulate_revenue_all_strategies() -> None:
    df = _make_eval_df(200)
    result = simulate_revenue(df, value_per_click=5.0)
    assert "truthful" in result
    assert "conservative" in result
    assert "aggressive" in result
    assert "uniform" in result
    for r in result.values():
        assert r.total_clicks >= 0
        assert r.total_impressions >= 0


def test_full_evaluation_report_keys() -> None:
    df = _make_eval_df(300)
    report = full_evaluation_report(
        df,
        segment_cols=["restaurant_cuisine", "ad_position"],
        group_ndcg_col="campaign_id",
    )
    assert "auc" in report
    assert "logloss" in report
    assert "global_ndcg" in report
    assert "grouped_ndcg" in report
    assert "segment_auc" in report
    assert "segment_calibration" in report
