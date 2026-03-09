"""Tests for the bidding optimization module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.bidding.optimizer import BidOptimizer, BudgetPacer


def _make_auction_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "impression_id": [f"imp_{i}" for i in range(n)],
            "y_prob": np.clip(rng.normal(0.06, 0.03, size=n), 0.001, 0.999),
            "bid_amount": rng.lognormal(0.7, 0.5, size=n),
            "click": rng.binomial(1, 0.06, size=n),
        }
    )


def test_compute_bid_second_price() -> None:
    opt = BidOptimizer(value_per_click=5.0, auction_type="second_price")
    bid = opt.compute_bid(predicted_ctr=0.10)
    assert abs(bid - 0.50) < 1e-6


def test_compute_bid_first_price_shaded() -> None:
    opt = BidOptimizer(value_per_click=5.0, auction_type="first_price", shade_factor=0.7)
    bid = opt.compute_bid(predicted_ctr=0.10)
    assert abs(bid - 0.35) < 1e-6


def test_compute_bid_clamps_to_range() -> None:
    opt = BidOptimizer(value_per_click=5.0, min_bid=0.01, max_bid=50.0)
    assert opt.compute_bid(0.0001) == 0.01
    assert opt.compute_bid(1.0) <= 50.0


def test_compute_bids_array() -> None:
    opt = BidOptimizer(value_per_click=10.0)
    ctrs = np.array([0.01, 0.05, 0.10, 0.50])
    bids = opt.compute_bids(ctrs)
    assert len(bids) == 4
    assert all(bids[i] <= bids[i + 1] for i in range(3))


def test_invalid_auction_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown auction_type"):
        BidOptimizer(auction_type="invalid")


def test_budget_pacer_remaining() -> None:
    pacer = BudgetPacer(total_budget=100.0, n_periods=10)
    assert pacer.remaining_budget == 100.0
    pacer.record_spend(30.0)
    assert abs(pacer.remaining_budget - 70.0) < 1e-6


def test_budget_pacer_advance_adjusts_multiplier() -> None:
    pacer = BudgetPacer(total_budget=100.0, n_periods=10)
    pacer.record_spend(5.0)  # under-spent (target is 10)
    m = pacer.advance_period()
    assert m > 1.0  # should increase multiplier to catch up


def test_simulate_auction_basic() -> None:
    df = _make_auction_df(100)
    opt = BidOptimizer(value_per_click=5.0)
    results, summary = opt.simulate_auction(df)
    assert len(results) == 100
    assert summary["total_impressions"] == 100
    assert summary["won_impressions"] >= 0
    assert summary["total_clicks"] >= 0


def test_simulate_auction_with_budget() -> None:
    df = _make_auction_df(200)
    opt = BidOptimizer(value_per_click=5.0)
    _, summary = opt.simulate_auction(df, budget=10.0, n_periods=5)
    assert summary["total_cost"] <= 12.0  # small overshoot allowed from granularity
