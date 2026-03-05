"""Tests for auction-aware bid optimizer: BudgetPacer, OptimalBidder, CampaignSimulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.auction.bid_optimizer import BudgetPacer, CampaignSimulator, OptimalBidder


# ---------------------------------------------------------------------------
# BudgetPacer
# ---------------------------------------------------------------------------

class TestBudgetPacer:
    def test_initial_multiplier_is_one(self) -> None:
        pacer = BudgetPacer(daily_budget=100.0, n_slots=24)
        assert pacer.pacing_multiplier == 1.0

    def test_remaining_budget_decreases(self) -> None:
        pacer = BudgetPacer(daily_budget=100.0, n_slots=10)
        pacer.record_spend(30.0)
        assert abs(pacer.remaining_budget - 70.0) < 1e-9

    def test_remaining_budget_never_negative(self) -> None:
        pacer = BudgetPacer(daily_budget=50.0, n_slots=10)
        pacer.record_spend(60.0)
        assert pacer.remaining_budget == 0.0

    def test_multiplier_clamped_to_bounds(self) -> None:
        pacer = BudgetPacer(daily_budget=100.0, n_slots=10, alpha=0.3)

        # Extreme underspend → multiplier should rise but stay <= 3.0
        for _ in range(50):
            pacer.update(actual_spend=0.0, time_elapsed=0.9)
        assert pacer.pacing_multiplier <= BudgetPacer.MAX_MULTIPLIER

        # Extreme overspend → multiplier should drop but stay >= 0.1
        pacer2 = BudgetPacer(daily_budget=100.0, n_slots=10, alpha=0.3)
        for _ in range(50):
            pacer2.update(actual_spend=200.0, time_elapsed=0.1)
        assert pacer2.pacing_multiplier >= BudgetPacer.MIN_MULTIPLIER

    def test_underspend_increases_multiplier(self) -> None:
        pacer = BudgetPacer(daily_budget=100.0, n_slots=10)
        pm_before = pacer.pacing_multiplier
        pacer.update(actual_spend=0.0, time_elapsed=0.5)
        assert pacer.pacing_multiplier > pm_before

    def test_overspend_decreases_multiplier(self) -> None:
        pacer = BudgetPacer(daily_budget=100.0, n_slots=10)
        pm_before = pacer.pacing_multiplier
        pacer.update(actual_spend=80.0, time_elapsed=0.3)
        assert pacer.pacing_multiplier < pm_before

    def test_on_track_keeps_multiplier_near_one(self) -> None:
        pacer = BudgetPacer(daily_budget=100.0, n_slots=10, alpha=0.3)
        pacer.update(actual_spend=50.0, time_elapsed=0.5)
        assert abs(pacer.pacing_multiplier - 1.0) < 0.05

    def test_update_with_zero_time_elapsed_is_noop(self) -> None:
        pacer = BudgetPacer(daily_budget=100.0, n_slots=10)
        pm = pacer.pacing_multiplier
        pacer.update(actual_spend=10.0, time_elapsed=0.0)
        assert pacer.pacing_multiplier == pm

    def test_alpha_parameter_respected(self) -> None:
        """Higher alpha → faster response to underspend."""
        slow = BudgetPacer(daily_budget=100.0, n_slots=10, alpha=0.1)
        fast = BudgetPacer(daily_budget=100.0, n_slots=10, alpha=0.9)
        slow.update(actual_spend=0.0, time_elapsed=0.5)
        fast.update(actual_spend=0.0, time_elapsed=0.5)
        assert fast.pacing_multiplier > slow.pacing_multiplier


# ---------------------------------------------------------------------------
# OptimalBidder
# ---------------------------------------------------------------------------

class TestOptimalBidder:
    def test_basic_bid(self) -> None:
        bidder = OptimalBidder()
        bid = bidder.compute_bid(value_per_click=5.0, predicted_ctr=0.10)
        assert abs(bid - 0.50) < 1e-6

    def test_bid_with_pacing(self) -> None:
        bidder = OptimalBidder()
        bid = bidder.compute_bid(
            value_per_click=5.0, predicted_ctr=0.10, pacing_multiplier=2.0
        )
        assert abs(bid - 1.0) < 1e-6

    def test_bid_floored_at_min(self) -> None:
        bidder = OptimalBidder(min_bid=0.01)
        bid = bidder.compute_bid(value_per_click=0.01, predicted_ctr=0.001)
        assert bid == pytest.approx(0.01)

    def test_bid_capped_at_max(self) -> None:
        bidder = OptimalBidder(max_bid=10.0)
        bid = bidder.compute_bid(
            value_per_click=100.0, predicted_ctr=0.50, pacing_multiplier=3.0
        )
        assert bid == pytest.approx(10.0)

    def test_bid_always_non_negative(self) -> None:
        bidder = OptimalBidder()
        for vpc in [0.0, -1.0, 5.0]:
            for ctr in [0.0, -0.1, 0.5]:
                bid = bidder.compute_bid(vpc, ctr)
                assert bid >= 0.0

    def test_bid_monotonic_in_ctr(self) -> None:
        bidder = OptimalBidder()
        bids = [bidder.compute_bid(5.0, ctr) for ctr in [0.01, 0.05, 0.10, 0.30]]
        assert all(bids[i] <= bids[i + 1] for i in range(len(bids) - 1))


# ---------------------------------------------------------------------------
# CampaignSimulator
# ---------------------------------------------------------------------------

class TestCampaignSimulator:
    def test_returns_trajectory_dataframe(self) -> None:
        sim = CampaignSimulator(n_rounds=5, impressions_per_round=10, seed=42)
        df = sim.simulate()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        for col in [
            "round", "round_spend", "round_clicks", "round_impressions",
            "total_spend", "total_clicks", "total_impressions",
            "budget_utilization", "actual_CPA", "impression_share",
            "pacing_multiplier",
        ]:
            assert col in df.columns

    def test_deterministic(self) -> None:
        sim1 = CampaignSimulator(seed=99)
        sim2 = CampaignSimulator(seed=99)
        pd.testing.assert_frame_equal(sim1.simulate(), sim2.simulate())

    def test_budget_utilization_above_85_pct(self) -> None:
        sim = CampaignSimulator(seed=42)
        df = sim.simulate()
        assert df.iloc[-1]["budget_utilization"] > 0.85

    def test_spend_never_exceeds_budget(self) -> None:
        sim = CampaignSimulator(seed=42)
        df = sim.simulate()
        assert df.iloc[-1]["total_spend"] <= sim.daily_budget + 1e-6

    def test_campaign_stops_when_budget_exhausted(self) -> None:
        sim = CampaignSimulator(
            daily_budget=2.0,
            n_rounds=24,
            impressions_per_round=50,
            seed=42,
        )
        df = sim.simulate()
        final = df.iloc[-1]
        assert final["total_spend"] <= 2.0 + 1e-6
        # Should stop gaining impressions once budget is gone
        late_rounds = df.iloc[-5:]
        assert (late_rounds["round_impressions"] == 0).any()

    def test_cumulative_metrics_monotonically_increase(self) -> None:
        sim = CampaignSimulator(seed=42)
        df = sim.simulate()
        for col in ["total_spend", "total_clicks", "total_impressions"]:
            values = df[col].values
            assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    def test_target_cpa_equals_value_per_click(self) -> None:
        sim = CampaignSimulator(value_per_click=7.0)
        assert sim.target_cpa == 7.0


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

class TestAcceptance:
    def test_default_simulation_meets_criteria(self) -> None:
        """Budget utilization > 90% and CPA within 20% of target."""
        sim = CampaignSimulator(seed=42)
        df = sim.simulate()
        final = df.iloc[-1]

        util = final["budget_utilization"]
        cpa = final["actual_CPA"]
        cpa_deviation = abs(cpa - sim.target_cpa) / sim.target_cpa

        assert util > 0.90, f"Budget utilization {util:.1%} <= 90%"
        assert cpa_deviation < 0.20, (
            f"CPA {cpa:.2f} deviates {cpa_deviation:.1%} from "
            f"target {sim.target_cpa:.2f} (> 20%)"
        )
