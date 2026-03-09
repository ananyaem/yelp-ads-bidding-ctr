"""Tests for the GSP auction engine and simulation."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from src.auction.gsp import AuctionSimulator, GSPAuction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate(
    restaurant_id: str,
    bid: float,
    predicted_ctr: float,
    campaign_id: str,
) -> dict:
    return {
        "restaurant_id": restaurant_id,
        "bid": bid,
        "predicted_ctr": predicted_ctr,
        "campaign_id": campaign_id,
    }


def _five_bidders() -> list[dict]:
    return [
        _candidate("r1", bid=2.00, predicted_ctr=0.10, campaign_id="c1"),  # score 0.20
        _candidate("r2", bid=3.00, predicted_ctr=0.20, campaign_id="c2"),  # score 0.60
        _candidate("r3", bid=1.50, predicted_ctr=0.30, campaign_id="c3"),  # score 0.45
        _candidate("r4", bid=5.00, predicted_ctr=0.05, campaign_id="c4"),  # score 0.25
        _candidate("r5", bid=4.00, predicted_ctr=0.15, campaign_id="c5"),  # score 0.60
    ]


# ---------------------------------------------------------------------------
# GSPAuction.rank_ads
# ---------------------------------------------------------------------------


class TestRankAds:
    def test_normal_ranking_order(self) -> None:
        auction = GSPAuction()
        ranked = auction.rank_ads(_five_bidders())

        scores = [ad["rank_score"] for ad in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_five_bidders_expected_order(self) -> None:
        auction = GSPAuction()
        ranked = auction.rank_ads(_five_bidders())
        ids = [ad["restaurant_id"] for ad in ranked]
        # scores: r2=0.60, r5=0.60, r3=0.45, r4=0.25, r1=0.20
        # r2 and r5 tie at 0.60; r2 bid=3.0 < r5 bid=4.0 → r2 first
        assert ids == ["r2", "r5", "r3", "r4", "r1"]

    def test_zero_ctr_skipped(self) -> None:
        auction = GSPAuction()
        candidates = [
            _candidate("r1", bid=5.0, predicted_ctr=0.0, campaign_id="c1"),
            _candidate("r2", bid=2.0, predicted_ctr=0.10, campaign_id="c2"),
        ]
        ranked = auction.rank_ads(candidates)
        assert len(ranked) == 1
        assert ranked[0]["restaurant_id"] == "r2"

    def test_negative_ctr_skipped(self) -> None:
        auction = GSPAuction()
        candidates = [
            _candidate("r1", bid=5.0, predicted_ctr=-0.05, campaign_id="c1"),
        ]
        ranked = auction.rank_ads(candidates)
        assert ranked == []

    def test_empty_candidates(self) -> None:
        auction = GSPAuction()
        assert auction.rank_ads([]) == []

    def test_tied_scores_broken_by_lower_bid(self) -> None:
        auction = GSPAuction()
        candidates = [
            _candidate("r_high_bid", bid=4.0, predicted_ctr=0.10, campaign_id="c1"),  # 0.40
            _candidate("r_low_bid", bid=2.0, predicted_ctr=0.20, campaign_id="c2"),  # 0.40
        ]
        ranked = auction.rank_ads(candidates)
        assert ranked[0]["restaurant_id"] == "r_low_bid"
        assert ranked[1]["restaurant_id"] == "r_high_bid"


# ---------------------------------------------------------------------------
# GSPAuction.compute_prices
# ---------------------------------------------------------------------------


class TestComputePrices:
    def test_five_bidders_prices(self) -> None:
        auction = GSPAuction(reserve_price=0.10, epsilon=0.01)
        ranked = auction.rank_ads(_five_bidders())
        prices = auction.compute_prices(ranked)

        assert len(prices) == 5
        # Order: r2, r5, r3, r4, r1 — GSP + bid cap
        assert prices[0] == pytest.approx(3.00)  # min(3, 0.60/0.20 + 0.01)
        assert prices[1] == pytest.approx(3.01)  # min(4, 0.45/0.15 + 0.01)
        assert prices[2] == pytest.approx(0.25 / 0.30 + 0.01)  # min(1.5, …)
        assert prices[3] == pytest.approx(4.01)  # min(5, 0.20/0.05 + 0.01)
        assert prices[-1] == pytest.approx(0.10)  # reserve

    def test_single_bidder_pays_reserve(self) -> None:
        auction = GSPAuction(reserve_price=0.10)
        ranked = auction.rank_ads([_candidate("r1", bid=5.0, predicted_ctr=0.20, campaign_id="c1")])
        prices = auction.compute_prices(ranked)
        assert len(prices) == 1
        assert prices[0] == pytest.approx(0.10)

    def test_no_winner_pays_more_than_bid(self) -> None:
        auction = GSPAuction(reserve_price=0.10, epsilon=0.01)
        ranked = auction.rank_ads(_five_bidders())
        prices = auction.compute_prices(ranked)

        for ad, price in zip(ranked, prices):
            assert price <= ad["bid"] + 1e-9

    def test_empty_list_returns_empty(self) -> None:
        auction = GSPAuction()
        assert auction.compute_prices([]) == []

    def test_price_ordering_is_non_increasing(self) -> None:
        """Higher-ranked ads should generally pay >= lower-ranked ads."""
        auction = GSPAuction(reserve_price=0.10, epsilon=0.01)
        candidates = [
            _candidate(f"r{i}", bid=float(5 - i), predicted_ctr=0.05 * (5 - i), campaign_id=f"c{i}")
            for i in range(5)
        ]
        ranked = auction.rank_ads(candidates)
        prices = auction.compute_prices(ranked)
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1] - 1e-9

    def test_gsp_price_formula(self) -> None:
        """Manually verify the GSP price formula for a two-bidder case."""
        auction = GSPAuction(reserve_price=0.10, epsilon=0.01)
        candidates = [
            _candidate("r1", bid=3.0, predicted_ctr=0.20, campaign_id="c1"),  # score 0.60
            _candidate("r2", bid=2.0, predicted_ctr=0.10, campaign_id="c2"),  # score 0.20
        ]
        ranked = auction.rank_ads(candidates)
        prices = auction.compute_prices(ranked)

        # r1 pays: score(r2)/ctr(r1) + eps = 0.20/0.20 + 0.01 = 1.01
        assert prices[0] == pytest.approx(1.01)
        # r2 pays reserve
        assert prices[1] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# GSPAuction.apply_budget_constraints
# ---------------------------------------------------------------------------


class TestBudgetConstraints:
    def test_exhausted_campaign_removed(self) -> None:
        auction = GSPAuction()
        ranked = auction.rank_ads(_five_bidders())
        budgets = {"c2": 0.0, "c5": 10.0}
        filtered = auction.apply_budget_constraints(ranked, budgets)

        campaign_ids = {ad["campaign_id"] for ad in filtered}
        assert "c2" not in campaign_ids
        assert "c5" in campaign_ids

    def test_unlisted_campaigns_pass_through(self) -> None:
        auction = GSPAuction()
        ranked = auction.rank_ads(
            [_candidate("r1", bid=2.0, predicted_ctr=0.10, campaign_id="c_unknown")]
        )
        filtered = auction.apply_budget_constraints(ranked, {"c_other": 0.0})
        assert len(filtered) == 1

    def test_all_budgets_exhausted_returns_empty(self) -> None:
        auction = GSPAuction()
        ranked = auction.rank_ads(
            [
                _candidate("r1", bid=2.0, predicted_ctr=0.10, campaign_id="c1"),
                _candidate("r2", bid=3.0, predicted_ctr=0.20, campaign_id="c2"),
            ]
        )
        budgets = {"c1": 0.0, "c2": 0.0}
        filtered = auction.apply_budget_constraints(ranked, budgets)
        assert filtered == []


# ---------------------------------------------------------------------------
# GSPAuction.run_auction (integration)
# ---------------------------------------------------------------------------


class TestRunAuction:
    def test_run_auction_end_to_end(self) -> None:
        auction = GSPAuction(max_slots=3)
        winners, prices = auction.run_auction(_five_bidders())
        assert len(winners) == 3
        assert len(prices) == 3

    def test_run_auction_empty_candidates(self) -> None:
        auction = GSPAuction()
        winners, prices = auction.run_auction([])
        assert winners == []
        assert prices == []


# ---------------------------------------------------------------------------
# AuctionSimulator
# ---------------------------------------------------------------------------


class TestAuctionSimulator:
    def test_basic_simulation(self) -> None:
        sim = AuctionSimulator(seed=42)
        requests = [_five_bidders() for _ in range(20)]
        summary = sim.simulate(requests)

        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty
        for col in [
            "campaign_id",
            "spend",
            "impressions",
            "clicks",
            "revenue",
            "avg_cpc",
            "social_welfare",
            "advertiser_roi",
        ]:
            assert col in summary.columns

    def test_deterministic_with_same_seed(self) -> None:
        requests = [_five_bidders() for _ in range(50)]

        s1 = AuctionSimulator(seed=99).simulate(requests)
        s2 = AuctionSimulator(seed=99).simulate(requests)

        pd.testing.assert_frame_equal(s1, s2)

    def test_different_seed_gives_different_results(self) -> None:
        requests = [_five_bidders() for _ in range(200)]

        s1 = AuctionSimulator(seed=1).simulate(requests)
        s2 = AuctionSimulator(seed=2).simulate(requests)

        assert not s1["clicks"].equals(s2["clicks"])

    def test_empty_requests(self) -> None:
        sim = AuctionSimulator()
        summary = sim.simulate([])
        assert summary.empty
        assert list(summary.columns) == AuctionSimulator._SUMMARY_COLUMNS

    def test_empty_inner_candidates(self) -> None:
        sim = AuctionSimulator()
        summary = sim.simulate([[], [], []])
        assert summary.empty

    def test_budget_exhaustion_mid_simulation(self) -> None:
        auction = GSPAuction(reserve_price=0.10, max_slots=1)
        sim = AuctionSimulator(auction=auction, seed=42)

        candidates = [
            _candidate("r1", bid=2.0, predicted_ctr=0.30, campaign_id="c1"),
        ]
        requests = [candidates for _ in range(100)]
        budgets = {"c1": 0.50}

        summary = sim.simulate(requests, campaign_budgets=budgets)

        if not summary.empty:
            c1_row = summary.loc[summary["campaign_id"] == "c1"]
            if not c1_row.empty:
                assert float(c1_row["spend"].iloc[0]) <= 0.50 + 1e-9

    def test_caller_budgets_not_mutated(self) -> None:
        sim = AuctionSimulator(seed=42)
        requests = [_five_bidders() for _ in range(10)]
        original_budgets = {"c1": 100.0, "c2": 100.0}
        budgets_copy = dict(original_budgets)

        sim.simulate(requests, campaign_budgets=budgets_copy)
        assert budgets_copy == original_budgets

    def test_aggregate_metrics(self) -> None:
        sim = AuctionSimulator(seed=42)
        requests = [_five_bidders() for _ in range(50)]
        summary = sim.simulate(requests)
        metrics = sim.compute_aggregate_metrics(summary)

        assert "total_revenue" in metrics
        assert "avg_cpc" in metrics
        assert "social_welfare" in metrics
        assert "advertiser_roi" in metrics
        assert metrics["total_revenue"] >= 0
        assert metrics["social_welfare"] >= 0

    def test_aggregate_metrics_empty(self) -> None:
        sim = AuctionSimulator()
        empty = pd.DataFrame(columns=AuctionSimulator._SUMMARY_COLUMNS)
        metrics = sim.compute_aggregate_metrics(empty)
        assert metrics == {
            "total_revenue": 0.0,
            "avg_cpc": 0.0,
            "social_welfare": 0.0,
            "advertiser_roi": 0.0,
        }


# ---------------------------------------------------------------------------
# Price verification: no winner pays more than their bid
# ---------------------------------------------------------------------------


class TestPriceNeverExceedsBid:
    def test_across_many_random_auctions(self) -> None:
        rng = np.random.default_rng(123)
        auction = GSPAuction(reserve_price=0.10, epsilon=0.01)

        for _ in range(500):
            n = rng.integers(1, 10)
            candidates = [
                _candidate(
                    f"r{i}",
                    bid=float(rng.uniform(0.5, 10.0)),
                    predicted_ctr=float(rng.uniform(0.01, 0.50)),
                    campaign_id=f"c{i}",
                )
                for i in range(n)
            ]
            winners, prices = auction.run_auction(candidates)
            for ad, price in zip(winners, prices):
                assert price <= ad["bid"] + 1e-9, (
                    f"Price {price} exceeds bid {ad['bid']} "
                    f"for restaurant_id={ad['restaurant_id']}"
                )


# ---------------------------------------------------------------------------
# Performance: 10K requests in < 30 seconds
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_10k_requests_under_30s(self) -> None:
        rng = np.random.default_rng(0)
        requests: list[list[dict]] = []
        for _ in range(10_000):
            n_candidates = int(rng.integers(3, 8))
            candidates = [
                _candidate(
                    f"r{j}",
                    bid=float(rng.uniform(0.5, 5.0)),
                    predicted_ctr=float(rng.uniform(0.01, 0.30)),
                    campaign_id=f"c{j % 20}",
                )
                for j in range(n_candidates)
            ]
            requests.append(candidates)

        sim = AuctionSimulator(seed=42)
        start = time.perf_counter()
        summary = sim.simulate(requests)
        elapsed = time.perf_counter() - start

        assert elapsed < 30.0, f"Took {elapsed:.1f}s (limit: 30s)"
        assert not summary.empty
