"""Auction-aware bid optimization with exponential-smoothing budget pacing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.auction.gsp import GSPAuction

logger = logging.getLogger(__name__)


class BudgetPacer:
    """Exponential-smoothing budget pacer.

    Distributes a daily budget across time slots and adjusts a
    *pacing_multiplier* so cumulative spend tracks the ideal linear
    spend curve.

    Parameters
    ----------
    daily_budget : float
        Total budget for the day.
    n_slots : int
        Number of time slots the budget is spread across.
    alpha : float
        Smoothing factor for exponential-smoothing updates.
    """

    MIN_MULTIPLIER = 0.1
    MAX_MULTIPLIER = 3.0

    def __init__(
        self,
        daily_budget: float,
        n_slots: int,
        alpha: float = 0.3,
    ) -> None:
        self.daily_budget = daily_budget
        self.n_slots = max(n_slots, 1)
        self.alpha = alpha
        self.pacing_multiplier: float = 1.0
        self._total_spent: float = 0.0

    @property
    def remaining_budget(self) -> float:
        return max(self.daily_budget - self._total_spent, 0.0)

    def record_spend(self, amount: float) -> None:
        self._total_spent += amount

    def update(self, actual_spend: float, time_elapsed: float) -> float:
        """Adjust *pacing_multiplier* via exponential smoothing.

        Uses a forward-looking formulation: compares the *required*
        spend rate (remaining budget / remaining time) against the
        ideal rate, so the pacer naturally accelerates when behind
        schedule and brakes when ahead.

        Parameters
        ----------
        actual_spend : float
            Cumulative spend observed so far.
        time_elapsed : float
            Fraction of total time elapsed, in ``(0, 1]``.

        Returns
        -------
        float
            Updated pacing multiplier.
        """
        if time_elapsed <= 0 or self.daily_budget < 1e-9:
            return self.pacing_multiplier

        remaining_budget = max(self.daily_budget - actual_spend, 0.0)
        remaining_time = max(1.0 - time_elapsed, 1e-6)

        required_rate = remaining_budget / remaining_time
        ideal_rate = self.daily_budget  # spend-per-unit-time at uniform pace

        rate_ratio = required_rate / ideal_rate
        target = self.pacing_multiplier * rate_ratio

        self.pacing_multiplier = (
            self.alpha * target + (1 - self.alpha) * self.pacing_multiplier
        )
        self.pacing_multiplier = float(
            np.clip(self.pacing_multiplier, self.MIN_MULTIPLIER, self.MAX_MULTIPLIER)
        )
        return self.pacing_multiplier


class OptimalBidder:
    """Pacing-aware optimal bidder.

    ``bid = value_per_click * predicted_ctr * pacing_multiplier``,
    floored at *min_bid* and capped at *max_bid*.
    """

    def __init__(
        self,
        min_bid: float = 0.01,
        max_bid: float = 10.0,
    ) -> None:
        self.min_bid = min_bid
        self.max_bid = max_bid

    def compute_bid(
        self,
        value_per_click: float,
        predicted_ctr: float,
        pacing_multiplier: float = 1.0,
    ) -> float:
        raw = value_per_click * predicted_ctr * pacing_multiplier
        return float(np.clip(raw, self.min_bid, self.max_bid))


class CampaignSimulator:
    """Simulate a single campaign competing in GSP auctions over a day.

    Each *round* represents a time slot (e.g. one hour of a 24-hour
    day).  Within each round the campaign participates in
    *impressions_per_round* independent auctions.  After every round
    the :class:`BudgetPacer` re-calibrates the pacing multiplier.

    Parameters
    ----------
    daily_budget : float
        Total budget for the simulated day.
    value_per_click : float
        Advertiser's value per click (also the target CPA).
    n_rounds : int
        Number of time-slot rounds.
    impressions_per_round : int
        Auction opportunities per round.
    n_competitors : int
        Number of competing advertisers in each auction.
    max_slots : int
        Winning ad slots per auction.
    our_ctr_mean / our_ctr_std : float
        Distribution of our restaurant's predicted CTR.
    seed : int
        RNG seed for full reproducibility.
    """

    def __init__(
        self,
        daily_budget: float = 100.0,
        value_per_click: float = 5.0,
        n_rounds: int = 24,
        impressions_per_round: int = 50,
        n_competitors: int = 4,
        max_slots: int = 1,
        our_ctr_mean: float = 0.08,
        our_ctr_std: float = 0.005,
        seed: int = 42,
    ) -> None:
        self.daily_budget = daily_budget
        self.value_per_click = value_per_click
        self.n_rounds = n_rounds
        self.impressions_per_round = impressions_per_round
        self.n_competitors = n_competitors
        self.max_slots = max_slots
        self.our_ctr_mean = our_ctr_mean
        self.our_ctr_std = our_ctr_std
        self.seed = seed

    @property
    def target_cpa(self) -> float:
        """Target cost-per-click equals the advertiser's value per click."""
        return self.value_per_click

    def simulate(self) -> pd.DataFrame:
        """Run the full campaign simulation.

        Returns
        -------
        pd.DataFrame
            One row per round with columns: round, round_spend,
            round_clicks, round_impressions, round_auctions,
            total_spend, total_clicks, total_impressions,
            budget_utilization, actual_CPA, impression_share,
            pacing_multiplier.
        """
        rng = np.random.default_rng(self.seed)
        # Price all candidates (no truncation) so GSP second-price
        # logic applies correctly; only the top *max_slots* positions
        # are considered winning slots.
        auction = GSPAuction(
            reserve_price=0.10,
            epsilon=0.01,
            max_slots=None,
        )
        pacer = BudgetPacer(self.daily_budget, self.n_rounds)
        bidder = OptimalBidder()

        total_auctions = 0
        total_spend = 0.0
        total_clicks = 0
        total_impressions = 0

        records: list[dict] = []

        for round_idx in range(self.n_rounds):
            round_spend = 0.0
            round_clicks = 0
            round_impressions = 0

            for _ in range(self.impressions_per_round):
                total_auctions += 1

                if pacer.remaining_budget <= 0:
                    continue

                our_ctr = float(
                    np.clip(
                        rng.normal(self.our_ctr_mean, self.our_ctr_std),
                        0.005,
                        0.50,
                    )
                )
                our_bid = bidder.compute_bid(
                    self.value_per_click, our_ctr, pacer.pacing_multiplier
                )

                candidates: list[dict] = [
                    {
                        "restaurant_id": "our_restaurant",
                        "bid": our_bid,
                        "predicted_ctr": our_ctr,
                        "campaign_id": "our_campaign",
                    }
                ]

                for j in range(self.n_competitors):
                    comp_vpc = rng.uniform(4.0, 6.0)
                    comp_ctr = float(
                        np.clip(rng.normal(0.075, 0.02), 0.005, 0.50)
                    )
                    comp_bid = float(np.clip(comp_vpc * comp_ctr, 0.05, 8.0))
                    candidates.append(
                        {
                            "restaurant_id": f"comp_{j}",
                            "bid": comp_bid,
                            "predicted_ctr": comp_ctr,
                            "campaign_id": f"comp_{j}",
                        }
                    )

                winners, prices = auction.run_auction(candidates)

                # Only top max_slots positions are actual ad placements
                for i in range(min(self.max_slots, len(winners))):
                    if winners[i]["campaign_id"] == "our_campaign":
                        p = prices[i]
                        if p <= pacer.remaining_budget:
                            pacer.record_spend(p)
                            round_spend += p
                            round_impressions += 1
                            if rng.random() < our_ctr:
                                round_clicks += 1
                        break

            total_spend += round_spend
            total_clicks += round_clicks
            total_impressions += round_impressions

            time_elapsed = (round_idx + 1) / self.n_rounds
            pacer.update(total_spend, time_elapsed)

            records.append(
                {
                    "round": round_idx,
                    "round_spend": round_spend,
                    "round_clicks": round_clicks,
                    "round_impressions": round_impressions,
                    "round_auctions": self.impressions_per_round,
                    "total_spend": total_spend,
                    "total_clicks": total_clicks,
                    "total_impressions": total_impressions,
                    "budget_utilization": total_spend / self.daily_budget,
                    "actual_CPA": total_spend / max(total_clicks, 1),
                    "impression_share": total_impressions / max(total_auctions, 1),
                    "pacing_multiplier": pacer.pacing_multiplier,
                }
            )

        return pd.DataFrame(records)
