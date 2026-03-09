"""CTR-based bid optimization with budget pacing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BidResult:
    """Result for a single impression bid decision."""

    impression_id: str
    predicted_ctr: float
    expected_value: float
    bid: float
    won: bool
    cost: float
    clicked: bool
    profit: float


class BudgetPacer:
    """PID-controller based budget pacing.

    Tracks spend over time and adjusts a *pacing_multiplier* that scales
    bid amounts so the campaign spends its budget evenly across a time window.
    """

    def __init__(
        self,
        total_budget: float,
        n_periods: int,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
    ) -> None:
        self.total_budget = total_budget
        self.n_periods = n_periods
        self.target_spend_per_period = total_budget / max(n_periods, 1)

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.pacing_multiplier: float = 1.0
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._period_spend: float = 0.0
        self._total_spent: float = 0.0
        self._period: int = 0

    @property
    def remaining_budget(self) -> float:
        return max(self.total_budget - self._total_spent, 0.0)

    def record_spend(self, amount: float) -> None:
        self._period_spend += amount
        self._total_spent += amount

    def advance_period(self) -> float:
        """Call at the end of each time period to update the pacing multiplier."""
        error = self.target_spend_per_period - self._period_spend
        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error

        adjustment = self.kp * error + self.ki * self._integral + self.kd * derivative
        scale = adjustment / max(self.target_spend_per_period, 1e-9)
        self.pacing_multiplier = float(np.clip(self.pacing_multiplier + scale, 0.1, 3.0))

        self._period_spend = 0.0
        self._period += 1
        return self.pacing_multiplier


class BidOptimizer:
    """Computes optimal bids from calibrated CTR predictions.

    Supports three auction models:
    - **second_price**: bid truthfully at expected value (dominant strategy).
    - **first_price**: shade bid below expected value by *shade_factor*.
    - **vcg**: bid truthfully (strategy-proof under VCG).
    """

    def __init__(
        self,
        value_per_click: float = 5.0,
        auction_type: str = "second_price",
        shade_factor: float = 0.7,
        min_bid: float = 0.01,
        max_bid: float = 50.0,
    ) -> None:
        if auction_type not in ("second_price", "first_price", "vcg"):
            raise ValueError(f"Unknown auction_type: {auction_type}")
        self.value_per_click = value_per_click
        self.auction_type = auction_type
        self.shade_factor = shade_factor
        self.min_bid = min_bid
        self.max_bid = max_bid

    def compute_bid(self, predicted_ctr: float) -> float:
        """Compute the optimal bid for a single impression."""
        ev = predicted_ctr * self.value_per_click

        if self.auction_type == "first_price":
            bid = ev * self.shade_factor
        else:
            bid = ev

        return float(np.clip(bid, self.min_bid, self.max_bid))

    def compute_bids(
        self,
        predicted_ctrs: np.ndarray,
        pacer: BudgetPacer | None = None,
    ) -> np.ndarray:
        """Compute bids for an array of impressions."""
        ctrs = np.asarray(predicted_ctrs, dtype=float)
        evs = ctrs * self.value_per_click

        if self.auction_type == "first_price":
            bids = evs * self.shade_factor
        else:
            bids = evs.copy()

        if pacer is not None:
            bids *= pacer.pacing_multiplier
            mask = np.cumsum(bids) > pacer.remaining_budget
            bids[mask] = 0.0

        return np.clip(bids, self.min_bid, self.max_bid)

    def simulate_auction(
        self,
        df: pd.DataFrame,
        predicted_ctr_col: str = "y_prob",
        market_price_col: str = "bid_amount",
        click_col: str = "click",
        impression_id_col: str = "impression_id",
        budget: float | None = None,
        n_periods: int = 10,
    ) -> tuple[list[BidResult], dict[str, float]]:
        """Run a full auction simulation on a dataset.

        Returns per-impression BidResults and aggregate summary metrics.
        """
        ctrs = df[predicted_ctr_col].to_numpy(dtype=float)
        market_prices = df[market_price_col].to_numpy(dtype=float)
        clicks = df[click_col].to_numpy(dtype=int)
        imp_ids = df[impression_id_col].astype(str).to_numpy()

        bids = self.compute_bids(ctrs)

        pacer = None
        if budget is not None:
            pacer = BudgetPacer(budget, n_periods)
            period_size = max(len(df) // n_periods, 1)

        results: list[BidResult] = []
        total_cost = 0.0
        total_revenue = 0.0
        total_clicks = 0
        total_won = 0

        for i in range(len(df)):
            if pacer is not None and i > 0 and i % period_size == 0:
                pacer.advance_period()
                remaining_bids = self.compute_bids(ctrs[i:], pacer)
                bids[i:] = remaining_bids

            bid = float(bids[i])
            mp = float(market_prices[i])
            won = bid >= mp

            if won and pacer is not None and pacer.remaining_budget < mp:
                won = False

            cost = mp if won else 0.0
            clicked = bool(clicks[i]) and won
            revenue = self.value_per_click if clicked else 0.0
            profit = revenue - cost

            if won and pacer is not None:
                pacer.record_spend(cost)

            total_cost += cost
            total_revenue += revenue
            total_clicks += int(clicked)
            total_won += int(won)

            results.append(
                BidResult(
                    impression_id=str(imp_ids[i]),
                    predicted_ctr=float(ctrs[i]),
                    expected_value=float(ctrs[i] * self.value_per_click),
                    bid=bid,
                    won=won,
                    cost=cost,
                    clicked=clicked,
                    profit=profit,
                )
            )

        total_profit = total_revenue - total_cost
        summary = {
            "total_impressions": len(df),
            "won_impressions": total_won,
            "win_rate": total_won / max(len(df), 1),
            "total_clicks": total_clicks,
            "total_cost": total_cost,
            "total_revenue": total_revenue,
            "total_profit": total_profit,
            "avg_cpc": total_cost / max(total_clicks, 1),
            "roi": total_profit / max(total_cost, 1e-9),
            "budget_utilization": total_cost / budget if budget else float("nan"),
        }
        return results, summary
