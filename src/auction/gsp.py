"""Generalized Second-Price (GSP) auction engine and batch simulation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GSPAuction:
    """Generalized Second-Price auction for ad ranking and pricing.

    Parameters
    ----------
    reserve_price : float
        Minimum price the last winning ad must pay.
    epsilon : float
        Small increment added to the GSP price to maintain strict ordering.
    max_slots : int or None
        Maximum ad slots awarded per auction.  ``None`` = unlimited.
    """

    def __init__(
        self,
        reserve_price: float = 0.10,
        epsilon: float = 0.01,
        max_slots: int | None = None,
    ) -> None:
        self.reserve_price = reserve_price
        self.epsilon = epsilon
        self.max_slots = max_slots

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def rank_ads(self, candidates: list[dict]) -> list[dict]:
        """Sort candidates by ``bid * predicted_ctr`` descending.

        * Ads with ``predicted_ctr <= 0`` are dropped (avoids division-by-zero
          in pricing) and a warning is logged.
        * Tied rank scores are broken by *lower* bid (rewards efficiency).
        """
        if not candidates:
            return []

        n = len(candidates)
        bids = np.empty(n, dtype=np.float64)
        ctrs = np.empty(n, dtype=np.float64)
        for i, ad in enumerate(candidates):
            bids[i] = float(ad["bid"])
            ctrs[i] = float(ad.get("predicted_ctr", 0.0))

        for i in range(n):
            if ctrs[i] <= 0:
                ad = candidates[i]
                logger.warning(
                    "Skipping ad restaurant_id=%s campaign=%s: " "predicted_ctr=%.6f <= 0",
                    ad.get("restaurant_id"),
                    ad.get("campaign_id"),
                    float(ctrs[i]),
                )

        valid_idx = np.flatnonzero(ctrs > 0)
        if valid_idx.size == 0:
            return []

        vb = bids[valid_idx]
        vc = ctrs[valid_idx]
        scores = vb * vc
        order = np.lexsort((vb, -scores))
        ranked_idx = valid_idx[order]

        return [{**candidates[int(i)], "rank_score": float(bids[i] * ctrs[i])} for i in ranked_idx]

    def compute_prices(self, ranked_ads: list[dict]) -> list[float]:
        """GSP pricing for a ranked ad list.

        Position *i* pays ``rank_score[i+1] / ctr[i] + epsilon``.
        The last winner pays ``reserve_price``.
        No winner ever pays more than its own bid.
        """
        if not ranked_ads:
            return []

        n = len(ranked_ads)
        rank_scores = np.array([float(ad["rank_score"]) for ad in ranked_ads], dtype=np.float64)
        ctrs = np.array([float(ad["predicted_ctr"]) for ad in ranked_ads], dtype=np.float64)
        own_bids = np.array([float(ad["bid"]) for ad in ranked_ads], dtype=np.float64)

        raw = np.empty(n, dtype=np.float64)
        raw[:-1] = rank_scores[1:] / ctrs[:-1] + self.epsilon
        raw[-1] = self.reserve_price
        capped = np.minimum(raw, own_bids)
        return capped.tolist()

    def apply_budget_constraints(
        self,
        ranked_ads: list[dict],
        campaign_budgets: dict[str, float],
    ) -> list[dict]:
        """Filter out ads whose campaign budget is exhausted.

        Only campaigns explicitly present in *campaign_budgets* are
        constrained; unlisted campaigns pass through freely.

        *campaign_budgets* is updated **in place**: values are coerced to
        ``max(0, value)`` so exhausted / negative balances stay well-defined.
        Impression spend is still deducted in :meth:`AuctionSimulator.simulate`.
        """
        for k in list(campaign_budgets.keys()):
            campaign_budgets[k] = max(0.0, float(campaign_budgets[k]))

        return [
            ad for ad in ranked_ads if campaign_budgets.get(ad["campaign_id"], float("inf")) > 0
        ]

    def run_auction(
        self,
        candidates: list[dict],
        campaign_budgets: dict[str, float] | None = None,
    ) -> tuple[list[dict], list[float]]:
        """Full pipeline: rank -> budget filter -> slot limit -> price.

        Returns ``(winners, prices)`` where both lists share the same
        length and ordering.
        """
        ranked = self.rank_ads(candidates)

        if campaign_budgets is not None:
            ranked = self.apply_budget_constraints(ranked, campaign_budgets)

        if self.max_slots is not None:
            ranked = ranked[: self.max_slots]

        prices = self.compute_prices(ranked)
        return ranked, prices


class AuctionSimulator:
    """Batch GSP auction runner with per-campaign metric tracking.

    Parameters
    ----------
    auction : GSPAuction or None
        Auction engine to use.  Defaults to ``GSPAuction()``.
    value_per_click : float
        Monetary value assigned to each click (for social-welfare
        computation).
    seed : int
        RNG seed — guarantees deterministic click simulation.
    """

    _SUMMARY_COLUMNS = [
        "campaign_id",
        "spend",
        "impressions",
        "clicks",
        "revenue",
        "avg_cpc",
        "social_welfare",
        "advertiser_roi",
    ]

    def __init__(
        self,
        auction: GSPAuction | None = None,
        value_per_click: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.auction = auction or GSPAuction()
        self.value_per_click = value_per_click
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        requests: list[list[dict]],
        campaign_budgets: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Run auctions for a batch of ad requests.

        Parameters
        ----------
        requests :
            Each element is a list of candidate dicts for one auction
            slot.  A candidate dict must contain ``restaurant_id``,
            ``bid``, ``predicted_ctr``, and ``campaign_id``.
        campaign_budgets :
            Optional mapping ``campaign_id -> remaining budget``.
            A *copy* is made internally so the caller's dict is unchanged.

        Returns
        -------
        pd.DataFrame
            Per-campaign summary with columns: campaign_id, spend,
            impressions, clicks, revenue, avg_cpc, social_welfare,
            advertiser_roi.
        """
        budgets = dict(campaign_budgets) if campaign_budgets is not None else None

        # Pre-draw all random values for speed and determinism.
        max_outcomes = sum(len(r) for r in requests)
        random_draws = self.rng.random(max_outcomes)
        draw_idx = 0

        records: list[dict] = []

        for candidates in requests:
            if not candidates:
                continue

            winners, prices = self.auction.run_auction(candidates, budgets)

            for winner, price in zip(winners, prices):
                ctr = winner["predicted_ctr"]
                cid = winner["campaign_id"]

                # Budget gate — deduct spend in-place
                if budgets is not None and cid in budgets:
                    if budgets[cid] < price:
                        draw_idx += 1
                        continue
                    budgets[cid] -= price

                clicked = bool(random_draws[draw_idx] < ctr)
                draw_idx += 1

                records.append(
                    {
                        "campaign_id": cid,
                        "restaurant_id": winner["restaurant_id"],
                        "bid": winner["bid"],
                        "predicted_ctr": ctr,
                        "rank_score": winner["rank_score"],
                        "price": price,
                        "clicked": clicked,
                        "click_value": self.value_per_click if clicked else 0.0,
                    }
                )

        if not records:
            return pd.DataFrame(columns=self._SUMMARY_COLUMNS)

        df = pd.DataFrame(records)

        summary = df.groupby("campaign_id", as_index=False).agg(
            spend=("price", "sum"),
            impressions=("price", "count"),
            clicks=("clicked", "sum"),
            revenue=("click_value", "sum"),
        )
        summary["avg_cpc"] = summary["spend"] / summary["clicks"].clip(lower=1)
        summary["social_welfare"] = summary["revenue"]
        summary["advertiser_roi"] = (summary["revenue"] - summary["spend"]) / summary["spend"].clip(
            lower=1e-9
        )

        return summary

    def compute_aggregate_metrics(self, summary: pd.DataFrame) -> dict[str, float]:
        """Derive platform-level metrics from a per-campaign summary."""
        if summary.empty:
            return {
                "total_revenue": 0.0,
                "avg_cpc": 0.0,
                "social_welfare": 0.0,
                "advertiser_roi": 0.0,
            }

        total_spend = float(summary["spend"].sum())
        total_clicks = float(summary["clicks"].sum())
        total_sw = float(summary["social_welfare"].sum())

        return {
            "total_revenue": total_spend,
            "avg_cpc": total_spend / max(total_clicks, 1),
            "social_welfare": total_sw,
            "advertiser_roi": (total_sw - total_spend) / max(total_spend, 1e-9),
        }
