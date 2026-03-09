"""Extended evaluation metrics: NDCG, per-segment calibration, and revenue simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


# ---------------------------------------------------------------------------
# NDCG for ranking quality
# ---------------------------------------------------------------------------


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute DCG@k from a relevance array (already sorted by model score)."""
    relevance = np.asarray(relevance, dtype=float)[:k]
    if len(relevance) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(relevance)) + 2)
    return float(np.sum(relevance / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@k.

    Items are ranked by *y_score* descending, and *y_true* provides the
    ground-truth relevance (typically binary click labels).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score)
    sorted_relevance = y_true[order]

    actual_dcg = dcg_at_k(sorted_relevance, k)
    ideal_relevance = np.sort(y_true)[::-1]
    ideal_dcg = dcg_at_k(ideal_relevance, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def grouped_ndcg(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "click",
    score_col: str = "y_prob",
    k: int = 10,
) -> dict[str, float]:
    """Compute NDCG@k per group (e.g. per query / per user session).

    Returns a dict mapping group_value -> NDCG and a special key
    ``"__mean__"`` for the macro-averaged NDCG.
    """
    results: dict[str, float] = {}
    for name, grp in df.groupby(group_col):
        if len(grp) < 2:
            continue
        results[str(name)] = ndcg_at_k(grp[label_col].to_numpy(), grp[score_col].to_numpy(), k=k)
    if results:
        results["__mean__"] = float(np.mean(list(results.values())))
    else:
        results["__mean__"] = 0.0
    return results


# ---------------------------------------------------------------------------
# Per-segment AUC
# ---------------------------------------------------------------------------


def per_segment_auc(
    df: pd.DataFrame,
    segment_col: str,
    label_col: str = "click",
    score_col: str = "y_prob",
    top_n: int = 20,
) -> dict[str, float]:
    """Compute AUC per segment (e.g. per cuisine, per city).

    For categorical columns, only the *top_n* most frequent values are used.
    For numeric columns, quartile-based buckets are created.
    """
    results: dict[str, float] = {}
    series = df[segment_col]

    if pd.api.types.is_numeric_dtype(series):
        try:
            buckets = pd.qcut(series.rank(method="first"), q=4, duplicates="drop")
        except ValueError:
            return results
        for bucket, sub in df.groupby(buckets, observed=False):
            results[str(bucket)] = _safe_auc(sub[label_col].to_numpy(), sub[score_col].to_numpy())
    else:
        top_vals = series.astype(str).value_counts().head(top_n).index
        for v in top_vals:
            sub = df[series.astype(str) == v]
            if len(sub) >= 10:
                results[str(v)] = _safe_auc(sub[label_col].to_numpy(), sub[score_col].to_numpy())
    return results


# ---------------------------------------------------------------------------
# Per-segment calibration (ECE)
# ---------------------------------------------------------------------------


@dataclass
class SegmentCalibration:
    segment: str
    ece: float
    mean_pred: float
    mean_true: float
    count: int


def per_segment_ece(
    df: pd.DataFrame,
    segment_col: str,
    label_col: str = "click",
    score_col: str = "y_prob",
    n_bins: int = 10,
    top_n: int = 20,
) -> list[SegmentCalibration]:
    """Compute ECE per segment to detect systematic miscalibration."""
    from src.models.calibration import compute_ece

    results: list[SegmentCalibration] = []
    series = df[segment_col]

    if pd.api.types.is_numeric_dtype(series):
        try:
            buckets = pd.qcut(series, q=4, duplicates="drop")
        except ValueError:
            return results
        for bucket, sub in df.groupby(buckets, observed=False):
            rel = compute_ece(sub[label_col].to_numpy(), sub[score_col].to_numpy(), n_bins=n_bins)
            results.append(
                SegmentCalibration(
                    segment=str(bucket),
                    ece=rel.ece,
                    mean_pred=float(sub[score_col].mean()),
                    mean_true=float(sub[label_col].mean()),
                    count=len(sub),
                )
            )
    else:
        top_vals = series.astype(str).value_counts().head(top_n).index
        for v in top_vals:
            sub = df[series.astype(str) == v]
            if len(sub) >= 50:
                rel = compute_ece(
                    sub[label_col].to_numpy(), sub[score_col].to_numpy(), n_bins=n_bins
                )
                results.append(
                    SegmentCalibration(
                        segment=str(v),
                        ece=rel.ece,
                        mean_pred=float(sub[score_col].mean()),
                        mean_true=float(sub[label_col].mean()),
                        count=len(sub),
                    )
                )
    return results


# ---------------------------------------------------------------------------
# Revenue simulation for bidding strategies
# ---------------------------------------------------------------------------


@dataclass
class RevenueResult:
    strategy: str
    total_revenue: float
    total_cost: float
    total_profit: float
    total_clicks: int
    total_impressions: int
    avg_cpc: float
    roi: float


def simulate_revenue(
    df: pd.DataFrame,
    predicted_ctr_col: str = "y_prob",
    bid_col: str = "bid_amount",
    click_col: str = "click",
    value_per_click: float = 5.0,
    budget: float | None = None,
) -> dict[str, RevenueResult]:
    """Simulate revenue under different bidding strategies.

    Strategies:
    - **truthful**: bid = expected_value = predicted_ctr * value_per_click
    - **conservative**: bid = 0.7 * expected_value
    - **aggressive**: bid = 1.3 * expected_value
    - **uniform**: bid = median(bid_amount), ignoring CTR prediction

    A row is "won" if the strategy's bid >= the existing *bid_amount* (proxy
    for market clearing price). Revenue is value_per_click per actual click.
    """
    frame = df.copy()
    pred_ctr = frame[predicted_ctr_col].to_numpy(dtype=float)
    market_price = frame[bid_col].to_numpy(dtype=float)
    actual_click = frame[click_col].to_numpy(dtype=int)
    ev = pred_ctr * value_per_click

    strategies: dict[str, np.ndarray] = {
        "truthful": ev,
        "conservative": 0.7 * ev,
        "aggressive": 1.3 * ev,
        "uniform": np.full_like(ev, float(np.median(market_price))),
    }

    results: dict[str, RevenueResult] = {}
    for name, bids in strategies.items():
        won = bids >= market_price
        if budget is not None:
            cumulative_cost = np.cumsum(market_price * won)
            within_budget = cumulative_cost <= budget
            won = won & within_budget

        clicks_won = int((actual_click * won).sum())
        impressions_won = int(won.sum())
        total_cost = float((market_price * won).sum())
        total_revenue = clicks_won * value_per_click
        total_profit = total_revenue - total_cost
        avg_cpc = total_cost / max(clicks_won, 1)
        roi = total_profit / max(total_cost, 1e-9)

        results[name] = RevenueResult(
            strategy=name,
            total_revenue=total_revenue,
            total_cost=total_cost,
            total_profit=total_profit,
            total_clicks=clicks_won,
            total_impressions=impressions_won,
            avg_cpc=avg_cpc,
            roi=roi,
        )
    return results


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------


def full_evaluation_report(
    df: pd.DataFrame,
    label_col: str = "click",
    score_col: str = "y_prob",
    segment_cols: list[str] | None = None,
    group_ndcg_col: str | None = None,
    ndcg_k: int = 10,
) -> dict[str, Any]:
    """Produce a comprehensive evaluation report."""
    y_true = df[label_col].to_numpy(dtype=float)
    y_prob = df[score_col].to_numpy(dtype=float)

    report: dict[str, Any] = {
        "auc": _safe_auc(y_true, y_prob),
        "logloss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))),
        "global_ndcg": ndcg_at_k(y_true, y_prob, k=ndcg_k),
    }

    if group_ndcg_col and group_ndcg_col in df.columns:
        report["grouped_ndcg"] = grouped_ndcg(df, group_ndcg_col, label_col, score_col, k=ndcg_k)

    segment_cols = segment_cols or []
    segment_auc: dict[str, dict[str, float]] = {}
    segment_calibration: dict[str, list[dict[str, Any]]] = {}
    for col in segment_cols:
        if col not in df.columns:
            continue
        segment_auc[col] = per_segment_auc(df, col, label_col, score_col)
        seg_cal = per_segment_ece(df, col, label_col, score_col)
        segment_calibration[col] = [
            {
                "segment": s.segment,
                "ece": s.ece,
                "mean_pred": s.mean_pred,
                "mean_true": s.mean_true,
                "count": s.count,
            }
            for s in seg_cal
        ]

    report["segment_auc"] = segment_auc
    report["segment_calibration"] = segment_calibration
    return report
