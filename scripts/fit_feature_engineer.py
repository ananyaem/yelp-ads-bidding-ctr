"""Fit FeatureEngineer on processed Yelp parquets and save models/feature_engineer.pkl.

``FeatureEngineer.fit`` scales the StandardScaler using ``transform()`` over every
review row passed in; that path is row-Python-heavy. Use ``--max-reviews`` (default
500k) for a reproducible subsample, or ``--max-reviews 0`` for all rows (slow).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import (
    BUSINESS_PARQUET_PATH,
    MODELS_DIR,
    REVIEW_PARQUET_PATH,
    USER_PARQUET_PATH,
)
from src.features.engineer import FeatureEngineer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit FeatureEngineer and save pickle.")
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=500_000,
        help="Cap reviews passed to fit (random sample). Use 0 for no cap (all rows; slow).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for subsampling.")
    parser.add_argument(
        "--rare-cuisine-threshold",
        type=int,
        default=50,
        help="Minimum business count per cuisine before bucketing to Other.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "feature_engineer.pkl",
        help="Output pickle path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    t0 = time.perf_counter()

    for path in (BUSINESS_PARQUET_PATH, REVIEW_PARQUET_PATH, USER_PARQUET_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing parquet: {path}")

    business_df = pd.read_parquet(BUSINESS_PARQUET_PATH)
    review_df = pd.read_parquet(REVIEW_PARQUET_PATH)
    user_df = pd.read_parquet(USER_PARQUET_PATH)

    n_rev = len(review_df)
    if args.max_reviews > 0 and n_rev > args.max_reviews:
        review_df = review_df.sample(n=args.max_reviews, random_state=args.seed).reset_index(
            drop=True
        )
        logging.info("Subsampled reviews: %s -> %s", n_rev, len(review_df))
    else:
        logging.info("Using all %s reviews for scaler fit inside FeatureEngineer.fit", n_rev)

    logging.info(
        "Fitting on business=%s review=%s user=%s rows",
        len(business_df),
        len(review_df),
        len(user_df),
    )
    fe = FeatureEngineer(rare_cuisine_threshold=args.rare_cuisine_threshold)
    fe.fit(business_df, review_df, user_df)
    out = fe.save(args.output)
    elapsed = time.perf_counter() - t0
    logging.info("Saved %s (%.1fs)", out, elapsed)
    logging.info(
        "Vocab sizes: cuisines=%s cities=%s top_cuisine=%s",
        len(fe.cuisine_to_idx),
        len(fe.city_to_idx),
        len(fe.top_cuisine_to_idx),
    )


if __name__ == "__main__":
    main()
