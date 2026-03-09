from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from src.features.engineer import FeatureEngineer


def _build_business_df() -> pd.DataFrame:
    rows = []
    # 55 Japanese restaurants -> cuisine is in-vocab, not grouped to Other.
    for i in range(55):
        rows.append(
            {
                "business_id": f"b_jp_{i}",
                "city": "Phoenix",
                "categories": "Restaurants, Sushi Bars, Japanese",
                "stars": 4.0 + (i % 3) * 0.5,
                "review_count": 20 + i,
                "attributes": {"RestaurantsPriceRange2": "2"},
            }
        )
    # One rare cuisine (Thai) and one missing price restaurant.
    rows.append(
        {
            "business_id": "b_rare",
            "city": "Phoenix",
            "categories": "Restaurants, Thai",
            "stars": 3.5,
            "review_count": 10,
            "attributes": {"RestaurantsPriceRange2": "1"},
        }
    )
    rows.append(
        {
            "business_id": "b_missing_price",
            "city": "Phoenix",
            "categories": "Restaurants, Japanese",
            "stars": 4.5,
            "review_count": 80,
            "attributes": {},
        }
    )
    return pd.DataFrame(rows)


def _build_review_df() -> pd.DataFrame:
    rows = []
    base_date = pd.Timestamp("2024-01-01 12:00:00")

    # User with >3 reviews: u1
    rows.extend(
        [
            {
                "review_id": "r1",
                "user_id": "u1",
                "business_id": "b_jp_1",
                "stars": 5.0,
                "date": base_date,
            },
            {
                "review_id": "r2",
                "user_id": "u1",
                "business_id": "b_jp_2",
                "stars": 4.0,
                "date": base_date + pd.Timedelta(days=1),
            },
            {
                "review_id": "r3",
                "user_id": "u1",
                "business_id": "b_jp_3",
                "stars": 4.0,
                "date": base_date + pd.Timedelta(days=2),
            },
        ]
    )
    # User with <3 reviews: u2
    rows.append(
        {
            "review_id": "r4",
            "user_id": "u2",
            "business_id": "b_jp_4",
            "stars": 3.0,
            "date": base_date + pd.Timedelta(days=3),
        }
    )
    return pd.DataFrame(rows)


def _build_user_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "yelping_since": "2018-01-01"},
            {"user_id": "u2", "yelping_since": "2020-01-01"},
            {"user_id": "u0", "yelping_since": "2021-01-01"},  # 0 reviews user
        ]
    )


def test_oov_cuisine_maps_to_other() -> None:
    fe = FeatureEngineer(rare_cuisine_threshold=50)
    fe.fit(_build_business_df(), _build_review_df(), _build_user_df())

    interactions = pd.DataFrame(
        [
            {
                "review_id": "rt1",
                "user_id": "u1",
                "business_id": "b_rare",  # Thai appears < 50 times, should map to Other
                "date": "2024-02-01 10:00:00",
            }
        ]
    )
    out = fe.transform(interactions)
    assert int(out.iloc[0]["primary_cuisine"]) == fe.cuisine_to_idx["Other"]


def test_missing_price_is_imputed_and_flag_set() -> None:
    fe = FeatureEngineer(rare_cuisine_threshold=50)
    fe.fit(_build_business_df(), _build_review_df(), _build_user_df())

    interactions = pd.DataFrame(
        [
            {
                "review_id": "rt2",
                "user_id": "u1",
                "business_id": "b_missing_price",
                "date": "2024-02-01 21:00:00",
            }
        ]
    )
    out = fe.transform(interactions)
    assert int(out.iloc[0]["price_missing"]) == 1
    assert pd.notna(out.iloc[0]["price_range"])


def test_user_with_zero_reviews_gets_city_default_cuisine() -> None:
    fe = FeatureEngineer(rare_cuisine_threshold=50)
    fe.fit(_build_business_df(), _build_review_df(), _build_user_df())

    # u0 has 0 reviews; should get city-level top cuisine fallback.
    interactions = pd.DataFrame(
        [
            {
                "review_id": "rt3",
                "user_id": "u0",
                "business_id": "b_jp_10",
                "date": "2024-02-03 13:30:00",
            }
        ]
    )
    out = fe.transform(interactions)
    expected = fe.city_top_cuisine["Phoenix"]
    expected_idx = fe.top_cuisine_to_idx[expected]
    assert int(out.iloc[0]["top_cuisine_preference"]) == expected_idx


def test_feature_config_vocab_sizes_and_serializable(tmp_path: Path) -> None:
    fe = FeatureEngineer(rare_cuisine_threshold=50)
    fe.fit(_build_business_df(), _build_review_df(), _build_user_df())

    cfg = fe.feature_config
    assert cfg["primary_cuisine"]["vocab_size"] == len(fe.cuisine_to_idx)
    assert cfg["city_encoded"]["vocab_size"] == len(fe.city_to_idx)
    assert cfg["top_cuisine_preference"]["vocab_size"] == len(fe.top_cuisine_to_idx)
    assert cfg["time_of_day"]["vocab_size"] == len(fe.time_of_day_to_idx)
    assert cfg["day_of_week"]["vocab_size"] == len(fe.day_of_week_to_idx)

    # JSON-safe requirement.
    json.dumps(cfg)

    # Pickle size requirement.
    model_path = tmp_path / "feature_engineer.pkl"
    fe.save(model_path)
    assert model_path.stat().st_size < 5 * 1024 * 1024
