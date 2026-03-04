"""Feature engineering pipeline for Yelp CTR modeling."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, model_validator
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_HPARAMS


def _safe_log1p(value: Any) -> float:
    try:
        return float(np.log1p(max(float(value), 0.0)))
    except Exception:
        return 0.0


def _parse_categories(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    items = [x.strip() for x in str(raw).split(",") if x and x.strip()]
    # Remove very broad bucket for stronger cuisine signal.
    return [x for x in items if x.lower() != "restaurants"]


def _pick_primary_cuisine(categories: list[str]) -> str:
    if not categories:
        return "Other"
    # Heuristic for "most specific": longest non-generic token.
    return sorted(categories, key=lambda x: (len(x), x.lower()), reverse=True)[0]


def _extract_price_range(attributes: Any) -> float | None:
    if attributes is None or (isinstance(attributes, float) and np.isnan(attributes)):
        return None
    if not isinstance(attributes, dict):
        return None
    raw = attributes.get("RestaurantsPriceRange2")
    if raw is None:
        return None
    raw_str = str(raw).strip().strip("'").strip('"')
    if raw_str == "":
        return None
    try:
        return float(int(float(raw_str)))
    except Exception:
        return None


def _bucket_time_of_day(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if 5 <= hour <= 10:
        return "morning"
    if 11 <= hour <= 14:
        return "lunch"
    if 15 <= hour <= 19:
        return "evening"
    if 20 <= hour <= 23:
        return "night"
    return "late_night"


@dataclass
class _SchemaSpec:
    required_columns: list[str]
    required_dtypes: dict[str, tuple[str, ...]]
    no_nan_columns: list[str]


class _DataFrameSchemaValidator(BaseModel):
    """Pydantic wrapper for DataFrame schema checks."""

    frame_name: str
    columns: list[str]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    spec: _SchemaSpec

    @model_validator(mode="after")
    def _validate(self) -> "_DataFrameSchemaValidator":
        missing = [c for c in self.spec.required_columns if c not in self.columns]
        if missing:
            raise ValueError(f"{self.frame_name}: missing required columns: {missing}")

        dtype_errors: list[str] = []
        for col, allowed in self.spec.required_dtypes.items():
            if col in self.dtypes:
                actual = self.dtypes[col]
                if not any(token in actual for token in allowed):
                    dtype_errors.append(f"{col} (expected one of {allowed}, got {actual})")
        if dtype_errors:
            raise ValueError(f"{self.frame_name}: wrong dtypes: {dtype_errors}")

        nan_errors = [col for col in self.spec.no_nan_columns if self.null_counts.get(col, 0) > 0]
        if nan_errors:
            details = {k: self.null_counts.get(k, 0) for k in nan_errors}
            raise ValueError(f"{self.frame_name}: NaN found in required fields: {details}")

        return self


class FeatureEngineer:
    """Builds restaurant/user/context features and model-ready metadata."""

    def __init__(self, rare_cuisine_threshold: int = 50, embedding_dim: int = DEFAULT_HPARAMS.embedding_dim) -> None:
        self.rare_cuisine_threshold = rare_cuisine_threshold
        self.embedding_dim = embedding_dim

        self.is_fitted = False
        self.scaler = StandardScaler()

        self.business_feature_map: dict[str, dict[str, Any]] = {}
        self.user_feature_map: dict[str, dict[str, Any]] = {}
        self.city_top_cuisine: dict[str, str] = {}

        self.cuisine_to_idx: dict[str, int] = {}
        self.city_to_idx: dict[str, int] = {}
        self.time_of_day_to_idx: dict[str, int] = {}
        self.day_of_week_to_idx: dict[str, int] = {}
        self.top_cuisine_to_idx: dict[str, int] = {}

        self.feature_config: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _validate_df(df: pd.DataFrame, frame_name: str, spec: _SchemaSpec) -> None:
        payload = {
            "frame_name": frame_name,
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "null_counts": {c: int(v) for c, v in df.isna().sum().to_dict().items()},
            "spec": spec,
        }
        try:
            _DataFrameSchemaValidator.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"{frame_name} schema validation failed: {exc}") from exc

    def fit(self, business_df: pd.DataFrame, review_df: pd.DataFrame, user_df: pd.DataFrame) -> "FeatureEngineer":
        """Fit feature mappings and scaler using train split only."""
        business_spec = _SchemaSpec(
            required_columns=["business_id", "city", "categories", "stars", "review_count"],
            required_dtypes={
                "business_id": ("object", "string"),
                "city": ("object", "string"),
                "categories": ("object", "string"),
                "stars": ("float", "int"),
                "review_count": ("float", "int"),
            },
            no_nan_columns=["business_id", "city", "stars", "review_count"],
        )
        review_spec = _SchemaSpec(
            required_columns=["review_id", "user_id", "business_id", "stars", "date"],
            required_dtypes={
                "review_id": ("object", "string"),
                "user_id": ("object", "string"),
                "business_id": ("object", "string"),
                "stars": ("float", "int"),
                "date": ("datetime", "object", "string"),
            },
            no_nan_columns=["review_id", "user_id", "business_id", "stars", "date"],
        )
        user_spec = _SchemaSpec(
            required_columns=["user_id", "yelping_since"],
            required_dtypes={
                "user_id": ("object", "string"),
                "yelping_since": ("datetime", "object", "string"),
            },
            no_nan_columns=["user_id", "yelping_since"],
        )
        self._validate_df(business_df, "business_df", business_spec)
        self._validate_df(review_df, "review_df", review_spec)
        self._validate_df(user_df, "user_df", user_spec)

        biz = business_df.copy()
        rev = review_df.copy()
        usr = user_df.copy()

        biz["business_id"] = biz["business_id"].astype(str)
        biz["city"] = biz["city"].astype(str).str.strip().replace("", "Unknown")
        biz["parsed_categories"] = biz["categories"].apply(_parse_categories)
        biz["primary_cuisine_raw"] = biz["parsed_categories"].apply(_pick_primary_cuisine)

        cuisine_counts = biz["primary_cuisine_raw"].value_counts()
        rare_cuisines = set(cuisine_counts[cuisine_counts < self.rare_cuisine_threshold].index)
        biz["primary_cuisine"] = biz["primary_cuisine_raw"].apply(lambda x: "Other" if x in rare_cuisines else x)

        city_avg_stars = biz.groupby("city")["stars"].mean().to_dict()
        biz["rating_vs_city_avg"] = biz["stars"] - biz["city"].map(city_avg_stars).fillna(biz["stars"].mean())
        biz["log_review_count"] = biz["review_count"].apply(_safe_log1p)

        biz["price_range_raw"] = biz.get("attributes", pd.Series([None] * len(biz))).apply(_extract_price_range)
        biz["price_missing"] = biz["price_range_raw"].isna().astype(int)

        city_cuisine_price_med = (
            biz.groupby(["city", "primary_cuisine"])["price_range_raw"].median().to_dict()
        )
        city_price_med = biz.groupby("city")["price_range_raw"].median().to_dict()
        cuisine_price_med = biz.groupby("primary_cuisine")["price_range_raw"].median().to_dict()
        global_price_med = float(biz["price_range_raw"].median()) if not math.isnan(float(biz["price_range_raw"].median())) else 2.0

        def _impute_price(row: pd.Series) -> float:
            val = row["price_range_raw"]
            if pd.notna(val):
                return float(val)
            city = row["city"]
            cuisine = row["primary_cuisine"]
            return float(
                city_cuisine_price_med.get((city, cuisine))
                or city_price_med.get(city)
                or cuisine_price_med.get(cuisine)
                or global_price_med
            )

        biz["price_range"] = biz.apply(_impute_price, axis=1)

        # City -> dominant cuisine mapping (for user default).
        city_top = (
            biz.groupby(["city", "primary_cuisine"])["business_id"]
            .nunique()
            .reset_index(name="cnt")
            .sort_values(["city", "cnt"], ascending=[True, False])
            .drop_duplicates(subset=["city"], keep="first")
        )
        self.city_top_cuisine = dict(zip(city_top["city"], city_top["primary_cuisine"]))

        # User aggregates.
        rev["user_id"] = rev["user_id"].astype(str)
        rev["business_id"] = rev["business_id"].astype(str)
        rev["date"] = pd.to_datetime(rev["date"], errors="coerce")
        rev = rev.dropna(subset=["date"])

        user_agg = (
            rev.groupby("user_id")
            .agg(avg_rating_given=("stars", "mean"), review_count=("review_id", "count"))
            .reset_index()
        )
        user_agg["log_review_count"] = user_agg["review_count"].apply(_safe_log1p)

        usr["user_id"] = usr["user_id"].astype(str)
        usr["yelping_since"] = pd.to_datetime(usr["yelping_since"], errors="coerce")
        if usr["yelping_since"].isna().any():
            bad = int(usr["yelping_since"].isna().sum())
            raise ValueError(f"user_df has invalid yelping_since values: {bad}")
        reference_ts = rev["date"].max() if not rev.empty else pd.Timestamp.utcnow()
        usr["account_age_years"] = (reference_ts - usr["yelping_since"]).dt.days / 365.25

        user_features = usr[["user_id", "account_age_years"]].merge(user_agg, on="user_id", how="left")
        user_features["review_count"] = user_features["review_count"].fillna(0).astype(int)
        user_features["avg_rating_given"] = user_features["avg_rating_given"].fillna(float(rev["stars"].mean()) if not rev.empty else 3.5)
        user_features["log_review_count"] = user_features["log_review_count"].fillna(0.0)

        # Top cuisine preference from reviewed restaurants.
        review_with_biz = rev.merge(
            biz[["business_id", "primary_cuisine", "city"]],
            on="business_id",
            how="left",
        )
        user_cuisine_pref = (
            review_with_biz.groupby(["user_id", "primary_cuisine"])["review_id"]
            .count()
            .reset_index(name="cnt")
            .sort_values(["user_id", "cnt"], ascending=[True, False])
            .drop_duplicates(subset=["user_id"], keep="first")
            .rename(columns={"primary_cuisine": "top_cuisine_preference"})
        )
        user_features = user_features.merge(user_cuisine_pref[["user_id", "top_cuisine_preference"]], on="user_id", how="left")

        # Build per-business map.
        for row in biz.itertuples(index=False):
            self.business_feature_map[str(row.business_id)] = {
                "city": str(row.city),
                "primary_cuisine": str(row.primary_cuisine),
                "rating_vs_city_avg": float(row.rating_vs_city_avg),
                "log_review_count": float(row.log_review_count),
                "price_range": float(row.price_range),
                "price_missing": int(row.price_missing),
            }

        # Build per-user map.
        for row in user_features.itertuples(index=False):
            self.user_feature_map[str(row.user_id)] = {
                "avg_rating_given": float(row.avg_rating_given),
                "log_review_count": float(row.log_review_count),
                "account_age_years": float(row.account_age_years),
                "review_count": int(row.review_count),
                "top_cuisine_preference": (
                    str(row.top_cuisine_preference) if pd.notna(row.top_cuisine_preference) else None
                ),
            }

        # Vocabularies for sparse features.
        cuisines = sorted(set(biz["primary_cuisine"].astype(str).tolist()) | {"Other"})
        cities = sorted(set(biz["city"].astype(str).tolist()) | {"Unknown"})
        top_cuisine_vocab = sorted(set(cuisines) | {"Other"})
        self.cuisine_to_idx = {v: i for i, v in enumerate(cuisines)}
        self.city_to_idx = {v: i for i, v in enumerate(cities)}
        self.top_cuisine_to_idx = {v: i for i, v in enumerate(top_cuisine_vocab)}
        self.time_of_day_to_idx = {v: i for i, v in enumerate(["morning", "lunch", "evening", "night", "late_night"])}
        self.day_of_week_to_idx = {v: i for i, v in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}

        # Fit scaler using train interactions.
        train_features = self.transform(rev[["review_id", "user_id", "business_id", "date"]], fit_mode=True)
        dense_cols = [
            "rating_vs_city_avg",
            "restaurant_log_review_count",
            "price_range",
            "avg_rating_given",
            "user_log_review_count",
            "account_age_years",
        ]
        self.scaler.fit(train_features[dense_cols].astype(float))

        self.feature_config = self._build_feature_config()
        # Must be JSON-safe per acceptance criteria.
        json.dumps(self.feature_config)

        self.is_fitted = True
        return self

    def transform(self, interactions_df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """Transform interactions into model-ready features."""
        if not fit_mode and not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform().")

        spec = _SchemaSpec(
            required_columns=["user_id", "business_id", "date"],
            required_dtypes={"user_id": ("object", "string"), "business_id": ("object", "string"), "date": ("datetime", "object", "string")},
            no_nan_columns=["user_id", "business_id", "date"],
        )
        self._validate_df(interactions_df, "interactions_df", spec)

        frame = interactions_df.copy()
        frame["user_id"] = frame["user_id"].astype(str)
        frame["business_id"] = frame["business_id"].astype(str)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            bad = int(frame["date"].isna().sum())
            raise ValueError(f"interactions_df has invalid date values: {bad}")

        rows: list[dict[str, Any]] = []
        global_city = "Unknown"
        global_cuisine = "Other"
        for row in frame.itertuples(index=False):
            biz_feat = self.business_feature_map.get(row.business_id)
            if biz_feat is None:
                biz_feat = {
                    "city": global_city,
                    "primary_cuisine": global_cuisine,
                    "rating_vs_city_avg": 0.0,
                    "log_review_count": 0.0,
                    "price_range": 2.0,
                    "price_missing": 1,
                }
            user_feat = self.user_feature_map.get(row.user_id)
            if user_feat is None:
                user_feat = {
                    "avg_rating_given": 3.5,
                    "log_review_count": 0.0,
                    "account_age_years": 0.0,
                    "review_count": 0,
                    "top_cuisine_preference": None,
                }

            city = str(biz_feat["city"])
            primary_cuisine = str(biz_feat["primary_cuisine"])
            default_city_cuisine = self.city_top_cuisine.get(city, "Other")
            top_cuisine_pref = (
                user_feat["top_cuisine_preference"]
                if user_feat["review_count"] >= 3 and user_feat["top_cuisine_preference"]
                else default_city_cuisine
            )
            if top_cuisine_pref not in self.top_cuisine_to_idx:
                top_cuisine_pref = "Other"

            time_bucket = _bucket_time_of_day(row.date)
            day_name = row.date.day_name()
            is_weekend = 1 if day_name in {"Saturday", "Sunday"} else 0

            rows.append(
                {
                    "user_id": row.user_id,
                    "business_id": row.business_id,
                    "date": row.date,
                    "primary_cuisine": self.cuisine_to_idx.get(primary_cuisine, self.cuisine_to_idx.get("Other", 0)),
                    "city_encoded": self.city_to_idx.get(city, self.city_to_idx.get("Unknown", 0)),
                    "top_cuisine_preference": self.top_cuisine_to_idx.get(top_cuisine_pref, self.top_cuisine_to_idx.get("Other", 0)),
                    "time_of_day": self.time_of_day_to_idx.get(time_bucket, self.time_of_day_to_idx["late_night"]),
                    "day_of_week": self.day_of_week_to_idx.get(day_name, 0),
                    "is_weekend": is_weekend,
                    "rating_vs_city_avg": float(biz_feat["rating_vs_city_avg"]),
                    "restaurant_log_review_count": float(biz_feat["log_review_count"]),
                    "price_range": float(biz_feat["price_range"]),
                    "price_missing": int(biz_feat["price_missing"]),
                    "avg_rating_given": float(user_feat["avg_rating_given"]),
                    "user_log_review_count": float(user_feat["log_review_count"]),
                    "account_age_years": float(user_feat["account_age_years"]),
                }
            )

        out = pd.DataFrame(rows)

        dense_cols = [
            "rating_vs_city_avg",
            "restaurant_log_review_count",
            "price_range",
            "avg_rating_given",
            "user_log_review_count",
            "account_age_years",
        ]
        if not fit_mode:
            out[dense_cols] = self.scaler.transform(out[dense_cols].astype(float))
        return out

    def _build_feature_config(self) -> dict[str, dict[str, Any]]:
        config = {
            "primary_cuisine": {
                "name": "primary_cuisine",
                "type": "sparse",
                "vocab_size": int(len(self.cuisine_to_idx)),
                "embedding_dim": int(self.embedding_dim),
            },
            "city_encoded": {
                "name": "city_encoded",
                "type": "sparse",
                "vocab_size": int(len(self.city_to_idx)),
                "embedding_dim": int(self.embedding_dim),
            },
            "top_cuisine_preference": {
                "name": "top_cuisine_preference",
                "type": "sparse",
                "vocab_size": int(len(self.top_cuisine_to_idx)),
                "embedding_dim": int(self.embedding_dim),
            },
            "time_of_day": {
                "name": "time_of_day",
                "type": "sparse",
                "vocab_size": int(len(self.time_of_day_to_idx)),
                "embedding_dim": int(self.embedding_dim),
            },
            "day_of_week": {
                "name": "day_of_week",
                "type": "sparse",
                "vocab_size": int(len(self.day_of_week_to_idx)),
                "embedding_dim": int(self.embedding_dim),
            },
            "rating_vs_city_avg": {"name": "rating_vs_city_avg", "type": "dense", "vocab_size": None, "embedding_dim": 0},
            "restaurant_log_review_count": {"name": "restaurant_log_review_count", "type": "dense", "vocab_size": None, "embedding_dim": 0},
            "price_range": {"name": "price_range", "type": "dense", "vocab_size": None, "embedding_dim": 0},
            "price_missing": {"name": "price_missing", "type": "dense", "vocab_size": None, "embedding_dim": 0},
            "avg_rating_given": {"name": "avg_rating_given", "type": "dense", "vocab_size": None, "embedding_dim": 0},
            "user_log_review_count": {"name": "user_log_review_count", "type": "dense", "vocab_size": None, "embedding_dim": 0},
            "account_age_years": {"name": "account_age_years", "type": "dense", "vocab_size": None, "embedding_dim": 0},
            "is_weekend": {"name": "is_weekend", "type": "dense", "vocab_size": None, "embedding_dim": 0},
        }
        return config

    def enrich_ad_impressions(self, ad_df: pd.DataFrame) -> pd.DataFrame:
        """Merge FeatureEngineer's business/user features into ad impression data.

        Bridges the review-based pipeline with the ad pipeline by looking up
        precomputed features for each (business_id, user_id) pair. Features
        already present in *ad_df* are preserved; only missing ones are added.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before enrich_ad_impressions().")

        out = ad_df.copy()
        out["business_id"] = out["business_id"].astype(str)
        out["user_id"] = out["user_id"].astype(str)

        biz_rows: list[dict[str, Any]] = []
        for bid in out["business_id"]:
            feat = self.business_feature_map.get(bid, {})
            biz_rows.append({
                "fe_rating_vs_city_avg": feat.get("rating_vs_city_avg", 0.0),
                "fe_restaurant_log_review_count": feat.get("log_review_count", 0.0),
                "fe_price_range": feat.get("price_range", 2.0),
                "fe_price_missing": feat.get("price_missing", 1),
                "fe_primary_cuisine_idx": self.cuisine_to_idx.get(
                    feat.get("primary_cuisine", "Other"),
                    self.cuisine_to_idx.get("Other", 0),
                ),
                "fe_city_idx": self.city_to_idx.get(
                    feat.get("city", "Unknown"),
                    self.city_to_idx.get("Unknown", 0),
                ),
            })
        biz_feat_df = pd.DataFrame(biz_rows, index=out.index)

        user_rows: list[dict[str, Any]] = []
        for uid in out["user_id"]:
            feat = self.user_feature_map.get(uid, {})
            top_pref = feat.get("top_cuisine_preference")
            if top_pref and top_pref in self.top_cuisine_to_idx:
                pref_idx = self.top_cuisine_to_idx[top_pref]
            else:
                pref_idx = self.top_cuisine_to_idx.get("Other", 0)

            user_rows.append({
                "fe_avg_rating_given": feat.get("avg_rating_given", 3.5),
                "fe_user_log_review_count": feat.get("log_review_count", 0.0),
                "fe_account_age_years": feat.get("account_age_years", 0.0),
                "fe_top_cuisine_preference_idx": pref_idx,
            })
        user_feat_df = pd.DataFrame(user_rows, index=out.index)

        for col in biz_feat_df.columns:
            if col not in out.columns:
                out[col] = biz_feat_df[col]
        for col in user_feat_df.columns:
            if col not in out.columns:
                out[col] = user_feat_df[col]

        return out

    def save(self, path: str | Path) -> Path:
        """Persist fitted pipeline as pickle."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted FeatureEngineer.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return target

    @staticmethod
    def load(path: str | Path) -> "FeatureEngineer":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, FeatureEngineer):
            raise TypeError("Pickle does not contain a FeatureEngineer instance.")
        return obj
