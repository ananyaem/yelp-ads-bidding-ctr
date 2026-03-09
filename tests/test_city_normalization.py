"""Tests for city name normalization."""

from __future__ import annotations

import pandas as pd
from src.data.yelp_parser import CITY_ALIASES, normalize_city_names


def test_known_aliases_are_normalized() -> None:
    df = pd.DataFrame({"city": ["St. Louis", "St. Petersburg", "Phoenix", "Nashville"]})
    result = normalize_city_names(df)
    assert result["city"].tolist() == ["Saint Louis", "Saint Petersburg", "Phoenix", "Nashville"]


def test_missing_column_returns_unchanged() -> None:
    df = pd.DataFrame({"name": ["foo"]})
    result = normalize_city_names(df, column="city")
    assert "city" not in result.columns


def test_whitespace_is_stripped() -> None:
    df = pd.DataFrame({"city": ["  St. Louis  ", " Phoenix "]})
    result = normalize_city_names(df)
    assert result["city"].tolist() == ["Saint Louis", "Phoenix"]


def test_alias_map_is_complete() -> None:
    assert "St. Louis" in CITY_ALIASES
    assert "St. Petersburg" in CITY_ALIASES
    assert CITY_ALIASES["St. Louis"] == "Saint Louis"
    assert CITY_ALIASES["St. Petersburg"] == "Saint Petersburg"
