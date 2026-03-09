"""Tests for Yelp JSON parsing helpers (excluded from coverage omit for this module — tested in isolation)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from src.data.yelp_parser import _read_json_lines, normalize_city_names


def test_normalize_city_aliases() -> None:
    df = pd.DataFrame({"city": ["Phx", "St. Louis", "Boston"]})
    out = normalize_city_names(df, "city")
    assert out["city"].tolist() == ["Phoenix", "Saint Louis", "Boston"]


def test_normalize_city_missing_column() -> None:
    df = pd.DataFrame({"x": [1]})
    out = normalize_city_names(df, "city")
    assert out is df


def test_read_json_lines_missing_file(tmp_path: Path) -> None:
    assert _read_json_lines(tmp_path / "missing.jsonl") is None


def test_read_json_lines_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    rows = [{"a": 1, "b": "z"}, {"a": 2, "b": "y"}]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    df = _read_json_lines(p)
    assert df is not None
    assert len(df) == 2
