"""Tests for project configuration helpers."""

from __future__ import annotations

import numpy as np
from src.config import DEFAULT_HPARAMS, HyperParams, set_seed


def test_set_seed_is_deterministic_numpy() -> None:
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    np.testing.assert_array_equal(a, b)


def test_default_hparams_frozen_fields() -> None:
    assert DEFAULT_HPARAMS.embedding_dim >= 4
    assert isinstance(DEFAULT_HPARAMS.dnn_layers, list)
    hp = HyperParams(seed=7)
    assert hp.seed == 7
