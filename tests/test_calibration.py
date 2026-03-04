"""Tests for PlattScaler, compute_ece, and PositionDebiaser."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.calibration import PlattScaler, PositionDebiaser, compute_ece
from src.models.deepfm import DeepFM


def _feature_config() -> dict:
    return {
        "restaurant_cuisine": {"name": "restaurant_cuisine", "type": "sparse", "vocab_size": 10, "embedding_dim": 8},
        "campaign_id": {"name": "campaign_id", "type": "sparse", "vocab_size": 5, "embedding_dim": 8},
        "ad_position": {"name": "ad_position", "type": "dense", "vocab_size": None, "embedding_dim": 0},
        "bid_amount": {"name": "bid_amount", "type": "dense", "vocab_size": None, "embedding_dim": 0},
    }


def _make_dummy_df(n: int = 200, position: float | None = None) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "restaurant_cuisine": rng.randint(0, 10, n),
        "campaign_id": rng.randint(0, 5, n),
        "ad_position": position if position is not None else rng.randint(1, 11, n).astype(float),
        "bid_amount": rng.lognormal(0.7, 0.5, n),
        "click": rng.randint(0, 2, n),
    })
    return df


# ---- PlattScaler tests ----

def test_platt_scaler_output_in_0_1():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 1000).astype(float)
    raw_probs = np.clip(rng.beta(2, 5, 1000), 0.01, 0.99)

    scaler = PlattScaler()
    scaler.fit(y_true, raw_probs)
    calibrated = scaler.calibrate(raw_probs)

    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_platt_scaler_improves_calibration():
    rng = np.random.RandomState(1)
    n = 5000
    true_p = rng.uniform(0.05, 0.95, n)
    y_true = (rng.rand(n) < true_p).astype(float)
    raw_probs = np.clip(true_p * 0.5 + 0.1, 0.01, 0.99)

    ece_before = compute_ece(y_true, raw_probs).ece

    scaler = PlattScaler()
    scaler.fit(y_true, raw_probs)
    cal_probs = scaler.calibrate(raw_probs)

    ece_after = compute_ece(y_true, cal_probs).ece
    assert ece_after < ece_before


def test_platt_scaler_save_load():
    rng = np.random.RandomState(2)
    y_true = rng.randint(0, 2, 500).astype(float)
    raw_probs = np.clip(rng.beta(2, 5, 500), 0.01, 0.99)

    scaler = PlattScaler()
    scaler.fit(y_true, raw_probs)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "scaler.pkl"
        scaler.save(path)

        assert path.stat().st_size < 1024 * 1024  # < 1 MB

        loaded = PlattScaler.load(path)
        np.testing.assert_allclose(loaded.a, scaler.a)
        np.testing.assert_allclose(loaded.b, scaler.b)

        original_cal = scaler.calibrate(raw_probs)
        loaded_cal = loaded.calibrate(raw_probs)
        np.testing.assert_allclose(original_cal, loaded_cal)


# ---- compute_ece tests ----

def test_ece_perfect_calibration_is_zero():
    y_prob = np.array([0.1] * 100 + [0.9] * 100)
    y_true = np.array([0] * 90 + [1] * 10 + [0] * 10 + [1] * 90)
    result = compute_ece(y_true, y_prob, n_bins=10)
    assert result.ece < 1e-9


def test_ece_returns_bin_data():
    rng = np.random.RandomState(3)
    y_true = rng.randint(0, 2, 500).astype(float)
    y_prob = rng.uniform(0, 1, 500)
    result = compute_ece(y_true, y_prob, n_bins=10)

    assert len(result.bin_edges) == 11
    assert len(result.bin_mean_pred) == 10
    assert len(result.bin_mean_true) == 10
    assert len(result.bin_count) == 10
    assert result.bin_count.sum() == 500


# ---- PositionDebiaser tests ----

def test_position_debiaser_constant_across_positions():
    fc = _feature_config()
    model = DeepFM(feature_config=fc, dnn_layers=[32, 16], dropout=0.0)
    model.eval()

    scaler = PlattScaler()
    scaler.a = 1.0
    scaler.b = 0.0
    scaler._fitted = True

    debiaser = PositionDebiaser(model=model, feature_config=fc, scaler=scaler)

    base_df = _make_dummy_df(n=50, position=1.0)

    predictions = []
    for pos in range(1, 11):
        df = base_df.copy()
        df["ad_position"] = float(pos)
        preds = debiaser.predict(df, calibrate=True)
        predictions.append(preds)

    stacked = np.stack(predictions, axis=0)
    max_deviation = np.ptp(stacked, axis=0).max()
    assert max_deviation < 1e-5, f"Debiased predictions vary by {max_deviation}"


def test_position_debiaser_raw_model_varies_with_position():
    """Without debiasing, the model should produce different predictions at different positions."""
    fc = _feature_config()
    model = DeepFM(feature_config=fc, dnn_layers=[64, 32], dropout=0.0)

    torch.manual_seed(99)
    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.uniform_(p, -0.5, 0.5)
    model.eval()

    base_df = _make_dummy_df(n=100, position=1.0)

    from src.training.trainer import AdClickDataset, _collate_batch
    from torch.utils.data import DataLoader

    means = []
    for pos in [1.0, 5.0, 10.0]:
        df = base_df.copy()
        df["ad_position"] = pos
        ds = AdClickDataset(df, fc)
        loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=_collate_batch)
        all_p = []
        with torch.no_grad():
            for batch in loader:
                out = model(batch["sparse"], batch["dense"])
                all_p.append(out["prediction"].view(-1))
        means.append(torch.cat(all_p).mean().item())

    assert not (abs(means[0] - means[1]) < 1e-6 and abs(means[1] - means[2]) < 1e-6), \
        "Model should vary with position before debiasing"
