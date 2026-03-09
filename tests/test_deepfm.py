from __future__ import annotations

import torch
from src.models.deepfm import DeepFM


def _feature_config() -> dict:
    return {
        "primary_cuisine": {
            "name": "primary_cuisine",
            "type": "sparse",
            "vocab_size": 20,
            "embedding_dim": 8,
        },
        "city_encoded": {
            "name": "city_encoded",
            "type": "sparse",
            "vocab_size": 10,
            "embedding_dim": 8,
        },
        "time_of_day": {
            "name": "time_of_day",
            "type": "sparse",
            "vocab_size": 5,
            "embedding_dim": 8,
        },
        "rating_vs_city_avg": {
            "name": "rating_vs_city_avg",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
        "restaurant_log_review_count": {
            "name": "restaurant_log_review_count",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
        "price_range": {
            "name": "price_range",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
        "price_missing": {
            "name": "price_missing",
            "type": "dense",
            "vocab_size": None,
            "embedding_dim": 0,
        },
    }


def _build_inputs(batch_size: int, device: torch.device):
    sparse_inputs = {
        "primary_cuisine": torch.randint(0, 20, (batch_size,), device=device),
        "city_encoded": torch.randint(0, 10, (batch_size,), device=device),
        "time_of_day": torch.randint(0, 5, (batch_size,), device=device),
    }
    dense_inputs = torch.randn(batch_size, 4, device=device)
    return sparse_inputs, dense_inputs


def test_forward_output_shape_and_prediction_range_cpu():
    model = DeepFM(feature_config=_feature_config())
    model.eval()
    sparse_inputs, dense_inputs = _build_inputs(batch_size=32, device=torch.device("cpu"))

    out = model(sparse_inputs, dense_inputs)
    assert out["prediction"].shape == (32, 1)
    assert out["fm_out"].shape == (32, 1)
    assert out["dnn_out"].shape[0] == 32

    assert torch.all(out["prediction"] > 0.0)
    assert torch.all(out["prediction"] < 1.0)


def test_gradient_flows_all_parameters():
    model = DeepFM(feature_config=_feature_config())
    model.train()
    sparse_inputs, dense_inputs = _build_inputs(batch_size=16, device=torch.device("cpu"))

    out = model(sparse_inputs, dense_inputs)
    loss = out["prediction"].mean()
    loss.backward()

    missing_grad = [
        name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing_grad, f"Parameters with no gradient: {missing_grad}"


def test_single_sample_batch_size_one():
    model = DeepFM(feature_config=_feature_config())
    model.eval()
    sparse_inputs, dense_inputs = _build_inputs(batch_size=1, device=torch.device("cpu"))

    out = model(sparse_inputs, dense_inputs)
    assert out["prediction"].shape == (1, 1)


def test_oov_index_clamped_no_crash():
    model = DeepFM(feature_config=_feature_config())
    model.eval()
    sparse_inputs, dense_inputs = _build_inputs(batch_size=4, device=torch.device("cpu"))

    # Inject OOV and negative indices intentionally.
    sparse_inputs["primary_cuisine"][0] = 999
    sparse_inputs["city_encoded"][1] = -7

    out = model(sparse_inputs, dense_inputs)
    assert out["prediction"].shape == (4, 1)


def test_model_summary_and_expected_param_count():
    model = DeepFM(feature_config=_feature_config())
    counts = model.parameter_count()
    summary = model.model_summary()

    assert counts["total"] > 0
    assert counts["trainable"] > 0
    assert "params(total=" in summary


def test_forward_on_gpu_if_available():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    model = DeepFM(feature_config=_feature_config()).to(device)
    model.eval()
    sparse_inputs, dense_inputs = _build_inputs(batch_size=8, device=device)

    out = model(sparse_inputs, dense_inputs)
    assert out["prediction"].shape == (8, 1)
