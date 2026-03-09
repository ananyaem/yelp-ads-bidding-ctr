"""Export DeepFM to ONNX and verify parity with PyTorch (atol=1e-5)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.config import DEFAULT_HPARAMS, MODELS_DIR
from src.models.deepfm import DeepFM


def _infer_dnn_layer_sizes(state_dict: dict[str, torch.Tensor]) -> list[int]:
    sizes: list[int] = []
    i = 0
    prefix = "dnn_layer.network."
    while True:
        key = f"{prefix}{i}.weight"
        if key not in state_dict:
            break
        w = state_dict[key]
        if w.dim() == 2:
            sizes.append(int(w.shape[0]))
        i += 4  # Linear, BatchNorm, ReLU, Dropout per block
        if i > 200:
            break
    return sizes


class DeepFMOnnxWrapper(nn.Module):
    """Flat sparse indices + dense vector for ONNX export."""

    def __init__(self, model: DeepFM) -> None:
        super().__init__()
        self.model = model
        self.sparse_keys = [k for k, v in model.feature_config.items() if v.get("type") == "sparse"]

    def forward(self, sparse_flat: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        # sparse_flat: [B, n_sparse] int64
        sparse_inputs = {name: sparse_flat[:, i] for i, name in enumerate(self.sparse_keys)}
        out = self.model(sparse_inputs, dense.float())
        return out["prediction"]


def load_deepfm_from_checkpoint(
    checkpoint_path: str | Path,
    dnn_layers: list[int] | None = None,
    dropout: float | None = None,
) -> tuple[DeepFM, dict[str, Any]]:
    path = Path(checkpoint_path)
    load_kw: dict[str, Any] = {"map_location": "cpu"}
    try:
        ckpt = torch.load(path, **load_kw, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, **load_kw)

    fc = ckpt["feature_config"]
    inferred = _infer_dnn_layer_sizes(ckpt["model_state_dict"])
    layers = (
        dnn_layers
        if dnn_layers is not None
        else (inferred if inferred else list(DEFAULT_HPARAMS.dnn_layers))
    )
    dr = dropout if dropout is not None else DEFAULT_HPARAMS.dropout
    model = DeepFM(fc, dnn_layers=layers, dropout=dr)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, fc


def export_deepfm_onnx(
    checkpoint_path: str | Path,
    onnx_out: str | Path,
    opset: int = 17,
) -> Path:
    """Export DeepFM checkpoint to ONNX. Returns output path."""
    model, _ = load_deepfm_from_checkpoint(checkpoint_path)
    wrapper = DeepFMOnnxWrapper(model)
    wrapper.eval()

    n_sparse = len(wrapper.sparse_keys)
    n_dense = len([k for k, v in model.feature_config.items() if v.get("type") == "dense"])
    batch = 2
    sparse_flat = torch.zeros(batch, n_sparse, dtype=torch.long)
    dense = torch.zeros(batch, n_dense, dtype=torch.float32)

    out_path = Path(onnx_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (sparse_flat, dense),
        str(out_path),
        input_names=["sparse_flat", "dense"],
        output_names=["prediction"],
        dynamic_axes={
            "sparse_flat": {0: "batch"},
            "dense": {0: "batch"},
            "prediction": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return out_path


def verify_onnx_matches_pytorch(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    batch_size: int = 16,
    atol: float = 1e-5,
    seed: int = 0,
) -> None:
    """Raise AssertionError if ONNX Runtime output differs from PyTorch beyond *atol*."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("Install onnxruntime to verify ONNX export.") from exc

    model, fc = load_deepfm_from_checkpoint(checkpoint_path)
    wrapper = DeepFMOnnxWrapper(model)
    wrapper.eval()

    rng = np.random.default_rng(seed)
    sparse_keys = wrapper.sparse_keys
    dense_keys = [k for k, v in fc.items() if v.get("type") == "dense"]

    sparse_flat = np.zeros((batch_size, len(sparse_keys)), dtype=np.int64)
    for i, name in enumerate(sparse_keys):
        vs = int(fc[name]["vocab_size"])
        sparse_flat[:, i] = rng.integers(0, max(vs, 1), size=batch_size)

    dense = rng.standard_normal((batch_size, len(dense_keys))).astype(np.float32)

    with torch.no_grad():
        st = torch.as_tensor(sparse_flat, dtype=torch.long)
        dt = torch.as_tensor(dense, dtype=torch.float32)
        torch_out = np.asarray(
            wrapper(st, dt).detach().cpu().view(-1).tolist(),
            dtype=np.float64,
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        None,
        {"sparse_flat": sparse_flat.astype(np.int64), "dense": dense},
    )[
        0
    ].reshape(-1)

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    assert np.allclose(
        torch_out, onnx_out, atol=atol
    ), f"ONNX mismatch: max_abs_diff={max_diff:.3e} (atol={atol})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DeepFM checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODELS_DIR / "best_deepfm.pt",
        help="Path to PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "deepfm.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true", help="Skip ORT verification")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint.resolve()}")

    out = export_deepfm_onnx(args.checkpoint, args.output, opset=args.opset)
    print(f"Wrote ONNX model to {out}")

    if not args.no_verify:
        verify_onnx_matches_pytorch(args.checkpoint, out)
        print("Verification OK: ONNX matches PyTorch within atol=1e-5")


if __name__ == "__main__":
    main()
