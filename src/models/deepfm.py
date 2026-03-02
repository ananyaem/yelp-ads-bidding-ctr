"""DeepFM model components for CTR prediction."""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """Maps sparse feature indices to learned embeddings."""

    def __init__(self, feature_config: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.feature_config = feature_config
        self.sparse_features = [k for k, v in feature_config.items() if v.get("type") == "sparse"]

        self.embeddings = nn.ModuleDict()
        self.linear_embeddings = nn.ModuleDict()
        self.embedding_dims: dict[str, int] = {}
        self.vocab_sizes: dict[str, int] = {}

        for feature_name in self.sparse_features:
            spec = feature_config[feature_name]
            vocab_size = int(spec["vocab_size"])
            embed_dim = int(spec["embedding_dim"])
            self.embedding_dims[feature_name] = embed_dim
            self.vocab_sizes[feature_name] = vocab_size

            emb = nn.Embedding(vocab_size, embed_dim)
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embeddings[feature_name] = emb

            linear_emb = nn.Embedding(vocab_size, 1)
            nn.init.xavier_uniform_(linear_emb.weight)
            self.linear_embeddings[feature_name] = linear_emb

        unique_dims = set(self.embedding_dims.values())
        if len(unique_dims) > 1:
            raise ValueError(f"All sparse features must share embedding_dim for FM; got {unique_dims}")

    def _clamp_indices(self, indices: torch.Tensor, feature_name: str) -> torch.Tensor:
        max_idx = self.vocab_sizes[feature_name] - 1
        clamped = indices.clamp(min=0, max=max_idx)
        if not torch.equal(indices, clamped):
            warnings.warn(
                f"OOV indices detected for feature '{feature_name}'. Clamped to [0, {max_idx}].",
                RuntimeWarning,
                stacklevel=2,
            )
        return clamped

    def forward(self, sparse_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fm_embeddings: list[torch.Tensor] = []
        linear_terms: list[torch.Tensor] = []

        for feature_name in self.sparse_features:
            if feature_name not in sparse_inputs:
                raise KeyError(f"Missing sparse input feature: {feature_name}")

            raw_idx = sparse_inputs[feature_name].long()
            if raw_idx.dim() == 2 and raw_idx.size(1) == 1:
                raw_idx = raw_idx.squeeze(1)
            clamped_idx = self._clamp_indices(raw_idx, feature_name)

            fm_embeddings.append(self.embeddings[feature_name](clamped_idx))
            linear_terms.append(self.linear_embeddings[feature_name](clamped_idx))

        fm_stack = torch.stack(fm_embeddings, dim=1)  # [B, F, D]
        linear_sparse = torch.stack(linear_terms, dim=1).sum(dim=1)  # [B, 1]
        return {"fm_embeddings": fm_stack, "linear_sparse": linear_sparse}


class FMLayer(nn.Module):
    """Second-order FM interactions with O(k*n) complexity."""

    def forward(self, embeddings: torch.Tensor, feature_values: torch.Tensor | None = None) -> torch.Tensor:
        # embeddings: [B, F, D]
        if feature_values is not None:
            # feature_values expected shape [B, F] or [B, F, 1].
            if feature_values.dim() == 2:
                feature_values = feature_values.unsqueeze(-1)
            vx = embeddings * feature_values
        else:
            vx = embeddings

        summed = torch.sum(vx, dim=1)  # [B, D]
        summed_square = summed * summed
        square_summed = torch.sum(vx * vx, dim=1)  # [B, D]

        fm_out = 0.5 * torch.sum(summed_square - square_summed, dim=1, keepdim=True)  # [B, 1]
        return fm_out


class DNNLayer(nn.Module):
    """Configurable MLP tower with BatchNorm, ReLU, and Dropout."""

    def __init__(self, input_dim: int, layer_sizes: list[int] | None = None, dropout: float = 0.3) -> None:
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [256, 128, 64]
        if not layer_sizes:
            raise ValueError("layer_sizes must be non-empty.")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in layer_sizes:
            linear = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = out_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = layer_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepFM(nn.Module):
    """DeepFM architecture combining linear, FM, and DNN components."""

    def __init__(
        self,
        feature_config: dict[str, dict[str, Any]],
        dnn_layers: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_config = feature_config
        self.sparse_features = [k for k, v in feature_config.items() if v.get("type") == "sparse"]
        self.dense_features = [k for k, v in feature_config.items() if v.get("type") == "dense"]

        self.embedding_layer = EmbeddingLayer(feature_config=feature_config)
        self.fm_layer = FMLayer()

        self.dense_linear = nn.Linear(len(self.dense_features), 1)
        nn.init.xavier_uniform_(self.dense_linear.weight)
        nn.init.zeros_(self.dense_linear.bias)

        common_embed_dim = next(iter(self.embedding_layer.embedding_dims.values())) if self.sparse_features else 0
        dnn_input_dim = len(self.dense_features) + len(self.sparse_features) * common_embed_dim
        self.dnn_layer = DNNLayer(input_dim=dnn_input_dim, layer_sizes=dnn_layers, dropout=dropout)
        self.dnn_out_linear = nn.Linear(self.dnn_layer.output_dim, 1)
        nn.init.xavier_uniform_(self.dnn_out_linear.weight)
        nn.init.zeros_(self.dnn_out_linear.bias)

    def forward(self, sparse_inputs: dict[str, torch.Tensor], dense_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if dense_inputs.dim() == 1:
            dense_inputs = dense_inputs.unsqueeze(0)

        emb_out = self.embedding_layer(sparse_inputs)
        fm_embeddings = emb_out["fm_embeddings"]
        linear_sparse = emb_out["linear_sparse"]  # [B, 1]

        fm_out = self.fm_layer(fm_embeddings)  # [B, 1]
        linear_dense = self.dense_linear(dense_inputs.float())  # [B, 1]

        dnn_in = torch.cat([fm_embeddings.flatten(start_dim=1), dense_inputs.float()], dim=1)
        dnn_hidden = self.dnn_layer(dnn_in)
        dnn_logit = self.dnn_out_linear(dnn_hidden)  # [B, 1]

        logit = linear_sparse + linear_dense + fm_out + dnn_logit
        prediction = torch.sigmoid(logit)

        return {
            "prediction": prediction,
            "fm_out": fm_out,
            "dnn_out": dnn_hidden,
        }

    def parameter_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}

    def model_summary(self) -> str:
        counts = self.parameter_count()
        return (
            f"DeepFM(sparse={len(self.sparse_features)}, dense={len(self.dense_features)}), "
            f"params(total={counts['total']}, trainable={counts['trainable']})"
        )
