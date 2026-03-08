"""Inference utilities: sponsored-listing pipeline and CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.inference.pipeline import InferencePipeline

__all__ = ["InferencePipeline"]


def __getattr__(name: str):
    if name == "InferencePipeline":
        from src.inference.pipeline import InferencePipeline

        return InferencePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
