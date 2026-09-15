"""Checkpoint helpers that avoid storing frozen pretrained weights."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def model_state_for_checkpoint(model: nn.Module) -> tuple[dict[str, torch.Tensor], bool]:
    if hasattr(model, "trainable_state_dict"):
        return model.trainable_state_dict(), True
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}, False


def load_model_checkpoint_state(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    compact = bool(checkpoint.get("compact_model_state", False))
    incompatible = model.load_state_dict(checkpoint["model"], strict=not compact)
    if compact and incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected compact-checkpoint keys: {incompatible.unexpected_keys}")

