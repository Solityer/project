#!/usr/bin/env python3
"""
Planetoid (Cora/Citeseer/Pubmed) GAT training utilities.

This module provides load_checkpoint() used by export_torch_gat_bundle.py to
load PyTorch GAT model checkpoints saved during training.

Checkpoint format (torch.save):
    {
        "bundle":     { metadata dict: L, C, d_in_profile, hidden_profile, ... },
        "state_dict": model.state_dict(),
    }
"""
from __future__ import annotations

import pathlib
from typing import Dict, Union

import torch


def load_checkpoint(
    path: Union[str, pathlib.Path],
    device: torch.device,
) -> Dict[str, object]:
    """Load a saved GAT checkpoint and return {'bundle': ..., 'state_dict': ...}."""
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint at {path} must be a dict, got {type(checkpoint)}")
    if "bundle" not in checkpoint:
        raise ValueError(f"checkpoint at {path} is missing 'bundle' metadata")
    if "state_dict" not in checkpoint:
        raise ValueError(f"checkpoint at {path} is missing 'state_dict'")
    return checkpoint
