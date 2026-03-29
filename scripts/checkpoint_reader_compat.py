#!/usr/bin/env python3
from __future__ import annotations

import inspect
from collections import namedtuple
from pathlib import Path
from typing import Dict

import numpy as np


def _patch_runtime() -> None:
    if not hasattr(inspect, "ArgSpec"):
        inspect.ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_


def load_checkpoint_reader():
    _patch_runtime()
    from tensorflow_checkpoint_reader.py_checkpoint_reader import NewCheckpointReader

    return NewCheckpointReader


def read_checkpoint_tensors(
    checkpoint_prefix: str | Path,
    include_optimizer: bool = False,
) -> Dict[str, np.ndarray]:
    reader_cls = load_checkpoint_reader()
    reader = reader_cls(str(checkpoint_prefix))
    shape_map = reader.get_variable_to_shape_map()
    tensors: Dict[str, np.ndarray] = {}
    for name in sorted(shape_map):
        if not include_optimizer and (
            name.endswith("/Adam")
            or name.endswith("/Adam_1")
            or name in {"beta1_power", "beta2_power"}
        ):
            continue
        tensors[name] = reader.get_tensor(name)
    return tensors
