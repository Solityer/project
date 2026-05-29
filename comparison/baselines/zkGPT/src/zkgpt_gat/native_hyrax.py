from __future__ import annotations

import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .field import write_field_vector
from .util import read_json, stable_env


@dataclass(frozen=True)
class NativeHyrax:
    zk_root: pathlib.Path

    @property
    def binary(self) -> pathlib.Path:
        return self.zk_root / "cmake-build-release" / "zkgpt_hyrax_mle"

    def require_binary(self) -> None:
        if not self.binary.exists():
            raise FileNotFoundError(
                f"missing Hyrax/MLE backend: {self.binary}; build target zkgpt_hyrax_mle first"
            )

    def commit_open(
        self,
        label: str,
        values: Iterable[int],
        point: Iterable[int],
        work_dir: pathlib.Path,
    ) -> Dict[str, Any]:
        self.require_binary()
        work_dir.mkdir(parents=True, exist_ok=True)
        vector_path = work_dir / f"{_safe(label)}.vector.fr"
        point_path = work_dir / f"{_safe(label)}.point.fr"
        out_path = work_dir / f"{_safe(label)}.hyrax.json"
        write_field_vector(vector_path, values)
        write_field_vector(point_path, point)
        completed = subprocess.run(
            [str(self.binary), "commit-open", label, str(vector_path), str(point_path), str(out_path)],
            cwd=str(self.zk_root),
            env=stable_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Hyrax/MLE backend failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return read_json(out_path)

    def verify_open(
        self,
        label: str,
        proof_path: pathlib.Path,
        point: Iterable[int],
        work_dir: pathlib.Path,
    ) -> Dict[str, Any]:
        self.require_binary()
        work_dir.mkdir(parents=True, exist_ok=True)
        point_path = work_dir / f"{_safe(label)}.verify.point.fr"
        out_path = work_dir / f"{_safe(label)}.verify.json"
        write_field_vector(point_path, point)
        completed = subprocess.run(
            [str(self.binary), "verify-open", label, str(proof_path), str(point_path), str(out_path)],
            cwd=str(self.zk_root),
            env=stable_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode not in {0, 3}:
            raise RuntimeError(
                "Hyrax/MLE verifier failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return read_json(out_path)


def mle_point_from_u64(challenge: int, variable_count: int) -> List[int]:
    # BN254 field elements are represented as decimal strings for the native CLI.
    # The xorshift stream is deterministic and bound to the Fiat-Shamir challenge.
    state = int(challenge) & ((1 << 64) - 1)
    out: List[int] = []
    for _ in range(variable_count):
        state ^= (state << 13) & ((1 << 64) - 1)
        state ^= state >> 7
        state ^= (state << 17) & ((1 << 64) - 1)
        out.append(state)
    return out


def _safe(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
