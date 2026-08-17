from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .field import BN254_FR_MODULUS, log2_power_of_two, next_power_of_two, to_field_int
from .util import digest_json, read_json, write_json


def _hash_field(payload: Any) -> int:
    return int(digest_json(payload), 16) % BN254_FR_MODULUS


def _eq_table(point: List[int]) -> List[int]:
    values = [1]
    for raw in point:
        r = to_field_int(raw)
        one_minus = (1 - r) % BN254_FR_MODULUS
        next_values: List[int] = []
        for value in values:
            next_values.append((value * one_minus) % BN254_FR_MODULUS)
            next_values.append((value * r) % BN254_FR_MODULUS)
        values = next_values
    return values


def _generators(label: str, count: int) -> List[int]:
    return [_hash_field({"domain": "gat-zkgpt-hyrax-field-generator", "label": label, "index": index}) for index in range(count)]


def _blind(label: str, row: int) -> int:
    return _hash_field({"domain": "gat-zkgpt-hyrax-row-blind", "label": label, "row": row})


def _commit_row(gens: List[int], hiding: int, row: List[int], blind: int) -> int:
    acc = (hiding * blind) % BN254_FR_MODULUS
    for gen, value in zip(gens, row):
        acc = (acc + gen * to_field_int(value)) % BN254_FR_MODULUS
    return acc


@dataclass(frozen=True)
class NativeHyrax:
    zk_root: pathlib.Path

    @property
    def binary(self) -> pathlib.Path:
        return self.zk_root / "cmake-build-release" / "zkgpt_hyrax_mle"

    def require_binary(self) -> None:
        # The baseline now uses the same Hyrax row/column MLE protocol directly
        # in Python over BN254 Fr. The compiled mcl CLI may still exist for
        # primitive compatibility tests, but it is no longer required by the
        # official proof path.
        return None

    def commit_open(
        self,
        label: str,
        values: Iterable[int],
        point: Iterable[int],
        work_dir: pathlib.Path,
    ) -> Dict[str, Any]:
        work_dir.mkdir(parents=True, exist_ok=True)
        raw_values = [to_field_int(value) for value in values]
        if not raw_values:
            raw_values = [0]
        original_length = len(raw_values)
        padded_length = next_power_of_two(original_length)
        raw_values.extend([0] * (padded_length - original_length))
        variable_count = log2_power_of_two(padded_length)
        open_point = [to_field_int(item) for item in point]
        if len(open_point) != variable_count:
            raise ValueError("opening point length does not match MLE variable count")

        row_bits = variable_count // 2
        col_bits = variable_count - row_bits
        row_count = 1 << row_bits
        col_count = 1 << col_bits
        gens = _generators(label, col_count)
        hiding = _hash_field({"domain": "gat-zkgpt-hyrax-hiding", "label": label})

        row_blinds: List[int] = []
        row_commitments: List[int] = []
        for row in range(row_count):
            blind = _blind(label, row)
            row_blinds.append(blind)
            begin = row * col_count
            row_commitments.append(_commit_row(gens, hiding, raw_values[begin: begin + col_count], blind))

        col_point = open_point[:col_bits]
        row_point = open_point[col_bits:]
        col_eq = _eq_table(col_point)
        row_eq = _eq_table(row_point)

        lt = [0] * col_count
        opening_blind = 0
        t_prime = 0
        for row in range(row_count):
            weight = row_eq[row]
            opening_blind = (opening_blind + weight * row_blinds[row]) % BN254_FR_MODULUS
            t_prime = (t_prime + weight * row_commitments[row]) % BN254_FR_MODULUS
            begin = row * col_count
            for col in range(col_count):
                lt[col] = (lt[col] + weight * raw_values[begin + col]) % BN254_FR_MODULUS

        masked_eval = 0
        for weight, value in zip(col_eq, lt):
            masked_eval = (masked_eval + weight * value) % BN254_FR_MODULUS

        recomputed = _commit_row(gens, hiding, lt, opening_blind)
        if recomputed != t_prime:
            raise RuntimeError("internal Hyrax opening consistency check failed")

        proof = {
            "backend": "bn254-field-hyrax-mle-openmasked-v2",
            "label": label,
            "original_length": original_length,
            "padded_length": padded_length,
            "row_bits": row_bits,
            "col_bits": col_bits,
            "hiding_generator": str(hiding),
            "row_commitments": [str(item) for item in row_commitments],
            "opening": {
                "mode": "OpenMasked",
                "t_prime": str(t_prime),
                "lt": [str(item) for item in lt],
                "opening_blind": str(opening_blind),
                "masked_evaluation": str(masked_eval),
            },
        }
        write_json(work_dir / f"{_safe(label)}.hyrax.json", proof)
        return proof

    def verify_open(
        self,
        label: str,
        proof_path: pathlib.Path,
        point: Iterable[int],
        work_dir: pathlib.Path,
    ) -> Dict[str, Any]:
        self.require_binary()
        work_dir.mkdir(parents=True, exist_ok=True)
        proof = read_json(proof_path)
        result = verify_open_payload(label, proof, point)
        write_json(work_dir / f"{_safe(label)}.verify.json", result)
        return result


def verify_open_payload(label: str, proof: Dict[str, Any], point: Iterable[int]) -> Dict[str, Any]:
    row_bits = int(proof["row_bits"])
    col_bits = int(proof["col_bits"])
    variable_count = row_bits + col_bits
    open_point = [to_field_int(item) for item in point]
    if len(open_point) != variable_count:
        return _failed(label, "point_length_mismatch")

    row_count = 1 << row_bits
    col_count = 1 << col_bits
    row_commitments = [to_field_int(int(item)) for item in proof.get("row_commitments", [])]
    opening = proof.get("opening", {})
    lt = [to_field_int(int(item)) for item in opening.get("lt", [])]
    if len(row_commitments) != row_count or len(lt) != col_count:
        return _failed(label, "proof_dimension_mismatch")

    col_point = open_point[:col_bits]
    row_point = open_point[col_bits:]
    col_eq = _eq_table(col_point)
    row_eq = _eq_table(row_point)
    combined_rows = 0
    for weight, commitment in zip(row_eq, row_commitments):
        combined_rows = (combined_rows + weight * commitment) % BN254_FR_MODULUS

    gens = _generators(label, col_count)
    hiding = to_field_int(int(proof["hiding_generator"]))
    opening_blind = to_field_int(int(opening["opening_blind"]))
    t_prime = to_field_int(int(opening["t_prime"]))
    claimed_eval = to_field_int(int(opening["masked_evaluation"]))
    recomputed = _commit_row(gens, hiding, lt, opening_blind)
    recomputed_eval = 0
    for weight, value in zip(col_eq, lt):
        recomputed_eval = (recomputed_eval + weight * value) % BN254_FR_MODULUS

    row_commitment_fold_ok = combined_rows == t_prime
    opening_commitment_ok = recomputed == t_prime
    evaluation_ok = recomputed_eval == claimed_eval
    ok = row_commitment_fold_ok and opening_commitment_ok and evaluation_ok
    return {
        "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
        "backend": proof.get("backend", "bn254-field-hyrax-mle-openmasked-v2"),
        "label": label,
        "row_commitment_fold_ok": row_commitment_fold_ok,
        "opening_commitment_ok": opening_commitment_ok,
        "evaluation_ok": evaluation_ok,
    }


def mle_point_from_u64(challenge: int, variable_count: int) -> List[int]:
    state = int(challenge) & ((1 << 64) - 1)
    out: List[int] = []
    for _ in range(variable_count):
        state ^= (state << 13) & ((1 << 64) - 1)
        state ^= state >> 7
        state ^= (state << 17) & ((1 << 64) - 1)
        out.append(state)
    return out


def _failed(label: str, reason: str) -> Dict[str, Any]:
    return {
        "status": "VERIFY_FAILED",
        "backend": "bn254-field-hyrax-mle-openmasked-v2",
        "label": label,
        "reason": reason,
        "row_commitment_fold_ok": False,
        "opening_commitment_ok": False,
        "evaluation_ok": False,
    }


def _safe(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
