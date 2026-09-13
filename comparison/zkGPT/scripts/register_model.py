#!/usr/bin/env python3
"""Register one canonical model through the native zkGPT C++ entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from common import (
    BACKENDS,
    DATASETS,
    ChainError,
    atomic_write_text,
    baseline_root,
    config_with_auth,
    launch_native,
    load_json,
    run_directory,
    verify_checkpoint,
)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="cpu")
    args = parser.parse_args(argv)
    root = baseline_root()
    try:
        config_path, config = config_with_auth(root, args.dataset)
        if config.get("setup_mode") != "hyrax_native_local" or config.get("setup_resource_scope") != "baseline-local":
            raise ChainError("model registration requires the baseline-local Hyrax setup declaration")
        verify_checkpoint(root, args.dataset)
        run_relative, run_path = run_directory(root, args.dataset, args.backend, config)
        config_relative = str(config_path.relative_to(root))
        completed = launch_native(root, args.dataset, args.backend, config_relative, operation="register")
        atomic_write_text(run_path / "registration.stdout.log", completed.stdout)
        atomic_write_text(run_path / "registration.stderr.log", completed.stderr)
        if completed.returncode != 0:
            raise ChainError(
                f"native registration failed ({completed.returncode}); see {run_relative}/registration.stderr.log"
            )
        registration_path = run_path / "model_registration.json"
        registration = load_json(registration_path, "model registration")
        if not isinstance(registration, dict) or registration.get("schema") != "zkgpt-gat-model-registration-v1":
            raise ChainError(f"native registration has an unsupported schema: {registration_path}")
        if registration.get("status") != "registered_long_lived_com_W":
            raise ChainError(f"native registration did not produce a long-lived commitment: {registration_path}")
        required = (
            "dataset", "model_arch_id", "quant_cfg_id", "static_tables_id", "field_id",
            "field_modulus", "manifest_digest", "model_digest", "bounds_digest",
            "quant_digest", "layout_digest", "selector", "opening_policy",
            "binding_digest", "registration_digest", "point_hex",
        )
        for key in required:
            if not isinstance(registration.get(key), str) or not registration[key]:
                raise ChainError(f"native registration is missing {key}: {registration_path}")
        if registration["dataset"] != args.dataset:
            raise ChainError(f"native registration dataset mismatch: {registration_path}")
        if registration["field_id"] != "BN254":
            raise ChainError(f"native registration field is not BN254: {registration_path}")
        if registration["selector"] != "00:X_pub,01:W,10:Aux,11:L_qry":
            raise ChainError(f"native registration selector is not canonical: {registration_path}")
        if registration["opening_policy"] != "same_r_three_openings_v1":
            raise ChainError(f"native registration opening policy is not canonical: {registration_path}")
        try:
            n_raw = int(registration["n_raw"])
            n_in = int(registration["n_in"])
            logical_size = int(registration["logical_size"])
            padded_size = int(registration["padded_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ChainError(f"native registration layout/domain numbers are invalid: {registration_path}") from exc
        if n_raw <= 0 or n_in <= 0 or padded_size != n_in:
            raise ChainError(f"native registration does not use one shared N_in: {registration_path}")
        segments = registration.get("input_segments")
        if not isinstance(segments, list) or len(segments) != 4:
            raise ChainError(f"native registration must contain exactly four input segments: {registration_path}")
        names = ("X_pub", "W", "Aux", "L_qry")
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict) or segment.get("name") != names[index]:
                raise ChainError(f"native registration segment {index} is not canonical: {registration_path}")
            try:
                if int(segment["offset"]) != index * n_in or int(segment["padded_length"]) != n_in:
                    raise ChainError(f"native registration segment {index} does not share N_in: {registration_path}")
            except (KeyError, TypeError, ValueError) as exc:
                raise ChainError(f"native registration segment {index} numbers are invalid: {registration_path}") from exc
        if logical_size != int(segments[1].get("logical_length", -1)) or logical_size > n_in:
            raise ChainError(f"native registration W logical size does not match layout: {registration_path}")
        print(f"REGISTERED dataset={args.dataset} backend={args.backend} path={run_relative}/model_registration.json")
        return 0
    except (ChainError, OSError, ValueError) as exc:
        print(f"REGISTRATION_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
