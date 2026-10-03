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
    validate_registration_artifact,
    require_prover_secret,
    run_directory,
    verify_cache,
    verify_checkpoint,
    verify_local_resources,
)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="cpu")
    args = parser.parse_args(argv)
    root = baseline_root()
    try:
        config_path, config = config_with_auth(root, args.dataset)
        resource_errors = verify_local_resources(root, (args.dataset,))
        if resource_errors:
            raise ChainError("resource authentication failed: " + " | ".join(resource_errors))

        verify_cache(root, args.dataset)
        verify_checkpoint(root, args.dataset)
        if config.get("setup_mode") != "hyrax_native_local" or config.get("setup_resource_scope") != "baseline-local":
            raise ChainError("model registration requires the baseline-local Hyrax setup declaration")
        run_relative, run_path = run_directory(root, args.dataset, args.backend, config)
        registration_relative = f"{run_relative}/model_registration.json"
        secret_relative = f"{run_relative}/com_W.prover.secret"
        registration_path = run_path / "model_registration.json"
        secret_path = run_path / "com_W.prover.secret"
        if registration_path.exists() != secret_path.exists():
            raise ChainError(f"incomplete model registration pair in {run_relative}")
        config_relative = str(config_path.relative_to(root))
        completed = launch_native(
            root, args.dataset, args.backend, config_relative, operation="register",
            registration_path_value=registration_relative,
            com_w_secret_path_value=secret_relative,
        )
        atomic_write_text(run_path / "registration.stdout.log", completed.stdout)
        atomic_write_text(run_path / "registration.stderr.log", completed.stderr)
        if completed.returncode != 0:
            raise ChainError(
                f"native registration failed ({completed.returncode}); see {run_relative}/registration.stderr.log"
            )
        validate_registration_artifact(
            root, args.dataset, args.backend, config, registration_path, secret_path
        )
        raw_registration = registration_path.read_text(encoding="utf-8")
        if "blinder" in raw_registration or "padding_mask" in raw_registration:
            raise ChainError(f"public model registration contains prover-only secret material: {registration_path}")
        print(f"REGISTERED dataset={args.dataset} backend={args.backend} path={run_relative}/model_registration.json")
        return 0
    except (ChainError, OSError, ValueError) as exc:
        print(f"REGISTRATION_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
