#!/usr/bin/env python3
"""Generate and authenticate the baseline-local zkDL PP/resource manifest.

This is the only setup entry point used by the formal runner.  It invokes the
native CPU binary for PP generation/inspection and writes the staged resource
manifest only after every staged bundle and the PP have been authenticated.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from runner_contract import (
    BASELINE_ROOT,
    PP_DEFAULT_MAX_VECTOR_LEN,
    PP_SETUP_ID,
    STAGED_RESOURCE_MANIFEST_PATH,
    atomic_write_json,
    build_staged_resource_manifest,
    ensure_contained,
    ensure_regular_file,
    load_resource_manifest,
    load_resource_policy,
    pp_path_from_manifest,
    validate_pp_file,
)


def _relative(path: Path) -> str:
    return ensure_contained(path, label="formal command path").relative_to(BASELINE_ROOT).as_posix()


def _native_command(binary: Path, subcommand: str) -> list[object]:
    """Build a native invocation with the mandatory exact baseline root."""
    return [
        _relative(binary),
        "--project-root",
        str(BASELINE_ROOT),
        subcommand,
    ]


def _run(command: list[object], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=BASELINE_ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )


def _remove_staged_manifest() -> None:
    path = ensure_contained(STAGED_RESOURCE_MANIFEST_PATH, label="staged resource manifest")
    if path.is_symlink():
        raise ValueError("staged resource manifest may not be a symlink")
    if path.exists():
        ensure_regular_file(path, label="staged resource manifest")
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=BASELINE_ROOT / "bin" / "zkdl-formal-cpu",
        help="native CPU formal binary produced by scripts/build.sh",
    )
    parser.add_argument("--max-vector-len", type=int, default=None)
    parser.add_argument("--seed-label", default=PP_SETUP_ID)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    binary = ensure_regular_file(
        args.binary if args.binary.is_absolute() else BASELINE_ROOT / args.binary,
        label="CPU formal binary",
    )
    policy, _ = load_resource_policy()
    policy_pp = policy["pp"]
    requested_max = policy_pp["max_vector_len"] if args.max_vector_len is None else args.max_vector_len
    if requested_max != policy_pp["max_vector_len"]:
        raise SystemExit("--max-vector-len must equal the source policy max_vector_len")
    pp_path = ensure_contained(BASELINE_ROOT / policy_pp["path"], label="canonical PP path")
    pp_path.parent.mkdir(parents=True, exist_ok=True)
    if pp_path.exists() or pp_path.is_symlink():
        ensure_regular_file(pp_path, label="canonical PP")
        if args.replace:
            _run(
                _native_command(binary, "setup")
                + [
                    "--output",
                    _relative(pp_path),
                    "--manifest",
                    "artifacts/setup/setup_manifest.json",
                    "--max-vector-len",
                    requested_max,
                    "--seed-label",
                    args.seed_label,
                ]
            )
    else:
        _run(
            _native_command(binary, "setup")
            + [
                "--output",
                _relative(pp_path),
                "--manifest",
                "artifacts/setup/setup_manifest.json",
                "--max-vector-len",
                requested_max,
                "--seed-label",
                args.seed_label,
            ]
        )

    pp_info = json.loads(
        _run(
            _native_command(binary, "pp-info")
            + [
                "--pp",
                _relative(pp_path),
                "--manifest",
                "artifacts/setup/pp_info.json",
            ],
            capture=True,
        ).stdout
    )
    staged = build_staged_resource_manifest(require_all=True, pp_info=pp_info)
    validate_pp_file(pp_path, staged, pp_info)
    _remove_staged_manifest()
    atomic_write_json(STAGED_RESOURCE_MANIFEST_PATH, staged)
    print(
        "ZKDL_RESOURCES_READY "
        f"manifest_id={staged['manifest_id']} pp_id={staged['pp']['pp_id']} "
        f"datasets={len(staged['datasets'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
