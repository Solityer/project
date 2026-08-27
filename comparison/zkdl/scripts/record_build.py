#!/usr/bin/env python3
"""Record an authenticated release-binary identity after scripts/build.sh."""
from __future__ import annotations

import argparse
from pathlib import Path

from runner_contract import BASELINE_ROOT, DEVICES, write_build_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=DEVICES, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary if args.binary.is_absolute() else BASELINE_ROOT / args.binary
    payload = write_build_manifest(args.device, binary)
    print(f"BUILD_MANIFEST_OK device={args.device} build_identity={payload['build_identity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
