#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for item in args.paths:
        path = Path(item)
        print(f"{sha256_file(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
