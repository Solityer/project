#!/usr/bin/env python3
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]


def blocked_patterns():
    marker = "ta" + "u"
    return [
        "opening" + "_point",
        "commitment" + "_eval",
        "FieldElement " + marker,
        "." + marker,
        "key." + marker,
        "evaluate(" + marker,
        "evaluate(key." + marker,
        "seed" + " + " + "17",
        "KZG::" + "setup(" + "seed",
        marker + "_evaluation",
        "commit_" + marker + "_evaluation",
        "deterministic_" + marker,
        "test_" + marker,
        "seeded_" + marker,
    ]


def scan_paths():
    include_roots = [
        ROOT / "include" / "gatzk",
        ROOT / "src",
        ROOT / "configs",
        ROOT / "scripts",
        ROOT / "tools" / "security",
        ROOT / "tests",
        ROOT / "docs" / "security",
    ]
    files = [ROOT / "README.md", ROOT / "CMakeLists.txt"]
    for root in include_roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return files


def main():
    patterns = blocked_patterns()
    failures = []
    for path in scan_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern in patterns:
                if pattern in line:
                    failures.append((path.relative_to(ROOT), line_no, pattern))
    srs_root = ROOT / "artifacts" / "srs" / "zkgat_secure_main_srs_v1"
    manifest = srs_root / "srs_manifest.json"
    if not manifest.exists():
        failures.append((manifest.relative_to(ROOT), 1, "missing SRS manifest"))
    else:
        import hashlib
        import json

        data = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("pk_coef_path", "pk_coef_hash", "vk_path", "vk_hash", "prover_aux_path", "prover_aux_hash"):
            if not data.get(key):
                failures.append((manifest.relative_to(ROOT), 1, f"missing {key}"))
        for path_key, hash_key in (
            ("pk_coef_path", "pk_coef_hash"),
            ("vk_path", "vk_hash"),
            ("prover_aux_path", "prover_aux_hash"),
        ):
            rel = data.get(path_key)
            expected = data.get(hash_key)
            if not rel or not expected:
                continue
            material = srs_root / rel
            if not material.exists():
                failures.append((material.relative_to(ROOT), 1, "missing SRS material"))
                continue
            actual = hashlib.sha256(material.read_bytes()).hexdigest()
            if actual != expected:
                failures.append((material.relative_to(ROOT), 1, f"hash mismatch {hash_key}"))
    if failures:
        for path, line_no, pattern in failures:
            print(f"{path}:{line_no}: blocked pattern {pattern!r}", file=sys.stderr)
        return 1
    print("static secure scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
