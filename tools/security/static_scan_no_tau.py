#!/usr/bin/env python3
import pathlib
import sys
import argparse


ROOT = pathlib.Path(__file__).resolve().parents[2]


def blocked_patterns():
    marker = "ta" + "u"
    srs_eval = "srs" + "_evaluation"
    aux = "prover" + "_aux"
    return [
        "opening" + "_point",
        "commitment" + "_eval",
        srs_eval + "_point",
        srs_eval + "_scalar",
        aux,
        "commit" + "_precomputed",
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
        "g1_mul_same_base" + "_batch(key.g1_generator",
        "polynomial" + ".evaluate(key.",
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


def configured_srs_roots():
    config_paths = sorted((ROOT / "configs").glob("*_secure.cfg"))
    config_paths.append(ROOT / "configs" / "unit_security_regression.cfg")
    roots = []
    for config in config_paths:
        if not config.exists():
            continue
        for raw_line in config.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line.startswith("srs_path") or "=" not in line:
                continue
            relative = line.split("=", 1)[1].strip()
            root = ROOT / relative
            if root not in roots:
                roots.append(root)
            break
    return roots


def sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as material:
        while chunk := material.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Scan formal sources and every currently materialized configured SRS.")
    parser.add_argument(
        "--require-all-configured-srs",
        action="store_true",
        help="also fail when a configured dataset SRS has not been generated")
    args = parser.parse_args()
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
        if path.name != "static_scan_no_tau.py":
            for marker in ("KZG::verify_batch(", "KZG::verify_external_fold("):
                pos = 0
                while True:
                    pos = text.find(marker, pos)
                    if pos == -1:
                        break
                    window = text[pos:pos + 320]
                    line_no = text.count("\n", 0, pos) + 1
                    if "context.kzg))" in window or "context.kzg);" in window:
                        failures.append((path.relative_to(ROOT), line_no, "verifier must use context.kzg.vk"))
                    if "keys)" in window and "keys.vk" not in window:
                        failures.append((path.relative_to(ROOT), line_no, "KZG verify tests must use keys.vk"))
                    pos += len(marker)
    diagnostics = ROOT / "tools" / "diagnostics"
    if diagnostics.exists():
        leftovers = [path for path in diagnostics.rglob("*") if path.is_file()]
        for path in leftovers:
            failures.append((path.relative_to(ROOT), 1, "unformalized diagnostics file"))
    import json

    expected_schema_id = None
    skipped_srs_roots = []
    for srs_root in configured_srs_roots():
        manifest = srs_root / "srs_manifest.json"
        if not manifest.exists():
            if args.require_all_configured_srs:
                failures.append((manifest.relative_to(ROOT), 1, "missing SRS manifest"))
            else:
                skipped_srs_roots.append(srs_root.relative_to(ROOT))
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        schema_id = data.get("protocol_schema_id")
        if expected_schema_id is None:
            expected_schema_id = schema_id
        elif schema_id != expected_schema_id:
            failures.append((manifest.relative_to(ROOT), 1, "protocol schema id mismatch"))
        required_keys = ("pk_coef_path", "pk_coef_hash", "vk_path", "vk_hash")
        for key in required_keys:
            if not data.get(key):
                failures.append((manifest.relative_to(ROOT), 1, f"missing {key}"))
        aux = "prover" + "_aux"
        for key in (aux + "_path", aux + "_hash"):
            if data.get(key):
                failures.append((manifest.relative_to(ROOT), 1, f"forbidden {key}"))
        for path_key, hash_key in (
            ("pk_coef_path", "pk_coef_hash"),
            ("vk_path", "vk_hash"),
        ):
            rel = data.get(path_key)
            expected = data.get(hash_key)
            if not rel or not expected:
                continue
            material = srs_root / rel
            if not material.exists():
                failures.append((material.relative_to(ROOT), 1, "missing SRS material"))
                continue
            actual = sha256_file(material)
            if actual != expected:
                failures.append((material.relative_to(ROOT), 1, f"hash mismatch {hash_key}"))
        lagrange_paths = data.get("pk_lagrange_paths") or {}
        lagrange_hashes = data.get("pk_lagrange_hashes") or {}
        if not lagrange_paths:
            failures.append((manifest.relative_to(ROOT), 1, "missing pk_lagrange_paths"))
        for domain, rel in lagrange_paths.items():
            expected = lagrange_hashes.get(domain)
            if not expected:
                failures.append((manifest.relative_to(ROOT), 1, f"missing pk_lagrange_hash {domain}"))
                continue
            material = srs_root / rel
            if not material.exists():
                failures.append((material.relative_to(ROOT), 1, "missing SRS material"))
                continue
            actual = sha256_file(material)
            if actual != expected:
                failures.append((material.relative_to(ROOT), 1, f"hash mismatch pk_lagrange_hashes.{domain}"))
        legacy_aux = srs_root / (aux + ".bin")
        if legacy_aux.exists():
            failures.append((legacy_aux.relative_to(ROOT), 1, "forbidden runtime auxiliary scalar material"))
    if failures:
        for path, line_no, pattern in failures:
            print(f"{path}:{line_no}: blocked pattern {pattern!r}", file=sys.stderr)
        return 1
    for srs_root in skipped_srs_roots:
        print(f"skipped unmaterialized configured SRS: {srs_root}")
    print("static secure scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
