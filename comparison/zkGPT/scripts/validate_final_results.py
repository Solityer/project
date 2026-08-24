#!/usr/bin/env python3
"""Independent artifact/manifest validator for the formal result matrix."""
from __future__ import annotations

import json
import hashlib
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.gpu_execution import validate_gpu_execution_evidence


def _config_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            entries[key.strip()] = value.strip()
    return entries


def _resolve_public_y(config_path: Path, dataset: str, backend: str, project_root: Path, public_y_root: Path) -> Path:
    entries = _config_entries(config_path)
    candidates: list[Path] = []
    configured = entries.get("public_y_path")
    if configured:
        path = Path(configured)
        candidates.append(path if path.is_absolute() else project_root / path)
    root = public_y_root.expanduser().resolve()
    if root.is_file():
        candidates.append(root)
    elif root.is_dir():
        candidates.extend(root / name for name in (
            f"{dataset}.json", f"{dataset}.npy", f"{dataset}.txt",
            f"{backend}/{dataset}.json", f"{backend}/{dataset}.npy", f"{backend}/{dataset}.txt",
        ))
    else:
        raise FileNotFoundError(f"public-Y root does not exist: {root}")
    reference = Path(entries.get("reference_output_dir", ""))
    if not reference.is_absolute():
        reference = project_root / reference
    candidates.extend(reference / name for name in ("public_y.json", "public_y.npy", "Y.json", "Y.npy", "logits.npy"))
    existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            existing.append(resolved)
    if len(existing) != 1:
        raise RuntimeError(f"expected exactly one independent public-Y source for {dataset}/{backend}, found {len(existing)}: {existing}")
    return existing[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the final zkGPT result matrix with the strict external verifier")
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--public-y-root", type=Path, required=True, help="independent public-Y file or directory")
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    result_path = ROOT / "results" / "final_zkgpt_results.json"
    errors: list[str] = []
    if not result_path.exists():
        errors.append("result_matrix_missing")
        rows = []
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = result.get("rows", [])
        if len(rows) != 10:
            errors.append("matrix_row_count_not_10")
        expected_pairs = {
            (backend, dataset)
            for backend in ("cpu", "gpu")
            for dataset in ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
        }
        actual_pairs = {(str(row.get("backend")), str(row.get("dataset"))) for row in rows}
        if actual_pairs != expected_pairs:
            errors.append("matrix_dataset_backend_coverage_invalid")
        for row in rows:
            dataset = row.get("dataset")
            backend = row.get("backend")
            if row.get("status") != "VERIFY_OK" or row.get("proof_generated") is not True or row.get("verified") is not True:
                errors.append(f"run_not_verified:{dataset}/{backend}")
            manifest_path = ROOT / "results" / "runs" / str(backend) / str(dataset) / "run_manifest.json"
            if row.get("status") == "VERIFY_OK" and not manifest_path.exists():
                errors.append(f"manifest_missing:{dataset}/{backend}")
                continue
            if not manifest_path.exists():
                continue
            artifact_dir = manifest_path.parent
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata_path = artifact_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            config_value = manifest.get("config")
            config_path = Path(str(config_value)) if config_value else Path()
            if not config_path.is_absolute():
                config_path = project_root / config_path
            registry_value = metadata.get("model_registry", {}).get("public_path")
            registry_path = ROOT / str(registry_value) if registry_value and not Path(str(registry_value)).is_absolute() else Path(str(registry_value or ""))
            try:
                public_y_path = _resolve_public_y(config_path, str(dataset), str(backend), project_root, args.public_y_root)
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"independent_context_inputs_invalid:{dataset}/{backend}:{exc}")
                continue
            replay = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "verify_proof.py"),
                    "--project-root", str(project_root),
                    "--config", str(config_path),
                    "--trusted-registry-public", str(registry_path),
                    "--artifact-dir", str(artifact_dir),
                    "--public-y", str(public_y_path),
                ],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if replay.returncode != 0:
                errors.append(f"independent_replay_failed:{dataset}/{backend}")
                continue
            independent_path = artifact_dir / "independent_verification.json"
            independent = json.loads(independent_path.read_text(encoding="utf-8")) if independent_path.is_file() else {}
            if independent.get("status") != "VERIFY_OK":
                errors.append(f"independent_status_failed:{dataset}/{backend}")
            if manifest.get("independent_verification_sha256") != _sha256(independent_path):
                errors.append(f"independent_digest_mismatch:{dataset}/{backend}")
            if manifest.get("full_gat_forward_proved") is not True or manifest.get("final_logits_bound") is not True or manifest.get("scope_mismatch") is not False:
                errors.append(f"proof_scope_invalid:{dataset}/{backend}")
            if backend == "gpu":
                gpu = manifest.get("gpu_backend", {})
                gpu_contract_valid, gpu_contract_errors = validate_gpu_execution_evidence(gpu)
                if not gpu_contract_valid:
                    errors.append(
                        f"gpu_proof_primitives_invalid:{dataset}/{backend}:"
                        + ",".join(gpu_contract_errors)
                    )
    audit = {
        "schema": "zkGPT_gat_formal_validation_v3",
        "baseline": "zkGPT",
        "status": "verified" if not errors else "not_verified",
        "formal_experiment_eligible": not errors,
        "errors": errors,
        "rows_checked": len(rows),
    }
    output = ROOT / "results" / "local_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"zkgpt_formal_validation={audit['status']} errors={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
