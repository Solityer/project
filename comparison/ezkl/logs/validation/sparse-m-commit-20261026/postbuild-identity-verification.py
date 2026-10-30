"""Recompute the nine official EZKL GPU build identity checks read-only."""
import json
import os
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "scripts"))
from formal_common import (  # noqa: E402
    CONTRACT,
    FORMAL_GPU_EXECUTION_POLICY,
    canonical_json_digest,
    native_backend_identity,
    resource_manifest_hash,
    sha256,
    verify_native_source_tree_hash,
)

validation = root / "logs/validation/sparse-m-commit-20261026"
build_log = (validation / "official-gpu-build-full.log").read_text()
sidecar_match = re.search(r"BUILD_STAGE=verify-native-source-identity sidecar=(\S+)", build_log)
if sidecar_match is None:
    raise SystemExit("current official build log lacks source identity sidecar")
sidecar = Path(sidecar_match.group(1))
receipt_path = root / "artifacts/build/gpu/build_receipt.json"
receipt = json.loads(receipt_path.read_text())
selected = receipt["selected_cuda_device"]
os.environ["EZKL_FORMAL_SELECTED_CUDA_DEVICE"] = json.dumps(selected, sort_keys=True, separators=(",", ":"))
os.environ["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
os.environ["ICICLE_DEVICE_ID"] = "0"

def check(label: str, expected: str, actual: str) -> None:
    if expected != actual:
        raise SystemExit(f"FAIL {label} expected={expected} actual={actual}")
    print(f"PASS {label} expected={expected} actual={actual}")

source_sha = verify_native_source_tree_hash(sidecar)
check("source_tree_sha256", receipt["source_tree_sha256"], source_sha)
binary_sha = sha256(root / receipt["binary_path"])
check("binary_sha256", receipt["binary_sha256"], binary_sha)
native = native_backend_identity()
check("native_backend_manifest_sha256", receipt["native_backend_manifest_sha256"], native["native_backend_manifest_sha256"])
for library, digest in sorted(native["native_backend_libraries"].items()):
    check(f"native_library:{Path(library).name}", receipt["native_backend_libraries"][library], digest)
check("resource_manifest_sha256", receipt["resource_manifest_sha256"], resource_manifest_hash())
check("adapter_contract_rawbytes_sha256", receipt["adapter_contract_sha256"], sha256(CONTRACT))
environment = dict(receipt["build_environment"])
embedded_digest = environment.pop("build_environment_sha256", None)
canonical_digest = canonical_json_digest(environment)
check("build_environment_canonical_json_sha256", receipt["build_environment_sha256"], canonical_digest)
check("build_environment_embedded_sha256", receipt["build_environment_sha256"], embedded_digest)
expected_policy = "halo2_icicle-bn254-gpu-hgraph-fused-ntt-16-25-msm-lagrange-resident-f4-sparse-m-cpu32-v6"
if receipt["native_execution_policy"] != expected_policy or native["native_execution_policy"] != expected_policy or FORMAL_GPU_EXECUTION_POLICY != expected_policy:
    raise SystemExit("FAIL native_execution_policy expected approved v6 composite")
print(f"PASS native_execution_policy={expected_policy}")
print("IDENTITY_CHECKS=9/9")
