#!/usr/bin/env python3
"""Check the Icicle CUDA backend required by the strict GPU route."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EZKL_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_LIBS = [
    "libicicle_backend_cuda_device.so",
    "libicicle_backend_cuda_curve_bn254.so",
    "libicicle_backend_cuda_field_bn254.so",
    "libicicle_backend_cuda_hash.so",
]


def run_command(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {
            "command": " ".join(command),
            "returncode": None,
            "error": f"not_found: {exc.filename}",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "returncode": None,
            "error": "timeout",
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
        }


def tail_text(value: object, limit: int = 4000) -> str:
    text = value if isinstance(value, str) else ""
    return text[-limit:]


def classify_arch(file_output: str) -> str:
    lowered = file_output.lower()
    if "x86-64" in lowered or "x86_64" in lowered:
        return "x86_64"
    if "aarch64" in lowered or "arm aarch64" in lowered:
        return "aarch64"
    if "arm64" in lowered:
        return "aarch64"
    return "unknown"


def host_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    return machine


def candidate_backend_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get("ICICLE_BACKEND_INSTALL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    for root in [EZKL_ROOT / ".icicle_backend_local", EZKL_ROOT / ".gpu_backend_local"]:
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.is_dir())
    halo2_dirs = [
        path
        for path in (EZKL_ROOT / ".cargo_local" / "git" / "checkouts").glob(
            "halo2-*/**/icicle/backend/cuda"
        )
        if path.is_dir()
    ]
    halo2_dirs.sort(key=lambda path: (0 if "1dd2090" in str(path) else 1, str(path)))
    candidates.extend(halo2_dirs)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def choose_backend_dir() -> Path | None:
    for path in candidate_backend_dirs():
        if all((path / name).exists() for name in REQUIRED_LIBS):
            return path
    return None


def library_report(backend_dir: Path | None, expected_arch: str) -> dict[str, object]:
    if backend_dir is None:
        return {
            "available": False,
            "required_libraries": REQUIRED_LIBS,
            "missing_libraries": REQUIRED_LIBS,
            "arch_match": False,
        }
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for name in REQUIRED_LIBS:
        path = backend_dir / name
        if not path.exists():
            missing.append(name)
            rows.append({"name": name, "path": str(path), "exists": False})
            continue
        file_result = run_command(["file", str(path)])
        ldd_result = run_command(["ldd", str(path)])
        file_output = str(file_result.get("stdout") or file_result.get("stderr") or "")
        arch = classify_arch(file_output)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": True,
                "file": file_output,
                "ldd_returncode": ldd_result.get("returncode"),
                "ldd_stdout": tail_text(ldd_result.get("stdout")),
                "ldd_stderr": tail_text(ldd_result.get("stderr")),
                "arch": arch,
                "arch_matches_host": arch == expected_arch,
            }
        )
    present_arches = sorted({str(row.get("arch")) for row in rows if row.get("exists")})
    return {
        "available": backend_dir.exists(),
        "path": str(backend_dir),
        "required_libraries": REQUIRED_LIBS,
        "missing_libraries": missing,
        "libraries": rows,
        "library_arches": present_arches,
        "arch_match": not missing and bool(rows) and all(row.get("arch_matches_host") for row in rows),
    }


def gpu_report() -> dict[str, object]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version",
        "--format=csv,noheader",
    ]
    result = run_command(query)
    devices: list[dict[str, str]] = []
    if result.get("returncode") == 0:
        for line in str(result.get("stdout", "")).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                devices.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "driver_version": parts[2],
                    }
                )
    return {
        "available": result.get("returncode") == 0,
        "device_count": len(devices),
        "devices": devices,
        "command": result.get("command"),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def cuda_report() -> dict[str, object]:
    nvcc = run_command(["nvcc", "--version"])
    smi = run_command(["nvidia-smi"])
    return {
        "nvcc": nvcc,
        "nvidia_smi": {
            "command": smi.get("command"),
            "returncode": smi.get("returncode"),
            "stdout": tail_text(smi.get("stdout")),
            "stderr": tail_text(smi.get("stderr")),
        },
    }


def ezkl_env(backend_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["EZKL_REQUIRE_GPU_BACKEND"] = "1"
    env["ICICLE_BACKEND_INSTALL_DIR"] = str(backend_dir)
    icicle_lib = EZKL_ROOT / "target" / "release" / "deps" / "icicle" / "lib"
    ld_parts = []
    if icicle_lib.exists():
        ld_parts.append(str(icicle_lib))
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


def initialization_report(backend_dir: Path | None) -> dict[str, object]:
    ezkl = EZKL_ROOT / "target" / "release" / "ezkl"
    if backend_dir is None:
        return {"attempted": False, "status": "backend_dir_missing"}
    if not ezkl.exists():
        return {"attempted": False, "status": "ezkl_binary_missing", "binary": str(ezkl)}
    with tempfile.TemporaryDirectory(prefix=".gpu_backend_check_", dir=EZKL_ROOT) as work_dir_name:
        work_dir = Path(work_dir_name)
        command = [
            str(ezkl),
            "gen-srs",
            "--srs-path",
            "gpu_backend_check.srs",
            "--logrows",
            "1",
        ]
        result = run_command(command, cwd=work_dir, env=ezkl_env(backend_dir), timeout=60)
    stderr = str(result.get("stderr", ""))
    stdout = str(result.get("stdout", ""))
    return {
        "attempted": True,
        "status": "ok" if result.get("returncode") == 0 else "failed",
        "returncode": result.get("returncode"),
        "command": result.get("command"),
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }


def make_report() -> dict[str, object]:
    arch = host_arch()
    backend_dir = choose_backend_dir()
    libraries = library_report(backend_dir, arch)
    gpu = gpu_report()
    init = initialization_report(backend_dir)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host_arch": arch,
        "platform_machine": platform.machine(),
        "gpu": gpu,
        "cuda": cuda_report(),
        "icicle_backend_install_dir": str(backend_dir) if backend_dir else "",
        "backend_libraries": libraries,
        "initialization": init,
        "usable": bool(
            gpu.get("available")
            and gpu.get("device_count", 0) > 0
            and libraries.get("arch_match")
            and init.get("status") == "ok"
        ),
    }


def print_text(report: dict[str, object]) -> None:
    libraries = report["backend_libraries"]
    init = report["initialization"]
    gpu = report["gpu"]
    print(f"host_arch={report['host_arch']}")
    print(f"gpu_available={gpu.get('available')} gpu_device_count={gpu.get('device_count')}")
    print(f"icicle_backend_install_dir={report.get('icicle_backend_install_dir')}")
    print(f"backend_arch_match={libraries.get('arch_match')}")
    for row in libraries.get("libraries", []):
        if isinstance(row, dict):
            print(f"{row.get('name')}: arch={row.get('arch')} match={row.get('arch_matches_host')}")
    print(f"initialization_attempted={init.get('attempted')} status={init.get('status')}")
    print(f"usable={report.get('usable')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = make_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 0 if report.get("usable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
