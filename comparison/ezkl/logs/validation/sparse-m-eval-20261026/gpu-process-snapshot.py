"""Read-only process and physical-device snapshot for official build."""
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out_dir = Path(__file__).resolve().parent
selected_uuid = "GPU-3cde57ba-ae43-4452-bd59-7b156b36db71"
ps = subprocess.run(
    ["ps", "-eo", "pid=,ppid=,etimes=,%cpu=,%mem=,stat=,args="],
    check=True, capture_output=True, text=True,
)
needles = (
    "comparison/zkdl/scripts/formal_runner.py",
    "comparison/zkdl/gat_runner",
    "comparison/zkGPT",
    "comparison/ezkl",
    "build_ezkl.sh",
    "nvcc",
    "cargo",
)
lines = [line for line in ps.stdout.splitlines() if any(n in line for n in needles)]
process_text = "\n".join(lines) + "\n"
(out_dir / "prebuild-process-snapshot.txt").write_text(process_text, encoding="utf-8")
gpu = subprocess.run(
    [
        "/usr/bin/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader",
    ],
    check=True, capture_output=True, text=True,
)
apps = subprocess.run(
    ["/usr/bin/nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
    check=True, capture_output=True, text=True,
)
gpu_text = gpu.stdout.strip() + "\n" + apps.stdout.strip() + "\n"
(out_dir / "prebuild-gpu-snapshot.csv").write_text(gpu_text, encoding="utf-8")
if selected_uuid not in gpu.stdout:
    raise SystemExit(f"selected physical GPU UUID not present in snapshot: {selected_uuid}")
print(f"SELECTED_UUID_MATCH={selected_uuid}")
print("PROCESS_SNAPSHOT_BEGIN")
print(process_text, end="")
print("PROCESS_SNAPSHOT_END")
print("GPU_SNAPSHOT_BEGIN")
print(gpu_text, end="")
print("GPU_SNAPSHOT_END")
