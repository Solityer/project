#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
phase=""
session_dir=""
dataset=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --internal-phase)
            [[ "$#" -ge 2 && -z "$phase" ]] || { echo "run-five: malformed --internal-phase" >&2; exit 2; }
            phase="$2"
            shift 2
            ;;
        --session-dir)
            [[ "$#" -ge 2 && -z "$session_dir" ]] || { echo "run-five: malformed --session-dir" >&2; exit 2; }
            session_dir="$2"
            shift 2
            ;;
        --dataset)
            [[ "$#" -ge 2 && -z "$dataset" ]] || { echo "run-five: malformed --dataset" >&2; exit 2; }
            dataset="$2"
            shift 2
            ;;
        *)
            echo "run-five: usage: $0 [--internal-phase prepare|cell|finalize --session-dir ABS [--dataset CANONICAL]]" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$phase" && -z "$session_dir" && -z "$dataset" ]]; then
    echo "run-five: the public full-matrix entry is /home/pzh/project/comparison/run_formal.sh" >&2
    exit 2
fi
if [[ "$phase" != "prepare" && "$phase" != "cell" && "$phase" != "finalize" ]]; then
    echo "run-five: internal phase must be prepare, cell, or finalize" >&2
    exit 2
fi
if [[ -z "$session_dir" ]]; then
    echo "run-five: --session-dir is required for internal phases" >&2
    exit 2
fi
if [[ "$phase" == "cell" ]]; then
    case "$dataset" in
        cora|citeseer|pubmed|ppi|ogbn_arxiv) ;;
        *) echo "run-five: cell phase requires one canonical --dataset" >&2; exit 2 ;;
    esac
elif [[ -n "$dataset" ]]; then
    echo "run-five: --dataset is accepted only for cell phase" >&2
    exit 2
fi

args=(--internal-phase "$phase" --session-dir "$session_dir")
if [[ -n "$dataset" ]]; then args+=(--dataset "$dataset"); fi
exec python3 "$script_dir/formal_phase.py" "${args[@]}"
