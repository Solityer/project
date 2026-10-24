#!/usr/bin/env python3
"""Create the per-run config and resource paths for one canonical dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")


def read_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        if "=" not in content:
            raise ValueError(f"malformed config line {number}: {path}")
        key, value = (part.strip() for part in content.split("=", 1))
        if not key or not value or key in values:
            raise ValueError(f"invalid or duplicate config key at line {number}: {path}")
        values[key] = value
    return values


def session_updates(dataset: str, session: str) -> dict[str, str]:
    if dataset not in DATASETS:
        raise ValueError("formal dataset is not canonical")
    resources = session + "/resources"
    return {
        "export_dir": session + "/" + dataset,
        "static_table_descriptor": resources + "/canonical/tables/" + dataset + "_gat_static_tables.json",
        "sorted_lookup_descriptor": resources + "/canonical/tables/" + dataset + "_gat_sorted_lookup.json",
        "integer_bound_ledger": resources + "/canonical/ledgers/" + dataset + "_gat_integer_ledger.json",
        "srs_id": Path(session).name + "_" + dataset + "_formal_srs",
        "srs_path": resources + "/srs/" + dataset,
        "srs_manifest_digest": "CEREMONY_REQUIRED",
        "model_registry_path": resources + "/verifier_registry/" + dataset + ".json",
        "verifier_registry_digest": "CEREMONY_REQUIRED",
        "model_blinder_sidecar_path": resources + "/model_blinders/" + dataset + ".json",
        "auto_prepare_dataset": "false",
        "dump_trace": "false",
    }


def configure(root: Path, session: Path, dataset: str, source_config: Path | None = None) -> Path:
    root = root.resolve(strict=True)
    session = Path(os.path.abspath(session))
    if session.parent != root / "runs" or not session.is_dir() or session.is_symlink():
        raise ValueError(f"formal session directory is unavailable: {session}")
    relative = session.relative_to(root).as_posix()
    original = source_config or root / "configs" / f"{dataset}_secure.cfg"
    values = read_config(original)
    if values.get("dataset") != dataset:
        raise ValueError("formal config/dataset mismatch")

    updates = session_updates(dataset, relative)
    values.update(updates)
    config_path = root / "configs" / f"{session.name}_{dataset}.cfg"
    config_path.write_text("".join(f"{key} = {value}\n" for key, value in values.items()), encoding="utf-8")

    run = root / updates["export_dir"]
    run.mkdir(parents=True, exist_ok=True)
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    if args.config is not None and args.dataset is not None:
        parser.error("--dataset cannot be combined with --config")
    root = Path(__file__).resolve().parent.parent
    session = Path(os.path.abspath(args.session))
    if session.parent != root / "runs" or not session.is_dir() or session.is_symlink():
        raise SystemExit(f"formal-session: session directory is unavailable: {session}")
    if args.config is not None:
        source_config = args.config.resolve(strict=True)
        if source_config.parent != root / "configs":
            raise SystemExit("formal-session: config must be under baseline-local configs/")
        dataset = read_config(source_config).get("dataset")
        if dataset not in DATASETS:
            raise SystemExit("formal-session: noncanonical dataset")
        configure(root, session, dataset, source_config)
        return 0
    if args.dataset is not None:
        configure(root, session, args.dataset)
        print(f"formal_session_config=READY dataset={args.dataset}", flush=True)
        return 0
    for dataset in DATASETS:
        configure(root, session, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
