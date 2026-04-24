from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "manifests" / "gat_formal_datasets.json"


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    output_relpath: str
    config_relpath: str
    cache_relpath: str
    checkpoint_relpath: str
    reference_run_manifest_relpath: str
    task_type: str
    report_unit: str
    batching_rule: str
    self_loop_rule: str
    edge_sort_rule: str
    chunking_rule: str
    deduplicate_edges: bool
    symmetrize_edges: bool
    batch_graphs: int
    protocol_id: str
    preprocess_id: str
    attestation_mode: str
    encoding_id: str
    proof_system_tag: str
    quant_cfg_id: str
    degree_bound_id: str

    def resolve(self) -> "ResolvedDatasetSpec":
        return ResolvedDatasetSpec(
            **self.__dict__,
            repo_root=REPO_ROOT,
            output_root=REPO_ROOT / self.output_relpath,
            config_path=REPO_ROOT / self.config_relpath,
            cache_path=REPO_ROOT / self.cache_relpath,
            checkpoint_path=REPO_ROOT / self.checkpoint_relpath,
            reference_run_manifest_path=REPO_ROOT / self.reference_run_manifest_relpath,
        )


@dataclass(frozen=True)
class ResolvedDatasetSpec(DatasetSpec):
    repo_root: Path
    output_root: Path
    config_path: Path
    cache_path: Path
    checkpoint_path: Path
    reference_run_manifest_path: Path


def load_dataset_spec(dataset: str) -> ResolvedDatasetSpec:
    raw = json.loads(MANIFEST_PATH.read_text())
    entry = raw["datasets"].get(dataset)
    if entry is None:
        known = ", ".join(sorted(raw["datasets"]))
        raise KeyError(f"unknown dataset '{dataset}', expected one of: {known}")
    return DatasetSpec(**entry).resolve()
