from __future__ import annotations

import argparse
import json
from pathlib import Path

from .formal_exporter import export_dataset
from .manifest import load_dataset_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal full-dataset GAT exporter for ezkl")
    parser.add_argument("--dataset", required=True, choices=["cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"])
    args = parser.parse_args()

    spec = load_dataset_spec(args.dataset)
    print(json.dumps(export_dataset(spec), indent=2))


if __name__ == "__main__":
    main()
