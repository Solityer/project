from __future__ import annotations

DATASET_ORDER = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]

CONFIG_BY_DATASET = {
    "cora": "cora_full.cfg",
    "citeseer": "citeseer_full.cfg",
    "pubmed": "pubmed_full.cfg",
    "ppi": "ppi_batch_formal.cfg",
    "ogbn-arxiv": "ogbn_arxiv_full.cfg",
}

CONFIG_DATASET_VALUE = {
    "cora": "cora",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
    "ppi": "ppi",
    "ogbn-arxiv": "ogbn_arxiv",
}

PROTOCOL_ID = "GAT-ZKML-zkGPT"
PROOF_BACKEND_ID = "zkgpt-style-gkr-lasso-matmul-hyrax-mle"
ENCODING_ID = "gat-zkgpt-fixed-order-v1"
PADDING_RULE_ID = "protocol-defined-family-padding+selector-mask-v1"
CIRCUIT_LAYOUT_ID = "gat-zkgpt-four-segment-Xpub-W-Aux-L-v1"
LASSO_CFG_ID = "gat-zkgpt-native-lasso-packed-tuples-range-padding-v1"
MATRIX_PROOF_CFG_ID = "gat-zkgpt-dedicated-exact-integer-matmul-sumcheck-v1"
COMMITMENT_BACKEND_ID = "zkGPT-hyrax-mle-evaluation-commitment-v2"

TRANSCRIPT_DOMAIN_ORDER = [
    "gat-zkgpt/meta",
    "gat-zkgpt/model-commit",
    "gat-zkgpt/static-table",
    "gat-zkgpt/input-commit/aux",
    "gat-zkgpt/input-commit/lasso",
    "gat-zkgpt/lasso",
    "gat-zkgpt/matmul",
    "gat-zkgpt/gkr",
    "gat-zkgpt/open",
]

METADATA_FIELD_ORDER = [
    "protocol_id",
    "proof_backend_id",
    "model_arch_id",
    "model_param_id",
    "static_table_id",
    "quant_cfg_id",
    "domain_cfg",
    "dim_cfg",
    "encoding_id",
    "padding_rule_id",
    "circuit_layout_id",
    "lasso_cfg_id",
    "matrix_proof_cfg_id",
]

# X_pub is reconstructed by the verifier and is intentionally uncommitted.
# W is registered preprocessing state; Aux and L are fresh per-inference
# segments.  This order is part of the public input-layer selector relation.
COMMITMENT_ORDER = ["com_W", "com_Aux", "com_L"]
