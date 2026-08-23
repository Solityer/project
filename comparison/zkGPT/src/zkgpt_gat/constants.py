from __future__ import annotations

DATASET_ORDER = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]

CONFIG_BY_DATASET = {
    "cora": "cora_secure.cfg",
    "citeseer": "citeseer_secure.cfg",
    "pubmed": "pubmed_secure.cfg",
    "ppi": "ppi_secure.cfg",
    "ogbn-arxiv": "ogbn_arxiv_secure.cfg",
}

CONFIG_DATASET_VALUE = {
    "cora": "cora",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
    "ppi": "ppi",
    "ogbn-arxiv": "ogbn_arxiv",
}

PROTOCOL_ID = "gat-zkgpt-v1"
PROOF_BACKEND_ID = "zkgpt-style-gkr-lasso-matmul-hyrax-mle"
ENCODING_ID = "gat-zkgpt-fixed-order-v1"
PADDING_RULE_ID = "zero-pad+selector-mask"
CIRCUIT_LAYOUT_ID = "gat-zkgpt-seven-segment-layout-v1"
LASSO_CFG_ID = "gat-zkgpt-lasso-lrelu-elu-exp-range-structured-limbs-v1"
MATRIX_PROOF_CFG_ID = "gat-zkgpt-batched-matmul-sumcheck-v1"
COMMITMENT_BACKEND_ID = "bn254-field-hyrax-mle-openmasked-v2"

TRANSCRIPT_DOMAIN_ORDER = [
    "gat-zkgpt/meta",
    "gat-zkgpt/model-commit",
    "gat-zkgpt/static-table",
    "gat-zkgpt/input-commit",
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

# The retained research graft uses a seven-segment layout and private com_T_H;
# it is intentionally not the protocol's formal X|W|Aux|L commitment order.
REFERENCE_GRAFT_COMMITMENT_ORDER = ["com_T_H", "com_W", "com_base", "com_comp", "com_psq", "com_qry"]
COMMITMENT_ORDER = REFERENCE_GRAFT_COMMITMENT_ORDER
