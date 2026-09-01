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

# The comparison baseline consumes the same canonical numerical GAT shape as
# the current formal model. Keep this contract in one production module so
# preflight, result validation, and resource-bound metadata all reject stale
# L2/single-head configurations consistently.
FORMAL_GAT_LAYER_COUNT = 3
FORMAL_GAT_HIDDEN_PROFILE = ((8, 8), (8, 8))
FORMAL_GAT_D_IN_PROFILE = {
    "cora": (1433, 64),
    "citeseer": (3703, 64),
    "pubmed": (500, 64),
    "ppi": (50, 64),
    "ogbn-arxiv": (128, 64),
}
FORMAL_GAT_K_OUT = 2
FORMAL_GAT_CLASS_COUNTS = {
    "cora": 7,
    "citeseer": 6,
    "pubmed": 3,
    "ppi": 121,
    "ogbn-arxiv": 40,
}
FORMAL_GAT_MODEL_ARCH_IDS = {
    dataset: f"gat_family_L3_hid8x8-8x8_kout2_c{class_count}"
    for dataset, class_count in FORMAL_GAT_CLASS_COUNTS.items()
}

PROTOCOL_ID = "GAT-ZKML-zkGPT"
PROOF_BACKEND_ID = "zkgpt-style-gkr-lasso-matmul-hyrax-mle"
ROUNDING_MODE_NEAREST = "nearest_ties_away"
ROUNDING_MODE_TOWARD_ZERO = "toward_zero"
ROUNDING_MODES = (ROUNDING_MODE_NEAREST, ROUNDING_MODE_TOWARD_ZERO)
ENCODING_ID = "gat-zkgpt-fixed-order-v1"
PADDING_RULE_ID = "protocol-defined-family-padding+selector-mask-v1"
CIRCUIT_LAYOUT_ID = "gat-zkgpt-four-segment-Xpub-W-Aux-L-v1"
LASSO_CFG_ID = "gat-zkgpt-native-lasso-packed-tuples-range-padding-v1"
MATRIX_PROOF_CFG_ID = "gat-zkgpt-dedicated-exact-integer-matmul-sumcheck-v1"
COMMITMENT_BACKEND_ID = "zkGPT-hyrax-mle-evaluation-commitment-v2"

# Cross-module wire/schema identities.  Keep these in one production module
# so the producer, fresh verifier, and result chain cannot silently drift to
# different historical strings.
EXTERNAL_PUBLIC_STATEMENT_SCHEMA = "zkGPT_external_public_statement_v1"
EXTERNAL_VERIFIER_CONTEXT_SCHEMA = "zkGPT_external_verifier_context_v1"
STATIC_TABLE_VK_SCHEMA = "zkGPT-static-table-vk-v1"
VERIFIER_SETUP_SCHEMA = "zkGPT-gat-verifier-setup-v1"
LAYOUT_SCHEMA = "zkGPT_gat_four_segment_layout_v1"
GPU_PROOF_MESSAGE_SCHEMA = "zkGPT_gpu_proof_messages_v3"
RUN_MANIFEST_SCHEMA = "zkGPT_gat_run_manifest_v4"
INDEPENDENT_REPORT_SCHEMA = "zkGPT_independent_verification_v4"
RESULT_MATRIX_SCHEMA = "zkGPT_gat_formal_result_matrix_v4"
FORMAL_PREFLIGHT_SCHEMA = "zkGPT_gat_formal_preflight_v3"
INTEGER_BOUND_LEDGER_SCHEMA = "zkGPT_gat_integer_bound_ledger_v2"
LASSO_LAYOUT_SCHEMA = "zkGPT_lasso_family_lc_layout_v1"

# The external verifier envelope has one canonical wire shape.  Keep the
# field sets next to the schema identities so producer, fresh verifier, and
# test fixtures cannot quietly grow historical aliases or private fields.
PUBLIC_STATEMENT_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "dataset",
        "dataset_display_name",
        "task_type",
        "features",
        "graph",
        "public_output",
        "input_preprocess_id",
        "rounding_mode",
        "integer_bound_ledger_digest",
    }
)
PUBLIC_STATEMENT_ENVELOPE_FIELDS = PUBLIC_STATEMENT_FIELDS | {"statement_digest"}
STATIC_TABLE_VK_FIELDS = frozenset(
    {
        "schema",
        "rounding_mode",
        "static_table_id",
        "quant_cfg_id",
        "input_preprocess_id",
        "static_table_descriptor_schema",
        "static_table_content_digest",
        "static_table_descriptor_digest",
        "integer_bound_ledger_digest",
        "table_registry_digest",
        "vk_digest",
    }
)
VERIFIER_SETUP_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "proof_backend_id",
        "commitment_backend_id",
        "circuit_layout_id",
        "model_arch_id",
        "model_param_id",
        "quant_cfg_id",
        "input_preprocess_id",
        "rounding_mode",
        "static_table_id",
        "registry_digest",
        "Layout_in_digest",
        "domain_cfg",
        "dim_cfg",
        "integer_bound_ledger_digest",
        "setup_digest",
    }
)
EXTERNAL_VERIFIER_CONTEXT_FIELDS = frozenset(
    {
        "schema",
        "public_statement",
        "model_registry",
        "static_table_vk",
        "setup_context",
        "context_digest",
    }
)
STATIC_TABLE_FAMILY_IDS = ("T_LReLU", "T_ELU", "T_exp", "T_range")
FUNCTION_LOOKUP_DOMAIN_BITS = 63
LOOKUP_TUPLE_ENCODING_ID = "Pack_f_offset_base_v1"

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
    "input_preprocess_id",
    "rounding_mode",
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
