//! Construction of a canonical GAT `Model` for the official ezkl graph path.

use std::collections::BTreeMap;

use crate::circuit::{Input, InputType};
use crate::graph::{Model, Node, NodeType, ParsedNodes, SupportedOp, VarVisibility, Visibility};
use crate::RunArgs;

use super::op::GatOp;
use super::relation::RoundingMode;
use super::schema::CanonicalGat;

/// Public input order used by a model built with [`build_model_for_run_args`].
pub const PUBLIC_INPUT_NAMES: [&str; 6] = [
    "feature_table",
    "node_ids",
    "src",
    "dst",
    "node_ptr",
    "edge_ptr",
];

/// The private model closure is one flat runtime input.  It is deliberately
/// not represented by per-tensor Constant nodes: the graph circuit hashes and
/// consumes the same advice cells, so a serialized compiled graph contains no
/// model values.
pub const RUNTIME_PARAM_INPUT_NAME: &str = "model_params";

/// Build a native ezkl model whose final node is one canonical `GatOp`.
///
/// This convenience entry point intentionally fails closed.  The official
/// graph model does not carry the runtime `RunArgs::param_visibility` into a
/// `Model` constructor, so returning a model here could make the `Hashed`
/// declaration look like a real hash module while the caller still runs with
/// private parameters.  Use [`build_model_for_run_args`] when the caller has
/// the exact runtime settings that will be passed to `GraphCircuit::forward`.
pub fn build_model(_spec: CanonicalGat, _rounding: RoundingMode) -> Result<Model, String> {
    Err("unsupported_exact_relation: build_model requires an explicit runtime param_visibility=Hashed{hash_is_public:true}".into())
}

/// Build the canonical model only when the caller has selected ezkl's native
/// public-Hashed parameter module for the same runtime settings used later by
/// compile/forward/prove.  The returned model still follows the official
/// `GraphCircuit` path; this check prevents a private-parameter fallback.
pub fn build_model_for_run_args(
    spec: CanonicalGat,
    rounding: RoundingMode,
    run_args: &RunArgs,
) -> Result<Model, String> {
    if !run_args.param_visibility.is_hashed_public() {
        return Err(
            "unsupported_exact_relation: canonical GAT weights require runtime param_visibility=Hashed{hash_is_public:true}".into(),
        );
    }
    build_model_with_hashed_runtime(spec, rounding)
}

fn build_model_with_hashed_runtime(
    spec: CanonicalGat,
    rounding: RoundingMode,
) -> Result<Model, String> {
    spec.validate_for_compile()?;
    let mut nodes = BTreeMap::new();
    let mut next = 0usize;
    let graph = &spec.graph;
    let public_shapes = [
        (
            vec![graph.n_total, spec.feature_dim],
            InputType::F32,
            spec.scales.s_input,
        ),
        (vec![graph.n], InputType::Int, 0),
        (vec![graph.e], InputType::Int, 0),
        (vec![graph.e], InputType::Int, 0),
        (vec![graph.g_batch + 1], InputType::Int, 0),
        (vec![graph.g_batch + 1], InputType::Int, 0),
    ];
    let mut inputs = Vec::with_capacity(public_shapes.len());
    for (dims, datum_type, scale) in public_shapes {
        let idx = next;
        next += 1;
        nodes.insert(
            idx,
            NodeType::Node(Node {
                opkind: SupportedOp::Input(Input {
                    scale,
                    datum_type,
                    decomp: false,
                }),
                out_scale: scale,
                inputs: vec![],
                out_dims: dims,
                idx,
                num_uses: 1,
            }),
        );
        inputs.push(idx);
    }
    let mut gat_inputs = inputs.iter().map(|idx| (*idx, 0)).collect::<Vec<_>>();
    let parameter_input = next;
    let parameter_len = spec.model.flattened_params().len();
    if parameter_len == 0 {
        return Err("canonical GAT model has an empty private parameter closure".into());
    }
    nodes.insert(
        parameter_input,
        NodeType::Node(Node {
            opkind: SupportedOp::Input(Input {
                // A flat closure can contain multiple canonical scales; each
                // tensor slice is checked by GatOp before use.
                scale: spec.scales.s_w_hidden[0][0],
                datum_type: InputType::F32,
                decomp: false,
            }),
            out_scale: spec.scales.s_w_hidden[0][0],
            inputs: vec![],
            out_dims: vec![parameter_len],
            idx: parameter_input,
            num_uses: 1,
        }),
    );
    gat_inputs.push((parameter_input, 0));
    next += 1;
    let gat_idx = next;
    let gat = GatOp::new(spec.clone(), rounding)?;
    nodes.insert(
        gat_idx,
        NodeType::Node(Node {
            opkind: SupportedOp::Gat(gat),
            out_scale: spec.scales.s_out,
            inputs: gat_inputs,
            out_dims: vec![graph.n, spec.model.classes],
            idx: gat_idx,
            num_uses: 1,
        }),
    );
    let graph = ParsedNodes::from_parts_with_param_inputs(
        nodes,
        inputs,
        vec![parameter_input],
        vec![(gat_idx, 0)],
        vec![InputType::F32],
    );
    Ok(Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            params: Visibility::Hashed {
                hash_is_public: true,
                outlets: vec![],
            },
            output: Visibility::Public,
        },
    })
}
