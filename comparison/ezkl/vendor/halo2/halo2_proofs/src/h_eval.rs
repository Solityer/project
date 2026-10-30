use group::ff::PrimeField;

/// One fixed/advice/instance polynomial query used by a fused row program.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GateQuery {
    /// 0=fixed, 1=advice, 2=instance.
    pub(crate) kind: u32,
    /// Column index in the selected source.
    pub(crate) column: usize,
    /// Halo2 row rotation.
    pub(crate) rotation: i32,
}

/// A straight-line gate program executed once per row on the selected device.
#[derive(Clone, Debug)]
pub(crate) struct GateProgram<F: PrimeField> {
    /// Eight-u32 instructions; word zero is the opcode.
    pub(crate) instructions: Vec<[u32; 8]>,
    /// Canonical field constants referenced by load-constant instructions.
    pub(crate) constants: Vec<F>,
    /// Query descriptors indexed by load-query instructions.
    pub(crate) queries: Vec<GateQuery>,
    /// Number of row-major scratch slots.
    pub(crate) scratch_slots: usize,
    /// Number of ordered Horner folds in the instruction stream.
    pub(crate) fold_count: usize,
    /// Lookup folds appended to a straight-line graph program. The output
    /// slots begin at `output_base`; each lookup stores all input values and
    /// its table value there before the CUDA kernel applies the fused fold.
    pub(crate) mv_lookup: Option<MvLookupProgram>,
}

#[derive(Clone, Debug)]
pub(crate) struct MvLookupProgram {
    pub(crate) output_base: usize,
    pub(crate) input_counts: Vec<usize>,
    pub(crate) graph_query_count: usize,
}

pub(crate) const GATE_OP_LOAD_QUERY: u32 = 0;
pub(crate) const GATE_OP_LOAD_CONST: u32 = 1;
pub(crate) const GATE_OP_ADD: u32 = 2;
pub(crate) const GATE_OP_SUB: u32 = 3;
pub(crate) const GATE_OP_MUL: u32 = 4;
pub(crate) const GATE_OP_SQUARE: u32 = 5;
pub(crate) const GATE_OP_DOUBLE: u32 = 6;
pub(crate) const GATE_OP_NEGATE: u32 = 7;
pub(crate) const GATE_OP_FOLD: u32 = 8;
/// Scale the H accumulator by a canonical constant (instruction word 1).
pub(crate) const GATE_OP_SCALE_ACCUMULATOR: u32 = 9;
/// Fold one gate into the current dependency group's local accumulator.
pub(crate) const GATE_OP_FOLD_BLOCK: u32 = 10;
/// Add the local group accumulator times a canonical constant (word 1) to H.
pub(crate) const GATE_OP_COMMIT_BLOCK: u32 = 11;
/// Copy a computed graph root into its reserved row-major output slot.
pub(crate) const GATE_OP_STORE_OUTPUT: u32 = 12;
/// Apply one multi-lookup H fold using the saved graph roots and auxiliary queries.
pub(crate) const GATE_OP_MV_LOOKUP_FOLD: u32 = 13;
