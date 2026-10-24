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
