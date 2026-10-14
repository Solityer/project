//! Compact native GAT bounded-range constraints and their row schedule.
//!
//! Large public intervals use two byte-bounded nonnegative witnesses
//! `x - lo` and `hi - x`. Their sum is the public interval span, so the byte
//! decompositions enforce exactly `lo <= x <= hi` without a wide fixed range
//! table or signed-difference bookkeeping tensors.

use std::collections::BTreeMap;

use halo2_proofs::{
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
    poly::Rotation,
};
use halo2curves::{bn256::Fr as Fp, ff::PrimeField};

use crate::{
    circuit::{table::{Range, RangeCheck}, BaseConfig, CircuitError},
    fieldutils::integer_rep_to_felt,
    tensor::VarTensor,
};

/// Version authenticated by formal GAT settings for this compact range gate.
pub const GAT_RANGE_GADGET_VERSION: u32 = 2;

/// Conservative usable-row reserve for the supported compact range gates.
///
/// Each bounded-range input advice column is queried once at rotation zero by
/// the byte lookup and at rotations -m..=-1 by the end-row gate. The legacy
/// supported bound is m <= 15, so the maximum distinct query count is 16 and
/// Halo2's blinding-factor formula reserves max(3, 16) + 1 = 17 rows. The
/// versioned GAT layout reserves this amount before advice columns are
/// allocated; the configured circuit checks that its actual query count fits.
pub const GAT_RANGE_BLINDING_FACTOR_RESERVE: usize = 17;

/// The byte table used by both slack decompositions.
pub const BYTE_RANGE: Range = (0, 255);

/// Maximum advice cells assigned in one range batch, bounding adapter-side
/// temporary tensors while preserving whole value groups.
const MAX_BATCH_CELLS: usize = 1 << 18;

#[derive(Clone, Debug, Default)]
pub struct GatRangeChecks {
    version: Option<u32>,
    byte_selectors: BTreeMap<(usize, usize), Selector>,
    end_selectors: BTreeMap<(Range, usize, usize), Selector>,
    col_size: usize,
    num_inner_cols: usize,
}

impl GatRangeChecks {
    pub(crate) fn version(&self) -> Option<u32> {
        self.version
    }

    pub(crate) fn byte_selector(&self, block: usize, lane: usize) -> Option<&Selector> {
        self.byte_selectors.get(&(block, lane))
    }

    pub(crate) fn end_selector(
        &self,
        range: Range,
        block: usize,
        lane: usize,
    ) -> Option<&Selector> {
        self.end_selectors.get(&(range, block, lane))
    }

    pub(crate) fn storage_shape(&self) -> Option<(usize, usize)> {
        (self.version == Some(GAT_RANGE_GADGET_VERSION))
            .then_some((self.col_size, self.num_inner_cols))
    }
}

/// One batch of whole `limbs + 1`-row groups. The item indices map to lanes
/// in ascending order. Unused lanes in the last group are zero-filled and
/// have no selectors enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RangeBatch {
    pub start_cursor: usize,
    pub first_item: usize,
    pub item_count: usize,
    pub groups: usize,
    pub cells: usize,
}

/// Shared geometry/live cursor scheduler for the compact bounded-range gate.
/// Each lane holds one value; its byte rows and final reconstruction row stay
/// within a single physical advice block.
#[derive(Clone, Debug)]
pub(crate) struct RangeSchedule {
    cursor: usize,
    next_item: usize,
    item_count: usize,
    col_size: usize,
    lanes: usize,
    limbs: usize,
}

impl RangeSchedule {
    pub(crate) fn new(
        start_cursor: usize,
        item_count: usize,
        col_size: usize,
        lanes: usize,
        limbs: usize,
    ) -> Result<Self, String> {
        if col_size == 0 || lanes == 0 || limbs == 0 {
            return Err("GAT compact range schedule requires positive dimensions".into());
        }
        let rows_per_group = limbs
            .checked_add(1)
            .ok_or_else(|| "GAT compact range row count overflow".to_string())?;
        if rows_per_group > col_size {
            return Err(format!(
                "GAT compact range group of {} rows exceeds native column capacity {}",
                rows_per_group, col_size
            ));
        }
        Ok(Self {
            cursor: start_cursor,
            next_item: 0,
            item_count,
            col_size,
            lanes,
            limbs,
        })
    }

    pub(crate) fn cursor(&self) -> usize {
        self.cursor
    }

    pub(crate) fn next_batch(&mut self) -> Result<Option<RangeBatch>, String> {
        if self.next_item >= self.item_count {
            return Ok(None);
        }
        let rows_per_group = self
            .limbs
            .checked_add(1)
            .ok_or_else(|| "GAT compact range row count overflow".to_string())?;
        let block_size = self
            .col_size
            .checked_mul(self.lanes)
            .ok_or_else(|| "GAT compact range block size overflow".to_string())?;

        let lane_offset = self.cursor % self.lanes;
        if lane_offset != 0 {
            self.cursor = self
                .cursor
                .checked_add(self.lanes - lane_offset)
                .ok_or_else(|| "GAT compact range lane alignment overflow".to_string())?;
        }

        let local = self.cursor % block_size;
        let row = local / self.lanes;
        if row + rows_per_group > self.col_size {
            self.cursor = self
                .cursor
                .checked_add(block_size - local)
                .ok_or_else(|| "GAT compact range block alignment overflow".to_string())?;
        }

        let row = (self.cursor % block_size) / self.lanes;
        let groups_in_block = (self.col_size - row) / rows_per_group;
        if groups_in_block == 0 {
            return Err("GAT compact range scheduler could not fit a group".into());
        }
        let cells_per_group = rows_per_group
            .checked_mul(self.lanes)
            .ok_or_else(|| "GAT compact range group cell count overflow".to_string())?;
        let budget_groups = (MAX_BATCH_CELLS / cells_per_group).max(1);
        let remaining_items = self.item_count - self.next_item;
        let remaining_groups = remaining_items
            .checked_add(self.lanes - 1)
            .ok_or_else(|| "GAT compact range group count overflow".to_string())?
            / self.lanes;
        let groups = groups_in_block.min(budget_groups).min(remaining_groups);
        let cells = groups
            .checked_mul(cells_per_group)
            .ok_or_else(|| "GAT compact range batch cell count overflow".to_string())?;
        let covered_items = groups
            .checked_mul(self.lanes)
            .ok_or_else(|| "GAT compact range covered item count overflow".to_string())?;
        let active_items = covered_items.min(remaining_items);

        let batch = RangeBatch {
            start_cursor: self.cursor,
            first_item: self.next_item,
            item_count: active_items,
            groups,
            cells,
        };
        self.cursor = self
            .cursor
            .checked_add(cells)
            .ok_or_else(|| "GAT compact range cursor overflow".to_string())?;
        self.next_item = self
            .next_item
            .checked_add(active_items)
            .ok_or_else(|| "GAT compact range item cursor overflow".to_string())?;
        Ok(Some(batch))
    }
}

/// Return the byte width required to represent any value in the public
/// interval, enforcing the existing signed range-gadget safety limits.
pub(crate) fn limb_count(range: Range) -> Result<usize, String> {
    if range.0 > range.1 {
        return Err(format!("invalid GAT bounded range {:?}", range));
    }
    let span = range
        .1
        .checked_sub(range.0)
        .ok_or_else(|| "GAT bounded range span overflow".to_string())?;
    if span <= 1023 {
        return Err("compact GAT range gadget is only for spans greater than 1023".into());
    }
    let significant_bits = 128 - (span as u128).leading_zeros() as usize;
    let limbs = significant_bits
        .checked_add(7)
        .ok_or_else(|| "GAT range limb count overflow".to_string())?
        / 8;
    let no_wrap_bits = limbs
        .checked_mul(8)
        .and_then(|bits| bits.checked_add(1))
        .ok_or_else(|| "GAT range limb width overflow".to_string())?;
    if limbs == 0 || limbs > 15 || no_wrap_bits > 253 {
        return Err("GAT bounded range cannot be represented by the canonical limb relation".into());
    }
    Ok(limbs)
}

/// Configure the adapter-owned byte lookups and simple end-row gates for all
/// large authenticated bounds in this GAT settings object.
pub(crate) fn configure(
    meta: &mut ConstraintSystem<Fp>,
    base: &mut BaseConfig<Fp>,
    ranges: &[Range],
    logrows: usize,
) -> Result<(), CircuitError> {
    let mut range_gadget = GatRangeChecks {
        version: Some(GAT_RANGE_GADGET_VERSION),
        ..GatRangeChecks::default()
    };
    let large_ranges = ranges
        .iter()
        .copied()
        .filter(|range| range.1.checked_sub(range.0).is_some_and(|span| span > 1023))
        .collect::<std::collections::BTreeSet<_>>();
    if large_ranges.is_empty() {
        base.gat_range_checks = range_gadget;
        return Ok(());
    }

    for range in &large_ranges {
        limb_count(*range).map_err(CircuitError::DimMismatch)?;
    }
    let byte_table = if let Some(table) = base.range_checks.ranges.get(&BYTE_RANGE) {
        table.clone()
    } else {
        let table = RangeCheck::<Fp>::configure(meta, BYTE_RANGE, logrows);
        base.range_checks.ranges.insert(BYTE_RANGE, table.clone());
        table
    };
    if byte_table.inputs.len() != 1 {
        return Err(CircuitError::DimMismatch(
            "GAT byte range table must occupy exactly one lookup column".into(),
        ));
    }
    let table_column = byte_table.inputs[0];

    let input_a = base
        .custom_gates
        .inputs
        .first()
        .ok_or(CircuitError::UnsupportedOp)?;
    let input_b = base
        .custom_gates
        .inputs
        .get(1)
        .ok_or(CircuitError::UnsupportedOp)?;
    let output = &base.custom_gates.output;
    if !input_a.is_advice() || !input_b.is_advice() || !output.is_advice() {
        return Err(CircuitError::WrongColumnType("GAT range advice".into()));
    }
    if input_a.num_blocks() != input_b.num_blocks()
        || input_a.num_blocks() != output.num_blocks()
        || input_a.num_inner_cols() != input_b.num_inner_cols()
        || input_a.num_inner_cols() != output.num_inner_cols()
        || input_a.col_size() != input_b.col_size()
        || input_a.col_size() != output.col_size()
    {
        return Err(CircuitError::DimMismatch(
            "GAT range advice columns do not share one native storage geometry".into(),
        ));
    }
    let (a_columns, b_columns, output_columns) = match (input_a, input_b, output) {
        (
            VarTensor::Advice { inner: a, .. },
            VarTensor::Advice { inner: b, .. },
            VarTensor::Advice { inner: out, .. },
        ) => (a, b, out),
        _ => return Err(CircuitError::WrongColumnType("GAT range advice".into())),
    };

    for block in 0..output.num_blocks() {
        for lane in 0..output.num_inner_cols() {
            let selector = meta.complex_selector();
            let a_column = a_columns[block][lane];
            let b_column = b_columns[block][lane];
            meta.lookup("gat_bounded_range_lower_byte", |cells| {
                let q = cells.query_selector(selector);
                let byte = cells.query_advice(a_column, Rotation(0));
                vec![(q * byte, table_column)]
            });
            meta.lookup("gat_bounded_range_upper_byte", |cells| {
                let q = cells.query_selector(selector);
                let byte = cells.query_advice(b_column, Rotation(0));
                vec![(q * byte, table_column)]
            });
            range_gadget.byte_selectors.insert((block, lane), selector);
        }
    }

    for range in large_ranges {
        let limbs = limb_count(range).map_err(CircuitError::DimMismatch)?;
        let lower = integer_rep_to_felt::<Fp>(range.0);
        let upper = integer_rep_to_felt::<Fp>(range.1);
        for block in 0..output.num_blocks() {
            for lane in 0..output.num_inner_cols() {
                let selector = meta.selector();
                let a_column = a_columns[block][lane];
                let b_column = b_columns[block][lane];
                let output_column = output_columns[block][lane];
                meta.create_gate("gat_bounded_range_end", |cells| {
                    let value = cells.query_advice(output_column, Rotation(0));
                    let mut lower_reconstruction = Expression::Constant(lower);
                    let mut upper_reconstruction = Expression::Constant(upper);
                    for limb in 0..limbs {
                        let rotation = Rotation(-((limbs - limb) as i32));
                        let coefficient = Expression::Constant(Fp::from_u128(1_u128 << (8 * limb)));
                        lower_reconstruction = lower_reconstruction
                            + cells.query_advice(a_column, rotation) * coefficient.clone();
                        upper_reconstruction = upper_reconstruction
                            - cells.query_advice(b_column, rotation) * coefficient;
                    }
                    let q = cells.query_selector(selector);
                    Constraints::with_selector(
                        q,
                        vec![
                            value.clone() - lower_reconstruction,
                            upper_reconstruction - value,
                        ],
                    )
                });
                range_gadget
                    .end_selectors
                    .insert((range, block, lane), selector);
            }
        }
    }

    range_gadget.col_size = output.col_size();
    range_gadget.num_inner_cols = output.num_inner_cols();
    base.gat_range_checks = range_gadget;
    Ok(())
}
