//! Utility gadgets.

use ff::{Field, PrimeField, PrimeFieldBits};
use halo2_proofs::{
    circuit::{AssignedCell, Cell, Layouter, Value},
    plonk::{Advice, Column, Error, Expression},
};
use std::marker::PhantomData;
use std::ops::Range;

/// A type that has a value at either keygen or proving time.
pub trait FieldValue<F: Field> {
    /// Returns the value of this type.
    fn value(&self) -> Value<&F>;
}

impl<F: Field> FieldValue<F> for Value<F> {
    fn value(&self) -> Value<&F> {
        self.as_ref()
    }
}

impl<F: Field> FieldValue<F> for AssignedCell<F, F> {
    fn value(&self) -> Value<&F> {
        self.value()
    }
}

/// Trait for a variable in the circuit.
pub trait Var<F: Field>: Clone + std::fmt::Debug + From<AssignedCell<F, F>> {
    /// The cell at which this variable was allocated.
    fn cell(&self) -> Cell;

    /// The value allocated to this variable.
    fn value(&self) -> Value<F>;
}

impl<F: Field> Var<F> for AssignedCell<F, F> {
    fn cell(&self) -> Cell {
        self.cell()
    }

    fn value(&self) -> Value<F> {
        self.value().cloned()
    }
}

/// Trait for utilities used across circuits.
pub trait UtilitiesInstructions<F: Field> {
    /// Variable in the circuit.
    type Var: Var<F>;

    /// Load a variable.
    fn load_private(
        &self,
        mut layouter: impl Layouter<F>,
        column: Column<Advice>,
        value: Value<F>,
    ) -> Result<Self::Var, Error> {
        layouter.assign_region(
            || "load private",
            |mut region| {
                region
                    .assign_advice(|| "load private", column, 0, || value)
                    .map(Self::Var::from)
            },
        )
    }
}

/// A type representing a range-constrained field element.
#[derive(Clone, Copy, Debug)]
pub struct RangeConstrained<F: Field, T: FieldValue<F>> {
    inner: T,
    num_bits: usize,
    _phantom: PhantomData<F>,
}

impl<F: Field, T: FieldValue<F>> RangeConstrained<F, T> {
    /// Returns the range-constrained inner type.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Returns the number of bits to which this cell is constrained.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }
}

impl<F: PrimeFieldBits> RangeConstrained<F, Value<F>> {
    /// Constructs a `RangeConstrained<Value<F>>` as a bitrange of the given value.
    pub fn bitrange_of(value: Value<&F>, bitrange: Range<usize>) -> Self {
        let num_bits = bitrange.len();
        Self {
            inner: value.map(|value| bitrange_subset(value, bitrange)),
            num_bits,
            _phantom: PhantomData,
        }
    }
}

impl<F: Field> RangeConstrained<F, AssignedCell<F, F>> {
    /// Constructs a `RangeConstrained<AssignedCell<F, F>>` without verifying that the
    /// cell is correctly range constrained.
    ///
    /// This API only exists to ease with integrating this type into existing circuits,
    /// and will likely be removed in future.
    pub fn unsound_unchecked(cell: AssignedCell<F, F>, num_bits: usize) -> Self {
        Self {
            inner: cell,
            num_bits,
            _phantom: PhantomData,
        }
    }

    /// Extracts the range-constrained value from this range-constrained cell.
    pub fn value(&self) -> RangeConstrained<F, Value<F>> {
        RangeConstrained {
            inner: self.inner.value().copied(),
            num_bits: self.num_bits,
            _phantom: PhantomData,
        }
    }
}

/// Checks that an expression is either 1 or 0.
pub fn bool_check<F: PrimeField>(value: Expression<F>) -> Expression<F> {
    range_check(value, 2)
}

/// If `a` then `b`, else `c`. Returns (a * b) + (1 - a) * c.
///
/// `a` must be a boolean-constrained expression.
pub fn ternary<F: Field>(a: Expression<F>, b: Expression<F>, c: Expression<F>) -> Expression<F> {
    let one_minus_a = Expression::Constant(F::ONE) - a.clone();
    a * b + one_minus_a * c
}

/// Takes a specified subsequence of the little-endian bit representation of a field element.
/// The bits are numbered from 0 for the LSB.
pub fn bitrange_subset<F: PrimeFieldBits>(field_elem: &F, bitrange: Range<usize>) -> F {
    // We can allow a subsequence of length NUM_BITS, because
    // field_elem.to_le_bits() returns canonical bitstrings.
    assert!(bitrange.end <= F::NUM_BITS as usize);

    field_elem
        .to_le_bits()
        .iter()
        .by_vals()
        .skip(bitrange.start)
        .take(bitrange.end - bitrange.start)
        .rev()
        .fold(F::ZERO, |acc, bit| {
            if bit {
                acc.double() + F::ONE
            } else {
                acc.double()
            }
        })
}

/// Check that an expression is in the small range [0..range),
/// i.e. 0 ≤ word < range.
pub fn range_check<F: PrimeField>(word: Expression<F>, range: usize) -> Expression<F> {
    (1..range).fold(word.clone(), |acc, i| {
        acc * (Expression::Constant(F::from(i as u64)) - word.clone())
    })
}

/// Decompose a word `alpha` into `window_num_bits` bits (little-endian)
/// For a window size of `w`, this returns [k_0, ..., k_n] where each `k_i`
/// is a `w`-bit value, and `scalar = k_0 + k_1 * w + k_n * w^n`.
///
/// # Panics
///
/// We are returning a `Vec<u8>` which means the window size is limited to
/// <= 8 bits.
pub fn decompose_word<F: PrimeFieldBits>(
    word: &F,
    word_num_bits: usize,
    window_num_bits: usize,
) -> Vec<u8> {
    assert!(window_num_bits <= 8);

    // Pad bits to multiple of window_num_bits
    let padding = (window_num_bits - (word_num_bits % window_num_bits)) % window_num_bits;
    let bits: Vec<bool> = word
        .to_le_bits()
        .into_iter()
        .take(word_num_bits)
        .chain(std::iter::repeat(false).take(padding))
        .collect();
    assert_eq!(bits.len(), word_num_bits + padding);

    bits.chunks_exact(window_num_bits)
        .map(|chunk| chunk.iter().rev().fold(0, |acc, b| (acc << 1) + (*b as u8)))
        .collect()
}

/// The u64 integer represented by an L-bit little-endian bitstring.
///
/// # Panics
///
/// Panics if the bitstring is longer than 64 bits.
pub fn lebs2ip<const L: usize>(bits: &[bool; L]) -> u64 {
    assert!(L <= 64);
    bits.iter()
        .enumerate()
        .fold(0u64, |acc, (i, b)| acc + if *b { 1 << i } else { 0 })
}

/// The sequence of bits representing a u64 in little-endian order.
///
/// # Panics
///
/// Panics if the expected length of the sequence `NUM_BITS` exceeds
/// 64.
pub fn i2lebsp<const NUM_BITS: usize>(int: u64) -> [bool; NUM_BITS] {
    /// Takes in an FnMut closure and returns a constant-length array with elements of
    /// type `Output`.
    fn gen_const_array<Output: Copy + Default, const LEN: usize>(
        closure: impl FnMut(usize) -> Output,
    ) -> [Output; LEN] {
        let mut ret: [Output; LEN] = [Default::default(); LEN];
        for (bit, val) in ret.iter_mut().zip((0..LEN).map(closure)) {
            *bit = val;
        }
        ret
    }
    assert!(NUM_BITS <= 64);
    gen_const_array(|mask: usize| (int & (1 << mask)) != 0)
}
