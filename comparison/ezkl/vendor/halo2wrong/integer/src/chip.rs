use std::rc::Rc;

use super::{AssignedInteger, AssignedLimb, UnassignedInteger};
use crate::instructions::{IntegerInstructions, Range};
use crate::rns::{Common, Integer, Rns};
use halo2::halo2curves::ff::PrimeField;
use halo2::plonk::Error;
use maingate::{halo2, AssignedCondition, AssignedValue, MainGateInstructions, RegionCtx};
use maingate::{MainGate, MainGateConfig};
use maingate::{RangeChip, RangeConfig};

mod add;
mod assert_in_field;
mod assert_not_zero;
mod assert_zero;
mod assign;
mod div;
mod invert;
mod mul;
mod reduce;
mod square;

/// Configuration for [`IntegerChip`]
#[derive(Clone, Debug)]
pub struct IntegerConfig {
    /// Configuration for [`RangeChip`]
    range_config: RangeConfig,
    /// Configuration for [`MainGate`]
    main_gate_config: MainGateConfig,
}

impl IntegerConfig {
    // Creates a new [`IntegerConfig`] from a [`RangeConfig`] and a
    /// [`MainGateConfig`]
    pub fn new(range_config: RangeConfig, main_gate_config: MainGateConfig) -> Self {
        Self {
            range_config,
            main_gate_config,
        }
    }
}

/// Chip for integer instructions
#[derive(Clone, Debug)]
pub struct IntegerChip<
    W: PrimeField,
    N: PrimeField,
    const NUMBER_OF_LIMBS: usize,
    const BIT_LEN_LIMB: usize,
> {
    /// RangeChip
    range_chip: RangeChip<N>,
    /// MainGate
    main_gate: MainGate<N>,
    /// Residue number system used to represent the integers
    rns: Rc<Rns<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>>,
}

impl<W: PrimeField, N: PrimeField, const NUMBER_OF_LIMBS: usize, const BIT_LEN_LIMB: usize>
    IntegerChip<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>
{
    fn sublimb_bit_len() -> usize {
        let number_of_lookup_limbs = 4;
        assert!(BIT_LEN_LIMB % number_of_lookup_limbs == 0);
        BIT_LEN_LIMB / number_of_lookup_limbs
    }

    /// Creates a new [`AssignedInteger`] from its limb representation and its
    /// native value
    pub(crate) fn new_assigned_integer(
        &self,
        limbs: &[AssignedLimb<N>; NUMBER_OF_LIMBS],
        native_value: AssignedValue<N>,
    ) -> AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB> {
        AssignedInteger::new(Rc::clone(&self.rns), limbs, native_value)
    }
}

impl<W: PrimeField, N: PrimeField, const NUMBER_OF_LIMBS: usize, const BIT_LEN_LIMB: usize>
    IntegerInstructions<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>
    for IntegerChip<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>
{
    fn reduce_external<T: PrimeField>(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        // TODO: external integer might have different parameter settings
        a: &AssignedInteger<T, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let to_be_reduced = self.new_assigned_integer(a.limbs(), a.native().clone());
        self.reduce(ctx, &to_be_reduced)
    }

    fn assign_integer(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        integer: UnassignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        range: Range,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        self.assign_integer_generic(ctx, integer, range)
    }

    fn assign_constant(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        integer: W,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        self.assign_constant_generic(ctx, integer)
    }

    fn decompose(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        integer: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<Vec<AssignedCondition<N>>, Error> {
        self.assert_in_field(ctx, integer)?;

        let main_gate = self.main_gate();

        let mut decomposed = Vec::new();
        for idx in 0..NUMBER_OF_LIMBS {
            let number_of_bits = if idx == NUMBER_OF_LIMBS - 1 {
                self.rns.wrong_modulus.bits() as usize % BIT_LEN_LIMB
            } else {
                BIT_LEN_LIMB
            };
            let decomposed_limb = main_gate.to_bits(ctx, integer.limb(idx), number_of_bits)?;
            decomposed.extend(decomposed_limb);
        }

        assert_eq!(decomposed.len(), self.rns.wrong_modulus.bits() as usize);

        Ok(decomposed)
    }

    fn add(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let (a, b) = (
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, a)?,
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, b)?,
        );
        self.add_generic(ctx, a, b)
    }

    fn add_add(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b_0: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b_1: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let (a, b_0, b_1) = (
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, a)?,
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, b_0)?,
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, b_1)?,
        );
        self.add_add_generic(ctx, a, b_0, b_1)
    }

    fn add_constant(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &Integer<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let a = &self.reduce_if_limb_values_exceeds_unreduced(ctx, a)?;
        self.add_constant_generic(ctx, a, b)
    }

    fn mul2(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        self.mul2_generic(ctx, a)
    }

    fn mul3(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        self.mul3_generic(ctx, a)
    }

    fn sub(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let (a, b) = (
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, a)?,
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, b)?,
        );
        self.sub_generic(ctx, a, b)
    }

    fn sub_sub(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b_0: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b_1: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let (a, b_0, b_1) = (
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, a)?,
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, b_0)?,
            &self.reduce_if_limb_values_exceeds_unreduced(ctx, b_1)?,
        );
        self.sub_sub_generic(ctx, a, b_0, b_1)
    }

    fn neg(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let a = &self.reduce_if_limb_values_exceeds_unreduced(ctx, a)?;
        self.neg_generic(ctx, a)
    }

    fn mul(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let (a, b) = (
            &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?,
            &self.reduce_if_limb_values_exceeds_reduced(ctx, b)?,
        );
        let (a, b) = (
            &self.reduce_if_max_operand_value_exceeds(ctx, a)?,
            &self.reduce_if_max_operand_value_exceeds(ctx, b)?,
        );
        self.mul_generic(ctx, a, b)
    }

    fn mul_constant(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &Integer<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let a = &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?;
        let a = &self.reduce_if_max_operand_value_exceeds(ctx, a)?;
        self.mul_constant_generic(ctx, a, b)
    }

    fn mul_into_one(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let (a, b) = (
            &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?,
            &self.reduce_if_limb_values_exceeds_reduced(ctx, b)?,
        );
        let (a, b) = (
            &self.reduce_if_max_operand_value_exceeds(ctx, a)?,
            &self.reduce_if_max_operand_value_exceeds(ctx, b)?,
        );
        self.mul_into_one_generic(ctx, a, b)
    }

    fn square(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let a = &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?;
        let a = &self.reduce_if_max_operand_value_exceeds(ctx, a)?;
        self.square_generic(ctx, a)
    }

    fn div(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<
        (
            AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
            AssignedCondition<N>,
        ),
        Error,
    > {
        let (a, b) = (
            &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?,
            &self.reduce_if_limb_values_exceeds_reduced(ctx, b)?,
        );
        let (a, b) = (
            &self.reduce_if_max_operand_value_exceeds(ctx, a)?,
            &self.reduce_if_max_operand_value_exceeds(ctx, b)?,
        );
        self.div_generic(ctx, a, b)
    }

    fn div_incomplete(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let (a, b) = (
            &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?,
            &self.reduce_if_limb_values_exceeds_reduced(ctx, b)?,
        );
        let (a, b) = (
            &self.reduce_if_max_operand_value_exceeds(ctx, a)?,
            &self.reduce_if_max_operand_value_exceeds(ctx, b)?,
        );
        self.div_incomplete_generic(ctx, a, b)
    }

    fn invert(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<
        (
            AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
            AssignedCondition<N>,
        ),
        Error,
    > {
        let a = &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?;
        let a = &self.reduce_if_max_operand_value_exceeds(ctx, a)?;
        self.invert_generic(ctx, a)
    }

    fn invert_incomplete(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let a = &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?;
        let a = &self.reduce_if_max_operand_value_exceeds(ctx, a)?;
        self.invert_incomplete_generic(ctx, a)
    }

    fn reduce(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        self.reduce_generic(ctx, a)
    }

    fn assert_equal(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let c = &self.sub(ctx, a, b)?;
        self.assert_zero_generic(ctx, c)?;
        Ok(())
    }

    fn assert_strict_equal(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let main_gate = self.main_gate();
        for idx in 0..NUMBER_OF_LIMBS {
            main_gate.assert_equal(ctx, a.limb(idx), b.limb(idx))?;
        }
        Ok(())
    }

    fn assert_not_equal(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let c = &self.sub(ctx, a, b)?;
        self.assert_not_zero_generic(ctx, c)?;
        Ok(())
    }

    fn assert_not_zero(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let a = &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?;
        let a = &self.reduce_if_max_operand_value_exceeds(ctx, a)?;
        self.assert_not_zero_generic(ctx, a)?;
        Ok(())
    }

    fn assert_zero(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let a = &self.reduce_if_max_operand_value_exceeds(ctx, a)?;
        let main_gate = self.main_gate();
        for limb in a.limbs() {
            main_gate.assert_zero(ctx, &limb.into())?;
        }
        Ok(())
    }

    fn assert_strict_zero(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let main_gate = self.main_gate();
        for limb in a.limbs() {
            main_gate.assert_zero(ctx, &limb.into())?;
        }
        Ok(())
    }

    fn assert_strict_one(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let main_gate = self.main_gate();
        for i in 1..NUMBER_OF_LIMBS {
            main_gate.assert_zero(ctx, a.limb(i))?;
        }
        main_gate.assert_one(ctx, a.limb(0))
    }

    fn assert_strict_bit(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let main_gate = self.main_gate();
        for i in 1..NUMBER_OF_LIMBS {
            main_gate.assert_zero(ctx, a.limb(i))?;
        }
        main_gate.assert_bit(ctx, a.limb(0))
    }

    fn select(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        cond: &AssignedCondition<N>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let main_gate = self.main_gate();

        let mut limbs: Vec<AssignedLimb<N>> = Vec::with_capacity(NUMBER_OF_LIMBS);
        for i in 0..NUMBER_OF_LIMBS {
            let res = main_gate.select(ctx, a.limb(i), b.limb(i), cond)?;

            let max_val = if a.limbs[i].max_val > b.limbs[i].max_val {
                a.limbs[i].max_val.clone()
            } else {
                b.limbs[i].max_val.clone()
            };

            limbs.push(AssignedLimb::from(res, max_val));
        }

        let native_value = main_gate.select(ctx, a.native(), b.native(), cond)?;

        Ok(self.new_assigned_integer(&limbs.try_into().unwrap(), native_value))
    }

    fn select_or_assign(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        b: &Integer<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        cond: &AssignedCondition<N>,
    ) -> Result<AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let main_gate = self.main_gate();

        let mut limbs: Vec<AssignedLimb<N>> = Vec::with_capacity(NUMBER_OF_LIMBS);
        for i in 0..NUMBER_OF_LIMBS {
            let b_limb = b.limb(i);

            let res = main_gate.select_or_assign(ctx, a.limb(i), b_limb.fe(), cond)?;

            // here we assume given constant is always in field
            let max_val = a.limbs[i].max_val();
            limbs.push(AssignedLimb::from(res, max_val));
        }

        let native_value = main_gate.select_or_assign(ctx, a.native(), b.native(), cond)?;

        Ok(self.new_assigned_integer(&limbs.try_into().unwrap(), native_value))
    }

    fn assert_in_field(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let a = &self.reduce_if_limb_values_exceeds_reduced(ctx, a)?;
        let a = &self.reduce_if_max_operand_value_exceeds(ctx, a)?;
        self.assert_in_field_generic(ctx, a)
    }

    fn sign(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        a: &AssignedInteger<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedCondition<N>, Error> {
        self.assert_in_field(ctx, a)?;
        self.main_gate().sign(ctx, a.limb(0))
    }
}

impl<W: PrimeField, N: PrimeField, const NUMBER_OF_LIMBS: usize, const BIT_LEN_LIMB: usize>
    IntegerChip<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>
{
    /// Create new ['IntegerChip'] with the configuration and a shared [`Rns`]
    pub fn new(config: IntegerConfig, rns: Rc<Rns<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>>) -> Self {
        IntegerChip {
            range_chip: RangeChip::new(config.range_config),
            main_gate: MainGate::new(config.main_gate_config),
            rns,
        }
    }

    /// Getter for [`RangeChip`]
    pub fn range_chip(&self) -> &RangeChip<N> {
        &self.range_chip
    }

    /// Getter for [`MainGate`]
    pub fn main_gate(&self) -> &MainGate<N> {
        &self.main_gate
    }

    /// Getter for [`Rns`]
    pub fn rns(&self) -> Rc<Rns<W, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>> {
        Rc::clone(&self.rns)
    }
}
