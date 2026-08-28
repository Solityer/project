use super::{make_mul_aux, AssignedPoint, EccConfig, MulAux, Point};
use crate::halo2;
use crate::integer::rns::{Integer, Rns};
use crate::integer::{IntegerChip, IntegerInstructions, Range, UnassignedInteger};
use crate::maingate;
use halo2::arithmetic::CurveAffine;
use halo2::circuit::{Layouter, Value};
use halo2::halo2curves::ff::PrimeField;
use halo2::plonk::Error;
use integer::maingate::RegionCtx;
use maingate::{AssignedCondition, MainGate};
use std::collections::BTreeMap;
use std::rc::Rc;

mod add;
mod mul;

/// Constaints elliptic curve operations such as assigment, addition and
/// multiplication
#[derive(Clone, Debug)]
#[allow(clippy::type_complexity)]
pub struct GeneralEccChip<
    Emulated: CurveAffine,
    N: PrimeField,
    const NUMBER_OF_LIMBS: usize,
    const BIT_LEN_LIMB: usize,
> {
    /// `IntegerChip` for the base field of the EC
    base_field_chip: IntegerChip<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    /// `IntegerChip` for the scalar field of the EC
    scalar_field_chip: IntegerChip<Emulated::Scalar, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    /// Auxiliary point for optimized multiplication algorithm
    aux_generator: Option<(
        AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        Value<Emulated>,
    )>,
    /// Auxiliary points for optimized multiplication for each (window_size,
    /// n_pairs) pairs
    aux_registry:
        BTreeMap<(usize, usize), AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>>,
}

impl<
        Emulated: CurveAffine,
        N: PrimeField,
        const NUMBER_OF_LIMBS: usize,
        const BIT_LEN_LIMB: usize,
    > GeneralEccChip<Emulated, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>
{
    /// Residue numeral system
    /// Used to emulate the base field `Emulated::Base` and the scalar
    /// field `Emulated::Scalar` over the native field `N`
    pub fn rns() -> (
        Rns<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        Rns<Emulated::Scalar, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) {
        (Rns::construct(), Rns::construct())
    }

    /// Return `GeneralEccChip` from `EccConfig`
    pub fn new(config: EccConfig) -> Self {
        let (rns_base_field, rns_scalar_field) = Self::rns();
        let integer_config = config.integer_chip_config();
        Self {
            base_field_chip: IntegerChip::new(integer_config.clone(), Rc::new(rns_base_field)),
            scalar_field_chip: IntegerChip::new(integer_config, Rc::new(rns_scalar_field)),
            aux_generator: None,
            aux_registry: BTreeMap::new(),
        }
    }

    /// Residue numeral system for the base field of the curve
    /// Return new refence for chips' rns base field
    pub fn rns_base(&self) -> Rc<Rns<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>> {
        self.base_field_chip.rns()
    }

    /// Residue numeral system for the scalar field of the curve
    /// Return new refence for chips' rns scalar field
    pub fn rns_scalar(&self) -> Rc<Rns<Emulated::Scalar, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>> {
        self.scalar_field_chip.rns()
    }

    /// Assign Rns base for chip
    pub fn new_unassigned_base(
        &self,
        e: Value<Emulated::Base>,
    ) -> UnassignedInteger<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB> {
        e.map(|e| Integer::from_fe(e, self.rns_base())).into()
    }

    /// Assign Rns Scalar for chip
    pub fn new_unassigned_scalar(
        &self,
        e: Value<Emulated::Scalar>,
    ) -> UnassignedInteger<Emulated::Scalar, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB> {
        e.map(|e| Integer::from_fe(e, self.rns_scalar())).into()
    }

    /// Return `IntegerChip` for the base field of the EC
    pub fn base_field_chip(
        &self,
    ) -> &IntegerChip<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB> {
        &self.base_field_chip
    }

    /// Return `IntegerChip` for the scalar field of the EC
    pub fn scalar_field_chip(
        &self,
    ) -> &IntegerChip<Emulated::Scalar, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB> {
        &self.scalar_field_chip
    }

    /// Return `Maingate` of the `GeneralEccChip`
    pub fn main_gate(&self) -> &MainGate<N> {
        self.base_field_chip.main_gate()
    }

    /// Returns a `Point` (Rns representation) from a point in the emulated EC
    pub fn to_rns_point(
        &self,
        point: Emulated,
    ) -> Point<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB> {
        let coords = point.coordinates();
        // disallow point of infinity
        // it will not pass assing point enforcement
        let coords = coords.unwrap();

        let x = Integer::from_fe(*coords.x(), self.rns_base());
        let y = Integer::from_fe(*coords.y(), self.rns_base());
        Point { x, y }
    }

    /// Returns emulated EC constant $b$
    fn parameter_b(&self) -> Integer<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB> {
        Integer::from_fe(Emulated::b(), self.rns_base())
    }

    /// Auxilary point for optimized multiplication algorithm
    fn get_mul_aux(
        &self,
        window_size: usize,
        number_of_pairs: usize,
    ) -> Result<MulAux<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        // Gets chips' aux generator
        let to_add = match self.aux_generator.clone() {
            Some((assigned, _)) => Ok(assigned),
            None => Err(Error::Synthesis),
        }?;
        let to_sub = match self.aux_registry.get(&(window_size, number_of_pairs)) {
            Some(aux) => Ok(aux.clone()),
            None => Err(Error::Synthesis),
        }?;
        // to_add the equivalent of AuxInit and to_sub AuxFin
        // see https://hackmd.io/ncuKqRXzR-Cw-Au2fGzsMg?view
        Ok(MulAux::new(to_add, to_sub))
    }
}

impl<
        Emulated: CurveAffine,
        N: PrimeField,
        const NUMBER_OF_LIMBS: usize,
        const BIT_LEN_LIMB: usize,
    > GeneralEccChip<Emulated, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>
{
    /// Expose `AssignedPoint` as Public Input
    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<N>,
        point: AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        offset: usize,
    ) -> Result<(), Error> {
        use integer::maingate::MainGateInstructions;
        let main_gate = self.main_gate();

        let mut offset = offset;
        for limb in point.x().limbs().iter() {
            main_gate.expose_public(layouter.namespace(|| "x coords"), limb.into(), offset)?;
            offset += 1;
        }
        for limb in point.y().limbs().iter() {
            main_gate.expose_public(layouter.namespace(|| "y coords"), limb.into(), offset)?;
            offset += 1;
        }
        Ok(())
    }

    /// Takes `Point` and assign its coordiantes as constant
    /// Returned as `AssignedPoint`
    pub fn assign_constant(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        point: Emulated,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let coords = point.coordinates();
        // disallow point of infinity
        let coords = coords.unwrap();
        let base_field_chip = self.base_field_chip();
        let x = base_field_chip.assign_constant(ctx, *coords.x())?;
        let y = base_field_chip.assign_constant(ctx, *coords.y())?;

        Ok(AssignedPoint::new(x, y))
    }

    /// Takes `Point` of the EC and returns it as `AssignedPoint`
    pub fn assign_point(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        point: Value<Emulated>,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let integer_chip = self.base_field_chip();

        let point = point.map(|point| self.to_rns_point(point));
        let (x, y) = point
            .map(|point| (point.x().clone(), point.y().clone()))
            .unzip();

        let x = integer_chip.assign_integer(ctx, x.into(), Range::Remainder)?;
        let y = integer_chip.assign_integer(ctx, y.into(), Range::Remainder)?;

        let point = AssignedPoint::new(x, y);
        self.assert_is_on_curve(ctx, &point)?;
        Ok(point)
    }

    /// Assigns the auxiliary generator point
    pub fn assign_aux_generator(
        &mut self,
        ctx: &mut RegionCtx<'_, N>,
        aux_generator: Value<Emulated>,
    ) -> Result<(), Error> {
        let aux_generator_assigned = self.assign_point(ctx, aux_generator)?;
        self.aux_generator = Some((aux_generator_assigned, aux_generator));
        Ok(())
    }

    /// Assigns multiplication auxiliary point for a pair of (window_size,
    /// n_pairs)
    pub fn assign_aux(
        &mut self,
        ctx: &mut RegionCtx<'_, N>,
        window_size: usize,
        number_of_pairs: usize,
    ) -> Result<(), Error> {
        match self.aux_generator {
            Some((_, point)) => {
                let aux = point.map(|point| make_mul_aux(point, window_size, number_of_pairs));
                let aux = self.assign_point(ctx, aux)?;
                self.aux_registry
                    .insert((window_size, number_of_pairs), aux);
                Ok(())
            }
            // aux generator is not assigned yet
            None => Err(Error::Synthesis),
        }
    }

    /// Constraints to ensure `AssignedPoint` is on curve
    pub fn assert_is_on_curve(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        point: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let integer_chip = self.base_field_chip();

        let y_square = &integer_chip.square(ctx, point.y())?;
        let x_square = &integer_chip.square(ctx, point.x())?;
        let x_cube = &integer_chip.mul(ctx, point.x(), x_square)?;
        let x_cube_b = &integer_chip.add_constant(ctx, x_cube, &self.parameter_b())?;
        integer_chip.assert_equal(ctx, x_cube_b, y_square)?;
        Ok(())
    }

    /// Constraints assert two `AssignedPoint`s are equal
    pub fn assert_equal(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        p0: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        p1: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<(), Error> {
        let integer_chip = self.base_field_chip();
        integer_chip.assert_equal(ctx, p0.x(), p1.x())?;
        integer_chip.assert_equal(ctx, p0.y(), p1.y())
    }

    /// Selects between 2 `AssignedPoint` determined by an `AssignedCondition`
    pub fn select(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        c: &AssignedCondition<N>,
        p1: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        p2: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let integer_chip = self.base_field_chip();
        let x = integer_chip.select(ctx, p1.x(), p2.x(), c)?;
        let y = integer_chip.select(ctx, p1.y(), p2.y(), c)?;
        Ok(AssignedPoint::new(x, y))
    }

    /// Selects between an `AssignedPoint` and a point on the EC `Emulated`
    /// determined by an `AssignedCondition`
    pub fn select_or_assign(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        c: &AssignedCondition<N>,
        p1: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        p2: Emulated,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let integer_chip = self.base_field_chip();
        let p2 = self.to_rns_point(p2);
        let x = integer_chip.select_or_assign(ctx, p1.x(), p2.x(), c)?;
        let y = integer_chip.select_or_assign(ctx, p1.y(), p2.y(), c)?;
        Ok(AssignedPoint::new(x, y))
    }

    /// Normalizes an `AssignedPoint` by reducing each of its coordinates
    pub fn normalize(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        point: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let integer_chip = self.base_field_chip();
        let x = integer_chip.reduce(ctx, point.x())?;
        let y = integer_chip.reduce(ctx, point.y())?;
        Ok(AssignedPoint::new(x, y))
    }

    /// Adds 2 distinct `AssignedPoints`
    pub fn add(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        p0: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        p1: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        // guarantees that p0 != p1 or p0 != p1
        // so that we can use unsafe addition formula which assumes operands are not
        // equal addition to that we strictly disallow addition result to be
        // point of infinity
        self.base_field_chip()
            .assert_not_equal(ctx, p0.x(), p1.x())?;

        self._add_incomplete_unsafe(ctx, p0, p1)
    }

    /// Doubles an `AssignedPoint`
    pub fn double(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        p: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        // point must be asserted to be in curve and not infinity
        self._double_incomplete(ctx, p)
    }

    /// Given an `AssignedPoint` $P$ computes P * 2^logn
    pub fn double_n(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        p: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        logn: usize,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let mut acc = p.clone();
        for _ in 0..logn {
            acc = self._double_incomplete(ctx, &acc)?;
        }
        Ok(acc)
    }

    /// Wrapper for `_ladder_incomplete`
    /// Given 2 `AssignedPoint` $P$ and $Q$ efficiently computes $2*P + Q$
    pub fn ladder(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        to_double: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
        to_add: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        self._ladder_incomplete(ctx, to_double, to_add)
    }

    /// Returns the negative or inverse of an `AssignedPoint`
    pub fn neg(
        &self,
        ctx: &mut RegionCtx<'_, N>,
        p: &AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedPoint<Emulated::Base, N, NUMBER_OF_LIMBS, BIT_LEN_LIMB>, Error> {
        let integer_chip = self.base_field_chip();
        let y_neg = integer_chip.neg(ctx, p.y())?;
        Ok(AssignedPoint::new(p.x().clone(), y_neg))
    }
}
