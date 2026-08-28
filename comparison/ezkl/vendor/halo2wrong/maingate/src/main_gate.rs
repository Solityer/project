//! `main_gate` is a five width stardart like PLONK gate that constrains the
//! equation below
//!
//! q_a * a + q_b * b + q_c * c + q_d * d + q_e * e +
//! q_mul_ab * a * b +
//! q_mul_cd * c * d +
//! q_e_next * e +
//! public_input +
//! q_constant = 0

use crate::halo2::circuit::{Chip, Layouter};
use crate::halo2::halo2curves::ff::PrimeField;
use crate::halo2::plonk::{Advice, Column, ConstraintSystem, Error, Fixed, Instance};
use crate::halo2::poly::Rotation;
use crate::instructions::{CombinationOptionCommon, MainGateInstructions, Term};
use crate::{AssignedCondition, AssignedValue};
use halo2wrong::halo2::circuit::Value;
use halo2wrong::RegionCtx;
use std::{iter, marker::PhantomData};

const WIDTH: usize = 5;

/// `ColumnTags` is an helper to find special columns that are frequently used
/// across gates
pub trait ColumnTags<Column> {
    /// Next row accumulator
    fn next() -> Column;
    /// First column
    fn first() -> Column;
    /// Index that last term should in linear combination
    fn last_term_index() -> usize;
}

/// Enumerates columns of the main gate
#[derive(Debug)]
pub enum MainGateColumn {
    /// A
    A = 0,
    /// B
    B = 1,
    /// C
    C = 2,
    /// D
    D = 3,
    /// E
    E = 4,
}

impl ColumnTags<MainGateColumn> for MainGateColumn {
    fn first() -> Self {
        MainGateColumn::A
    }

    fn next() -> Self {
        MainGateColumn::E
    }

    fn last_term_index() -> usize {
        Self::first() as usize
    }
}

/// Config defines fixed and witness columns of the main gate
#[derive(Clone, Debug)]
pub struct MainGateConfig {
    pub(crate) a: Column<Advice>,
    pub(crate) b: Column<Advice>,
    pub(crate) c: Column<Advice>,
    pub(crate) d: Column<Advice>,
    pub(crate) e: Column<Advice>,

    pub(crate) sa: Column<Fixed>,
    pub(crate) sb: Column<Fixed>,
    pub(crate) sc: Column<Fixed>,
    pub(crate) sd: Column<Fixed>,
    pub(crate) se: Column<Fixed>,

    pub(crate) se_next: Column<Fixed>,

    pub(crate) s_mul_ab: Column<Fixed>,
    pub(crate) s_mul_cd: Column<Fixed>,

    pub(crate) s_constant: Column<Fixed>,
    /// column for public input
    pub instance: Column<Instance>,
}

impl MainGateConfig {
    /// Returns advice columns of `MainGateConfig`
    pub fn advices(&self) -> [Column<Advice>; WIDTH] {
        [self.a, self.b, self.c, self.d, self.e]
    }
    /// Returns fixed columns of `MainGateConfig`
    pub fn fixed(&self) -> [Column<Fixed>; WIDTH] {
        [self.sa, self.sb, self.sc, self.sd, self.se]
    }
    /// Returns instance column of `MainGateConfig`
    pub fn instance(&self) -> Column<Instance> {
        self.instance
    }
}

/// MainGate implements instructions with [`MainGateConfig`]
#[derive(Clone, Debug)]
pub struct MainGate<F: PrimeField> {
    config: MainGateConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Chip<F> for MainGate<F> {
    type Config = MainGateConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

/// Additional combination customisations for this gate with two multiplication
#[derive(Clone, Debug)]
pub enum CombinationOption<F: PrimeField> {
    /// Wrapper for common combination options
    Common(CombinationOptionCommon<F>),
    /// Activates both of the multiplication gate
    OneLinerDoubleMul(F),
    /// Activates both multiplication gate and combines the result to the next
    /// row
    CombineToNextDoubleMul(F),
}

impl<F: PrimeField> From<CombinationOptionCommon<F>> for CombinationOption<F> {
    fn from(option: CombinationOptionCommon<F>) -> Self {
        CombinationOption::Common(option)
    }
}

impl<F: PrimeField> MainGateInstructions<F, WIDTH> for MainGate<F> {
    type CombinationOption = CombinationOption<F>;
    type MainGateColumn = MainGateColumn;

    fn expose_public(
        &self,
        mut layouter: impl Layouter<F>,
        value: AssignedValue<F>,
        row: usize,
    ) -> Result<(), Error> {
        let config = self.config();
        layouter.constrain_instance(value.cell(), config.instance, row)
    }

    fn assign_to_column(
        &self,
        ctx: &mut RegionCtx<'_, F>,
        unassigned: Value<F>,
        column: MainGateColumn,
    ) -> Result<AssignedValue<F>, Error> {
        let column = match column {
            MainGateColumn::A => self.config.a,
            MainGateColumn::B => self.config.b,
            MainGateColumn::C => self.config.c,
            MainGateColumn::D => self.config.d,
            MainGateColumn::E => self.config.e,
        };
        let cell = ctx.assign_advice(|| "assign value", column, unassigned)?;
        // proceed to the next row
        self.no_operation(ctx)?;
        Ok(cell)
    }

    fn sub_sub_with_constant(
        &self,
        ctx: &mut RegionCtx<'_, F>,
        a: &AssignedValue<F>,
        b_0: &AssignedValue<F>,
        b_1: &AssignedValue<F>,
        constant: F,
    ) -> Result<AssignedValue<F>, Error> {
        let c = a
            .value()
            .zip(b_0.value())
            .zip(b_1.value())
            .map(|((a, b_0), b_1)| *a - *b_0 - *b_1 + constant);

        Ok(self
            .apply(
                ctx,
                [
                    Term::assigned_to_add(a),
                    Term::assigned_to_sub(b_0),
                    Term::assigned_to_sub(b_1),
                    Term::unassigned_to_sub(c),
                ],
                constant,
                CombinationOptionCommon::OneLinerAdd.into(),
            )?
            .swap_remove(3))
    }

    fn select(
        &self,
        ctx: &mut RegionCtx<'_, F>,
        a: &AssignedValue<F>,
        b: &AssignedValue<F>,
        cond: &AssignedCondition<F>,
    ) -> Result<AssignedValue<F>, Error> {
        // We should satisfy the equation below with bit asserted condition flag
        // c (a-b) + b - res = 0
        // cond * a - cond * b + b - res = 0

        // Witness layout:
        // | A   | B   | C | D   | E  |
        // | --- | --- | - | --- | ---|
        // | c   | a   | c | b   | res|

        let res = a
            .value()
            .zip(b.value())
            .zip(cond.value())
            .map(|((a, b), cond)| {
                if *cond == F::ONE {
                    *a
                } else {
                    assert_eq!(*cond, F::ZERO);
                    *b
                }
            });

        let mut assigned = self.apply(
            ctx,
            [
                Term::assigned_to_mul(cond),
                Term::assigned_to_mul(a),
                Term::assigned_to_mul(cond),
                Term::assigned_to_add(b),
                Term::unassigned_to_sub(res),
            ],
            F::ZERO,
            CombinationOption::OneLinerDoubleMul(-F::ONE),
        )?;
        ctx.constrain_equal(assigned[0].cell(), assigned[2].cell())?;
        Ok(assigned.swap_remove(4))
    }

    fn select_or_assign(
        &self,
        ctx: &mut RegionCtx<'_, F>,
        a: &AssignedValue<F>,
        b: F,
        cond: &AssignedCondition<F>,
    ) -> Result<AssignedValue<F>, Error> {
        // We should satisfy the equation below with bit asserted condition flag
        // c (a-b_constant) + b_constant - res = 0

        // Witness layout:
        // | A   | B   | C | D   |
        // | --- | --- | - | --- |
        // | dif | a   | - | -   |
        // | c   | dif | - | res |

        let (dif, res) = a
            .value()
            .zip(cond.value())
            .map(|(a, cond)| {
                (
                    *a - b,
                    if *cond == F::ONE {
                        *a
                    } else {
                        assert_eq!(*cond, F::ZERO);
                        b
                    },
                )
            })
            .unzip();

        // a - b - dif = 0
        let dif = &self.apply(
            ctx,
            [Term::assigned_to_add(a), Term::unassigned_to_sub(dif)],
            -b,
            CombinationOptionCommon::OneLinerAdd.into(),
        )?[1];

        // cond * dif + b + a_or_b  = 0
        let res = self
            .apply(
                ctx,
                [
                    Term::assigned_to_mul(dif),
                    Term::assigned_to_mul(cond),
                    Term::unassigned_to_sub(res),
                ],
                b,
                CombinationOptionCommon::OneLinerMul.into(),
            )?
            .swap_remove(2);

        Ok(res)
    }

    fn apply<'t>(
        &self,
        ctx: &mut RegionCtx<'_, F>,
        terms: impl IntoIterator<Item = Term<'t, F>> + 't,
        constant: F,
        option: CombinationOption<F>,
    ) -> Result<Vec<AssignedValue<F>>, Error> {
        let terms = terms.into_iter().collect::<Vec<_>>();
        debug_assert!(terms.len() <= WIDTH);

        let assigned = [
            self.config.a,
            self.config.b,
            self.config.c,
            self.config.d,
            self.config.e,
        ]
        .into_iter()
        .zip([
            self.config.sa,
            self.config.sb,
            self.config.sc,
            self.config.sd,
            self.config.se,
        ])
        .zip(terms.iter().chain(iter::repeat(&Term::Zero)))
        .enumerate()
        .map(|(idx, ((coeff, base), term))| {
            let assigned = ctx.assign_advice(|| format!("coeff_{idx}"), coeff, term.coeff())?;
            ctx.assign_fixed(|| format!("base_{idx}"), base, term.base())?;
            Ok(assigned)
        })
        .collect::<Result<Vec<_>, Error>>()?;

        ctx.assign_fixed(|| "s_constant", self.config.s_constant, constant)?;

        // Given specific option configure multiplication and rotation gates
        match option {
            CombinationOption::Common(option) => match option {
                // q_a * a + q_b * b + q_c * c + q_d * d + q_e * e +
                // a * b +
                // q_e_next * e +
                // q_constant = 0
                CombinationOptionCommon::CombineToNextMul(next) => {
                    ctx.assign_fixed(|| "s_mul_ab", self.config.s_mul_ab, F::ONE)?;
                    ctx.assign_fixed(|| "s_mul_cd", self.config.s_mul_cd, F::ZERO)?;
                    ctx.assign_fixed(|| "se_next", self.config.se_next, next)?;
                }

                // q_a * a + q_b * b + q_c * c + q_d * d + q_e * e +
                // q_mul_ab * a * b +
                // q_e_next * e +
                // q_constant = 0
                CombinationOptionCommon::CombineToNextScaleMul(next, n) => {
                    ctx.assign_fixed(|| "s_mul_ab", self.config.s_mul_ab, n)?;
                    ctx.assign_fixed(|| "s_mul_cd", self.config.s_mul_cd, F::ZERO)?;
                    ctx.assign_fixed(|| "se_next", self.config.se_next, next)?;
                }

                // q_a * a + q_b * b + q_c * c + q_d * d + q_e * e +
                // q_e_next * e +
                // q_constant = 0
                CombinationOptionCommon::CombineToNextAdd(next) => {
                    ctx.assign_fixed(|| "s_mul_ab", self.config.s_mul_ab, F::ZERO)?;
                    ctx.assign_fixed(|| "s_mul_cd", self.config.s_mul_cd, F::ZERO)?;
                    ctx.assign_fixed(|| "se_next", self.config.se_next, next)?;
                }

                // q_a * a + q_b * b + q_c * c + q_d * d + q_e * e +
                // q_mul_ab * a * b +
                // q_constant = 0
                CombinationOptionCommon::OneLinerMul => {
                    ctx.assign_fixed(|| "s_mul_ab", self.config.s_mul_ab, F::ONE)?;
                    ctx.assign_fixed(|| "s_mul_cd", self.config.s_mul_cd, F::ZERO)?;
                    ctx.assign_fixed(|| "se_next", self.config.se_next, F::ZERO)?;
                }

                // q_a * a + q_b * b + q_c * c + q_d * d + q_e * e +
                // q_constant = 0
                CombinationOptionCommon::OneLinerAdd => {
                    ctx.assign_fixed(|| "se_next", self.config.se_next, F::ZERO)?;
                    ctx.assign_fixed(|| "s_mul_ab", self.config.s_mul_ab, F::ZERO)?;
                    ctx.assign_fixed(|| "s_mul_cd", self.config.s_mul_cd, F::ZERO)?;
                }
            },

            // q_a * a + q_b * b + q_c * c + q_d * d + q_e * e +
            // q_mul_ab * a * b +
            // q_mul_cd * c * d +
            // q_e_next * e +
            // q_constant = 0
            CombinationOption::CombineToNextDoubleMul(next) => {
                ctx.assign_fixed(|| "s_mul_ab", self.config.s_mul_ab, F::ONE)?;
                ctx.assign_fixed(|| "s_mul_cd", self.config.s_mul_cd, F::ONE)?;
                ctx.assign_fixed(|| "se_next", self.config.se_next, next)?;
            }

            // q_a * a + q_b * b + q_c * c + q_d * d + q_e * e +
            // q_mul_ab * a * b +
            // q_mul_cd * c * d +
            // q_constant = 0
            CombinationOption::OneLinerDoubleMul(e) => {
                ctx.assign_fixed(|| "s_mul_ab", self.config.s_mul_ab, F::ONE)?;
                ctx.assign_fixed(|| "s_mul_cd", self.config.s_mul_cd, e)?;
                ctx.assign_fixed(|| "se_next", self.config.se_next, F::ZERO)?;
            }
        };

        // If given witness is already assigned apply copy constains
        for (term, rhs) in terms.iter().zip(assigned.iter()) {
            if let Term::Assigned(lhs, _) = term {
                ctx.constrain_equal(lhs.cell(), rhs.cell())?;
            }
        }

        ctx.next();

        Ok(assigned)
    }

    /// Skip this row without any operation
    fn no_operation(&self, ctx: &mut RegionCtx<'_, F>) -> Result<(), Error> {
        ctx.assign_fixed(|| "s_mul_ab", self.config.s_mul_ab, F::ZERO)?;
        ctx.assign_fixed(|| "s_mul_cd", self.config.s_mul_cd, F::ZERO)?;
        ctx.assign_fixed(|| "sc", self.config.sc, F::ZERO)?;
        ctx.assign_fixed(|| "sa", self.config.sa, F::ZERO)?;
        ctx.assign_fixed(|| "sb", self.config.sb, F::ZERO)?;
        ctx.assign_fixed(|| "sd", self.config.sd, F::ZERO)?;
        ctx.assign_fixed(|| "se", self.config.se, F::ZERO)?;
        ctx.assign_fixed(|| "se_next", self.config.se_next, F::ZERO)?;
        ctx.assign_fixed(|| "s_constant", self.config.s_constant, F::ZERO)?;
        ctx.next();
        Ok(())
    }
}

impl<F: PrimeField> MainGate<F> {
    /// Create new main gate with given config
    pub fn new(config: MainGateConfig) -> Self {
        MainGate {
            config,
            _marker: PhantomData,
        }
    }

    /// Configures polynomial relationships and returns the resuiting config
    pub fn configure(meta: &mut ConstraintSystem<F>) -> MainGateConfig {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let d = meta.advice_column();
        let e = meta.advice_column();

        let sa = meta.fixed_column();
        let sb = meta.fixed_column();
        let sc = meta.fixed_column();
        let sd = meta.fixed_column();
        let se = meta.fixed_column();

        let s_mul_ab = meta.fixed_column();
        let s_mul_cd = meta.fixed_column();

        let se_next = meta.fixed_column();
        let s_constant = meta.fixed_column();

        let instance = meta.instance_column();

        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);
        meta.enable_equality(d);
        meta.enable_equality(e);
        meta.enable_equality(instance);

        meta.create_gate("main_gate", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let d = meta.query_advice(d, Rotation::cur());
            let e_next = meta.query_advice(e, Rotation::next());
            let e = meta.query_advice(e, Rotation::cur());

            let sa = meta.query_fixed(sa, Rotation::cur());
            let sb = meta.query_fixed(sb, Rotation::cur());
            let sc = meta.query_fixed(sc, Rotation::cur());
            let sd = meta.query_fixed(sd, Rotation::cur());
            let se = meta.query_fixed(se, Rotation::cur());

            let se_next = meta.query_fixed(se_next, Rotation::cur());

            let s_mul_ab = meta.query_fixed(s_mul_ab, Rotation::cur());
            let s_mul_cd = meta.query_fixed(s_mul_cd, Rotation::cur());

            let s_constant = meta.query_fixed(s_constant, Rotation::cur());

            vec![
                a.clone() * sa
                    + b.clone() * sb
                    + c.clone() * sc
                    + d.clone() * sd
                    + e * se
                    + a * b * s_mul_ab
                    + c * d * s_mul_cd
                    + se_next * e_next
                    + s_constant,
            ]
        });

        MainGateConfig {
            a,
            b,
            c,
            d,
            e,
            sa,
            sb,
            sc,
            sd,
            se,
            se_next,
            s_constant,
            s_mul_ab,
            s_mul_cd,
            instance,
        }
    }
}
