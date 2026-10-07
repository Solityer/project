use crate::plonk::{permutation, Any, ProvingKey};

#[cfg(feature = "mv-lookup")]
use crate::plonk::mv_lookup as lookup;

#[cfg(not(feature = "mv-lookup"))]
use crate::plonk::lookup;

use crate::poly::Basis;
use crate::{
    arithmetic::{parallelize, CurveAffine},
    poly::{Coeff, EvaluationDomain, ExtendedLagrangeCoeff, Polynomial, Rotation},
};

use group::ff::{Field, PrimeField, WithSmallOrderMulGroup};
#[cfg(feature = "mv-lookup")]
use maybe_rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::collections::BTreeSet;
use std::io;

use super::{shuffle, ConstraintSystem, Expression};

#[cfg(feature = "mv-lookup")]
use ff::BatchInvert;

#[cfg(feature = "mv-lookup")]
use maybe_rayon::slice::ParallelSliceMut;

/// Return the index in the polynomial of size `isize` after rotation `rot`.
fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub enum ValueSource {
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
    /// This is a challenge
    Challenge(usize),
    /// beta
    Beta(),
    /// gamma
    Gamma(),
    /// theta
    Theta(),
    /// y
    Y(),
    /// Previous value
    PreviousValue(),
}

impl Default for ValueSource {
    fn default() -> Self {
        ValueSource::Constant(0)
    }
}

impl ValueSource {
    /// Get the value for this source
    #[allow(clippy::too_many_arguments)]
    pub fn get<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        match self {
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Challenge(index) => challenges[*index],
            ValueSource::Beta() => *beta,
            ValueSource::Gamma() => *gamma,
            ValueSource::Theta() => *theta,
            ValueSource::Y() => *y,
            ValueSource::PreviousValue() => *previous_value,
        }
    }
}

/// Calculation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a square
    Square(ValueSource),
    /// This is a double
    Double(ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is Horner's rule: `val = a; val = val * c + b[]`
    Horner(ValueSource, Vec<ValueSource>, ValueSource),
    /// This is a simple assignment
    Store(ValueSource),
}

impl Calculation {
    /// Get the resulting value of this calculation
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        let get_value = |value: &ValueSource| {
            value.get(
                rotations,
                constants,
                intermediates,
                fixed_values,
                advice_values,
                instance_values,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            )
        };
        match self {
            Calculation::Add(a, b) => get_value(a) + get_value(b),
            Calculation::Sub(a, b) => get_value(a) - get_value(b),
            Calculation::Mul(a, b) => get_value(a) * get_value(b),
            Calculation::Square(v) => get_value(v).square(),
            Calculation::Double(v) => get_value(v).double(),
            Calculation::Negate(v) => -get_value(v),
            Calculation::Horner(start_value, parts, factor) => {
                let factor = get_value(factor);
                let mut value = get_value(start_value);
                for part in parts.iter() {
                    value = value * factor + get_value(part);
                }
                value
            }
            Calculation::Store(v) => get_value(v),
        }
    }
}

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<C: CurveAffine> {
    /// Custom gate constraints, kept in native gate/polynomial order so the
    /// prover can evaluate and release one constraint group's cosets at a time.
    pub custom_gates: Vec<GraphEvaluator<C>>,
    ///  Lookups evalution
    #[cfg(feature = "mv-lookup")]
    pub lookups: Vec<(Vec<GraphEvaluator<C>>, GraphEvaluator<C>)>,
    #[cfg(not(feature = "mv-lookup"))]
    pub lookups: Vec<GraphEvaluator<C>>,
    ///  Shuffle evalution
    pub shuffles: Vec<GraphEvaluator<C>>,
}

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<C: CurveAffine> {
    /// Constants
    pub constants: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

/// EvaluationData
#[derive(Default, Debug)]
pub struct EvaluationData<C: CurveAffine> {
    /// Intermediates
    pub intermediates: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<usize>,
}

/// CaluclationInfo
#[derive(Clone, Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// Target
    pub target: usize,
}

#[derive(Default)]
struct ColumnDependencies {
    fixed: BTreeSet<usize>,
    advice: BTreeSet<usize>,
    instance: BTreeSet<usize>,
}

impl ColumnDependencies {
    fn add(&mut self, source: &ValueSource) {
        match source {
            ValueSource::Fixed(column, _) => {
                self.fixed.insert(*column);
            }
            ValueSource::Advice(column, _) => {
                self.advice.insert(*column);
            }
            ValueSource::Instance(column, _) => {
                self.instance.insert(*column);
            }
            _ => {}
        }
    }

    fn extend(&mut self, other: &Self) {
        self.fixed.extend(other.fixed.iter().copied());
        self.advice.extend(other.advice.iter().copied());
        self.instance.extend(other.instance.iter().copied());
    }
}

/// Reconstruct only the cosets referenced by the active native constraint
/// group. Unused entries are zero-length sentinels; graph dependency discovery
/// guarantees that evaluation never indexes one of them.
fn load_cosets<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    polys: &[Polynomial<F, Coeff>],
    columns: &BTreeSet<usize>,
) -> io::Result<Vec<Polynomial<F, ExtendedLagrangeCoeff>>> {
    let mut cosets = (0..polys.len())
        .map(|_| Polynomial::from_vec(Vec::new()))
        .collect::<Vec<_>>();
    for &column in columns {
        let poly = polys.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "constraint references an absent polynomial",
            )
        })?;
        cosets[column] = domain.coeff_to_extended_spilled(poly)?;
    }
    Ok(cosets)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_graph_group<C: CurveAffine>(
    graph: &GraphEvaluator<C>,
    pk: &ProvingKey<C>,
    advice_polys: &[Polynomial<C::ScalarExt, Coeff>],
    instance_polys: &[Polynomial<C::ScalarExt, Coeff>],
    challenges: &[C::ScalarExt],
    values: &mut Polynomial<C::ScalarExt, ExtendedLagrangeCoeff>,
    y: C::ScalarExt,
    beta: C::ScalarExt,
    gamma: C::ScalarExt,
    theta: C::ScalarExt,
    rot_scale: i32,
    isize: i32,
) -> io::Result<()> {
    let dependencies = graph.column_dependencies();
    let domain = &pk.vk.domain;
    let fixed = load_cosets(domain, &pk.fixed_polys, &dependencies.fixed)?;
    let advice = load_cosets(domain, advice_polys, &dependencies.advice)?;
    let instance = load_cosets(domain, instance_polys, &dependencies.instance)?;
    parallelize(&mut values[..], |h, start| {
        let mut data = graph.instance();
        for (offset, value) in h.iter_mut().enumerate() {
            let idx = start + offset;
            let contribution = graph.evaluate(
                &mut data,
                &fixed,
                &advice,
                &instance,
                challenges,
                &beta,
                &gamma,
                &theta,
                &y,
                &C::ScalarExt::ZERO,
                idx,
                rot_scale,
                isize,
            );
            *value = *value * y + contribution;
        }
    });
    Ok(())
}

impl<C: CurveAffine> Evaluator<C> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<C::ScalarExt>) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut custom_gates = Vec::new();
        for gate in cs.gates.iter() {
            for poly in gate.polynomials() {
                let mut graph = GraphEvaluator::default();
                let root = graph.add_expression(poly);
                graph.add_calculation(Calculation::Store(root));
                custom_gates.push(graph);
            }
        }
        ev.custom_gates = custom_gates;

        // Lookups
        #[cfg(feature = "mv-lookup")]
        for lookup in cs.lookups.iter() {
            let mut graph_table = GraphEvaluator::default();
            let mut graph_inputs: Vec<_> = (0..lookup.inputs_expressions.len())
                .map(|_| GraphEvaluator::default())
                .collect();

            let evaluate_lc = |graph: &mut GraphEvaluator<C>, expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Inputs cosets
            for (input_expressions, graph_input) in lookup
                .inputs_expressions
                .iter()
                .zip(graph_inputs.iter_mut())
            {
                let compressed_input_coset = evaluate_lc(graph_input, input_expressions);

                graph_input.add_calculation(Calculation::Add(
                    compressed_input_coset,
                    ValueSource::Beta(),
                ));
            }

            // table coset
            let compressed_table_coset = evaluate_lc(&mut graph_table, &lookup.table_expressions);

            graph_table.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Beta(),
            ));

            /*
                a) f_i + beta
                b) t + beta
            */
            ev.lookups.push((graph_inputs.to_vec(), graph_table));
        }

        #[cfg(not(feature = "mv-lookup"))]
        // Lookups
        for lookup in cs.lookups.iter() {
            let mut graph = GraphEvaluator::default();

            let mut evaluate_lc = |expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = graph.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Gamma(),
            ));
            let lc = graph.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Beta(),
            ));
            graph.add_calculation(Calculation::Mul(lc, right_gamma));

            ev.lookups.push(graph);
        }

        // Shuffles
        for shuffle in cs.shuffles.iter() {
            let evaluate_lc = |expressions: &Vec<Expression<_>>, graph: &mut GraphEvaluator<C>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            let mut graph_input = GraphEvaluator::default();
            let compressed_input_coset = evaluate_lc(&shuffle.input_expressions, &mut graph_input);
            let _ = graph_input.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Gamma(),
            ));

            let mut graph_shuffle = GraphEvaluator::default();
            let compressed_shuffle_coset =
                evaluate_lc(&shuffle.shuffle_expressions, &mut graph_shuffle);
            let _ = graph_shuffle.add_calculation(Calculation::Add(
                compressed_shuffle_coset,
                ValueSource::Gamma(),
            ));

            ev.shuffles.push(graph_input);
            ev.shuffles.push(graph_shuffle);
        }

        ev
    }

    /// Evaluate the native quotient polynomial a constraint group at a time.
    /// The Horner order remains gate -> permutation -> lookup -> shuffle, while
    /// each group's extended-domain inputs are reconstructed in mapped storage
    /// and released before the next group starts.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn evaluate_h(
        &self,
        pk: &ProvingKey<C>,
        advice_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        instance_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        challenges: &[C::ScalarExt],
        y: C::ScalarExt,
        beta: C::ScalarExt,
        gamma: C::ScalarExt,
        theta: C::ScalarExt,
        lookups: &[Vec<lookup::prover::Committed<C>>],
        shuffles: &[Vec<shuffle::prover::Committed<C>>],
        permutations: &[permutation::prover::Committed<C>],
    ) -> Result<Polynomial<C::ScalarExt, ExtendedLagrangeCoeff>, super::Error> {
        let domain = &pk.vk.domain;
        let size = domain.extended_len();
        let rot_scale = 1 << (domain.extended_k() - domain.k());
        let isize = size as i32;
        let one = C::ScalarExt::ONE;
        let l0 = &pk.l0;
        let l_last = &pk.l_last;
        let l_active_row = &pk.l_active_row;
        let p = &pk.vk.cs.permutation;
        let extended_omega = domain.get_extended_omega();
        let mut values: Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> =
            Polynomial::mapped_filled(size, C::ScalarExt::ZERO)
                .map_err(super::Error::NativeStorage)?;

        for (((advice, instance), lookup_sets), (shuffle_sets, permutation)) in advice_polys
            .iter()
            .zip(instance_polys.iter())
            .zip(lookups.iter())
            .zip(shuffles.iter().zip(permutations.iter()))
        {
            // Custom gate polynomials are folded in their existing native order.
            for graph in &self.custom_gates {
                evaluate_graph_group(
                    graph,
                    pk,
                    advice,
                    instance,
                    challenges,
                    &mut values,
                    y,
                    beta,
                    gamma,
                    theta,
                    rot_scale,
                    isize,
                )?;
            }

            // Permutation boundary and copy constraints.
            let sets = &permutation.sets;
            if !sets.is_empty() {
                let chunk_len = pk.vk.cs.degree() - 2;
                let column_groups = p.columns.chunks(chunk_len).collect::<Vec<_>>();
                if sets.len() != column_groups.len() {
                    return Err(super::Error::NativeStorage(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "permutation product set count does not match circuit columns",
                    )));
                }
                let last_rotation = Rotation(-((pk.vk.cs.blinding_factors() + 1) as i32));

                let first_z = domain
                    .coeff_to_extended_spilled(&sets[0].permutation_product_poly)
                    .map_err(super::Error::NativeStorage)?;
                parallelize(&mut values[..], |h, start| {
                    for (offset, value) in h.iter_mut().enumerate() {
                        let idx = start + offset;
                        *value = *value * y + ((one - first_z[idx]) * l0[idx]);
                    }
                });
                drop(first_z);

                let last_z = domain
                    .coeff_to_extended_spilled(&sets.last().unwrap().permutation_product_poly)
                    .map_err(super::Error::NativeStorage)?;
                parallelize(&mut values[..], |h, start| {
                    for (offset, value) in h.iter_mut().enumerate() {
                        let idx = start + offset;
                        let z = last_z[idx];
                        *value = *value * y + ((z.square() - z) * l_last[idx]);
                    }
                });
                drop(last_z);

                if sets.len() > 1 {
                    let mut previous_z = domain
                        .coeff_to_extended_spilled(&sets[0].permutation_product_poly)
                        .map_err(super::Error::NativeStorage)?;
                    for set in sets.iter().skip(1) {
                        let current_z = domain
                            .coeff_to_extended_spilled(&set.permutation_product_poly)
                            .map_err(super::Error::NativeStorage)?;
                        parallelize(&mut values[..], |h, start| {
                            for (offset, value) in h.iter_mut().enumerate() {
                                let idx = start + offset;
                                let r_last =
                                    get_rotation_idx(idx, last_rotation.0, rot_scale, isize);
                                *value =
                                    *value * y + ((current_z[idx] - previous_z[r_last]) * l0[idx]);
                            }
                        });
                        previous_z = current_z;
                    }
                }

                // One set at a time, and one source/sigma pair at a time. The
                // running products are mapped arrays; no all-column coset bank
                // or all-set product coset bank is retained.
                let mut column_index = 0usize;
                for (set, columns) in sets.iter().zip(column_groups.into_iter()) {
                    let z = domain
                        .coeff_to_extended_spilled(&set.permutation_product_poly)
                        .map_err(super::Error::NativeStorage)?;
                    let mut left: Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> =
                        Polynomial::mapped_filled(size, C::ScalarExt::ONE)
                            .map_err(super::Error::NativeStorage)?;
                    let mut right: Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> =
                        Polynomial::mapped_filled(size, C::ScalarExt::ONE)
                            .map_err(super::Error::NativeStorage)?;
                    parallelize(&mut left[..], |out, start| {
                        for (offset, value) in out.iter_mut().enumerate() {
                            let idx = start + offset;
                            let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                            *value = z[r_next];
                        }
                    });
                    parallelize(&mut right[..], |out, start| {
                        out.copy_from_slice(&z[start..start + out.len()]);
                    });
                    drop(z);

                    let mut delta_power = C::ScalarExt::ONE;
                    for column in columns {
                        let source = match column.column_type() {
                            Any::Advice(_) => &advice[column.index()],
                            Any::Fixed => &pk.fixed_polys[column.index()],
                            Any::Instance => &instance[column.index()],
                        };
                        let source_coset = domain
                            .coeff_to_extended_spilled(source)
                            .map_err(super::Error::NativeStorage)?;
                        let sigma_poly =
                            pk.permutation.polys().get(column_index).ok_or_else(|| {
                                super::Error::NativeStorage(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "permutation sigma polynomial is missing",
                                ))
                            })?;
                        let sigma_coset = domain
                            .coeff_to_extended_spilled(sigma_poly)
                            .map_err(super::Error::NativeStorage)?;

                        parallelize(&mut left[..], |out, start| {
                            for (offset, value) in out.iter_mut().enumerate() {
                                let idx = start + offset;
                                *value *= source_coset[idx] + beta * sigma_coset[idx] + gamma;
                            }
                        });
                        let delta_start = beta * C::ScalarExt::ZETA * delta_power;
                        parallelize(&mut right[..], |out, start| {
                            let mut beta_term =
                                delta_start * extended_omega.pow_vartime([start as u64]);
                            for (offset, value) in out.iter_mut().enumerate() {
                                let idx = start + offset;
                                *value *= source_coset[idx] + beta_term + gamma;
                                beta_term *= &extended_omega;
                            }
                        });
                        delta_power *= &C::ScalarExt::DELTA;
                        column_index += 1;
                        drop(sigma_coset);
                        drop(source_coset);
                    }

                    parallelize(&mut values[..], |h, start| {
                        for (offset, value) in h.iter_mut().enumerate() {
                            let idx = start + offset;
                            *value = *value * y + ((left[idx] - right[idx]) * l_active_row[idx]);
                        }
                    });
                    drop(left);
                    drop(right);
                }
            }

            #[cfg(feature = "mv-lookup")]
            for (lookup_index, lookup) in lookup_sets.iter().enumerate() {
                let (input_graphs, table_graph) = &self.lookups[lookup_index];
                if input_graphs.is_empty() {
                    return Err(super::Error::NativeStorage(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "multi-lookup argument has no input expressions",
                    )));
                }
                let mut dependencies = ColumnDependencies::default();
                for graph in input_graphs.iter().chain(std::iter::once(table_graph)) {
                    dependencies.extend(&graph.column_dependencies());
                }
                let fixed = load_cosets(domain, &pk.fixed_polys, &dependencies.fixed)
                    .map_err(super::Error::NativeStorage)?;
                let advice_cosets = load_cosets(domain, advice, &dependencies.advice)
                    .map_err(super::Error::NativeStorage)?;
                let instance_cosets = load_cosets(domain, instance, &dependencies.instance)
                    .map_err(super::Error::NativeStorage)?;
                let phi_coset = domain
                    .coeff_to_extended_spilled(&lookup.phi_poly)
                    .map_err(super::Error::NativeStorage)?;
                let m_coset = domain
                    .coeff_to_extended_spilled(&lookup.m_poly)
                    .map_err(super::Error::NativeStorage)?;
                const LOOKUP_ROW_CHUNK: usize = 4096;
                (&mut values[..])
                    .par_chunks_mut(LOOKUP_ROW_CHUNK)
                    .enumerate()
                    .for_each(|(chunk_index, h)| {
                        let start = chunk_index * LOOKUP_ROW_CHUNK;
                        let mut input_data = input_graphs
                            .iter()
                            .map(GraphEvaluator::instance)
                            .collect::<Vec<_>>();
                        let mut inverse_values = Vec::with_capacity(h.len() * input_graphs.len());
                        for offset in 0..h.len() {
                            let idx = start + offset;
                            for (graph, data) in input_graphs.iter().zip(input_data.iter_mut()) {
                                inverse_values.push(graph.evaluate(
                                    data,
                                    &fixed,
                                    &advice_cosets,
                                    &instance_cosets,
                                    challenges,
                                    &beta,
                                    &gamma,
                                    &theta,
                                    &y,
                                    &C::ScalarExt::ZERO,
                                    idx,
                                    rot_scale,
                                    isize,
                                ));
                            }
                        }
                        inverse_values.batch_invert();

                        let mut input_data = input_graphs
                            .iter()
                            .map(GraphEvaluator::instance)
                            .collect::<Vec<_>>();
                        let mut table_data = table_graph.instance();
                        for (offset, h_value) in h.iter_mut().enumerate() {
                            let idx = start + offset;
                            let mut inputs_product = C::ScalarExt::ONE;
                            for (graph, data) in input_graphs.iter().zip(input_data.iter_mut()) {
                                inputs_product *= graph.evaluate(
                                    data,
                                    &fixed,
                                    &advice_cosets,
                                    &instance_cosets,
                                    challenges,
                                    &beta,
                                    &gamma,
                                    &theta,
                                    &y,
                                    &C::ScalarExt::ZERO,
                                    idx,
                                    rot_scale,
                                    isize,
                                );
                            }
                            let inverse_sum = inverse_values
                                [offset * input_graphs.len()..(offset + 1) * input_graphs.len()]
                                .iter()
                                .fold(C::ScalarExt::ZERO, |sum, inverse| sum + inverse);
                            let table_value = table_graph.evaluate(
                                &mut table_data,
                                &fixed,
                                &advice_cosets,
                                &instance_cosets,
                                challenges,
                                &beta,
                                &gamma,
                                &theta,
                                &y,
                                &C::ScalarExt::ZERO,
                                idx,
                                rot_scale,
                                isize,
                            );
                            let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                            let lhs =
                                table_value * inputs_product * (phi_coset[r_next] - phi_coset[idx]);
                            let rhs = inputs_product * (table_value * inverse_sum - m_coset[idx]);
                            *h_value = *h_value * y + l0[idx] * phi_coset[idx];
                            *h_value = *h_value * y + l_last[idx] * phi_coset[idx];
                            *h_value = *h_value * y + (lhs - rhs) * l_active_row[idx];
                        }
                    });
                drop(m_coset);
                drop(phi_coset);
                drop(instance_cosets);
                drop(advice_cosets);
                drop(fixed);
            }

            #[cfg(not(feature = "mv-lookup"))]
            for (lookup_index, lookup) in lookup_sets.iter().enumerate() {
                let graph = &self.lookups[lookup_index];
                let dependencies = graph.column_dependencies();
                let fixed = load_cosets(domain, &pk.fixed_polys, &dependencies.fixed)
                    .map_err(super::Error::NativeStorage)?;
                let advice_cosets = load_cosets(domain, advice, &dependencies.advice)
                    .map_err(super::Error::NativeStorage)?;
                let instance_cosets = load_cosets(domain, instance, &dependencies.instance)
                    .map_err(super::Error::NativeStorage)?;
                let product_coset = domain
                    .coeff_to_extended_spilled(&lookup.product_poly)
                    .map_err(super::Error::NativeStorage)?;
                let input_coset = domain
                    .coeff_to_extended_spilled(&lookup.permuted_input_poly)
                    .map_err(super::Error::NativeStorage)?;
                let table_coset = domain
                    .coeff_to_extended_spilled(&lookup.permuted_table_poly)
                    .map_err(super::Error::NativeStorage)?;
                parallelize(&mut values[..], |h, start| {
                    let mut data = graph.instance();
                    for (offset, h_value) in h.iter_mut().enumerate() {
                        let idx = start + offset;
                        let table_value = graph.evaluate(
                            &mut data,
                            &fixed,
                            &advice_cosets,
                            &instance_cosets,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &y,
                            &C::ScalarExt::ZERO,
                            idx,
                            rot_scale,
                            isize,
                        );
                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        let r_prev = get_rotation_idx(idx, -1, rot_scale, isize);
                        let input_minus_table = input_coset[idx] - table_coset[idx];
                        *h_value = *h_value * y + ((one - product_coset[idx]) * l0[idx]);
                        *h_value = *h_value * y
                            + ((product_coset[idx].square() - product_coset[idx]) * l_last[idx]);
                        *h_value = *h_value * y
                            + ((product_coset[r_next]
                                * (input_coset[idx] + beta)
                                * (table_coset[idx] + gamma)
                                - product_coset[idx] * table_value)
                                * l_active_row[idx]);
                        *h_value = *h_value * y + (input_minus_table * l0[idx]);
                        *h_value = *h_value * y
                            + (input_minus_table
                                * (input_coset[idx] - input_coset[r_prev])
                                * l_active_row[idx]);
                    }
                });
            }

            for (shuffle_index, shuffle) in shuffle_sets.iter().enumerate() {
                let input_graph = &self.shuffles[2 * shuffle_index];
                let shuffle_graph = &self.shuffles[2 * shuffle_index + 1];
                let mut dependencies = input_graph.column_dependencies();
                dependencies.extend(&shuffle_graph.column_dependencies());
                let fixed = load_cosets(domain, &pk.fixed_polys, &dependencies.fixed)
                    .map_err(super::Error::NativeStorage)?;
                let advice_cosets = load_cosets(domain, advice, &dependencies.advice)
                    .map_err(super::Error::NativeStorage)?;
                let instance_cosets = load_cosets(domain, instance, &dependencies.instance)
                    .map_err(super::Error::NativeStorage)?;
                let product_coset = domain
                    .coeff_to_extended_spilled(&shuffle.product_poly)
                    .map_err(super::Error::NativeStorage)?;
                parallelize(&mut values[..], |h, start| {
                    let mut input_data = input_graph.instance();
                    let mut shuffle_data = shuffle_graph.instance();
                    for (offset, h_value) in h.iter_mut().enumerate() {
                        let idx = start + offset;
                        let input_value = input_graph.evaluate(
                            &mut input_data,
                            &fixed,
                            &advice_cosets,
                            &instance_cosets,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &y,
                            &C::ScalarExt::ZERO,
                            idx,
                            rot_scale,
                            isize,
                        );
                        let shuffle_value = shuffle_graph.evaluate(
                            &mut shuffle_data,
                            &fixed,
                            &advice_cosets,
                            &instance_cosets,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &y,
                            &C::ScalarExt::ZERO,
                            idx,
                            rot_scale,
                            isize,
                        );
                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        *h_value = *h_value * y + ((one - product_coset[idx]) * l0[idx]);
                        *h_value = *h_value * y
                            + ((product_coset[idx].square() - product_coset[idx]) * l_last[idx]);
                        *h_value = *h_value * y
                            + l_active_row[idx]
                                * (product_coset[r_next] * shuffle_value
                                    - product_coset[idx] * input_value);
                    }
                });
                drop(product_coset);
                drop(instance_cosets);
                drop(advice_cosets);
                drop(fixed);
            }
        }

        Ok(values)
    }
}

impl<C: CurveAffine> Default for GraphEvaluator<C> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![
                C::ScalarExt::ZERO,
                C::ScalarExt::ONE,
                C::ScalarExt::from(2u64),
            ],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<C: CurveAffine> GraphEvaluator<C> {
    fn column_dependencies(&self) -> ColumnDependencies {
        let mut dependencies = ColumnDependencies::default();
        for calc in &self.calculations {
            match &calc.calculation {
                Calculation::Add(a, b) | Calculation::Sub(a, b) | Calculation::Mul(a, b) => {
                    dependencies.add(a);
                    dependencies.add(b);
                }
                Calculation::Square(value)
                | Calculation::Double(value)
                | Calculation::Negate(value)
                | Calculation::Store(value) => dependencies.add(value),
                Calculation::Horner(start, parts, factor) => {
                    dependencies.add(start);
                    dependencies.add(factor);
                    for part in parts {
                        dependencies.add(part);
                    }
                }
            }
        }
        dependencies
    }

    /// Adds a rotation
    fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    fn add_constant(&mut self, constant: &C::ScalarExt) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    fn add_expression(&mut self, expr: &Expression<C::ScalarExt>) -> ValueSource {
        match expr {
            Expression::Constant(scalar) => self.add_constant(scalar),
            Expression::Selector(_selector) => unreachable!(),
            Expression::Fixed(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Advice(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Instance(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Challenge(challenge) => self.add_calculation(Calculation::Store(
                ValueSource::Challenge(challenge.index()),
            )),
            Expression::Negated(a) => match **a {
                Expression::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            Expression::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    Expression::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            self.add_calculation(Calculation::Negate(result_b))
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            Expression::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_b))
                } else if result_b == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_a))
                } else if result_a == result_b {
                    self.add_calculation(Calculation::Square(result_a))
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            Expression::Scaled(a, f) => {
                if *f == C::ScalarExt::ZERO {
                    ValueSource::Constant(0)
                } else if *f == C::ScalarExt::ONE {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }

    /// Creates a new evaluation structure
    pub fn instance(&self) -> EvaluationData<C> {
        EvaluationData {
            intermediates: vec![C::ScalarExt::ZERO; self.num_intermediates],
            rotations: vec![0usize; self.rotations.len()],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<B: Basis>(
        &self,
        data: &mut EvaluationData<C>,
        fixed: &[Polynomial<C::ScalarExt, B>],
        advice: &[Polynomial<C::ScalarExt, B>],
        instance: &[Polynomial<C::ScalarExt, B>],
        challenges: &[C::ScalarExt],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        previous_value: &C::ScalarExt,
        idx: usize,
        rot_scale: i32,
        isize: i32,
    ) -> C::ScalarExt {
        // All rotation index values
        for (rot_idx, rot) in self.rotations.iter().enumerate() {
            data.rotations[rot_idx] = get_rotation_idx(idx, *rot, rot_scale, isize);
        }

        // All calculations, with cached intermediate results
        for calc in self.calculations.iter() {
            data.intermediates[calc.target] = calc.calculation.evaluate(
                &data.rotations,
                &self.constants,
                &data.intermediates,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            );
        }

        // Return the result of the last calculation (if any)
        if let Some(calc) = self.calculations.last() {
            data.intermediates[calc.target]
        } else {
            C::ScalarExt::ZERO
        }
    }
}

/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: Basis>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|query| {
                    fixed[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}
