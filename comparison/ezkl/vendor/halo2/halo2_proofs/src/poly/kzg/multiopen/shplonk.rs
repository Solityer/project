mod prover;
mod verifier;

use crate::{poly::query::Query, transcript::ChallengeScalar};
use ff::Field;
use maybe_rayon::{iter::ParallelIterator, prelude::IntoParallelIterator};
pub use prover::ProverSHPLONK;
use std::collections::BTreeSet;
pub use verifier::VerifierSHPLONK;

#[derive(Clone, Copy, Debug)]
struct U {}
type ChallengeU<F> = ChallengeScalar<F, U>;

#[derive(Clone, Copy, Debug)]
struct V {}
type ChallengeV<F> = ChallengeScalar<F, V>;

#[derive(Clone, Copy, Debug)]
struct Y {}
type ChallengeY<F> = ChallengeScalar<F, Y>;

#[derive(Debug, Clone, PartialEq)]
struct Commitment<F: Field, T: PartialEq + Clone>((T, Vec<F>));

impl<F: Field, T: PartialEq + Clone> Commitment<F, T> {
    fn get(&self) -> T {
        self.0 .0.clone()
    }

    fn evals(&self) -> Vec<F> {
        self.0 .1.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RotationSet<F: Field, T: PartialEq + Clone> {
    commitments: Vec<Commitment<F, T>>,
    points: Vec<F>,
}

#[derive(Debug, PartialEq)]
struct IntermediateSets<F: Field, Q: Query<F>> {
    rotation_sets: Vec<RotationSet<F, Q::Commitment>>,
    super_point_set: BTreeSet<F>,
}

fn construct_intermediate_sets<F: Field + Ord, I, Q: Query<F, Eval = F>>(
    queries: I,
) -> IntermediateSets<F, Q>
where
    I: IntoIterator<Item = Q> + Clone,
{
    let queries = queries.into_iter().collect::<Vec<_>>();

    // Find evaluation of a commitment at a rotation
    let get_eval = |commitment: Q::Commitment, rotation: F| -> F {
        queries
            .iter()
            .find(|query| query.get_commitment() == commitment && query.get_point() == rotation)
            .unwrap()
            .get_eval()
    };

    // All points that appear in queries
    let mut super_point_set = BTreeSet::new();

    // Collect rotation sets for each commitment
    // Example elements in the vector:
    // (C_0, {r_5}),
    // (C_1, {r_1, r_2, r_3}),
    // (C_2, {r_2, r_3, r_4}),
    // (C_3, {r_2, r_3, r_4}),
    // ...
    let mut commitment_rotation_set_map: Vec<(Q::Commitment, BTreeSet<F>)> = vec![];
    for query in queries.iter() {
        let rotation = query.get_point();
        super_point_set.insert(rotation);
        if let Some(commitment_rotation_set) = commitment_rotation_set_map
            .iter_mut()
            .find(|(commitment, _)| *commitment == query.get_commitment())
        {
            let (_, rotation_set) = commitment_rotation_set;
            rotation_set.insert(rotation);
        } else {
            commitment_rotation_set_map.push((
                query.get_commitment(),
                BTreeSet::from_iter(std::iter::once(rotation)),
            ));
        };
    }

    // Flatten rotation sets and collect commitments that opens against each commitment set
    // Example elements in the vector:
    // {r_5}: [C_0],
    // {r_1, r_2, r_3} : [C_1]
    // {r_2, r_3, r_4} : [C_2, C_3],
    // ...
    // NOTE: we want to make the order of the collection of rotation sets independent of the opening points, to ease the verifier computation
    let mut rotation_set_commitment_map: Vec<(BTreeSet<F>, Vec<Q::Commitment>)> = vec![];
    for (commitment, rotation_set) in commitment_rotation_set_map.into_iter() {
        if let Some(rotation_set_commitment) = rotation_set_commitment_map
            .iter_mut()
            .find(|(set, _)| set == &rotation_set)
        {
            let (_, commitments) = rotation_set_commitment;
            commitments.push(commitment);
        } else {
            rotation_set_commitment_map.push((rotation_set, vec![commitment]));
        };
    }

    let rotation_sets = rotation_set_commitment_map
        .into_par_iter()
        .map(|(rotations, commitments)| {
            let rotations_vec = rotations.iter().collect::<Vec<_>>();
            let commitments: Vec<Commitment<F, Q::Commitment>> = commitments
                .into_par_iter()
                .map(|commitment| {
                    let evals: Vec<F> = rotations_vec
                        .as_slice()
                        .into_par_iter()
                        .map(|&&rotation| get_eval(commitment, rotation))
                        .collect();
                    Commitment((commitment, evals))
                })
                .collect();

            RotationSet {
                commitments,
                points: rotations.into_iter().collect(),
            }
        })
        .collect::<Vec<RotationSet<_, _>>>();

    IntermediateSets {
        rotation_sets,
        super_point_set,
    }
}
