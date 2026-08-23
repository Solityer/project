use crate::algebra::{eq_weights, interpolate_at, Scalar};
use crate::hyrax::{
    commit_with_backend, prove_evaluation, verify_evaluation, CommitmentParameters,
    CommitmentWitness, EvaluationProof, VectorCommitment,
};
use crate::prover_backend::ProverBackend;
use crate::registry::{
    ColumnSource, OperationFamilySpec, RelationRegistry, Visibility, WitnessPackage,
};
use crate::transcript::Transcript;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckRound {
    pub evaluations: Vec<Scalar>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FamilyOpening {
    pub family_id: String,
    pub combined_claim: Scalar,
    pub proof: EvaluationProof,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationProof {
    pub family_id: String,
    pub mask_commitment: VectorCommitment,
    pub rounds: Vec<SumcheckRound>,
    pub terminal_columns: Vec<Scalar>,
    pub mask_terminal: Scalar,
    pub family_openings: Vec<FamilyOpening>,
    pub mask_opening: EvaluationProof,
    pub transcript_digest: [u8; 32],
}

fn eval_residuals(
    op: &OperationFamilySpec,
    values: &[Scalar],
    lambdas: &[Scalar],
) -> Result<Scalar> {
    let mut out = Scalar::zero();
    for (e, &l) in op.residuals.iter().zip(lambdas) {
        out += e.eval(values)? * l;
    }
    Ok(out)
}
fn build_columns(
    registry: &RelationRegistry,
    op: &OperationFamilySpec,
    witness: &WitnessPackage,
) -> Result<Vec<Vec<Scalar>>> {
    op.columns
        .iter()
        .map(|c| match &c.source {
            ColumnSource::Public { values } => Ok(values.clone()),
            ColumnSource::TensorView {
                family_id,
                canonical_index_by_row,
            } => {
                let spec = registry.tensor(family_id)?;
                let t = if spec.visibility == Visibility::Public {
                    registry.public_tensors.get(family_id)
                } else {
                    witness.tensors.get(family_id)
                }
                .ok_or_else(|| anyhow::anyhow!("missing family {family_id}"))?;
                (0..op.rows)
                    .map(|row| Ok(t[canonical_index_by_row.at(row)?]))
                    .collect()
            }
        })
        .collect()
}
fn absorb_operation_header(t: &mut Transcript, op: &OperationFamilySpec, mask: &VectorCommitment) {
    t.label("zk-sumcheck-operation-v1");
    t.label(&op.family_id);
    t.label(&op.operation_kind);
    t.u64(b"rows", op.rows as u64);
    t.u64(b"degree", op.degree_bound as u64);
    t.point(b"mask-commitment", mask.point);
}

// Commitment witnesses remain proof-local and are never serialized.
pub fn prove_operation(
    pp: &CommitmentParameters,
    registry: &RelationRegistry,
    op: &OperationFamilySpec,
    witness: &WitnessPackage,
    tensor_commitments: &BTreeMap<String, VectorCommitment>,
    opening_witnesses: &BTreeMap<String, CommitmentWitness>,
    transcript: &mut Transcript,
    backend: &ProverBackend,
) -> Result<OperationProof> {
    let mut columns = build_columns(registry, op, witness)?;
    for row in 0..op.rows {
        let vals = columns.iter().map(|x| x[row]).collect::<Vec<_>>();
        for e in &op.residuals {
            ensure!(
                e.eval(&vals)? == Scalar::zero(),
                "operation {} residual nonzero at row {}",
                op.family_id,
                row
            );
        }
    }
    let mask_values = (0..op.rows / 2)
        .map(|_| Scalar::random())
        .collect::<Result<Vec<_>>>()?;
    let (mask_commitment, mask_witness) =
        commit_with_backend(pp, mask_values, Scalar::random()?, backend)?;
    absorb_operation_header(transcript, op, &mask_commitment);
    let lambdas = (0..op.residuals.len())
        .map(|i| {
            transcript.u64(b"residual-index", i as u64);
            transcript.challenge("operation-residual-compression")
        })
        .collect::<Vec<_>>();
    let nr = op.rows.trailing_zeros() as usize;
    let tau = (0..nr)
        .map(|i| {
            transcript.u64(b"tau-index", i as u64);
            transcript.challenge("operation-query-compression")
        })
        .collect::<Vec<_>>();
    let mut beta = eq_weights(&tau);
    let mut mask = mask_witness.values.clone();
    let mut factor = Scalar::one();
    let mut rounds = vec![];
    let mut rs = vec![];
    let mut running = Scalar::zero();
    for round in 0..nr {
        let degree = op.degree_bound.max(2);
        #[cfg(feature = "gpu")]
        let evaluations = if let Some(gpu) = backend.gpu() {
            gpu.sumcheck_evaluations(
                &columns,
                &beta,
                &mask,
                round == 0,
                factor,
                &lambdas,
                &op.residuals,
                degree,
            )?
        } else {
            cpu_round_evaluations(op, &columns, &beta, &mask, round, factor, &lambdas, degree)?
        };
        #[cfg(not(feature = "gpu"))]
        let evaluations =
            cpu_round_evaluations(op, &columns, &beta, &mask, round, factor, &lambdas, degree)?;
        ensure!(
            evaluations[0] + evaluations[1] == running,
            "sumcheck endpoints"
        );
        for (i, v) in evaluations.iter().enumerate() {
            transcript.u64(b"round-eval-index", i as u64);
            transcript.scalar(b"round-eval", *v);
        }
        let r = transcript.challenge("zk-sumcheck-round");
        running = interpolate_at(&evaluations, r)?;
        rounds.push(SumcheckRound { evaluations });
        rs.push(r);
        columns = columns
            .iter()
            .map(|c| backend.fold_scalars(c, Scalar::one() - r, r))
            .collect::<Result<Vec<_>>>()?;
        beta = backend.fold_scalars(&beta, Scalar::one() - r, r)?;
        if round == 0 {
            factor = r * (Scalar::one() - r)
        } else {
            mask = backend.fold_scalars(&mask, Scalar::one() - r, r)?
        }
    }
    let terminal_columns = columns.iter().map(|x| x[0]).collect::<Vec<_>>();
    let mask_terminal = mask[0];
    ensure!(
        running
            == beta[0] * eval_residuals(op, &terminal_columns, &lambdas)? + factor * mask_terminal,
        "sumcheck terminal"
    );
    let basis = eq_weights(&rs);
    let mut groups: BTreeMap<String, Vec<(usize, &crate::registry::ViewIndex)>> = BTreeMap::new();
    for (ci, c) in op.columns.iter().enumerate() {
        if let ColumnSource::TensorView {
            family_id,
            canonical_index_by_row,
        } = &c.source
        {
            if registry.tensor(family_id)?.visibility != Visibility::Public {
                groups
                    .entry(family_id.clone())
                    .or_default()
                    .push((ci, canonical_index_by_row));
            }
        }
    }
    let mut family_openings = vec![];
    for (family_id, claims) in groups {
        let spec = registry.tensor(&family_id)?;
        let mut weights = vec![Scalar::zero(); spec.logical_len];
        let mut claim = Scalar::zero();
        for (k, (ci, map)) in claims.iter().enumerate() {
            transcript.u64(b"fac-claim-index", k as u64);
            let rho = transcript.challenge("fac-line8-rho");
            claim += rho * terminal_columns[*ci];
            for q in 0..op.rows {
                weights[map.at(q)?] += rho * basis[q];
            }
        }
        let c = tensor_commitments.get(&family_id).unwrap();
        let w = opening_witnesses.get(&family_id).unwrap();
        let p = prove_evaluation(
            pp,
            &format!("{}:fac-line8:{}", op.family_id, family_id),
            c,
            w,
            &weights,
            transcript,
            backend,
        )?;
        ensure!(
            p.claimed_value == claim,
            "FAC Line8 combined claim mismatch"
        );
        family_openings.push(FamilyOpening {
            family_id,
            combined_claim: claim,
            proof: p,
        });
    }
    let mask_weights = eq_weights(&rs[1..]);
    let mask_opening = prove_evaluation(
        pp,
        &format!("{}:sumcheck-mask", op.family_id),
        &mask_commitment,
        &mask_witness,
        &mask_weights,
        transcript,
        backend,
    )?;
    ensure!(
        mask_opening.claimed_value == mask_terminal,
        "mask terminal mismatch"
    );
    let digest = transcript.digest();
    Ok(OperationProof {
        family_id: op.family_id.clone(),
        mask_commitment,
        rounds,
        terminal_columns,
        mask_terminal,
        family_openings,
        mask_opening,
        transcript_digest: digest,
    })
}

fn cpu_round_evaluations(
    op: &OperationFamilySpec,
    columns: &[Vec<Scalar>],
    beta: &[Scalar],
    mask: &[Scalar],
    round: usize,
    factor: Scalar,
    lambdas: &[Scalar],
    degree: usize,
) -> Result<Vec<Scalar>> {
    let half = beta.len() / 2;
    let mut evaluations = vec![];
    for ti in 0..=degree {
        let x = Scalar::from_u64(ti as u64);
        let mut sum = Scalar::zero();
        for i in 0..half {
            let vals = columns
                .iter()
                .map(|c| c[i] * (Scalar::one() - x) + c[i + half] * x)
                .collect::<Vec<_>>();
            let eb = beta[i] * (Scalar::one() - x) + beta[i + half] * x;
            sum += eb * eval_residuals(op, &vals, lambdas)?;
            sum += if round == 0 {
                mask[i] * x * (Scalar::one() - x)
            } else {
                (mask[i] * (Scalar::one() - x) + mask[i + half] * x) * factor
            };
        }
        evaluations.push(sum);
    }
    Ok(evaluations)
}

pub fn verify_operation(
    pp: &CommitmentParameters,
    registry: &RelationRegistry,
    op: &OperationFamilySpec,
    tensor_commitments: &BTreeMap<String, VectorCommitment>,
    proof: &OperationProof,
    transcript: &mut Transcript,
) -> Result<()> {
    ensure!(
        proof.family_id == op.family_id,
        "operation proof family mismatch"
    );
    absorb_operation_header(transcript, op, &proof.mask_commitment);
    let lambdas = (0..op.residuals.len())
        .map(|i| {
            transcript.u64(b"residual-index", i as u64);
            transcript.challenge("operation-residual-compression")
        })
        .collect::<Vec<_>>();
    let nr = op.rows.trailing_zeros() as usize;
    let tau = (0..nr)
        .map(|i| {
            transcript.u64(b"tau-index", i as u64);
            transcript.challenge("operation-query-compression")
        })
        .collect::<Vec<_>>();
    ensure!(proof.rounds.len() == nr, "sumcheck round count");
    let mut running = Scalar::zero();
    let mut rs = vec![];
    for round in &proof.rounds {
        ensure!(
            round.evaluations.len() == op.degree_bound.max(2) + 1,
            "sumcheck round degree"
        );
        ensure!(
            round.evaluations[0] + round.evaluations[1] == running,
            "sumcheck endpoint rejected"
        );
        for (i, v) in round.evaluations.iter().enumerate() {
            transcript.u64(b"round-eval-index", i as u64);
            transcript.scalar(b"round-eval", *v);
        }
        let r = transcript.challenge("zk-sumcheck-round");
        running = interpolate_at(&round.evaluations, r)?;
        rs.push(r);
    }
    ensure!(
        proof.terminal_columns.len() == op.columns.len(),
        "terminal column count"
    );
    let beta = eq_weights(&tau);
    let beta_terminal = crate::algebra::inner_product(&beta, &eq_weights(&rs))?;
    let factor = rs[0] * (Scalar::one() - rs[0]);
    ensure!(
        running
            == beta_terminal * eval_residuals(op, &proof.terminal_columns, &lambdas)?
                + factor * proof.mask_terminal,
        "sumcheck terminal rejected"
    );
    // Public columns are reconstructed locally; prover cannot choose their terminal value.
    let basis = eq_weights(&rs);
    for (ci, c) in op.columns.iter().enumerate() {
        match &c.source {
            ColumnSource::Public { values } => ensure!(
                crate::algebra::inner_product(values, &basis)? == proof.terminal_columns[ci],
                "public terminal mismatch"
            ),
            ColumnSource::TensorView {
                family_id,
                canonical_index_by_row,
            } if registry.tensor(family_id)?.visibility == Visibility::Public => {
                let values = registry.public_tensors.get(family_id).unwrap();
                let view = (0..op.rows)
                    .map(|row| Ok(values[canonical_index_by_row.at(row)?]))
                    .collect::<Result<Vec<_>>>()?;
                ensure!(
                    crate::algebra::inner_product(&view, &basis)? == proof.terminal_columns[ci],
                    "public tensor terminal mismatch"
                );
            }
            _ => {}
        }
    }
    let mut groups: BTreeMap<String, Vec<(usize, &crate::registry::ViewIndex)>> = BTreeMap::new();
    for (ci, c) in op.columns.iter().enumerate() {
        if let ColumnSource::TensorView {
            family_id,
            canonical_index_by_row,
        } = &c.source
        {
            if registry.tensor(family_id)?.visibility != Visibility::Public {
                groups
                    .entry(family_id.clone())
                    .or_default()
                    .push((ci, canonical_index_by_row));
            }
        }
    }
    ensure!(
        groups.len() == proof.family_openings.len(),
        "FAC opening family count"
    );
    for ((family_id, claims), opening) in groups.into_iter().zip(&proof.family_openings) {
        ensure!(opening.family_id == family_id, "FAC family order mismatch");
        let spec = registry.tensor(&family_id)?;
        let mut weights = vec![Scalar::zero(); spec.logical_len];
        let mut claim = Scalar::zero();
        for (k, (ci, map)) in claims.iter().enumerate() {
            transcript.u64(b"fac-claim-index", k as u64);
            let rho = transcript.challenge("fac-line8-rho");
            claim += rho * proof.terminal_columns[*ci];
            for q in 0..op.rows {
                weights[map.at(q)?] += rho * basis[q];
            }
        }
        ensure!(
            opening.combined_claim == claim && opening.proof.claimed_value == claim,
            "FAC combined claim rejected"
        );
        verify_evaluation(
            pp,
            &format!("{}:fac-line8:{}", op.family_id, family_id),
            tensor_commitments
                .get(&family_id)
                .ok_or_else(|| anyhow::anyhow!("missing family commitment"))?,
            &weights,
            &opening.proof,
            transcript,
        )?;
    }
    let mask_weights = eq_weights(&rs[1..]);
    ensure!(
        proof.mask_opening.claimed_value == proof.mask_terminal,
        "mask opening value mismatch"
    );
    verify_evaluation(
        pp,
        &format!("{}:sumcheck-mask", op.family_id),
        &proof.mask_commitment,
        &mask_weights,
        &proof.mask_opening,
        transcript,
    )?;
    ensure!(
        proof.transcript_digest == transcript.digest(),
        "operation transcript digest mismatch"
    );
    Ok(())
}
