use crate::algebra::{eq_weights, interpolate_at, Scalar};
use crate::hyrax::{
    commit_with_backend, prove_commitment_evaluation_with_domain, prove_evaluation,
    verify_commitment_evaluation_with_domain, verify_evaluation, CommitmentEvaluationProof,
    CommitmentLinkProof, CommitmentParameters, CommitmentWitness, EvaluationProof,
    VectorCommitment,
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
    pub terminal_claim: Scalar,
    pub canonical_proof: CommitmentEvaluationProof,
    pub padding_commitment: VectorCommitment,
    pub padding_proof: CommitmentEvaluationProof,
    pub link: CommitmentLinkProof,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationProof {
    pub family_id: String,
    pub mask_commitment: VectorCommitment,
    pub padding_commitments: Vec<Option<VectorCommitment>>,
    pub rounds: Vec<SumcheckRound>,
    /// Public columns carry their verifier-computed terminal value; private
    /// columns carry the value after an independent random padding polynomial
    /// has been added.  Canonical claims are never serialized here.
    pub masked_terminal_columns: Vec<Scalar>,
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
) -> Result<(Vec<Vec<Scalar>>, Vec<Vec<Scalar>>)> {
    let mut columns = Vec::with_capacity(op.columns.len());
    let mut padding = Vec::with_capacity(op.columns.len());
    for c in &op.columns {
        let (values, pads) = match &c.source {
            ColumnSource::Public { values } => (values.clone(), vec![Scalar::zero(); op.rows]),
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
                let mut values = Vec::with_capacity(op.rows);
                let mut pads = vec![Scalar::zero(); op.rows];
                for row in 0..op.rows {
                    if row < op.real_rows && spec.visibility == Visibility::Public {
                        values.push(t[canonical_index_by_row.at(row)?]);
                    } else if row < op.real_rows {
                        values.push(t[canonical_index_by_row.at(row)?]);
                    } else if spec.visibility == Visibility::Public {
                        values.push(Scalar::zero());
                    } else {
                        let value = Scalar::random()?;
                        values.push(value);
                        pads[row] = value;
                    }
                }
                (values, pads)
            }
        };
        ensure!(
            values.len() == op.rows,
            "operation column row shape mismatch"
        );
        columns.push(values);
        padding.push(pads);
    }
    Ok((columns, padding))
}
fn absorb_operation_header(t: &mut Transcript, op: &OperationFamilySpec, mask: &VectorCommitment) {
    t.label("zk-sumcheck-operation-v1");
    t.label(&op.family_id);
    t.label(&op.operation_kind);
    t.u64(b"rows", op.rows as u64);
    t.u64(b"degree", op.degree_bound as u64);
    t.point(b"mask-commitment", mask.point);
}

fn absorb_padding_headers(
    transcript: &mut Transcript,
    op: &OperationFamilySpec,
    commitments: &[Option<VectorCommitment>],
) -> Result<()> {
    ensure!(
        commitments.len() == op.columns.len(),
        "padding commitment column count mismatch"
    );
    transcript.label("zkDL-private-padding-commitments-v1");
    for (index, commitment) in commitments.iter().enumerate() {
        transcript.u64(b"padding-column-index", index as u64);
        match commitment {
            Some(value) => {
                ensure!(
                    value.logical_len == op.rows,
                    "padding commitment shape mismatch"
                );
                transcript.point(b"padding-commitment", value.point);
            }
            None => transcript.label("padding-public-column"),
        }
    }
    Ok(())
}

fn terminal_link_challenge(
    transcript: &mut Transcript,
    op: &OperationFamilySpec,
    family_id: &str,
    terminal_claim: Scalar,
    canonical: crate::algebra::Point,
    padding: crate::algebra::Point,
    nonce: crate::algebra::Point,
) -> Scalar {
    transcript.label("zkDL-terminal-claim-padding-link-v1");
    transcript.label(&op.family_id);
    transcript.label(family_id);
    transcript.u64(b"terminal-rows", op.rows as u64);
    transcript.u64(b"terminal-real-rows", op.real_rows as u64);
    transcript.scalar(b"masked-terminal-claim", terminal_claim);
    transcript.point(b"canonical-evaluation-commitment", canonical);
    transcript.point(b"padding-evaluation-commitment", padding);
    transcript.point(b"terminal-link-nonce", nonce);
    transcript.challenge("terminal-claim-padding-schnorr")
}

fn prove_terminal_link_with_pp(
    pp: &CommitmentParameters,
    transcript: &mut Transcript,
    op: &OperationFamilySpec,
    family_id: &str,
    terminal_claim: Scalar,
    canonical: crate::algebra::Point,
    padding: crate::algebra::Point,
    canonical_blinder: Scalar,
    padding_blinder: Scalar,
) -> Result<CommitmentLinkProof> {
    let generator = pp.claim_g(op.rows);
    let nonce = loop {
        let value = Scalar::random()?;
        if value != Scalar::zero() {
            break value;
        }
    };
    let nonce_commitment = generator.mul(nonce)?;
    let challenge = terminal_link_challenge(
        transcript,
        op,
        family_id,
        terminal_claim,
        canonical,
        padding,
        nonce_commitment,
    );
    Ok(CommitmentLinkProof {
        nonce_commitment,
        response: nonce + challenge * (canonical_blinder + padding_blinder),
    })
}

fn verify_terminal_link_with_pp(
    pp: &CommitmentParameters,
    transcript: &mut Transcript,
    op: &OperationFamilySpec,
    family_id: &str,
    terminal_claim: Scalar,
    canonical: crate::algebra::Point,
    padding: crate::algebra::Point,
    proof: &CommitmentLinkProof,
) -> Result<()> {
    let generator = pp.claim_g(op.rows);
    let challenge = terminal_link_challenge(
        transcript,
        op,
        family_id,
        terminal_claim,
        canonical,
        padding,
        proof.nonce_commitment,
    );
    let expected = proof.nonce_commitment.add(
        canonical
            .add(padding)?
            .add(pp.u().mul(-terminal_claim)?)?
            .mul(challenge)?,
    )?;
    ensure!(
        generator.mul(proof.response)?.eq_checked(expected)?,
        "terminal canonical/padding commitment link rejected"
    );
    Ok(())
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
    let (mut columns, padding_columns) = build_columns(registry, op, witness)?;
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
    // Every private column's random half is committed before any Fiat-Shamir
    // challenge is sampled.  The FAC rho challenges therefore cannot choose
    // an aggregate padding vector adaptively after seeing the sumcheck point.
    let mut padding_commitments = vec![None; op.columns.len()];
    let mut padding_blinders = vec![None; op.columns.len()];
    for (ci, column) in op.columns.iter().enumerate() {
        let private = match &column.source {
            ColumnSource::TensorView { family_id, .. } => {
                registry.tensor(family_id)?.visibility != Visibility::Public
            }
            ColumnSource::Public { .. } => false,
        };
        if private {
            let blinder = Scalar::random()?;
            let (commitment, _witness) =
                commit_with_backend(pp, padding_columns[ci].clone(), blinder, backend)?;
            padding_commitments[ci] = Some(commitment);
            padding_blinders[ci] = Some(blinder);
        }
    }
    absorb_operation_header(transcript, op, &mask_commitment);
    absorb_padding_headers(transcript, op, &padding_commitments)?;
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
    let masked_terminal_columns = columns.iter().map(|x| x[0]).collect::<Vec<_>>();
    let mask_terminal = mask[0];
    ensure!(
        running
            == beta[0] * eval_residuals(op, &masked_terminal_columns, &lambdas)?
                + factor * mask_terminal,
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
        let mut padding_values = vec![Scalar::zero(); op.rows];
        let mut terminal_claim = Scalar::zero();
        let mut aggregate_point = crate::algebra::Point::identity();
        let mut aggregate_blinder = Scalar::zero();
        for (k, (ci, map)) in claims.iter().enumerate() {
            transcript.u64(b"fac-claim-index", k as u64);
            let rho = transcript.challenge("fac-line8-rho");
            terminal_claim += rho * masked_terminal_columns[*ci];
            let commitment = padding_commitments[*ci]
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("private column is missing padding commitment"))?;
            let blinder = padding_blinders[*ci]
                .ok_or_else(|| anyhow::anyhow!("private column is missing padding blinder"))?;
            aggregate_point = aggregate_point.add(commitment.point.mul(rho)?)?;
            aggregate_blinder += rho * blinder;
            for q in 0..op.real_rows {
                weights[map.at(q)?] += rho * basis[q];
            }
            for q in op.real_rows..op.rows {
                padding_values[q] += rho * padding_columns[*ci][q];
            }
        }
        let c = tensor_commitments.get(&family_id).unwrap();
        let w = opening_witnesses.get(&family_id).unwrap();
        let (canonical_proof, canonical_blinder) = prove_commitment_evaluation_with_domain(
            pp,
            &format!("{}:fac-line8:{}", op.family_id, family_id),
            c,
            w,
            &weights,
            op.rows,
            transcript,
            backend,
        )?;
        let padding_commitment = VectorCommitment {
            logical_len: op.rows,
            point: aggregate_point,
        };
        let padding_witness = CommitmentWitness {
            values: padding_values,
            blinder: aggregate_blinder,
        };
        let (padding_proof, padding_eval_blinder) = prove_commitment_evaluation_with_domain(
            pp,
            &format!("{}:fac-line8-padding:{}", op.family_id, family_id),
            &padding_commitment,
            &padding_witness,
            &basis,
            op.rows,
            transcript,
            backend,
        )?;
        drop(padding_witness);
        let link = prove_terminal_link_with_pp(
            pp,
            transcript,
            op,
            &family_id,
            terminal_claim,
            canonical_proof.claim_commitment,
            padding_proof.claim_commitment,
            canonical_blinder,
            padding_eval_blinder,
        )?;
        family_openings.push(FamilyOpening {
            family_id,
            terminal_claim,
            canonical_proof,
            padding_commitment,
            padding_proof,
            link,
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
        padding_commitments,
        rounds,
        masked_terminal_columns,
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
    ensure!(
        proof.padding_commitments.len() == op.columns.len(),
        "padding commitment column count"
    );
    for (ci, column) in op.columns.iter().enumerate() {
        let private = match &column.source {
            ColumnSource::TensorView { family_id, .. } => {
                registry.tensor(family_id)?.visibility != Visibility::Public
            }
            ColumnSource::Public { .. } => false,
        };
        match (private, &proof.padding_commitments[ci]) {
            (true, Some(commitment)) => ensure!(
                commitment.logical_len == op.rows,
                "private padding commitment shape mismatch"
            ),
            (false, None) => {}
            (true, None) => ensure!(false, "private column missing padding commitment"),
            (false, Some(_)) => ensure!(false, "public column carries padding commitment"),
        }
    }
    absorb_operation_header(transcript, op, &proof.mask_commitment);
    absorb_padding_headers(transcript, op, &proof.padding_commitments)?;
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
        proof.masked_terminal_columns.len() == op.columns.len(),
        "terminal column count"
    );
    let beta = eq_weights(&tau);
    let beta_terminal = crate::algebra::inner_product(&beta, &eq_weights(&rs))?;
    let factor = rs[0] * (Scalar::one() - rs[0]);
    ensure!(
        running
            == beta_terminal * eval_residuals(op, &proof.masked_terminal_columns, &lambdas)?
                + factor * proof.mask_terminal,
        "sumcheck terminal rejected"
    );
    // Public columns are reconstructed locally; prover cannot choose their terminal value.
    let basis = eq_weights(&rs);
    for (ci, c) in op.columns.iter().enumerate() {
        match &c.source {
            ColumnSource::Public { values } => ensure!(
                crate::algebra::inner_product(values, &basis)? == proof.masked_terminal_columns[ci],
                "public terminal mismatch"
            ),
            ColumnSource::TensorView {
                family_id,
                canonical_index_by_row,
            } if registry.tensor(family_id)?.visibility == Visibility::Public => {
                let values = registry.public_tensors.get(family_id).unwrap();
                let view = (0..op.rows)
                    .map(|row| {
                        if row < op.real_rows {
                            Ok(values[canonical_index_by_row.at(row)?])
                        } else {
                            Ok(Scalar::zero())
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                ensure!(
                    crate::algebra::inner_product(&view, &basis)?
                        == proof.masked_terminal_columns[ci],
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
        let mut terminal_claim = Scalar::zero();
        let mut expected_padding_point = crate::algebra::Point::identity();
        for (k, (ci, map)) in claims.iter().enumerate() {
            transcript.u64(b"fac-claim-index", k as u64);
            let rho = transcript.challenge("fac-line8-rho");
            terminal_claim += rho * proof.masked_terminal_columns[*ci];
            let commitment = proof.padding_commitments[*ci]
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("private column is missing padding commitment"))?;
            expected_padding_point = expected_padding_point.add(commitment.point.mul(rho)?)?;
            for q in 0..op.real_rows {
                weights[map.at(q)?] += rho * basis[q];
            }
        }
        ensure!(
            opening.terminal_claim == terminal_claim,
            "FAC combined claim rejected"
        );
        let commitment = tensor_commitments
            .get(&family_id)
            .ok_or_else(|| anyhow::anyhow!("missing family commitment"))?;
        ensure!(
            opening.padding_commitment.logical_len == op.rows,
            "FAC padding commitment shape mismatch"
        );
        ensure!(
            opening
                .padding_commitment
                .point
                .eq_checked(expected_padding_point)?,
            "FAC padding commitment is not the rho aggregate of column commitments"
        );
        verify_commitment_evaluation_with_domain(
            pp,
            &format!("{}:fac-line8:{}", op.family_id, family_id),
            commitment,
            &weights,
            op.rows,
            &opening.canonical_proof,
            transcript,
        )?;
        verify_commitment_evaluation_with_domain(
            pp,
            &format!("{}:fac-line8-padding:{}", op.family_id, family_id),
            &opening.padding_commitment,
            &basis,
            op.rows,
            &opening.padding_proof,
            transcript,
        )?;
        verify_terminal_link_with_pp(
            pp,
            transcript,
            op,
            &family_id,
            terminal_claim,
            opening.canonical_proof.claim_commitment,
            opening.padding_proof.claim_commitment,
            &opening.link,
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
