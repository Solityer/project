use crate::algebra::{msm, Point, Scalar};
use crate::prover_backend::ProverBackend;
use crate::transcript::Transcript;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentParameters {
    pub protocol: String,
    pub generator_seed: [u8; 32],
    pub max_vector_len: usize,
    pub security_bits: u32,
    pub parameters_id: String,
}

impl CommitmentParameters {
    pub fn new(seed: [u8; 32], max_vector_len: usize) -> Result<Self> {
        ensure!(max_vector_len > 0, "PP max_vector_len must be positive");
        let mut out = Self {
            protocol: "zkDL-hiding-Hyrax-IPA-BLS12-381-v1".into(),
            generator_seed: seed,
            max_vector_len,
            security_bits: 128,
            parameters_id: String::new(),
        };
        out.parameters_id = hex::encode(out.canonical_digest());
        // Eagerly validate distinct, non-identity base domains.
        out.g(0).checked()?;
        out.h(0).checked()?;
        out.u().checked()?;
        ensure!(
            !out.g(0).eq_checked(out.h(0))?
                && !out.g(0).eq_checked(out.u())?
                && !out.h(0).eq_checked(out.u())?,
            "PP generator domains collided"
        );
        Ok(out)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.protocol == "zkDL-hiding-Hyrax-IPA-BLS12-381-v1",
            "unsupported PP protocol"
        );
        ensure!(
            self.security_bits == 128 && self.max_vector_len > 0,
            "invalid PP security/size"
        );
        ensure!(
            self.parameters_id == hex::encode(self.canonical_digest()),
            "PP digest mismatch"
        );
        self.g(0).checked()?;
        self.h(0).checked()?;
        self.u().checked()?;
        Ok(())
    }
    fn canonical_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"zkDL-Hyrax-PP-v1\0");
        h.update(self.protocol.as_bytes());
        h.update(self.generator_seed);
        h.update((self.max_vector_len as u64).to_be_bytes());
        h.update(self.security_bits.to_be_bytes());
        h.finalize().into()
    }
    fn point(&self, family: &[u8], index: usize) -> Point {
        let mut msg = Vec::with_capacity(32 + family.len() + 8);
        msg.extend_from_slice(&self.generator_seed);
        msg.extend_from_slice(family);
        msg.extend_from_slice(&(index as u64).to_be_bytes());
        Point::hash_to_curve(b"BLS12381G1_XMD:SHA-256_SSWU_RO_ZKDL_", &msg)
    }
    pub fn g(&self, i: usize) -> Point {
        self.point(b"G", i)
    }
    pub fn h(&self, i: usize) -> Point {
        self.point(b"H", i)
    }
    pub fn u(&self) -> Point {
        self.point(b"U", 0)
    }
    fn proof_len(&self, logical_len: usize) -> Result<usize> {
        ensure!(
            logical_len > 0 && logical_len <= self.max_vector_len,
            "Hyrax vector exceeds PP"
        );
        Ok((logical_len + 1).next_power_of_two())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorCommitment {
    pub logical_len: usize,
    pub point: Point,
}

#[derive(Clone, Debug)]
pub struct CommitmentWitness {
    pub values: Vec<Scalar>,
    pub blinder: Scalar,
}

impl Drop for CommitmentWitness {
    fn drop(&mut self) {
        for v in &mut self.values {
            v.0.fill(0);
        }
        self.blinder.0.fill(0);
    }
}

pub fn commit(
    pp: &CommitmentParameters,
    values: Vec<Scalar>,
    blinder: Scalar,
) -> Result<(VectorCommitment, CommitmentWitness)> {
    let n = pp.proof_len(values.len())?;
    let mut scalars = values.clone();
    scalars.push(blinder);
    scalars.resize(n, Scalar::zero());
    let points = (0..n).map(|i| pp.g(i)).collect::<Vec<_>>();
    let point = msm(&points, &scalars)?;
    Ok((
        VectorCommitment {
            logical_len: values.len(),
            point,
        },
        CommitmentWitness { values, blinder },
    ))
}

pub fn commit_with_backend(
    pp: &CommitmentParameters,
    values: Vec<Scalar>,
    blinder: Scalar,
    backend: &ProverBackend,
) -> Result<(VectorCommitment, CommitmentWitness)> {
    let n = pp.proof_len(values.len())?;
    let mut scalars = values.clone();
    scalars.push(blinder);
    scalars.resize(n, Scalar::zero());
    let points = (0..n).map(|i| pp.g(i)).collect::<Vec<_>>();
    let point = backend.msm(&points, &scalars)?;
    Ok((
        VectorCommitment {
            logical_len: values.len(),
            point,
        },
        CommitmentWitness { values, blinder },
    ))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationProof {
    pub logical_len: usize,
    pub claimed_value: Scalar,
    pub l: Vec<Point>,
    pub r: Vec<Point>,
    pub final_a: Scalar,
    pub final_b: Scalar,
    pub transcript_digest: [u8; 32],
}

fn absorb_eval_header(
    transcript: &mut Transcript,
    pp: &CommitmentParameters,
    label: &str,
    commitment: &VectorCommitment,
    weights: &[Scalar],
    claimed: Scalar,
) {
    transcript.label("hyrax-evaluation-v1");
    transcript.label(label);
    transcript.bytes(b"pp-id", pp.parameters_id.as_bytes());
    transcript.u64(b"logical-len", commitment.logical_len as u64);
    transcript.point(b"commitment", commitment.point);
    let mut h = Sha256::new();
    h.update(b"zkDL-Hyrax-weights-v1\0");
    for w in weights {
        h.update(w.0);
    }
    transcript.bytes(b"weights-digest", &h.finalize());
    transcript.scalar(b"claimed-value", claimed);
}

pub fn prove_evaluation(
    pp: &CommitmentParameters,
    label: &str,
    commitment: &VectorCommitment,
    witness: &CommitmentWitness,
    weights: &[Scalar],
    transcript: &mut Transcript,
    backend: &ProverBackend,
) -> Result<EvaluationProof> {
    pp.validate()?;
    ensure!(
        witness.values.len() == commitment.logical_len && weights.len() == commitment.logical_len,
        "Hyrax witness/weight shape mismatch"
    );
    let (recomputed, _) =
        commit_with_backend(pp, witness.values.clone(), witness.blinder, backend)?;
    ensure!(
        recomputed.point.eq_checked(commitment.point)?,
        "Hyrax commitment witness mismatch"
    );
    let claimed = backend.inner_product(&witness.values, weights)?;
    absorb_eval_header(transcript, pp, label, commitment, weights, claimed);
    let n = pp.proof_len(commitment.logical_len)?;
    let mut a = witness.values.clone();
    a.push(witness.blinder);
    a.resize(n, Scalar::zero());
    let mut b = weights.to_vec();
    b.push(Scalar::zero());
    b.resize(n, Scalar::zero());
    let mut g = (0..n).map(|i| pp.g(i)).collect::<Vec<_>>();
    let mut h = (0..n).map(|i| pp.h(i)).collect::<Vec<_>>();
    let u = pp.u();
    let mut proof = EvaluationProof {
        logical_len: commitment.logical_len,
        claimed_value: claimed,
        l: vec![],
        r: vec![],
        final_a: Scalar::zero(),
        final_b: Scalar::zero(),
        transcript_digest: [0; 32],
    };
    while a.len() > 1 {
        let half = a.len() / 2;
        let (a_l, a_r) = a.split_at(half);
        let (b_l, b_r) = b.split_at(half);
        let (g_l, g_r) = g.split_at(half);
        let (h_l, h_r) = h.split_at(half);
        let c_l = backend.inner_product(a_l, b_r)?;
        let c_r = backend.inner_product(a_r, b_l)?;
        let l = backend.msm(&[g_r, h_l, &[u]].concat(), &[a_l, b_r, &[c_l]].concat())?;
        let r = backend.msm(&[g_l, h_r, &[u]].concat(), &[a_r, b_l, &[c_r]].concat())?;
        transcript.point(b"ipa-L", l);
        transcript.point(b"ipa-R", r);
        let x = transcript.challenge("hyrax-ipa-round");
        let x_inv = x.inverse()?;
        let na = backend.fold_scalars(&a, x, x_inv)?;
        let nb = backend.fold_scalars(&b, x_inv, x)?;
        let ng = backend.fold_points(&g, x_inv, x)?;
        let nh = backend.fold_points(&h, x, x_inv)?;
        proof.l.push(l);
        proof.r.push(r);
        a = na;
        b = nb;
        g = ng;
        h = nh;
    }
    proof.final_a = a[0];
    proof.final_b = b[0];
    transcript.scalar(b"ipa-final-a", proof.final_a);
    transcript.scalar(b"ipa-final-b", proof.final_b);
    proof.transcript_digest = transcript.digest();
    Ok(proof)
}

pub fn verify_evaluation(
    pp: &CommitmentParameters,
    label: &str,
    commitment: &VectorCommitment,
    weights: &[Scalar],
    proof: &EvaluationProof,
    transcript: &mut Transcript,
) -> Result<()> {
    pp.validate()?;
    ensure!(
        commitment.logical_len == proof.logical_len && weights.len() == commitment.logical_len,
        "Hyrax proof shape mismatch"
    );
    let n = pp.proof_len(commitment.logical_len)?;
    ensure!(
        proof.l.len() == n.trailing_zeros() as usize && proof.r.len() == proof.l.len(),
        "Hyrax IPA round count mismatch"
    );
    absorb_eval_header(
        transcript,
        pp,
        label,
        commitment,
        weights,
        proof.claimed_value,
    );
    let mut b = weights.to_vec();
    b.push(Scalar::zero());
    b.resize(n, Scalar::zero());
    let mut g = (0..n).map(|i| pp.g(i)).collect::<Vec<_>>();
    let mut h = (0..n).map(|i| pp.h(i)).collect::<Vec<_>>();
    let u = pp.u();
    let bh = msm(&h, &b)?;
    let mut p = commitment.point.add(bh)?.add(u.mul(proof.claimed_value)?)?;
    for (&l, &r) in proof.l.iter().zip(&proof.r) {
        transcript.point(b"ipa-L", l);
        transcript.point(b"ipa-R", r);
        let x = transcript.challenge("hyrax-ipa-round");
        let x_inv = x.inverse()?;
        p = l.mul(x * x)?.add(p)?.add(r.mul(x_inv * x_inv)?)?;
        let half = b.len() / 2;
        let (b_l, b_r) = b.split_at(half);
        let (g_l, g_r) = g.split_at(half);
        let (h_l, h_r) = h.split_at(half);
        let mut nb = Vec::with_capacity(half);
        let mut ng = Vec::with_capacity(half);
        let mut nh = Vec::with_capacity(half);
        for i in 0..half {
            nb.push(b_l[i] * x_inv + b_r[i] * x);
            ng.push(g_l[i].mul(x_inv)?.add(g_r[i].mul(x)?)?);
            nh.push(h_l[i].mul(x)?.add(h_r[i].mul(x_inv)?)?);
        }
        b = nb;
        g = ng;
        h = nh;
    }
    ensure!(
        b[0] == proof.final_b,
        "Hyrax folded public coefficient mismatch"
    );
    let rhs = g[0]
        .mul(proof.final_a)?
        .add(h[0].mul(proof.final_b)?)?
        .add(u.mul(proof.final_a * proof.final_b)?)?;
    ensure!(p.eq_checked(rhs)?, "Hyrax evaluation opening rejected");
    transcript.scalar(b"ipa-final-a", proof.final_a);
    transcript.scalar(b"ipa-final-b", proof.final_b);
    ensure!(
        proof.transcript_digest == transcript.digest(),
        "Hyrax transcript digest mismatch"
    );
    Ok(())
}
