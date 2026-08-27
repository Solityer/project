use crate::algebra::{msm, Point, Scalar};
use crate::prover_backend::ProverBackend;
use crate::transcript::Transcript;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentParameters {
    /// Stable public schema for the serialized baseline-local parameter file.
    /// This is intentionally part of the authenticated object rather than a
    /// convention kept only in the runner.
    pub schema: String,
    pub setup_id: String,
    pub security_id: String,
    pub protocol: String,
    pub generator_seed: [u8; 32],
    pub max_vector_len: usize,
    pub security_bits: u32,
    /// The untagged digest used by the native CLI/resource manifest.
    pub pp_id: String,
    /// Tagged spelling of the same authenticated PP identity.
    pub digest: String,
    /// Kept as an internal compatibility alias for the existing proof code.
    pub parameters_id: String,
}

pub const PP_SCHEMA: &str = "zkDL_gat_pp_info_v1";
pub const PP_SETUP_ID: &str = "zkDL-GAT-transparent-PP-v1";
pub const PP_SECURITY_ID: &str = "zkdl_gat_bls12_381_multi_session_v2";

impl CommitmentParameters {
    pub fn new(seed: [u8; 32], max_vector_len: usize) -> Result<Self> {
        ensure!(max_vector_len > 0, "PP max_vector_len must be positive");
        let mut out = Self {
            schema: PP_SCHEMA.into(),
            setup_id: PP_SETUP_ID.into(),
            security_id: PP_SECURITY_ID.into(),
            protocol: "zkDL-hiding-Hyrax-IPA-BLS12-381-v1".into(),
            generator_seed: seed,
            max_vector_len,
            security_bits: 128,
            pp_id: String::new(),
            digest: String::new(),
            parameters_id: String::new(),
        };
        let identity = hex::encode(out.canonical_digest());
        out.pp_id = identity.clone();
        out.parameters_id = identity.clone();
        out.digest = format!("sha256:{identity}");
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
            self.schema == PP_SCHEMA
                && self.setup_id == PP_SETUP_ID
                && self.security_id == PP_SECURITY_ID,
            "unsupported PP schema/setup/security identity"
        );
        ensure!(
            self.protocol == "zkDL-hiding-Hyrax-IPA-BLS12-381-v1",
            "unsupported PP protocol"
        );
        ensure!(
            self.security_bits == 128
                && self.max_vector_len > 0
                && self.max_vector_len.is_power_of_two(),
            "invalid PP security/size"
        );
        ensure!(
            self.parameters_id == self.pp_id
                && self.pp_id == hex::encode(self.canonical_digest())
                && self.digest == format!("sha256:{}", self.pp_id),
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
        h.update((self.schema.len() as u64).to_be_bytes());
        h.update(self.schema.as_bytes());
        h.update((self.setup_id.len() as u64).to_be_bytes());
        h.update(self.setup_id.as_bytes());
        h.update((self.security_id.len() as u64).to_be_bytes());
        h.update(self.security_id.as_bytes());
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
    pub fn claim_g(&self, logical_len: usize) -> Point {
        self.point(b"CLAIM-G", logical_len)
    }
    pub fn claim_h(&self, logical_len: usize) -> Point {
        self.point(b"CLAIM-H", logical_len)
    }
    fn proof_len(&self, logical_len: usize) -> Result<usize> {
        ensure!(
            logical_len > 0 && logical_len <= self.max_vector_len,
            "Hyrax vector exceeds PP"
        );
        Ok((logical_len + 1).next_power_of_two())
    }

    /// Generator used for the dedicated Pedersen blinder slot.  The blinder
    /// is appended immediately after the logical vector and is therefore
    /// independent of all value generators used by the committed tensor.
    pub fn blinder_generator(&self, logical_len: usize) -> Result<Point> {
        self.proof_len(logical_len)?;
        let generator = self.g(logical_len);
        generator.checked()?;
        Ok(generator)
    }

    /// Re-randomize a registered model commitment without changing its
    /// committed vector.  The caller must retain the fresh delta as part of
    /// the proof-local opening witness.
    pub fn rerandomize_commitment(
        &self,
        commitment: &VectorCommitment,
        delta: Scalar,
    ) -> Result<VectorCommitment> {
        let blinder = self.blinder_generator(commitment.logical_len)?;
        Ok(VectorCommitment {
            logical_len: commitment.logical_len,
            point: commitment.point.add(blinder.mul(delta)?)?,
        })
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentEvaluationProof {
    pub logical_len: usize,
    pub claim_domain_len: usize,
    pub claim_commitment: Point,
    pub l: Vec<Point>,
    pub r: Vec<Point>,
    pub final_a: Scalar,
    pub final_b: Scalar,
    pub transcript_digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentLinkProof {
    pub nonce_commitment: Point,
    pub response: Scalar,
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

fn absorb_commitment_eval_header(
    transcript: &mut Transcript,
    pp: &CommitmentParameters,
    label: &str,
    base: &VectorCommitment,
    claim_domain_len: usize,
    claim_commitment: Point,
    weights: &[Scalar],
) {
    transcript.label("hyrax-commitment-valued-evaluation-v1");
    transcript.label(label);
    transcript.bytes(b"pp-id", pp.parameters_id.as_bytes());
    transcript.u64(b"logical-len", base.logical_len as u64);
    transcript.u64(b"claim-domain-len", claim_domain_len as u64);
    transcript.point(b"base-commitment", base.point);
    transcript.point(b"claim-commitment", claim_commitment);
    let mut h = Sha256::new();
    h.update(b"zkDL-Hyrax-weights-v2\0");
    for weight in weights {
        h.update(weight.0);
    }
    transcript.bytes(b"weights-digest", &h.finalize());
}

fn claim_generators(
    pp: &CommitmentParameters,
    logical_len: usize,
    claim_domain_len: usize,
) -> Result<(Vec<Point>, Vec<Point>, usize, usize)> {
    let n = pp.proof_len(logical_len)?;
    let ext_n = (n + 1).next_power_of_two();
    let mut g = (0..n).map(|i| pp.g(i)).collect::<Vec<_>>();
    let mut h = (0..n).map(|i| pp.h(i)).collect::<Vec<_>>();
    g.push(pp.claim_g(claim_domain_len));
    h.push(pp.claim_h(claim_domain_len));
    while g.len() < ext_n {
        let index = g.len();
        g.push(pp.g(index));
        h.push(pp.h(index));
    }
    Ok((g, h, n, ext_n))
}

pub fn prove_commitment_evaluation_with_domain(
    pp: &CommitmentParameters,
    label: &str,
    base_commitment: &VectorCommitment,
    witness: &CommitmentWitness,
    weights: &[Scalar],
    claim_domain_len: usize,
    transcript: &mut Transcript,
    backend: &ProverBackend,
) -> Result<(CommitmentEvaluationProof, Scalar)> {
    pp.validate()?;
    ensure!(
        witness.values.len() == base_commitment.logical_len
            && weights.len() == base_commitment.logical_len,
        "commitment-valued Hyrax witness/weight shape mismatch"
    );
    let (recomputed, _) =
        commit_with_backend(pp, witness.values.clone(), witness.blinder, backend)?;
    ensure!(
        recomputed.point.eq_checked(base_commitment.point)?,
        "commitment-valued Hyrax base commitment witness mismatch"
    );
    let value = backend.inner_product(&witness.values, weights)?;
    let claim_blinder = loop {
        let candidate = Scalar::random()?;
        if candidate != Scalar::zero() {
            break candidate;
        }
    };
    let claim_commitment = pp
        .u()
        .mul(value)?
        .add(pp.claim_g(claim_domain_len).mul(claim_blinder)?)?;
    absorb_commitment_eval_header(
        transcript,
        pp,
        label,
        base_commitment,
        claim_domain_len,
        claim_commitment,
        weights,
    );
    let (mut g, mut h, n, ext_n) =
        claim_generators(pp, base_commitment.logical_len, claim_domain_len)?;
    let mut a = witness.values.clone();
    a.push(witness.blinder);
    a.resize(n, Scalar::zero());
    a.push(claim_blinder);
    a.resize(ext_n, Scalar::zero());
    let mut b = weights.to_vec();
    b.push(Scalar::zero());
    b.resize(n, Scalar::zero());
    b.push(Scalar::zero());
    b.resize(ext_n, Scalar::zero());
    let u = pp.u();
    let mut proof = CommitmentEvaluationProof {
        logical_len: base_commitment.logical_len,
        claim_domain_len,
        claim_commitment,
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
        let x = transcript.challenge("hyrax-commitment-ipa-round");
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
    Ok((proof, claim_blinder))
}

pub fn verify_commitment_evaluation_with_domain(
    pp: &CommitmentParameters,
    label: &str,
    base_commitment: &VectorCommitment,
    weights: &[Scalar],
    claim_domain_len: usize,
    proof: &CommitmentEvaluationProof,
    transcript: &mut Transcript,
) -> Result<()> {
    pp.validate()?;
    ensure!(
        base_commitment.logical_len == proof.logical_len
            && weights.len() == base_commitment.logical_len,
        "commitment-valued Hyrax proof shape mismatch"
    );
    absorb_commitment_eval_header(
        transcript,
        pp,
        label,
        base_commitment,
        claim_domain_len,
        proof.claim_commitment,
        weights,
    );
    ensure!(
        proof.claim_domain_len == claim_domain_len,
        "commitment-valued claim domain mismatch"
    );
    let (mut g, mut h, n, ext_n) =
        claim_generators(pp, base_commitment.logical_len, claim_domain_len)?;
    ensure!(
        proof.l.len() == ext_n.trailing_zeros() as usize && proof.r.len() == proof.l.len(),
        "commitment-valued Hyrax IPA round count mismatch"
    );
    let mut b = weights.to_vec();
    b.push(Scalar::zero());
    b.resize(n, Scalar::zero());
    b.push(Scalar::zero());
    b.resize(ext_n, Scalar::zero());
    let u = pp.u();
    let bh = msm(&h, &b)?;
    let mut p = base_commitment.point.add(bh)?.add(proof.claim_commitment)?;
    for (&l, &r) in proof.l.iter().zip(&proof.r) {
        transcript.point(b"ipa-L", l);
        transcript.point(b"ipa-R", r);
        let x = transcript.challenge("hyrax-commitment-ipa-round");
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
        "commitment-valued folded coefficient mismatch"
    );
    let rhs = g[0]
        .mul(proof.final_a)?
        .add(h[0].mul(proof.final_b)?)?
        .add(u.mul(proof.final_a * proof.final_b)?)?;
    ensure!(
        p.eq_checked(rhs)?,
        "commitment-valued Hyrax opening rejected"
    );
    transcript.scalar(b"ipa-final-a", proof.final_a);
    transcript.scalar(b"ipa-final-b", proof.final_b);
    ensure!(
        proof.transcript_digest == transcript.digest(),
        "commitment-valued Hyrax transcript digest mismatch"
    );
    Ok(())
}
