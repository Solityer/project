use crate::algebra::Scalar;
use crate::hyrax::{commit, commit_with_backend, CommitmentParameters, VectorCommitment};
use crate::prover_backend::{GpuTelemetry,ProverBackend};
use crate::registry::{RelationRegistry, Visibility, WitnessPackage};
use crate::sumcheck::{prove_operation,verify_operation,OperationProof};
use crate::transcript::Transcript;
use anyhow::{ensure,Context,Result};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const PROOF_FORMAT:&str="zkDL-GAT-proof-envelope-v2";

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct PublicStatement {
    pub protocol_id:String,pub proof_task:String,pub privacy_policy:String,pub input_privacy:String,pub output_mode:String,
    pub dataset:String,pub pp_id:String,pub registry:RelationRegistry,
    pub graph_features_digest:String,pub topology_digest:String,pub quant_cfg_id:String,pub static_tables_id:String,pub integer_bound_ledger_id:String,
    pub model_arch_id:String,pub model_param_id:String,pub public_logits:Vec<i64>,pub statement_id:String,
}
impl PublicStatement{
    pub fn canonical_digest(&self)->Result<[u8;32]>{let mut x=self.clone();x.statement_id.clear();let bytes=bincode::serialize(&x)?;let mut h=Sha256::new();h.update(b"zkDL-GAT-public-statement-v1\0");h.update(bytes);Ok(h.finalize().into())}
    pub fn seal(mut self)->Result<Self>{self.registry.validate()?;self.statement_id=hex::encode(self.canonical_digest()?);self.validate()?;Ok(self)}
    pub fn validate(&self)->Result<()>{ensure!(self.protocol_id=="GAT-ZKML-zkDL"&&self.proof_task=="node_classification_full_gat","statement task mismatch");ensure!(self.privacy_policy=="model_and_trace_hidden"&&self.input_privacy=="public_graph_public_features"&&self.output_mode=="public_logits","statement privacy/output mismatch");ensure!(!self.public_logits.is_empty(),"public logits missing");ensure!(self.statement_id==hex::encode(self.canonical_digest()?),"statement digest mismatch");self.registry.validate()?;Ok(())}
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct ModelRegistry {
    pub format:String,pub pp_id:String,pub registry_id:String,pub model_arch_id:String,pub model_param_id:String,pub certified_bounds_id:String,pub commitments:BTreeMap<String,VectorCommitment>,pub registry_object_id:String,
}
impl ModelRegistry{
    fn digest(&self)->Result<[u8;32]>{let mut x=self.clone();x.registry_object_id.clear();let b=bincode::serialize(&x)?;let mut h=Sha256::new();h.update(b"zkDL-GAT-model-registry-v1\0");h.update(b);Ok(h.finalize().into())}
    fn seal(mut self)->Result<Self>{self.registry_object_id=hex::encode(self.digest()?);Ok(self)}
    pub fn validate(&self,pp:&CommitmentParameters,s:&PublicStatement)->Result<()>{ensure!(self.format=="zkDL-GAT-registered-model-v1"&&self.pp_id==pp.parameters_id&&self.registry_id==s.registry.registry_id,"model registry context mismatch");ensure!(self.model_arch_id==s.model_arch_id&&self.model_param_id==s.model_param_id,"model identity mismatch");ensure!(self.registry_object_id==hex::encode(self.digest()?),"model registry digest mismatch");for t in &s.registry.tensor_families{if t.visibility==Visibility::RegisteredModel{let c=self.commitments.get(&t.family_id).ok_or_else(||anyhow::anyhow!("missing registered model family"))?;ensure!(c.logical_len==t.logical_len,"model commitment shape mismatch");}}Ok(())}
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct ModelOpeningSecret {pub registry_object_id:String,pub blinders:BTreeMap<String,Scalar>}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct ProofEnvelope {
    pub format:String,pub statement_id:String,pub model_registry_id:String,pub device:String,pub fresh_nonce:[u8;32],
    pub dynamic_commitments:BTreeMap<String,VectorCommitment>,pub operations:Vec<OperationProof>,pub gpu_telemetry:Option<GpuTelemetry>,pub transcript_digest:[u8;32],
}

fn absorb_gpu_telemetry(transcript:&mut Transcript,device:&str,telemetry:&Option<GpuTelemetry>)->Result<()> {
    transcript.label("formal-prover-device-telemetry-v1");transcript.label(device);
    match telemetry { Some(value)=>transcript.bytes(b"gpu-telemetry",&bincode::serialize(value)?),None=>transcript.bytes(b"gpu-telemetry",b"cpu-no-gpu-telemetry") }
    Ok(())
}

fn validate_gpu_telemetry(device:&str,telemetry:&Option<GpuTelemetry>)->Result<()> {
    if device=="cpu" { ensure!(telemetry.is_none(),"CPU proof carries GPU telemetry");return Ok(()); }
    let value=telemetry.as_ref().ok_or_else(||anyhow::anyhow!("GPU proof is missing native kernel telemetry"))?;
    ensure!(value.provider_id=="zkDL-native-CUDA-BLS12-381-v1"&&value.probe_kernel_invocations==1&&value.cross_backend_kat_passed,"GPU provider identity/probe/KAT telemetry rejected");
    ensure!(value.kat_vectors_digest.len()==64&&value.kat_vectors_digest.bytes().all(|x|x.is_ascii_hexdigit()),"GPU cross-backend KAT digest rejected");
    ensure!(value.msm_kernel_invocations>0&&value.field_inner_product_invocations>0&&value.scalar_fold_invocations>0&&value.point_fold_invocations>0&&value.sumcheck_kernel_invocations>0,"GPU proof does not attest full commitment/IPA/Sumcheck kernel coverage");
    ensure!(!value.device_name.is_empty()&&!value.compute_capability.is_empty(),"GPU device identity missing");Ok(())
}

fn begin_transcript(s:&PublicStatement,m:&ModelRegistry,nonce:[u8;32],device:&str,commitments:&BTreeMap<String,VectorCommitment>)->Transcript{
    let mut t=Transcript::new(PROOF_FORMAT);t.bytes(b"statement-id",s.statement_id.as_bytes());t.bytes(b"model-registry-id",m.registry_object_id.as_bytes());t.bytes(b"proof-nonce",&nonce);t.label(device);
    for family in &s.registry.tensor_families{if family.visibility!=Visibility::Public{let c=commitments.get(&family.family_id).expect("validated commitment map");t.label(&family.family_id);t.u64(b"tensor-len",c.logical_len as u64);t.point(b"tensor-commitment",c.point);}}
    t
}

pub fn register_model(pp:&CommitmentParameters,s:&PublicStatement,w:&WitnessPackage,certified_bounds_id:String)->Result<(ModelRegistry,ModelOpeningSecret)>{pp.validate()?;s.validate()?;w.validate(&s.registry)?;let mut commitments=BTreeMap::new();let mut blinders=BTreeMap::new();for family in &s.registry.tensor_families{if family.visibility==Visibility::RegisteredModel{let r=Scalar::random()?;let(c,_)=commit(pp,w.tensors[&family.family_id].clone(),r)?;commitments.insert(family.family_id.clone(),c);blinders.insert(family.family_id.clone(),r);}}
    let registry=ModelRegistry{format:"zkDL-GAT-registered-model-v1".into(),pp_id:pp.parameters_id.clone(),registry_id:s.registry.registry_id.clone(),model_arch_id:s.model_arch_id.clone(),model_param_id:s.model_param_id.clone(),certified_bounds_id,commitments,registry_object_id:String::new()}.seal()?;let secret=ModelOpeningSecret{registry_object_id:registry.registry_object_id.clone(),blinders};Ok((registry,secret))}

pub fn prove(pp:&CommitmentParameters,s:&PublicStatement,m:&ModelRegistry,secret:&ModelOpeningSecret,w:&WitnessPackage,device:&str)->Result<ProofEnvelope>{pp.validate()?;s.validate()?;m.validate(pp,s)?;w.validate(&s.registry)?;ensure!(secret.registry_object_id==m.registry_object_id,"model opening secret mismatch");ensure!(device=="cpu"||device=="gpu","device must be cpu|gpu");
    let backend=ProverBackend::open(device)?;
    let mut all=BTreeMap::new();let mut openings=BTreeMap::new();let mut dynamic=BTreeMap::new();for family in &s.registry.tensor_families{match family.visibility{Visibility::Public=>{},Visibility::RegisteredModel=>{let r=*secret.blinders.get(&family.family_id).ok_or_else(||anyhow::anyhow!("missing model blinder"))?;let(c,cw)=commit_with_backend(pp,w.tensors[&family.family_id].clone(),r,&backend)?;ensure!(c.point.eq_checked(m.commitments[&family.family_id].point)?,"model witness does not open registered commitment");all.insert(family.family_id.clone(),c);openings.insert(family.family_id.clone(),cw);},Visibility::DynamicPrivate=>{let(c,cw)=commit_with_backend(pp,w.tensors[&family.family_id].clone(),Scalar::random()?,&backend)?;dynamic.insert(family.family_id.clone(),c.clone());all.insert(family.family_id.clone(),c);openings.insert(family.family_id.clone(),cw);}}}
    let mut nonce=[0u8;32];getrandom::getrandom(&mut nonce).map_err(|e|anyhow::anyhow!("OS randomness failed: {e}"))?;let mut transcript=begin_transcript(s,m,nonce,device,&all);let mut operations=Vec::with_capacity(s.registry.operation_families.len());for op in &s.registry.operation_families{operations.push(prove_operation(pp,&s.registry,op,w,&all,&openings,&mut transcript,&backend)?);}let gpu_telemetry=backend.finish_gpu()?;validate_gpu_telemetry(device,&gpu_telemetry)?;absorb_gpu_telemetry(&mut transcript,device,&gpu_telemetry)?;let transcript_digest=transcript.digest();Ok(ProofEnvelope{format:PROOF_FORMAT.into(),statement_id:s.statement_id.clone(),model_registry_id:m.registry_object_id.clone(),device:device.into(),fresh_nonce:nonce,dynamic_commitments:dynamic,operations,gpu_telemetry,transcript_digest})}

pub fn verify(pp:&CommitmentParameters,s:&PublicStatement,m:&ModelRegistry,p:&ProofEnvelope,expected_device:Option<&str>)->Result<()> {pp.validate()?;s.validate()?;m.validate(pp,s)?;ensure!(p.format==PROOF_FORMAT&&p.statement_id==s.statement_id&&p.model_registry_id==m.registry_object_id,"proof envelope context mismatch");if let Some(d)=expected_device{ensure!(p.device==d,"proof device mismatch");}ensure!(p.device=="cpu"||p.device=="gpu","invalid proof device");validate_gpu_telemetry(&p.device,&p.gpu_telemetry)?;let mut all=m.commitments.clone();for family in &s.registry.tensor_families{if family.visibility==Visibility::DynamicPrivate{let c=p.dynamic_commitments.get(&family.family_id).ok_or_else(||anyhow::anyhow!("missing dynamic commitment"))?;ensure!(c.logical_len==family.logical_len,"dynamic commitment shape mismatch");all.insert(family.family_id.clone(),c.clone());}}ensure!(p.dynamic_commitments.len()==s.registry.tensor_families.iter().filter(|x|x.visibility==Visibility::DynamicPrivate).count(),"unexpected dynamic commitment");ensure!(p.operations.len()==s.registry.operation_families.len(),"operation proof count mismatch");let mut transcript=begin_transcript(s,m,p.fresh_nonce,&p.device,&all);for(op,proof)in s.registry.operation_families.iter().zip(&p.operations){verify_operation(pp,&s.registry,op,&all,proof,&mut transcript)?;}absorb_gpu_telemetry(&mut transcript,&p.device,&p.gpu_telemetry)?;ensure!(p.transcript_digest==transcript.digest(),"proof transcript digest mismatch");Ok(())}

pub fn read_bin<T:for<'a>Deserialize<'a>>(path:&Path)->Result<T>{let bytes=fs::read(path).with_context(||format!("read {}",path.display()))?;Ok(bincode::deserialize(&bytes).with_context(||format!("parse {}",path.display()))?)}
pub fn write_bin<T:Serialize>(path:&Path,value:&T,private:bool)->Result<()> {if let Some(parent)=path.parent(){fs::create_dir_all(parent)?;}let bytes=bincode::serialize(value)?;fs::write(path,&bytes)?;#[cfg(unix)]if private{use std::os::unix::fs::PermissionsExt;fs::set_permissions(path,fs::Permissions::from_mode(0o600))?;}Ok(())}
