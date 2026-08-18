use crate::algebra::Scalar;
use crate::protocol::PublicStatement;
use crate::registry::*;
use anyhow::{ensure,Result};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::collections::BTreeMap;

pub const CANONICAL_ADAPTER_SCHEMA:&str="zkDL_gat_canonical_adapter_v2";

#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(untagged)]pub enum FieldValue{Signed(i64),CanonicalHex(String)}
impl FieldValue{fn field(&self)->Result<Scalar>{match self{Self::Signed(v)=>Ok(Scalar::from_i64(*v)),Self::CanonicalHex(v)=>Scalar::from_be_hex(v)}}}
#[derive(Clone,Debug,Serialize,Deserialize)]pub enum IntegerViewIndex{Explicit(Vec<u32>),Affine{offset:u32,stride:u32},PaddedAffine{offset:u32,stride:u32,logical_rows:u32,padding_index:u32}}
#[derive(Clone,Debug,Serialize,Deserialize)]pub enum IntegerColumnSource{TensorView{family_id:String,canonical_index_by_row:IntegerViewIndex},Public{values:Vec<FieldValue>}}
#[derive(Clone,Debug,Serialize,Deserialize)]pub struct IntegerOperationColumn{pub column_id:String,pub source:IntegerColumnSource}
#[derive(Clone,Debug,Serialize,Deserialize)]pub enum IntegerExpr{Const(FieldValue),Column(u32),Add(Box<IntegerExpr>,Box<IntegerExpr>),Sub(Box<IntegerExpr>,Box<IntegerExpr>),Mul(Box<IntegerExpr>,Box<IntegerExpr>),PublicTableLookup{values:Vec<FieldValue>,bit_columns:Vec<u32>}}
impl IntegerExpr{fn field(&self)->Result<Expr>{Ok(match self{Self::Const(v)=>Expr::Const(v.field()?),Self::Column(i)=>Expr::Column(*i),Self::Add(a,b)=>Expr::Add(Box::new(a.field()?),Box::new(b.field()?)),Self::Sub(a,b)=>Expr::Sub(Box::new(a.field()?),Box::new(b.field()?)),Self::Mul(a,b)=>Expr::Mul(Box::new(a.field()?),Box::new(b.field()?)),Self::PublicTableLookup{values,bit_columns}=>Expr::PublicTableLookup{values:values.iter().map(FieldValue::field).collect::<Result<Vec<_>>>()?,bit_columns:bit_columns.clone()}})}}
#[derive(Clone,Debug,Serialize,Deserialize)]pub struct IntegerOperationFamily{pub family_id:String,pub operation_kind:String,pub rows:usize,pub degree_bound:usize,pub columns:Vec<IntegerOperationColumn>,pub residuals:Vec<IntegerExpr>,pub requires_line8:bool}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct RelationCompilerInput{
 pub schema:String,pub dataset:String,pub pp_id:String,pub graph_features_digest:String,pub topology_digest:String,pub quant_cfg_id:String,pub static_tables_id:String,pub integer_bound_ledger_id:String,pub model_arch_id:String,pub model_param_id:String,pub public_logits:Vec<i64>,pub tensor_families:Vec<TensorFamilySpec>,pub operation_families:Vec<IntegerOperationFamily>,pub tensors:BTreeMap<String,Vec<FieldValue>>,pub source_input_id:String,
}

fn source_digest(input:&RelationCompilerInput)->Result<String>{let mut copy=input.clone();copy.source_input_id.clear();let bytes=serde_json::to_vec(&copy)?;let mut h=Sha256::new();h.update(b"zkDL-GAT-canonical-adapter-input-v2\0");h.update(bytes);Ok(format!("sha256:{}",hex::encode(h.finalize())))}

pub fn compile(input:RelationCompilerInput)->Result<(PublicStatement,WitnessPackage)>{
 ensure!(input.schema==CANONICAL_ADAPTER_SCHEMA,"only the baseline-local canonical GAT adapter schema is accepted");ensure!(input.source_input_id==source_digest(&input)?,"canonical adapter source digest mismatch");
 let operations=input.operation_families.iter().map(|o|->Result<OperationFamilySpec>{Ok(OperationFamilySpec{family_id:o.family_id.clone(),operation_kind:o.operation_kind.clone(),rows:o.rows,degree_bound:o.degree_bound,columns:o.columns.iter().map(|c|->Result<OperationColumn>{Ok(OperationColumn{column_id:c.column_id.clone(),source:match &c.source{IntegerColumnSource::TensorView{family_id,canonical_index_by_row}=>ColumnSource::TensorView{family_id:family_id.clone(),canonical_index_by_row:match canonical_index_by_row{IntegerViewIndex::Explicit(v)=>ViewIndex::Explicit(v.clone()),IntegerViewIndex::Affine{offset,stride}=>ViewIndex::Affine{offset:*offset,stride:*stride},IntegerViewIndex::PaddedAffine{offset,stride,logical_rows,padding_index}=>ViewIndex::PaddedAffine{offset:*offset,stride:*stride,logical_rows:*logical_rows,padding_index:*padding_index}}},IntegerColumnSource::Public{values}=>ColumnSource::Public{values:values.iter().map(FieldValue::field).collect::<Result<Vec<_>>>()?}}})}).collect::<Result<Vec<_>>>()?,residuals:o.residuals.iter().map(IntegerExpr::field).collect::<Result<Vec<_>>>()?,requires_line8:o.requires_line8})}).collect::<Result<Vec<_>>>()?;
 let all_tensors=input.tensors.iter().map(|(k,v)|Ok((k.clone(),v.iter().map(FieldValue::field).collect::<Result<Vec<_>>>()?))).collect::<Result<BTreeMap<_,_>>>()?;
 ensure!(all_tensors.len()==input.tensor_families.len(),"canonical adapter tensor set mismatch");let mut public_tensors=BTreeMap::new();let mut private_tensors=BTreeMap::new();for spec in &input.tensor_families{let values=all_tensors.get(&spec.family_id).ok_or_else(||anyhow::anyhow!("missing compiler tensor {}",spec.family_id))?.clone();if spec.visibility==Visibility::Public{public_tensors.insert(spec.family_id.clone(),values);}else{private_tensors.insert(spec.family_id.clone(),values);}}
 let registry=RelationRegistry{schema:"zkDL_gat_fac_registry_v1".into(),tensor_families:input.tensor_families.clone(),public_tensors,operation_families:operations,mandatory_operation_kinds:MANDATORY_GAT_OPERATIONS.iter().map(|x|x.to_string()).collect(),registry_id:String::new()}.seal()?;
 let witness=WitnessPackage{registry_id:registry.registry_id.clone(),tensors:private_tensors,witness_id:input.source_input_id.clone()};witness.validate(&registry)?;
 let statement=PublicStatement{protocol_id:"GAT-ZKML-zkDL".into(),proof_task:"node_classification_full_gat".into(),privacy_policy:"model_and_trace_hidden".into(),input_privacy:"public_graph_public_features".into(),output_mode:"public_logits".into(),dataset:input.dataset,pp_id:input.pp_id,registry,graph_features_digest:input.graph_features_digest,topology_digest:input.topology_digest,quant_cfg_id:input.quant_cfg_id,static_tables_id:input.static_tables_id,integer_bound_ledger_id:input.integer_bound_ledger_id,model_arch_id:input.model_arch_id,model_param_id:input.model_param_id,public_logits:input.public_logits,statement_id:String::new()}.seal()?;Ok((statement,witness))}
