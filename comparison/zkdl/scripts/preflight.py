#!/usr/bin/env python3
"""Protocol preflight and integer-bound ledger for the formal zkDL GAT relation."""
from __future__ import annotations
import argparse, ast, hashlib, json, math, re, struct
from pathlib import Path

from runner_contract import (
    BASELINE_ROOT,
    atomic_write_json,
    config_digest,
    ensure_contained,
    load_resource_manifest,
    protocol_digest,
    resource_inventory_digest,
    model_shape_payload,
    static_table_row_cap_blocker,
    validate_checkpoint_structure,
    validate_checkpoint_tensor_shapes,
    validate_quantization_config,
    validate_config_resources,
)

ROOT=Path(__file__).resolve().parents[1]
FR=0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
SUPPORTED_ROUNDING_MODE='nearest_ties_away'

def cfg(path:Path):
 out={}
 for raw in path.read_text().splitlines():
  line=raw.split('#',1)[0].strip()
  if not line or '=' not in line: continue
  k,v=line.split('=',1);out[k.strip()]=v.strip()
 return out
def sha(path:Path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tensors(path:Path):
 tok=path.read_text().split();i=0;out={}
 while i<len(tok):
  if tok[i]!='TENSOR':raise ValueError('invalid tensor marker')
  name=tok[i+1];rank=int(tok[i+2]);i+=3
  if name in out:raise ValueError(f'duplicate tensor name: {name}')
  shape=list(map(int,tok[i:i+rank]));i+=rank;n=int(tok[i]);i+=1;values=list(map(float,tok[i:i+n]));i+=n
  product=1
  for dim in shape:product*=dim
  if product!=n:raise ValueError(f'tensor shape mismatch: {name}')
  out[name]=(shape,values)
 if i!=len(tok):raise ValueError('trailing checkpoint tensor tokens')
 return out
def q(value,scale):return math.floor(value*scale+.5) if value>=0 else math.ceil(value*scale-.5)
def qmax(values,scale):return max((abs(q(x,scale)) for x in values),default=0)
def public_feature_bound(cache:Path,scale:int):
 meta={}
 for line in (cache/'meta.cfg').read_text().splitlines():
  if '=' in line:
   k,v=line.split('=',1);meta[k.strip()]=v.strip()
 n=int(meta['num_nodes']);d=int(meta['num_features'])
 bound=0;digest_state=hashlib.sha256();digest_state.update(b'zkDL-GAT-public-features-v1\0');digest_state.update(struct.pack('>QQ',n,d))
 if (cache/'features.txt').exists():
  with (cache/'features.txt').open() as f:
   if list(map(int,f.readline().split()))!=[n,d]:raise ValueError('features.txt header mismatch')
   rows=0
   for line in f:
    dense=[0]*d;seen=set()
    for item in line.split():
     column,value=item.split(':',1);column=int(column)
     if not 0<=column<d or column in seen:raise ValueError('features.txt column malformed')
     seen.add(column);dense[column]=q(float(value),scale)
    for quantized in dense:
     bound=max(bound,abs(quantized));digest_state.update(struct.pack('>q',quantized))
    rows+=1
   if rows!=n:raise ValueError('features.txt row count mismatch')
 else:
  with (cache/'features.npy').open('rb') as f:
   if f.read(6)!=b'\x93NUMPY':raise ValueError('bad npy magic')
   major,minor=f.read(2);hlen=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0];header=ast.literal_eval(f.read(hlen).decode('latin1'));shape=tuple(header['shape'])
   if shape!=(n,d) or header.get('fortran_order'):raise ValueError('features.npy shape/order mismatch')
   fmt='f' if header['descr'].endswith('f4') else 'd' if header['descr'].endswith('f8') else None
   if fmt is None:raise ValueError('features.npy dtype mismatch')
   width=4 if fmt=='f' else 8
   for _ in range(n):
    raw=f.read(d*width)
    if len(raw)!=d*width:raise ValueError('truncated features.npy')
    vals=struct.unpack('<'+fmt*d,raw)
    for v in vals:
     quantized=q(v,scale);bound=max(bound,abs(quantized));digest_state.update(struct.pack('>q',quantized))
 return bound,digest_state.hexdigest(),n,d
def graph_info(cache:Path,n:int):
 edges=[]
 for stable,line in enumerate((cache/'edges.txt').read_text().splitlines()):
  p=list(map(int,line.split()))
  if len(p)==2:g,s,d=0,p[0],p[1]
  elif len(p)==3:g,s,d=p
  else:raise ValueError('malformed edge')
  if s>=n or d>=n:raise ValueError('edge endpoint out of range')
  edges.append((g,s,d,stable))
 if (cache/'node_ptr.txt').exists():node_ptr=list(map(int,(cache/'node_ptr.txt').read_text().split()))
 else:node_ptr=[0,n]
 if not node_ptr or node_ptr[0]!=0 or node_ptr[-1]!=n or any(a>=b for a,b in zip(node_ptr,node_ptr[1:])):raise ValueError('invalid node_ptr')
 node_gid=[0]*n
 for g,(lo,hi) in enumerate(zip(node_ptr,node_ptr[1:])):
  for i in range(lo,hi):node_gid[i]=g
 for g,s,d,_ in edges:
  if node_gid[s]!=g or node_gid[d]!=g:raise ValueError('cross-graph edge or graph_id mismatch')
 self_nodes={s for _,s,d,_ in edges if s==d}
 for i in range(n):
  if i not in self_nodes: edges.append((node_gid[i],i,i,len(edges)))
 edges.sort(key=lambda e:(e[0],e[2],e[3]));deg=[0]*n
 for _,_,d,_ in edges:deg[d]+=1
 if min(deg)<1:raise ValueError('self-loop normalization failed')
 edge_ptr=[0]
 for graph_id in range(len(node_ptr)-1):
  edge_ptr.append(edge_ptr[-1]+sum(1 for g,_,_,_ in edges if g==graph_id))
 src=[s for _,s,_,_ in edges];dst=[d for _,_,d,_ in edges]
 h=hashlib.sha256();h.update(b'zkDL-GAT-public-topology-v1\0');h.update(struct.pack('>QQQ',n,len(edges),len(node_ptr)-1))
 for values in (node_ptr,edge_ptr,src,dst):
  h.update(struct.pack('>Q',len(values)))
  for value in values:h.update(struct.pack('>Q',value))
 return len(edges),max(deg),h.hexdigest(),node_ptr,edge_ptr
def signed_round_bound(a:int,d_min:int):return (2*a+d_min)//(2*d_min)
def lrelu(x:int):return x if x>=0 else -signed_round_bound(-x,5)
def elu(x:int):return x if x>=0 else q(16.0*(math.exp(x/16.0)-1.0),1)
def exp_neg(x:int):return max(1,q(16.0*math.exp(-x/256.0),1))
def division_bounds(bx:int,bd:int,bq:int):
 ba=abs(int(bx));bd=abs(int(bd));bq=abs(int(bq))
 if bd<=0:raise ValueError('SignedRoundDiv denominator bound must be positive')
 br=2*ba+bd+2*bd*bq
 bdelta=2*bd-1+br
 return {'B_a':ba,'B_d':bd,'B_q':bq,'B_r':br,'B_delta':bdelta,'B_2a_plus_d':2*ba+bd,'B_2dq_plus_r':2*bd*bq+br,'B_r_plus_delta':br+bdelta}
def static_table_descriptor(range_bits:int,source_id:str,lrelu_bound:int,elu_bound:int,exp_bound:int,rounding_mode:str):
 if rounding_mode!=SUPPORTED_ROUNDING_MODE:raise ValueError('unsupported rounding_mode')
 configured=1<<range_bits
 # range_bits is a lower bound.  The actual table domains are derived from
 # the conservative integer ledger below, so valid large instances are not
 # rejected by a fixed preflight range.  This descriptor deliberately does
 # not enumerate rows; callers can apply the native 2^26 cap first.
 lbound=max(configured,abs(int(lrelu_bound)));ebound=max(configured,abs(int(elu_bound)));exp_max=max(configured-1,int(exp_bound))
 return {'schema':'zkDL_gat_static_tables_v1','source_static_table_id':source_id,'generator_id':'main_gatzk_quantized_tables_v1','rounding_mode':rounding_mode,'canonical_order':'kind_then_signed_input_ascending','lrelu':{'min':-lbound,'max':lbound,'input_scale':256,'output_scale':256,'rows':2*lbound+1},'elu':{'min':-ebound,'max':ebound,'input_scale':16,'output_scale':16,'rows':2*ebound+1},'exp':{'min':0,'max':exp_max,'input_scale':256,'output_scale':16,'rows':exp_max+1}}
def static_table_profile(range_bits:int,source_id:str,lrelu_bound:int,elu_bound:int,exp_bound:int,rounding_mode:str):
 spec=static_table_descriptor(range_bits,source_id,lrelu_bound,elu_bound,exp_bound,rounding_mode)
 blocker=static_table_row_cap_blocker(spec)
 if blocker:raise ValueError('static table profile blocked: '+json.dumps(blocker,sort_keys=True,separators=(',',':')))
 lmin=spec['lrelu']['min'];lmax=spec['lrelu']['max'];emin=spec['elu']['min'];emax=spec['elu']['max'];xmin=spec['exp']['min'];xmax=spec['exp']['max']
 h=hashlib.sha256();h.update(b'zkDL-GAT-static-table-rows-v1\0')
 for kind,lo,hi,fn in [('lrelu',lmin,lmax,lrelu),('elu',emin,emax,elu),('exp',xmin,xmax,exp_neg)]:
  h.update(kind.encode()+b'\0')
  for x in range(lo,hi+1):h.update(struct.pack('>qq',x,fn(x)))
 spec['rows_digest']='sha256:'+h.hexdigest()
 canonical=json.dumps(spec,sort_keys=True,separators=(',',':')).encode();spec['static_tables_id']='sha256:'+hashlib.sha256(b'zkDL-GAT-static-table-profile-v1\0'+canonical).hexdigest()
 return spec
def build_static_tables(range_bits:int,source_id:str,lrelu_bound:int,elu_bound:int,exp_bound:int,rounding_mode:str):
 descriptor=static_table_descriptor(range_bits,source_id,lrelu_bound,elu_bound,exp_bound,rounding_mode)
 blocker=static_table_row_cap_blocker(descriptor)
 if blocker:return descriptor,blocker
 return static_table_profile(range_bits,source_id,lrelu_bound,elu_bound,exp_bound,rounding_mode),None
def next_power_two(n:int):
 return max(2,1<<(max(1,int(n))-1).bit_length())
def static_table_rows_digest(tables):
 """Derive the exact public lookup-row digest used by the canonical adapter."""
 blocker=static_table_row_cap_blocker(tables)
 if blocker:raise ValueError('static table digest blocked: '+json.dumps(blocker,sort_keys=True,separators=(',',':')))
 entries=[]
 for kind in ('lrelu','elu','exp'):
  spec=tables[kind];lo=int(spec['min']);hi=int(spec['max']);fn=lrelu if kind=='lrelu' else elu if kind=='elu' else exp_neg
  rows=[int(fn(x))%FR for x in range(lo,hi+1)];padded=next_power_two(len(rows));rows += [0]*(padded-len(rows));bits=tuple(range(3,3+padded.bit_length()-1));entries.append((tuple(rows),bits))
  selector=tuple([1]*(hi-lo+1)+[0]*(padded-(hi-lo+1)));entries.append((selector,bits))
 ordered=sorted(set(entries),key=lambda item:(len(item[0]),item[0],len(item[1]),item[1]))
 h=hashlib.sha256();h.update(b'zkDL-GAT-registry-static-table-rows-v1\0')
 for values,bits in ordered:
  h.update(struct.pack('>Q',len(values)))
  for value in values:h.update(int(value).to_bytes(32,'big'))
  h.update(struct.pack('>Q',len(bits)))
  for bit in bits:h.update(struct.pack('>I',bit))
 return 'sha256:'+h.hexdigest()
def validate_checkpoint_config(c,manifest,graph_count,feature_dim):
 return validate_checkpoint_structure(c,manifest,graph_count=graph_count,feature_dim=feature_dim)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);ap.add_argument('--project-root',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
 root=ensure_contained(a.project_root,label='project-root');out=ensure_contained(a.output,label='preflight output');base=ROOT
 if root != BASELINE_ROOT:raise SystemExit('project-root must be comparison/zkdl; stage resources locally first')
 config_path=ensure_contained(a.config,label='formal config')
 c=cfg(config_path)
 if c.get('rounding_mode')!=SUPPORTED_ROUNDING_MODE:raise SystemExit('unsupported rounding_mode; only nearest_ties_away is implemented')
 validate_quantization_config(c)
 resource_manifest,resource_manifest_id=load_resource_manifest(require_staged=True)
 validate_config_resources(c.get('dataset',''),c,resource_manifest)
 cache=root/c['cache'];cp=root/c['checkpoint_bundle'];manifest=json.loads((cp/'manifest.json').read_text());ts=tensors(cp/'tensors.txt')
 scale_keys=['feature_scale','hidden_weight_scale','hidden_attention_scale','hidden_projection_scale','delta_scale','exp_scale','hidden_recovery_num_scale','hidden_recovery_den_scale','elu_input_scale','elu_output_scale','agg_scale','output_weight_scale','output_attention_scale','output_projection_scale','output_pre_scale','output_head_scale','output_bias_scale','output_scale']
 scales={k:int(c[k]) for k in scale_keys};bh,features_digest,n,d=public_feature_bound(cache,scales['feature_scale']);e,maxdeg,topo_digest,node_ptr,edge_ptr=graph_info(cache,n);checkpoint_binding=validate_checkpoint_config(c,manifest,len(node_ptr)-1,d);validate_checkpoint_tensor_shapes(manifest,{name:shape for name,(shape,_) in ts.items()})
 failures=[];heads=[];current_bh=bh
 hidden_by_layer={}
 for spec in manifest['hidden_head_specs']:hidden_by_layer.setdefault(int(spec['layer_index']),[]).append(spec)
 for layer,layer_meta in enumerate(manifest['hidden_layers']):
  next_bounds=[];din=int(layer_meta['input_dim']);dh=int(layer_meta['head_dim'])
  for spec in hidden_by_layer[layer]:
   bw=qmax(ts[spec['seq_kernel']][1],scales['hidden_weight_scale']);bas=qmax(ts[spec['attn_src_kernel']][1],scales['hidden_attention_scale']);bad=qmax(ts[spec['attn_dst_kernel']][1],scales['hidden_attention_scale']);bhp=din*current_bh*bw;bes=dh*bhp*bas;bed=dh*bhp*bad;bs=bes+bed;bz=max(abs(lrelu(-bs)),abs(lrelu(bs)));bdelta=2*bz;bu=exp_neg(0);bsum=maxdeg*bu;bhnum=maxdeg*bu*bhp;bx=scales['hidden_recovery_num_scale']*bhnum;bd=scales['hidden_recovery_den_scale']*bsum;bq=signed_round_bound(bx,bd)
   rec={'kind':'hidden','layer':layer,'head':int(spec['local_head_index']),'B_H':current_bh,'B_W':bw,'B_a_src':bas,'B_a_dst':bad,'B_H_prime':bhp,'B_E_src':bes,'B_E_dst':bed,'B_S':bs,'B_delta_raw':bdelta,'B_U':bu,'B_Sum':bsum,'B_H_num':bhnum,'B_x_div':bx,'B_d_div':bd,'B_q_div':bq}
   rec.update(division_bounds(bx,bd,bq))
   heads.append(rec);next_bounds.append(max(abs(elu(-bq)),abs(elu(bq))))
  current_bh=max(next_bounds)
 din=int(manifest['hidden_layers'][-1]['head_count'])*int(manifest['hidden_layers'][-1]['head_dim']);classes=int(manifest['C'])
 for spec in manifest['output_head_specs']:
  bw=qmax(ts[spec['seq_kernel']][1],scales['output_weight_scale']);bas=qmax(ts[spec['attn_src_kernel']][1],scales['output_attention_scale']);bad=qmax(ts[spec['attn_dst_kernel']][1],scales['output_attention_scale']);bb=qmax(ts[spec['output_bias']][1],scales['output_bias_scale']);byp=din*current_bh*bw;bes=classes*byp*bas;bed=classes*byp*bad;bs=bes+bed;bz=max(abs(lrelu(-bs)),abs(lrelu(bs)));bdelta=2*bz;bu=exp_neg(0);bsum=maxdeg*bu;bnum=maxdeg*bu*byp;bx=scales['output_pre_scale']*bnum;bd=scales['output_projection_scale']*bsum;bq=signed_round_bound(bx,bd);balign=signed_round_bound(scales['output_head_scale']*bq,scales['output_pre_scale']);bhead=balign+bb
  rec={'kind':'output','head':int(spec['head_index']),'B_H':current_bh,'B_W':bw,'B_a_src':bas,'B_a_dst':bad,'B_b':bb,'B_Y_prime':byp,'B_E_src':bes,'B_E_dst':bed,'B_S':bs,'B_delta_raw':bdelta,'B_U':bu,'B_Sum':bsum,'B_Y_num':bnum,'B_x_div':bx,'B_d_div':bd,'B_q_div':bq,'B_Y_head':bhead}
  rec.update(division_bounds(bx,bd,bq))
  heads.append(rec)
  # Output alignment is a second SignedRoundDiv family (canonical denominator)
  # and must receive its own certified integer bounds.
  balign_x=scales['output_head_scale']*bq
  alignment_d=scales['output_pre_scale']
  balign=signed_round_bound(balign_x,alignment_d)
  alignment={'kind':'output_alignment','head':int(spec['head_index']),'B_x':balign_x,'B_d':alignment_d,'B_q':balign}
  alignment.update(division_bounds(balign_x,alignment_d,balign));heads.append(alignment)
 # The final public output is another division over the sum of all output
 # heads; include it in the same ledger as every per-head division.
 output_head_bound=sum(int(head['B_Y_head']) for head in heads if head['kind']=='output')
 output_count=len(manifest['output_head_specs'])
 if output_count <= 0:raise ValueError('checkpoint manifest has no output heads')
 final_q=signed_round_bound(output_head_bound,output_count)
 final_average={'kind':'output_average','head_count':output_count,'B_x':output_head_bound,'B_d':output_count,'B_q':final_q}
 final_average.update(division_bounds(output_head_bound,output_count,final_q));heads.append(final_average)
 max_score=max((abs(int(head['B_S'])) for head in heads if 'B_S' in head),default=0)
 max_delta=max((int(head['B_delta_raw']) for head in heads if 'B_delta_raw' in head),default=0)
 max_elu=max((abs(int(head['B_q_div'])) for head in heads if head['kind']=='hidden'),default=0)
 shape_payload=model_shape_payload(manifest,feature_dim=d)
 model_shape_id='sha256:'+hashlib.sha256(json.dumps(shape_payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 checkpoint_manifest_id='sha256:'+sha(cp/'manifest.json')
 tables,capacity_blocker=build_static_tables(int(c['range_bits']),manifest['static_table_id'],max_score,max_elu,max_delta,c['rounding_mode'])
 if capacity_blocker:
  blocked={'schema':'zkDL_gat_preflight_v1','status':'blocked_static_table_rows','dataset':c['dataset'],'L':int(manifest['L']),'hidden_profile':checkpoint_binding['hidden_profile'],'d_in_profile':checkpoint_binding['d_in_profile'],'K_out':int(manifest['K_out']),'node_count':n,'edge_count':e,'graph_count':len(node_ptr)-1,'feature_dim':d,'class_count':int(manifest['C']),'config_sha256':config_digest(config_path),'protocol_sha256':protocol_digest(),'resource_manifest_id':resource_manifest_id,'checkpoint_manifest_id':checkpoint_manifest_id,'checkpoint_manifest_sha256':sha(cp/'manifest.json'),'checkpoint_tensors_sha256':sha(cp/'tensors.txt'),'model_shape_id':model_shape_id,'model_arch_id':manifest['model_arch_id'],'model_param_id':manifest['model_param_id'],'static_table_descriptor':tables,'blocker':capacity_blocker}
  atomic_write_json(out,blocked)
  raise SystemExit('preflight blocked: '+json.dumps(capacity_blocker,sort_keys=True,separators=(',',':')))
 lmin=tables['lrelu']['min'];lmax=tables['lrelu']['max'];exp_max=tables['exp']['max'];elu_min=tables['elu']['min'];elu_max=tables['elu']['max']
 static_table_rows_id=static_table_rows_digest(tables)
 for head in heads:
  if 'B_S' in head and (-int(head['B_S'])<lmin or int(head['B_S'])>lmax):failures.append(f"{head['kind']}[{head.get('layer','')},{head.get('head','')}] LReLU domain exceeds derived table")
  if 'B_delta_raw' in head and int(head['B_delta_raw'])>exp_max:failures.append(f"{head['kind']}[{head.get('layer','')},{head.get('head','')}] ExpMap domain exceeds derived table")
  if head['kind']=='hidden' and (-int(head['B_q_div'])<elu_min or int(head['B_q_div'])>elu_max):failures.append(f"hidden[{head.get('layer','')},{head['head']}] ELU domain exceeds derived table")
 field_half=int('73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001',16)//2
 max_bound=max((int(value) for head in heads for key,value in head.items() if key.startswith('B_') and isinstance(value,(int,float))),default=0)
 if max_bound>=field_half:failures.append('conservative integer ledger exceeds signed BLS12-381 field half-modulus')
 bit_width=max(2,max_bound.bit_length()+1)
 ledger={'schema':'zkDL_gat_integer_bound_ledger_v3','dataset':c['dataset'],'rounding_mode':c['rounding_mode'],'L':int(manifest['L']),'hidden_profile':checkpoint_binding['hidden_profile'],'d_in_profile':checkpoint_binding['d_in_profile'],'K_out':int(manifest['K_out']),'node_count':n,'edge_count':e,'max_in_degree':maxdeg,'range_bits':int(c['range_bits']),'bit_width':bit_width,'table_domains':{'lrelu':[lmin,lmax],'elu':[elu_min,elu_max],'exp':[0,exp_max]},'scales':scales,'heads':heads,'field':'BLS12-381-Fr','field_half_modulus':str(field_half),'field_fit':'every declared B_a/B_q/B_r/B_delta/B_d and composite equality bound is below the signed field half-modulus; the Rust compiler rechecks each serialized witness value','failures':failures,'graph_count':len(node_ptr)-1,'feature_dim':d,'class_count':int(manifest['C']),'resource_manifest_id':resource_manifest_id,'config_sha256':config_digest(config_path),'checkpoint_manifest_id':checkpoint_manifest_id,'model_shape_id':model_shape_id,'static_tables_id':tables['static_tables_id'],'static_table_rows_id':static_table_rows_id,'model_arch_id':manifest['model_arch_id'],'model_param_id':manifest['model_param_id']}
 report_metadata={'status':'accepted' if not failures else 'rejected_integer_bound','dataset':c['dataset'],'rounding_mode':c['rounding_mode'],'L':int(manifest['L']),'hidden_profile':checkpoint_binding['hidden_profile'],'d_in_profile':checkpoint_binding['d_in_profile'],'K_out':int(manifest['K_out']),'node_count':n,'edge_count':e,'graph_count':len(node_ptr)-1,'feature_dim':d,'class_count':int(manifest['C']),'config_sha256':config_digest(config_path),'protocol_sha256':protocol_digest(),'resource_manifest_id':resource_manifest_id,'resource_inventory_digest':resource_inventory_digest(c['dataset'],resource_manifest),'graph_features_digest':'sha256:'+features_digest,'topology_digest':'sha256:'+topo_digest,'checkpoint_manifest_id':checkpoint_manifest_id,'checkpoint_manifest_sha256':sha(cp/'manifest.json'),'checkpoint_tensors_sha256':sha(cp/'tensors.txt'),'source_quant_cfg_id':manifest['quant_cfg_id'],'quant_cfg_id':'sha256:'+hashlib.sha256(json.dumps(scales,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'model_shape_id':model_shape_id,'static_tables_id':tables['static_tables_id'],'static_table_rows_id':static_table_rows_id,'static_tables':tables,'model_arch_id':manifest['model_arch_id'],'model_param_id':manifest['model_param_id'],'checkpoint_binding':checkpoint_binding,'public_instance':{'node_count':n,'edge_count':e,'graph_count':len(node_ptr)-1,'feature_dim':d,'class_count':int(manifest['C']),'node_ptr':node_ptr,'edge_ptr':edge_ptr}}
 canonical=json.dumps(ledger,sort_keys=True,separators=(',',':')).encode();ledger['integer_bound_ledger_id']='sha256:'+hashlib.sha256(canonical).hexdigest()
 report={**report_metadata,'ledger':ledger}
 atomic_write_json(out,report)
 if failures:raise SystemExit('preflight rejected formal instance: '+failures[0])
 print('PREFLIGHT_OK',ledger['integer_bound_ledger_id'])
if __name__=='__main__':main()
