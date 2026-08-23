#!/usr/bin/env python3
"""Protocol preflight and integer-bound ledger for the formal zkDL GAT relation."""
from __future__ import annotations
import argparse, ast, hashlib, json, math, re, struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

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
  name=tok[i+1];rank=int(tok[i+2]);i+=3;shape=list(map(int,tok[i:i+rank]));i+=rank;n=int(tok[i]);i+=1;values=list(map(float,tok[i:i+n]));i+=n
  product=1
  for dim in shape:product*=dim
  if product!=n:raise ValueError(f'tensor shape mismatch: {name}')
  out[name]=(shape,values)
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
def exp_neg(x:int):return max(0,q(16.0*math.exp(-x/256.0),1))
def static_table_profile(range_bits:int,large:bool,source_id:str):
 configured=1<<range_bits
 range_size=max(configured,(1<<23)-2) if large else configured
 lmin=-(3<<18) if large else -range_size
 lmax=((1<<23)-2)-(3<<18)-1 if large else range_size
 elu_bound=max((1<<20) if large else configured,configured)
 exp_max=range_size-1
 h=hashlib.sha256();h.update(b'zkDL-GAT-static-table-rows-v1\0')
 counts={}
 for kind,lo,hi,fn in [('lrelu',lmin,lmax,lrelu),('elu',-elu_bound,elu_bound,elu),('exp',0,exp_max,exp_neg)]:
  counts[kind]=hi-lo+1
  h.update(kind.encode()+b'\0')
  for x in range(lo,hi+1):h.update(struct.pack('>qq',x,fn(x)))
 rows_digest='sha256:'+h.hexdigest()
 spec={'schema':'zkDL_gat_static_tables_v1','source_static_table_id':source_id,'generator_id':'main_gatzk_quantized_tables_v1','rounding_mode':'nearest_ties_away','canonical_order':'kind_then_signed_input_ascending','lrelu':{'min':lmin,'max':lmax,'input_scale':256,'output_scale':256,'rows':counts['lrelu']},'elu':{'min':-elu_bound,'max':elu_bound,'input_scale':16,'output_scale':16,'rows':counts['elu']},'exp':{'min':0,'max':exp_max,'input_scale':256,'output_scale':16,'rows':counts['exp']},'rows_digest':rows_digest}
 canonical=json.dumps(spec,sort_keys=True,separators=(',',':')).encode();spec['static_tables_id']='sha256:'+hashlib.sha256(b'zkDL-GAT-static-table-profile-v1\0'+canonical).hexdigest()
 return spec
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);ap.add_argument('--project-root',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
 root=a.project_root.resolve();out=a.output.resolve();base=ROOT.resolve()
 if out!=base and base not in out.parents:raise SystemExit('output must stay inside comparison/zkdl')
 c=cfg(a.config);cache=root/c['cache'];cp=root/c['checkpoint_bundle'];manifest=json.loads((cp/'manifest.json').read_text());ts=tensors(cp/'tensors.txt')
 scale_keys=['feature_scale','hidden_weight_scale','hidden_attention_scale','hidden_projection_scale','delta_scale','exp_scale','hidden_recovery_num_scale','hidden_recovery_den_scale','elu_input_scale','elu_output_scale','agg_scale','output_weight_scale','output_attention_scale','output_projection_scale','output_pre_scale','output_head_scale','output_bias_scale','output_scale']
 scales={k:int(c[k]) for k in scale_keys};bh,features_digest,n,d=public_feature_bound(cache,scales['feature_scale']);e,maxdeg,topo_digest,node_ptr,edge_ptr=graph_info(cache,n)
 large=c['batching_rule']=='whole_graph_single' and (e>=(1<<18) or n*d>=(1<<24));tables=static_table_profile(int(c['range_bits']),large,manifest['static_table_id']);lmin=tables['lrelu']['min'];lmax=tables['lrelu']['max'];exp_max=tables['exp']['max'];elu_min=tables['elu']['min'];elu_max=tables['elu']['max']
 failures=[];heads=[];current_bh=bh
 hidden_by_layer={}
 for spec in manifest['hidden_head_specs']:hidden_by_layer.setdefault(int(spec['layer_index']),[]).append(spec)
 for layer,layer_meta in enumerate(manifest['hidden_layers']):
  next_bounds=[];din=int(layer_meta['input_dim']);dh=int(layer_meta['head_dim'])
  for spec in hidden_by_layer[layer]:
   bw=qmax(ts[spec['seq_kernel']][1],scales['hidden_weight_scale']);bas=qmax(ts[spec['attn_src_kernel']][1],scales['hidden_attention_scale']);bad=qmax(ts[spec['attn_dst_kernel']][1],scales['hidden_attention_scale']);bhp=din*current_bh*bw;bes=dh*bhp*bas;bed=dh*bhp*bad;bs=bes+bed;bz=max(abs(lrelu(-bs)),abs(lrelu(bs)));bdelta=2*bz;bu=max(exp_neg(x) for x in (0,bdelta));bsum=maxdeg*bu;bhnum=maxdeg*bu*bhp;bx=scales['hidden_recovery_num_scale']*bhnum;bd=scales['hidden_recovery_den_scale']*bsum;bq=signed_round_bound(bx,max(1,bd))
   rec={'kind':'hidden','layer':layer,'head':int(spec['local_head_index']),'B_H':current_bh,'B_W':bw,'B_a_src':bas,'B_a_dst':bad,'B_H_prime':bhp,'B_E_src':bes,'B_E_dst':bed,'B_S':bs,'B_delta_raw':bdelta,'B_U':bu,'B_Sum':bsum,'B_H_num':bhnum,'B_x_div':bx,'B_d_div':bd,'B_q_div':bq}
   if -bs<lmin or bs>lmax:failures.append(f"hidden[{layer},{spec['local_head_index']}] LReLU domain [-{bs},{bs}] exceeds [{lmin},{lmax}]")
   if bdelta>exp_max:failures.append(f"hidden[{layer},{spec['local_head_index']}] ExpMap delta bound {bdelta} exceeds [0,{exp_max}]")
   if -bq<elu_min or bq>elu_max:failures.append(f"hidden[{layer},{spec['local_head_index']}] ELU input bound [-{bq},{bq}] exceeds [{elu_min},{elu_max}]")
   heads.append(rec);next_bounds.append(max(abs(elu(-bq)),abs(elu(bq))))
  current_bh=max(next_bounds)
 din=int(manifest['hidden_layers'][-1]['head_count'])*int(manifest['hidden_layers'][-1]['head_dim']);classes=int(manifest['C'])
 for spec in manifest['output_head_specs']:
  bw=qmax(ts[spec['seq_kernel']][1],scales['output_weight_scale']);bas=qmax(ts[spec['attn_src_kernel']][1],scales['output_attention_scale']);bad=qmax(ts[spec['attn_dst_kernel']][1],scales['output_attention_scale']);bb=qmax(ts[spec['output_bias']][1],scales['output_bias_scale']);byp=din*current_bh*bw;bes=classes*byp*bas;bed=classes*byp*bad;bs=bes+bed;bz=max(abs(lrelu(-bs)),abs(lrelu(bs)));bdelta=2*bz;bu=max(exp_neg(x) for x in (0,bdelta));bsum=maxdeg*bu;bnum=maxdeg*bu*byp;bx=scales['output_pre_scale']*bnum;bd=scales['output_projection_scale']*bsum;bq=signed_round_bound(bx,max(1,bd));balign=signed_round_bound(scales['output_head_scale']*bq,scales['output_pre_scale']);bhead=balign+bb
  rec={'kind':'output','head':int(spec['head_index']),'B_H':current_bh,'B_W':bw,'B_a_src':bas,'B_a_dst':bad,'B_b':bb,'B_Y_prime':byp,'B_E_src':bes,'B_E_dst':bed,'B_S':bs,'B_delta_raw':bdelta,'B_U':bu,'B_Sum':bsum,'B_Y_num':bnum,'B_x_div':bx,'B_d_div':bd,'B_q_div':bq,'B_Y_head':bhead}
  if -bs<lmin or bs>lmax:failures.append(f"output[{spec['head_index']}] LReLU domain [-{bs},{bs}] exceeds [{lmin},{lmax}]")
  if bdelta>exp_max:failures.append(f"output[{spec['head_index']}] ExpMap delta bound {bdelta} exceeds [0,{exp_max}]")
  heads.append(rec)
 ledger={'schema':'zkDL_gat_integer_bound_ledger_v2','dataset':c['dataset'],'node_count':n,'edge_count':e,'max_in_degree':maxdeg,'range_bits':int(c['range_bits']),'table_domains':{'lrelu':[lmin,lmax],'elu':[elu_min,elu_max],'exp':[0,exp_max]},'scales':scales,'heads':heads,'field':'BLS12-381-Fr','field_fit':'all listed integers must be below the canonical signed BLS12-381 half-modulus; formal compiler rechecks actual witness values','failures':failures}
 canonical=json.dumps(ledger,sort_keys=True,separators=(',',':')).encode();ledger['integer_bound_ledger_id']='sha256:'+hashlib.sha256(canonical).hexdigest()
 report={'status':'accepted' if not failures else 'rejected_table_domain','dataset':c['dataset'],'graph_features_digest':'sha256:'+features_digest,'topology_digest':'sha256:'+topo_digest,'public_instance':{'node_count':n,'edge_count':e,'graph_count':len(node_ptr)-1,'feature_dim':d,'class_count':int(manifest['C']),'node_ptr':node_ptr,'edge_ptr':edge_ptr},'checkpoint_manifest_sha256':sha(cp/'manifest.json'),'checkpoint_tensors_sha256':sha(cp/'tensors.txt'),'source_quant_cfg_id':manifest['quant_cfg_id'],'quant_cfg_id':'sha256:'+hashlib.sha256(json.dumps(scales,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'static_tables_id':tables['static_tables_id'],'static_tables':tables,'model_arch_id':manifest['model_arch_id'],'model_param_id':manifest['model_param_id'],'ledger':ledger}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
 if failures:raise SystemExit('preflight rejected formal instance: '+failures[0])
 print('PREFLIGHT_OK',ledger['integer_bound_ledger_id'])
if __name__=='__main__':main()
