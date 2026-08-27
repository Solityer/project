#!/usr/bin/env python3
"""Build the fixed, baseline-local integer GAT witness and FAC relation plan.

This adapter accepts only the repository's canonical dataset cache, checkpoint
bundle, formal config and authenticated preflight report.  It never accepts an
operation registry or expression from a caller: every tensor view and residual
below is generated from the fixed GAT program.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math,struct
from pathlib import Path
import numpy as np

from runner_contract import (
    BASELINE_ROOT,
    atomic_write_bytes,
    ensure_contained,
    load_resource_manifest,
    validate_config_resources,
)

ROOT=Path(__file__).resolve().parents[1]
FR=0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
ADAPTER_SCHEMA='zkDL_gat_canonical_adapter_v2'
ADAPTER_VERSION=2
PROTOCOL_PATH=ROOT/'GAT-ZKML-zkDL.md'
CANONICAL_RELATION_SCHEMA='zkDL_gat_canonical_relation_schema_v1'
CANONICAL_SCHEMA_KEYS=('schema','version','architecture','tensor_family_order','tensor_families','operation_family_order','operation_families','fac_plan','terminal_plan')
CANONICAL_ARCHITECTURE_KEYS=('model_arch_id','model_param_id','node_count','edge_count','graph_count','feature_dim','class_count','output_head_count','bit_width','lookup_bits','hidden_layers','hidden_head_specs','output_head_specs')

def exact_keys(obj,expected,label):
 if not isinstance(obj,dict) or set(obj)!=set(expected):raise ValueError(f'{label} has unknown or missing fields')

def cfg(path):
 out={}
 for raw in path.read_text().splitlines():
  line=raw.split('#',1)[0].strip()
  if line and '=' in line:
   k,v=line.split('=',1);out[k.strip()]=v.strip()
 return out
def q(v,s):return math.floor(v*s+.5) if v>=0 else math.ceil(v*s-.5)
def sdiv(x,d):
 if d<=0:raise ValueError('SignedRoundDiv denominator is not positive')
 a=abs(int(x));y=(2*a+d)//(2*d);return -y if x<0 else y
def lrelu(x):return x if x>=0 else -sdiv(-x,5)
def elu(x):return x if x>=0 else q(16.0*(math.exp(x/16.0)-1.0),1)
def expneg(x):
 if x<0:raise ValueError('negative ExpMap input')
 return max(1,q(16.0*math.exp(-x/256.0),1))
def npow(n):return max(2,1<<(max(1,n)-1).bit_length())
def hx(v):return '0x'+(int(v)%FR).to_bytes(32,'big').hex()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sha256_tagged(domain,payload):return 'sha256:'+hashlib.sha256(domain+payload).hexdigest()
def protocol_digest():return 'sha256:'+hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
def canonical_json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def source_input_digest(obj):
 material=dict(obj);material.pop('source_input_id',None);material.pop('canonical_relation_digest',None)
 return sha256_tagged(b'zkDL-GAT-canonical-adapter-input-v2\0',canonical_json(material))
def relation_digest(obj):
 material=dict(obj);material.pop('canonical_relation_digest',None)
 return sha256_tagged(b'zkDL-GAT-canonical-relation-v2\0',canonical_json(material))
def static_table_rows_digest(ops):
 entries=set()
 def visit(expr):
  if 'PublicTableLookup' in expr:
   table=expr['PublicTableLookup'];vals=tuple(int(v)%FR for v in table['values']);bits=tuple(int(v) for v in table['bit_columns']);entries.add((vals,bits));return
  for key in ('Add','Sub','Mul'):
   if key in expr:
    visit(expr[key][0]);visit(expr[key][1]);return
 for op in ops:
  for expr in op['residuals']:visit(expr)
 ordered=sorted(entries,key=lambda item:(len(item[0]),item[0],len(item[1]),item[1]))
 h=hashlib.sha256();h.update(b'zkDL-GAT-registry-static-table-rows-v1\0')
 for vals,bits in ordered:
  h.update(struct.pack('>Q',len(vals)))
  for value in vals:h.update(int(value).to_bytes(32,'big'))
  h.update(struct.pack('>Q',len(bits)))
  for bit in bits:h.update(struct.pack('>I',bit))
 return 'sha256:'+h.hexdigest()
def registered_family_set_digest(spec):
 h=hashlib.sha256();h.update(b'zkDL-GAT-registered-family-set-v1\0')
 for tensor in spec:
  if tensor['visibility']!='RegisteredModel':continue
  fid=tensor['family_id'].encode();h.update(struct.pack('>Q',len(fid)));h.update(fid)
  h.update(struct.pack('>QQ',int(tensor['logical_len']),int(tensor['padded_len'])))
  shape=tensor['logical_shape'];h.update(struct.pack('>Q',len(shape)))
  for value in shape:h.update(struct.pack('>Q',int(value)))
  order=tensor['mle_order'];h.update(struct.pack('>Q',len(order)))
  for value in order:
   raw=value.encode();h.update(struct.pack('>Q',len(raw)));h.update(raw)
 return 'sha256:'+h.hexdigest()
def canonical_tensor_digest(spec,tensors):
 h=hashlib.sha256();h.update(b'zkDL-GAT-canonical-tensor-v1\0')
 for tensor in spec:
  fid=tensor['family_id'].encode();h.update(struct.pack('>Q',len(fid)));h.update(fid)
  values=tensors[fid.decode()];h.update(struct.pack('>Q',len(values)))
  for value in values:
   raw=int(value,0) if isinstance(value,str) else int(value)
   h.update((raw%FR).to_bytes(32,'big'))
 return 'sha256:'+h.hexdigest()

def load_features(cache,n,d,scale):
 p=cache/'features.txt'
 if p.exists():
  out=np.zeros((n,d),dtype=np.int64)
  with p.open() as f:
   if tuple(map(int,f.readline().split()))!=(n,d):raise ValueError('feature header mismatch')
   for row,line in enumerate(f):
    seen=set()
    for item in line.split():
     col,val=item.split(':',1);col=int(col)
     if not 0<=col<d or col in seen:raise ValueError('features.txt column malformed')
     seen.add(col);out[row,col]=q(float(val),scale)
  if row+1!=n:raise ValueError('feature row count mismatch')
  return out,p
 raw=np.load(cache/'features.npy',mmap_mode='r')
 if raw.shape!=(n,d):raise ValueError('feature shape mismatch')
 scaled=raw*scale;out=np.where(scaled>=0,np.floor(scaled+.5),np.ceil(scaled-.5)).astype(np.int64)
 return out,cache/'features.npy'
def load_edges(cache,n):
 raw=[]
 for stable,line in enumerate((cache/'edges.txt').read_text().splitlines()):
  p=list(map(int,line.split()));g,s,d=(0,p[0],p[1]) if len(p)==2 else tuple(p)
  raw.append((g,s,d,stable))
 ptr=list(map(int,(cache/'node_ptr.txt').read_text().split())) if (cache/'node_ptr.txt').exists() else [0,n]
 if not ptr or ptr[0]!=0 or ptr[-1]!=n or any(a>=b for a,b in zip(ptr,ptr[1:])):raise ValueError('invalid node_ptr')
 gid=np.zeros(n,dtype=np.int64)
 for g,(lo,hi) in enumerate(zip(ptr,ptr[1:])):gid[lo:hi]=g
 for g,s,d,_ in raw:
  if not (0<=s<n and 0<=d<n):raise ValueError('edge endpoint out of range')
  if not (0<=g<len(ptr)-1) or int(gid[s])!=g or int(gid[d])!=g:raise ValueError('cross-graph edge or graph_id mismatch')
 loops={s for _,s,d,_ in raw if s==d}
 for node in range(n):
  if node not in loops:raw.append((int(gid[node]),node,node,len(raw)))
 raw.sort(key=lambda e:(e[0],e[2],e[3]))
 src=np.asarray([e[1] for e in raw],dtype=np.int64);dst=np.asarray([e[2] for e in raw],dtype=np.int64)
 start=np.zeros(len(raw),dtype=np.int64);end=np.zeros(len(raw),dtype=np.int64)
 for i in range(len(raw)):start[i]=int(i==0 or dst[i-1]!=dst[i]);end[i]=int(i+1==len(raw) or dst[i+1]!=dst[i])
 edge_ptr=[0]
 for graph_id in range(len(ptr)-1):edge_ptr.append(edge_ptr[-1]+sum(1 for g,_,_,_ in raw if g==graph_id))
 return src,dst,start,end,np.asarray(ptr,dtype=np.int64),np.asarray(edge_ptr,dtype=np.int64)
def load_tensors(path):
 tok=path.read_text().split();i=0;out={}
 while i<len(tok):
  if tok[i]!='TENSOR':raise ValueError('checkpoint tensor marker mismatch')
  name=tok[i+1];rank=int(tok[i+2]);i+=3;shape=tuple(map(int,tok[i:i+rank]));i+=rank;count=int(tok[i]);i+=1;vals=np.asarray(list(map(float,tok[i:i+count])),dtype=np.float64);i+=count
  if math.prod(shape)!=count:raise ValueError('checkpoint tensor shape mismatch')
  out[name]=(shape,vals)
 return out

def C(v):return {'Const':int(v) if isinstance(v,(int,np.integer)) else v}
def X(i):return {'Column':i}
def A(a,b):return {'Add':[a,b]}
def S(a,b):return {'Sub':[a,b]}
def M(a,b):return {'Mul':[a,b]}
def sumexpr(xs):
 out=C(0)
 for x in xs:out=A(out,x)
 return out
def valid(expr,v):return M(X(v),expr)

class Builder:
 def __init__(self):self.spec=[];self.tensors={};self.ops=[]
 def tensor(self,name,values,shape,visibility):
  flat=np.asarray(values,dtype=object).reshape(-1).tolist()
  if not flat:raise ValueError('empty canonical tensor')
  if name in self.tensors:raise ValueError(f'duplicate canonical tensor family: {name}')
  if not shape or any(int(x)<=0 for x in shape) or math.prod(map(int,shape))!=len(flat):raise ValueError(f'canonical tensor shape mismatch: {name}')
  flat=[int(x) if isinstance(x,(int,np.integer)) else x for x in flat]
  if visibility not in ('Public','RegisteredModel','DynamicPrivate'):raise ValueError(f'unknown tensor visibility: {visibility}')
  self.tensors[name]=flat;self.spec.append({'family_id':name,'logical_len':len(flat),'padded_len':npow(len(flat)),'logical_shape':[int(x) for x in shape],'mle_order':['row_major_lsb_first'],'instance_order':['canonical_program_order'],'visibility':visibility});return name
 def view(self,name,family,mapping):
  if family not in self.tensors:raise ValueError(f'unknown canonical tensor family: {family}')
  return {'column_id':name,'source':{'TensorView':{'family_id':family,'canonical_index_by_row':mapping}}}
 def affine(self,name,family,offset,stride,logical,rows,pad=0):
  if pad!=0:raise ValueError('canonical tensor padding index must be zero')
  index={'Affine':{'offset':offset,'stride':stride}} if logical==rows else {'PaddedAffine':{'offset':offset,'stride':stride,'logical_rows':logical,'padding_index':pad}}
  return self.view(name,family,index)
 def explicit(self,name,family,index,rows):
  if len(index)>rows:raise ValueError(f'canonical explicit view exceeds operation rows: {name}')
  if any(int(x)<0 for x in index):raise ValueError(f'canonical explicit view has negative index: {name}')
  return self.view(name,family,{'Explicit':[int(x) for x in index]+[0]*(rows-len(index))})
 def public(self,name,values,rows):
  if len(values)>rows:raise ValueError(f'canonical public column exceeds operation rows: {name}')
  return {'column_id':name,'source':{'Public':list(map(int,values))+[0]*(rows-len(values))}}
 def op(self,kind,label,logical,columns,residuals,degree=None):
  if logical<=0 or not columns or not residuals:raise ValueError(f'canonical operation is empty: {label}')
  base=npow(logical);rows=base*2
  normalized=[]
  seen=set()
  for column in list(columns):
   if column['column_id'] in seen:raise ValueError(f'duplicate operation column: {label}.{column["column_id"]}')
   seen.add(column['column_id'])
   source=column['source']
   if 'Public' in source:
    values=list(source['Public'])+[0]*(rows-len(source['Public']))
    column={'column_id':column['column_id'],'source':{'Public':values}}
   elif 'TensorView' in source:
    view=source['TensorView']['canonical_index_by_row']
    if 'Affine' in view:
     affine=view['Affine']
     view={'PaddedAffine':{'offset':int(affine['offset']),'stride':int(affine['stride']),'logical_rows':int(logical),'padding_index':0}}
     column={'column_id':column['column_id'],'source':{'TensorView':{'family_id':source['TensorView']['family_id'],'canonical_index_by_row':view}}}
    elif 'Explicit' in view:
     view={'Explicit':list(view['Explicit'])+[0]*max(0,rows-len(view['Explicit']))}
     column={'column_id':column['column_id'],'source':{'TensorView':{'family_id':source['TensorView']['family_id'],'canonical_index_by_row':view}}}
   normalized.append(column)
  columns=normalized+[self.public('valid',[1]*logical,rows),self.public('domain_selector',[1]*logical,rows)]
  valid_index=len(columns)-2;selector_index=len(columns)-1
  res=[M(X(selector_index),valid(x,valid_index)) for x in residuals]
  deg=max((expr_degree(x) for x in res),default=1)+1
  self.ops.append({'family_id':label,'operation_kind':kind,'rows':rows,'real_rows':logical,'degree_bound':max(degree or deg,deg),'columns':columns,'residuals':res,'requires_line8':sum('TensorView' in c['source'] for c in columns)>1})

def expr_degree(e):
 if 'Const' in e:return 0
 if 'Column' in e:return 1
 if 'Add' in e:return max(map(expr_degree,e['Add']))
 if 'Sub' in e:return max(map(expr_degree,e['Sub']))
 if 'Mul' in e:return sum(map(expr_degree,e['Mul']))
 if 'PublicTableLookup' in e:return len(e['PublicTableLookup']['bit_columns'])
 raise ValueError('unknown expression')

def canonical_fac_plan(ops,spec):
 visibility={x['family_id']:x['visibility'] for x in spec}
 plans=[]
 for op in ops:
  groups={}
  for index,column in enumerate(op['columns']):
   source=column['source']
   if 'TensorView' in source and visibility[source['TensorView']['family_id']]!='Public':
    groups.setdefault(source['TensorView']['family_id'],[]).append((index,column))
  claims=[]
  for family,columns in sorted(groups.items()):
   non_identity=False
   family_len=next(x['logical_len'] for x in spec if x['family_id']==family)
   for index,column in columns:
    view=column['source']['TensorView']['canonical_index_by_row']
    padded=view.get('PaddedAffine')
    if not (padded and int(padded['offset'])==0 and int(padded['stride'])==1 and int(padded['logical_rows'])==op['real_rows'] and int(padded['padding_index'])==0 and family_len==op['real_rows']):
     non_identity=True
   claims.append({'family_id':family,'column_indices':[i for i,_ in columns],'column_ids':[c['column_id'] for _,c in columns],'view_count':len(columns),'requires_reindex':non_identity or len(columns)>1,'requires_line8':True})
  plans.append({'operation_id':op['family_id'],'terminal_claims':claims,'terminal_target':'canonical_tensor_family'})
 return {'schema':'zkDL_gat_fac_reindex_plan_v1','operations':plans}

def validate_canonical_schema(schema,spec,ops,architecture):
 exact_keys(schema,CANONICAL_SCHEMA_KEYS,'canonical relation schema')
 if schema['schema']!=CANONICAL_RELATION_SCHEMA or schema['version']!=1:raise ValueError('canonical relation schema identity mismatch')
 exact_keys(architecture,CANONICAL_ARCHITECTURE_KEYS,'canonical architecture profile')
 if not isinstance(architecture['hidden_layers'],list) or not architecture['hidden_layers']:raise ValueError('canonical hidden layer profile is empty')
 for index,layer in enumerate(architecture['hidden_layers']):
  exact_keys(layer,('layer_index','input_dim','head_count','head_dim'),f'canonical hidden layer {index}')
  if int(layer['layer_index'])!=index or int(layer['head_count'])<=0 or int(layer['head_dim'])<=0:raise ValueError('canonical hidden layer order/dimensions are invalid')
 hidden_count=sum(int(layer['head_count']) for layer in architecture['hidden_layers'])
 if not isinstance(architecture['hidden_head_specs'],list) or len(architecture['hidden_head_specs'])!=hidden_count:raise ValueError('canonical hidden head specification count mismatch')
 hidden_keys=('layer_index','local_head_index','global_head_index','seq_kernel','attn_dst_kernel','attn_dst_bias','attn_src_kernel','attn_src_bias','output_bias')
 cursor=0
 for layer in architecture['hidden_layers']:
  for local in range(int(layer['head_count'])):
   head=architecture['hidden_head_specs'][cursor];exact_keys(head,hidden_keys,f'canonical hidden head {cursor}')
   if int(head['layer_index'])!=int(layer['layer_index']) or int(head['local_head_index'])!=local or int(head['global_head_index'])!=cursor:raise ValueError('canonical hidden head order/index mapping is invalid')
   if any(not isinstance(head[key],str) or not head[key] for key in hidden_keys[3:]):raise ValueError('canonical hidden head tensor binding is malformed')
   cursor+=1
 if not isinstance(architecture['output_head_specs'],list) or len(architecture['output_head_specs'])!=int(architecture['output_head_count']) or not architecture['output_head_specs']:raise ValueError('canonical output head specification count mismatch')
 output_keys=('head_index','seq_kernel','attn_dst_kernel','attn_dst_bias','attn_src_kernel','attn_src_bias','output_bias')
 for index,head in enumerate(architecture['output_head_specs']):
  exact_keys(head,output_keys,f'canonical output head {index}')
  if int(head['head_index'])!=index or any(not isinstance(head[key],str) or not head[key] for key in output_keys[1:]):raise ValueError('canonical output head order/binding is malformed')
 if schema['tensor_family_order']!=[x['family_id'] for x in spec] or schema['tensor_families']!=spec:raise ValueError('canonical tensor descriptor is not isomorphic to adapter output')
 if schema['operation_family_order']!=[x['family_id'] for x in ops] or schema['operation_families']!=ops:raise ValueError('canonical operation descriptor is not isomorphic to adapter output')
 if schema['fac_plan']!=canonical_fac_plan(ops,spec):raise ValueError('canonical FAC descriptor mismatch')
 exact_keys(schema['terminal_plan'],('public_logits_operation','public_logits_family','target'),'canonical terminal plan')
 if schema['terminal_plan']!={'public_logits_operation':'output.public_logits','public_logits_family':'public.logits','target':'verifier_local_public_y'}:raise ValueError('canonical terminal descriptor mismatch')

def field_inverse(x):
 if x%FR==0:raise ValueError('zero field inverse')
 return pow(int(x),FR-2,FR)
def lookup_aux(x,lo,size):
 idx=np.asarray(x,dtype=object)-lo;bits=max(1,npow(size).bit_length()-1)
 return idx,[np.asarray([(int(v)>>j)&1 for v in idx],dtype=np.int64) for j in range(bits)]
def add_lookup(b,kind,label,xname,yname,x,lo,hi,fn):
 logical=len(x);rows=npow(logical);idx,bits=lookup_aux(x,lo,hi-lo+1);iname=b.tensor(label+'.idx',idx,(logical,),'DynamicPrivate');bnames=[b.tensor(f'{label}.bit_{j}',v,(logical,),'DynamicPrivate') for j,v in enumerate(bits)]
 cols=[b.affine('x',xname,0,1,logical,rows),b.affine('y',yname,0,1,logical,rows),b.affine('idx',iname,0,1,logical,rows)]
 for j,n in enumerate(bnames):cols.append(b.affine(f'bit_{j}',n,0,1,logical,rows))
 bitcols=list(range(3,3+len(bits)));reconstruct=sumexpr([M(C(1<<j),X(ci)) for j,ci in enumerate(bitcols)])
 table=[fn(v) for v in range(lo,hi+1)];padded=npow(len(table));selector=[1]*len(table)+[0]*(padded-len(table));table+= [0]*(padded-len(table))
 lookup={'PublicTableLookup':{'values':table,'bit_columns':bitcols}};select={'PublicTableLookup':{'values':selector,'bit_columns':bitcols}}
 res=[S(X(2),S(X(0),C(lo))),S(X(2),reconstruct),S(X(1),lookup),S(C(1),select)]+[M(X(ci),S(X(ci),C(1))) for ci in bitcols]
 b.op(kind,label,logical,cols,res,degree=len(bits)+2)

def add_dot(b,kind,label,left_name,right_name,out_name,n,d,left_public=False):
 rows=npow(n);cols=[]
 for k in range(d):
  cols.append(b.affine(f'l{k}',left_name,k,d,n,rows));cols.append(b.affine(f'r{k}',right_name,k,0,n,rows))
 cols.append(b.affine('out',out_name,0,1,n,rows));products=[M(X(2*k),X(2*k+1)) for k in range(d)];b.op(kind,label,n,cols,[S(X(2*d),sumexpr(products))])

def div_aux(x,d):
 data={k:[] for k in ['sigma','z','a','d','a_inv','d_inv','q','r','delta','y']}
 for xv,dv in zip(np.asarray(x,dtype=object).reshape(-1),np.asarray(d,dtype=object).reshape(-1)):
  xv=int(xv);dv=int(dv);yv=sdiv(xv,dv);a=abs(xv);qv=abs(yv);sig=int(xv<0);z=int(a==0);r=2*a+dv-2*dv*qv;delta=2*dv-1-r
  vals=(sig,z,a,dv,0 if z else hx(field_inverse(a)),hx(field_inverse(dv)),qv,r,delta,yv)
  for k,v in zip(data,vals):data[k].append(v)
 return data
def bits_for(values,width):return [np.asarray([(int(v)>>j)&1 for v in values],dtype=np.int64) for j in range(width)]
def add_div(b,label,xname,dname,yname,x,d,width):
 aux=div_aux(x,d);logical=len(aux['a']);rows=npow(logical);names={}
 for k in ['sigma','z','a','d','a_inv','d_inv','q','r','delta']:names[k]=b.tensor(f'{label}.{k}',aux[k],(logical,),'DynamicPrivate')
 bitnames={}
 for k in ['a','d','q','r','delta']:
  bitnames[k]=[b.tensor(f'{label}.{k}_bit_{j}',v,(logical,),'DynamicPrivate') for j,v in enumerate(bits_for(aux[k],width))]
 cols=[b.affine('x',xname,0,1,logical,rows),b.affine('d',dname,0,1,logical,rows),b.affine('y',yname,0,1,logical,rows)]
 pos={}
 for k in names:pos[k]=len(cols);cols.append(b.affine(k,names[k],0,1,logical,rows))
 bitpos={}
 for k,arr in bitnames.items():
  bitpos[k]=[]
  for j,n in enumerate(arr):bitpos[k].append(len(cols));cols.append(b.affine(f'{k}_bit_{j}',n,0,1,logical,rows))
 sig=X(pos['sigma']);z=X(pos['z']);a=X(pos['a']);dd=X(pos['d']);ainv=X(pos['a_inv']);dinv=X(pos['d_inv']);qq=X(pos['q']);rr=X(pos['r'])
 sign=S(C(1),M(C(2),sig));res=[S(X(0),M(sign,a)),S(X(2),M(sign,qq)),M(sig,S(sig,C(1))),M(z,S(z,C(1))),M(z,a),S(M(a,ainv),S(C(1),z)),M(z,sig),M(z,ainv),S(M(dd,dinv),C(1)),S(A(M(C(2),a),dd),A(M(C(2),M(dd,qq)),rr)),S(A(rr,X(pos['delta'])),S(M(C(2),dd),C(1)))]
 for k in bitpos:
  res.append(S(X(pos[k]),sumexpr([M(C(1<<j),X(ci)) for j,ci in enumerate(bitpos[k])])))
  res.extend(M(X(ci),S(X(ci),C(1))) for ci in bitpos[k])
 b.op('signed_round_div',label,logical,cols,res,degree=4)

def group_acc(values,dst,n):
 arr=np.asarray(values,dtype=object);flat=arr.reshape(len(dst),-1);acc=np.empty_like(flat)
 running=[0]*flat.shape[1];prev=-1
 for e,node in enumerate(dst):
  if int(node)!=prev:running=[0]*flat.shape[1];prev=int(node)
  for j in range(flat.shape[1]):running[j]+=int(flat[e,j]);acc[e,j]=running[j]
 out=np.zeros((n,flat.shape[1]),dtype=object)
 for e,node in enumerate(dst):out[int(node)]=acc[e]
 return acc.reshape(arr.shape),out

def build_head(b,prefix,H,Hname,W,asrc,adst,src,dst,start,end,hidden,bias,tables,width):
 n,din=H.shape;dout=W.shape[1];e=len(src);proj=(H.astype(object)@W.astype(object));pname=b.tensor(prefix+'.projection',proj,proj.shape,'DynamicPrivate')
 for j in range(dout):add_dot(b,'projection_matmul',f'{prefix}.projection.col_{j}',Hname,b.tensor(f'{prefix}.Wcol_{j}',W[:,j],(din,),'RegisteredModel'),b.tensor(f'{prefix}.Pcol_{j}',proj[:,j],(n,),'DynamicPrivate'),n,din)
 es=proj.astype(object)@asrc.astype(object);ed=proj.astype(object)@adst.astype(object);esn=b.tensor(prefix+'.E_src',es,(n,),'DynamicPrivate');edn=b.tensor(prefix+'.E_dst',ed,(n,),'DynamicPrivate')
 add_dot(b,'attention_dot',prefix+'.attention_src',pname,b.tensor(prefix+'.a_src',asrc,(dout,),'RegisteredModel'),esn,n,dout);add_dot(b,'attention_dot',prefix+'.attention_dst',pname,b.tensor(prefix+'.a_dst',adst,(dout,),'RegisteredModel'),edn,n,dout)
 ers=es[src];erd=ed[dst];ersn=b.tensor(prefix+'.route_E_src',ers,(e,),'DynamicPrivate');erdn=b.tensor(prefix+'.route_E_dst',erd,(e,),'DynamicPrivate');rows=npow(e)
 b.op('public_sparse_route',prefix+'.route_src',e,[b.explicit('node',esn,src,rows),b.affine('edge',ersn,0,1,e,rows)],[S(X(1),X(0))]);b.op('public_sparse_route',prefix+'.route_dst',e,[b.explicit('node',edn,dst,rows),b.affine('edge',erdn,0,1,e,rows)],[S(X(1),X(0))])
 score=ers+erd;sn=b.tensor(prefix+'.score',score,(e,),'DynamicPrivate');b.op('score_add',prefix+'.score_add',e,[b.affine('src',ersn,0,1,e,rows),b.affine('dst',erdn,0,1,e,rows),b.affine('score',sn,0,1,e,rows)],[S(X(2),A(X(0),X(1)))])
 z=np.asarray([lrelu(int(x)) for x in score],dtype=object);zn=b.tensor(prefix+'.z',z,(e,),'DynamicPrivate');add_lookup(b,'lrelu_indexed_public_table',prefix+'.lrelu',sn,zn,score,tables['lrelu']['min'],tables['lrelu']['max'],lrelu)
 maximum=np.zeros(n,dtype=object);selector=np.zeros(e,dtype=object)
 for i in range(e):
  if start[i]:maximum[int(dst[i])]=z[i]
  elif z[i]>maximum[int(dst[i])]:maximum[int(dst[i])]=z[i]
 for i in range(e):
  if z[i]==maximum[int(dst[i])] and not any(selector[j] for j in range(i) if dst[j]==dst[i]):selector[i]=1
 delta=np.asarray([int(maximum[int(dst[i])])-int(z[i]) for i in range(e)],dtype=object);mn=b.tensor(prefix+'.max',maximum,(n,),'DynamicPrivate');dn=b.tensor(prefix+'.delta',delta,(e,),'DynamicPrivate');seln=b.tensor(prefix+'.max_selector',selector,(e,),'DynamicPrivate');selacc,_=group_acc(selector,dst,n);san=b.tensor(prefix+'.max_selector_acc',selacc,(e,),'DynamicPrivate')
 dbits=bits_for(delta,width);dbn=[b.tensor(f'{prefix}.delta_bit_{j}',v,(e,),'DynamicPrivate') for j,v in enumerate(dbits)];cols=[b.explicit('max',mn,dst,rows),b.affine('z',zn,0,1,e,rows),b.affine('delta',dn,0,1,e,rows),b.affine('selector',seln,0,1,e,rows),b.affine('sel_acc',san,0,1,e,rows),b.public('start',start,rows),b.public('end',end,rows)]
 for j,x in enumerate(dbn):cols.append(b.affine(f'delta_bit_{j}',x,0,1,e,rows))
 bitp=list(range(7,7+len(dbn)));prev=[0]+list(range(e-1));cols.append(b.explicit('prev_acc',san,prev,rows));pa=len(cols)-1
 res=[S(X(2),S(X(0),X(1))),M(X(3),S(X(3),C(1))),M(X(3),X(2)),S(X(2),sumexpr([M(C(1<<j),X(ci)) for j,ci in enumerate(bitp)])),S(X(4),A(M(S(C(1),X(5)),X(pa)),X(3))),M(X(6),S(X(4),C(1)))]+[M(X(ci),S(X(ci),C(1))) for ci in bitp]
 b.op('exact_group_max',prefix+'.max_relation',e,cols,res,degree=4)
 u=np.asarray([expneg(int(x)) for x in delta],dtype=object);un=b.tensor(prefix+'.u',u,(e,),'DynamicPrivate');add_lookup(b,'exp_indexed_public_table',prefix+'.exp',dn,un,delta,tables['exp']['min'],tables['exp']['max'],expneg)
 uacc,sumv=group_acc(u,dst,n);uan=b.tensor(prefix+'.u_acc',uacc,(e,),'DynamicPrivate');sumn=b.tensor(prefix+'.sum',sumv[:,0],(n,),'DynamicPrivate');prev=[0]+list(range(e-1));cols=[b.affine('u',un,0,1,e,rows),b.affine('acc',uan,0,1,e,rows),b.explicit('prev',uan,prev,rows),b.public('start',start,rows),b.public('end',end,rows),b.explicit('sum',sumn,dst,rows)];b.op('group_sum',prefix+'.group_sum',e,cols,[S(X(1),A(M(S(C(1),X(3)),X(2)),X(0))),M(X(4),S(X(1),X(5)))])
 inv=np.asarray([hx(field_inverse(int(x))) for x in sumv[:,0]],dtype=object);invn=b.tensor(prefix+'.inv',inv,(n,),'DynamicPrivate');alpha=np.asarray([hx(int(u[k])*field_inverse(int(sumv[int(dst[k]),0]))) for k in range(e)],dtype=object);aln=b.tensor(prefix+'.alpha',alpha,(e,),'DynamicPrivate')
 nr=npow(n);b.op('inverse_alpha',prefix+'.inverse',n,[b.affine('sum',sumn,0,1,n,nr),b.affine('inv',invn,0,1,n,nr)],[S(M(X(0),X(1)),C(1))]);b.op('inverse_alpha',prefix+'.alpha',e,[b.affine('alpha',aln,0,1,e,rows),b.affine('u',un,0,1,e,rows),b.explicit('inv',invn,dst,rows)],[S(X(0),M(X(1),X(2)))])
 routed=proj[src];rpn=b.tensor(prefix+'.route_projection',routed,routed.shape,'DynamicPrivate');logical=e*dout;rr=npow(logical);idx=np.arange(logical);node_idx=(src.repeat(dout)*dout+np.tile(np.arange(dout),e)).tolist();b.op('public_sparse_route',prefix+'.route_projection_relation',logical,[b.explicit('node',pname,node_idx,rr),b.affine('edge',rpn,0,1,logical,rr)],[S(X(1),X(0))])
 umsg=u[:,None]*routed;umn=b.tensor(prefix+'.u_message',umsg,umsg.shape,'DynamicPrivate');amsg=np.asarray([[hx((int(alpha[k],16)*int(routed[k,j]))%FR) for j in range(dout)] for k in range(e)],dtype=object);amn=b.tensor(prefix+'.alpha_message',amsg,amsg.shape,'DynamicPrivate')
 emap=np.repeat(np.arange(e),dout).tolist();b.op('u_message_hadamard',prefix+'.u_message_relation',logical,[b.explicit('u',un,emap,rr),b.affine('h',rpn,0,1,logical,rr),b.affine('msg',umn,0,1,logical,rr)],[S(X(2),M(X(0),X(1)))]);b.op('alpha_message_hadamard',prefix+'.alpha_message_relation',logical,[b.explicit('alpha',aln,emap,rr),b.affine('h',rpn,0,1,logical,rr),b.affine('msg',amn,0,1,logical,rr)],[S(X(2),M(X(0),X(1)))])
 numacc,num=group_acc(umsg,dst,n);fac_fields=np.asarray([[int(v,16) for v in row] for row in amsg],dtype=object);facc,fagg=group_acc(fac_fields,dst,n);fan=b.tensor(prefix+'.field_aggregate',[[hx(v) for v in row] for row in fagg],fagg.shape,'DynamicPrivate');nan=b.tensor(prefix+'.integer_numerator',num,num.shape,'DynamicPrivate');naccn=b.tensor(prefix+'.num_acc',numacc,numacc.shape,'DynamicPrivate');faccn=b.tensor(prefix+'.field_acc',[[hx(v) for v in row] for row in facc],facc.shape,'DynamicPrivate')
 prev_idx=[0]*logical
 for k in range(e):
  for j in range(dout):prev_idx[k*dout+j]=(k-1)*dout+j if k and dst[k-1]==dst[k] else k*dout+j
 start_dim=np.repeat(start,dout);end_dim=np.repeat(end,dout);dst_dim=(dst.repeat(dout)*dout+np.tile(np.arange(dout),e)).tolist()
 b.op('integer_numerator',prefix+'.integer_aggregate',logical,[b.affine('msg',umn,0,1,logical,rr),b.affine('acc',naccn,0,1,logical,rr),b.explicit('prev',naccn,prev_idx,rr),b.public('start',start_dim,rr),b.public('end',end_dim,rr),b.explicit('out',nan,dst_dim,rr)],[S(X(1),A(M(S(C(1),X(3)),X(2)),X(0))),M(X(4),S(X(1),X(5)))])
 b.op('field_aggregate',prefix+'.field_aggregate_relation',logical,[b.affine('msg',amn,0,1,logical,rr),b.affine('acc',faccn,0,1,logical,rr),b.explicit('prev',faccn,prev_idx,rr),b.public('start',start_dim,rr),b.public('end',end_dim,rr),b.explicit('out',fan,dst_dim,rr)],[S(X(1),A(M(S(C(1),X(3)),X(2)),X(0))),M(X(4),S(X(1),X(5)))])
 nd=n*dout;ndr=npow(nd);sum_idx=np.repeat(np.arange(n),dout).tolist();b.op('dual_aggregate_consistency',prefix+'.dual_consistency',nd,[b.affine('num',nan,0,1,nd,ndr),b.explicit('sum',sumn,sum_idx,ndr),b.affine('field',fan,0,1,nd,ndr)],[S(X(0),M(X(1),X(2)))])
 xscale=1 if hidden else 16;dscale=16 if hidden else 256;x=(np.asarray(num,dtype=object)*xscale).reshape(-1);dv=np.repeat(np.asarray(sumv[:,0],dtype=object)*dscale,dout);pre=np.asarray([sdiv(int(a),int(d)) for a,d in zip(x,dv)],dtype=object);xn=b.tensor(prefix+'.recovery_x',x,(nd,),'DynamicPrivate');ddn=b.tensor(prefix+'.recovery_d',dv,(nd,),'DynamicPrivate');pren=b.tensor(prefix+'.pre',pre,(n,dout),'DynamicPrivate');add_div(b,prefix+'.recovery_div',xn,ddn,pren,x,dv,width)
 if hidden:
  agg=np.asarray([elu(int(v)) for v in pre],dtype=object).reshape(n,dout);aggn=b.tensor(prefix+'.aggregate',agg,agg.shape,'DynamicPrivate');add_lookup(b,'elu_indexed_public_table',prefix+'.elu',pren,aggn,pre,tables['elu']['min'],tables['elu']['max'],elu);return agg,aggn
 align_x=pre*16;align_d=np.full(nd,16,dtype=object);axn=b.tensor(prefix+'.alignment_x',align_x,(nd,),'DynamicPrivate');adn=b.tensor(prefix+'.alignment_d',align_d,(nd,),'Public');aligned=np.asarray([sdiv(int(a),16) for a in align_x],dtype=object);algn=b.tensor(prefix+'.aligned',aligned,(n,dout),'DynamicPrivate');add_div(b,prefix+'.alignment_div',axn,adn,algn,align_x,align_d,width)
 bn=b.tensor(prefix+'.bias',bias,(dout,),'RegisteredModel');head=aligned.reshape(n,dout)+bias.reshape(1,dout);hn=b.tensor(prefix+'.head_output',head,head.shape,'DynamicPrivate');bias_idx=np.tile(np.arange(dout),n).tolist();b.op('output_bias',prefix+'.bias_add',nd,[b.affine('aligned',algn,0,1,nd,ndr),b.explicit('bias',bn,bias_idx,ndr),b.affine('head',hn,0,1,nd,ndr)],[S(X(2),A(X(0),X(1)))]);return head,hn

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--preflight',type=Path,required=True);ap.add_argument('--pp-id',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 config_path=ensure_contained(a.config,label='formal config');
 if config_path.relative_to(ROOT).parts[:1] != ('configs',):raise SystemExit('formal config must be baseline-local configs/<dataset>.cfg')
 c=cfg(config_path);root=ensure_contained(a.project_root,label='project-root');
 if root != BASELINE_ROOT:raise SystemExit('project-root must be comparison/zkdl; stage resources locally first')
 resource_manifest,resource_manifest_id=load_resource_manifest(require_staged=True)
 validate_config_resources(c.get('dataset',''),c,resource_manifest)
 preflight_path=ensure_contained(a.preflight,label='preflight report');
 if preflight_path.relative_to(ROOT).parts[:1] != ('runs',):raise SystemExit('preflight must be written below baseline runs/')
 report=json.loads(preflight_path.read_text());
 if report['status']!='accepted':raise SystemExit('canonical adapter refuses a preflight-rejected formal instance')
 if report.get('resource_manifest_id') != resource_manifest_id:raise ValueError('preflight/resource manifest mismatch')
 cache=root/c['cache'];cp=root/c['checkpoint_bundle'];manifest=json.loads((cp/'manifest.json').read_text());raw=load_tensors(cp/'tensors.txt');n=int((cache/'meta.cfg').read_text().split('num_nodes = ')[1].splitlines()[0]);d=int((cache/'meta.cfg').read_text().split('num_features = ')[1].splitlines()[0]);H,feature_path=load_features(cache,n,d,int(c['feature_scale']));src,dst,start,end,node_ptr,edge_ptr=load_edges(cache,n);tables=report['static_tables'];width=max(2,int(report['ledger'].get('bit_width',int(c['range_bits'])+16)));b=Builder();Hname=b.tensor('public.features',H,H.shape,'Public');b.tensor('public.edge_src',src,src.shape,'Public');b.tensor('public.edge_dst',dst,dst.shape,'Public');b.tensor('public.node_ptr',node_ptr,node_ptr.shape,'Public');b.tensor('public.edge_ptr',edge_ptr,edge_ptr.shape,'Public')
 expected=report['public_instance']
 if [int(x) for x in node_ptr]!=expected['node_ptr'] or [int(x) for x in edge_ptr]!=expected['edge_ptr'] or len(src)!=expected['edge_count']:raise ValueError('canonical topology differs from authenticated preflight instance')
 H2,H2n=H,Hname
 hidden_specs=manifest['hidden_head_specs']
 for layer_meta in manifest['hidden_layers']:
  layer=int(layer_meta['layer_index']);layer_specs=sorted((x for x in hidden_specs if int(x['layer_index'])==layer),key=lambda x:int(x['local_head_index']))
  if len(layer_specs)!=int(layer_meta['head_count']) or H2.shape[1]!=int(layer_meta['input_dim']):raise ValueError(f'hidden layer {layer} manifest dimension mismatch')
  hidden=[]
  for spec in layer_specs:
   dout=int(layer_meta['head_dim']);W=np.asarray([q(v,int(c['hidden_weight_scale'])) for v in raw[spec['seq_kernel']][1]],dtype=object).reshape(H2.shape[1],dout);aa=np.asarray([q(v,int(c['hidden_attention_scale'])) for v in raw[spec['attn_src_kernel']][1]],dtype=object);ad=np.asarray([q(v,int(c['hidden_attention_scale'])) for v in raw[spec['attn_dst_kernel']][1]],dtype=object);prefix=f"hidden.layer_{layer}.head_{spec['local_head_index']}";hidden.append(build_head(b,prefix,H2,H2n,W,aa,ad,src,dst,start,end,True,None,tables,width))
  next_h=np.concatenate([x[0] for x in hidden],axis=1);next_name=b.tensor(f'hidden.layer_{layer}.concat',next_h,next_h.shape,'DynamicPrivate');cursor=0
  for head,(vals,name) in enumerate(hidden):
   for j in range(vals.shape[1]):
    idx=(np.arange(n)*vals.shape[1]+j).tolist();outidx=(np.arange(n)*next_h.shape[1]+cursor+j).tolist();b.op('hidden_concat',f'hidden.layer_{layer}.concat.head_{head}.col_{j}',n,[b.explicit('head',name,idx,npow(n)),b.explicit('concat',next_name,outidx,npow(n))],[S(X(1),X(0))])
   cursor+=vals.shape[1]
  H2,H2n=next_h,next_name
 outputs=[]
 for spec in manifest['output_head_specs']:
  classes=int(manifest['C']);W=np.asarray([q(v,int(c['output_weight_scale'])) for v in raw[spec['seq_kernel']][1]],dtype=object).reshape(H2.shape[1],classes);aa=np.asarray([q(v,int(c['output_attention_scale'])) for v in raw[spec['attn_src_kernel']][1]],dtype=object);ad=np.asarray([q(v,int(c['output_attention_scale'])) for v in raw[spec['attn_dst_kernel']][1]],dtype=object);bias=np.asarray([q(v,int(c['output_bias_scale'])) for v in raw[spec['output_bias']][1]],dtype=object);prefix=f"output.head_{spec['head_index']}";outputs.append(build_head(b,prefix,H2,H2n,W,aa,ad,src,dst,start,end,False,bias,tables,width))
 heads=[x[0] for x in outputs];ysum=sum(heads[1:],heads[0].copy());ysn=b.tensor('output.head_sum',ysum,ysum.shape,'DynamicPrivate');final=np.asarray([sdiv(int(v),len(heads)) for v in ysum.reshape(-1)],dtype=object).reshape(ysum.shape);final_private=b.tensor('output.final_logits',final,final.shape,'DynamicPrivate');yn=b.tensor('public.logits',final,final.shape,'Public');nd=final.size;rows=npow(nd);cols=[b.affine(f'head_{i}',name,0,1,nd,rows) for i,(_,name) in enumerate(outputs)]+[b.affine('sum',ysn,0,1,nd,rows)];b.op('head_sum_final_average','output.head_sum',nd,cols,[S(X(len(outputs)),sumexpr([X(i) for i in range(len(outputs))]))]);kn=b.tensor('public.k_out',[len(outputs)]*nd,(nd,),'Public');add_div(b,'output.final_average',ysn,kn,final_private,ysum.reshape(-1),np.full(nd,len(outputs),dtype=object),width);b.ops[-1]['operation_kind']='head_sum_final_average';b.op('public_logits_terminal','output.public_logits',nd,[b.affine('private_final',final_private,0,1,nd,rows),b.affine('public_y',yn,0,1,nd,rows)],[S(X(0),X(1))])
 families={x['operation_kind'] for x in b.ops};missing=set(['projection_matmul','attention_dot','public_sparse_route','score_add','lrelu_indexed_public_table','exact_group_max','exp_indexed_public_table','group_sum','inverse_alpha','u_message_hadamard','alpha_message_hadamard','field_aggregate','integer_numerator','dual_aggregate_consistency','signed_round_div','elu_indexed_public_table','hidden_concat','output_bias','head_sum_final_average','public_logits_terminal'])-families
 if missing:raise ValueError('canonical relation family missing: '+','.join(sorted(missing)))
 tensor_digest=canonical_tensor_digest(b.spec,b.tensors)
 shape_payload={'hidden_layers':manifest['hidden_layers'],'hidden_head_specs':manifest['hidden_head_specs'],'output_head_specs':manifest['output_head_specs'],'class_count':int(manifest['C']),'feature_dim':d}
 model_shape_id='sha256:'+hashlib.sha256(json.dumps(shape_payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 checkpoint_manifest_id='sha256:'+digest(cp/'manifest.json');static_rows_id=static_table_rows_digest(b.ops);commitment_set_id=registered_family_set_digest(b.spec);config_id='sha256:'+hashlib.sha256(config_path.read_bytes()).hexdigest()
 for key,value in [('node_count',n),('edge_count',len(src)),('graph_count',len(node_ptr)-1),('feature_dim',d),('class_count',int(manifest['C'])),('checkpoint_manifest_id',checkpoint_manifest_id),('model_shape_id',model_shape_id),('static_tables_id',report['static_tables_id']),('static_table_rows_id',static_rows_id),('model_arch_id',manifest['model_arch_id']),('model_param_id',manifest['model_param_id']),('config_sha256',config_id)]:
  if key not in report:raise ValueError('preflight metadata missing: '+key)
  if report[key] != value:raise ValueError('canonical/preflight metadata mismatch: '+key)
 if report.get('static_table_rows_id') != static_rows_id:raise ValueError('canonical/preflight static table row digest mismatch')
 for key in ('graph_count','feature_dim','class_count','resource_manifest_id','config_sha256','checkpoint_manifest_id','model_shape_id','static_tables_id','static_table_rows_id','model_arch_id','model_param_id'):
  if report['ledger'].get(key) != report[key]:raise ValueError('preflight ledger metadata mismatch: '+key)
 lookup_bits={name:max(1,npow(int(tables[name]['rows'])).bit_length()-1) for name in ('lrelu','elu','exp')}
 architecture={'model_arch_id':manifest['model_arch_id'],'model_param_id':manifest['model_param_id'],'node_count':n,'edge_count':len(src),'graph_count':len(node_ptr)-1,'feature_dim':d,'class_count':int(manifest['C']),'output_head_count':len(manifest['output_head_specs']),'bit_width':width,'lookup_bits':lookup_bits,'hidden_layers':manifest['hidden_layers'],'hidden_head_specs':manifest['hidden_head_specs'],'output_head_specs':manifest['output_head_specs']}
 canonical_schema={'schema':CANONICAL_RELATION_SCHEMA,'version':1,'architecture':architecture,'tensor_family_order':[x['family_id'] for x in b.spec],'tensor_families':b.spec,'operation_family_order':[x['family_id'] for x in b.ops],'operation_families':b.ops,'fac_plan':canonical_fac_plan(b.ops,b.spec),'terminal_plan':{'public_logits_operation':'output.public_logits','public_logits_family':'public.logits','target':'verifier_local_public_y'}}
 validate_canonical_schema(canonical_schema,b.spec,b.ops,architecture)
 obj={'schema':ADAPTER_SCHEMA,'adapter_schema':ADAPTER_SCHEMA,'adapter_version':ADAPTER_VERSION,'protocol_sha256':protocol_digest(),'rounding_mode':c['rounding_mode'],'canonical_relation_digest':'','canonical_schema':canonical_schema,'dataset':c['dataset'],'pp_id':a.pp_id,'node_count':n,'edge_count':len(src),'graph_count':len(node_ptr)-1,'feature_dim':d,'class_count':int(manifest['C']),'public_features':[int(x) for x in H.reshape(-1)],'node_ptr':[int(x) for x in node_ptr],'edge_ptr':[int(x) for x in edge_ptr],'src':[int(x) for x in src],'dst':[int(x) for x in dst],'graph_features_digest':report['graph_features_digest'],'topology_digest':report['topology_digest'],'checkpoint_manifest_id':checkpoint_manifest_id,'canonical_tensor_digest':tensor_digest,'model_shape_id':model_shape_id,'commitment_set_id':commitment_set_id,'quant_cfg_id':report['quant_cfg_id'],'static_tables_id':report['static_tables_id'],'static_table_rows_id':static_rows_id,'static_table_profile':tables,'integer_bound_ledger_id':report['ledger']['integer_bound_ledger_id'],'integer_bound_ledger':report['ledger'],'model_arch_id':manifest['model_arch_id'],'model_param_id':manifest['model_param_id'],'public_logits':[int(x) for x in final.reshape(-1)],'tensor_families':b.spec,'operation_families':b.ops,'tensors':dict(sorted(b.tensors.items())),'resource_manifest_id':resource_manifest_id,'config_sha256':config_id,'source_input_id':''}
 obj['source_input_id']=source_input_digest(obj);obj['canonical_relation_digest']=relation_digest(obj);output=ensure_contained(a.output,label='canonical relation output');
 if output.relative_to(ROOT).parts[:1] != ('runs',):raise SystemExit('canonical relation output must be written below baseline runs/')
 atomic_write_bytes(output,(json.dumps(obj,separators=(',',':'),ensure_ascii=False)+'\n').encode(),mode=0o600);print('CANONICAL_ADAPTER_OK',obj['source_input_id'],'max_tensor_len',max(x['logical_len'] for x in b.spec))
if __name__=='__main__':main()
