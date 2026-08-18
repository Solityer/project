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

ROOT=Path(__file__).resolve().parents[1]
FR=0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

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
 return max(0,q(16.0*math.exp(-x/256.0),1))
def npow(n):return max(2,1<<(max(1,n)-1).bit_length())
def hx(v):return '0x'+(int(v)%FR).to_bytes(32,'big').hex()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_features(cache,n,d):
 p=cache/'features.txt'
 if p.exists():
  out=np.zeros((n,d),dtype=np.int64)
  with p.open() as f:
   if tuple(map(int,f.readline().split()))!=(n,d):raise ValueError('feature header mismatch')
   for row,line in enumerate(f):
    for item in line.split():
     col,val=item.split(':',1);out[row,int(col)]=q(float(val),1)
  if row+1!=n:raise ValueError('feature row count mismatch')
  return out,p
 raw=np.load(cache/'features.npy',mmap_mode='r')
 if raw.shape!=(n,d):raise ValueError('feature shape mismatch')
 out=np.where(raw>=0,np.floor(raw+.5),np.ceil(raw-.5)).astype(np.int64)
 return out,cache/'features.npy'
def load_edges(cache,n):
 raw=[]
 for stable,line in enumerate((cache/'edges.txt').read_text().splitlines()):
  p=list(map(int,line.split()));g,s,d=(0,p[0],p[1]) if len(p)==2 else tuple(p)
  raw.append((g,s,d,stable))
 ptr=list(map(int,(cache/'node_ptr.txt').read_text().split())) if (cache/'node_ptr.txt').exists() else [0,n]
 gid=np.zeros(n,dtype=np.int64)
 for g,(lo,hi) in enumerate(zip(ptr,ptr[1:])):gid[lo:hi]=g
 loops={s for _,s,d,_ in raw if s==d}
 for node in range(n):
  if node not in loops:raw.append((int(gid[node]),node,node,len(raw)))
 raw.sort(key=lambda e:(e[0],e[2],e[3]))
 src=np.asarray([e[1] for e in raw],dtype=np.int64);dst=np.asarray([e[2] for e in raw],dtype=np.int64)
 start=np.zeros(len(raw),dtype=np.int64);end=np.zeros(len(raw),dtype=np.int64)
 for i in range(len(raw)):start[i]=int(i==0 or dst[i-1]!=dst[i]);end[i]=int(i+1==len(raw) or dst[i+1]!=dst[i])
 return src,dst,start,end
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
  flat=[int(x) if isinstance(x,(int,np.integer)) else x for x in flat]
  self.tensors[name]=flat;self.spec.append({'family_id':name,'logical_len':len(flat),'padded_len':npow(len(flat)),'logical_shape':list(shape),'mle_order':['row_major_lsb_first'],'instance_order':['canonical_program_order'],'visibility':visibility});return name
 def view(self,name,family,mapping):return {'column_id':name,'source':{'TensorView':{'family_id':family,'canonical_index_by_row':mapping}}}
 def affine(self,name,family,offset,stride,logical,rows,pad=0):
  index={'Affine':{'offset':offset,'stride':stride}} if logical==rows else {'PaddedAffine':{'offset':offset,'stride':stride,'logical_rows':logical,'padding_index':pad}}
  return self.view(name,family,index)
 def explicit(self,name,family,index,rows):return self.view(name,family,{'Explicit':[int(x) for x in index]+[0]*(rows-len(index))})
 def public(self,name,values,rows):return {'column_id':name,'source':{'Public':list(map(int,values))+[0]*(rows-len(values))}}
 def op(self,kind,label,logical,columns,residuals,degree=None):
  rows=npow(logical);columns=list(columns)+[self.public('valid',[1]*logical,rows)];v=len(columns)-1;res=[valid(x,v) for x in residuals]
  deg=max((expr_degree(x) for x in res),default=1)+1
  self.ops.append({'family_id':label,'operation_kind':kind,'rows':rows,'degree_bound':max(degree or deg,deg),'columns':columns,'residuals':res,'requires_line8':sum('TensorView' in c['source'] for c in columns)>1})

def expr_degree(e):
 if 'Const' in e:return 0
 if 'Column' in e:return 1
 if 'Add' in e:return max(map(expr_degree,e['Add']))
 if 'Sub' in e:return max(map(expr_degree,e['Sub']))
 if 'Mul' in e:return sum(map(expr_degree,e['Mul']))
 if 'PublicTableLookup' in e:return len(e['PublicTableLookup']['bit_columns'])
 raise ValueError('unknown expression')

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
 data={k:[] for k in ['sigma','z','a','a_inv','d_inv','q','r','delta','y']}
 for xv,dv in zip(np.asarray(x,dtype=object).reshape(-1),np.asarray(d,dtype=object).reshape(-1)):
  xv=int(xv);dv=int(dv);yv=sdiv(xv,dv);a=abs(xv);qv=abs(yv);sig=int(xv<0);z=int(a==0);r=2*a+dv-2*dv*qv;delta=2*dv-1-r
  vals=(sig,z,a,0 if z else hx(field_inverse(a)),hx(field_inverse(dv)),qv,r,delta,yv)
  for k,v in zip(data,vals):data[k].append(v)
 return data
def bits_for(values,width):return [np.asarray([(int(v)>>j)&1 for v in values],dtype=np.int64) for j in range(width)]
def add_div(b,label,xname,dname,yname,x,d,width):
 aux=div_aux(x,d);logical=len(aux['a']);rows=npow(logical);names={}
 for k in ['sigma','z','a','a_inv','d_inv','q','r','delta']:names[k]=b.tensor(f'{label}.{k}',aux[k],(logical,),'DynamicPrivate')
 bitnames={}
 for k in ['a','q','r','delta']:
  bitnames[k]=[b.tensor(f'{label}.{k}_bit_{j}',v,(logical,),'DynamicPrivate') for j,v in enumerate(bits_for(aux[k],width))]
 cols=[b.affine('x',xname,0,1,logical,rows),b.affine('d',dname,0,1,logical,rows),b.affine('y',yname,0,1,logical,rows)]
 pos={}
 for k in names:pos[k]=len(cols);cols.append(b.affine(k,names[k],0,1,logical,rows))
 bitpos={}
 for k,arr in bitnames.items():
  bitpos[k]=[]
  for j,n in enumerate(arr):bitpos[k].append(len(cols));cols.append(b.affine(f'{k}_bit_{j}',n,0,1,logical,rows))
 sig=X(pos['sigma']);z=X(pos['z']);a=X(pos['a']);ainv=X(pos['a_inv']);dinv=X(pos['d_inv']);qq=X(pos['q']);rr=X(pos['r']);dd=X(1)
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
 c=cfg(a.config);root=a.project_root.resolve();report=json.loads(a.preflight.read_text());
 if report['status']!='accepted':raise SystemExit('canonical adapter refuses a preflight-rejected formal instance')
 cache=root/c['cache'];cp=root/c['checkpoint_bundle'];manifest=json.loads((cp/'manifest.json').read_text());raw=load_tensors(cp/'tensors.txt');n=int((cache/'meta.cfg').read_text().split('num_nodes = ')[1].splitlines()[0]);d=int((cache/'meta.cfg').read_text().split('num_features = ')[1].splitlines()[0]);H,feature_path=load_features(cache,n,d);src,dst,start,end=load_edges(cache,n);tables=report['static_tables'];width=max(2,int(c['range_bits'])+16);b=Builder();Hname=b.tensor('public.features',H,H.shape,'Public');b.tensor('public.edge_src',src,src.shape,'Public');b.tensor('public.edge_dst',dst,dst.shape,'Public')
 hidden=[]
 for spec in manifest['hidden_head_specs']:
  W=np.asarray([q(v,16) for v in raw[spec['seq_kernel']][1]],dtype=object).reshape(d,int(manifest['hidden_layers'][0]['head_dim']));aa=np.asarray([q(v,16) for v in raw[spec['attn_src_kernel']][1]],dtype=object);ad=np.asarray([q(v,16) for v in raw[spec['attn_dst_kernel']][1]],dtype=object);prefix=f"hidden.layer_0.head_{spec['local_head_index']}";hidden.append(build_head(b,prefix,H,Hname,W,aa,ad,src,dst,start,end,True,None,tables,width))
 H2=np.concatenate([x[0] for x in hidden],axis=1);H2n=b.tensor('hidden.concat',H2,H2.shape,'DynamicPrivate');logical=H2.size;rows=npow(logical);cols=[];cursor=0
 for head,(vals,name) in enumerate(hidden):
  for j in range(vals.shape[1]):
   idx=(np.arange(n)*vals.shape[1]+j).tolist();outidx=(np.arange(n)*H2.shape[1]+cursor+j).tolist();b.op('hidden_concat',f'hidden.concat.head_{head}.col_{j}',n,[b.explicit('head',name,idx,npow(n)),b.explicit('concat',H2n,outidx,npow(n))],[S(X(1),X(0))])
  cursor+=vals.shape[1]
 outputs=[]
 for spec in manifest['output_head_specs']:
  classes=int(manifest['C']);W=np.asarray([q(v,16) for v in raw[spec['seq_kernel']][1]],dtype=object).reshape(H2.shape[1],classes);aa=np.asarray([q(v,1) for v in raw[spec['attn_src_kernel']][1]],dtype=object);ad=np.asarray([q(v,1) for v in raw[spec['attn_dst_kernel']][1]],dtype=object);bias=np.asarray([q(v,16) for v in raw[spec['output_bias']][1]],dtype=object);prefix=f"output.head_{spec['head_index']}";outputs.append(build_head(b,prefix,H2,H2n,W,aa,ad,src,dst,start,end,False,bias,tables,width))
 heads=[x[0] for x in outputs];ysum=sum(heads[1:],heads[0].copy());ysn=b.tensor('output.head_sum',ysum,ysum.shape,'DynamicPrivate');final=np.asarray([sdiv(int(v),len(heads)) for v in ysum.reshape(-1)],dtype=object).reshape(ysum.shape);yn=b.tensor('public.logits',final,final.shape,'Public');nd=final.size;rows=npow(nd);cols=[b.affine(f'head_{i}',name,0,1,nd,rows) for i,(_,name) in enumerate(outputs)]+[b.affine('sum',ysn,0,1,nd,rows)];b.op('head_sum_final_average','output.head_sum',nd,cols,[S(X(len(outputs)),sumexpr([X(i) for i in range(len(outputs))]))]);kn=b.tensor('public.k_out',[len(outputs)]*nd,(nd,),'Public');add_div(b,'output.final_average',ysn,kn,yn,ysum.reshape(-1),np.full(nd,len(outputs),dtype=object),width);b.ops[-1]['operation_kind']='head_sum_final_average';b.op('public_logits_terminal','output.public_logits',nd,[b.affine('private_sum',yn,0,1,nd,rows),b.affine('public_y',yn,0,1,nd,rows)],[S(X(0),X(1))])
 families={x['operation_kind'] for x in b.ops};missing=set(['projection_matmul','attention_dot','public_sparse_route','score_add','lrelu_indexed_public_table','exact_group_max','exp_indexed_public_table','group_sum','inverse_alpha','u_message_hadamard','alpha_message_hadamard','field_aggregate','integer_numerator','dual_aggregate_consistency','signed_round_div','elu_indexed_public_table','hidden_concat','output_bias','head_sum_final_average','public_logits_terminal'])-families
 if missing:raise ValueError('canonical relation family missing: '+','.join(sorted(missing)))
 obj={'schema':'zkDL_gat_canonical_adapter_v2','dataset':c['dataset'],'pp_id':a.pp_id,'graph_features_digest':report['graph_features_digest'],'topology_digest':report['topology_digest'],'quant_cfg_id':report['quant_cfg_id'],'static_tables_id':report['static_tables_id'],'integer_bound_ledger_id':report['ledger']['integer_bound_ledger_id'],'model_arch_id':manifest['model_arch_id'],'model_param_id':manifest['model_param_id'],'public_logits':[int(x) for x in final.reshape(-1)],'tensor_families':b.spec,'operation_families':b.ops,'tensors':dict(sorted(b.tensors.items())),'source_input_id':''}
 canonical=json.dumps(obj,separators=(',',':'),ensure_ascii=False).encode();obj['source_input_id']='sha256:'+hashlib.sha256(b'zkDL-GAT-canonical-adapter-input-v2\0'+canonical).hexdigest();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(obj,separators=(',',':'))+'\n');print('CANONICAL_ADAPTER_OK',obj['source_input_id'],'max_tensor_len',max(x['logical_len'] for x in b.spec))
if __name__=='__main__':main()
