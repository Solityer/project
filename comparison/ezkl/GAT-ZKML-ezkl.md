# GAT-ZKML ezkl 协议笔记

## 0. 符号

### 0.0 证明关系与对比实验范围

本文定义的 comparison relation 记为：

$$
R_{GAT\text{-}ezkl}.
$$

其逻辑公共输入为：

$$
Public_{ezkl}=(M_{pub},T_H,I,src,dst,node\_ptr,edge\_ptr,Y,H(W),CircuitRef),
$$

其中 $CircuitRef$ 唯一引用本次实验使用的 `compiled_circuit_id`、`settings_id` 与 $vk_{ezkl}$；$Y$ 是完整 $N\times C$ 公开 logits。

其私有见证为：

$$
Witness_{ezkl}=(W,Trace_{GAT},Aux_{max},Aux_{div},Aux_{lookup}).
$$

本文的比较对象不是“把主实验 KZG 协议改名为 ezkl”，而是把同一 GAT 定点关系编译到 ezkl/Halo2 后端。原论文直接提供的机制包括：ONNX-to-circuit、cumulative arguments、elementwise arguments、logUP lookup、Halo2 proof、private model weights 与公开模型哈希 $H(W)$。GAT 稀疏路由、stable Softmax、主实验 exact table、`SignedRoundDiv` 和完整多头输出属于本文为 GAT 增加的 backend adaptation；这些 adaptation 只能由 ezkl/Halo2 原生 gate/lookup/argument 组合实现。

因此本文同时满足两项要求：

1. 被证明的数值函数与主实验相同；
2. 用于证明该函数的密码学机制属于 ezkl/Halo2，而不是主实验证明系统。


### 0.1 模型结构

GAT 的前向固定为一个可参数化的多层多头结构。本文改变的是证明后端，不改变 GAT 的计算关系。

1. 设注意力层总数为 $L$。其中前 $L-1$ 层为隐藏注意力层，第 $L$ 层为最终输出注意力层。公开模型结构参数必须满足：

$$
L\ge2,\qquad K_{out}\ge1,\qquad C\ge1,
$$

并且对所有 $\ell\in\{1,\ldots,L-1\}$，

$$
K_{hid}^{(\ell)},\ d_h^{(\ell)},\ d_{in}^{(\ell)}\in\mathbb Z_{>0}.
$$

2. 对隐藏层 $\ell\in\{1,\ldots,L-1\}$，头数为 $K_{hid}^{(\ell)}$，每头输出维度为 $d_h^{(\ell)}$。每个隐藏头依次执行投影、源/目标 attention score、LeakyReLU、stable Softmax、聚合、定点整数恢复和 ELU。

3. 同一隐藏层的全部头在 ELU 后沿特征维拼接：

$$
d_{cat}^{(\ell)}=K_{hid}^{(\ell)}d_h^{(\ell)}.
$$

4. 层间维度满足：

$$
d_{in}^{(1)}=d_{in},\qquad d_{in}^{(\ell+1)}=d_{cat}^{(\ell)}.
$$

5. 最终输出层有 $K_{out}\ge1$ 个完整 GAT 输出注意力头。每个输出头仍执行 projection、attention、stable Softmax 与聚合；输出头不施加 ELU，而是在聚合结果完成定点尺度对齐以后加入本头偏置 $b^{(out,s)}$。

6. 全部输出头在共同尺度上求和后，用规范整数除法执行按头平均，得到公开 logits：

$$
Y\in\mathbb Z^{N\times C}.
$$

本对比实验的公开/私有边界固定为：

- `proof_task = node_classification_full_gat`；
- `privacy_policy = model_and_trace_hidden`；
- `input_privacy = public_graph_public_features`；
- `output_mode = public_logits`。

隐藏头编号记为：

$$
(\ell,r),\qquad \ell\in\{1,\ldots,L-1\},\quad r\in\{0,\ldots,K_{hid}^{(\ell)}-1\}.
$$

输出头编号记为：

$$
(out,s),\qquad s\in\{0,\ldots,K_{out}-1\}.
$$

正文为了公式紧凑可以局部省略 $\ell$；协议中的 tensor、commitment、proof label 与 transcript tag 均必须保留完整层号与头号。

### 0.2 ezkl / Halo2 证明后端

本文采用 ezkl 论文的 ONNX-to-zkSNARK 路线。证明系统把 ML 运算分解为若干可组合 argument，并由 Halo2 / Plonkish 后端生成最终 proof。

本文明确使用 ezkl 原论文中的四类基础 argument：

1. cumulative dot-product argument；
2. cumulative sum / product argument；
3. elementwise add / mul / sub argument；
4. lookup argument。

ezkl 对 Halo2 lookup 做了修改，非线性 lookup 使用 ezkl 自身的 logUP lookup。这里的 `logUP` 是 ezkl backend 原生 lookup argument，不是主实验的 LogUp。

复杂关系由上述 argument 组合。例如原论文给出的 max argument 是由 claimed max、减法、ReLU lookup、boolean selector 与“至少一个元素等于最大值”的关系组合得到，而不是引入另一套 lookup proof system。

量化后的 GAT 计算图记为：

$$
G_{ezkl}=ExportGAT_{ezkl}(M_{pub},quant\_cfg,Topo).
$$

编译电路：

$$
C_{ezkl}=Compile_{ezkl}(G_{ezkl},settings).
$$

setup / compile 生成：

$$
(pk_{ezkl},vk_{ezkl},C_{ezkl},settings\_id).
$$

所选 ezkl/Halo2 的密码学参数必须由公开的 `ezkl_security_profile_id` 唯一固定。该 profile 至少绑定：pinned ezkl/Halo2 version、实际使用的曲线/群与标量域、PCS/commitment backend、Fiat--Shamir hash/transcript suite、setup 参数以及该 native configuration 对应的安全级别声明。本文不预设一个脱离实际工具链的曲线名称或安全位数；正式实验记录并比较所使用版本的真实 native security profile，不得为了与其他 backend 表面一致而替换曲线或改写安全级别。

Halo2 内部 polynomial commitment、quotient、opening 与 Fiat--Shamir transcript 由 `ezkl_security_profile_id` 所绑定的 ezkl/Halo2 原生协议参数定义；本文不把主实验 KZG 结构移植进来。

### 0.3 图、局部编号、多图 batch 与稳定排序

记：

- 全局节点数 $N_{total}$；
- 当前实例节点数 $N$；
- 当前实例边数 $E$；
- batch 内图数 $G_{batch}$；
- 节点绝对编号序列 $I=(I_0,\ldots,I_{N-1})$；
- 边源/目标索引 $src,dst$；
- 节点前缀指针 $node\_ptr=(\nu_0,\ldots,\nu_{G_{batch}})$；
- 边前缀指针 $edge\_ptr=(\epsilon_0,\ldots,\epsilon_{G_{batch}})$。

要求公共实例首先满足：

$$
N_{total}\ge N\ge1,\qquad E\ge1,\qquad G_{batch}\ge1,
$$

$$
|I|=N,\qquad 0\le I_i<N_{total},
$$

并且节点绝对编号不得重复：

$$
I_i\ne I_j\qquad(i\ne j).
$$

边数组与前缀指针长度必须满足：

$$
|src|=|dst|=E,
$$

$$
|node\_ptr|=|edge\_ptr|=G_{batch}+1.
$$

同时：

$$
0=\nu_0<\cdots<\nu_{G_{batch}}=N,
$$

$$
0=\epsilon_0<\cdots<\epsilon_{G_{batch}}=E.
$$

定义：

$$
node\_gid(i)=g\iff \nu_g\le i<\nu_{g+1},
$$

$$
edge\_gid(k)=g\iff \epsilon_g\le k<\epsilon_{g+1}.
$$

每条有效边满足：

$$
src(k),dst(k)\in\{0,\ldots,N-1\},
$$

$$
node\_gid(src(k))=node\_gid(dst(k))=edge\_gid(k).
$$

公开预处理执行：

1. 必要时加入自环，使每个有效节点至少有一条入边；
2. 先按 `edge_gid` 分图块；
3. 每个图块内部按 $dst$ 做确定性稳定非降序排序；
4. 生成 `node_ptr,edge_ptr` 和当前 backend 所需 selector / padding layout。

若 $G_{batch}=1$，仍显式使用：

$$
node\_ptr=(0,N),\qquad edge\_ptr=(0,E).
$$

所有 group max、Softmax 分母、聚合和组状态机必须限制在同一图、同一 $dst$ 组内。

定义 batch-aware 组起点和组末选择器：

$$
Q_{grp,new}[k]=1
$$

当且仅当 $k$ 是某个图块内某个 $dst$ 组的第一条有效边；类似地 $Q_{grp,end}[k]=1$ 当且仅当 $k$ 是该组最后一条有效边。两者由公开 $src,dst,node\_ptr,edge\_ptr$ 唯一重建。

### 0.4 静态表、定点尺度与规范整数语义

#### 0.4.1 公开特征与函数表

全局输入特征表为公开数据：

$$
T_H\in\mathbb Z^{N_{total}\times d_{in}}.
$$

当前实例第一层输入：

$$
H_{i,j}^{(1)}=T_H[I_i,j].
$$

静态函数表包括：

$$
T_{LReLU}\subseteq\mathbb Z\times\mathbb Z,
$$

$$
T_{ELU}\subseteq\mathbb Z\times\mathbb Z,
$$

$$
T_{exp}\subseteq\mathbb Z_{\ge0}\times\mathbb Z_{>0}.
$$

指数表输入为 stable Softmax 的非负差分：

$$
\Delta^+=M-Z\ge0.
$$

定义：

$$
(\Delta,ExpMap(\Delta))\in T_{exp},
$$

其中：

$$
ExpMap(\Delta)=QuantExpNeg(\Delta;quant\_cfg),
$$

其目标定点语义为：

$$
ExpMap(\Delta)\approx S_{exp}\exp(-\Delta/S_\Delta).
$$

因此不得把 $T_{exp}$ 改成正指数表。

`static_tables_id` 必须绑定表生成器版本、输入输出尺度、认证区间、LeakyReLU 负半轴斜率、ELU/ExpMap 近似参数、舍入模式、clipping/saturation 规则、canonical row order 与完整表摘要。

#### 0.4.2 Backend 标量域中的有符号整数解释

当前 backend 标量域记为 $\mathbb F_{p_B}$。域元素标准代表 $z\in[0,p_B-1]$ 的规范带符号解释为：

$$
Dec_{p_B}(z)=
\begin{cases}
z,&0\le z\le(p_B-1)/2,\\
z-p_B,&(p_B-1)/2<z<p_B.
\end{cases}
$$

规范整数 $v$ 的域编码为：

$$
Enc_{p_B}(v)=v\bmod p_B.
$$

凡需要整数含义的变量，都必须通过 `integer_bound_ledger` 保证其绝对值及相关中间乘积严格小于 $(p_B-1)/2$。

#### 0.4.3 完整尺度链

本文直接继承主实验 `quant_cfg_id` 的尺度语义，不允许由 comparison backend 自行重新标定。公开尺度至少包括：

$$
S_{\Delta},\quad S_{exp},\quad
S_H^{(\ell)},\quad S_W^{(\ell,r)},\quad S_{H'}^{(\ell,r)},
$$

$$
S_{a,src}^{(\ell,r)},\quad S_{a,dst}^{(\ell,r)},\quad
S_{agg,pre},\quad S_{agg},\quad S_{cat},
$$

$$
S_W^{(out,s)},\quad S_{Y'}^{(s)},\quad
S_{a,src}^{(out,s)},\quad S_{a,dst}^{(out,s)},
$$

$$
S_b^{(out,s)},\quad S_{out,pre},\quad S_{out,head},\quad S_{out}.
$$

除：

$$
S_{inv}=\texttt{field\_exact},
\qquad
S_{\alpha}=\texttt{field\_exact},
$$

以外，全部数值尺度均为正整数，并且第一层输入尺度固定为：

$$
S_H^{(1)}=S_{input}.
$$

对每个隐藏头 $(\ell,r)$：

$$
S_{H'}^{(\ell,r)}
=
S_H^{(\ell)}S_W^{(\ell,r)}.
$$

源、目标 attention score 必须落到同一差分尺度：

$$
S_{H'}^{(\ell,r)}S_{a,src}^{(\ell,r)}
=
S_{H'}^{(\ell,r)}S_{a,dst}^{(\ell,r)}
=
S_{\Delta}.
$$

静态函数表的尺度绑定固定为：

$$
Scale_{in}(T_{LReLU})
=
Scale_{out}(T_{LReLU})
=
S_{\Delta},
$$

$$
Scale_{in}(T_{exp})=S_{\Delta},
\qquad
Scale_{out}(T_{exp})=S_{exp},
$$

$$
Scale_{in}(T_{ELU})=S_{agg,pre},
\qquad
Scale_{out}(T_{ELU})=S_{agg}.
$$

隐藏层定点恢复严格使用：

$$
H_{ELU,in}
=
SignedRoundDiv
\left(
S_{agg,pre}H_{agg,num},
S_{H'}Sum
\right).
$$

这里 $U=ExpMap(\Delta^+)$ 与 $Sum=\sum U$ 都处在同一 $S_{exp}$ 输出尺度，因此 $S_{exp}$ 在整数聚合分子 / 分母的比值中抵消；不能再额外插入一个未登记的 $S_{exp}$ rescale。

隐藏头经 $T_{ELU}$ 后输出尺度为 $S_{agg}$，拼接不改变单坐标尺度：

$$
S_{cat}=S_{agg},
\qquad
S_H^{(\ell+1)}=S_{cat}
\quad
(1\le\ell<L).
$$

对输出头 $s$：

$$
S_{Y'}^{(s)}
=
S_H^{(L)}S_W^{(out,s)},
$$

$$
S_{Y'}^{(s)}S_{a,src}^{(out,s)}
=
S_{Y'}^{(s)}S_{a,dst}^{(out,s)}
=
S_{\Delta}.
$$

输出聚合首先恢复到 $S_{out,pre}$：

$$
Y^{q}
=
SignedRoundDiv
\left(
S_{out,pre}Y_{agg,num},
S_{Y'}Sum
\right).
$$

随后显式对齐到输出头共同尺度：

$$
\widetilde Y^{pre}
=
SignedRoundDiv
\left(
S_{out,head}Y^{q},
S_{out,pre}
\right).
$$

输出偏置必须具有同一尺度：

$$
S_b^{(out,s)}=S_{out,head}.
$$

最终所有输出头在共同尺度 $S_{out,head}$ 上做整数平均，并固定：

$$
S_{out}=S_{out,head}.
$$

若：

$$
S_{out,pre}=S_{out,head},
$$

第二个 `SignedRoundDiv` 退化为恒等关系，但仍由同一 `quant_cfg_id` 决定，不能由 backend 私自删除后改变 proof statement。

不得在 comparison backend 中额外插入主实验没有的 projection rescale、attention rescale、hidden-head rescale、函数表重标定或 final-average 后 rescale。

#### 0.4.4 `SignedRoundDiv`

对带符号整数 $x$ 和正整数 $d$，定义：

$$
y=SignedRoundDiv(x,d).
$$

辅助见证：

$$
(\sigma,z,a,a^{-1},d^{-1},q,r,\delta),
$$

其中：

$$
\sigma,z\in\{0,1\},\qquad a,q,r,\delta\in\mathbb Z_{\ge0}.
$$

在 backend 标量域中必须显式满足：

$$
x=(1-2\sigma)a,\qquad y=(1-2\sigma)q,
$$

$$
\sigma(\sigma-1)=0,\qquad z(z-1)=0,\qquad za=0,
$$

$$
aa^{-1}=1-z,
$$

$$
z\sigma=0,\qquad za^{-1}=0,
$$

$$
dd^{-1}=1.
$$

若 `rounding_mode=toward_zero`：

$$
a=dq+r,
$$

$$
r+\delta=d-1.
$$

若 `rounding_mode=nearest_ties_away`：

$$
2a+d=2dq+r,
$$

$$
r+\delta=2d-1.
$$

$a,q,r,\delta$ 必须证明非负；动态 denominator 必须证明为正且非零。

#### 0.4.5 `integer_bound_ledger`

`integer_bound_ledger` 是当前 instance 的公开整数上界证书。它必须逐 hidden/output attention family 登记表输入区间、表输出区间、聚合界、`SignedRoundDiv` 界与 no-wrap 条件，而不能只给出一个全局“不会溢出”的声明。

公开最大入度：

$$
\Delta_{in}^{max}
=
\max_i\#\{k\mid dst(k)=i\}.
$$

对 hidden head $(\ell,r)$，设当前输入、权重和 attention vector 的公开绝对值上界分别为：

$$
B_H^{(\ell)},\quad
B_W^{(\ell,r)},\quad
B_{a,src}^{(\ell,r)},\quad
B_{a,dst}^{(\ell,r)}.
$$

则：

$$
B_{H'}^{(\ell,r)}
=
d_{in}^{(\ell)}
B_H^{(\ell)}
B_W^{(\ell,r)}.
$$

$$
B_{E,src}^{(\ell,r)}
=
d_h^{(\ell)}
B_{H'}^{(\ell,r)}
B_{a,src}^{(\ell,r)},
$$

$$
B_{E,dst}^{(\ell,r)}
=
d_h^{(\ell)}
B_{H'}^{(\ell,r)}
B_{a,dst}^{(\ell,r)},
$$

$$
B_S^{(\ell,r)}
=
B_{E,src}^{(\ell,r)}
+
B_{E,dst}^{(\ell,r)}.
$$

对 output head $s$：

$$
B_{Y'}^{(s)}
=
d_{in}^{(L)}
B_H^{(L)}
B_W^{(out,s)},
$$

$$
B_{E,src}^{(out,s)}
=
C
B_{Y'}^{(s)}
B_{a,src}^{(out,s)},
$$

$$
B_{E,dst}^{(out,s)}
=
C
B_{Y'}^{(s)}
B_{a,dst}^{(out,s)},
$$

$$
B_S^{(out,s)}
=
B_{E,src}^{(out,s)}
+
B_{E,dst}^{(out,s)}.
$$

下面统一用 $g$ 表示任意一个 hidden head $(\ell,r)$ 或 output head $(out,s)$。设 `static_tables_id` 认证的 $T_{LReLU}$ 输入整数区间为：

$$
[L_{S,min},L_{S,max}],
$$

$T_{exp}$ 的认证输入整数区间为：

$$
[0,L_{\Delta,max}].
$$

每个 attention family 必须首先通过 **function-table domain gate**：

$$
[-B_S^{(g)},B_S^{(g)}]
\subseteq
[L_{S,min},L_{S,max}].
$$

在该 family 实际可能使用的 score 区间上定义：

$$
B_Z^{(g)}
=
\max_{\substack{(x,z)\in T_{LReLU}\\
|x|\le B_S^{(g)}}}
|z|.
$$

组最大值只从同一 family 的 $Z$ 中选择，因此：

$$
B_M^{(g)}=B_Z^{(g)}.
$$

由：

$$
\Delta^+=M-Z
$$

得到保守上界：

$$
B_{\Delta,raw}^{(g)}
=
B_M^{(g)}+B_Z^{(g)}
\le
2B_Z^{(g)}.
$$

每个 family 还必须通过 $T_{exp}$ 输入域检查：

$$
0\le
B_{\Delta,raw}^{(g)}
\le
L_{\Delta,max}.
$$

同时 $M-Z$ 的整数减法两侧和结果都必须满足当前 backend field 的 no-wrap 条件。允许 ledger 根据已认证的非对称 score/output 区间给出更紧的 $B_{\Delta}^{(g)}$，但必须满足：

$$
B_{\Delta}^{(g)}
\le
B_{\Delta,raw}^{(g)}
\le
L_{\Delta,max}.
$$

该 family 的指数输出上界定义为：

$$
B_U^{(g)}
=
\max_{0\le\delta\le B_{\Delta}^{(g)}}
ExpMap(\delta).
$$

因此 hidden head $(\ell,r)$ 满足：

$$
B_{Sum}^{(\ell,r)}
=
\Delta_{in}^{max}
B_U^{(\ell,r)},
$$

$$
B_{H,num}^{(\ell,r)}
=
\Delta_{in}^{max}
B_U^{(\ell,r)}
B_{H'}^{(\ell,r)}.
$$

hidden aggregation recovery 的 signed dividend 与 positive denominator 分别满足：

$$
B_{x,H}^{(\ell,r)}
=
S_{agg,pre}
B_{H,num}^{(\ell,r)},
$$

$$
B_{d,H}^{(\ell,r)}
=
S_{H'}^{(\ell,r)}
B_{Sum}^{(\ell,r)}.
$$

$H_{ELU,in}$ 的 quotient bound 必须落在 $T_{ELU}$ 的认证输入整数区间内；$H_{agg}$ 的上界取 $T_{ELU}$ 在该认证输入区间上的最大绝对输出。隐藏 concat 不改变单坐标上界，因此下一层 $B_H^{(\ell+1)}$ 取本层所有 hidden heads 的公开保守最大值。

对 output head $s$：

$$
B_{Sum}^{(out,s)}
=
\Delta_{in}^{max}
B_U^{(out,s)},
$$

$$
B_{Y,num}^{(s)}
=
\Delta_{in}^{max}
B_U^{(out,s)}
B_{Y'}^{(s)}.
$$

output aggregation recovery：

$$
B_{x,Y}^{(s)}
=
S_{out,pre}
B_{Y,num}^{(s)},
$$

$$
B_{d,Y}^{(s)}
=
S_{Y'}^{(s)}
B_{Sum}^{(out,s)}.
$$

bias 前 scale alignment 的 dividend 上界为：

$$
B_{x,align}^{(s)}
=
S_{out,head}
B_{Y^q}^{(s)},
$$

其 denominator 为公开正整数 $S_{out,pre}$。若输出偏置绝对值上界为 $B_b^{(s)}$，则：

$$
B_{Y,head}^{(s)}
\le
B_{\widetilde Y^{pre}}^{(s)}
+
B_b^{(s)}.
$$

定义：

$$
B_{Y,head}^{max}
=
\max_s
B_{Y,head}^{(s)}.
$$

最终头和：

$$
B_{Y,sum}
=
K_{out}
B_{Y,head}^{max}.
$$

最终平均的 `SignedRoundDiv` dividend 上界即 $B_{Y,sum}$，denominator 为公开正整数 $K_{out}$。

对每个 `SignedRoundDiv` instance $\nu$，ledger 还必须分别给出：

$$
B_{a,\nu},\quad
B_{q,\nu},\quad
B_{r,\nu},\quad
B_{\delta,\nu},\quad
B_{d,\nu},
$$

并验证当前 rounding equation 的**两侧和中间乘积**都不会发生模回绕。特别地，`nearest_ties_away` 必须覆盖：

$$
2a+d,
\qquad
2dq+r,
\qquad
2d-1,
$$

`toward_zero` 必须覆盖：

$$
a,
\qquad
dq+r,
\qquad
d-1.
$$

如果 range proof 使用每 limb $B_{limb}$ 位分解：

$$
V
=
\sum_{h=0}^{\ell_V-1}
2^{B_{limb}h}V^{[h]},
$$

则：

$$
0\le V^{[h]}<2^{B_{limb}},
$$

并固定：

$$
\ell_V
=
\left\lceil
\frac{\operatorname{bitlen}(B_V+1)}
{B_{limb}}
\right\rceil.
$$

最终 no-wrap 条件是：所有需要按规范整数解释的值、function-table 输入、整数 equality 两侧以及 ledger 登记的中间乘积都必须严格满足：

$$
|v|
<
\frac{p_B-1}{2}.
$$

任一 hidden/output family 无法同时通过 table-domain gate、range gate 和 field-fit gate 时，当前 dataset/model/quant/backend-field 组合必须拒绝；不得通过修改 scale、静态表、rounding 或 GAT 前向来迁就 backend。

### 0.5 模型参数

隐藏头参数：

$$
W^{(\ell,r)}\in\mathbb Z^{d_{in}^{(\ell)}\times d_h^{(\ell)}},
$$

$$
a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}\in\mathbb Z^{d_h^{(\ell)}}.
$$

输出头参数：

$$
W^{(out,s)}\in\mathbb Z^{d_{in}^{(L)}\times C},
$$

$$
a_{src}^{(out,s)},a_{dst}^{(out,s)},b^{(out,s)}\in\mathbb Z^C.
$$

模型参数值属于私有见证。公开模型绑定对象由当前 backend 单独定义。

为保证不同 backend 证明同一模型关系，本文把**规范模型参数闭包**固定为：

$$
\mathcal W_{GAT}
=
\Big(
\{W^{(\ell,r)},a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}\}_{\ell,r},
\{W^{(out,s)},a_{src}^{(out,s)},a_{dst}^{(out,s)},b^{(out,s)}\}_{s}
\Big).
$$

后文若某个原论文或 backend 用单个符号 $W$ 表示“model weights / model witness”，一律指上述 $\mathcal W_{GAT}$ 的 canonical concatenation，而不只指 projection matrix。

除输出头 $b^{(out,s)}$ 外，本文的规范模型关系不包含任何额外 hidden/output bias。任何未出现在 $\mathcal W_{GAT}$ 中的参数都不属于模型见证，也不得进入 GAT 前向或证明关系；若未来规范模型增加新的参数族，必须同步升级 `model_arch_id` 与协议 schema。

### 0.5.1 私有模型参数上界与模型注册边界

第 0.4 节 `integer_bound_ledger` 中的

$$
B_W,\quad B_{a,src},\quad B_{a,dst},\quad B_b
$$

必须与本次证明实际使用的同一规范模型 $\mathcal W_{GAT}$ 绑定，不能由 prover 在单次 inference proof 中自由声明。

为与主实验保持同一模型注册边界，本对比协议固定采用**注册时认证模型上界、推理时复用认证结果**的方式：

1. 注册阶段读取 canonical $\mathcal W_{GAT}$，检查各参数族的规范有符号整数上界；
2. 把上界摘要与同一电路内模型绑定值 $H(W)$、模型结构、`quant_cfg_id`、backend 标量域和 ezkl/Halo2 circuit identity 绑定；
3. inference proof 只允许使用已经注册的模型身份，不在每次推理中重复证明整份模型参数范围；
4. 若 proof 中实际使用的私有权重与注册的 $H(W)$ 不一致，则电路内模型哈希关系必须使验证失败。

这与主实验“固定模型在参数生成/验证键边界完成范围认证”的口径一致，也避免给 comparison 额外增加主实验不存在的 per-inference 模型范围证明。

公开特征 $T_H$ 的范围由 verifier 直接依据公开 canonical integers 检查。任何未与当前模型绑定对象共同认证的 $B_W,B_a,B_b$ 都不能用于 no-wrap 论证。

### 0.5.2 公共 statement 的密码学绑定

$M_{pub}$ 只包含会改变 verifier 接受语义的公共信息。至少应唯一确定：

- GAT 结构、层数、各层头数与维度；
- 本次公开 graph/topology 与公开 features 的 canonical identity；
- `quant_cfg_id` 与 `static_tables_id`；
- 当前模型的公开绑定值 $H(W)$；
- ezkl/Halo2 compiled circuit、settings 与 verification key 的身份；
- `ezkl_security_profile_id` 及其绑定的 native curve/field、PCS、transcript 与 security configuration；
- 完整公开 logits $Y$ 的 canonical encoding。

凡使用摘要绑定上述对象，必须采用 collision-resistant hash、明确的 domain separation 和唯一 canonical encoding。若对象已经由 Halo2 verification key、compiled-circuit identity 或电路内 commitment/hash 原生绑定，则直接引用该原生对象即可，不再人为叠加第二套证明关系。


### 0.6 全部中间变量

本节列出的对象是 **GAT 逻辑 trace**。在 ezkl/Halo2 关系中，它们映射为编译图 tensor、advice/fixed/public-instance cells 及对应 argument；本节不会为这些对象引入主实验式 KZG polynomial/commitment 命名。相同的逻辑公式在三个 baseline 中保持一致，是为了证明同一个 GAT 函数，而不是要求三个 backend 使用相同的证明数据结构。

#### 0.6.1 第一层输入

$$
H^{(1)}=T_H[I].
$$

对 $\ell\ge2$：

$$
H^{(\ell)}=H_{cat}^{(\ell-1)}.
$$

#### 0.6.2 隐藏头节点/边变量

对固定 $(\ell,r)$：

$$
H_{i,j}'^{(\ell,r)}
=
\sum_mH_{i,m}^{(\ell)}W_{m,j}^{(\ell,r)}.
$$

$$
E_{src,i}^{(\ell,r)}
=
\sum_jH_{i,j}'^{(\ell,r)}a_{src,j}^{(\ell,r)},
$$

$$
E_{dst,i}^{(\ell,r)}
=
\sum_jH_{i,j}'^{(\ell,r)}a_{dst,j}^{(\ell,r)}.
$$

对边 $k$：

$$
S_k^{(\ell,r)}
=
E_{src,src(k)}^{(\ell,r)}+E_{dst,dst(k)}^{(\ell,r)}.
$$

$$
(S_k^{(\ell,r)},Z_k^{(\ell,r)})\in T_{LReLU}.
$$

组最大值：

$$
M_i^{(\ell,r)}=\max_{dst(k)=i}Z_k^{(\ell,r)}.
$$

非负差分：

$$
\Delta_k^{+(\ell,r)}=M_{dst(k)}^{(\ell,r)}-Z_k^{(\ell,r)}\ge0.
$$

每个组至少存在一个有效边满足 $\Delta_k^+=0$。

指数：

$$
(\Delta_k^{+(\ell,r)},U_k^{(\ell,r)})\in T_{exp}.
$$

分母：

$$
Sum_i^{(\ell,r)}=\sum_{dst(k)=i}U_k^{(\ell,r)}>0.
$$

域内 inverse：

$$
Sum_i^{(\ell,r)}inv_i^{(\ell,r)}=1.
$$

$$
\alpha_k^{(\ell,r)}=U_k^{(\ell,r)}inv_{dst(k)}^{(\ell,r)}.
$$

有限域聚合：

$$
H_{agg,pre,i,j}^{(\ell,r)}
=
\sum_{dst(k)=i}\alpha_k^{(\ell,r)}H_{src(k),j}'^{(\ell,r)}.
$$

整数聚合分子：

$$
H_{agg,num,i,j}^{(\ell,r)}
=
\sum_{dst(k)=i}U_k^{(\ell,r)}H_{src(k),j}'^{(\ell,r)}.
$$

二者必须满足域内一致性：

$$
H_{agg,num,i,j}^{(\ell,r)}
=
Sum_i^{(\ell,r)}H_{agg,pre,i,j}^{(\ell,r)}.
$$

真正进入 ELU 的 fixed-point integer：

$$
H_{ELU,in,i,j}^{(\ell,r)}
=
SignedRoundDiv\!\left(
S_{agg,pre}H_{agg,num,i,j}^{(\ell,r)},
S_{H'}^{(\ell,r)}Sum_i^{(\ell,r)}
\right).
$$

ELU：

$$
(H_{ELU,in,i,j}^{(\ell,r)},H_{agg,i,j}^{(\ell,r)})\in T_{ELU}.
$$

`inv` 与 $\alpha$ 不得直接替代上述整数恢复。

#### 0.6.3 隐藏拼接

$$
H_{cat,i,r d_h^{(\ell)}+j}^{(\ell)}
=
H_{agg,i,j}^{(\ell,r)}.
$$

#### 0.6.4 输出头变量

对输出头 $s$：

$$
Y_{i,c}'^{(s)}
=
\sum_mH_{i,m}^{(L)}W_{m,c}^{(out,s)}.
$$

$$
E_{src,i}^{(out,s)}=\sum_cY_{i,c}'^{(s)}a_{src,c}^{(out,s)},
$$

$$
E_{dst,i}^{(out,s)}=\sum_cY_{i,c}'^{(s)}a_{dst,c}^{(out,s)}.
$$

$$
S_k^{(out,s)}
=
E_{src,src(k)}^{(out,s)}+E_{dst,dst(k)}^{(out,s)}.
$$

$$
(S_k^{(out,s)},Z_k^{(out,s)})\in T_{LReLU}.
$$

$$
M_i^{(out,s)}=\max_{dst(k)=i}Z_k^{(out,s)}.
$$

$$
\Delta_k^{+(out,s)}=M_{dst(k)}^{(out,s)}-Z_k^{(out,s)}.
$$

$$
(\Delta_k^{+(out,s)},U_k^{(out,s)})\in T_{exp}.
$$

$$
Sum_i^{(out,s)}=\sum_{dst(k)=i}U_k^{(out,s)}.
$$

$$
Sum_i^{(out,s)}inv_i^{(out,s)}=1,
$$

$$
\alpha_k^{(out,s)}=U_k^{(out,s)}inv_{dst(k)}^{(out,s)}.
$$

有限域聚合：

$$
Y_{i,c}^{pre(s)}
=
\sum_{dst(k)=i}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}.
$$

整数分子：

$$
Y_{agg,num,i,c}^{(s)}
=
\sum_{dst(k)=i}U_k^{(out,s)}Y_{src(k),c}'^{(s)}.
$$

域内一致性：

$$
Y_{agg,num,i,c}^{(s)}
=
Sum_i^{(out,s)}Y_{i,c}^{pre(s)}.
$$

输出聚合的整数恢复：

$$
Y_{i,c}^{q(s)}
=
SignedRoundDiv\!\left(
S_{out,pre}Y_{agg,num,i,c}^{(s)},
S_{Y'}^{(s)}Sum_i^{(out,s)}
\right).
$$

bias 前尺度对齐：

$$
\widetilde Y_{i,c}^{pre(s)}
=
SignedRoundDiv\!\left(
S_{out,head}Y_{i,c}^{q(s)},
S_{out,pre}
\right).
$$

每头输出：

$$
Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}.
$$

最终头和：

$$
Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}.
$$

最终 logits：

$$
Y_{i,c}=SignedRoundDiv(Y_{i,c}^{sum},K_{out}).
$$

不得使用 $K_{out}^{-1}Y_{i,c}^{sum}$ 作为最终整数平均。

### 0.7 ONNX / NativeGAT 计算图、公开实例与 witness 布局

#### 0.7.1 计算图

量化 GAT 图记为：

$$
G_{ezkl}=ExportGAT_{ezkl}(M_{pub},Topo,quant\_cfg,static\_tables\_id).
$$

计算图必须显式或以等价 custom op 表达第 0.6 节全部语义，包括：

- hidden/output projection；
- public sparse gather / route；
- LeakyReLU；
- per-$dst$ stable max；
- $\Delta^+$ 与 ExpMap；
- $Sum,inv,\alpha$；
- field aggregate 与 integer numerator 两条聚合；
- hidden/output `SignedRoundDiv`；
- ELU；
- hidden concat；
- output scale alignment、bias 与最终整数平均。

不得以宿主机预先计算 $Y$、然后只在电路中验证哈希或少量摘要替代上述完整关系。

#### 0.7.2 稀疏 topology 表示

本文使用固定 shape 的公开 CSR / grouped-edge 布局。至少由公开 $src,dst,node\_ptr,edge\_ptr$ 唯一得到：

- `row_ptr` / group offsets；
- `src_index`；
- group target index；
- `Q_edge_valid`；
- `Q_grp_new,Q_grp_end`；
- 每个节点组公开容量与 padding selector。

本协议把公开 topology 编译为 circuit wiring，并要求其 canonical identity 进入 `compiled_circuit_id`。图的值仍是公开 statement 的组成部分；验证者通过 circuit identity 检查该 wiring 与公开 $src,dst,node\_ptr,edge\_ptr$ 完全一致。

#### 0.7.3 Public instance

验证者从公开 $T_H,I$ 本地重建：

$$
H^{(1)}=T_H[I].
$$

public instance 必须直接包含 $H^{(1)}$ 或包含能使 verifier 逐元素重建并比较的等价编码，同时绑定：

- topology identity；
- 完整 $Y$；
- `quant_cfg_id`；
- `static_tables_id`；
- $H(W)$；
- `compiled_circuit_id/settings_id/vk_id`。

禁止把 $H^{(1)}$ 转成 private advice 后只留下一个 proof 外 hash。

#### 0.7.4 Private witness

严格模式 private witness 包括：

- $W,a_{src},a_{dst},b$；
- 第 0.6 节全部 private trace；
- max argument 辅助量；
- `SignedRoundDiv` auxiliary；
- logUP / range 内部 witness。

#### 0.7.5 exact GAT function table 与 max 辅助 ReLU

GAT 数值语义使用的

$$
T_{LReLU},\quad T_{ELU},\quad T_{exp}
$$

integer rows 由 `static_tables_id` 固定。ezkl calibration 不得重新决定这些函数的 slope、近似、scale、rounding 或 clipping。

ezkl 原论文 composite max argument 还需要普通 ReLU：

$$
ReLU(x)=\max(0,x).
$$

它不是 GAT 前向新增的一层激活，因此不并入主实验 `static_tables_id` 的 GAT 函数语义；本文把它登记为 backend-only 辅助表：

$$
T_{ReLU}^{max}
=
\{(x,\max(0,x))\mid x\in\mathcal D_{max,ReLU}\}.
$$

$\mathcal D_{max,ReLU}$ 必须由 `integer_bound_ledger` 唯一给出，并覆盖 composite max 中实际出现的：

$$
w=x-(m-1)
$$

以及：

$$
1-\sum_i y_i.
$$

`T_{ReLU}^{max}` 的 canonical rows、输入区间、encoding 和摘要进入：

`backend_aux_tables_id`

并由 `settings_id` 与 `compiled_circuit_id` 绑定。它只服务 ezkl max argument，不改变 $T_{LReLU}$、$T_{ELU}$、$T_{exp}$ 的主实验语义。

### 0.8 Padding、组宽与 backend capability

对每个公开目标节点组，允许把入边 padding 到公开容量 $D_i$ 或全局 $\Delta_{in}^{max}$。padding edge 必须满足 selector 为 0，并从 max、Sum、aggregation、lookup 和 `SignedRoundDiv` denominator 中排除。

如果所选 ezkl/Halo2 协议参数无法用 native arithmetic、logUP 或等价 Halo2 argument **精确**表达下列任一关系：

- sparse public routing；
- exact group max；
- exact $T_{exp}$；
- dynamic-denominator `SignedRoundDiv`；
- final integer head average；

则该配置标记 `unsupported_exact_relation`，不能改成浮点、域逆元平均或证明关系之外的未约束计算。

### 0.9 模型绑定与公共 statement

原 ezkl 论文的 inference proof把模型权重作为 private input，并在 circuit 中计算公开模型权重哈希；Appendix C.1 明确说明该关系使用 ZKG hashing。因此本对比协议保持这一机制：

$$
H(W)=ZKG\big(CanonicalModelParams(\mathcal W_{GAT})\big),
$$

且该等式必须由**同一个 ezkl/Halo2 proof circuit**约束。proof 外单独计算一个模型摘要不能替代电路内模型绑定。

本对比协议的逻辑公共 statement 为：

$$
Public_{ezkl}
=
(M_{pub},T_H,I,src,dst,node\_ptr,edge\_ptr,Y,H(W),CircuitRef),
$$

其中 $CircuitRef$ 唯一确定本次 compiled circuit、settings 与 verification key。逻辑私有见证仍为：

$$
Witness_{ezkl}=(\mathcal W_{GAT},Trace_{GAT},Aux_{max},Aux_{div},Aux_{lookup}).
$$

公开 topology 由固定 circuit wiring 承载，其 canonical identity 由 compiled circuit 绑定；公开 features 与完整 $Y$ 仍作为本次 proof 的公开输入/输出。

如果某个 ezkl 版本只能把模型权重固化为 compile-time constants，而不能把 $\mathcal W_{GAT}$ 作为 private witness 并证明上述 $H(W)$ 关系，则它证明的是另一种隐私边界，不属于本文与主实验对齐的严格 comparison relation。

## 1. 参数生成

### 1.1 输入

参数生成输入为安全参数 $\lambda$、由 `ezkl_security_profile_id` 唯一确定的 ezkl/Halo2 native backend 参数、GAT architecture、公开 topology shape、`quant_cfg_id`、`static_tables_id`、模型参数 shape/visibility、`integer_bound_ledger` 与 canonical encoding。

### 1.2 ONNX / NativeGAT 图生成

将第 0.6 节 relation 编译为 ONNX 或等价的 NativeGAT circuit representation。标准 ONNX 缺少的 segment/group primitive 可以分解为固定 shape 子图；若引入 custom relation，其语义必须由 Halo2 arithmetic/logUP constraints 完整约束。

图生成后计算：

$$
model\_graph\_id=Hash(CanonicalIR(G_{ezkl})).
$$

### 1.3 Argument registry

编译前登记每类关系使用的 ezkl argument：

| GAT relation | ezkl/Halo2 argument |
|---|---|
| projection / attention dot | cumulative dot-product |
| group sum | cumulative sum |
| $UH'$ / $\alpha H'$ / bias / consistency | elementwise add/mul/sub |
| LeakyReLU / ELU / ExpMap / range | logUP lookup / native range |
| max auxiliary ReLU | `T_ReLU^max` + ezkl logUP |
| max | 第 3.5 节 composite max argument |
| `SignedRoundDiv` | 第 3.7 节 composite arithmetic+range gadget |

该 registry 进入 `compiled_circuit_id`，防止同名 circuit 使用不同 relation。

### 1.4 Settings 与量化冻结

ezkl/Halo2 的编译 settings 只能从已经冻结的 `quant_cfg_id` 导出。若 backend 需要 calibration，则其结果必须满足：

$$
QuantCfg_{ezkl}=QuantCfg_{main}
$$

在所有 scale、rounding、table row、clipping endpoint 和最终 $Y$ 上逐项相等。否则 setup 失败。

### 1.5 固定 shape 与 topology binding

根据 $N,E,\{d_{in}^{(\ell)},d_h^{(\ell)},d_{cat}^{(\ell)}\},C,K_{out}$ 生成固定 circuit shape。公开 topology wiring 或其 runtime-public input layout 写入 circuit manifest。

### 1.6 模型哈希 relation

严格 private-weight 模式下，电路内必须约束：

$$
h_W=H(W),
$$

并把 $h_W$ 放入 public instance。任何仅在 proof 外独立计算、但未由同一电路关系约束的哈希都不构成模型绑定。

### 1.6.1 模型参数范围绑定

模型注册阶段对 canonical $\mathcal W_{GAT}$ 检查第 0.4 节所需的全部参数上界，并把这些 bounds 与同一电路内 $H(W)$、模型结构、量化配置和 circuit identity 共同认证。inference proof 不重复证明整份模型参数范围。

因此，验证者使用 `integer_bound_ledger` 时依赖的是已经与当前 $H(W)$ 绑定的注册上界，而不是 prover 在本次 proof 中临时提供的自由元数据。

### 1.7 ezkl / Halo2 setup

执行 backend 原生 setup/compile，得到：

$$
(pk_{ezkl},vk_{ezkl},C_{ezkl}).
$$

同时固定 compiled-circuit identity、settings identity、verification-key identity 与 model-graph identity。若当前 Halo2 backend 需要公共参数，则使用其原生参数，不复用主实验 SRS。

### 1.8 参数生成输出

最终 setup context：

$$
SetupCtx_{ezkl}=(C_{ezkl},pk_{ezkl},vk_{ezkl},settings,Manifest_{ezkl}).
$$

## 2. 见证生成

### 2.1 公开输入

验证者可重建的第一层输入固定为：

$$
H^{(1)}=T_H[I].
$$

prover 使用同一 canonical $H^{(1)}$ 生成 public-instance encoding；不另行量化一份不同输入。

### 2.2 Hidden-head witness

对每个 $(\ell,r)$，按第 0.6 节顺序生成：

$$
H',E_{src},E_{dst},S,Z,M,\Delta^+,U,Sum,inv,\alpha,
$$

$$
H_{agg,pre},H_{agg,num},H_{ELU,in},H_{agg}.
$$

`inv` 和 $\alpha$ 只作为 field witness；`H_{ELU,in}` 必须由 integer numerator 的 `SignedRoundDiv` 得到。

### 2.3 Max composite witness

对每个公开 $dst$ group，设组内 LeakyReLU 输出向量为 $x$，claimed max 为 $m$。生成：

$$
w=x-(m-1),
$$

$$
y=ReLU(w),
$$

以及证明“至少一个 $y_i=1$”所需的 group-sum / ReLU auxiliary。该 witness 只为 max argument 服务，不改变 $M=\max x$ 的语义。

### 2.4 logUP query witness

生成 exact query：

$$
(S,Z),\qquad(\Delta^+,U),\qquad(H_{ELU,in},H_{agg}),
$$

以及 `SignedRoundDiv` range/limb queries。所有 query 必须来自同一 canonical trace。

### 2.5 `SignedRoundDiv` witness

对 hidden aggregation、output aggregation、output scale alignment、final head average 的每个坐标生成：

$$
(\sigma,z,a,a^{-1},d^{-1},q,r,\delta).
$$

其中 dynamic denominator 例如 $S_{H'}Sum$、$S_{Y'}Sum$ 必须从当前 trace 得到，不能由 prover 提供未绑定的自由 denominator。

### 2.6 Hidden concat 与 output witness

逐层生成 $H_{cat}^{(\ell)}$，最终对每个输出头生成：

$$
Y',E_{src}^{out},E_{dst}^{out},S^{out},Z^{out},M^{out},\Delta^{+,out},U^{out},Sum^{out},inv^{out},\alpha^{out},
$$

$$
Y^{pre},Y_{agg,num},Y^q,\widetilde Y^{pre},Y^{(s)}.
$$

最后生成：

$$
Y^{sum},\qquad Y=SignedRoundDiv(Y^{sum},K_{out}).
$$

### 2.7 Witness 与 public-instance 输出

形成 backend 所需的 witness assignments 与 public instance。完整 private trace 不属于 verifier-facing proof。

## 3. 证明生成

### 3.1 输入与输出

输入：

$$
(SetupCtx_{ezkl},Public_{ezkl},Witness_{ezkl}).
$$

输出 Halo2/ezkl native proof：

$$
\pi_{ezkl}.
$$

### 3.2 Cumulative sum / product argument

按原论文，若向量 $x=(x_0,\ldots,x_{n-1})$，累计状态 $m$ 满足：

$$
m_0=0,
$$

累计和关系：

$$
m_{t+1}=m_t+x_t,
$$

累计积关系使用对应乘法递推。GAT 的 $Sum$、组内 $UH'$ 累计和等关系在公开 group boundary 上重置状态；padding row 由 selector 屏蔽。

### 3.3 Cumulative dot-product argument

对 $x,y$：

$$
m_0=0,
$$

$$
m_{t+1}=m_t+x_ty_t.
$$

最终 $m_n$ 绑定 dot product。hidden/output projection 和 attention vector dot 使用该 argument 或 ezkl 编译器对等价 Einsum 的原生实现。

### 3.4 Elementwise argument

对等长 $x,y$ 与输出 $v$：

$$
v_t=x_t\circ y_t,
$$

其中 $\circ$ 为当前 relation 指定的 add/mul/sub。用于 score add、$UH'$、$\alpha H'$、bias、field/integer consistency 与 `SignedRoundDiv` 局部算术。

### 3.4.1 Public sparse routing / gather

GAT 的 $src/dst$ gather 必须被 Halo2 电路约束，而不能只在证明关系之外计算。本协议把公开 $src,dst,Q_{grp,new},Q_{grp,end}$ 与 node-to-edge wiring 编译进 circuit，并由 `compiled_circuit_id` 绑定其 canonical topology identity。

因此对每条有效边，电路中的 routed view 必须严格引用公开 wiring 指定的节点：

$$
H_{src}^{\prime edge}[k,j]=H'[src(k),j],
$$

$$
E_{src}^{edge}[k]=E_{src}[src(k)],\qquad
E_{dst}^{edge}[k]=E_{dst}[dst(k)],
$$

输出层同理绑定 $Y_{src}^{\prime edge}[k,c]$。$M^{edge},Sum^{edge},inv^{edge}$ 与聚合结果的 edge/node views 也只能通过同一公开 $dst$ wiring 连接。后续 score、$\Delta^+$、$\alpha$、integer numerator 与 group states 只能引用这些已绑定 routed views。

### 3.5 ezkl 论文的 composite max argument

对 group $x$ 和 claimed $m$：

1. 约束 $w=x-(m-1)$；
2. logUP lookup 约束 $y=ReLU(w)$；
3. 约束：
   $$y_i(y_i-1)=0;$$
4. 因此不存在 $x_i>m$；
5. 再构造：
   $$z=ReLU\left(1-\sum_i y_i\right),$$
   并约束：
   $$z=0.$$

若 $\sum_i y_i=0$，则 $z=1$，验证失败；因此至少一个 $x_i=m$。两部分共同得到 $m=\max_i x_i$。

GAT 对每个 `(graph_id,dst)` group 独立实例化；padding 不进入 group。

### 3.6 logUP lookup

使用 ezkl 原生 logUP 对 exact public tables 证明：

$$
(S,Z)\in T_{LReLU},
$$

$$
(\Delta^+,U)\in T_{exp},
$$

$$
(H_{ELU,in},H_{agg})\in T_{ELU}.
$$

如果 range/limb 采用 lookup，其表也由 `static_tables_id` 所绑定的 range relation 固定。这里的 logUP 与主实验 LogUp 不共享证明关系或 transcript。

### 3.7 `SignedRoundDiv` Halo2 composite gadget

对每个实例建立第 0.4.4 节的全部 equality/boolean/range constraints。至少包括：

$$
x=(1-2\sigma)a,
\qquad y=(1-2\sigma)q,
$$

$$
aa^{-1}=1-z,
\qquad dd^{-1}=1,
$$

以及当前 rounding mode 的 quotient/remainder relation。

必须在四类位置全部实例化：hidden recovery、output recovery、output scale alignment、final $K_{out}$ average。不得用 field inverse 或工具链默认 division 代替。

### 3.8 双聚合 consistency

电路同时约束：

$$
H_{agg,pre,i,j}
=
\sum_{dst(k)=i}\alpha_k H_{src}^{\prime edge}[k,j],
$$

$$
H_{agg,num,i,j}
=
\sum_{dst(k)=i}U_k H_{src}^{\prime edge}[k,j],
$$

$$
H_{agg,num}=Sum\cdot H_{agg,pre},
$$

输出层使用已经约束的 $Y_{src}^{\prime edge}$ 完全同理：

$$
Y^{pre}_{i,c}
=
\sum_{dst(k)=i}\alpha_kY_{src}^{\prime edge}[k,c],
\qquad
Y_{agg,num,i,c}
=
\sum_{dst(k)=i}U_kY_{src}^{\prime edge}[k,c].
$$

该双路径是主实验 fixed-point semantics 与 ezkl field arithmetic 的连接点，不能只保留其中一条，也不能在这里重新使用未约束的外部 gather。

### 3.9 Public input、output 与 $H(W)$

完整公开 graph/features、完整 $Y$ 与公开 $H(W)$ 必须由同一个 ezkl/Halo2 verifier statement 绑定。验证者依据认证的 circuit/settings 把公开输入编码为 native public instance，并执行 native Halo2 verification。

其中：

- $H^{(1)}=T_H[I]$ 由公开数据确定；
- 若 topology 编译进 circuit，则 circuit identity 必须唯一绑定该 topology；
- $Y$ 必须作为完整公开 logits 参与 native public-output binding；
- $H(W)$ 必须来自电路内对实际私有 $\mathcal W_{GAT}$ 的 ZKG relation。

外层摘要可以用于对象索引，但不能替代 native public-instance relation。

### 3.10 Halo2 transcript 与 PCS

Halo2 内部 commitment、quotient、opening 与 Fiat--Shamir challenge 顺序完全由固定的 ezkl/Halo2 backend version 定义。本文不在其外再叠加主实验 KZG/LogUp transcript。

### 3.11 最终证明对象

逻辑上：

$$
\pi_{GAT}^{ezkl}
=
(M_{pub},PublicStatement,CircuitRef,\pi_{ezkl}).
$$

其中：

$$
CircuitRef=(compiled\_circuit\_id,settings\_id,vk\_id).
$$

`PublicStatement` 包含本次验证所需的公开 graph/features identity、完整 $Y$ 与公开 $H(W)$。公开 graph/features 的原始值由 verifier 作为 public input 获得，并按照 `CircuitRef/settings` 唯一重建 native public instance；它们的 canonical identity 必须与 $M_{pub}$ 中的 public-input binding 一致。

真正的计算正确性由 $\pi_{ezkl}$ 的 native verification 给出；外围字段只能描述已经由 native public instance 或认证 circuit identity 固定的 statement identity。

## 4. 验证

### 4.1 Verifier context

验证者持有：$vk_{ezkl}$、compiled-circuit identity、编译 settings、`ezkl_security_profile_id` 及其认证参数、公开 graph/features、`static_tables_id` 与 model-binding object。

### 4.2 公共 statement 与 circuit identity

检查 $M_{pub}$、公开 graph/features、完整 $Y$、$H(W)$ 与 $CircuitRef$ 的 canonical identity，并确认所用 verification key/settings 对应第 0 节定义的同一 GAT relation。

### 4.3 公开输入重建

在执行任何 public reconstruction 或 native Halo2 verification 之前，验证者必须检查第 0.1 节与第 0.3 节定义的全部模型维度和公共 instance 合法性条件；任一条件不满足必须立即拒绝。

验证者从公开 $T_H,I$ 重建 $H^{(1)}$，从 $src,dst,node\_ptr,edge\_ptr$ 重建 topology identity、group layout 与 padding selectors，并检查它们与 compiled circuit 绑定的 wiring 一致，再根据认证 settings/circuit layout 生成唯一 native public instance。

### 4.4 Quant/table/bound gate

验证者检查 `quant_cfg_id` 与 `static_tables_id`，重新计算公开输入、图入度和所有可公开推导的 field-fit 条件，并确认模型参数 bounds 来自与当前 $H(W)$ 共同认证的模型注册对象。任一 table-domain、range 或 no-wrap 条件失败都必须拒绝。

### 4.5 Circuit identity

检查 compiled-circuit identity、settings、verification key 与 model-graph identity 一致，避免用另一电路的有效 proof 替代当前 GAT relation。

### 4.6 Native ezkl/Halo2 verification

执行：

$$
Verify_{ezkl}(vk_{ezkl},PublicInstance,\pi_{ezkl})=1.
$$

native verifier 失败立即拒绝。

### 4.7 模型与输出绑定

检查 $H(W)$ 是本次 circuit public output/instance 的已约束字段，并直接核对完整公开 logits $Y$ 与 native public instance 的绑定。验证关系以完整 $Y$ 为准，不以额外摘要替代。

### 4.8 接受条件

只有 metadata、public reconstruction、bound gate、circuit identity、native proof、model hash 与完整 logits binding 全部通过才输出 $Accept$。

## 5. 完备性、可靠性与零知识边界

### 5.1 完备性

若 exact GAT witness 满足第 0 章 relation，并被编译 circuit 完整表达，则按 ezkl/Halo2 原生算法生成的 proof 被接受。

### 5.2 计算可靠性

可靠性依赖固定 ezkl/Halo2 backend 的 SNARK soundness、PCS binding/opening soundness、logUP soundness、compiled-circuit identity 与 public/model binding。custom GAT op 只有在其所有 Halo2 constraints 都进入 compiled circuit 时才属于该可靠性声明。

### 5.3 Zero knowledge

$W$ 与 intermediate witness 均为 private circuit witness；public graph/features、$H(W)$ 和完整 $Y$ 明文可见。Halo2/ezkl 的零知识 blinding 必须按 backend native protocol 启用。模型参数上界只在第 0.5.1 节已认证的固定模型注册边界中公开，不公开参数值。

### 5.4 $H(W)$ 的额外泄露

公开 deterministic $H(W)$ 会暴露模型相等性/可链接性，并在模型候选空间较小时形成额外 metadata leakage。它是 ezkl 原论文模型一致性机制的一部分，因此 comparison 应保留并单独声明，而不能称为与主实验完全相同的泄露界面。

### 5.5 Compiled-constant 路线

如果权重只能固化为 circuit constants，则该关系不再满足本文固定的 `model_and_trace_hidden` 隐私边界，因此不属于本对比协议。

### 5.6 Native security parameters

comparison 必须报告 `ezkl_security_profile_id` 实际绑定的 native curve/field、PCS、transcript suite 与对应安全级别。该安全参数只描述本次 pinned ezkl/Halo2 configuration，不要求与 zkDL、zkGPT 或主实验使用相同曲线，也不得把不同 backend 的安全级别人为写成同一个数值。若所用 ezkl 版本或实验工件不能唯一恢复这些参数，则该 run 不进入正式主比较。

### 5.7 Backend capability

任何 exact GAT relation 若无法由当前 ezkl/Halo2 工具链表达，允许的结论是 `unsupported_exact_relation`，而不是修改 GAT forward 以适配 backend。
