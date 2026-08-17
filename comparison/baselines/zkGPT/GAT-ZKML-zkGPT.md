# GAT-ZKML zkGPT 协议笔记

## 0. 符号

### 0.0 证明关系与对比实验范围

本文定义：

$$
R_{GAT\text{-}zkGPT}.
$$

逻辑公共输入为：

$$
Public_{zkGPT}=(M_{pub},X_{pub},com_W,StaticTables,Y),
$$

私有见证为：

$$
Witness_{zkGPT}
=
(W,Aux,L_{qry},Aux_{Lasso},R_{com}),
\qquad
Aux_{Lasso}=\{M_f\}_f,
\qquad
R_{com}=(\rho_W,\rho_{aux},\rho_L),
$$

其中 $Aux$ 包含完整 GAT 中间 trace、sparse accumulation advice、max auxiliary 和 `SignedRoundDiv` auxiliary；$L_{qry}$ 是按 `lookup_tuple_encoding_id` 规范编码的 Lasso function/range query；$Aux_{Lasso}$ 是第 2.6.1 节定义的 Lasso auxiliary matrices，只作为 Lasso proof witness，不属于规范 GAT 数值 trace。

本文沿用原 zkGPT 的核心结构：public input、private model weights、input-layer advice、Lasso、ML-friendly GKR、dedicated matrix-multiplication Sumcheck、input-layer claim combine、Hyrax commitment、constraint fusion 与 circuit squeeze。GAT 稀疏注意力和 exact `SignedRoundDiv` 是本文 adaptation；它们必须映射到上述 zkGPT 原生原语，不能移植主实验 PSQ/LogUp/KZG。

原论文把 input embedding 作为 public input；因此当前 GAT 的 $T_H/H$ 同样固定为 public。正式协议不存在 private `com_{T_H}`。


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

### 0.2 有限域、MLE、Hyrax、ML-friendly GKR、Lasso 与 MatMul Sumcheck

本文采用 zkGPT 论文的核心证明路线：

- Hyrax-style MLE commitment；
- ML-friendly GKR；
- Lasso lookup / range proof；
- dedicated matrix multiplication Sumcheck；
- advice-based non-arithmetic checking；
- constraint fusion；
- circuit squeeze。

原论文的核心 input-layer 结构是 public input 与 private witness 共存：输入 embedding 为 public，model weights、rounding/intermediate advice 与 lookup queries 为 witness。本文把公开 GAT graph/features 对应到该 public-input 角色。

原 zkGPT 采用 BN254，并报告约 100-bit security；本 comparison 保持这一真实安全参数。

#### 0.2.1 MLE / Hyrax

对 padding 后的输入向量 $V$：

$$
\widetilde V(u)=\sum_b\widetilde{eq}(u,b)V(b).
$$

承诺：

$$
com_V=zkPC.Commit(\widetilde V;pp,r_V).
$$

长期 model commitment 在 preprocessing 中生成并跨 proof 复用；input-dependent aux / lookup queries 每次 proof 重新承诺。

#### 0.2.2 ML-friendly GKR

普通 arithmetic relation 被组织为电路层，并通过 GKR 把输出层 MLE claim 逐层归约到输入层 MLE claim。允许 gate 同时连接上一电路层和 input-layer advice，从而使 rounding advice 等非线性结果作为输入层 witness 被多个 constraint subcircuit 使用。

#### 0.2.3 Lasso

Lasso 用于公开 table 上的 range 与 exponentiation/函数 lookup。lookup query 属于 private input-layer witness，必须与 arithmetic circuit 中真实变量保持一致。

#### 0.2.4 Dedicated MatMul Sumcheck

真正的大规模矩阵乘法使用独立矩阵乘法 Sumcheck，而不是展开成普通 GKR gates。原论文的 dedicated MatMul proof 可聚合多个矩阵乘法 claim；这种聚合只改变证明组织方式，不能改变 canonical integer matrix product。

#### 0.2.5 零知识边界

原 zkGPT 正文给出的 Protocol 1 主要描述 Hyrax commitment、Lasso、ML-friendly GKR、dedicated MatMul Sumcheck 与 Fiat--Shamir 非交互化。论文同时说明：可以采用与文献 [69] 类似的 masking polynomials 为整体协议加入 zero knowledge，但没有在 Protocol 1 中展开一套完整的 masking 轮次。

本文固定的隐私策略是 `model_and_trace_hidden`，因此严格 comparison 除了第 0--4 节的核心计算关系外，还要求采用论文所指向的 masking-polynomial 路线，把 witness-dependent GKR/MatMul/Lasso 消息以及 private terminal commitment evaluations 纳入同一零知识变换。masking 只允许改变 verifier view，不得改变公开 statement、GAT 数值关系、lookup queries 或最终 $Y$。

原论文没有在 Protocol 1 中展开完整 masking 轮次，因此本文不会把未掩蔽的 Protocol 1 自动称为零知识。若没有给出覆盖上述全部 witness-dependent view 的具体 masking construction，则该实例只能证明计算可靠性，不能满足本笔记第 0.1 节固定的隐私 relation，也不能作为严格的 zkGPT 对比协议。

严格隐私 profile 记为 `masked_full`，并由公开的 `privacy_extension_profile_id` 唯一确定。只有当该 profile 同时满足下列条件时，才允许在主比较中声明 `model_and_trace_hidden` zero knowledge：

1. masking 覆盖全部 witness-dependent GKR、dedicated MatMul Sumcheck、Lasso auxiliary commitments/messages、lookup-input consistency、input-layer combine，以及 $W/Aux/L$ 三个 private segment 的 terminal evaluation/opening view；
2. 不同 proof、不同 relation family 和需要独立随机性的 masking polynomial 使用 fresh randomness，且 per-proof masking/opening randomness 不得由长期 $com_W$ 的 commitment randomness、旧 transcript 或可预测标签派生；
3. 各子协议 masking、Hyrax terminal binding 与 Fiat--Shamir 非交互化具有可组合的 random-oracle-model 零知识论证，而不是只对某一个子协议单独成立；
4. 因为 $com_W$ 在 preprocessing 中长期复用，必须对共享同一 $com_W$ 的多项式数量 inference proofs 给出 multi-session zero-knowledge 保证，使其联合 verifier view 仍满足同一隐私定义。

原 zkGPT 论文并未直接给出上述针对 Lasso、terminal openings 与长期 $com_W$ 复用的完整 multi-session composition theorem。因此这些条件属于本文严格 privacy-parity comparison 对原协议建议 masking 路线的显式化要求，而不是声称原论文已经证明。若选定的 `privacy_extension_profile_id` 不能同时满足任一条件，则该配置必须标记 `unsupported_privacy_profile`；不得退化到未掩蔽或仅部分掩蔽 transcript 后仍把结果列入严格零知识主比较。

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

第 0.4 节使用的 $B_W,B_{a,src},B_{a,dst},B_b$ 必须与长期模型 commitment $com_W$ 所绑定的同一 canonical $\mathcal W_{GAT}$ 一致。

为与主实验模型注册边界一致，本对比协议固定：

1. preprocessing 时对 canonical $\mathcal W_{GAT}$ 检查各参数族的规范整数上界；
2. 用 Hyrax-style commitment 生成长期 $com_W$；
3. 把模型上界、模型结构、`quant_cfg_id` 与同一 $com_W$ 共同认证；
4. 每次 inference proof 复用该 $com_W$，不重复证明整份模型参数 range。

任何与当前 $com_W$ 无共同认证关系的模型 bounds 都不能用于 no-wrap 论证。公开特征 $T_H$ 的范围由 verifier 从公开 canonical integers 直接检查。

### 0.5.2 公共 statement 的密码学绑定

$M_{pub}$ 只包含会改变 verifier 接受语义的公共信息，至少唯一确定：GAT 结构、公开 graph/features、`quant_cfg_id`、`static_tables_id`、长期 $com_W$、input-layer layout、Lasso/GKR/MatMul public parameters、`privacy_extension_profile_id`、完整公开 $Y$ 以及所用 BN254 security parameters。

凡使用摘要绑定这些对象，必须采用 collision-resistant hash、明确 domain separation 与唯一 canonical encoding；已有 Hyrax commitment 或公开电路参数原生绑定的对象不再人为叠加第二套 hash relation。


### 0.6 全部中间变量

本节对象是 **GAT 逻辑 trace**。在 zkGPT 中，公开 features/topology 的 runtime 部分进入 $X_{pub}$ 或固定 circuit description，规范模型进入 $W/com_W$，中间 trace 与 rounding/max advice 进入 $Aux$，函数/range 查询进入 $L_{qry}$。MatMul Sumcheck、ML-friendly GKR、Lasso 和 Hyrax 分别证明这些 segment 之间的关系；这里不会复制主实验的 KZG/PSQ/LogUp 对象布局。

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

### 0.7 Input-layer layout、public input 与 private witness

原 zkGPT Protocol 1 的输入层逻辑为：

$$
X\|W\|aux\|L.
$$

本文保持这一四段结构：

$$
V_{in}=X_{pub}\|W\|Aux\|L_{qry}.
$$

先定义未补齐的最大 segment 长度：

$$
N_{raw}
=
\max\{|X_{pub}|,|W|,|Aux|,|L_{qry}|\}.
$$

由于四段后续都作为 MLE segment 使用，规范 padded segment 长度固定为：

$$
N_{in}
=
2^{\left\lceil\log_2\max(1,N_{raw})\right\rceil}.
$$

四个 segment 都按各自公开 selector / padding rule 补齐到同一个 $N_{in}$，因此逻辑 input layer 总长度为：

$$
|V_{in}|=4N_{in},
$$

且 $\log_2N_{in}$ 必为整数。后文用于 $com_W,com_{aux},com_L$ 与 Step (7) opening 的：

$$
\widetilde W,\qquad \widetilde{Aux},\qquad \widetilde L
$$

均指**长度恰为 $N_{in}$ 的规范 padded segment MLE**，不是各逻辑向量在另一维度上的独立 MLE。逻辑值占各 segment 的规范前缀/注册 coordinates；其余 coordinates 的 padding 规则由 `Layout_in` 固定。由于 $com_W$ 在 preprocessing 中长期复用，$N_{in}$ 与 padding/layout 语义必须与该长期 model commitment 共同绑定；不同 $N_{in}$ 不能在没有额外 embedding proof 的情况下直接复用同一 $com_W$。

令最后两个 selector variables 为 $(s_1,s_2)$，其余：

$$
r^{-}\in\mathbb F^{\log_2N_{in}}.
$$

四段 MLE 的组合必须按固定 segment order：

$$
\widetilde{in}(r^{-},s_1,s_2)
=
\widetilde X_{pub}(r^{-})(1-s_1)(1-s_2)
+
\widetilde W(r^{-})(1-s_1)s_2
+
\widetilde{Aux}(r^{-})s_1(1-s_2)
+
\widetilde L(r^{-})s_1s_2.
$$

该 selector 公式定义本文的规范四段 input-layer 语义；segment order 是协议关系的一部分，不能被任意重排。

原论文 Protocol 1 的文字定义明确把：

$$
X\|W\|aux\|L
$$

排列成长度 $4N$ 的 input layer。其 Step (7) 打印公式却把 $L$ 项再次写成与 $X$ 项相同的 $(1-s_1)(1-s_2)$ selector；若逐字执行，该公式无法形成四个互斥且完备的 segment。本文依据原协议已经明确的四段连续拼接语义，对这一处内部不一致作最小修正。修正后的 canonical 2-bit selector 为：

$$
00\to X,\qquad
01\to W,\qquad
10\to Aux,\qquad
11\to L.
$$

这里的 $11\to L$ 是对原论文 Step (7) 打印不一致的显式 repair，而不是声称论文原公式已经如此书写。

原 Protocol 1 Step (1) 又明确生成 $com_L$，Step (6) 把 Lasso 的 $\widetilde L_j(r_j)$ claims 与其他 input-layer claims 合并，而 Step (7) 最终检查的四段 $\widetilde{in}(r)$ 仍含 $\widetilde L(r^-)$。但打印的 Step (7) 只显式调用 $W$ 与 $aux$ 的 polynomial-commitment opening，没有给出把这个**最终随机点** $\widetilde L(r^-)$ 绑定回 $com_L$ 的 opening。前面的 Lasso claims 位于各自 $r_f$，不能自动替代任意新的 $r^-$ commitment evaluation。

因此正式 comparison 采用第二个显式最小 repair，并要求 Step (7) 对：

$$
com_W,\qquad com_{aux},\qquad com_L
$$

分别验证最终 input-combine 所要求的 terminal evaluation/opening。该 repair 不改变 Lasso、GKR 或 GAT relation，只补齐已被 Step (1) commitment 与 Step (7) 四段公式要求的最终 commitment binding。它**不是**声称原论文打印的 Step (7) 已经显式包含 $Open_L$。

若采用不同 segment 布局或另一种 Step (7) commitment-binding 关系，必须重新定义相应公共 relation，并证明其与这里的四段输入绑定等价。

#### 0.7.1 $X_{pub}$

$X_{pub}$ 只表示 **GAT 计算的公开输入层数值**，包含或确定性引用：

- $H^{(1)}=T_H[I]$；
- dimension/padding selectors；
- 当前计算必须读取的其他 public scalars。

公开 $src,dst,node\_ptr,edge\_ptr$ 的 logical statement 由认证 GKR circuit wiring 的 canonical topology identity 绑定，因此不需要把整套 topology 再复制为 private 或独立 committed segment。

`quant_cfg_id/static_tables_id` 属于 public statement / circuit identity；只有当具体 circuit 把其数值编码当作 runtime input 使用时才进入 $X_{pub}$。

**完整公开输出 $Y$ 不属于 $X_{pub}$。** 它是 output claim / public statement，由 verifier 从 $Y$ 的随机 MLE claim 开始向输入层归约。把 $Y$ 塞进 input segment 会混淆原 Protocol 1 的“public input $X$”与“public prediction/output $y$”。

#### 0.7.2 $W$

$W$ 是原 zkGPT 记号在本文中的 private **whole-model segment**，即第 0.5 节 $\mathcal W_{GAT}$ 的 canonical concatenation；它同时包含 projection matrices、attention vectors 与输出偏置，而不是只包含名字为 `W` 的矩阵。由长期 Hyrax commitment：

$$
com_W
$$

绑定。

#### 0.7.3 $Aux$

$Aux$ 是原论文 `aux` 在 GAT 中的 adaptation，固定内部子布局：

$$
Aux=Aux_{trace}\|Aux_{group}\|Aux_{max}\|Aux_{div}.
$$

其中：

- $Aux_{trace}$：第 0.6 节 canonical private intermediates；
- $Aux_{group}$：$R_U,R_{num},R_{field}$ 等 sparse group state；
- $Aux_{max}$：$M,\Delta^+,s_{max}$ 与 group-count advice；
- $Aux_{div}$：`SignedRoundDiv` 的 $\sigma,z,a,a^{-1},d^{-1},q,r,\delta$。

整个 $Aux$ 由 input-dependent commitment $com_{aux}$ 绑定；若拆分为多个 commitments，则它们的联合关系必须与同一逻辑 $Aux$ 等价。

#### 0.7.4 $L_{qry}$

$L_{qry}$ 保存 Lasso 所需 canonical scalar queries：

- function table 使用第 3.3 节 `lookup_tuple_encoding_id` 的 deterministic packed value；
- range table 直接使用规范非负整数 / limb query。

对每个 family，记真正送入 Lasso 的公开 scalar table 为：

$$
T_f^{Lasso}
=
\begin{cases}
T_f^{pack},&\text{function family},\\
T_f,&\text{scalar range/limb family}.
\end{cases}
$$

`Layout_in` 为每个 family 预留第 2.6.1 节规定的完整 padded query block；这些 block 与其 padding coordinates 全部属于同一个 committed $L_{qry}$ segment。

每个**真实** query 都必须通过 GKR arithmetic relation 绑定到 $Aux$ 中真实变量，不能由 prover 另行选择一个只满足 lookup、却不对应本次 GAT trace 的独立向量。padding query 不代表额外 GAT 运算，其值与 table-row 选择由第 2.6.1 节固定。
### 0.8 Advice 非算术 relation 与 exact `SignedRoundDiv`

原论文对 quantized division 给出的典型 advice 检查为：

$$
z=round\left(\frac{C_1}{C_2}\frac{q_x}{q_y}\right),
$$

并转成范围不等式：

$$
C_2q_y(2q_z-1)\le2C_1q_x<C_2q_y(2q_z+1).
$$

这体现 zkGPT 的“把非算术结果作为 advice，再用 range relation 验证”思想。但当前 GAT comparison 的整数函数固定为第 0.4.4 节 `SignedRoundDiv`，其符号、零和 tie 规则更具体。因此：

`division_relation = current_gat_signed_round_div`。

本文沿用 zkGPT 的原生分工：$y$ 和 auxiliary 放入 $Aux$；GKR 检查 arithmetic equalities；Lasso 证明非负/range/limb。论文原始 interval relation只能作为设计来源，不能单独替代 exact relation。

### 0.9 Constraint fusion 与 circuit squeeze

#### 0.9.1 Constraint fusion

原论文通过合并相邻 rounding constraints 减少昂贵 range proof。本文只允许 **exact-equivalent** fusion：fusion on/off 时所有主实验规范整数 checkpoint 必须逐元素相同。

不得跨越或删除：

- $H_{ELU,in}$；
- $Y^q$；
- $\widetilde Y^{pre}$；
- final $K_{out}$ average；
- `quant_cfg_id` 声明的其他 rounding checkpoint。

不能证明 exact equivalence 的候选 fusion：

`disabled_for_semantic_equivalence`。

#### 0.9.2 Circuit squeeze

允许把只依赖 input-layer advice 的独立 constraint subcircuits 压浅/加宽以增加 GKR Sumcheck 并行度。squeeze 只能改变 proof circuit topology，不能改变 relation registry、input-layer claims 或 canonical GAT trace。

### 0.10 公共 statement 与协议边界

公共 statement 固定为：

$$
Public_{zkGPT}=(M_{pub},X_{pub},com_W,StaticTables,Y),
$$

私有见证为：

$$
Witness_{zkGPT}
=
(\mathcal W_{GAT},Aux,L_{qry},Aux_{Lasso},R_{com}).
$$

$X_{pub}$ 承担原论文 public input embedding 的角色；$com_W$ 绑定完整规范模型；$Aux$ 绑定中间 trace、group/max/div advice；$L_{qry}$ 绑定 Lasso function/range queries。完整 $Y$ 是 public output claim，不属于 $X_{pub}$ private segment。

本对比协议保留原 zkGPT 的 Hyrax、ML-friendly GKR、Lasso、dedicated MatMul Sumcheck、advice 与 circuit squeeze。为了证明与主实验**同一个**定点 GAT，constraint fusion 只有在不删除任何规范 `SignedRoundDiv`/rounding checkpoint 且保持全部整数输出逐点相同时才允许；原论文会改变嵌套 rounding 语义的 fusion 在本 comparison 中禁用。

原论文 Protocol 1 的 Step (7) 打印公式存在两处内部不一致：四段 input selector 中 $L$ 项与 $X$ 项使用了同一 selector，而且最终随机点的 $\widetilde L(r^-)$ 没有显式打开 $com_L$。本文按原协议已经明确的 $X\|W\|aux\|L$ 四段输入语义采用第 0.7 节的最小修复：$11\to L$，并在最终 input-layer check 中打开 $com_L$。这两项 repair 不改变 GAT function 或 Lasso relation，只闭合原 Protocol 1 所需的 input commitment binding。

原 zkGPT 采用 BN254，并报告约 100-bit security。论文仅说明 zero knowledge 可通过类似 [69] 的 masking polynomials 加入；因此是否声明完整零知识取决于第 0.2.5 节所述 masking 是否实际实例化，不能由协议名称自动推出。

## 1. 参数生成

### 1.1 `zkGPT.KeyGen`

执行原论文抽象接口：

$$
pp\leftarrow zkPC.KeyGen(1^\lambda).
$$

本文使用原 zkGPT Hyrax/BN254 security profile。

### 1.2 `zkGPT.Commit`：模型 preprocessing

对规范展平后的 private logical model $W$，按第 0.7 / 1.6 节 `Layout_in` 嵌入长度 $N_{in}$ 的 model segment：

$$
W^{pad}=Pad_{Layout}(W;N_{in}),
$$

并把后文：

$$
\widetilde W
$$

固定解释为 $W^{pad}$ 的 $N_{in}$-point Boolean-hypercube MLE。独立采样 commitment blinder $\rho_W$：

$$
com_W\leftarrow zkPC.Commit(\widetilde W,pp,\rho_W).
$$

$com_W$ 在 preprocessing 中生成并跨具有**相同 model/layout reuse key** 的 inference proofs 复用。模型注册对象除逻辑 model shape/order/scale 外还必须绑定 $N_{in}$、padding rule 与 $com_W$ identity。若另一个公共实例导致 $N_{in}$ 改变，则不能把旧 $com_W$ 当作新 padded polynomial 的 commitment；除非另行证明两个 embedding 与同一规范模型的绑定等价。

randomness lifetime 随对象 ownership 固定：$\rho_W$ 在模型 preprocessing/registration 时采样，并与该长期复用的 $com_W$ 一起固定；$\rho_{aux}$、$\rho_L$ 属于每次 inference 的 dynamic commitments，每个 proof 必须重新独立采样。即使复用 $com_W$，本次 Hyrax evaluation/opening、GKR/Lasso masking polynomial 等零知识随机量仍必须 per-proof fresh，不能从 $\rho_W$ 派生或跨 proof 重用。
### 1.2.1 模型参数上界与注册

为与主实验固定模型的注册边界一致，本对比协议把模型参数范围检查放在一次性模型注册阶段，而不在每次 inference 中重复证明整份固定模型参数范围。

注册阶段对 canonical $\mathcal W_{GAT}$ 检查：

$$
|W|\le B_W,\qquad
|a_{src}|\le B_{a,src},\qquad
|a_{dst}|\le B_{a,dst},\qquad
|b|\le B_b,
$$

并把这些 bounds 与同一个长期 $com_W$、模型结构、`quant_cfg_id`、input-layer 布局以及 BN254 标量域共同认证。verifier 只接受与本次 $com_W$ 完全一致的注册对象；prover 不能在单次 proof 中自由更换参数上界。

原 zkGPT 论文在其实验协议中对 committed quantized weights 使用 range proof。本文不把这一 per-inference cost 混入主比较，因为主实验同样把固定模型参数范围放在模型注册边界；这里明确将该检查前移到同等级的一次性认证注册中。除此之外，$com_W$ 的 Hyrax binding、终点 evaluation/opening 以及 GAT 计算 relation 均保持不变。

### 1.3 Lasso table preprocessing

为 exact $T_{LReLU},T_{ELU},T_{exp}$ 与 range relation 建立 Lasso public-table preprocessing。table row 必须由 `static_tables_id` 确定，不得为 zkGPT 重新近似一套函数。

二元 function table 使用公开、单射的整数 tuple encoding 生成标量 lookup table $T_f^{pack}$；编码基数、偏移和认证范围必须由公共 relation 固定，从而不能把不同二元行映射到同一标量 query。

### 1.4 Dedicated MatMul plan

Dedicated MatMul relation覆盖 hidden/output projection 以及 hidden/output source/destination attention dot。多个矩阵乘法只能在 shape 与语义兼容时聚合；聚合不得改变任一原始矩阵乘法 claim。

### 1.5 ML-friendly GKR circuit

构建 arithmetic constraint circuit，至少包含：public route、score add、group states、$Sum\cdot inv=1$、$\alpha=Uinv$、$UH'$、$\alpha H'$、双聚合 consistency、max selector arithmetic、`SignedRoundDiv` arithmetic、concat、bias、head sum 和 Lasso-query consistency。

### 1.6 Input-layer layout

固定：

$$
Layout_{in}=Layout(X_{pub}\|W\|Aux\|L_{qry}).
$$

先记录第 0.7 节的 $N_{raw}$，再把四段统一 padding 到：

$$
N_{in}=2^{\lceil\log_2\max(1,N_{raw})\rceil}.
$$

$Layout_{in}$ 固定每段 offset、logical length、统一 padded length $N_{in}$、MLE variable order、visibility 与 padding selector。$W,Aux,L_{qry}$ 三个 private segment 的 commitments 都以同一个 $N_{in}$ MLE domain 为准，并必须有对应 terminal commitment binding；不能把 logical-length polynomial 的 commitment 直接当作 $N_{in}$-point MLE 的 commitment。

### 1.7 Constraint fusion、circuit squeeze 与隐私配置

公共协议关系固定以下约束变换边界：

- `constraint fusion`：仅保留与主实验定点整数关系严格等价的 fusion；会消去或改变 $H_{ELU,in}$、$Y^q$、$\widetilde Y^{pre}$ 或最终 $K_{out}$ 整数平均的 fusion 禁用；
- `circuit squeeze`：允许把彼此拓扑独立的约束子电路压浅/加宽，因为它只改变 GKR circuit topology，不改变 relation；
- `zero-knowledge masking`：严格 comparison 必须覆盖第 0.2.5 节规定的全部 witness-dependent transcript；若不能满足，则该协议不满足本文固定的隐私 relation。

### 1.8 参数生成输出

$$
SetupCtx_{zkGPT}=(pp,com_W,VK_{Lasso},C_{GKR},C_{MatMul},Layout_{in},SqueezeLayout).
$$

## 2. 见证生成

### 2.1 Public input

从公开 $T_H,I$ 得到：

$$
H^{(1)}=T_H[I],
$$

并构造 public topology/group selectors。$X_{pub}$ 由 verifier 可独立重建。

### 2.2 Canonical GAT trace

按第 0.6 节生成完整定点 trace。正式 witness 必须包含 $H_{agg,num},H_{ELU,in},Y_{agg,num},Y^q,\widetilde Y^{pre},Y^{sum}$，并严格满足本文 exact table 与整数恢复语义；任何近似浮点前向都不属于证明关系。

### 2.3 Sparse group advice

对 hidden/output 每个 head 生成：

- $R_U$：累计 $U$；
- $R_{num}$：累计 $UH'$ / $UY'$；
- $R_{field}$：累计 $\alpha H'$ / $\alpha Y'$。

三个状态的 logical edge domain 都是 $k\in\{0,\ldots,E-1\}$。统一定义域前初值：

$$
R_U[-1]=0,\qquad
R_{num}[-1,j]=0,\qquad
R_{field}[-1,j]=0.
$$

第一条有效边必为其 $(graph\_id,dst)$ group 的起点，因此 $Q_{grp,new}[0]=1$。每个新 group（包括每个新图块的第一组）在 `Q_grp_new=1` 时重置，在 `Q_grp_end=1` 时绑定节点输出；不会读取前一图块的状态。若这些 state tensors 的物理布局为二次幂 padding，则所有 $k\ge E$ 的 padding coordinates 固定为 0，且不参与 group recurrence 或 terminal binding。

### 2.4 Max advice

生成 $M,\Delta^+,s_{max}$ 与 group-count / selector witness。

### 2.5 Exact division advice

对 hidden recovery、output recovery、output alignment、final average 生成第 0.4.4 节全部 auxiliary，并写入 $Aux_{div}$。

### 2.6 Lasso canonical query

构造 $L_{qry}$：

- function-table query 存储第 3.3 节 $Pack_f(x,y)$；
- range/limb query 存储规范 scalar。

每个 packed function query 必须由 committed $Aux$ 中真实 $(x,y)$ 通过 GKR arithmetic relation计算；不能由 prover 独立选择未绑定的 query。

### 2.6.1 Lasso auxiliary matrices

把第 2.6 节 canonical lookup queries 按 function/range family 分成语义 query vector：

$$
A_f^{real}
=
(A_f[0],\ldots,A_f[m_f-1]).
$$

其中 $m_f$ 是真实 query 数。为了让 $A_f$ 具有规范 Boolean-hypercube MLE domain，定义：

$$
\bar m_f
=
2^{\left\lceil\log_2\max(1,m_f)\right\rceil}.
$$

`LassoPreprocess_f` 必须为每个 family 固定一个公开、已认证且确实属于该 table 的 padding value：

$$
a_f^{pad}\in T_f^{Lasso}.
$$

规范 padded query vector 为：

$$
A_f[q]
=
\begin{cases}
A_f^{real}[q],&0\le q<m_f,\\
a_f^{pad},&m_f\le q<\bar m_f.
\end{cases}
$$

这些 padding coordinates **不是 out-of-band Lasso witness**：它们必须占据 committed $L_{qry}$ 的规范 padding coordinates，并由 `Layout_in` / family registry 固定。第 3.4 节公开 $C'_f$ 必须覆盖 $0\le q<\bar m_f$ 的全部 coordinates；语义 GAT query count 仍只统计 $q<m_f$。

按照原 zkGPT Protocol 1 的 Lasso relation，为每个 family 构造相应私有辅助矩阵：

$$
M_f\in\{0,1\}^{\bar m_f\times \bar N_f},
$$

其中 $\bar N_f$ 是该 $T_f^{Lasso}$ 的 Lasso-native preprocessing 固定的 padded table domain。对 $q<m_f$，第 $q$ 行选择真实 query 对应 table row；对 $q\ge m_f$，该行必须选择认证的 $a_f^{pad}$ 对应 row。由此 padding 不会制造“0 必须属于所有 function tables”之类的额外语义假设。

此外，`LassoPreprocess_f` 必须登记公开 padding selector：

$$
Q_f^{pad}(q)=1\iff m_f\le q<\bar m_f.
$$

comparison arithmetic/input-consistency relation必须证明：

$$
Q_f^{pad}(q)\big(A_f[q]-a_f^{pad}\big)=0
\qquad
\forall q\in\{0,\ldots,\bar m_f-1\}.
$$

因此 padding coordinates 的 query value被固定为 $a_f^{pad}$；若 $T_f^{Lasso}$ 中该值有重复行，Lasso auxiliary matrix可以按 preprocessing 选定任一规范等价 row，但不得改变 $A_f[q]$。这样 $L_{qry}$ 的 family padding 是 protocol-defined canonical value，而不是额外自由 witness。

$M_f$ 只属于 Lasso proof witness，不替代 $A_f$、$L_{qry}$ 或 $Aux$ 的 canonical commitment；后续仍必须通过第 3.4 节把 Lasso claim 绑定回 four-segment input 中的同一个 committed $L_{qry}$。

若具体 Lasso structured-table realization把 $M_f$ 分解为多个 native committed polynomials，则 `LassoPreprocess_f` 必须规定与上述概念矩阵等价的 canonical decomposition；不能改变 query-vector、padding 或 input-binding relation。

### 2.7 Dynamic commitments

按原 zkGPT Protocol 1 的 `aux,L` 结构，先依据本次已冻结的 `Layout_in` 构造长度均为 $N_{in}$ 的：

$$
Aux^{pad}=Pad_{Layout}(Aux;N_{in}),
$$

$$
L^{pad}=Pad_{Layout}(L_{qry};N_{in}),
$$

其中 $L^{pad}$ 内各 family 的 padding block遵循第 2.6.1 节，family block之外的剩余 tail padding遵循 `Layout_in` 固定值。后文 $\widetilde{Aux},\widetilde L$ 均指这两个 padded segments 的 MLE。

对它们生成 input-dependent private commitments：

$$
com_{aux}\leftarrow zkPC.Commit(\widetilde{Aux},pp,\rho_{aux}),
$$

$$
com_L\leftarrow zkPC.Commit(\widetilde L,pp,\rho_L).
$$

$com_W$ 使用第 1.2 节同一 $N_{in}$ domain 的 preprocessing commitment；$X_{pub}$ 不做 private commitment。若三者任一 commitment 的 MLE domain/layout 与本次 `Layout_in` 不一致，proof relation 直接无效。

## 3. 证明生成

### 3.1 输入与输出

输入：

$$
(SetupCtx_{zkGPT},Public_{zkGPT},Witness_{zkGPT}).
$$

其中 $Witness_{zkGPT}$ 按第 0.0 / 0.10 节展开为 $(W,Aux,L_{qry},Aux_{Lasso},R_{com})$，避免证明算法的输入签名遗漏第 2.6.1 节所需的 $M_f$ witness。

输出：

$$
\pi_{GAT}^{zkGPT}.
$$

### 3.2 与原 zkGPT Protocol 1 对齐的整体过程

本文按原论文 `X\|W\|aux\|L` 协议结构进行 GAT adaptation。原论文打印 Step (7) 的 selector 与 $L$ terminal commitment binding 存在第 0.7 节说明的内部缺口；本文的严格 relation 使用两个显式 repair，而不把 repair 误写成论文原文已经存在。

总体过程固定为：

1. prover 计算 public prediction $Y$、private $Aux$ 与 $L_{qry}$；
2. prover 发送 $com_{aux},com_L$ 与 $Y$；$com_W$ 已在 preprocessing 中固定；
3. prover 先发送**全部** lookup families 的 Lasso-native commitments $\{LC_f\}_f$；只有全部 commitments 固定后才生成各 $r_f$，随后执行各 family 的完整 Lasso proof；
4. verifier 从公开 $Y$ 的随机 MLE claim 开始验证计算；
5. 对 registered matrix-multiplication layer/claim 执行 dedicated MatMul Sumcheck；
6. 对其他 arithmetic constraint layers 执行 ML-friendly GKR；
7. 到达 input layer 后，把 MatMul/GKR 对 $W/Aux$ 的 claims 与 Lasso 已验证的 $\{\ell_f=\widetilde A_f(r_f)\}_f$，通过第 3.4 与 3.14 节 consistency/combine relation 一并归约为唯一 $\widetilde{in}(r)$ claim；
8. verifier 本地计算 $\widetilde X_{pub}(r^-)$，并对 $com_W,com_{aux},com_L$ 的相应 terminal evaluations/openings 全部验证，再用第 0.7 节的四段 selector 重新计算 $\widetilde{in}(r)$。

第 8 步中的 $Open_L$ 是本文为补齐原打印 Step (7) 最终 $\widetilde L(r^-)$ commitment binding 所做的显式 repair；它不是重复执行第 3 步的 Lasso membership，也不能由位于其他随机点 $r_f$ 的 Lasso claims替代。

### 3.3 Lasso function/range proof

原 zkGPT/Lasso 的 lookup vector 是标量向量。GAT 的 $T_{LReLU},T_{ELU},T_{exp}$ 是二元函数表，因此本文固定使用公开、确定性的单射整数 packing：

对 function family $f$，公开 bounds 为：

$$
-B_{f,x}\le x\le B_{f,x},
\qquad
-B_{f,y}\le y\le B_{f,y}.
$$

定义：

$$
x^+=x+B_{f,x},\qquad y^+=y+B_{f,y},
$$

$$
B_f^{pack}=2B_{f,y}+1,
$$

$$
Pack_f(x,y)=x^+B_f^{pack}+y^+.
$$

为了让上述整数表示真正 injective，不能只检查 packed scalar 本身。GKR/Lasso range relation必须同时证明：

$$
-B_{f,x}\le x\le B_{f,x},
\qquad
-B_{f,y}\le y\le B_{f,y}.
$$

于是：

$$
0\le x^+\le 2B_{f,x},
\qquad
0\le y^+\le 2B_{f,y}<B_f^{pack}.
$$

因此 $Pack_f$ 在认证 rectangle 上是一一映射。`integer_bound_ledger` 还必须证明最大可能 packed integer：

$$
B_{Pack,f}
=
2B_{f,x}(2B_{f,y}+1)+2B_{f,y}
<
\frac{p_{zkGPT}-1}{2},
$$

并对 GKR 中的乘法 $x^+B_f^{pack}$ 检查同一 no-wrap 条件。否则一个越界 $(x,y)$ 可能通过“进位/借位”与合法 table row 形成相同 packed scalar，不能把 Lasso scalar membership 解释为原二元 tuple membership。

Lasso preprocessing 构造：

$$
T_f^{pack}=\{Pack_f(x_t,y_t)\}_t.
$$

第 2.6.1 节把每个 family 的真实 query 数 $m_f$ padding 到 $\bar m_f$，并把 padding coordinates作为 committed $L_{qry}$ 的规范部分。Lasso random point 因此属于：

$$
r_f\in\mathbb F^{\log_2\bar m_f}.
$$

对每个 lookup family $f$，先按原 zkGPT Protocol 1 Step (2)--(3) 执行 Lasso membership 阶段；随后在 Protocol 1 Step (5) 的 GKR arithmetic relation 中证明 $L_{qry}$ 中该 packed query 确实等于 committed $Aux$ 中真实 $(x,y)$ 的 $Pack_f(x,y)$。两者的验证顺序必须保持这一 paper-native 顺序，不能为了叙述方便把后面的 GKR relation 提前到 Lasso challenge 之前。

与原 Protocol 1 Step (2) 一致，本次 proof 的**全部** lookup families 必须先固定各自 Lasso-native commitment：

$$
LC_f=LassoCommit(M_f).
$$

只有当完整有序集合 $\{LC_f\}_f$ 都已经进入 transcript 后，才允许派生任一 family 的 $r_f$。随后对每个 family：

1. prover 给出 canonical padded query-vector claim：
   $$
   \ell_f=\widetilde A_f(r_f);
   $$
2. prover/verifier执行该 family 的**完整 native Lasso argument**：
   $$
   \pi_f^{Lasso},
   $$
   其 terminal relation 包含：
   $$
   \sum_{y\in\{0,1\}^{\log_2\bar N_f}}
   \widetilde M_f(r_f,y)\,T_f^{Lasso}[y]
   =
   \ell_f.
   $$

上式只是 Lasso argument 的 terminal membership equation，不能把“检查这一条等式”当作完整 Lasso protocol 的替代。$\pi_f^{Lasso}$ 必须包含 native Lasso soundness 所要求的全部 commitment/Sumcheck/opening messages。

因此 $\Pi_{Lasso}$ 至少包含每个 family 的 $(LC_f,\ell_f,\pi_f^{Lasso})$；$r_f$ 由 transcript 重算，不是 prover 自选字段。缺少 $LC_f$、在全部 $LC$ 固定之前派生任一 $r_f$、只验证 terminal equality 而没有完整 Lasso argument，或没有执行第 2.6.1 节 padding binding，均不等价于原 Protocol 1 的 Step (2)--(3)。

由上述 relation分别证明：

$$
(S,Z)\in T_{LReLU},\qquad
(\Delta^+,U)\in T_{exp},\qquad
(H_{ELU,in},H_{agg})\in T_{ELU}.
$$

range queries 继续覆盖 $\Delta^+$、max auxiliary、`SignedRoundDiv` 的 $a,q,r,\delta$ 与 no-wrap limbs。


### 3.4 Lookup-vector 与 input-layer consistency

仅验证第 3.3 节 Lasso terminal relation 还不足以证明 $\ell_f$ 来自本次 committed $L_{qry}$。必须进一步使用原 zkGPT Appendix A 的 lookup-vector/input consistency relation。

令完整 input-layer 向量为：

$$
V_{in}=X_{pub}\|W\|Aux\|L_{qry}.
$$

对每个 lookup family $f$，定义由 `Layout_in` 和 family segment registry 唯一确定的公开 selector/incidence matrix：

$$
C'_f(x,z)\in\{0,1\},
$$

其中 $C'_f(x,z)=1$ 当且仅当 padded query vector $A_f[x]$ 正是 $V_{in}[z]$ 中对应的 canonical $L_{qry}$ coordinate。第 2.6.1 节的 padding coordinates 同样必须映射到 committed $L_{qry}$ 的 padding coordinates；$C'_f$ 只描述公开位置映射，不携带 private lookup value。

在全部 $(LC_f,\ell_f)$ 已固定且完整 Lasso arguments 已按第 3.3 节执行后，verifier / Fiat--Shamir 生成独立随机组合系数 $\eta_f$，并证明：

$$
\sum_f \eta_f\,\widetilde A_f(r_f)
=
\sum_{z}
\widetilde V_{in}(z)
\left(
\sum_f
\eta_f\,\widetilde C'_f(r_f,z)
\right).
$$

左侧的 $\widetilde A_f(r_f)$ 必须逐 family 等于第 3.3 节 Lasso 已绑定的 $\ell_f$。该 Sumcheck 把全部 Lasso query-vector claims 归约到同一个 four-segment $V_{in}$ claim，从而排除“另外构造一个满足 table membership、但与本次 GAT witness/input 无关的 query vector”。

此外，GKR arithmetic relation仍必须证明每个**真实** packed function query 是 committed $Aux$ 中真实 $(x,y)$ 的 $Pack_f(x,y)$，range family也必须绑定到其 canonical range/div/max auxiliary；padding coordinates则必须通过第 2.6.1 节登记的 $Q_f^{pad}$ / padding-row relation绑定为公开认证 $a_f^{pad}$。Lasso membership、$A_f\leftrightarrow V_{in}$ consistency、真实 query 的 $Aux\leftrightarrow A_f$ arithmetic relation与 padding-value binding 缺一不可。

第 3.4 节只把 Lasso claims 绑定到 four-segment input polynomial；它不替代第 3.15 节在最终随机点 $r^-$ 对 $com_L$ 的 terminal opening。

### 3.5 Dedicated MatMul Sumcheck

对矩阵关系：

$$
C=AB,
$$

在随机输出点 $(u,v)$ 检查：

$$
\widetilde C(u,v)=\sum_z\widetilde A(u,z)\widetilde B(z,v).
$$

使用原论文的 dedicated matrix-multiplication Sumcheck，并仅聚合语义与 shape 兼容的矩阵乘法 claims。

本 relation 用于：

$$
H'=HW,
$$

$$
E_{src}=H'a_{src},\qquad E_{dst}=H'a_{dst},
$$

及输出对应 relation。

### 3.5.1 Public sparse routing / gather

本协议把公开 $src,dst,Q_{grp,new},Q_{grp,end}$ 与 node-to-edge wiring 固定在认证的 GKR circuit description 中，并由 circuit identity 绑定该公开 topology。逻辑公共 statement 仍包含 $src,dst,node\_ptr,edge\_ptr$；验证者必须检查其 canonical identity 与该 circuit wiring 一致。

因此 routed edge views 在 GKR relation 中满足：

$$
H_{src}^{\prime edge}[k,j]=H'[src(k),j],
$$

$$
E_{src}^{edge}[k]=E_{src}[src(k)],\qquad
E_{dst}^{edge}[k]=E_{dst}[dst(k)],
$$

并以同一公开 $dst$ wiring 绑定 $M^{edge},Sum^{edge},inv^{edge}$、$H_{agg,num}^{edge}$ / $Y_{agg,num}^{edge}$ 与输出层 $Y_{src}^{\prime edge}$。后续 score、$\Delta^+$、$\alpha$、integer numerator 与 group states 只能引用这些已绑定 routed views。

### 3.6 Sparse GKR：group sum

对所有有效边 $0\le k<E$ 定义边顺序上的状态：

$$
R_U[k]=Q_{grp,new}[k]U[k]+(1-Q_{grp,new}[k])(R_U[k-1]+U[k]),
$$

其中 $k=0$ 使用第 2.3 节约定的 $R_U[-1]=0$。

组末：

$$
Q_{grp,end}[k]\big(R_U[k]-Sum^{edge}[k]\big)=0,
$$

其中 $Sum^{edge}[k]$ 已由第 3.5.1 节 destination broadcast 绑定到 $Sum[dst(k)]$。

### 3.7 Sparse GKR：integer numerator

隐藏层：

$$
P_U[k,j]=U[k]H_{src}^{\prime edge}[k,j],
$$

对所有有效边 $0\le k<E$：

$$
R_{num}[k,j]=Q_{grp,new}[k]P_U[k,j]+(1-Q_{grp,new}[k])(R_{num}[k-1,j]+P_U[k,j]),
$$

其中 $k=0$ 使用 $R_{num}[-1,j]=0$。

组末：

$$
Q_{grp,end}[k]
\big(
R_{num}[k,j]-H_{agg,num}^{edge}[k,j]
\big)
=0,
$$

其中 $H_{agg,num}^{edge}$ 通过同一 public destination mapping 与 canonical node tensor $H_{agg,num}$ 绑定。输出层对应使用 $Y_{agg,num}^{edge}$。

输出层把 $H'$ 换成 $Y'$，得到 $Y_{agg,num}$。

### 3.8 Sparse GKR：field aggregate

对有效边定义：

$$
P_\alpha[k,j]=\alpha[k]H_{src}^{\prime edge}[k,j],
$$

$$
R_{field}[k,j]
=
Q_{grp,new}[k]P_\alpha[k,j]
+
(1-Q_{grp,new}[k])
\big(R_{field}[k-1,j]+P_\alpha[k,j]\big),
$$

其中 $k=0$ 使用 $R_{field}[-1,j]=0$。在组末约束：

$$
Q_{grp,end}[k]
\big(
R_{field}[k,j]-H_{agg,pre}^{edge}[k,j]
\big)=0.
$$

输出层采用同一状态关系，把 $H'$ 替换为 $Y'$，终点绑定到 $Y^{pre}$。

### 3.9 Inverse 与双聚合 consistency

GKR 检查：

$$
Sum\cdot inv=1,
$$

$$
\alpha_k
=
U_k\,inv^{edge}[k],
$$

$$
H_{agg,num}=Sum\cdot H_{agg,pre},
$$

$$
Y_{agg,num}=Sum\cdot Y^{pre}.
$$

### 3.10 Exact max

GKR 检查：

$$
\Delta_k^+
=
M^{edge}[k]-Z_k,
$$

$$
s_{max}(s_{max}-1)=0,
$$

组内：

$$
\sum_{dst(k)=i}s_{max,k}=1,
$$

$$
s_{max,k}\Delta_k^+=0.
$$

Lasso range 证明 $\Delta^+\ge0$。

### 3.11 Exact `SignedRoundDiv`

GKR 逐项检查第 0.4.4 节 arithmetic identities；Lasso/range 检查非负和 limbs。论文的 division interval 只说明 advice 思路，不替代该 relation。

### 3.12 Constraint fusion

仅对形式证明 exact-equivalent 的 adjacent arithmetic constraints 启用。所有 required rounding checkpoint 作为不可跨越 boundary；否则标记 `disabled_for_semantic_equivalence`。

### 3.13 Circuit squeeze

把独立 constraint subcircuits 重排为更少、更宽的 GKR layers。squeeze 前后 output claim、input-layer claim set 与 relation IDs 必须一致。

### 3.14 Input-layer combine

设 Lasso/MatMul/GKR 最终产生 claims：

$$
\{\widetilde V_j(r_j)=v_j\}_{j=0}^{m-1},
$$

其中 $V_j$ 属于 $W,Aux,L_{qry}$ 或 public $X_{pub}$。使用 fresh random coefficients与 Sumcheck 把这些 claims归约到 four-segment input polynomial 的唯一 terminal claim：

$$
\widetilde{in}(r),\qquad
r=(r^-,s_1,s_2).
$$

Lasso claims不能仅因为“出现在 Lasso transcript 中”就被接受。第 3.4 节 exact consistency relation先把：

$$
\{\ell_f=\widetilde A_f(r_f)\}_f
$$

通过公开 $C'_f$ 绑定到同一个 $V_{in}$；该 relation可以与其他 input-layer claims在同一 randomized Sumcheck 中联合执行，也可以等价分阶段执行。

必须区分两个不同目标：

1. 第 3.4 节证明 Lasso query claims 属于 committed-input 所声明的 $L_{qry}$ **位置关系**；
2. 本节最终得到新的随机点 $r^-$ 后，第 3.15 节还必须证明用于四段 $\widetilde{in}(r)$ 重构的 $\widetilde L(r^-)$ 确实是 Step (1) 中 $com_L$ 所承诺 polynomial 的 evaluation。

因此最终 input-layer binding 必须满足：

- $W$ terminal claim绑定到 $com_W$；
- $Aux$ terminal claim绑定到 $com_{aux}$；
- $L_{qry}$ terminal claim绑定到 $com_L$；
- public $X_{pub}$ 由 verifier 本地 evaluation；
- 第 0.7 节的四段 selector 把四段重构成唯一 $\widetilde{in}(r)$ claim。

只执行前面的 Lasso consistency、却省略最终 $com_L$ opening 的 printed Step (7) 不能闭合 input commitment binding，因此不属于本文的严格 relation。

### 3.15 Hyrax terminal binding 与零知识 masking

input-layer combine 结束后，验证者得到唯一随机点 $r^-$. 正式 Step (7) repair 要求分别把三个 private segment 的 terminal evaluations 绑定回：

$$
com_W,\qquad com_{aux},\qquad com_L.
$$

即验证相应的 Hyrax evaluation/opening proofs，并由 verifier 本地计算 $\widetilde X_{pub}(r^-)$，再检查修复后的四段 selector：

$$
\widetilde{in}(r)
=
\widetilde X_{pub}(r^-)(1-s_1)(1-s_2)
+
\widetilde W(r^-)(1-s_1)s_2
+
\widetilde{Aux}(r^-)s_1(1-s_2)
+
\widetilde L(r^-)s_1s_2.
$$

前面的 Lasso claims 位于各自随机点 $r_f$，不能替代这里对新随机点 $r^-$ 的 $com_L$ binding。

为满足第 0.1 节固定的 `model_and_trace_hidden` 隐私 relation，这些 terminal evaluations 以及前面的 witness-dependent GKR/MatMul/Lasso transcript 必须进入第 0.2.5 节要求的同一 masking construction。由于 $com_W$ 跨 inference proof 长期复用，本次 proof 的 masking、dynamic commitment 与 terminal-opening randomness 必须 fresh，并且该 construction 必须满足第 0.2.5 节规定的 multi-session composition。若没有完整 masking，或只能证明单次 transcript 隐私而不能支持长期 $com_W$ 的联合 verifier view，则本节只能提供 commitment binding 与 soundness，不能构成本文的严格 zkGPT 对比协议，并应标记 `unsupported_privacy_profile`。

### 3.16 Fiat--Shamir 非交互化

原 zkGPT 明确把交互式 public-coin protocols 通过 Fiat--Shamir 转为可离线验证的非交互 proof。每个 challenge 必须由密码学安全、domain-separated 的 hash-to-field 从当前 public statement、commitments 与此前全部 prover messages 派生。

尤其必须保证：Lasso-native commitments 在其随机 query point 之前固定；GKR/MatMul/Sumcheck 每一轮 message 在下一轮 challenge 之前吸收；input-layer combine 与最终 Hyrax openings 继续沿同一 transcript 顺序。任何 challenge 都不能由 prover 自由指定。

### 3.17 最终证明对象

抽象地：

$$
\pi_{GAT}^{zkGPT}
=
(com_{aux},com_L,\Pi_{Lasso},\Pi_{MatMul},\Pi_{GKR},\Pi_{combine},\Pi_{open}).
$$

长期 $com_W$、公开 $X_{pub}$、静态表与完整 $Y$ 来自 verifier context / public statement；proof 中各子协议通过同一 Fiat--Shamir transcript 绑定。masking messages 属于对应子协议 proof 的一部分，但不改变该抽象计算 relation。

## 4. 验证

### 4.1 Verifier context

验证者持有 $pp$、认证 $com_W$、Lasso public tables、GKR/MatMul relation、$Layout_{in}$、公开 graph/features 与完整 $Y$。

### 4.2 公共 relation 检查

检查 GAT 结构、公开 graph/features、量化与静态表、长期 $com_W$、四段 input layout、Lasso/GKR/MatMul public parameters、Step (7) 两项 repair 以及完整 $Y$ 是否与当前 verifier context 一致。

### 4.3 Public input reconstruction

在执行任何 public reconstruction、MLE evaluation、Lasso/MatMul/GKR verification 或 Hyrax terminal binding 之前，验证者必须检查第 0.1 节与第 0.3 节定义的全部模型维度和公共 instance 合法性条件；任一条件不满足必须立即拒绝。

从公开 $T_H,I,src,dst,node\_ptr,edge\_ptr$ 重建 $X_{pub}$、group selectors、padding 和 public MLE evaluator。协议中不存在 private $com_{T_H}$；公开特征必须由 verifier 直接重建。

### 4.4 Transcript replay

按第 3.16 节逐消息重放所有 Fiat--Shamir challenges。

### 4.5 Lasso

按第 3.16 节 transcript 顺序验证每个 function/range family 的完整 native Lasso proof $\pi_f^{Lasso}$，并验证其 $LC_f$、$r_f$、$\ell_f$、$T_f^{Lasso}$、$Q_f^{pad}$ 与第 2.6.1 节 padding relation一致；同时验证 $Q_f^{pad}(A_f-a_f^{pad})=0$ 的 arithmetic/input-consistency relation。不能只检查 terminal membership equality。

随后验证第 3.4 节 lookup-vector/input consistency，使所有真实与 padding query coordinates都归约到 committed canonical $L_{qry}$ segment。

### 4.6 MatMul

验证全部 registered dedicated matrix multiplication Sumchecks，其 terminal claims 必须指向 $X_{pub},W,Aux$ 对应 input-layer segments。

### 4.7 GKR

验证 arithmetic circuit，至少覆盖 sparse group states、inverse/alpha、$UH'$/$\alpha H'$、双聚合 consistency、max selector、exact `SignedRoundDiv`、concat、bias、head sum 和 lookup-query consistency。

### 4.8 Combine

验证 Lasso/MatMul/GKR terminal input claims 的 combine Sumcheck，得到唯一：

$$
\widetilde{in}(r),\qquad r=(r^-,s_1,s_2).
$$

第 3.4 节的 lookup/input consistency必须作为该 combine relation 的组成部分得到验证。combine 结束后，$W,Aux,L$ 三个 private segment在同一 $r^-$ 上的 terminal values仍必须由第 4.9 节各自 commitment opening绑定。

### 4.9 Hyrax terminal binding

验证 $com_W,com_{aux},com_L$ 在最终随机点 $r^-$ 的 evaluation/opening proofs，并由 verifier 本地计算 $\widetilde X_{pub}(r^-)$。随后检查：

$$
\widetilde{in}(r)
=
\widetilde X_{pub}(r^-)(1-s_1)(1-s_2)
+
\widetilde W(r^-)(1-s_1)s_2
+
\widetilde{Aux}(r^-)s_1(1-s_2)
+
\widetilde L(r^-)s_1s_2.
$$

缺失 $com_L$ 的最终 binding、使用错误 commitment，或 selector 不满足 $00\to X,01\to W,10\to Aux,11\to L$ 时直接拒绝。若声明 zero knowledge，还必须额外验证第 0.2.5/3.15 节规定的 masking construction。

### 4.10 Final output

最后必须确认完整 public：

$$
Y=SignedRoundDiv(Y^{sum},K_{out})
$$

由已验证 GKR/lookup/matmul relation 推导。output hash 不能替代 full output binding。

### 4.11 接受条件

所有 metadata、public reconstruction、Lasso、MatMul、GKR、combine、openings 和 final output 全部通过才输出 $Accept$。

## 5. 完备性、可靠性与零知识边界

### 5.1 完备性

正确 GAT trace 可生成满足完整 Lasso、MatMul、GKR、lookup-input consistency、input combine 与 $W/Aux/L$ 三段 terminal Hyrax openings 的 proof；诚实 verifier 按修复后的四段 Step (7) relation 接受。

### 5.2 可靠性

可靠性依赖 Hyrax commitment binding、完整 Lasso lookup/range soundness、dedicated matrix Sumcheck、ML-friendly GKR、input-layer combine 与 Fiat--Shamir 非交互化。

对每个 lookup family：

1. 全部 $LC_f$ 必须在任一 $r_f$ 之前固定；
2. 完整 $\pi_f^{Lasso}$ 把 $\ell_f$ 绑定到公开 table relation；
3. 第 3.4 节通过公开 $C'_f$ 把同一个 $\ell_f$ 绑定到 canonical padded $L_{qry}$ / four-segment $V_{in}$；
4. GKR arithmetic relation把真实 query value绑定到真实 $Aux$；
5. final input combine产生新随机点 $r^-$ 后，$Open_L$ 再把 $\widetilde L(r^-)$ 绑定到 Step (1) 的 $com_L$。

因此所有 advice/query 最终必须归约到已承诺 $W/Aux/L$ 或 verifier-known $X_{pub}$。若省略第 5 步，只保留论文打印的 no-$L$-opening Step (7)，本文不把该协议变体视为已闭合的正式 soundness relation。

### 5.3 Zero knowledge

原 zkGPT 论文说明 zero knowledge 可以通过类似文献 [69] 的 masking polynomials 加入，但 Protocol 1 本身没有给出完整 masking transcript。因此本 comparison 只在实际协议明确实例化并验证该 masking 时声明完整 zero knowledge。

需要隐藏的 verifier view 至少包括与 $\mathcal W_{GAT}$、$Aux$、$L_{qry}$ 有关的 GKR/MatMul/Lasso witness-dependent messages以及最终 Hyrax terminal evaluations。仅使用 hiding $com_W/com_{aux}/com_L$，但让普通 Sumcheck/GKR transcript 暴露 witness-dependent evaluations，不足以支持完整 ZK claim。

严格 `masked_full` 还必须满足第 0.2.5 节定义的 composition 与 multi-session 条件：共享同一长期 $com_W$ 的多份 inference proofs，其联合 verifier view 仍须保持对 $\mathcal W_{GAT}$、$Aux$、$L_{qry}$ 与 terminal evaluations 的零知识；每份 proof 使用 fresh masking/opening randomness。原论文没有直接给出这一完整 multi-session theorem，因此只有登记的 `privacy_extension_profile_id` 实际提供相应论证时才声明完整 zero knowledge。

未启用 masking、仅部分 masking，或无法证明上述 multi-session 条件的核心关系仍可以具有计算可靠性和模型 commitment binding，但它不满足本文固定的 `model_and_trace_hidden` 零知识要求，并必须标记 `unsupported_privacy_profile`。masking 不得改变第 0 节的 GAT 数值 relation。

### 5.4 Native security parameters

原 zkGPT 使用 Hyrax PCS 与 BN254，论文报告约 100-bit security。comparison 使用真实 native security 参数，不把它写成 128-bit 以追求表面一致。

### 5.5 Constraint fusion 可靠性边界

原论文 constraint fusion 以减少 rounding/range relation 为优化目标；本 comparison 只有在所有主实验规范整数 checkpoint 完全相同时才允许启用。数值误差“没有变差”不足以证明 semantic equivalence。

### 5.6 Circuit squeeze 边界

squeeze 只改变 GKR circuit topology；如果某个 relation 在 squeeze 后不再出现在 verifier 检查集合，则该变换不再与原关系等价。
