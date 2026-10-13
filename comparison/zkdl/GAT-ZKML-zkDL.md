# GAT-ZKML zkDL 协议笔记

## 0. 符号

### 0.0 证明关系与对比实验范围

本文定义的 relation 记为：

$$
R_{GAT\text{-}zkDL}.
$$

公共输入为：

$$
Public_{zkDL}=(M_{pub},Topo,T_H,StaticTables,VK_{model}^{zkDL},Y),
$$

私有见证为：

$$
Witness_{zkDL}
=
(W,Trace_{GAT},Aux_{lookup},Aux_{max},Aux_{div},
\{S_i\}_{i=0}^{N_T-1},
R_{model},R_{dyn}).
$$

其中：

- $R_{model}=\{r_i^{model}\}$ 是模型注册时为长期 model-family commitments 独立采样并由 prover 私下保留的 opening randomness；
- $R_{dyn}=\{r_i^{dyn}\}$ 是本次 inference 对其余 dynamic private tensor-family stacks 独立采样的 fresh commitment randomness。

两类 randomness 都不进入 public statement；任何两个独立 private commitments 不共享同一个 blinder。$S_i$ 是第 0.7 节定义的 FAC4DNN tensor-family stack。

原 zkDL 论文证明的是深度学习训练，核心是 zkReLU、FAC4DNN、MLE/Sumcheck 与 Pedersen/Hyrax commitment；本文只证明 GAT inference。因此本文固定称为 `zkDL-style GAT inference adaptation`，不包含训练反向传播、梯度或参数更新，也不把 LeakyReLU/ELU/ExpMap 称作原论文 zkReLU。

GAT 新增的 table membership、稀疏 route、group max 和 `SignedRoundDiv` 均使用 Hyrax/MLE/Sumcheck/FAC4DNN 体系内的 relation 表达；不得引入主实验 LogUp、KZG quotient 或 zkGPT Lasso。


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

### 0.2 有限域、MLE、Hyrax、Sumcheck 与 FAC4DNN

本文采用 zkDL 论文的核心证明机制，但证明对象是 GAT inference，而不是训练。因此本文固定称为：

`zkDL-style GAT inference adaptation`。

原 zkDL 论文的核心组件为：

- tensor 的多线性扩展 MLE；
- Sumcheck；
- Pedersen/Hyrax-style hiding commitment；
- tensor MLE evaluation proof；
- FAC4DNN；
- zkReLU；

原论文 zkReLU 专门证明 ReLU 及其 backpropagation，不等于本文的 LeakyReLU、ELU 或 ExpMap；这些 GAT 非线性必须使用本文定义的 compatibility relation。

原论文使用 BLS12-381 参数并报告 128-bit security；本 comparison 保持该原生安全参数边界。

原论文的 threat model 明确假设 verifier 为 `semi-honest`：验证者遵循协议，但试图获知 prover 持有的私有数据和模型参数。本文的 GAT adaptation 可以继续要求对作弊 prover 的计算可靠性，但不能仅因为增加 hiding commitment / masking 就把原论文的 privacy theorem 自动升级为“任意恶意 verifier 下零知识”。若要声明更强 verifier model，必须另外给出相应 zero-knowledge argument；否则隐私结论固定在原论文的 semi-honest verifier 边界。

#### 0.2.1 MLE

对 padding 后长度 $2^m$ 的 tensor $T$：

$$
\widetilde T(u)=\sum_{b\in\{0,1\}^m}\widetilde{eq}(u,b)T(b),
$$

$$
\widetilde{eq}(u,b)=\prod_t(u_tb_t+(1-u_t)(1-b_t)).
$$

#### 0.2.2 Hyrax commitment 与 terminal evaluation

私有 tensor 使用带独立随机量的 Pedersen/Hyrax-style hiding commitment：

$$
Com(T;r)\leftarrow HyraxCommit(PP_{com},T,r).
$$

Sumcheck / FAC4DNN 最终归约得到 tensor 的随机点评值 claim：

$$
v=\widetilde T(u).
$$

随后按照原 zkDL Protocol 1 Line 11，对该 claim 与同一个 commitment 执行 proof of evaluation：

$$
VerifyEval(VP_{com},Com(T),u,v,\pi_{eval})=1.
$$

本文的零知识边界不额外发明另一套“masked evaluation commitment”或替换 Line 11。零知识性按照原 zkDL Theorem C.2 的组合边界理解：commitment 使用 zero-knowledge Pedersen/Hyrax hiding，witness-dependent Sumcheck 使用 zero-knowledge Sumcheck variant，而 proof of evaluation 仍作为 `zkDL-Prove` 的组成部分被同一零知识模拟论证覆盖。

#### 0.2.3 FAC4DNN

对 $R$ 个同类型矩阵乘法实例 $Z^{(i)}=X^{(i)}Y^{(i)}$，把它们沿新实例维堆叠为 $\mathcal X,\mathcal Y,\mathcal Z$。原论文的聚合式可写成：

$$
\widetilde{\mathcal Z}(w,u_1,u_3)
=
\sum_i\widetilde\beta(w,i)
\sum_j\widetilde{\mathcal X}(i,u_1,j)\widetilde{\mathcal Y}(i,j,u_3).
$$

先在实例维执行 Sumcheck，把多个实例压缩到随机点 $v$，再继续单个 operation 的 Sumcheck。FAC4DNN 的作用是聚合同类型 relation 的 proof，不改变原始 GAT 的层间依赖和明文计算顺序。

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

第 0.4 节 `integer_bound_ledger` 使用的 $B_W,B_{a,src},B_{a,dst},B_b$ 必须与实际参与 proof 的同一规范模型 $\mathcal W_{GAT}$ 绑定。

为与主实验固定模型的注册边界一致，本对比协议固定采用一次性模型注册：

1. 把模型参数按 zkDL tensor-family 规则组成长期 model stacks；
2. 注册阶段检查各参数族的规范有符号整数上界；
3. 生成并认证对应的长期 Pedersen/Hyrax commitments $VK_{model}^{zkDL}$；
4. 将参数上界与 $VK_{model}^{zkDL}$、模型结构、`quant_cfg_id` 和 BLS12-381 标量域共同绑定；
5. 每次 inference proof 复用这些认证 commitments，不再重复证明整份模型参数 range。

因此本次 proof 中不能由 prover 自由声明新的 $B_W,B_a,B_b$。公开特征 $T_H$ 的范围由 verifier 从公开 canonical integers 直接检查。

### 0.5.2 公共 statement 的密码学绑定

$M_{pub}$ 只保留会改变 verifier relation 的公共信息，至少唯一确定：GAT 结构、公开 graph/features、`quant_cfg_id`、`static_tables_id`、长期 $VK_{model}^{zkDL}$、tensor/operation-family 组织方式、公开输出 $Y$ 以及当前 commitment/security parameters。

凡用摘要绑定上述对象，必须使用 collision-resistant hash、明确 domain separation 与唯一 canonical encoding；已有 Pedersen/Hyrax commitment 或公开参数原生绑定的对象不再重复引入第二套 hash proof。


### 0.6 全部中间变量

本节对象在 zkDL 中解释为 **canonical private/public tensor families**。私有 trace 后续进入 MLE、Hyrax commitment 和 FAC4DNN operation-family Sumcheck；公开 topology/table 由 verifier-known tensor/selector 参与 relation。这里保留与主实验相同的逻辑变量，是为了固定被证明的 GAT 计算，不意味着复制主实验的 polynomial/LogUp/PSQ 表示。

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

### 0.7 Tensor family、MLE 布局与 FAC4DNN

#### 0.7.1 Tensor family

按照原 zkDL Protocol 1 的思想，把本次 GAT 中所有 private tensor 按“相同性质 + 兼容维度”划分为 $N_T$ 个 tensor families：

$$
\mathcal T_0,\mathcal T_1,\ldots,\mathcal T_{N_T-1}.
$$

对 family $\mathcal T_i$，按 `TensorFamilyRegistry` 的固定实例顺序堆叠：

$$
S_i=Stack(T_{i,0},T_{i,1},\ldots,T_{i,n_i-1}).
$$

$S_i$ 是该 family 的 canonical committed representation。每个 entry 登记 logical shape、stack axis、instance tag、padding shape、MLE variable order 与 visibility。

公开 $T_H$、topology 和 $Y$ 不作为秘密 tensor family；它们的 terminal MLE value 由 verifier 本地计算。

#### 0.7.2 Operation family

类似地，把需要证明的 relation 划分为 $N_A$ 个 operation families：

$$
\mathcal A_0,\ldots,\mathcal A_{N_A-1}.
$$

至少包括：projection matmul、attention dot、public route、score add、LeakyReLU membership、max/range、ExpMap membership、group sum、inverse/alpha、$UH'$、$\alpha H'$、field aggregate、integer numerator、consistency、`SignedRoundDiv`、ELU membership、concat、output bias、head sum/final average。

只有 operation type 和 shape 足够兼容、能够共享同一个 aggregated Sumcheck polynomial 的实例才进入同一 family；FAC4DNN 不要求也不允许为了聚合而改变 GAT 前向。

#### 0.7.3 MLE

所有 $S_i$ padding 到 family 固定 power-of-two shape，使用第 0.2 节 MLE。实例维是 FAC4DNN 新增的聚合维，其随机点记为 $w_i$。

#### 0.7.4 Canonical stack、operation view 与 FAC4DNN re-indexing

原 zkDL 对同一 tensor family 的不同子序列/排列使用公开 re-index mapping 和 Sumcheck，把 operation-view claims 归约回 canonical stacked tensor。本文沿用这一关系。设 canonical stack：

$$
S=(T_0,T_1,\ldots,T_{N-1}),
$$

operation view：

$$
S_k=(T_{j_{k,0}},T_{j_{k,1}},\ldots,T_{j_{k,N_k-1}}),
$$

其中 $j_{k,q}$ 由 `OperationFamilyRegistry` 公开确定。定义：

$$
p_k(i,q)=
\begin{cases}
1,&i=j_{k,q},\\
0,&\text{otherwise}.
\end{cases}
$$

对 claims $\widetilde S_k(u_k,u)$ 和 verifier 随机系数 $\rho_k$，证明：

$$
\sum_k\rho_k\widetilde S_k(u_k,u)
=
\sum_{i=0}^{N-1}
\left(
\sum_k\sum_{q=0}^{N_k-1}
\rho_k\widetilde\beta(u_k,q)p_k(i,q)
\right)
\widetilde S(i,u).
$$

对 $i$ 执行 Sumcheck 后，operation-view claims 被随机组合并绑定回 canonical committed $S$ 的 terminal MLE claim。这里 Equation (30) 同时承担**re-indexing 与同一 canonical tensor family 多个 terminal view claims 的随机压缩**，因此不能仅以“mapping 是 identity”为理由删除该步骤。

对某一 operation family，只有同时满足以下条件时，Line 8 / Equation (30) 才允许退化为无消息的恒等绑定：

1. Line 7 对该 canonical tensor family 只产生一个 terminal claim；
2. 该 claim 已经按相同 MLE variable order、相同 padding 与相同 evaluation point 直接等于 canonical $\widetilde S(u)$；
3. 不存在需要通过 $\rho_k$ 合并的第二个 operation-view claim。

若存在多个 terminal claims，即使每个 view 的 index mapping 都是 identity，也必须执行 Equation (30) 的随机组合/归约；若存在 non-identity view，则更必须执行完整 re-index Sumcheck。只有在使用一个已证明等价、且保持同一 commitment binding 的 multi-point batching relation 时，才允许替代该步骤。

任何 operation view 的 packed/reordered representation 都必须通过上述 FAC4DNN re-index Sumcheck 与 canonical family 绑定；该绑定关系本身属于证明 statement。

#### 0.7.5 GAT compatibility lookup

原 zkDL 的 zkReLU 只针对 ReLU/反向传播，不能直接承担 GAT 的 LeakyReLU、ELU、ExpMap。主实验三张函数表在 `quant_cfg_id` 认证的整数输入区间内均为 deterministic map，因此本文使用只依赖 MLE/Hyrax/Sumcheck 的 indexed-public-table relation，并对同一函数 family 的多条 query 以 verifier 随机权重进行联合压缩。

对函数表：

$$
T_f=\{(x_t,y_t)\}_{t=0}^{m_f-1},
$$

令认证输入区间为连续整数：

$$
x\in[x_f^{min},x_f^{max}],
\qquad
m_f=x_f^{max}-x_f^{min}+1.
$$

按 canonical 输入升序把输出写成公开数组 $T_{f,y}[0],\ldots,T_{f,y}[m_f-1]$，padding 到：

$$
n_f=2^{d_f},
\qquad
d_f=\lceil\log_2m_f\rceil.
$$

公开 table-row selector $Q_f^{table}[u]$ 在 $u<m_f$ 时为 $1$，padding row 为 $0$。

对第 $q$ 个 private query $(x_q,y_q)$ 定义：

$$
idx_q=x_q-x_f^{min},
$$

以及 private bits：

$$
b_{q,0},\ldots,b_{q,d_f-1},
\qquad
b_{q,j}(b_{q,j}-1)=0,
$$

$$
idx_q=\sum_{j=0}^{d_f-1}2^jb_{q,j}.
$$

`integer_bound_ledger` 必须登记 $idx_q$、$x_q-x_f^{min}$、bit reconstruction 及相关乘积的 no-wrap 界。

对 $u\in\{0,1\}^{d_f}$ 定义：

$$
\chi_b(q,u)
=
\prod_{j=0}^{d_f-1}
\left(
u_jb_{q,j}+(1-u_j)(1-b_{q,j})
\right).
$$

对单个有效 query，目标关系为：

$$
\sum_u\chi_b(q,u)Q_f^{table}(u)=1,
$$

$$
y_q=\sum_u\chi_b(q,u)T_{f,y}(u).
$$

第一式拒绝 padding index；第二式把 $y_q$ 唯一绑定到第 $idx_q$ 行。

但正式 GAT 中同一函数 family 有大量 query，**不得逐 query 各生成一份独立 proof，也不得只检查一个未经随机压缩的总和**。设本 family 有 $Q_f$ 个有效 query，padding 到：

$$
n_q=2^{d_q},
$$

公开 query selector：

$$
Q_f^{qry}(q)=
\begin{cases}
1,&q<Q_f,\\
0,&q\ge Q_f.
\end{cases}
$$

padding query 的 $x,y,idx,b$ 均按 `LookupProfile` 固定为 canonical zero/padding representation。

在相关 query/bit tensor commitments 已吸入 transcript 后，验证者通过 Fiat--Shamir 得到：

$$
\tau_f\in\mathbb F^{d_q},
$$

并令：

$$
\beta_f(q)=\widetilde{eq}(\tau_f,q).
$$

同时为 bit-index dimension 生成独立 challenge $\omega_f$；若 $d_f$ 不是 2 的幂，则 bit dimension 也按 registry padding，并使用公开 `Q_bit` 排除 padding bits。

首先随机压缩 index 与 boolean relation：

$$
\sum_q
\beta_f(q)Q_f^{qry}(q)
\left(
idx_q-x_q+x_f^{min}
\right)
=0,
$$

$$
\sum_q
\beta_f(q)Q_f^{qry}(q)
\left(
idx_q-\sum_j2^jb_{q,j}
\right)
=0.
$$

boolean relation 使用 query/bit 两维随机权重：

$$
\sum_{q,j}
\beta_f(q)\,
\widetilde{eq}(\omega_f,j)\,
Q_f^{qry}(q)Q_{bit}(j)\,
b_{q,j}(b_{q,j}-1)
=0.
$$

随后对 query-index 与 table-index 两个维度执行 batched Sumcheck：

$$
\sum_{q,u}
\beta_f(q)
Q_f^{qry}(q)
\chi_b(q,u)
Q_f^{table}(u)
=
\sum_q
\beta_f(q)Q_f^{qry}(q),
$$

以及：

$$
\sum_{q,u}
\beta_f(q)
Q_f^{qry}(q)
\chi_b(q,u)
T_{f,y}(u)
=
\sum_q
\beta_f(q)Q_f^{qry}(q)y_q.
$$

这样任何一个 query 的非零 residual 都只能以 Schwartz--Zippel 概率逃过随机 query compression，而不能通过不同 query 之间的确定性抵消隐藏错误。

$\chi_b(q,u)$ 的乘积 degree、query/bit padding shape、$Q_f$、$n_q$、$d_f$、公开 table identity 与 Sumcheck round degree bound 必须由公共 relation 固定。若采用 multiplication tree 或其他等价 MLE circuit 计算 $\chi_b$，必须证明其保持同一 lookup relation。

所有 $x,y,idx,b$ tensor 继续通过第 0.7.4 节 FAC re-index Sumcheck 绑定回 canonical GAT trace。

### 0.8 Max、range、`SignedRoundDiv` 与 padding

对 max 使用 $\Delta^+$ 与 $s_{max}$；对非负整数/range 使用 bit 或 limb tensor。若：

$$
V=\sum_{h=0}^{b-1}2^hb_h,
$$

则：

$$
b_h(b_h-1)=0.
$$

`SignedRoundDiv` 的 $\sigma,z,a,a^{-1},d^{-1},q,r,\delta$ 按 relation family 堆叠，并通过 arithmetic Sumcheck + range tensor 证明。

padding 只发生在 tensor/MLE 物理形状层；所有 padding value 固定为 0，且不能进入真实 group max、sum、lookup query 或公开 $Y$。

### 0.9 证明 relation 与公共 statement

逻辑公共输入固定为：

$$
Public_{zkDL}=(M_{pub},Topo,T_H,StaticTables,VK_{model}^{zkDL},Y),
$$

私有见证为：

$$
Witness_{zkDL}
=
(\mathcal W_{GAT},Trace_{GAT},Aux_{lookup},Aux_{max},Aux_{div},\{S_i\},R_{model},R_{dyn}).
$$

其中 $\{S_i\}$ 是 canonical tensor-family stacks；$R_{model}$ 与 $R_{dyn}$ 分别是长期模型 commitments 和本次动态 commitments 的私有随机性。独立 commitments 必须使用独立 randomness。

本协议是 **zkDL-style GAT inference adaptation**：它保留原论文的 MLE、Sumcheck、FAC4DNN、Pedersen/Hyrax commitment 和 proof-of-evaluation 结构，但证明对象从训练关系改为第 0 节定义的完整 GAT inference。

原 zkDL 论文的 verifier threat model 是 semi-honest；其零知识定理要求 zero-knowledge Pedersen commitment 与 zero-knowledge Sumcheck variant。本文只有在这些前提实际成立时才继承相同的 privacy 结论，不把普通 Sumcheck 或只带 hiding commitment 的变体写成完整零知识。

原论文使用 BLS12-381 并报告 128-bit security；本对比协议保持该原生安全参数，而不强行与其他 backend 的曲线参数对齐。

## 1. 参数生成

### 1.1 输入

输入安全参数、BLS12-381/native backend 参数、GAT architecture、公开 graph shape、`quant_cfg_id`、`static_tables_id`、model tensor shapes、tensor/operation family policy 与 `integer_bound_ledger`。

### 1.2 Commitment 参数

生成 Pedersen/Hyrax-style commitment 参数：

$$
(PP_{com},VP_{com}).
$$

commitment/group 参数固定为原论文对应的 BLS12-381 安全参数。

### 1.3 模型注册

模型参数按隐藏层/头和输出头顺序组成 model tensor families。对每个长期 model family 独立采样：

$$
r_i^{model}\leftarrow\mathbb F,
$$

并生成：

$$
com_i^{model}=Com(S_i^{model};r_i^{model}),
\qquad
VK_{model}^{zkDL}=\{com_i^{model}\}_i.
$$

公开注册对象只包含 $VK_{model}^{zkDL}$ 及其认证 metadata；prover 私下保留：

$$
R_{model}=\{r_i^{model}\}_i,
$$

以便第 3.12 节对长期 model commitments 生成原 Protocol 1 Line 11 的 proof of evaluation。$R_{model}$ 不进入 public statement，也不因不同 inference proof 而重新采样。

若模型 tensor 与动态 tensor 在同一 logical operation view 中出现，模型部分仍必须与认证的长期 model commitment/reference 保持同一 identity；prover 不能在保持同一 `model_param_id` 的情况下替换模型。

### 1.3.1 模型参数上界与注册

模型注册阶段对 canonical $\mathcal W_{GAT}$ 检查：

$$
|W|\le B_W,\qquad
|a_{src}|\le B_{a,src},\qquad
|a_{dst}|\le B_{a,dst},\qquad
|b|\le B_b,
$$

并把这些 bounds 与同一组 $VK_{model}^{zkDL}$ commitments、模型结构、量化配置和标量域共同认证。inference proof 直接复用该注册边界；任何未与当前长期 commitment 共同认证的模型 bound 都无效。

### 1.4 公共 tensor 注册

注册公开 $T_H$、topology、$T_{LReLU},T_{ELU},T_{exp}$、range relation 和完整 $Y$ 的 canonical encoding。它们可有公开 deterministic digest/commitment，但不使用“隐藏 public input”的随机承诺语义。

### 1.5 TensorFamilyRegistry

为每个 $\mathcal T_i$ 固定：

$$
(family\_id,instance\_order,logical\_shape,padded\_shape,stack\_axis,MLE\_order,visibility).
$$

### 1.6 OperationFamilyRegistry

为每个 $\mathcal A_j$ 固定：operation polynomial、输入/输出 tensor family、实例 mapping、padding selector、Sumcheck variable order、terminal claims 与 verifier local-public claims。

### 1.7 FAC4DNN plan

原论文 FAC4DNN 将多个相似 operation 放到额外实例维并用随机 $w$ 压缩。本文对 hidden heads、hidden layers、output heads 中兼容 operation 采用同样规则；不兼容 shape 分开 family，避免为大 operation 强行 padding 到不公平规模。

### 1.8 参数生成输出

$$
SetupCtx_{zkDL}
=
(PP_{com},VP_{com},VK_{model}^{zkDL},ModelBoundRef,
TensorFamilyRegistry,OperationFamilyRegistry,LookupProfile,
StackBindingProfile).
$$

## 2. 见证生成

### 2.1 公开特征与 topology

从公开：

$$
H^{(1)}=T_H[I]
$$

生成公开 route/group selectors。公开数据不进入 hiding witness family。

### 2.2 Canonical GAT trace

按第 0.6 节逐层生成完整 integer trace，特别保留：

$$
H_{agg,pre},H_{agg,num},H_{ELU,in},
$$

$$
Y^{pre},Y_{agg,num},Y^q,\widetilde Y^{pre},Y^{sum}.
$$

hard-coded scale 或浮点 Softmax 不属于本文 formal witness relation。

模型注册与 inference proof 的 randomness 生命周期必须区分：长期 `VK_model` 的 $R_{model}$ 在模型注册时采样并随该注册对象固定；每次 inference 的 dynamic private tensor-family commitments 使用新的独立 $R_{dyn}$；zero-knowledge Sumcheck 所需的 masking randomness 每个 proof 重新采样。不得跨 proof 重用 dynamic commitment 或 Sumcheck masking randomness。

### 2.3 Non-arithmetic auxiliary

indexed-public-table relation 需要以下辅助见证：

- LeakyReLU/ELU/ExpMap query 的 $idx$、bit tensor 与 public-table MLE evaluation auxiliary；
- $s_{max}$ 与 $\Delta^+$ range bits；
- `SignedRoundDiv` auxiliary 与 range bits/limbs。

这些对象同样进入 tensor families，并通过第 0.7.4 节 FAC re-index relation 绑定回 canonical GAT trace。


### 2.4 Tensor-family stacks

对每个 private family 构造：

$$
S_i=Stack(T_{i,0},\ldots,T_{i,n_i-1}),
$$

并按 registry 完成 zero padding。stack order 是协议语义的一部分。

### 2.5 Commitments

对本次 inference 的 dynamic private $S_i$ 使用 $R_{dyn}$ 中各自独立的 randomness：

$$
com_i^{dyn}\leftarrow Commit(S_i;r_i^{dyn}).
$$

模型长期 family 不生成第二份可替换的 model commitment，而是引用已认证 $VK_{model}^{zkDL}$；其 terminal canonical MLE claim 最终通过第 3.12 节的 proof of evaluation 绑定到同一长期 commitment。

### 2.6 派生 operation views

operation family 按公开 $j_{k,q}$ 从 canonical $S_i$ 定义相应 operation view，并固定第 0.7.4 节的公开 re-index mapping。此阶段只确定 view 的值、shape 与 mapping，不生成任何依赖 verifier / Fiat--Shamir challenge 的 Sumcheck prover message；这些消息只能在第 3 章 commitment 已固定后按协议轮次生成。

## 3. 证明生成

### 3.1 由原 zkDL Protocol 1 适配的总体协议

原论文 Protocol 1 对每个 tensor family 的 stack 先做 commitment，再对每个 operation family 执行 aggregated Sumcheck，最后对被归约出的 stack MLE evaluations 执行 proof of evaluation。GAT inference 没有 training-step 维，因此本文把聚合实例维定义为 compatible layer/head/operation instances。

协议输入为 $SetupCtx_{zkDL},Public_{zkDL},Witness_{zkDL}$。

### 3.2 Tensor-family commitment phase

对动态 private tensor families 按 registry 顺序发送：

$$
Com_{tensor}=(com_0,com_1,\ldots,com_{N_T-1}).
$$

长期 model commitments 通过 `VK_model` 引用；公开 tensor 不发送 hiding commitment。

commitments 固定以后才生成任何依赖 stack 的 random evaluation point。

### 3.3 Operation-family Sumcheck phase

对每个 $\mathcal A_j$，定义其聚合 relation $f_j$ 并执行 Sumcheck。family 顺序由 registry 固定。

以 $R$ 个矩阵乘法实例 $Z^{(i)}=X^{(i)}Y^{(i)}$ 为例：

$$
\widetilde{\mathcal Z}(w,u_1,u_3)
=
\sum_{i=0}^{R-1}\widetilde\beta(w,i)
\sum_j\widetilde{\mathcal X}(i,u_1,j)\widetilde{\mathcal Y}(i,j,u_3).
$$

先对实例维 $i$ 做 Sumcheck，压缩到随机点 $v$；随后继续单个矩阵乘法 Sumcheck。hidden/output projection 与 attention dot 只能在 shape compatible 时合并。

### 3.4 Public sparse route relation

route mapping 全部由公开 topology 确定。对 routed edge tensor $R_{src}(H')$、$R_{dst}(E)$，Sumcheck 的 terminal public route polynomial evaluation 由 verifier 本地计算。

关系至少覆盖全部正式 node-to-edge broadcast：

$$
H_{src}^{\prime edge}[k,j]=H'[src(k),j],
$$

$$
E_{src}^{edge}[k]=E_{src}[src(k)],
\qquad
E_{dst}^{edge}[k]=E_{dst}[dst(k)],
$$

$$
M^{edge}[k]=M[dst(k)],
$$

$$
Sum^{edge}[k]=Sum[dst(k)],
\qquad
inv^{edge}[k]=inv[dst(k)].
$$

公开 route operator 由 verifier 根据 $src/dst$ 本地构造；private routed tensor 的 terminal claim 必须通过 public operator 的 MLE/Sumcheck relation 绑定到 canonical node tensor，不能依赖证明关系之外的未约束 gather。

### 3.5 Group aggregation relation

分别证明 Hadamard：

$$
P_U[k,j]=U_kH_{src}^{\prime edge}[k,j],
$$

$$
P_\alpha[k,j]
=
\alpha_kH_{src}^{\prime edge}[k,j],
$$

再由公开 group mapping 证明：

$$
H_{agg,num,i,j}=\sum_{dst(k)=i}P_U[k,j],
$$

$$
H_{agg,pre,i,j}=\sum_{dst(k)=i}P_\alpha[k,j].
$$

输出层同理，并且输出层的 $UY'$ 与 $\alpha Y'$ 只能使用已由 public-route relation 绑定的 $Y_{src}^{\prime edge}$。

### 3.6 Function-table compatibility Sumcheck

正式 `zkDL_gat_indexed_public_table_v1` 与 `zkDL_gat_indexed_public_table_batch_v1` 共同定义 lookup proof。

对每个 function-query family $f$，把 canonical query tensor padding 到 $n_q=2^{d_q}$，提交/绑定：

$$
x,\quad y,\quad idx,\quad \{b_j\}_{j=0}^{d_f-1}.
$$

在这些 commitments 已进入 transcript 后生成独立：

$$
\tau_f,\omega_f.
$$

执行四类 batched relation：

1. index shift：
   $$
   \sum_q\beta_f(q)Q_f^{qry}(q)
   (idx_q-x_q+x_f^{min})=0;
   $$

2. bit reconstruction：
   $$
   \sum_q\beta_f(q)Q_f^{qry}(q)
   \left(idx_q-\sum_j2^jb_{q,j}\right)=0;
   $$

3. query/bit boolean compression：
   $$
   \sum_{q,j}
   \beta_f(q)\widetilde{eq}(\omega_f,j)
   Q_f^{qry}(q)Q_{bit}(j)
   b_{q,j}(b_{q,j}-1)=0;
   $$

4. query/table indexed membership：
   $$
   \sum_{q,u}
   \beta_f(q)Q_f^{qry}(q)
   \chi_b(q,u)Q_f^{table}(u)
   =
   \sum_q\beta_f(q)Q_f^{qry}(q),
   $$
   $$
   \sum_{q,u}
   \beta_f(q)Q_f^{qry}(q)
   \chi_b(q,u)T_{f,y}(u)
   =
   \sum_q\beta_f(q)Q_f^{qry}(q)y_q.
   $$

这些 relation 使用 arithmetic Sumcheck 归约到 operation-view 中的 private tensor claims 与公开 table MLE evaluations。后续是否执行 Protocol 1 Line 8 / Equation (30) 统一服从第 3.11 节触发规则：存在 non-identity view，或同一 canonical tensor family 产生多个 terminal view claims 时，必须先执行 FAC re-index/compression Sumcheck 归约到 canonical tensor-family claims；只有“恰好一个且已经是同点 canonical claim”的退化情形可以直接进入第 3.12 节 Protocol 1 Line 11 的 Hyrax proof of evaluation。

三个 exact table family：

$$
T_{LReLU},\qquad T_{ELU},\qquad T_{exp}
$$

分别拥有独立 `function_query_family_id`、query count/shape、$\tau_f,\omega_f$ challenge labels 和 degree bound。不同 function family 不共享未标记 challenge。

FAC4DNN re-index relation随后把 operation-view 中的 $x,y,idx,b$ claims 绑定回 canonical tensor families，防止 prover 为 lookup proof 构造与真实 GAT trace 不同的 query vector。

### 3.7 Exact max

证明：

$$
\Delta_k^+=M_{dst(k)}-Z_k,
$$

$\Delta^+$ 的 bit/limb decomposition，以及：

$$
s_{max,k}(s_{max,k}-1)=0,
$$

$$
\sum_{dst(k)=i}s_{max,k}=1,
$$

$$
s_{max,k}\Delta_k^+=0.
$$

### 3.8 Inverse、alpha 与双聚合 consistency

Sumcheck 证明：

$$
Sum\cdot inv=1,
$$

$$
\alpha=U\cdot inv_{dst},
$$

$$
H_{agg,num}=Sum\cdot H_{agg,pre},
$$

$$
Y_{agg,num}=Sum\cdot Y^{pre}.
$$

### 3.9 `SignedRoundDiv`

对每个 registered division family，arithmetic Sumcheck 证明第 0.4.4 节 equality；bit/limb family 证明 $a,q,r,\delta$ 非负与公开位宽。dynamic $d$ 必须由 committed $Sum$/scale relation绑定。

### 3.10 Operation-view terminal claims

第 3.3--3.9 节中每个 operation-family Sumcheck 对应原 zkDL Protocol 1 Line 7。该 Sumcheck 结束时产生的是本 operation family 实际读取的 **operation-view MLE claims**，例如：

$$
\widetilde S_k(u_k,u).
$$

这些 claims 此时仍属于 operation view，尚不能直接视为 canonical committed $S_i$ 的 evaluation claim。本节只收集同一 operation family 的 terminal view claims，不引入额外“tensor-claim compression”证明阶段。

对每个 canonical tensor family，若当前 Line 7 产生多个 terminal view claims，则即使各 view 的 mapping 均为 identity，也必须把这些 claims 送入第 3.11 节 Line 8 / Equation (30) 的随机组合；只有“恰好一个且已经是同点 canonical claim”的退化情形可以直接进入第 3.12 节。

### 3.11 FAC4DNN re-indexing proof

本节对应原 zkDL Protocol 1 **Line 8 / Equation (30)**。对当前 operation family 中需要 re-index 或需要随机合并的 terminal views，按照第 0.7.4 节公开 mapping 与 verifier 随机系数 $\rho_k$ 执行：

$$
\sum_k\rho_k\widetilde S_k(u_k,u)
=
\sum_{i=0}^{N-1}
\left(
\sum_k\sum_{q=0}^{N_k-1}
\rho_k\widetilde\beta(u_k,q)p_k(i,q)
\right)
\widetilde S(i,u).
$$

该 Sumcheck 完成后，operation-view claims 才被归约为 canonical tensor-family MLE claims，并进入第 3.12 节 proof of evaluation。

Line 8 的触发规则固定为：

- 只要存在 non-identity view，必须执行；
- 只要同一 canonical tensor family 存在两个或以上 terminal view claims，也必须执行，即使 mapping 全部是 identity；
- 只有一个 terminal claim、且它已经按同一 MLE layout 和同一 evaluation point等于 canonical claim 时，Equation (30) 才退化为无消息恒等绑定。

Line 8 / Equation (30) 的 re-index/compression Sumcheck 必须直接把 operation-view terminal claims 归约到同一 canonical tensor family 的 terminal claim；不能用脱离 canonical commitment 的独立摘要替代。

若不同 operation families 最终对同一 canonical $S_i$ 产生多个 evaluation points，则这些 points 均属于 Line 11 待验证 claims；除非 Hyrax evaluation relation 本身定义了保持同一 commitment binding 的 multi-point batching，否则本文不额外假定未定义的跨-family claim compression。

### 3.12 Proof of evaluation

与原论文 Protocol 1 Line 11 对应，对每个由 zero-knowledge Sumcheck / FAC re-index relation 归约得到的 private tensor-family terminal claim：

$$
(S_i,u_i,v_i),\qquad v_i=\widetilde S_i(u_i),
$$

生成原生 Hyrax evaluation proof，并验证：

$$
VerifyEval(VP_{com},com_i,u_i,v_i,\pi_i^{eval})=1.
$$

这里的 evaluation proof 只负责把前述 terminal claim 绑定回同一个 hiding commitment；它不是一个独立于 Sumcheck 的额外 GAT 计算 relation。本文对该步骤的零知识主张直接采用原 zkDL Theorem C.2 对 Protocol 1 Line 7、8、11 组合的安全边界，不另行假定新的 evaluation-opening primitive。

公开 $T_H$, topology, tables 与 $Y$ 的 MLE 值由 verifier 本地计算。

### 3.13 非交互化边界

原 zkDL Protocol 1 以交互式 Sumcheck 描述。若对比实验需要把完整 transcript 固化为可离线验证的单个 proof，可以对其 public-coin challenges 使用 Fiat--Shamir transform；该步骤属于本文 comparison adaptation，而不是原论文 Protocol 1 的文字定义。

Fiat--Shamir 必须使用密码学安全、domain-separated 的 hash-to-field，并在每一轮挑战产生前吸收当前公共 statement、相关 commitments 和此前全部 prover messages。不同 operation family、不同 round 与不同 lookup batch 必须使用不同标签。

原 zkDL Theorem C.2 直接证明的是交互协议的零知识。因此，Fiat--Shamir 后对象可以据此主张非交互零知识，仅当所选 zero-knowledge Sumcheck / commitment-evaluation 组合具有相应 random-oracle-model 安全论证；否则只主张其非交互 soundness，并把 privacy 结论限定为原交互协议。

### 3.14 最终证明对象

抽象地，zkDL-style GAT proof 由以下部分组成：

$$
\pi_{GAT}^{zkDL}
=
(Com_{dyn},\Pi_{op},\Pi_{reindex},\Pi_{eval}),
$$

其中：

- $Com_{dyn}$：本次 inference 的动态 private tensor-family commitments；
- $\Pi_{op}$：各 operation-family 的 FAC4DNN/Sumcheck proofs，包括 lookup compatibility、max/range、稀疏 group relation、`SignedRoundDiv` 和最终输出关系；
- $\Pi_{reindex}$：需要时由 Equation (30) 产生的 re-index/compression proofs；
- $\Pi_{eval}$：把所有 private terminal MLE claims 绑定回长期模型 commitments 或本次动态 commitments 的 Hyrax evaluation proofs。

公共 graph/features、静态表与完整 $Y$ 不作为 private commitments 发送；验证者直接由公共 statement 计算其所需 MLE evaluations。若采用第 3.13 节的 Fiat--Shamir 编译，则上述交互消息按同一逻辑顺序固化到 proof transcript。

## 4. 验证

### 4.1 Verifier context

验证者持有 $PP_{com},VP_{com}$、$VK_{model}^{zkDL}$ 及其认证模型上界、公共 tensor/operation relation、公开 graph/features/tables 与完整 $Y$。

### 4.2 公共 relation 与注册对象

检查公开 GAT 结构、graph/features、量化与静态表、长期 $VK_{model}^{zkDL}$、tensor/operation-family 定义以及完整 $Y$ 是否与本次 verifier context 一致。

### 4.3 Public reconstruction

在执行任何 public reconstruction、MLE evaluation、Sumcheck 或 Hyrax verification 之前，验证者必须检查第 0.1 节与第 0.3 节定义的全部模型维度和公共 instance 合法性条件；任一条件不满足必须立即拒绝。

从公开 $T_H,I,src,dst,node\_ptr,edge\_ptr$ 重建 $H^{(1)}$、route/group mappings、selectors 与所有 public MLE evaluator。

### 4.4 Fiat--Shamir replay

仅当采用第 3.13 节非交互化时，验证者按原交互协议的消息顺序重放所有 hash-to-field challenges；保持原交互协议时则直接使用 verifier 采样的随机挑战。

### 4.5 Operation-family Sumcheck

按 registry 验证每个 aggregated operation Sumcheck，包括 matmul、route、lookup compatibility、max/range、group aggregation、inverse/alpha、consistency、`SignedRoundDiv`、concat、bias 与 final average。

### 4.6 Canonical/derived stack binding

按第 3.11 节触发规则验证 `BindingBlock` 中的 `fac_reindex_sumcheck_v1`：

- 存在任一 non-identity view 时，`BindingBlock` 必须包含对应 Line 8 / Equation (30) proof；
- 同一 canonical tensor family 有多个 terminal view claims 时，即使 mapping 全部为 identity，仍必须包含随机组合/归约 proof；
- 只有恰好一个 terminal claim、且已经是同点 canonical claim 时，对应 binding 才允许为空。

所有 re-index/compression terminal claims 都必须落到 canonical committed tensor family。不能仅根据 “view 是 identity” 判断 `BindingBlock` 应为空。

### 4.7 Terminal evaluation

对 private canonical terminal claims 验证 Hyrax evaluation proof。长期 model-family 的 proof of evaluation 由 prover 使用私下保留的 $R_{model}$ 生成，dynamic family 的 proof of evaluation 由 prover 使用本次 $R_{dyn}$ 生成；验证者只接收相应 commitment、terminal point/value 与 evaluation proof，并分别相对于认证 $VK_{model}^{zkDL}$ 或本次 dynamic commitment 验证，验证者不得接收、恢复或使用 $R_{model},R_{dyn}$。对 public tensor terminal value，verifier 本地求值后直接参与 Sumcheck terminal check。

### 4.8 完整输出

最终 relation 必须证明：

$$
Y=SignedRoundDiv(Y^{sum},K_{out}).
$$

完整 $Y\in\mathbb Z^{N\times C}$ 作为 public output 随 canonical proof envelope 提供一次。验证者从 canonical $Y$ padding/encoding 本地构造：

$$
\widetilde Y_{pub}.
$$

在 dynamic tensor commitments 已固定后，由 transcript 生成：

$$
r_Y\in\mathbb F^{\log n_N+\log n_C}.
$$

验证者本地计算：

$$
v_Y=\widetilde Y_{pub}(r_Y).
$$

最终 output-average operation family 必须把 output-side relation 归约到 verifier 本地计算的同一个 public terminal claim $v_Y$。其 operation-view terminal claims 同样服从第 3.11 节 Line 8 / Equation (30) 触发规则：存在 non-identity view 或同一 canonical family 有多个 terminal claims 时必须完成 re-index/compression binding；只有单一同点 canonical claim 才允许退化为空步骤。接受路径最终绑定的是完整公开 $Y$ 的随机 MLE evaluation，而不是 prover 自己提供的第二份输出向量。


### 4.9 接受条件

仅当 metadata、全部 mandatory families、Sumcheck、stack binding、Hyrax evaluations、range/div relations 和 public $Y$ 全部通过时返回 $Accept$。

## 5. 完备性、可靠性与零知识边界

### 5.1 适配范围

本文不是原 zkDL training relation 的复刻，而是使用其 proof architecture 证明 GAT inference。zkReLU 的原始 ReLU/backprop relation不被误用于 LeakyReLU、ELU 或 ExpMap。

### 5.2 完备性

若所有 tensor-family stacks来自正确 GAT trace，operation-family equations 成立，且 evaluation proofs 正确，则诚实 proof 被接受。

### 5.3 可靠性

可靠性依赖 Pedersen/Hyrax commitment binding、Sumcheck soundness、FAC4DNN aggregated relation 的随机压缩、public route/table identity 与 terminal evaluation proofs。operation stack/view 必须与 committed tensor families绑定，不能只证明一个脱离 canonical GAT trace 的独立 stack。

### 5.4 Zero knowledge

原 zkDL Theorem C.2 的零知识结论依赖两个明确前提：

1. private tensors 使用 zero-knowledge Pedersen/Hyrax-style hiding commitments；
2. witness-dependent Sumcheck 使用 zero-knowledge variant。

该定理把 Protocol 1 Line 7、Line 8 与 Line 11 的 proof of evaluation 作为同一组合协议分析。因此本文的 GAT inference adaptation只有在对应 operation-family Sumchecks、re-index Sumchecks 和 terminal evaluation proofs 都满足这些前提时，才称为 zero-knowledge zkDL-style proof。

原论文的 verifier model 是 semi-honest；本文不把该结论自动升级为 arbitrary malicious-verifier zero knowledge。长期模型 commitment 的复用属于本 comparison 的注册方式，不能据此额外声称原论文已经给出新的多会话隐私定理。

当同一长期 $VK_{model}^{zkDL}$ 所引用的 model-family commitments 被多个 inference proofs 重复使用时，严格 `model_and_trace_hidden` profile 只有在所选 hiding commitment、zero-knowledge Sumcheck 与 terminal evaluation-proof composition 对该 repeated registered-model statement 具有 multi-session zero-knowledge 论证时才成立。每次 inference proof 所需的 masking / Sumcheck randomness 必须 fresh，且不得由长期 commitment randomness、旧 transcript 或可预测 label 派生。若不能给出该 multi-session 组合论证，则 reusable-registry profile 必须标记为 `unsupported_privacy_profile`；原论文单次交互 Protocol 1 的零知识结论只能作为明确标注的 paper-native secondary privacy profile，不能进入要求严格 privacy parity 的主比较。

若采用第 3.13 节 Fiat--Shamir 编译，零知识结论还需要相应 ROM 编译条件；没有该论证时，仍可使用非交互 transcript 做 soundness comparison，但必须把 privacy claim 限定为原交互协议。

### 5.5 GAT function-table adaptation

`T_{LReLU},T_{ELU},T_{exp}` 的 indexed public-table relation 是本文为 GAT 增加的 compatibility relation，不是原 zkDL 论文中的 zkReLU。它只能使用 MLE/Sumcheck/FAC4DNN/Hyrax 体系内的原语，并且必须把 lookup query 绑定回同一 canonical GAT trace。

因此本文不会把 LeakyReLU、ELU 或 ExpMap 误写成原论文 zkReLU；zkReLU 只作为 zkDL 处理非算术关系和组织 FAC4DNN 的设计来源。

### 5.6 Security parameters 与 threat-model 边界

原论文采用 BLS12-381 并声明 128-bit security。本 baseline 保持该真实安全参数，不能为了与 BN254/其他 backend 对齐而把不同曲线参数写成同一安全位数。

隐私威胁模型固定为 semi-honest verifier。

与原论文 threat model 一致。soundness 仍要求错误 GAT statement 不能被作弊 prover 通过 Sumcheck/commitment/evaluation proof 接受；privacy claim 则只在登记的 verifier model 与实际 masking/zero-knowledge Sumcheck 前提下成立。

本文不声明 arbitrary malicious-verifier zero knowledge；若要得到该结论，必须另给独立安全证明。

### 5.7 当前 CPU 执行实现边界

当前 baseline 的初始 CPU G1 commitment MSM 使用固定大小分块的 BLST Pippenger；预计算 affine basis 在同一证明或验证上下文内只读共享，有序归并保持相同的群元素结果。分块回执绑定 segment、family 与 sorted-lookup 列，验证端仍检查每一项和顺序。Hyrax prover 保留协议原有的展开 projective G 向量和逐轮 fold 方程；每轮在同一有界持久工作池上并行计算各 G 点对的两次 BLST 标量乘法与一次加法，并将 `a/b` fold 结果写回固定索引。内积输入先完整检查规范标量，再按固定连续块在 BLST Montgomery 域中累加，并按块序归并；零乘数只在完整输入检查后跳过乘法。独立验证端仍由 proof transcript 重建折叠系数和最终 G 项。上述实现没有采用展开 indexed-basis fold 作为正式 prover 路径，也不改变挑战、随机数、transcript、wire 编码、证明范围或验证条件。CPU 工作池并行度有界，调用者工作区计入资源上界。sorted-prefix 索引按规范化后的全局前缀边界处理最后一个分块，不改变 lookup 语义。

Canonical operation descriptors 可在单次 relation-plan 构造期间共享内容完全相同的 selector index/mask backing；每个 operation 的逻辑引用、维度、约束和规范化序列化仍保持原值。mask 与 index-map 的共享采用写时复制，因此 mutation 不会穿过别名修改其他 operation。Proof 中仍序列化完整展开的 descriptor arrays；独立验证的接收侧 descriptor 仍按未共享树计入内存上界。上述实现优化不改变 zkDL proof protocol、Fiat--Shamir/transcript 顺序或接受条件。

新增 `gat_runner resource-plan` 只用于静态容量分析：它从已认证配置形状、checkpoint tensor-index、规范化图边和静态表描述符计算带 `ANALYSIS_ONLY` 标记的上界，再复用正式执行的 host/disk 计算公式。它不准备数据、不运行正式 admission、不生成或验证 proof，也不产生实验结果；只有正式入口成功完成实际独立验证和结果收集后，才能声称实验完成。
