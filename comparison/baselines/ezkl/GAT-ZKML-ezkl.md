# GAT-ZKML（EZKL 方案）

证明系统采用 **EZKL → ONNX → Halo2 → zkSNARK** 路线，并固定模型语义、图输入语义、量化语义、参数族定义、前向中间对象定义、参数生成、见证生成、证明生成与验证规则。

---

## 0. 符号

### 0.1 模型结构

GAT 的前向固定为一个可参数化的多层结构：

1. 设注意力层总层数为 $L$。其中前 $L-1$ 层是隐藏层，第 $L$ 层是最终输出层。

2. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，记该层注意力头数量为 $K_{hid}^{(\ell)}$，每个头的输出维度为 $d_h^{(\ell)}$。该层每个头完成注意力聚合后逐元素施加 ELU；随后将全部头的 ELU 后输出沿特征维拼接，得到

$$d_{cat}^{(\ell)}=K_{hid}^{(\ell)}\cdot d_h^{(\ell)}.$$ 

3. 第 $\ell$ 个隐藏层输入维度记为 $d_{in}^{(\ell)}$。其中 $d_{in}^{(1)}=d_{in}$，且对任意 $\ell\ge 2$，都有

$$d_{in}^{(\ell)}=d_{cat}^{(\ell-1)}.$$ 

4. 最终输出层的注意力头数量记为 $K_{out}$。输出层不是“线性层替代注意力”，而是由 $K_{out}$ 个输出注意力头组成。每个输出头先完成注意力聚合，再加上该头自己的输出偏置向量；若 $K_{out}>1$，则全部输出头的后偏置输出按头维做算术平均，得到最终 logits。

5. 输出注意力头的激活函数固定为恒等映射，不再施加 ELU；但恒等映射输入是已经加上该输出头偏置后的注意力聚合结果。

6. 对任意多层 GAT，证明者与验证者都必须按层序 $\ell=1,\ldots,L-1$ 逐层实例化隐藏层模板；第 $\ell-1$ 层拼接输出 $H_{cat}^{(\ell-1)}$ 必须作为第 $\ell$ 层的唯一输入表示。

为避免公式中引入过多层指标，除非特别声明，隐藏层模板统一采用如下局部简写：

- 隐藏层模板索引：$\ell\in\{1,2,\ldots,L-1\}$
- 第 $\ell$ 个隐藏层中第 $r$ 个注意力头编号：$r\in\{0,1,\ldots,K_{hid}-1\}$
- 当前隐藏层模板输入维度：$d_{in}:=d_{in}^{(\ell)}$
- 当前隐藏层模板每头输出维度：$d_h:=d_h^{(\ell)}$
- 当前隐藏层模板头数：$K_{hid}:=K_{hid}^{(\ell)}$
- 当前隐藏层模板拼接维：$d_{cat}:=d_{cat}^{(\ell)}$
- 输出类别数：$C$
- 最终输出层第 $s$ 个输出头编号：$s\in\{0,1,\ldots,K_{out}-1\}$

凡正文中为了公式紧凑而省略的 $(\ell)$、$(\ell,r)$、$(out,s)$ 标记，只能在数学推导中省略；在任何实现接口、配置字段、日志字段与 artifact 命名中，都必须恢复完整指标。

### 0.2 有限域、定点量化、Halo2、KZG 与 Fiat–Shamir

记有限域为 $\mathbb F_p$，其中 $p$ 为大素数。所有电路中的整数化运算都在 $\mathbb F_p$ 上进行。

采用定点数量化语义：若域元素 $x\in\mathbb F_p$ 的标准代表满足 $x>\frac{p-1}{2}$，则在实数语义下将其解释为负数 $x-p$。

证明系统采用ezkl 通用 zkML 路线：

1. 先将模型导出为 ONNX 计算图；
2. 再将 ONNX 计算图编译为 Halo2 电路；
3. 再由 Halo2 后端生成 zkSNARK 推理证明；
4. strict route 公开绑定输入、输出与模型权重哈希 $H(W)$；timing route 公开绑定输出、route metadata 与固定模型身份，完整输入作为 witness 参与电路满足性证明；
5. 只有在同一已冻结的 $onnx\_graph\_id$、同一输入形状、同一预处理口径、同一编译设置、同一量化配置与同一 $proof\_route\_id / visibility\_id$ 下，才可复用同一组 $(pk,vk)$ 进行重复推理证明；
6. 多个单样本推理证明可进一步组织为 evaluation attestation。

记 KZG 型 proving setup 的证明键和验证键分别为 $pk$ 与 $vk$。其底层依赖结构化参考字符串与多项式承诺，但本文不再把每个中间多项式对象单独暴露为 proof object，而只把最终电路 satisfiability proof 暴露给验证者。

随机预言机统一记作 $H_{FS}(\cdot)$。对电路内部承诺吸入、挑战派生、folding 与 opening 次序，本文不重新定义 ezkl / Halo2 后端的内部 transcript，而统一以对应 backend 实现为准；本文仅固定外部公开对象 $M_{pub}$、公开输出、$h_W$、$proof\_route\_id$、$visibility\_id$，以及在 strict route 中公开输入的绑定语义。

### 0.3 图、局部子图、多图批处理与排序

- 全局节点总数：$N_{total}$
- 当前证明实例中的总节点数：$N$
- 当前证明实例中的总边数：$E$
- 当前证明实例中的图个数：$G_{batch}$
- 局部子图或 batch 后的节点绝对编号序列：$I=(I_0,I_1,\ldots,I_{N-1})$
- 边源索引序列：$src=(src(0),src(1),\ldots,src(E-1))$
- 边目标索引序列：$dst=(dst(0),dst(1),\ldots,dst(E-1))$
- 批处理中各图的节点前缀指针：$node\_ptr=(\nu_0,\nu_1,\ldots,\nu_{G_{batch}})$
- 批处理中各图的边前缀指针：$edge\_ptr=(\epsilon_0,\epsilon_1,\ldots,\epsilon_{G_{batch}})$

对任意节点索引 $i\in\{0,1,\ldots,N-1\}$，定义其图编号 $node\_gid(i)=g$，其中 $g$ 满足 $\nu_g\le i<\nu_{g+1}$。

对任意边索引 $k\in\{0,1,\ldots,E-1\}$，定义其图编号 $edge\_gid(k)=g$，其中 $g$ 满足 $\epsilon_g\le k<\epsilon_{g+1}$。

固定要求边序列满足以下两级稳定排序：

1. 先按 $edge\_gid(k)$ 非降序排列；
2. 在每个固定图块内部，再按目标节点索引 $dst(k)$ 做确定性稳定排序。

若原始输入不满足这一条件，则在见证生成前必须先完成：

- 图内节点重编号或局部子图抽取；
- 按图分块的稳定排序；
- 每个图块内部按 $dst$ 的稳定排序；
- 必要时显式加入自环。

实现必须严格继承主仓库 cfg 中的 `batching_rule / subgraph_rule / self_loop_rule / edge_sort_rule / chunking_rule`，不得自行定义第二套预处理语义。

当前 comparison/ezkl 正式计时主线必须使用与主仓库一致的全量正式数据集口径；文中的 `subgraph_rule` / 局部子图术语仅表示继承主仓库 cfg 字段语义，不得被实现为 toy、sample、smoke、抽样或降规模替代路线。

### 0.4 静态表与量化尺度

静态表包括：

1. 全局特征表：$T_H\in\mathbb F_p^{N_{total}\times d_{in}}$
2. LeakyReLU 查表：$T_{LReLU}\subseteq \mathbb F_p\times\mathbb F_p$
3. ELU 查表：$T_{ELU}\subseteq \mathbb F_p\times\mathbb F_p$
4. 指数查表：$T_{exp}\subseteq \mathbb F_p\times\mathbb F_p$
5. 范围表：$T_{range}=\{0,1,2,\ldots,2^B-1\}$，其中 $B$ 是范围检查位宽

与 Softmax 链路有关的量化尺度全部视为公开参数：

- 差分反量化尺度：$S_{\Delta}$
- 指数输出尺度：$S_{exp}$
- 逆元见证尺度：$S_{inv}$
- 归一化权重尺度：$S_{\alpha}$
- 隐藏层聚合前尺度：$S_{agg,pre}$
- 隐藏层 ELU 后尺度：$S_{agg}$
- 拼接后表示尺度：$S_{cat}$
- 输出层每个头的聚合前尺度：$S_{out,pre}$
- 输出层每个头的加偏置后尺度：$S_{out,head}$
- 输出层最终多头平均 logits 尺度：$S_{out}$

舍入方式（向最近整数舍入或截断）、乘法后的 rescale 规则也全部是公开参数。

进一步固定输出层的量化与平均语义如下：

1. 对每个输出头 $s$，聚合前输出 $Y^{pre(s)}$ 在与偏置向量 $b^{(out,s)}$ 相加之前，必须已经被公开量化规则对齐到与 $b^{(out,s)}$ 相同的输出头尺度 $S_{out,head}$。若实现中注意力聚合自然产出的尺度是 $S_{out,pre}$，则必须先施加公开的对齐 / 回缩算子：

$$\widetilde Y^{pre(s)} = Rescale_{out}\!\big(Y^{pre(s)}; S_{out,pre}\rightarrow S_{out,head}\big).$$

然后正式定义：

$$Y^{(s)}=\widetilde Y^{pre(s)}+b^{(out,s)}.$$

2. 最终多头平均必须在共同尺度上执行。本文协议固定要求最终平均发生在输出头共同尺度 $S_{out,head}$ 上，并且最终 logits 继续保持这一共同尺度，即 $S_{out}=S_{out,head}$，因而先在域内计算

$$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)},$$

再乘以公开整数 $K_{out}$ 在 $\mathbb F_p$ 中的逆元 $K_{out}^{-1}$，得到最终 logits：

$$Y_{i,c}=K_{out}^{-1}\cdot Y_{i,c}^{sum}.$$

3. 上述 $Rescale_{out}$、舍入方式以及乘以 $K_{out}^{-1}$ 的平均语义，全部必须并入 $quant\_cfg\_id$ 所标识的公开量化配置；证明者与验证者不得各自选择不同实现。

### 0.5 模型参数

#### 0.5.1 隐藏层参数族

对任意隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，以及该层中的每个注意力头 $r\in\{0,1,\ldots,K_{hid}^{(\ell)}-1\}$，定义：

- 投影矩阵 $W^{(\ell,r)}\in\mathbb F_p^{d_{in}^{(\ell)}\times d_h^{(\ell)}}$
- 源方向注意力向量 $a_{src}^{(\ell,r)}\in\mathbb F_p^{d_h^{(\ell)}}$
- 目标方向注意力向量 $a_{dst}^{(\ell,r)}\in\mathbb F_p^{d_h^{(\ell)}}$

#### 0.5.2 输出层参数

对最终输出层中的每个输出注意力头 $s\in\{0,1,\ldots,K_{out}-1\}$，定义：

- 输出投影矩阵 $W^{(out,s)}\in\mathbb F_p^{d_{in}^{(L)}\times C}$
- 输出层源方向注意力向量 $a_{src}^{(out,s)}\in\mathbb F_p^C$
- 输出层目标方向注意力向量 $a_{dst}^{(out,s)}\in\mathbb F_p^C$
- 输出层偏置向量 $b^{(out,s)}\in\mathbb F_p^C$

其中 $d_{in}^{(L)}=d_{cat}^{(L-1)}$。

记全部模型权重集合为

$$W=\Big(\{W^{(\ell,r)},a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}\}_{\ell,r},\{W^{(out,s)},a_{src}^{(out,s)},a_{dst}^{(out,s)},b^{(out,s)}\}_{s=0}^{K_{out}-1}\Big).$$

定义公开模型权重哈希为

$$h_W = H(W).$$

### 0.6 全部中间变量

#### 0.6.1 原始特征

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 与每个输入维索引 $j\in\{0,1,\ldots,d_{in}-1\}$，定义

$$H_{i,j}=T_H[I_i,j].$$

于是 $H\in\mathbb F_p^{N\times d_{in}}$。

#### 0.6.2 单个隐藏层模板中第 $r$ 个注意力头的节点域变量

对某个固定隐藏层 $\ell$ 的单层模板，在该模板中，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$，定义：

1. 投影结果：
   $$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}.$$
2. 节点域源注意力：
   $$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)}.$$
3. 节点域目标注意力：
   $$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}.$$
4. 节点域组最大值：
   $$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}.$$
5. 节点域分母：
   $$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}.$$
6. 节点域逆元：
   $$inv_i^{(r)}=\mathrm{QuantInv}(Sum_i^{(r)}).$$
7. 聚合前隐藏矩阵：
   $$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}.$$
8. ELU 后隐藏矩阵：
   $$H_{agg,i,j}^{(r)}=\mathrm{QuantELU}(H_{agg,pre,i,j}^{(r)}).$$

#### 0.6.3 单个隐藏层模板中第 $r$ 个注意力头的边域变量

对某个固定隐藏层 $\ell$ 的单层模板，在该模板中，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$，定义：

1. 源注意力广播：
   $$E_{src,k}^{edge(r)}=E_{src,src(k)}^{(r)}.$$
2. 目标注意力广播：
   $$E_{dst,k}^{edge(r)}=E_{dst,dst(k)}^{(r)}.$$
3. 线性打分：
   $$S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)}.$$
4. LeakyReLU 后打分：
   $$Z_k^{(r)}=\mathrm{QuantLReLU}(S_k^{(r)}).$$
5. 最大值广播：
   $$M_k^{edge(r)}=M_{dst(k)}^{(r)}.$$
6. 非负差分：
   $$\Delta_k^{+(r)}=M_k^{edge(r)}-Z_k^{(r)}.$$
7. 指数输出：
   $$U_k^{(r)}=\mathrm{QuantExp}(\Delta_k^{+(r)}).$$
8. 分母广播：
   $$Sum_k^{edge(r)}=Sum_{dst(k)}^{(r)}.$$
9. 逆元广播：
   $$inv_k^{edge(r)}=inv_{dst(k)}^{(r)}.$$
10. 归一化权重：
    $$\alpha_k^{(r)}=\mathrm{Rescale}\big(U_k^{(r)}\cdot inv_k^{edge(r)}\big).$$

#### 0.6.4 隐藏层模板的拼接阶段变量

定义拼接结果 $H_{cat}\in\mathbb F_p^{N\times d_{cat}}$，其中对任意 $i,r,j$，有

$$H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}.$$

#### 0.6.5 输出层变量

对每个输出头 $s\in\{0,1,\ldots,K_{out}-1\}$，定义：

1. 输出投影结果：
   $$Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out,s)}.$$
2. 输出层源注意力：
   $$E_{src,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{src,c}^{(out,s)}.$$
3. 输出层目标注意力：
   $$E_{dst,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{dst,c}^{(out,s)}.$$
4. 输出层线性打分：
   $$S_k^{(out,s)}=E_{src,src(k)}^{(out,s)}+E_{dst,dst(k)}^{(out,s)}.$$
5. 输出层 LeakyReLU 后打分：
   $$Z_k^{(out,s)}=\mathrm{QuantLReLU}(S_k^{(out,s)}).$$
6. 输出层组最大值：
   $$M_i^{(out,s)}=\max\{Z_k^{(out,s)}\mid dst(k)=i\}.$$
7. 输出层非负差分：
   $$\Delta_k^{+(out,s)}=M_{dst(k)}^{(out,s)}-Z_k^{(out,s)}.$$
8. 输出层指数输出：
   $$U_k^{(out,s)}=\mathrm{QuantExp}(\Delta_k^{+(out,s)}).$$
9. 输出层分母：
   $$Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)}.$$
10. 输出层逆元：
    $$inv_i^{(out,s)}=\mathrm{QuantInv}(Sum_i^{(out,s)}).$$
11. 输出层归一化权重：
    $$\alpha_k^{(out,s)}=\mathrm{Rescale}\big(U_k^{(out,s)}\cdot inv_{dst(k)}^{(out,s)}\big).$$
12. 输出层聚合前输出：
    $$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}.$$
13. 尺度对齐后的聚合前输出：
    $$\widetilde Y_{i,c}^{pre(s)}=\mathrm{Rescale}\big(Y_{i,c}^{pre(s)};S_{out,pre}^{(s)}\rightarrow S_{out,head}^{(s)}\big).$$
14. 输出头加偏置后的输出：
    $$Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}.$$
15. 最终输出：
    $$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)},\qquad Y_{i,c}=K_{out}^{-1}\cdot Y_{i,c}^{sum}.$$

### 0.7 电路基础约束说明

为与论文中的 ONNX-to-circuit 设计保持一致，本文仅按功能把编译后电路中的约束理解为以下几类基础约束组合；这是一种实现规约，用于说明 witness 与前向语义如何落入电路，不把这些对象重新暴露为外部 proof schema。

#### 0.7.1 累积点积约束

设 $x,y\in\mathbb F_p^n$，引入累加器向量 $m\in\mathbb F_p^{n+1}$，满足

$$m_0=0,\qquad x_i y_i + m_i = m_{i+1}\quad \forall i\in\{0,1,\ldots,n-1\}.$$

则

$$m_n=\sum_{i=0}^{n-1}x_i y_i.$$

该约束用于矩阵乘法、线性投影、注意力向量内积等操作。

#### 0.7.2 累积和 / 累积积约束

设 $x\in\mathbb F_p^n$，引入累加器向量 $m\in\mathbb F_p^{n+1}$，满足

$$m_0=0,\qquad x_i \oplus m_i = m_{i+1}\quad \forall i\in\{0,1,\ldots,n-1\},$$

其中 $\oplus$ 表示加法或乘法。

#### 0.7.3 逐元素加减乘约束

设 $x,y,z\in\mathbb F_p^n$，逐元素关系写为

$$x_i\circ y_i = z_i\quad \forall i\in\{0,1,\ldots,n-1\},$$

其中 $\circ\in\{+,-,\times\}$。

#### 0.7.4 lookup 约束

设量化非线性函数 $\phi$ 对应查表集合

$$T_\phi = \{(u,v)\mid v=\phi(u)\}.$$

则对输入输出对 $(u_i,v_i)$ 的约束写为

$$(u_i,v_i)\in T_\phi\quad \forall i.$$ 

这覆盖 $\mathrm{QuantLReLU}$、$\mathrm{QuantELU}$、$\mathrm{QuantExp}$ 以及范围检查等。

#### 0.7.5 最大值约束

对每个目标节点的组内最大值，采用由比较、二值选择与 lookup 组合得到的 max gadget，实现

$$M_i = \max\{Z_k\mid dst(k)=i\}.$$

### 0.8 strict route 下的公共输入、私有输入与见证

本节定义 `ezkl_strict_public_input_full` 路线下的公开输入、私有输入与见证；`ezkl_timing_private_input_public_output_full` 的可见性差异见 0.9 节。

对单个样本证明，strict route 的公开输入和私有输入固定如下：

- 公开元数据：
  $$M_{pub}.$$

- 公开输入张量：
  $$\tilde x=(I,src,dst,node\_ptr,edge\_ptr,H,quant\_cfg,model\_arch\_id).$$

- 公开输出张量：
  $$\tilde y.$$

其中 $\tilde y$ 表示**公开量化输出张量**。若普通推理路径先得到明文输出 $y$，则进入证明系统之前必须按当前冻结的量化配置把它唯一确定地映射为公开输出张量 $\tilde y$。因此，后文凡写 $\tilde y$，均指单样本证明实际绑定的公开量化输出，而不是未量化的明文 logits。

- 公开模型权重哈希：
  $$h_W.$$

- 私有输入：
  $$W.$$

- 见证：
  $$\omega=(W,\text{全部中间张量与全部量化中间值}).$$

于是单样本电路关系写为

$$R_{GAT\text{-}ezkl}(M_{pub},\tilde x,\tilde y,h_W;W)=1.$$

更具体地，若把 ONNX 编译后的 Halo2 电路记为 $\mathcal C_{GAT}$，则要求

$$\mathcal C_{GAT}(M_{pub},\tilde x,\tilde y,h_W;\omega)=1,\qquad H(W)=h_W,\qquad f_{GAT}^{quant}(\tilde x;W)=\tilde y.$$

### 0.9 对比实验路线分层：strict 证明路线与 runnable timing route

0.8 节定义的是本文的 **strict public-input equivalence route**，其路线标识固定为：

$$proof\_route\_id=ezkl\_strict\_public\_input\_full.$$

在该路线中，$\tilde x=(I,src,dst,node\_ptr,edge\_ptr,H,quant\_cfg,model\_arch\_id)$ 必须作为公开输入绑定到 proof / verify 过程；$\tilde y$、$h_W$ 与 $M_{pub}$ 也必须公开绑定。该路线是本文与主仓库证明语义最严格对齐的证明路线，也是默认协议语义。除非显式声明 $proof\_route\_id$ 为其它值，后续第 1--4 节均按该 strict route 理解。

#### 0.9.1 strict route 与 timing route 的使用边界

`ezkl_strict_public_input_full` 是本文默认的严格证明路线，用于表达“公开输入、公开输出与模型身份均被 proof / verify 直接绑定”的证明语义。

若在具体 ezkl / Halo2 后端实现中，strict route 因 public instance 规模、domain 约束、setup 资源或 backend 能力边界无法覆盖全部正式 benchmark，则 comparison 实验允许启用 `ezkl_timing_private_input_public_output_full` 作为独立的 runnable timing route。

timing route 的目的仅是获得 ezkl 在相同模型、相同 checkpoint、相同数据集、相同图规则与相同 benchmark 口径下的运行时间开销。它不得被解释为 strict public-input route 的成功替代，也不得被用于声称完整公开输入绑定已经跑通。

两条路线必须在 metadata、结果表和最终报告中通过 `proof_route_id` 与 `visibility_id` 明确区分。


#### 0.9.2 runnable timing route

为了获得五个全量正式数据集在 ezkl 后端上的可运行时间开销，本文在不改变主仓库模型、数据、图规则、checkpoint、量化语义和 benchmark 口径的前提下，额外定义一条 **runnable timing route**。该路线仅用于 runtime / cost measurement，不得冒充 0.8 节 strict public-input equivalence route 的成功结果。

该路线标识固定为：

$$proof\_route\_id=ezkl\_timing\_private\_input\_public\_output\_full.$$

该路线采用以下 ezkl visibility 适配：

1. $input\_visibility=private$：$\tilde x=(I,src,dst,node\_ptr,edge\_ptr,H,quant\_cfg,model\_arch\_id)$ 作为 witness input 进入电路，不再作为 public instance 暴露；
2. $output\_visibility=public$：量化 logits 与 graph consistency checks 仍作为公开输出绑定；

   其中，graph consistency checks 至少应覆盖由主仓库正式预处理规则导出的图结构一致性校验摘要，包括节点数、边数、batch 指针、排序规则、自环规则与当前 $preprocess\_id$ 的一致性；其具体字段必须由实现中的正式 metadata 固定，且不得替代或修改主仓库图预处理语义。
3. $param\_visibility=fixed$：权重参数仍由同一 checkpoint 导出并固定到当前 ONNX / ezkl compiled circuit；
4. $h_W$ 与 $M_{pub}$ 仍公开记录，用于绑定模型身份、配置身份、route 身份与 artifact 身份；
5. 输入图、节点顺序、边顺序、batch 规则、自环规则、checkpoint、GAT 前向语义、输出层多头平均语义与主仓库保持一致。

在 timing route 中，$\tilde x$ 的可见性变化属于 ezkl 证明接口 / witness organization 的必要适配；它不改变 $f_{GAT}^{quant}(\tilde x;W)=\tilde y$ 的计算语义，也不改变任一数据集、图结构、权重或 benchmark 口径。

#### 0.9.3 结果解释规则

comparison 结果必须区分两类路线：

1. `ezkl_strict_public_input_full`：用于判断原始公开输入绑定证明方案是否能完整替换主证明方案；若未跑通，只能报告为 strict route capability boundary；
2. `ezkl_timing_private_input_public_output_full`：用于获得 ezkl 在同一模型、同一数据、同一图规则与同一 checkpoint 下的 full-dataset runtime cost。

最终 timing 表必须显式记录 $proof\_route\_id$、$input\_visibility$、$output\_visibility$、$param\_visibility$、$final\_K$、`num_inner_cols`、SRS 策略、setup / prove / verify 时间与 proof size。任何 timing route 结果都不得被写成 strict public-input route 的成功结果。

## 1. 参数生成

### 1.1 输入

参数生成算法输入为：

- 安全参数 $\lambda$
- ONNX 模型文件 $M_{onnx}$
- 图输入配置与预处理规则
- 模型结构配置 $(L,\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1},K_{out},C)$
- 量化配置 $quant\_cfg$
- 数据集和 benchmark 配置
- 模型权重集合 $W$

### 1.2 输出

参数生成算法输出：

- 编译后的电路 $\mathcal C_{GAT}$
- proving key：$pk$
- verification key：$vk$
- 模型权重哈希：$h_W=H(W)$
- 公共元数据块：$M_{pub}$

### 1.3 ONNX 编译

将多层多头 GAT 前向导出为 ONNX 计算图。导出时必须满足：

1. 图输入、局部节点顺序、边顺序和 batch 规则与 benchmark 任务完全一致；
2. 隐藏层的 concat 语义保持不变；
3. 输出层“每头先偏置后平均”的语义保持不变；
4. 全部定点量化和 rescale 规则由 $quant\_cfg$ 唯一决定；
5. 若某一步在 ONNX 中只能以近似形式表达，则该近似必须被唯一固定并写入公共元数据。

编译器把 ONNX 图中的每个算子降低为 Halo2 电路中的约束。多数线性运算可规约为 Einstein summation；逐元素算子规约为 elementwise 约束；非线性规约为 lookup 或其组合；分组求和、点积和指标规约为累计式约束。

### 1.4 公共元数据

公共元数据固定为

$$M_{pub}=(protocol\_id,model\_arch\_id,model\_param\_id,onnx\_graph\_id,quant\_cfg\_id,preprocess\_id,attestation\_mode,encoding\_id,degree\_bound\_id,proof\_system\_tag,ezkl\_version,proof\_route\_id,visibility\_id).$$

其中：

- $protocol\_id$：固定标识本文证明流程版本
- $model\_arch\_id$：固定绑定层数、头数、维度、输出头平均语义
- $model\_param\_id$：唯一标识当前参数族
- $onnx\_graph\_id$：唯一标识 ONNX 导出图
- $quant\_cfg\_id$：唯一标识量化与校准规则
- $preprocess\_id$：唯一标识图排序、自环、batch 和子图规则
- $attestation\_mode$：标识采用哪种 attestation 打包方式
- $encoding\_id$：标识 proof 编码格式
- $degree\_bound\_id$：标识后端 proving system 的相关配置
- $proof\_system\_tag$：固定标识当前 proving backend，本文最终版固定为 ezkl / Halo2 / KZG 路线
- $ezkl\_version$：固定标识当前 comparison 实验所使用的 ezkl 版本
- $proof\_route\_id$：在 comparison artifact / metadata 中额外记录当前证明 / 计时路线，取值必须为 `ezkl_strict_public_input_full` 或 `ezkl_timing_private_input_public_output_full`
- $visibility\_id$：在 comparison artifact / metadata 中额外记录 ezkl 输入、输出和参数可见性，例如 `public_public_fixed` 或 `private_public_fixed`

### 1.5 模型权重哈希绑定

除推理电路外，额外引入模型权重哈希子电路，要求：

$$H(W)=h_W.$$

因此，单样本证明的绑定对象必须按 $proof\_route\_id$ 区分：

1. 在 `ezkl_strict_public_input_full` 中，证明同时绑定公开输入张量、公开输出张量与模型权重哈希；
2. 在 `ezkl_timing_private_input_public_output_full` 中，证明公开绑定输出张量、模型身份、$proof\_route\_id$、$visibility\_id$ 与相关公共元数据；完整输入张量作为 witness input 参与电路约束，但不作为 public instance 暴露；
3. 两条路线都必须绑定同一 checkpoint 导出的固定权重身份，且不得改变模型、图规则、数据集或量化语义。

### 1.6 校准与 trusted setup

编译阶段还必须执行量化校准。记浮点模型输出为 $y^{fp}$，量化输出为 $\tilde y$，则校准目标是在资源开销与语义偏移之间取得平衡，要求对 benchmark 任务满足：

$$\|\tilde y-y^{fp}\|\le \varepsilon_{quant}$$

其中阈值 $\varepsilon_{quant}$ 必须由实验配置预先固定。

trusted setup 阶段为 proving system 生成结构化参考字符串，之后得到 $(pk,vk)$。只有在同一冻结后的 $onnx\_graph\_id$、同一输入形状、同一预处理口径、同一编译设置、同一量化配置、同一 $proof\_route\_id$ 与同一 $visibility\_id$ 下，才允许对不同样本重复使用同一组 $(pk,vk)$。一旦其中任一项发生变化，必须重新执行对应 setup。

## 2. 见证生成

### 2.0 总体顺序

工程实现必须按以下顺序执行：

1. 按公开固定的预处理规则完成图抽取、重编号、加自环、排序、batch 构造；
2. 明文推理：在普通推理路径上运行 GAT，得到输入输出对 $(x_i,y_i)$；
3. 量化与 witness 生成：将 $(x_i,y_i)$ 量化为 $(\tilde x_i,\tilde y_i)$，并构造全部中间张量见证；
4. 按 $proof\_route\_id$ 生成单样本 zkSNARK 证明：strict route 将 $\tilde x_i$ 与 $\tilde y_i$ 作为 public instance 绑定；timing route 将 $\tilde x_i$ 写入 witness 文件，仅将 $\tilde y_i$ 与 route metadata 作为 public instance 绑定；
5. 将全体单样本证明打包为与当前 route 匹配的 evaluation attestation。

### 2.1 数据加载与特征检索

对每个样本，先根据 $I$ 从特征表 $T_H$ 中取出局部特征：

$$H_{i,j}=T_H[I_i,j].$$

随后将 $(H,src,dst,node\_ptr,edge\_ptr)$ 编码为 ONNX/Halo2 所消费的输入张量。此阶段不生成任何显式多项式承诺，而只生成 witness 所需的张量数据。

### 2.2 隐藏层模板中第 $r$ 个注意力头的见证生成

以下步骤对应某个固定隐藏层 $\ell\in\{1,2,\ldots,L-1\}$ 的模板。在该模板中，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$ 依次生成见证。

#### 2.2.1 投影

对每个节点 $i$ 与每个隐藏维索引 $j$，定义

$$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}.$$

#### 2.2.2 源 / 目标注意力

对每个节点 $i$，定义

$$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)},$$
$$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}.$$

#### 2.2.3 边打分与 LeakyReLU

对每个边索引 $k$，定义广播量

$$E_{src,k}^{edge(r)}=E_{src,src(k)}^{(r)},\qquad E_{dst,k}^{edge(r)}=E_{dst,dst(k)}^{(r)}.$$

再定义线性打分与 LeakyReLU 后打分：

$$S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)},$$
$$Z_k^{(r)}=\mathrm{QuantLReLU}(S_k^{(r)}).$$

#### 2.2.4 组最大值、指数与 softmax 分母

对每个目标节点 $i$，定义

$$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}.$$

对每个边索引 $k$，定义

$$\Delta_k^{+(r)}=M_{dst(k)}^{(r)}-Z_k^{(r)},$$
$$U_k^{(r)}=\mathrm{QuantExp}(\Delta_k^{+(r)}).$$

然后对每个目标节点 $i$ 定义 softmax 分母

$$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}.$$

#### 2.2.5 归一化权重与聚合

定义

$$inv_i^{(r)}=\mathrm{QuantInv}(Sum_i^{(r)}),$$
$$\alpha_k^{(r)}=\mathrm{Rescale}\big(U_k^{(r)}\cdot inv_{dst(k)}^{(r)}\big).$$

再对每个节点 $i$ 与每个隐藏维索引 $j$，定义聚合前隐藏矩阵：

$$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}.$$

#### 2.2.6 ELU 输出

对每个节点 $i$ 与每个隐藏维索引 $j$，定义

$$H_{agg,i,j}^{(r)}=\mathrm{QuantELU}(H_{agg,pre,i,j}^{(r)}).$$

### 2.3 隐藏层拼接

对任意节点 $i$、任意头 $r$ 与局部维索引 $j$，定义

$$H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}.$$

该拼接在 ONNX 图中表现为 tensor concatenate 操作，在电路中通过张量索引和复制一致性约束实现。

### 2.4 输出层见证生成

#### 2.4.1 输出投影

对每个输出头 $s$，定义

$$Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out,s)}.$$

#### 2.4.2 输出层源 / 目标注意力

对每个节点 $i$，定义

$$E_{src,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{src,c}^{(out,s)},$$
$$E_{dst,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{dst,c}^{(out,s)}.$$

#### 2.4.3 输出层 softmax

对每个边索引 $k$，定义

$$S_k^{(out,s)}=E_{src,src(k)}^{(out,s)}+E_{dst,dst(k)}^{(out,s)},$$
$$Z_k^{(out,s)}=\mathrm{QuantLReLU}(S_k^{(out,s)}).$$

再对每个目标节点 $i$ 定义

$$M_i^{(out,s)}=\max\{Z_k^{(out,s)}\mid dst(k)=i\},$$
$$\Delta_k^{+(out,s)}=M_{dst(k)}^{(out,s)}-Z_k^{(out,s)},$$
$$U_k^{(out,s)}=\mathrm{QuantExp}(\Delta_k^{+(out,s)}),$$
$$Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)},$$
$$inv_i^{(out,s)}=\mathrm{QuantInv}(Sum_i^{(out,s)}),$$
$$\alpha_k^{(out,s)}=\mathrm{Rescale}\big(U_k^{(out,s)}\cdot inv_{dst(k)}^{(out,s)}\big).$$

#### 2.4.4 输出层聚合、尺度对齐与偏置

对每个节点 $i$ 与类别索引 $c$，定义

$$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}.$$

然后做尺度对齐：

$$\widetilde Y_{i,c}^{pre(s)}=\mathrm{Rescale}\big(Y_{i,c}^{pre(s)};S_{out,pre}^{(s)}\rightarrow S_{out,head}^{(s)}\big).$$

再做偏置相加：

$$Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}.$$

#### 2.4.5 多头平均

全部输出头完成后，对每个节点 $i$ 与类别索引 $c$，定义

$$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)},$$
$$Y_{i,c}=K_{out}^{-1}\cdot Y_{i,c}^{sum}.$$

最终输出 $Y$ 即为该量化 GAT 模型的 logits。

### 2.5 witness 文件与单样本证明

对每个样本 $i$，先在普通推理路径上得到输入输出对 $(x_i,y_i)$，再量化为 $(\tilde x_i,\tilde y_i)$。构造 witness：

$$\omega_i=(W,\text{全部中间张量},\tilde x_i,\tilde y_i,h_W).$$

然后生成单样本证明。协议层可统一记为：

$$\pi_i \leftarrow \mathrm{Prove}_{proof\_route\_id}(pk,W,\tilde x_i,\tilde y_i),$$

这里的 $\mathrm{Prove}_{proof\_route\_id}(pk,W,\tilde x_i,\tilde y_i)$ 是**协议层抽象记号**，用于表达“使用与当前冻结模型图、route 与 visibility 对应的 proving key、私有权重、该样本输入输出和 witness 文件生成单样本推理证明”的语义。实际 ezkl 工程实现中，prove 阶段的直接输入应落实为：编译后的电路 / settings、与该样本对应的 witness 文件，以及 proving key $pk$；而不是把上述四元组机械理解为某个固定外部 API 的字面形参列表。

在 strict route 中，$\tilde x_i$ 与 $\tilde y_i$ 均作为 public instance 绑定；在 timing route 中，$\tilde x_i$ 位于 witness，$\tilde y_i$、$proof\_route\_id$ 与 $visibility\_id$ 作为公开对象绑定。

并要求

$$\pi_i \supset \{h_W,\tilde y_i,proof\_route\_id,visibility\_id\}.$$

等价地，单样本证明满足如下 route-aware 验证关系：

strict route 中：

$$\mathrm{Verify}_{strict}(vk,\tilde x_i,\tilde y_i,h_W,\pi_i)=1.$$

timing route 中：

$$\mathrm{Verify}_{time}(vk,\tilde y_i,h_W,proof\_route\_id,visibility\_id,\pi_i)=1.$$

其中 timing route 的 $\tilde x_i$ 不是 verifier 接收的 public instance，而是 witness satisfiability 中被约束的输入。

### 2.6 evaluation attestation 生成

设评测数据集共有 $N_{eval}$ 个样本。对每个样本生成证明 $\pi_i$ 后，论文允许把这些单样本 inference proofs 继续组织为 verifiable evaluation attestation。为保证当前 comparison/ezkl 对比实验的边界清晰，本文最终版本固定采用 **直接打包模式**，不在第一版实验中引入额外 aggregation circuit。

#### 2.6.1 直接打包模式

定义 strict route 的直接打包 attestation：

$$\Pi_{eval}^{bundle}=\Big(M_{pub},h_W,proof\_route\_id,visibility\_id,\{(\tilde x_i,\tilde y_i,\pi_i)\}_{i=1}^{N_{eval}}\Big).$$

定义 timing route 的直接打包 attestation：

$$\Pi_{eval,time}^{bundle}=\Big(M_{pub},h_W,proof\_route\_id,visibility\_id,\{(\tilde y_i,\pi_i)\}_{i=1}^{N_{eval}}\Big).$$

在 timing route 中，$\tilde x_i$ 不作为 public instance 打包；它只作为 witness input 参与每个证明的电路满足性验证。上述两类直接打包模式都最接近逐样本基线，验证者可逐个验证全部 proof，并自行计算 accuracy、bias metric 或其他公开指标。

#### 2.6.2 Attestation 边界

虽然论文同时讨论了 proof aggregation 与 metric-only aggregation 两类更进一步的组织方式，但它们都会额外引入第二层聚合电路、额外 setup 与额外审计负担，因此本文不把这两类 aggregation 机制纳入协议主线，也不纳入 benchmark 主表。

本文后续凡提及 evaluation attestation，若无特别声明，均指与当前 $proof\_route\_id$ 对应的直接打包模式：strict route 使用 $\Pi_{eval}^{bundle}$，timing route 使用 $\Pi_{eval,time}^{bundle}$。

## 3. 证明生成

### 3.1 输入与输出

证明算法输入为：

- 编译后的电路与参数 $(\mathcal C_{GAT},pk,vk)$
- 与 $proof\_route\_id$ 匹配的公开对象：strict route 使用 $(M_{pub},\tilde x,\tilde y,h_W)$；timing route 使用 $(M_{pub},\tilde y,h_W,proof\_route\_id,visibility\_id)$
- 私有权重 $W$
- 全部 witness 张量；timing route 中 $\tilde x$ 也包含在 witness 张量中

输出为单样本 zkSNARK 证明 $\pi$；evaluation attestation 在本文最终 comparison 实验中固定采用直接打包模式，不额外生成 aggregation proof。

### 3.2 Fiat–Shamir 挑战顺序

本文不重新规定 ezkl / Halo2 后端内部的 Fiat–Shamir transcript 细节。凡电路内部承诺吸入、挑战派生、多轮折叠与 opening 次序，统一以具体 backend 实现为准。

本文只固定以下外部绑定语义：

1. 单样本证明必须绑定公共元数据块 $M_{pub}$；
2. 单样本证明必须绑定 $proof\_route\_id$ 与 $visibility\_id$；
3. strict route 的单样本证明必须绑定公开输入 $\tilde x$；
4. timing route 的单样本证明不把完整 $\tilde x$ 作为 public instance，而要求 $\tilde x$ 作为 witness input 满足当前冻结电路；
5. 单样本证明必须绑定公开输出 $\tilde y$；
6. 单样本证明必须绑定模型权重哈希 $h_W$；
7. evaluation attestation 中全部单样本 proof 必须绑定到同一 $M_{pub}$、同一 $h_W$、同一 $proof\_route\_id$ 与同一 $visibility\_id$。

### 3.3 电路约束族

本节只对 0.7 节给出的基础约束说明做面向证明阶段的归纳，不再额外引入新的外部 proof 对象。


#### 3.3.1 投影与线性层约束

对任意矩阵乘法或向量点积，都使用累积点积约束。

这覆盖：

- 隐藏层投影 $H\mapsto H'^{(r)}$
- 隐藏层源 / 目标注意力投影
- 输出层投影 $H_{cat}\mapsto Y'^{(s)}$
- 输出层源 / 目标注意力投影

#### 3.3.2 分组求和与聚合约束

对任意组内累积和，都使用累积和约束。

这覆盖：

- softmax 分母 $Sum_i^{(r)}$
- 聚合前隐藏表示 $H_{agg,pre}^{(r)}$
- 输出层聚合前输出 $Y^{pre(s)}$
- 多头最终求和 $Y^{sum}$

#### 3.3.3 逐元素约束

对任意逐元素运算，统一写成

$$u_i\circ v_i=w_i.$$

这覆盖：

- 打分相加 $S=E_{src}^{edge}+E_{dst}^{edge}$
- 差分 $\Delta^+=M^{edge}-Z$
- 归一化乘法 $\alpha = \mathrm{Rescale}(U\cdot inv)$
- 偏置相加 $Y^{(s)}=\widetilde Y^{pre(s)}+b^{(out,s)}$
- 多头求和 $Y^{sum}$

#### 3.3.4 lookup 约束

对任意量化非线性算子 $\phi$，统一写为

$$(u_i,v_i)\in T_\phi.$$

这覆盖：

- $\mathrm{QuantLReLU}$
- $\mathrm{QuantELU}$
- $\mathrm{QuantExp}$
- 范围检查与定点校准辅助查表

#### 3.3.5 最大值约束

对每个目标节点组内的最大值，使用 0.7.5 节中的 max gadget。对每个分组 $G_i=\{k\mid dst(k)=i\}$，要求：

$$M_i = \max\{Z_k\mid k\in G_i\}.$$

该约束对隐藏层和输出层分别实例化。

#### 3.3.6 量化一致性约束

对所有 rescale、截断、舍入和固定点格式，要求存在由 $quant\_cfg$ 唯一决定的电路约束，使得：

$$\tilde z = \mathrm{Rescale}(z;S_a\rightarrow S_b)$$

在所有样本上使用同一规则。任何与 $quant\_cfg$ 不一致的近似都不得被接受。

#### 3.3.7 模型权重哈希约束

额外要求：

$$H(W)=h_W.$$

因此，即使输出正确，若使用了不同权重，也无法通过验证。

#### 3.3.8 单样本推理关系

综合全部约束后，strict route 的单样本电路满足以下关系：

$$R_{GAT\text{-}ezkl}^{strict}(M_{pub},\tilde x,\tilde y,h_W;W)=1$$

timing route 的单样本电路满足以下关系：

$$R_{GAT\text{-}ezkl}^{time}(M_{pub},\tilde y,h_W,proof\_route\_id,visibility\_id;W,\tilde x)=1.$$

两条路线当且仅当以下条件成立：

1. 权重哈希满足 $H(W)=h_W$；
2. 量化 GAT 前向满足 $f_{GAT}^{quant}(\tilde x;W)=\tilde y$；
3. 全部中间张量满足隐藏层、输出层、softmax、非线性与量化一致性约束；
4. 两条路线的差异只在 $\tilde x$ 的 public instance / witness 可见性，而不在模型、数据、图规则或前向语义。

### 3.4 单样本 proof object

strict route 的单样本证明对象固定记为

$$\pi=(M_{pub},h_W,\tilde x,\tilde y,\Pi_{snark}).$$

其中：

- $M_{pub}$ 是公共元数据
- $h_W$ 是公开模型权重哈希
- $\tilde x$ 是公开输入张量
- $\tilde y$ 是公开输出张量
- $\Pi_{snark}$ 是 Halo2/ezkl 后端给出的 succinct proof

上述 $\pi=(M_{pub},h_W,\tilde x,\tilde y,\Pi_{snark})$ 是 `ezkl_strict_public_input_full` 的单样本 proof object。

对 `ezkl_timing_private_input_public_output_full`，单样本 timing proof object 固定记为：

$$\pi^{time}=(M_{pub},h_W,\tilde y,proof\_route\_id,visibility\_id,\Pi_{snark}).$$

其中 $\tilde x$ 不再作为 public instance 暴露，而是作为 witness input 参与电路满足性证明。该对象只用于 ezkl timing baseline，不得解释为 strict public-input proof object。

### 3.5 evaluation attestation object

本文只定义直接打包模式。

strict route 的 evaluation attestation object 为：

$$\Pi_{eval}^{bundle}=\Big(M_{pub},h_W,proof\_route\_id,visibility\_id,\{(\tilde x_i,\tilde y_i,\pi_i)\}_{i=1}^{N_{eval}}\Big).$$

timing route 的 evaluation attestation object 为：

$$\Pi_{eval,time}^{bundle}=\Big(M_{pub},h_W,proof\_route\_id,visibility\_id,\{(\tilde y_i,\pi_i)\}_{i=1}^{N_{eval}}\Big).$$

其中每个 $\pi_i$ 都是单样本推理证明，且全部证明共享同一公开元数据 $M_{pub}$、同一模型权重哈希 $h_W$、同一 $proof\_route\_id$ 与同一 $visibility\_id$。timing route 的 object 不包含 public $\tilde x_i$，不得解释为 strict public-input attestation。

## 4. 验证

### 4.1 元数据与电路一致性检查

验证者首先检查：

1. $protocol\_id$、$model\_arch\_id$、$model\_param\_id$、$onnx\_graph\_id$、$quant\_cfg\_id$、$preprocess\_id$、$proof\_system\_tag$、$ezkl\_version$、$proof\_route\_id$ 与 $visibility\_id$ 与本地期望一致；
2. strict route 中，验证者直接检查公开输入图的排序、自环和 batch 规则与公开固定的预处理规则一致；timing route 中，输入图作为 witness input 进入电路，其排序、自环、batch 规则由编译电路、公开 metadata、公开 graph consistency checks 与 witness satisfiability 共同约束；
3. strict route 中，proof 对象中的公开输入输出张量维度与模型结构一致；timing route 中，proof 对象中的公开输出张量、route metadata 与模型结构一致，完整输入张量维度由 witness / compiled circuit 约束；
4. 模型权重哈希 $h_W$ 与目标 attestation 声称的模型身份一致。

### 4.2 单样本证明验证

对任意单样本证明 $\pi_i$，验证者按当前 $proof\_route\_id$ 执行对应验证。

strict route 验证为：

$$\mathrm{Verify}_{strict}(vk,\tilde x_i,\tilde y_i,h_W,\pi_i)=1?$$

timing route 验证为：

$$\mathrm{Verify}_{time}(vk,\tilde y_i,h_W,proof\_route\_id,visibility\_id,\pi_i)=1?$$

其中 timing route 的 $\tilde x_i$ 仅存在于 witness / circuit satisfiability 中，不作为 verifier 的 public input。

若验证成功，则接受以下事实：

1. 存在一组私有权重 $W$，满足 $H(W)=h_W$；
2. 存在一组 witness 中间张量，使得编译后的 GAT 电路约束全部成立；
3. 公开输出 $\tilde y_i$ 确实是某个满足当前 route、visibility 与 witness 约束的输入 $\tilde x_i$ 在该权重下的量化推理结果；
4. 只有 strict route 可进一步解释为“验证者给定的公开输入 $\tilde x_i$ 被 proof 直接绑定”。

### 4.2.1 timing route 的验证语义边界

对于 `ezkl_timing_private_input_public_output_full`，验证者执行的是 ezkl 后端对 public output、fixed params 与 witness satisfiability 的验证。此时 public instance 不包含完整 $\tilde x$；因此 timing route 的验证结论只能解释为：存在某个 witness input $\tilde x$ 与固定权重 / 固定电路，使得公开输出 $\tilde y$ 满足当前 ONNX / Halo2 电路约束。它不等价于 0.8 节 strict route 中“公开输入 $\tilde x$ 被 proof 直接绑定”的语义。

### 4.3 直接打包 attestation 验证

对 strict route 的 $\Pi_{eval}^{bundle}$，验证者逐个验证所有样本证明：

$$\forall i\in\{1,\ldots,N_{eval}\},\qquad \mathrm{Verify}_{strict}(vk,\tilde x_i,\tilde y_i,h_W,\pi_i)=1.$$

对 timing route 的 $\Pi_{eval,time}^{bundle}$，验证者逐个验证所有样本证明：

$$\forall i\in\{1,\ldots,N_{eval}\},\qquad \mathrm{Verify}_{time}(vk,\tilde y_i,h_W,proof\_route\_id,visibility\_id,\pi_i)=1.$$

全部通过后，再根据公开标签 $\hat y_i$ 计算公开指标：

$$Acc = \frac{1}{N_{eval}}\sum_{i=1}^{N_{eval}}\mathbf 1\{\operatorname{argmax}(\tilde y_i)=\hat y_i\}.$$

若需要 fairness、precision、recall 或其他统计量，则同理直接在公开输出上计算。timing route 的公开指标可用于 runtime / cost comparison，但不得解释为 strict public-input route 的成功证明。

### 4.4 未来推理挑战验证（扩展语义）

对未来用户给定的输入 $x^*$ 和模型提供者返回的输出 $y^*$，模型提供者需按当前 $proof\_route\_id$ 重新生成单样本证明：

$$\pi^*\leftarrow \mathrm{Prove}_{proof\_route\_id}(pk,W,\tilde x^*,\tilde y^*).$$

这里的 $\mathrm{Prove}_{proof\_route\_id}(pk,W,\tilde x^*,\tilde y^*)$ 同样只是协议层抽象记号。实际工程中，应理解为：在与当前冻结配置、route 与 visibility 一致的编译电路 / settings 下，使用 proving key $pk$ 与该次 future inference 对应的 witness 文件生成新的单样本证明。

strict route 中，验证者执行：

$$\mathrm{Verify}_{strict}(vk,\tilde x^*,\tilde y^*,h_W,\pi^*)=1.$$

timing route 中，验证者执行：

$$\mathrm{Verify}_{time}(vk,\tilde y^*,h_W,proof\_route\_id,visibility\_id,\pi^*)=1.$$

若验证成功，则说明当前输出与之前发布 evaluation attestation 的模型身份一致，即它来自同一 $h_W$ 对应的模型；但 timing route 不提供公开输入 $\tilde x^*$ 被直接绑定的 strict 语义。

### 4.5 验证结论

若元数据一致、单样本 proof 全部通过、直接打包 attestation 中全部样本证明都绑定到同一模型权重哈希 $h_W$、同一 $proof\_route\_id$ 与同一 $visibility\_id$，且公开统计指标与公开标签计算结果一致，则接受当前 route 下的证明；否则拒绝。对 timing route，接受结论仅限于 runtime / cost comparison 所定义的 private-input/public-output 证明接口，不得提升解释为 strict public-input route 成功。
