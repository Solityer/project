# GAT-ZKML zkDL 协议笔记

## 符号

### 0.0 协议关系与证明目标

本文定义的证明关系记为：

$R_{GAT\text{-}zkDL}$

其公共输入为：

$Public=(M_{pub},\mathsf{Topo},VK_{static}^{zkDL},VK_{model}^{zkDL},Y)$

其中 $M_{pub}$ 固定协议版本、模型结构、维度、量化配置、工作域与证明范围；$\mathsf{Topo}$ 固定局部子图、边序、batch 分组与公开预处理结果；$VK_{static}^{zkDL}$ 与 $VK_{model}^{zkDL}$ 分别绑定静态表和模型参数；$Y$ 是本次证明声明的最终 logits。

其私有见证为：

$Witness=(H,\mathsf{T}_{hid},\mathsf{T}_{cat},\mathsf{T}_{out},\mathsf{T}_{aux},\mathsf{T}_{agg},\mathsf r)$

其中 $\mathsf{T}_{hid}$、$\mathsf{T}_{cat}$、$\mathsf{T}_{out}$ 分别表示隐藏层、拼接阶段与输出层的全部前向中间张量；$\mathsf{T}_{aux}$ 表示非线性、范围、最大值、逆元与重缩放所需的辅助见证；$\mathsf{T}_{agg}$ 表示 FAC4DNN 聚合张量；$\mathsf r$ 表示各张量承诺随机数。

证明目标是使验证者确信存在上述私有见证，使得以下条件同时成立：

1. 特征检索与公开拓扑一致；
2. 全部隐藏层、拼接层与输出层满足第 0.1 节定义的 GAT 前向语义；
3. LeakyReLU、ELU、ExpMap、范围、最大值、逆元与重缩放等非纯乘加关系满足各自辅助子协议；
4. 所有参与约束的私有张量均被承诺绑定，并在 sumcheck 末端通过张量评值证明绑定；
5. 最终公开输出 $Y$ 等于输出层多头平均得到的 logits。

本协议的证明范围固定为 GAT 前向计算证明，不包含训练反向传播、梯度计算或参数更新语义。

### 0.1 模型结构

GAT 的前向固定为一个可参数化的多层结构：

1. 设注意力层的总层数为：$L$。其中前 $L-1$ 层是隐藏层，第 $L$ 层是最终输出层。

2. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，记该层的注意力头数量为 $K_{hid}^{(\ell)}$，每个头的输出维度为 $d_h^{(\ell)}$。

	该层每个头完成注意力聚合后，逐元素施加 ELU；

	随后把本层全部头的 ELU 输出沿特征维进行拼接，得到：$d_{cat}^{(\ell)}=K_{hid}^{(\ell)}\cdot d_h^{(\ell)}$ 

3. 第 $\ell$ 个隐藏层的输入维度记为 $d_{in}^{(\ell)}$。其中 $d_{in}^{(1)}=d_{in}$，且对任意 $\ell\ge 2$，都有 $d_{in}^{(\ell)}=d_{cat}^{(\ell-1)}$。

4. 最终输出层（第 $L$ 层）的注意力头数量记为 $K_{out}$。

	输出层不是“仿射线性层 + 偏置”替代注意力，而是由 $K_{out}$ 个输出注意力头组成。每个输出注意力头先完成注意力聚合，再加上该头自己的输出偏置向量；

	若 $K_{out}>1$，则全部输出头的后偏置输出按头维做算术平均，得到最终 logits。

5. 输出注意力头的激活函数固定为恒等映射，不再施加 ELU；但恒等映射的输入是已经加上该输出头偏置后的注意力聚合结果。

6. 后续对隐藏层的明文对象、见证对象、非线性辅助证明、稀疏路由张量、MLE 一致性绑定与 sumcheck 推导，都严格对应“单个隐藏层模板”。

	对任意多层 GAT，证明者与验证者都必须按层序 $\ell=1,\ldots,L-1$ 逐层实例化该模板；第 $\ell-1$ 层的拼接输出 $H_{cat}^{(\ell-1)}$ 必须作为第 $\ell$ 层的唯一输入表示。

7. 具体证明实例使用的 $L,\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1},K_{out}$ 必须完全由公共元数据 $M_{pub}$ 唯一确定；协议正文不设置默认结构参数。

为避免公式中引入过多层指标，除非特别声明，隐藏层模板统一采用如下局部简写：

- 隐藏层模板索引：$\ell\in\{1,2,\ldots,L-1\}$
- 第 $\ell$ 个隐藏层中第 $r$ 个注意力头的编号：$r\in\{0,1,\ldots,K_{hid}-1\}$ 
- 当前隐藏层模板的输入维度：$d_{in}:=d_{in}^{(\ell)}$ 
- 当前隐藏层模板的每头输出维度：$d_h:=d_h^{(\ell)}$ 
- 当前隐藏层模板的头数：$K_{hid}:=K_{hid}^{(\ell)}$ 
- 当前隐藏层模板的拼接维：$d_{cat}:=d_{cat}^{(\ell)}=K_{hid}\cdot d_h$
- 输出类别数：$C$
- 最终输出层第 $s$ 个输出头编号：$s\in\{0,1,\ldots,K_{out}-1\}$

换言之，后文若出现 $H_{cat}$、$d_h$、$K_{hid}$ 等未显式带层指标的记号，在隐藏层上下文中一律解释为“当前正在实例化的那一层”的数学局部记号；正式协议对象名、承诺标签、评值标签与子证明标签一律必须显式带层指标 $(\ell)$，不得把这些局部记号直接作为无层标对象使用。

同理，凡正文中为了公式紧凑而省略的 $(\ell)$、$(\ell,r)$、$(out,s)$ 标记，只能在数学推导中省略；在任何协议对象、配置字段、承诺名、评值名与子证明名中，都必须恢复完整指标。

### 0.2 有限域、承诺、MLE、Sumcheck 与 Fiat–Shamir

记有限域为 $\mathbb F_p$，其中 $p$ 为大素数。所有域内运算都在 $\mathbb F_p$ 上进行。

采用定点量化语义：若域元素 $x\in\mathbb F_p$ 的标准代表满足 $x>\frac{p-1}{2}$，则在实数语义下把它解释为负数 $x-p$。

LeakyReLU、ELU、范围比较、指数查表、量化回缩与 Softmax 中的最大值 / 分母 / 逆元语义，均在上述有符号域语义下定义。上述语义作为本文协议的固定有符号域解释。

动态见证、静态表、模型参数、FAC4DNN 聚合张量与辅助见证均作为张量或展平向量承诺。

承诺后端固定为 Hyrax 型张量承诺。本文记承诺与评值验证算法为：

$Com(T;r)\leftarrow HyraxCommit(PP_{com},T,r)$

$VerifyEval(VP_{com},Com(T),u,y,\pi_{eval})\in\{0,1\}$

其中 $T$ 是一维向量、矩阵或高阶张量按固定展平规则得到的域元素序列，$r\in\mathbb F_p$ 是承诺随机数，$u$ 是 MLE 评值点，$y=\widetilde T(u)$ 是声明评值。本文协议版本只使用这一类张量承诺语义；若承诺后端发生改变，应视为新的协议版本。

#### 0.2.1 承诺后端抽象语义

本文在协议层只依赖 Hyrax 型张量承诺的抽象接口。该接口必须满足：

1. **绑定性**：在承诺后，证明者不能为同一承诺打开出两个不同张量或两个不一致的 MLE 评值。
2. **隐藏性**：承诺与评值证明除协议声明的评值外，不泄露私有张量内容。
3. **评值正确性**：若 $Com(T)$ 绑定张量 $T$，则通过 $\pi_{eval}(T,u,y)$ 验证当且仅当 $y=\widetilde T(u)$。
4. **批量一致性**：若多个评值点或多个张量被批量打开，批量验证语义必须等价于逐项验证所有声明评值。

协议正文把 $Com(T)$ 与 $VerifyEval(VP_{com},Com(T),u,y,\pi_{eval})$ 视为基本原语。任何群元素编码、生成元派生、批量评值证明格式或字节级序列化规则均由 $commitment\_backend\_id$ 所标识的后端规范决定；这些规则不改变本文定义的证明关系。

### 0.3 图、局部子图、多图批处理与排序

- 全局节点总数：$N_{total}$

- 当前证明实例中的总节点数：$N$

- 当前证明实例中的总边数：$E$

- 当前证明实例中的图个数：$G_{batch}$

- 局部子图 / 批处理后的节点绝对编号序列：$I=(I_0,I_1,\ldots,I_{N-1})$

- 边源索引序列：$src=(src(0),src(1),\ldots,src(E-1))$

- 边目标索引序列：$dst=(dst(0),dst(1),\ldots,dst(E-1))$

- 批处理中各图的节点前缀指针：$node\_ptr=(\nu_0,\nu_1,\ldots,\nu_{G_{batch}})$，其中 $0=\nu_0<\nu_1<\cdots<\nu_{G_{batch}}=N$
  
- 批处理中各图的边前缀指针：$edge\_ptr=(\epsilon_0,\epsilon_1,\ldots,\epsilon_{G_{batch}})$，其中 $0=\epsilon_0<\epsilon_1<\cdots<\epsilon_{G_{batch}}=E$

对任意节点索引 $i\in\{0,1,\ldots,N-1\}$，定义其图编号 $node\_gid(i)=g$，其中 $g\in\{0,1,\ldots,G_{batch}-1\}$ 满足 $\nu_g\le i<\nu_{g+1}$ 

对任意边索引 $k\in\{0,1,\ldots,E-1\}$，定义其图编号 $edge\_gid(k)=g$，其中 $g\in\{0,1,\ldots,G_{batch}-1\}$ 满足 $\epsilon_g\le k<\epsilon_{g+1}$ 

因此，对任意边索引 $k$，都有：

$src(k),dst(k)\in\{0,1,\ldots,N-1\},\quad node\_gid(src(k))=edge\_gid(k),\quad node\_gid(dst(k))=edge\_gid(k)$ 

若 $G_{batch}=1$，则本文的公共输入仍然必须显式写成 $node\_ptr=(0,N),\quad edge\_ptr=(0,E)$，不得省略，也不得以其他记号替代。

固定要求边序列满足以下两级稳定排序：

1. 先按 $edge\_gid(k)$ 非降序排列，使不同图的边在公开边序列中保持连续块；
2. 在每个固定图块内部，再按目标节点索引 $dst(k)$ 做确定性稳定排序。

等价地，若 $0\le k_1<k_2\le E-1$，则必须满足：

$edge\_gid(k_1)<edge\_gid(k_2)$ 或 $(edge\_gid(k_1)=edge\_gid(k_2)$  且 $dst(k_1)\le dst(k_2)$ 

若原始输入不满足这一条件，则在见证生成前先做：

- 图内节点重编号 / 局部子图抽取；
- 按图分块的稳定排序；
- 每个图块内部按 $dst$ 的稳定排序。

验证者只接收已经按此规范生成的公开序列。除非某条公式显式写出“按图”量或“图局部”量，否则后文中的 $N,E$ 一律指当前证明实例中的总节点数与总边数；在多图批处理情形下，它们分别等于所有图的节点数与边数总和。

### 0.4 静态表与量化尺度

静态表包括：

1. 全局特征表：$T_H\in\mathbb F_p^{N_{total}\times d_{in}}$

2. LeakyReLU 表：$T_{LReLU}\subseteq \mathbb F_p\times\mathbb F_p$

3. ELU 表：$T_{ELU}\subseteq \mathbb F_p\times\mathbb F_p$

4. 指数表：$T_{exp}\subseteq \mathbb F_p\times\mathbb F_p$

为避免多值歧义，要求：

- 对任意 $x\in\mathbb F_p$，在 $T_{LReLU}$ 中至多存在一个 $y\in\mathbb F_p$ 使得 $(x,y)\in T_{LReLU}$。
- 对任意 $x\in\mathbb F_p$，在 $T_{ELU}$ 中至多存在一个 $y\in\mathbb F_p$ 使得 $(x,y)\in T_{ELU}$。
- 对任意 $u\in\mathbb F_p$，在 $T_{exp}$ 中至多存在一个 $v\in\mathbb F_p$ 使得 $(u,v)\in T_{exp}$。

指数表的输入语义固定为非负差分。对任意有效查询 $\Delta^+$，$T_{exp}$ 中的配对关系表示：

$ExpMap(\Delta^+)=QuantExp(-\Delta^+)$

即 Softmax 稳定化中使用的指数项是 $\exp(Z-M)$ 的量化结果，而不是 $\exp(M-Z)$ 的量化结果。

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

舍入方式（向最近整数舍入或截断）、乘法后的重缩放规则也全部为公开参数。

进一步固定输出层的量化与平均语义如下：

1. 对每个输出头 $s$，聚合前输出 $Y^{pre(s)}$ 在与偏置向量 $b^{(out,s)}$ 相加之前，必须已经被公开量化规则对齐到与 $b^{(out,s)}$ 相同的输出头尺度 $S_{out,head}$。若注意力聚合自然产出的尺度是 $S_{out,pre}$，则必须先施加公开的对齐 / 回缩算子：$\widetilde Y^{pre(s)} = Rescale_{out}\!\big(Y^{pre(s)}; S_{out,pre}\rightarrow S_{out,head}\big)$

   然后将输出头内部的偏置相加正式定义为：$Y^{(s)}=\widetilde Y^{pre(s)}+b^{(out,s)}$ 

   当且仅当 $quant\_cfg\_id$ 指定 $S_{out,pre}=S_{out,head}$ 时，$Rescale_{out}$ 定义为恒等映射；否则 $Rescale_{out}$ 必须按 $quant\_cfg\_id$ 唯一指定的舍入 / 截断规则执行，不存在第二种解释。

2. 最终多头平均必须在共同尺度上执行。本文协议固定要求最终平均发生在输出头共同尺度 $S_{out,head}$ 上，并且最终 logits 继续保持这一共同尺度，即 $S_{out}=S_{out,head}$，因而先在域内计算：$Y^{sum}_{i,c}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}$，再乘以公开整数 $K_{out}$ 在 $\mathbb F_p$ 中的逆元 $K_{out}^{-1}$，得到域内平均值与最终 logits：$Y_{i,c}=Y_{i,c}^{avg}=K_{out}^{-1}\cdot Y^{sum}_{i,c}$

   换言之，本协议严格禁止在最终平均之后再额外引入第二个平均后重缩放子步骤。本文协议中，最终平均的唯一正式语义就是上式；任何需要平均后重缩放的方案都必须定义为新的协议版本。

3. 上述 $Rescale_{out}$、舍入方式，以及乘以 $K_{out}^{-1}$ 的平均语义，全部必须并入 $quant\_cfg\_id$ 所标识的公开量化配置；证明者与验证者不得各自选择不同的平均语义。

### 0.5 模型参数

#### 0.5.1 隐藏层参数族

对任意隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，以及该层中的每个注意力头 $r\in\{0,1,\ldots,K_{hid}^{(\ell)}-1\}$，定义：

- 投影矩阵 $W^{(\ell,r)}\in\mathbb F_p^{d_{in}^{(\ell)}\times d_h^{(\ell)}}$
- 源方向注意力向量 $a_{src}^{(\ell,r)}\in\mathbb F_p^{d_h^{(\ell)}}$
- 目标方向注意力向量 $a_{dst}^{(\ell,r)}\in\mathbb F_p^{d_h^{(\ell)}}$

在后文“单个隐藏层模板”的数学推导中，统一把 $W^{(\ell,r)},a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}$，记作 $W^{(r)},a_{src}^{(r)},a_{dst}^{(r)}$。这一重写只服务于公式书写；正式协议对象名、承诺标签、评值标签与子证明标签都必须保留完整层指标 $(\ell,r)$。

#### 0.5.2 输出层参数

对最终输出层中的每个输出注意力头 $s\in\{0,1,\ldots,K_{out}-1\}$，定义：

- 输出投影矩阵 $W^{(out,s)}\in\mathbb F_p^{d_{in}^{(L)}\times C}$
- 输出层源方向注意力向量 $a_{src}^{(out,s)}\in\mathbb F_p^C$
- 输出层目标方向注意力向量 $a_{dst}^{(out,s)}\in\mathbb F_p^C$
- 输出层偏置向量 $b^{(out,s)}\in\mathbb F_p^C$

其中 $d_{in}^{(L)}=d_{cat}^{(L-1)}$，也就是最终输出层接收最后一个隐藏层拼接后的表示。

本文采用把 GAT 注意力向量按源/目标方向分拆的等价写法：$\mathbf a^{(out,s)\top}[u\,\|\,v] = a_{src}^{(out,s)\top}u + a_{dst}^{(out,s)\top}v$。其中 $[u\,\|\,v]$ 表示拼接。该写法与 GAT 前向语义等价。

所有模型参数在多次证明中固定，因此统一预处理进入模型验证键 $VK_{model}$。在多层实例中，$VK_{model}$ 必须同时包含：

- 全部隐藏层参数族：$\{W^{(\ell,r)},a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}\}_{\ell,r}$；
- 最终输出层参数族：$\{W^{(out,s)},a_{src}^{(out,s)},a_{dst}^{(out,s)},b^{(out,s)}\}_{s=0}^{K_{out}-1}$。

### 0.6 全部中间变量

#### 0.6.1 原始特征

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 与每个输入维索引 $j\in\{0,1,\ldots,d_{in}-1\}$，定义：$H_{i,j}=T_H[I_i,j]$，于是：$H\in\mathbb F_p^{N\times d_{in}}$

#### 0.6.2 单个隐藏层模板中第 $r$ 个注意力头的节点域变量

以下对象对应某个固定隐藏层 $\ell$ 的单层模板；在该模板中，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$，定义：

1. 投影结果：$H'^{(r)}\in\mathbb F_p^{N\times d_h}, \quad H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}$

2. 节点域源注意力：$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)}$

3. 节点域目标注意力：$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}$

4. 节点域压缩特征：$H^{\star(r)}\in\mathbb F_p^N, \quad H_i^{\star(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}(\xi^{(r)})^j$

5. 节点域组最大值：$M^{(r)}\in\mathbb F_p^N \quad M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}$

6. 节点域分母：$Sum^{(r)}\in\mathbb F_p^N \quad Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}$

7. 节点域逆元：$inv^{(r)}\in\mathbb F_p^N \quad inv_i^{(r)}=(Sum_i^{(r)})^{-1}$

8. 聚合前隐藏矩阵：$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}$

9. 聚合前压缩特征：$H_{agg,pre,i}^{\star(r)}\in\mathbb F_p^N \quad H_{agg,pre,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,pre,i,j}^{(r)}(\xi^{(r)})^j$

10. ELU 后隐藏矩阵：$H_{agg}^{(r)}\in\mathbb F_p^{N\times d_h} \quad H_{agg,i,j}^{(r)}=ELU(H_{agg,pre,i,j}^{(r)})$

11. ELU 后压缩特征：$H_{agg}^{\star(r)}\in\mathbb F_p^N \quad H_{agg,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,i,j}^{(r)}(\xi^{(r)})^j$

#### 0.6.3 单个隐藏层模板中第 $r$ 个注意力头的边域变量

以下对象对应某个固定隐藏层 $\ell$ 的单层模板；在该模板中，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$，定义：

1. 源注意力广播：$E_{src}^{edge(r)}\in\mathbb F_p^E \qquad E_{src,k}^{edge(r)}=E_{src,src(k)}^{(r)}$

2. 目标注意力广播：$E_{dst}^{edge(r)}\in\mathbb F_p^E \qquad E_{dst,k}^{edge(r)}=E_{dst,dst(k)}^{(r)}$

3. 源压缩特征广播：$H_{src}^{\star,edge(r)}\in\mathbb F_p^E \qquad H_{src,k}^{\star,edge(r)}=H_{src(k)}^{\star(r)}$

4. 聚合前压缩特征广播：$H_{agg,pre}^{\star,edge(r)}\in\mathbb F_p^E \qquad H_{agg,pre,k}^{\star,edge(r)}=H_{agg,pre,dst(k)}^{\star(r)}$

5. 聚合后压缩特征广播：$H_{agg}^{\star,edge(r)}\in\mathbb F_p^E \qquad H_{agg,k}^{\star,edge(r)}=H_{agg,dst(k)}^{\star(r)}$

6. 线性打分：$S^{(r)}\in\mathbb F_p^E \qquad S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)}$

7. LeakyReLU 后打分：$Z^{(r)}\in\mathbb F_p^E \qquad Z_k^{(r)}=LReLU(S_k^{(r)})$

8. 最大值广播：$M^{edge(r)}\in\mathbb F_p^E \qquad M_k^{edge(r)}=M_{dst(k)}^{(r)}$

9. 非负差分：$\Delta^{+(r)}\in\mathbb F_p^E \qquad \Delta_k^{+(r)}=M_k^{edge(r)}-Z_k^{(r)}$

10. 指数输出：$U^{(r)}\in\mathbb F_p^E \qquad U_k^{(r)}=ExpMap(\Delta_k^{+(r)})=QuantExp(-\Delta_k^{+(r)})$

11. 分母广播：$Sum^{edge(r)}\in\mathbb F_p^E \qquad Sum_k^{edge(r)}=Sum_{dst(k)}^{(r)}$

12. 逆元广播：$inv^{edge(r)}\in\mathbb F_p^E \qquad inv_k^{edge(r)}=inv_{dst(k)}^{(r)}$

13. 归一化权重：$\alpha^{(r)}\in\mathbb F_p^E \qquad \alpha_k^{(r)}=U_k^{(r)}\cdot inv_k^{edge(r)}$

14. 聚合前压缩加权特征：$\widehat v_{pre}^{\star(r)}\in\mathbb F_p^E \qquad \widehat v_{pre,k}^{\star(r)}=\alpha_k^{(r)}H_{src,k}^{\star,edge(r)}$

#### 0.6.4 单个隐藏层模板的拼接阶段变量

定义拼接结果 $H_{cat}\in\mathbb F_p^{N\times d_{cat}}$，其中对任意 $i\in\{0,1,\ldots,N-1\}$、$r\in\{0,1,\ldots,K_{hid}-1\}$、$j\in\{0,1,\ldots,d_h-1\}$，都有 $H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}$

定义拼接压缩挑战：$\xi_{cat}\in\mathbb F_p$

定义拼接压缩特征：$H_{cat}^{\star}\in\mathbb F_p^N \quad H_{cat,i}^{\star}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}\xi_{cat}^m$

#### 0.6.5 输出层变量

输出层有两类共享维：一类是输入共享维 $d_{cat}$，另一类是类别共享维 $C$。因此我们不仅需要 $d_{cat}$ 域，还需要显式引入类别共享域。

定义输出层类别压缩挑战：$\xi_{out}\in\mathbb F_p$

对每个输出头 $s\in\{0,1,\ldots,K_{out}-1\}$，定义：

1. 输出投影结果：$Y'^{(s)}\in\mathbb F_p^{N\times C}, \qquad Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out,s)}$
2. 输出层源注意力：$E_{src}^{(out,s)}\in\mathbb F_p^N, \qquad E_{src,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{src,c}^{(out,s)}$
3. 输出层目标注意力：$E_{dst}^{(out,s)}\in\mathbb F_p^N, \qquad E_{dst,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{dst,c}^{(out,s)}$
4. 输出层源注意力广播：$E_{src}^{edge(out,s)}\in\mathbb F_p^E, \qquad E_{src,k}^{edge(out,s)}=E_{src,src(k)}^{(out,s)}$
5. 输出层目标注意力广播：$E_{dst}^{edge(out,s)}\in\mathbb F_p^E, \qquad E_{dst,k}^{edge(out,s)}=E_{dst,dst(k)}^{(out,s)}$
6. 输出层线性打分：$S^{(out,s)}\in\mathbb F_p^E, \qquad S_k^{(out,s)}=E_{src,k}^{edge(out,s)}+E_{dst,k}^{edge(out,s)}$
7. 输出层 LeakyReLU 后打分：$Z^{(out,s)}\in\mathbb F_p^E, \qquad Z_k^{(out,s)}=LReLU(S_k^{(out,s)})$
8. 输出层组最大值：$M^{(out,s)}\in\mathbb F_p^N, \qquad M_i^{(out,s)}=\max\{Z_k^{(out,s)}\mid dst(k)=i\}$
9. 输出层最大值广播：$M^{edge(out,s)}\in\mathbb F_p^E, \qquad M_k^{edge(out,s)}=M_{dst(k)}^{(out,s)}$
10. 输出层非负差分：$\Delta^{+(out,s)}\in\mathbb F_p^E, \qquad \Delta_k^{+(out,s)}=M_k^{edge(out,s)}-Z_k^{(out,s)}$
11. 输出层指数输出：$U^{(out,s)}\in\mathbb F_p^E, \qquad U_k^{(out,s)}=ExpMap(\Delta_k^{+(out,s)})=QuantExp(-\Delta_k^{+(out,s)})$
12. 输出层分母：$Sum^{(out,s)}\in\mathbb F_p^N, \qquad Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)}$
13. 输出层分母广播：$Sum^{edge(out,s)}\in\mathbb F_p^E, \qquad Sum_k^{edge(out,s)}=Sum_{dst(k)}^{(out,s)}$
14. 输出层逆元：$inv^{(out,s)}\in\mathbb F_p^N, \qquad inv_i^{(out,s)}=(Sum_i^{(out,s)})^{-1}$
15. 输出层逆元广播：$inv^{edge(out,s)}\in\mathbb F_p^E, \qquad inv_k^{edge(out,s)}=inv_{dst(k)}^{(out,s)}$
16. 输出层归一化权重：$\alpha^{(out,s)}\in\mathbb F_p^E, \qquad \alpha_k^{(out,s)}=U_k^{(out,s)}\cdot inv_k^{edge(out,s)}$
17. 输出层投影结果的类别压缩：$Y'^{\star(s)}\in\mathbb F_p^N, \qquad Y_i'^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}\xi_{out}^c$
18. 输出层投影结果的边域广播压缩：$Y'^{\star,edge(s)}\in\mathbb F_p^E, \qquad Y_{k}'^{\star,edge(s)}=Y_{src(k)}'^{\star(s)}$
19. 输出层压缩加权边特征：$\widehat y^{\star(s)}\in\mathbb F_p^E, \qquad \widehat y_k^{\star(s)}=\alpha_k^{(out,s)}Y_{k}'^{\star,edge(s)}$
20. 输出层聚合前输出：$Y^{pre(s)}\in\mathbb F_p^{N\times C}, \qquad Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}$
21. 输出层聚合前输出的类别压缩：$Y^{pre,\star(s)}\in\mathbb F_p^N, \qquad Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{pre(s)}\xi_{out}^c$
22. 输出层聚合前输出压缩广播：$Y^{pre,\star,edge(s)}\in\mathbb F_p^E, \qquad Y_k^{pre,\star,edge(s)}=Y_{dst(k)}^{pre,\star(s)}$
23. 尺度对齐后的聚合前输出：$\widetilde Y^{pre(s)}\in\mathbb F_p^{N\times C}$，其中：$\widetilde Y_{i,c}^{pre(s)}=Rescale_{out}\!\big(Y_{i,c}^{pre(s)}; S_{out,pre}\rightarrow S_{out,head}\big)$

24. 尺度对齐后的聚合前输出类别压缩：$\widetilde Y^{pre,\star(s)}\in\mathbb F_p^N, \qquad \widetilde Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}\widetilde Y_{i,c}^{pre(s)}\xi_{out}^c$
25. 输出层加偏置后的头输出：$Y^{(s)}\in\mathbb F_p^{N\times C}, \qquad Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}$

26. 输出层头输出的类别压缩：$Y^{\star(s)}\in\mathbb F_p^N, \qquad Y_i^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{(s)}\xi_{out}^c$
27. 输出层头输出压缩广播：$Y^{\star,edge(s)}\in\mathbb F_p^E, \qquad Y_k^{\star,edge(s)}=Y_{dst(k)}^{\star(s)}$
28. 最终输出：先定义 $Y^{sum}\in\mathbb F_p^{N\times C}$ 与域内平均 $Y^{avg}\in\mathbb F_p^{N\times C}$：$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}, \qquad Y_{i,c}^{avg}=K_{out}^{-1}\cdot Y_{i,c}^{sum}$

其中 $K_{out}^{-1}$ 是整数 $K_{out}$ 在 $\mathbb F_p$ 中的逆元。本文协议要求最终 logits 与输出头使用同一共同尺度 $S_{out}=S_{out,head}$，因此正式定义 $Y\in\mathbb F_p^{N\times C}, \quad Y_{i,c}=Y_{i,c}^{avg}$ 

29. 最终输出的类别压缩：$Y^{\star}\in\mathbb F_p^N, \qquad Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c$
30. 最终输出压缩广播：$Y^{\star,edge}\in\mathbb F_p^E, \qquad Y_k^{\star,edge}=Y_{dst(k)}^{\star}$

其中 $Y^{\star}$ 与 $Y^{\star,edge}$ 均由公共输出 $Y$、公开挑战 $\xi_{out}$ 与公开路由 $R_{dst}$ 唯一确定，不作为私有压缩张量承诺，也不提交张量评值证明。

### 0.7 张量域、公共辅助张量与展平规则

#### 0.7.1 张量域总表

本文协议使用七类逻辑域名称。协议中的维度、对象标签与数据加载口径必须与下列名称一致：

1. 特征检索域：$\mathbb H_{FH}$
2. 边域：$\mathbb H_{edge}$
3. 输入共享维域：$\mathbb H_{in}$
4. 隐藏层单头共享维域：$\mathbb H_{d_h}$
5. 拼接共享维域：$\mathbb H_{cat}$
6. 输出层类别共享维域：$\mathbb H_C$
7. 节点域：$\mathbb H_N$

这些名称表示张量各维度的逻辑索引集合及其补零后的 MLE 维度。协议对象标签必须使用这些域名。

对任意逻辑长度 $d$，记其二次幂补零长度为 $\widehat d=2^{\lceil\log_2 d\rceil}$。当 $d=1$ 时，约定 $\widehat d=1$。所有 MLE 张量均在补零后的布尔超立方上定义，填充区取零并由公共有效区张量屏蔽。

#### 0.7.2 公共有效区张量

有效区选择器解释为公共二值张量。定义：

1. 边域有效区张量 $Q_{edge}^{valid}$；
2. 节点域有效区张量 $Q_N$；
3. 每个隐藏层的输入共享维有效区张量 $Q_{in}^{valid,(\ell)}$；
4. 每个隐藏层的单头共享维有效区张量 $Q_{d_h}^{valid,(\ell)}$；
5. 每个隐藏层的拼接共享维有效区张量 $Q_{cat}^{valid,(\ell)}$；
6. 输出层输入共享维有效区张量 $Q_{cat}^{valid,(out)}$；
7. 输出类别共享维有效区张量 $Q_C^{valid}$；
8. 公共输出有效区张量 $Q_Y$，其中 $Q_Y(i,c)=Q_N(i)\cdot Q_C^{valid}(c)$；
9. 对每个查表函数 $f\in\{LReLU,ELU,exp\}$ 的表项有效区向量 $Q_f^{tbl}$；
10. 对每个查表函数 $f\in\{LReLU,ELU,exp\}$ 以及每个查询族标号 $\tau$ 的查询有效区向量 $Q_{f,\tau}^{qry}$。其中 $\tau$ 必须唯一绑定层编号、头编号或输出头编号、查询张量域、真实查询长度与补零长度。

这些张量均由验证者根据公开元数据本地重建，不由证明者作为动态见证提交。即使不同层的有效区取值相同，也必须按固定标签分别吸入 transcript，不能合并标签。

#### 0.7.3 公共拓扑张量

边序列、节点分块和 batch 边界语义由公开输入本地生成。公共张量包括：

- $src\in [N]^E$；
- $dst\in [N]^E$；
- $node\_gid\in [G_{batch}]^N$；
- $edge\_gid\in [G_{batch}]^E$；
- $Q_{graph\_new}^{edge},Q_{graph\_end}^{edge}$；
- $Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$；
- 节点与边的枚举张量 $Idx_N,Idx_{edge},Idx_{in},Idx_{d_h},Idx_{cat},Idx_C$。

为了用 sumcheck 证明广播和聚合关系，进一步把公开拓扑写成稀疏选择矩阵：

$R_{src}\in\{0,1\}^{E\times N},\quad R_{src}[k,i]=1\iff i=src(k)$

$R_{dst}\in\{0,1\}^{E\times N},\quad R_{dst}[k,i]=1\iff i=dst(k)$

于是源广播、目标广播、按目标节点求和与目标路由统一表示为：

$V^{edge}=R_{src}V,\qquad U^{edge}=R_{dst}U,\qquad Agg=R_{dst}^{\top}Z^{edge}$

其中 $R_{src}$ 与 $R_{dst}$ 是公共稀疏张量，不隐藏、不承诺，但必须在 transcript 中以公共拓扑哈希的形式绑定。

公共稀疏路由张量的 MLE 评值不得通过具体化 $E\times N$ 稠密矩阵计算。令 $\beta_{\widehat E}(u,k)$ 表示边索引补零域上的多线性等值基函数，$\beta_{\widehat N}(v,i)$ 表示节点索引补零域上的多线性等值基函数，则：

$\widetilde R_{src}(u_E,u_N)=\sum_{k=0}^{E-1}\beta_{\widehat E}(u_E,k)\,\beta_{\widehat N}(u_N,src(k))$

$\widetilde R_{dst}(u_E,u_N)=\sum_{k=0}^{E-1}\beta_{\widehat E}(u_E,k)\,\beta_{\widehat N}(u_N,dst(k))$

在所有 sumcheck 末端评值中，$R_{src}$ 与 $R_{dst}$ 的评值均按上述公式由公开 $src,dst$ 直接计算。该公式只依赖公开 $src,dst$，不要求把 $E\times N$ 稠密矩阵作为协议对象。

#### 0.7.4 批处理分组语义

多图批处理时，softmax 分组严格限制在图内。$Q_{grp\_new}^{edge}$ 与 $Q_{grp\_end}^{edge}$ 是唯一正式分组边界标签，并作为公共辅助张量参与下列张量承诺与 sumcheck 语句：

1. 每个目标组的最大值 $M_i$ 与所有同组 $Z_k$ 的范围关系；
2. 每个目标组的分母 $Sum_i=\sum_{dst(k)=i}U_k$；
3. 每个目标组的归一化权重 $\alpha_k=U_k\cdot inv_{dst(k)}$；
4. 每个目标组的聚合 $Agg_i=\sum_{dst(k)=i}\alpha_k V_{src(k)}$。

这些语句通过公共稀疏矩阵 $R_{dst}$ 与对应的辅助有效性证明表达，而不是通过逐边递推状态机表达。

#### 0.7.5 展平规则

展平规则固定如下。对任意矩阵 $M\in\mathbb F_p^{r\times c}$，定义行优先展平索引：

$\operatorname{flat}_{r,c}(i,j)=i\cdot c+j$

特别地：

- 对 $N\times d_{in}$ 矩阵，使用索引 $i\cdot d_{in}+j$；
- 对 $N\times d_h$ 矩阵，使用索引 $i\cdot d_h+j$；
- 对 $N\times d_{cat}$ 矩阵，使用索引 $i\cdot d_{cat}+m$；
- 对 $N\times C$ 矩阵，使用索引 $i\cdot C+c$。

在 MLE 中，多维索引先按上述行优先规则确定一维位置，再映射为补零长度上的二进制索引。所有承诺、评值证明、sumcheck 末端评值与序列化字段都必须使用同一展平规则。

#### 0.7.6 MLE 编码

对矩阵乘法：

$A\in\mathbb F_p^{m\times \ell},\quad B\in\mathbb F_p^{\ell\times n},\quad C=A\cdot B\in\mathbb F_p^{m\times n}$

矩阵乘法关系采用 MLE sumcheck：

$\widetilde C(u_i,u_j)=\sum_{t\in [\ell]}\widetilde A(u_i,t)\widetilde B(t,u_j)$

其中 $u_i,u_j$ 是验证者挑战得到的随机点评值，$t$ 在补零后的共享维布尔超立方上求和。向量点积、注意力打分、压缩特征、拼接、输出平均均按同一 MLE 口径表达。

对 $N_{inst}$ 个同形操作实例，例如不同层、不同头、不同输出头中形状相同的矩阵乘法或 Hadamard 乘法，使用 FAC4DNN 的堆叠方式，把实例索引并入额外维度：

$\mathcal A\in\mathbb F_p^{N_{inst}\times m\times \ell},\quad \mathcal B\in\mathbb F_p^{N_{inst}\times \ell\times n},\quad \mathcal C\in\mathbb F_p^{N_{inst}\times m\times n}$

并证明：

$\widetilde{\mathcal C}(w,u_i,u_j)=\sum_{t\in[\ell]}\widetilde{\mathcal A}(w,u_i,t)\widetilde{\mathcal B}(w,t,u_j)$

其中 $w$ 是实例维挑战。该写法只作用于证明聚合；每个实例的 GAT 前向计算均按第 0.1-0.6 节定义执行。

#### 0.7.7 FAC4DNN 聚合承诺规则

FAC4DNN 聚合只改变证明组织方式，不改变 GAT 前向计算顺序。证明者必须先按见证生成部分逐层、逐头生成每个单实例张量，再把形状相同且属于同一类代数关系的实例堆叠为聚合张量。

设某一类操作为 $op$，其同形实例集合为 $\mathcal I_{op}$。若每个实例都有输入张量 $X_{op,t}^{(a)}$ 与输出张量 $Y_{op,t}$，且这些张量的形状分别一致，则定义聚合张量：

$\mathcal X_{op}^{(a)}[t,\cdot]=X_{op,t}^{(a)},\qquad \mathcal Y_{op}[t,\cdot]=Y_{op,t}\qquad (t\in\mathcal I_{op})$

实例顺序由公共元数据唯一确定，按层编号、头编号、操作在模板中的位置和输入张量编号依次排序。聚合张量单独承诺：

$Com(\mathcal X_{op}^{(a)}),\qquad Com(\mathcal Y_{op})$

聚合 sumcheck 中出现的 $\widetilde{\mathcal X}_{op}^{(a)}$ 与 $\widetilde{\mathcal Y}_{op}$ 必须来自这些聚合承诺。若某个张量既作为单实例张量参与约束，又作为聚合张量的切片参与约束，则必须证明切片一致性：

$\widetilde{\mathcal X}_{op}^{(a)}(e_t,u)=\widetilde X_{op,t}^{(a)}(u)$

其中 $e_t$ 是实例编号 $t$ 的公开布尔编码。若某个张量只通过聚合张量参与所有约束，则可只保留聚合承诺；若它还被非聚合约束引用，则必须同时保持单实例承诺并给出切片一致性证明。

形状不同的实例不得放入同一个聚合张量。聚合描述、实例顺序、输入输出张量标签和切片绑定关系都属于证明对象的公开结构，并被吸入 transcript。

#### 0.7.8 张量绑定与评值子证明

所有参与证明的私有张量都必须先承诺，再在 sumcheck 末端用评值证明绑定。最终绑定子证明族记为：

$\Pi_{bind}^{zkDL}=\big(\pi_{eval}^{feat},\{\pi_{eval}^{hidden,(\ell,r)}\}_{\ell,r},\{\pi_{eval}^{concat,(\ell)}\}_{\ell},\{\pi_{eval}^{out,s}\}_{s},\pi_{eval}^{out,avg},\Pi_{agg},\Pi_{aux}^{GAT}\big)$

其中：

- $\pi_{eval}^{feat}$ 负责特征检索张量 $H$ 与静态特征表 $T_H$ 的评值绑定；
- $\pi_{eval}^{hidden,(\ell,r)}$ 负责第 $\ell$ 层第 $r$ 个隐藏头的投影、源注意力、目标注意力、Softmax 相关张量、聚合与 ELU 输出；
- $\pi_{eval}^{concat,(\ell)}$ 负责该层拼接张量与各头输出的绑定；
- $\pi_{eval}^{out,s}$ 负责第 $s$ 个输出头的投影、注意力、Softmax、聚合、尺度对齐与偏置输出；
- $\pi_{eval}^{out,avg}$ 负责最终公开输出 $Y$ 与各 $Y^{(s)}$ 的平均一致性；其中 $Y$ 的 MLE 评值由验证者从公共输入本地计算，不作为私有动态张量提交评值证明；
- $\Pi_{agg}$ 负责 FAC4DNN 聚合张量与单实例张量的切片一致性；
- $\Pi_{aux}^{GAT}$ 负责 LeakyReLU、ELU、指数、范围、最大值、重缩放与逆元等非纯乘加关系的辅助有效性证明。

每个子证明都必须使用独立的域分离标签初始化内部 transcript。

### 0.8 Padding、非线性辅助证明、Softmax 可行性与协议边界

#### 0.8.1 Padding 规则

对任意逻辑张量维度，若真实长度为 $L_{\mathcal D}$、MLE 补零长度为 $\widehat L_{\mathcal D}$，则所有离散列、矩阵和高阶张量必须补到长度 $\widehat L_{\mathcal D}$。

对所有填充位置，统一规定：

- 见证值补零；
- 静态表值补零；
- 公共辅助张量补零；
- 承诺输入序列补零；
- MLE 求和时由公共有效区张量屏蔽无效位置；
- 所有边域私有张量 $X_{edge}$ 必须满足 $(1-Q_{edge}^{valid})\odot X_{edge}=0$；
- 所有节点域私有张量 $X_N$ 必须满足 $(1-Q_N)\odot X_N=0$；
- 所有共享维或类别维辅助张量必须由对应有效区张量屏蔽，填充区固定为 $0$；
- 公共输出 $Y\in\mathbb F_p^{N\times C}$ 按节点域与类别域共同有效区解释。令 $Q_Y(i,c)=Q_N(i)\cdot Q_C^{valid}(c)$，则 $Y$ 在 $Q_Y=0$ 的位置固定为 $0$，其 MLE 评值按同一补零规则由验证者本地计算；
- 协议不得把填充区改写成其他等价但不同的随机值或哨兵值。

上述填充区零约束属于正式约束族。若某个张量只在更大的聚合张量中出现，也必须在聚合张量的对应切片上满足同一填充规则。

#### 0.8.2 非线性辅助证明、查表成员关系与零分母约束

非线性关系使用辅助有效性证明。每类非线性对象都必须提交对应辅助张量，并通过二值性、范围重构、查表成员关系和函数关系 sumcheck 证明其正确性。

需要覆盖的辅助证明包括：

1. $\Pi_{LReLU}$：证明 $Z=LReLU(S)$；
2. $\Pi_{ELU}$：证明 $H_{agg}=ELU(H_{agg,pre})$；
3. $\Pi_{exp}$：证明 $U=ExpMap(\Delta^+)=QuantExp(-\Delta^+)$；
4. $\Pi_{range}$：证明 $\Delta^+$、重缩放余数、最大值差分和其他有符号解释相关对象在公开位宽内；
5. $\Pi_{inv}$：证明有效区内 $Q_N\odot(Sum\odot inv-1)=0$，并证明节点填充区的 $Sum$ 与 $inv$ 为 $0$；
6. $\Pi_{rescale}$：证明各重缩放对象满足公开量化配置指定的舍入 / 截断规则；
7. $\Pi_{tbl}$：证明所有依赖静态函数表的输入输出对属于对应静态表承诺。

对每个辅助位张量 $bit$，均要求：

$bit\odot(bit-1)=0$

并由对应的有效区选择器约束填充区为 $0$。非线性辅助证明使用辅助输入把非算术函数关系表达为算术关系；本文协议的辅助对象包括 GAT 前向所需的 LeakyReLU、ELU、ExpMap、range、inverse 与 rescale 关系。

静态表查找统一采用张量化表成员关系证明。设某个静态二列表为：

$T_f=\{(x_t,y_t)\}_{t=0}^{T_f^{len}-1}$

其中 $f\in\{LReLU,ELU,exp\}$。令 $\widehat T_f$ 为该表承诺域的补零长度。表项有效区向量 $Q_f^{tbl}\in\{0,1\}^{\widehat T_f}$ 在真实表项上取 $1$，在补零表项上取 $0$；补零表项的 $x_t,y_t$ 固定为 $0$。对一组查询 $(X_q,V_q)_{q=0}^{Q_{f,\tau}-1}$，令 $\tau$ 表示该查询族标号，并定义公共有效区向量 $Q_{f,\tau}^{qry}\in\{0,1\}^{Q_{f,\tau}}$。证明者提交选择矩阵：

$Sel_{f,\tau}\in\{0,1\}^{Q_{f,\tau}\times \widehat T_f}$

并证明以下逐点关系：

$Sel_{f,\tau}[q,t]\cdot(Sel_{f,\tau}[q,t]-1)=0$

$(1-Q_{f,\tau}^{qry}[q])\cdot Sel_{f,\tau}[q,t]=0$

$Sel_{f,\tau}[q,t]\cdot(1-Q_f^{tbl}[t])=0$

$\sum_{t=0}^{\widehat T_f-1} Sel_{f,\tau}[q,t]=Q_{f,\tau}^{qry}[q]$

$X_q=\sum_{t=0}^{\widehat T_f-1}Sel_{f,\tau}[q,t]\cdot x_t$

$V_q=\sum_{t=0}^{\widehat T_f-1}Sel_{f,\tau}[q,t]\cdot y_t$

$(1-Q_{f,\tau}^{qry}[q])\cdot X_q=0,\qquad (1-Q_{f,\tau}^{qry}[q])\cdot V_q=0$

上述关系均用 MLE sumcheck 证明，并在 sumcheck 末端打开 $Com(Sel_{f,\tau})$、$Com(T_f)$、$Com(X)$ 与 $Com(V)$。静态表 $T_f$ 必须来自 $VK_{static}^{zkDL}$ 中对应承诺，不允许使用临时未绑定表。

选择矩阵查表是本文协议固定采用的查表方案。

范围检查固定采用位分解方式。对无符号对象 $x$，设其有效区向量为 $Q_x$，位宽为 $B_x$，位张量为 $bit_0,\ldots,bit_{B_x-1}$，则证明：

$bit_b\odot(bit_b-1)=0\quad(0\le b<B_x)$

$(1-Q_x)\odot bit_b=0\quad(0\le b<B_x)$

$Q_x\odot\left(x-\sum_{b=0}^{B_x-1}2^b bit_b\right)=0$

$(1-Q_x)\odot x=0$

对有符号对象 $x$，最高位采用负权重。设位宽为 $B_x$，则证明：

$Q_x\odot\left(x-\sum_{b=0}^{B_x-2}2^b bit_b+2^{B_x-1}bit_{B_x-1}\right)=0$

$(1-Q_x)\odot x=0$

其中等价地，$x$ 的有符号重构值为 $\sum_{b=0}^{B_x-2}2^b bit_b-2^{B_x-1}bit_{B_x-1}$。每个对象使用的 $Q_x$、$B_x$ 与最高位解释均由 $quant\_cfg\_id$ 固定。

重缩放关系采用商余数约束。对任意重缩放对象，设输入为 $x$，输出为 $y$，有效区为 $Q_{res}$，公开除数为 $\rho$，公开偏移为 $\delta$，余数为 $rem$。其中 $\rho,\delta$ 由 $quant\_cfg\_id$ 按该重缩放关系唯一确定。协议证明：

$Q_{res}\odot(x+\delta-\rho y-rem)=0$

$(1-Q_{res})\odot x=0,\qquad (1-Q_{res})\odot y=0,\qquad (1-Q_{res})\odot rem=0$

同时引入 $rem^{bd}=\rho-1-rem$，并证明：

$Q_{res}\odot(rem+rem^{bd}-(\rho-1))=0$

$(1-Q_{res})\odot rem^{bd}=0$

再通过无符号范围检查证明 $rem$ 与 $rem^{bd}$ 均在公开位宽内。该约束等价于 $0\le rem<\rho$。当该重缩放关系为恒等映射时，固定取 $\rho=1,\delta=0,rem=0,rem^{bd}=0$。

所有 Softmax 分母必须满足非零性：

$Sum_i^{(\ell,r)}\ne 0,\qquad Sum_i^{(out,s)}\ne 0$

本条件由公开预处理和量化可行性假设保证；证明层面通过有效区屏蔽后的逆元约束绑定：$Q_N\odot(Sum\odot inv-1)=0$，并同时要求 $(1-Q_N)\odot Sum=0$ 与 $(1-Q_N)\odot inv=0$。

#### 0.8.3 Softmax 可行性

要求每个有效节点至少有一条入边。公共输入构造阶段必须保证：

$\#\{k\mid dst(k)=i\}\ge 1\quad \forall i\in\{0,1,\ldots,N-1\}$

若原始图不满足该条件，则必须在公开预处理阶段显式加入自环后，才允许进入本文协议。

同时还必须满足：

1. 所有合法输入下的实数域最大分母严格小于模数 $p$；
2. 每个有效指数输出 $U_k$ 的量化值非零，因而每个有效节点的 $Sum_i$ 非零；
3. 所有辅助位分解的公开位宽足以覆盖本文协议量化配置可能产生的中间值。

自环、排序、量化和截断规则由公开配置唯一确定。

#### 0.8.4 协议复用边界

参数生成后，承诺公共参数、静态表承诺、模型承诺、公共拓扑重建规则、逻辑域补零长度以及非线性辅助子协议的公开配置可以跨多次证明复用。

上述复用不改变证明关系。单次证明必须仍然绑定本次的 $M_{pub}$、公开拓扑、动态张量承诺、sumcheck transcript 与张量评值证明。

#### 0.8.5 公开元数据与版本元数据字段表

为保证证明对象、静态表、模型参数与量化配置之间的一致性，证明对象中必须携带公开元数据 / 版本元数据。字段集合固定为：

1. $protocol\_id$：标识本文协议版本与 transcript 规则版本。
2. $proof\_scope\_id$：固定标识本文证明范围为 GAT 前向计算证明，不包含训练反向传播、梯度计算或参数更新语义。
3. $model\_arch\_id$：绑定总层数、隐藏层头数 / 维度、输出层头数、隐藏层 ELU、输出头偏置、最终多头平均与拼接维定义。
4. $model\_param\_id$：唯一标识 $VK_{model}^{zkDL}$ 对应的模型参数承诺集合。
5. $static\_table\_id$：唯一标识 $T_H,T_{LReLU},T_{ELU},T_{exp}$ 以及 $VK_{static}^{zkDL}$。
6. $quant\_cfg\_id$：唯一标识全部量化尺度、舍入规则、重缩放规则与符号解释规则。
7. $domain\_cfg$：固定定义为各逻辑域真实长度、补零长度和 MLE 维度计划。
8. $dim\_cfg$：固定定义为 $(G_{batch},N,E,N_{total},node\_ptr,edge\_ptr,L,\{d_{in}^{(\ell)}\}_{\ell=1}^{L},\{d_h^{(\ell)}\}_{\ell=1}^{L-1},\{d_{cat}^{(\ell)}\}_{\ell=1}^{L-1},C,B,K_{out})$。
9. $encoding\_id$：标识域元素字节序、承诺编码方式、张量展平顺序、MLE 挑战编码方式与 transcript 吸入顺序。
10. $padding\_rule\_id$：标识补零语义与公共有效区张量重建规则。
11. $tensor\_bound\_id$：标识本协议涉及的张量形状集合、补零形状、最大承诺长度、聚合实例维度与 sumcheck 轮数上界。
12. $commitment\_backend\_id$：标识张量承诺与评值证明语义。

上述字段是证明对象内部携带的公共字段。验证者必须从 $\pi_{GAT}^{zkDL}$ 中按固定顺序解析这些字段，并把它们纳入 transcript 与一致性检查。

#### 0.8.6 大图稀疏语义

面向大图数据集时，本文协议仅在公开真实边集上定义注意力打分、softmax、路由与聚合。禁止把 GAT 注意力改写为全图稠密两两注意力，也禁止把不同图的边混入同一 softmax 目标组。

排序、自环、局部子图抽取与多图 batch 分组属于公开预处理，并由 $M_{pub}$ 与 $\mathsf{Topo}$ 绑定。它们不改变 GAT 前向语义，只确定本次证明实例的公开图结构。

## 参数生成

### 1.1 输入

参数生成算法输入为：

- 安全参数 $\lambda$；
- 有限域模数 $p$；
- 局部子图规模上界 $(N,E)$；
- 全局节点数 $N_{total}$；
- 当前证明实例中的图个数 $G_{batch}$；
- 当前证明实例的节点 / 边前缀指针 $node\_ptr,edge\_ptr$；
- 总层数 $L$；
- 各隐藏层的结构配置 $\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1}$；
- 各层输入维配置 $\{d_{in}^{(\ell)}\}_{\ell=1}^{L}$，其中 $d_{in}^{(1)}=d_{in}$，且 $d_{in}^{(\ell+1)}=d_{cat}^{(\ell)}$；
- 输出类别数 $C$；
- 输出层头数 $K_{out}$；
- 范围检查位宽 $B$；
- 静态表 $T_H,T_{LReLU},T_{ELU},T_{exp}$；
- 全部隐藏层参数族 $\{W^{(\ell,r)},a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}\}_{\ell,r}$；
- 输出层参数族 $\{W^{(out,s)},a_{src}^{(out,s)},a_{dst}^{(out,s)},b^{(out,s)}\}_{s=0}^{K_{out}-1}$。

这些输入的含义、量化语义和公开 / 私有边界由本文协议固定。

### 1.2 输出

参数生成算法输出为：

- 张量承诺公共参数 $PP_{com}$；
- 张量承诺验证参数 $VP_{com}$；
- 静态表验证键 $VK_{static}^{zkDL}$；
- 模型验证键 $VK_{model}^{zkDL}$；
- 各逻辑域真实长度、补零长度和 MLE 维度计划；
- 公共拓扑张量重建规则；
- 非线性辅助证明所需的公开位宽、重构权重和表版本标识。

本文协议输出张量承诺公共参数、张量边界参数、静态表验证键与模型验证键。

### 1.3 张量域与 MLE 维度计划

为避免在多层 GAT 中为每一层重复维护不同长度的共享维张量域，本文继续采用全局共享域口径：

- 对全部隐藏层的输入共享维，统一用一个 $\mathbb H_{in}$；
- 对全部隐藏层单头输出共享维，统一用一个 $\mathbb H_{d_h}$；
- 对全部隐藏层拼接共享维与最终输出层输入共享维，统一用一个 $\mathbb H_{cat}$。

定义：

$d_{in}^{max}=\max\{d_{in}^{(1)},d_{in}^{(2)},\ldots,d_{in}^{(L)}\}$

$d_h^{max}=\max\{d_h^{(1)},d_h^{(2)},\ldots,d_h^{(L-1)}\}$

$d_{cat}^{max}=\max\{d_{cat}^{(1)},d_{cat}^{(2)},\ldots,d_{cat}^{(L-1)}\}$

各逻辑长度的补零长度定义为：

$\widehat n_{FH}=2^{\lceil\log_2 \max\{N_{total}d_{in}^{(1)},Nd_{in}^{(1)}\}\rceil}$

$\widehat n_{edge}=2^{\lceil\log_2 \max\{N,E,Nd_h^{max},|T_{LReLU}|,|T_{ELU}|,|T_{exp}|,2^B\}\rceil}$

$\widehat n_{in}=2^{\lceil\log_2 d_{in}^{max}\rceil}$

$\widehat n_{d_h}=2^{\lceil\log_2 d_h^{max}\rceil}$

$\widehat n_{cat}=2^{\lceil\log_2 d_{cat}^{max}\rceil}$

$\widehat n_C=2^{\lceil\log_2 C\rceil}$

$\widehat n_N=2^{\lceil\log_2 N\rceil}$

若某个长度本身已经是二次幂，则补零长度等于原长度。协议对象中必须记录真实长度和补零长度，不能只记录补零长度。

### 1.4 承诺公共参数初始化

初始化算法生成张量承诺所需的公共参数。公共参数必须能够承诺本文协议中出现的动态张量、静态张量、模型参数张量、辅助张量与 FAC4DNN 聚合张量，并支持对应的 MLE 随机点评值证明。

令 $\mathcal T_{shape}$ 表示由 $M_{pub}$、模型结构、静态表版本、量化配置和 FAC4DNN 聚合规则唯一确定的张量形状集合。该集合至少覆盖：

1. 静态表张量 $T_H,T_{LReLU},T_{ELU},T_{exp}$；
2. 全部隐藏层参数与输出层参数；
3. 输入、隐藏层、拼接阶段和输出层的全部动态前向张量及其压缩张量；
4. 位分解、查表选择、最大值代表边选择、重缩放余数和逆元等辅助张量；
5. FAC4DNN 聚合张量及必要的切片一致性对象。

对任意 $T\in\mathcal T_{shape}$，记其补零后的元素个数为 $|T|_{pad}$。定义最大承诺长度：

$D_{tensor}^{max}=\max_{T\in \mathcal T_{shape}} |T|_{pad}$

承诺公共参数 $PP_{com}$ 必须覆盖长度 $D_{tensor}^{max}$ 的张量承诺，并支持把矩阵和高阶张量按第 0.7.5 节的展平规则解释为承诺输入。$tensor\_bound\_id$ 必须绑定 $\mathcal T_{shape}$、最大承诺长度、补零维度和 sumcheck 轮数上界。

### 1.5 静态表与模型承诺

#### 1.5.1 静态表承诺

静态表继续包括：

$T_H,T_{LReLU},T_{ELU},T_{exp}$

静态表作为静态张量承诺：

$Com(T_H),\quad Com(T_{LReLU}),\quad Com(T_{ELU}),\quad Com(T_{exp})$

其中二列表 $T_{LReLU},T_{ELU},T_{exp}$ 使用 $(x,y)$ 行语义。范围检查由第 0.8.2 节的位分解约束给出。

#### 1.5.2 模型承诺

对每个隐藏层 $\ell$ 与隐藏头 $r$，承诺：

$Com(W^{(\ell,r)}),\quad Com(a_{src}^{(\ell,r)}),\quad Com(a_{dst}^{(\ell,r)})$

对每个输出头 $s$，承诺：

$Com(W^{(out,s)}),\quad Com(a_{src}^{(out,s)}),\quad Com(a_{dst}^{(out,s)}),\quad Com(b^{(out,s)})$

$VK_{model}^{zkDL}$ 按固定层序、头序和输出头顺序拼接这些承诺。任何协议不得因为 FAC4DNN 聚合而重排模型参数承诺。

#### 1.5.3 静态表验证键

$VK_{static}^{zkDL}$ 固定为：

$VK_{static}^{zkDL}=\{Com(T_H),Com(T_{LReLU}),Com(T_{ELU}),Com(T_{exp})\}$

静态表承诺在多次证明之间复用。单次证明对象中只携带这些承诺的版本标识和必要的公开哈希，不重复携带完整表内容。

### 1.6 公共拓扑张量与辅助公开张量

验证者根据公共输入本地重建：

1. 源索引张量 $src$；
2. 目标索引张量 $dst$；
3. 边有效区张量 $Q_{edge}^{valid}$；
4. 节点有效区张量 $Q_N$；
5. 各层共享维有效区张量；
6. 输出类别有效区张量；
7. 公共输出有效区张量 $Q_Y$；
8. 节点、边、共享维、类别枚举张量；
9. 特征检索表端 / 查询端的行列辅助张量；
10. 特征检索公开选择矩阵 $R_I$，其中 $R_I[i,v]=1\iff v=I_i$；
11. $node\_gid,edge\_gid$；
12. $Q_{graph\_new}^{edge},Q_{graph\_end}^{edge},Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$；
13. 稀疏路由矩阵 $R_{src},R_{dst}$；
14. 每个查表函数的表项有效区向量 $Q_f^{tbl}$，其中 $f\in\{LReLU,ELU,exp\}$；
15. 每个查表查询族的公共有效区向量 $Q_{f,\tau}^{qry}$，其中 $f\in\{LReLU,ELU,exp\}$，查询族标号 $\tau$ 由对应层、头、张量域、真实查询长度和补零长度唯一确定。

这些对象不作为动态承诺对象由证明者提交；验证者本地重建后吸入 transcript。若 $G_{batch}=1$，batch-aware 标签仍然必须存在并被吸入 transcript。$R_I,R_{src},R_{dst}$ 均按稀疏公共对象处理，不得作为稠密矩阵写入协议对象。

## 见证生成

### 2.0 总体顺序与协议约束

本协议执行顺序按 GAT 前向依赖展开，承诺和证明轨迹使用张量承诺与 MLE sumcheck 后端。协议执行顺序固定分为三阶段：

1. 前向阶段：只计算神经网络明文对象；
2. 辅助张量阶段：按固定 Fiat-Shamir 顺序生成挑战，并构造 MLE 评值点、非线性辅助张量、公共稀疏路由张量和 FAC4DNN 堆叠张量；
3. 承诺与证明阶段：生成张量承诺、sumcheck transcript 与张量评值证明。

证明者必须严格执行本文定义的 GAT 前向语义。数据集、边排序、自环、隐藏层 ELU、输出层偏置、多输出头平均、量化尺度、舍入方式和公开预处理规则均由公开配置固定。

### 2.1 数据加载与特征检索

#### 2.1.1 原始特征

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 与每个输入维索引 $j\in\{0,1,\ldots,d_{in}-1\}$，仍然定义：

$H_{i,j}=T_H[I_i,j]$

#### 2.1.2 特征检索张量关系

特征检索使用公开选择矩阵 $R_I\in\{0,1\}^{N\times N_{total}}$：

$R_I[i,v]=1\iff v=I_i$

并证明：

$H=R_I T_H$

MLE 形式为：

$\widetilde H(u_i,u_j)=\sum_{v\in[N_{total}]}\widetilde R_I(u_i,v)\widetilde T_H(v,u_j)$

令 $\beta_{\widehat N}(u,i)$ 表示节点补零域上的多线性等值基函数，$\beta_{\widehat N_{total}}(v,t)$ 表示全局节点补零域上的多线性等值基函数，则公开选择矩阵 $R_I$ 的 MLE 评值定义为：

$\widetilde R_I(u_N,u_T)=\sum_{i=0}^{N-1}\beta_{\widehat N}(u_N,i)\,\beta_{\widehat N_{total}}(u_T,I_i)$

其中 $R_I$ 由公开节点绝对编号 $I$ 本地重建。证明者承诺 $H$，验证者使用 $Com(T_H)$ 与 $Com(H)$ 的评值证明检查该关系。

### 2.2 单个隐藏层模板中第 $r$ 个注意力头的完整见证生成

以下步骤对应固定隐藏层 $\ell\in\{1,\ldots,L-1\}$ 的单层模板。证明者按层序和头序计算全部前向变量；证明后端可以把同形张量操作按 FAC4DNN 堆叠聚合，但承诺对象和对象标签必须显式带有 $(\ell,r)$ 标签。

#### 2.2.1 投影

前向计算定义为：

$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}$

对应约束为 MLE 矩阵乘法 sumcheck：

$\widetilde H'^{(r)}(u_i,u_j)=\sum_m \widetilde H(u_i,m)\widetilde W^{(r)}(m,u_j)$

证明者提交 $Com(H'^{(r)})$，并在 sumcheck 末端给出 $H,W,H'$ 的评值证明。

#### 2.2.2 源注意力绑定

前向计算定义为：

$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)}$

对应约束为向量点积 sumcheck：

$\widetilde E_{src}^{(r)}(u_i)=\sum_j \widetilde H'^{(r)}(u_i,j)\widetilde a_{src}^{(r)}(j)$

#### 2.2.3 目标注意力绑定

前向计算定义为：

$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}$

对应约束与源注意力相同，使用 $Com(E_{dst}^{(r)})$、$Com(H'^{(r)})$ 和模型承诺 $Com(a_{dst}^{(r)})$ 的评值绑定。

#### 2.2.4 压缩特征绑定

压缩特征定义为：

$H_i^{\star(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}(\xi^{(r)})^j$

$\xi^{(r)}$ 由 transcript 生成，证明者提交 $Com(H^{\star(r)})$，并用一条 MLE 线性组合 sumcheck 绑定 $H'^{(r)}$ 与 $H^{\star(r)}$。

#### 2.2.5 源广播

源广播前向语义定义为：

$E_{src,k}^{edge(r)}=E_{src,src(k)}^{(r)},\qquad H_{src,k}^{\star,edge(r)}=H_{src(k)}^{\star(r)}$

对应约束为公开稀疏矩阵路由：

$E_{src}^{edge(r)}=R_{src}E_{src}^{(r)},\qquad H_{src}^{\star,edge(r)}=R_{src}H^{\star(r)}$

这两条关系分别用 MLE sumcheck 证明。

#### 2.2.6 LeakyReLU、最大值代表边选择与范围检查

前向语义定义为：

$S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)}$

$Z_k^{(r)}=LReLU(S_k^{(r)})$

$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}$

本文协议提交 $Com(S^{(r)}),Com(Z^{(r)}),Com(M^{(r)}),Com(M^{edge(r)}),Com(\Delta^{+(r)})$，并使用辅助证明约束查表关系、最大值代表边选择和范围关系：

1. $\Pi_{LReLU}^{(\ell,r)}$ 证明 $Z=LReLU(S)$，其中查表成员关系按第 0.8.2 节的 $Sel_{LReLU,\tau}$ 选择矩阵证明；
2. $M^{edge}=R_{dst}M$ 由路由 sumcheck 证明；
3. $\Delta^+=M^{edge}-Z$ 由线性 sumcheck 证明；
4. $\Pi_{range}^{(\ell,r)}$ 证明 $\Delta^+\ge 0$ 且位宽合法；
5. 对每个目标组至少选择一条达到最大值的边，通过最大值代表边选择张量 $B_{max}^{(\ell,r)}$ 证明。

$B_{max}^{(\ell,r)}\in\{0,1\}^E$ 的约束固定为：

$B_{max}\odot(B_{max}-1)=0$

$B_{max}\odot\Delta^+=0$

$(1-Q_{edge}^{valid})\odot B_{max}=0$

$R_{dst}^{\top}B_{max}=Q_N$

其中 $Q_N$ 是节点有效区内取 $1$、填充区取 $0$ 的公共节点有效区向量。上述约束表示每个有效目标组恰好选择一条达到最大值的代表边，边填充区不能选择代表边，填充节点不选择代表边；它不要求数学最大值唯一，允许多个真实边同时达到最大值。

#### 2.2.7 指数、分母、逆元与归一化权重

前向语义定义为：

$U_k^{(r)}=ExpMap(\Delta_k^{+(r)})=QuantExp(-\Delta_k^{+(r)})$

$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}$

$inv_i^{(r)}=(Sum_i^{(r)})^{-1}$

$\alpha_k^{(r)}=U_k^{(r)}\cdot inv_{dst(k)}^{(r)}$

对应约束为：

- $\Pi_{exp}^{(\ell,r)}$ 证明 $U=ExpMap(\Delta^+)=QuantExp(-\Delta^+)$；
- $Sum=R_{dst}^{\top}U$ 用稀疏聚合 sumcheck 证明；
- $inv^{edge}=R_{dst}inv$ 用路由 sumcheck 证明；
- $Q_N\odot(Sum\odot inv-1)=0$ 用 Hadamard sumcheck 证明，并检查 $(1-Q_N)\odot Sum=0$、$(1-Q_N)\odot inv=0$；
- $\alpha=U\odot inv^{edge}$ 用 Hadamard sumcheck 证明。

#### 2.2.8 聚合前隐藏矩阵与压缩特征

前向语义定义为：

$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}$

对应约束为两步张量关系：

1. 源特征矩阵广播语义：$H_{src}^{edge(r)}=R_{src}H'^{(r)}$；
2. 加权聚合：$H_{agg,pre}^{(r)}=R_{dst}^{\top}(\alpha^{(r)}\odot H_{src}^{edge(r)})$。

其中 $H_{src}^{edge(r)}$ 是公开稀疏路由诱导的虚拟广播张量，本协议固定不单独承诺、不单独进入最终证明对象；聚合 sumcheck 直接使用 $R_{src}$ 与 $H'^{(r)}$ 的评值表达该广播关系。

默认 MLE 语句写为：

$\widetilde H_{agg,pre}^{(r)}(u_i,u_j)=\sum_{k\in[E]}\widetilde R_{dst}(k,u_i)\widetilde\alpha^{(r)}(k)\sum_{i'\in[N]}\widetilde R_{src}(k,i')\widetilde H'^{(r)}(i',u_j)$

聚合前压缩特征仍按本文协议定义：

$H_{agg,pre,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,pre,i,j}^{(r)}(\xi^{(r)})^j$

并用线性组合 sumcheck 绑定。

#### 2.2.9 ELU

前向语义定义为：

$H_{agg,i,j}^{(r)}=ELU(H_{agg,pre,i,j}^{(r)})$

本文协议提交 $Com(H_{agg}^{(r)})$ 与 $Com(H_{agg}^{\star(r)})$，使用 $\Pi_{ELU}^{(\ell,r)}$ 证明 ELU 函数关系，并用线性组合 sumcheck 证明：

$H_{agg,i}^{\star(r)}=\sum_j H_{agg,i,j}^{(r)}(\xi^{(r)})^j$

#### 2.2.10 目标广播

目标广播语义定义为：

$H_{agg,pre,k}^{\star,edge(r)}=H_{agg,pre,dst(k)}^{\star(r)},\qquad H_{agg,k}^{\star,edge(r)}=H_{agg,dst(k)}^{\star(r)}$

对应约束为：

$H_{agg,pre}^{\star,edge(r)}=R_{dst}H_{agg,pre}^{\star(r)}$

$H_{agg}^{\star,edge(r)}=R_{dst}H_{agg}^{\star(r)}$

### 2.3 隐藏层模板的拼接阶段

拼接前向语义定义为：

$H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}$

本文协议提交 $Com(H_{cat})$ 与 $Com(H_{cat}^{\star})$，并用公共拼接选择张量 $R_{cat}^{(\ell,r)}$ 证明每个头输出进入拼接张量的正确位置：

$H_{cat}=\sum_{r=0}^{K_{hid}-1}R_{cat}^{(\ell,r)}H_{agg}^{(r)}$

随后证明：

$H_{cat,i}^{\star}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}\xi_{cat}^m$

该层 $H_{cat}^{(\ell)}$ 作为下一隐藏层唯一输入，层间数据流按固定层序执行。

### 2.4 输出层的完整见证生成

输出层前向语义固定如下。对每个输出头 $s$，对应约束与隐藏头相同；共享维使用类别维 $C$，输入使用最后一层 $H_{cat}^{(L-1)}$。

#### 2.4.1 第 $s$ 个输出头的输出投影

前向计算定义为：

$Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out,s)}$

对应约束为 MLE 矩阵乘法 sumcheck，并提交 $Com(Y'^{(s)})$。

#### 2.4.2 第 $s$ 个输出头的源 / 目标注意力绑定

前向计算定义为：

$E_{src,i}^{(out,s)}=\sum_cY_{i,c}'^{(s)}a_{src,c}^{(out,s)}$

$E_{dst,i}^{(out,s)}=\sum_cY_{i,c}'^{(s)}a_{dst,c}^{(out,s)}$

对应约束为向量点积 sumcheck，并与模型承诺评值证明绑定。

#### 2.4.3 第 $s$ 个输出头的 Softmax 链路

输出层的源广播、目标广播、LeakyReLU、最大值、范围、指数、分母、逆元与归一化权重，全部按隐藏层同一套辅助有效性证明和稀疏路由 sumcheck 执行。所有输出头必须按 $s=0,1,\ldots,K_{out}-1$ 的固定顺序提交承诺和证明记录。

#### 2.4.4 第 $s$ 个输出头的聚合前输出与压缩绑定

前向计算定义为：

$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}$

对应约束为：

$Y_{src}^{\prime edge(s)}=R_{src}Y'^{(s)}$

$Y^{pre(s)}=R_{dst}^{\top}(\alpha^{(out,s)}\odot Y_{src}^{\prime edge(s)})$

其中 $Y_{src}^{\prime edge(s)}$ 与隐藏层的 $H_{src}^{edge(r)}$ 一样，固定解释为虚拟路由表达式，不单独承诺。聚合 sumcheck 直接使用 $R_{src}$ 与 $Y'^{(s)}$ 的评值展开源广播。

随后用线性组合 sumcheck 证明 $Y'^{\star(s)}$、$Y^{pre,\star(s)}$ 与对应矩阵的类别压缩一致性。

#### 2.4.5 第 $s$ 个输出头的尺度对齐、偏置与目标广播

前向语义定义为：

$\widetilde Y_{i,c}^{pre(s)}=Rescale_{out}(Y_{i,c}^{pre(s)};S_{out,pre}\rightarrow S_{out,head})$

$Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}$

本文协议用 $\Pi_{rescale}^{(out,s)}$ 证明尺度对齐，使用线性 sumcheck 证明加偏置关系，并用 $R_{dst}$ 证明输出头压缩广播。

#### 2.4.6 最终多头平均输出

最终平均语义定义为：

$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)},\qquad Y_{i,c}=K_{out}^{-1}\cdot Y_{i,c}^{sum}$

最终输出 $Y$ 是公共输入张量，不作为私有动态张量承诺。验证者首先检查 $Y$ 的形状、补零长度与公共有效区 $Q_Y(i,c)=Q_N(i)\cdot Q_C^{valid}(c)$ 一致，并要求 $Q_Y=0$ 的位置为 $0$。在 sumcheck 末端，验证者直接从公共 $Y$ 计算所需的 MLE 评值，并检查该公共评值与全部被承诺的 $Y^{(s)}$ 满足平均关系。

给定挑战 $\xi_{out}$ 时，公共压缩值定义为：

$Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c$

该压缩值同样由验证者从公共 $Y$ 本地计算，不作为私有压缩张量承诺。公共 $Y$ 及其补零语义必须在生成所有依赖 $\xi_{out}$ 的挑战之前进入 Fiat-Shamir transcript。

至此，所有明文前向对象均由张量承诺、MLE sumcheck、公共输出评值和辅助有效性证明绑定。

## 证明生成

### 3.1 输入与输出

证明生成算法输入：

- 承诺公共参数 $PP_{com}$；
- 静态表验证键 $VK_{static}^{zkDL}$；
- 模型验证键 $VK_{model}^{zkDL}$；
- 公开元数据 $M_{pub}$；
- 公共拓扑张量；
- 公开输出张量 $Y$；
- 见证生成部分得到的全部 GAT 前向见证张量；
- 见证生成部分得到的全部非线性辅助张量；
- 承诺随机数。

输出 GAT-ZKML 证明：

$\pi_{GAT}^{zkDL}=\big(M_{pub},\mathsf{Topo},\mathsf{VKRef},\mathsf{Com}_{base},\mathsf{Com}_{comp},\mathsf{Com}_{aux},\mathsf{Com}_{tbl},\mathsf{AggDesc},\mathsf{Com}_{agg},\Pi_{slice},\mathsf{SC},\mathsf{Eval},\mathsf{Aux},\mathsf{digest}\big)$

其中 $\mathsf{Topo}$ 绑定公共拓扑张量与公共选择矩阵，$\mathsf{VKRef}$ 绑定静态表承诺与模型承诺，$\mathsf{Com}_{base}$、$\mathsf{Com}_{comp}$、$\mathsf{Com}_{aux}$、$\mathsf{Com}_{tbl}$ 和 $\mathsf{Com}_{agg}$ 分别对应基础动态张量、压缩张量、非线性辅助张量、查表选择张量和 FAC4DNN 聚合张量的承诺；$\Pi_{slice}$ 是聚合张量与单实例张量的一致性证明，$\mathsf{SC}$ 是 sumcheck transcript，$\mathsf{Eval}$ 是张量评值证明，$\mathsf{Aux}$ 是非线性辅助有效性证明。

### 3.2 Fiat-Shamir 挑战顺序

挑战顺序固定如下：

1. 吸入 $protocol\_id,proof\_scope\_id,commitment\_backend\_id,model\_arch\_id,model\_param\_id,static\_table\_id,quant\_cfg\_id,domain\_cfg,dim\_cfg,encoding\_id,padding\_rule\_id,tensor\_bound\_id$；
2. 吸入公共输出张量 $Y$ 的规范编码哈希、公共拓扑张量哈希与公共有效区张量哈希，包括 $I,src,dst,node\_ptr,edge\_ptr,R_I,R_{src},R_{dst},Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$ 以及第 0.7.2 节定义的全部有效区张量；
3. 吸入 $VK_{static}^{zkDL}$ 与 $VK_{model}^{zkDL}$；
4. 依次吸入基础动态张量承诺块、非线性辅助张量承诺块与查表选择张量承诺块。基础动态张量承诺块包括不依赖压缩挑战的前向张量；非线性辅助张量承诺块包括辅助位张量、最大值代表边选择张量、重缩放余数张量和逆元相关张量；查表选择张量承诺块包括全部 $Sel_{f,\tau}$；
5. 生成各层压缩挑战 $\xi^{(\ell,r)},\xi_{cat}^{(\ell)},\xi_{out}$；
6. 证明者基于第 5 步挑战生成并提交压缩张量承诺，包括 $H^{\star},H_{agg,pre}^{\star},H_{agg}^{\star},H_{cat}^{\star},Y'^{\star},Y^{pre,\star},Y^{\star(s)}$ 及其边域广播压缩对象；公共 $Y^{\star}$ 由验证者从公共 $Y$ 与 $\xi_{out}$ 本地计算，不进入压缩张量承诺块；
7. 吸入全部压缩张量承诺；
8. 根据 $\mathcal T_{shape}$ 和 FAC4DNN 聚合规则构造聚合张量，吸入全部聚合标识、聚合描述和聚合承诺。聚合只能引用已经在第 4 步或第 6 步完成承诺的张量；若某个聚合包含压缩对象，则必须在压缩张量承诺完成后构造；若某个聚合只包含基础对象，也仍按本步骤统一延后吸入 transcript，禁止提前生成聚合挑战；
9. 生成 FAC4DNN 堆叠实例维挑战 $w_{op}$；
10. 对每类 sumcheck 生成其内部轮挑战；
11. 吸入 sumcheck 末端声明，生成张量评值点；
12. 吸入全部评值证明；
13. 输出最终 transcript digest。

任何协议不得因为并行生成不同头、不同层或不同聚合的证明而改变上述吸入顺序。并行只允许发生在局部计算层面，最终 transcript 必须按固定层序、头序、输出头序、聚合序序列化。

### 3.3 约束族与 sumcheck 语句

所有约束族使用以下 sumcheck 语句。

#### 3.3.1 线性张量关系

覆盖对象包括：特征检索、源 / 目标广播、目标广播、拼接、偏置相加、最终平均、压缩特征、类别压缩、尺度对齐后的线性部分。

通用形式为：

$Z=\sum_t A_tX_t+B$

在随机点 $u$ 上证明：

$\widetilde Z(u)=\sum_t a_t\widetilde X_t(\phi_t(u))+\widetilde B(u)$

其中 $\phi_t$ 是由展平规则和公共拓扑决定的索引映射。

#### 3.3.2 矩阵乘法与点积

隐藏层投影、输出层投影、源注意力、目标注意力均使用 MLE 矩阵乘法或点积 sumcheck。对形状相同的实例，优先按 FAC4DNN 堆叠聚合；对形状不同的实例，按形状聚合分别证明。

#### 3.3.3 稀疏路由与聚合

源广播和目标广播使用公共矩阵 $R_{src},R_{dst}$：

$X^{src}=R_{src}X,\qquad X^{dst}=R_{dst}X$

按目标节点聚合使用：

$Agg=R_{dst}^{\top}V^{edge}$

$R_{src}$ 与 $R_{dst}$ 在协议中的参与方式以第 0.7.3 节的公开稀疏 MLE 评值公式为准；$E\times N$ 稠密矩阵不属于协议对象。

#### 3.3.4 Hadamard 乘法

覆盖对象包括：

- $\alpha=U\odot inv^{edge}$；
- $\widehat V=\alpha\odot V^{edge}$；
- $Q_N\odot(Sum\odot inv-1)=0$；
- 辅助二值性 $aux\odot(aux-1)=0$；
- 最大值辅助张量中 $B_{max}\odot\Delta^+=0$。

所有 Hadamard 关系按 MLE 乘法 sumcheck 证明。

#### 3.3.5 非线性辅助证明

$\Pi_{aux}^{GAT}$ 包括：

1. $\Pi_{LReLU}$：对 $S,Z$ 的有符号输入位分解、静态表成员关系、输出重构和函数关系；
2. $\Pi_{ELU}$：对 $H_{agg,pre},H_{agg}$ 的输入位分解、静态表成员关系、输出重构和函数关系；
3. $\Pi_{exp}$：对 $\Delta^+,U$ 的范围位分解、静态表成员关系和 $U=QuantExp(-\Delta^+)$ 的输出重构；
4. $\Pi_{range}$：对所有差分、余数和重缩放中间值的位宽合法性；
5. $\Pi_{inv}$：对有效区分母非零性的乘法绑定；
6. $\Pi_{rescale}$：对公开量化配置指定的重缩放规则进行商余数证明；
7. $\Pi_{max}$：对每个目标组的最大值代表边选择进行二值性、选中边零差分和组内恰好选择一条边的证明。

静态表成员关系统一使用选择矩阵。对 $f\in\{LReLU,ELU,exp\}$，设查询输入输出为 $(X_q,V_q)$，静态表为 $T_f=(T_{f,x},T_{f,y})$，选择矩阵为 $Sel_{f,\tau}$，查询族标号为 $\tau$，查询有效区向量为 $Q_{f,\tau}^{qry}$。以下矩阵式是第 0.8.2 节逐点查表关系的 MLE 形式：

$Sel_{f,\tau}\odot(Sel_{f,\tau}-1)=0$

$((1-Q_{f,\tau}^{qry})\mathbf 1^\top)\odot Sel_{f,\tau}=0$

$Sel_{f,\tau}\odot((1-Q_f^{tbl})^\top)=0$

$Sel_{f,\tau}\mathbf 1=Q_{f,\tau}^{qry}$

$X=Sel_{f,\tau}T_{f,x}$

$V=Sel_{f,\tau}T_{f,y}$

$(1-Q_{f,\tau}^{qry})\odot X=0,\qquad (1-Q_{f,\tau}^{qry})\odot V=0$

所有关系均用 MLE sumcheck 证明，并通过 $Com(Sel_{f,\tau})$、$Com(T_f)$、$Com(X)$、$Com(V)$ 的评值证明绑定。$T_f$ 必须是 $VK_{static}^{zkDL}$ 中对应静态表承诺。

范围证明对每个被检查对象绑定其有效区、位宽和符号解释。无符号对象使用：

$Q_x\odot\left(x-\sum_{b=0}^{B_x-1}2^b bit_b\right)=0$

有符号对象使用：

$Q_x\odot\left(x-\sum_{b=0}^{B_x-2}2^b bit_b+2^{B_x-1}bit_{B_x-1}\right)=0$

所有位张量满足二值性，且在无效区为 $0$；被检查对象在无效区也为 $0$。

重缩放证明按第 0.8.2 节的商余数约束执行。对每个重缩放关系，证明对象必须绑定公开除数 $\rho$、公开偏移 $\delta$、输入 $x$、输出 $y$、余数 $rem$ 以及边界余量 $rem^{bd}$；并证明主商余数关系、边界闭合关系、无效区零约束和公开无符号范围约束。$\rho,\delta$ 与余数位宽由 $quant\_cfg\_id$ 唯一确定。

最大值证明不要求最大值唯一。对每个 softmax 目标组，证明者提交 $B_{max}$，并证明：

$B_{max}\odot(B_{max}-1)=0$

$B_{max}\odot\Delta^+=0$

$(1-Q_{edge}^{valid})\odot B_{max}=0$

$R_{dst}^{\top}B_{max}=Q_N$

其中 $Q_N$ 是公共节点有效区向量。配合 $\Delta^+\ge 0$ 的范围证明，上述约束足以证明每个有效节点的 $M_i$ 是该目标组内某个实际达到的最大值，并保证边填充区与节点填充区不产生代表边选择。

这些证明的共同要求是：辅助张量先承诺，再证明二值性、范围重构、查表成员关系和目标函数关系。若某个函数由公开查找表给出，则表承诺必须是 $VK_{static}^{zkDL}$ 中对应静态表承诺，不能使用临时未绑定表。

#### 3.3.6 FAC4DNN 聚合示例

对同形操作实例使用额外实例维堆叠。例如，所有形状相同的隐藏层投影可组成：

$\mathcal H\in\mathbb F_p^{N_{inst}\times N\times d_{in}}$

$\mathcal W\in\mathbb F_p^{N_{inst}\times d_{in}\times d_h}$

$\mathcal H'\in\mathbb F_p^{N_{inst}\times N\times d_h}$

并用实例挑战 $w$ 同时证明所有实例：

$\widetilde{\mathcal H'}(w,u_i,u_j)=\sum_m\widetilde{\mathcal H}(w,u_i,m)\widetilde{\mathcal W}(w,m,u_j)$

实例维只用于证明聚合，不改变 GAT 前向依赖顺序。聚合张量必须是已承诺对象；若聚合切片还被单实例约束引用，则切片一致性由 $\Pi_{agg}$ 证明。

#### 3.3.7 FAC4DNN 聚合等价类

FAC4DNN 聚合按“代数关系类型 + 张量形状 + 公共参数版本 + 辅助子协议类型”划分等价类。只有同时满足以下条件的实例才能放入同一聚合张量：

1. 代数关系相同，例如矩阵乘法、点积、Hadamard 乘法、稀疏路由、线性压缩、查表成员关系或重缩放关系；
2. 输入张量和输出张量的真实形状、补零形状与 MLE 维度完全相同；
3. 使用的公共拓扑、静态表版本、模型参数版本和量化配置一致；
4. 子协议语义一致，例如 LeakyReLU、ELU、ExpMap、range、max、inverse 与 rescale 不得互相混入同一等价类；
5. 实例顺序可由 $M_{pub}$ 唯一确定。

对每个等价类 $\mathcal C_{op}$，证明者把其中实例按公开顺序堆叠为聚合张量，并对聚合张量承诺。若某个单实例张量同时被非聚合约束引用，则必须给出聚合切片与单实例承诺之间的一致性证明：

$\widetilde{\mathcal T}_{agg}(e_t,u)=\widetilde T_{single}^{(t)}(u)$

其中 $e_t$ 是实例 $t$ 的公开布尔编码。

聚合只改变证明组织方式，不改变 GAT 前向计算的层序依赖；证明者必须先按前向语义生成所有单实例见证，再按等价类堆叠。

### 3.4 张量评值与开放

每条 sumcheck 在末端产生若干张量随机点评值声明。证明者必须对所有被引用的私有张量提供评值证明：

$\pi_{eval}(T,u,\widetilde T(u))$

同一个张量在多个 sumcheck 中被打开到多个点时，多个评值点的验证语义必须等价于逐项验证；具体采用批量打开或逐项打开由 $commitment\_backend\_id$ 标识的承诺后端规范决定。

静态表和模型参数也通过其静态承诺给出评值证明，但不重新提交表值或模型值。

### 3.5 最终证明对象

最终证明对象固定由以下块按顺序组成：

1. 元数据块 $M_{pub}$；
2. 公共拓扑绑定块 $\mathsf{Topo}$；
3. 静态表承诺与模型承诺引用块 $\mathsf{VKRef}$；
4. 基础动态张量承诺块 $\mathsf{Com}_{base}$；
5. 压缩张量承诺块 $\mathsf{Com}_{comp}$；
6. 非线性辅助张量承诺块 $\mathsf{Com}_{aux}$；
7. 查表选择张量承诺块 $\mathsf{Com}_{tbl}$；
8. FAC4DNN 聚合描述块 $\mathsf{AggDesc}$；
9. FAC4DNN 聚合张量承诺块 $\mathsf{Com}_{agg}$；
10. 聚合切片一致性证明块 $\Pi_{slice}$；
11. sumcheck transcript 块 $\mathsf{SC}$；
12. sumcheck 末端评值声明块；
13. 张量评值证明块 $\mathsf{Eval}$；
14. 辅助有效性证明块 $\mathsf{Aux}$；
15. 最终 transcript 摘要 $\mathsf{digest}$。

基础动态张量承诺块包含不依赖压缩挑战的前向张量，但不包含公共输出 $Y$；压缩张量承诺块包含由 $\xi^{(\ell,r)},\xi_{cat}^{(\ell)},\xi_{out}$ 派生的私有压缩张量及其边域广播压缩对象；公共 $Y^{\star}$ 由验证者从公共 $Y$ 本地计算，不属于压缩张量承诺块。非线性辅助张量承诺块包含位分解、最大值代表边选择、重缩放余数、逆元辅助等对象；查表选择张量承诺块包含全部 $\{Sel_{f,\tau}\}_{f,\tau}$；聚合张量承诺块包含 FAC4DNN 聚合证明所需的输入聚合张量和输出聚合张量。

$H_{src}^{edge(r)}$ 与 $Y_{src}^{\prime edge(s)}$ 属于由 $R_{src}$ 诱导的虚拟路由表达式，不单独承诺、不单独打开评值。

公共输出 $Y$ 不属于私有动态张量承诺块，也不属于证明对象的私有见证部分；它作为验证输入按第 3.2 节进入 transcript。证明对象按上述顺序编码并纳入 transcript；任何块的缺失、重复或顺序错误都导致验证失败。

### 3.6 规范编码与 transcript 绑定

本文只要求存在由 $encoding\_id$ 唯一确定的规范编码函数：

$Enc(\cdot)$

所有公共元数据、公共输出 $Y$、公共拓扑哈希、承诺、聚合描述、sumcheck 消息、末端评值声明和评值证明都必须先经 $Enc(\cdot)$ 编码，再吸入 Fiat-Shamir transcript。

验证者必须使用同一个 $Enc(\cdot)$ 重放 transcript。若同一数学对象存在多种编码，或者证明对象中存在缺失、重复、顺序不一致的协议对象，验证者拒绝。

## 验证

### 4.1 元数据与对象完整性检查

验证者首先解析 $\pi_{GAT}^{zkDL}$，检查：

- $protocol\_id$ 是否为本文协议版本；
- $proof\_scope\_id$ 是否固定为 GAT 前向计算证明范围；
- $commitment\_backend\_id$ 是否与承诺后端语义一致；
- $model\_arch\_id,model\_param\_id,static\_table\_id,quant\_cfg\_id$ 是否匹配公开配置；
- $domain\_cfg,dim\_cfg,padding\_rule\_id,tensor\_bound\_id$ 是否能唯一重建所有公共张量；
- 公共输出 $Y$ 的形状、补零长度和有效区零约束是否与 $Q_Y$ 一致；
- 动态张量承诺是否完整，是否按固定层序、头序、输出头序出现；
- 辅助张量承诺是否覆盖所有 LeakyReLU、ELU、ExpMap、range、inverse 和 rescale 关系。

若任一字段缺失、重复、顺序错误或与公开配置不一致，立即拒绝。

### 4.2 重放 Fiat-Shamir transcript

验证者按第 3.2 节固定顺序重放 transcript，并把公共输出 $Y$ 按规范编码吸入 transcript，重新生成所有挑战，包括压缩挑战、实例维挑战、sumcheck 轮挑战和评值点。证明对象中携带的任何挑战值不得作为验证输入。

### 4.3 sumcheck transcript 验证

验证者逐个检查所有 sumcheck：

1. 首轮声明是否等于当前约束族要求的目标声明；
2. 每轮多项式的端点和是否等于上一轮声明；
3. 挑战是否由 transcript 正确生成；
4. 末端声明是否与后续张量评值证明中的评值一致；
5. FAC4DNN 聚合中的实例维、张量形状和操作类型是否一致。

若某个 sumcheck 失败，立即拒绝。

### 4.4 张量评值证明验证

对每个承诺张量 $Com(T)$ 和每个声明评值 $\widetilde T(u)=y$，验证者调用承诺后端的评值验证算法：

$VerifyEval(VP_{com},Com(T),u,y,\pi_{eval})=1$

静态表、模型参数和动态见证全部使用同一评值语义。若一个张量被多个 sumcheck 引用，所有引用的评值点和值必须一致绑定到同一个承诺。

### 4.5 FAC4DNN 聚合与切片一致性检查

验证者检查所有 FAC4DNN 聚合对象：

1. 聚合描述是否与公开张量形状集合、实例顺序和 transcript 中的聚合承诺一致；
2. 每个聚合张量中的实例形状是否完全相同，不同形状实例是否被拆入不同聚合张量；
3. 聚合实例顺序是否严格遵守第 0.7.7 节规定的公共顺序；
4. 聚合承诺是否在第 3.2 节规定位置吸入 transcript；
5. 当某个聚合张量切片还被单实例约束引用时，是否存在对应的切片一致性证明；
6. 切片一致性证明是否绑定同一个单实例张量承诺与同一个聚合张量承诺，即验证 $\widetilde{\mathcal T}_{agg}(e_t,u)=\widetilde T_{single}^{(t)}(u)$。

若聚合描述、聚合承诺、实例顺序或切片一致性证明任一项不一致，立即拒绝。

### 4.6 非线性辅助证明检查

验证者检查 $\Pi_{aux}^{GAT}$ 中的全部辅助证明：

- 辅助位张量二值性；
- 输入有符号解释与公开位宽；
- LeakyReLU、ELU、ExpMap 的函数关系；
- $\Delta^+$ 的非负范围；
- 最大值代表边选择关系 $B_{max}\odot(B_{max}-1)=0$、$B_{max}\odot\Delta^+=0$、$(1-Q_{edge}^{valid})\odot B_{max}=0$ 与 $R_{dst}^{\top}B_{max}=Q_N$；
- $Q_N\odot(Sum\odot inv-1)=0$，并且 $Sum,inv$ 在节点填充区为 $0$；
- 重缩放余数范围与公开舍入 / 截断规则；
- 查表选择矩阵 $Sel_{f,\tau}$ 的二值性、单行唯一选择、输入输出重构关系；
- 静态表版本与 $VK_{static}^{zkDL}$ 的绑定。

若任一辅助证明与公开量化配置或静态表版本不一致，立即拒绝。

### 4.7 GAT 语义绑定

验证者接受时，证明对象中的各约束族共同绑定以下 GAT 语义：

输入特征张量由 $T_H$ 与公开局部节点编号 $I$ 唯一确定；边集、排序、自环与 batch 分组由 $\mathsf{Topo}$ 唯一确定；隐藏层张量按 $\ell=1,\ldots,L-1$ 的层序依次绑定，且每层内按 $r=0,\ldots,K_{hid}^{(\ell)}-1$ 的头序绑定；每个隐藏头的投影、注意力打分、Softmax、聚合与 ELU 输出均满足第 0.6 节定义；输出层保持注意力头语义，每个输出头先完成注意力聚合，再按 $quant\_cfg\_id$ 做尺度对齐并加入本头偏置；最终公开 $Y$ 由全部输出头在共同尺度上做域内算术平均得到。所有量化尺度、舍入、截断与符号解释均由同一个 $quant\_cfg\_id$ 绑定。

### 4.8 验证结论

当且仅当元数据检查、transcript 重放、所有 sumcheck、所有张量评值证明、FAC4DNN 聚合与切片一致性检查、所有非线性辅助证明以及 GAT 语义绑定条件全部通过时，验证者接受：

$Verify_{zkDL}(\pi_{GAT}^{zkDL},M_{pub},\mathsf{Topo},VK_{static}^{zkDL},VK_{model}^{zkDL},Y)=1$

否则拒绝。

以下三节给出验证性质。

### 4.9 验证性质：完备性

若证明者按照 $M_{pub}$、公开拓扑、静态表承诺与模型承诺生成全部 GAT 前向张量、辅助张量、FAC4DNN 聚合张量和承诺随机数，并且所有张量满足前述约束，则验证算法必然接受：

$Verify_{zkDL}(\pi_{GAT}^{zkDL},M_{pub},\mathsf{Topo},VK_{static}^{zkDL},VK_{model}^{zkDL},Y)=1$

### 4.10 验证性质：可靠性

在张量承诺绑定性、sumcheck 可靠性以及 Fiat-Shamir 随机预言机模型成立的前提下，任意多项式时间证明者若能使验证者接受，则除可忽略概率外，存在一组私有见证张量满足关系 $R_{GAT\text{-}zkDL}$。

具体地，接受证明同时蕴含：

1. 公开输出 $Y$ 与全部被承诺的输出头张量 $\{Y^{(s)}\}_{s=0}^{K_{out}-1}$ 的域内平均一致；
2. 这些被承诺的输出头张量及其前置张量由第 0.1 节定义的 GAT 前向语义逐层得到；
3. 所有非线性、查表、范围、最大值、逆元与重缩放关系均满足对应辅助子协议；
4. FAC4DNN 聚合张量与其单实例切片一致。

### 4.11 验证性质：零知识

在张量承诺隐藏性、辅助子协议零知识性以及 sumcheck 零知识变体成立的前提下，证明对象除公共输入、公共拓扑、静态 / 模型承诺、协议中被打开的必要评值、声明输出 $Y$ 以及协议公开的形状信息外，不泄露公共输入未包含的私有特征值、私有中间张量、辅助见证或承诺随机数。
