# GAT-ZKML

## 0. 符号

### 0.1 模型结构

 GAT的前向固定为一个可参数化的多层结构：

1. 设注意力层的总层数为：$L$。其中前 $L-1$ 层是隐藏层，第 $L$ 层是最终输出层。

2. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，记该层的注意力头数量为 $K_{hid}^{(\ell)}$，每个头的输出维度为 $d_h^{(\ell)}$。该层每个头完成注意力聚合后，逐元素施加 ELU；

	随后把本层全部头的ELU后的输出沿特征维进行拼接，得到：$d_{cat}^{(\ell)}=K_{hid}^{(\ell)}\cdot d_h^{(\ell)}$。

3. 第 $\ell$ 个隐藏层的输入维度记为 $d_{in}^{(\ell)}$。其中 $d_{in}^{(1)}=d_{in}$，且对任意 $\ell\ge 2$，都有 $d_{in}^{(\ell)}=d_{cat}^{(\ell-1)}$。

4. 最终输出层（第 $L$ 层）的注意力头数量记为 $K_{out}$。输出层不是“仿射线性层 + 偏置”替代注意力，而是由 $K_{out}$ 个输出注意力头组成。每个输出注意力头先完成注意力聚合，再加上该头自己的输出偏置向量；若 $K_{out}>1$，则全部输出头的后偏置输出按头维做算术平均，得到最终logits。

5. 输出注意力头的激活函数固定为恒等映射，不再施加 ELU；但恒等映射的输入是已经加上该输出头偏置后的注意力聚合结果。

6. 后续对隐藏层的明文对象、见证对象、查找、PSQ、路由、一致性绑定约束与商多项式推导，都严格对应“单个隐藏层模板”。

	对任意多层 GAT，证明者与验证者都必须按层序 $\ell=1,\ldots,L-1$ 逐层实例化该模板；第 $\ell-1$ 层的拼接输出 $H_{cat}^{(\ell-1)}$ 必须作为第 $\ell$ 层的唯一输入表示。

7. 下列基准仅作为非规范实例说明，不构成协议参数默认值；实际实验使用的 $L,\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1},K_{out}$ 必须完全由公共元数据 $M_{pub}$ （这次证明所绑定的公开配置总包）唯一确定：
   - Cora / Citeseer 风格两层 GAT：$L=2$，在某些基准测试的配置文件中可取 $K_{hid}^{(1)}=8,\ K_{out}=1$；
   - Pubmed 风格两层 GAT：$L=2$；若某次实验把它设为两层多输出头实例，则对应的 $K_{out}>1$ 也必须由该次实验的 $M_{pub}$ 唯一固定；
   - PPI 风格多层 GAT：$L=3$ 或更深，仍保持“隐藏层拼接、最终层平均“的官方语义。

为避免公式中引入过多层指标，除非特别声明，隐藏层模板统一采用如下局部简写：

- 隐藏层模板索引：$\ell\in\{1,2,\ldots,L-1\}$

- 第 $\ell$ 个隐藏层中第 $r$ 个注意力头的编号：$r\in\{0,1,\ldots,K_{hid}-1\}$ 
  
- 当前隐藏层模板的输入维度：$d_{in}:=d_{in}^{(\ell)}$ 
  
- 当前隐藏层模板的每头输出维度：$d_h:=d_h^{(\ell)}$ 
  
- 当前隐藏层模板的头数：$K_{hid}:=K_{hid}^{(\ell)}$ 
  
- 当前隐藏层模板的拼接维：$d_{cat}:=d_{cat}^{(\ell)}=K_{hid}\cdot d_h$
  
- 输出类别数：$C$
  
- 最终输出层第 $s$ 个输出头编号：$s\in\{0,1,\ldots,K_{out}-1\}$

换言之，后文若出现 $H_{cat}$、$d_h$、$K_{hid}$ 等未显式带层指标的记号，在隐藏层上下文中一律解释为“当前正在实例化的那一层”的数学局部记号；代码实现中的对象名、序列化器字段名、证明标签以及日志字段名，一律必须显式带层指标 $(\ell)$，不得把这些局部记号直接搬入实现。

同理，凡正文中为了公式紧凑而省略的 $(\ell)$、$(\ell,r)$、$(out,s)$ 标记，只能在数学推导中省略；在任何实现接口、配置字段、承诺名、评值名与子证明名中，都必须恢复完整指标。

### 0.2 有限域、双线性群、安全 SRS、KZG 与 Fiat--Shamir

记有限域为 $\mathbb F_p$，其中 $p$ 为大素数。所有协议内算术均在 $\mathbb F_p$ 上进行。

采用定点数量化语义：若域元素 $x\in\mathbb F_p$ 的标准代表满足 $x>\frac{p-1}{2}$，则在实数语义下解释为有符号整数 $x-p$；否则解释为非负整数 $x$。LeakyReLU、ELU、范围比较、指数查表、量化回缩和 Softmax 归一化均在该有符号域语义下定义。

记双线性群为 $(\mathbb G_1,\mathbb G_2,\mathbb G_T,e)$，其中 $e:\mathbb G_1\times\mathbb G_2\to\mathbb G_T$ 是非退化双线性映射，生成元分别为 $G_1\in\mathbb G_1$ 与 $G_2\in\mathbb G_2$。

正式协议使用KZG 多项式承诺。对次数严格小于 $D$ 的多项式 $P(X)=\sum_{i=0}^{D-1}p_iX^i\in\mathbb F_p[X]$，数学上的承诺定义为：$$[P]=P(\tau)G_1$$。其中 $\tau\in\mathbb F_p$ 是参数生成阶段采样的隐藏陷门。该公式只定义承诺语义，不允许实现层在证明阶段直接读取、保存、打印、序列化或由 seed 派生 $\tau$。正式实现必须满足：

1. $\tau$ 只在 trusted setup 或 MPC ceremony 的临时内存中出现；
2. 生成证明键和验证键后，$\tau$ 必须销毁；
3. 证明者只持有 $PK$ 或经过预处理的工作域 SRS；
4. 验证者只持有 $VK_{KZG}$；
5. 主实验代码中不得存在 `seed -> tau`、`tau = seed + c`、`polynomial.evaluate(tau)` 作为承诺主路径的实现。

KZG 验证键为：$$VK_{KZG}=\{[1]_2,[\tau]_2\}=\{G_2,\tau G_2\}$$；证明键至少包含基于系数的SRS：$$PK_{coef}=\{[\tau^0]_1,[\tau^1]_1,\ldots,[\tau^{D_{max}}]_1\}$$。

若多项式以系数形式给出，则承诺实现必须使用：$$[P]=\sum_{i=0}^{D-1}p_i[\tau^i]_1.$$

为了适配大量工作域插值多项式，正式实现可以进一步预计算每个工作域 $\mathbb H_{\mathcal D}$ 的基于拉格朗日的SRS：$$PK_{lag}^{(\mathcal D)}=\{[L_0^{(\mathcal D)}(\tau)]_1,[L_1^{(\mathcal D)}(\tau)]_1,\ldots,[L_{n_{\mathcal D}-1}^{(\mathcal D)}(\tau)]_1\}$$。

若多项式以工作域$\mathbb H_{\mathcal D}$上的评估形式给出，且 $P(X)=\sum_{i=0}^{n_{\mathcal D}-1}P(\omega_{\mathcal D}^i)L_i^{(\mathcal D)}(X)$，则承诺实现应使用：$$[P]=\sum_{i=0}^{n_{\mathcal D}-1}P(\omega_{\mathcal D}^i)[L_i^{(\mathcal D)}(\tau)]_1$$.

该基于拉格朗日承诺路径是主实验的推荐实现，因为它避免在每个 trace 列上重复执行 IFFT，同时不暴露 $\tau$。

#### 0.2.1 隐私、隐藏承诺与掩蔽 opening

正式目标是零知识可验证推理，而不是仅证明计算正确性。所有非公开见证多项式在进入承诺和 opening 之前都必须启用 hiding/blinding 规则。本文采用 domain-vanishing blinding：对定义在工作域 $\mathbb H_{\mathcal D}$ 上的私有多项式 $P$，证明者实际承诺的对象为$$P^{zk}(X)=P(X)+Z_{\mathcal D}(X)R_P(X)$$。

其中 $Z_{\mathcal D}(X)=X^{n_{\mathcal D}}-1$，$R_P(X)$ 是按 `privacy_policy` 规定采样的随机低次数掩蔽多项式。由于 $Z_{\mathcal D}$ 在工作域内恒为 0，$P^{zk}$ 与 $P$ 在工作域上的 trace 值完全一致，不改变所有域内约束语义；而在随机 opening 点上，验证者看到的是 $P^{zk}(z)$，不是裸见证值 $P(z)$。

所有商构造、opening bundle、外点绑定子证明和 transcript 都必须消费 $P^{zk}$ 的承诺与评值。若某个绑定关系需要使用裸语义值，必须通过零知识绑定子证明 $\Pi_{bind}$ 给出，不得把裸折叠向量、裸中间激活、裸注意力权重或裸模型参数评值写入 proof object。

公开多项式不需要掩蔽，包括由验证者本地重建的拓扑列、选择器列、索引列、公开元数据和最终被声明公开的输出 $Y$。若最终 logits $Y$ 被定义为公开输出，则 $P_Y$ 和 $P_{Y^\star}$ 可按公开输出策略不掩蔽；若某次实验要求只公开 label 而隐藏 logits，则必须把 logits 改为私有 witness 并额外证明 label 与 logits 的 argmax 关系。

随机预言机统一写作 $H_{FS}(\cdot)$。所有 Fiat--Shamir 挑战必须按本文固定顺序生成，禁止重排、回退、漏吸入或重复吸入。`srs_id`、`srs_type`、`privacy_policy`、`proof_task`、`encoding` 和所有 public metadata 必须在任何动态承诺挑战生成之前吸入 transcript。

### 0.3 图、局部子图、多图批处理与排序

- 全局节点总数：$N_{total}$

- 当前证明实例中的总节点数：$N$

- 当前证明实例中的总边数：$E$

- 当前证明实例中的图个数：$G_{batch}$

- 局部子图 / 批处理后的节点绝对编号序列：$I=(I_0,I_1,\ldots,I_{N-1})$

- 边源索引序列：$src=(src(0),src(1),\ldots,src(E-1))$

- 边目标索引序列：$dst=(dst(0),dst(1),\ldots,dst(E-1))$

- 批处理中各图的节点前缀指针：$node\_ptr=(\nu_0,\nu_1,\ldots,\nu_{G_{batch}})$，其中$0=\nu_0<\nu_1<\cdots<\nu_{G_{batch}}=N$
  
- 批处理中各图的边前缀指针：$edge\_ptr=(\epsilon_0,\epsilon_1,\ldots,\epsilon_{G_{batch}})$，其中 $0=\epsilon_0<\epsilon_1<\cdots<\epsilon_{G_{batch}}=E$

对任意节点索引 $i\in\{0,1,\ldots,N-1\}$，定义其图编号 $node\_gid(i)=g$，其中 $g\in{0,1,\ldots,G_{batch}-1}$ 满足 $\nu_g\le i<\nu_{g+1}$ 

对任意边索引 $k\in\{0,1,\ldots,E-1\}$，定义其图编号 $edge\_gid(k)=g$，其中 $g\in{0,1,\ldots,G_{batch}-1}$ 满足 $\epsilon_g\le k<\epsilon_{g+1}$ 

因此，对任意边索引 $k$，都有：

$src(k),dst(k)\in\{0,1,\ldots,N-1\},\quad node\_gid(src(k))=edge\_gid(k),\quad node\_gid(dst(k))=edge\_gid(k)$ 

若 $G_{batch}=1$，公共输入仍然必须显式写成 $node\_ptr=(0,N),\quad edge\_ptr=(0,E)$，不得省略，也不得以其他记号替代。

固定要求边序列满足以下两级稳定排序：

1. 先按 $edge\_gid(k)$ 非降序排列，使不同图的边在公开边序列中保持连续块；

2. 在每个固定图块内部，再按目标节点索引 $dst(k)$ 做确定性稳定排序。

等价地，若 $0\le k_1<k_2\le E-1$，则必须满足：$edge\_gid(k_1)<edge\_gid(k_2)$ 或 $(edge\_gid(k_1)=edge\_gid(k_2)$  且 $dst(k_1)\le dst(k_2)$ 

若原始输入不满足这一条件，则在见证生成前先做：

- 图内节点重编号 / 局部子图抽取；
  
- 按图分块的稳定排序；

- 每个图块内部按 $dst$ 的稳定排序。

验证者只接收已经按此规范生成的公开序列。除非某条公式显式写出“按图”量或“图局部”量，否则后文中的 $N,E$ 一律指当前证明实例中的总节点数与总边数；在多图批处理情形下，它们分别等于所有图的节点数与边数总和。

### 0.4 静态表与量化尺度

静态表包括：

1. 全局特征表：$T_H\in\mathbb F_p^{N_{total}\times d_{in}}$

2. LeakyReLU表：$T_{LReLU}\subseteq \mathbb F_p\times\mathbb F_p$

3. ELU表：$T_{ELU}\subseteq \mathbb F_p\times\mathbb F_p$

4. 指数表：$T_{exp}\subseteq \mathbb F_p\times\mathbb F_p$

5. 范围表：$T_{range}=\{0,1,2,\ldots,2^B-1\}$，其中 $B$ 是范围检查位宽。


为避免多值歧义，要求：

- 对任意 $x\in\mathbb F_p$，在 $T_{LReLU}$ 中至多存在一个 $y\in\mathbb F_p$ 使得 $(x,y)\in T_{LReLU}$。
  
- 对任意 $x\in\mathbb F_p$，在 $T_{ELU}$ 中至多存在一个 $y\in\mathbb F_p$ 使得 $(x,y)\in T_{ELU}$。
  
- 对任意 $u\in\mathbb F_p$，在 $T_{exp}$ 中至多存在一个 $v\in\mathbb F_p$ 使得 $(u,v)\in T_{exp}$。

与Softmax链路有关的量化尺度全部视为公开参数：

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

舍入方式（向最近整数舍入或截断）、乘法后的重缩放规则也全部是公开参数。

进一步固定输出层的量化与平均语义如下：

1. 对每个输出头 $s$，聚合前输出 $Y^{pre(s)}$ 在与偏置向量 $b^{(out,s)}$ 相加之前，必须已经被公开量化规则对齐到与 $b^{(out,s)}$ 相同的输出头尺度 $S_{out,head}$。若实现中注意力聚合自然产出的尺度是 $S_{out,pre}$，则必须先施加公开的对齐 / 回缩算子：$\widetilde Y^{pre(s)} = Rescale_{out}\!\big(Y^{pre(s)}; S_{out,pre}\rightarrow S_{out,head}\big)$

   然后将输出头内部的偏置相加正式定义为：$Y^{(s)}=\widetilde Y^{pre(s)}+b^{(out,s)}$ 

   当且仅当 $quant\_cfg\_id$ 指定 $S_{out,pre}=S_{out,head}$ 时，$Rescale_{out}$ 定义为恒等映射；否则 $Rescale_{out}$ 必须按 $quant\_cfg\_id$ 唯一指定的舍入 / 截断规则执行，不存在第二种解释。

2. 最终多头平均必须在共同尺度上执行。协议固定要求最终平均发生在输出头共同尺度 $S_{out,head}$ 上，并且最终 logits 继续保持这一共同尺度，即$S_{out}=S_{out,head}$，因而先在域内计算：$Y^{sum}_{i,c}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}$，再乘以公开整数 $K_{out}$ 在 $\mathbb F_p$ 中的逆元 $K_{out}^{-1}$，得到域内平均值与最终 logits：$Y_{i,c}=Y_{i,c}^{avg}=K_{out}^{-1}\cdot Y^{sum}_{i,c}$

   换言之，本协议严格禁止在最终平均之后再额外引入第二个平均后重缩放子步骤。当前主协议中，最终平均的唯一正式语义就是上式；任何需要平均后重缩放的方案都不属于本文当前协议，必须另行定义为新的协议版本。

3. 上述 $Rescale_{out}$、舍入方式，以及乘以 $K_{out}^{-1}$ 的平均语义，全部必须并入 $quant\_cfg\_id$ 所标识的公开量化配置；证明者与验证者不得各自选择不同的平均语义。

### 0.5 模型参数

#### 0.5.1 隐藏层参数族

对任意隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，以及该层中的每个注意力头 $r\in\{0,1,\ldots,K_{hid}^{(\ell)}-1\}$，定义：

- 投影矩阵 $W^{(\ell,r)}\in\mathbb F_p^{d_{in}^{(\ell)}\times d_h^{(\ell)}}$
  
- 源方向注意力向量 $a_{src}^{(\ell,r)}\in\mathbb F_p^{d_h^{(\ell)}}$
  
- 目标方向注意力向量 $a_{dst}^{(\ell,r)}\in\mathbb F_p^{d_h^{(\ell)}}$

在“单个隐藏层模板”章节的数学推导中，统一把 $W^{(\ell,r)},a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}$，记作 $W^{(r)},a_{src}^{(r)},a_{dst}^{(r)}$。这一重写只用于公式书写；代码实现中的对象名、序列化器字段名、证明标签以及日志字段名都必须保留完整层指标 $(\ell,r)$。

#### 0.5.2 输出层参数

对最终输出层中的每个输出注意力头 $s\in\{0,1,\ldots,K_{out}-1\}$，定义：

- 输出投影矩阵 $W^{(out,s)}\in\mathbb F_p^{d_{in}^{(L)}\times C}$
  
- 输出层源方向注意力向量 $a_{src}^{(out,s)}\in\mathbb F_p^C$
  
- 输出层目标方向注意力向量 $a_{dst}^{(out,s)}\in\mathbb F_p^C$
  
- 输出层偏置向量 $b^{(out,s)}\in\mathbb F_p^C$

其中 $d_{in}^{(L)}=d_{cat}^{(L-1)}$，也就是最终输出层接收最后一个隐藏层拼接后的表示。

本文沿用把官方注意力向量按源/目标方向分拆的等价写法：$\mathbf a^{(out,s)\top}[u\,\|\,v] = a_{src}^{(out,s)\top}u + a_{dst}^{(out,s)\top}v$。其中 $[u\,\|\,v]$ 表示拼接。这只是代数重写，不改变官方GAT的前向语义。

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

10. 指数输出：$U^{(r)}\in\mathbb F_p^E \qquad U_k^{(r)}=ExpMap(\Delta_k^{+(r)})$

11. 分母广播：$Sum^{edge(r)}\in\mathbb F_p^E \qquad Sum_k^{edge(r)}=Sum_{dst(k)}^{(r)}$

12. 逆元广播：$inv^{edge(r)}\in\mathbb F_p^E \qquad inv_k^{edge(r)}=inv_{dst(k)}^{(r)}$

13. 归一化权重：$\alpha^{(r)}\in\mathbb F_p^E \qquad \alpha_k^{(r)}=U_k^{(r)}\cdot inv_k^{edge(r)}$

14. 聚合前压缩加权特征：$\widehat v_{pre}^{\star(r)}\in\mathbb F_p^E \qquad \widehat v_{pre,k}^{\star(r)}=\alpha_k^{(r)}H_{src,k}^{\star,edge(r)}$

#### 0.6.4 单个隐藏层模板的拼接阶段变量

定义拼接结果为：$H_{cat}\in\mathbb F_p^{N\times d_{cat}}$

其中对任意 $i\in\{0,1,\ldots,N-1\}$、$r\in\{0,1,\ldots,K_{hid}-1\}$、$j\in\{0,1,\ldots,d_h-1\}$，都有 $H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}$

定义拼接压缩挑战：$\xi_{cat}\in\mathbb F_p$

定义拼接压缩特征：$H_{cat}^{\star}\in\mathbb F_p^N \quad H_{cat,i}^{\star}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}\xi_{cat}^m$

#### 0.6.5 输出层变量

输出层有两类共享维：一类是输入共享维 $d_{cat}$，另一类是类别共享维 $C$。 因此我们不仅需要 $d_{cat}$ 域，还需要显式引入类别共享域。

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
11. 输出层指数输出：$U^{(out,s)}\in\mathbb F_p^E, \qquad U_k^{(out,s)}=ExpMap(\Delta_k^{+(out,s)})$
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
25. 输出层加偏置后的 头 输出：$Y^{(s)}\in\mathbb F_p^{N\times C}, \qquad Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}$

26. 输出层 头 输出的类别压缩：$Y^{\star(s)}\in\mathbb F_p^N, \qquad Y_i^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{(s)}\xi_{out}^c$
27. 输出层 头 输出压缩广播：$Y^{\star,edge(s)}\in\mathbb F_p^E, \qquad Y_k^{\star,edge(s)}=Y_{dst(k)}^{\star(s)}$
28. 最终输出：先定义 $Y^{sum}\in\mathbb F_p^{N\times C}$ 与域内平均 $Y^{avg}\in\mathbb F_p^{N\times C}$：$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}, \qquad Y_{i,c}^{avg}=K_{out}^{-1}\cdot Y_{i,c}^{sum}$

其中 $K_{out}^{-1}$ 是整数 $K_{out}$ 在 $\mathbb F_p$ 中的逆元。本文协议要求最终 logits 与输出头使用同一共同尺度 $S_{out}=S_{out,head}$，因此正式定义 $Y\in\mathbb F_p^{N\times C}, \quad Y_{i,c}=Y_{i,c}^{avg}$ 

29. 最终输出的类别压缩：$Y^{\star}\in\mathbb F_p^N, \qquad Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c$

30. 最终输出压缩广播：$Y^{\star,edge}\in\mathbb F_p^E, \qquad Y_k^{\star,edge}=Y_{dst(k)}^{\star}$

### 0.7 工作域、选择器、辅助公开列与展平规则

#### 0.7.1 工作域总表

使用七类工作域：

1. 特征检索域：$\mathbb H_{FH}$

2. 边域：$\mathbb H_{edge}$

3. 输入共享维域：$\mathbb H_{in}$

4. 隐藏层单头共享维域：$\mathbb H_{d_h}$

5. 拼接共享维域：$\mathbb H_{cat}$

6. 输出层类别共享维域：$\mathbb H_C$

7. 节点域：$\mathbb H_N$


对任意工作域 $\mathbb H_{\mathcal D}$，都记其大小为 $n_{\mathcal D}$，生成元为 $\omega_{\mathcal D}$，零化多项式为：$Z_{\mathcal D}(X)=X^{n_{\mathcal D}}-1$

#### 0.7.2 拉格朗日基函数与首尾指示多项式

对任意工作域 $\mathbb H_{\mathcal D}$，记其拉格朗日基函数为：$L_0^{(\mathcal D)}(X),L_1^{(\mathcal D)}(X),\ldots,L_{n_{\mathcal D}-1}^{(\mathcal D)}(X)$。（特点：在域的第$i$取值$1$，在其他点上取值$0$）

首点指示多项式记为：$First_{\mathcal D}(X)=L_0^{(\mathcal D)}(X)$

末点指示多项式记为：$Last_{\mathcal D}(X)=L_{n_{\mathcal D}-1}^{(\mathcal D)}(X)$

#### 0.7.3 公共有效区选择器与按层实例化规则 

1. 边域有效区选择器：$Q_{edge}^{valid}[k]= \begin{cases} 1 &0\le k\le E-1,\\ 0,&E\le k\le n_{edge}-1 \end{cases}$

2. 节点域有效区选择器：$Q_N[i]= \begin{cases} 1 &0\le i\le N-1,\\ 0,&N\le i\le n_N-1 \end{cases}$

3. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，输入共享维域有效区选择器族定义为：$Q_{in}^{valid,(\ell)}[m]= \begin{cases} 1 &0\le m\le d_{in}^{(\ell)}-1 \\ 0 &d_{in}^{(\ell)}\le m\le n_{in}-1 \end{cases}$
   
4. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，隐藏层单头共享维域有效区选择器族定义为：$Q_{d_h}^{valid,(\ell)}[j]= \begin{cases} 1 &0\le j\le d_h^{(\ell)}-1 \\ 0 &d_h^{(\ell)}\le j\le n_{d_h}-1 \end{cases}$ 
   
5. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，拼接共享维域有效区选择器族定义为：$Q_{cat}^{valid,(\ell)}[m]= \begin{cases} 1 &0\le m\le d_{cat}^{(\ell)}-1 \\ 0 &d_{cat}^{(\ell)}\le m\le n_{cat}-1 \end{cases}$
   
6. 输出层输入共享维有效区选择器定义为：$Q_{cat}^{valid,(out)}[m]= \begin{cases} 1 &0\le m\le d_{in}^{(L)}-1,\\ 0,&d_{in}^{(L)}\le m\le n_{cat}-1 \end{cases}$
   
   虽然 $d_{in}^{(L)}=d_{cat}^{(L-1)}$，正式协议仍然固定把 $Q_{cat}^{valid,(out)}$ 与 $Q_{cat}^{valid,(L-1)}$ 视为两个不同的公共对象。序列化器、解析器、transcript、商多项式构造器与验证者都必须将它们按两个不同的标签处理不得复用，不得合并，也不得依赖字节串相等来省略其中之一。
   
7. 输出层类别共享维域有效区选择器：$Q_C^{valid}[c]= \begin{cases} 1 &0\le c\le C-1,\\ 0,&C\le c\le n_C-1 \end{cases}$

8. 在当前隐藏层模板的数学推导中，统一采用以下局部记号：$Q_{in}^{valid}:=Q_{in}^{valid,(\ell)} \quad Q_{d_h}^{valid}:=Q_{d_h}^{valid,(\ell)} \quad
   Q_{cat}^{valid}:=Q_{cat}^{valid,(\ell)}$ 
   
   这一定义仅用于公式书写。正式协议中的 transcript、序列化器、证明对象清单、商多项式构造器以及验证者/解析器，必须始终使用带层指标的选择器族；任何实现都不得将不同层的选择器视为同一个公共对象。


#### 0.7.4 分组选择器与 batch 边界选择器

由于公开边序列已经先按 $edge\_gid$ 分块、再在每个图块内部按 $dst(k)$ 非降序排列，因此定义：

1. 图内目标组起点选择器：$Q_{new}^{edge}[k]= \begin{cases} 1 &k=0 \\ 1 &1\le k\le E-1\ \text{且}\ dst(k)\ne dst(k-1) \\ 0 &1\le k\le E-1\ \text{且}\ dst(k)=dst(k-1) \\ 0 &k\ge E  \end{cases}$

2. 图内目标组末尾选择器：$Q_{end}^{edge}[k]= \begin{cases} 1 &k=E-1 \\ 1 &0\le k\le E-2\ \text{且}\ dst(k+1)\ne dst(k) \\ 0 &0\le k\le E-2\ \text{且}\ dst(k+1)=dst(k) \\ 0 &k\ge E  \end{cases}$

3. 图块起点选择器：$Q_{graph\_new}^{edge}[k]= \begin{cases} 1 &k=0 \\ 1 &1\le k\le E-1\ \text{且}\ edge\_gid(k)\ne edge\_gid(k-1) \\ 0 &1\le k\le E-1\ \text{且}\ edge\_gid(k)=edge\_gid(k-1) \\ 0 &k\ge E \end{cases}$

4. 图块末尾选择器：$Q_{graph\_end}^{edge}[k]= \begin{cases} 1 &k=E-1 \\ 1 &0\le k\le E-2\ \text{且}\ edge\_gid(k+1)\ne edge\_gid(k) \\ 0 &0\le k\le E-2\ \text{且}\ edge\_gid(k+1)=edge\_gid(k) \\ 0 &k\ge E \end{cases}$

5. batch-aware 组起点选择器：$Q_{new,batch}^{edge}[k]=1-\big(1-Q_{graph\_new}^{edge}[k]\big)\big(1-Q_{new}^{edge}[k]\big)$
   
6. 批量感知组末尾选择器：$Q_{end,batch}^{edge}[k]=1-\big(1-Q_{graph\_end}^{edge}[k]\big)\big(1-Q_{end}^{edge}[k]\big)$

7. 正式协议统一记号：$Q_{grp\_new}^{edge}:=\begin{cases}Q_{new}^{edge} & G_{batch}=1 \\Q_{new,batch}^{edge} & G_{batch}>1 \end{cases}\quad Q_{grp\_end}^{edge}:=\begin{cases}Q_{end}^{edge} & G_{batch}=1 \\Q_{end,batch}^{edge} & G_{batch}>1 \end{cases}$ 

因此，代码实现、序列化器、商多项式构造器与验证者/解析器中，凡是“组起点 / 组末尾”相关的状态机，都必须以 $Q_{grp\_new}^{edge}$、$Q_{grp\_end}^{edge}$ 为唯一正式输入；即使 $G_{batch}=1$，实现也必须使用这两个批处理感知标签，而不得回退到旧的 $Q_{new}^{edge},Q_{end}^{edge}$ 标签。

由此，后文所有隐藏层 / 输出层的 $C_{max}$ 状态机、$PSQ$ 状态机、对应的商多项式恒等式，以及验证者对这些恒等式的检查，除非该条公式标题中显式写出“单图局部定义”，否则都必须统一使用批处理感知选择器 $Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$ 

#### 0.7.5 公共枚举列与索引辅助列

为了把所有“由位置直接导出的值”也代数化，定义以下公共辅助列：

1. 节点域枚举列：$Idx_N[i]=i \quad 0\le i\le n_N-1$

2. 输入共享维枚举列：$Idx_{in}[m]=m \quad 0\le m\le n_{in}-1$

3. 隐藏层共享维枚举列：$Idx_{d_h}[j]=j \quad 0\le j\le n_{d_h}-1$

4. 拼接共享维枚举列：$Idx_{cat}[m]=m \quad 0\le m\le n_{cat}-1$

5. 输出层类别枚举列：$Idx_C[c]=c \quad 0\le c\le n_C-1$

6. 特征检索表端行索引列：对每个 $u=v\cdot d_{in}+j$，定义：$Row_{feat}^{tbl}[u]=v \quad Col_{feat}^{tbl}[u]=j$

7. 特征检索查询端局部节点索引列：对每个 $q=i\cdot d_{in}+j$，定义：$Row_{feat}^{qry}[q]=i \quad Col_{feat}^{qry}[q]=j$


对这些离散列做插值即可得到相应公共多项式。

#### 0.7.6 展平规则

对任意矩阵 $M\in\mathbb F_p^{r\times c}$，定义行优先展平索引：$\operatorname{flat}_{r,c}(i,j)=i\cdot c+j$

特别地：

- 对 $N\times d_{in}$ 矩阵，使用索引 $i\cdot d_{in}+j$；
  
- 对 $N\times d_h$ 矩阵，使用索引 $i\cdot d_h+j$；
  
- 对 $N\times d_{cat}$ 矩阵，使用索引 $i\cdot d_{cat}+m$；
  
- 对 $N\times C$ 矩阵，使用索引 $i\cdot C+c$。

#### 0.7.7 CRPC 编码

对矩阵乘法：$A\in\mathbb F_p^{m\times\ell} \quad B\in\mathbb F_p^{\ell\times n} \quad C=A\cdot B\in\mathbb F_p^{m\times n}$ 

定义输出系数多项式：$P_C(X)=\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}C_{i,j}X^{i\cdot n+j}$

对共享维 $t\in\{0,1,\ldots,\ell-1\}$，定义：$A_t^{\langle n\rangle}(X)=\sum_{i=0}^{m-1}A_{i,t}X^{i\cdot n} \quad B_t(X)=\sum_{j=0}^{n-1}B_{t,j}X^j$

于是：$P_C(X)=\sum_{t=0}^{\ell-1}A_t^{\langle n\rangle}(X) \cdot B_t(X)$

#### 0.7.8 张量绑定子证明

所有形如：$a_t=\sum_i M_{i,t}\chi_i(y)$ 或 $b_t=\sum_j W_{t,j}\psi_j(y)$ 的一维折叠向量，都必须与其来源矩阵承诺强绑定。

最终绑定子证明族记为：$\Pi_{bind}= \big( \pi_{bind}^{feat},\ \{\pi_{bind}^{hidden,(\ell,r)}\}_{\ell,r\ \text{按层序与头序}},\ \{\pi_{bind}^{concat,(\ell)}\}_{\ell=1}^{L-1},\ \pi_{bind}^{out,0},\ldots,\pi_{bind}^{out,K_{out}-1},\ \pi_{bind}^{out,avg} \big)$

其中：

- 对每个隐藏层 $\ell$、每个隐藏头 $r$，$\pi_{bind}^{hidden,(\ell,r)}$ 负责该头的投影、源注意力、目标注意力、压缩与聚合绑定；
  
- 对每个隐藏层 $\ell$，$\pi_{bind}^{concat,(\ell)}$ 负责该层拼接对象与压缩对象的一致性绑定；
  
- 对每个输出头 $s$，$\pi_{bind}^{out,s}$ 负责输出投影、源注意力、目标注意力、聚合前压缩、加偏置后输出压缩等绑定；
  
- $\pi_{bind}^{out,avg}$ 负责最终多头平均输出 $Y$ 与各个 $Y^{(s)}$ 之间的一致性绑定。

每个子证明都必须使用独立的域分离标签初始化内部 transcript。

主实验中的绑定证明必须采用固定接口语义。每个绑定证明都要明确写出绑定对象名称、来源承诺、折叠后承诺、masked evaluation、挑战点和绑定模式。证明对象中只允许出现辅助承诺、masked opening 和 KZG opening 证明；不得出现裸折叠向量、裸模型参数评值、裸中间激活或裸内积结果。验证者必须检查绑定模式、对象标签、可见性、承诺顺序、opening 类型和 transcript 域分离标签是否全部一致。

### 0.8 填充、零分母、Softmax 可行性与缓存要求

#### 0.8.1 填充规则

对任意工作域 $\mathbb H_{\mathcal D}$，若真实长度为 $L_{\mathcal D}$、工作域长度为 $n_{\mathcal D}$，则所有离散列必须补到长度 $n_{\mathcal D}$。

对所有 $t\in\{L_{\mathcal D},L_{\mathcal D}+1,\ldots,n_{\mathcal D}-1\}$，统一规定：

- 见证值补零；
  
- 表值补零；
  
- 查询值补零；
  
- 重数补零；
  
- 由有效区选择器屏蔽无效位置；
  
- 所有状态机在填充区都必须保持常值；实现时不得改写成其他等价但不同形态的递推规则。

#### 0.8.2 零分母冲突

对任意 LogUp 子系统 $\mathcal L$，零分母集合定义为：$Bad_{\mathcal L} = \{-Table[t]\mid t\text{ 在表有效区}\} \cup \{-Query[t]\mid t\text{ 在查询有效区}\}$ 

要求：每个 lookup / 路由子系统的挑战必须在决定其基础值的承诺固定之后再生成，并且语义上必须满足：$\beta_{\mathcal L}\notin Bad_{\mathcal L}$

#### 0.8.3 Softmax 可行性

要求每个有效节点至少有一条入边。公共输入构造阶段必须保证：$\#\{k\mid dst(k)=i\}\ge 1 \quad \forall i\in\{0,1,\ldots,N-1\}$。若原始图不满足该条件，则必须在公开预处理阶段显式加入自环后，才允许进入本文协议。

同时还必须满足：

1. 所有合法输入下的实数域最大分母严格小于模数 $p$；
   
2. 量化后最小聚合值至少为 $1$。

这样才能保证：$Sum_i^{(r)}\ne 0$ 以及 $Sum_i^{(out,s)}\ne 0\;\forall s\in\{0,1,\ldots,K_{out}-1\}$ 在域内成立。

#### 0.8.4 热路径对象与缓存要求

热路径对象包括：所有域元素、所有挑战值、所有曲线点、SRS、静态表承诺、模型承诺、全部 FFT / IFFT 计划以及商计算的扩展域计划。

程序启动后必须一次性缓存：

- $PK$；
  
- $VK_{KZG}$；

- $VK_{static}$；
  
- $VK_{model}$；
  
- $\mathbb H_{FH},\mathbb H_{edge},\mathbb H_{in},\mathbb H_{d_h},\mathbb H_{cat},\mathbb H_C,\mathbb H_N$ 的 FFT / IFFT 计划；
  
- 商多项式计算所需的扩展域执行计划

单次证明或验证期间不得重复构造这些对象。

#### 0.8.5 公开元数据与安全设置字段

为保证证明对象、SRS、静态表、模型参数、量化配置和隐私策略之间一致，证明对象中必须携带一份公开元数据。命名保持直接、易读，实现时必须按下列字段固定顺序序列化并吸入 transcript。

1. `protocol_name`：协议名称。主实验固定为 `zkgat`。

2. `proof_task`：证明任务。主实验固定为 `node_classification_full_gat`，表示证明完整多层多头 GAT 前向、输出注意力头、输出偏置和最终平均。

3. `srs_id`：本次使用的 SRS 标识，绑定 SRS 文件、曲线参数、次数界、工作域大小和 verifier key。

4. `srs_type`：SRS 类型。主实验只能使用 `trusted_srs` 或 `mpc_srs`，不得出现 `seeded_tau`、`deterministic_tau`、`test_tau` 或任何证明者可见陷门模式。

5. `srs_degree_bound`：SRS 次数界，必须覆盖所有工作域、多项式、商多项式和 blinding 后的次数界。

6. `privacy_policy`：隐私策略。主实验固定为 `model_and_trace_hidden`，表示隐藏模型参数和全部中间推理状态；图结构和节点特征是否隐藏由 `input_privacy` 单独声明。

7. `input_privacy`：输入隐私。主实验固定为 `public_graph_public_features`，即图拓扑和节点特征作为公开输入或公开承诺对象，不默认隐藏。若以后需要隐藏输入特征或图结构，必须另写新的实现笔记和证明范围。

8. `output_mode`：输出公开方式。主实验固定为 `public_logits`，最终 logits 公开并被证明绑定。若以后只公开 label，需要额外实现 argmax 证明。

9. `opening_mode`：KZG opening 策略。主实验固定为 `domain_grouped_batch_opening`。

10. `binding_mode`：外点评值绑定策略。主实验固定为 `zk_subproof_for_crpc`，其中矩阵乘法、注意力向量点积和压缩特征绑定使用零知识子证明。

11. `model_arch`：模型结构标识，绑定层数、隐藏头数、隐藏维度、输出头数、输出偏置和最终平均语义。

12. `model_params`：模型参数承诺集合标识。

13. `static_tables`：静态表承诺集合标识，绑定特征表、LeakyReLU 表、ELU 表、指数表和范围表。

14. `quant_config`：量化配置，绑定尺度、舍入、重缩放、符号解释和输出平均语义。

15. `domain_cfg`：工作域大小和生成元配置。

16. `dim_cfg`：图规模、batch 指针、层数、维度、类别数、范围位宽、输出头数等配置。

17. `encoding`：域元素、曲线点、数组长度、字节序、proof block tag 和 transcript 吸入顺序。

18. `padding_rule`：填充规则、有效区选择器和填充区状态机保持规则。

19. `degree_bound`：动态多项式、商多项式、blinding 多项式和 opening proof 的次数界策略。

验证者必须拒绝任何字段缺失、重复、顺序错误、值不匹配或出现未声明字段的 proof。

#### 0.8.6 大图稀疏执行与批处理假设

面向大图数据集时，本文固定采用如下执行假设；这些假设不改变 GAT 前向，只约束证明系统的工程实现：

1. 仅在真实边集上具体化注意力计算
   
   隐藏层层与输出层的注意力打分、softmax、PSQ、路由与 lookup 全部仅在公开边集 $E$ 上展开；禁止为了实现方便而将带掩码的注意力改写为全图稠密的两两计算。
   
2. 边域是主热路径
   
   对大图证明，边相关对象（如 $S,Z,M^{edge},\Delta^+,U,Sum^{edge},inv^{edge},\alpha$ 以及输出层对应族）必须以边并行的方式生成、缓存并进行开放证明；节点域和共享维域对象只作为必要的聚合与绑定接口存在。
   
3. 排序、自环与局部子图抽取属于公开预处理 
   
   为满足 $dst$ 非降序、softmax 至少一条入边、以及局部子图抽取要求，见证生成之前必须先完成稳定排序、加自环、局部节点重编号等公开预处理；这些步骤属于公开输入构造过程的一部分，不属于证明语义本身，但在实验报告中必须与核心证明成本分开统计。
   
4. 多图批处理中，softmax分组必须严格限制在图内
   
   若对多个图同时做批处理，则每个图的边序列、$dst$ 分组、PSQ 状态机与路由总和都必须在各图内部独立成立；禁止把不同图的边混入同一个softmax归一化组。
   
5. 缓存策略必须对大图友好

   对于大图实验，必须缓存 SRS、静态承诺、模型承诺、FFT/IFFT 执行计划、MSM 预处理以及商多项式扩展域执行计划；禁止在每次证明中重复构造这些对象。本文主协议不包含证明级分块语义：单个证明的承诺对象、transcript 吸入顺序与开放证明目标都必须一次性按照全局对象定义。若实现层为了节省内存而在内部采用分块生成见证值的工程手段，也不得将分块边界写入证明对象，不得改变最终承诺对象的全局定义，也不得改变 transcript 语义。


### 0.9 多项式可见性登记表与 proof payload 允许集合

为避免 proof object 泄露未授权 witness，本协议固定所有多项式的可见性类别。实现中不得依赖默认推断；每个动态多项式、静态承诺对象、外点评值、域内点评值、折叠标量与子证明字段都必须在 `ProofSchema` 中登记其 `PolynomialVisibility`。

可见性枚举为：

1. `Public`：验证者可由公共输入本地重建，或其值本身不携带私有witness。
2. `PublicOutput`：协议显式声明公开的输出对象。主实验默认公开最终logits；若改为只公开label，则必须改写输出证明范围。
3. `PrivateWitness`：模型参数、输入特征私有化场景下的特征值、中间激活、注意力分数、softmax 权重、查找 query、路由 query、PSQ 状态、折叠向量、累加器、中间矩阵和所有不属于 public / public-output 的对象。

主实验默认分类如下：

| 对象族 | 代表对象 | 可见性 | proof payload 规则 |
|---|---|---|---|
| 公开元数据 | `protocol_name, srs_id, srs_type, privacy_policy, proof_task, model_arch, quant_config, domain_cfg, dim_cfg` | `Public` | 明文写入并吸入 transcript |
| 公开拓扑 | `src,dst,node_ptr,edge_ptr,node_gid,edge_gid,Q_*^{valid},Q_{grp_new},Q_{grp_end},Idx_*` | `Public` | 验证者本地重建；若写入 proof 只允许写 hash / metadata |
| SRS / VK 摘要 | `srs_id, pk_hash, vk_hash, domain_srs_hash` | `Public` | 明文写入 metadata |
| 静态表承诺 | `[V_{T_H}], [V_{T_LReLU}], [V_{T_ELU}], [V_{T_exp}], [V_{T_range}]` | `Public` for commitments, table values按实验策略 | proof 只写承诺 / hash，不写完整私有表 |
| 模型承诺 | `[V_W], [V_a], [V_b]` | commitment 为 `Public`，参数值为 `PrivateWitness` | proof 只写承诺，不写裸参数 |
| 原始特征 | `P_H` | 主实验为 `Public`，因为 `input_privacy = public_graph_public_features`；只有未来切换为隐藏输入特征时才是 `PrivateWitness` | 主实验中由验证者按公开输入重建或按公开承诺绑定；隐藏输入特征版本必须另写规则并启用 masked opening |
| 隐藏层投影 | `P_{H'}, P_{E_src}, P_{E_dst}, P_{H^\star}` | `PrivateWitness` | 只允许承诺与 masked evaluation |
| 边域注意力链 | `P_S,P_Z,P_M^{edge},P_{\Delta^+},P_U,P_{Sum}^{edge},P_{inv}^{edge},P_\alpha` | `PrivateWitness` | 禁止输出完整列或裸随机点评值 |
| lookup 对象 | `P_Table,P_Query,P_m,P_R` | `PrivateWitness`，公开表端可按策略为 `Public` | query、multiplicity、accumulator 默认 masked |
| route 对象 | `P_Table^{src/dst},P_Query^{src/dst},P_m,P_R` | `PrivateWitness` | 默认 masked |
| PSQ / 状态机 | `P_{PSQ},P_{C_max},P_{Acc},P_w,P_T` | `PrivateWitness` | 默认 masked |
| 折叠向量 | `P_a,P_b,P_Acc, \mu_*` | `PrivateWitness`，公开参数派生的常量向量可按策略 public | 不允许裸 `mu`；必须 tagged masked scalar 或子证明内部值 |
| 拼接与输出头 | `P_{H_cat},P_{Y'},P_{Y^{pre}},P_{\tilde Y^{pre}},P_{Y^{(s)}}` | `PrivateWitness` | 默认 masked |
| 最终 logits | `P_Y,P_{Y^\star}` | 主实验为 `PublicOutput` | 可明文公开并绑定；若只公开 label，则改为 `PrivateWitness` 并证明 argmax |
| 商多项式 | `Q_{\mathcal D}` | `PrivateWitness` | 承诺与 opening 均 masked |
| opening value | `Eval_dom, Eval_ext` | 随对象继承 | private 对象必须为 `MaskedPrivateValue` |

实现验收规则：

1. `ProofSchema` 必须以固定顺序列出全部对象、域名、层编号、头编号、可见性、是否进入 commitment、是否进入 opening、是否进入 quotient。
2. Serializer 输出任何 `FieldElement` 前必须带 `EvaluationKind`，取值只能是 `PublicValue` 或 `MaskedPrivateValue`。
3. Verifier 必须执行 `private_plain_value_count == 0` 检查。
4. 未登记对象、重复对象、未知 tag、缺失可见性字段、private 对象明文输出，均为拒绝条件。

### 0.10 SRS manifest、文件格式与 setup 绑定

主实验不允许使用内存中临时生成且不可审计的 SRS。每个 SRS 必须由 `srs_manifest.json` 描述，并与证明对象中的 `srs_id` 绑定。manifest 至少包含：

| 字段 | 含义 |
|---|---|
| `srs_id` | SRS 全局唯一标识，建议为 manifest canonical bytes 的 SHA-256 或 Blake2 摘要 |
| `curve_id` | 曲线与 pairing 后端标识 |
| `field_modulus_id` | 有限域模数标识 |
| `srs_type` | 只能为 `trusted_srs` 或 `mpc_srs` |
| `ceremony_transcript_hash` | trusted setup / MPC ceremony transcript 的摘要；若为内部生成测试材料，不能进入主实验 |
| `degree_bound` | `D_max` |
| `domain_sizes` | `{FH, edge, in, d_h, cat, C, N}` 的大小 |
| `pk_coef_hash` | coefficient-basis SRS 文件摘要 |
| `pk_lagrange_hashes` | 各工作域 Lagrange-basis SRS 文件摘要 |
| `vk_hash` | verifier key 文件摘要 |
| `encoding` | 曲线点、域元素、数组长度和字节序规则 |
| `generated_at` | 生成时间，仅用于审计，不进入性能指标 |
| `generator_version` | SRS 生成器版本 |
| `security_level` | 目标安全强度，例如 128-bit |

文件布局固定为一个以 `srs_id` 命名的 SRS 目录。该目录必须包含 SRS manifest、coefficient-basis proving key、七个工作域对应的 Lagrange-basis proving key，以及 verifier key。文件名必须稳定、可校验，并与 manifest 中的 hash 一一对应。

加载规则：

1. Prover 加载 `pk_coef.bin` 与所需 `pk_lagrange_*.bin`，不得加载或生成 `tau`。
2. Verifier 只加载 `vk.bin` 与 `srs_manifest.json`。
3. Proof metadata 中的 `srs_id`、`srs_type`、`srs_degree_bound`、`encoding` 必须与本地 manifest 一致。
4. 任何 SRS 文件 hash 不匹配、degree bound 不足、domain size 不匹配、曲线 ID 不匹配，验证者必须拒绝。
5. SRS manifest 不计入 proof size，但其 hash 必须进入 transcript。
6. SRS 生成工具必须与 prover、verifier、benchmark 主程序分离。最终主实验程序只能加载 SRS 文件，不能在证明或验证时生成、恢复或派生 tau。最终提交分支不包含 `seed -> tau`、`deterministic_tau`、`test_tau` 代码。


### 0.11 与当前项目实现对齐的主实验边界

本笔记用于指导当前 zkGAT 项目代码整理，而不是重新定义一个与现有工程无关的新系统。主实验必须保持当前项目已经实现并验证过的 GAT 前向语义：隐藏层多头注意力、隐藏层 ELU、隐藏层头维拼接、输出层注意力头、输出头偏置、最终输出头平均，以及节点分类任务口径都不得改变。

本次重构的主要范围集中在密码协议与实验合规层：安全 SRS、KZG 承诺与 opening、模型参数隐藏承诺、私有 trace 掩蔽、ProofSchema 可见性登记、公开元数据、验证器安全 gate、安全测试、baseline proof-scope 对齐和结果导出。除非为了接入掩蔽多项式、metadata 或 proof schema 标签，GAT 前向、route、PSQ、lookup、边排序、batch 语义和量化语义都不应被重写。

主实验固定为节点分类。图结构和节点特征按公开输入处理；模型参数、中间激活、注意力分数、softmax 权重、lookup witness、route witness、PSQ 状态和折叠向量按私有见证处理；最终 logits 按公开输出处理。这一边界必须同时体现在配置、public metadata、transcript、proof schema、verifier gate、实验结果表和论文叙述中。

现有 ezkl、zkDL、zkGPT 对比实验不需要迁移到 zkGAT 的 SRS 或 opening 方案；它们只需要补齐 proof scope、输入输出口径、setup 假设、proof size 统计口径和硬件环境说明。主实验与对比实验的所有结果都必须在同一模型、同一数据预处理、同一量化配置、同一节点分类任务下重新导出。

## 1. 参数生成

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
- 静态表 $T_H,T_{LReLU},T_{ELU},T_{exp},T_{range}$；
- 全部隐藏层参数族：对每个隐藏层$\ell=1,\ldots,L-1$，包含该层全部 $r=0,\ldots,K_{hid}^{(\ell)}-1$ 个头的 $W^{(\ell,r)},a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}$ 
- 输出层参数族 $\{W^{(out,s)},a_{src}^{(out,s)},a_{dst}^{(out,s)},b^{(out,s)}\}_{s=0}^{K_{out}-1}$ 
- SRS 类型 $srs\_type$，主实验必须为 `trusted_srs` 或 `mpc_srs`；
- SRS 标识 $srs\_id$ 与 SRS 次数界 $srs\_degree\_bound$；
- 隐私策略 $privacy\_policy$，主实验固定为 `model_and_trace_hidden`，包括私有多项式分类、domain-vanishing blinding 次数和 masked opening 规则；
- 证明任务 $proof\_task$，主实验固定为 `node_classification_full_gat`。


### 1.2 输出

参数生成算法输出：

- KZG 证明键：$PK_{coef}$ 与各工作域基于拉格朗日的SRS $\{PK_{lag}^{(\mathcal D)}\}_{\mathcal D\in\{FH,edge,in,d_h,cat,C,N\}}$

- KZG 验证键：$VK_{KZG}$

- SRS 安全元数据：$(srs\_id,srs\_type,srs\_degree\_bound)$

- 隐私策略元数据：$privacy\_policy$

- 静态表验证键：$VK_{static}$

- 模型验证键：$VK_{model}$

- 各工作域：$\mathbb H_{FH},\mathbb H_{edge},\mathbb H_{in},\mathbb H_{d_h},\mathbb H_{cat},\mathbb H_C,\mathbb H_N$

- 各工作域零化多项式：$Z_{FH},Z_{edge},Z_{in},Z_{d_h},Z_{cat},Z_C,Z_N$

### 1.3 工作域

为避免在多层GAT的实现中为每一层重复使用一套不同长度的共享维工作域，正式协议统一采用全局共享域口径：

- 对全部隐藏层的输入共享维，统一用一个 $\mathbb H_{in}$；
  
- 对全部隐藏层单头输出共享维，统一用一个 $\mathbb H_{d_h}$；
  
- 对全部隐藏层拼接共享维与最终输出层输入共享维，统一用一个 $\mathbb H_{cat}$。

因此，先定义三类全局最大配置概况：

- $d_{in}^{max}=\max\{d_{in}^{(1)},d_{in}^{(2)},\ldots,d_{in}^{(L)}\}$；
  
- $d_h^{max}=\max\{d_h^{(1)},d_h^{(2)},\ldots,d_h^{(L-1)}\}$；
  
- $d_{cat}^{max}=\max\{d_{cat}^{(1)},d_{cat}^{(2)},\ldots,d_{cat}^{(L-1)}\}$。

在任何具体层 $\ell$ 的实例化中：

- 第 $\ell$ 层隐藏模板只占用 $\mathbb H_{in}$ 的前 $d_{in}^{(\ell)}$ 个有效位置；
  
- 第 $\ell$ 层隐藏层头模板只占用 $\mathbb H_{d_h}$ 的前 $d_h^{(\ell)}$ 个有效位置；
  
- 第 $\ell$ 层拼接模板与最终输出层投影模板，只占用 $\mathbb H_{cat}$ 的前 $d_{cat}^{(\ell)}$ 或 $d_{in}^{(L)}=d_{cat}^{(L-1)}$ 个有效位置；
  
- 其余位置统一由公共有效区选择器屏蔽，并按第 0.8.1 节的填充规则补零。

#### 1.3.1 特征检索域

取最小二次幂长度 $n_{FH}$ 满足：$n_{FH}\ge \max\{N_{total}d_{in}^{(1)},Nd_{in}^{(1)}\}+2$

定义：$\mathbb H_{FH}=\{1,\omega_{FH},\omega_{FH}^2,\ldots,\omega_{FH}^{n_{FH}-1}\} \quad Z_{FH}(X)=X^{n_{FH}}-1$

#### 1.3.2 边域

取最小二次幂长度 $n_{edge}$ 满足：$n_{edge}\ge \max\{N,E,Nd_h^{max},|T_{LReLU}|,|T_{ELU}|,|T_{exp}|,2^B\}+2$

定义：$\mathbb H_{edge}=\{1,\omega_{edge},\omega_{edge}^2,\ldots,\omega_{edge}^{n_{edge}-1}\} \quad Z_{edge}(X)=X^{n_{edge}}-1$

#### 1.3.3 输入共享维域

取最小二次幂长度 $n_{in}$ 满足：$n_{in}\ge d_{in}^{max}+2$

定义：$\mathbb H_{in}=\{1,\omega_{in},\omega_{in}^2,\ldots,\omega_{in}^{n_{in}-1}\} \quad Z_{in}(X)=X^{n_{in}}-1$

#### 1.3.4 隐藏层单头共享维域

取最小二次幂长度 $n_{d_h}$ 满足：$n_{d_h}\ge d_h^{max}+2$

定义：$\mathbb H_{d_h}=\{1,\omega_{d_h},\omega_{d_h}^2,\ldots,\omega_{d_h}^{n_{d_h}-1}\} \quad Z_{d_h}(X)=X^{n_{d_h}}-1$

#### 1.3.5 拼接共享维域

取最小二次幂长度 $n_{cat}$ 满足：$n_{cat}\ge d_{cat}^{max}+2$

定义：$\mathbb H_{cat}=\{1,\omega_{cat},\omega_{cat}^2,\ldots,\omega_{cat}^{n_{cat}-1}\} \quad Z_{cat}(X)=X^{n_{cat}}-1$

#### 1.3.6 输出层类别共享维域

取最小二次幂长度 $n_C$ 满足：$n_C\ge C+2$

定义：$\mathbb H_C=\{1,\omega_C,\omega_C^2,\ldots,\omega_C^{n_C-1}\} \quad Z_C(X)=X^{n_C}-1$

#### 1.3.7 节点域

取最小二次幂长度 $n_N$ 满足：$n_N\ge N+2$

定义：$\mathbb H_N=\{1,\omega_N,\omega_N^2,\ldots,\omega_N^{n_N-1}\} \quad Z_N(X)=X^{n_N}-1.$

### 1.4 安全 KZG 初始化与 SRS 生成

参数生成阶段执行可信设置或 MPC ceremony：$$\tau\xleftarrow{\$}\mathbb F_p$$

该陷门只允许在 setup 临时上下文中存在。生成 $PK$ 与 $VK_{KZG}$ 后，setup 程序必须销毁 $\tau$，并输出可验证的 `srs_id`。主实验代码、prover、verifier、benchmark runner、serializer、manifest exporter 和 debug logger 均不得保存或访问 $\tau$。

取次数上界时，必须覆盖：

- 七个工作域上的全部 lookup / 路由 / 状态机 / 绑定 / 商多项式；
- 全部隐藏层族对象；
- 最终输出层的全部输出头族对象；
- 最终多头平均输出 $Y,Y^{\star},Y^{\star,edge}$ 及其对应约束；
- domain-vanishing blinding 后的多项式次数界。

在多层 GAT 中，$L$ 与各层头数会增加对象个数，但不会改变单个对象所属工作域的次数界模板。记：

- $d_{in}^{max}=\max\{d_{in}^{(1)},d_{in}^{(2)},\ldots,d_{in}^{(L)}\}$；
- $d_h^{max}=\max\{d_h^{(1)},d_h^{(2)},\ldots,d_h^{(L-1)}\}$；
- $d_{cat}^{max}=\max\{d_{cat}^{(1)},d_{cat}^{(2)},\ldots,d_{cat}^{(L-1)}\}$。

基础次数界为：

$$D_{base}= \max\{ 3n_{FH}+8,\ 3n_{edge}+8,\ 2n_{in}+8,\ 2n_{d_h}+8,\ 2n_{cat}+8,\ 2n_C+8,\ 2n_N+8,\ Nd_{cat}^{max}+\max\{d_{in}^{max},C\},\ NC+1 \}$$

若 `privacy_policy` 指定每个私有多项式使用盲化次数$d_{blind}$，则正式 SRS 次数界为：$$D_{max}=D_{base}+d_{blind}+\max_{\mathcal D} n_{\mathcal D}$$

输出基于系数形式的证明键：$$PK_{coef}=\{[\tau^0]_1,[\tau^1]_1,\ldots,[\tau^{D_{max}}]_1\}$$

对每个工作域 $\mathbb H_{\mathcal D}$，输出或预计算基于拉格朗日基的 SRS：$$PK_{lag}^{(\mathcal D)}=\{[L_0^{(\mathcal D)}(\tau)]_1,\ldots,[L_{n_{\mathcal D}-1}^{(\mathcal D)}(\tau)]_1\}$$

输出验证键：$$VK_{KZG}=\{[1]_2,[\tau]_2\}=\{G_2,\tau G_2\}$$

实现要求：

1. `KZGKeyPair` 不得在主实验路径中包含 `tau` 字段；
2. `commit` 不得调用 `polynomial.evaluate(tau)`；
3. 基于系数形式的多项式使用 $PK_{coef}$ 做多基 MSM；
4. 求值基多项式使用对应工作域的 $PK_{lag}^{(\mathcal D)}$ 做多基 MSM；
5. 若某个对象没有可用工作域 Lagrange SRS，则必须先转为系数形式并使用 $PK_{coef}$；
6. 所有 SRS 与 $VK_{KZG}$ 的摘要必须写入 `srs_id` 并纳入 transcript。

### 1.5 静态表与模型承诺

#### 1.5.1 全局特征表多项式

定义：$P_{T_H}(X)=\sum_{v=0}^{N_{total}-1}\sum_{j=0}^{d_{in}-1}T_H[v,j]X^{v\cdot d_{in}+j}$

#### 1.5.2 LeakyReLU 表多项式

设 $T_{LReLU}$ 的第 $t$ 行是 $(T_{LReLU}[t,0],T_{LReLU}[t,1])$，定义：

$P_{T_{LReLU},x}(X)=\sum_{t=0}^{|T_{LReLU}|-1}T_{LReLU}[t,0]L_t^{(edge)}(X)$

$P_{T_{LReLU},y}(X)=\sum_{t=0}^{|T_{LReLU}|-1}T_{LReLU}[t,1]L_t^{(edge)}(X)$

#### 1.5.3 ELU 表多项式

设 $T_{ELU}$ 的第 $t$ 行是 $(T_{ELU}[t,0],T_{ELU}[t,1])$，定义：

$P_{T_{ELU},x}(X)=\sum_{t=0}^{|T_{ELU}|-1}T_{ELU}[t,0]L_t^{(edge)}(X)$

$P_{T_{ELU},y}(X)=\sum_{t=0}^{|T_{ELU}|-1}T_{ELU}[t,1]L_t^{(edge)}(X)$

#### 1.5.4 指数表多项式

设 $T_{exp}$ 的第 $t$ 行是 $(T_{exp}[t,0],T_{exp}[t,1])$，定义：

$P_{T_{exp},x}(X)=\sum_{t=0}^{|T_{exp}|-1}T_{exp}[t,0]L_t^{(edge)}(X)$

$P_{T_{exp},y}(X)=\sum_{t=0}^{|T_{exp}|-1}T_{exp}[t,1]L_t^{(edge)}(X)$

#### 1.5.5 范围表多项式

定义：$P_{T_{range}}(X)=\sum_{t=0}^{2^B-1}t\,L_t^{(edge)}(X)$

#### 1.5.6 静态表验证键

定义：$VK_{static} = \{ [V_{T_H}], [V_{T_{LReLU},x}], [V_{T_{LReLU},y}], [V_{T_{ELU},x}], [V_{T_{ELU},y}], [V_{T_{exp},x}], [V_{T_{exp},y}], [V_{T_{range}}] \}$

#### 1.5.7 模型承诺与模型验证键

对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$、以及该层中的每个注意力头 $r\in\{0,1,\ldots,K_{hid}^{(\ell)}-1\}$，承诺：$[V_{W^{(\ell,r)}}],[V_{a_{src}^{(\ell,r)}}],[V_{a_{dst}^{(\ell,r)}}]$

对每个输出头 $s\in\{0,1,\ldots,K_{out}-1\}$，承诺：$[V_{W^{(out,s)}}],\quad [V_{a_{src}^{(out,s)}}],\quad [V_{a_{dst}^{(out,s)}}],\quad [V_{b^{(out,s)}}]$

因此，$VK_{model}$ 由两部分按固定顺序拼接而成：先按层序和头序写入全部隐藏层承诺 $[V_{W^{(\ell,r)}}],[V_{a_{src}^{(\ell,r)}}],[V_{a_{dst}^{(\ell,r)}}]$；再按 $s=0,\ldots,K_{out}-1$ 写入全部输出层承诺 $[V_{W^{(out,s)}}],[V_{a_{src}^{(out,s)}}],[V_{a_{dst}^{(out,s)}}],[V_{b^{(out,s)}}]$。

模型参数值属于私有见证。`VK_model`中只能出现模型参数的隐藏承诺和承诺摘要，不能出现裸参数值，也不能出现裸参数的 opening。若模型参数承诺需要在多次证明中复用，复用对象仍然必须是隐藏承诺；验证者只使用承诺和后续零知识绑定证明检查模型一致性。

实现时，模型参数应先展平成独立的参数多项式，并在参数域上使用与动态私有多项式一致的隐藏承诺规则。`VK_model` 只保存隐藏后的模型承诺和摘要。所有投影矩阵、注意力向量和输出偏置在证明中的使用，都必须通过固定的零知识绑定证明完成；proof 不能打开裸模型参数，也不能把模型参数的随机点评值作为明文字段输出。

### 1.6 公共拓扑多项式与辅助公开多项式

定义：

1. 源索引多项式：$P_{src}(X)=\sum_{k=0}^{E-1}src(k)L_k^{(edge)}(X)$

2. 目标索引多项式：$P_{dst}(X)=\sum_{k=0}^{E-1}dst(k)L_k^{(edge)}(X)$

3. 组起点选择器多项式：$P_{Q_{new}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{new}^{edge}[k]L_k^{(edge)}(X)$

4. 组末尾选择器多项式：$P_{Q_{end}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{end}^{edge}[k]L_k^{(edge)}(X)$

5. 边域有效区选择器多项式：$P_{Q_{edge}^{valid}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{edge}^{valid}[k]L_k^{(edge)}(X)$

6. 节点域有效区选择器多项式：$P_{Q_N}(X)=\sum_{i=0}^{n_N-1}Q_N[i]L_i^{(N)}(X)$

7. 对每个隐藏层 $\ell=1,\ldots,L-1$，输入共享维有效区选择器多项式：$P_{Q_{in}^{valid,(\ell)}}(X)=\sum_{m=0}^{n_{in}-1}Q_{in}^{valid,(\ell)}[m]L_m^{(in)}(X)$
   
8. 对每个隐藏层 $\ell=1,\ldots,L-1$，隐藏层共享维有效区选择器多项式：$P_{Q_{d_h}^{valid,(\ell)}}(X)=\sum_{j=0}^{n_{d_h}-1}Q_{d_h}^{valid,(\ell)}[j]L_j^{(d_h)}(X)$
   
9. 对每个隐藏层 $\ell=1,\ldots,L-1$，拼接共享维有效区选择器多项式：$P_{Q_{cat}^{valid,(\ell)}}(X)=\sum_{m=0}^{n_{cat}-1}Q_{cat}^{valid,(\ell)}[m]L_m^{(cat)}(X)$
   
10. 输出层输入共享维有效区选择器多项式：$P_{Q_{cat}^{valid,(out)}}(X)=\sum_{m=0}^{n_{cat}-1}Q_{cat}^{valid,(out)}[m]L_m^{(cat)}(X)$
    
11. 输出层类别共享维有效区选择器多项式：$P_{Q_C^{valid}}(X)=\sum_{c=0}^{n_C-1}Q_C^{valid}[c]L_c^{(C)}(X)$

12. 节点域枚举多项式：$P_{Idx_N}(X)=\sum_{i=0}^{n_N-1}Idx_N[i]L_i^{(N)}(X)$

13. 输入共享维枚举多项式：$P_{Idx_{in}}(X)=\sum_{m=0}^{n_{in}-1}Idx_{in}[m]L_m^{(in)}(X)$

14. 隐藏层共享维枚举多项式：$P_{Idx_{d_h}}(X)=\sum_{j=0}^{n_{d_h}-1}Idx_{d_h}[j]L_j^{(d_h)}(X)$

15. 拼接共享维枚举多项式：$P_{Idx_{cat}}(X)=\sum_{m=0}^{n_{cat}-1}Idx_{cat}[m]L_m^{(cat)}(X)$

16. 输出类别枚举多项式：$P_{Idx_C}(X)=\sum_{c=0}^{n_C-1}Idx_C[c]L_c^{(C)}(X)$

17. 特征检索表端行列辅助多项式：

	$P_{Row_{feat}^{tbl}}(X)=\sum_{u=0}^{n_{FH}-1}Row_{feat}^{tbl}[u]L_u^{(FH)}(X)$

	$P_{Col_{feat}^{tbl}}(X)=\sum_{u=0}^{n_{FH}-1}Col_{feat}^{tbl}[u]L_u^{(FH)}(X)$

18. 特征检索查询端局部行列辅助多项式：

	$P_{Row_{feat}^{qry}}(X)=\sum_{q=0}^{n_{FH}-1}Row_{feat}^{qry}[q]L_q^{(FH)}(X)$

	$P_{Col_{feat}^{qry}}(X)=\sum_{q=0}^{n_{FH}-1}Col_{feat}^{qry}[q]L_q^{(FH)}(X)$

19. 特征检索查询端绝对节点编号辅助多项式：

	对每个 $q=i\cdot d_{in}+j$，定义$I_{feat}^{qry}[q]=I_i$，然后插值得到：$P_{I_{feat}^{qry}}(X)=\sum_{q=0}^{n_{FH}-1}I_{feat}^{qry}[q]L_q^{(FH)}(X)$ 

20. batch 节点图编号辅助多项式：

	定义：$P_{node\_gid}(X)=\sum_{i=0}^{n_N-1}node\_gid(i)L_i^{(N)}(X)$

21. batch 边图编号辅助多项式：

	定义：$P_{edge\_gid}(X)=\sum_{k=0}^{n_{edge}-1}edge\_gid(k)L_k^{(edge)}(X)$

22. 图块起点 / 末尾选择器多项式：

	$P_{Q_{graph\_new}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{graph\_new}^{edge}[k]L_k^{(edge)}(X)$

	$P_{Q_{graph\_end}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{graph\_end}^{edge}[k]L_k^{(edge)}(X)$

23. batch-aware 组起点 / 组末尾选择器多项式：

	$P_{Q_{grp\_new}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{grp\_new}^{edge}[k]L_k^{(edge)}(X)$

	$P_{Q_{grp\_end}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{grp\_end}^{edge}[k]L_k^{(edge)}(X)$

这些对象全部由验证者根据公共输入本地重建，不属于动态承诺对象；其中与$Q_{in}^{valid,(\ell)},Q_{d_h}^{valid,(\ell)},Q_{cat}^{valid,(\ell)},Q_{cat}^{valid,(out)}$ 对应的选择器族，必须按层指标逐一重建，不得以“共享最大工作域”为由把它们压缩成一套无层标选择器。

$P_{node\_gid},P_{edge\_gid},P_{Q_{graph\_new}^{edge}},P_{Q_{graph\_end}^{edge}},P_{Q_{grp\_new}^{edge}},P_{Q_{grp\_end}^{edge}}$ 一律属于验证者必须本地重建并吸入 transcript 的公共拓扑对象。若 $G_{batch}=1$，则它们的取值按单图公式唯一确定；实现不得因为单图而省略这些对象。

在上述公共拓扑对象中，$P_{Q_{new}^{edge}},P_{Q_{end}^{edge}}$ 只允许作为图内基础辅助对象存在；所有正式状态机、商多项式恒等式以及验证者检查都必须只消费 $P_{Q_{grp\_new}^{edge}},P_{Q_{grp\_end}^{edge}}$，不得直接消费旧的单图选择器


## 2. 见证生成与承诺

### 2.0 总体顺序与实现约束

按“前向计算 → 挑战与轨迹列生成 → 多项式编码 → 提交承诺”的顺序执行。 工程实现必须严格分为三阶段：

1. 前向阶段：只计算神经网络明文对象；
2. 挑战与轨迹阶段：按固定 Fiat–Shamir 顺序生成挑战，并据此构造表列、查询列、重数列、累加器列、状态机列；
3. 承诺与证明阶段：集中进行 FFT / MSM，生成全部承诺与全部开放证明。

禁止把前向计算与承诺生成耦合在一起。

#### 2.0.1 安全承诺、掩蔽与 opening 统一规则

凡记为 $[P]$ 的动态承诺，其实际含义为 $[P^{zk}]$，其中：$$P^{zk}(X)=P(X)+Z_{\mathcal D}(X)R_P(X)$$。

若 $P$ 是公开多项式，或 `privacy_policy` 明确标记为 public-output polynomial，则 $R_P=0$。除此之外，所有模型相关、激活相关、注意力相关、路由相关、lookup query 相关、PSQ 相关、折叠向量相关和状态机相关多项式都视为 private witness polynomial，必须采样独立 blinding 多项式。

证明者必须先完成前向阶段，随后生成所有基础 trace 值，再依据 `privacy_policy` 构造 $P^{zk}$ 并提交承诺。任何实现不得为了性能把裸 $P$ 的外点评值、折叠向量、完整中间矩阵或模型参数评值写入证明对象。

所有 `Eval_ext`、`Eval_dom` 和 `Open_dom` 均以 $P^{zk}$ 为 opening 对象。若某条代数关系在语义上需要裸 $P$，必须通过 quotient 中的 blinding correction 或 $\Pi_{bind}$ 的零知识子证明处理，不能通过暴露裸评值处理。


### 2.1 数据加载与特征检索

#### 2.1.1 原始特征

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 与每个输入维索引 $j\in\{0,1,\ldots,d_{in}-1\}$，定义：$H_{i,j}=T_H[I_i,j]$

#### 2.1.2 特征检索表端与查询端

生成挑战值：$\eta_{feat},\beta_{feat}$ 

> - $\eta_{feat}$：把多个字段压成一个域元素
> - $\beta_{feat}$：给后面的 LogUp/累加器公式使用，避免分母冲突。

对每个全局表端索引定义：$u=v\cdot d_{in}+j \quad 0\le v\le N_{total}-1 \quad 0\le j\le d_{in}-1$

定义表端离散列：$Table^{feat}[u]=v+\eta_{feat}j+\eta_{feat}^2T_H[v,j]$ 

对每个查询索引：$q=i\cdot d_{in}+j \quad 0\le i\le N-1 \quad 0\le j\le d_{in}-1$

定义查询端离散列：$Query^{feat}[q]=I_i+\eta_{feat}j+\eta_{feat}^2H_{i,j}$

> 把当前证明里实际声称取到的条目$(I_i, j, H_{i,j})$也压成一个域元素，表示“这次实例实际访问了哪些特征项”
>

#### 2.1.3 重数列

对每个全局条目索引 $u=v\cdot d_{in}+j$，定义：$m_{feat}[u]=\#\{i\in\{0,1,\ldots,N-1\}\mid I_i=v\}$

#### 2.1.4 表端与查询端有效区选择器

定义表端有效区选择器：$Q_{tbl}^{feat}[t]= \begin{cases} 1 &0\le t\le N_{total}d_{in}-1 \\ 0 &N_{total}d_{in}\le t\le n_{FH}-1 \end{cases}$

定义查询端有效区选择器：$Q_{qry}^{feat}[t]= \begin{cases} 1 & 0\le t\le Nd_{in}-1 \\ 0 &Nd_{in}\le t\le n_{FH}-1 \end{cases}$ 

#### 2.1.5 特征检索累加器

定义累加器离散列：$R_{feat}[0]=0$

对所有 $t\in\{0,1,\ldots,n_{FH}-2\}$，定义：$R_{feat}[t+1] = R_{feat}[t] + Q_{tbl}^{feat}[t]\cdot\frac{m_{feat}[t]}{Table^{feat}[t]+\beta_{feat}} - Q_{qry}^{feat}[t]\cdot\frac{1}{Query^{feat}[t]+\beta_{feat}}$

填充区保持常值。如果查询端和表端完全匹配，而且重数也对，那么一路累加下来，最终会抵消为 0。

#### 2.1.6 多项式编码

定义：

1. 节点绝对编号多项式：$P_I(X)=\sum_{i=0}^{N-1}I_iL_i^{(N)}(X)$

2. 原始特征系数多项式：$P_H(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{d_{in}-1}H_{i,j}X^{i\cdot d_{in}+j}$

3. 表端多项式：$P_{Table^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Table^{feat}[t]L_t^{(FH)}(X)$

4. 查询端多项式：$P_{Query^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Query^{feat}[t]L_t^{(FH)}(X)$

5. 重数多项式：$P_{m_{feat}}(X)=\sum_{t=0}^{n_{FH}-1}m_{feat}[t]L_t^{(FH)}(X)$

6. 有效区选择器多项式：

	$P_{Q_{tbl}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Q_{tbl}^{feat}[t]L_t^{(FH)}(X)$

	$P_{Q_{qry}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Q_{qry}^{feat}[t]L_t^{(FH)}(X)$

7. 累加器多项式：$P_{R_{feat}}(X)=\sum_{t=0}^{n_{FH}-1}R_{feat}[t]L_t^{(FH)}(X)$ 


#### 2.1.7 承诺

证明者提交承诺：$[P_H],\ [P_{Table^{feat}}],\ [P_{Query^{feat}}],\ [P_{m_{feat}}],\ [P_{Q_{tbl}^{feat}}],\ [P_{Q_{qry}^{feat}}],\ [P_{R_{feat}}]$

其中$[P_I]$不作为动态承诺对象由证明者提交。验证者根据公共输入本地重建$[P_I]$，并在第 3.2.1 节规定的位置将其吸入 transcript。

在主实验中，`input_privacy = public_graph_public_features`，因此 $P_H$、特征检索表端列、查询端列、重数列、有效区选择器列和特征检索累加器列都按公开输入策略处理。若这些对象进入动态承诺块，其掩蔽多项式固定为 0，不作为私有输入见证处理，也不得把这一公开特征检索路径误实现成隐藏输入证明。若未来需要隐藏节点特征或图结构，必须另行定义新的输入隐私策略和证明范围，不能复用当前主实验口径。

### 2.2 单个隐藏层模板中第 $r$ 个注意力头的完整见证生成

以下步骤对应某个固定隐藏层 $\ell\in\{1,2,\ldots,L-1\}$ 的单层模板，在该模板内，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$ 的隐藏层注意力头分别定义：

数学定义中的头顺序固定为 $r=0,1,\ldots,K_{hid}-1$；实现是否并行不属于协议语义，但任何实现都必须产出与这一固定头顺序一致的承诺对象、外点评值与子证明顺序。对多层 GAT，证明者与验证者都必须先对隐藏层 $\ell=1$ 实例化第 2.2 节与第 2.3 节，再把 $H_{cat}^{(1)}$ 作为下一隐藏层输入，依次实例化到 $\ell=L-1$；最后再进入第 2.4 节的最终输出层。

#### 2.2.1 投影

对每个节点 $i$ 与每个隐藏维索引 $j\in\{0,1,\ldots,d_h-1\}$，定义：$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}$ 

定义输出系数多项式：$P_{H'^{(r)}}(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}X^{i\cdot d_h+j}$ 

> $X$的指数就是矩阵元素在内存中的顺序编号

对每个共享维索引 $m\in\{0,1,\ldots,d_{in}-1\}$，定义：$A_m^{proj(r)}(X)=\sum_{i=0}^{N-1}H_{i,m}X^{i\cdot d_h} \quad B_m^{proj(r)}(X)=\sum_{j=0}^{d_h-1}W_{m,j}^{(r)}X^j$ 

> 对特征矩阵按列提取并整理出每个维度（每列）的系数多项式；对权重矩阵按行提取，同理输出每个维度的系数多项式

于是：$P_{H'^{(r)}}(X)=\sum_{m=0}^{d_{in}-1}A_m^{proj(r)}(X)B_m^{proj(r)}(X)$

生成挑战：$y_{proj}^{(r)}=H_{FS}(\text{transcript},[P_H],[P_{H'^{(r)}}],[V_{W^{(r)}}])$

定义折叠向量：（计算所有多项式在挑战点处的值）

$a_m^{proj(r)}=A_m^{proj(r)}(y_{proj}^{(r)})=\sum_{i=0}^{N-1}H_{i,m}(y_{proj}^{(r)})^{i\cdot d_h}$ 

$b_m^{proj(r)}=B_m^{proj(r)}(y_{proj}^{(r)})=\sum_{j=0}^{d_h-1}W_{m,j}^{(r)}(y_{proj}^{(r)})^j$

定义外点评值：$\mu_{proj}^{(r)}=\sum_{m=0}^{d_{in}-1}a_m^{proj(r)}b_m^{proj(r)}$

要求：$P_{H'^{(r)}}(y_{proj}^{(r)})=\mu_{proj}^{(r)}$ 

定义共享维累加器：$Acc_{proj}^{(r)}[0]=0$

$Acc_{proj}^{(r)}[m+1]=Acc_{proj}^{(r)}[m]+a_m^{proj(r)}b_m^{proj(r)} \quad 0\le m\le d_{in}-1$ 

对 $m\ge d_{in}$ 的填充区，统一规定：$a_m^{proj(r)}=0 \quad b_m^{proj(r)}=0 \quad Acc_{proj}^{(r)}[m+1]=Acc_{proj}^{(r)}[m]$

定义多项式：

$P_{a^{proj(r)}}(X)=\sum_{m=0}^{n_{in}-1}a_m^{proj(r)}L_m^{(in)}(X)$

$P_{b^{proj(r)}}(X)=\sum_{m=0}^{n_{in}-1}b_m^{proj(r)}L_m^{(in)}(X)$

$P_{Acc^{proj(r)}}(X)=\sum_{m=0}^{n_{in}-1}Acc_{proj}^{(r)}[m]L_m^{(in)}(X)$

提交承诺：$[P_{H'^{(r)}}],\ [P_{a^{proj(r)}}],\ [P_{b^{proj(r)}}],\ [P_{Acc^{proj(r)}}]$

#### 2.2.2 源注意力绑定

对每个节点 $i$，定义：$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)}$ 

把 $E_{src}^{(r)}$ 填充到长度 $n_N$，用拉格朗日把数组插值成多项式：$P_{E_{src}^{(r)}}(X)=\sum_{i=0}^{n_N-1}E_{src}^{(r)}[i]L_i^{(N)}(X)$ 

生成挑战：$y_{src}^{(r)}=H_{FS}(\text{transcript},[P_{H'^{(r)}}],[P_{E_{src}^{(r)}}],[V_{a_{src}^{(r)}}])$ 

定义折叠向量：$a_j^{src(r)}=\sum_{i=0}^{N-1}H_{i,j}'^{(r)}L_i^{(N)}(y_{src}^{(r)}) \quad b_j^{src(r)}=a_{src,j}^{(r)}$ 

> $a_j^{src(r)}$：固定一个列索引$j$，从矩阵$H'$中取出第 $j$ 列的所有元素，每个元素乘以对应的拉格朗日基在随机点处的值$ L_i^{(N)}(y_{src})$。然后对所有节点$i$求和，得到一个数。
>
> $b_j^{src(r)}$：直接等于注意力向量$a_{src}$的第$ j $个分量，原封不动，不做任何压缩。
>

定义外点评值：$\mu_{src}^{(r)}=\sum_{j=0}^{d_h-1}a_j^{src(r)}b_j^{src(r)}$

要求：$P_{E_{src}^{(r)}}(y_{src}^{(r)})=\mu_{src}^{(r)}$

> 左边：提交的“源注意力结果列”在随机点的值
>
> 右边：由投影矩阵$H'^{(r)}$和参数向量$a^{src(r)}$重新	折叠计算出来的值

定义共享维累加器：$Acc_{src}^{(r)}[0]=0$

$Acc_{src}^{(r)}[j+1]=Acc_{src}^{(r)}[j]+a_j^{src(r)}b_j^{src(r)} \quad 0\le j\le d_h-1$

对填充区统一规定：$a_j^{src(r)}=0 \quad b_j^{src(r)}=0 \quad Acc_{src}^{(r)}[j+1]=Acc_{src}^{(r)}[j]$ 

定义：$P_{a^{src(r)}}(X) \quad P_{b^{src(r)}}(X) \quad P_{Acc^{src(r)}}(X)$，分别为其在 $\mathbb H_{d_h}$ 上插值多项式。

提交承诺：$[P_{E_{src}^{(r)}}],\ [P_{a^{src(r)}}],\ [P_{b^{src(r)}}],\ [P_{Acc^{src(r)}}]$

#### 2.2.3 目标注意力绑定

对每个节点 $i$，定义：$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}$

把 $E_{dst}^{(r)}$ 填充到长度 $n_N$，用拉格朗日把数组插值成多项式：$P_{E_{dst}^{(r)}}(X)=\sum_{i=0}^{n_N-1}E_{dst}^{(r)}[i]L_i^{(N)}(X)$

生成挑战：$y_{dst}^{(r)}=H_{FS}(\text{transcript},[P_{H'^{(r)}}],[P_{E_{dst}^{(r)}}],[V_{a_{dst}^{(r)}}])$

定义折叠向量：$a_j^{dst(r)}=\sum_{i=0}^{N-1}H_{i,j}'^{(r)}L_i^{(N)}(y_{dst}^{(r)}) \quad b_j^{dst(r)}=a_{dst,j}^{(r)}$

定义外点评值：$\mu_{dst}^{(r)}=\sum_{j=0}^{d_h-1}a_j^{dst(r)}b_j^{dst(r)}$

要求：$P_{E_{dst}^{(r)}}(y_{dst}^{(r)})=\mu_{dst}^{(r)}$

定义共享维累加器：$Acc_{dst}^{(r)}[0]=0$

$Acc_{dst}^{(r)}[j+1]=Acc_{dst}^{(r)}[j]+a_j^{dst(r)}b_j^{dst(r)} \quad 0\le j\le d_h-1$

定义：$P_{a^{dst(r)}}(X) \quad P_{b^{dst(r)}}(X) \quad P_{Acc^{dst(r)}}(X)$，分别为其在 $\mathbb H_{d_h}$ 上插值多项式。

提交承诺：$[P_{E_{dst}^{(r)}}],\ [P_{a^{dst(r)}}],\ [P_{b^{dst(r)}}],\ [P_{Acc^{dst(r)}}]$

#### 2.2.4 压缩特征绑定

生成压缩挑战：$\xi^{(r)}=H_{FS}(\text{transcript},[P_{H'^{(r)}}])$ 

对每个节点 $i$，定义：$H_i^{\star(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}(\xi^{(r)})^j$ 

> 原来节点$i$在第$r$个头上有一个$d_h$维向量，现在用$\xi^{(r)}$的幂作为权重，把这$d_h$个分量线性组合成一个数

插值成：$P_{H^{\star(r)}}(X)=\sum_{i=0}^{n_N-1}H^{\star(r)}[i]L_i^{(N)}(X)$ 

生成挑战：$y_{\star}^{(r)}=H_{FS}(\text{transcript},[P_{H'^{(r)}}],[P_{H^{\star(r)}}])$

定义折叠向量：$a_j^{\star(r)}=\sum_{i=0}^{N-1}H_{i,j}'^{(r)}L_i^{(N)}(y_{\star}^{(r)}) \quad b_j^{\star(r)}=(\xi^{(r)})^j$

定义外点评值：$\mu_{\star}^{(r)}=\sum_{j=0}^{d_h-1}a_j^{\star(r)}b_j^{\star(r)}$

要求：$P_{H^{\star(r)}}(y_{\star}^{(r)})=\mu_{\star}^{(r)}$ 

定义共享维累加器：$Acc_{\star}^{(r)}[0]=0$

$Acc_{\star}^{(r)}[j+1]=Acc_{\star}^{(r)}[j]+a_j^{\star(r)}b_j^{\star(r)}$

定义：$P_{a^{\star(r)}}(X) \quad P_{b^{\star(r)}}(X) \quad P_{Acc^{\star(r)}}(X)$，为其在 $\mathbb H_{d_h}$ 上插值多项式。

提交承诺：$[P_{H^{\star(r)}}],\ [P_{a^{\star(r)}}],\ [P_{b^{\star(r)}}],\ [P_{Acc^{\star(r)}}]$

#### 2.2.5 源路由

对每个边索引 $k$，定义：$E_{src,k}^{edge(r)}=E_{src,src(k)}^{(r)} \quad H_{src,k}^{\star,edge(r)}=H_{src(k)}^{\star(r)}$ 

> 这条边的源注意力广播值，等于它源节点的源注意力
>
> 这条边的源压缩特征广播值，等于它源节点的压缩特征
>
> 也就是把节点值按 `src(k)` 路由到边上

生成挑战：$\eta_{src}^{(r)},\beta_{src}^{(r)}$

对每个节点 $i$，定义表端：$Table^{src(r)}[i] = i+\eta_{src}^{(r)}E_{src,i}^{(r)}+(\eta_{src}^{(r)})^2H_i^{\star(r)}$

定义节点重数：$m_{src}^{(r)}[i]=\#\{k\mid src(k)=i\}$

对每个边索引 $k$，定义查询端：$Query^{src(r)}[k] = src(k)+\eta_{src}^{(r)}E_{src,k}^{edge(r)}+(\eta_{src}^{(r)})^2H_{src,k}^{\star,edge(r)}$

定义公开总和：$S_{src}^{(r)} = \sum_{i=0}^{N-1}\frac{m_{src}^{(r)}[i]}{Table^{src(r)}[i]+\beta_{src}^{(r)}} = \sum_{k=0}^{E-1}\frac{1}{Query^{src(r)}[k]+\beta_{src}^{(r)}}$

> 左边：表端每条记录按“应该出现的次数”计入
>
> 右边：查询端每条边实际出现一次计入

定义节点端累加器：$R_{src}^{node(r)}[0]=0$

$R_{src}^{node(r)}[i+1] = R_{src}^{node(r)}[i] + Q_N[i]\cdot\frac{m_{src}^{(r)}[i]}{Table^{src(r)}[i]+\beta_{src}^{(r)}} \quad 0\le i\le n_N-2$ 

定义边端累加器：$R_{src}^{edge(r)}[0]=0$

$R_{src}^{edge(r)}[k+1] = R_{src}^{edge(r)}[k] + Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{src(r)}[k]+\beta_{src}^{(r)}} \quad 0\le k\le n_{edge}-2$

插值得到：$P_{Table^{src(r)}},\ P_{Query^{src(r)}},\ P_{m_{src}^{(r)}},\ P_{R_{src}^{node(r)}},\ P_{R_{src}^{edge(r)}}$

提交承诺：$[P_{Table^{src(r)}}],\ [P_{Query^{src(r)}}],\ [P_{m_{src}^{(r)}}],\ [P_{R_{src}^{node(r)}}],\ [P_{R_{src}^{edge(r)}}]$

#### 2.2.6 LeakyReLU、最大值、唯一性与范围检查

对每个边索引 $k$，定义：$S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)} \quad Z_k^{(r)}=LReLU(S_k^{(r)})$

生成挑战：$\eta_L^{(r)},\beta_L^{(r)}$

定义 LeakyReLU 表端：$Table^{L(r)}[t]=T_{LReLU}[t,0]+\eta_L^{(r)}T_{LReLU}[t,1]$ 

> LeakyReLU 静态表里每一行是：（$T_{LReLU}[t,0]$，$T_{LReLU}[t,1]$），表示（输入值，对应输出值）

查询端：$Query^{L(r)}[k]=S_k^{(r)}+\eta_L^{(r)}Z_k^{(r)}$ 

定义表端有效区选择器：$Q_{tbl}^{L(r)}[t]= \begin{cases} 1 &0\le t\le |T_{LReLU}|-1 \\ 0 &|T_{LReLU}|\le t\le n_{edge}-1 \end{cases}$

查询端有效区选择器：$Q_{qry}^{L(r)}[k]=Q_{edge}^{valid}[k]$

对每个表端索引 $t$，定义重数：$m_L^{(r)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid (S_k^{(r)},Z_k^{(r)})=(T_{LReLU}[t,0],T_{LReLU}[t,1])\}$

定义累加器：$R_L^{(r)}[0]=0$

$R_L^{(r)}[k+1] = R_L^{(r)}[k] + Q_{tbl}^{L(r)}[k]\cdot\frac{m_L^{(r)}[k]}{Table^{L(r)}[k]+\beta_L^{(r)}} - Q_{qry}^{L(r)}[k]\cdot\frac{1}{Query^{L(r)}[k]+\beta_L^{(r)}} \quad 0\le k\le n_{edge}-2$，对填充区保持常值。

插值得到：$P_{Table^{L(r)}},\ P_{Query^{L(r)}},\ P_{m_L^{(r)}},\ P_{Q_{tbl}^{L(r)}},\ P_{Q_{qry}^{L(r)}},\ P_{R_L^{(r)}}$

定义组最大值：$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}$

定义最大值广播与非负差分：$M_k^{edge(r)}=M_{dst(k)}^{(r)} \quad \Delta_k^{+(r)}=M_k^{edge(r)}-Z_k^{(r)}$ 

> 如果$M$真的是组最大值，那对同组所有边都应该有：$\Delta_k^{+(r)} ≥ 0$ 

引入指示变量：$s_{max}^{(r)}[k]\in\{0,1\}$

要求：

1. 二值性：$s_{max}^{(r)}[k]\big(s_{max}^{(r)}[k]-1\big)=0$

2. 被选中的位置必须零差分：$s_{max}^{(r)}[k]\cdot \Delta_k^{+(r)}=0$


定义组内计数状态机：$C_{max}^{(r)}[0]=Q_{edge}^{valid}[0]\cdot s_{max}^{(r)}[0]$ 

对 $0\le k\le n_{edge}-2$，定义：$C_{max}^{(r)}[k+1] = (1-Q_{edge}^{valid}[k+1])C_{max}^{(r)}[k] + Q_{edge}^{valid}[k+1] \Big( Q_{grp\_new}^{edge}[k+1]s_{max}^{(r)}[k+1] + (1-Q_{grp\_new}^{edge}[k+1])\big(C_{max}^{(r)}[k]+s_{max}^{(r)}[k+1]\big) \Big)$ 

对每个边索引 $k$，要求组末约束：$Q_{grp\_end}^{edge}[k]\cdot\big(C_{max}^{(r)}[k]-1\big)=0$

生成范围检查挑战：$\beta_R^{(r)}$

定义范围表端：$Table^{R(r)}[t]=t$

范围查询端：$Query^{R(r)}[k]=\Delta_k^{+(r)}$

定义表端有效区选择器：$Q_{tbl}^{R(r)}[t]= \begin{cases} 1 &0\le t\le 2^B-1 \\ 0 &2^B\le t\le n_{edge}-1 \end{cases}$

查询端有效区选择器：$Q_{qry}^{R(r)}[k]=Q_{edge}^{valid}[k]$ 

对每个表端索引 $t$，定义重数：$m_R^{(r)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid \Delta_k^{+(r)}=t\}$

定义累加器：$R_R^{(r)}[0]=0$

$R_R^{(r)}[k+1] = R_R^{(r)}[k] + Q_{tbl}^{R(r)}[k]\cdot\frac{m_R^{(r)}[k]}{Table^{R(r)}[k]+\beta_R^{(r)}} - Q_{qry}^{R(r)}[k]\cdot\frac{1}{Query^{R(r)}[k]+\beta_R^{(r)}} \quad 0\le k\le n_{edge}-2$，对填充区保持常值。

插值得到：$P_{Table^{R(r)}},\ P_{Query^{R(r)}},\ P_{m_R^{(r)}},\ P_{Q_{tbl}^{R(r)}},\ P_{Q_{qry}^{R(r)}},\ P_{R_R^{(r)}}$

把$S^{(r)},\ Z^{(r)},\ M^{(r)},\ M^{edge(r)},\ \Delta^{+(r)},\ s_{max}^{(r)},\ C_{max}^{(r)}$，分别插值成：$P_{S^{(r)}},\ P_{Z^{(r)}},\ P_{M^{(r)}},\ P_{M^{edge(r)}},\ P_{\Delta^{+(r)}},\ P_{s_{max}^{(r)}},\ P_{C_{max}^{(r)}}$

提交承诺：

$[P_{S^{(r)}}],\ [P_{Z^{(r)}}],\ [P_{M^{(r)}}],\ [P_{M^{edge(r)}}],\ [P_{\Delta^{+(r)}}],\ [P_{s_{max}^{(r)}}],\ [P_{C_{max}^{(r)}}],\ [P_{Table^{L(r)}}],\ [P_{Query^{L(r)}}],\\ [P_{m_L^{(r)}}],\ [P_{Q_{tbl}^{L(r)}}],\ [P_{Q_{qry}^{L(r)}}],\ [P_{R_L^{(r)}}],\ [P_{Table^{R(r)}}],\ [P_{Query^{R(r)}}],\ [P_{m_R^{(r)}}],\ [P_{Q_{tbl}^{R(r)}}],\ [P_{Q_{qry}^{R(r)}}],\ [P_{R_R^{(r)}}]$ 

#### 2.2.7 指数、分母、逆元与归一化权重

对每个边索引 $k$，定义：$U_k^{(r)}=ExpMap(\Delta_k^{+(r)})$ 

> 把上一节得到的非负差分$ \Delta^+ $送进指数查表，得到 softmax 分子项$U$ 

生成挑战：$\eta_{exp}^{(r)},\beta_{exp}^{(r)}$ 

定义指数表端：$Table^{exp(r)}[t]=T_{exp}[t,0]+\eta_{exp}^{(r)}T_{exp}[t,1]$

查询端：$Query^{exp(r)}[k]=\Delta_k^{+(r)}+\eta_{exp}^{(r)}U_k^{(r)}$

定义表端有效区选择器：$Q_{tbl}^{exp(r)}[t]= \begin{cases} 1 &0\le t\le |T_{exp}|-1 \\ 0 &|T_{exp}|\le t\le n_{edge}-1 \end{cases}$

查询端有效区选择器：$Q_{qry}^{exp(r)}[k]=Q_{edge}^{valid}[k]$

对每个表端索引 $t$，定义重数：$m_{exp}^{(r)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid (\Delta_k^{+(r)},U_k^{(r)})=(T_{exp}[t,0],T_{exp}[t,1])\}$

定义累加器：$R_{exp}^{(r)}[0]=0$

$R_{exp}^{(r)}[k+1] = R_{exp}^{(r)}[k] + Q_{tbl}^{exp(r)}[k]\cdot\frac{m_{exp}^{(r)}[k]}{Table^{exp(r)}[k]+\beta_{exp}^{(r)}} - Q_{qry}^{exp(r)}[k]\cdot\frac{1}{Query^{exp(r)}[k]+\beta_{exp}^{(r)}} \quad 0\le k\le n_{edge}-2$，对填充区保持常值。

插值得到：$P_{Table^{exp(r)}},\ P_{Query^{exp(r)}},\ P_{m_{exp}^{(r)}},\ P_{Q_{tbl}^{exp(r)}},\ P_{Q_{qry}^{exp(r)}},\ P_{R_{exp}^{(r)}}$

对每个节点 $i$，定义：$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)} \quad inv_i^{(r)}=(Sum_i^{(r)})^{-1}$

> 把同一个$dst=i$的那组边的所有$U_k$加起来，得到这个节点 softmax 的分母

对每个边索引 $k$，定义广播：$Sum_k^{edge(r)}=Sum_{dst(k)}^{(r)} \quad inv_k^{edge(r)}=inv_{dst(k)}^{(r)}$

定义归一化权重：$\alpha_k^{(r)}=U_k^{(r)}\cdot inv_k^{edge(r)}$

把：$U^{(r)},\ Sum^{(r)},\ Sum^{edge(r)},\ inv^{(r)},\ inv^{edge(r)},\alpha^{(r)}$ 分别插值成：$P_{U^{(r)}},\ P_{Sum^{(r)}},\ P_{Sum^{edge(r)}},\ P_{inv^{(r)}},\ P_{inv^{edge(r)}},\ P_{\alpha^{(r)}}$

提交承诺：$[P_{U^{(r)}}],\ [P_{Sum^{(r)}}],\ [P_{Sum^{edge(r)}}],\ [P_{inv^{(r)}}],\ [P_{inv^{edge(r)}}],\ [P_{\alpha^{(r)}}],\ [P_{Table^{exp(r)}}],\ [P_{Query^{exp(r)}}],\ [P_{m_{exp}^{(r)}}],\ [P_{Q_{tbl}^{exp(r)}}],\ [P_{Q_{qry}^{exp(r)}}],\ [P_{R_{exp}^{(r)}}]$

#### 2.2.8 聚合前隐藏矩阵、聚合前压缩特征与 PSQ

对每个节点 $i$ 与每个隐藏维索引 $j$，定义：$H_{agg,pre,i,j}^{(r)} = \sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}$

定义压缩特征：$H_{agg,pre,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,pre,i,j}^{(r)}(\xi^{(r)})^j$

定义边级压缩加权特征：$\widehat v_{pre,k}^{\star(r)}=\alpha_k^{(r)}H_{src,k}^{\star,edge(r)}$ 

定义广播：$H_{agg,pre,k}^{\star,edge(r)}=H_{agg,pre,dst(k)}^{\star(r)}$ 

在决定 $U^{(r)},\ Sum^{(r)},\ Sum^{edge(r)},\ \widehat v_{pre}^{\star(r)},\ H_{agg,pre}^{\star(r)}$ 这些基础值的相关承诺对象全部固定之后，生成挑战$\lambda_{psq}^{(r)}$。

定义：$w_{psq}^{(r)}[k]=U_k^{(r)}+\lambda_{psq}^{(r)}\widehat v_{pre,k}^{\star(r)}$

定义节点端目标值：$T_{psq}^{(r)}[i]=Sum_i^{(r)}+\lambda_{psq}^{(r)}H_{agg,pre,i}^{\star(r)}$

定义边端广播目标值：$T_{psq}^{edge(r)}[k]=Sum_k^{edge(r)}+\lambda_{psq}^{(r)}H_{agg,pre,k}^{\star,edge(r)}$

定义 PSQ 状态机：$PSQ^{(r)}[0]=Q_{edge}^{valid}[0]\cdot w_{psq}^{(r)}[0]$

对 $0\le k\le n_{edge}-2$，定义：$PSQ^{(r)}[k+1] = (1-Q_{edge}^{valid}[k+1])PSQ^{(r)}[k] + Q_{edge}^{valid}[k+1] \Big( Q_{grp\_new}^{edge}[k+1]w_{psq}^{(r)}[k+1] + (1-Q_{grp\_new}^{edge}[k+1])\big(PSQ^{(r)}[k]+w_{psq}^{(r)}[k+1]\big) \Big)$

对每个边索引 $k$，施加组末约束：$Q_{grp\_end}^{edge}[k]\cdot\big(PSQ^{(r)}[k]-T_{psq}^{edge(r)}[k]\big)=0$

组末约束等价地同时强制了：$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)} \quad H_{agg,pre,i}^{\star(r)}=\sum_{\{k\mid dst(k)=i\}}\widehat v_{pre,k}^{\star(r)}$

插值得到：$P_{H_{agg,pre}^{(r)}},\ P_{H_{agg,pre}^{\star(r)}},\ P_{\widehat v_{pre}^{\star(r)}},\ P_{w_{psq}^{(r)}},\ P_{T_{psq}^{(r)}},\ P_{T_{psq}^{edge(r)}},\ P_{PSQ^{(r)}}$

生成聚合前压缩绑定挑战：$y_{agg,pre}^{(r)}=H_{FS}(\text{transcript},[P_{H_{agg,pre}^{(r)}}],[P_{H_{agg,pre}^{\star(r)}}])$

要求：$P_{H_{agg,pre}^{\star(r)}}(y_{agg,pre}^{(r)}) = \sum_{j=0}^{d_h-1} \Big( \sum_{i=0}^{N-1}H_{agg,pre,i,j}^{(r)}L_i^{(N)}(y_{agg,pre}^{(r)}) \Big) (\xi^{(r)})^j$

提交承诺：$[P_{H_{agg,pre}^{(r)}}],\ [P_{H_{agg,pre}^{\star(r)}}],\ [P_{\widehat v_{pre}^{\star(r)}}],\ [P_{w_{psq}^{(r)}}],\ [P_{T_{psq}^{(r)}}],\ [P_{T_{psq}^{edge(r)}}],\ [P_{PSQ^{(r)}}]$

#### 2.2.9 ELU

本节 ELU 查表复用边域$\mathbb H_{edge}$作为工作域；下文“ELU 工作域”均指$\mathbb H_{edge}$。

对每个节点 $i$ 与每个隐藏维索引 $j$，定义：$H_{agg,i,j}^{(r)}=ELU(H_{agg,pre,i,j}^{(r)})$

将 $H_{agg,pre}^{(r)}$ 与 $H_{agg}^{(r)}$ 展平到 ELU 工作域，生成挑战：$\eta_{ELU}^{(r)},\beta_{ELU}^{(r)}$

定义 ELU 表端：$Table^{ELU(r)}[t]=T_{ELU}[t,0]+\eta_{ELU}^{(r)}T_{ELU}[t,1]$

定义 ELU 查询端：$Query^{ELU(r)}[q]=H_{agg,pre}^{(r)}[q]+\eta_{ELU}^{(r)}H_{agg}^{(r)}[q]$

定义表端有效区选择器：$Q_{tbl}^{ELU(r)}[t]= \begin{cases} 1 &0\le t\le |T_{ELU}|-1 \\ 0 &|T_{ELU}|\le t\le n_{edge}-1 \end{cases}$

查询端有效区选择器：$Q_{qry}^{ELU(r)}[q]= \begin{cases} 1 &0\le q\le Nd_h-1 \\ 0 &Nd_h\le q\le n_{edge}-1 \end{cases}$

对每个表端索引 $t$，定义重数：$m_{ELU}^{(r)}[t] = \#\{q\in\{0,1,\ldots,Nd_h-1\}\mid (H_{agg,pre}^{(r)}[q],H_{agg}^{(r)}[q])=(T_{ELU}[t,0],T_{ELU}[t,1])\}$

定义累加器：$R_{ELU}^{(r)}[0]=0$

$R_{ELU}^{(r)}[q+1] = R_{ELU}^{(r)}[q] + Q_{tbl}^{ELU(r)}[q]\cdot\frac{m_{ELU}^{(r)}[q]}{Table^{ELU(r)}[q]+\beta_{ELU}^{(r)}} - Q_{qry}^{ELU(r)}[q]\cdot\frac{1}{Query^{ELU(r)}[q]+\beta_{ELU}^{(r)}}, \qquad 0\le q\le n_{edge}-2$

插值得到：$P_{Table^{ELU(r)}},\ P_{Query^{ELU(r)}},\ P_{m_{ELU}^{(r)}},\ P_{Q_{tbl}^{ELU(r)}},\ P_{Q_{qry}^{ELU(r)}},\ P_{R_{ELU}^{(r)}}$

定义聚合后压缩特征：$H_{agg,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,i,j}^{(r)}(\xi^{(r)})^j$

定义边域广播：$H_{agg,k}^{\star,edge(r)}=H_{agg,dst(k)}^{\star(r)}$

生成聚合后压缩绑定挑战：$y_{agg}^{(r)}=H_{FS}(\text{transcript},[P_{H_{agg}^{(r)}}],[P_{H_{agg}^{\star(r)}}])$

要求：$P_{H_{agg}^{\star(r)}}(y_{agg}^{(r)}) = \sum_{j=0}^{d_h-1} \Big( \sum_{i=0}^{N-1}H_{agg,i,j}^{(r)}L_i^{(N)}(y_{agg}^{(r)}) \Big) (\xi^{(r)})^j$

提交：$[P_{H_{agg}^{(r)}}],\ [P_{H_{agg}^{\star(r)}}],\ [P_{Table^{ELU(r)}}],\ [P_{Query^{ELU(r)}}],\ [P_{m_{ELU}^{(r)}}],\ [P_{R_{ELU}^{(r)}}]$

#### 2.2.10 目标路由的延迟定稿

目标路由需要同时绑定：

- 目标注意力；
- 组最大值；
- 分母；
- 逆元；
- ELU 后压缩聚合特征。

因此在 $P_{E_{dst}^{(r)}},\ P_{M^{(r)}},\ P_{Sum^{(r)}},\ P_{inv^{(r)}},\ P_{H_{agg}^{\star(r)}}$ 及其边域广播对象全部固定之后，统一生成挑战：$\eta_{dst}^{(r)},\beta_{dst}^{(r)}$ 

对每个节点 $i$，定义表端：$Table^{dst(r)}[i] = i +\eta_{dst}^{(r)}E_{dst,i}^{(r)} +(\eta_{dst}^{(r)})^2M_i^{(r)} +(\eta_{dst}^{(r)})^3Sum_i^{(r)} +(\eta_{dst}^{(r)})^4inv_i^{(r)} +(\eta_{dst}^{(r)})^5H_{agg,i}^{\star(r)}$

对每个边索引 $k$，定义查询端：

$Query^{dst(r)}[k] = dst(k) +\eta_{dst}^{(r)}E_{dst,k}^{edge(r)} +(\eta_{dst}^{(r)})^2M_k^{edge(r)} +(\eta_{dst}^{(r)})^3Sum_k^{edge(r)} +(\eta_{dst}^{(r)})^4inv_k^{edge(r)} +(\eta_{dst}^{(r)})^5H_{agg,k}^{\star,edge(r)}$

定义节点重数：$m_{dst}^{(r)}[i]=\#\{k\mid dst(k)=i\}$

定义公开总和：$S_{dst}^{(r)} = \sum_{i=0}^{N-1}\frac{m_{dst}^{(r)}[i]}{Table^{dst(r)}[i]+\beta_{dst}^{(r)}} = \sum_{k=0}^{E-1}\frac{1}{Query^{dst(r)}[k]+\beta_{dst}^{(r)}}$

定义节点端累加器：$R_{dst}^{node(r)}[0]=0$

$R_{dst}^{node(r)}[i+1] = R_{dst}^{node(r)}[i] + Q_N[i]\cdot\frac{m_{dst}^{(r)}[i]}{Table^{dst(r)}[i]+\beta_{dst}^{(r)}}$

定义边端累加器：$R_{dst}^{edge(r)}[0]=0$

$R_{dst}^{edge(r)}[k+1] = R_{dst}^{edge(r)}[k] + Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{dst(r)}[k]+\beta_{dst}^{(r)}}$

插值得到：$P_{Table^{dst(r)}},\ P_{Query^{dst(r)}},\ P_{m_{dst}^{(r)}},\ P_{R_{dst}^{node(r)}},\ P_{R_{dst}^{edge(r)}}$

提交承诺：$[P_{Table^{dst(r)}}],\ [P_{Query^{dst(r)}}],\ [P_{m_{dst}^{(r)}}],\ [P_{R_{dst}^{node(r)}}],\ [P_{R_{dst}^{edge(r)}}]$

### 2.3 隐藏层模板的拼接阶段

本节同样对应某个固定隐藏层 $\ell$ 的模板拼接输出。在多层实例中，该层的拼接结果与其承诺对象严格记为：$H_{cat}^{(\ell)},H_{cat}^{\star(\ell)},P_{H_{cat}}^{(\ell)},P_{H_{cat}^{\star}}^{(\ell)}$。
在本节的数学推导中，局部简写为 $H_{cat},H_{cat}^{\star},P_{H_{cat}},P_{H_{cat}^{\star}}$；但代码实现、serializer 字段名、proof label 与日志字段名都必须保留层指标 $(\ell)$。

对每个节点 $i$、每个隐藏层注意力头 $r$、每个局部维索引 $j$，定义：$H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}$

定义拼接系数多项式：$P_{H_{cat}}(X)=\sum_{i=0}^{N-1}\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}X^{i\cdot d_{cat}+m}$

生成拼接压缩挑战：$\xi_{cat}=H_{FS}(\text{transcript},[P_{H_{cat}}])$

对每个节点 $i$，定义：$H_{cat,i}^{\star}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}\xi_{cat}^m$

插值得到：$P_{H_{cat}^{\star}}(X)=\sum_{i=0}^{n_N-1}H_{cat}^{\star}[i]L_i^{(N)}(X)$

生成拼接绑定挑战：$y_{cat}=H_{FS}(\text{transcript},[P_{H_{cat}}],[P_{H_{cat}^{\star}}])$

要求：$P_{H_{cat}^{\star}}(y_{cat}) = \sum_{m=0}^{d_{cat}-1} \Big( \sum_{i=0}^{N-1}H_{cat,i,m}L_i^{(N)}(y_{cat}) \Big)\xi_{cat}^m$

提交：$[P_{H_{cat}}],\ [P_{H_{cat}^{\star}}]$

### 2.4 输出层的完整见证生成

输出层由 $K_{out}$ 个输出注意力头组成。对每个输出头 $s\in\{0,1,\ldots,K_{out}-1\}$，先按单头 注意力 完成 projection / score / softmax / aggregation / bias-add；全部输出头完成后，再做多头平均，得到最终输出 $Y$。

#### 2.4.1 第 $s$ 个输出头的输出投影

对每个节点 $i$ 与类别索引 $c$，定义：$Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out,s)}$。

定义系数多项式：$P_{Y'^{(s)}}(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}X^{i\cdot C+c}$。

按 CRPC 定义：$A_m^{proj(out,s)}(X)=\sum_{i=0}^{N-1}H_{cat,i,m}X^{i\cdot C},\quad
B_m^{proj(out,s)}(X)=\sum_{c=0}^{C-1}W_{m,c}^{(out,s)}X^c$。

于是：$P_{Y'^{(s)}}(X)=\sum_{m=0}^{d_{cat}-1}A_m^{proj(out,s)}(X)B_m^{proj(out,s)}(X)$。

生成输出投影绑定挑战：$y_{proj}^{(out,s)}=H_{FS}(\text{transcript},[P_{H_{cat}}],[P_{Y'^{(s)}}],[V_{W^{(out,s)}}])$。

定义折叠向量：

$a_m^{proj(out,s)}=A_m^{proj(out,s)}(y_{proj}^{(out,s)})=\sum_{i=0}^{N-1}H_{cat,i,m}(y_{proj}^{(out,s)})^{i\cdot C}$

$b_m^{proj(out,s)}=B_m^{proj(out,s)}(y_{proj}^{(out,s)})=\sum_{c=0}^{C-1}W_{m,c}^{(out,s)}(y_{proj}^{(out,s)})^c$ 

定义外点评值：$\mu_{proj}^{(out,s)}=\sum_{m=0}^{d_{cat}-1}a_m^{proj(out,s)}b_m^{proj(out,s)}$

要求：$P_{Y'^{(s)}}(y_{proj}^{(out,s)})=\mu_{proj}^{(out,s)}$ 

定义共享维累加器：$Acc_{proj}^{(out,s)}[0]=0$

$Acc_{proj}^{(out,s)}[m+1]=Acc_{proj}^{(out,s)}[m]+a_m^{proj(out,s)}b_m^{proj(out,s)}
\quad 0\le m\le d_{cat}-1$ 

对 $m\ge d_{cat}$ 的填充区，统一规定：

$a_m^{proj(out,s)}=0 \quad b_m^{proj(out,s)}=0 \quad Acc_{proj}^{(out,s)}[m+1]=Acc_{proj}^{(out,s)}[m]$ 

定义多项式：

$P_{a^{proj(out,s)}}(X)=\sum_{m=0}^{n_{cat}-1}a_m^{proj(out,s)}L_m^{(cat)}(X)$

$P_{b^{proj(out,s)}}(X)=\sum_{m=0}^{n_{cat}-1}b_m^{proj(out,s)}L_m^{(cat)}(X)$

$P_{Acc^{proj(out,s)}}(X)=\sum_{m=0}^{n_{cat}-1}Acc_{proj}^{(out,s)}[m]L_m^{(cat)}(X)$。

提交承诺：$[P_{Y'^{(s)}}],\ [P_{a^{proj(out,s)}}],\ [P_{b^{proj(out,s)}}],\ [P_{Acc^{proj(out,s)}}]$。

#### 2.4.2 第 $s$ 个输出头的源 / 目标注意力绑定

对每个节点 $i$，定义：

- $E_{src,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{src,c}^{(out,s)}$
- $E_{dst,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{dst,c}^{(out,s)}$

把 $E_{src}^{(out,s)}$ 与 $E_{dst}^{(out,s)}$ 分别填充到长度 $n_N$，插值成：

$P_{E_{src}^{(out,s)}}(X)=\sum_{i=0}^{n_N-1}E_{src}^{(out,s)}[i]L_i^{(N)}(X) \quad P_{E_{dst}^{(out,s)}}(X)=\sum_{i=0}^{n_N-1}E_{dst}^{(out,s)}[i]L_i^{(N)}(X)$ 

生成源注意力绑定挑战：$y_{src}^{(out,s)}=H_{FS}(\text{transcript},[P_{Y'^{(s)}}],[P_{E_{src}^{(out,s)}}],[V_{a_{src}^{(out,s)}}])$

定义折叠向量：$a_c^{src(out,s)}=\sum_{i=0}^{N-1}Y_{i,c}'^{(s)}L_i^{(N)}(y_{src}^{(out,s)}) \quad b_c^{src(out,s)}=a_{src,c}^{(out,s)}$

定义外点评值：$\mu_{src}^{(out,s)}=\sum_{c=0}^{C-1}a_c^{src(out,s)}b_c^{src(out,s)}$

要求：$P_{E_{src}^{(out,s)}}(y_{src}^{(out,s)})=\mu_{src}^{(out,s)}$

定义共享维累加器：$Acc_{src}^{(out,s)}[0]=0$

$Acc_{src}^{(out,s)}[c+1]=Acc_{src}^{(out,s)}[c]+a_c^{src(out,s)}b_c^{src(out,s)}
\qquad 0\le c\le C-1$ 

对填充区统一规定：$a_c^{src(out,s)}=0 \quad b_c^{src(out,s)}=0 \quad Acc_{src}^{(out,s)}[c+1]=Acc_{src}^{(out,s)}[c] \quad (c\ge C)$

定义：$P_{a^{src(out,s)}}(X) \quad P_{b^{src(out,s)}}(X) \quad P_{Acc^{src(out,s)}}(X)$，分别为其在 $\mathbb H_C$ 上的插值多项式。

生成目标注意力绑定挑战：$y_{dst}^{(out,s)}=H_{FS}(\text{transcript},[P_{Y'^{(s)}}],[P_{E_{dst}^{(out,s)}}],[V_{a_{dst}^{(out,s)}}])$

定义折叠向量：$a_c^{dst(out,s)}=\sum_{i=0}^{N-1}Y_{i,c}'^{(s)}L_i^{(N)}(y_{dst}^{(out,s)}) \quad
b_c^{dst(out,s)}=a_{dst,c}^{(out,s)}$ 

定义外点评值：$\mu_{dst}^{(out,s)}=\sum_{c=0}^{C-1}a_c^{dst(out,s)}b_c^{dst(out,s)}$

要求：$P_{E_{dst}^{(out,s)}}(y_{dst}^{(out,s)})=\mu_{dst}^{(out,s)}$

定义共享维累加器：$Acc_{dst}^{(out,s)}[0]=0$

$Acc_{dst}^{(out,s)}[c+1]=Acc_{dst}^{(out,s)}[c]+a_c^{dst(out,s)}b_c^{dst(out,s)} \quad 0\le c\le C-1$

对填充区统一规定：$a_c^{dst(out,s)}=0 \quad b_c^{dst(out,s)}=0 \quad Acc_{dst}^{(out,s)}[c+1]=Acc_{dst}^{(out,s)}[c] \quad (c\ge C)$

定义：$P_{a^{dst(out,s)}}(X) \quad P_{b^{dst(out,s)}}(X) \quad P_{Acc^{dst(out,s)}}(X)$

分别为其在 $\mathbb H_C$ 上的插值多项式。

提交承诺：$[P_{E_{src}^{(out,s)}}],\ [P_{a^{src(out,s)}}],\ [P_{b^{src(out,s)}}],\ [P_{Acc^{src(out,s)}}],\ [P_{E_{dst}^{(out,s)}}],\ [P_{a^{dst(out,s)}}],\ [P_{b^{dst(out,s)}}],\ [P_{Acc^{dst(out,s)}}]$ 

#### 2.4.3 第 $s$ 个输出头的源路由、LeakyReLU、最大值、范围检查、指数、分母、逆元与归一化权重

对每个边索引 $k$，先定义输出层第 $s$ 个 头 的源路由广播对象：$E_{src,k}^{edge(out,s)}=E_{src,src(k)}^{(out,s)} \quad Y_{k}'^{\star,edge(s)}=Y_{src(k)}'^{\star(s)}$

生成挑战：$\eta_{src}^{(out,s)},\beta_{src}^{(out,s)}$

对每个节点 $i$，定义表端：$Table^{src(out,s)}[i]=i+\eta_{src}^{(out,s)}E_{src,i}^{(out,s)}+(\eta_{src}^{(out,s)})^2Y_i'^{\star(s)}$

定义节点重数：$m_{src}^{(out,s)}[i]=\#\{k\mid src(k)=i\}$

对每个边索引 $k$，定义查询端：$Query^{src(out,s)}[k]=src(k)+\eta_{src}^{(out,s)}E_{src,k}^{edge(out,s)}+(\eta_{src}^{(out,s)})^2Y_{k}'^{\star,edge(s)}$

定义公开总和：

$S_{src}^{(out,s)}=\sum_{i=0}^{N-1}\frac{m_{src}^{(out,s)}[i]}{Table^{src(out,s)}[i]+\beta_{src}^{(out,s)}}=\sum_{k=0}^{E-1}\frac{1}{Query^{src(out,s)}[k]+\beta_{src}^{(out,s)}}$

定义节点端累加器：$R_{src}^{node(out,s)}[0]=0$

$R_{src}^{node(out,s)}[i+1]=R_{src}^{node(out,s)}[i]+Q_N[i]\cdot\frac{m_{src}^{(out,s)}[i]}{Table^{src(out,s)}[i]+\beta_{src}^{(out,s)}}$

定义边端累加器：$R_{src}^{edge(out,s)}[0]=0$

$R_{src}^{edge(out,s)}[k+1]=R_{src}^{edge(out,s)}[k]+Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{src(out,s)}[k]+\beta_{src}^{(out,s)}}$

插值得到：$P_{Table^{src(out,s)}},\ P_{Query^{src(out,s)}},\ P_{m_{src}^{(out,s)}},\ P_{R_{src}^{node(out,s)}},\ P_{R_{src}^{edge(out,s)}}$

接着对每个边索引 $k$，定义：

$S_k^{(out,s)}=E_{src,k}^{edge(out,s)}+E_{dst,k}^{edge(out,s)}, \quad Z_k^{(out,s)}=LReLU(S_k^{(out,s)})$

生成挑战：$\eta_L^{(out,s)},\beta_L^{(out,s)}$

定义 LeakyReLU 表端：$Table^{L(out,s)}[t]=T_{LReLU}[t,0]+\eta_L^{(out,s)}T_{LReLU}[t,1]$

定义查询端：$Query^{L(out,s)}[k]=S_k^{(out,s)}+\eta_L^{(out,s)}Z_k^{(out,s)}$

定义表端有效区选择器：$Q_{tbl}^{L(out,s)}[t]=\begin{cases}1,&0\le t\le |T_{LReLU}|-1\\0,&|T_{LReLU}|\le t\le n_{edge}-1\end{cases}$

定义查询端有效区选择器：$Q_{qry}^{L(out,s)}[k]=Q_{edge}^{valid}[k]$

定义重数：$m_L^{(out,s)}[t]=\#\{k\in\{0,1,\ldots,E-1\}\mid (S_k^{(out,s)},Z_k^{(out,s)})=(T_{LReLU}[t,0],T_{LReLU}[t,1])\}$

定义累加器：$R_L^{(out,s)}[0]=0$

$R_L^{(out,s)}[k+1]=R_L^{(out,s)}[k]+Q_{tbl}^{L(out,s)}[k]\cdot\frac{m_L^{(out,s)}[k]}{Table^{L(out,s)}[k]+\beta_L^{(out,s)}}-Q_{qry}^{L(out,s)}[k]\cdot\frac{1}{Query^{L(out,s)}[k]+\beta_L^{(out,s)}}$

插值得到：$P_{Table^{L(out,s)}},\ P_{Query^{L(out,s)}},\ P_{m_L^{(out,s)}},\ P_{Q_{tbl}^{L(out,s)}},\ P_{Q_{qry}^{L(out,s)}},\ P_{R_L^{(out,s)}}$

定义组最大值：$M_i^{(out,s)}=\max\{Z_k^{(out,s)}\mid dst(k)=i\}$

定义最大值广播与差分：$M_k^{edge(out,s)}=M_{dst(k)}^{(out,s)}, \qquad \Delta_k^{+(out,s)}=M_k^{edge(out,s)}-Z_k^{(out,s)}$

引入二值指示变量：$s_{max}^{(out,s)}[k]\in\{0,1\}$，并要求：

$s_{max}^{(out,s)}[k](s_{max}^{(out,s)}[k]-1)=0, \quad s_{max}^{(out,s)}[k]\cdot \Delta_k^{+(out,s)}=0$

定义组内计数状态机：$C_{max}^{(out,s)}[0]=Q_{edge}^{valid}[0]\cdot s_{max}^{(out,s)}[0]$

$C_{max}^{(out,s)}[k+1]=(1-Q_{edge}^{valid}[k+1])C_{max}^{(out,s)}[k]+Q_{edge}^{valid}[k+1]\Big(Q_{grp\_new}^{edge}[k+1]s_{max}^{(out,s)}[k+1]+(1-Q_{grp\_new}^{edge}[k+1])(C_{max}^{(out,s)}[k]+s_{max}^{(out,s)}[k+1])\Big)$

并要求组末唯一性约束：$Q_{grp\_end}^{edge}[k]\cdot(C_{max}^{(out,s)}[k]-1)=0$

生成范围检查挑战：$\beta_R^{(out,s)}$

定义范围表端与查询端：$Table^{R(out,s)}[t]=t, \quad Query^{R(out,s)}[k]=\Delta_k^{+(out,s)}$

定义有效区选择器：$Q_{tbl}^{R(out,s)}[t]=\begin{cases}1,&0\le t\le 2^B-1\\0,&2^B\le t\le n_{edge}-1\end{cases}$，$Q_{qry}^{R(out,s)}[k]=Q_{edge}^{valid}[k]$

定义重数与累加器：$m_R^{(out,s)}[t]=\#\{k\in\{0,1,\ldots,E-1\}\mid \Delta_k^{+(out,s)}=t\}$，$R_R^{(out,s)}[0]=0$

$R_R^{(out,s)}[k+1]=R_R^{(out,s)}[k]+Q_{tbl}^{R(out,s)}[k]\cdot\frac{m_R^{(out,s)}[k]}{Table^{R(out,s)}[k]+\beta_R^{(out,s)}}-Q_{qry}^{R(out,s)}[k]\cdot\frac{1}{Query^{R(out,s)}[k]+\beta_R^{(out,s)}}$

插值得到：$P_{Table^{R(out,s)}},\ P_{Query^{R(out,s)}},\ P_{m_R^{(out,s)}},\ P_{Q_{tbl}^{R(out,s)}},\ P_{Q_{qry}^{R(out,s)}},\ P_{R_R^{(out,s)}}$

生成指数查表挑战：$\eta_{exp}^{(out,s)},\beta_{exp}^{(out,s)}$

定义：$U_k^{(out,s)}=ExpMap(\Delta_k^{+(out,s)})$

$Table^{exp(out,s)}[t]=T_{exp}[t,0]+\eta_{exp}^{(out,s)}T_{exp}[t,1]$

$Query^{exp(out,s)}[k]=\Delta_k^{+(out,s)}+\eta_{exp}^{(out,s)}U_k^{(out,s)}$

$Q_{tbl}^{exp(out,s)}[t]=\begin{cases}1,&0\le t\le |T_{exp}|-1\\0,&|T_{exp}|\le t\le n_{edge}-1\end{cases}$，$Q_{qry}^{exp(out,s)}[k]=Q_{edge}^{valid}[k]$

$m_{exp}^{(out,s)}[t]=\#\{k\in\{0,1,\ldots,E-1\}\mid (\Delta_k^{+(out,s)},U_k^{(out,s)})=(T_{exp}[t,0],T_{exp}[t,1])\}$，$R_{exp}^{(out,s)}[0]=0$

$R_{exp}^{(out,s)}[k+1]=R_{exp}^{(out,s)}[k]+Q_{tbl}^{exp(out,s)}[k]\cdot\frac{m_{exp}^{(out,s)}[k]}{Table^{exp(out,s)}[k]+\beta_{exp}^{(out,s)}}-Q_{qry}^{exp(out,s)}[k]\cdot\frac{1}{Query^{exp(out,s)}[k]+\beta_{exp}^{(out,s)}}$

插值得到：$P_{Table^{exp(out,s)}},\ P_{Query^{exp(out,s)}},\ P_{m_{exp}^{(out,s)}},\ P_{Q_{tbl}^{exp(out,s)}},\ P_{Q_{qry}^{exp(out,s)}},\ P_{R_{exp}^{(out,s)}}$

最后定义 softmax 分母、逆元与归一化权重：

$Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)}, \qquad inv_i^{(out,s)}=(Sum_i^{(out,s)})^{-1}$

$Sum_k^{edge(out,s)}=Sum_{dst(k)}^{(out,s)}, \qquad inv_k^{edge(out,s)}=inv_{dst(k)}^{(out,s)}$

$\alpha_k^{(out,s)}=U_k^{(out,s)}\cdot inv_k^{edge(out,s)}$

把 $U^{(out,s)},\ Sum^{(out,s)},\ Sum^{edge(out,s)},\ inv^{(out,s)},\ inv^{edge(out,s)},\alpha^{(out,s)}$ 分别插值成 $P_{U^{(out,s)}},\ P_{Sum^{(out,s)}},\ P_{Sum^{edge(out,s)}},\ P_{inv^{(out,s)}},\ P_{inv^{edge(out,s)}},\ P_{\alpha^{(out,s)}}$

提交承诺：$[P_{Table^{src(out,s)}}],\ [P_{Query^{src(out,s)}}],\ [P_{m_{src}^{(out,s)}}],\ [P_{R_{src}^{node(out,s)}}],\ [P_{R_{src}^{edge(out,s)}}],\ [P_{S^{(out,s)}}],\ [P_{Z^{(out,s)}}],\ [P_{M^{(out,s)}}],\ [P_{M^{edge(out,s)}}],\ [P_{\Delta^{+(out,s)}}],\ [P_{s_{max}^{(out,s)}}],\ [P_{C_{max}^{(out,s)}}],\ [P_{Table^{L(out,s)}}],\ [P_{Query^{L(out,s)}}],\ [P_{m_L^{(out,s)}}],\ [P_{Q_{tbl}^{L(out,s)}}],\ [P_{Q_{qry}^{L(out,s)}}],\ [P_{R_L^{(out,s)}}],\ [P_{Table^{R(out,s)}}],\ [P_{Query^{R(out,s)}}],\ [P_{m_R^{(out,s)}}],\ [P_{Q_{tbl}^{R(out,s)}}],\ [P_{Q_{qry}^{R(out,s)}}],\ [P_{R_R^{(out,s)}}],\ [P_{Table^{exp(out,s)}}],\ [P_{Query^{exp(out,s)}}],\ [P_{m_{exp}^{(out,s)}}],\ [P_{Q_{tbl}^{exp(out,s)}}],\ [P_{Q_{qry}^{exp(out,s)}}],\ [P_{R_{exp}^{(out,s)}}],\ [P_{U^{(out,s)}}],\ [P_{Sum^{(out,s)}}],\ [P_{Sum^{edge(out,s)}}],\ [P_{inv^{(out,s)}}],\ [P_{inv^{edge(out,s)}}],\ [P_{\alpha^{(out,s)}}]$

#### 2.4.4 第 $s$ 个输出头的聚合前输出、PSQ 与压缩绑定

对每个节点 $i$，定义：

1. 输出投影结果的类别压缩：$Y_i'^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}\xi_{out}^c$
   
2. 聚合前输出：$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}$
   
3. 聚合前输出的类别压缩：$Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{pre(s)}\xi_{out}^c$

对每个边索引 $k$，定义：

4. 输出投影结果的边域广播压缩：$Y_{k}'^{\star,edge(s)}=Y_{src(k)}'^{\star(s)}$
   
5. 压缩加权边特征：$\widehat y_k^{\star(s)}=\alpha_k^{(out,s)}Y_{k}'^{\star,edge(s)}$
   
6. 聚合前输出压缩广播：$Y_k^{pre,\star,edge(s)}=Y_{dst(k)}^{pre,\star(s)}$

在决定 $U^{(out,s)},Sum^{(out,s)},Sum^{edge(out,s)},Y'^{\star,edge(s)},\widehat y^{\star(s)},Y^{pre,\star(s)}$ 这些基础值及其相关承诺对象全部固定之后

生成 PSQ 挑战：$\lambda_{out}^{(s)}$

定义：

- $w_{out}^{(s)}[k]=U_k^{(out,s)}+\lambda_{out}^{(s)}\widehat y_k^{\star(s)}$
- $T_{out}^{pre(s)}[i]=Sum_i^{(out,s)}+\lambda_{out}^{(s)}Y_i^{pre,\star(s)}$
- $T_{out}^{pre,edge(s)}[k]=Sum_k^{edge(out,s)}+\lambda_{out}^{(s)}Y_k^{pre,\star,edge(s)}$

定义状态机：$PSQ^{(out,s)}[0]=Q_{edge}^{valid}[0]\cdot w_{out}^{(s)}[0]$

对 $0\le k\le n_{edge}-2$，定义：

$PSQ^{(out,s)}[k+1]=(1-Q_{edge}^{valid}[k+1])PSQ^{(out,s)}[k]+Q_{edge}^{valid}[k+1]\Big(Q_{grp\_new}^{edge}[k+1]w_{out}^{(s)}[k+1]+(1-Q_{grp\_new}^{edge}[k+1])\big(PSQ^{(out,s)}[k]+w_{out}^{(s)}[k+1]\big)\Big)$

并施加组末约束：

$Q_{grp\_end}^{edge}[k]\cdot\big(PSQ^{(out,s)}[k]-T_{out}^{pre,edge(s)}[k]\big)=0$

该组末约束等价地同时强制：

$Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)},\qquad
Y_i^{pre,\star(s)}=\sum_{\{k\mid dst(k)=i\}}\widehat y_k^{\star(s)}$

定义多项式：

$P_{Y'^{\star(s)}}(X)=\sum_{i=0}^{n_N-1}Y'^{\star(s)}[i]L_i^{(N)}(X)$

$P_{Y'^{\star,edge(s)}}(X)=\sum_{k=0}^{n_{edge}-1}Y'^{\star,edge(s)}[k]L_k^{(edge)}(X)$

$P_{Y^{pre(s)}}(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}Y_{i,c}^{pre(s)}X^{i\cdot C+c}$

$P_{Y^{pre,\star(s)}}(X)=\sum_{i=0}^{n_N-1}Y^{pre,\star(s)}[i]L_i^{(N)}(X)$

$P_{Y^{pre,\star,edge(s)}}(X)=\sum_{k=0}^{n_{edge}-1}Y^{pre,\star,edge(s)}[k]L_k^{(edge)}(X)$

$P_{\widehat y^{\star(s)}}(X)=\sum_{k=0}^{n_{edge}-1}\widehat y^{\star(s)}[k]L_k^{(edge)}(X)$

$P_{w_{out}^{(s)}}(X)=\sum_{k=0}^{n_{edge}-1}w_{out}^{(s)}[k]L_k^{(edge)}(X)$

$P_{T_{out}^{pre(s)}}(X)=\sum_{i=0}^{n_N-1}T_{out}^{pre(s)}[i]L_i^{(N)}(X)$

$P_{T_{out}^{pre,edge(s)}}(X)=\sum_{k=0}^{n_{edge}-1}T_{out}^{pre,edge(s)}[k]L_k^{(edge)}(X)$

$P_{PSQ^{(out,s)}}(X)=\sum_{k=0}^{n_{edge}-1}PSQ^{(out,s)}[k]L_k^{(edge)}(X)$

生成聚合前输出压缩绑定挑战：$y_{out,pre}^{(s)}=H_{FS}(\text{transcript},[P_{Y^{pre(s)}}],[P_{Y^{pre,\star(s)}}])$

要求：$P_{Y^{pre,\star(s)}}(y_{out,pre}^{(s)})=\sum_{c=0}^{C-1}\Big(\sum_{i=0}^{N-1}Y_{i,c}^{pre(s)}L_i^{(N)}(y_{out,pre}^{(s)})\Big)\xi_{out}^c$

提交承诺：$[P_{Y'^{\star(s)}}],\ [P_{Y'^{\star,edge(s)}}],\ [P_{Y^{pre(s)}}],\ [P_{Y^{pre,\star(s)}}],\ [P_{Y^{pre,\star,edge(s)}}],\ [P_{\widehat y^{\star(s)}}],\ [P_{w_{out}^{(s)}}],\ [P_{T_{out}^{pre(s)}}],\ [P_{T_{out}^{pre,edge(s)}}],\ [P_{PSQ^{(out,s)}}]$

#### 2.4.5 第 $s$ 个输出头的输出偏置、加偏置后压缩绑定与目标路由

对每个节点 $i$ 与类别索引 $c$，正式定义：

- 先做公开尺度对齐：$\widetilde Y_{i,c}^{pre(s)} = Rescale_{out}\!\big(Y_{i,c}^{pre(s)}; S_{out,pre}\rightarrow S_{out,head}\big)$
- 再做 bias-add：$Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}$

对每个节点 $i$，进一步定义：

- 尺度对齐后的聚合前压缩：$\widetilde Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}\widetilde Y_{i,c}^{pre(s)}\xi_{out}^c$
- 输出头压缩：$Y_i^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{(s)}\xi_{out}^c$
- 输出头压缩广播：$Y_k^{\star,edge(s)}=Y_{dst(k)}^{\star(s)}$

定义输出头偏置压缩常数：$b_{\star}^{(out,s)}=\sum_{c=0}^{C-1}b_c^{(out,s)}\xi_{out}^c$

当且仅当 $quant\_cfg\_id$ 指定 $S_{out,pre}=S_{out,head}$ 时，$Rescale_{out}$ 定义为恒等映射；否则，$Rescale_{out}$ 的舍入 / 截断规则必须完全由公开量化配置唯一决定。

定义多项式：

$P_{\widetilde Y^{pre(s)}}(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}\widetilde Y_{i,c}^{pre(s)}X^{i\cdot C+c}$

$P_{\widetilde Y^{pre,\star(s)}}(X)=\sum_{i=0}^{n_N-1}\widetilde Y^{pre,\star(s)}[i]L_i^{(N)}(X)$

$P_{Y^{(s)}}(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}Y_{i,c}^{(s)}X^{i\cdot C+c}$

$P_{Y^{\star(s)}}(X)=\sum_{i=0}^{n_N-1}Y^{\star(s)}[i]L_i^{(N)}(X)$

$P_{Y^{\star,edge(s)}}(X)=\sum_{k=0}^{n_{edge}-1}Y^{\star,edge(s)}[k]L_k^{(edge)}(X)$

生成输出偏置绑定挑战 $y_{bias}^{(out,s)}$，用于绑定 $P_{\widetilde Y^{pre,\star(s)}}$ 与 $P_{Y^{\star(s)}}$ 的 bias-add 关系；再生成压缩绑定挑战 $y_{out}^{(s)}$，用于绑定 $P_{Y^{(s)}}$ 与 $P_{Y^{\star(s)}}$ 的类别压缩关系。

随后在 $P_{E_{dst}^{(out,s)}},P_{M^{(out,s)}},P_{Sum^{(out,s)}},P_{inv^{(out,s)}},P_{Y^{\star(s)}}$ 固定后，生成目标路由挑战 $\eta_{dst}^{(out,s)},\beta_{dst}^{(out,s)}$，并定义：

$Table^{dst(out,s)}[i]=i+\eta_{dst}^{(out,s)}E_{dst,i}^{(out,s)}+(\eta_{dst}^{(out,s)})^2M_i^{(out,s)}+(\eta_{dst}^{(out,s)})^3Sum_i^{(out,s)}+(\eta_{dst}^{(out,s)})^4inv_i^{(out,s)}+(\eta_{dst}^{(out,s)})^5Y_i^{\star(s)}$

$Query^{dst(out,s)}[k]=dst(k)+\eta_{dst}^{(out,s)}E_{dst,k}^{edge(out,s)}+(\eta_{dst}^{(out,s)})^2M_k^{edge(out,s)}+(\eta_{dst}^{(out,s)})^3Sum_k^{edge(out,s)}+(\eta_{dst}^{(out,s)})^4inv_k^{edge(out,s)}+(\eta_{dst}^{(out,s)})^5Y_k^{\star,edge(s)}$

$m_{dst}^{(out,s)}[i]=\#\{k\mid dst(k)=i\}$

$S_{dst}^{(out,s)}=\sum_{i=0}^{N-1}\frac{m_{dst}^{(out,s)}[i]}{Table^{dst(out,s)}[i]+\beta_{dst}^{(out,s)}}=\sum_{k=0}^{E-1}\frac{1}{Query^{dst(out,s)}[k]+\beta_{dst}^{(out,s)}}$

$R_{dst}^{node(out,s)}[0]=0$

$R_{dst}^{node(out,s)}[i+1]=R_{dst}^{node(out,s)}[i]+Q_N[i]\cdot\frac{m_{dst}^{(out,s)}[i]}{Table^{dst(out,s)}[i]+\beta_{dst}^{(out,s)}}$

$R_{dst}^{edge(out,s)}[0]=0$

$R_{dst}^{edge(out,s)}[k+1]=R_{dst}^{edge(out,s)}[k]+Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{dst(out,s)}[k]+\beta_{dst}^{(out,s)}}$

插值得到：$P_{Table^{dst(out,s)}},\ P_{Query^{dst(out,s)}},\ P_{m_{dst}^{(out,s)}},\ P_{R_{dst}^{node(out,s)}},\ P_{R_{dst}^{edge(out,s)}}$

提交承诺：$[P_{\widetilde Y^{pre(s)}}],\ [P_{\widetilde Y^{pre,\star(s)}}],\ [P_{Y^{(s)}}],\ [P_{Y^{\star(s)}}],\ [P_{Y^{\star,edge(s)}}],\ [P_{Table^{dst(out,s)}}],\ [P_{Query^{dst(out,s)}}],\ [P_{m_{dst}^{(out,s)}}],\ [P_{R_{dst}^{node(out,s)}}],\ [P_{R_{dst}^{edge(out,s)}}]$

#### 2.4.6 最终多头平均输出

当全部输出头完成后，先定义最终求和与域内平均：

$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)} \quad Y_{i,c}^{avg}=K_{out}^{-1}\cdot Y_{i,c}^{sum}$

其中 $K_{out}^{-1}$ 是整数 $K_{out}$ 在 $\mathbb F_p$ 中的逆元。

本文协议固定要求最终平均保持输出头共同尺度，即 $S_{out}=S_{out,head}$。因此最终输出定义为：$Y_{i,c}=Y_{i,c}^{avg}$

定义最终输出系数多项式：$P_Y(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}Y_{i,c}X^{i\cdot C+c}$

定义最终压缩输出与广播：

- $Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c$
- $Y_k^{\star,edge}=Y_{dst(k)}^{\star}$
- $P_{Y^{\star}}(X)=\sum_{i=0}^{n_N-1}Y^{\star}[i]L_i^{(N)}(X)$
- $P_{Y^{\star,edge}}(X)=\sum_{k=0}^{n_{edge}-1}Y^{\star,edge}[k]L_k^{(edge)}(X)$

生成最终平均输出压缩绑定挑战 $y_{avg}=H_{FS}(\text{transcript},[P_Y],[P_{Y^{\star}}])$，要求：

$P_{Y^{\star}}(y_{avg}) = \sum_{c=0}^{C-1}\Big(\sum_{i=0}^{N-1}Y_{i,c}L_i^{(N)}(y_{avg})\Big)\xi_{out}^c$

最终平均一致性在后续节点域约束中写成：

$K_{out}P_{Y^{\star}}-\sum_{s=0}^{K_{out}-1}P_{Y^{\star(s)}}=0$


## 3. 证明生成

### 3.1 输入与输出

证明算法输入是：

- 参数生成输出：$(PK,VK_{KZG},VK_{static},VK_{model})$

- 公共输入：$(I,src,dst,G_{batch},node\_ptr,edge\_ptr,N,E,N_{total},L,\{K_{hid}^{(\ell)}\}_{\ell=1}^{L-1},\{d_{in}^{(\ell)}\}_{\ell=1}^{L},\{d_h^{(\ell)}\}_{\ell=1}^{L-1},\{d_{cat}^{(\ell)}\}_{\ell=1}^{L-1},C,B,K_{out})$

- 全部动态见证对象

输出最终证明：$\pi_{GAT}$

### 3.2 Fiat–Shamir 挑战顺序

验证者与证明者必须严格按以下顺序重放。

#### 3.2.1 全局阶段

1. 先从证明对象 $\pi_{GAT}$ 中按第 3.5 节规定的固定顺序解析元数据块 $M_{pub}$，并把 $M_{pub}$ 与全部公开量化参数一起吸入 transcript。随后再吸入 $N,E,N_{total},L,\{d_{in}^{(\ell)}\}_{\ell=1}^{L},\{d_h^{(\ell)}\}_{\ell=1}^{L-1},\{d_{cat}^{(\ell)}\}_{\ell=1}^{L-1},C,B,K_{out}$，

  以及 $[P_I],[P_{src}],[P_{dst}],[P_{node\_gid}],[P_{edge\_gid}],[P_{Q_{graph\_new}^{edge}}],[P_{Q_{graph\_end}^{edge}}],[P_{Q_{grp\_new}^{edge}}],[P_{Q_{grp\_end}^{edge}}]$ 

  再吸入基础动态承诺 $[P_H]$ 与静态特征表承诺，生成：$\eta_{feat},\beta_{feat}$。

2. 对每个隐藏层 $\ell=1,2,\ldots,L-1$，按层顺序重放该层全部 hidden- head挑战；在每一层内部，再按 $r=0,1,\ldots,K_{hid}^{(\ell)}-1$ 的顺序生成该层全部隐藏层挑战。

3. 对每个隐藏层 $\ell=1,2,\ldots,L-1$，其拼接阶段都严格分两步执行：先吸入该层的 $[P_{H_{cat}}^{(\ell)}]$，生成 $\xi_{cat}^{(\ell)}$，再吸入 $[P_{H_{cat}^{\star}}^{(\ell)}]$，生成 $y_{cat}^{(\ell)}$。

4. 最终输出层阶段按输出头编号 $s=0,1,\ldots,K_{out}-1$ 依次生成：

  $y_{proj}^{(out,s)},y_{src}^{(out,s)},y_{dst}^{(out,s)},\eta_{src}^{(out,s)},\beta_{src}^{(out,s)},\eta_L^{(out,s)},\beta_L^{(out,s)},\beta_R^{(out,s)},\eta_{exp}^{(out,s)},\beta_{exp}^{(out,s)},\lambda_{out}^{(s)},y_{out,pre}^{(s)},y_{bias}^{(out,s)},y_{out}^{(s)},\eta_{dst}^{(out,s)},\beta_{dst}^{(out,s)}$。

5. 全部输出头相关对象固定后，吸入 $[P_Y],[P_{Y^{\star}}]$，生成最终平均输出压缩挑战：$y_{avg}$。

6. 吸入全部动态承诺与全部静态承诺，生成商聚合挑战 $\alpha_{quot}$。

7. 生成工作域开放点：$z_{FH},z_{edge},z_{in},z_{d_h},z_{cat},z_C,z_N$。

8. 生成各工作域 batch opening 折叠挑战：$v_{FH},v_{edge},v_{in},v_{d_h},v_{cat},v_C,v_N$。

9. 生成外点评值批量折叠挑战：$\rho_{ext}$。


### 3.3 约束族与商多项式

为避免省略，本节先给出每类约束的显式公式，再给出每个工作域商多项式的完全展开表达。

#### 3.3.1 通用 LogUp 起点、递推、终点模板

设某个 LogUp 子系统定义在工作域 $\mathbb H_{\mathcal D}$ 上，含有：

- 表多项式 $P_{Table}(X)$；
- 查询多项式 $P_{Query}(X)$；
- 重数多项式 $P_m(X)$；
- 表端有效区选择器 $P_{Q_{tbl}}(X)$；
- 查询端有效区选择器 $P_{Q_{qry}}(X)$；
- 累加器 $P_R(X)$；
- 挑战 $\beta$。

则其三条主约束为：

1. 起点约束：$C_{lookup,0}(X)=First_{\mathcal D}(X)\cdot P_R(X)$

  

2. 递推约束：$\begin{aligned} C_{lookup,1}(X) ={}& \big(P_R(\omega_{\mathcal D}X)-P_R(X)\big) \big(P_{Table}(X)+\beta\big) \big(P_{Query}(X)+\beta\big)\\ &- P_{Q_{tbl}}(X)P_m(X)\big(P_{Query}(X)+\beta\big)\\ &+ P_{Q_{qry}}(X)\big(P_{Table}(X)+\beta\big) \end{aligned}$

	

3. 终点约束：$C_{lookup,2}(X)=Last_{\mathcal D}(X)\cdot P_R(X)$


若表列或查询列还要与基础对象绑定，则必须额外加入表绑定约束与查询绑定约束。

#### 3.3.2 通用节点路由模板

设某个节点端路由子系统定义在 $\mathbb H_N$ 上，含有：

- 表多项式 $P_{Table}(X)$；
- 节点重数多项式 $P_m(X)$；
- 节点累加器 $P_{R^{node}}(X)$；
- 节点有效区选择器 $P_{Q_N}(X)$；
- 公共总和 $S$；
- 挑战 $\beta$。

则三条主约束为：

1. 起点约束：$C_{route,node,0}(X)=First_N(X)\cdot P_{R^{node}}(X)$

2. 递推约束：$C_{route,node,1}(X) = \big(P_{R^{node}}(\omega_NX)-P_{R^{node}}(X)\big)\big(P_{Table}(X)+\beta\big) - P_{Q_N}(X)P_m(X)$

3. 终点约束：$C_{route,node,2}(X)=Last_N(X)\cdot\big(P_{R^{node}}(X)-S\big)$


#### 3.3.3 通用边路由模板

设某个边端路由子系统定义在 $\mathbb H_{edge}$ 上，含有：

- 查询多项式 $P_{Query}(X)$；
- 边端累加器 $P_{R^{edge}}(X)$；
- 边域有效区选择器 $P_{Q_{edge}^{valid}}(X)$；
- 公共总和 $S$；
- 挑战 $\beta$。

则三条主约束为：

1. 起点约束：$C_{route,edge,0}(X)=First_{edge}(X)\cdot P_{R^{edge}}(X)$

2. 递推约束：$C_{route,edge,1}(X) = \big(P_{R^{edge}}(\omega_{edge}X)-P_{R^{edge}}(X)\big)\big(P_{Query}(X)+\beta\big) - P_{Q_{edge}^{valid}}(X)$

3. 终点约束：$C_{route,edge,2}(X)=Last_{edge}(X)\cdot\big(P_{R^{edge}}(X)-S\big)$


#### 3.3.4 最大值唯一性约束

设包含：

- 二值指示多项式 $P_s(X)$；
- 差分多项式 $P_{\Delta}(X)$；
- 计数状态机 $P_C(X)$；
- 组起点选择器 $P_{Q_{grp\_new}^{edge}}(X)$；
- 组末尾选择器 $P_{Q_{grp\_end}^{edge}}(X)$；
- 边域有效区选择器 $P_{Q_{edge}^{valid}}(X)$。

则约束为：

1. 二值性：$C_{max,bin}(X)=P_{Q_{edge}^{valid}}(X)\cdot P_s(X)\big(P_s(X)-1\big)$

2. 被选中位置零差分：$C_{max,zero}(X)=P_{Q_{edge}^{valid}}(X)\cdot P_s(X)\cdot P_{\Delta}(X)$

3. 起点计数：$C_{max,0}(X)=First_{edge}(X)\cdot\big(P_C(X)-P_{Q_{edge}^{valid}}(X)P_s(X)\big)$

4. 递推：$\begin{aligned} C_{max,1}(X) ={}& P_C(\omega_{edge}X) - \Big( \big(1-P_{Q_{edge}^{valid}}(\omega_{edge}X)\big)P_C(X)\\ &\qquad\qquad+ P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_s(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ \big(1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)\big)\big(P_C(X)+P_s(\omega_{edge}X)\big) \big) \Big) \end{aligned}$

5. 组末唯一性：$C_{max,end}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_C(X)-1\big)$


#### 3.3.5 PSQ 状态机约束

设包含：

- 边级权值多项式 $P_w(X)$；
- 边级目标值多项式 $P_{T^{edge}}(X)$；
- 状态机多项式 $P_{PSQ}(X)$；
- 组起点选择器 $P_{Q_{grp\_new}^{edge}}(X)$；
- 组末尾选择器 $P_{Q_{grp\_end}^{edge}}(X)$；
- 边域有效区选择器 $P_{Q_{edge}^{valid}}(X)$。

则约束为：

1. 起点约束：$C_{psq,0}(X)=First_{edge}(X)\cdot\big(P_{PSQ}(X)-P_{Q_{edge}^{valid}}(X)P_w(X)\big)$

	

2. 递推约束：$\begin{aligned} C_{psq,1}(X) ={}& P_{PSQ}(\omega_{edge}X) - \Big( \big(1-P_{Q_{edge}^{valid}}(\omega_{edge}X)\big)P_{PSQ}(X)\\ &\qquad\qquad+ P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_w(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ \big(1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)\big)\big(P_{PSQ}(X)+P_w(\omega_{edge}X)\big) \big) \Big) \end{aligned}$

	

3. 组末约束：$C_{psq,end}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{PSQ}(X)-P_{T^{edge}}(X)\big)$


#### 3.3.6 逆元约束

设节点域分母、逆元分别为 $P_{Sum}(X)$、$P_{inv}(X)$，则节点域逆元约束为：$C_{inv}(X)=P_{Q_N}(X)\cdot\big(P_{Sum}(X)P_{inv}(X)-1\big)$

本文正式协议不采用复合多项式替换 $P_{Sum}(P_{dst}(X)) \quad P_{inv}(P_{dst}(X))$ 来表达边域广播一致性。

本文唯一采用的正式口径是：

1. 先把边域广播列 $Sum^{edge}$、$inv^{edge}$ 显式作为独立见证列提交；
2. 再通过目标路由查询端绑定、PSQ 目标值绑定以及后续对应的边域 / 节点域约束，强制这些广播列与对应节点域列保持一致。

因此，在 quotient 约束族中，$3.3.8$、$3.3.9$、$3.3.13$、$3.3.14$ 已经给出的目标路由与输出聚合约束，就是本文用于保证广播一致性的唯一正式约束来源；本节不再额外引入第二套复合多项式广播约束。

#### 3.3.7 特征检索域商多项式

特征检索域上的全部约束定义为：

1. lookup 起点约束：$C_{feat,0}(X)=First_{FH}(X)\cdot P_{R_{feat}}(X)$

	

2. lookup 递推约束：$\begin{aligned} C_{feat,1}(X) ={}& \big(P_{R_{feat}}(\omega_{FH}X)-P_{R_{feat}}(X)\big) \big(P_{Table^{feat}}(X)+\beta_{feat}\big) \big(P_{Query^{feat}}(X)+\beta_{feat}\big)\\ &- P_{Q_{tbl}^{feat}}(X)P_{m_{feat}}(X)\big(P_{Query^{feat}}(X)+\beta_{feat}\big)\\ &+ P_{Q_{qry}^{feat}}(X)\big(P_{Table^{feat}}(X)+\beta_{feat}\big) \end{aligned}$

	

3. lookup 终点约束：$C_{feat,2}(X)=Last_{FH}(X)\cdot P_{R_{feat}}(X)$

4. 表端绑定约束：$C_{feat,tbl}(X)=P_{Table^{feat}}(X)-\Big(P_{Row_{feat}^{tbl}}(X)+\eta_{feat}P_{Col_{feat}^{tbl}}(X)+\eta_{feat}^2P_{T_H}(X)\Big)$

5. 查询端绑定约束：$C_{feat,qry}(X)=P_{Query^{feat}}(X)-\Big(P_{I_{feat}^{qry}}(X)+\eta_{feat}P_{Col_{feat}^{qry}}(X)+\eta_{feat}^2P_H(X)\Big)$


于是特征检索域商多项式完全展开为：

$t_{FH}(X) = \frac{ \alpha_{quot}^{e_{feat,0}}C_{feat,0}(X) + \alpha_{quot}^{e_{feat,1}}C_{feat,1}(X) + \alpha_{quot}^{e_{feat,2}}C_{feat,2}(X) + \alpha_{quot}^{e_{feat,tbl}}C_{feat,tbl}(X) + \alpha_{quot}^{e_{feat,qry}}C_{feat,qry}(X) }{ Z_{FH}(X) }$

#### 3.3.8 单个隐藏层模板中第 $r$ 个注意力头在边域上的全部约束

以下约束对应某个固定隐藏层 $\ell$ 的模板；在该模板中，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$，在边域 $\mathbb H_{edge}$ 上定义以下约束。

##### （一）源路由边端约束

$C_{src,edge,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{src}^{edge(r)}}(X)$

$C_{src,edge,1}^{(r)}(X) = \big(P_{R_{src}^{edge(r)}}(\omega_{edge}X)-P_{R_{src}^{edge(r)}}(X)\big)\big(P_{Query^{src(r)}}(X)+\beta_{src}^{(r)}\big)-P_{Q_{edge}^{valid}}(X)$

$C_{src,edge,2}^{(r)}(X)=Last_{edge}(X)\cdot\big(P_{R_{src}^{edge(r)}}(X)-S_{src}^{(r)}\big)$

##### （二）LeakyReLU 约束

$C_{L,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_L^{(r)}}(X)$

$\begin{aligned} C_{L,1}^{(r)}(X) ={}& \big(P_{R_L^{(r)}}(\omega_{edge}X)-P_{R_L^{(r)}}(X)\big) \big(P_{Table^{L(r)}}(X)+\beta_L^{(r)}\big) \big(P_{Query^{L(r)}}(X)+\beta_L^{(r)}\big)\\ &- P_{Q_{tbl}^{L(r)}}(X)P_{m_L^{(r)}}(X)\big(P_{Query^{L(r)}}(X)+\beta_L^{(r)}\big)\\ &+ P_{Q_{qry}^{L(r)}}(X)\big(P_{Table^{L(r)}}(X)+\beta_L^{(r)}\big) \end{aligned}$

$C_{L,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_L^{(r)}}(X)$

$C_{L,tbl}^{(r)}(X)=P_{Table^{L(r)}}(X)-\big(P_{T_{LReLU},x}(X)+\eta_L^{(r)}P_{T_{LReLU},y}(X)\big)$

$C_{L,qry}^{(r)}(X)=P_{Query^{L(r)}}(X)-\big(P_{S^{(r)}}(X)+\eta_L^{(r)}P_{Z^{(r)}}(X)\big)$

##### （三）最大值唯一性约束

$C_{max,bin}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(r)}}(X)\big(P_{s_{max}^{(r)}}(X)-1\big)$

$C_{max,zero}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(r)}}(X)P_{\Delta^{+(r)}}(X)$

$C_{max,0}^{(r)}(X)=First_{edge}(X)\cdot\big(P_{C_{max}^{(r)}}(X)-P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(r)}}(X)\big)$

$\begin{aligned} C_{max,1}^{(r)}(X) ={}& P_{C_{max}^{(r)}}(\omega_{edge}X)\\ &- \Big( (1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{C_{max}^{(r)}}(X) + P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_{s_{max}^{(r)}}(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ (1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X))(P_{C_{max}^{(r)}}(X)+P_{s_{max}^{(r)}}(\omega_{edge}X)) \big) \Big) \end{aligned}$

$C_{max,end}^{(r)}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{C_{max}^{(r)}}(X)-1\big)$

##### （四）范围检查约束

$C_{R,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_R^{(r)}}(X)$

$\begin{aligned} C_{R,1}^{(r)}(X) ={}& \big(P_{R_R^{(r)}}(\omega_{edge}X)-P_{R_R^{(r)}}(X)\big) \big(P_{Table^{R(r)}}(X)+\beta_R^{(r)}\big) \big(P_{Query^{R(r)}}(X)+\beta_R^{(r)}\big)\\ &- P_{Q_{tbl}^{R(r)}}(X)P_{m_R^{(r)}}(X)\big(P_{Query^{R(r)}}(X)+\beta_R^{(r)}\big)\\ &+ P_{Q_{qry}^{R(r)}}(X)\big(P_{Table^{R(r)}}(X)+\beta_R^{(r)}\big) \end{aligned}$

$C_{R,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_R^{(r)}}(X)$

$C_{R,tbl}^{(r)}(X)=P_{Table^{R(r)}}(X)-P_{T_{range}}(X)$

$C_{R,qry}^{(r)}(X)=P_{Query^{R(r)}}(X)-P_{\Delta^{+(r)}}(X)$

##### （五）指数查表约束

$C_{exp,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{exp}^{(r)}}(X)$

$\begin{aligned} C_{exp,1}^{(r)}(X) ={}& \big(P_{R_{exp}^{(r)}}(\omega_{edge}X)-P_{R_{exp}^{(r)}}(X)\big) \big(P_{Table^{exp(r)}}(X)+\beta_{exp}^{(r)}\big) \big(P_{Query^{exp(r)}}(X)+\beta_{exp}^{(r)}\big)\\ &- P_{Q_{tbl}^{exp(r)}}(X)P_{m_{exp}^{(r)}}(X)\big(P_{Query^{exp(r)}}(X)+\beta_{exp}^{(r)}\big)\\ &+ P_{Q_{qry}^{exp(r)}}(X)\big(P_{Table^{exp(r)}}(X)+\beta_{exp}^{(r)}\big) \end{aligned}$

$C_{exp,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_{exp}^{(r)}}(X)$

$C_{exp,tbl}^{(r)}(X)=P_{Table^{exp(r)}}(X)-\big(P_{T_{exp},x}(X)+\eta_{exp}^{(r)}P_{T_{exp},y}(X)\big)$

$C_{exp,qry}^{(r)}(X)=P_{Query^{exp(r)}}(X)-\big(P_{\Delta^{+(r)}}(X)+\eta_{exp}^{(r)}P_{U^{(r)}}(X)\big)$

##### （六）聚合前 PSQ 约束

$C_{psq,0}^{(r)}(X)=First_{edge}(X)\cdot\big(P_{PSQ^{(r)}}(X)-P_{Q_{edge}^{valid}}(X)P_{w_{psq}^{(r)}}(X)\big)$

$\begin{aligned} C_{psq,1}^{(r)}(X) ={}& P_{PSQ^{(r)}}(\omega_{edge}X)\\ &- \Big( (1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{PSQ^{(r)}}(X) + P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_{w_{psq}^{(r)}}(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ (1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X))(P_{PSQ^{(r)}}(X)+P_{w_{psq}^{(r)}}(\omega_{edge}X)) \big) \Big) \end{aligned}$

$C_{psq,end}^{(r)}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{PSQ^{(r)}}(X)-P_{T_{psq}^{edge(r)}}(X)\big)$

##### （七）ELU 约束

$C_{ELU,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{ELU}^{(r)}}(X)$

$\begin{aligned} C_{ELU,1}^{(r)}(X) ={}& \big(P_{R_{ELU}^{(r)}}(\omega_{edge}X)-P_{R_{ELU}^{(r)}}(X)\big) \big(P_{Table^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big) \big(P_{Query^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big)\\ &- P_{Q_{tbl}^{ELU(r)}}(X)P_{m_{ELU}^{(r)}}(X)\big(P_{Query^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big)\\ &+ P_{Q_{qry}^{ELU(r)}}(X)\big(P_{Table^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big) \end{aligned}$

$C_{ELU,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_{ELU}^{(r)}}(X)$

$C_{ELU,tbl}^{(r)}(X)=P_{Table^{ELU(r)}}(X)-\big(P_{T_{ELU},x}(X)+\eta_{ELU}^{(r)}P_{T_{ELU},y}(X)\big)$

$C_{ELU,qry}^{(r)}(X)=P_{Query^{ELU(r)}}(X)-\big(P_{H_{agg,pre}^{(r)}}(X)+\eta_{ELU}^{(r)}P_{H_{agg}^{(r)}}(X)\big)$

##### （八）目标路由边端约束

$C_{dst,edge,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{dst}^{edge(r)}}(X)$

$C_{dst,edge,1}^{(r)}(X) = \big(P_{R_{dst}^{edge(r)}}(\omega_{edge}X)-P_{R_{dst}^{edge(r)}}(X)\big)\big(P_{Query^{dst(r)}}(X)+\beta_{dst}^{(r)}\big)-P_{Q_{edge}^{valid}}(X)$

$C_{dst,edge,2}^{(r)}(X)=Last_{edge}(X)\cdot\big(P_{R_{dst}^{edge(r)}}(X)-S_{dst}^{(r)}\big)$

##### （九）边域绑定约束

$C_{src,qry}^{(r)}(X)=P_{Query^{src(r)}}(X)-\Big(P_{src}(X)+\eta_{src}^{(r)}P_{E_{src}^{edge(r)}}(X)+(\eta_{src}^{(r)})^2P_{H_{src}^{\star,edge(r)}}(X)\Big)$

$C_{dst,qry}^{(r)}(X)=P_{Query^{dst(r)}}(X)-\Big( P_{dst}(X)+\eta_{dst}^{(r)}P_{E_{dst}^{edge(r)}}(X)+(\eta_{dst}^{(r)})^2P_{M^{edge(r)}}(X)+(\eta_{dst}^{(r)})^3P_{Sum^{edge(r)}}(X)+(\eta_{dst}^{(r)})^4P_{inv^{edge(r)}}(X)+(\eta_{dst}^{(r)})^5P_{H_{agg}^{\star,edge(r)}}(X) \Big)$

#### 3.3.9 单个隐藏层模板中第 $r$ 个注意力头在节点域上的全部约束

以下约束对应某个固定隐藏层 $\ell$ 的模板；对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$，在节点域 $\mathbb H_N$ 上定义：

1. 源路由节点端三条约束：

	$C_{src,node,0}^{(r)}(X)=First_N(X)\cdot P_{R_{src}^{node(r)}}(X)$

	$C_{src,node,1}^{(r)}(X)=\big(P_{R_{src}^{node(r)}}(\omega_NX)-P_{R_{src}^{node(r)}}(X)\big)\big(P_{Table^{src(r)}}(X)+\beta_{src}^{(r)}\big)-P_{Q_N}(X)P_{m_{src}^{(r)}}(X)$

	$C_{src,node,2}^{(r)}(X)=Last_N(X)\cdot\big(P_{R_{src}^{node(r)}}(X)-S_{src}^{(r)}\big)$

2. 目标路由节点端三条约束：

	$C_{dst,node,0}^{(r)}(X)=First_N(X)\cdot P_{R_{dst}^{node(r)}}(X)$

	$C_{dst,node,1}^{(r)}(X)=\big(P_{R_{dst}^{node(r)}}(\omega_NX)-P_{R_{dst}^{node(r)}}(X)\big)\big(P_{Table^{dst(r)}}(X)+\beta_{dst}^{(r)}\big)-P_{Q_N}(X)P_{m_{dst}^{(r)}}(X)$

	$C_{dst,node,2}^{(r)}(X)=Last_N(X)\cdot\big(P_{R_{dst}^{node(r)}}(X)-S_{dst}^{(r)}\big)$

3. 分母逆元约束：$C_{inv}^{(r)}(X)=P_{Q_N}(X)\cdot\big(P_{Sum^{(r)}}(X)P_{inv^{(r)}}(X)-1\big)$

4. 源路由表端绑定：$C_{src,tbl}^{(r)}(X)=P_{Table^{src(r)}}(X)-\Big(P_{Idx_N}(X)+\eta_{src}^{(r)}P_{E_{src}^{(r)}}(X)+(\eta_{src}^{(r)})^2P_{H^{\star(r)}}(X)\Big)$

5. 目标路由表端绑定：$\begin{aligned} C_{dst,tbl}^{(r)}(X) ={}& P_{Table^{dst(r)}}(X) - \Big( P_{Idx_N}(X) + \eta_{dst}^{(r)}P_{E_{dst}^{(r)}}(X)\\ &\qquad +(\eta_{dst}^{(r)})^2P_{M^{(r)}}(X) +(\eta_{dst}^{(r)})^3P_{Sum^{(r)}}(X) +(\eta_{dst}^{(r)})^4P_{inv^{(r)}}(X) +(\eta_{dst}^{(r)})^5P_{H_{agg}^{\star(r)}}(X) \Big) \end{aligned}$


#### 3.3.10 第 $r$ 个隐藏层注意力头在共享维域上的全部绑定约束

##### （一）输入共享维域 $\mathbb H_{in}$

$C_{proj,0}^{(r)}(X)=First_{in}(X)\cdot P_{Acc^{proj(r)}}(X)$

$C_{proj,1}^{(r)}(X)=P_{Acc^{proj(r)}}(\omega_{in}X)-P_{Acc^{proj(r)}}(X)-P_{a^{proj(r)}}(X)P_{b^{proj(r)}}(X)$

$C_{proj,2}^{(r)}(X)=Last_{in}(X)\cdot\big(P_{Acc^{proj(r)}}(X)-\mu_{proj}^{(r)}\big)$

##### （二）隐藏层单头共享维域 $\mathbb H_{d_h}$

源注意力绑定：

$C_{srcbind,0}^{(r)}(X)=First_{d_h}(X)\cdot P_{Acc^{src(r)}}(X)$

$C_{srcbind,1}^{(r)}(X)=P_{Acc^{src(r)}}(\omega_{d_h}X)-P_{Acc^{src(r)}}(X)-P_{a^{src(r)}}(X)P_{b^{src(r)}}(X)$

$C_{srcbind,2}^{(r)}(X)=Last_{d_h}(X)\cdot\big(P_{Acc^{src(r)}}(X)-\mu_{src}^{(r)}\big)$

目标注意力绑定：

$C_{dstbind,0}^{(r)}(X)=First_{d_h}(X)\cdot P_{Acc^{dst(r)}}(X)$

$C_{dstbind,1}^{(r)}(X)=P_{Acc^{dst(r)}}(\omega_{d_h}X)-P_{Acc^{dst(r)}}(X)-P_{a^{dst(r)}}(X)P_{b^{dst(r)}}(X)$

$C_{dstbind,2}^{(r)}(X)=Last_{d_h}(X)\cdot\big(P_{Acc^{dst(r)}}(X)-\mu_{dst}^{(r)}\big)$

压缩特征绑定：

$C_{starbind,0}^{(r)}(X)=First_{d_h}(X)\cdot P_{Acc^{\star(r)}}(X)$

$C_{starbind,1}^{(r)}(X)=P_{Acc^{\star(r)}}(\omega_{d_h}X)-P_{Acc^{\star(r)}}(X)-P_{a^{\star(r)}}(X)P_{b^{\star(r)}}(X)$

$C_{starbind,2}^{(r)}(X)=Last_{d_h}(X)\cdot\big(P_{Acc^{\star(r)}}(X)-\mu_{\star}^{(r)}\big)$

先对每个隐藏维索引 $j\in\{0,1,\ldots,d_h-1\}$ 定义节点域列多项式：

$P_{A_{agg,pre,j}^{(r)}}(X)=\sum_{i=0}^{n_N-1}H_{agg,pre,i,j}^{(r)}L_i^{(N)}(X)$

$P_{A_{agg,j}^{(r)}}(X)=\sum_{i=0}^{n_N-1}H_{agg,i,j}^{(r)}L_i^{(N)}(X)$

聚合前压缩绑定：$C_{aggpre}^{(r)}(X)=P_{H_{agg,pre}^{\star(r)}}(X)-\sum_{j=0}^{d_h-1}P_{A_{agg,pre,j}^{(r)}}(X)(\xi^{(r)})^j$

聚合后压缩绑定：$C_{agg}^{(r)}(X)=P_{H_{agg}^{\star(r)}}(X)-\sum_{j=0}^{d_h-1}P_{A_{agg,j}^{(r)}}(X)(\xi^{(r)})^j$

#### 3.3.11 拼接共享维域约束

在 $\mathbb H_{cat}$ 上，除每个隐藏层拼接压缩绑定约束外，还必须显式写出最终输出层投影的共享维绑定约束。

对任意隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，其拼接压缩绑定写为：

$C_{cat}^{(\ell)}(X)=P_{H_{cat}^{\star(\ell)}}(X)-\sum_{m=0}^{d_{cat}^{(\ell)}-1}P_{A_{cat,m}^{(\ell)}}(X)(\xi_{cat}^{(\ell)})^m$


其中
$P_{A_{cat,m}^{(\ell)}}(X)=\sum_{i=0}^{n_N-1}H_{cat,i,m}^{(\ell)}L_i^{(N)}(X)$

对每个输出头 $s\in\{0,1,\ldots,K_{out}-1\}$，输出投影的共享维绑定显式定义为：


$C_{outproj,0}^{(s)}(X)=First_{cat}(X)\cdot P_{Acc^{proj(out,s)}}(X)$


$C_{outproj,1}^{(s)}(X)=P_{Acc^{proj(out,s)}}(\omega_{cat}X)-P_{Acc^{proj(out,s)}}(X)-P_{a^{proj(out,s)}}(X)P_{b^{proj(out,s)}}(X)$


$C_{outproj,2}^{(s)}(X)=Last_{cat}(X)\cdot\big(P_{Acc^{proj(out,s)}}(X)-\mu_{proj}^{(out,s)}\big)$

#### 3.3.12 输出层类别共享维域约束

在 $\mathbb H_C$ 上，对每个输出头 $s$ 显式定义：

##### （一）输出层源注意力绑定


$C_{outsrc,0}^{(s)}(X)=First_C(X)\cdot P_{Acc^{src(out,s)}}(X)$


$C_{outsrc,1}^{(s)}(X)=P_{Acc^{src(out,s)}}(\omega_CX)-P_{Acc^{src(out,s)}}(X)-P_{a^{src(out,s)}}(X)P_{b^{src(out,s)}}(X)$


$C_{outsrc,2}^{(s)}(X)=Last_C(X)\cdot\big(P_{Acc^{src(out,s)}}(X)-\mu_{src}^{(out,s)}\big)$

##### （二）输出层目标注意力绑定


$C_{outdst,0}^{(s)}(X)=First_C(X)\cdot P_{Acc^{dst(out,s)}}(X)$


$C_{outdst,1}^{(s)}(X)=P_{Acc^{dst(out,s)}}(\omega_CX)-P_{Acc^{dst(out,s)}}(X)-P_{a^{dst(out,s)}}(X)P_{b^{dst(out,s)}}(X)$


$C_{outdst,2}^{(s)}(X)=Last_C(X)\cdot\big(P_{Acc^{dst(out,s)}}(X)-\mu_{dst}^{(out,s)}\big)$

##### （三）尺度对齐后聚合前输出压缩绑定

先对每个类别索引 $c$ 定义节点域列多项式：$P_{A_{\widetilde Y,c}^{(s)}}(X)=\sum_{i=0}^{n_N-1}\widetilde Y_{i,c}^{pre(s)}L_i^{(N)}(X)$

则有：$C_{out\widetilde Y}^{(s)}(X)=P_{\widetilde Y^{pre,\star(s)}}(X)-\sum_{c=0}^{C-1}P_{A_{\widetilde Y,c}^{(s)}}(X)\xi_{out}^c$

##### （四）输出头压缩绑定

先对每个类别索引 $c$ 定义节点域列多项式：$P_{A_{Y,c}^{(s)}}(X)=\sum_{i=0}^{n_N-1}Y_{i,c}^{(s)}L_i^{(N)}(X)$

则有：$C_{outY}^{(s)}(X)=P_{Y^{\star(s)}}(X)-\sum_{c=0}^{C-1}P_{A_{Y,c}^{(s)}}(X)\xi_{out}^c$

##### （五）最终平均输出压缩绑定

先对每个类别索引 $c$ 定义：$P_{A_{Y,c}}(X)=\sum_{i=0}^{n_N-1}Y_{i,c}L_i^{(N)}(X)$

则有：$C_{outY}^{avg}(X)=P_{Y^{\star}}(X)-\sum_{c=0}^{C-1}P_{A_{Y,c}}(X)\xi_{out}^c$

#### 3.3.13 输出层在边域上的全部约束

在边域 $\mathbb H_{edge}$ 上，对每个输出头 $s$，显式定义如下约束。

##### （一）源路由边端约束


$C_{out,src,edge,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_{src}^{edge(out,s)}}(X)$


$C_{out,src,edge,1}^{(s)}(X)=\big(P_{R_{src}^{edge(out,s)}}(\omega_{edge}X)-P_{R_{src}^{edge(out,s)}}(X)\big)\big(P_{Query^{src(out,s)}}(X)+\beta_{src}^{(out,s)}\big)-P_{Q_{edge}^{valid}}(X)$


$C_{out,src,edge,2}^{(s)}(X)=Last_{edge}(X)\cdot\big(P_{R_{src}^{edge(out,s)}}(X)-S_{src}^{(out,s)}\big)$

##### （二）LeakyReLU 约束


$C_{out,L,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_L^{(out,s)}}(X)$


$\begin{aligned}
C_{out,L,1}^{(s)}(X)={}&\big(P_{R_L^{(out,s)}}(\omega_{edge}X)-P_{R_L^{(out,s)}}(X)\big)
\big(P_{Table^{L(out,s)}}(X)+\beta_L^{(out,s)}\big)
\big(P_{Query^{L(out,s)}}(X)+\beta_L^{(out,s)}\big)\\
&-P_{Q_{tbl}^{L(out,s)}}(X)P_{m_L^{(out,s)}}(X)\big(P_{Query^{L(out,s)}}(X)+\beta_L^{(out,s)}\big)\\
&+P_{Q_{qry}^{L(out,s)}}(X)\big(P_{Table^{L(out,s)}}(X)+\beta_L^{(out,s)}\big)
\end{aligned}$


$C_{out,L,2}^{(s)}(X)=Last_{edge}(X)\cdot P_{R_L^{(out,s)}}(X)$


$C_{out,L,tbl}^{(s)}(X)=P_{Table^{L(out,s)}}(X)-\big(P_{T_{LReLU},x}(X)+\eta_L^{(out,s)}P_{T_{LReLU},y}(X)\big)$


$C_{out,L,qry}^{(s)}(X)=P_{Query^{L(out,s)}}(X)-\big(P_{S^{(out,s)}}(X)+\eta_L^{(out,s)}P_{Z^{(out,s)}}(X)\big)$

##### （三）最大值唯一性约束


$C_{out,max,bin}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(out,s)}}(X)\big(P_{s_{max}^{(out,s)}}(X)-1\big)$


$C_{out,max,zero}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(out,s)}}(X)P_{\Delta^{+(out,s)}}(X)$


$C_{out,max,0}^{(s)}(X)=First_{edge}(X)\cdot\big(P_{C_{max}^{(out,s)}}(X)-P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(out,s)}}(X)\big)$


$\begin{aligned}
C_{out,max,1}^{(s)}(X)={}&P_{C_{max}^{(out,s)}}(\omega_{edge}X)\\
&-\Big((1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{C_{max}^{(out,s)}}(X)\\
&\qquad+P_{Q_{edge}^{valid}}(\omega_{edge}X)\big(P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_{s_{max}^{(out,s)}}(\omega_{edge}X)\\
&\qquad\qquad+(1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X))(P_{C_{max}^{(out,s)}}(X)+P_{s_{max}^{(out,s)}}(\omega_{edge}X))\big)\Big)
\end{aligned}$


$C_{out,max,end}^{(s)}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{C_{max}^{(out,s)}}(X)-1\big)$

##### （四）范围检查约束


$C_{out,R,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_R^{(out,s)}}(X)$


$\begin{aligned}
C_{out,R,1}^{(s)}(X)={}&\big(P_{R_R^{(out,s)}}(\omega_{edge}X)-P_{R_R^{(out,s)}}(X)\big)
\big(P_{Table^{R(out,s)}}(X)+\beta_R^{(out,s)}\big)
\big(P_{Query^{R(out,s)}}(X)+\beta_R^{(out,s)}\big)\\
&-P_{Q_{tbl}^{R(out,s)}}(X)P_{m_R^{(out,s)}}(X)\big(P_{Query^{R(out,s)}}(X)+\beta_R^{(out,s)}\big)\\
&+P_{Q_{qry}^{R(out,s)}}(X)\big(P_{Table^{R(out,s)}}(X)+\beta_R^{(out,s)}\big)
\end{aligned}$


$C_{out,R,2}^{(s)}(X)=Last_{edge}(X)\cdot P_{R_R^{(out,s)}}(X)$


$C_{out,R,tbl}^{(s)}(X)=P_{Table^{R(out,s)}}(X)-P_{T_{range}}(X)$


$C_{out,R,qry}^{(s)}(X)=P_{Query^{R(out,s)}}(X)-P_{\Delta^{+(out,s)}}(X)$

##### （五）指数查表约束


$C_{out,exp,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_{exp}^{(out,s)}}(X)$


$\begin{aligned}
C_{out,exp,1}^{(s)}(X)={}&\big(P_{R_{exp}^{(out,s)}}(\omega_{edge}X)-P_{R_{exp}^{(out,s)}}(X)\big)
\big(P_{Table^{exp(out,s)}}(X)+\beta_{exp}^{(out,s)}\big)
\big(P_{Query^{exp(out,s)}}(X)+\beta_{exp}^{(out,s)}\big)\\
&-P_{Q_{tbl}^{exp(out,s)}}(X)P_{m_{exp}^{(out,s)}}(X)\big(P_{Query^{exp(out,s)}}(X)+\beta_{exp}^{(out,s)}\big)\\
&+P_{Q_{qry}^{exp(out,s)}}(X)\big(P_{Table^{exp(out,s)}}(X)+\beta_{exp}^{(out,s)}\big)
\end{aligned}$


$C_{out,exp,2}^{(s)}(X)=Last_{edge}(X)\cdot P_{R_{exp}^{(out,s)}}(X)$


$C_{out,exp,tbl}^{(s)}(X)=P_{Table^{exp(out,s)}}(X)-\big(P_{T_{exp},x}(X)+\eta_{exp}^{(out,s)}P_{T_{exp},y}(X)\big)$


$C_{out,exp,qry}^{(s)}(X)=P_{Query^{exp(out,s)}}(X)-\big(P_{\Delta^{+(out,s)}}(X)+\eta_{exp}^{(out,s)}P_{U^{(out,s)}}(X)\big)$

##### （六）聚合前输出 PSQ 约束


$C_{out,psq,0}^{(s)}(X)=First_{edge}(X)\cdot\big(P_{PSQ^{(out,s)}}(X)-P_{Q_{edge}^{valid}}(X)P_{w_{out}^{(s)}}(X)\big)$


$\begin{aligned}
C_{out,psq,1}^{(s)}(X)={}&P_{PSQ^{(out,s)}}(\omega_{edge}X)\\
&-\Big((1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{PSQ^{(out,s)}}(X)\\
&\qquad+P_{Q_{edge}^{valid}}(\omega_{edge}X)\big(P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_{w_{out}^{(s)}}(\omega_{edge}X)\\
&\qquad\qquad+(1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X))(P_{PSQ^{(out,s)}}(X)+P_{w_{out}^{(s)}}(\omega_{edge}X))\big)\Big)
\end{aligned}$


$C_{out,psq,end}^{(s)}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{PSQ^{(out,s)}}(X)-P_{T_{out}^{pre,edge(s)}}(X)\big)$

##### （七）目标路由边端约束


$C_{out,dst,edge,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_{dst}^{edge(out,s)}}(X)$


$C_{out,dst,edge,1}^{(s)}(X)=\big(P_{R_{dst}^{edge(out,s)}}(\omega_{edge}X)-P_{R_{dst}^{edge(out,s)}}(X)\big)\big(P_{Query^{dst(out,s)}}(X)+\beta_{dst}^{(out,s)}\big)-P_{Q_{edge}^{valid}}(X)$


$C_{out,dst,edge,2}^{(s)}(X)=Last_{edge}(X)\cdot\big(P_{R_{dst}^{edge(out,s)}}(X)-S_{dst}^{(out,s)}\big)$

##### （八）边域绑定约束


$C_{out,src,qry}^{(s)}(X)=P_{Query^{src(out,s)}}(X)-\Big(P_{src}(X)+\eta_{src}^{(out,s)}P_{E_{src}^{edge(out,s)}}(X)+(\eta_{src}^{(out,s)})^2P_{Y'^{\star,edge(s)}}(X)\Big)$

$\begin{aligned}
C_{out,dst,qry}^{(s)}(X) ={} & P_{Query^{dst(out,s)}}(X) - \Big( P_{dst}(X) + \eta_{dst}^{(out,s)}P_{E_{dst}^{edge(out,s)}}(X) \\
& + (\eta_{dst}^{(out,s)})^2 P_{M^{edge(out,s)}}(X) + (\eta_{dst}^{(out,s)})^3 P_{Sum^{edge(out,s)}}(X) \\
& + (\eta_{dst}^{(out,s)})^4 P_{inv^{edge(out,s)}}(X) + (\eta_{dst}^{(out,s)})^5 P_{Y^{\star,edge(s)}}(X) \Big)
\end{aligned}$ 


#### 3.3.14 输出层在节点域上的全部约束

在节点域 $\mathbb H_N$ 上，对每个输出头 $s$ 显式定义：

##### （一）源路由节点端约束


$C_{out,src,node,0}^{(s)}(X)=First_N(X)\cdot P_{R_{src}^{node(out,s)}}(X)$


$C_{out,src,node,1}^{(s)}(X)=\big(P_{R_{src}^{node(out,s)}}(\omega_NX)-P_{R_{src}^{node(out,s)}}(X)\big)\big(P_{Table^{src(out,s)}}(X)+\beta_{src}^{(out,s)}\big)-P_{Q_N}(X)P_{m_{src}^{(out,s)}}(X)$


$C_{out,src,node,2}^{(s)}(X)=Last_N(X)\cdot\big(P_{R_{src}^{node(out,s)}}(X)-S_{src}^{(out,s)}\big)$

##### （二）目标路由节点端约束


$C_{out,dst,node,0}^{(s)}(X)=First_N(X)\cdot P_{R_{dst}^{node(out,s)}}(X)$


$C_{out,dst,node,1}^{(s)}(X)=\big(P_{R_{dst}^{node(out,s)}}(\omega_NX)-P_{R_{dst}^{node(out,s)}}(X)\big)\big(P_{Table^{dst(out,s)}}(X)+\beta_{dst}^{(out,s)}\big)-P_{Q_N}(X)P_{m_{dst}^{(out,s)}}(X)$


$C_{out,dst,node,2}^{(s)}(X)=Last_N(X)\cdot\big(P_{R_{dst}^{node(out,s)}}(X)-S_{dst}^{(out,s)}\big)$

##### （三）逆元约束


$C_{out,inv}^{(s)}(X)=P_{Q_N}(X)\cdot\big(P_{Sum^{(out,s)}}(X)P_{inv^{(out,s)}}(X)-1\big)$

##### （四）表端绑定约束


$C_{out,src,tbl}^{(s)}(X)=P_{Table^{src(out,s)}}(X)-\Big(P_{Idx_N}(X)+\eta_{src}^{(out,s)}P_{E_{src}^{(out,s)}}(X)+(\eta_{src}^{(out,s)})^2P_{Y'^{\star(s)}}(X)\Big)$


$\begin{aligned}
C_{out,dst,tbl}^{(s)}(X)={}&P_{Table^{dst(out,s)}}(X)-\Big(
P_{Idx_N}(X)+\eta_{dst}^{(out,s)}P_{E_{dst}^{(out,s)}}(X)
+(\eta_{dst}^{(out,s)})^2P_{M^{(out,s)}}(X)\\
&\qquad+(\eta_{dst}^{(out,s)})^3P_{Sum^{(out,s)}}(X)
+(\eta_{dst}^{(out,s)})^4P_{inv^{(out,s)}}(X)
+(\eta_{dst}^{(out,s)})^5P_{Y^{\star(s)}}(X)\Big)
\end{aligned}$

##### （五）最终平均输出一致性约束

本文正式协议要求最终平均与输出头共享同一共同尺度 $S_{out}=S_{out,head}$，因此最终节点域平均一致性可直接写为：


$C_{out,avg}(X)=K_{out}P_{Y^{\star}}(X)-\sum_{s=0}^{K_{out}-1}P_{Y^{\star(s)}}(X)$

本文当前协议严格禁止在最终平均后再额外执行独立回缩。若未来需要该能力，只能通过定义新的公开量化算子与新的协议版本实现；当前版本中不得沿用上式之外的其他最终平均语义。


#### 3.3.15 七个工作域的商多项式

七个工作域的商多项式保持原 formal 结构，但在多层 / 多头场景下，必须按**层序 + 头 序 + avg 对象**的固定顺序聚合。具体写法如下。

##### （一）边域商多项式

$\begin{aligned}
t_{edge}(X)=\frac{1}{Z_{edge}(X)}\Bigg[
&\sum_{\ell=1}^{L-1}\sum_{r=0}^{K_{hid}^{(\ell)}-1}\Big(
\alpha_{quot}^{e_{src,edge,0}^{(\ell,r)}}C_{src,edge,0}^{(\ell,r)}
+\alpha_{quot}^{e_{src,edge,1}^{(\ell,r)}}C_{src,edge,1}^{(\ell,r)}
+\alpha_{quot}^{e_{src,edge,2}^{(\ell,r)}}C_{src,edge,2}^{(\ell,r)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{L,0}^{(\ell,r)}}C_{L,0}^{(\ell,r)}
+\alpha_{quot}^{e_{L,1}^{(\ell,r)}}C_{L,1}^{(\ell,r)}
+\alpha_{quot}^{e_{L,2}^{(\ell,r)}}C_{L,2}^{(\ell,r)}
+\alpha_{quot}^{e_{L,tbl}^{(\ell,r)}}C_{L,tbl}^{(\ell,r)}
+\alpha_{quot}^{e_{L,qry}^{(\ell,r)}}C_{L,qry}^{(\ell,r)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{max,bin}^{(\ell,r)}}C_{max,bin}^{(\ell,r)}
+\alpha_{quot}^{e_{max,zero}^{(\ell,r)}}C_{max,zero}^{(\ell,r)}
+\alpha_{quot}^{e_{max,0}^{(\ell,r)}}C_{max,0}^{(\ell,r)}
+\alpha_{quot}^{e_{max,1}^{(\ell,r)}}C_{max,1}^{(\ell,r)}
+\alpha_{quot}^{e_{max,end}^{(\ell,r)}}C_{max,end}^{(\ell,r)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{R,0}^{(\ell,r)}}C_{R,0}^{(\ell,r)}
+\alpha_{quot}^{e_{R,1}^{(\ell,r)}}C_{R,1}^{(\ell,r)}
+\alpha_{quot}^{e_{R,2}^{(\ell,r)}}C_{R,2}^{(\ell,r)}
+\alpha_{quot}^{e_{R,tbl}^{(\ell,r)}}C_{R,tbl}^{(\ell,r)}
+\alpha_{quot}^{e_{R,qry}^{(\ell,r)}}C_{R,qry}^{(\ell,r)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{exp,0}^{(\ell,r)}}C_{exp,0}^{(\ell,r)}
+\alpha_{quot}^{e_{exp,1}^{(\ell,r)}}C_{exp,1}^{(\ell,r)}
+\alpha_{quot}^{e_{exp,2}^{(\ell,r)}}C_{exp,2}^{(\ell,r)}
+\alpha_{quot}^{e_{exp,tbl}^{(\ell,r)}}C_{exp,tbl}^{(\ell,r)}
+\alpha_{quot}^{e_{exp,qry}^{(\ell,r)}}C_{exp,qry}^{(\ell,r)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{psq,0}^{(\ell,r)}}C_{psq,0}^{(\ell,r)}
+\alpha_{quot}^{e_{psq,1}^{(\ell,r)}}C_{psq,1}^{(\ell,r)}
+\alpha_{quot}^{e_{psq,end}^{(\ell,r)}}C_{psq,end}^{(\ell,r)}
+\alpha_{quot}^{e_{ELU,0}^{(\ell,r)}}C_{ELU,0}^{(\ell,r)}
+\alpha_{quot}^{e_{ELU,1}^{(\ell,r)}}C_{ELU,1}^{(\ell,r)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{ELU,2}^{(\ell,r)}}C_{ELU,2}^{(\ell,r)}
+\alpha_{quot}^{e_{ELU,tbl}^{(\ell,r)}}C_{ELU,tbl}^{(\ell,r)}
+\alpha_{quot}^{e_{ELU,qry}^{(\ell,r)}}C_{ELU,qry}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,edge,0}^{(\ell,r)}}C_{dst,edge,0}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,edge,1}^{(\ell,r)}}C_{dst,edge,1}^{(\ell,r)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{dst,edge,2}^{(\ell,r)}}C_{dst,edge,2}^{(\ell,r)}
+\alpha_{quot}^{e_{src,qry}^{(\ell,r)}}C_{src,qry}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,qry}^{(\ell,r)}}C_{dst,qry}^{(\ell,r)}
\Big)\\
&+\sum_{s=0}^{K_{out}-1}\Big(
\alpha_{quot}^{e_{out,src,edge,0}^{(s)}}C_{out,src,edge,0}^{(s)}
+\alpha_{quot}^{e_{out,src,edge,1}^{(s)}}C_{out,src,edge,1}^{(s)}
+\alpha_{quot}^{e_{out,src,edge,2}^{(s)}}C_{out,src,edge,2}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{out,L,0}^{(s)}}C_{out,L,0}^{(s)}
+\alpha_{quot}^{e_{out,L,1}^{(s)}}C_{out,L,1}^{(s)}
+\alpha_{quot}^{e_{out,L,2}^{(s)}}C_{out,L,2}^{(s)}
+\alpha_{quot}^{e_{out,L,tbl}^{(s)}}C_{out,L,tbl}^{(s)}
+\alpha_{quot}^{e_{out,L,qry}^{(s)}}C_{out,L,qry}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{out,max,bin}^{(s)}}C_{out,max,bin}^{(s)}
+\alpha_{quot}^{e_{out,max,zero}^{(s)}}C_{out,max,zero}^{(s)}
+\alpha_{quot}^{e_{out,max,0}^{(s)}}C_{out,max,0}^{(s)}
+\alpha_{quot}^{e_{out,max,1}^{(s)}}C_{out,max,1}^{(s)}
+\alpha_{quot}^{e_{out,max,end}^{(s)}}C_{out,max,end}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{out,R,0}^{(s)}}C_{out,R,0}^{(s)}
+\alpha_{quot}^{e_{out,R,1}^{(s)}}C_{out,R,1}^{(s)}
+\alpha_{quot}^{e_{out,R,2}^{(s)}}C_{out,R,2}^{(s)}
+\alpha_{quot}^{e_{out,R,tbl}^{(s)}}C_{out,R,tbl}^{(s)}
+\alpha_{quot}^{e_{out,R,qry}^{(s)}}C_{out,R,qry}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{out,exp,0}^{(s)}}C_{out,exp,0}^{(s)}
+\alpha_{quot}^{e_{out,exp,1}^{(s)}}C_{out,exp,1}^{(s)}
+\alpha_{quot}^{e_{out,exp,2}^{(s)}}C_{out,exp,2}^{(s)}
+\alpha_{quot}^{e_{out,exp,tbl}^{(s)}}C_{out,exp,tbl}^{(s)}
+\alpha_{quot}^{e_{out,exp,qry}^{(s)}}C_{out,exp,qry}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{out,psq,0}^{(s)}}C_{out,psq,0}^{(s)}
+\alpha_{quot}^{e_{out,psq,1}^{(s)}}C_{out,psq,1}^{(s)}
+\alpha_{quot}^{e_{out,psq,end}^{(s)}}C_{out,psq,end}^{(s)}
+\alpha_{quot}^{e_{out,dst,edge,0}^{(s)}}C_{out,dst,edge,0}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{out,dst,edge,1}^{(s)}}C_{out,dst,edge,1}^{(s)}
+\alpha_{quot}^{e_{out,dst,edge,2}^{(s)}}C_{out,dst,edge,2}^{(s)}
+\alpha_{quot}^{e_{out,src,qry}^{(s)}}C_{out,src,qry}^{(s)}
+\alpha_{quot}^{e_{out,dst,qry}^{(s)}}C_{out,dst,qry}^{(s)}
\Big)
\Bigg]
\end{aligned}$ 

##### （二）拼接共享维域商多项式


$t_{cat}(X)=\frac{1}{Z_{cat}(X)}\Bigg[
\sum_{\ell=1}^{L-1}\alpha_{quot}^{e_{cat}^{(\ell)}}C_{cat}^{(\ell)}(X)
+\sum_{s=0}^{K_{out}-1}\Big(
\alpha_{quot}^{e_{outproj,0}^{(s)}}C_{outproj,0}^{(s)}
+\alpha_{quot}^{e_{outproj,1}^{(s)}}C_{outproj,1}^{(s)}
+\alpha_{quot}^{e_{outproj,2}^{(s)}}C_{outproj,2}^{(s)}
\Big)
\Bigg]$

##### （三）输出层类别共享维域商多项式


$\begin{aligned}
t_C(X)=\frac{1}{Z_C(X)}\Bigg[
&\sum_{s=0}^{K_{out}-1}\Big(
\alpha_{quot}^{e_{outsrc,0}^{(s)}}C_{outsrc,0}^{(s)}
+\alpha_{quot}^{e_{outsrc,1}^{(s)}}C_{outsrc,1}^{(s)}
+\alpha_{quot}^{e_{outsrc,2}^{(s)}}C_{outsrc,2}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{outdst,0}^{(s)}}C_{outdst,0}^{(s)}
+\alpha_{quot}^{e_{outdst,1}^{(s)}}C_{outdst,1}^{(s)}
+\alpha_{quot}^{e_{outdst,2}^{(s)}}C_{outdst,2}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{out\widetilde Y}^{(s)}}C_{out\widetilde Y}^{(s)}
+\alpha_{quot}^{e_{outY}^{(s)}}C_{outY}^{(s)}
\Big)
+\alpha_{quot}^{e_{outY}^{avg}}C_{outY}^{avg}
\Bigg]
\end{aligned}$

##### （四）节点域商多项式

$t_N(X)=\frac{1}{Z_N(X)}\Bigg[
\sum_{\ell=1}^{L-1}\sum_{r=0}^{K_{hid}^{(\ell)}-1}\Big(
\alpha_{quot}^{e_{src,node,0}^{(\ell,r)}}C_{src,node,0}^{(\ell,r)}
+\alpha_{quot}^{e_{src,node,1}^{(\ell,r)}}C_{src,node,1}^{(\ell,r)}
+\alpha_{quot}^{e_{src,node,2}^{(\ell,r)}}C_{src,node,2}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,node,0}^{(\ell,r)}}C_{dst,node,0}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,node,1}^{(\ell,r)}}C_{dst,node,1}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,node,2}^{(\ell,r)}}C_{dst,node,2}^{(\ell,r)}
+\alpha_{quot}^{e_{inv}^{(\ell,r)}}C_{inv}^{(\ell,r)}
+\alpha_{quot}^{e_{src,tbl}^{(\ell,r)}}C_{src,tbl}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,tbl}^{(\ell,r)}}C_{dst,tbl}^{(\ell,r)}
\Big)\\
+\sum_{s=0}^{K_{out}-1}\Big(
\alpha_{quot}^{e_{out,src,node,0}^{(s)}}C_{out,src,node,0}^{(s)}
+\alpha_{quot}^{e_{out,src,node,1}^{(s)}}C_{out,src,node,1}^{(s)}
+\alpha_{quot}^{e_{out,src,node,2}^{(s)}}C_{out,src,node,2}^{(s)}
+\alpha_{quot}^{e_{out,dst,node,0}^{(s)}}C_{out,dst,node,0}^{(s)}
+\alpha_{quot}^{e_{out,dst,node,1}^{(s)}}C_{out,dst,node,1}^{(s)}
+\alpha_{quot}^{e_{out,dst,node,2}^{(s)}}C_{out,dst,node,2}^{(s)}
+\alpha_{quot}^{e_{out,inv}^{(s)}}C_{out,inv}^{(s)}
+\alpha_{quot}^{e_{out,src,tbl}^{(s)}}C_{out,src,tbl}^{(s)}
+\alpha_{quot}^{e_{out,dst,tbl}^{(s)}}C_{out,dst,tbl}^{(s)}
\Big)
+\alpha_{quot}^{e_{out,avg}}C_{out,avg}
\Bigg]$

其余工作域 $t_{FH},t_{in},t_{d_h}$ 保持原 formal 结构不变；在实现中，必须仍按“先 hidden family、再输出层 family、最后 avg 对象”的固定顺序把相关点评值吸入 transcript 与 serializer。


### 3.4 掩蔽外点评值、域内点评值与开放

外点评值集合必须至少包括下列语义对象；但正式 proof object 中编码的是对应 $P^{zk}$ 的 masked evaluation，而不是裸 $P$ 的 witness evaluation：

1. 特征检索相关外点评值；
2. 对每个隐藏层 $\ell=1,\ldots,L-1$、以及该层每个注意力头 $r=0,\ldots,K_{hid}^{(\ell)}-1$：
   $P_{H'^{(\ell,r)}}(y_{proj}^{(\ell,r)}),P_{E_{src}^{(\ell,r)}}(y_{src}^{(\ell,r)}),P_{E_{dst}^{(\ell,r)}}(y_{dst}^{(\ell,r)}),P_{H^{\star(\ell,r)}}(y_{\star}^{(\ell,r)}),P_{H_{agg,pre}^{\star(\ell,r)}}(y_{agg,pre}^{(\ell,r)}),P_{H_{agg}^{\star(\ell,r)}}(y_{agg}^{(\ell,r)})$；
3. 对每个隐藏层 $\ell=1,\ldots,L-1$ 的拼接阶段：$P_{H_{cat}^{\star(\ell)}}(y_{cat}^{(\ell)})$；
4. 对每个输出头 $s$：
   $P_{Y'^{(s)}}(y_{proj}^{(out,s)}),P_{E_{src}^{(out,s)}}(y_{src}^{(out,s)}),P_{E_{dst}^{(out,s)}}(y_{dst}^{(out,s)}),P_{Y^{pre,\star(s)}}(y_{out,pre}^{(s)}),P_{\widetilde Y^{pre,\star(s)}}(y_{bias}^{(out,s)}),P_{Y^{\star(s)}}(y_{bias}^{(out,s)}),P_{Y^{\star(s)}}(y_{out}^{(s)})$；
5. 最终平均输出：$P_{Y^{\star}}(y_{avg}),P_Y(y_{avg})$。

对每个工作域 $\mathcal D\in\{FH,edge,in,d_h,cat,C,N\}$，都在点评集合 $\{z_{\mathcal D},z_{\mathcal D}\omega_{\mathcal D}\}$ 上执行多点评 batch opening。所有进入 opening 的私有多项式必须使用 $P^{zk}$；验证者只检查 $[P^{zk}]$ 与 masked evaluation 的一致性，不接收裸 witness evaluation。

对全部外点评值统一用 $\rho_{ext}$ 做折叠，形成唯一的外点评值批量开放。



#### 3.4.1 masked 外点评值绑定的主实验固定规则

由于 $P^{zk}(X)=P(X)+Z_{\mathcal D}(X)R_P(X)$ 在工作域外通常不等于 $P(X)$，外点评值绑定不能直接沿用旧的裸评值等式。主实验不允许让实现者自由选择绑定方式，必须按下表固定执行。

| 关系族 | 固定实现方式 | 说明 |
|---|---|---|
| CRPC 投影绑定 | `ZKSubproofMode` | 不输出裸折叠向量和裸 $\mu$ |
| 源注意力 / 目标注意力绑定 | `ZKSubproofMode` | 只输出承诺、masked evaluation 和子证明 |
| 压缩特征绑定 | `ZKSubproofMode` | 只输出承诺、masked evaluation 和子证明 |
| 拼接绑定 | `DomainQuotientMode` | 在工作域内用商多项式约束 |
| lookup / LogUp 累加器 | `DomainQuotientMode` | 使用域内商多项式约束 |
| route 一致性 | `DomainQuotientMode` | 使用域内商多项式约束 |
| PSQ 状态机 | `DomainQuotientMode` | 使用域内商多项式约束 |
| 最大值唯一性 / 范围 / 逆元 / 归一化 | `DomainQuotientMode` | 使用域内商多项式约束 |
| 输出平均绑定 | `DomainQuotientMode` | 使用域内商多项式约束 |
| 最终 logits 绑定 | `PublicOutputMode` | 主实验公开 logits 并绑定到 proof |

实现要求：

1. 每个 `Pi_bind` 必须带 `binding_mode`，且其值必须与上表一致。
2. Verifier 必须拒绝未声明 `binding_mode` 或模式与上表不一致的 proof。
3. Private binding 不得输出裸折叠向量、裸模型参数评值、裸中间激活或裸 $\mu$。
4. 子证明必须使用独立 domain-separation tag，例如 `bind:hidden:<layer>:<head>:proj`。
5. 若某个绑定关系暂时无法按上表实现，则该关系不得进入主实验。

#### 3.4.2 KZG opening 的正式算法接口与验证方程

本协议允许单点 opening、同点 batch opening 和多点 batch opening。所有 opening 都针对已承诺的 $P^{zk}$ 或 public polynomial；private polynomial 的 claimed value 必须标记为 `MaskedPrivateValue`。

单点 opening：

给定承诺 $C=[P]$、点 $z$ 和 claimed value $v=P(z)$，证明者构造 witness polynomial：

$$W(X)=\frac{P(X)-v}{X-z}.$$

证明为 $[W]$。验证者检查：

$$e(C-vG_1,G_2)=e([W], [\tau]_2-zG_2).$$

同点 batch opening：

对多项式 $\{P_i\}_{i=0}^{m-1}$、同一点 $z$、claimed values $\{v_i\}$，Fiat--Shamir 生成 folding challenge $\rho$，定义：

$$P_\rho(X)=\sum_{i=0}^{m-1}\rho^iP_i(X),\quad v_\rho=\sum_{i=0}^{m-1}\rho^iv_i.$$

证明者打开 $P_\rho$ 于 $z$，验证者用折叠承诺

$$C_\rho=\sum_{i=0}^{m-1}\rho^iC_i$$

执行单点 pairing 检查。

多点 batch opening：

对 $(P_i,z_j)$ 集合，主实现应采用分域批处理：同一工作域内先按点折叠，再按多项式折叠，最终生成一个或少量 witness commitments。若实现使用更通用的 KZG multiproof，必须在 `opening_mode` 中固定算法、排序、folding challenge 顺序和 proof size 统计口径。

实现规则：

1. Opening witness polynomial 可以在 coefficient basis 中构造；evaluation-basis 多项式可先通过 IFFT 转换，或使用等价的 barycentric quotient 构造。
2. 构造 witness 时不得使用 $\tau$。
3. Verifier 只使用 commitments、claimed values、points、folding challenges 和 $VK_{KZG}$。
4. 所有 folding challenge 必须吸入：metadata hash、commitment list、point list、visibility tags、claimed value tags。
5. `proof_size_bytes` 统计 opening proof curve points、claimed values、必要的 folding metadata；不统计 SRS、manifest、日志、运行时间、路径字符串。

#### 3.4.3 reference forward 一致性要求

为了证明 secure 主路径仍然对应原始 GAT 推理，主实验必须为每个数据集保存 reference forward 摘要：

1. 明文 GAT forward 的最终 logits hash；
2. 量化后 GAT forward 的最终 logits hash；
3. secure proof public output 的 logits hash；
4. `model_arch`、`model_params`、`quant_config`、`static_tables`；
5. 是否启用 self-loop、edge sort、multi-graph batch。

验收条件：

- 若 `public_output=logits`，secure proof 的 public logits 必须与量化 reference logits 完全一致。
- 若 `public_output=label`，必须额外证明 argmax，且 public label 必须与 reference label 一致。
- reference forward 时间不计入 proving time，但必须在实验 artifact 中保留 hash。

### 3.5 最终证明对象

最终证明对象：$\pi_{GAT}$，按如下**固定顺序**序列化为有序元组：

$\pi_{GAT} = \Big( M_{pub},\  \mathbf{Com}_{dyn}^{zk},\  \mathbf{S}_{route},\  \mathbf{Eval}_{ext}^{zk},\  \mathbf{Eval}_{dom}^{zk},\  \mathbf{Com}_{quot}^{zk},\  \mathbf{Open}_{dom}^{zk},\  W_{ext}^{zk},\  \Pi_{bind}^{zk} \Big)$

其中各大块的内部顺序必须完全固定为：

#### 3.5.1 元数据块

$M_{pub} = \big(protocol\_id,\ model\_arch\_id,\ model\_param\_id,\ static\_table\_id,\ quant\_cfg\_id,\ domain\_cfg,\ dim\_cfg,\ encoding\_id,\ padding\_rule\_id,\ degree\_bound\_id\big)$

#### 3.5.2 动态承诺块

$\mathbf{Com}_{dyn}$ 按以下顺序拼接：

1. 特征检索承诺块 $\mathbf{Com}_{feat}$，其内部顺序严格按第 2.1.7 节中承诺出现顺序：

   $\mathbf{Com}_{feat}=([P_H],[P_{Table^{feat}}],[P_{Query^{feat}}],[P_{m_{feat}}],[P_{Q_{tbl}^{feat}}],[P_{Q_{qry}^{feat}}],[P_{R_{feat}}])$

2. 隐藏层 family：对每个隐藏层 $\ell=1,\ldots,L-1$，依次写入
   
   $\mathbf{Com}_{hidden}^{(\ell,0)},\mathbf{Com}_{hidden}^{(\ell,1)},\ldots,\mathbf{Com}_{hidden}^{(\ell,K_{hid}^{(\ell)}-1)}$
   
   其中每个 $\mathbf{Com}_{hidden}^{(\ell,r)}$ 的内部顺序，**唯一且完全**由第 3.5.2.1 节的显式枚举定义；实现时不得再使用“按第 2.2 节自动平铺”的口径。

3. 隐藏层拼接块 family：对每个隐藏层 $\ell=1,\ldots,L-1$，在该层全部 头 承诺块之后立即写入
   
   $\mathbf{Com}_{cat}^{(\ell)}=([P_{H_{cat}}^{(\ell)}],[P_{H_{cat}^{\star}}^{(\ell)}])$
   
   不得把某层的拼接块移动到下一层 hidden family 之后。

4. 输出层承诺块 family：
   
   $\mathbf{Com}_{out}^{(0)},\mathbf{Com}_{out}^{(1)},\ldots,\mathbf{Com}_{out}^{(K_{out}-1)}$
   
   其中每个 $\mathbf{Com}_{out}^{(s)}$ 的内部顺序，**唯一且完全**由第 3.5.2.2 节的显式枚举定义；不同输出头的对象不得交叉混排。

5. 最终平均输出承诺块：$\mathbf{Com}_{out}^{avg}=([P_Y],[P_{Y^{\star}}],[P_{Y^{\star,edge}}])$
   

因此：$\mathbf{Com}_{dyn}=\big(\mathbf{Com}_{feat},\ \{\mathbf{Com}_{hidden}^{(\ell,r)}\}_{\ell,r\ \text{按层序与头序}},\ \{\mathbf{Com}_{cat}^{(\ell)}\}_{\ell=1}^{L-1},\ \mathbf{Com}_{out}^{(0)},\ldots,\mathbf{Com}_{out}^{(K_{out}-1)},\ \mathbf{Com}_{out}^{avg}\big)$


##### 3.5.2.1 单个隐藏头承诺块 $\mathbf{Com}_{hidden}^{(\ell,r)}$ 的完全枚举

对任意隐藏层 $\ell$、任意 hidden 头 $r$，正式固定：


$\begin{aligned}
\mathbf{Com}_{hidden}^{(\ell,r)}=\big(
&[P_{H'^{(\ell,r)}}],[P_{a^{proj(\ell,r)}}],[P_{b^{proj(\ell,r)}}],[P_{Acc^{proj(\ell,r)}}],\\
&[P_{E_{src}^{(\ell,r)}}],[P_{a^{src(\ell,r)}}],[P_{b^{src(\ell,r)}}],[P_{Acc^{src(\ell,r)}}],\\
&[P_{E_{dst}^{(\ell,r)}}],[P_{a^{dst(\ell,r)}}],[P_{b^{dst(\ell,r)}}],[P_{Acc^{dst(\ell,r)}}],\\
&[P_{H^{\star(\ell,r)}}],[P_{a^{\star(\ell,r)}}],[P_{b^{\star(\ell,r)}}],[P_{Acc^{\star(\ell,r)}}],\\
&[P_{Table^{src(\ell,r)}}],[P_{Query^{src(\ell,r)}}],[P_{m_{src}^{(\ell,r)}}],[P_{R_{src}^{node(\ell,r)}}],[P_{R_{src}^{edge(\ell,r)}}],\\
&[P_{S^{(\ell,r)}}],[P_{Z^{(\ell,r)}}],[P_{M^{(\ell,r)}}],[P_{M^{edge(\ell,r)}}],[P_{\Delta^{+(\ell,r)}}],[P_{s_{max}^{(\ell,r)}}],[P_{C_{max}^{(\ell,r)}}],\\
&[P_{Table^{L(\ell,r)}}],[P_{Query^{L(\ell,r)}}],[P_{m_L^{(\ell,r)}}],[P_{Q_{tbl}^{L(\ell,r)}}],[P_{Q_{qry}^{L(\ell,r)}}],[P_{R_L^{(\ell,r)}}],\\
&[P_{Table^{R(\ell,r)}}],[P_{Query^{R(\ell,r)}}],[P_{m_R^{(\ell,r)}}],[P_{Q_{tbl}^{R(\ell,r)}}],[P_{Q_{qry}^{R(\ell,r)}}],[P_{R_R^{(\ell,r)}}],\\
&[P_{U^{(\ell,r)}}],[P_{Sum^{(\ell,r)}}],[P_{Sum^{edge(\ell,r)}}],[P_{inv^{(\ell,r)}}],[P_{inv^{edge(\ell,r)}}],[P_{\alpha^{(\ell,r)}}],\\
&[P_{Table^{exp(\ell,r)}}],[P_{Query^{exp(\ell,r)}}],[P_{m_{exp}^{(\ell,r)}}],[P_{Q_{tbl}^{exp(\ell,r)}}],[P_{Q_{qry}^{exp(\ell,r)}}],[P_{R_{exp}^{(\ell,r)}}],\\
&[P_{H_{agg,pre}^{(\ell,r)}}],[P_{H_{agg,pre}^{\star(\ell,r)}}],[P_{\widehat v_{pre}^{\star(\ell,r)}}],[P_{w_{psq}^{(\ell,r)}}],[P_{T_{psq}^{(\ell,r)}}],[P_{T_{psq}^{edge(\ell,r)}}],[P_{PSQ^{(\ell,r)}}],\\
&[P_{H_{agg}^{(\ell,r)}}],[P_{H_{agg}^{\star(\ell,r)}}],[P_{Table^{ELU(\ell,r)}}],[P_{Query^{ELU(\ell,r)}}],[P_{m_{ELU}^{(\ell,r)}}],[P_{R_{ELU}^{(\ell,r)}}],\\
&[P_{Table^{dst(\ell,r)}}],[P_{Query^{dst(\ell,r)}}],[P_{m_{dst}^{(\ell,r)}}],[P_{R_{dst}^{node(\ell,r)}}],[P_{R_{dst}^{edge(\ell,r)}}]
\big).
\end{aligned}$

实现时不得再使用“按第 2.2 节出现顺序自动平铺”的模糊口径；serializer / parser 必须以上述枚举为准。

##### 3.5.2.2 单个输出头承诺块 $\mathbf{Com}_{out}^{(s)}$ 的完全枚举

对任意输出头 $s$，正式固定：


$\begin{aligned}
\mathbf{Com}_{out}^{(s)}=\big(
&[P_{Y'^{(s)}}],[P_{a^{proj(out,s)}}],[P_{b^{proj(out,s)}}],[P_{Acc^{proj(out,s)}}],\\
&[P_{E_{src}^{(out,s)}}],[P_{a^{src(out,s)}}],[P_{b^{src(out,s)}}],[P_{Acc^{src(out,s)}}],\\
&[P_{E_{dst}^{(out,s)}}],[P_{a^{dst(out,s)}}],[P_{b^{dst(out,s)}}],[P_{Acc^{dst(out,s)}}],\\
&[P_{Table^{src(out,s)}}],[P_{Query^{src(out,s)}}],[P_{m_{src}^{(out,s)}}],[P_{R_{src}^{node(out,s)}}],[P_{R_{src}^{edge(out,s)}}],\\
&[P_{S^{(out,s)}}],[P_{Z^{(out,s)}}],[P_{M^{(out,s)}}],[P_{M^{edge(out,s)}}],[P_{\Delta^{+(out,s)}}],[P_{s_{max}^{(out,s)}}],[P_{C_{max}^{(out,s)}}],\\
&[P_{Table^{L(out,s)}}],[P_{Query^{L(out,s)}}],[P_{m_L^{(out,s)}}],[P_{Q_{tbl}^{L(out,s)}}],[P_{Q_{qry}^{L(out,s)}}],[P_{R_L^{(out,s)}}],\\
&[P_{Table^{R(out,s)}}],[P_{Query^{R(out,s)}}],[P_{m_R^{(out,s)}}],[P_{Q_{tbl}^{R(out,s)}}],[P_{Q_{qry}^{R(out,s)}}],[P_{R_R^{(out,s)}}],\\
&[P_{Table^{exp(out,s)}}],[P_{Query^{exp(out,s)}}],[P_{m_{exp}^{(out,s)}}],[P_{Q_{tbl}^{exp(out,s)}}],[P_{Q_{qry}^{exp(out,s)}}],[P_{R_{exp}^{(out,s)}}],\\
&[P_{U^{(out,s)}}],[P_{Sum^{(out,s)}}],[P_{Sum^{edge(out,s)}}],[P_{inv^{(out,s)}}],[P_{inv^{edge(out,s)}}],[P_{\alpha^{(out,s)}}],\\
&[P_{Y'^{\star(s)}}],[P_{Y'^{\star,edge(s)}}],[P_{Y^{pre(s)}}],[P_{Y^{pre,\star(s)}}],[P_{Y^{pre,\star,edge(s)}}],[P_{\widehat y^{\star(s)}}],[P_{w_{out}^{(s)}}],[P_{T_{out}^{pre(s)}}],[P_{T_{out}^{pre,edge(s)}}],[P_{PSQ^{(out,s)}}],\\
&[P_{\widetilde Y^{pre(s)}}],[P_{\widetilde Y^{pre,\star(s)}}],[P_{Y^{(s)}}],[P_{Y^{\star(s)}}],[P_{Y^{\star,edge(s)}}],\\
&[P_{Table^{dst(out,s)}}],[P_{Query^{dst(out,s)}}],[P_{m_{dst}^{(out,s)}}],[P_{R_{dst}^{node(out,s)}}],[P_{R_{dst}^{edge(out,s)}}]
\big).
\end{aligned}$

最终平均输出承诺块保持为：


$\mathbf{Com}_{out}^{avg}=([P_Y],[P_{Y^{\star}}],[P_{Y^{\star,edge}}])$


#### 3.5.3 路由公开总和块

$\mathbf{S}_{route}$ 按如下顺序排列：

1. hidden 层 family：对每个隐藏层 $\ell=1,\ldots,L-1$，依次写入
   
   $\big(S_{src}^{(\ell,0)},S_{dst}^{(\ell,0)},S_{src}^{(\ell,1)},S_{dst}^{(\ell,1)},\ldots,S_{src}^{(\ell,K_{hid}^{(\ell)}-1)},S_{dst}^{(\ell,K_{hid}^{(\ell)}-1)}\big)$
   
2. 输出层 family：
   
   $\big(S_{src}^{(out,0)},S_{dst}^{(out,0)},S_{src}^{(out,1)},S_{dst}^{(out,1)},\ldots,S_{src}^{(out,K_{out}-1)},S_{dst}^{(out,K_{out}-1)}\big)$

整体上，$\mathbf{S}_{route}$ 必须先按层序写完全部 hidden 层，再写输出层；不得把输出层总和块插入到任一隐藏层 family 中间。


#### 3.5.4 外点评值块

$\mathbf{Eval}_{ext}$ 按如下顺序排列：

1. hidden 层 family：对每个隐藏层 $\ell=1,\ldots,L-1$、以及该层每个 头 $r=0,\ldots,K_{hid}^{(\ell)}-1$，依次写入与
   $y_{proj}^{(\ell,r)},y_{src}^{(\ell,r)},y_{dst}^{(\ell,r)},y_{\star}^{(\ell,r)},y_{agg,pre}^{(\ell,r)},y_{agg}^{(\ell,r)}$ 对应的外点评值。
   
2. hidden 层拼接阶段：对每个隐藏层 $\ell=1,\ldots,L-1$，写入$P_{H_{cat}^{\star(\ell)}}(y_{cat}^{(\ell)})$ 及其绑定所需外点评值。
   
3. 输出层 family：对每个 $s=0,\ldots,K_{out}-1$，依次写入与$y_{proj}^{(out,s)},y_{src}^{(out,s)},y_{dst}^{(out,s)},y_{out,pre}^{(s)},y_{bias}^{(out,s)},y_{out}^{(s)}$ 对应的外点评值。
   
4. 最终平均输出：最后写入 $(P_{Y^{\star}}(y_{avg}),P_Y(y_{avg}))$ 以及 $\pi_{bind}^{out,avg}$ 所需的外点评值。

##### 3.5.4.1 单个 hidden-head 外点评值块的完全枚举

对任意隐藏层 $\ell$、任意 hidden 头 $r$，其外点评值块固定为：


$\mathbf{Eval}_{ext}^{hidden,(\ell,r)}=\big(
P_{H'^{(\ell,r)}}(y_{proj}^{(\ell,r)}),\
P_{E_{src}^{(\ell,r)}}(y_{src}^{(\ell,r)}),\
P_{E_{dst}^{(\ell,r)}}(y_{dst}^{(\ell,r)}),\
P_{H^{\star(\ell,r)}}(y_{\star}^{(\ell,r)}),\ P_{H_{agg,pre}^{\star(\ell,r)}}(y_{agg,pre}^{(\ell,r)}),\
P_{H_{agg}^{\star(\ell,r)}}(y_{agg}^{(\ell,r)})
\big)$

##### 3.5.4.2 单个输出头外点评值块的完全枚举

对任意输出头 $s$，其外点评值块固定为：


$\mathbf{Eval}_{ext}^{out,s}=\big(
P_{Y'^{(s)}}(y_{proj}^{(out,s)}),\
P_{E_{src}^{(out,s)}}(y_{src}^{(out,s)}),\
P_{E_{dst}^{(out,s)}}(y_{dst}^{(out,s)}),\
P_{Y^{pre,\star(s)}}(y_{out,pre}^{(s)}),\
P_{\widetilde Y^{pre,\star(s)}}(y_{bias}^{(out,s)}),\ P_{Y^{\star(s)}}(y_{bias}^{(out,s)}),\
P_{Y^{\star(s)}}(y_{out}^{(s)})
\big)$

本文正式证明对象不暴露任何额外的折叠向量点评值。第 3.5.4 节已经枚举的外点评值列表就是完整列表；serializer 与 parser 不得再增加未枚举的额外点评值。

#### 3.5.5 域内点评值块

对任意定义在工作域 $\mathbb H_{\mathcal D}$ 上并且进入 proof object 的已承诺多项式 $P$，统一记其域内点评值对为：$\operatorname{ev}_{\mathcal D}(P):=\big(P(z_{\mathcal D}),P(z_{\mathcal D}\omega_{\mathcal D})\big)$。

若某条 quotient identity 还依赖验证者本地重建的公共多项式（例如公共 selector、公共索引列、公共拓扑多项式），则这些公共对象的域内点值**不**在 $\mathbf{Eval}_{dom}$ 中单独编码，而由验证者按与下文相同的对象顺序本地重建并插入检查。

$\mathbf{Eval}_{dom}$ 在本文正式协议中的唯一固定顺序定义为：

$\mathbf{Eval}_{dom}=(\mathbf{Eval}_{FH},\mathbf{Eval}_{edge},\mathbf{Eval}_{in},\mathbf{Eval}_{d_h},\mathbf{Eval}_{cat},\mathbf{Eval}_{C},\mathbf{Eval}_{N})$。

##### 3.5.5.1 特征检索域块


$\mathbf{Eval}_{FH}=\big(
\operatorname{ev}_{FH}(P_H),\
\operatorname{ev}_{FH}(P_{Table^{feat}}),\ \operatorname{ev}_{FH}(P_{Query^{feat}}),\
\operatorname{ev}_{FH}(P_{m_{feat}}),\
\operatorname{ev}_{FH}(P_{Q_{tbl}^{feat}}),\ \operatorname{ev}_{FH}(P_{Q_{qry}^{feat}}),\
\operatorname{ev}_{FH}(P_{R_{feat}})
\big)$

##### 3.5.5.2 边域块

边域块固定为：$\mathbf{Eval}_{edge}=\big(
\{\mathbf{Eval}_{edge}^{hidden,(\ell,r)}\}_{\ell=1}^{L-1}\!{}_{r=0}^{K_{hid}^{(\ell)}-1},\ \{\mathbf{Eval}_{edge}^{out,s}\}_{s=0}^{K_{out}-1},\
\mathbf{Eval}_{edge}^{avg}
\big)$

其中，对任意隐藏层 $\ell$、任意 hidden 头 $r$，固定定义：


$\mathbf{Eval}_{edge}^{hidden,(\ell,r)}=\big(
\operatorname{ev}_{edge}(P_{Query^{src(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{R_{src}^{edge(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{S^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Z^{(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{M^{edge(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{\Delta^{+(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{s_{max}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{C_{max}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Table^{L(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Query^{L(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{m_L^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Q_{tbl}^{L(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{Q_{qry}^{L(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{R_L^{(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{Table^{R(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Query^{R(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{m_R^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Q_{tbl}^{R(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Q_{qry}^{R(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{R_R^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{U^{(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{Sum^{edge(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{inv^{edge(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{\alpha^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Table^{exp(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{Query^{exp(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{m_{exp}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Q_{tbl}^{exp(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{Q_{qry}^{exp(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{R_{exp}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{\widehat v_{pre}^{\star(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{w_{psq}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{T_{psq}^{edge(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{PSQ^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Table^{ELU(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Query^{ELU(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{m_{ELU}^{(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{R_{ELU}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Query^{dst(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{R_{dst}^{edge(\ell,r)}})
\big)$

对任意输出头 $s$，固定定义：


$\mathbf{Eval}_{edge}^{out,s}=\big(
\operatorname{ev}_{edge}(P_{Query^{src(out,s)}}),\
\operatorname{ev}_{edge}(P_{R_{src}^{edge(out,s)}}),\ \operatorname{ev}_{edge}(P_{S^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Z^{(out,s)}}),\ \operatorname{ev}_{edge}(P_{M^{edge(out,s)}}),\
\operatorname{ev}_{edge}(P_{\Delta^{+(out,s)}}),\ \operatorname{ev}_{edge}(P_{s_{max}^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{C_{max}^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Table^{L(out,s)}}),\
\operatorname{ev}_{edge}(P_{Query^{L(out,s)}}),\
\operatorname{ev}_{edge}(P_{m_L^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Q_{tbl}^{L(out,s)}}),\ \operatorname{ev}_{edge}(P_{Q_{qry}^{L(out,s)}}),\
\operatorname{ev}_{edge}(P_{R_L^{(out,s)}}),\ \operatorname{ev}_{edge}(P_{Table^{R(out,s)}}),\
\operatorname{ev}_{edge}(P_{Query^{R(out,s)}}),\ \operatorname{ev}_{edge}(P_{m_R^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Q_{tbl}^{R(out,s)}}),\
\operatorname{ev}_{edge}(P_{Q_{qry}^{R(out,s)}}),\ \operatorname{ev}_{edge}(P_{R_R^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{U^{(out,s)}}),\ \operatorname{ev}_{edge}(P_{Sum^{edge(out,s)}}),\
\operatorname{ev}_{edge}(P_{inv^{edge(out,s)}}),\ \operatorname{ev}_{edge}(P_{\alpha^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Table^{exp(out,s)}}),\ \operatorname{ev}_{edge}(P_{Query^{exp(out,s)}}),\
\operatorname{ev}_{edge}(P_{m_{exp}^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Q_{tbl}^{exp(out,s)}}),\ \operatorname{ev}_{edge}(P_{Q_{qry}^{exp(out,s)}}),\
\operatorname{ev}_{edge}(P_{R_{exp}^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Y'^{\star,edge(s)}}),\
\operatorname{ev}_{edge}(P_{\widehat y^{\star(s)}}),\
\operatorname{ev}_{edge}(P_{w_{out}^{(s)}}),\ \operatorname{ev}_{edge}(P_{T_{out}^{pre,edge(s)}}),\
\operatorname{ev}_{edge}(P_{PSQ^{(out,s)}}),\ \operatorname{ev}_{edge}(P_{Query^{dst(out,s)}}),\
\operatorname{ev}_{edge}(P_{R_{dst}^{edge(out,s)}}),\
\operatorname{ev}_{edge}(P_{Y^{pre,\star,edge(s)}}),\
\operatorname{ev}_{edge}(P_{Y^{\star,edge(s)}})
\big)$

最终平均输出在边域上的唯一块固定为：$\mathbf{Eval}_{edge}^{avg}=\big( \operatorname{ev}_{edge}(P_{Y^{\star,edge}}) \big)$

##### 3.5.5.3 输入共享维域块

输入共享维域块固定为：$\mathbf{Eval}_{in}=\big(\{\mathbf{Eval}_{in}^{hidden,(\ell,r)}\}_{\ell=1}^{L-1}\!{}_{r=0}^{K_{hid}^{(\ell)}-1} \big)$

其中：$\mathbf{Eval}_{in}^{hidden,(\ell,r)}=\big(
\operatorname{ev}_{in}(P_{a^{proj(\ell,r)}}),\
\operatorname{ev}_{in}(P_{b^{proj(\ell,r)}}),\
\operatorname{ev}_{in}(P_{Acc^{proj(\ell,r)}})
\big)$

输入共享维域不包含输出层 family，也不包含 avg 块。

##### 3.5.5.4 隐藏层单头共享维域块

隐藏层单头共享维域块固定为：$\mathbf{Eval}_{d_h}=\big(
\{\mathbf{Eval}_{d_h}^{hidden,(\ell,r)}\}_{\ell=1}^{L-1}\!{}_{r=0}^{K_{hid}^{(\ell)}-1}
\big)$

其中

$\mathbf{Eval}_{d_h}^{hidden,(\ell,r)}=\big(
\operatorname{ev}_{d_h}(P_{a^{src(\ell,r)}}),\
\operatorname{ev}_{d_h}(P_{b^{src(\ell,r)}}),\
\operatorname{ev}_{d_h}(P_{Acc^{src(\ell,r)}}),\
\operatorname{ev}_{d_h}(P_{a^{dst(\ell,r)}}),\
\operatorname{ev}_{d_h}(P_{b^{dst(\ell,r)}}),\
\operatorname{ev}_{d_h}(P_{Acc^{dst(\ell,r)}}),\
\operatorname{ev}_{d_h}(P_{a^{\star(\ell,r)}}),\
\operatorname{ev}_{d_h}(P_{b^{\star(\ell,r)}}),\
\operatorname{ev}_{d_h}(P_{Acc^{\star(\ell,r)}})
\big)$

该域不包含输出层 family，也不包含 avg 块。

##### 3.5.5.5 拼接共享维域块

拼接共享维域块固定为：


$\mathbf{Eval}_{cat}=\big(
\{\mathbf{Eval}_{cat}^{concat,\ell}\}_{\ell=1}^{L-1},\
\{\mathbf{Eval}_{cat}^{out,s}\}_{s=0}^{K_{out}-1}
\big).$

其中，对任意隐藏层 $\ell$：


$\mathbf{Eval}_{cat}^{concat,\ell}=\big(
\operatorname{ev}_{cat}(P_{H_{cat}^{(\ell)}})
\big),$

对任意输出头 $s$：


$\mathbf{Eval}_{cat}^{out,s}=\big(
\operatorname{ev}_{cat}(P_{a^{proj(out,s)}}),\
\operatorname{ev}_{cat}(P_{b^{proj(out,s)}}),\
\operatorname{ev}_{cat}(P_{Acc^{proj(out,s)}})
\big).$

该域不包含 avg 块。

##### 3.5.5.6 输出类别共享维域块

输出类别共享维域块固定为：$\mathbf{Eval}_{C}=\big(\{\mathbf{Eval}_{C}^{out,s}\}_{s=0}^{K_{out}-1}\big)$

其中：


$\mathbf{Eval}_{C}^{out,s}=\big(
\operatorname{ev}_{C}(P_{a^{src(out,s)}}),\
\operatorname{ev}_{C}(P_{b^{src(out,s)}}),\
\operatorname{ev}_{C}(P_{Acc^{src(out,s)}}),\
\operatorname{ev}_{C}(P_{a^{dst(out,s)}}),\
\operatorname{ev}_{C}(P_{b^{dst(out,s)}}),\
\operatorname{ev}_{C}(P_{Acc^{dst(out,s)}})
\big).$

该域不包含隐藏层 family，也不包含 avg 块。

##### 3.5.5.7 节点域块

节点域块固定为：


$\mathbf{Eval}_{N}=\big(
\{\mathbf{Eval}_{N}^{hidden,(\ell,r)}\}_{\ell=1}^{L-1}\!{}_{r=0}^{K_{hid}^{(\ell)}-1},\ \{\mathbf{Eval}_{N}^{concat,\ell}\}_{\ell=1}^{L-1},\
\{\mathbf{Eval}_{N}^{out,s}\}_{s=0}^{K_{out}-1},\
\mathbf{Eval}_{N}^{avg}
\big).$

其中，对任意隐藏层 $\ell$、任意 hidden 头 $r$：


$\mathbf{Eval}_{N}^{hidden,(\ell,r)}=\big(
\operatorname{ev}_{N}(P_{E_{src}^{(\ell,r)}}),\ \operatorname{ev}_{N}(P_{E_{dst}^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{H^{\star(\ell,r)}}),\ \operatorname{ev}_{N}(P_{Table^{src(\ell,r)}}),\
\operatorname{ev}_{N}(P_{m_{src}^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{R_{src}^{node(\ell,r)}}),\ \operatorname{ev}_{N}(P_{M^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{Sum^{(\ell,r)}}),\ \operatorname{ev}_{N}(P_{inv^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{H_{agg,pre}^{\star(\ell,r)}}),\
\operatorname{ev}_{N}(P_{T_{psq}^{(\ell,r)}}),\ \operatorname{ev}_{N}(P_{H_{agg}^{\star(\ell,r)}}),\
\operatorname{ev}_{N}(P_{Table^{dst(\ell,r)}}),\ \operatorname{ev}_{N}(P_{m_{dst}^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{R_{dst}^{node(\ell,r)}})
\big).$

对任意隐藏层 $\ell$：


$\mathbf{Eval}_{N}^{concat,\ell}=\big(
\operatorname{ev}_{N}(P_{H_{cat}^{\star(\ell)}})
\big).$

对任意输出头 $s$：


$\mathbf{Eval}_{N}^{out,s}=\big(
\operatorname{ev}_{N}(P_{E_{src}^{(out,s)}}),\ \operatorname{ev}_{N}(P_{E_{dst}^{(out,s)}}),\
\operatorname{ev}_{N}(P_{Table^{src(out,s)}}),\ \operatorname{ev}_{N}(P_{m_{src}^{(out,s)}}),\
\operatorname{ev}_{N}(P_{R_{src}^{node(out,s)}}),\
\operatorname{ev}_{N}(P_{M^{(out,s)}}),\
\operatorname{ev}_{N}(P_{Sum^{(out,s)}}),\
\operatorname{ev}_{N}(P_{inv^{(out,s)}}),\
\operatorname{ev}_{N}(P_{Y'^{\star(s)}}),\
\operatorname{ev}_{N}(P_{Y^{pre,\star(s)}}),\
\operatorname{ev}_{N}(P_{T_{out}^{pre(s)}}),\
\operatorname{ev}_{N}(P_{\widetilde Y^{pre,\star(s)}}),\ \operatorname{ev}_{N}(P_{Y^{\star(s)}}),\
\operatorname{ev}_{N}(P_{Table^{dst(out,s)}}),\ \operatorname{ev}_{N}(P_{m_{dst}^{(out,s)}}),\
\operatorname{ev}_{N}(P_{R_{dst}^{node(out,s)}})
\big).$

最终平均输出在节点域上的唯一块固定为：


$\mathbf{Eval}_{N}^{avg}=\big(
\operatorname{ev}_{N}(P_{Y^{\star}})
\big).$

上述 $\mathbf{Eval}_{FH},\mathbf{Eval}_{edge},\mathbf{Eval}_{in},\mathbf{Eval}_{d_h},\mathbf{Eval}_{cat},\mathbf{Eval}_{C},\mathbf{Eval}_{N}$ 的块数、块顺序与块内字段，构成本文协议对 $\mathbf{Eval}_{dom}$ 的**唯一完整枚举**。serializer、parser、prover 与 verifier 都必须严格按该枚举实现；不得增加、删除、换序或合并任何域内点评值块。
#### 3.5.6 商承诺块

$\mathbf{Com}_{quot}$ 按七个工作域固定为：

$\mathbf{Com}_{quot}=([t_{FH}],[t_{edge}],[t_{in}],[t_{d_h}],[t_{cat}],[t_C],[t_N])$

其中与输出层 family 和平均输出相关的全部约束，分别已经被吸收到 $t_{edge},t_C,t_N$ 中。

#### 3.5.7 域内 opening 块

$\mathbf{Open}_{dom}$ 按七个工作域固定为：

$\mathbf{Open}_{dom}=(Open_{FH},Open_{edge},Open_{in},Open_{d_h},Open_{cat},Open_C,Open_N)$

各工作域 opening 的对象归属规则：

为避免实现时误把对象放入错误工作域，固定如下归属：

1. $Open_{FH}$：

   只包含特征检索子系统对象，即$[P_H],[P_{Table^{feat}}],[P_{Query^{feat}}],[P_{m_{feat}}],[P_{Q_{tbl}^{feat}}],[P_{Q_{qry}^{feat}}],[P_{R_{feat}}]$及其对应 quotient 约束点评值。

2. $Open_{edge}$：
   
   包含全部边域对象，尤其是 hidden family 与 output family 中所有source / target route 查询列、LeakyReLU / range / exp lookup 列、$S,Z,M^{edge},\Delta^+,s_{max},C_{max},U,Sum^{edge},inv^{edge},\alpha$、PSQ 状态机与其边域目标值，以及$Y'^{\star,edge(s)},Y^{pre,\star,edge(s)},Y^{\star,edge(s)},Y^{\star,edge}$。
   
3. $Open_{in}$：
   
   包含全部定义在 $\mathbb H_{in}$ 上的 projection 折叠向量、共享维累加器，以及$P_{Q_{in}^{valid,(\ell)}}$ family。

4. $Open_{d_h}$：
   
   包含全部定义在 $\mathbb H_{d_h}$ 上的 hidden-head 绑定折叠向量、共享维累加器，以及$P_{Q_{d_h}^{valid,(\ell)}}$ family。

5. $Open_{cat}$：
   
   包含 hidden concat 相关对象、输出层输入共享维相关对象，以及$P_{Q_{cat}^{valid,(\ell)}}$ family 与$P_{Q_{cat}^{valid,(out)}}$。
   
6. $Open_C$：

	包含全部定义在 $\mathbb H_C$ 上的输出层类别维绑定对象、类别枚举对象，以及最终 average 所需的类别维绑定接口对象。

7. $Open_N$：
   
   包含全部节点域对象，尤其是$P_I$（本地重建并吸入 transcript）、全部$E_{src},E_{dst},M,Sum,inv,H^{\star},H_{agg,pre}^{\star},H_{agg}^{\star},H_{cat}^{\star},Y'^{\star},Y^{pre,\star},\widetilde Y^{pre,\star},Y^{\star}$，各类 node-route 累加器，以及最终输出$P_{Y^{\star}}$。

$\mathbf{Eval}_{dom}$ 与 $\mathbf{Open}_{dom}$ 的内部顺序必须逐域一致；任何实现都不得以“该对象可由别的域对象推出”为由省略本域 opening。

#### 3.5.8 外点评值折叠见证

$W_{ext}$ 是按挑战 $\rho_{ext}$ 折叠后的统一外点 opening 见证。

#### 3.5.9 张量绑定子证明块

$\Pi_{bind}$ 的固定顺序为：


$\Pi_{bind}= \big( \pi_{bind}^{feat},\ \{\pi_{bind}^{hidden,(\ell,r)}\}_{\ell,r\ \text{按层序与头序}},\ \{\pi_{bind}^{concat,(\ell)}\}_{\ell=1}^{L-1},\ \pi_{bind}^{out,0},\ldots,\pi_{bind}^{out,K_{out}-1},\ \pi_{bind}^{out,avg} \big)$

其中：

- $\pi_{bind}^{hidden,(\ell,r)}$ 只绑定第 $\ell$ 个隐藏层中第 $r$ 个 头 的内部对象；
- $\pi_{bind}^{concat,(\ell)}$ 只绑定第 $\ell$ 个隐藏层的拼接对象；
- $\pi_{bind}^{out,s}$ 只绑定第 $s$ 个输出头内部对象；
- $\pi_{bind}^{out,avg}$ 只绑定最终平均输出 $Y$ 与各 $Y^{(s)}$ 的关系；
- 任何实现都不得把 $\pi_{bind}^{out,avg}$ 插入到某个单独 hidden / concat / output-head 子证明之前。

#### 3.5.10 Serializer / Parser 的固定字节级规则

为消除实现歧义，proof object 的字节级编码固定遵循下列规则；这些规则属于正式协议的一部分，而不是实现建议。

1. 所有整数统一按无符号 little-endian 编码。

2. 所有域元素统一编码成固定长度 `field_bytes`，其长度由 `encoding` 唯一确定；同一 proof 中不得混用不同长度的域元素编码。

3. 所有曲线点统一采用 compressed encoding，具体编码格式由 `encoding` 唯一确定；serializer 与 parser 不得一端使用 compressed、另一端使用 uncompressed。

4. 所有 variable-length 向量、family 块、子证明列表，都必须显式写入长度前缀；parser 不得通过“读到下一个 tag”来猜测边界。

5. 对所有 hidden family、concat family、output family，必须显式携带：
   - `layer_id`
   - `head_id`（若适用）
   即使协议顺序已经固定，也不得依赖“隐式位置下标”来恢复对象身份。

6. `M_pub`、`Com_dyn`、`S_route`、`Eval_ext`、`Eval_dom`、`Com_quot`、`Open_dom`、`W_ext`、`Pi_bind` 这九个顶层块，必须各自携带固定 tag；tag 的字节串取值由 `encoding` 唯一确定。

7. parser 若遇到下列任一情形，必须直接拒绝该 proof，而不是“容错继续解析”：
   - 未声明字段；
   - 顶层块缺失；
   - family 长度与 `dim_cfg` 不一致；
   - 对象顺序与第 3.5 节固定顺序不一致；
   - 出现多余对象；
   - 出现重复 tag。

8. verifier replay transcript 时，吸入顺序必须与 serializer 输出顺序逐字节一致；不得使用“按对象类型重新分组后再吸入”的实现。

9. 本文前五章中凡出现 `固定顺序`、`唯一顺序`、`完全枚举` 的地方，若与某份代码实现不一致，应以本文笔记为准修正代码，而不是反过来修改 parser 逻辑。

## 4. 验证

验证算法输入为：

- 参数生成输出：$(VK_{KZG},VK_{static},VK_{model},srs\_id,srs\_type,srs\_degree\_bound,privacy\_policy)$
- 公共输入：$(I,src,dst,G_{batch},node\_ptr,edge\_ptr,N,E,N_{total},L,\{K_{hid}^{(\ell)}\}_{\ell=1}^{L-1},\{d_{in}^{(\ell)}\}_{\ell=1}^{L},\{d_h^{(\ell)}\}_{\ell=1}^{L-1},\{d_{cat}^{(\ell)}\}_{\ell=1}^{L-1},C,B,K_{out})$
- 证明对象：$\pi_{GAT}$

验证者按下列步骤执行。

### 4.1 元数据与对象完整性检查

1. 解析 $M_{pub}$，逐项检查 $protocol\_id$、$proof\_task$、$srs\_id$、$srs\_type$、$srs\_degree\_bound$、$privacy\_policy$、$model\_arch\_id$、$model\_param\_id$、$static\_table\_id$、$quant\_cfg\_id$、$domain\_cfg$、$dim\_cfg$、$encoding\_id$、$padding\_rule\_id$、$degree\_bound\_id$ 与本地期望完全一致；不得只检查其中子集。
2. 无条件检查 $node\_ptr,edge\_ptr$ 与公开图块边界的一致性，并本地重建 $edge\_gid,node\_gid,Q_{graph\_new}^{edge},Q_{graph\_end}^{edge},Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$。当 $G_{batch}=1$ 时，这些 batch-aware 公共拓扑对象按单图退化公式唯一确定；验证者仍然必须显式重建，不得省略。
3. 检查 $\mathbf{Com}_{dyn}$、$\mathbf{S}_{route}$、$\mathbf{Eval}_{ext}$、$\mathbf{Eval}_{dom}$、$\mathbf{Com}_{quot}$、$\mathbf{Open}_{dom}$、$W_{ext}$、$\Pi_{bind}$ 的对象数量与顺序是否与第 3.5 节完全一致。
4. 若 parser 读到任何未声明字段、重复字段、缺失字段、长度前缀不一致、family 个数与 `dim_cfg` 不一致，或顶层块 tag 与第 3.5.10 节不一致，则验证必须立即拒绝，不得继续做 opening 验证。

### 4.2 重放 Fiat–Shamir transcript

验证者必须按第 3.2.1 节的固定顺序重放全部挑战，包括：

- 全部隐藏层 family 的挑战；
- 每个隐藏层各自的拼接阶段挑战；
- 全部输出头挑战；
- 最终平均输出压缩挑战 $y_{avg}$；
- 商聚合挑战、各工作域 opening 折叠挑战、外点评值折叠挑战。


### 4.3 域内 opening 验证

对七个工作域 $FH,edge,in,d_h,cat,C,N$ 的所有 batch opening 逐一验证，确保：

- 特征检索对象、全部隐藏层对象、各隐藏层拼接对象、每个输出头对象、最终平均输出对象，全部在对应工作域上正确开放；
- 对每个隐藏层 $\ell$，hidden family 的对象数量恰好为 $K_{hid}^{(\ell)}$；
- 输出头 family 的对象数量恰好为 $K_{out}$；
- 每个隐藏层的拼接对象都只出现一次，且位于该层全部 hidden-head family 之后；
- 最终平均输出对象仅出现一次，且位于全部输出头 family 之后。


### 4.4 外点评值折叠 opening 验证

对 $\mathbf{Eval}_{ext}$ 执行统一折叠 opening 验证，确保：

- $\mathbf{Eval}_{ext}$ 的块数、块顺序与块内字段，必须与第 3.5.4 节给出的完整枚举完全一致；多一个、少一个、换序、插入未枚举字段，验证都必须立即拒绝；
- 每个隐藏层 hidden-head 外点评值块都与其对应的 $(\ell,r)$ 身份一致；
- 每个输出头的投影、源注意力、目标注意力、聚合前压缩输出、尺度对齐后聚合前压缩输出、加偏置后输出压缩外点评值均正确；
- 最终平均输出的 $P_{Y^{\star}}(y_{avg})$ 与 $P_Y(y_{avg})$ 正确。

### 4.5 quotient 身份检查

验证者在对应工作域点评上检查：

- 特征检索域约束；
- hidden head family 的全部边域 / 节点域 / 共享维域约束；
- 拼接域约束；
- 输出头 family 的全部投影、源 / 目标注意力、LeakyReLU、最大值、范围检查、指数、PSQ、目标路由、逆元、压缩绑定约束；
- 最终平均输出约束 $C_{out,avg}(z_N)=0$。

其中，最终平均输出约束是：

$K_{out}P_{Y^{\star}}(z_N)-\sum_{s=0}^{K_{out}-1}P_{Y^{\star(s)}}(z_N)=0$。

### 4.6 张量绑定子证明检查

验证者必须把 $\Pi_{bind}$ 当作严格 schema 块来解析与检查，而不是“收集若干绑定证明后逐个尝试验证”。正式要求如下：

1. $\Pi_{bind}$ 的块数、块顺序与块身份，必须与第 0.7.8 节和第 3.5.9 节给出的固定顺序**完全一致**；多一个、少一个、换序、重复、插入未枚举块，验证都必须立即拒绝。

2. 验证者必须按如下唯一顺序检查：

   - $\pi_{bind}^{feat}$；
   - 全部 $\pi_{bind}^{hidden,(\ell,r)}$，其中层索引按 $\ell=1,\ldots,L-1$ 递增，且每层内部头索引按 $r=0,\ldots,K_{hid}^{(\ell)}-1$ 递增；
   - 全部 $\pi_{bind}^{concat,(\ell)}$，其中 $\ell=1,\ldots,L-1$；
   - $\pi_{bind}^{out,0},\ldots,\pi_{bind}^{out,K_{out}-1}$；
   - $\pi_{bind}^{out,avg}$。

3. 对每个 $\pi_{bind}^{hidden,(\ell,r)}$、$\pi_{bind}^{concat,(\ell)}$ 与 $\pi_{bind}^{out,s}$，其内部身份标签必须与所在块位置完全一致；parser 不得接受“块位置属于 $(\ell,r)$，但块内部标签声称是 $(\ell',r')$”之类的不一致对象。

4. $\pi_{bind}^{out,avg}$ 必须位于 $\Pi_{bind}$ 的最后一个位置，并且必须把最终输出 $Y$ 与各输出头 $Y^{(s)}$ 的算术平均关系绑定到同一 transcript；不得把 $\pi_{bind}^{out,avg}$ 插入到任何 hidden / concat / output-head 子证明之前。

5. 所有张量绑定子证明都必须通过各自的内部检查；任一子证明失败，或任一子证明与其对应对象身份不一致，验证都必须立即拒绝。
### 4.7 验证结论

若以上检查全部通过，则接受证明；否则拒绝。

验证者还必须执行 proof leakage audit gate：proof object 中不得出现未被 `privacy_policy` 声明为 public 的裸模型参数、裸中间激活、裸注意力分数、裸 softmax 权重、裸 lookup witness 或裸私有多项式随机点评值；否则直接拒绝。


## 5. 安全模型与证明目标

### 5.1 参与方与攻击者

证明者 $\mathcal P$ 是 GAT 推理服务端，持有模型参数、模型验证键对应的承诺开口信息以及一次推理中的所有中间 witness。验证者 $\mathcal V$ 是客户端或审计方，持有公开输入、$VK_{KZG}$、$VK_{static}$、$VK_{model}$、公开元数据和最终输出声明。

攻击者是恶意证明者，可以尝试：替换模型参数、篡改节点特征检索、跳过或修改注意力 softmax、伪造 lookup、伪造路由、伪造 PSQ、篡改输出头平均、改变 public metadata、重排承诺块、复用其他证明的 opening 或构造与最终 logits 不一致的 proof。

### 5.2 公开输入与私有见证

公开输入包括：图拓扑与 batch 边界、节点索引 $I$、$src,dst,node\_ptr,edge\_ptr$、任务和结构元数据、量化配置、静态表标识、模型结构标识、SRS 标识、公开输出策略以及最终声明输出。若实验选择公开 logits，则 $Y$ 为公开输出；若只公开 label，则 $Y$ 为私有 witness，必须额外证明 label 与 $Y$ 的 argmax 关系。

私有见证包括：模型参数开口、隐藏层和输出层全部投影结果、注意力打分、lookup query、lookup multiplicity、路由辅助列、Softmax 中间量、PSQ 状态、聚合前后激活、输出头中间输出以及所有 private polynomial 的 blinding randomness。

### 5.3 完备性

若证明者按本文定义的多层多头 GAT 前向、公开预处理、量化配置、lookup 表、路由规则、PSQ 状态机和 KZG SRS 生成证明，则验证者接受。

### 5.4 可靠性 / 知识可靠性

在 KZG binding 假设、Fiat--Shamir 随机预言机模型以及所有 lookup / route / quotient identity 正确构造的条件下，任何多项式时间恶意证明者若能使验证者接受，则其承诺对象必须与某个满足本文 GAT 前向关系的 witness 一致。该 witness 的最终输出必须等于 proof 中声明的输出。

### 5.5 零知识与泄露边界

在 trusted/MPC SRS 的隐藏陷门假设、private polynomial domain-vanishing blinding 和 masked opening 规则下，证明对象除公开元数据、公开图输入、公开输出和必要承诺外，不泄露模型参数或中间推理状态。证明对象中所有 private polynomial 的随机点评值均以 masked evaluation 出现；裸 witness 值不得序列化。

本文不默认隐藏图结构和节点特征。若需要隐藏图结构或节点特征，必须另行定义私有输入承诺、私有拓扑 lookup、私有特征 opening 和相应的泄露模型，不属于当前主协议。

### 5.6 SRS 假设

主实验只接受 `trusted_srs` 或 `mpc_srs`。任何 `seeded_tau`、`deterministic_tau`、`test_tau` 或证明者可见陷门模式都不得出现在最终主实验分支。开发阶段若曾经使用测试 SRS，只能保留在私有备份分支，不能进入最终提交代码、benchmark runner、paper experiment runner 或结果目录。


### 5.7 定理式安全声明

**定理 1（完备性）.** 若证明者按公开元数据、公开图拓扑、量化配置、模型承诺、静态表承诺和 GAT 前向语义诚实生成所有 trace、blinded polynomials、quotient polynomials 与 KZG openings，则验证者接受。

证明思路：所有 witness 多项式在各自工作域上满足第 3.3 节的商多项式恒等式；domain-vanishing blinding 不改变工作域内取值；KZG opening 对承诺和 claimed values 完备；Fiat--Shamir challenge 由相同 transcript 重放得到，因此所有 gate 均通过。

**定理 2（知识可靠性 / 计算可靠性）.** 在 KZG commitment binding、所用曲线离散对数困难、Fiat--Shamir 随机预言机模型和 SRS 隐藏陷门假设下，任意多项式时间恶意证明者若能以非忽略概率让验证者接受，则存在一组与公开承诺一致的 witness 多项式，使其满足完整 zkGAT 前向关系、lookup 关系、route 关系、PSQ 状态机关系、输出平均关系与公开输出绑定关系。

证明思路：KZG binding 固定每个承诺对应的低次数多项式；opening soundness 保证 claimed masked evaluations 与承诺一致；quotient identities 在随机点评价成立则除非高次数差异多项式在随机点偶然为零，否则其在工作域上满足约束；schema gate 固定层、头、对象顺序和公共元数据；因此通过验证等价于存在满足完整前向关系的 witness。

**定理 3（零知识 / witness 隐私）.** 对所有标记为 `PrivateWitness` 的模型参数、中间激活、注意力分数、softmax 权重、lookup query、route query、PSQ 状态机和折叠向量，若其承诺与 opening 均按 `privacy_policy` 进行 domain-vanishing blinding，并且 proof object 只输出 masked evaluations 与零知识子证明，则验证者除公开元数据、公开拓扑、公开输出、承诺和允许泄露的大小信息外，不获得关于私有 witness 的额外信息。

证明思路：在工作域外，$Z_{\mathcal D}(z)R_P(z)$ 对每个 private polynomial 提供独立随机掩蔽；proof payload 不包含裸 private evaluations；所有绑定关系通过 quotient、masked correction 或 ZK subproof 验证；因此可由模拟器在给定公开输入和公开输出的情况下模拟 masked evaluations 与承诺分布。完整形式化证明依赖底层 KZG hiding variant 或 masked-opening 策略的模拟安全性。

**泄露边界.** 本协议默认泄露并允许论文报告以下信息：图规模 $(N,E,G_{batch})$、公开拓扑或其承诺、模型结构 ID、量化配置 ID、静态表 ID、SRS ID、proof size、运行时间、最终公开 logits 或 label。若某次应用要求隐藏图结构、节点特征或 logits，则必须定义新的 `proof_task` 和 `privacy_policy`，不能复用主实验安全声明。


## 6. 实现与实验合规要求

### 6.1 主实验代码要求

1. 主分支中不得存在 `seed -> tau` 主路径；
2. prover/verifier 数据结构不得保存 $\tau$；
3. KZG commitment 必须基于 SRS powers 或 Lagrange-basis SRS；
4. private polynomial 必须使用 $P^{zk}=P+Z_{\mathcal D}R_P$；
5. proof serializer 必须区分 public value 与 masked value；
6. verifier 必须执行 metadata gate、schema gate、opening gate、quotient gate、binding gate 和 leakage audit gate；
7. benchmark 结果必须来自 secure 主路径。

### 6.2 必需实验

主实验必须重新运行五个节点分类数据集：Cora、Citeseer、Pubmed、PPI、ogbn-arxiv。每个数据集至少报告 CPU full 与 GPU full 的 commitment time、prove time、verify time、proof size。若开启 masked opening 和 secure SRS 后结果变化，论文只能使用新结果。

安全实验必须包括：tamper rejection、proof leakage audit、SRS metadata mismatch rejection、proof-scope mismatch rejection、commitment order tamper、opening value tamper、challenge tamper、lookup query tamper、route relation tamper、PSQ state tamper 和 final logits tamper。

对比实验必须额外给出 proof-scope 对齐表，说明 ezkl、zkDL、zkGPT 是否证明 full GAT forward、是否绑定 final logits、是否使用 full dataset、verifier 是否读取 witness、proof size 统计口径和 setup assumption。



### 6.3 正确性一致性测试

secure 主路径重构后，所有数据集必须执行 reference consistency test：

1. `reference_fp32_forward`：原始模型前向；
2. `reference_quant_forward`：量化语义前向；
3. `secure_trace_forward`：证明系统 trace 前向；
4. `proof_public_output`：proof object 中公开输出。

验收条件：

- `reference_quant_forward_hash == secure_trace_forward_hash`；
- 若公开 logits，则 `secure_trace_forward_hash == proof_public_output_hash`；
- 若只公开 label，则必须额外验证 argmax proof；
- 所有 hash、配置 ID 和数据集 ID 必须进入 run manifest。

### 6.4 最终论文实验验收标准

最终论文主表只能读取满足以下条件的 run：

1. `protocol_name == zkgat`；
2. `srs_type in {trusted_srs, mpc_srs}`；
3. `srs_id` 与本地 SRS manifest 一致；
4. `privacy_policy` 启用 private polynomial blinding；
5. `private_plain_value_count == 0`；
6. `verifier_reads_witness == false`；
7. `tamper_tests_passed == true`；
8. `reference_consistency_passed == true`；
9. `node_classification_full_gat == true`；
10. proof size 只统计 proof payload，不统计 SRS、manifest、日志、路径和时间戳。

未满足上述任意条件的 run 只能作为开发诊断材料，不能进入论文正文、附录或最终结果表。


### 6.5 论文结果来源锁定

所有进入论文正文、附录和最终结果表的实验结果，必须来自同一个最终主实验分支，并统一导出到 `results/paper_secure/`。

每条结果至少包含以下字段：`protocol_name`、`srs_type`、`srs_id`、`privacy_policy`、`input_privacy`、`output_mode`、`opening_mode`、`binding_mode`、`private_plain_value_count`、`reference_consistency_passed`、`tamper_tests_passed`、`verifier_reads_witness`。

只有满足以下条件的 run 才能进入论文表格：

1. `srs_type` 是 `trusted_srs` 或 `mpc_srs`；
2. `srs_id` 与本地 SRS manifest 一致；
3. `privacy_policy == model_and_trace_hidden`；
4. `input_privacy == public_graph_public_features`；
5. `output_mode == public_logits`；
6. `opening_mode == domain_grouped_batch_opening`；
7. `binding_mode == zk_subproof_for_crpc`；
8. `private_plain_value_count == 0`；
9. `verifier_reads_witness == false`；
10. `reference_consistency_passed == true`；
11. `tamper_tests_passed == true`；
12. proof size 只统计 proof payload，不统计 SRS、manifest、日志、路径、时间戳和调试输出。

不满足任意条件的运行结果只能作为开发诊断材料，不能进入论文。
