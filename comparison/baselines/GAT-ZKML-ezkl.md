# GAT-ZKML

## 0. 符号

### 0.1 模型结构

GAT 的前向固定为一个可参数化的多层结构。本文只替换证明方案，不改变 GAT 的明文前向语义。

1. 设注意力层的总层数为：$L$。其中前 $L-1$ 层是隐藏层，第 $L$ 层是最终输出层。

2. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，记该层的注意力头数量为 $K_{hid}^{(\ell)}$，每个头的输出维度为 $d_h^{(\ell)}$。

	该层每个头完成注意力聚合后，逐元素施加 ELU；

	随后把本层全部头的 ELU 后输出沿特征维进行拼接，得到：$d_{cat}^{(\ell)}=K_{hid}^{(\ell)}\cdot d_h^{(\ell)}$。

3. 第 $\ell$ 个隐藏层的输入维度记为 $d_{in}^{(\ell)}$。其中 $d_{in}^{(1)}=d_{in}$，且对任意 $\ell\ge 2$，都有 $d_{in}^{(\ell)}=d_{cat}^{(\ell-1)}$。

4. 最终输出层（第 $L$ 层）的注意力头数量记为 $K_{out}$。

	输出层不是“仿射线性层 + 偏置”替代注意力，而是由 $K_{out}$ 个输出注意力头组成。每个输出注意力头先完成注意力聚合，再加上该头自己的输出偏置向量；

	若 $K_{out}>1$，则全部输出头的后偏置输出按头维做算术平均，得到最终 logits。

5. 输出注意力头的激活函数固定为恒等映射，不再施加 ELU；但恒等映射的输入是已经加上该输出头偏置后的注意力聚合结果。

6. 本文采用 ezkl 式证明方案。后续不再手写每一层的 KZG 多项式承诺、LogUp、PSQ、路由、一致性绑定约束与商多项式，而是把完整 GAT 前向导出为固定形状的 ONNX 计算图，由 ezkl 将 ONNX 图中的算子编译成 Halo2/zkSNARK 电路约束。

	对任意多层 GAT，明文前向仍然必须按层序 $\ell=1,\ldots,L-1$ 逐层执行；第 $\ell-1$ 层的拼接输出 $H_{cat}^{(\ell-1)}$ 必须作为第 $\ell$ 层的唯一输入表示。证明系统只验证该固定计算图的执行正确性，不改变该执行顺序。

7. 下列基准仅作为非规范实例说明，不构成协议参数默认值；实际实验使用的 $L,\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1},K_{out}$ 必须完全由公开配置 $manifest\_cfg$（用于生成本次证明的 $manifest\_body$ 与 $manifest\_id$ 的配置总包）唯一确定：
   - Cora / Citeseer 风格两层 GAT：$L=2$，在某些基准测试的配置文件中可取 $K_{hid}^{(1)}=8,\ K_{out}=1$；
   - Pubmed 风格两层 GAT：$L=2$；若某次实验把它设为两层多输出头实例，则对应的 $K_{out}>1$ 也必须由该次实验的 $manifest\_cfg$ 唯一固定；
   - PPI 风格多层 GAT：$L=3$ 或更深，仍保持“隐藏层拼接、最终层平均”的官方语义。

为避免公式中引入过多层指标，除非特别声明，隐藏层模板统一采用如下局部简写：

- 隐藏层模板索引：$\ell\in\{1,2,\ldots,L-1\}$
- 第 $\ell$ 个隐藏层中第 $r$ 个注意力头的编号：$r\in\{0,1,\ldots,K_{hid}-1\}$
- 当前隐藏层模板的输入维度：$d_{in}:=d_{in}^{(\ell)}$
- 当前隐藏层模板的每头输出维度：$d_h:=d_h^{(\ell)}$
- 当前隐藏层模板的头数：$K_{hid}:=K_{hid}^{(\ell)}$
- 当前隐藏层模板的拼接维：$d_{cat}:=d_{cat}^{(\ell)}=K_{hid}\cdot d_h$
- 输出类别数：$C$
- 最终输出层第 $s$ 个输出头编号：$s\in\{0,1,\ldots,K_{out}-1\}$

换言之，后文若出现 $H_{cat}$、$d_h$、$K_{hid}$ 等未显式带层指标的记号，在隐藏层上下文中一律解释为“当前正在实例化的那一层”的数学局部记号；代码实现中的 ONNX 输入名、张量名、运行日志字段名、proof artifact 字段名与 attestation manifest 字段名，一律必须显式带层指标或可唯一反推出层指标，不得把这些局部记号直接搬入实现。

同理，凡正文中为了公式紧凑而省略的 $(\ell)$、$(\ell,r)$、$(out,s)$ 标记，只能在数学推导中省略；在任何实现接口、序列化字段、模型导出字段、公开元数据字段与验证脚本字段中，都必须恢复完整指标。

### 0.2 有限域、固定点编码与 Halo2 后端

记有限域为 $\mathbb F_p$，其中 $p$ 为大素数。ezkl 后端在 Halo2/Plonkish 证明系统中把固定点张量值编码为域元素；所有电路约束中的算术均在 $\mathbb F_p$ 上执行。

为避免实数、整数和域元素之间出现语义混用，本文固定三层表示：

1. 实数张量：$X\in\mathbb R^{r\times c}$；
2. 定点整数张量：$\widetilde X\in\mathbb Z^{r\times c}$；
3. 域内编码张量：$\langle \widetilde X\rangle_p\in\mathbb F_p^{r\times c}$。

对任意实数标量 $x$ 和公开尺度 $S_X>0$，量化、反量化和域编码分别定义为：

$Q_X(x)=\operatorname{Clip}_{[q_{min},q_{max}]}\big(\operatorname{Round}(S_X\cdot x)\big)$，

$DQ_X(\widetilde x)=\widetilde x/S_X$，

$Enc_p(\widetilde x)=\begin{cases}\widetilde x,&\widetilde x\ge 0,\\ p+\widetilde x,&\widetilde x<0.\end{cases}$

其中 $q_{min},q_{max}$、$\operatorname{Round}$ 的具体舍入方式、clip 边界、溢出处理和符号解释均由 $quant\_cfg\_id$ 唯一固定。反向解释时，若域元素 $z\in\mathbb F_p$ 的标准代表满足 $z>\frac{p-1}{2}$，则解释为整数 $z-p$；否则解释为整数 $z$。因此：

$Dec_p(z)=\begin{cases}z,&0\le z\le\frac{p-1}{2},\\z-p,&\frac{p-1}{2}<z<p.\end{cases}$

浮点模型 $f_W$ 进入证明前必须转换为固定点模型 $\widetilde f_{\widetilde W}$。对任意张量运算 $T=A\star B$，电路中实际验证的是其定点版本：

$\widetilde T=Q_T\big(DQ_A(\widetilde A)\star DQ_B(\widetilde B)\big)$，

或等价地验证由 $settings.json$ 指定的整数重缩放关系：

$\widetilde T=\operatorname{Rescale}_{A,B\to T}(\widetilde A,\widetilde B;quant\_cfg\_id)$。

LeakyReLU、ELU、Softmax 中的指数、除法 / 逆元、范围比较与量化回缩均不再由本文手写查找表证明，而是由 ezkl 根据 ONNX 算子、settings 配置和 Halo2 lookup / arithmetic gadget 自动生成对应约束。若某个非线性算子被实现为查找表，则该查找表属于 ezkl 编译电路的一部分，其版本必须由 $settings\_id$ 与 $compiled\_circuit\_id$ 共同绑定。

记底层证明系统使用的公共结构化参考串为 $SRS$。若后端使用 KZG 多项式承诺，则 $SRS$ 必须来自公开可信设置或协议指定的公共参数；证明者不得自行采样 KZG 陷门 $\tau$，也不得把自采样陷门作为安全假设的一部分。本文不再把 KZG opening、商多项式或 Fiat-Shamir 挑战顺序作为手写协议暴露给验证者。

随机预言机统一写作 $H_{FS}(\cdot)$。Fiat-Shamir transcript 由 ezkl/Halo2 证明系统内部确定；本文只要求实现层保证 $model.onnx$、$settings.json$、编译电路摘要、验证键、公开输入、公开输出、$H(W)$ 和 $manifest\_id$ 的编码顺序由 $encoding\_id$ 唯一确定，避免不同实现产生不同 transcript 语义。

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

对任意节点索引 $i\in\{0,1,\ldots,N-1\}$，定义其图编号 $node\_gid(i)=g$，其中 $g\in\{0,1,\ldots,G_{batch}-1\}$ 满足 $\nu_g\le i<\nu_{g+1}$。

对任意边索引 $k\in\{0,1,\ldots,E-1\}$，定义其图编号 $edge\_gid(k)=g$，其中 $g\in\{0,1,\ldots,G_{batch}-1\}$ 满足 $\epsilon_g\le k<\epsilon_{g+1}$。

因此，对任意边索引 $k$，都有：

$src(k),dst(k)\in\{0,1,\ldots,N-1\},\quad node\_gid(src(k))=edge\_gid(k),\quad node\_gid(dst(k))=edge\_gid(k)$。

若 $G_{batch}=1$，则本文的公共输入仍然必须显式写成 $node\_ptr=(0,N),\quad edge\_ptr=(0,E)$，不得省略，也不得以其他记号替代。

固定要求边序列满足以下两级稳定排序：

1. 先按 $edge\_gid(k)$ 非降序排列，使不同图的边在公开边序列中保持连续块；
2. 在每个固定图块内部，再按目标节点索引 $dst(k)$ 做确定性稳定排序。

等价地，若 $0\le k_1<k_2\le E-1$，则必须满足：

$edge\_gid(k_1)<edge\_gid(k_2)$ 或 $(edge\_gid(k_1)=edge\_gid(k_2)$ 且 $dst(k_1)\le dst(k_2)$。

若原始输入不满足这一条件，则在见证生成前先做：

- 图内节点重编号 / 局部子图抽取；
- 按图分块的稳定排序；
- 每个图块内部按 $dst$ 的稳定排序；
- 必要时加入自环，使每个有效节点至少有一条入边。

验证者只接收已经按此规范生成的公开序列。除非某条公式显式写出“按图”量或“图局部”量，否则后文中的 $N,E$ 一律指当前证明实例中的总节点数与总边数；在多图批处理情形下，它们分别等于所有图的节点数与边数总和。

在 ezkl 方案中，上述排序不再用于手写边域状态机，而是用于保证 ONNX 输入张量、witness 文件、公开输入哈希与验证脚本之间的确定性一致。任何改变边顺序的实现都必须同步改变 $manifest$ 与输入张量摘要；否则视为不同证明实例。

### 0.4 编译工件与量化尺度

本文不再维护单独的 $T_{LReLU}$、$T_{ELU}$、$T_{exp}$、$T_{range}$ 多项式承诺。ezkl 方案中的静态对象改为以下公开或可哈希绑定的工件：

1. 模型计算图：$model.onnx$

2. ezkl 配置：$settings.json$

3. 编译后电路：$C_{ezkl}$

4. 证明键：$pk$

5. 验证键：$vk$

6. 量化后的模型权重集合：$\widetilde W$，其原文不公开，公开绑定值为模型权重哈希 $H(W)$ 或 $H(\widetilde W)$。本文统一记为 $H(W)$，其精确定义由 $model\_param\_id$ 和 $quant\_cfg\_id$ 固定。

7. 当前证明实例的公开输入张量：$x$，包含节点特征、边索引、图指针、必要 mask 与批处理辅助张量。

8. 当前证明实例的公开输出张量：$y$，通常是最终 logits、类别概率或后处理后的预测结果，具体公开范围由 $visibility\_cfg$ 固定。

9. 数据集标签 $label$。标签不进入单次推理正确性证明，除非采用聚合证明直接证明 accuracy、macro-F1、混淆矩阵或其他评价指标。

与 Softmax 链路有关的量化尺度全部视为公开参数，并由 $settings.json$ 与 $quant\_cfg\_id$ 唯一绑定：

- 输入特征尺度：$S_{in}$
- 权重尺度：$S_W$
- 注意力打分尺度：$S_{score}$
- LeakyReLU 后打分尺度：$S_{lrelu}$
- 指数或 Softmax 近似尺度：$S_{softmax}$
- 归一化注意力权重尺度：$S_{\alpha}$
- 隐藏层聚合前尺度：$S_{agg,pre}$
- 隐藏层 ELU 后尺度：$S_{agg}$
- 拼接后表示尺度：$S_{cat}$
- 输出层每个头的聚合前尺度：$S_{out,pre}$
- 输出层每个头的加偏置后尺度：$S_{out,head}$
- 输出层最终多头平均 logits 尺度：$S_{out}$

舍入方式（向最近整数舍入或截断）、乘法后的重缩放规则、除法或逆元近似规则、非线性函数近似规则、输入 / 输出可见性规则也全部是公开参数。

进一步固定输出层的量化与平均语义如下：

1. 对每个输出头 $s$，聚合前输出 $Y^{pre(s)}$ 在与偏置向量 $b^{(out,s)}$ 相加之前，必须已经被公开量化规则对齐到与 $b^{(out,s)}$ 相同的输出头尺度 $S_{out,head}$。若实现中注意力聚合自然产出的尺度是 $S_{out,pre}$，则必须先施加公开的对齐 / 回缩算子：$\widetilde Y^{pre(s)}=Rescale_{out}(Y^{pre(s)};S_{out,pre}\rightarrow S_{out,head})$。

   然后将输出头内部的偏置相加正式定义为：$Y^{(s)}=\widetilde Y^{pre(s)}+b^{(out,s)}$。

   当且仅当 $quant\_cfg\_id$ 指定 $S_{out,pre}=S_{out,head}$ 时，$Rescale_{out}$ 定义为恒等映射；否则 $Rescale_{out}$ 必须按 $quant\_cfg\_id$ 唯一指定的舍入 / 截断规则执行，不存在第二种解释。

2. 最终多头平均必须在共同尺度上执行。本文固定要求最终平均发生在输出头共同尺度 $S_{out,head}$ 上，并且最终 logits 继续保持这一共同尺度，即 $S_{out}=S_{out,head}$。先计算实数语义下的头和：$Y^{sum}_{i,c}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}$，再定义实数平均：$Y_{i,c}^{avg}=K_{out}^{-1}\cdot Y^{sum}_{i,c}$。

   在定点整数语义下，不直接把有限域逆元默认解释为整数除法，而是定义公开平均量化算子：

   $\widetilde Y_{i,c}=AvgQuant_{K_{out}}\left(\sum_{s=0}^{K_{out}-1}\widetilde Y_{i,c}^{(s)};K_{out},quant\_cfg\_id\right)$。

   $AvgQuant_{K_{out}}$ 的除法、舍入、截断、溢出处理和域编码方式全部由 $quant\_cfg\_id$ 唯一固定。只有当 $quant\_cfg\_id$ 明确声明采用域内精确平均，且输出解释也按域元素解释时，才允许把平均写作 $K_{out}^{-1}\cdot \sum_s\widetilde Y_{i,c}^{(s)}$。

3. 上述 $Rescale_{out}$、舍入方式、$AvgQuant_{K_{out}}$，以及必要时的域内 $K_{out}^{-1}$ 解释，全部必须并入 $quant\_cfg\_id$ 所标识的公开量化配置；证明者与验证者不得各自选择不同的平均语义。

### 0.5 模型参数

#### 0.5.1 隐藏层参数族

对任意隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，以及该层中的每个注意力头 $r\in\{0,1,\ldots,K_{hid}^{(\ell)}-1\}$，定义：

- 投影矩阵 $W^{(\ell,r)}\in\mathbb R^{d_{in}^{(\ell)}\times d_h^{(\ell)}}$
- 源方向注意力向量 $a_{src}^{(\ell,r)}\in\mathbb R^{d_h^{(\ell)}}$
- 目标方向注意力向量 $a_{dst}^{(\ell,r)}\in\mathbb R^{d_h^{(\ell)}}$

量化后分别记为 $\widetilde W^{(\ell,r)}$、$\widetilde a_{src}^{(\ell,r)}$、$\widetilde a_{dst}^{(\ell,r)}$。在电路内部，实际被约束的是量化后的固定点张量。

在后文“单个隐藏层模板”的数学推导中，统一把 $W^{(\ell,r)},a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}$ 记作 $W^{(r)},a_{src}^{(r)},a_{dst}^{(r)}$。这一重写只服务于公式书写；代码实现中的对象名、ONNX initializer 名、proof artifact 字段名以及日志字段名都必须保留完整层指标 $(\ell,r)$。

#### 0.5.2 输出层参数

对最终输出层中的每个输出注意力头 $s\in\{0,1,\ldots,K_{out}-1\}$，定义：

- 输出投影矩阵 $W^{(out,s)}\in\mathbb R^{d_{in}^{(L)}\times C}$
- 输出层源方向注意力向量 $a_{src}^{(out,s)}\in\mathbb R^C$
- 输出层目标方向注意力向量 $a_{dst}^{(out,s)}\in\mathbb R^C$
- 输出层偏置向量 $b^{(out,s)}\in\mathbb R^C$

其中 $d_{in}^{(L)}=d_{cat}^{(L-1)}$，也就是最终输出层接收最后一个隐藏层拼接后的表示。

本文沿用把官方注意力向量按源 / 目标方向分拆的等价写法：$\mathbf a^{(out,s)\top}[u\,\|\,v]=a_{src}^{(out,s)\top}u+a_{dst}^{(out,s)\top}v$。其中 $[u\,\|\,v]$ 表示拼接。这只是代数重写，不改变官方 GAT 的前向语义。

全部模型参数在一次 attestation 中固定。本文不再定义手写 $VK_{model}$；模型参数的公开绑定由以下对象共同完成：

- $model\_arch\_id$：绑定层数、头数、维度、激活、输出头平均语义；
- $model\_param\_id$：绑定量化后权重字节串、权重哈希算法和哈希输出；
- $H(W)$：证明中公开暴露或由聚合证明公开暴露的模型权重哈希；
- $vk$：ezkl 编译后电路的验证键；
- $compiled\_circuit\_id$：编译电路摘要，用于排除“同名但不同约束”的实现替换。

### 0.6 全部中间变量

#### 0.6.1 原始特征

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 与每个输入维索引 $j\in\{0,1,\ldots,d_{in}-1\}$，定义：$H_{i,j}=T_H[I_i,j]$，于是：$H\in\mathbb R^{N\times d_{in}}$。

若实验直接把局部节点特征矩阵作为输入张量，则 $T_H$ 只作为数据集来源，不作为证明系统中的独立查找表。验证者检查的是公开输入张量 $x$ 的哈希和 manifest 中记录的数据集版本，而不是手写特征表 LogUp 关系。

#### 0.6.2 单个隐藏层模板中第 $r$ 个注意力头的节点域与边域变量

以下对象对应某个固定隐藏层 $\ell$ 的单层模板；在该模板中，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$，定义：

1. 投影结果：
$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}$。

2. 节点域源注意力：
$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)}$。

3. 节点域目标注意力：
$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}$。

4. 边域线性打分：
$S_k^{(r)}=E_{src,src(k)}^{(r)}+E_{dst,dst(k)}^{(r)}$。

5. LeakyReLU 后打分：
$Z_k^{(r)}=LReLU(S_k^{(r)})$。

6. 每个目标节点的组最大值：
$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}$。

7. 稳定 Softmax 指数项：
$U_k^{(r)}=\exp(Z_k^{(r)}-M_{dst(k)}^{(r)})$。

   若实现使用非负差分 $\Delta_k^{+(r)}=M_{dst(k)}^{(r)}-Z_k^{(r)}$，则对应指数查找或近似必须被定义为 $U_k^{(r)}=\exp(-\Delta_k^{+(r)})$，不得把 $\exp(\Delta_k^{+(r)})$ 误作为 Softmax 分子。

8. 每个目标节点的 Softmax 分母：
$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}$。

9. 归一化注意力权重：
$\alpha_k^{(r)}=U_k^{(r)}\cdot (Sum_{dst(k)}^{(r)})^{-1}$。

10. 聚合前隐藏表示：
$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}$。

11. ELU 后隐藏表示：
$H_{agg,i,j}^{(r)}=ELU(H_{agg,pre,i,j}^{(r)})$。

上述变量在证明中作为 ezkl witness 的内部赋值或由 witness 生成器隐式生成；它们不再分别编码为外部多项式，也不再分别提交 KZG 承诺。

#### 0.6.3 单个隐藏层模板的拼接阶段变量

定义拼接结果 $H_{cat}\in\mathbb R^{N\times d_{cat}}$，其中对任意 $i\in\{0,1,\ldots,N-1\}$、$r\in\{0,1,\ldots,K_{hid}-1\}$、$j\in\{0,1,\ldots,d_h-1\}$，都有：

$H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}$。

拼接操作必须在 ONNX 图中显式出现，或由导出器生成等价的固定轴 concat 算子。拼接顺序必须与 $model\_arch\_id$ 中的头顺序一致。

#### 0.6.4 输出层变量

输出层有两类共享维：一类是输入共享维 $d_{cat}$，另一类是类别共享维 $C$。

对每个输出头 $s\in\{0,1,\ldots,K_{out}-1\}$，定义：

1. 输出投影结果：
$Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out,s)}$。

2. 输出层源注意力：
$E_{src,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{src,c}^{(out,s)}$。

3. 输出层目标注意力：
$E_{dst,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{dst,c}^{(out,s)}$。

4. 输出层线性打分：
$S_k^{(out,s)}=E_{src,src(k)}^{(out,s)}+E_{dst,dst(k)}^{(out,s)}$。

5. 输出层 LeakyReLU 后打分：
$Z_k^{(out,s)}=LReLU(S_k^{(out,s)})$。

6. 输出层组最大值：
$M_i^{(out,s)}=\max\{Z_k^{(out,s)}\mid dst(k)=i\}$。

7. 输出层指数项与分母：
$U_k^{(out,s)}=\exp(Z_k^{(out,s)}-M_{dst(k)}^{(out,s)})$，
$Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)}$。

8. 输出层归一化权重：
$\alpha_k^{(out,s)}=U_k^{(out,s)}\cdot (Sum_{dst(k)}^{(out,s)})^{-1}$。

9. 输出层聚合前输出：
$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}$。

10. 尺度对齐后的聚合前输出：
$\widetilde Y_{i,c}^{pre(s)}=Rescale_{out}(Y_{i,c}^{pre(s)};S_{out,pre}\rightarrow S_{out,head})$。

11. 输出层加偏置后的头输出：
$Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}$。

12. 最终输出：
$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}$，
$Y_{i,c}^{avg}=K_{out}^{-1}\cdot Y_{i,c}^{sum}$。

定点证明中最终输出记为：

$\widetilde Y_{i,c}=AvgQuant_{K_{out}}\left(\sum_{s=0}^{K_{out}-1}\widetilde Y_{i,c}^{(s)};K_{out},quant\_cfg\_id\right)$。

若 $quant\_cfg\_id$ 明确采用域内精确平均，则可把上式等价实现为 $\widetilde Y_{i,c}=K_{out}^{-1}\cdot\sum_s\widetilde Y_{i,c}^{(s)}$；否则必须使用 $AvgQuant_{K_{out}}$ 指定的整数除法与舍入规则。

最终 GAT 推理函数记为：

$f_W(x)=Y$。

量化后实际被证明的函数记为：

$\widetilde f_{\widetilde W}(\widetilde x)=\widetilde Y$。

### 0.7 ONNX 计算图、公开张量与展平规则

#### 0.7.1 计算图

将完整 GAT 前向导出为 ONNX 计算图：

$G_{onnx}=(\mathcal V,\mathcal E_{onnx},\mathcal O)$。

其中 $\mathcal V$ 是张量节点，$\mathcal E_{onnx}$ 是 ONNX 数据依赖边，$\mathcal O$ 是 ONNX 算子集合。每个 ONNX 算子必须属于当前 ezkl 版本支持的算子集合，或被导出器分解为受支持算子的组合。若原始稀疏 GAT 代码包含 ezkl 不直接支持的 segment-softmax、scatter、gather 或动态 shape 操作，则必须在导出阶段改写为等价的固定形状 ONNX 子图；不得在证明阶段临时改变前向语义。

#### 0.7.2 输入张量

单次证明实例的输入张量记为：

$x=(H,src,dst,node\_ptr,edge\_ptr,mask,\operatorname{aux})$。

其中 $mask$ 用于标识 padding 位置，$\operatorname{aux}$ 包含为了固定形状 ONNX 图所需的公开辅助张量。若某些拓扑张量被固化为 ONNX 常量而不是运行时输入，则其字节串必须进入 $compiled\_circuit\_id$ 或 $manifest\_id$ 的哈希绑定。

#### 0.7.3 量化输入与输出

按 $quant\_cfg\_id$ 对输入和权重量化；证明输出由量化模型或 ezkl witness 生成器产生：

$\widetilde x=Quantize(x;quant\_cfg\_id),\quad \widetilde W=Quantize(W;quant\_cfg\_id)$，

$\widetilde y=\widetilde f_{\widetilde W}(\widetilde x)$。

浮点输出只作为对照量记为 $y^{float}=f_W(x)$；不得把 $Quantize(y^{float};quant\_cfg\_id)$ 默认解释为 proof 输出，除非 $quant\_cfg\_id$ 与导出 / 校准记录明确证明该路径和量化模型路径等价。

对每个公开或私有张量 $T$，统一写作：

$\widetilde T=Q_T(T),\quad \langle\widetilde T\rangle_p=Enc_p(\widetilde T),\quad T^{deq}=DQ_T(\widetilde T)$。

若 $T$ 是输入特征，则 $Q_T$ 使用 $S_{in}$；若 $T$ 是权重，则使用 $S_W$；若 $T$ 是注意力打分、Softmax 权重、隐藏层聚合值或输出 logits，则分别使用第 0.4 节列出的相应尺度。所有尺度、舍入和重缩放规则必须由 $settings.json$ 与 $quant\_cfg\_id$ 固定。

证明语句不是“浮点模型 $f_W(x)=y$ 精确成立”，而是：

$\widetilde f_{\widetilde W}(\widetilde x)=\widetilde y$

在 ezkl 编译电路中成立。实验报告中的准确率、损失或其他指标必须说明使用浮点输出、量化输出，还是量化输出反量化后的结果；不得在同一指标中混用。

为避免输出解释冲突，若公开 logits，则公开值为 $\widetilde y$ 或 $DQ_{out}(\widetilde y)$ 中的一种，并由 $visibility\_cfg$ 固定；若公开类别，则必须由电路或公开后处理脚本固定：

$\widehat c_i=\arg\max_{0\le c<C}\widetilde Y_{i,c}$。

若 $\widehat c_i$ 在 proof 外计算，则验证者只能验证 logits 绑定；若 $\widehat c_i$ 在 proof 内公开，则 verifier 可以直接验证类别输出绑定。

#### 0.7.4 见证

单次推理 witness 分为 ezkl 原生 witness 与外部公开绑定记录两部分。ezkl 原生 witness 记为：

$w_i^{ezkl}=(priv_i,inst_i,trace_i)$。

其中 $inst_i$ 是 ezkl verifier 实际读取的 public instance 向量，$priv_i$ 是不公开的 witness 字段，$trace_i$ 是 witness 生成器产生的中间张量赋值。外部公开绑定记录统一记为：

$public_i=(inst_i,\operatorname{Pub}(\widetilde x_i),\operatorname{Pub}(\widetilde y_i),input\_hash_i,output\_hash_i,H(W),manifest\_id,artifact\_ids_i,binding\_digest_i,visibility\_cfg)$。

其中 $artifact\_ids_i$ 至少包含：

$artifact\_ids_i=(onnx\_model\_id,settings\_id,compiled\_circuit\_id,vk\_id,quant\_cfg\_id,dataset\_id)$。

若需要证明实验可复现性，也可以额外包含 $pk\_id$；但验证者不需要持有 $pk$。输入和输出摘要定义为：

$input\_hash_i=Hash(encoding(\widetilde x_i,visibility\_cfg,encoding\_id))$，

$output\_hash_i=Hash(encoding(\widetilde y_i,visibility\_cfg,encoding\_id))$。

外部公开字段的绑定摘要定义为：

$binding\_digest_i=Hash(encoding(\operatorname{Pub}(\widetilde x_i),\operatorname{Pub}(\widetilde y_i),input\_hash_i,output\_hash_i,H(W),manifest\_id,artifact\_ids_i,visibility\_cfg,encoding\_id))$。

凡是不直接进入 $inst_i$ 的外部公开字段，都必须通过 $binding\_digest_i$ 绑定到 $inst_i$，即要求 $binding\_digest_i\in inst_i$，或在聚合模式下由 $public_{agg}$ 与聚合电路公开绑定；否则该外部公开记录不得作为 proof 的验证依据。

若还需要绑定原始浮点输入或数据集文件，则另设 $raw\_input\_hash_i=Hash(encoding(x_i,dataset\_encoding\_id))$；$raw\_input\_hash_i$ 不得与 $input\_hash_i$ 混用。

权重绑定字段写作 $weight\_binding_i$。若权重作为私有 witness，则：

$priv_i=(\widetilde W,\widetilde x_i^{priv},\widetilde y_i^{priv},trace_i,\ldots),\quad weight\_binding_i=\widetilde W$，

并且电路必须把如下等式作为可验证约束公开绑定：

$H(W)=Hash(model\_param\_encoding(\widetilde W,quant\_cfg\_id,model\_arch\_id))$。

若实现没有在电路内计算或验证该哈希关系，则不能声称采用私有权重 witness 模式；此时只能采用“权重固化为电路常量”的绑定模式。

若权重作为编译期常量固化进电路，则：

$priv_i=(\widetilde x_i^{priv},\widetilde y_i^{priv},trace_i,\ldots)$，

$weight\_binding_i=(compiled\_circuit\_id,model\_param\_id,H(W))$。

这时 witness 中不重复携带权重原文，但验证者必须检查 $compiled\_circuit\_id$、$vk\_id$、$model\_param\_id$ 与公开 $H(W)$ 同时匹配。两种权重绑定方式不得在同一个 attestation 中混用。

本文主实验默认采用私有权重 witness 模式：$\widetilde W$ 作为私有见证参与推理电路，$H(W)$ 作为公开绑定值参与验证；因此实现必须在 proof 内证明私有 $\widetilde W$ 与公开 $H(W)$ 的哈希一致性。若实际 ezkl 工程实现选择把权重固化为电路常量，则该实验必须在 $manifest\_cfg$、$model\_param\_id$ 和实验报告中显式声明为“权重常量模式”，并不得与主实验默认模式混写。

#### 0.7.5 展平规则

对任意矩阵 $M\in\mathbb R^{r\times c}$，定义行优先展平索引：

$\operatorname{flat}_{r,c}(i,j)=i\cdot c+j$。

特别地：

- 对 $N\times d_{in}$ 矩阵，使用索引 $i\cdot d_{in}+j$；
- 对 $N\times d_h$ 矩阵，使用索引 $i\cdot d_h+j$；
- 对 $N\times d_{cat}$ 矩阵，使用索引 $i\cdot d_{cat}+m$；
- 对 $N\times C$ 矩阵，使用索引 $i\cdot C+c$。

所有 ONNX 输入、witness JSON、公开输入向量和 verifier 解析器必须采用同一展平规则。若 ezkl 运行时采用不同张量布局，则必须在 $encoding\_id$ 中明确记录转换关系。

#### 0.7.6 证明语句

单次推理的 ezkl 内部证明语句固定为：

$\operatorname{Verify}_{ezkl}(vk,\pi_i,inst_i)=1$。

其知识语义写作：

$\exists\ priv_i,trace_i:\ C_{ezkl}(priv_i,inst_i)=1$。

其中 $C_{ezkl}$ 是由 $model.onnx$ 与 $settings.json$ 编译得到的电路。对本文 GAT 推理，$C_{ezkl}$ 至少表达如下关系：

$C_{ezkl}(priv_i,inst_i)=1\Longrightarrow \widetilde f_{\widetilde W}(\widetilde x_i)=\widetilde y_i$，

并且公开绑定记录满足：

$public_i=(inst_i,\operatorname{Pub}(\widetilde x_i),\operatorname{Pub}(\widetilde y_i),input\_hash_i,output\_hash_i,H(W),manifest\_id,artifact\_ids_i,binding\_digest_i,visibility\_cfg)$。

$\operatorname{Pub}(\cdot)$ 表示由 $visibility\_cfg$ 指定的公开部分。主实验默认公开输入张量、最终输出 $Y$ 与模型权重哈希 $H(W)$；隐藏层张量不公开。若需要隐藏部分输入或输出，必须在 $visibility\_cfg$ 中显式声明，并在验证章节说明可验证结论的范围。$H(W)$、$manifest\_id$ 与 $artifact\_ids_i$ 必须由 proof artifact、manifest 或聚合证明公开绑定；它们不必然都作为 ezkl verifier 的原生 public instance 字段逐项出现。但若这些字段未逐项进入 $inst_i$，则 $binding\_digest_i$ 必须进入 $inst_i$ 或进入聚合证明公开输入，用一个不可替换的摘要把外部字段与 proof 绑定。

### 0.8 Padding、零分母、Softmax 可行性与缓存要求

#### 0.8.1 Padding 规则

由于 ezkl / ONNX 证明电路通常要求固定 shape，本文固定采用静态 padding 规则。对任意真实长度 $L_{\mathcal D}$ 与电路长度上界 $n_{\mathcal D}$，所有输入张量必须补到长度 $n_{\mathcal D}$。

对所有 $t\in\{L_{\mathcal D},L_{\mathcal D}+1,\ldots,n_{\mathcal D}-1\}$，统一规定：

- 特征值补零；
- 边索引补零；
- mask 置为无效；
- padding 边不得参与有效节点 Softmax；
- padding 节点不得进入最终指标统计；
- 由 ONNX 图中的 mask 算子或等价固定形状逻辑排除 padding 对有效输出的影响。

#### 0.8.2 Softmax 可行性

要求每个有效节点至少有一条入边。公共输入构造阶段必须保证：

$\#\{k\mid dst(k)=i\}\ge 1 \quad \forall i\in\{0,1,\ldots,N-1\}$。

若原始图不满足该条件，则必须在公开预处理阶段显式加入自环后，才允许进入本文协议。

同时还必须满足：

1. 量化后的 Softmax 分母在电路域中非零；
2. 指数、除法 / 逆元与回缩的近似误差在 $quant\_cfg\_id$ 给定的容忍范围内；
3. 量化后输出的评价指标必须由相同量化配置解释。

#### 0.8.3 热路径对象与缓存要求

热路径对象包括：$SRS$、$model.onnx$、$settings.json$、$C_{ezkl}$、$pk$、$vk$、量化参数、witness 生成缓存、FFT/MSM 预处理以及证明系统内部表布局。

程序启动后必须一次性缓存：

- $SRS$；
- $pk$；
- $vk$；
- $C_{ezkl}$；
- $settings.json$；
- ONNX runtime / ezkl witness 生成所需的静态执行计划；
- 数据集输入张量的公开预处理结果；
- 证明生成所需的硬件资源配置。

单次证明或验证期间不得重复构造这些对象。若实现为了节省内存而采用内部 chunk 生成 witness 的工程手段，也不得改变最终 ONNX 输入、公开输出、$H(W)$、proof artifact 或 transcript 语义。

#### 0.8.4 公开元数据与版本元数据字段表

为保证证明对象、模型参数、量化配置、ONNX 图和 ezkl 电路之间的一致性，证明对象或 attestation manifest 中必须携带一份公开元数据 / 版本元数据，其字段集合固定为：

1. 协议标识字段：$protocol\_id$

	该字段固定标识本文协议版本、ezkl 后端版本、Halo2 后端版本与 verifier 调用规则版本。

2. 模型结构版本字段：$model\_arch\_id$

	该字段固定绑定：总层数 $L$、各隐藏层的头数 / 维度配置概况 $\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1}$、输出层头数 $K_{out}$、隐藏层使用 ELU、每个输出头在注意力聚合后加上该头的偏置向量、最终输出按全部输出头做算术平均、以及各层拼接维定义 $d_{cat}^{(\ell)}=K_{hid}^{(\ell)}d_h^{(\ell)}$。

3. 模型参数版本字段：$model\_param\_id$

	该字段唯一标识量化权重集合 $\widetilde W$、权重哈希算法与公开模型权重哈希 $H(W)$。

4. ONNX 模型字段：$onnx\_model\_id$

	该字段唯一标识 $model.onnx$ 的字节哈希、导出器版本、opset 版本与输入 / 输出张量名。

5. ezkl 设置字段：$settings\_id$

	该字段唯一标识 $settings.json$、量化尺度、lookup 配置、visibility 配置、logrows、scale 与校准策略。

6. 编译电路字段：$compiled\_circuit\_id$

	该字段唯一标识 ezkl 编译后的电路工件 $C_{ezkl}$，用于绑定 ONNX 图、settings 与 Halo2 约束布局。

7. 证明键字段：$pk\_id$

	该字段唯一标识证明键 $pk$。$pk$ 不需要公开给验证者，但其摘要必须进入 manifest 以保证实验可复现。

8. 验证键字段：$vk\_id$

	该字段唯一标识验证键 $vk$，验证者必须使用与该字段匹配的 $vk$。

9. 量化配置版本字段：$quant\_cfg\_id$

	该字段唯一标识全部量化尺度、舍入规则、重缩放规则、符号解释规则和误差容忍边界。

10. 维度配置字段：$dim\_cfg$

	该字段固定定义为：$(G_{batch},N,E,N_{total},node\_ptr,edge\_ptr,L,\{d_{in}^{(\ell)}\}_{\ell=1}^{L},\{d_h^{(\ell)}\}_{\ell=1}^{L-1},\{d_{cat}^{(\ell)}\}_{\ell=1}^{L-1},C,K_{out})$。

11. 编码与序列化字段：$encoding\_id$

	该字段固定标识域元素字节序、整数符号解释、张量展平顺序、ONNX 输入顺序、witness JSON 字段顺序、proof artifact 编码方式以及公开输入向量编码方式。

12. Padding 与 mask 规则字段：$padding\_rule\_id$

	该字段固定标识第 0.8.1 节规定的填充语义、mask 语义、自环添加规则与 padding 输出排除规则。

13. 数据集字段：$dataset\_id$

	该字段唯一标识评测数据集、局部子图抽样规则、标签文件版本、公开预处理脚本版本和输入张量摘要。

14. 评估声明字段：$eval\_id,metric\_script\_id,metric\_value,label\_hash,sample\_order\_id,output\_interpretation\_id$

	这些字段固定评估任务、指标脚本、声明的指标值、标签文件摘要、样本顺序以及输出解释规则。若当前对象只是单次生产推理挑战而不是评估声明，则这些字段取空值 $\bot$，但仍保留字段位置，避免 manifest 编码歧义。

本文统一把 $manifest\_id=Hash(manifest\_body)$ 作为公开元数据摘要。本文不再设置第二个公开元数据哈希字段。

上述字段是证明对象或 attestation manifest 内部携带的正式公共字段。验证者必须按固定顺序解析这些字段，并把它们与 proof artifact、公开输入、验证键和评价指标一起检查。这些字段不是建议性注释，而是参与一致性检查的正式公共对象。

## 1. 参数生成

### 1.1 输入

参数生成算法输入为：

- 安全参数 $\lambda$；
- 底层证明系统公共参数 $SRS$；
- 已训练的 GAT 浮点模型 $f_W$；
- GAT 结构配置 $L,\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1},K_{out},C$；
- 局部子图规模上界 $(N,E)$；
- 全局节点数 $N_{total}$；
- 当前证明实例中的图个数 $G_{batch}$；
- 当前证明实例的节点 / 边前缀指针 $node\_ptr,edge\_ptr$；
- ONNX 导出配置，包括 opset、输入输出名、固定 shape、mask 张量与导出器版本；
- ezkl 设置配置，包括量化 scale、logrows、visibility、lookup 配置和校准策略；
- 校准样本或代表性输入样本，用于确定量化与资源权衡；
- 公开 manifest 配置 $manifest\_cfg$，用于固定协议版本、模型结构、量化配置、数据集、公开输入解释规则与评估声明字段。

### 1.2 输出

参数生成算法输出：

- ONNX 模型文件：$model.onnx$；

- ezkl 设置文件：$settings.json$；

- 编译后的 ezkl 电路工件：$C_{ezkl}$；

- 证明键：$pk$；

- 验证键：$vk$；

- 模型权重哈希：$H(W)$；

- 工件清单：$manifest$ 与其摘要 $manifest\_id$，其中 $manifest$ 包含 $protocol\_id,model\_arch\_id,model\_param\_id,onnx\_model\_id,settings\_id,compiled\_circuit\_id,pk\_id,vk\_id,quant\_cfg\_id,dim\_cfg,encoding\_id,padding\_rule\_id,dataset\_id,eval\_id,metric\_script\_id,metric\_value,label\_hash,sample\_order\_id,output\_interpretation\_id,H(W)$；

- 可选聚合电路工件：$C_{agg}$ 与对应的 $pk_{agg},vk_{agg}$，仅在需要生成聚合证明或指标私有化证明时输出。

### 1.3 固定 shape 与电路规模

本文不再定义 $\mathbb H_{FH},\mathbb H_{edge},\mathbb H_{in},\mathbb H_{d_h},\mathbb H_{cat},\mathbb H_C,\mathbb H_N$ 等手写多项式工作域。ezkl 方案中的“工作域”由固定形状张量、电路行数和 Halo2 内部表布局共同决定。

为避免在多层 GAT 的实现中为每一层重复维护不同 shape，本文正式协议统一采用全局 shape 上界口径：

- 对全部隐藏层的输入共享维，统一取 $d_{in}^{max}=\max\{d_{in}^{(1)},d_{in}^{(2)},\ldots,d_{in}^{(L)}\}$；
- 对全部隐藏层单头输出共享维，统一取 $d_h^{max}=\max\{d_h^{(1)},d_h^{(2)},\ldots,d_h^{(L-1)}\}$；
- 对全部隐藏层拼接共享维与最终输出层输入共享维，统一取 $d_{cat}^{max}=\max\{d_{cat}^{(1)},d_{cat}^{(2)},\ldots,d_{cat}^{(L-1)}\}$。

在任何具体层 $\ell$ 的实例化中：

- 第 $\ell$ 层隐藏模板只占用固定 shape 的前 $d_{in}^{(\ell)}$ 个有效输入维；
- 第 $\ell$ 层隐藏层头模板只占用固定 shape 的前 $d_h^{(\ell)}$ 个有效输出维；
- 第 $\ell$ 层拼接模板与最终输出层投影模板，只占用前 $d_{cat}^{(\ell)}$ 或 $d_{in}^{(L)}=d_{cat}^{(L-1)}$ 个有效位置；
- 其余位置由 mask 排除，并按第 0.8.1 节的填充规则补零。

若使用固定最大 shape 造成额外约束成本，实验报告必须把该成本归入 ezkl 证明开销，不得把 padding 位置当作有效计算结果。

### 1.4 SRS 与 ezkl Setup

ezkl/Halo2 后端若使用 KZG 承诺，则参数生成阶段必须使用公开 $SRS$。本文不再执行“证明者采样隐藏陷门 $\tau$”的步骤。

参数生成中的核心 setup 语义写作：

$Setup_{ezkl}(1^\lambda,W,f,manifest\_cfg,SRS)\rightarrow(model.onnx,settings.json,C_{ezkl},pk,vk,H(W),manifest)$。

展开为以下确定性或可复现实验步骤：

1. ONNX 导出：

$model.onnx=Export_{onnx}(f_W,model\_arch\_id,shape\_cfg,opset,encoding\_id)$。

2. 量化与校准：

$settings.json=Calibrate(model.onnx,calibration\_data,quant\_cfg\_id,visibility\_cfg,resource\_cfg)$。

3. 电路编译：

$C_{ezkl}=Compile_{ezkl}(model.onnx,settings.json,SRS)$。

4. 键生成：

$(pk,vk)=KeyGen_{ezkl}(SRS,C_{ezkl})$。

5. 权重哈希：

$H(W)=Hash(model\_param\_encoding(\widetilde W,quant\_cfg\_id,model\_arch\_id))$。

6. 工件摘要：

$onnx\_model\_id=Hash(model.onnx)$，

$settings\_id=Hash(settings.json)$，

$compiled\_circuit\_id=Hash(C_{ezkl})$，

$pk\_id=Hash(pk)$，

$vk\_id=Hash(vk)$。

7. manifest 生成：

$manifest\_body=Encode(protocol\_id,model\_arch\_id,model\_param\_id,onnx\_model\_id,settings\_id,compiled\_circuit\_id,pk\_id,vk\_id,quant\_cfg\_id,dim\_cfg,encoding\_id,padding\_rule\_id,dataset\_id,eval\_id,metric\_script\_id,metric\_value,label\_hash,sample\_order\_id,output\_interpretation\_id,H(W))$。

$manifest\_id=Hash(manifest\_body)$。

$manifest=(manifest\_body,manifest\_id)$。

后文统一使用 $manifest\_id$ 作为公开元数据摘要。$pk$ 由模型提供方持有并用于生成推理证明；$vk$、$manifest$、$manifest\_id$、$H(W)$ 与必要公开输入一起发布给验证者。$pk$ 的体积可能远大于 $vk$，这是 ezkl 方案的工程成本，不影响验证者的简洁验证接口。

### 1.5 编译工件、非线性函数与模型绑定

#### 1.5.1 输入数据绑定

全局特征表 $T_H$ 不再被编码为单独 KZG 多项式。对 transductive 图数据集，实验实现必须选择以下二者之一，并在 $dataset\_id$ 中固定：

1. 公开输入模式：把当前证明实例的局部特征矩阵 $H$ 直接作为公开输入张量的一部分，并公开其字节哈希；
2. 数据集哈希模式：公开完整数据集或其 Merkle / 文件哈希，并在输入生成脚本中确定性抽取 $H$。

主实验默认采用公开输入模式。若采用数据集哈希模式但不证明抽取过程，则验证者只能确认 proof 对某个公开输入张量成立，不能仅凭 proof 推出该张量必然来自完整数据集。

#### 1.5.2 非线性函数绑定

LeakyReLU、ELU、Softmax 指数、除法 / 逆元、范围检查不再作为本文外部静态表承诺出现。它们由 ezkl 根据 ONNX 算子和 settings 编译到 $C_{ezkl}$ 中。验证者通过 $compiled\_circuit\_id$、$settings\_id$ 与 $vk$ 检查这些函数的约束版本，不单独检查表多项式。

#### 1.5.3 模型权重绑定

定义模型权重编码：

$model\_param\_encoding(\widetilde W,quant\_cfg\_id,model\_arch\_id)=Encode(\mathsf{domain}=\textsf{GAT-ZKML-weight},model\_arch\_id,quant\_cfg\_id,\{(name,shape,dtype,bytes(\widetilde T))\}_{\widetilde T\in\widetilde W})$。

其中参数张量必须按以下固定顺序编码：

$\{\widetilde W^{(\ell,r)},\widetilde a_{src}^{(\ell,r)},\widetilde a_{dst}^{(\ell,r)}\}_{\ell=1,r=0}^{L-1,K_{hid}^{(\ell)}-1}$，

$\{\widetilde W^{(out,s)},\widetilde a_{src}^{(out,s)},\widetilde a_{dst}^{(out,s)},\widetilde b^{(out,s)}\}_{s=0}^{K_{out}-1}$。

模型权重哈希定义为：

$H(W)=Hash(model\_param\_encoding(\widetilde W,quant\_cfg\_id,model\_arch\_id))$。

若权重作为私有 witness，则证明系统必须在电路内证明存在同一份 $\widetilde W$ 使得：

$\widetilde f_{\widetilde W}(\widetilde x_i)=\widetilde y_i$，

$H(W)=Hash(model\_param\_encoding(\widetilde W,quant\_cfg\_id,model\_arch\_id))$。

由于 $\widetilde W$ 不公开，外部 manifest 不能独立检查上述哈希等式；该等式必须由电路内哈希 gadget、ZK-friendly 哈希约束或等价的 proof 内绑定机制证明。若实现只在 proof 外记录 $H(W)$，而没有证明私有权重和 $H(W)$ 的一致性，则不得采用私有权重 witness 模式。

若权重固化为电路常量，则证明系统必须保证：

$Hash(C_{ezkl})=compiled\_circuit\_id,\quad Hash(vk)=vk\_id,\quad model\_param\_id\mapsto H(W)$，

并由 manifest 固定 $compiled\_circuit\_id$、$vk\_id$、$model\_param\_id$ 与 $H(W)$ 的对应关系。两种方式都可以保持权重原文不公开，但不能在同一个评估声明中混合使用。本文主实验默认采用私有权重 witness 模式；若具体实现改用权重固化为电路常量的模式，则必须把该选择写入 $manifest\_cfg$、$protocol\_id$ 或 $model\_param\_id$ 所绑定的实现配置中。若具体实现使用 ZK-friendly hashing 或其他电路内哈希，哈希算法、输入顺序与输出编码必须由 $model\_param\_id$ 唯一固定。单次证明对应的外部公开记录 $public_i$ 必须包含 $H(W)$，或聚合证明 $\Pi$ 的公开记录 $public_{agg}$ 必须包含所有被聚合证明共同的 $H(W)$。

#### 1.5.4 验证键绑定

验证键 $vk$ 绑定编译电路而不是单独的手写约束清单。验证者必须检查：

$Hash(vk)=vk\_id,\quad Hash(C_{ezkl})=compiled\_circuit\_id,\quad Hash(settings.json)=settings\_id,\quad Hash(model.onnx)=onnx\_model\_id$。

若任一哈希不一致，则拒绝验证。

### 1.6 公开拓扑张量与辅助公开输入

本文不再生成公共拓扑多项式 $P_{src},P_{dst},P_{Q}^{edge}$ 等对象。对应信息改由公开输入张量和 manifest 字段承载：

1. 源索引张量：$src$；

2. 目标索引张量：$dst$；

3. 节点前缀张量：$node\_ptr$；

4. 边前缀张量：$edge\_ptr$；

5. 有效节点 mask：$node\_mask$；

6. 有效边 mask：$edge\_mask$；

7. padding mask：$pad\_mask$；

8. batch 图编号张量：$node\_gid,edge\_gid$，若 ONNX 图需要显式使用；

9. 输入张量摘要：$input\_hash=Hash(encoding(\widetilde x,visibility\_cfg,encoding\_id))$。

这些对象全部由验证者根据公开输入本地解析，或从 manifest 中读取其哈希后与公开输入文件比对。凡是会影响 GAT 前向的拓扑对象，都必须进入 $public_i$ 或 $manifest\_body$，并由 $manifest\_id=Hash(manifest\_body)$ 绑定；不得只写在实验说明里而不进入验证绑定。

## 2. 见证生成与公开绑定

### 2.0 总体顺序与实现约束

按“公开预处理 → 标准推理 → 量化校准 → witness 生成 → proof 生成”的顺序书写。工程实现必须严格分为四阶段：

1. 公开预处理阶段：稳定排序、加自环、局部节点重编号、固定 shape padding、mask 构造、公开输入张量编码；
2. 前向推理阶段：用同一个 $model.onnx$ 和同一份量化配置生成 $\widetilde y$；
3. witness 阶段：生成 ezkl witness 文件，绑定 $\widetilde x,\widetilde y,H(W),manifest\_id$ 与 visibility 配置；
4. 证明阶段：调用 ezkl prover，用 $pk$、$C_{ezkl}$ 与 witness 生成 $\pi_i$。

禁止把公开预处理规则、量化规则或 ONNX 导出规则留作人工解释。它们必须进入 manifest，并在验证阶段被机器检查。

本章中的“公开绑定”不表示证明者逐项提交 KZG 多项式承诺；在 ezkl 方案中，承诺由 proof artifact 内部的 Halo2/KZG 机制以及公开的 $H(W)$、$input\_hash$、$compiled\_circuit\_id$、$vk\_id$ 共同承担。证明者不得额外提交与验证逻辑无关的手写承诺对象。

### 2.1 数据加载与特征检索

#### 2.1.1 原始特征

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 与每个输入维索引 $j\in\{0,1,\ldots,d_{in}-1\}$，定义：

$H_{i,j}=T_H[I_i,j]$。

若 $T_H$ 是公开数据集的一部分，则 $H$ 的构造脚本必须由 $dataset\_id$ 绑定；若 $H$ 是用户提供的生产输入，则 $H$ 本身就是本次证明的输入张量。

#### 2.1.2 特征输入绑定方式

ezkl 方案不再设置特征检索 LogUp 表端与查询端。特征合法性通过以下方式处理：

- 主实验模式：$H$ 作为公开输入直接进入 ONNX 图，验证者检查 proof 对该 $H$ 成立；
- 数据集绑定模式：验证者额外检查 $Hash(T_H)$、$I$ 和公开预处理脚本摘要，但不在本 proof 内证明 $H=T_H[I]$；
- 若必须在 proof 内证明 $H=T_H[I]$，则需要把该检索逻辑导出为 ONNX 支持的电路子图，作为 $C_{ezkl}$ 的一部分，而不是恢复手写 LogUp 证明。

#### 2.1.3 输入张量生成

对每个证明实例生成：

$x_i=(H_i,src_i,dst_i,node\_ptr_i,edge\_ptr_i,node\_mask_i,edge\_mask_i,\operatorname{aux}_i)$。

其中 $H_i$ 可以表示单图、局部子图或多图 batch 的节点特征张量。所有张量必须按照 $encoding\_id$ 中规定的顺序、shape 和 dtype 序列化。

先按 $quant\_cfg\_id$ 生成量化输入 $\widetilde x_i=Q_x(x_i)$，再定义输入摘要：

$input\_hash_i=Hash(encoding(\widetilde x_i,visibility\_cfg,encoding\_id))$。

若还需要绑定原始浮点输入、原始图文件或完整数据集文件，则另设：

$raw\_input\_hash_i=Hash(encoding(x_i,dataset\_encoding\_id))$。

$input\_hash_i$ 只绑定证明系统实际使用的量化输入；$raw\_input\_hash_i$ 只绑定 proof 外的数据来源，两者不得互相替代。若 $visibility\_cfg$ 指定输入完全公开，则 $public_i$ 中可直接包含 $\widetilde x_i$；若指定输入隐藏，则 $public_i$ 至少必须包含 $input\_hash_i$ 或其他足以绑定输入的公开承诺。

#### 2.1.4 标准推理与量化输出

先在普通推理环境中运行浮点模型，得到只用于误差分析和实验对照的参考输出：

$y_i^{float}=f_W(x_i)$。

随后按 $quant\_cfg\_id$ 生成量化输入和量化权重：

$\widetilde x_i=Q_x(x_i),\quad \widetilde W=Q_W(W)$。

proof 绑定的量化输出必须由量化模型或 ezkl witness 生成器得到：

$\widetilde y_i=\widetilde f_{\widetilde W}(\widetilde x_i)$。

不得把 $Q_y(y_i^{float})$ 默认当作 proof 输出；只有当 $quant\_cfg\_id$ 明确证明两条路径等价时，才允许把 $Q_y(y_i^{float})$ 作为 $\widetilde y_i$ 的生成方式。

对节点分类输出，若保留 logits，则：

$\widetilde y_i=\widetilde Y_i\in\mathbb Z^{N\times C}$。

若只保留类别预测，则类别必须由固定规则生成：

$\widehat c_{i,u}=\arg\max_{0\le c<C}\widetilde Y_{i,u,c}$，

$\widetilde y_i=(\widehat c_{i,0},\ldots,\widehat c_{i,N-1})$。

也可以直接运行量化后的 ONNX / ezkl witness 生成器得到 $\widetilde y_i$。无论采用哪种工程路径，最终 proof 只证明：

$\widetilde f_{\widetilde W}(\widetilde x_i)=\widetilde y_i$。

因此，实验记录必须同时保存：

$y_i^{float}=f_W(x_i)$，

$\widetilde y_i=\widetilde f_{\widetilde W}(\widetilde x_i)$，

$y_i^{deq}=DQ_y(\widetilde y_i)$，

$err_i=\|y_i^{float}-y_i^{deq}\|$。

若量化导致评价指标变化，该变化必须归入对比实验结果，不得把浮点指标与量化证明指标混称为同一个数。

#### 2.1.5 witness 文件

单个 witness 文件必须包含或绑定：

1. 量化输入 $\widetilde x_i$；
2. 量化输出 $\widetilde y_i$；
3. 模型权重哈希 $H(W)$；
4. manifest 摘要 $manifest\_id$；
5. visibility 配置；
6. 必要的私有权重或私有中间赋值；
7. proof 生成所需的 ezkl 内部 witness trace。

witness 文件可以是 proof 生成的临时文件，不必公开；但其中被声明为 public 的字段必须能由验证者从 proof artifact 中读取或从公开输入文件中重构。

### 2.2 单个隐藏层模板中第 $r$ 个注意力头的完整见证生成

以下步骤对应某个固定隐藏层 $\ell\in\{1,2,\ldots,L-1\}$ 的单层模板，在该模板内，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$ 的隐藏层注意力头分别定义。

数学定义中的头顺序固定为 $r=0,1,\ldots,K_{hid}-1$；实现是否并行不属于协议语义，但任何实现都必须产出与这一固定头顺序一致的 ONNX 算子顺序、输出张量顺序与 manifest 字段顺序。对多层 GAT，证明者与验证者都必须先对隐藏层 $\ell=1$ 实例化第 2.2 节与第 2.3 节，再把 $H_{cat}^{(1)}$ 作为下一隐藏层输入，依次实例化到 $\ell=L-1$；最后再进入第 2.4 节的最终输出层。

#### 2.2.1 投影

对每个节点 $i$ 与每个隐藏维索引 $j\in\{0,1,\ldots,d_h-1\}$，定义：

$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}$。

在 ezkl 电路中，该矩阵乘法由 ONNX MatMul / Gemm / Einsum 或等价算子表示，并被编译为累计点积约束或等价 arithmetic constraints。证明者不再提交 $[P_{H'^{(r)}}]$、折叠向量或外点评值。

#### 2.2.2 源注意力绑定

对每个节点 $i$，定义：

$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)}$。

该点积必须在 ONNX 图中显式出现。若导出器把源注意力和目标注意力合并为一个更大的线性算子，则必须保证合并后的图与本公式代数等价，并由 $onnx\_model\_id$ 绑定。

#### 2.2.3 目标注意力绑定

对每个节点 $i$，定义：

$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}$。

该点积与源注意力采用相同量化语义。不同头之间不得复用错误的注意力向量；头编号必须由 ONNX initializer 名或 manifest 明确绑定。

#### 2.2.4 边域打分与 LeakyReLU

对每条有效边 $k$，定义：

$S_k^{(r)}=E_{src,src(k)}^{(r)}+E_{dst,dst(k)}^{(r)}$，

$Z_k^{(r)}=LReLU(S_k^{(r)})$。

Gather / index select 操作必须由固定 shape ONNX 图表示。LeakyReLU 由 ezkl 编译为非线性 lookup 或等价约束，其斜率、截断范围和量化尺度由 $settings\_id$ 唯一固定。

#### 2.2.5 Softmax 归一化

对每个目标节点 $i$，定义：

$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}$，

$U_k^{(r)}=\exp(Z_k^{(r)}-M_{dst(k)}^{(r)})$，

$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}$，

$\alpha_k^{(r)}=U_k^{(r)}\cdot (Sum_{dst(k)}^{(r)})^{-1}$。

在 ONNX 图中，Softmax 可以实现为显式 segment-softmax 分解，也可以实现为 masked dense softmax；二者必须在有效边集合上与上述公式一致。若为了 ezkl 支持性采用 masked dense 实现，padding 和非边位置必须被 mask 到不影响有效边归一化的值，且该 mask 语义必须进入 $padding\_rule\_id$。

#### 2.2.6 聚合与 ELU

对每个节点 $i$ 与每个隐藏维 $j$，定义：

$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}$，

$H_{agg,i,j}^{(r)}=ELU(H_{agg,pre,i,j}^{(r)})$。

聚合操作由 ONNX 中的 gather、scatter、matmul、einsum 或等价固定 shape 算子表示。ELU 的近似、查找表范围或分段实现由 $settings\_id$ 固定。证明者不再构造 PSQ 状态机或路由总和状态机；这些语义由 ONNX 图和 ezkl 约束共同保证。

### 2.3 隐藏层模板的拼接阶段

对每个隐藏层 $\ell$，所有隐藏头完成后，按头编号从小到大拼接：

$H_{cat,i,r\cdot d_h+j}^{(\ell)}=H_{agg,i,j}^{(\ell,r)}$。

拼接后的 $H_{cat}^{(\ell)}$ 是下一隐藏层的唯一输入。若 $\ell=L-1$，则 $H_{cat}^{(L-1)}$ 是最终输出层的输入。

ezkl witness 生成器必须记录拼接算子的内部赋值，但 proof artifact 不单独公开每个头的隐藏输出，除非 $visibility\_cfg$ 显式要求公开中间层。主实验默认公开输入张量、最终输出 $Y$ 与 $H(W)$；隐藏层张量不公开，除非 $visibility\_cfg$ 显式要求。

### 2.4 输出层的完整见证生成

对最终输出层中的每个输出头 $s$，见证生成按以下顺序执行。

1. 输出投影：

$Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out,s)}$。

定点电路中对应为：

$\widetilde Y_{i,c}'^{(s)}=\operatorname{Rescale}_{cat,W\to outproj}\left(\sum_{m=0}^{d_{cat}-1}\widetilde H_{cat,i,m}\widetilde W_{m,c}^{(out,s)}\right)$。

2. 输出注意力打分：

$E_{src,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{src,c}^{(out,s)}$，

$E_{dst,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{dst,c}^{(out,s)}$，

$S_k^{(out,s)}=E_{src,src(k)}^{(out,s)}+E_{dst,dst(k)}^{(out,s)}$，

$Z_k^{(out,s)}=LReLU(S_k^{(out,s)})$。

3. 输出层 Softmax：

$M_i^{(out,s)}=\max\{Z_k^{(out,s)}\mid dst(k)=i\}$，

$U_k^{(out,s)}=\exp(Z_k^{(out,s)}-M_{dst(k)}^{(out,s)})$，

$Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)}$，

$\alpha_k^{(out,s)}=U_k^{(out,s)}\cdot(Sum_{dst(k)}^{(out,s)})^{-1}$。

定点实现若使用非负差分，则必须满足：

$\Delta_k^{+(out,s)}=M_{dst(k)}^{(out,s)}-Z_k^{(out,s)},\quad U_k^{(out,s)}=\exp(-\Delta_k^{+(out,s)})$。

4. 输出聚合：

$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}$。

定点电路中对应为：

$\widetilde Y_{i,c}^{pre(s)}=\operatorname{Rescale}_{\alpha,outproj\to outpre}\left(\sum_{\{k\mid dst(k)=i\}}\widetilde\alpha_k^{(out,s)}\widetilde Y_{src(k),c}'^{(s)}\right)$。

5. 输出头尺度对齐与加偏置：

$\bar Y_{i,c}^{pre(s)}=Rescale_{out}(Y_{i,c}^{pre(s)};S_{out,pre}\rightarrow S_{out,head})$，

$Y_{i,c}^{(s)}=\bar Y_{i,c}^{pre(s)}+b_c^{(out,s)}$。

为避免与第 0.6 节中的 $\widetilde Y$ 记号冲突，本节用 $\bar Y^{pre(s)}$ 表示“输出头尺度对齐后的聚合前输出”；最终被 proof 约束的仍是对应的固定点张量。

6. 多输出头平均：

$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}$，

$Y_{i,c}^{avg}=K_{out}^{-1}\cdot Y_{i,c}^{sum}$。

定点 witness 中公开或私有的最终输出必须按公开平均量化算子生成：

$\widetilde Y_{i,c}=AvgQuant_{K_{out}}\left(\sum_{s=0}^{K_{out}-1}\widetilde Y_{i,c}^{(s)};K_{out},quant\_cfg\_id\right)$。

若 $K_{out}=1$，则 $AvgQuant_{1}$ 为恒等映射，$Y_{i,c}=Y_{i,c}^{(0)}$。实现不得在 $K_{out}=1$ 时改变输出层语义或跳过 manifest 中的输出头字段。

7. 公开输出生成：

若任务是节点分类，公开输出可以是 logits $Y$、argmax 类别 $\widehat y_i=\arg\max_cY_{i,c}$，或二者同时公开。具体选择必须由 $visibility\_cfg$ 固定。若只公开 argmax，不公开 logits，则验证者只能验证 argmax 是电路输出或固定后处理规则的公开结果，不能在 proof 外重新计算 logits 级指标。

8. witness 绑定：

最终 witness 与公开记录必须满足：

$w_i^{ezkl}=(priv_i,inst_i,trace_i)$，

$public_i=(inst_i,\operatorname{Pub}(\widetilde x_i),\operatorname{Pub}(\widetilde y_i),input\_hash_i,output\_hash_i,H(W),manifest\_id,artifact\_ids_i,binding\_digest_i,visibility\_cfg)$。

其中 $trace_i$ 是 ezkl 内部 witness trace，包含上述所有中间张量的固定点赋值。$trace_i$ 不公开，但 proof 验证保证其满足 $C_{ezkl}$ 的全部约束。$priv_i$ 是否包含 $\widetilde W$ 取决于第 0.7.4 节定义的权重绑定模式。

## 3. 证明生成

### 3.1 输入与输出

证明生成算法输入：

- 编译电路 $C_{ezkl}$；
- 证明键 $pk$；
- 单次推理 witness $w_i^{ezkl}=(priv_i,inst_i,trace_i)$；
- manifest；
- 公开输入文件、公开输入向量，或可生成 $inst_i$ 的公开字段。

证明生成算法输出：

- 单次推理证明 $\pi_i$；
- 与 $\pi_i$ 绑定的 ezkl public instance $inst_i$；
- 外部公开记录 $public_i$；
- 可选的 witness 摘要 $witness\_id_i$；
- 可选的聚合证明输入项 $agg\_item_i=(\pi_i,inst_i,public_i,vk\_id,H(W))$。

形式化写作：

$Prove_{ezkl}(pk,C_{ezkl},priv_i,inst_i,trace_i)\rightarrow(\pi_i,inst_i,public_i)$。

正确性要求写作：

$\operatorname{Verify}_{ezkl}(vk,\pi_i,inst_i)=1\Longrightarrow \exists priv_i,trace_i:\ C_{ezkl}(priv_i,inst_i)=1$。

这里的 $public_i$ 表示 proof artifact 或 attestation 中对 $H(W)$、公开输出、输入摘要和 manifest 摘要的公开绑定；它不表示 proof 文件明文包含全部私有见证，也不要求所有这些字段都直接进入 ezkl verifier 的原生 public instance。

### 3.2 ezkl Prover 与 transcript 绑定

本文不再手写 Fiat-Shamir 挑战顺序。证明者调用 ezkl prover，Halo2 transcript 内部按后端规定吸入以下对象：

1. 验证键和电路布局绑定信息；
2. ezkl public instance $inst_i$；
3. 内部 witness 多项式承诺；
4. permutation / lookup / arithmetic argument 所需的中间承诺；
5. quotient 与 opening 相关消息；
6. Fiat-Shamir 随机挑战。

本文只在外层规定 transcript 绑定条件：

$transcript\_id_i=Hash(encoding\_id,vk\_id,compiled\_circuit\_id,inst_i,public_i)$。

若 $encoding\_id$、公开输入顺序、曲线点编码、域元素字节序、$vk$ 或 $inst_i$ 中任一项改变，则必须得到不同的 $transcript\_id_i$ 或不同的验证结果。实现不得把不同编码语义的 proof 伪装成同一协议版本。

### 3.3 电路约束族与公式展开

ezkl 的灵活性来自把 ONNX 计算图中的 ML 运算规约为少量通用证明约束。本文不再逐项列出手写 KZG opening 或自定义商多项式恒等式，而是采用 ezkl/Halo2 后端生成的 arithmetic、lookup、permutation 与 quotient 约束。为了避免“只写名称、不写公式”，本节给出本文 GAT 前向会用到的约束语义展开；具体行布局和 selector 由 $compiled\_circuit\_id$ 绑定。

1. 累计点积约束。

对长度为 $n$ 的两个向量 $a,b$，点积 $c=\sum_{q=0}^{n-1}a_qb_q$ 可用累加器 $m_0,\ldots,m_n$ 表示：

$m_0=0,\quad m_{q+1}=m_q+a_qb_q\quad(0\le q<n),\quad m_n=c$。

矩阵乘法、源注意力点积、目标注意力点积、输出投影和加权聚合都可归约到该形式。

2. 累计求和约束。

对 $s=\sum_{q=0}^{n-1}u_q$，使用累加器：

$r_0=0,\quad r_{q+1}=r_q+u_q\quad(0\le q<n),\quad r_n=s$。

Softmax 分母、多输出头求和、batch 级正确数统计和指标求和都属于该类型。

3. 逐元素算术约束。

对同 shape 张量 $A,B,C$，逐元素加法、减法、乘法和线性重缩放分别写作：

$C_t=A_t+B_t$，

$C_t=A_t-B_t$，

$C_t=A_t\cdot B_t$，

$C_t=\operatorname{Rescale}_{A\to C}(A_t;S_A\rightarrow S_C)$。

4. lookup 或分段非线性约束。

对非线性算子 $g$，若 settings 使用查找表 $T_g$，则每个有效位置必须满足：

$(u_t,v_t)\in T_g,\quad v_t\approx Q_{out}(g(DQ_{in}(u_t)))$。

该形式覆盖 $LReLU$、$ELU$、$\exp$ 近似、范围检查、比较、max/argmax 辅助约束和某些除法 / 逆元近似。

5. copy / permutation / gather 约束。

若 ONNX 图要求 $B_t=A_{\sigma(t)}$，则电路必须验证：

$B_t=A_{\sigma(t)}\quad \forall t\in\mathcal I_{valid}$。

在 Halo2 内部该关系可由 permutation/copy constraints 或等价算术选择矩阵实现。对公开索引 $src,dst$ 的 gather，可写作：

$Gather(A,src)_k=A_{src(k)}$，

$Gather(A,dst)_k=A_{dst(k)}$。

6. 权重身份绑定。

证明对象必须把本次推理使用的权重与公开哈希 $H(W)$ 绑定。若权重为私有 witness，则 proof 内部必须证明：

$H(W)=Hash(model\_param\_encoding(\widetilde W,quant\_cfg\_id,model\_arch\_id))$。

该关系不能只写在外部 JSON 或实验说明中；否则验证者无法排除 witness 中实际权重与公开 $H(W)$ 不一致的情况。

若权重为电路常量，则验证者检查：

$(compiled\_circuit\_id,vk\_id,model\_param\_id,H(W))\in manifest$。

两种方式都必须能排除“评估时证明一个模型、生产挑战时换另一个模型”的情况。

#### 3.3.1 GAT 投影约束

对隐藏层投影，电路必须验证每个有效节点 $i$、头 $r$ 和隐藏维 $j$：

$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}$。

定点展开为：

$acc_{i,j,0}^{(r)}=0$，

$acc_{i,j,m+1}^{(r)}=acc_{i,j,m}^{(r)}+\widetilde H_{i,m}\widetilde W_{m,j}^{(r)}\quad(0\le m<d_{in})$，

$\widetilde H_{i,j}'^{(r)}=\operatorname{Rescale}_{in,W\to h'}(acc_{i,j,d_{in}}^{(r)})$。

源注意力和目标注意力同理：

$e^{src,(r)}_{i,0}=0,\quad e^{src,(r)}_{i,j+1}=e^{src,(r)}_{i,j}+\widetilde H_{i,j}'^{(r)}\widetilde a_{src,j}^{(r)}$，

$\widetilde E_{src,i}^{(r)}=\operatorname{Rescale}_{h',a\to score}(e^{src,(r)}_{i,d_h})$，

$e^{dst,(r)}_{i,0}=0,\quad e^{dst,(r)}_{i,j+1}=e^{dst,(r)}_{i,j}+\widetilde H_{i,j}'^{(r)}\widetilde a_{dst,j}^{(r)}$，

$\widetilde E_{dst,i}^{(r)}=\operatorname{Rescale}_{h',a\to score}(e^{dst,(r)}_{i,d_h})$。

输出层投影使用同一形式：

$Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out,s)}$，

$\widetilde Y_{i,c}'^{(s)}=\operatorname{Rescale}_{cat,W\to outproj}\left(\sum_{m=0}^{d_{cat}-1}\widetilde H_{cat,i,m}\widetilde W_{m,c}^{(out,s)}\right)$。

手写方案中的 CRPC 折叠、共享维累加器开放和张量绑定子证明全部删除；它们的验证功能由上述 ONNX 算子约束和 Halo2 内部 opening 统一承担。

#### 3.3.2 注意力打分与非线性约束

对每条有效边 $k$，隐藏层注意力打分必须满足：

$E_{src,k}^{edge(r)}=E_{src,src(k)}^{(r)}$，

$E_{dst,k}^{edge(r)}=E_{dst,dst(k)}^{(r)}$，

$S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)}$。

定点电路中为：

$\widetilde S_k^{(r)}=\operatorname{Rescale}_{score}\left(\widetilde E_{src,src(k)}^{(r)}+\widetilde E_{dst,dst(k)}^{(r)}\right)$。

LeakyReLU 的实数语义为：

$LReLU_\rho(t)=\begin{cases}t,&t\ge 0,\\ \rho t,&t<0,\end{cases}$

其中负半轴斜率 $\rho$ 必须与训练 / 导出模型一致。电路约束为：

$\widetilde Z_k^{(r)}=Q_{lrelu}\left(LReLU_\rho(DQ_{score}(\widetilde S_k^{(r)}))\right)$，

或等价 lookup 关系：

$(\widetilde S_k^{(r)},\widetilde Z_k^{(r)})\in T_{LReLU}^{settings}$。

输出层 $S_k^{(out,s)}$、$Z_k^{(out,s)}$ 采用同一公式，只需把 $(r)$ 替换为 $(out,s)$。

#### 3.3.3 Softmax 约束

隐藏层每个目标节点 $i$ 的 Softmax 约束为：

$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}$，

$\Delta_k^{+(r)}=M_{dst(k)}^{(r)}-Z_k^{(r)}$，

$U_k^{(r)}=\exp(-\Delta_k^{+(r)})$，

$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}$，

$\alpha_k^{(r)}=U_k^{(r)}\cdot (Sum_{dst(k)}^{(r)})^{-1}$。

定点展开为：

$\widetilde U_k^{(r)}=Q_{softmax}\left(\exp(-DQ_{score}(\widetilde\Delta_k^{+(r)}))\right)$，

$sum_{i,0}^{(r)}=0$，

$sum_{i,t+1}^{(r)}=sum_{i,t}^{(r)}+\widetilde U_{k_t}^{(r)}\quad(k_t\in\{k\mid dst(k)=i\})$，

$\widetilde{Sum}_i^{(r)}=sum_{i,deg(i)}^{(r)}$，

$\widetilde{inv}_i^{(r)}\approx Q_{inv}\left((DQ_{softmax}(\widetilde{Sum}_i^{(r)}))^{-1}\right)$，

$\widetilde\alpha_k^{(r)}=\operatorname{Rescale}_{softmax,inv\to \alpha}\left(\widetilde U_k^{(r)}\cdot \widetilde{inv}_{dst(k)}^{(r)}\right)$。

其中 $\approx$ 的误差界由 $quant\_cfg\_id$ 和 $settings\_id$ 规定；若实现使用精确域内逆元，则对应约束可写为：

$\widetilde{Sum}_i^{(r)}\cdot inv_i^{(r)}=1\quad\text{in }\mathbb F_p$。

若使用 masked dense softmax，则非边位置必须由 mask 排除，满足：

$edge\_mask_k=0\Longrightarrow \alpha_k^{(r)}=0$。

输出层 Softmax 对每个输出头 $s$ 同理：

$\alpha_k^{(out,s)}=\frac{\exp(Z_k^{(out,s)}-M_{dst(k)}^{(out,s)})}{\sum_{k':dst(k')=dst(k)}\exp(Z_{k'}^{(out,s)}-M_{dst(k)}^{(out,s)})}$。

#### 3.3.4 聚合、ELU、拼接与输出约束

隐藏层聚合必须满足：

$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}$。

定点展开为：

$g_{i,j,0}^{(r)}=0$，

$g_{i,j,t+1}^{(r)}=g_{i,j,t}^{(r)}+\widetilde\alpha_{k_t}^{(r)}\widetilde H_{src(k_t),j}'^{(r)}\quad(k_t\in\{k\mid dst(k)=i\})$，

$\widetilde H_{agg,pre,i,j}^{(r)}=\operatorname{Rescale}_{\alpha,h'\to aggpre}(g_{i,j,deg(i)}^{(r)})$。

ELU 的实数语义为：

$ELU_\eta(t)=\begin{cases}t,&t\ge 0,\\ \eta(\exp(t)-1),&t<0.\end{cases}$

电路约束为：

$\widetilde H_{agg,i,j}^{(r)}=Q_{agg}\left(ELU_\eta(DQ_{aggpre}(\widetilde H_{agg,pre,i,j}^{(r)}))\right)$，

或等价 lookup 关系：

$(\widetilde H_{agg,pre,i,j}^{(r)},\widetilde H_{agg,i,j}^{(r)})\in T_{ELU}^{settings}$。

拼接约束为纯 copy/reshape 关系：

$H_{cat,i,r\cdot d_h+j}^{(\ell)}=H_{agg,i,j}^{(\ell,r)}$。

输出层聚合、尺度对齐、偏置和平均约束为：

$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}$，

$\bar Y_{i,c}^{pre(s)}=Rescale_{out}(Y_{i,c}^{pre(s)};S_{out,pre}\rightarrow S_{out,head})$，

$Y_{i,c}^{(s)}=\bar Y_{i,c}^{pre(s)}+b_c^{(out,s)}$，

$Y_{i,c}^{avg}=K_{out}^{-1}\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}$。

定点电路中最终 logits 必须满足：

$\widetilde Y_{i,c}=AvgQuant_{K_{out}}\left(\sum_{s=0}^{K_{out}-1}\widetilde Y_{i,c}^{(s)};K_{out},quant\_cfg\_id\right)$。

若 $K_{out}=1$，最后一式退化为 $\widetilde Y_{i,c}=\widetilde Y_{i,c}^{(0)}$；实现不得在 $K_{out}=1$ 时把输出层改写成没有注意力的线性层。

#### 3.3.5 评价指标聚合约束

单次推理证明只证明推理正确性。若要生成“可验证评估声明”，必须选择以下一种包装方式。

1. 朴素打包。发布所有 $\{\pi_i,inst_i,public_i\}_{i=1}^{n}$、$vk$、$manifest$、标签与评价脚本。验证者逐个验证推理 proof，再在 proof 外计算指标。

对节点分类，若公开类别预测 $\widehat c_i$，准确率定义为：

$Acc=\frac{1}{n_{eval}}\sum_{i=1}^{n_{eval}}\mathbf 1[\widehat c_i=label_i]$。

若公开 logits，则先按固定规则得到：

$\widehat c_i=\arg\max_{0\le c<C}\widetilde Y_{i,c}$，

再计算 $Acc$。宏平均 F1 可写为：

$TP_c=\sum_i\mathbf 1[\widehat c_i=c\wedge label_i=c]$，

$FP_c=\sum_i\mathbf 1[\widehat c_i=c\wedge label_i\ne c]$，

$FN_c=\sum_i\mathbf 1[\widehat c_i\ne c\wedge label_i=c]$，

$F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}$，

$MacroF1=\frac{1}{C}\sum_{c=0}^{C-1}F1_c$。

2. ezkl 聚合。使用 ezkl aggregation toolkit 或递归证明，把多个推理 proof 聚合为一个证明 $\Pi$。聚合证明至少必须检查：

$\operatorname{Verify}_{ezkl}(vk,\pi_i,inst_i)=1\quad\forall i$，

$H(W)_1=H(W)_2=\cdots=H(W)_n$，

$vk\_id_1=\cdots=vk\_id_n,\quad compiled\_circuit\_id_1=\cdots=compiled\_circuit\_id_n$。

3. 指标私有化聚合。额外编写一个 Halo2 / ezkl 兼容指标电路，只公开最终 accuracy、混淆矩阵、bias 指标或其他声明值，不公开逐样本输出。该电路必须证明：

$metric=Metric(\{\widetilde y_i,label_i\}_{i=1}^{n};metric\_script\_id)$。

该方案需要额外审计指标电路，且必须在 $protocol\_id$ 中声明为不同证明模式。

### 3.4 proof 字节、公开实例与审计统计

本文不再生成手写外点评值、域内点评值与 KZG opening 清单。对应过程由 ezkl/Halo2 proof 内部完成。验证者只接收：

- proof 字节串 $\pi_i$ 或聚合 proof $\Pi$；
- 验证键 $vk$ 或 $vk_{agg}$；
- ezkl public instance $inst_i$ 与外部公开记录 $public_i$，或聚合公开输入 $public_{agg}$；
- manifest。

如果需要调试或审计 proof 内部开销，可以记录 ezkl 的 constraint count、lookup count、logrows、proof size、proving time、verify time、pk size、vk size；这些统计量不参与数学验证，但必须与同一 $compiled\_circuit\_id$ 绑定，不能跨模型混用。

### 3.5 最终证明对象

最终证明对象记为 $\pi_{GAT}$。它有两种规范形式。

第一种是朴素评估包：

$\pi_{GAT}^{bundle}=(manifest,vk,\{(\pi_i,inst_i,public_i)\}_{i=1}^{n},labels,metric\_script\_id,metric\_value)$。

它必须满足：

$\operatorname{Verify}_{ezkl}(vk,\pi_i,inst_i)=1\quad\forall i\in\{1,\ldots,n\}$，

$H(W)_1=H(W)_2=\cdots=H(W)_n$，

$manifest\_id_1=manifest\_id_2=\cdots=manifest\_id_n$，

$public_i.artifact\_ids_i.vk\_id=Hash(vk),\quad public_i.artifact\_ids_i.compiled\_circuit\_id=manifest.compiled\_circuit\_id\quad\forall i$。

验证者逐个验证 $\pi_i$，检查所有 $H(W)$、$public_i.artifact\_ids_i.vk\_id$、$public_i.artifact\_ids_i.compiled\_circuit\_id$、$public_i.artifact\_ids_i.settings\_id$ 和 $manifest\_id$ 一致，再运行 $metric\_script\_id$ 指定的评价脚本得到最终指标。

第二种是聚合评估声明：

$\pi_{GAT}^{agg}=(manifest,vk_{agg},\Pi,public_{agg})$。

其中 $public_{agg}$ 至少包含：

- $H(W)$；
- $vk\_id$ 或被聚合推理 proof 的验证键摘要；
- $compiled\_circuit\_id$；
- $settings\_id$；
- $dataset\_id$ 或输入集合摘要；
- 评价指标 $metric\_value$；
- 指标计算规则摘要；
- 聚合证明模式标识。

聚合语义写作：

$\operatorname{Verify}_{agg}(vk_{agg},\Pi,public_{agg})=1$，

并且：

$public_{agg}.H(W)=H(W)_i\quad\forall i$，

$public_{agg}.metric\_value=Metric(\{\widetilde y_i,label_i\}_{i=1}^{n};metric\_script\_id)$。

朴素评估包更容易审计逐样本输出，但会公开更多测试集推理结果；聚合评估声明可以隐藏逐样本输出，但需要额外聚合电路和指标电路。两种形式都不得声称验证了训练过程；本文只证明固定模型权重下的推理与评估声明。

## 4. 验证

### 4.1 元数据与对象完整性检查

验证者首先解析 $\pi_{GAT}$ 中的 manifest，并检查：

1. $protocol\_id$ 是否为本文 ezkl 证明方案版本；
2. $model\_arch\_id$ 是否与声明的 GAT 结构一致；
3. $onnx\_model\_id$ 是否等于 $Hash(model.onnx)$；
4. $settings\_id$ 是否等于 $Hash(settings.json)$；
5. $compiled\_circuit\_id$ 是否等于 $Hash(C_{ezkl})$；
6. $vk\_id$ 是否等于 $Hash(vk)$；
7. $quant\_cfg\_id$ 是否与实验报告中的量化配置一致；
8. $dataset\_id$、$dim\_cfg$、$padding\_rule\_id$ 是否与公开输入张量一致；
9. $encoding\_id$ 是否与 verifier 解析方式一致。

任一检查失败，则直接拒绝，不进入 proof 验证。

### 4.2 ezkl / Halo2 verifier 调用

验证者不手动重放本文自定义 transcript；验证者从外部公开记录 $public_i$ 中解析 $inst_i$，并调用 ezkl/Halo2 verifier：

$\operatorname{Verify}_{ezkl}(vk,\pi_i,inst_i)$。

该 verifier 内部会按 $encoding\_id$ 与后端规则重建 Fiat-Shamir transcript。本文层面的要求是 proof 字节串、ezkl public instance、外部公开记录和验证键必须全部由 manifest 绑定；如果 proof 能通过 verifier，但 manifest 哈希不一致，仍然拒绝。

### 4.3 单次推理 proof 验证

本文不再暴露手写域内 opening。该步骤替换为 ezkl 推理 proof 验证。

对朴素评估包中的每个样本 $i$，验证者先从 $public_i$ 中解析 $inst_i$，再执行：

$ok_i=\operatorname{Verify}_{ezkl}(vk,\pi_i,inst_i)$。

要求：

$ok_i=1\quad\forall i\in\{1,\ldots,n\}$。

同时检查 $public_i$ 中的公开输入、公开输出、输入输出哈希、$H(W)$、$manifest\_id$ 与 $artifact\_ids_i$ 均与 manifest 一致：

$public_i.H(W)=manifest.H(W)$，

$public_i.manifest\_id=manifest.manifest\_id$，

$public_i.artifact\_ids_i.vk\_id=Hash(vk)$，

$public_i.artifact\_ids_i.compiled\_circuit\_id=manifest.compiled\_circuit\_id$，

$public_i.artifact\_ids_i.settings\_id=manifest.settings\_id$，

$public_i.artifact\_ids_i.quant\_cfg\_id=manifest.quant\_cfg\_id$，

$public_i.binding\_digest_i=Hash(encoding(\operatorname{Pub}(\widetilde x_i),\operatorname{Pub}(\widetilde y_i),input\_hash_i,output\_hash_i,H(W),manifest\_id,artifact\_ids_i,visibility\_cfg,encoding\_id))$，

$public_i.binding\_digest_i\in inst_i$ 或已由聚合证明公开绑定。

若任何单个 proof 验证失败，则整个朴素评估包无效。若只需要验证某个生产输入的挑战证明，则只验证对应的 $\pi^\ast$。

### 4.4 公开输入、输出与哈希绑定检查

本文不再暴露手写外点评值折叠 opening。该步骤替换为公开输入和输出绑定检查。

1. 若输入公开，验证者按 $encoding\_id$ 重建 $\widetilde x_i$ 并检查：

$Hash(encoding(\widetilde x_i,visibility\_cfg,encoding\_id))=input\_hash_i$。

2. 若输入隐藏，验证者检查 $input\_hash_i$ 与承诺的输入摘要一致，但不能得知输入原文。

3. 若输出公开，验证者读取 $\widetilde y_i$ 并检查：

$Hash(encoding(\widetilde y_i,visibility\_cfg,encoding\_id))=output\_hash_i$。

4. 验证者检查所有 proof 的模型权重哈希一致：

$H(W)_1=H(W)_2=\cdots=H(W)_n$。

5. 验证者检查所有 proof 的电路和量化配置一致：

$public_1.artifact\_ids.vk\_id=\cdots=public_n.artifact\_ids.vk\_id$，

$public_1.artifact\_ids.compiled\_circuit\_id=\cdots=public_n.artifact\_ids.compiled\_circuit\_id$，

$public_1.artifact\_ids.settings\_id=\cdots=public_n.artifact\_ids.settings\_id$，

$public_1.artifact\_ids.quant\_cfg\_id=\cdots=public_n.artifact\_ids.quant\_cfg\_id$。

这些检查保证评价声明中使用的是同一套模型权重、同一份电路、同一组量化规则，而不是不同 proof 混合拼接出的结果。

### 4.5 评价指标与聚合证明检查

本文不再列出手写 quotient 身份。该步骤替换为聚合证明或指标计算检查。

若使用朴素评估包，验证者在所有单次 proof 通过后，按 $metric\_script\_id$ 对公开输出与标签重新计算指标：

$metric'=Metric(\{\widetilde y_i,label_i\}_{i=1}^{n};metric\_script\_id)$。

然后检查：

$metric'=manifest.metric\_value$。

若指标脚本、标签、样本顺序、输出解释规则、mask 或 padding 规则不一致，则拒绝。

若使用聚合评估声明，验证者执行：

$\operatorname{Verify}_{agg}(vk_{agg},\Pi,public_{agg})=1$。

并检查：

$public_{agg}.H(W)=manifest.H(W)$，

$public_{agg}.vk\_id=manifest.vk\_id$，

$public_{agg}.compiled\_circuit\_id=manifest.compiled\_circuit\_id$，

$public_{agg}.dataset\_id=manifest.dataset\_id$，

$public_{agg}.metric\_value=manifest.metric\_value$。

聚合 proof 必须证明所有被聚合的推理 proof 有效，并且指标计算规则正确执行；否则不得把 $\Pi$ 称为可验证评估声明。

### 4.6 模型、输入与输出一致性检查

本文不再存在单独的 $\Pi_{bind}$ 张量绑定子证明族。对应语义由以下检查合并承担：

1. ONNX 图和 settings 由 $onnx\_model\_id$、$settings\_id$、$compiled\_circuit\_id$ 绑定；
2. 权重由 $H(W)$、$model\_param\_id$ 与 proof 公开输出或 manifest 绑定；
3. 输入由公开输入张量或 $input\_hash_i$ 绑定；
4. 输出由 proof 的 public output 或 $output\_hash_i$ 绑定；
5. 中间张量由 ezkl witness trace 和 Halo2 约束内部绑定。

如果验证者需要确认生产模型与评估模型一致，则必须要求提供新的挑战证明：

$Prove_{ezkl}(pk,C_{ezkl},priv^\ast,inst^\ast,trace^\ast)\rightarrow(\pi^\ast,inst^\ast,public^\ast)$，

其中 $priv^\ast,inst^\ast,trace^\ast$ 对应生产输入 $x^\ast$ 与输出 $y^\ast$。验证者接受该生产推理当且仅当：

$\operatorname{Verify}_{ezkl}(vk,\pi^\ast,inst^\ast)=1$

且：

$public^\ast.H(W)=manifest.H(W)$，

$public^\ast.artifact\_ids.compiled\_circuit\_id=manifest.compiled\_circuit\_id$，

$public^\ast.artifact\_ids.vk\_id=manifest.vk\_id$。

这只能证明被挑战的该次推理存在同一权重哈希下的有效 witness；它不能自动证明未被挑战的历史推理也全部正确。若需要生产环境中的持续保证，必须采用每次推理都证明、随机挑战、或“先预测后证明”的审计策略。

### 4.7 验证结论

验证者输出：

$\operatorname{Accept}$，当且仅当以下条件全部成立：

1. manifest 完整性检查通过；
2. $vk$、$model.onnx$、$settings.json$、$C_{ezkl}$ 与各自哈希字段一致；
3. 每个单次推理 proof 或聚合 proof 验证通过；
4. 所有外部公开记录中的 $H(W)$ 与声明的模型权重哈希一致；
5. 输入、输出、标签、样本顺序、padding、mask 与量化解释规则一致；
6. 评价指标由公开输出重新计算得到，或由聚合 proof 正确证明；
7. 生产挑战场景下，新 proof 的 $H(W)$ 与原评估 attestation 中的 $H(W)$ 一致。

否则输出：

$\operatorname{Reject}$。

最终可验证结论应表述为：

“在 $manifest$ 绑定的 ONNX 计算图、ezkl 设置、量化配置、验证键和公开输入集合下，存在与公开模型权重哈希 $H(W)$ 一致的模型权重，使得量化 GAT 推理 $\widetilde f_{\widetilde W}(\widetilde x_i)=\widetilde y_i$ 对所有已验证样本成立；对应评价指标按声明规则计算得到。”

不得把该结论扩大解释为：

- 证明了训练过程正确；
- 证明了浮点模型与量化模型完全等价；
- 证明了未进入 attestation 的数据集样本也满足同样指标；
- 证明了生产环境中未被挑战的所有历史推理都使用同一模型；
- 证明了 benchmark 本身充分、公平或无偏。

本文对比实验的核心替换点到此完成：GAT 前向、数据预处理、层数与多头语义保持不变；原手写 KZG/LogUp/PSQ/商多项式证明方案整体替换为 ezkl 的 ONNX 编译、电路 setup、witness 生成、zkSNARK 推理证明、模型权重哈希绑定与评估 attestation 验证流程。
