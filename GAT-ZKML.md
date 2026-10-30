# GAT-ZKML

## 0. 符号

### 0.1 模型结构

 GAT 的前向固定为一个可参数化的多层结构。公开结构参数必须满足 $L\ge2$、$K_{out}\ge1$、$C\ge1$，且每个隐藏层的 $K_{hid}^{(\ell)},d_h^{(\ell)},d_{in}^{(\ell)}$ 均为正整数：

1. 设注意力层的总层数为：$L$。其中前 $L-1$ 层是隐藏层，第 $L$ 层是最终输出层。

2. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，记该层的注意力头数量为 $K_{hid}^{(\ell)}$，每个头的输出维度为 $d_h^{(\ell)}$。该层每个头完成注意力聚合后，逐元素施加 ELU；

	随后把本层全部头的ELU后的输出沿特征维进行拼接，得到：$d_{cat}^{(\ell)}=K_{hid}^{(\ell)}\cdot d_h^{(\ell)}$。

3. 第 $\ell$ 个隐藏层的输入维度记为 $d_{in}^{(\ell)}$。其中 $d_{in}^{(1)}=d_{in}$，且对任意 $\ell\ge 2$，都有 $d_{in}^{(\ell)}=d_{cat}^{(\ell-1)}$。

4. 最终输出层（第 $L$ 层）的注意力头数量记为 $K_{out}$。输出层不是“仿射线性层 + 偏置”替代注意力，而是由 $K_{out}$ 个输出注意力头组成。每个输出注意力头先完成注意力聚合，再加上该头自己的输出偏置向量；若 $K_{out}>1$，则全部输出头的后偏置输出按头维做算术平均，得到最终 logits。

5. 输出注意力头的激活函数固定为恒等映射，不再施加 ELU；但恒等映射的输入是已经加上该输出头偏置后的注意力聚合结果。

6. 后续对隐藏层的明文对象、见证对象、查找、PSQ、路由、一致性绑定约束与商多项式推导，都严格对应“单个隐藏层模板”。

	对任意多层 GAT，证明者与验证者都必须按层序 $\ell=1,\ldots,L-1$ 逐层实例化该模板；第 $\ell-1$ 层的拼接输出 $H_{cat}^{(\ell-1)}$ 必须作为第 $\ell$ 层的唯一输入表示。

7. 协议不预设具体的网络深度、隐藏层头数、单头维度或输出头数。每次证明采用的
   $L,\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1},K_{out}$
   必须由公共元数据 $M_{pub}$ 唯一确定，并作为当前证明实例所绑定的模型架构参数。
   Prover 与 Verifier 必须基于同一 $M_{pub}$ 确定完整的层级结构、各隐藏层多头配置及前后层维度链；
   任意层数、隐藏层头数、单头维度、输出头数或层间输入输出维度与 $M_{pub}$ 不一致的实例，
   均必须在进入正式证明或验证流程前被拒绝。

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

记双线性群为 $(\mathbb G_1,\mathbb G_2,\mathbb G_T,e)$。三群的素数阶均为 $p$，即 KZG 多项式域 $\mathbb F_p$ 必须与曲线标量域完全一致；禁止把基域模数、曲线坐标域模数或其他有限域误用为协议标量域。$e:\mathbb G_1\times\mathbb G_2\to\mathbb G_T$ 是非退化双线性映射，生成元分别为 $G_1\in\mathbb G_1$ 与 $G_2\in\mathbb G_2$。所有外部输入曲线点在进入 transcript 前必须通过规范解码、曲线上点和素数阶子群检查。

正式协议使用 KZG 多项式承诺。对次数不超过 $D$ 的多项式 $P(X)=\sum_{i=0}^{D}p_iX^i\in\mathbb F_p[X]$，数学上的承诺定义为：$$[P]=P(\tau)G_1$$。其中 $\tau\in\mathbb F_p$ 是参数生成阶段采样的隐藏陷门。该公式只定义承诺语义，不允许实现层在证明阶段直接读取、保存、打印、序列化或由 seed 派生 $\tau$。正式实现必须满足：

1. $\tau$ 只在独立 trusted setup 的临时内存中出现；
2. 生成证明键和验证键后，$\tau$ 必须销毁；
3. 证明者只持有 $PK$ 或经过预处理的工作域 SRS；
4. 验证者只持有 $VK_{KZG}$；
5. 主实验代码中不得存在 `seed -> tau`、`tau = seed + c`、`polynomial.evaluate(tau)` 作为承诺主路径的实现。

正式协议统一采用次数上界约定 $\deg P\le D_{max}$。KZG 验证键固定为：

$$VK_{KZG}=\Big\{[1]_2,[\tau]_2,\{[Z_{\mathcal D,n}]_2\}_{(\mathcal D,n)\in domain\_size\_catalog}\Big\},$$

其中 $[Z_{\mathcal D,n}]_2=Z_{\mathcal D,n}(\tau)G_2$ 形成 manifest 中的零化多项式 G2 目录；当前 proof 只按 `domain_cfg` 选择七个对应元素，用于证明长期模型承诺到 proof-local 承诺的工作域零化多项式可除关系。证明键包含普通 coefficient-basis SRS：

$$PK_{coef}=\{[\tau^0]_1,[\tau^1]_1,\ldots,[\tau^{D_{max}}]_1\}.$$

本文只使用单生成元 $G_1$ 下的标准 KZG commitment、标准 KZG opening、七域 quotient 和 CRPC；不得引入第二独立生成元、Pedersen-style 隐藏承诺、IPA、sumcheck 或其他承诺体系。验证键中的任何元素都不包含、恢复或派生 $\tau$。

若多项式以系数形式给出，则承诺实现必须使用：$$[P]=\sum_{i=0}^{D}p_i[\tau^i]_1.$$

为了适配大量工作域插值多项式，SRS 工件必须为 `domain_generator_catalog` 中每个固定了工作域标签、大小、生成元和行顺序的目录项预计算拉格朗日 SRS：

$$PK_{lag}^{(\mathcal D,n)}=\{[L_0^{(\mathcal D,n)}(\tau)]_1,[L_1^{(\mathcal D,n)}(\tau)]_1,\ldots,[L_{n-1}^{(\mathcal D,n)}(\tau)]_1\},\qquad DomainEntry_{\mathcal D,n}\in domain\_generator\_catalog.$$

当前实例选定 $n_{\mathcal D}$ 后，正文简写 $PK_{lag}^{(\mathcal D)}:=PK_{lag}^{(\mathcal D,n_{\mathcal D})}$。

若多项式以工作域$\mathbb H_{\mathcal D}$上的评估形式给出，且 $P(X)=\sum_{i=0}^{n_{\mathcal D}-1}P(\omega_{\mathcal D}^i)L_i^{(\mathcal D)}(X)$，则承诺实现应使用：$$[P]=\sum_{i=0}^{n_{\mathcal D}-1}P(\omega_{\mathcal D}^i)[L_i^{(\mathcal D)}(\tau)]_1$$.

该基于拉格朗日承诺路径是主实验的固定实现，因为它避免在每个 trace 列上重复执行 IFFT，同时不暴露 $\tau$。

#### 0.2.1 隐私、隐藏承诺与掩蔽 opening

正式目标是零知识可验证推理，而不是仅证明计算正确性。所有非公开见证多项式在进入承诺和 opening 之前都必须启用 hiding/blinding 规则。私有工作域多项式采用 domain-vanishing blinding：对定义在工作域 $\mathbb H_{\mathcal D}$ 上的私有多项式 $P$，证明者实际承诺的对象为

$$P^{zk}(X)=P(X)+Z_{\mathcal D}(X)R_P(X).$$

设 $Z_{\mathcal D}(X)R_P(X)=\sum_j u_jX^j$，则使用 Lagrange-basis SRS 与 coefficient-basis SRS 构造同一个隐藏 commitment：

$$[P^{zk}]=\sum_{i=0}^{n_{\mathcal D}-1}P(\omega_{\mathcal D}^i)[L_i^{(\mathcal D)}(\tau)]_1+\sum_j u_j[\tau^j]_1.$$

主实验加载阶段必须同时校验七个工作域的 Lagrange-basis SRS；任一工作域文件缺失、摘要不匹配或与 coefficient-basis SRS 不一致时直接拒绝。普通 canonical trace commitment 的主路径不得退回逐列 IFFT 加 coefficient-basis MSM；coefficient-basis MSM 只用于本身按系数登记的对象、SRS 一致性审计，以及第 0.7.8 节 `KZGShiftRelationProof` 明确定义的 proof-local 平移系数物化。该平移路径不得替代原 canonical commitment 的 Lagrange-basis 承诺，也不得改变原对象的 commitment。

其中 $Z_{\mathcal D}(X)=X^{n_{\mathcal D}}-1$，$R_P(X)$ 是按 `privacy_policy` 规定采样的随机低次数掩蔽多项式。由于 $Z_{\mathcal D}$ 在工作域内恒为 0，$P^{zk}$ 与 $P$ 在工作域上的 trace 值完全一致，不改变所有域内约束语义；在域外 opening 点上，证明者构造并打开完整的 $P^{zk}$，验证者接收 $P^{zk}(z)$ 而不是裸见证值 $P(z)$。

动态 witness 的 $R_P$ 必须在不同对象、层、头、工作域和 proof 之间独立采样。对每个私有多项式，由 `ObservationLedger` 计算其全部 commitments、`MaskedPrivateValue` 和 opening witnesses 在全部 transcript 阶段满足语义覆盖条件所需的最大 blinder 基长度 $q_P^{max}$，诚实证明者固定满足

$$\deg(R_P)+1\ge q_P^{max}.$$

该下界不只计算互异评值点，也包括 commitment、point-grouped witness和平移关系产生的观察。长期模型 commitment 本身不得在单份 proof 中执行私有域外 opening；每份 proof 必须按第 0.7.8 节生成与其工作域取值相同、但使用独立随机性的 proof-local 再盲化 commitment。任何 fresh proof 都不得复用动态 witness 或 proof-local 模型再盲化的随机性。

所有商构造、opening bundle、外点绑定子证明和 transcript 都必须消费 $P^{zk}$ 的承诺与评值。若某个绑定关系需要使用语义值，必须通过第 0.7.8 节的 KZG 同点平移归一化关系、coefficient-tail CRPC 或工作域 quotient 给出，不得把裸折叠向量、裸中间激活、裸注意力权重、裸路由终值或裸模型参数评值写入 proof object。

对 coefficient-basis 私有对象，使用第 0.7.8 节规定的 coefficient-tail blinding；不定义参数域，也不允许把 coefficient-basis 对象视为未绑定的独立见证。对工作域商多项式，使用第 3.3 节规定的校正商和独立商掩码。

公开多项式不需要掩蔽，包括由验证者本地重建的拓扑列、选择器列、索引列、公开元数据和完整最终 logits $Y$。主实验的输出模式固定为 `public_logits`；$P_Y$ 是唯一的动态 `PublicOutput` 多项式，其 commitment 由证明者在输出绑定挑战之前提交，验证者从完整 $Y_{public}$ 计算随机点评值并通过 KZG opening 绑定，逐输出头结果、平均前求和、中间压缩结果和模型参数始终属于私有见证。

随机预言机统一写作 $H_{FS}(\cdot)$。`encoding` 必须固定 `hash_suite_id`、type-tagged length-prefix transcript 编码、hash-to-field 算法、非零挑战拒绝规则和 counter 的字节宽度；禁止使用无长度前缀的字符串拼接、宿主语言默认序列化或不同平台不一致的 hash-to-field。所有 Fiat--Shamir 挑战必须按本文固定顺序生成，禁止重排、回退、漏吸入或重复吸入。`srs_id`、`srs_type`、`privacy_policy`、`proof_task`、`encoding` 和所有 public metadata 必须在任何动态承诺挑战生成之前吸入 transcript。

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

对任意节点索引 $i\in\{0,1,\ldots,N-1\}$，定义其图编号 $node\_gid(i)=g$，其中 $g\in\{0,1,\ldots,G_{batch}-1\}$ 满足 $\nu_g\le i<\nu_{g+1}$

对任意边索引 $k\in\{0,1,\ldots,E-1\}$，定义其图编号 $edge\_gid(k)=g$，其中 $g\in\{0,1,\ldots,G_{batch}-1\}$ 满足 $\epsilon_g\le k<\epsilon_{g+1}$

因此，对任意边索引 $k$，都有：

$src(k),dst(k)\in\{0,1,\ldots,N-1\},\quad node\_gid(src(k))=edge\_gid(k),\quad node\_gid(dst(k))=edge\_gid(k)$

若 $G_{batch}=1$，公共输入仍然必须显式写成 $node\_ptr=(0,N),\quad edge\_ptr=(0,E)$，不得省略，也不得以其他记号替代。

固定要求边序列满足以下两级稳定排序：

1. 先按 $edge\_gid(k)$ 非降序排列，使不同图的边在公开边序列中保持连续块；

2. 在每个固定图块内部，再按目标节点索引 $dst(k)$ 做确定性稳定排序。

等价地，若 $0\le k_1<k_2\le E-1$，则必须满足：$edge\_gid(k_1)<edge\_gid(k_2)$ 或 $\big(edge\_gid(k_1)=edge\_gid(k_2)\ \text{且}\ dst(k_1)\le dst(k_2)\big)$

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

对 `quant_config` 登记的每个有效输入整数，$T_{LReLU},T_{ELU},T_{exp}$ 必须分别存在且只存在一条输出记录；表中不得出现认证输入区间之外的键。$T_{range}$ 必须逐项且仅包含 $0,1,\ldots,2^B-1$。因此查表函数在认证区间上是全函数，不能以缺失行作为隐式 clipping 或异常分支。

指数表的输入语义固定为非负最大值差 $\delta$，输出语义固定为负指数定点映射，而不是正指数映射。对 `quant_config` 认证区间内的每个非负整数 $\delta$，定义

$$(\delta,ExpMap(\delta))\in T_{exp},\qquad ExpMap(\delta):=QuantExpNeg(\delta;quant\_config),$$

其中表生成目标固定为

$$QuantExpNeg(\delta;quant\_config)\approx S_{exp}\exp\left(-\frac{\delta}{S_{\Delta}}\right).$$

上述近似式只说明确定性表生成器的数学目标；协议实际证明的是 `static_tables_id` 所绑定整数表值的精确执行。$T_{exp}$ 的第一列必须按规范顺序编码 $\delta$，第二列编码 $ExpMap(\delta)$。表生成规范必须固定负号、输入解释、近似算法、舍入、clipping、saturation、最小正输出和端点处理。认证区间内必须满足 $ExpMap(0)$ 为最大指数输出，并对任意 $0\le\delta_1<\delta_2$ 满足

$$ExpMap(\delta_1)\ge ExpMap(\delta_2)\ge1.$$

因此后文的 $U_k=ExpMap(M_{dst(k)}-Z_k)$ 唯一表示对 $\exp((Z_k-M_{dst(k)})/S_{\Delta})$ 的认证定点映射；禁止把 `ExpMap` 实现为 $\exp(+\delta/S_{\Delta})$ 或其他符号约定。

本文证明的激活语义固定为由 `static_tables_id` 唯一确定的表驱动定点 GAT。`static_tables_id` 不是任意表文件名，而是下列规范对象的 canonical bytes 摘要：

1. `table_generator_id` 与生成器代码 / 规范摘要；
2. 每张表的输入尺度、输出尺度、规范有符号输入区间和输出区间；
3. LeakyReLU 负半轴斜率的有理数表示、ELU 与指数函数采用的公开近似算法及其参数；
4. clipping、saturation、溢出拒绝、舍入模式和端点处理规则；
5. 表行的唯一排序规则、重复键拒绝规则、canonical field encoding 与完整表摘要。

参数生成器必须从 `quant_config` 和 `table_generator_id` 确定性重建 $T_{LReLU},T_{ELU},T_{exp},T_{range}$，逐字节核对其 canonical bytes 与 `VK_{static}` 中的注册摘要；验证者至少核对 `static_tables_id`、生成器规范摘要、尺度 / 区间证书和 commitments。协议所证明的是该认证表驱动定点网络的精确执行；除非注册规范同时给出并验证了相应误差定理，否则不额外声称表值与未指定的实数 LeakyReLU、ELU 或指数函数完全相等。

与 Softmax、聚合和输出有关的量化尺度全部视为公开参数：

- 差分反量化尺度：$S_{\Delta}$

- 指数输出尺度：$S_{exp}$

- 逆元见证语义标记：$S_{inv}=\texttt{field\_exact}$

- 归一化权重语义标记：$S_{\alpha}=\texttt{field\_exact}$

- 第 $\ell$ 层输入表示尺度：$S_H^{(\ell)}$，其中 $S_H^{(1)}=S_{input}$

- 第 $\ell$ 个隐藏层第 $r$ 个头的权重尺度：$S_W^{(\ell,r)}$

- 第 $\ell$ 个隐藏层第 $r$ 个头的投影输出尺度：$S_{H'}^{(\ell,r)}$

- 隐藏层注意力向量尺度：$S_{a,src}^{(\ell,r)},S_{a,dst}^{(\ell,r)}$

- 隐藏层 ELU 输入尺度：$S_{agg,pre}$

- 隐藏层 ELU 后尺度：$S_{agg}$

- 拼接后表示尺度：$S_{cat}$

- 第 $s$ 个输出头的权重尺度：$S_W^{(out,s)}$

- 第 $s$ 个输出头的投影输出尺度：$S_{Y'}^{(s)}$

- 输出层注意力向量尺度：$S_{a,src}^{(out,s)},S_{a,dst}^{(out,s)}$

- 第 $s$ 个输出偏置尺度：$S_b^{(out,s)}$

- 输出层每个头的聚合前尺度：$S_{out,pre}$

- 输出层每个头的加偏置后尺度：$S_{out,head}$

- 输出层最终多头平均 logits 尺度：$S_{out}$

除固定为 `field_exact` 的 $S_{inv},S_{\alpha}$ 外，全部数值尺度均为正整数。当前前向公式不包含未声明的投影后回缩，因此 `quant_config` 必须满足下列唯一尺度链：

$$S_{H'}^{(\ell,r)}=S_H^{(\ell)}S_W^{(\ell,r)},\qquad
S_{H'}^{(\ell,r)}S_{a,src}^{(\ell,r)}
=S_{H'}^{(\ell,r)}S_{a,dst}^{(\ell,r)}=S_{\Delta},$$

$$S_{cat}=S_{agg},\qquad
S_H^{(\ell+1)}=S_{cat}\quad(1\le\ell<L),$$

$$S_{Y'}^{(s)}=S_H^{(L)}S_W^{(out,s)},\qquad
S_{Y'}^{(s)}S_{a,src}^{(out,s)}
=S_{Y'}^{(s)}S_{a,dst}^{(out,s)}=S_{\Delta},\qquad
S_b^{(out,s)}=S_{out,head}.$$

所有隐藏头和输出头的源 / 目标打分及差分都处于共同尺度 $S_{\Delta}$，因而允许直接相加并复用同一个 $T_{LReLU}$ 与 $T_{exp}$。`quant_config` 固定 $T_{LReLU}$ 的输入 / 输出尺度均为 $S_{\Delta}$、$T_{exp}$ 的输入尺度为 $S_{\Delta}$ 且输出尺度为 $S_{exp}$、$T_{ELU}$ 的输入尺度为 $S_{agg,pre}$ 且输出尺度为 $S_{agg}$。若某个模型需要投影后回缩、不同注意力向量尺度对齐或不同隐藏头输出尺度对齐，则必须在模型导出阶段预先量化为满足上述尺度链的整数参数；本文证明语义不允许在实现中隐式插入未被约束的回缩步骤。

舍入方式、每个中间量的有符号范围和无模回绕上界由 $quant\_cfg\_id$ 唯一确定。有限域元素用于承载这些整数关系，但任何整数除法、比较、范围与舍入关系都必须同时证明其规范有符号代表和无模回绕条件，不能仅验证模 $p$ 等式。验证者在吸入动态 commitment 前重算并检查全部尺度等式；任一尺度缺失、为零或链路不一致时拒绝。

**有限域 Softmax 关系与定点输出关系**

有限域 Softmax 关系定义为：

$$Sum_i\cdot inv_i=1,$$

$$\alpha_k=U_k\cdot inv_{dst(k)}.$$

其中 $inv_i$ 和 $\alpha_k$ 是有限域中的辅助见证，用于证明分母非零以及有限域 Softmax 代数关系。它们不直接作为 ELU 表、偏置加法或最终 logits 的定点整数输入。

对隐藏层第 $(\ell,r)$ 个头，定义整数聚合分子：

$$H_{agg,num,i,j}^{(\ell,r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(\ell,r)}H_{src(k),j}'^{(\ell,r)}.$$

有限域聚合值定义为：

$$H_{agg,pre,i,j}^{(\ell,r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(\ell,r)}H_{src(k),j}'^{(\ell,r)}.$$

由 $Sum_i inv_i=1$ 可得：

$$H_{agg,num,i,j}^{(\ell,r)}=Sum_i^{(\ell,r)}H_{agg,pre,i,j}^{(\ell,r)}$$

在 $\mathbb F_p$ 中成立。ELU 的定点整数输入定义为：

$$H_{ELU,in,i,j}^{(\ell,r)}=
SignedRoundDiv\!\left(
S_{agg,pre}\,H_{agg,num,i,j}^{(\ell,r)},
S_{H'}^{(\ell,r)}\,Sum_i^{(\ell,r)}
\right).$$

指数输出中的尺度 $S_{exp}$ 同时出现在聚合分子与分母中，因而在该比值中抵消。只有由上述整数除法关系得到的 $H_{ELU,in}^{(\ell,r)}$ 才允许进入 $T_{ELU}$。

对输出层第 $s$ 个头，定义：

$$Y_{agg,num,i,c}^{(s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)}Y_{src(k),c}'^{(s)},$$

$$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)},$$

并证明：

$$Y_{agg,num,i,c}^{(s)}=Sum_i^{(out,s)}Y_{i,c}^{pre(s)}.$$

输出层聚合结果在尺度 $S_{out,pre}$ 上的整数表示定义为：

$$Y_{i,c}^{q(s)}=
SignedRoundDiv\!\left(
S_{out,pre}\,Y_{agg,num,i,c}^{(s)},
S_{Y'}^{(s)}\,Sum_i^{(out,s)}
\right).$$

在与偏置相加前，将该结果对齐到 $S_{out,head}$：

$$\widetilde Y_{i,c}^{pre(s)}=
SignedRoundDiv\!\left(
S_{out,head}\,Y_{i,c}^{q(s)},
S_{out,pre}
\right),$$

$$Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}.$$

当 $S_{out,pre}=S_{out,head}$ 时，第二个除法关系退化为恒等关系。

最终多头平均发生在共同尺度 $S_{out,head}$ 上，并保持 $S_{out}=S_{out,head}$。定义：

$$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)},$$

$$Y_{i,c}=SignedRoundDiv\!\left(Y_{i,c}^{sum},K_{out}\right).$$

最终平均不得用 $K_{out}^{-1}Y_{i,c}^{sum}$ 的有限域乘法替代，也不得在该平均之后增加未由 $quant\_cfg\_id$ 声明的回缩步骤。

**$SignedRoundDiv$ 见证与约束**

对带符号整数 $x$ 和正整数 $d$，$y=SignedRoundDiv(x,d)$ 使用见证

$$(\sigma,z,a,a^{-1},d^{-1},q,r,\delta),$$

其中 $\sigma,z\in\{0,1\}$，$a,q,r,\delta$ 是非负整数，并满足：

$$x=(1-2\sigma)a,$$

$$y=(1-2\sigma)q,$$

$$z(z-1)=0,$$

$$za=0,$$

$$a\cdot a^{-1}=1-z,$$

$$z\sigma=0,$$

$$z\cdot a^{-1}=0,$$

$$d\cdot d^{-1}=1.$$

因此 $z=1$ 当且仅当输入绝对值 $a=0$，且输入零只有 $\sigma=0,a^{-1}=0$ 这一种规范辅助表示；当 $a\ne0$ 时，$a^{-1}$ 由 $aa^{-1}=1$ 唯一确定。当非零输入经过舍入得到 $q=0$ 时，输出域元素仍唯一为 $0$；$\sigma$ 仅编码输入符号，不作为零输出的独立编码。

当公开舍入方式为向零截断时，要求：

$$a=dq+r,$$

$$r+\delta=d-1.$$

当公开舍入方式为向最近整数舍入且半数远离零时，要求：

$$2a+d=2dq+r,$$

$$r+\delta=2d-1.$$

两种模式均通过 $r,\delta\ge0$ 保证余数范围。分母 $d$ 必须为正整数；当 $d$ 是动态分母时，还要提供其非零逆元见证并证明 $d\cdot d^{-1}=1$。

范围证明统一采用以 $B$ 位为基数的 limb 分解，不存在未登记的替代范围协议。对任一非负范围对象 $V$，`quant_cfg_id` 固定公开 limb 数 $\ell_V\ge1$，证明者生成与 $V$ 同索引的 limb 列 $V^{[0]},\ldots,V^{[\ell_V-1]}$，并满足

$$V=\sum_{h=0}^{\ell_V-1}2^{Bh}V^{[h]},\qquad 0\le V^{[h]}<2^B.$$

每个 $V^{[h]}$ 独立通过 $T_{range}$ 的 LogUp 实例证明；重建等式进入同一工作域商多项式。若 $\ell_V=1$，该规则退化为对 $V$ 的直接范围查表。所有高位未使用区域固定为零；公开上界统一按 $0\le V<2^{B\ell_V}$ 计算。$a,d-1,q,r,\delta$、各尺度乘积、聚合分子和需要整数解释的累加结果都必须登记对应 limb 数。

每个 $SignedRoundDiv$ 实例在其自然工作域上生成以下多项式约束。记有效区选择器为 $Q_{div}$，相应多项式统一记为 $P_x,P_d,P_y,P_{\sigma},P_z,P_a,P_{a^{-1}},P_{d^{-1}},P_q,P_r,P_{\delta}$，则：

$$C_{div,\sigma}(X)=P_{Q_{div}}(X)P_{\sigma}(X)(P_{\sigma}(X)-1),$$

$$C_{div,abs}(X)=P_{Q_{div}}(X)\big(P_x(X)-(1-2P_{\sigma}(X))P_a(X)\big),$$

$$C_{div,zbin}(X)=P_{Q_{div}}(X)P_z(X)(P_z(X)-1),$$

$$C_{div,zero}(X)=P_{Q_{div}}(X)P_z(X)P_a(X),$$

$$C_{div,nz}(X)=P_{Q_{div}}(X)\big(P_a(X)P_{a^{-1}}(X)-(1-P_z(X))\big),$$

$$C_{div,zsign}(X)=P_{Q_{div}}(X)P_z(X)P_{\sigma}(X),$$

$$C_{div,zinv}(X)=P_{Q_{div}}(X)P_z(X)P_{a^{-1}}(X),$$

$$C_{div,dnz}(X)=P_{Q_{div}}(X)\big(P_d(X)P_{d^{-1}}(X)-1\big),$$

$$C_{div,out}(X)=P_{Q_{div}}(X)\big(P_y(X)-(1-2P_{\sigma}(X))P_q(X)\big).$$

向零截断模式使用：

$$C_{div,eq}(X)=P_{Q_{div}}(X)\big(P_a(X)-P_d(X)P_q(X)-P_r(X)\big),$$

$$C_{div,rem}(X)=P_{Q_{div}}(X)\big(P_r(X)+P_{\delta}(X)-P_d(X)+1\big).$$

向最近整数舍入且半数远离零模式使用：

$$C_{div,eq}(X)=P_{Q_{div}}(X)\big(2P_a(X)+P_d(X)-2P_d(X)P_q(X)-P_r(X)\big),$$

$$C_{div,rem}(X)=P_{Q_{div}}(X)\big(P_r(X)+P_{\delta}(X)-2P_d(X)+1\big).$$

这些约束与相应的范围 LogUp 约束必须进入该工作域的商多项式。所有合法配置必须保证上述整数等式的两侧及中间乘积的绝对值严格小于 $p/2$；不满足无模回绕界的实例必须在生成动态承诺前拒绝。

**工作域放置**

不增加新的主工作域。对于隐藏层 $N\times d_h^{(\ell)}$ 和输出层 $N\times C$ 的量化关系，使用行优先展平索引并放置在 $\mathbb H_{edge}$：

- 隐藏层第 $(\ell,r)$ 个头使用有效区 $0\le q<Nd_h^{(\ell)}$；

- 输出层各头和最终平均使用有效区 $0\le q<NC$。

因此 $n_{edge}$ 必须同时覆盖 $Nd_h^{max}$ 与 $NC$。对应的公开有效区选择器、广播后的 $Sum$、有限域聚合值、整数聚合分子、定点输出以及 $SignedRoundDiv$ 辅助列都在 $\mathbb H_{edge}$ 上插值。其行优先 coefficient-basis 多项式与这些 evaluation trace 的一致性由第 0.7.8 节的张量绑定子证明验证。

`quant_cfg_id` 必须携带规范的整数上界登记表 `integer_bound_ledger`。记公开图的最大入度为

$$\Delta_{in}^{max}=\max_i\#\{k\mid dst(k)=i\},$$

指数表输出上界为 $B_U=\max T_{exp}[t,1]$。对每个隐藏层和输出头，登记并验证下列保守上界：

$$B_{H'}^{(\ell,r)}=d_{in}^{(\ell)}B_{H,in}^{(\ell)}B_W^{(\ell,r)},$$

$$B_{Sum}^{(\ell,r)}=\Delta_{in}^{max}B_U,\qquad B_{H,num}^{(\ell,r)}=\Delta_{in}^{max}B_U B_{H'}^{(\ell,r)},$$

$$B_{Y'}^{(s)}=d_{in}^{(L)}B_{H,cat}B_W^{(out,s)},$$

$$B_{Sum}^{(out,s)}=\Delta_{in}^{max}B_U,\qquad B_{Y,num}^{(s)}=\Delta_{in}^{max}B_U B_{Y'}^{(s)}.$$

其中 $B_{H,in}^{(1)}$ 由公开输入特征的规范有符号代表确定；后续隐藏层输入上界由上一层 $T_{ELU}$ 输出上界和拼接规则确定；$B_{H,cat}$ 由最后一个隐藏层输出上界确定；$B_W^{(\ell,r)},B_W^{(out,s)}$、注意力向量上界与偏置上界由模型登记阶段验证并绑定到 $VK_{model}$ 摘要。

对注意力打分和查表输入还必须登记

$$B_{E,src}^{(\ell,r)}=d_h^{(\ell)}B_{H'}^{(\ell,r)}B_{a,src}^{(\ell,r)},\qquad
B_{E,dst}^{(\ell,r)}=d_h^{(\ell)}B_{H'}^{(\ell,r)}B_{a,dst}^{(\ell,r)},$$

$$B_S^{(\ell,r)}=B_{E,src}^{(\ell,r)}+B_{E,dst}^{(\ell,r)},$$

并对输出头登记相同形式的 $B_{E,src}^{(out,s)},B_{E,dst}^{(out,s)},B_S^{(out,s)}$。`integer_bound_ledger` 必须同时覆盖 $T_{LReLU}$ 输出、组最大值、差分、指数表输入 / 输出、偏置相加和拼接后的全部规范有符号代表；这些界与静态表声明的输入 / 输出区间必须一致。

登记表还必须给出 $S_{agg,pre}B_{H,num}$、$S_{H'}B_{Sum}$、$S_{out,pre}B_{Y,num}$、$S_{Y'}B_{Sum}$、尺度对齐乘积、偏置加法、$K_{out}$ 头求和以及两种舍入等式两侧的上界。全部上界必须严格小于 $(p-1)/2$；每个范围对象的 $B\ell_V$ 位上界也必须严格小于 $(p-1)/2$。参数生成在提交模型 commitments 前检查模型参数上界，证明生成在提交动态 commitments 前检查公开输入、图入度、静态表和维度上界；验证者根据公共输入、`quant_cfg_id` 和 $VK_{model}$ 中的上界摘要重复检查。任一检查失败时必须拒绝。

在这些上界下，$H_{agg,num}$ 与 $Y_{agg,num}$ 的有限域值分别等于其规范有符号整数求和结果，不存在模回绕；因此第 0.4 节规定的 $SignedRoundDiv$ 关系具有唯一整数语义。

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

本文将官方注意力向量按源/目标方向分拆为等价写法：$\mathbf a^{(out,s)\top}[u\,\|\,v] = a_{src}^{(out,s)\top}u + a_{dst}^{(out,s)\top}v$。其中 $[u\,\|\,v]$ 表示拼接。这只是代数重写，不改变官方 GAT 的前向语义。

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

8. 有限域聚合前隐藏矩阵：$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}$

9. 整数聚合分子：$H_{agg,num,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}H_{src(k),j}'^{(r)}=Sum_i^{(r)}H_{agg,pre,i,j}^{(r)}$

10. ELU 定点输入：$H_{ELU,in,i,j}^{(r)}=SignedRoundDiv\!\left(S_{agg,pre}H_{agg,num,i,j}^{(r)},S_{H'}^{(\ell,r)}Sum_i^{(r)}\right)$

11. 聚合前压缩特征：$H_{agg,pre,i}^{\star(r)}\in\mathbb F_p^N \quad H_{agg,pre,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,pre,i,j}^{(r)}(\xi^{(r)})^j$

12. ELU 后隐藏矩阵：$H_{agg}^{(r)}\in\mathbb F_p^{N\times d_h} \quad H_{agg,i,j}^{(r)}=ELU(H_{ELU,in,i,j}^{(r)})$

13. ELU 后压缩特征：$H_{agg}^{\star(r)}\in\mathbb F_p^N \quad H_{agg,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,i,j}^{(r)}(\xi^{(r)})^j$


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

13. 有限域归一化权重：$\alpha^{(r)}\in\mathbb F_p^E \qquad \alpha_k^{(r)}=U_k^{(r)}\cdot inv_k^{edge(r)}$。该对象用于有限域 Softmax 与 PSQ 关系，不直接作为 ELU 的定点整数输入

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
20. 输出层有限域聚合前输出：$Y^{pre(s)}\in\mathbb F_p^{N\times C}, \qquad Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}$
21. 输出层整数聚合分子：$Y_{agg,num,i,c}^{(s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)}Y_{src(k),c}'^{(s)}=Sum_i^{(out,s)}Y_{i,c}^{pre(s)}$
22. 输出层聚合前定点结果：$Y_{i,c}^{q(s)}=SignedRoundDiv\!\left(S_{out,pre}Y_{agg,num,i,c}^{(s)},S_{Y'}^{(s)}Sum_i^{(out,s)}\right)$
23. 输出层聚合前输出的类别压缩：$Y^{pre,\star(s)}\in\mathbb F_p^N, \qquad Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{pre(s)}\xi_{out}^c$
24. 输出层聚合前输出压缩广播：$Y^{pre,\star,edge(s)}\in\mathbb F_p^E, \qquad Y_k^{pre,\star,edge(s)}=Y_{dst(k)}^{pre,\star(s)}$
25. 输出层聚合前定点结果的类别压缩：$Y^{q,\star(s)}\in\mathbb F_p^N, \qquad Y_i^{q,\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{q(s)}\xi_{out}^c$
26. 尺度对齐后的聚合前输出：$\widetilde Y^{pre(s)}\in\mathbb F_p^{N\times C}$，其中：$\widetilde Y_{i,c}^{pre(s)}=SignedRoundDiv\!\left(S_{out,head}Y_{i,c}^{q(s)},S_{out,pre}\right)$

27. 尺度对齐后的聚合前输出类别压缩：$\widetilde Y^{pre,\star(s)}\in\mathbb F_p^N, \qquad \widetilde Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}\widetilde Y_{i,c}^{pre(s)}\xi_{out}^c$
28. 输出层加偏置后的头输出：$Y^{(s)}\in\mathbb F_p^{N\times C}, \qquad Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}$

29. 输出层头输出的类别压缩：$Y^{\star(s)}\in\mathbb F_p^N, \qquad Y_i^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{(s)}\xi_{out}^c$
30. 输出层头输出压缩广播：$Y^{\star,edge(s)}\in\mathbb F_p^E, \qquad Y_k^{\star,edge(s)}=Y_{dst(k)}^{\star(s)}$
31. 最终输出：先定义 $Y^{sum}\in\mathbb F_p^{N\times C}$ 与整数平均 $Y^{avg}\in\mathbb F_p^{N\times C}$：$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}, \qquad Y_{i,c}^{avg}=SignedRoundDiv\!\left(Y_{i,c}^{sum},K_{out}\right)$

本文协议要求最终 logits 与输出头使用同一共同尺度 $S_{out}=S_{out,head}$，因此正式定义 $Y\in\mathbb F_p^{N\times C}, \quad Y_{i,c}=Y_{i,c}^{avg}$。

32. 最终输出的类别压缩：$Y^{\star}\in\mathbb F_p^N, \qquad Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c$

33. 最终输出压缩广播：$Y^{\star,edge}\in\mathbb F_p^E, \qquad Y_k^{\star,edge}=Y_{dst(k)}^{\star}$

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


对任意工作域 $\mathbb H_{\mathcal D}$，都记其大小为 $n_{\mathcal D}$，生成元为 $\omega_{\mathcal D}$，零化多项式为 $Z_{\mathcal D}(X)=X^{n_{\mathcal D}}-1$。$(n_{\mathcal D},\omega_{\mathcal D})$ 必须来自 `domain_cfg` 引用的 manifest 目录项；实例注册和加载阶段验证 $n_{\mathcal D}\mid(p-1)$、$\omega_{\mathcal D}^{n_{\mathcal D}}=1$，且对 $n_{\mathcal D}$ 的每个素因子 $q$ 都有 $\omega_{\mathcal D}^{n_{\mathcal D}/q}\ne1$。实现不得为同一大小现场选择另一生成元或另一行顺序。全部工作域大小必须小于 $p$，并满足 $2^B<p$。

#### 0.7.2 拉格朗日基函数与首尾指示多项式

对任意工作域 $\mathbb H_{\mathcal D}$，记其拉格朗日基函数为：$L_0^{(\mathcal D)}(X),L_1^{(\mathcal D)}(X),\ldots,L_{n_{\mathcal D}-1}^{(\mathcal D)}(X)$。（特点：在域的第$i$取值$1$，在其他点上取值$0$）

首点指示多项式记为：$First_{\mathcal D}(X)=L_0^{(\mathcal D)}(X)$

末点指示多项式记为：$Last_{\mathcal D}(X)=L_{n_{\mathcal D}-1}^{(\mathcal D)}(X)$

末点对应的公开工作域点记为：

$$z_{last,\mathcal D}=\omega_{\mathcal D}^{n_{\mathcal D}-1}.$$

#### 0.7.3 公共有效区选择器与按层实例化规则

1. 边域有效区选择器：$Q_{edge}^{valid}[k]= \begin{cases} 1 &0\le k\le E-1,\\ 0,&E\le k\le n_{edge}-1 \end{cases}$

2. 节点域有效区选择器：$Q_N[i]= \begin{cases} 1 &0\le i\le N-1,\\ 0,&N\le i\le n_N-1 \end{cases}$

3. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，输入共享维域有效区选择器族定义为：$Q_{in}^{valid,(\ell)}[m]= \begin{cases} 1 &0\le m\le d_{in}^{(\ell)}-1 \\ 0 &d_{in}^{(\ell)}\le m\le n_{in}-1 \end{cases}$

4. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，隐藏层单头共享维域有效区选择器族定义为：$Q_{d_h}^{valid,(\ell)}[j]= \begin{cases} 1 &0\le j\le d_h^{(\ell)}-1 \\ 0 &d_h^{(\ell)}\le j\le n_{d_h}-1 \end{cases}$

5. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，拼接共享维域有效区选择器族定义为：$Q_{cat}^{valid,(\ell)}[m]= \begin{cases} 1 &0\le m\le d_{cat}^{(\ell)}-1 \\ 0 &d_{cat}^{(\ell)}\le m\le n_{cat}-1 \end{cases}$

6. 输出层输入共享维有效区选择器定义为：$Q_{cat}^{valid,(out)}[m]= \begin{cases} 1 &0\le m\le d_{in}^{(L)}-1,\\ 0,&d_{in}^{(L)}\le m\le n_{cat}-1 \end{cases}$

   虽然 $d_{in}^{(L)}=d_{cat}^{(L-1)}$，正式协议仍然固定把 $Q_{cat}^{valid,(out)}$ 与 $Q_{cat}^{valid,(L-1)}$ 视为两个不同的公共对象。序列化器、解析器、transcript、商多项式构造器与验证者都必须将它们按两个不同的标签处理不得复用，不得合并，也不得依赖字节串相等来省略其中之一。

7. 输出层类别共享维域有效区选择器：$Q_C^{valid}[c]= \begin{cases} 1 &0\le c\le C-1,\\ 0,&C\le c\le n_C-1 \end{cases}$

8. 对每个隐藏层 $\ell$，定义行优先隐藏矩阵有效区选择器：$Q_{Nd_h}^{valid,(\ell)}[q]=1$ 当且仅当 $0\le q<Nd_h^{(\ell)}$，其余位置为 $0$。该选择器定义在 $\mathbb H_{edge}$ 上。

9. 定义输出矩阵有效区选择器：$Q_{NC}^{valid}[q]=1$ 当且仅当 $0\le q<NC$，其余位置为 $0$。该选择器定义在 $\mathbb H_{edge}$ 上。

10. 在当前隐藏层模板的数学推导中，统一采用以下局部记号：$Q_{in}^{valid}:=Q_{in}^{valid,(\ell)} \quad Q_{d_h}^{valid}:=Q_{d_h}^{valid,(\ell)} \quad
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

因此，代码实现、序列化器、商多项式构造器与验证者/解析器中，凡是“组起点 / 组末尾”相关的状态机，都必须以 $Q_{grp\_new}^{edge}$、$Q_{grp\_end}^{edge}$ 为唯一正式输入；即使 $G_{batch}=1$，实现也必须使用这两个批处理感知标签，而不得使用未带 batch 语义的 $Q_{new}^{edge},Q_{end}^{edge}$ 作为正式状态机标签。

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

公共辅助列的填充模式固定分为两类：

- `FullDomainEnum`：$Idx_N,Idx_{in},Idx_{d_h},Idx_{cat},Idx_C$ 在各自整个工作域上保持 $0,1,\ldots,n_{\mathcal D}-1$ 的枚举语义，不执行零填充；
- `PublicZeroPad`：只在真实有效支撑上有语义的公共数据列和索引辅助列，在无效区统一取零。该类至少包括 $src,dst,node\_gid,edge\_gid,Row_{feat}^{tbl},Col_{feat}^{tbl},Row_{feat}^{qry},Col_{feat}^{qry},I_{feat}^{qry}$，以及 `ProofSchema` 登记的其他公开 row-major、batch 和 lookup helper。具体地，$node\_gid[i]=0$ 对 $N\le i<n_N$，$edge\_gid[k]=0$ 对 $E\le k<n_{edge}$；特征检索辅助列在各自表端或查询端有效长度之外均取零。

选择器列按其定义在无效区取零，不归入 `FullDomainEnum`。`padding_rule` 必须逐对象登记上述分类，`protocol_schema_id` 绑定完整公共列填充注册表；prover、serializer 和 verifier 不得使用最后一个有效值、宿主语言默认值或任意哨兵替代规范零值。

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

所有形如 $a_t=\sum_i M_{i,t}\chi_i(y)$、$b_t=\sum_j W_{t,j}\psi_j(y)$ 的一维折叠向量，以及 CRPC 使用的 coefficient-basis relation view，都必须与其来源 tensor commitment 强绑定。

绑定子证明族固定为：

$$\Pi_{bind}= \big( \pi_{bind}^{feat},\ \{\pi_{bind}^{hidden,(\ell,r)}\}_{\ell,r\ \text{按层序与头序}},\ \{\pi_{bind}^{concat,(\ell)}\}_{\ell=1}^{L-1},\ \pi_{bind}^{out,0},\ldots,\pi_{bind}^{out,K_{out}-1},\ \pi_{bind}^{out,avg} \big).$$

其中：

- 对每个隐藏层 $\ell$、每个隐藏头 $r$，$\pi_{bind}^{hidden,(\ell,r)}$ 负责该头的投影、源注意力、目标注意力、压缩、有限域聚合、ELU 输入量化与聚合输出绑定；

- 对每个隐藏层 $\ell$，$\pi_{bind}^{concat,(\ell)}$ 负责该层各头输出、拼接矩阵与压缩对象的一致性绑定；

- 对每个输出头 $s$，$\pi_{bind}^{out,s}$ 负责输出投影、源注意力、目标注意力、有限域聚合、定点回缩、私有偏置和输出压缩绑定；

- $\pi_{bind}^{out,avg}$ 负责各输出头、最终整数平均、最终完整 logits $Y$ 及其压缩对象的一致性绑定。

$\Pi_{bind}$ 只使用本文的 KZG、七个工作域、LogUp、CRPC、商多项式和 Fiat--Shamir，不引入其他承诺体系或其他内积证明系统。

**KZG 同点平移归一化线性关系**

设公开常数、公开点、公开系数和已承诺多项式给出待证线性关系。构造 `RelationMeta` 前，验证者必须删除系数为零的项，并把同一点上的全部项合并为一个规范线性组合多项式 $F_j$，得到

$$\lambda_0+\sum_{j=1}^{m}F_j(z_j)=0,$$

其中规范化后的 $z_1,\ldots,z_m$ 两两互异，每个 $[F_j]$ 由来源 commitments 与公开系数唯一线性重建，外层系数固定为 $1$。$[F_j]$ 是 `MetadataReference` 指向的虚拟线性组合 commitment，不在 proof 中重复序列化；prover 按同一公开线性组合构造其 opening polynomial。规范点序按 `encoding` 的 field-element canonical bytes 升序确定，不允许证明者选择。

当 $m=0$ 时，验证者直接检查 $\lambda_0=0$，proof 不得携带关系字段。

当 $m=1$ 时，令 $G_{rel}(X)=\lambda_0+F_1(X)$，并把标准零值 claim

$$([G_{rel}],z_1,0)$$

登记到 $z_1$ 所属的普通 `StandardPointGroup`。proof 不生成关系专用 witness，实际 witness 只出现在该点唯一的 point-grouped KZG opening 中。

当 $m\ge2$ 时，规范公共枢轴固定为

$$z_\star=z_1.$$

令 $\Delta_1=0$、$\widetilde F_1=F_1$；对每个 $j=2,\ldots,m$，定义平移量与平移多项式

$$\Delta_j=z_j-z_\star,$$

$$\widetilde F_j(X)=F_j(X+\Delta_j).$$

平移保持次数不变，即 $\deg\widetilde F_j=\deg F_j\le D_{max}$。证明者使用普通 $PK_{coef}$ 提交 $[\widetilde F_j]$；不得为该关系加载或生成第二套 SRS。$\widetilde F_j$ 必须由实际已承诺的隐藏多项式 $F_j$ 平移得到，不能改用未掩蔽语义多项式。

在全部来源 commitments、$[\widetilde F_j]$、关系标签、点序和 $\Delta_j$ 固定后，按关系标签生成承诺后挑战 $y_{rel}$。挑战使用规范公开 counter，直至同时满足：

1. 对每个 $j\ge2$，$y_{rel}$ 与 $y_{rel}+\Delta_j$ 均采用 canonical field encoding；
2. 对 $F_j$ 中每个采用 domain-vanishing blinding 的私有来源对象 $P^{zk}=P+Z_{\mathcal D}R_P$，都有 $Z_{\mathcal D}(y_{rel}+\Delta_j)\ne0$；
3. 对每个 coefficient-tail 私有来源对象 $\widehat P=P+X^LR$，都有 $y_{rel}+\Delta_j\ne0$；
4. 满足 `RandomIdentityLedger` 为该平移关系登记的全部其他公开排除条件。

令 $E_{rel}(Y)$ 为上述公开排除多项式去重后的乘积，$B_{rel}=\deg E_{rel}$，$d_{rel}^{max}$ 为账本按全局 $D_{max}$ 计算的平移差多项式最坏次数。`RelationMeta` 必须登记并由验证者重建 $(E_{rel},B_{rel},d_{rel}^{max})$，参数生成必须检查

$$d_{rel}^{max}<p-B_{rel}.$$

规范候选集合为

$$\mathcal A_{rel}=\{y\in\mathbb F_p:E_{rel}(y)\ne0\},\qquad |\mathcal A_{rel}|\ge p-B_{rel}>d_{rel}^{max}.$$

`ObservationLedger` 中仍依赖形式陷门 $\mathsf T$ 的秩退化不进入公开 counter；它属于第 1.4 节定义的 `BadCRS_obs`，并只在零知识安全界中计入。

$y_{rel}$ 通过随机预言机在 $\mathcal A_{rel}$ 上执行公开拒绝采样。上述条件只依赖公开 metadata、来源对象的隐藏类型、关系点和候选挑战，验证者能够独立重放。点固定后若 $[\tau-y_{rel}]_2$ 或任一 $[\tau-(y_{rel}+\Delta_j)]_2$ 为单位元，必须返回 `SRS_COMPROMISED` 并撤销该 SRS，不得通过 counter 改点继续使用。

对每个 $j=2,\ldots,m$，证明者计算唯一的平移检查值

$$u_{rel,j}=\widetilde F_j(y_{rel})=F_j(y_{rel}+\Delta_j).$$

若 $F_j$ 含私有来源，则 $u_{rel,j}$ 的字段类型固定为 `MaskedPrivateValue`；仅当 $F_j$ 完全由公开多项式构成时才允许标记为 `PublicValue`。该字段在 proof 中只序列化一次，并被下列两个普通 KZG claims 共同引用：

$$([\widetilde F_j],y_{rel},u_{rel,j}),$$

$$([F_j],y_{rel}+\Delta_j,u_{rel,j}).$$

两项 claim 分别进入其所在点的唯一 `StandardPointGroup`。若

$$D_j(X)=\widetilde F_j(X)-F_j(X+\Delta_j)$$

不是零多项式，则在 commitments 固定后从 $\mathcal A_{rel}$ 生成的 $y_{rel}$ 上通过检查的概率至多为

$$\frac{\deg D_j}{|\mathcal A_{rel}|}\le\frac{\deg D_j}{p-B_{rel}}.$$

因此全部平移检查通过时，除该 Schwartz--Zippel 可忽略错误外，均有

$$\widetilde F_j(z_\star)=F_j(z_j).$$

最后定义普通 KZG 关系多项式

$$G_{rel}(X)=\lambda_0+\sum_{j=1}^{m}\widetilde F_j(X),$$

其 commitment 由验证者本地线性重建：

$$[G_{rel}]=\lambda_0G_1+\sum_{j=1}^{m}[\widetilde F_j].$$

proof 把零值 claim

$$([G_{rel}],z_\star,0)$$

登记到 $z_\star$ 的唯一 `StandardPointGroup`。验证者通过标准 KZG 方程验证该零值 claim。结合全部平移检查，可得

$$0=G_{rel}(z_\star)=\lambda_0+\sum_{j=1}^{m}F_j(z_j).$$

对 $m\ge2$，逻辑字段固定为

$$KZGShiftRelationProof=\Big(\{[\widetilde F_j]\}_{j=2}^{m},\{u_{rel,j}\}_{j=2}^{m},MetadataReference(\mathcal O_{rel}^{std})\Big),$$

其中 $\mathcal O_{rel}^{std}$ 是上述平移检查 claims 与关系零值 claim 在普通 point-grouped opening registry 中的规范引用集合。`RelationMeta` 固定来源聚合映射、规范点序、$z_\star$、$\Delta_j$、来源隐藏类型、$y_{rel}$ 域分离标签和 opening 顺序。proof 不编码 $F_j(z_j)$、裸 route terminal、裸折叠结果或关系专用局部 quotient。

平移 commitment、$u_{rel,j}$ 及其 opening witnesses 对来源 blinder 的全部代数依赖必须进入 `ObservationLedger`；平移不会建立新的独立随机源。所有私有来源对象的 blinding 预算必须覆盖这些额外观察。该构造只使用普通单生成元 KZG commitments、普通 KZG openings 与现有 transcript，不增加工作域、顶层 proof block 或独立证明系统。

**平移 commitment 的唯一实现路径**

主实验固定 `shift_algorithm_id = coefficient_taylor_shift`。对每个关系和每个 $j\ge2$，prover 必须按以下顺序构造 $[\widetilde F_j]$：

1. 按 `RelationMeta` 的来源顺序取得实际 blinded source polynomial。若来源原生为工作域 evaluation trace，则先使用其已缓存的原生工作域 IFFT plan，把域内语义取值唯一物化为次数小于 $n_{\mathcal D}$ 的插值多项式 $P$；随后使用该对象在本 proof 中已经登记的 blinder coefficient，显式相加 $Z_{\mathcal D}R_P$，得到完整的 $P^{zk}$ coefficient vector。不得试图仅由工作域评值恢复 $Z_{\mathcal D}R_P$。coefficient-tail 对象直接使用其完整低阶语义区、gap-zero 区和随机尾 coefficient；proof-local 模型对象使用注册时保存的模型 coefficient 与本 proof 的 $Z_{\mathcal D}S_{model}$ 合成。相同 `(proof_nonce,object_id)` 只允许物化一次并缓存。该 IFFT 仅服务于平移关系，不能替代原对象使用 Lagrange SRS 的 canonical commitment。
2. 实例注册器必须从 `degree_resource_registry` 为该关系固定公开次数上界 $d_{shift,j}^{cfg}$。证明者按公开线性系数合并来源对象后，必须物化固定长度 $d_{shift,j}^{cfg}+1$ 的 coefficient vector：
   $$F_j(X)=\sum_{i=0}^{d_{shift,j}^{cfg}}f_iX^i.$$
   若语义多项式的高位系数为零，则在固定向量中规范补零；禁止根据当前 witness、proof-local blinder、最高非零系数或线性组合抵消结果裁剪 coefficient 长度。对公开平移量 $\Delta_j$，固定计算
   $$a_i=f_i\,i!,\qquad a_i^{rev}=a_{d_{shift,j}^{cfg}-i},\qquad b_i=\frac{\Delta_j^i}{i!},$$
   对 $a^{rev}$ 与 $b$ 做一次卷积得到 $c$，并输出
   $$\widetilde f_k=\frac{c_{d_{shift,j}^{cfg}-k}}{k!},\qquad k=0,\ldots,d_{shift,j}^{cfg}.$$
   由此唯一得到 $\widetilde F_j(X)=F_j(X+\Delta_j)$。factorial 与 inverse-factorial 表按 $0,\ldots,D_{max}$ 预计算并由 `protocol_schema_id` 绑定。
3. 该关系的卷积临时域固定为
   $$n_{shift,j}^{cfg}=2^{\left\lceil\log_2\left(2d_{shift,j}^{cfg}+1\right)\right\rceil}.$$
   参数编译必须验证 $n_{shift,j}^{cfg}\mid(p-1)$，并按 `encoding` 的 canonical primitive-root 导出规则选择唯一根。该临时域只用于 prover 内部 FFT 卷积，不是第八个证明工作域，不需要 Lagrange SRS，也不进入 proof payload。若不存在该临时域，实例注册直接拒绝，不允许切换朴素平移或其他后端。
4. 用 $PK_{coef}$ 对固定长度 $(\widetilde f_0,\ldots,\widetilde f_{d_{shift,j}^{cfg}})$ 执行 MSM，生成唯一的 $[\widetilde F_j]$。MSM 的逻辑输入长度、profile 计数和缓存布局固定为 $d_{shift,j}^{cfg}+1$，不得根据私有高位零系数缩短。禁止使用未经登记的 shifted-Lagrange SRS、逐点评值插值、近似平移或其他等价实现。
5. coefficient materialization、blinder 合成与 Taylor-shift 卷积计入 `prove_time`；生成 $[\widetilde F_j]$ 的 MSM 计入 `commitment_time`。缓存不得跨 proof 复用包含私有 coefficient 或 blinder 的物化结果。

`shift_algorithm_id`、factorial 编码、临时根导出规则、实现规范摘要和最大 $n_{shift}^{cfg}$ 进入 `protocol_schema_id`；当前关系的 $d_{shift,j}^{cfg}$、固定 $n_{shift,j}^{cfg}$、来源对象、公开线性组合规则和缓存键导出规则进入 `instance_schema_hash`。当前 proof 的实际最高非零次数不得进入 metadata、transcript、schema hash、FFT 计划、MSM 长度或计时分支。

**七域 canonical trace**

每个私有 tensor 只允许有一个 canonical trace：

- 动态节点矩阵按列放置在 $\mathbb H_N$；需要对完整行优先数组执行范围、量化或查表时，同时建立 $\mathbb H_{edge}$ 上的行优先 evaluation trace，并通过本节的 indexed-view 关系与列 trace 绑定；

- 边向量放置在 $\mathbb H_{edge}$；

- 隐藏层 $W^{(\ell,r)}$ 的每个输出列放置在 $\mathbb H_{in}$；

- $a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}$ 放置在 $\mathbb H_{d_h}$；

- 输出层 $W^{(out,s)}$ 的每个类别列放置在 $\mathbb H_{cat}$；

- $a_{src}^{(out,s)},a_{dst}^{(out,s)},b^{(out,s)}$ 放置在 $\mathbb H_C$；

- 拼接矩阵、输出矩阵和最终 logits 的列 trace 放置在 $\mathbb H_N$。

对跨层传递、参与矩阵乘法、注意力、拼接或公开输出绑定的动态矩阵 $M\in\mathbb F_p^{N\times d}$，其唯一 canonical 表示固定为按列的节点域多项式族

$$P_{M[:,j]}^{can}(X)=\sum_{i=0}^{n_N-1}M_{i,j}L_i^{(N)}(X),\qquad j=0,\ldots,d-1.$$

正文中不带 `flat` 且以行优先幂次定义的 $P_M(X)=\sum_{i,j}M_{i,j}X^{id+j}$ 一律是 coefficient-basis relation view，不是任何工作域上的 evaluation trace；带 `flat` 的对象是明确登记在 $\mathbb H_{edge}$ 上的 row-major evaluation view。三类表示不得共用同一个 commitment 字段。对 $H',H_{agg,pre},H_{agg},H_{cat},Y',Y^{pre},\widetilde Y^{pre},Y^{(s)},Y$ 等网络状态矩阵，$\Pi_{bind}$ 的 $\mathbf{Com}_{view}$ 必须先按对象顺序和列序编码全部 $[P_{M[:,j]}^{can}]$，再编码 coefficient-basis、row-major、broadcast 或压缩 view commitments；所有非 canonical 表示通过 indexed-view、basis link 或工作域 quotient 与该列族绑定。

只在逐坐标量化关系中使用、且不作为后续层输入或独立矩阵关系来源的 $H_{agg,num},H_{ELU,in},Y_{agg,num},Y^q,Y^{sum}$，canonical 表示固定为 $\mathbb H_{edge}$ 上的 `flat` evaluation trace，不再生成第二份 coefficient-basis commitment或节点域列 commitment。由这些对象导出的局部压缩量可以在见证生成中计算，但除非 `ProofSchema` 明确登记为后续关系输入，否则不形成独立 commitment。canonical 列的 claimed values 只在所属 $\Pi_{bind}$ 的 $\mathbf{Eval}_{bind}^{zk}$ 中出现，不在顶层 $\mathbf{Eval}_{dom}$ 重复编码；对应 KZG claims 必须进入全局 `StandardPointGroupRegistry`，实际 witness 只在 $\mathbf{Open}_{dom}$ 或 $W_{ext}$ 的唯一 owner entry 中编码，$\Pi_{bind}$ 仅保存 point-group `MetadataReference`。

所有动态私有 canonical trace 使用 $P^{zk}=P+Z_{\mathcal D}R_P$。长期模型对象在 $VK_{model}$ 中只保留 setup 阶段生成的隐藏 commitment $[P_{model}^{vk}]$，该 commitment 不执行 proof-local 私有 opening。每份 proof 对每个模型 canonical trace 独立采样多项式 $S_{model}$，构造

$$P_{model}^{proof}(X)=P_{model}^{vk}(X)+Z_{\mathcal D}(X)S_{model}(X),$$

并提交 $[P_{model}^{proof}]$ 与 $[S_{model}]$。验证者使用 setup 中的 $[Z_{\mathcal D}]_2$ 检查

$$e\!\left([P_{model}^{proof}]-[P_{model}^{vk}],G_2\right)
=e\!\left([S_{model}],[Z_{\mathcal D}]_2\right).$$

该方程必须按第 1.4 节 `ShiftedKZG-Divisibility(D_{max},n_{\mathcal D})` 假设解释；它与 $e([S_{model}],[\tau^{n_{\mathcal D}}]_2)=e([P_{model}^{proof}]-[P_{model}^{vk}]+[S_{model}],G_2)$ 完全等价，不是仅在隐藏点 $\tau$ 上未经次数边界的启发式相等检查。

设 `ObservationLedger` 对该 proof-local 模型 canonical 多项式给出的最大所需 blinder 基长度为 $q_{model}^{max}$，诚实证明者必须满足

$$\deg S_{model}+1\ge q_{model}^{max},\qquad \deg S_{model}+n_{\mathcal D}\le D_{max}.$$

两项上界必须进入逐对象资源登记。证明者在发布 proof-local commitment 前重新采样使 $[S_{model}]\ne\mathcal O_{\mathbb G_1}$。验证者可直接拒绝的条件是：proof 打开 $S_{model}$、省略可除检查、字段集合超过 schema 预算或后续关系仍引用长期 commitment；随机系数不足及跨 proof 复用属于诚实证明者生成错误，由合规审计和测试拒绝，不声称可由单份 proof 识别。

任一 coefficient-basis relation view、广播 view、行优先 view 或折叠 view 都必须通过下述关系绑定到对应的动态 canonical trace或 proof-local 模型 canonical trace，不能作为独立见证来源。

**indexed-view 绑定**

设 canonical trace 的原生工作域为 $\mathbb H_{\mathcal D_{can}}$，relation view 的原生工作域为 $\mathbb H_{\mathcal D_{view}}$。两者可以属于同一工作域，也可以属于两个不同的既有工作域。公开索引编码分别记为 $idx_{can}[q]$ 和 $idx_{view}[q]$。根据公开索引映射，先唯一确定表端有效区选择器 $Q_{view}^{can}$、查询端有效区选择器 $Q_{view}^{qry}$ 和表端重数 $m_{view}^{can}$。

在 canonical commitment、view value commitment、对象身份、两个工作域、层号、头号和 chunk 描述全部固定后生成 $\eta_{view}$。分别定义：

$$Table_{view}^{can}[q]=Q_{view}^{can}[q]\big(idx_{can}[q]+\eta_{view}\,value_{can}[q]\big),$$

$$Query_{view}[q]=Q_{view}^{qry}[q]\big(idx_{view}[q]+\eta_{view}\,value_{view}[q]\big).$$

因此无效区固定满足 $Table_{view}^{can}[q]=Query_{view}[q]=0$；公开索引列可以在整个原生工作域上按其 canonical 规则定义，但只有选择器为 $1$ 的位置进入 tuple 编码。不得通过令无效区索引取任意特殊值来替代上述选择器门控。

$m_{view}^{can}$ 是由 `RelationMeta` 中的公开索引映射唯一重建的 `Public` 多项式，不允许证明者自由选择。证明者先提交 $Table_{view}^{can},Query_{view},m_{view}^{can}$ 和 view 值 commitments；验证者随后一次性生成 $\beta_{view}$。挑战生成不得吸入任何依赖 $\beta_{view}$ 的逆元或累加器 commitment。$m_{view}^{can}$ 的 commitment 必须在该 indexed-view 所有 quotient 点和至少一个承诺后随机点以验证者本地计算的 `PublicValue` 打开；任一评值不一致时拒绝。

挑战固定后，证明者构造分母逆元列：

$$J_{view}^{can}[t]=
\begin{cases}
(Table_{view}^{can}[t]+\beta_{view})^{-1},&Q_{view}^{can}[t]=1,\\
0,&Q_{view}^{can}[t]=0,
\end{cases}$$

$$J_{view}^{qry}[t]=
\begin{cases}
(Query_{view}[t]+\beta_{view})^{-1},&Q_{view}^{qry}[t]=1,\\
0,&Q_{view}^{qry}[t]=0.
\end{cases}$$

两个累加器定义为：

$$R_{view}^{can}[0]=0,$$

$$R_{view}^{can}[t+1]=R_{view}^{can}[t]+Q_{view}^{can}[t]m_{view}^{can}[t]J_{view}^{can}[t],$$

$$R_{view}^{qry}[0]=0,$$

$$R_{view}^{qry}[t+1]=R_{view}^{qry}[t]+Q_{view}^{qry}[t]J_{view}^{qry}[t].$$

逆元非零约束、无效区归零、起点和递推分别进入两个原生工作域商多项式；工作域最后一点到首点的循环转移由 $1-Last_{\mathcal D}$ 屏蔽。两个终点满足：

$$P_{R_{view}^{can}}(z_{last,\mathcal D_{can}})-P_{R_{view}^{qry}}(z_{last,\mathcal D_{view}})=0,$$

该跨域终点等式使用本节的 KZG 同点平移归一化关系验证。点互异时 proof 编码 `KZGShiftRelationProof`，点合并后只登记普通零值 claim；任何分支都不编码未掩蔽私有终点。若 canonical trace 与 view 的长度或重复次数不同，必须使用公开重数和有效区选择器；跨图、跨层、跨头或跨对象的索引标签不得相互匹配。

proof 不传输裸 tensor 值、裸 view 值或裸 LogUp 终点。该构造只使用所述两个工作域，不增加主工作域或顶层 proof block。

**coefficient-tail blinding 与 basis link**

对长度为 $L_M$ 的私有 coefficient-basis 对象：

$$P_M(X)=\sum_{q=0}^{L_M-1}m_qX^q,$$

实际承诺：

$$\widehat P_M(X)=P_M(X)+X^{L_M}R_M^{coef}(X).$$

随机尾项从第 $L_M$ 个系数开始，不改变低阶语义系数。$R_M^{coef}$ 必须由密码学安全随机源独立采样，并满足 `privacy_policy` 登记的次数界。该规则适用于 CRPC 的输出多项式、共享维 relation view、模型矩阵 relation view、动态矩阵 relation view 和相应辅助 quotient。

设 `ObservationLedger` 对同一 coefficient object 给出的最大所需随机尾长度为 $q_M^{max}$。`privacy_policy` 必须固定唯一尾长度 $L_{tail,M}=\deg(R_M^{coef})+1$，并满足

$$L_{tail,M}\ge q_M^{max}.$$

该秩同时覆盖 commitment、masked evaluations、basis / CRPC openings和平移关系 observations。该对象还必须携带由形状、布局和该尾长度唯一重建的三个互斥公开选择器：$Q_{sem}$ 标识与 canonical tensor 一一对应的语义系数位置，$Q_{gap}$ 标识 $0,\ldots,L_M-1$ 内未使用的低阶空隙，$Q_{tail}$ 标识 $L_M,\ldots,L_M+L_{tail,M}-1$ 的随机尾位置。对象有效区之外的系数固定为零。三者在对象有效区内满足

$$Q_{sem}+Q_{gap}+Q_{tail}=1,$$

且两两乘积为零。语义位置通过 indexed-view 与 canonical trace 绑定；空隙位置必须在系数 chunk 的承载工作域中证明

$$Q_{gap}(X)P_{Coeff_M}(X)=0;$$

尾位置只允许由 $R_M^{coef}$ 占用。任何未被三类选择器覆盖的系数、低阶空隙非零或将随机尾写入语义区的对象都必须拒绝。

将 $\widehat P_M$ 的完整系数向量固定放置到现有边域 $\mathbb H_{edge}$ 的 coefficient chunk 中，不允许证明者在七个工作域之间选择承载域。令单个 chunk 的固定容量为 $C_{chunk}=n_{edge}-1$，总系数长度为 $L_{all}$，则 chunk 数、偏移和长度唯一确定为

$$n_{chunk}=\left\lceil\frac{L_{all}}{C_{chunk}}\right\rceil,\qquad offset_c=cC_{chunk},\qquad L_c=\min\{C_{chunk},L_{all}-offset_c\}.$$

第 $c$ 个 chunk 的系数 trace 记为 $\widehat m_{c,t}$，统一定义在 $\mathbb H_{edge}$ 上；最后一行保留为终点行。定义公开步进选择器与终点：

$$Q_c^{step}[t]=\begin{cases}1,&0\le t<L_c,\\0,&L_c\le t<n_{edge},\end{cases}
\qquad z_{last,c}=\omega_{edge}^{L_c}.$$

低阶语义系数通过 indexed-view 关系与 canonical trace 绑定；随机尾系数只承担隐藏作用。

在全部 coefficient commitments 和 chunk commitments 固定后生成非零挑战 $\gamma_{basis}$。验证者本地重建公开幂次列 $\Gamma_c[t]=\gamma_{basis}^{t}$。对每个 chunk 构造：

$$A_{basis,c}[0]=0,$$

$$A_{basis,c}[t+1]=A_{basis,c}[t]+Q_c^{step}[t]\,\widehat m_{c,t}\,\Gamma_c[t],
\qquad 0\le t<n_{edge}-1.$$

因此 $A_{basis,c}[L_c]=\sum_{t=0}^{L_c-1}\widehat m_{c,t}\gamma_{basis}^{t}$，并且累加器在 $L_c$ 之后的填充区保持常值。于是：

$$\widehat P_M(\gamma_{basis})-
\sum_c\gamma_{basis}^{offset_c}A_{basis,c}(z_{last,c})=0.$$

该关系使用本节的 KZG 同点平移归一化关系验证；点互异时 proof 编码 `KZGShiftRelationProof`，并只提交平移 commitments、masked 检查值与普通 KZG witness 引用，不传输 $\widehat P_M(\gamma_{basis})$、$A_{basis,c}(z_{last,c})$ 或其他私有域元素。由此，coefficient-basis commitment 的全部低阶系数被绑定到七域 canonical trace，不需要参数域。

**CRPC 绑定**

对 $C=A\cdot B$，令 $L_C=mn$。为保证随机尾项不进入 CRPC 的前 $L_C$ 个语义系数，三个对象统一从 $X^{L_C}$ 开始加入独立随机尾项：

$$\widehat P_C(X)=P_C(X)+X^{L_C}R_C^{coef}(X),$$

$$\widehat A_t^{\langle n\rangle}(X)=A_t^{\langle n\rangle}(X)+X^{L_C}R_{A,t}^{coef}(X),$$

$$\widehat B_t(X)=B_t(X)+X^{L_C}R_{B,t}^{coef}(X).$$

这里允许 $A_t^{\langle n\rangle}$ 或 $B_t$ 的语义次数远小于 $L_C$；$0$ 到 $L_C-1$ 之间未使用的系数固定为零。定义：

$$D_{CRPC}(X)=\widehat P_C(X)-\sum_{t=0}^{\ell-1}\widehat A_t^{\langle n\rangle}(X)\widehat B_t(X).$$

所有包含随机尾项的交叉项次数均不小于 $L_C$，因此 $D_{CRPC}$ 的前 $L_C$ 个系数与语义 CRPC 关系一致并全部为零，于是：

$$Q_{CRPC}(X)=\frac{D_{CRPC}(X)}{X^{L_C}}$$

是多项式。

为隐藏 $Q_{CRPC}$，令 $d_Q=b_{Q_{CRPC}}^{hon}$ 为诚实构造在 $\mathcal R_{deg}$ 中登记的资源次数上界，并固定其语义长度 $L_Q=d_Q+1$。因此诚实生成的 $\deg(Q_{CRPC})<L_Q$；恶意 proof 的可靠性不假设这一较小次数，而使用全局 $D_{max}$ 和 `RandomIdentityLedger`。采样独立 $R_Q^{coef}$，并承诺：

$$\widehat Q_{CRPC}(X)=Q_{CRPC}(X)+X^{L_Q}R_Q^{coef}(X).$$

在全部相关 commitments 固定后生成 $\gamma_{CRPC}$。验证关系为：

$$\widehat P_C(\gamma_{CRPC})
-\sum_t\widehat A_t^{\langle n\rangle}(\gamma_{CRPC})\widehat B_t(\gamma_{CRPC})
-\gamma_{CRPC}^{L_C}\widehat Q_{CRPC}(\gamma_{CRPC})
+\gamma_{CRPC}^{L_C+L_Q}R_Q^{coef}(\gamma_{CRPC})=0.$$

所有参与对象都在同一非零点 $\gamma_{CRPC}$ 提供对 coefficient-tail 隐藏多项式的标准 KZG openings；proof 编码这些隐藏多项式在 $\gamma_{CRPC}$ 的 `MaskedPrivateValue`，验证者逐项验证 opening 后，按固定顺序直接计算上述乘积与线性组合并检查结果为零。CRPC 恒等式不使用跨点线性关系证明。$\widehat A_t$、$\widehat B_t$ 和 $\widehat P_C$ 的低阶系数分别通过 basis link 与 indexed-view 关系绑定到输入 tensor、模型 commitments 和输出 tensor。因此，证明者不能用未绑定的折叠向量或未绑定的系数多项式替代真实矩阵。

**私有输出偏置绑定**

输出偏置 $b^{(out,s)}$ 以 $VK_{model}$ 中 $\mathbb H_C$ 上的长期隐藏 canonical commitment 为锚点，并在每份 proof 中引用通过可除方程验证的 proof-local canonical commitment。对行优先索引 $q=iC+c$，构造偏置广播 view：

$$b_{flat}^{(out,s)}[q]=b_c^{(out,s)}.$$

该广播 view 通过 indexed-view 关系与本份 proof 的偏置 canonical commitment 绑定；该 proof-local commitment 已通过 $[Z_C]_2$ 可除方程绑定到 $VK_{model}$ 中的长期偏置锚点。在 $\mathbb H_{edge}$ 上证明：

$$Q_{NC}^{valid}(X)\big(
P_{Y^{(s)},flat}(X)-P_{\widetilde Y^{pre(s)},flat}(X)-P_{b_{flat}^{(out,s)}}(X)
\big)=0.$$

验证者只读取偏置 commitment、广播 view commitment、商 commitment 和 KZG witnesses，不读取偏置明文或偏置随机点评值。

**聚合关系绑定**

聚合矩阵必须在对应输入矩阵、归一化权重和聚合输出 commitments 全部固定后，使用独立的承诺后压缩挑战证明完整逐坐标关系。该挑战不得与前向阶段用于构造压缩列的挑战复用。

对隐藏层第 $(\ell,r)$ 个头，在 $[P_{H'^{(\ell,r)}}]$、$[P_{\alpha^{(\ell,r)}}]$ 和 $[P_{H_{agg,pre}^{(\ell,r)}}]$ 固定后生成

$$\xi_{agg,bind}^{(\ell,r)}=H_{FS}(\text{transcript},\texttt{bind:hidden:agg},\ell,r,[P_{H'}],[P_{\alpha}],[P_{H_{agg,pre}}]).$$

定义节点压缩列、源广播列、边乘积列和目标广播列：

$$H_{src,bind,k}^{\star(\ell,r)}=\sum_{j=0}^{d_h^{(\ell)}-1}H_{src(k),j}'^{(\ell,r)}(\xi_{agg,bind}^{(\ell,r)})^j,$$

$$H_{agg,bind,i}^{\star(\ell,r)}=\sum_{j=0}^{d_h^{(\ell)}-1}H_{agg,pre,i,j}^{(\ell,r)}(\xi_{agg,bind}^{(\ell,r)})^j,$$

$$V_{agg,bind,k}^{(\ell,r)}=\alpha_k^{(\ell,r)}H_{src,bind,k}^{\star(\ell,r)},$$

$$T_{agg,bind,k}^{edge(\ell,r)}=H_{agg,bind,dst(k)}^{\star(\ell,r)}.$$

$H_{src,bind}^{\star}$ 与 $H_{agg,bind}^{\star}$ 分别通过 indexed-view 与 $H'$、$H_{agg,pre}$ 的 canonical trace 绑定。源广播使用节点表端和边查询端：

$$Table_{agg,src}^{(\ell,r)}[i]=Q_N[i]\big(i+\eta_{agg,src}^{(\ell,r)}H_{src,bind,i}^{\star(\ell,r)}\big),$$

$$Query_{agg,src}^{(\ell,r)}[k]=Q_{edge}^{valid}[k]\big(src(k)+\eta_{agg,src}^{(\ell,r)}H_{src,bind,k}^{\star,edge(\ell,r)}\big),$$

表端重数为 $m_{src}[i]=\#\{k\mid src(k)=i\}$。目标广播使用：

$$Table_{agg,dst}^{(\ell,r)}[i]=Q_N[i]\big(i+\eta_{agg,dst}^{(\ell,r)}H_{agg,bind,i}^{\star(\ell,r)}\big),$$

$$Query_{agg,dst}^{(\ell,r)}[k]=Q_{edge}^{valid}[k]\big(dst(k)+\eta_{agg,dst}^{(\ell,r)}T_{agg,bind,k}^{edge(\ell,r)}\big),$$

表端重数为 $m_{dst}[i]=\#\{k\mid dst(k)=i\}$。两组关系分别按第 0.7.8 节规定的 indexed-view 绑定构造 LogUp commitments、累加器和商约束。边域乘积约束为

$$C_{agg,bind,mul}^{(\ell,r)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{V_{agg,bind}^{(\ell,r)}}(X)-P_{\alpha^{(\ell,r)}}(X)P_{H_{src,bind}^{\star,edge(\ell,r)}}(X)\Big).$$

定义聚合绑定状态机：

$$PSQ_{agg,bind}^{(\ell,r)}[0]=Q_{edge}^{valid}[0]V_{agg,bind,0}^{(\ell,r)},$$

$$\begin{aligned}
PSQ_{agg,bind}^{(\ell,r)}[k+1]={}&(1-Q_{edge}^{valid}[k+1])PSQ_{agg,bind}^{(\ell,r)}[k]\\
&+Q_{edge}^{valid}[k+1]\Big(Q_{grp\_new}^{edge}[k+1]V_{agg,bind}^{(\ell,r)}[k+1]\\
&\qquad +(1-Q_{grp\_new}^{edge}[k+1])\big(PSQ_{agg,bind}^{(\ell,r)}[k]+V_{agg,bind}^{(\ell,r)}[k+1]\big)\Big),
\end{aligned}$$

并约束

$$Q_{grp\_end}^{edge}[k]\Big(PSQ_{agg,bind}^{(\ell,r)}[k]-T_{agg,bind}^{edge(\ell,r)}[k]\Big)=0.$$

对输出头 $s$，在 $[P_{Y'^{(s)}}]$、$[P_{\alpha^{(out,s)}}]$ 和 $[P_{Y^{pre(s)}}]$ 固定后，以独立挑战 $\xi_{out,agg,bind}^{(s)}$ 定义

$$Y_{src,bind,k}^{\star(s)}=\sum_{c=0}^{C-1}Y_{src(k),c}'^{(s)}(\xi_{out,agg,bind}^{(s)})^c,$$

$$Y_{pre,bind,i}^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{pre(s)}(\xi_{out,agg,bind}^{(s)})^c,$$

$$V_{out,agg,bind,k}^{(s)}=\alpha_k^{(out,s)}Y_{src,bind,k}^{\star(s)},\qquad
T_{out,agg,bind,k}^{edge(s)}=Y_{pre,bind,dst(k)}^{\star(s)}.$$

$Y_{src,bind}^{\star}$ 与 $Y_{pre,bind}^{\star}$ 分别通过 indexed-view 与 $Y'$、$Y^{pre}$ 的 canonical trace 绑定。源广播使用：

$$Table_{out,agg,src}^{(s)}[i]=Q_N[i]\big(i+\eta_{out,agg,src}^{(s)}Y_{src,bind,i}^{\star(s)}\big),$$

$$Query_{out,agg,src}^{(s)}[k]=Q_{edge}^{valid}[k]\big(src(k)+\eta_{out,agg,src}^{(s)}Y_{src,bind,k}^{\star,edge(s)}\big),$$

表端重数为 $m_{src}[i]$。目标广播使用：

$$Table_{out,agg,dst}^{(s)}[i]=Q_N[i]\big(i+\eta_{out,agg,dst}^{(s)}Y_{pre,bind,i}^{\star(s)}\big),$$

$$Query_{out,agg,dst}^{(s)}[k]=Q_{edge}^{valid}[k]\big(dst(k)+\eta_{out,agg,dst}^{(s)}T_{out,agg,bind,k}^{edge(s)}\big).$$

表端重数为 $m_{dst}[i]$。两组关系分别按第 0.7.8 节规定的 indexed-view 绑定构造 LogUp commitments、累加器和商约束。边域乘积约束为

$$C_{out,agg,bind,mul}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{V_{out,agg,bind}^{(s)}}(X)-P_{\alpha^{(out,s)}}(X)P_{Y_{src,bind}^{\star,edge(s)}}(X)\Big).$$

定义输出聚合绑定状态机：

$$PSQ_{out,agg,bind}^{(s)}[0]=Q_{edge}^{valid}[0]V_{out,agg,bind,0}^{(s)},$$

$$\begin{aligned}
PSQ_{out,agg,bind}^{(s)}[k+1]={}&(1-Q_{edge}^{valid}[k+1])PSQ_{out,agg,bind}^{(s)}[k]\\
&+Q_{edge}^{valid}[k+1]\Big(Q_{grp\_new}^{edge}[k+1]V_{out,agg,bind}^{(s)}[k+1]\\
&\qquad +(1-Q_{grp\_new}^{edge}[k+1])\big(PSQ_{out,agg,bind}^{(s)}[k]+V_{out,agg,bind}^{(s)}[k+1]\big)\Big),
\end{aligned}$$

并约束

$$Q_{grp\_end}^{edge}[k]\Big(PSQ_{out,agg,bind}^{(s)}[k]-T_{out,agg,bind}^{edge(s)}[k]\Big)=0.$$

上述辅助 commitments、状态机和约束进入 $\mathbb H_{edge}$ 与 $\mathbb H_N$ 的对应关系登记表，不增加工作域、顶层 proof block 或独立证明系统。

**子证明 transcript 与载荷**

每个绑定子证明使用独立域分离标签，并按下列顺序生成：

1. 吸入对象身份、层号、头号、工作域、canonical commitments、proof-local 模型 commitments、coefficient commitments、全部 coefficient chunk commitments、view value commitments、聚合输入 / 输出 commitments 和 chunk 描述，生成 $\eta_{view}$、非零 $\gamma_{basis}$ 以及承诺后聚合压缩挑战；
2. 提交 indexed-view 的表端、查询端、重数 commitments、basis accumulators、聚合绑定辅助 commitments、$[\widehat Q_{CRPC}]$、$[R_Q^{coef}]$ 和公开选择器摘要；在各 indexed-view 的 $Table,Query,m$ 固定后生成对应 $\beta_{view}$；在全部 CRPC 输入、输出与 quotient 来源 commitments 固定后生成非零 $\gamma_{CRPC}$；
3. 使用各 $\beta_{view}$ 构造并提交分母逆元与 LogUp 累加器 commitments。此时只登记其约束族和次数界，不提交任何局部 quotient commitment；全部 $C_{bind,\mathcal D}$ 在全局 $\alpha_{quot}$ 生成后统一并入第 3.3 节的七域商；
4. 对每个规范化线性关系，$m=0$ 不提交字段；$m=1$ 只登记普通零值 claim；$m\ge2$ 时按规范点序提交 $\{[\widetilde F_j]\}_{j=2}^{m}$，随后生成该关系的 $y_{rel}$，登记每个平移检查的两个普通 KZG claims 与 $([G_{rel}],z_\star,0)$。本阶段不提交 opening witness；不得提交原点评值、裸终点或关系专用局部商；
5. 全部动态与绑定辅助 commitments 固定后生成全局 $\alpha_{quot}$，提交七域商；所有显式 opening 随后按第 3.4.2 节以“相同点内批量、不同点分组”的唯一算法生成。不同点不得共用一个单点 quotient witness。

$\gamma_{basis}$ 与 $\gamma_{CRPC}$ 必须非零；工作域外 masked evaluation 点必须满足其原生工作域零化多项式在该点非零。每个子证明固定记录对象标签、来源 commitment、view commitment、工作域、层号、头号、chunk offset、公开线性系数、关系点、全局 quotient 关系引用，以及对应全局 point group 的 `MetadataReference(group_id)`。实际 `OpeningWitness` 只由 $\mathbf{Open}_{dom}$ 或 $W_{ext}$ 中的唯一 owner entry 编码。证明对象不得包含裸折叠向量、裸模型参数评值、裸中间激活、裸偏置、裸内积结果或裸 route terminal。验证者必须检查子证明身份、对象顺序、可见性、挑战重放、proof-local 模型再盲化、indexed-view LogUp、coefficient 区域选择器、basis link、CRPC identity、聚合关系、偏置绑定、平移一致性 openings、关系零值 opening 和全部 KZG 方程。

### 0.8 填充、零分母、Softmax 可行性与缓存要求

#### 0.8.1 填充规则

对任意工作域 $\mathbb H_{\mathcal D}$，若真实长度为 $L_{\mathcal D}$、工作域长度为 $n_{\mathcal D}$，则所有离散列必须补到长度 $n_{\mathcal D}$。

所有使用 $t\mapsto t+1$ 递推并在 $Last_{\mathcal D}$ 检查终点的对象，其有效表端和查询端支撑必须包含于 $\{0,\ldots,n_{\mathcal D}-2\}$；第 $n_{\mathcal D}-1$ 行强制为填充行。参数生成必须保证每个工作域为最长有效序列至少保留一个终点行，否则拒绝该配置。

对所有 $t\in\{L_{\mathcal D},L_{\mathcal D}+1,\ldots,n_{\mathcal D}-1\}$，统一规定：

- 非状态见证值补零；

- 表值补零；

- 查询值补零；

- 重数补零；

- 由有效区选择器屏蔽无效位置；

- 所有状态机在填充区都必须保持常值；实现时不得改写成其他等价但不同形态的递推规则。

`ProofSchema` 必须为每个工作域 trace 显式登记 `padding_mode`。非状态值列使用 `ZeroPad`，其有效区选择器记为 $Q_P$，并进入原工作域 quotient：

$$C_{padding,P}(X)=\big(1-P_{Q_P}(X)\big)P_P(X).$$

LogUp、route、PSQ、最大值计数和共享维累加器使用 `HoldLastState`；其填充区常值性由带 $1-Last_{\mathcal D}$ 屏蔽的递推和有效区选择器共同证明，不再附加零填充约束。coefficient-tail 对象使用本节之外的 `CoefficientRegionPad`，由 $Q_{sem},Q_{gap},Q_{tail},Q_{chunk}$ 约束。任何对象缺失 `padding_mode`、模式与对象类型不一致或填充约束未进入 quotient 时必须拒绝。

#### 0.8.2 零分母冲突

每个 LogUp / route / indexed-view 实例都必须在表端、查询端、重数和公开选择器 commitments 固定后生成一次挑战 $\beta$。挑战不得依赖逆元列或累加器 commitment。

对有效表端和查询端分别生成分母逆元列 $J_{tbl},J_{qry}$，并在对应工作域证明：

$$Q_{tbl}\big((Table+\beta)J_{tbl}-1\big)=0,$$

$$(1-Q_{tbl})J_{tbl}=0,$$

$$Q_{qry}\big((Query+\beta)J_{qry}-1\big)=0,$$

$$(1-Q_{qry})J_{qry}=0.$$

因此任何有效位置出现零分母时，诚实或恶意证明者都无法生成可接受 proof。协议不定义依赖私有 $Table/Query$ 值的公开 `Bad` 集合，也不允许把由私有零分母决定的拒绝计数写入 transcript。

证明生成使用公开的 prover-chosen `proof_nonce` 初始化 transcript。诚实证明者必须用 CSPRNG 采样该值，但密码学验证者只检查长度、规范编码和 transcript 绑定，不能从字节串验证其随机来源。应用层可以另外维护 anti-replay cache，但缓存命中不得改变本文密码学 verifier 对同一 statement/proof 的接受语义，也不进入 `protocol_schema_id`。若诚实证明在极小概率下遇到零分母，必须丢弃该次未发布 proof，重新采样新的 `proof_nonce` 和全部 proof-local blinding，再从头生成；proof 不记录失败次数。工作域外点、非零公开挑战和域零化多项式非零等仅依赖公开信息的条件，可以使用规范公开 counter 拒绝采样，验证者能够独立重放。

#### 0.8.3 Softmax 可行性

要求每个有效节点至少有一条入边。公共输入构造阶段必须保证：

$$\#\{k\mid dst(k)=i\}\ge 1\qquad\forall i\in\{0,1,\ldots,N-1\}.$$

若原始图不满足该条件，必须在公开预处理阶段显式加入自环后才允许进入本文协议。

指数表的有效输出必须是正整数，并由公开量化配置给出上界 $U_{max}$：

$$1\le U_k\le U_{max}.$$

节点分母按整数关系计算：

$$Sum_i=\sum_{\{k\mid dst(k)=i\}}U_k.$$

该求和及其所有中间值必须通过范围约束证明没有模回绕。因此：

$$Sum_i>0,$$

有限域逆元 $inv_i$ 只用于证明该非零关系和有限域 Softmax 关系；ELU 输入与输出层定点值由第 0.4 节的整数聚合分子和 $SignedRoundDiv$ 关系确定。

$quant\_cfg\_id$ 还必须给出 $U_k$、$Sum_i$、$H_{agg,num}$、$Y_{agg,num}$、尺度乘积、最终求和和全部除法辅助量的范围上界。任何上界不能保证规范有符号代表唯一且绝对值小于 $p/2$ 的配置都必须拒绝。

#### 0.8.4 热路径对象与缓存要求

热路径对象包括：所有域元素、所有挑战值、所有曲线点、SRS、静态表承诺、模型承诺、全部 FFT / IFFT 计划以及商计算的扩展域计划。

程序启动后必须一次性缓存：

- $PK_{coef}$ 与当前实例选中的七个 $PK_{lag}^{(\mathcal D)}$；

- $VK_{KZG}$；

- $VK_{static}$；

- $VK_{model}$；

- $\mathbb H_{FH},\mathbb H_{edge},\mathbb H_{in},\mathbb H_{d_h},\mathbb H_{cat},\mathbb H_C,\mathbb H_N$ 的 FFT / IFFT 计划；

- 商多项式计算所需的扩展域执行计划

单次证明或验证期间不得重复构造这些对象。

#### 0.8.5 公开元数据与安全设置字段

为保证证明对象、SRS、静态表、模型参数、量化配置和隐私策略之间一致，证明对象中必须携带一份公开元数据。实现按下列字段固定顺序序列化并吸入 transcript：

1. `protocol_name`：主实验固定为 `zkgat`。
2. `proof_nonce`：证明者选择的公开 256-bit 域分离值。诚实证明者使用 CSPRNG 独立采样，且不得编码 witness、失败次数或数据相关状态；验证者只检查固定长度、规范编码和 transcript 绑定，不声称验证随机来源。
3. `proof_task`：主实验固定为 `node_classification_full_gat`。
4. `srs_id`：只绑定曲线、标量域、$D_{max}$、`domain_size_catalog`、`domain_generator_catalog`、`kzg_assumption_profile`、普通 coefficient SRS、Lagrange SRS 目录和 verifier key；不绑定具体数据集、模型或实例对象数量。
5. `protocol_schema_id`：绑定协议级 `ProtocolCompilerArtifact` 的 canonical bytes，包括七个工作域、九个顶层 proof block、关系模式、transcript 顺序、字段类型、verifier 方程、最大三份账本模板、constraint / point-group / degree / padding registries、KZG 假设 profile 和 `shift_algorithm_id`。
6. `instance_schema_hash`：绑定当前实例展开后的 `ProtocolCompilerArtifact` canonical bytes，包括层数、头数、维度、对象 family、固定 coefficient chunk 数、逐对象资源上界、实际 `ProofSchema`、三份账本、constraint registry、全局 point-group registry 模板实例和公共列 padding registry；进入 transcript，但不进入 `srs_id`。
7. `srs_type`：唯一合法值为 `trusted_srs`；serializer 不得生成其他值，parser 与 verifier 必须拒绝其他值。
8. `srs_degree_bound`：必须覆盖当前实例全部多项式、商和 blinding 后次数上界。
9. `kzg_assumption_profile`：引用 manifest 中认证的普通 KZG 与 `ShiftedKZG-Divisibility` 假设 profile；其允许域大小、$D_{max}$ 和安全位数必须与当前实例一致。
10. `privacy_policy`：主实验固定为 `model_and_trace_hidden`。
11. `input_privacy`：主实验固定为 `public_graph_public_features`。
12. `output_mode`：主实验固定为 `public_logits`；`M_pub` 按行优先编码完整 $N\times C$ logits 及摘要。
13. `opening_mode`：主实验固定为 `point_grouped_batch_opening`。
14. `binding_mode`：主实验固定为 `zk_subproof_for_crpc`。
15. `model_arch`：绑定层数、隐藏头数、隐藏维度、输出头数、偏置和最终平均语义。
16. `model_params`：模型参数 commitments 集合标识。
17. `static_tables`：`static_tables_id`、`table_generator_id`、确定性生成规范摘要、尺度 / 区间证书和静态表 commitments 集合标识。
18. `quant_config`：绑定尺度、舍入、重缩放、符号解释、输出平均、limb 数和 `integer_bound_ledger`。
19. `domain_cfg`：当前实例从 SRS 目录选择的七个工作域大小与生成元配置。
20. `dim_cfg`：图规模、batch 指针、层数、维度、类别数、范围位宽和输出头数。
21. `encoding`：域元素、曲线点、数组长度、字节序、proof block tag、`hash_suite_id`、type-tagged length-prefix transcript 编码、hash-to-field 和 counter 宽度。
22. `padding_rule`：填充规则、有效区选择器和状态机保持规则。
23. `degree_bound`：逐对象诚实 prover 资源上界、全局可验证 $D_{max}$ 和 `RandomIdentityLedger` 最坏次数策略。
24. `security_budget`：`SecurityBudgetLedger` 的 canonical 摘要、$Q_{FS}^{max}$、目标安全位数和精确总优势上界。
25. `public_output`：完整 $Y_{public}$ 的形状、canonical field encoding 和摘要。

`public_output` 的摘要算法固定为 `output_digest`：

$$\begin{aligned}
output\_digest=H_{suite}(&\operatorname{ASCII}(\texttt{"zkgat:public-output"})\parallel enc_{u64}(N)\parallel enc_{u64}(C)\parallel enc_{u64}(NC)\parallel\\
&\bigparallel_{i=0}^{N-1}\bigparallel_{c=0}^{C-1}enc_{\mathbb F_p}(Y_{public}[i,c])).
\end{aligned}$$

摘要生成前必须使用无界整数或至少 128 位无符号临时整数计算 $NC$，并检查

$$0<N<2^{64},\qquad 0<C<2^{64},\qquad 0<NC<2^{64}.$$

任一条件失败时必须在生成动态 commitments 前拒绝；禁止先在 64 位整数中发生乘法溢出，再把截断结果传给 $enc_{u64}$。

其中 $H_{suite}$、摘要字节长度、$enc_{u64}$ 的无符号 little-endian 规则和固定宽度 $enc_{\mathbb F_p}$ 均由 `encoding.hash_suite_id` 唯一确定；域分离标签、$N$、$C$ 和元素数量必须全部进入摘要。禁止使用 JSON 文本、宿主语言数组序列化、无长度拼接或 transcript 当前内部状态替代该算法。reference forward、$M_{pub}$ 和实验结果中的公开输出摘要必须使用同一 `output_digest`。

验证者必须拒绝任何字段缺失、重复、顺序错误、值不匹配或出现未声明字段的 proof。

#### 0.8.6 大图稀疏执行与批处理假设

面向大图数据集时，本文固定采用如下执行假设；这些假设不改变 GAT 前向，只约束证明系统的工程实现：

1. 仅在真实边集上具体化注意力计算

   隐藏层与输出层的注意力打分、softmax、PSQ、路由与 lookup 全部仅在公开边集 $E$ 上展开；禁止为了实现方便而将带掩码的注意力改写为全图稠密的两两计算。

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
2. `PublicOutput`：协议显式声明公开的输出对象。完整最终 logits 固定为公开输出。
3. `PrivateWitness`：模型参数、输入特征私有化场景下的特征值、中间激活、注意力分数、softmax 权重、查找 query、路由 query、PSQ 状态、折叠向量、累加器、中间矩阵和所有不属于 public / public-output 的对象。

对象分类固定如下：

| 对象族 | 代表对象 | 可见性 | proof payload 规则 |
|---|---|---|---|
| 公开元数据 | `protocol_name, srs_id, protocol_schema_id, instance_schema_hash, srs_type, kzg_assumption_profile, privacy_policy, proof_task, model_arch, quant_config, domain_cfg, dim_cfg, security_budget` | `Public` | 明文写入并吸入 transcript |
| 公开拓扑 | `src,dst,node_ptr,edge_ptr,node_gid,edge_gid,Q_*^{valid},Q_{grp_new},Q_{grp_end},Idx_*` | `Public` | 验证者本地重建；若写入 proof 只允许写 hash / metadata |
| SRS / VK 摘要 | `srs_id, pk_coef_hash, pk_lagrange_catalog_hash, vk_hash` | `Public` | 由 manifest 摘要绑定并吸入 transcript；proof 不重复编码 SRS 文件 |
| 静态表与承诺 | `T_H,T_{LReLU},T_{ELU},T_{exp},T_{range}` 及其 commitments | `Public` | 验证者按 `static_tables` 标识加载或重建；proof 只写承诺与摘要，不重复编码完整表 |
| 模型承诺 | `[V_W], [V_a], [V_b]` | commitment 为 `Public`，参数值为 `PrivateWitness` | proof 只写长期锚点承诺，不写裸参数 |
| proof-local 模型再盲化 | `[P_{model}^{proof}],[S_{model}]` | commitment 为 `Public`，多项式为 `PrivateWitness` | $P_{model}^{proof}$ 只允许 masked openings；$S_{model}$ 只允许 commitment 与可除 pairing，禁止任何 opening；禁止复用 blinder |
| 原始特征 | `P_H,P_H^{feat}` | `Public` | 验证者按公开输入分别重建 coefficient-basis 与 $\mathbb H_{FH}$ evaluation-basis 编码 |
| 隐藏层投影 | `P_{H'}, P_{E_src}, P_{E_dst}, P_{H^\star}` | `PrivateWitness` | 只允许承诺与 masked evaluation |
| 边域注意力链 | `P_S,P_Z,P_M^{edge},P_{\Delta^+},P_U,P_{Sum}^{edge},P_{inv}^{edge},P_\alpha` | `PrivateWitness` | 禁止输出完整列或裸随机点评值 |
| lookup 对象 | 静态表端、特征检索对象、动态 `P_Table,P_Query,P_m,P_R` | 静态表端和特征检索对象为 `Public`；由私有 trace 构造的 table、query、multiplicity、accumulator 为 `PrivateWitness` | 私有 lookup 对象只允许隐藏 commitment 与 masked evaluation |
| route 对象 | `P_Table^{src/dst},P_Query^{src/dst},P_R` | `PrivateWitness` | 只允许隐藏 commitment 与 `MaskedPrivateValue` opening |
| 拓扑重数 | `P_{m_{src}},P_{m_{dst}},P_{m_{feat}},P_{m_{view}^{can}}` | `Public` | 由公开索引映射唯一重建；commitment 在全部使用点以本地 `PublicValue` 打开，禁止作为自由 witness |
| PSQ / 状态机 | `P_{PSQ},P_{C_max},P_{Acc},P_w,P_T` | `PrivateWitness` | 只允许隐藏 commitment 与 `MaskedPrivateValue` opening |
| 折叠向量 | `P_a,P_b,P_Acc, \mu_*` | 来源于私有 tensor 或模型参数的折叠向量与累加器为 `PrivateWitness`；仅由公开挑战幂构成的常量向量为 `Public` | 不允许编码裸 `mu`；私有对象只允许隐藏 commitment、masked evaluation 和子证明 witness |
| KZG 平移关系对象 | $[\widetilde F_j],u_{rel,j}$ 及普通 point-grouped witness 引用 | 平移多项式为 `PrivateWitness`；commitment 为 `Public`；私有检查值为 `MaskedPrivateValue` | 只允许平移 commitment、单次编码的 masked 检查值和普通 KZG witness 引用；禁止编码 $F_j(z_j)$ 或裸终点 |
| 拼接与输出头 | `P_{H_cat},P_{Y'},P_{Y^{pre}},P_{\tilde Y^{pre}},P_{Y^{(s)}}` | `PrivateWitness` | 只允许隐藏 commitment 与 `MaskedPrivateValue` opening |
| 最终 logits | `Y_public,P_Y` | `PublicOutput` | 完整数组在 `M_pub` 中规范编码；验证者计算 $P_Y$ 的公开挑战点评值并验证 commitment opening |
| 最终输出压缩对象 | `P_{Y^\star},P_{Y^{\star,edge}}` | `Public` | 由公开 logits 与公开挑战确定并参与绑定 |
| 商多项式 | `Q_{\mathcal D}` | `PrivateWitness` | 承诺与 opening 均 masked |
| opening value | `Eval_dom, Eval_ext` | 随对象继承 | private 对象必须为 `MaskedPrivateValue` |

实现验收规则：

1. `ProofSchema` 必须以固定顺序列出全部对象、域名、层编号、头编号、可见性、是否进入 commitment、是否进入 opening、是否进入 quotient。
2. Serializer 输出任何对象前必须按 `ProofSchema` 写入固定字段类型。`FieldElement` 的 `EvaluationKind` 只能是 `PublicValue` 或 `MaskedPrivateValue`；曲线点字段只能是 `Commitment` 或 `OpeningWitness`；跨块对象引用只能是 `MetadataReference`。关系证明不定义独立响应标量类型。
3. Verifier 必须执行 `private_plain_value_count == 0` 检查。
4. 未登记对象、重复对象、未知 tag、缺失可见性字段、private 对象明文输出，均为拒绝条件。

### 0.10 SRS manifest、文件格式与 setup 绑定

主实验不允许使用内存中临时生成且不可审计的 SRS。SRS 必须经独立 trusted setup 发布渠道认证，每个 SRS 由 `srs_manifest.json` 描述并与 proof 中的 `srs_id` 绑定。`srs_type` 的唯一合法值为 `trusted_srs`；serializer 不得生成其他值，parser 与 verifier 必须拒绝其他值。`srs_id` 只标识可跨数据集复用的 KZG 工件，不包含当前 `ProofSchema`、模型 commitment、数据集标识或实例维度。

manifest 至少包含：

| 字段 | 含义 |
|---|---|
| `srs_id` | 删除 `srs_id` 与 `generated_at` 后，对全部安全字段 canonical bytes 的摘要 |
| `curve_id` | 曲线与 pairing 后端标识 |
| `field_modulus_id` | KZG 标量域模数标识 |
| `srs_type` | 唯一合法值为 `trusted_srs`；其他值均为非法编码 |
| `ceremony_transcript_hash` | ceremony transcript 摘要 |
| `degree_bound` | 全局 $D_{max}$ |
| `kzg_assumption_profile` | 普通 KZG、shifted-KZG 可除性假设、允许域大小与各自安全位数的认证 profile |
| `domain_size_catalog` | 七类工作域允许使用的大小目录 |
| `domain_generator_catalog` | 每个 $(\mathcal D,n)$ 的 `entry_id`、$\omega_{\mathcal D,n}$ canonical bytes / hash、`root_derivation_id` 与固定 `row_order` |
| `pk_coef_hash` | `pk_coef.bin` 摘要 |
| `pk_lagrange_hashes` | 按 `entry_id` 排列的 Lagrange SRS 文件摘要目录 |
| `vk_hash` | `vk.bin` 摘要；包含 $G_2,[\tau]_2$ 与全部 $[Z_{\mathcal D,n}]_2$ |
| `vanishing_g2_hash` | 零化多项式 G2 commitment 目录摘要 |
| `srs_consistency_hash` | coefficient powers、目录生成元、Lagrange SRS 和零化多项式 G2 commitment 的一致性审计摘要 |
| `encoding` | 曲线点、域元素、数组长度、字节序和 canonical JSON 规则 |
| `generated_at` | 审计时间，不进入性能指标 |
| `generator_id` | SRS 生成工具标识 |
| `security_level` | 目标安全强度 |

每个 `domain_generator_catalog` 条目必须满足：$n$ 为允许大小，$\omega^n=1$，对 $n$ 的每个素因子 $q$ 有 $\omega^{n/q}\ne1$，`row_order` 固定为 $(1,\omega,\ldots,\omega^{n-1})$。若采用确定性根导出，manifest 还必须绑定 primitive-root 来源和导出公式；不同根或不同行序是不同目录条目，不能共享同一 `entry_id`。

文件布局固定为以 `srs_id` 命名的目录，包含 manifest、`pk_coef.bin`、`vk.bin`，以及每个目录项对应的 `pk_lagrange_<entry_id>.bin`。同一 manifest 可供不同数据集引用不同目录项；当前 `domain_cfg` 只能编码七个 `entry_id`，验证者从 manifest 重建 $(\mathcal D,n,\omega,row\_order)$，不得接受 proof 自行给出的替代生成元。

SRS 发布前执行确定性一致性审计：

1. 对 $0\le i<D_{max}$ 检查
   $$e([\tau^i]_1,[\tau]_2)=e([\tau^{i+1}]_1,G_2);$$
2. 对每个目录项，使用该项固定的 $\omega$ 和 `row_order` 重建全部 Lagrange 基系数，再用 `pk_coef.bin` 计算 $[L_i(\tau)]_1$，与对应文件逐点比较；
3. 使用 `pk_coef.bin` 计算 $[Z_{\mathcal D,n}]_1$，检查
   $$e([Z_{\mathcal D,n}]_1,G_2)=e(G_1,[Z_{\mathcal D,n}]_2);$$
4. 检查每个 Lagrange 文件、零化多项式目录和生成元目录使用相同 `entry_id`，禁止只按大小匹配；
5. 对全部 G1 / G2 元素执行规范解码、曲线上点、正确子群和必需非单位元检查。

审计失败时不得发布 manifest。审计报告 canonical bytes 不包含 `srs_id`、`srs_consistency_hash` 或 `generated_at`，避免摘要循环。

加载规则：

1. Prover 加载 `pk_coef.bin` 与当前 `domain_cfg` 引用的七个 Lagrange 文件，不得加载或生成 $\tau$；
2. Verifier 只加载 `vk.bin` 与 manifest，并按同一七个 `entry_id` 选择 $[Z_{\mathcal D,n}]_2$；
3. proof metadata 的 `srs_id`、`srs_type`、`srs_degree_bound`、`kzg_assumption_profile` 与 `encoding` 必须和 manifest 一致；`protocol_schema_id`、`instance_schema_hash` 分别与本地协议编译器和当前实例一致；
4. 任一文件摘要不匹配、目录项不存在、生成元 / 行序不一致、degree bound 不足、Lagrange SRS 或零化 G2 元素缺失、曲线 ID 不匹配或群元素非法时拒绝；
5. manifest 不计入 `proof_size_bytes`，但 `srs_id` 和七个 `entry_id` 必须在首个动态挑战前进入 transcript；
6. SRS 生成工具与 prover、verifier、benchmark 分离。主实验路径只能加载工件，不能生成、恢复或派生 $\tau$。

### 0.11 与当前项目实现对齐的主实验边界

GAT 前向语义固定为隐藏层多头注意力、隐藏层 ELU、隐藏层头维拼接、输出层注意力头、输出头偏置、最终输出头平均和节点分类任务。

协议范围包括安全 SRS、KZG commitment 与 opening、模型参数隐藏 commitment、私有 trace 掩蔽、`ProofSchema` 可见性登记、公开元数据、验证 gate、安全测试、proof-scope 对齐和结果导出。GAT 前向、route、PSQ、lookup、边排序、batch 语义和量化语义由本文公式唯一确定。

图结构和节点特征属于公开输入；模型参数、中间激活、注意力分数、Softmax 权重、lookup witness、route witness、PSQ 状态和折叠向量属于私有见证；完整最终 logits 属于公开输出。配置、public metadata、transcript、proof schema、verifier gate 和实验结果使用同一边界。

对比方法分别使用其公开声明的 setup 与证明后端，并报告 proof scope、输入输出口径、setup 假设、proof size 统计口径和硬件环境。同一数据集内的主方法与全部对比方法使用同一模型、同一数据预处理、同一量化配置和同一节点分类任务；不同数据集分别使用各自登记的模型与量化配置。

## 1. 参数生成

### 1.1 输入

参数生成由两个边界明确、顺序固定的阶段组成。SRS 工件生成只依赖协议级最大包络，不依赖具体数据集、模型参数或当前证明实例；实例注册只从已经发布并认证的 SRS 目录中选择工作域大小，不生成、更新或派生 $\tau$。

SRS 工件生成阶段输入为：

- 安全参数 $\lambda$ 与有限域模数 $p$；
- SRS 类型 $srs\_type$，固定为 `trusted_srs`；实例注册器必须拒绝其他值；
- 协议模板标识 $protocol\_schema\_id$；
- 七类工作域允许大小目录 $domain\_size\_catalog$，以及为每个目录项固定生成元、根导出算法和行顺序的 $domain\_generator\_catalog$；
- 协议允许的最大实例包络，包括 $N^{max},E^{max},N_{total}^{max},L^{max},K_{hid}^{max},K_{out}^{max},d_{in}^{max},d_h^{max},d_{cat}^{max},C^{max},B^{max}$、最大 coefficient object 长度与最大对象 family 数；
- 由协议模板最大包络生成的最大 `ProofSchema`、形式 SRS 坐标 `ObservationLedger`、`RandomIdentityLedger`、`SecurityBudgetLedger` 和逐对象资源上界模板；
- 编码、曲线、hash-to-field、Fiat--Shamir 域分离、`shift_algorithm_id`、`kzg_assumption_profile` 和安全级别配置。

实例注册阶段输入为：

- 已认证的 $srs\_id$、SRS manifest 与 $srs\_degree\_bound$；
- 当前实例规模 $(N,E,N_{total},G_{batch},node\_ptr,edge\_ptr)$；
- 总层数 $L$、各隐藏层 $\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1}$、各层输入维 $\{d_{in}^{(\ell)}\}_{\ell=1}^{L}$、输出类别数 $C$ 与输出头数 $K_{out}$；
- 完整 `quant_config`：包括 $S_{input},S_{\Delta},S_{exp},S_{agg,pre},S_{agg},S_{cat},S_{out,pre},S_{out,head},S_{out}$，全部隐藏 / 输出权重与注意力向量尺度、偏置尺度、静态表输入输出尺度、舍入模式、每个范围对象的公开 limb 数与完整 `integer_bound_ledger`；
- 范围检查位宽 $B$，并满足 $2^B<p$；
- 静态表 $T_H,T_{LReLU},T_{ELU},T_{exp},T_{range}$；
- 全部隐藏层参数族与输出层参数族；
- 隐私策略 $privacy\_policy$，主实验固定为 `model_and_trace_hidden`；
- 证明任务 $proof\_task$，主实验固定为 `node_classification_full_gat`。

实例注册器必须从上述输入选择唯一的七个 `domain_generator_catalog` 目录项，并确定 coefficient chunk 数、`ProofSchema`、`ObservationLedger`、`RandomIdentityLedger`、`SecurityBudgetLedger`、$\mathcal R_{deg}^{instance}$ 和 $instance\_schema\_hash$；全部对象必须落在 manifest 与 $protocol\_schema\_id$ 的最大包络内。`instance_schema_hash` 不是外部自由输入，也不得参与 SRS 工件生成。


### 1.2 输出

SRS 工件生成阶段输出：

- coefficient-basis 证明键 $PK_{coef}$；
- 目录中全部 $\{PK_{lag}^{(\mathcal D,n)}\}_{(\mathcal D,n)\in domain\_size\_catalog}$；
- 验证键 $VK_{KZG}=\{G_2,[\tau]_2,[Z_{\mathcal D,n}]_2\}_{(\mathcal D,n)\in domain\_size\_catalog}$；
- 经一致性审计的 trusted-setup SRS manifest 与 $(srs\_id,srs\_type=\texttt{trusted\_srs},srs\_degree\_bound,kzg\_assumption\_profile)$；
- 独立的只读 `ProtocolCompilerArtifact` 及其 `protocol_schema_id`；工件包含最大 `ProofSchema`、三份账本模板、constraint / point-group / degree / padding registries、`shift_algorithm_id` 与 KZG 假设 profile，但不作为 `srs_id` 的输入字段。

实例注册阶段输出：

- 当前实例选定的七个工作域与零化多项式；
- 当前实例展开后的只读 `ProtocolCompilerArtifact`、$instance\_schema\_hash$、`ProofSchema`、`ObservationLedger`、`RandomIdentityLedger`、`SecurityBudgetLedger` 与 $\mathcal R_{deg}^{instance}$；
- 隐私策略元数据 $privacy\_policy$；
- 静态表验证键 $VK_{static}$；
- 模型验证键 $VK_{model}$。

实例注册阶段不得输出新的 SRS 幂、修改 manifest 或改变 $srs\_id$。

### 1.3 工作域

为避免在多层GAT的实现中为每一层重复使用一套不同长度的共享维工作域，正式协议统一采用全局共享域口径：

- 对全部隐藏层的输入共享维，统一用一个 $\mathbb H_{in}$；

- 对全部隐藏层单头输出共享维，统一用一个 $\mathbb H_{d_h}$；

- 对全部隐藏层拼接共享维与最终输出层输入共享维，统一用一个 $\mathbb H_{cat}$。

当前实例的每个 $\omega_{\mathcal D}$ 必须由 `domain_cfg` 引用的 `domain_generator_catalog` 目录项给出；后续“取最小二次幂长度”的表述只选择目录大小，不重新选择生成元。若所需大小不存在对应目录项，实例注册失败，不允许现场生成新的根或 Lagrange SRS。

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

取最小二次幂长度 $n_{edge}$ 满足：$n_{edge}\ge \max\{N,E,Nd_h^{max},NC,|T_{LReLU}|,|T_{ELU}|,|T_{exp}|,2^B\}+2$。coefficient-basis 对象不要求整体装入单个边域；它们严格按第 0.7.8 节的固定容量 $n_{edge}-1$ 切分，因此矩阵完整系数长度不得再次作为 $n_{edge}$ 的下界

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

SRS 工件生成严格分为“协议编译”和“ceremony”两个顺序阶段。协议编译不接触 $\tau$，先根据 `protocol_schema_id`、最大实例包络、`domain_size_catalog` 与 `domain_generator_catalog` 确定 `ProofSchema` 模板、形式观察账本、随机身份账本和全局次数界；只有这些公开对象全部固定后，ceremony 才采样 $\tau$ 并生成 SRS。禁止先生成 SRS、再通过修改 schema 或账本扩大 $D_{max}$。

`domain_generator_catalog` 对每个 $(\mathcal D,n)$ 固定唯一目录项：

$$
DomainEntry_{\mathcal D,n}=
(\mathcal D,n,\omega_{\mathcal D,n},root\_derivation\_id,row\_order),
$$

其中 $\omega_{\mathcal D,n}$ 使用 canonical field encoding，阶必须恰为 $n$；`row_order` 固定为 $(1,\omega,\ldots,\omega^{n-1})$。生成元只能由 `root_derivation_id` 规定的确定性算法产生，或作为 manifest 安全字段直接登记；`domain_cfg` 只能引用目录项 ID，不得提交新的生成元。

**形式 SRS 坐标 ObservationLedger**

`ObservationLedger` 是协议编译器和安全证明使用的公开形式账本，不把未知陷门 $\tau$ 当作可计算字段，也不声称 verifier 可以从群元素恢复 blinder。对每个私有基础对象 $O$，把其独立随机系数按固定顺序写成

$$\mathbf r_O=(r_{O,0},\ldots,r_{O,h_O-1})^\top.$$

其随机贡献唯一写成

$$B_O(X)=\sum_{a=0}^{h_O-1}r_{O,a}b_{O,a}(X),$$

其中：domain-vanishing blinding 使用 $b_{O,a}(X)=Z_{\mathcal D}(X)X^a$；coefficient-tail blinding 使用 $b_{O,a}(X)=X^{L_O+a}$；proof-local 模型再盲化使用 $b_{O,a}(X)=Z_{\mathcal D}(X)X^a$。商掩码及其他明确登记独立随机性的对象使用其 schema 给出的公开基多项式。

引入只用于符号编译的形式不定元 $\mathsf T$，并在域

$$\mathbb K=\mathbb F_p(\mathsf T,\mathbf X_{FS})$$

上建立观察矩阵；$\mathbf X_{FS}$ 是全部尚未实例化的 Fiat--Shamir 挑战形式变量。每个实际序列化私有观察恰生成一行：

1. 对 commitment $[P+B_O]$，第 $a$ 列为 $b_{O,a}(\mathsf T)$；一个群元素对应一行形式指数线性式，不把其展开系数错误地当成多个可见群元素。
2. 对 `MaskedPrivateValue` $(P+B_O)(z)$，第 $a$ 列为 $b_{O,a}(z)$。
3. 对单点 KZG witness
   $$W_z(X)=\frac{(P+B_O)(X)-(P+B_O)(z)}{X-z},$$
   第 $a$ 列为
   $$\frac{b_{O,a}(\mathsf T)-b_{O,a}(z)}{\mathsf T-z}.$$
4. 对同点批量 witness，先按 `ProofSchema` 的 claim 顺序和形式 folding challenge 构造折叠多项式，再按上一条生成唯一一行；单项 witness 不再重复计入。
5. 对平移多项式 $\widetilde P(X)=P(X+\Delta)$，commitment、平移检查值和 opening witness 分别以 $b_{O,a}(\mathsf T+\Delta)$、$b_{O,a}(y+\Delta)$ 及其标准 quotient 形式生成；平移不产生新的 blinder 列。
6. 对线性组合、CRPC、basis link、七域商和 proof-local 模型对象，沿公开表达式 DAG 传播依赖；只有 `ProofSchema` 明确分配独立随机系数时才新建列。
7. `MetadataReference`、同一字段的重复引用以及验证者本地重建的 `PublicValue` 不生成新行。

跨多个私有来源的字段按全局变量顺序拼接。为避免把“blinder 自身有秩”误当成“私有语义已被覆盖”，账本同时建立两类变量：

- $\mathbf s$：每个基础私有 committed polynomial 的未掩蔽 coefficient 向量。为给出保守且可组合的泄露界，编译器先把这些 coefficient 视为相互独立的语义坐标，再沿公开表达式 DAG 传播到派生对象；
- $\mathbf r$：全部独立 blinder coefficient，按对象、层、头、域和 schema 顺序排列。

公开观察空间固定为直和

$$\mathcal V_{obs}=\mathbb K^{n_{field}}\oplus\mathbb K^{n_{group}},$$

其中 `FieldElement` 观察只进入第一分量，G1 commitment / opening witness 的形式指数只进入第二分量；编译器不得在两个分量之间执行行消元。对 transcript 的每个挑战阶段 $u$，由截至该阶段已序列化的规范字段生成

$$\mathbf o_u=A_{sec,u}(\mathsf T,\mathbf X_{FS})\mathbf s+A_{blind,u}(\mathsf T,\mathbf X_{FS})\mathbf r+\mathbf o_{pub,u}.$$

`ObservationLedger` 的核心覆盖条件不是单独检查 $A_{blind,u}$ 的秩，而是对每个阶段都要求

$$\operatorname{Im}(A_{sec,u})\subseteq\operatorname{Im}(A_{blind,u}),$$

等价地，

$$\operatorname{rank}\big[A_{blind,u}\mid A_{sec,u}\big]=\operatorname{rank}(A_{blind,u}).$$

协议编译器必须使用固定列序对 $A_{blind,u}$ 做 canonical RREF，并唯一求出字典序最小的有理函数矩阵 $K_u$，使

$$A_{sec,u}=A_{blind,u}K_u.$$

若任一阶段不存在该 $K_u$，则 schema 在 ceremony 前即判无效。对含 $(\mathsf T-z)^{-1}$ 的 witness 行，先按 claim 顺序约分公共因子，再乘 canonical 最小公分母；$K_u$ 的分母、RREF 枢轴 minor 和覆盖等式分子必须分别化为互素多项式，并把所有非恒零证书的摘要、次数界和阶段标签写入账本。`MetadataReference` 和同一点 point-grouped folding 只按最终序列化字段计一次。

对每个对象 $O$，$q_O^{max}$ 固定定义为：在全部 transcript 阶段同时满足上述覆盖条件时，该对象 blinder 基前缀所需长度的最大值。编译器按对象 canonical 顺序选择字典序最小的向量 $(q_O^{max})_O$；不得把“私有字段数量”“观察行数量”或 $\operatorname{rank}(A_{blind})$ 单独当作 $q_O^{max}$。由此必须拒绝例如

$$o_1=s+r,\qquad o_2=2s+r$$

这类 blinder 矩阵秩非零、但仍可通过 $o_2-o_1=s$ 消去掩码得到私有值的错误登记。

挑战实例化后，凡退化只由公开挑战决定的因子，必须进入对应挑战的公开排除多项式并由 counter 重采样；任何仍依赖 $\mathsf T$ 的 RREF 枢轴、覆盖矩阵 $K_u$ 分母或覆盖等式退化不能由 verifier 检查，统一记为 `BadCRS_obs`。令 $\mathcal S_\tau=\mathbb F_p^\ast\setminus\bigcup_{(\mathcal D,n)}\mathbb H_{\mathcal D,n}$，$N_\tau=|\mathcal S_\tau|$。若该对象全部形式证书关于 $\mathsf T$ 的去重根多项式次数上界为 $d_{O,crs}$，则其在从 $\mathcal S_\tau$ 均匀采样的隐藏 $\tau$ 上退化的概率至多为 $d_{O,crs}/N_\tau$。所有对象的 $d_{O,crs}$ 进入安全界。特别地，平移 blinder 中 $Z_{\mathcal D}(\mathsf T+\Delta)$ 的根事件必须计入 `BadCRS_obs`，不得伪装成公开 counter 条件。

诚实证明者固定使用

$$h_O=q_O^{max},$$

并以 CSPRNG 独立采样全部 $\mathbf r_O$。该长度、随机源和跨 proof 不复用属于诚实证明者生成与合规审计要求；密码学 verifier 强制检查的是 schema、字段集合、账本摘要、公开挑战排除条件和全局次数界，不声称从单份 proof 判断随机来源、实际最高非零系数或跨 proof 复用。

**RandomIdentityLedger 的唯一生成算法**

`RandomIdentityLedger` 登记每个承诺后随机检查，不依赖恶意证明者遵守较小的对象级诚实次数。对每个随机身份 $c$，固定记录：

$$
(label_c,expression\_dag_c,challenge\_stage_c,
\mathcal A_c,E_c(Y),B_c,d_c^{max},multiplicity_c).
$$

其中：

1. $expression\_dag_c$ 是验证者实际计算的多项式身份表达式；
2. 每个已承诺输入节点的恶意次数一律按全局 $D_{max}$，公开固定多项式按其真实公开次数；加法取最大次数，乘法取次数和，移位不改变次数，乘以 $Z_{\mathcal D}$ 增加 $n_{\mathcal D}$；
3. 由此得到 verifier 可依赖的最坏次数 $d_c^{max}$，不得用无法从 commitment 验证的实际 $\deg f_c$ 替代；
4. $E_c(Y)$ 是全部公开拒绝条件去重后的乘积，$B_c=\deg E_c$，规范候选集合为
   $$\mathcal A_c=\{y\in\mathbb F_p:E_c(y)\ne0\},\qquad |\mathcal A_c|\ge p-B_c;$$
5. 参数生成必须逐项检查
   $$d_c^{max}<p-B_c;$$
6. tuple compression 把两个不同长度至多为 $t_c$ 的元组压缩为 $\sum_{j=0}^{t_c-1}\eta^jx_j$ 时，其碰撞身份单独登记，次数上界为 $t_c-1$；
7. 同一挑战复用的多个身份按账本的 `multiplicity_c` 计入联合界，不得漏计或因同一 point-grouped witness 而错误合并代数身份。

账本必须覆盖七域 quotient identity、公开多项式绑定、KZG 平移一致性、关系零值、CRPC、basis link 的随机检查、所有 tuple compression、随机线性折叠和其他承诺后随机身份。每个 LogUp / route / indexed-view 多重集合关系还必须登记清分母后的有理终点身份：若表端和查询端有效项总数上界为 $q_{logup,c}$，则编译器从实际有理式表达式 DAG 计算其分子最坏次数并写入 $d_c^{max}$；该项与 tuple-compression 碰撞项分别计数。纯 KZG opening 方程的底层绑定优势单独计入 $\operatorname{Adv}_{KZG}$，不重复作为 Schwartz--Zippel 项。

对象级 $b_O$ 只用于诚实 prover 的内存、MSM 和次数规划，并由 `instance_schema_hash` 绑定；普通 KZG verifier 只强制每个 committed polynomial 的次数不超过全局 $D_{max}$。文档、代码和安全定理不得声称 verifier 能证明实际次数不超过 $b_O$。

**SecurityBudgetLedger 的唯一生成算法**

`security_level` 不能只作为 manifest 标签。协议编译器必须生成 `SecurityBudgetLedger`，并使用精确整数 / 有理数运算登记：

$$
(\lambda_{target},Q_{FS}^{max},\lambda_{KZG},\lambda_{register},\lambda_{RO},
\varepsilon_{alg},\varepsilon_{abort},\varepsilon_{crs},\varepsilon_{total}).
$$

其中

$$
\varepsilon_{alg}=Q_{FS}^{max}\sum_{c\in RandomIdentityLedger}
 multiplicity_c\frac{d_c^{max}}{p-B_c},
$$

$$
\varepsilon_{abort}=\sum_\nu\frac{q_{den,\nu}}p,
\qquad
\varepsilon_{crs}=\sum_O\frac{d_{O,crs}}{N_\tau},
$$

且

$$
\varepsilon_{total}
=2^{-\lambda_{KZG}}+2^{-\lambda_{register}}+2^{-\lambda_{RO}}
+\varepsilon_{alg}+\varepsilon_{abort}+\varepsilon_{crs}.
$$

$Q_{FS}^{max}$ 必须覆盖一次攻击中允许的随机预言机查询、公开 counter 尝试和 prover-chosen nonce grinding；曲线、KZG 假设、注册证明和 hash / random-oracle profile 分别提供可审计的安全位数下界。编译器必须检查 $\lambda_{KZG},\lambda_{register},\lambda_{RO}>0$，并以这些 profile 声明的直接优势上界执行唯一的总预算检查

$$\varepsilon_{total}\le2^{-\lambda_{target}}.$$

仅检查 $d_c^{max}<p-B_c$ 不足以通过安全 gate。所有分数必须以 canonical numerator / denominator 编码，禁止使用浮点近似；账本摘要进入 `protocol_schema_id`，当前实例裁剪后的实际项进入 `instance_schema_hash`。任何预算项缺失、重复、分母非正、总界超过目标或 manifest 的 `security_level` 与 $\lambda_{target}$ 不一致时，实例注册必须拒绝。

**ProtocolCompilerArtifact 与验证上下文加载规则**

协议编译器必须输出一个 canonical、只读的 `ProtocolCompilerArtifact`，而不能只输出其摘要。该工件至少包含：

- `proof_schema`；
- `observation_ledger`；
- `random_identity_ledger`；
- `security_budget_ledger`；
- `constraint_registry` 与逐工作域 quotient 登记顺序；
- `standard_point_group_registry_template`；
- `degree_resource_registry`；
- `shift_algorithm_spec`；
- `public_column_padding_registry`；
- `kzg_assumption_profile`；
- transcript 阶段表、对象 canonical 顺序和 serializer tag 表。

协议级模板部分的 canonical bytes 摘要固定为 `protocol_schema_id`；当前实例裁剪、展开并填入层数、头数、维度、对象 family、chunk、claim-group 模板、canonical claim IDs、owner 规则、点导出标签、counter 规则、碰撞合并规则和 witness 所有权规则后的 canonical bytes 摘要固定为 `instance_schema_hash`。Fiat--Shamir 运行时才产生的实际标量点及其碰撞合并后的最终 `StandardPointGroupRegistry` 不直接写入 `instance_schema_hash`，而由 transcript 和 proof 的 canonical registry 编码绑定。哈希只用于认证工件，不能替代工件内容。prover 和 verifier 必须从本地只读注册表按 ID 加载同一工件，或者由调用方显式提供完整工件；加载失败、工件字段缺失、canonical 重编码摘要不匹配、模板与实例部分交叉引用不闭合时必须拒绝。

**proof-local 模型再盲化的 shifted-KZG 可除性边界**

对工作域大小 $n=n_{\mathcal D}$，记

$$A(X)=P_{model}^{proof}(X)-P_{model}^{vk}(X),\qquad C_S=[S_{model}],$$

并由现有 commitments 本地重建

$$C_{shift}=[P_{model}^{proof}]-[P_{model}^{vk}]+[S_{model}].$$

验证者从目录元素计算

$$[\tau^n]_2=[Z_{\mathcal D}]_2+G_2,$$

并使用与正文可除 pairing 完全等价的 shifted degree-bound 方程

$$e(C_S,[\tau^n]_2)=e(C_{shift},G_2).$$

本文将其安全性明确建模为单生成元 KZG 的 `ShiftedKZG-Divisibility(D_{max},n)` 代数知识假设：对获得 $PK_{coef}$、$G_2,[\tau]_2,[\tau^n]_2$ 的概率多项式时间代数对手，若上述方程成立且 $C_S,C_{shift}$ 均为规范 SRS 线性组合，则除可忽略概率外存在

$$\deg S_{model}\le D_{max}-n$$

且

$$A(X)=(X^n-1)S_{model}(X).$$

该假设是现有 pairing 的可靠性解释，不新增 commitment、proof block、工作域或序列化字段。`kzg_assumption_profile` 必须绑定该假设、$D_{max}$、允许的全部 $n$ 和对应安全位数，并进入 SRS manifest、`protocol_schema_id` 与 `SecurityBudgetLedger`。若某个域满足 $D_{max}<n$，或 profile 不覆盖该 $(D_{max},n)$，则实例注册拒绝。

**全局次数界与 ceremony**

对每类工作域定义

$$n_{\mathcal D}^{max}=\max\{n:(\mathcal D,n)\in domain\_size\_catalog\}.$$

编译器先根据最大实例包络、诚实 blinder 系数数和每个构造的确定性代数公式，计算全部诚实生成对象的资源次数上界 $b_O^{hon}$：乘积取次数和，除以已证明可除的固定因子后减去其次数，opening witness 的上界为被打开多项式上界减一。全局 SRS 界唯一取为覆盖全部 $b_O^{hon}$ 的最小允许整数 $D_{max}$。随后 `RandomIdentityLedger` 在分析恶意 proof 时，把每个已承诺输入的可验证次数统一按该全局 $D_{max}$ 计入，得到 $d_c^{max}$ 并检查全部 $d_c^{max}<p-B_c$。不得用恶意输入的 $D_{max}$ 反向递推诚实 CRPC quotient 或商对象的资源次数，也不得建立循环次数方程。若 $D_{max}\ge p$、任一随机身份检查失败或超过曲线安全配置，协议模板无效。不同数据集的 $\mathcal R_{deg}^{instance}$ 必须落在该最大包络内，因而可以复用同一 `srs_id`。

完成上述编译后，ceremony 验证 `domain_generator_catalog` 中每个生成元的阶、canonical encoding、根导出规则和行顺序，并检查目录中全部工作域点的并集严格小于 $\mathbb F_p^\ast$。随后采样

$$\tau\xleftarrow{\$}\mathbb F_p^\ast\setminus\bigcup_{(\mathcal D,n)}\mathbb H_{\mathcal D,n}.$$

若任一 $Z_{\mathcal D,n}(\tau)=0$、$[\tau]_2=\mathcal O_{\mathbb G_2}$ 或待输出群元素非法，setup 在输出任何文件前重新采样。生成工件后必须销毁 $\tau$；主实验路径不得保存、恢复或派生它。

输出：

$$PK_{coef}=\{[\tau^0]_1,\ldots,[\tau^{D_{max}}]_1\},$$

$$PK_{lag}^{(\mathcal D,n)}=\{[L_0^{(\mathcal D,n)}(\tau)]_1,\ldots,[L_{n-1}^{(\mathcal D,n)}(\tau)]_1\},$$

$$VK_{KZG}=\Big\{G_2,[\tau]_2,\{[Z_{\mathcal D,n}]_2\}_{(\mathcal D,n)}\Big\}.$$

实现要求：

1. `KZGKeyPair` 不包含 `tau`；`commit` 不调用 `polynomial.evaluate(tau)`；
2. coefficient-basis 对象使用 $PK_{coef}$，工作域 evaluation 对象使用 manifest 指定的 $PK_{lag}^{(\mathcal D,n)}$；
3. `srs_id` 只绑定 SRS 工件和域目录，`protocol_schema_id` 绑定协议级 `ProtocolCompilerArtifact`，`instance_schema_hash` 绑定当前实例展开后的完整编译工件；
4. prover 和 verifier 必须从同一 canonical 编译器重建账本摘要，任何字段、挑战阶段、表达式 DAG、候选集合或次数上界不一致时拒绝；
5. `BadCRS_obs` 是安全证明中的坏 CRS 事件，不是 verifier 可通过读取 $\tau$ 检查的运行时 gate。

### 1.5 静态表与模型承诺

#### 1.5.1 全局特征表多项式

对 $u=v\cdot d_{in}+j$，定义特征表工作域取值 $T_H^{FH}[u]=T_H[v,j]$，填充区取零。静态特征表多项式固定为 $\mathbb H_{FH}$ 上的 evaluation-basis 插值：

$$P_{T_H}(X)=\sum_{u=0}^{n_{FH}-1}T_H^{FH}[u]L_u^{(FH)}(X).$$

该对象只用于特征检索表端约束。节点局部特征在矩阵乘法和 coefficient-basis 关系中使用独立的行优先系数多项式 $P_H$；两种编码不得共用同一标签。

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

定义：$VK_{static} = \{static\_tables\_id,table\_generator\_id,table\_spec\_hash,range\_and\_scale\_certificate,[V_{T_H}], [V_{T_{LReLU},x}], [V_{T_{LReLU},y}], [V_{T_{ELU},x}], [V_{T_{ELU},y}], [V_{T_{exp},x}], [V_{T_{exp},y}], [V_{T_{range}}] \}$。注册器必须按第 0.4 节的确定性生成规范重建全部表并核对 canonical bytes；proof 只能引用这一认证工件，不能提交同名替代表。

#### 1.5.7 模型承诺与模型验证键

对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$、以及该层中的每个注意力头 $r\in\{0,1,\ldots,K_{hid}^{(\ell)}-1\}$，承诺：$[V_{W^{(\ell,r)}}],[V_{a_{src}^{(\ell,r)}}],[V_{a_{dst}^{(\ell,r)}}]$

对每个输出头 $s\in\{0,1,\ldots,K_{out}-1\}$，承诺：$[V_{W^{(out,s)}}],\quad [V_{a_{src}^{(out,s)}}],\quad [V_{a_{dst}^{(out,s)}}],\quad [V_{b^{(out,s)}}]$

因此，$VK_{model}$ 由两部分按固定顺序拼接而成：先按层序和头序写入全部隐藏层承诺 $[V_{W^{(\ell,r)}}],[V_{a_{src}^{(\ell,r)}}],[V_{a_{dst}^{(\ell,r)}}]$；再按 $s=0,\ldots,K_{out}-1$ 写入全部输出层承诺 $[V_{W^{(out,s)}}],[V_{a_{src}^{(out,s)}}],[V_{a_{dst}^{(out,s)}}],[V_{b^{(out,s)}}]$。

模型参数值属于私有见证。`VK_model`中只能出现模型参数的隐藏承诺和承诺摘要，不能出现裸参数值，也不能出现裸参数的 opening。若模型参数承诺需要在多次证明中复用，复用对象仍然必须是隐藏承诺；验证者只使用承诺和后续零知识绑定证明检查模型一致性。

模型参数不使用独立参数域。隐藏层 $W^{(\ell,r)}$ 的每个输出列在 $\mathbb H_{in}$ 上形成 canonical trace；$a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}$ 在 $\mathbb H_{d_h}$ 上形成 canonical trace；输出层 $W^{(out,s)}$ 的每个类别列在 $\mathbb H_{cat}$ 上形成 canonical trace；$a_{src}^{(out,s)},a_{dst}^{(out,s)},b^{(out,s)}$ 在 $\mathbb H_C$ 上形成 canonical trace。每个 canonical trace 使用 domain-vanishing blinding。

生成这些 commitments 之前，模型登记器必须按 `integer_bound_ledger` 检查每个模型参数的规范有符号代表，并为每个参数族记录公开绝对值上界。`model_params` 摘要只哈希隐藏 commitments、公开上界、形状、布局、尺度配置和模型身份，不得哈希裸参数 canonical bytes、参数明文或 blinding randomness，以避免形成可离线枚举的确定性私有参数摘要。$VK_{model}$ 携带该摘要和公开上界，不携带参数明文。

模型登记是验证上下文的认证边界：验证者只接受由预先信任的模型登记流程生成并认证的 $VK_{model}$；该流程必须实际执行参数范围和尺度链检查。若部署不信任模型登记器，则必须在模型注册时使用本文已有的 KZG、范围 LogUp 和 quotient 机制生成一次性模型范围 / 尺度登记证明，并在接受 $VK_{model}$ 前验证；该注册证明不属于单次推理的九个顶层 proof block，也不计入单次推理 proof size。超出登记上界或尺度链不一致的模型参数不得登记为有效模型。

CRPC 使用的 coefficient-basis relation view 采用第 0.7.8 节规定的 coefficient-tail blinding，并通过 indexed-view 与 basis link 绑定到上述 canonical commitments。`VK_model` 只保存隐藏后的长期 canonical commitments、形状、布局、次数界和摘要。每份 proof 必须先按第 0.7.8 节生成 proof-local 模型 canonical commitments 及其 $[S_{model}]$ 可除证明；随后根据该次 CRPC 的 $L_C$ 生成使用独立随机尾项的 coefficient relation commitments，并将它们编码在对应的 $\Pi_{bind}$ 子块中。这些 proof-local commitments 不写入 `VK_model`。所有投影矩阵、注意力向量和输出偏置在证明中的使用都必须通过第 0.7.8 节的 KZG-only 绑定证明完成；proof 不能打开裸模型参数，也不能把模型参数的随机点评值作为明文字段输出。

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

12. 对每个隐藏层 $\ell$，行优先隐藏矩阵有效区选择器多项式：$P_{Q_{Nd_h}^{valid,(\ell)}}(X)=\sum_{q=0}^{n_{edge}-1}Q_{Nd_h}^{valid,(\ell)}[q]L_q^{(edge)}(X)$

13. 输出矩阵有效区选择器多项式：$P_{Q_{NC}^{valid}}(X)=\sum_{q=0}^{n_{edge}-1}Q_{NC}^{valid}[q]L_q^{(edge)}(X)$

14. 节点域枚举多项式：$P_{Idx_N}(X)=\sum_{i=0}^{n_N-1}Idx_N[i]L_i^{(N)}(X)$

15. 输入共享维枚举多项式：$P_{Idx_{in}}(X)=\sum_{m=0}^{n_{in}-1}Idx_{in}[m]L_m^{(in)}(X)$

16. 隐藏层共享维枚举多项式：$P_{Idx_{d_h}}(X)=\sum_{j=0}^{n_{d_h}-1}Idx_{d_h}[j]L_j^{(d_h)}(X)$

17. 拼接共享维枚举多项式：$P_{Idx_{cat}}(X)=\sum_{m=0}^{n_{cat}-1}Idx_{cat}[m]L_m^{(cat)}(X)$

18. 输出类别枚举多项式：$P_{Idx_C}(X)=\sum_{c=0}^{n_C-1}Idx_C[c]L_c^{(C)}(X)$

19. 特征检索表端行列辅助多项式：

	$P_{Row_{feat}^{tbl}}(X)=\sum_{u=0}^{n_{FH}-1}Row_{feat}^{tbl}[u]L_u^{(FH)}(X)$

	$P_{Col_{feat}^{tbl}}(X)=\sum_{u=0}^{n_{FH}-1}Col_{feat}^{tbl}[u]L_u^{(FH)}(X)$

20. 特征检索查询端局部行列辅助多项式：

	$P_{Row_{feat}^{qry}}(X)=\sum_{q=0}^{n_{FH}-1}Row_{feat}^{qry}[q]L_q^{(FH)}(X)$

	$P_{Col_{feat}^{qry}}(X)=\sum_{q=0}^{n_{FH}-1}Col_{feat}^{qry}[q]L_q^{(FH)}(X)$

21. 特征检索查询端绝对节点编号辅助多项式：

	对每个 $q=i\cdot d_{in}+j$，定义$I_{feat}^{qry}[q]=I_i$，然后插值得到：$P_{I_{feat}^{qry}}(X)=\sum_{q=0}^{n_{FH}-1}I_{feat}^{qry}[q]L_q^{(FH)}(X)$

22. batch 节点图编号辅助多项式：

	定义：$P_{node\_gid}(X)=\sum_{i=0}^{n_N-1}node\_gid(i)L_i^{(N)}(X)$

23. batch 边图编号辅助多项式：

	定义：$P_{edge\_gid}(X)=\sum_{k=0}^{n_{edge}-1}edge\_gid(k)L_k^{(edge)}(X)$

24. 图块起点 / 末尾选择器多项式：

	$P_{Q_{graph\_new}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{graph\_new}^{edge}[k]L_k^{(edge)}(X)$

	$P_{Q_{graph\_end}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{graph\_end}^{edge}[k]L_k^{(edge)}(X)$

25. batch-aware 组起点 / 组末尾选择器多项式：

	$P_{Q_{grp\_new}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{grp\_new}^{edge}[k]L_k^{(edge)}(X)$

	$P_{Q_{grp\_end}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{grp\_end}^{edge}[k]L_k^{(edge)}(X)$

这些对象全部由验证者根据公共输入本地重建，不属于动态承诺对象；其中与$Q_{in}^{valid,(\ell)},Q_{d_h}^{valid,(\ell)},Q_{cat}^{valid,(\ell)},Q_{cat}^{valid,(out)}$ 对应的选择器族，必须按层指标逐一重建，不得以“共享最大工作域”为由把它们压缩成一套无层标选择器。

本节全部公共辅助多项式必须遵守第 0.7.5 节的 `public_column_padding_registry`：全域枚举列保持完整枚举，$src,dst,node\_gid,edge\_gid,Row/Col_{feat}^{tbl/qry},I_{feat}^{qry}$ 等有效支撑列在无效区严格为零。验证者以同一规则重建完整长度数组后再插值；任何 padding byte、commitment claimed value 或本地评值不一致都必须拒绝。

$P_{node\_gid},P_{edge\_gid},P_{Q_{graph\_new}^{edge}},P_{Q_{graph\_end}^{edge}},P_{Q_{grp\_new}^{edge}},P_{Q_{grp\_end}^{edge}}$ 一律属于验证者必须本地重建并吸入 transcript 的公共拓扑对象。若 $G_{batch}=1$，则它们的取值按单图公式唯一确定；实现不得因为单图而省略这些对象。

在上述公共拓扑对象中，$P_{Q_{new}^{edge}},P_{Q_{end}^{edge}}$ 只允许作为图内基础辅助对象存在；所有正式状态机、商多项式恒等式以及验证者检查都必须只消费 $P_{Q_{grp\_new}^{edge}},P_{Q_{grp\_end}^{edge}}$，不得直接消费未带 batch 语义的单图选择器


## 2. 见证生成与承诺

### 2.0 总体顺序与实现约束

证明生成采用严格的多轮提交顺序，不允许在尚未提交挑战前置 commitments 时预先生成 Fiat–Shamir 挑战。工程实现分为下列逻辑阶段：

1. 前向阶段：计算不依赖证明挑战的 GAT 明文对象和基础 trace；
2. 基础提交阶段：对基础 trace、canonical tensor、proof-local 模型对象和其他挑战来源完成多项式编码与 commitment；
3. 挑战依赖提交阶段：严格按第 3.2 节执行“吸入前置 commitments → 生成挑战 → 构造依赖该挑战的列 → 提交这些列”的多轮过程；$Table,Query,m$ 必须先于 $\beta$，分母逆元和累加器必须后于 $\beta$；
4. 商与开放阶段：全部约束来源 commitments 固定后生成 $\alpha_{quot}$，提交七域校正商，再生成 opening 点与唯一 claimed-value 字段；随后汇总九个顶层块的全部普通 KZG claims，编译全局 `StandardPointGroupRegistry`，每个标量点只生成一个 point-grouped KZG witness。

前向数值计算可以与密码学 commitment 后端解耦，但 FFT / MSM 不得跨越其所属的 transcript 提交轮次延迟到挑战之后。任何实现若先生成挑战、后补交本应被该挑战绑定的 commitment，必须拒绝。

#### 2.0.1 安全承诺、掩蔽与 opening 统一规则

凡定义在工作域 $\mathbb H_{\mathcal D}$ 上的私有 evaluation-basis 动态承诺 $[P]$，其实际含义为第 0.2.1 节的 $[P^{zk}]$。凡使用行优先或 CRPC 系数编码的私有 coefficient-basis 动态承诺 $[P]$，其实际含义为第 0.7.8 节规定的 $[\widehat P]$。公开多项式和 `PublicOutput` 多项式的掩蔽项固定为零。

所有模型相关、激活相关、注意力相关、路由相关、lookup query 相关、PSQ 相关、折叠向量相关和状态机相关多项式都属于 private witness polynomial，必须按照其登记的 basis 与原生工作域采样独立 blinding 多项式。

证明者必须先完成前向阶段，随后生成所有基础 trace 值，再依据 `privacy_policy` 构造 $P^{zk}$ 并提交承诺。任何实现不得为了性能把裸 $P$ 的外点评值、折叠向量、完整中间矩阵或模型参数评值写入证明对象。

所有 `Eval_ext`、`Eval_dom` 和 `Open_dom` 均以 $P^{zk}$ 为 opening 对象。若某条代数关系在语义上需要裸 $P$，必须通过第 3.3 节规定的商校正或第 0.7.8 节的 KZG 同点平移归一化关系处理，不能通过暴露裸评值处理。

所有 private coefficient-basis 对象使用 coefficient-tail blinding，并与七域 canonical trace 绑定。所有工作域商 commitment 的实际对象为第 3.3 节定义的 $t_{\mathcal D}^{zk}$；证明者还必须提交对应商掩码 $R_{t,\mathcal D}$。不得在证明主路径中保留未盲化 coefficient commitment、未校正语义商 commitment 或可被 serializer 访问的裸多项式副本。


全文中任何写成 $1/(Table+\beta)$ 或 $1/(Query+\beta)$ 的见证公式均为第 0.8.2 节分母逆元列的数学简写。 Serializer、commitment registry 和 quotient registry 只接受显式 $J$ 列；分式记号不得形成独立执行路径。正式见证必须在 $\beta$ 生成后构造并承诺 $J_{tbl},J_{qry}$，使用乘法形式生成累加器；所有 commitment、evaluation、opening 和 quotient manifest 必须包含这两列。依赖 $\beta$ 的 $J$ 与累加器 commitments 不得在 $\beta$ 之前吸入 transcript。

长期模型 commitments 只作为 $VK_{model}$ 锚点。证明生成首先提交 proof-local 模型 canonical commitments 与 $[S_{model}]$，通过 $[Z_{\mathcal D}]_2$ 可除检查后，所有后续关系只引用 proof-local commitments。

符号 $[V_{W}], [V_{a}], [V_b]$ 在任何动态挑战输入中均表示有序二元组“$VK_{model}$ 中的长期锚点 commitment + 本 proof 的对应 proof-local commitment”；实际 CRPC、indexed-view、basis link 和 opening 只能引用 proof-local commitment。仅吸入长期锚点而遗漏 proof-local commitment，或在挑战产生后替换 proof-local commitment，均为拒绝条件。


### 2.1 数据加载与特征检索

#### 2.1.1 原始特征

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 与每个输入维索引 $j\in\{0,1,\ldots,d_{in}-1\}$，定义：$H_{i,j}=T_H[I_i,j]$

#### 2.1.2 特征检索表端与查询端

在 $P_I$、公开特征表 commitment 和局部特征输入固定后生成字段压缩挑战 $\eta_{feat}$。对每个全局表端索引

$$u=v\cdot d_{in}+j,$$

定义

$$Table^{feat}[u]=v+\eta_{feat}j+\eta_{feat}^2T_H[v,j].$$

对每个查询索引

$$q=i\cdot d_{in}+j,$$

定义

$$Query^{feat}[q]=I_i+\eta_{feat}j+\eta_{feat}^2H_{i,j}.$$

#### 2.1.3 重数列

对每个全局条目索引 $u=v\cdot d_{in}+j$，定义

$$m_{feat}[u]=\#\{i\in\{0,1,\ldots,N-1\}\mid I_i=v\}.$$

#### 2.1.4 表端与查询端有效区选择器

定义

$$Q_{tbl}^{feat}[t]=\begin{cases}1,&0\le t<N_{total}d_{in},\\0,&N_{total}d_{in}\le t<n_{FH},\end{cases}$$

$$Q_{qry}^{feat}[t]=\begin{cases}1,&0\le t<Nd_{in},\\0,&Nd_{in}\le t<n_{FH}.\end{cases}$$

提交 $[P_{Table^{feat}}],[P_{Query^{feat}}],[P_{m_{feat}}]$ 后，吸入这三个 commitments、两个公开选择器描述和对象身份，生成 $\beta_{feat}$。$\beta_{feat}$ 不得在表端、查询端和重数 commitments 固定之前生成，也不得吸入依赖自身的逆元或累加器 commitment。

#### 2.1.5 特征检索累加器

在 $\beta_{feat}$ 固定后生成

$$J_{tbl}^{feat}[t]=\begin{cases}(Table^{feat}[t]+\beta_{feat})^{-1},&Q_{tbl}^{feat}[t]=1,\\0,&Q_{tbl}^{feat}[t]=0,\end{cases}$$

$$J_{qry}^{feat}[t]=\begin{cases}(Query^{feat}[t]+\beta_{feat})^{-1},&Q_{qry}^{feat}[t]=1,\\0,&Q_{qry}^{feat}[t]=0.\end{cases}$$

定义 $R_{feat}[0]=0$，并对 $0\le t<n_{FH}-1$ 固定递推

$$R_{feat}[t+1]=R_{feat}[t]+Q_{tbl}^{feat}[t]m_{feat}[t]J_{tbl}^{feat}[t]-Q_{qry}^{feat}[t]J_{qry}^{feat}[t].$$

填充区保持常值。证明者随后提交 $[P_{J_{tbl}^{feat}}],[P_{J_{qry}^{feat}}],[P_{R_{feat}}]$。有效分母非零由第 3.3 节的逆元约束证明，不使用依赖私有表值或查询值的公开拒绝计数。

#### 2.1.6 多项式编码

定义：

1. 节点绝对编号多项式：
   $$P_I(X)=\sum_{i=0}^{N-1}I_iL_i^{(N)}(X).$$

2. 原始特征行优先系数多项式：
   $$P_H(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{d_{in}-1}H_{i,j}X^{i\cdot d_{in}+j}.$$

3. 特征检索查询值多项式：令 $H^{feat}[q]=H_{i,j}$，其中 $q=i\cdot d_{in}+j$，并定义
   $$P_H^{feat}(X)=\sum_{q=0}^{n_{FH}-1}H^{feat}[q]L_q^{(FH)}(X).$$
   $P_H^{feat}$ 只用于特征检索查询端；$P_H$ 只用于矩阵乘法与 coefficient-basis 关系。二者都由公开输入唯一重建，并通过行优先索引定义绑定。

4. 表端、查询端与重数多项式：
   $$P_{Table^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Table^{feat}[t]L_t^{(FH)}(X),$$
   $$P_{Query^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Query^{feat}[t]L_t^{(FH)}(X),$$
   $$P_{m_{feat}}(X)=\sum_{t=0}^{n_{FH}-1}m_{feat}[t]L_t^{(FH)}(X).$$

5. 有效区选择器多项式：
   $$P_{Q_{tbl}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Q_{tbl}^{feat}[t]L_t^{(FH)}(X),$$
   $$P_{Q_{qry}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Q_{qry}^{feat}[t]L_t^{(FH)}(X).$$

6. 分母逆元与累加器多项式：
   $$P_{J_{tbl}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}J_{tbl}^{feat}[t]L_t^{(FH)}(X),$$
   $$P_{J_{qry}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}J_{qry}^{feat}[t]L_t^{(FH)}(X),$$
   $$P_{R_{feat}}(X)=\sum_{t=0}^{n_{FH}-1}R_{feat}[t]L_t^{(FH)}(X).$$

特征子块固定包含 $[P_H],[P_H^{feat}],[P_{Table^{feat}}],[P_{Query^{feat}}],[P_{m_{feat}}],[P_{J_{tbl}^{feat}}],[P_{J_{qry}^{feat}}],[P_{R_{feat}}]$。其中 $P_H$ 与 $P_H^{feat}$ 为 `Public`；其余对象按第 0.9 节登记。

#### 2.1.7 承诺

证明者按固定顺序提交

$$[P_H],\ [P_H^{feat}],\ [P_{Table^{feat}}],\ [P_{Query^{feat}}],\ [P_{m_{feat}}],\ [P_{J_{tbl}^{feat}}],\ [P_{J_{qry}^{feat}}],\ [P_{R_{feat}}].$$

$P_I$、$P_{Q_{tbl}^{feat}}$ 和 $P_{Q_{qry}^{feat}}$ 由验证者根据公共输入本地重建，不作为动态承诺重复编码。主实验的图结构与节点特征公开，因此上述特征对象的掩蔽多项式固定为零；该公开路径不得被解析为隐藏输入证明。

### 2.2 单个隐藏层模板中第 $r$ 个注意力头的完整见证生成

以下步骤对应某个固定隐藏层 $\ell\in\{1,2,\ldots,L-1\}$ 的单层模板，在该模板内，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$ 的隐藏层注意力头分别定义：

数学定义中的头顺序固定为 $r=0,1,\ldots,K_{hid}-1$；实现是否并行不属于协议语义，但任何实现都必须产出与这一固定头顺序一致的承诺对象、外点评值与子证明顺序。对多层 GAT，证明者与验证者都必须先对隐藏层 $\ell=1$ 实例化第 2.2 节与第 2.3 节，再把 $H_{cat}^{(1)}$ 作为下一隐藏层输入，依次实例化到 $\ell=L-1$；最后再进入第 2.4 节的最终输出层。

第 $\ell=1$ 层中，本节的 $H$ 与 $P_H$ 只能引用第 2.1 节由公开特征建立的对象。对任意 $\ell\ge2$，本节所有 $H_{i,m}$、$P_H$、$A_m^{proj(r)}$ 和对应 canonical / coefficient view 必须直接引用上一层 $H_{cat}^{(\ell-1)}$，不得提交新的“层输入” commitment。输出层投影同样只能引用 $H_{cat}^{(L-1)}$。`RelationMeta` 必须记录该来源引用并由 parser 检查层号连续性。

本节记号 $\mu_{proj}^{(r)},\mu_{src}^{(r)},\mu_{dst}^{(r)},\mu_{\star}^{(r)},\mu_{agg,pre}^{(r)},\mu_{agg}^{(r)}$ 只表示证明者构造关系时的局部终值。proof 不序列化这些裸域元素；相应终点等式由第 0.7.8 节的 KZG 同点平移归一化关系和工作域商约束证明。

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
> 右边：由投影矩阵$H'^{(r)}$和参数向量$a^{src(r)}$重新折叠计算出来的值

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

> 节点 $i$ 在第 $r$ 个头上的 $d_h$ 维向量使用 $\xi^{(r)}$ 的幂进行线性压缩

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

将 $E_{src}^{edge(r)}$ 与 $H_{src}^{\star,edge(r)}$ 填充到 $n_{edge}$ 并插值为 $P_{E_{src}^{edge(r)}}$、$P_{H_{src}^{\star,edge(r)}}$。这两个广播 commitments 必须在 $\eta_{src}^{(r)}$ 生成前提交；其值由本节 source indexed-view 关系绑定到节点域 canonical 对象。

> 这条边的源注意力广播值，等于它源节点的源注意力
>
> 这条边的源压缩特征广播值，等于它源节点的压缩特征
>
> 也就是把节点值按 `src(k)` 路由到边上

先生成字段压缩挑战 $\eta_{src}^{(r)}$，提交由该挑战确定的表端、查询端和重数 commitments 后，再生成分母挑战 $\beta_{src}^{(r)}$

对每个节点 $i$，定义表端：$Table^{src(r)}[i] = Q_N[i]\big(i+\eta_{src}^{(r)}E_{src,i}^{(r)}+(\eta_{src}^{(r)})^2H_i^{\star(r)}\big)$

定义节点重数：$m_{src}^{(r)}[i]=\#\{k\mid src(k)=i\}$

对每个边索引 $k$，定义查询端：$Query^{src(r)}[k] = Q_{edge}^{valid}[k]\big(src(k)+\eta_{src}^{(r)}E_{src,k}^{edge(r)}+(\eta_{src}^{(r)})^2H_{src,k}^{\star,edge(r)}\big)$

节点端与边端的有理和相等：

$$\sum_{i=0}^{N-1}\frac{m_{src}^{(r)}[i]}{Table^{src(r)}[i]+\beta_{src}^{(r)}}
=
\sum_{k=0}^{E-1}\frac{1}{Query^{src(r)}[k]+\beta_{src}^{(r)}}.$$

该共同终值属于 `PrivateWitness`，不定义为 proof 中的明文标量。节点端与边端累加器最后有效位置的相等关系由第 0.7.8 节规定的 KZG 同点平移归一化关系。

源 / 目标 route 的表端和查询端统一采用选择器门控编码：节点端由 $Q_N$ 门控，边端由 $Q_{edge}^{valid}$ 门控。因而所有无效位置的 $Table,Query$ 都严格为零，并分别满足 `ZeroPad`；不得依赖填充区的 $Idx_N,src,dst$ 恰好取零来获得等价行为。

> 左边：表端每条记录按“应该出现的次数”计入
>
> 右边：查询端每条边实际出现一次计入

定义节点端累加器：$R_{src}^{node(r)}[0]=0$

$R_{src}^{node(r)}[i+1] = R_{src}^{node(r)}[i] + Q_N[i]\cdot\frac{m_{src}^{(r)}[i]}{Table^{src(r)}[i]+\beta_{src}^{(r)}} \quad 0\le i\le n_N-2$

定义边端累加器：$R_{src}^{edge(r)}[0]=0$

$R_{src}^{edge(r)}[k+1] = R_{src}^{edge(r)}[k] + Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{src(r)}[k]+\beta_{src}^{(r)}} \quad 0\le k\le n_{edge}-2$

插值得到：$P_{Table^{src(r)}},\ P_{Query^{src(r)}},\ P_{m_{src}^{(r)}},\ P_{J_{src}^{node(r)}},\ P_{J_{src}^{edge(r)}},\ P_{R_{src}^{node(r)}},\ P_{R_{src}^{edge(r)}}$

提交承诺：$[P_{E_{src}^{edge(r)}}],\ [P_{H_{src}^{\star,edge(r)}}],\ [P_{Table^{src(r)}}],\ [P_{Query^{src(r)}}],\ [P_{m_{src}^{(r)}}],\ [P_{J_{src}^{node(r)}}],\ [P_{J_{src}^{edge(r)}}],\ [P_{R_{src}^{node(r)}}],\ [P_{R_{src}^{edge(r)}}]$

#### 2.2.6 LeakyReLU、最大值、唯一性与范围检查

对每个边索引 $k$，先定义 $E_{dst,k}^{edge(r)}=E_{dst,dst(k)}^{(r)}$，填充并插值为 $P_{E_{dst}^{edge(r)}}$；该 broadcast commitment 必须在 $P_{S^{(r)}}$ 与 $\eta_L^{(r)}$ 之前固定，并在目标路由 indexed-view 中绑定。随后定义：$S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)} \quad Z_k^{(r)}=LReLU(S_k^{(r)})$

先生成字段压缩挑战 $\eta_L^{(r)}$，提交由该挑战确定的表端、查询端和重数 commitments 后，再生成分母挑战 $\beta_L^{(r)}$

定义 LeakyReLU 表端：$Table^{L(r)}[t]=T_{LReLU}[t,0]+\eta_L^{(r)}T_{LReLU}[t,1]$

> LeakyReLU 静态表里每一行是：（$T_{LReLU}[t,0]$，$T_{LReLU}[t,1]$），表示（输入值，对应输出值）

查询端：$Query^{L(r)}[k]=S_k^{(r)}+\eta_L^{(r)}Z_k^{(r)}$

定义表端有效区选择器：$Q_{tbl}^{L(r)}[t]= \begin{cases} 1 &0\le t\le |T_{LReLU}|-1 \\ 0 &|T_{LReLU}|\le t\le n_{edge}-1 \end{cases}$

查询端有效区选择器：$Q_{qry}^{L(r)}[k]=Q_{edge}^{valid}[k]$

对每个表端索引 $t$，定义重数：$m_L^{(r)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid (S_k^{(r)},Z_k^{(r)})=(T_{LReLU}[t,0],T_{LReLU}[t,1])\}$

定义累加器：$R_L^{(r)}[0]=0$

$R_L^{(r)}[k+1] = R_L^{(r)}[k] + Q_{tbl}^{L(r)}[k]\cdot\frac{m_L^{(r)}[k]}{Table^{L(r)}[k]+\beta_L^{(r)}} - Q_{qry}^{L(r)}[k]\cdot\frac{1}{Query^{L(r)}[k]+\beta_L^{(r)}} \quad 0\le k\le n_{edge}-2$，对填充区保持常值。

插值得到：$P_{Table^{L(r)}},\ P_{Query^{L(r)}},\ P_{m_L^{(r)}},\ P_{Q_{tbl}^{L(r)}},\ P_{Q_{qry}^{L(r)}},\ P_{J_{tbl}^{L(r)}},\ P_{J_{qry}^{L(r)}},\ P_{R_L^{(r)}}$

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

提交范围查询与重数 commitments 后生成范围检查挑战 $\beta_R^{(r)}$；随后生成对应 $J_{tbl},J_{qry}$ 与累加器 commitments

定义范围表端：$Table^{R(r)}[t]=t$

范围查询端：$Query^{R(r)}[k]=\Delta_k^{+(r)}$

定义表端有效区选择器：$Q_{tbl}^{R(r)}[t]= \begin{cases} 1 &0\le t\le 2^B-1 \\ 0 &2^B\le t\le n_{edge}-1 \end{cases}$

查询端有效区选择器：$Q_{qry}^{R(r)}[k]=Q_{edge}^{valid}[k]$

对每个表端索引 $t$，定义重数：$m_R^{(r)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid \Delta_k^{+(r)}=t\}$

定义累加器：$R_R^{(r)}[0]=0$

$R_R^{(r)}[k+1] = R_R^{(r)}[k] + Q_{tbl}^{R(r)}[k]\cdot\frac{m_R^{(r)}[k]}{Table^{R(r)}[k]+\beta_R^{(r)}} - Q_{qry}^{R(r)}[k]\cdot\frac{1}{Query^{R(r)}[k]+\beta_R^{(r)}} \quad 0\le k\le n_{edge}-2$，对填充区保持常值。

插值得到：$P_{Table^{R(r)}},\ P_{Query^{R(r)}},\ P_{m_R^{(r)}},\ P_{Q_{tbl}^{R(r)}},\ P_{Q_{qry}^{R(r)}},\ P_{J_{tbl}^{R(r)}},\ P_{J_{qry}^{R(r)}},\ P_{R_R^{(r)}}$

上述 $Q_{tbl}^{L(r)},Q_{qry}^{L(r)},Q_{tbl}^{R(r)},Q_{qry}^{R(r)}$ 均由静态表长度、范围位宽和公开有效区唯一重建，只作为公共多项式引用，不进入动态承诺或 proof evaluation。

把$S^{(r)},\ Z^{(r)},\ M^{(r)},\ M^{edge(r)},\ \Delta^{+(r)},\ s_{max}^{(r)},\ C_{max}^{(r)}$，分别插值成：$P_{S^{(r)}},\ P_{Z^{(r)}},\ P_{M^{(r)}},\ P_{M^{edge(r)}},\ P_{\Delta^{+(r)}},\ P_{s_{max}^{(r)}},\ P_{C_{max}^{(r)}}$

提交承诺：

$[P_{E_{dst}^{edge(r)}}],\ [P_{S^{(r)}}],\ [P_{Z^{(r)}}],\ [P_{M^{(r)}}],\ [P_{M^{edge(r)}}],\ [P_{\Delta^{+(r)}}],\ [P_{s_{max}^{(r)}}],\ [P_{C_{max}^{(r)}}],\ [P_{Table^{L(r)}}],\ [P_{Query^{L(r)}}],\ [P_{m_L^{(r)}}],\ [P_{J_{tbl}^{L(r)}}],\ [P_{J_{qry}^{L(r)}}],\ [P_{R_L^{(r)}}],\ [P_{Table^{R(r)}}],\ [P_{Query^{R(r)}}],\ [P_{m_R^{(r)}}],\ [P_{J_{tbl}^{R(r)}}],\ [P_{J_{qry}^{R(r)}}],\ [P_{R_R^{(r)}}]$

#### 2.2.7 指数、分母、逆元与归一化权重

对每个边索引 $k$，定义：$U_k^{(r)}=ExpMap(\Delta_k^{+(r)})$。

先生成字段压缩挑战 $\eta_{exp}^{(r)}$，提交由该挑战确定的表端、查询端和重数 commitments 后，再生成分母挑战 $\beta_{exp}^{(r)}$。

定义指数表端：

$$Table^{exp(r)}[t]=T_{exp}[t,0]+\eta_{exp}^{(r)}T_{exp}[t,1].$$

查询端：

$$Query^{exp(r)}[k]=\Delta_k^{+(r)}+\eta_{exp}^{(r)}U_k^{(r)}.$$

定义表端有效区选择器：

$$Q_{tbl}^{exp(r)}[t]= \begin{cases} 1 &0\le t\le |T_{exp}|-1 \\ 0 &|T_{exp}|\le t\le n_{edge}-1. \end{cases}$$

查询端有效区选择器：

$$Q_{qry}^{exp(r)}[k]=Q_{edge}^{valid}[k].$$

对每个表端索引 $t$，定义重数：

$$m_{exp}^{(r)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid (\Delta_k^{+(r)},U_k^{(r)})=(T_{exp}[t,0],T_{exp}[t,1])\}.$$

定义累加器：

$$R_{exp}^{(r)}[0]=0,$$

$$R_{exp}^{(r)}[k+1] = R_{exp}^{(r)}[k] + Q_{tbl}^{exp(r)}[k]\cdot\frac{m_{exp}^{(r)}[k]}{Table^{exp(r)}[k]+\beta_{exp}^{(r)}} - Q_{qry}^{exp(r)}[k]\cdot\frac{1}{Query^{exp(r)}[k]+\beta_{exp}^{(r)}}$$

对 $0\le k\le n_{edge}-2$ 成立，填充区保持常值。

插值得到：

$$P_{Table^{exp(r)}},\ P_{Query^{exp(r)}},\ P_{m_{exp}^{(r)}},\ P_{Q_{tbl}^{exp(r)}},\ P_{Q_{qry}^{exp(r)}},\ P_{J_{tbl}^{exp(r)}},\ P_{J_{qry}^{exp(r)}},\ P_{R_{exp}^{(r)}}.$$

对每个节点 $i$，定义：

$$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)},\qquad inv_i^{(r)}=(Sum_i^{(r)})^{-1}.$$

对每个边索引 $k$，定义广播：

$$Sum_k^{edge(r)}=Sum_{dst(k)}^{(r)},\qquad inv_k^{edge(r)}=inv_{dst(k)}^{(r)}.$$

定义有限域归一化权重：

$$\alpha_k^{(r)}=U_k^{(r)}\cdot inv_k^{edge(r)}.$$

$inv^{(r)}$ 与 $\alpha^{(r)}$ 只用于证明 $Sum\ne0$ 和有限域 Softmax 关系；ELU 的定点输入由第 2.2.8 节的整数聚合分子和 $SignedRoundDiv$ 关系确定。

把 $U^{(r)},Sum^{(r)},Sum^{edge(r)},inv^{(r)},inv^{edge(r)},\alpha^{(r)}$ 分别插值成：

$$P_{U^{(r)}},\ P_{Sum^{(r)}},\ P_{Sum^{edge(r)}},\ P_{inv^{(r)}},\ P_{inv^{edge(r)}},\ P_{\alpha^{(r)}}.$$

提交承诺：

$$[P_{U^{(r)}}],\ [P_{Sum^{(r)}}],\ [P_{Sum^{edge(r)}}],\ [P_{inv^{(r)}}],\ [P_{inv^{edge(r)}}],\ [P_{\alpha^{(r)}}],\ [P_{Table^{exp(r)}}],\ [P_{Query^{exp(r)}}],\ [P_{m_{exp}^{(r)}}],\ [P_{J_{tbl}^{exp(r)}}],\ [P_{J_{qry}^{exp(r)}}],\ [P_{R_{exp}^{(r)}}].$$

#### 2.2.8 聚合前隐藏矩阵、聚合前压缩特征与 PSQ

对每个节点 $i$ 与每个隐藏维索引 $j$，定义有限域聚合值：

$$H_{agg,pre,i,j}^{(r)} = \sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}.$$

定义压缩特征：

$$H_{agg,pre,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,pre,i,j}^{(r)}(\xi^{(r)})^j.$$

定义边级压缩加权特征：

$$\widehat v_{pre,k}^{\star(r)}=\alpha_k^{(r)}H_{src,k}^{\star,edge(r)}.$$

定义广播：

$$H_{agg,pre,k}^{\star,edge(r)}=H_{agg,pre,dst(k)}^{\star(r)}.$$

将 $H_{agg,pre}^{\star,edge(r)}$ 填充并插值为 $P_{H_{agg,pre}^{\star,edge(r)}}$，其 commitment 必须在 $\lambda_{psq}^{(r)}$ 生成前固定，并在目标路由 indexed-view 中与 $P_{H_{agg,pre}^{\star(r)}}$ 绑定。

在决定 $U^{(r)},Sum^{(r)},Sum^{edge(r)},\widehat v_{pre}^{\star(r)},H_{agg,pre}^{\star(r)}$ 这些基础值的相关承诺对象全部固定之后，生成挑战 $\lambda_{psq}^{(r)}$。

定义：

$$w_{psq}^{(r)}[k]=U_k^{(r)}+\lambda_{psq}^{(r)}\widehat v_{pre,k}^{\star(r)},$$

$$T_{psq}^{(r)}[i]=Sum_i^{(r)}+\lambda_{psq}^{(r)}H_{agg,pre,i}^{\star(r)},$$

$$T_{psq}^{edge(r)}[k]=Sum_k^{edge(r)}+\lambda_{psq}^{(r)}H_{agg,pre,k}^{\star,edge(r)}.$$

定义 PSQ 状态机：

$$PSQ^{(r)}[0]=Q_{edge}^{valid}[0]\cdot w_{psq}^{(r)}[0].$$

对 $0\le k\le n_{edge}-2$，定义：

$$PSQ^{(r)}[k+1] = (1-Q_{edge}^{valid}[k+1])PSQ^{(r)}[k] + Q_{edge}^{valid}[k+1] \Big( Q_{grp\_new}^{edge}[k+1]w_{psq}^{(r)}[k+1] + (1-Q_{grp\_new}^{edge}[k+1])\big(PSQ^{(r)}[k]+w_{psq}^{(r)}[k+1]\big) \Big).$$

对每个边索引 $k$，施加组末约束：

$$Q_{grp\_end}^{edge}[k]\cdot\big(PSQ^{(r)}[k]-T_{psq}^{edge(r)}[k]\big)=0.$$

组末约束同时强制：

$$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)},$$

$$H_{agg,pre,i}^{\star(r)}=\sum_{\{k\mid dst(k)=i\}}\widehat v_{pre,k}^{\star(r)}.$$

插值得到：

$$P_{H_{agg,pre}^{(r)}},\ P_{H_{agg,pre}^{\star(r)}},\ P_{\widehat v_{pre}^{\star(r)}},\ P_{w_{psq}^{(r)}},\ P_{T_{psq}^{(r)}},\ P_{T_{psq}^{edge(r)}},\ P_{PSQ^{(r)}}.$$

生成聚合前压缩绑定挑战：

$$y_{agg,pre}^{(r)}=H_{FS}(\text{transcript},[P_{H_{agg,pre}^{(r)}}],[P_{H_{agg,pre}^{\star(r)}}]).$$

要求：

$$P_{H_{agg,pre}^{\star(r)}}(y_{agg,pre}^{(r)}) = \sum_{j=0}^{d_h-1} \Big( \sum_{i=0}^{N-1}H_{agg,pre,i,j}^{(r)}L_i^{(N)}(y_{agg,pre}^{(r)}) \Big) (\xi^{(r)})^j.$$

有限域聚合关系与定点 ELU 输入关系同时成立。对行优先索引 $q=i\cdot d_h+j$ 定义 $\mathbb H_{edge}$ 上的辅助列：

$$Sum_{flat}^{(r)}[q]=Sum_i^{(r)},$$

$$H_{agg,pre,flat}^{(r)}[q]=H_{agg,pre,i,j}^{(r)},$$

$$H_{agg,num,flat}^{(r)}[q]=Sum_{flat}^{(r)}[q]\,H_{agg,pre,flat}^{(r)}[q],$$

$$H_{ELU,in,flat}^{(r)}[q]=SignedRoundDiv\!\left(
S_{agg,pre}H_{agg,num,flat}^{(r)}[q],
S_{H'}^{(\ell,r)}Sum_{flat}^{(r)}[q]
\right).$$

诚实见证还满足：

$$H_{agg,num,flat}^{(r)}[i\cdot d_h+j]=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}H_{src(k),j}'^{(r)}.$$

$Sum_{flat}^{(r)}$ 通过 indexed-view 与 $P_{Sum^{(r)}}$ 绑定；$H_{agg,pre,flat}^{(r)}$ 通过 basis link 与 $P_{H_{agg,pre}^{(r)}}$ 绑定。$H_{agg,num,flat}^{(r)}=Sum_{flat}^{(r)}H_{agg,pre,flat}^{(r)}$ 的乘法约束、$SignedRoundDiv$ 约束和全部范围 LogUp 进入边域商多项式。

对该除法实例生成第 0.4 节规定的 $SignedRoundDiv$ 辅助列：

$$\sigma_{agg}^{(r)},\ z_{agg}^{(r)},\ a_{agg}^{(r)},\ (a_{agg}^{(r)})^{-1},\ d_{agg}^{-1(r)},\ q_{agg}^{(r)},\ r_{agg}^{(r)},\ \delta_{agg}^{(r)}.$$

将 $Sum_{flat}^{(r)},H_{agg,pre,flat}^{(r)},H_{agg,num,flat}^{(r)},H_{ELU,in,flat}^{(r)}$ 和这些辅助列插值为 $\mathbb H_{edge}$ 上的多项式。

提交承诺：

$$[P_{H_{agg,pre}^{(r)}}],\ [P_{H_{agg,pre}^{\star(r)}}],\ [P_{H_{agg,pre}^{\star,edge(r)}}],\ [P_{\widehat v_{pre}^{\star(r)}}],\ [P_{w_{psq}^{(r)}}],\ [P_{T_{psq}^{(r)}}],\ [P_{T_{psq}^{edge(r)}}],\ [P_{PSQ^{(r)}}],$$

并提交定点输入辅助承诺：

$$[P_{Sum_{flat}^{(r)}}],\ [P_{H_{agg,pre,flat}^{(r)}}],\ [P_{H_{agg,num,flat}^{(r)}}],\ [P_{H_{ELU,in,flat}^{(r)}}],\ [P_{\sigma_{agg}^{(r)}}],\ [P_{z_{agg}^{(r)}}],\ [P_{a_{agg}^{(r)}}],\ [P_{(a_{agg}^{(r)})^{-1}}],\ [P_{d_{agg}^{-1(r)}}],\ [P_{q_{agg}^{(r)}}],\ [P_{r_{agg}^{(r)}}],\ [P_{\delta_{agg}^{(r)}}].$$

随后按 $V\in(a_{agg}^{(r)},d_{agg}^{(r)}-1,q_{agg}^{(r)},r_{agg}^{(r)},\delta_{agg}^{(r)})$ 的固定顺序，并在每个 $V$ 内按 $h=0,1,\ldots,\ell_{agg,V}-1$ 的顺序，提交第 3.5.2 节定义的定点除法辅助承诺族中的全部 limb、范围查询、重数、分母逆元和范围累加器 commitments；查询选择器由验证者本地重建。

#### 2.2.9 ELU

本节 ELU 查表复用边域 $\mathbb H_{edge}$ 作为工作域；下文“ELU 工作域”均指 $\mathbb H_{edge}$。

对每个节点 $i$ 与每个隐藏维索引 $j$，定义：

$$H_{agg,i,j}^{(r)}=ELU(H_{ELU,in,i,j}^{(r)}).$$

其中 $H_{ELU,in}^{(r)}$ 是第 2.2.8 节已证明的定点整数输入，不允许从独立浮点聚合、旁路舍入或未受约束的缓存生成。

将 $H_{ELU,in}^{(r)}$ 与 $H_{agg}^{(r)}$ 展平到 ELU 工作域。在来源 commitments 固定后生成 $\eta_{ELU}^{(r)}$；提交由该挑战确定的表端、查询端和重数 commitments 后生成 $\beta_{ELU}^{(r)}$，再生成分母逆元和累加器 commitments。

定义 ELU 表端：

$$Table^{ELU(r)}[t]=T_{ELU}[t,0]+\eta_{ELU}^{(r)}T_{ELU}[t,1].$$

定义 ELU 输出行优先列及其边域插值多项式：

$$H_{agg,flat}^{(r)}[i\cdot d_h+j]=H_{agg,i,j}^{(r)},$$

$$P_{H_{agg,flat}^{(r)}}(X)=\sum_{q=0}^{n_{edge}-1}H_{agg,flat}^{(r)}[q]L_q^{(edge)}(X).$$

同时显式定义行优先 coefficient-basis relation view：

$$P_{H_{agg}^{(r)}}(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{d_h-1}H_{agg,i,j}^{(r)}X^{i d_h+j}.$$

$P_{H_{agg}^{(r)}}$ 不是节点域 canonical 列族，也不是 $\mathbb H_{edge}$ 上的 flat evaluation trace；它只作为第 0.7.8 节登记的 coefficient-basis relation view。$P_{H_{agg,flat}^{(r)}}$ 通过该节规定的 basis link 与 $P_{H_{agg}^{(r)}}$ 绑定。

定义 ELU 查询端：

$$Query^{ELU(r)}[q]=H_{ELU,in,flat}^{(r)}[q]+\eta_{ELU}^{(r)}H_{agg,flat}^{(r)}[q].$$

定义表端有效区选择器：

$$Q_{tbl}^{ELU(r)}[t]= \begin{cases} 1 &0\le t\le |T_{ELU}|-1 \\ 0 &|T_{ELU}|\le t\le n_{edge}-1. \end{cases}$$

查询端有效区选择器：

$$Q_{qry}^{ELU(r)}[q]=Q_{Nd_h}^{valid,(\ell)}[q].$$

对每个表端索引 $t$，定义重数：

$$m_{ELU}^{(r)}[t] = \#\{q\in\{0,1,\ldots,Nd_h-1\}\mid (H_{ELU,in,flat}^{(r)}[q],H_{agg,flat}^{(r)}[q])=(T_{ELU}[t,0],T_{ELU}[t,1])\}.$$

定义累加器：

$$R_{ELU}^{(r)}[0]=0,$$

$$R_{ELU}^{(r)}[q+1] = R_{ELU}^{(r)}[q] + Q_{tbl}^{ELU(r)}[q]\cdot\frac{m_{ELU}^{(r)}[q]}{Table^{ELU(r)}[q]+\beta_{ELU}^{(r)}} - Q_{qry}^{ELU(r)}[q]\cdot\frac{1}{Query^{ELU(r)}[q]+\beta_{ELU}^{(r)}},$$

对 $0\le q\le n_{edge}-2$ 成立，填充区保持常值。

插值得到：

$$P_{Table^{ELU(r)}},\ P_{Query^{ELU(r)}},\ P_{m_{ELU}^{(r)}},\ P_{Q_{tbl}^{ELU(r)}},\ P_{Q_{qry}^{ELU(r)}},\ P_{J_{tbl}^{ELU(r)}},\ P_{J_{qry}^{ELU(r)}},\ P_{R_{ELU}^{(r)}}.$$

定义聚合后压缩特征：

$$H_{agg,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,i,j}^{(r)}(\xi^{(r)})^j.$$

定义边域广播：

$$H_{agg,k}^{\star,edge(r)}=H_{agg,dst(k)}^{\star(r)}.$$

将该广播填充并插值为 $P_{H_{agg}^{\star,edge(r)}}$，其 commitment 在目标路由字段压缩挑战之前固定。

生成聚合后压缩绑定挑战：

$$y_{agg}^{(r)}=H_{FS}(\text{transcript},[P_{H_{agg}^{(r)}}],[P_{H_{agg}^{\star(r)}}]).$$

要求：

$$P_{H_{agg}^{\star(r)}}(y_{agg}^{(r)}) = \sum_{j=0}^{d_h-1} \Big( \sum_{i=0}^{N-1}H_{agg,i,j}^{(r)}L_i^{(N)}(y_{agg}^{(r)}) \Big) (\xi^{(r)})^j.$$

提交：

$$[P_{H_{agg}^{(r)}}],\ [P_{H_{agg,flat}^{(r)}}],\ [P_{H_{agg}^{\star(r)}}],\ [P_{H_{agg}^{\star,edge(r)}}],\ [P_{Table^{ELU(r)}}],\ [P_{Query^{ELU(r)}}],\ [P_{m_{ELU}^{(r)}}],\ [P_{J_{tbl}^{ELU(r)}}],\ [P_{J_{qry}^{ELU(r)}}],\ [P_{R_{ELU}^{(r)}}].$$

#### 2.2.10 目标路由

目标路由需要同时绑定：

- 目标注意力；
- 组最大值；
- 分母；
- 逆元；
- 聚合前压缩特征；
- ELU 后压缩聚合特征。

因此在 $P_{E_{dst}^{(r)}},\ P_{M^{(r)}},\ P_{Sum^{(r)}},\ P_{inv^{(r)}},\ P_{H_{agg,pre}^{\star(r)}},\ P_{H_{agg}^{\star(r)}}$ 及其边域广播对象全部固定之后，统一先生成字段压缩挑战 $\eta_{dst}^{(r)}$，提交由该挑战确定的表端、查询端和重数 commitments 后，再生成分母挑战 $\beta_{dst}^{(r)}$

对每个节点 $i$，定义表端：$Table^{dst(r)}[i] = Q_N[i]\big(i +\eta_{dst}^{(r)}E_{dst,i}^{(r)} +(\eta_{dst}^{(r)})^2M_i^{(r)} +(\eta_{dst}^{(r)})^3Sum_i^{(r)} +(\eta_{dst}^{(r)})^4inv_i^{(r)} +(\eta_{dst}^{(r)})^5H_{agg,pre,i}^{\star(r)} +(\eta_{dst}^{(r)})^6H_{agg,i}^{\star(r)}\big)$

对每个边索引 $k$，定义查询端：

$Query^{dst(r)}[k] = Q_{edge}^{valid}[k]\big(dst(k) +\eta_{dst}^{(r)}E_{dst,k}^{edge(r)} +(\eta_{dst}^{(r)})^2M_k^{edge(r)} +(\eta_{dst}^{(r)})^3Sum_k^{edge(r)} +(\eta_{dst}^{(r)})^4inv_k^{edge(r)} +(\eta_{dst}^{(r)})^5H_{agg,pre,k}^{\star,edge(r)} +(\eta_{dst}^{(r)})^6H_{agg,k}^{\star,edge(r)}\big)$

定义节点重数：$m_{dst}^{(r)}[i]=\#\{k\mid dst(k)=i\}$

节点端与边端的有理和相等：

$$\sum_{i=0}^{N-1}\frac{m_{dst}^{(r)}[i]}{Table^{dst(r)}[i]+\beta_{dst}^{(r)}}
=
\sum_{k=0}^{E-1}\frac{1}{Query^{dst(r)}[k]+\beta_{dst}^{(r)}}.$$

该共同终值属于 `PrivateWitness`，不定义为 proof 中的明文标量。节点端与边端累加器最后有效位置的相等关系由第 0.7.8 节规定的 KZG 同点平移归一化关系。



定义节点端累加器：$R_{dst}^{node(r)}[0]=0$

$R_{dst}^{node(r)}[i+1] = R_{dst}^{node(r)}[i] + Q_N[i]\cdot\frac{m_{dst}^{(r)}[i]}{Table^{dst(r)}[i]+\beta_{dst}^{(r)}}$

定义边端累加器：$R_{dst}^{edge(r)}[0]=0$

$R_{dst}^{edge(r)}[k+1] = R_{dst}^{edge(r)}[k] + Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{dst(r)}[k]+\beta_{dst}^{(r)}}$

插值得到：$P_{Table^{dst(r)}},\ P_{Query^{dst(r)}},\ P_{m_{dst}^{(r)}},\ P_{J_{dst}^{node(r)}},\ P_{J_{dst}^{edge(r)}},\ P_{R_{dst}^{node(r)}},\ P_{R_{dst}^{edge(r)}}$

提交承诺：$[P_{Table^{dst(r)}}],\ [P_{Query^{dst(r)}}],\ [P_{m_{dst}^{(r)}}],\ [P_{J_{dst}^{node(r)}}],\ [P_{J_{dst}^{edge(r)}}],\ [P_{R_{dst}^{node(r)}}],\ [P_{R_{dst}^{edge(r)}}]$

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

提交：$[P_{H_{cat}}],\ [P_{H_{cat}^{\star}}]$。$\pi_{bind}^{concat,(\ell)}$ 还必须按 $(r,j)$ 的字典序登记节点域等式

$$Q_N(X)\Big(P_{H_{cat}[:,r d_h+j]}^{can}(X)-P_{H_{agg}^{(r)}[:,j]}^{can}(X)\Big)=0,$$

从而逐列证明拼接结果恰由本层全部头的 ELU 输出组成，而不是仅证明 $H_{cat}^{\star}$ 与一个自由矩阵自洽。上述等式进入 $\mathcal C_{bind,N}$。

### 2.4 输出层的完整见证生成

输出层由 $K_{out}$ 个输出注意力头组成。对每个输出头 $s\in\{0,1,\ldots,K_{out}-1\}$，先按单头注意力完成 projection / score / softmax / aggregation / bias-add；全部输出头完成后，再做多头平均，得到最终输出 $Y$。

本节记号 $\mu_{proj}^{(out,s)},\mu_{src}^{(out,s)},\mu_{dst}^{(out,s)},\mu_{out,pre}^{(s)},\mu_{bias}^{(out,s)},\mu_{out}^{(s)}$ 只表示证明者构造关系时的局部终值。proof 不序列化这些裸域元素；相应终点等式由第 0.7.8 节的 KZG 同点平移归一化关系和工作域商约束证明。

输出层见证生成采用两阶段固定顺序：先对所有 $s=0,\ldots,K_{out}-1$ 构造并提交输出投影 commitment $[P_{Y'^{(s)}}]$；全部输出投影 commitments 按头序固定后生成唯一全局类别压缩挑战

$$\xi_{out}=H_{FS}(\texttt{output:class-compress}\parallel transcript\parallel [P_{Y'^{(0)}}]\parallel\cdots\parallel[P_{Y'^{(K_{out}-1)}}]).$$

随后立即对每个输出头构造

$$Y_i'^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}\xi_{out}^c,$$

形成 $P_{Y'^{\star(s)}}$ 并提交其 commitment；输出源路由的 $Table/Query$ 只有在该 commitment 固定后才能生成。此后才能构造输出路由、输出聚合及其他依赖 $\xi_{out}$ 的压缩对象。不得在不同输出头中重新派生不同的 $\xi_{out}$，也不得在看到 $\xi_{out}$ 后替换任一 $[P_{Y'^{(s)}}]$ 或 $[P_{Y'^{\star(s)}}]$。

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

将 $E_{src}^{edge(out,s)}$ 与 $Y'^{\star,edge(s)}$ 填充并插值为边域多项式；两个 commitments 必须在 $\eta_{src}^{(out,s)}$ 前固定，并由 source indexed-view 绑定。

先生成字段压缩挑战 $\eta_{src}^{(out,s)}$，提交由该挑战确定的表端、查询端和重数 commitments 后，再生成分母挑战 $\beta_{src}^{(out,s)}$

对每个节点 $i$，定义表端：$Table^{src(out,s)}[i]=Q_N[i]\big(i+\eta_{src}^{(out,s)}E_{src,i}^{(out,s)}+(\eta_{src}^{(out,s)})^2Y_i'^{\star(s)}\big)$

定义节点重数：$m_{src}^{(out,s)}[i]=\#\{k\mid src(k)=i\}$

对每个边索引 $k$，定义查询端：$Query^{src(out,s)}[k]=Q_{edge}^{valid}[k]\big(src(k)+\eta_{src}^{(out,s)}E_{src,k}^{edge(out,s)}+(\eta_{src}^{(out,s)})^2Y_{k}'^{\star,edge(s)}\big)$

节点端与边端满足：

$$\sum_{i=0}^{N-1}\frac{m_{src}^{(out,s)}[i]}{Table^{src(out,s)}[i]+\beta_{src}^{(out,s)}}
=
\sum_{k=0}^{E-1}\frac{1}{Query^{src(out,s)}[k]+\beta_{src}^{(out,s)}}.$$

未掩蔽共同终值不进入 proof；其相等关系由 KZG 同点平移归一化关系验证。点互异时只编码平移 commitment、单次 masked 检查值和普通 point-grouped witness 引用；点相同时只登记普通零值 claim。

定义节点端累加器：$R_{src}^{node(out,s)}[0]=0$

$R_{src}^{node(out,s)}[i+1]=R_{src}^{node(out,s)}[i]+Q_N[i]\cdot\frac{m_{src}^{(out,s)}[i]}{Table^{src(out,s)}[i]+\beta_{src}^{(out,s)}}$

定义边端累加器：$R_{src}^{edge(out,s)}[0]=0$

$R_{src}^{edge(out,s)}[k+1]=R_{src}^{edge(out,s)}[k]+Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{src(out,s)}[k]+\beta_{src}^{(out,s)}}$

插值得到：$P_{Table^{src(out,s)}},\ P_{Query^{src(out,s)}},\ P_{m_{src}^{(out,s)}},\ P_{J_{src}^{node(out,s)}},\ P_{J_{src}^{edge(out,s)}},\ P_{R_{src}^{node(out,s)}},\ P_{R_{src}^{edge(out,s)}}$

接着对每个边索引 $k$，先定义 $E_{dst,k}^{edge(out,s)}=E_{dst,dst(k)}^{(out,s)}$，填充并插值为 $P_{E_{dst}^{edge(out,s)}}$；该 commitment 在 $P_{S^{(out,s)}}$ 与 $\eta_L^{(out,s)}$ 之前固定，并在目标路由 indexed-view 中绑定。随后定义：

$S_k^{(out,s)}=E_{src,k}^{edge(out,s)}+E_{dst,k}^{edge(out,s)}, \quad Z_k^{(out,s)}=LReLU(S_k^{(out,s)})$

先生成字段压缩挑战 $\eta_L^{(out,s)}$，提交由该挑战确定的表端、查询端和重数 commitments 后，再生成分母挑战 $\beta_L^{(out,s)}$

定义 LeakyReLU 表端：$Table^{L(out,s)}[t]=T_{LReLU}[t,0]+\eta_L^{(out,s)}T_{LReLU}[t,1]$

定义查询端：$Query^{L(out,s)}[k]=S_k^{(out,s)}+\eta_L^{(out,s)}Z_k^{(out,s)}$

定义表端有效区选择器：$Q_{tbl}^{L(out,s)}[t]=\begin{cases}1,&0\le t\le |T_{LReLU}|-1\\0,&|T_{LReLU}|\le t\le n_{edge}-1\end{cases}$

定义查询端有效区选择器：$Q_{qry}^{L(out,s)}[k]=Q_{edge}^{valid}[k]$

定义重数：$m_L^{(out,s)}[t]=\#\{k\in\{0,1,\ldots,E-1\}\mid (S_k^{(out,s)},Z_k^{(out,s)})=(T_{LReLU}[t,0],T_{LReLU}[t,1])\}$

定义累加器：$R_L^{(out,s)}[0]=0$

$R_L^{(out,s)}[k+1]=R_L^{(out,s)}[k]+Q_{tbl}^{L(out,s)}[k]\cdot\frac{m_L^{(out,s)}[k]}{Table^{L(out,s)}[k]+\beta_L^{(out,s)}}-Q_{qry}^{L(out,s)}[k]\cdot\frac{1}{Query^{L(out,s)}[k]+\beta_L^{(out,s)}}$

插值得到：$P_{Table^{L(out,s)}},\ P_{Query^{L(out,s)}},\ P_{m_L^{(out,s)}},\ P_{Q_{tbl}^{L(out,s)}},\ P_{Q_{qry}^{L(out,s)}},\ P_{J_{tbl}^{L(out,s)}},\ P_{J_{qry}^{L(out,s)}},\ P_{R_L^{(out,s)}}$

定义组最大值：$M_i^{(out,s)}=\max\{Z_k^{(out,s)}\mid dst(k)=i\}$

定义最大值广播与差分：$M_k^{edge(out,s)}=M_{dst(k)}^{(out,s)}, \qquad \Delta_k^{+(out,s)}=M_k^{edge(out,s)}-Z_k^{(out,s)}$

引入二值指示变量：$s_{max}^{(out,s)}[k]\in\{0,1\}$，并要求：

$s_{max}^{(out,s)}[k](s_{max}^{(out,s)}[k]-1)=0, \quad s_{max}^{(out,s)}[k]\cdot \Delta_k^{+(out,s)}=0$

定义组内计数状态机：$C_{max}^{(out,s)}[0]=Q_{edge}^{valid}[0]\cdot s_{max}^{(out,s)}[0]$

$C_{max}^{(out,s)}[k+1]=(1-Q_{edge}^{valid}[k+1])C_{max}^{(out,s)}[k]+Q_{edge}^{valid}[k+1]\Big(Q_{grp\_new}^{edge}[k+1]s_{max}^{(out,s)}[k+1]+(1-Q_{grp\_new}^{edge}[k+1])(C_{max}^{(out,s)}[k]+s_{max}^{(out,s)}[k+1])\Big)$

并要求组末唯一性约束：$Q_{grp\_end}^{edge}[k]\cdot(C_{max}^{(out,s)}[k]-1)=0$

提交范围查询与重数 commitments 后生成范围检查挑战 $\beta_R^{(out,s)}$；随后生成对应 $J_{tbl},J_{qry}$ 与累加器 commitments

定义范围表端与查询端：$Table^{R(out,s)}[t]=t, \quad Query^{R(out,s)}[k]=\Delta_k^{+(out,s)}$

定义有效区选择器：$Q_{tbl}^{R(out,s)}[t]=\begin{cases}1,&0\le t\le 2^B-1\\0,&2^B\le t\le n_{edge}-1\end{cases}$，$Q_{qry}^{R(out,s)}[k]=Q_{edge}^{valid}[k]$

定义重数与累加器：$m_R^{(out,s)}[t]=\#\{k\in\{0,1,\ldots,E-1\}\mid \Delta_k^{+(out,s)}=t\}$，$R_R^{(out,s)}[0]=0$

$R_R^{(out,s)}[k+1]=R_R^{(out,s)}[k]+Q_{tbl}^{R(out,s)}[k]\cdot\frac{m_R^{(out,s)}[k]}{Table^{R(out,s)}[k]+\beta_R^{(out,s)}}-Q_{qry}^{R(out,s)}[k]\cdot\frac{1}{Query^{R(out,s)}[k]+\beta_R^{(out,s)}}$

插值得到：$P_{Table^{R(out,s)}},\ P_{Query^{R(out,s)}},\ P_{m_R^{(out,s)}},\ P_{Q_{tbl}^{R(out,s)}},\ P_{Q_{qry}^{R(out,s)}},\ P_{J_{tbl}^{R(out,s)}},\ P_{J_{qry}^{R(out,s)}},\ P_{R_R^{(out,s)}}$

先生成指数表字段压缩挑战 $\eta_{exp}^{(out,s)}$；提交由该挑战确定的表端、查询端和重数 commitments 后生成 $\beta_{exp}^{(out,s)}$，再生成分母逆元与累加器 commitments

定义：$U_k^{(out,s)}=ExpMap(\Delta_k^{+(out,s)})$

$Table^{exp(out,s)}[t]=T_{exp}[t,0]+\eta_{exp}^{(out,s)}T_{exp}[t,1]$

$Query^{exp(out,s)}[k]=\Delta_k^{+(out,s)}+\eta_{exp}^{(out,s)}U_k^{(out,s)}$

$Q_{tbl}^{exp(out,s)}[t]=\begin{cases}1,&0\le t\le |T_{exp}|-1\\0,&|T_{exp}|\le t\le n_{edge}-1\end{cases}$，$Q_{qry}^{exp(out,s)}[k]=Q_{edge}^{valid}[k]$

$m_{exp}^{(out,s)}[t]=\#\{k\in\{0,1,\ldots,E-1\}\mid (\Delta_k^{+(out,s)},U_k^{(out,s)})=(T_{exp}[t,0],T_{exp}[t,1])\}$，$R_{exp}^{(out,s)}[0]=0$

$R_{exp}^{(out,s)}[k+1]=R_{exp}^{(out,s)}[k]+Q_{tbl}^{exp(out,s)}[k]\cdot\frac{m_{exp}^{(out,s)}[k]}{Table^{exp(out,s)}[k]+\beta_{exp}^{(out,s)}}-Q_{qry}^{exp(out,s)}[k]\cdot\frac{1}{Query^{exp(out,s)}[k]+\beta_{exp}^{(out,s)}}$

插值得到：$P_{Table^{exp(out,s)}},\ P_{Query^{exp(out,s)}},\ P_{m_{exp}^{(out,s)}},\ P_{Q_{tbl}^{exp(out,s)}},\ P_{Q_{qry}^{exp(out,s)}},\ P_{J_{tbl}^{exp(out,s)}},\ P_{J_{qry}^{exp(out,s)}},\ P_{R_{exp}^{(out,s)}}$

上述 $Q_{tbl}^{L(out,s)},Q_{qry}^{L(out,s)},Q_{tbl}^{R(out,s)},Q_{qry}^{R(out,s)},Q_{tbl}^{exp(out,s)},Q_{qry}^{exp(out,s)}$ 均由静态表长度、范围位宽和公开有效区唯一重建，只作为公共多项式引用，不进入动态承诺或 proof evaluation。

最后定义 softmax 分母、逆元与归一化权重：

$Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)}, \qquad inv_i^{(out,s)}=(Sum_i^{(out,s)})^{-1}$

$Sum_k^{edge(out,s)}=Sum_{dst(k)}^{(out,s)}, \qquad inv_k^{edge(out,s)}=inv_{dst(k)}^{(out,s)}$

$\alpha_k^{(out,s)}=U_k^{(out,s)}\cdot inv_k^{edge(out,s)}$

$inv^{(out,s)}$ 与 $\alpha^{(out,s)}$ 只用于分母非零和有限域 Softmax 关系；输出定点结果由第 2.4.4 节的整数聚合分子与 $SignedRoundDiv$ 关系确定。

把 $U^{(out,s)},\ Sum^{(out,s)},\ Sum^{edge(out,s)},\ inv^{(out,s)},\ inv^{edge(out,s)},\alpha^{(out,s)}$ 分别插值成 $P_{U^{(out,s)}},\ P_{Sum^{(out,s)}},\ P_{Sum^{edge(out,s)}},\ P_{inv^{(out,s)}},\ P_{inv^{edge(out,s)}},\ P_{\alpha^{(out,s)}}$

提交承诺：$[P_{E_{src}^{edge(out,s)}}],\ [P_{E_{dst}^{edge(out,s)}}],\ [P_{Y'^{\star,edge(s)}}],\ [P_{Table^{src(out,s)}}],\ [P_{Query^{src(out,s)}}],\ [P_{m_{src}^{(out,s)}}],\ [P_{J_{src}^{node(out,s)}}],\ [P_{J_{src}^{edge(out,s)}}],\ [P_{R_{src}^{node(out,s)}}],\ [P_{R_{src}^{edge(out,s)}}],\ [P_{S^{(out,s)}}],\ [P_{Z^{(out,s)}}],\ [P_{M^{(out,s)}}],\ [P_{M^{edge(out,s)}}],\ [P_{\Delta^{+(out,s)}}],\ [P_{s_{max}^{(out,s)}}],\ [P_{C_{max}^{(out,s)}}],\ [P_{Table^{L(out,s)}}],\ [P_{Query^{L(out,s)}}],\ [P_{m_L^{(out,s)}}],\ [P_{J_{tbl}^{L(out,s)}}],\ [P_{J_{qry}^{L(out,s)}}],\ [P_{R_L^{(out,s)}}],\ [P_{Table^{R(out,s)}}],\ [P_{Query^{R(out,s)}}],\ [P_{m_R^{(out,s)}}],\ [P_{J_{tbl}^{R(out,s)}}],\ [P_{J_{qry}^{R(out,s)}}],\ [P_{R_R^{(out,s)}}],\ [P_{Table^{exp(out,s)}}],\ [P_{Query^{exp(out,s)}}],\ [P_{m_{exp}^{(out,s)}}],\ [P_{J_{tbl}^{exp(out,s)}}],\ [P_{J_{qry}^{exp(out,s)}}],\ [P_{R_{exp}^{(out,s)}}],\ [P_{U^{(out,s)}}],\ [P_{Sum^{(out,s)}}],\ [P_{Sum^{edge(out,s)}}],\ [P_{inv^{(out,s)}}],\ [P_{inv^{edge(out,s)}}],\ [P_{\alpha^{(out,s)}}]$

#### 2.4.4 第 $s$ 个输出头的聚合前输出、PSQ 与压缩绑定

对每个节点 $i$，定义：

1. 输出投影结果的类别压缩使用第 2.4 节定义并在输出路由前提交的 $Y_i'^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}\xi_{out}^c$；本节不得重新生成或替换该对象

2. 有限域聚合前输出：$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}$

3. 整数聚合分子：$Y_{agg,num,i,c}^{(s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)}Y_{src(k),c}'^{(s)}=Sum_i^{(out,s)}Y_{i,c}^{pre(s)}$

4. 聚合前定点结果：$Y_{i,c}^{q(s)}=SignedRoundDiv\!\left(S_{out,pre}Y_{agg,num,i,c}^{(s)},S_{Y'}^{(s)}Sum_i^{(out,s)}\right)$

5. 聚合前输出的类别压缩：$Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{pre(s)}\xi_{out}^c$

6. 聚合前定点结果的类别压缩：$Y_i^{q,\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{q(s)}\xi_{out}^c$

对每个边索引 $k$，定义：

7. 输出投影结果的边域广播压缩：$Y_{k}'^{\star,edge(s)}=Y_{src(k)}'^{\star(s)}$

8. 压缩加权边特征：$\widehat y_k^{\star(s)}=\alpha_k^{(out,s)}Y_{k}'^{\star,edge(s)}$

9. 聚合前输出压缩广播：$Y_k^{pre,\star,edge(s)}=Y_{dst(k)}^{pre,\star(s)}$

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

$Y_{agg,num}^{(s)}$ 与 $Y^{q(s)}$ 是量化局部矩阵，其唯一 proof 表示分别为下文的 $P_{Y_{agg,num,flat}^{(s)}}$ 与 $P_{Y_{q,flat}^{(s)}}$；不生成同名 coefficient-basis commitment。$Y^{q,\star(s)}$ 仅为见证生成中的派生检查量，不进入 commitment、evaluation 或 opening registry。

$P_{Y^{pre,\star(s)}}(X)=\sum_{i=0}^{n_N-1}Y^{pre,\star(s)}[i]L_i^{(N)}(X)$

$P_{Y^{pre,\star,edge(s)}}(X)=\sum_{k=0}^{n_{edge}-1}Y^{pre,\star,edge(s)}[k]L_k^{(edge)}(X)$

$P_{\widehat y^{\star(s)}}(X)=\sum_{k=0}^{n_{edge}-1}\widehat y^{\star(s)}[k]L_k^{(edge)}(X)$

$P_{w_{out}^{(s)}}(X)=\sum_{k=0}^{n_{edge}-1}w_{out}^{(s)}[k]L_k^{(edge)}(X)$

$P_{T_{out}^{pre(s)}}(X)=\sum_{i=0}^{n_N-1}T_{out}^{pre(s)}[i]L_i^{(N)}(X)$

$P_{T_{out}^{pre,edge(s)}}(X)=\sum_{k=0}^{n_{edge}-1}T_{out}^{pre,edge(s)}[k]L_k^{(edge)}(X)$

$P_{PSQ^{(out,s)}}(X)=\sum_{k=0}^{n_{edge}-1}PSQ^{(out,s)}[k]L_k^{(edge)}(X)$

生成聚合前输出压缩绑定挑战：$y_{out,pre}^{(s)}=H_{FS}(\text{transcript},[P_{Y^{pre(s)}}],[P_{Y^{pre,\star(s)}}])$

要求：$P_{Y^{pre,\star(s)}}(y_{out,pre}^{(s)})=\sum_{c=0}^{C-1}\Big(\sum_{i=0}^{N-1}Y_{i,c}^{pre(s)}L_i^{(N)}(y_{out,pre}^{(s)})\Big)\xi_{out}^c$

提交承诺：$[P_{Y'^{\star(s)}}],\ [P_{Y'^{\star,edge(s)}}],\ [P_{Y^{pre(s)}}],\ [P_{Y^{pre,\star(s)}}],\ [P_{Y^{pre,\star,edge(s)}}],\ [P_{\widehat y^{\star(s)}}],\ [P_{w_{out}^{(s)}}],\ [P_{T_{out}^{pre(s)}}],\ [P_{T_{out}^{pre,edge(s)}}],\ [P_{PSQ^{(out,s)}}]$。

对行优先索引 $q=iC+c$，在 $\mathbb H_{edge}$ 上定义：

$$Sum_{flat}^{(out,s)}[q]=Sum_i^{(out,s)},$$

$$Y_{pre,flat}^{(s)}[q]=Y_{i,c}^{pre(s)},$$

$$Y_{agg,num,flat}^{(s)}[q]=Sum_{flat}^{(out,s)}[q]Y_{pre,flat}^{(s)}[q],$$

$$Y_{q,flat}^{(s)}[q]=SignedRoundDiv\!\left(S_{out,pre}Y_{agg,num,flat}^{(s)}[q],S_{Y'}^{(s)}Sum_{flat}^{(out,s)}[q]\right).$$

$Sum_{flat}^{(out,s)}$ 通过 indexed-view 与 $P_{Sum^{(out,s)}}$ 绑定，$Y_{pre,flat}^{(s)}$ 通过 basis link 与 $P_{Y^{pre(s)}}$ 绑定；$Y_{q,flat}^{(s)}$ 本身就是量化局部矩阵 $Y^{q(s)}$ 的 canonical row-major trace。乘法关系、$SignedRoundDiv$ 关系和范围 LogUp 进入边域商多项式。

将 $Sum_{flat}^{(out,s)},Y_{pre,flat}^{(s)},Y_{agg,num,flat}^{(s)},Y_{q,flat}^{(s)}$ 插值为 $\mathbb H_{edge}$ 上的多项式。对该除法实例生成 $\sigma_{outq}^{(s)},z_{outq}^{(s)},a_{outq}^{(s)},(a_{outq}^{(s)})^{-1},d_{outq}^{-1(s)},q_{outq}^{(s)},r_{outq}^{(s)},\delta_{outq}^{(s)}$。提交：

$$[P_{Sum_{flat}^{(out,s)}}],\ [P_{Y_{pre,flat}^{(s)}}],\ [P_{Y_{agg,num,flat}^{(s)}}],\ [P_{Y_{q,flat}^{(s)}}],$$

$$[P_{\sigma_{outq}^{(s)}}],\ [P_{z_{outq}^{(s)}}],\ [P_{a_{outq}^{(s)}}],\ [P_{(a_{outq}^{(s)})^{-1}}],\ [P_{d_{outq}^{-1(s)}}],\ [P_{q_{outq}^{(s)}}],\ [P_{r_{outq}^{(s)}}],\ [P_{\delta_{outq}^{(s)}}].$$

随后按 $V\in(a_{outq}^{(s)},d_{outq}^{(s)}-1,q_{outq}^{(s)},r_{outq}^{(s)},\delta_{outq}^{(s)})$ 的固定顺序，并在每个 $V$ 内按 $h=0,1,\ldots,\ell_{outq,V}-1$ 的顺序，提交第 3.5.2 节定义的定点除法辅助承诺族中的全部 limb、范围查询、重数、分母逆元和范围累加器 commitments；查询选择器由验证者本地重建。

#### 2.4.5 第 $s$ 个输出头的输出偏置、加偏置后压缩绑定与目标路由

对每个节点 $i$ 与类别索引 $c$，正式定义：

- 先做公开尺度对齐：$\widetilde Y_{i,c}^{pre(s)}=SignedRoundDiv\!\left(S_{out,head}Y_{i,c}^{q(s)},S_{out,pre}\right)$
- 再做 bias-add：$Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}$

对每个节点 $i$，进一步定义：

- 尺度对齐后的聚合前压缩：$\widetilde Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}\widetilde Y_{i,c}^{pre(s)}\xi_{out}^c$
- 输出头压缩：$Y_i^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{(s)}\xi_{out}^c$
- 输出头压缩广播：$Y_k^{\star,edge(s)}=Y_{dst(k)}^{\star(s)}$

定义输出头偏置压缩常数：$b_{\star}^{(out,s)}=\sum_{c=0}^{C-1}b_c^{(out,s)}\xi_{out}^c$

当 $S_{out,pre}=S_{out,head}$ 时，该 $SignedRoundDiv$ 实例退化为恒等关系；其他情况下，其舍入模式、范围和无模回绕界由 $quant\_cfg\_id$ 唯一决定。

定义多项式：

$P_{\widetilde Y^{pre(s)}}(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}\widetilde Y_{i,c}^{pre(s)}X^{i\cdot C+c}$

$P_{\widetilde Y^{pre,\star(s)}}(X)=\sum_{i=0}^{n_N-1}\widetilde Y^{pre,\star(s)}[i]L_i^{(N)}(X)$

$P_{Y^{(s)}}(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}Y_{i,c}^{(s)}X^{i\cdot C+c}$

$P_{Y^{\star(s)}}(X)=\sum_{i=0}^{n_N-1}Y^{\star(s)}[i]L_i^{(N)}(X)$

$P_{Y^{\star,edge(s)}}(X)=\sum_{k=0}^{n_{edge}-1}Y^{\star,edge(s)}[k]L_k^{(edge)}(X)$

对行优先索引 $q=iC+c$，定义：

$$Y_{q,flat}^{(s)}[q]=Y_{i,c}^{q(s)},$$

$$\widetilde Y_{flat}^{pre(s)}[q]=\widetilde Y_{i,c}^{pre(s)},$$

$$b_{flat}^{(out,s)}[q]=b_c^{(out,s)},$$

$$Y_{flat}^{(s)}[q]=Y_{i,c}^{(s)}.$$

将 $Y_{q,flat}^{(s)},\widetilde Y_{flat}^{pre(s)},b_{flat}^{(out,s)},Y_{flat}^{(s)}$ 插值为 $\mathbb H_{edge}$ 上的多项式。

$Y_{q,flat}^{(s)}$ 直接引用上一节定义并承诺的 canonical row-major trace；$\widetilde Y_{flat}^{pre(s)}$ 与 $P_{\widetilde Y^{pre(s)}}$、$Y_{flat}^{(s)}$ 与 $P_{Y^{(s)}}$ 分别通过 basis link 绑定。$b_{flat}^{(out,s)}$ 通过 indexed-view 与本份 proof 中 $\mathbb H_C$ 上的偏置 canonical commitment 绑定；后者已通过模型再盲化可除方程绑定到 $VK_{model}$。

尺度对齐除法生成 $\sigma_{align}^{(s)},z_{align}^{(s)},a_{align}^{(s)},(a_{align}^{(s)})^{-1},d_{align}^{-1(s)},q_{align}^{(s)},r_{align}^{(s)},\delta_{align}^{(s)}$，并在边域商中施加第 0.4 节规定的 $SignedRoundDiv$ 约束。偏置关系在 $\mathbb H_{edge}$ 上施加：

$$C_{out,bias}^{(s)}(X)=P_{Q_{NC}^{valid}}(X)\big(P_{Y_{flat}^{(s)}}(X)-P_{\widetilde Y_{flat}^{pre(s)}}(X)-P_{b_{flat}^{(out,s)}}(X)\big).$$

生成输出偏置绑定挑战 $y_{bias}^{(out,s)}$，用于绑定 $P_{\widetilde Y^{pre,\star(s)}}$ 与 $P_{Y^{\star(s)}}$ 的 bias-add 关系；再生成压缩绑定挑战 $y_{out}^{(s)}$，用于绑定 $P_{Y^{(s)}}$ 与 $P_{Y^{\star(s)}}$ 的类别压缩关系。

随后在 $P_{E_{dst}^{(out,s)}},P_{M^{(out,s)}},P_{Sum^{(out,s)}},P_{inv^{(out,s)}},P_{Y^{\star(s)}}$ 固定后先生成 $\eta_{dst}^{(out,s)}$；提交由该挑战确定的表端、查询端和重数 commitments 后生成 $\beta_{dst}^{(out,s)}$，再生成分母逆元与累加器 commitments。定义：

$Table^{dst(out,s)}[i]=Q_N[i]\big(i+\eta_{dst}^{(out,s)}E_{dst,i}^{(out,s)}+(\eta_{dst}^{(out,s)})^2M_i^{(out,s)}+(\eta_{dst}^{(out,s)})^3Sum_i^{(out,s)}+(\eta_{dst}^{(out,s)})^4inv_i^{(out,s)}+(\eta_{dst}^{(out,s)})^5Y_i^{pre,\star(s)}+(\eta_{dst}^{(out,s)})^6Y_i^{\star(s)}\big)$

$Query^{dst(out,s)}[k]=Q_{edge}^{valid}[k]\big(dst(k)+\eta_{dst}^{(out,s)}E_{dst,k}^{edge(out,s)}+(\eta_{dst}^{(out,s)})^2M_k^{edge(out,s)}+(\eta_{dst}^{(out,s)})^3Sum_k^{edge(out,s)}+(\eta_{dst}^{(out,s)})^4inv_k^{edge(out,s)}+(\eta_{dst}^{(out,s)})^5Y_k^{pre,\star,edge(s)}+(\eta_{dst}^{(out,s)})^6Y_k^{\star,edge(s)}\big)$

$m_{dst}^{(out,s)}[i]=\#\{k\mid dst(k)=i\}$

节点端与边端满足：

$$\sum_{i=0}^{N-1}\frac{m_{dst}^{(out,s)}[i]}{Table^{dst(out,s)}[i]+\beta_{dst}^{(out,s)}}
=
\sum_{k=0}^{E-1}\frac{1}{Query^{dst(out,s)}[k]+\beta_{dst}^{(out,s)}}.$$

未掩蔽共同终值不进入 proof；其相等关系由 KZG 同点平移归一化关系验证。点互异时只编码平移 commitment、单次 masked 检查值和普通 point-grouped witness 引用；点相同时只登记普通零值 claim。

$R_{dst}^{node(out,s)}[0]=0$

$R_{dst}^{node(out,s)}[i+1]=R_{dst}^{node(out,s)}[i]+Q_N[i]\cdot\frac{m_{dst}^{(out,s)}[i]}{Table^{dst(out,s)}[i]+\beta_{dst}^{(out,s)}}$

$R_{dst}^{edge(out,s)}[0]=0$

$R_{dst}^{edge(out,s)}[k+1]=R_{dst}^{edge(out,s)}[k]+Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{dst(out,s)}[k]+\beta_{dst}^{(out,s)}}$

插值得到：$P_{Table^{dst(out,s)}},\ P_{Query^{dst(out,s)}},\ P_{m_{dst}^{(out,s)}},\ P_{J_{dst}^{node(out,s)}},\ P_{J_{dst}^{edge(out,s)}},\ P_{R_{dst}^{node(out,s)}},\ P_{R_{dst}^{edge(out,s)}}$

提交承诺：$[P_{\widetilde Y^{pre(s)}}],\ [P_{\widetilde Y^{pre,\star(s)}}],\ [P_{Y^{(s)}}],\ [P_{Y^{\star(s)}}],\ [P_{Y^{\star,edge(s)}}],\ [P_{Table^{dst(out,s)}}],\ [P_{Query^{dst(out,s)}}],\ [P_{m_{dst}^{(out,s)}}],\ [P_{J_{dst}^{node(out,s)}}],\ [P_{J_{dst}^{edge(out,s)}}],\ [P_{R_{dst}^{node(out,s)}}],\ [P_{R_{dst}^{edge(out,s)}}]$，以及

$$[P_{Y_{q,flat}^{(s)}}],\ [P_{\widetilde Y_{flat}^{pre(s)}}],\ [P_{b_{flat}^{(out,s)}}],\ [P_{Y_{flat}^{(s)}}],\ [P_{\sigma_{align}^{(s)}}],\ [P_{z_{align}^{(s)}}],\ [P_{a_{align}^{(s)}}],\ [P_{(a_{align}^{(s)})^{-1}}],\ [P_{d_{align}^{-1(s)}}],\ [P_{q_{align}^{(s)}}],\ [P_{r_{align}^{(s)}}],\ [P_{\delta_{align}^{(s)}}].$$

随后按 $V\in(a_{align}^{(s)},d_{align}^{(s)}-1,q_{align}^{(s)},r_{align}^{(s)},\delta_{align}^{(s)})$ 的固定顺序，并在每个 $V$ 内按 $h=0,1,\ldots,\ell_{align,V}-1$ 的顺序，提交第 3.5.2 节定义的定点除法辅助承诺族中的全部 limb、范围查询、重数、分母逆元和范围累加器 commitments；查询选择器由验证者本地重建。

#### 2.4.6 最终多头平均输出

当全部输出头完成后，定义最终求和：

$$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)}.$$

最终平均使用第 0.4 节固定的带符号整数除法：

$$Y_{i,c}^{avg}=SignedRoundDiv\!\left(Y_{i,c}^{sum},K_{out}\right).$$

最终平均保持输出头共同尺度，即 $S_{out}=S_{out,head}$。最终输出定义为：

$$Y_{i,c}=Y_{i,c}^{avg}.$$

定义最终输出系数多项式：

$$P_Y(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}Y_{i,c}X^{i\cdot C+c}.$$

定义最终压缩输出与广播：

- $Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c$
- $Y_k^{\star,edge}=Y_{dst(k)}^{\star}$
- $P_{Y^{\star}}(X)=\sum_{i=0}^{n_N-1}Y^{\star}[i]L_i^{(N)}(X)$
- $P_{Y^{\star,edge}}(X)=\sum_{k=0}^{n_{edge}-1}Y^{\star,edge}[k]L_k^{(edge)}(X)$

对行优先索引 $q=iC+c$，在 $\mathbb H_{edge}$ 上定义：

$$Y_{sum,flat}[q]=\sum_{s=0}^{K_{out}-1}Y_{flat}^{(s)}[q],$$

$$Y_{flat}[q]=Y_{i,c}.$$

将 $Y_{sum,flat},Y_{flat}$ 插值为 $\mathbb H_{edge}$ 上的多项式。对该平均除法生成 $\sigma_{avg},z_{avg},a_{avg},a_{avg}^{-1},d_{avg}^{-1},q_{avg},r_{avg},\delta_{avg}$，并实例化：

$$Y_{flat}[q]=SignedRoundDiv(Y_{sum,flat}[q],K_{out}).$$

$Y_{flat}$ 通过 basis link 与公开输出 $P_Y$ 绑定；各 $Y_{flat}^{(s)}$ 已在第 2.4.5 节绑定到对应输出头。求和约束、$SignedRoundDiv$ 约束和范围 LogUp 全部进入边域商多项式。

先按固定顺序提交最终平均来源 commitment 总表：

$$\mathbf{Com}_{out}^{avg}=\Big([P_Y], [P_{Y^{\star}}], [P_{Y^{\star,edge}}], [P_{Y_{sum,flat}}], [P_{Y_{flat}}], [P_{\sigma_{avg}}], [P_{z_{avg}}], [P_{a_{avg}}], [P_{a_{avg}^{-1}}], [P_{d_{avg}^{-1}}], [P_{q_{avg}}], [P_{r_{avg}}], [P_{\delta_{avg}}], \mathbf{Com}_{range}^{avg}\Big),$$

其中 $\mathbf{Com}_{range}^{avg}$ 按 $V\in(a_{avg},d_{avg}-1,q_{avg},r_{avg},\delta_{avg})$ 的固定顺序，并在每个 $V$ 内按 $h=0,1,\ldots,\ell_{avg,V}-1$ 的顺序，编码第 3.5.2 节定义的全部 limb、范围查询、重数、分母逆元和范围累加器 commitments；查询选择器由验证者本地重建。

只有在 $\mathbf{Com}_{out}^{avg}$ 全部固定并吸入 transcript 后，才能生成最终平均输出压缩绑定挑战：

$$y_{avg}=H_{FS}\!\left(\text{transcript},\texttt{"zkgat:bind:output:avg"},\operatorname{CanonicalCommitmentList}(\mathbf{Com}_{out}^{avg})\right).$$

不得仅吸入 $[P_Y]$ 与 $[P_{Y^{\star}}]$，也不得在任何最终平均除法或范围辅助 commitment 固定之前生成 $y_{avg}$。

挑战固定后要求：

$$P_{Y^{\star}}(y_{avg}) = \sum_{c=0}^{C-1}\Big(\sum_{i=0}^{N-1}Y_{i,c}L_i^{(N)}(y_{avg})\Big)\xi_{out}^c.$$

$\pi_{bind}^{out,avg}$ 负责各输出头行优先 trace、$Y_{sum,flat}$、$Y_{flat}$、$P_Y$ 与 $P_{Y^{\star}}$ 的一致性；平均数值关系由上述边域整数除法约束证明。

## 3. 证明生成

### 3.1 输入与输出

证明算法输入是：

- 参数生成输出：$(PK,VK_{KZG},VK_{static},VK_{model})$，其中 $PK$ 由 $PK_{coef}$ 与当前 `domain_cfg` 选定的七个 $PK_{lag}^{(\mathcal D,n_{\mathcal D})}$ 组成

- 公共输入：$(I,src,dst,G_{batch},node\_ptr,edge\_ptr,N,E,N_{total},L,\{K_{hid}^{(\ell)}\}_{\ell=1}^{L-1},\{d_{in}^{(\ell)}\}_{\ell=1}^{L},\{d_h^{(\ell)}\}_{\ell=1}^{L-1},\{d_{cat}^{(\ell)}\}_{\ell=1}^{L-1},C,B,K_{out},Y_{public})$，其中 $Y_{public}\in\mathbb F_p^{N\times C}$ 为完整公开 logits

- 全部动态见证对象

输出最终证明：$\pi_{GAT}$

### 3.2 Fiat–Shamir 挑战顺序

验证者与证明者必须严格按以下顺序重放。任何依赖挑战 $c$ 的列或 commitment 都不得在 $c$ 生成之前吸入 transcript。

#### 3.2.1 全局阶段

1. 解析并吸入 $M_{pub}$。$M_{pub}$ 必须包含独立 `proof_nonce`、完整行优先 $Y_{public}$、其形状和规范摘要。验证者先检查数组长度为 $NC$，从 $Y_{public}$ 本地建立 $P_Y$ 的公开评值器，再吸入全部公开拓扑、维度、量化、SRS、静态表和模型锚点 commitments。
2. parser 先按 `ProofSchema` 定位并完成全部字段的长度、类型和重复性检查，但不得按原始字节出现顺序直接吸入 transcript。transcript 只按本节给出的依赖顺序吸入字段。首先提交不依赖动态挑战的基础 commitments，包括 proof-local 模型 commitments、公开输入 commitments、未压缩前向 trace、canonical tensor commitments、量化源对象和关系来源 commitments。
3. 隐藏层按 $\ell=1,\ldots,L-1$ 顺序执行。对当前层，先按头序固定全部 $H'^{(\ell,r)},E_{src}^{(\ell,r)},E_{dst}^{(\ell,r)}$ 来源 commitments；每个 $\xi^{(\ell,r)}$ 仅在对应 $[P_{H'^{(\ell,r)}}]$ 固定后生成，并立即提交 $H^{\star(\ell,r)}$。随后对每个头依次完成 source route、LeakyReLU、最大值 / 范围、指数、$Sum/inv/\alpha$、聚合前矩阵与 PSQ、`SignedRoundDiv`、ELU 和 target route。任一 LogUp / route 实例都严格执行“固定未压缩来源 commitments → 生成该实例 $\eta$（若存在）→ 提交 $Table,Query,m$ → 生成 $\beta$ → 提交分母逆元 → 提交累加器”的局部顺序；不得把后续头或后续层对象插入该实例内部。每个 route 的节点端和边端累加器 commitments 均固定后，若终点不同，立即按第 3.5.3 节提交对应平移 commitment 并生成该 route 的 $y_{rel}$；masked 检查值与 witnesses 延迟到 opening 阶段。当前层全部头完成后提交 $[P_{H_{cat}^{(\ell)}}]$，生成 $\xi_{cat}^{(\ell)}$，再提交 $[P_{H_{cat}^{\star(\ell)}}]$；该拼接输出成为下一层唯一输入。
4. 隐藏层全部完成后进入输出层。先按头序只提交全部 $[P_{Y'^{(s)}}]$，再从该有序列表唯一生成全局 $\xi_{out}$，随后按头序提交 $[P_{Y'^{\star(s)}}]$。对每个输出头依次完成源 / 目标注意力、source route、LeakyReLU、最大值 / 范围、指数、$Sum/inv/\alpha$、聚合前矩阵与 PSQ、输出量化、尺度对齐、偏置绑定、头输出压缩和 target route；每个 LogUp / route 仍使用第 3 项规定的局部顺序。每个输出头的 route 节点端和边端累加器 commitments 均固定后，若终点不同，立即按第 3.5.3 节提交对应平移 commitment并生成该 route 的 $y_{rel}$；masked 检查值与 witnesses 延迟到 opening 阶段。全部输出头完成后提交 $Y_{sum,flat},Y_{flat},P_Y,P_{Y^{\star}},P_{Y^{\star,edge}}$ 及最终平均除法对象；$y_{avg}$ 只能在这些来源 commitments 固定后生成。
5. 所有承诺后压缩挑战，包括聚合绑定、拼接绑定、输出绑定和各 $y_{proj},y_{src},y_{dst},y_{star},y_{cat},y_{out}$，必须在其全部来源 commitment 与结果 commitment 固定后生成。依赖该挑战的折叠列、basis accumulator、关系 view 或辅助 commitment 必须在挑战之后提交。任何挑战不得吸入由自身决定的对象。
6. 每个 $\Pi_{bind}$ 按顶层固定顺序依次执行：先固定来源、canonical、proof-local 模型、coefficient-basis relation view、view value 和全部 coefficient chunk commitments；随后生成 $\eta_{view}$、非零 $\gamma_{basis}$ 及承诺后聚合压缩挑战，并提交由这些挑战决定的表 / 查询、basis accumulator 和聚合辅助 commitments；在 $Table,Query,m$ 固定后生成 $\beta_{view}$，再提交分母逆元与累加器 commitments；固定 $[\widehat P_C]$、全部 $[\widehat A_t],[\widehat B_t]$、$[\widehat Q_{CRPC}]$ 与 $[R_Q^{coef}]$ 后生成非零 $\gamma_{CRPC}$。对规范化后 $m\ge2$ 的线性关系，在来源 commitments 固定后提交 $\{[\widetilde F_j]\}_{j=2}^{m}$，生成该关系的 $y_{rel}$，并登记平移检查 claims 与 $([G_{rel}],z_\star,0)$；$m=1$ 只登记普通零值 claim，$m=0$ 不提交关系字段。实际 opening witnesses 只能在第 9 步的普通 registry 完整固定后生成。任何后续对象不得反向影响前序挑战，且此阶段不提交局部 quotient commitment。
7. 吸入全部动态 commitments、关系证明 commitments、proof-local 模型再盲化 commitments 和静态 / 模型锚点 commitments，生成商聚合挑战 $\alpha_{quot}$。opening witnesses 不参与商聚合挑战。
8. 构造并提交七个 $[t_{\mathcal D}^{zk}]$ 与 $[R_{t,\mathcal D}]$，随后按 $FH,edge,in,d_h,cat,C,N$ 的固定顺序生成工作域开放点 $z_{FH},z_{edge},z_{in},z_{d_h},z_{cat},z_C,z_N$。对每个候选 $z_{\mathcal D}$，公开 counter 必须同时保证 $Z_{\mathcal D}(z_{\mathcal D})\ne0$，并保证新产生的 $z_{\mathcal D}$ 与 $z_{\mathcal D}\omega_{\mathcal D}$ 和此前全部工作域的两点均不相等；由此 14 个工作域 opening points 两两不同。全部排除因子进入相应 challenge 的公开候选集合、`RandomIdentityLedger` 与 `SecurityBudgetLedger`。
9. 汇总 $\mathbf{Eval}_{ext}$、$\mathbf S_{route}$、$\mathbf{Eval}_{dom}$ 与各 $\mathbf{Eval}_{bind}^{zk}$ 登记的全部普通 KZG opening claims。对每个规范点吸入有序 commitment、可见性标签和 claimed value 后生成非零 $\rho_z$，随后构造并提交该点唯一的 $[W_z]$。不同点不得共享 challenge 或 witness；同一 claim 不得跨块重复登记。

所有挑战使用域分离标签。每个挑战 $c$ 必须按 `RandomIdentityLedger` 的规范候选集合 $\mathcal A_c$ 派生：验证者从 transcript 和固定宽度 counter 重建候选，并仅对 $E_c(c)=0$ 等公开条件递增 counter。承担字段压缩、随机线性组合或折叠作用的挑战通常还要求非零，该条件作为因子 $Y$ 进入 $E_c$；没有公开排除条件的 $\beta$ 在完整 $\mathbb F_p$ 上采样，不要求非零，也绝不依据私有分母值拒绝采样。工作域外 opening 点还必须满足其登记的 $Z_{\mathcal D}(z)\ne0$。点固定后若 $[\tau-z]_2=\mathcal O_{\mathbb G_2}$，则公开值 $z$ 已等于陷门 $\tau$；证明生成与验证必须立即返回 `SRS_COMPROMISED`，停止使用并撤销该 `srs_id`，不得通过重新派生另一个点继续使用同一 SRS。

### 3.3 约束族与商多项式

为避免省略，本节先给出每类约束的显式公式，再给出每个工作域商多项式的完全展开表达。

#### 3.3.1 通用 LogUp 起点、递推、终点模板

设某个 LogUp 子系统定义在 $\mathbb H_{\mathcal D}$ 上，包含 $P_{Table},P_{Query},P_m,P_{Q_{tbl}},P_{Q_{qry}},P_{J_{tbl}},P_{J_{qry}},P_R$ 与挑战 $\beta$。固定约束为：

$$C_{lookup,0}=First_{\mathcal D}P_R,$$

$$C_{lookup,tblinv}=P_{Q_{tbl}}\big((P_{Table}+\beta)P_{J_{tbl}}-1\big),$$

$$C_{lookup,tblpad}=(1-P_{Q_{tbl}})P_{J_{tbl}},$$

$$C_{lookup,qryinv}=P_{Q_{qry}}\big((P_{Query}+\beta)P_{J_{qry}}-1\big),$$

$$C_{lookup,qrypad}=(1-P_{Q_{qry}})P_{J_{qry}},$$

$$\begin{aligned}
C_{lookup,1}(X)={}&(1-Last_{\mathcal D}(X))\Big[
(P_R(\omega_{\mathcal D}X)-P_R(X))(P_{Table}(X)+\beta)(P_{Query}(X)+\beta)\\
&-P_{Q_{tbl}}(X)P_m(X)(P_{Query}(X)+\beta)
+P_{Q_{qry}}(X)(P_{Table}(X)+\beta)\Big],
\end{aligned}$$

$$C_{lookup,2}=Last_{\mathcal D}P_R.$$

逆元约束单独保证所有有效分母非零；交叉相乘递推只证明有理和递推，不得替代逆元约束。无效区的 $J_{tbl},J_{qry}$ 必须为零。

#### 3.3.2 通用节点路由模板

节点端包含 $P_{Table},P_m,P_{J_{node}},P_{R^{node}}$ 与挑战 $\beta$。固定约束为：

$$C_{route,node,0}=First_NP_{R^{node}},$$

$$C_{route,node,inv}=P_{Q_N}\big((P_{Table}+\beta)P_{J_{node}}-1\big),$$

$$C_{route,node,pad}=(1-P_{Q_N})P_{J_{node}},$$

$$C_{route,node,1}=(1-Last_N)\Big((P_{R^{node}}(\omega_NX)-P_{R^{node}})(P_{Table}+\beta)-P_{Q_N}P_m\Big).$$

节点端不单独施加零终点；其终点与对应边端终点通过第 0.7.8 节的 KZG 同点平移归一化关系绑定。

#### 3.3.3 通用边路由模板

边端包含 $P_{Query},P_{J_{edge}},P_{R^{edge}}$。固定约束为：

$$C_{route,edge,0}=First_{edge}P_{R^{edge}},$$

$$C_{route,edge,inv}=P_{Q_{edge}^{valid}}\big((P_{Query}+\beta)P_{J_{edge}}-1\big),$$

$$C_{route,edge,pad}=(1-P_{Q_{edge}^{valid}})P_{J_{edge}},$$

$$C_{route,edge,1}=(1-Last_{edge})\Big((P_{R^{edge}}(\omega_{edge}X)-P_{R^{edge}})(P_{Query}+\beta)-P_{Q_{edge}^{valid}}\Big).$$

跨域终点关系

$$P_{R^{node}}(z_{last,N})-P_{R^{edge}}(z_{last,edge})=0$$

只由既有 $\mathbf S_{route}$ 中按点规范化选择的 同点零值 claim 或 `KZGShiftRelationProof` 验证，不输出共同终值。

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

4. 递推：$$\begin{aligned} C_{max,1}(X) ={}&\big(1-Last_{edge}(X)\big)\Big[ P_C(\omega_{edge}X) - \Big( \big(1-P_{Q_{edge}^{valid}}(\omega_{edge}X)\big)P_C(X)\\ &\qquad\qquad+ P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_s(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ \big(1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)\big)\big(P_C(X)+P_s(\omega_{edge}X)\big) \big) \Big) \Big]\end{aligned}$$

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



2. 递推约束：$$\begin{aligned} C_{psq,1}(X) ={}&\big(1-Last_{edge}(X)\big)\Big[ P_{PSQ}(\omega_{edge}X) - \Big( \big(1-P_{Q_{edge}^{valid}}(\omega_{edge}X)\big)P_{PSQ}(X)\\ &\qquad\qquad+ P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_w(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ \big(1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)\big)\big(P_{PSQ}(X)+P_w(\omega_{edge}X)\big) \big) \Big) \Big]\end{aligned}$$



3. 组末约束：$C_{psq,end}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{PSQ}(X)-P_{T^{edge}}(X)\big)$


#### 3.3.6 逆元约束

设节点域分母、逆元分别为 $P_{Sum}(X)$、$P_{inv}(X)$，则节点域逆元约束为：$C_{inv}(X)=P_{Q_N}(X)\cdot\big(P_{Sum}(X)P_{inv}(X)-1\big)$

本文正式协议不采用复合多项式替换 $P_{Sum}(P_{dst}(X)) \quad P_{inv}(P_{dst}(X))$ 来表达边域广播一致性。

本文唯一采用的正式口径是：

1. 先把边域广播列 $Sum^{edge}$、$inv^{edge}$ 显式作为独立见证列提交；
2. 再通过目标路由查询端绑定、PSQ 目标值绑定以及后续对应的边域 / 节点域约束，强制这些广播列与对应节点域列保持一致。

因此，在 quotient 约束族中，$3.3.8$、$3.3.9$、$3.3.13$、$3.3.14$ 已经给出的目标路由与输出聚合约束，就是本文用于保证广播一致性的唯一正式约束来源；本节不再额外引入第二套复合多项式广播约束。

**$SignedRoundDiv$ 约束模板**

对第 0.4 节定义的任一除法实例 $\nu$，设有效区选择器为 $P_{Q_\nu}$，输入、分母、输出和辅助多项式分别为

$$P_{x_\nu},P_{d_\nu},P_{y_\nu},P_{\sigma_\nu},P_{z_\nu},P_{a_\nu},P_{a_\nu^{-1}},P_{d_\nu^{-1}},P_{q_\nu},P_{r_\nu},P_{\delta_\nu}.$$

固定约束为：

$$C_{\nu,\sigma}=P_{Q_\nu}P_{\sigma_\nu}(P_{\sigma_\nu}-1),$$
$$C_{\nu,abs}=P_{Q_\nu}\big(P_{x_\nu}-(1-2P_{\sigma_\nu})P_{a_\nu}\big),$$
$$C_{\nu,zbin}=P_{Q_\nu}P_{z_\nu}(P_{z_\nu}-1),$$
$$C_{\nu,zero}=P_{Q_\nu}P_{z_\nu}P_{a_\nu},$$
$$C_{\nu,nz}=P_{Q_\nu}\big(P_{a_\nu}P_{a_\nu^{-1}}-(1-P_{z_\nu})\big),$$
$$C_{\nu,zsign}=P_{Q_\nu}P_{z_\nu}P_{\sigma_\nu},$$
$$C_{\nu,zinv}=P_{Q_\nu}P_{z_\nu}P_{a_\nu^{-1}},$$
$$C_{\nu,dnz}=P_{Q_\nu}\big(P_{d_\nu}P_{d_\nu^{-1}}-1\big),$$
$$C_{\nu,out}=P_{Q_\nu}\big(P_{y_\nu}-(1-2P_{\sigma_\nu})P_{q_\nu}\big).$$

向零截断模式使用：

$$C_{\nu,eq}=P_{Q_\nu}(P_{a_\nu}-P_{d_\nu}P_{q_\nu}-P_{r_\nu}),$$
$$C_{\nu,rem}=P_{Q_\nu}(P_{r_\nu}+P_{\delta_\nu}-P_{d_\nu}+1).$$

向最近整数舍入且半数远离零模式使用：

$$C_{\nu,eq}=P_{Q_\nu}(2P_{a_\nu}+P_{d_\nu}-2P_{d_\nu}P_{q_\nu}-P_{r_\nu}),$$
$$C_{\nu,rem}=P_{Q_\nu}(P_{r_\nu}+P_{\delta_\nu}-2P_{d_\nu}+1).$$

对每个 $V\in\{a_\nu,d_\nu-1,q_\nu,r_\nu,\delta_\nu\}$，`quant_cfg_id` 固定 limb 数 $\ell_{\nu,V}$。其中

$$P_{d_\nu-1}(X)=P_{d_\nu}(X)-1,$$

其余 $P_V$ 直接取相应见证列。证明者提交 limb 多项式 $P_{V^{[h]}}$，并施加重建约束

$$C_{R,recon}^{\nu,V}(X)=P_{Q_\nu}(X)\left(P_V(X)-\sum_{h=0}^{\ell_{\nu,V}-1}2^{Bh}P_{V^{[h]}}(X)\right).$$

对每个 $h=0,\ldots,\ell_{\nu,V}-1$，在 $\mathbb H_{edge}$ 上建立独立范围 LogUp。定义

$$P_{Query_R^{\nu,V,h}}(X)=P_{V^{[h]}}(X),\qquad P_{Q_{qry}^{\nu,V,h}}(X)=P_{Q_\nu}(X),$$

公开表为 $P_{Table_R}=P_{T_{range}}$，表选择器为 $P_{Q_{tbl}^{R}}$，相应重数和累加器为 $P_{m_R^{\nu,V,h}}$、$P_{R_R^{\nu,V,h}}$。固定约束为：

$$C_{R,0}^{\nu,V,h}(X)=First_{edge}(X)P_{R_R^{\nu,V,h}}(X),$$

$$\begin{aligned}
C_{R,1}^{\nu,V,h}(X)={}&\big(1-Last_{edge}(X)\big)\Big[\big(P_{R_R^{\nu,V,h}}(\omega_{edge}X)-P_{R_R^{\nu,V,h}}(X)\big)
\big(P_{T_{range}}(X)+\beta_R^{\nu,V,h}\big)
\big(P_{Query_R^{\nu,V,h}}(X)+\beta_R^{\nu,V,h}\big)\\
&-P_{Q_{tbl}^{R}}(X)P_{m_R^{\nu,V,h}}(X)\big(P_{Query_R^{\nu,V,h}}(X)+\beta_R^{\nu,V,h}\big)\\
&+P_{Q_\nu}(X)\big(P_{T_{range}}(X)+\beta_R^{\nu,V,h}\big)
\Big]\end{aligned}$$

$$C_{R,2}^{\nu,V,h}(X)=Last_{edge}(X)P_{R_R^{\nu,V,h}}(X),$$

$$C_{R,tblinv}^{\nu,V,h}(X)=P_{Q_{tbl}^{R}}(X)\Big((P_{Table_R}(X)+\beta_R^{\nu,V,h})P_{J_{tbl,R}^{\nu,V,h}}(X)-1\Big),$$

$$C_{R,tblpad}^{\nu,V,h}(X)=(1-P_{Q_{tbl}^{R}}(X))P_{J_{tbl,R}^{\nu,V,h}}(X),$$

$$C_{R,qryinv}^{\nu,V,h}(X)=P_{Q_\nu}(X)\Big((P_{Query_R^{\nu,V,h}}(X)+\beta_R^{\nu,V,h})P_{J_{qry,R}^{\nu,V,h}}(X)-1\Big),$$

$$C_{R,qrypad}^{\nu,V,h}(X)=(1-P_{Q_\nu}(X))P_{J_{qry,R}^{\nu,V,h}}(X),$$

$$C_{R,tbl}^{\nu,V,h}(X)=P_{Table_R}(X)-P_{T_{range}}(X),\qquad
C_{R,qry}^{\nu,V,h}(X)=P_{Query_R^{\nu,V,h}}(X)-P_{V^{[h]}}(X).$$

令 $\mathcal C_{div,\nu}$ 表示 $C_{\nu,\sigma},C_{\nu,abs},C_{\nu,zbin},C_{\nu,zero},C_{\nu,nz},C_{\nu,zsign},C_{\nu,zinv},C_{\nu,dnz},C_{\nu,out},C_{\nu,eq},C_{\nu,rem}$、五组 limb 重建约束以及全部 limb 范围 LogUp 约束按 `ProofSchema` 固定指数顺序用 $\alpha_{quot}$ 聚合后的完整分子。任何除法实例遗漏 $C_{\nu,zinv}$ 都属于 schema 错误。

**$\Pi_{bind}$ 七域约束模板**

对每个工作域 $\mathbb H_{\mathcal D}$，`ProofSchema` 固定关系登记表 $\mathcal R_{bind,\mathcal D}$。该登记表包含第 0.7.8 节在该工作域上执行的 indexed-view、basis accumulator、聚合绑定约束，以及在 $\mathbb H_N$ 上执行的 canonical 列压缩、逐列拼接和输出压缩约束。具体地，$\mathcal R_{bind,N}$ 必须按层序、头序登记全部 $C_{aggpre}^{(\ell,r)},C_{agg}^{(\ell,r)},C_{concat}^{(\ell,r,j)},C_{cat}^{(\ell)},C_{outY'}^{(s)},C_{outYpre}^{(s)},C_{out\widetilde Y}^{(s)},C_{outY}^{(s)},C_{outY}^{avg}$。其中 $C_{concat}^{(\ell,r,j)}$ 按 $(\ell,r,j)$ 字典序登记；任何一项缺失、重复或换序都属于 schema 错误。

对 indexed-view 关系 $\rho$，设 canonical 端工作域为 $\mathbb H_{\mathcal D_{can}}$，view 端工作域为 $\mathbb H_{\mathcal D_{view}}$。canonical 端固定：

$$C_{view,can,0}^{\rho}=First_{\mathcal D_{can}}P_{R_{view}^{can,\rho}},$$

$$\begin{aligned}
C_{view,can,1}^{\rho}={}&\big(1-Last_{\mathcal D_{can}}(X)\big)\Big[
\big(P_{R_{view}^{can,\rho}}(\omega_{\mathcal D_{can}}X)-P_{R_{view}^{can,\rho}}(X)\big)
\big(P_{Table_{view}^{can,\rho}}(X)+\beta_{view}^{\rho}\big)\\
&\qquad-P_{Q_{view}^{can,\rho}}(X)P_{m_{view}^{can,\rho}}(X)\Big],
\end{aligned}$$

$$C_{view,tbl}^{\rho}=P_{Table_{view}^{can,\rho}}-P_{Q_{view}^{can,\rho}}\big(P_{idx_{can}^{\rho}}+\eta_{view}^{\rho}P_{value_{can}^{\rho}}\big).$$

$$C_{view,tbl,pad}^{\rho}=(1-P_{Q_{view}^{can,\rho}})P_{Table_{view}^{can,\rho}},$$

$$C_{view,can,inv}^{\rho}=P_{Q_{view}^{can,\rho}}\big((P_{Table_{view}^{can,\rho}}+\beta_{view}^{\rho})P_{J_{view}^{can,\rho}}-1\big),$$

$$C_{view,can,Jpad}^{\rho}=(1-P_{Q_{view}^{can,\rho}})P_{J_{view}^{can,\rho}}.$$

view 端固定：

$$C_{view,qry,0}^{\rho}=First_{\mathcal D_{view}}P_{R_{view}^{qry,\rho}},$$

$$\begin{aligned}
C_{view,qry,1}^{\rho}={}&\big(1-Last_{\mathcal D_{view}}(X)\big)\Big[
\big(P_{R_{view}^{qry,\rho}}(\omega_{\mathcal D_{view}}X)-P_{R_{view}^{qry,\rho}}(X)\big)
\big(P_{Query_{view}^{\rho}}(X)+\beta_{view}^{\rho}\big)\\
&\qquad-P_{Q_{view}^{qry,\rho}}(X)\Big],
\end{aligned}$$

$$C_{view,qry}^{\rho}=P_{Query_{view}^{\rho}}-P_{Q_{view}^{qry,\rho}}\big(P_{idx_{view}^{\rho}}+\eta_{view}^{\rho}P_{value_{view}^{\rho}}\big).$$

$$C_{view,qry,pad}^{\rho}=(1-P_{Q_{view}^{qry,\rho}})P_{Query_{view}^{\rho}},$$

$$C_{view,qry,inv}^{\rho}=P_{Q_{view}^{qry,\rho}}\big((P_{Query_{view}^{\rho}}+\beta_{view}^{\rho})P_{J_{view}^{qry,\rho}}-1\big),$$

$$C_{view,qry,Jpad}^{\rho}=(1-P_{Q_{view}^{qry,\rho}})P_{J_{view}^{qry,\rho}}.$$

上述 canonical 端的编码、值填充、逆元填充、起点和递推约束全部进入 $t_{\mathcal D_{can}}$；view 端对应全部约束进入 $t_{\mathcal D_{view}}$。两个终点不设为零，而由 $\Pi_{bind}$ 中的 KZG 同点平移归一化关系检查：

$$P_{R_{view}^{can,\rho}}(z_{last,\mathcal D_{can}})-P_{R_{view}^{qry,\rho}}(z_{last,\mathcal D_{view}})=0.$$

对每个 coefficient-basis relation object $M$，`ProofSchema` 固定三个公开选择器 $Q_{sem}^M,Q_{gap}^M,Q_{tail}^M$，分别表示真实语义系数、低阶未使用系数和随机尾系数。它们在每个承载 chunk 上满足覆盖、二值和互斥约束：

$$C_{coef,cover}^M=Q_{chunk}^M-Q_{sem}^M-Q_{gap}^M-Q_{tail}^M,$$

$$C_{coef,bin,u}^M=Q_u^M(Q_u^M-1),\qquad u\in\{sem,gap,tail\},$$

$$C_{coef,disj,uv}^M=Q_u^MQ_v^M,\qquad u\ne v,$$

并对低阶空隙施加

$$C_{coef,gap}^M(X)=P_{Q_{gap}^M}(X)P_{\widehat m^M}(X),\qquad C_{coef,pad}^M(X)=\big(1-P_{Q_{chunk}^M}(X)\big)P_{\widehat m^M}(X).$$

$Q_{sem}^M$ 上的系数通过 indexed-view 与 canonical trace 绑定；$Q_{tail}^M$ 只允许承载 `privacy_policy` 登记长度的独立随机尾项；$Q_{chunk}^M=0$ 的位置由公开 padding selector 强制系数为零。任一位置未被三个区域唯一覆盖、空隙非零或随机尾侵入语义区域时必须拒绝。上述 coefficient 区域约束固定进入 $\mathcal C_{bind,edge}$。

对 `Public` 或 `PublicOutput` 的 coefficient-basis 对象，随机尾长度固定为零，$Q_{tail}^M=0$；真实公开系数属于 $Q_{sem}^M$，注册次数界内其余位置全部属于 $Q_{gap}^M$ 并强制为零。公开对象不得借用 coefficient-tail hiding，也不得保留未由公开输入确定的自由系数。

对 basis chunk $(M,c)$，固定：

$$C_{basis,0}^{M,c}=First_{edge}P_{A_{basis,c}^{M}},$$

$$C_{basis,1}^{M,c}=\big(1-Last_{edge}(X)\big)\Big(P_{A_{basis,c}^{M}}(\omega_{edge}X)-P_{A_{basis,c}^{M}}(X)-P_{Q_c^{step}}(X)P_{\widehat m_c^M}(X)P_{\Gamma_c^M}(X)\Big),$$

其中 $P_{Q_c^{step}}$ 是第 0.7.8 节定义的公开步进选择器，$P_{\Gamma_c^M}(\omega_{edge}^t)=\gamma_{basis}^{t}$ 由验证者根据挑战本地重建。$P_{Q_c^{step}}$ 与 $P_{\Gamma_c^M}$ 不作为动态 commitment；其工作域固定为 $\mathbb H_{edge}$，长度和 chunk offset 由 `RelationMeta` 唯一确定并进入 transcript。因子 $1-Last_{edge}$ 屏蔽工作域最后一点到首点的循环转移；$Q_c^{step}=0$ 的填充区保持累加器常值。basis 的跨点终值等式由 KZG 同点平移归一化关系检查，不作为裸常数进入商。

第 0.7.8 节规定的聚合关系在边域包含乘积约束，并分别以 $V_{agg,bind},T_{agg,bind}^{edge},PSQ_{agg,bind}$ 和 $V_{out,agg,bind},T_{out,agg,bind}^{edge},PSQ_{out,agg,bind}$ 实例化第 3.3.5 节的起点、带 $1-Last_{edge}$ 屏蔽的递推和组末约束；节点域包含压缩 canonical/view 绑定。令

$$\mathcal C_{bind,\mathcal D}(X)=
\sum_{\rho\in\mathcal R_{bind,\mathcal D}}
\alpha_{quot}^{e_{bind,\rho}}C_{bind}^{\rho}(X),$$

其中 $C_{bind}^{\rho}$ 按关系登记顺序依次枚举上述 $C_{view,*}$、$C_{coef,*}$、$C_{basis,*}$、$C_{agg,bind,mul}$、聚合绑定状态机、canonical 列压缩和逐列拼接约束。若某工作域登记表为空，则 $\mathcal C_{bind,\mathcal D}=0$。这些项并入原七个工作域商，不生成第八个工作域商。

#### 3.3.7 特征检索域商多项式

本域除下列特征检索约束外，还必须包含 $\mathcal C_{denom,FH}$ 与 $\mathcal C_{padding,FH}$。

特征检索域上的全部约束定义为：

1. lookup 起点约束：$C_{feat,0}(X)=First_{FH}(X)\cdot P_{R_{feat}}(X)$



2. lookup 递推约束：$$\begin{aligned} C_{feat,1}(X) ={}&\big(1-Last_{FH}(X)\big)\Big[ \big(P_{R_{feat}}(\omega_{FH}X)-P_{R_{feat}}(X)\big) \big(P_{Table^{feat}}(X)+\beta_{feat}\big) \big(P_{Query^{feat}}(X)+\beta_{feat}\big)\\ &- P_{Q_{tbl}^{feat}}(X)P_{m_{feat}}(X)\big(P_{Query^{feat}}(X)+\beta_{feat}\big)\\ &+ P_{Q_{qry}^{feat}}(X)\big(P_{Table^{feat}}(X)+\beta_{feat}\big) \Big]\end{aligned}$$



3. lookup 终点约束：$C_{feat,2}(X)=Last_{FH}(X)\cdot P_{R_{feat}}(X)$

4. 表端绑定约束：$C_{feat,tbl}(X)=P_{Table^{feat}}(X)-\Big(P_{Row_{feat}^{tbl}}(X)+\eta_{feat}P_{Col_{feat}^{tbl}}(X)+\eta_{feat}^2P_{T_H}(X)\Big)$

5. 查询端绑定约束：$C_{feat,qry}(X)=P_{Query^{feat}}(X)-\Big(P_{I_{feat}^{qry}}(X)+\eta_{feat}P_{Col_{feat}^{qry}}(X)+\eta_{feat}^2P_H^{feat}(X)\Big)$


于是特征检索域商多项式完全展开为：

$t_{FH}(X) = \frac{ \alpha_{quot}^{e_{feat,0}}C_{feat,0}(X) + \alpha_{quot}^{e_{feat,1}}C_{feat,1}(X) + \alpha_{quot}^{e_{feat,2}}C_{feat,2}(X) + \alpha_{quot}^{e_{feat,tbl}}C_{feat,tbl}(X) + \alpha_{quot}^{e_{feat,qry}}C_{feat,qry}(X) + \mathcal C_{bind,FH}(X) + \mathcal C_{denom,FH}(X)+\mathcal C_{padding,FH}(X)}{ Z_{FH}(X) }$

#### 3.3.8 单个隐藏层模板中第 $r$ 个注意力头在边域上的全部约束

以下约束对应某个固定隐藏层 $\ell$ 的模板；在该模板中，对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$，在边域 $\mathbb H_{edge}$ 上定义以下约束。

##### （一）源路由边端约束

$C_{src,edge,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{src}^{edge(r)}}(X)$

$C_{src,edge,1}^{(r)}(X) = \big(1-Last_{edge}(X)\big)\Big(\big(P_{R_{src}^{edge(r)}}(\omega_{edge}X)-P_{R_{src}^{edge(r)}}(X)\big)\big(P_{Query^{src(r)}}(X)+\beta_{src}^{(r)}\big)-P_{Q_{edge}^{valid}}(X)\Big)$

源路由终点不使用明文标量约束。令 $z_{last,N}=\omega_N^{n_N-1}$、$z_{last,edge}=\omega_{edge}^{n_{edge}-1}$，验证关系为

$$P_{R_{src}^{node(r)}}(z_{last,N})-P_{R_{src}^{edge(r)}}(z_{last,edge})=0,$$

并由 $\mathbf S_{route}$ 中对应的 KZG 同点平移归一化关系验证。

##### （二）LeakyReLU 约束

$C_{L,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_L^{(r)}}(X)$

$$\begin{aligned} C_{L,1}^{(r)}(X) ={}&\big(1-Last_{edge}(X)\big)\Big[ \big(P_{R_L^{(r)}}(\omega_{edge}X)-P_{R_L^{(r)}}(X)\big) \big(P_{Table^{L(r)}}(X)+\beta_L^{(r)}\big) \big(P_{Query^{L(r)}}(X)+\beta_L^{(r)}\big)\\ &- P_{Q_{tbl}^{L(r)}}(X)P_{m_L^{(r)}}(X)\big(P_{Query^{L(r)}}(X)+\beta_L^{(r)}\big)\\ &+ P_{Q_{qry}^{L(r)}}(X)\big(P_{Table^{L(r)}}(X)+\beta_L^{(r)}\big) \Big]\end{aligned}$$

$C_{L,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_L^{(r)}}(X)$

$C_{L,tbl}^{(r)}(X)=P_{Table^{L(r)}}(X)-\big(P_{T_{LReLU},x}(X)+\eta_L^{(r)}P_{T_{LReLU},y}(X)\big)$

$C_{L,qry}^{(r)}(X)=P_{Query^{L(r)}}(X)-\big(P_{S^{(r)}}(X)+\eta_L^{(r)}P_{Z^{(r)}}(X)\big)$

线性打分必须由两个已绑定广播列唯一确定：

$$C_{score}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{S^{(r)}}(X)-P_{E_{src}^{edge(r)}}(X)-P_{E_{dst}^{edge(r)}}(X)\Big).$$

##### （三）最大值唯一性约束

$C_{max,bin}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(r)}}(X)\big(P_{s_{max}^{(r)}}(X)-1\big)$

$C_{max,zero}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(r)}}(X)P_{\Delta^{+(r)}}(X)$

$C_{max,0}^{(r)}(X)=First_{edge}(X)\cdot\big(P_{C_{max}^{(r)}}(X)-P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(r)}}(X)\big)$

$$\begin{aligned} C_{max,1}^{(r)}(X) ={}&\big(1-Last_{edge}(X)\big)\Big[ P_{C_{max}^{(r)}}(\omega_{edge}X)\\ &- \Big( (1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{C_{max}^{(r)}}(X) + P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_{s_{max}^{(r)}}(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ (1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X))(P_{C_{max}^{(r)}}(X)+P_{s_{max}^{(r)}}(\omega_{edge}X)) \big) \Big) \Big]\end{aligned}$$

$C_{max,end}^{(r)}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{C_{max}^{(r)}}(X)-1\big)$

最大值广播与非负差分必须满足：

$$C_{delta}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{\Delta^{+(r)}}(X)-P_{M^{edge(r)}}(X)+P_{Z^{(r)}}(X)\Big).$$

##### （四）范围检查约束

$C_{R,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_R^{(r)}}(X)$

$$\begin{aligned} C_{R,1}^{(r)}(X) ={}&\big(1-Last_{edge}(X)\big)\Big[ \big(P_{R_R^{(r)}}(\omega_{edge}X)-P_{R_R^{(r)}}(X)\big) \big(P_{Table^{R(r)}}(X)+\beta_R^{(r)}\big) \big(P_{Query^{R(r)}}(X)+\beta_R^{(r)}\big)\\ &- P_{Q_{tbl}^{R(r)}}(X)P_{m_R^{(r)}}(X)\big(P_{Query^{R(r)}}(X)+\beta_R^{(r)}\big)\\ &+ P_{Q_{qry}^{R(r)}}(X)\big(P_{Table^{R(r)}}(X)+\beta_R^{(r)}\big) \Big]\end{aligned}$$

$C_{R,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_R^{(r)}}(X)$

$C_{R,tbl}^{(r)}(X)=P_{Table^{R(r)}}(X)-P_{T_{range}}(X)$

$C_{R,qry}^{(r)}(X)=P_{Query^{R(r)}}(X)-P_{\Delta^{+(r)}}(X)$

##### （五）指数查表约束

$C_{exp,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{exp}^{(r)}}(X)$

$$\begin{aligned} C_{exp,1}^{(r)}(X) ={}&\big(1-Last_{edge}(X)\big)\Big[ \big(P_{R_{exp}^{(r)}}(\omega_{edge}X)-P_{R_{exp}^{(r)}}(X)\big) \big(P_{Table^{exp(r)}}(X)+\beta_{exp}^{(r)}\big) \big(P_{Query^{exp(r)}}(X)+\beta_{exp}^{(r)}\big)\\ &- P_{Q_{tbl}^{exp(r)}}(X)P_{m_{exp}^{(r)}}(X)\big(P_{Query^{exp(r)}}(X)+\beta_{exp}^{(r)}\big)\\ &+ P_{Q_{qry}^{exp(r)}}(X)\big(P_{Table^{exp(r)}}(X)+\beta_{exp}^{(r)}\big) \Big]\end{aligned}$$

$C_{exp,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_{exp}^{(r)}}(X)$

$C_{exp,tbl}^{(r)}(X)=P_{Table^{exp(r)}}(X)-\big(P_{T_{exp},x}(X)+\eta_{exp}^{(r)}P_{T_{exp},y}(X)\big)$

$C_{exp,qry}^{(r)}(X)=P_{Query^{exp(r)}}(X)-\big(P_{\Delta^{+(r)}}(X)+\eta_{exp}^{(r)}P_{U^{(r)}}(X)\big)$

**有限域归一化权重约束**

$$C_{alpha}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{\alpha^{(r)}}(X)-P_{U^{(r)}}(X)P_{inv^{edge(r)}}(X)\Big).$$

聚合前 PSQ 的源值、压缩权值和边域目标值必须由其来源列唯一确定：

$$C_{psq,mul}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{\widehat v_{pre}^{\star(r)}}(X)-P_{\alpha^{(r)}}(X)P_{H_{src}^{\star,edge(r)}}(X)\Big),$$

$$C_{psq,w}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{w_{psq}^{(r)}}(X)-P_{U^{(r)}}(X)-\lambda^{(r)}P_{\widehat v_{pre}^{\star(r)}}(X)\Big),$$

$$C_{psq,Tedge}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{T_{psq}^{edge(r)}}(X)-P_{Sum^{edge(r)}}(X)-\lambda^{(r)}P_{H_{agg,pre}^{\star,edge(r)}}(X)\Big).$$

##### （六）聚合前 PSQ 约束

$C_{psq,0}^{(r)}(X)=First_{edge}(X)\cdot\big(P_{PSQ^{(r)}}(X)-P_{Q_{edge}^{valid}}(X)P_{w_{psq}^{(r)}}(X)\big)$

$$\begin{aligned} C_{psq,1}^{(r)}(X) ={}&\big(1-Last_{edge}(X)\big)\Big[ P_{PSQ^{(r)}}(\omega_{edge}X)\\ &- \Big( (1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{PSQ^{(r)}}(X) + P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_{w_{psq}^{(r)}}(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ (1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X))(P_{PSQ^{(r)}}(X)+P_{w_{psq}^{(r)}}(\omega_{edge}X)) \big) \Big) \Big]\end{aligned}$$

$C_{psq,end}^{(r)}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{PSQ^{(r)}}(X)-P_{T_{psq}^{edge(r)}}(X)\big)$

##### （七）ELU 约束

$C_{ELU,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{ELU}^{(r)}}(X)$

$$\begin{aligned} C_{ELU,1}^{(r)}(X) ={}&\big(1-Last_{edge}(X)\big)\Big[ \big(P_{R_{ELU}^{(r)}}(\omega_{edge}X)-P_{R_{ELU}^{(r)}}(X)\big) \big(P_{Table^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big) \big(P_{Query^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big)\\ &- P_{Q_{tbl}^{ELU(r)}}(X)P_{m_{ELU}^{(r)}}(X)\big(P_{Query^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big)\\ &+ P_{Q_{qry}^{ELU(r)}}(X)\big(P_{Table^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big) \Big]\end{aligned}$$

$C_{ELU,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_{ELU}^{(r)}}(X)$

$C_{ELU,tbl}^{(r)}(X)=P_{Table^{ELU(r)}}(X)-\big(P_{T_{ELU},x}(X)+\eta_{ELU}^{(r)}P_{T_{ELU},y}(X)\big)$

$C_{ELU,qry}^{(r)}(X)=P_{Query^{ELU(r)}}(X)-\big(P_{H_{ELU,in,flat}^{(r)}}(X)+\eta_{ELU}^{(r)}P_{H_{agg,flat}^{(r)}}(X)\big)$

##### （八）目标路由边端约束

$C_{dst,edge,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{dst}^{edge(r)}}(X)$

$C_{dst,edge,1}^{(r)}(X) = \big(1-Last_{edge}(X)\big)\Big(\big(P_{R_{dst}^{edge(r)}}(\omega_{edge}X)-P_{R_{dst}^{edge(r)}}(X)\big)\big(P_{Query^{dst(r)}}(X)+\beta_{dst}^{(r)}\big)-P_{Q_{edge}^{valid}}(X)\Big)$

目标路由终点不使用明文标量约束。验证关系为

$$P_{R_{dst}^{node(r)}}(z_{last,N})-P_{R_{dst}^{edge(r)}}(z_{last,edge})=0,$$

并由 $\mathbf S_{route}$ 中对应的 KZG 同点平移归一化关系验证。

##### （九）边域绑定约束

$C_{src,qry}^{(r)}(X)=P_{Query^{src(r)}}(X)-P_{Q_{edge}^{valid}}(X)\Big(P_{src}(X)+\eta_{src}^{(r)}P_{E_{src}^{edge(r)}}(X)+(\eta_{src}^{(r)})^2P_{H_{src}^{\star,edge(r)}}(X)\Big)$

$C_{dst,qry}^{(r)}(X)=P_{Query^{dst(r)}}(X)-P_{Q_{edge}^{valid}}(X)\Big( P_{dst}(X)+\eta_{dst}^{(r)}P_{E_{dst}^{edge(r)}}(X)+(\eta_{dst}^{(r)})^2P_{M^{edge(r)}}(X)+(\eta_{dst}^{(r)})^3P_{Sum^{edge(r)}}(X)+(\eta_{dst}^{(r)})^4P_{inv^{edge(r)}}(X)+(\eta_{dst}^{(r)})^5P_{H_{agg,pre}^{\star,edge(r)}}(X)+(\eta_{dst}^{(r)})^6P_{H_{agg}^{\star,edge(r)}}(X) \Big)$

**隐藏层聚合定点输入约束**

定义：

$$C_{agg,num}^{(r)}(X)=P_{Q_{Nd_h}^{valid,(\ell)}}(X)\big(P_{H_{agg,num,flat}^{(r)}}(X)-P_{Sum_{flat}^{(r)}}(X)P_{H_{agg,pre,flat}^{(r)}}(X)\big).$$

对除法实例 $\nu=agg,(\ell,r)$，取：

$$P_{x_\nu}=S_{agg,pre}P_{H_{agg,num,flat}^{(r)}},\qquad
P_{d_\nu}=S_{H'}^{(\ell,r)}P_{Sum_{flat}^{(r)}},\qquad
P_{y_\nu}=P_{H_{ELU,in,flat}^{(r)}}.$$

使用第 3.3.6 节后给出的完整 $\mathcal C_{div,agg}^{(\ell,r)}$。$P_{Sum_{flat}^{(r)}}$ 与 $P_{H_{agg,pre,flat}^{(r)}}$ 的 indexed-view 和 basis-link 约束由 $\pi_{bind}^{hidden,(\ell,r)}$ 验证。

#### 3.3.9 单个隐藏层模板中第 $r$ 个注意力头在节点域上的全部约束

以下约束对应某个固定隐藏层 $\ell$ 的模板；对每个 $r\in\{0,1,\ldots,K_{hid}-1\}$，在节点域 $\mathbb H_N$ 上定义：

1. 源路由节点端三条约束：

	$C_{src,node,0}^{(r)}(X)=First_N(X)\cdot P_{R_{src}^{node(r)}}(X)$

	$C_{src,node,1}^{(r)}(X)=\big(1-Last_N(X)\big)\Big(\big(P_{R_{src}^{node(r)}}(\omega_NX)-P_{R_{src}^{node(r)}}(X)\big)\big(P_{Table^{src(r)}}(X)+\beta_{src}^{(r)}\big)-P_{Q_N}(X)P_{m_{src}^{(r)}}(X)\Big)$

	源路由节点端与边端的终点相等关系由第 3.5.3 节的 KZG 同点平移归一化关系验证，不加入节点域 quotient。

2. 目标路由节点端三条约束：

	$C_{dst,node,0}^{(r)}(X)=First_N(X)\cdot P_{R_{dst}^{node(r)}}(X)$

	$C_{dst,node,1}^{(r)}(X)=\big(1-Last_N(X)\big)\Big(\big(P_{R_{dst}^{node(r)}}(\omega_NX)-P_{R_{dst}^{node(r)}}(X)\big)\big(P_{Table^{dst(r)}}(X)+\beta_{dst}^{(r)}\big)-P_{Q_N}(X)P_{m_{dst}^{(r)}}(X)\Big)$

	目标路由节点端与边端的终点相等关系由第 3.5.3 节的 KZG 同点平移归一化关系验证，不加入节点域 quotient。

3. 分母逆元约束：$C_{inv}^{(r)}(X)=P_{Q_N}(X)\cdot\big(P_{Sum^{(r)}}(X)P_{inv^{(r)}}(X)-1\big)$

4. 源路由表端绑定：$C_{src,tbl}^{(r)}(X)=P_{Table^{src(r)}}(X)-P_{Q_N}(X)\Big(P_{Idx_N}(X)+\eta_{src}^{(r)}P_{E_{src}^{(r)}}(X)+(\eta_{src}^{(r)})^2P_{H^{\star(r)}}(X)\Big)$

5. 目标路由表端绑定：$$\begin{aligned} C_{dst,tbl}^{(r)}(X) ={}& P_{Table^{dst(r)}}(X) - P_{Q_N}(X)\Big( P_{Idx_N}(X) + \eta_{dst}^{(r)}P_{E_{dst}^{(r)}}(X)\\ &\qquad +(\eta_{dst}^{(r)})^2P_{M^{(r)}}(X) +(\eta_{dst}^{(r)})^3P_{Sum^{(r)}}(X) +(\eta_{dst}^{(r)})^4P_{inv^{(r)}}(X) +(\eta_{dst}^{(r)})^5P_{H_{agg,pre}^{\star(r)}}(X) +(\eta_{dst}^{(r)})^6P_{H_{agg}^{\star(r)}}(X) \Big) \end{aligned}$$

6. PSQ 节点端目标值绑定：

$$C_{psq,Tnode}^{(r)}(X)=P_{Q_N}(X)\Big(P_{T_{psq}^{(r)}}(X)-P_{Sum^{(r)}}(X)-\lambda^{(r)}P_{H_{agg,pre}^{\star(r)}}(X)\Big).$$


#### 3.3.10 第 $r$ 个隐藏层注意力头在共享维域上的全部绑定约束

##### （一）输入共享维域 $\mathbb H_{in}$

$C_{proj,0}^{(r)}(X)=First_{in}(X)\cdot P_{Acc^{proj(r)}}(X)$

$C_{proj,1}^{(r)}(X)=\big(1-Last_{in}(X)\big)\Big(P_{Acc^{proj(r)}}(\omega_{in}X)-P_{Acc^{proj(r)}}(X)-P_{a^{proj(r)}}(X)P_{b^{proj(r)}}(X)\Big)$

投影累加器的终点与 $P_{H'^{(r)}}(y_{proj}^{(r)})$ 满足

$$P_{Acc^{proj(r)}}(z_{last,in})-P_{H'^{(r)}}(y_{proj}^{(r)})=0,$$

并由 $\pi_{bind}^{hidden,(\ell,r)}$ 的 KZG 同点平移归一化关系验证；该关系不引入裸 $\mu_{proj}^{(r)}$。

##### （二）隐藏层单头共享维域 $\mathbb H_{d_h}$

源注意力绑定：

$C_{srcbind,0}^{(r)}(X)=First_{d_h}(X)\cdot P_{Acc^{src(r)}}(X)$

$C_{srcbind,1}^{(r)}(X)=\big(1-Last_{d_h}(X)\big)\Big(P_{Acc^{src(r)}}(\omega_{d_h}X)-P_{Acc^{src(r)}}(X)-P_{a^{src(r)}}(X)P_{b^{src(r)}}(X)\Big)$

源注意力累加器的终点与 $P_{E_{src}^{(r)}}(y_{src}^{(r)})$ 通过 KZG 同点平移归一化关系绑定，不引入裸 $\mu_{src}^{(r)}$。

目标注意力绑定：

$C_{dstbind,0}^{(r)}(X)=First_{d_h}(X)\cdot P_{Acc^{dst(r)}}(X)$

$C_{dstbind,1}^{(r)}(X)=\big(1-Last_{d_h}(X)\big)\Big(P_{Acc^{dst(r)}}(\omega_{d_h}X)-P_{Acc^{dst(r)}}(X)-P_{a^{dst(r)}}(X)P_{b^{dst(r)}}(X)\Big)$

目标注意力累加器的终点与 $P_{E_{dst}^{(r)}}(y_{dst}^{(r)})$ 通过 KZG 同点平移归一化关系绑定，不引入裸 $\mu_{dst}^{(r)}$。

压缩特征绑定：

$C_{starbind,0}^{(r)}(X)=First_{d_h}(X)\cdot P_{Acc^{\star(r)}}(X)$

$C_{starbind,1}^{(r)}(X)=\big(1-Last_{d_h}(X)\big)\Big(P_{Acc^{\star(r)}}(\omega_{d_h}X)-P_{Acc^{\star(r)}}(X)-P_{a^{\star(r)}}(X)P_{b^{\star(r)}}(X)\Big)$

压缩累加器的终点与 $P_{H^{\star(r)}}(y_{\star}^{(r)})$ 通过 KZG 同点平移归一化关系绑定，不引入裸 $\mu_{\star}^{(r)}$。

先对每个隐藏维索引 $j\in\{0,1,\ldots,d_h-1\}$ 定义节点域列多项式：

$P_{A_{agg,pre,j}^{(r)}}(X)=\sum_{i=0}^{n_N-1}H_{agg,pre,i,j}^{(r)}L_i^{(N)}(X)$

$P_{A_{agg,j}^{(r)}}(X)=\sum_{i=0}^{n_N-1}H_{agg,i,j}^{(r)}L_i^{(N)}(X)$

聚合前压缩绑定：$C_{aggpre}^{(r)}(X)=P_{H_{agg,pre}^{\star(r)}}(X)-\sum_{j=0}^{d_h-1}P_{A_{agg,pre,j}^{(r)}}(X)(\xi^{(r)})^j$

聚合后压缩绑定：$C_{agg}^{(r)}(X)=P_{H_{agg}^{\star(r)}}(X)-\sum_{j=0}^{d_h-1}P_{A_{agg,j}^{(r)}}(X)(\xi^{(r)})^j$

上述两个等式的全部多项式都原生定义在 $\mathbb H_N$，因此它们登记为 $\mathcal R_{bind,N}$ 的成员并只进入 $\mathcal C_{bind,N}$；它们不属于 $\mathbb H_{d_h}$ 商。$\mathbb H_{d_h}$ 仅承载 $a,b,Acc$ 的共享维递推。

#### 3.3.11 拼接共享维域约束

本节同时列出拼接阶段的节点域压缩关系和最终输出层投影的共享维递推。拼接压缩与逐列拼接等式属于 $\mathbb H_N$ 的 $\mathcal C_{bind,N}$；$\mathbb H_{cat}$ 商只承载最终输出投影的共享维 $a,b,Acc$ 递推。

对任意隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，其拼接压缩绑定写为：

$C_{cat}^{(\ell)}(X)=P_{H_{cat}^{\star(\ell)}}(X)-\sum_{m=0}^{d_{cat}^{(\ell)}-1}P_{A_{cat,m}^{(\ell)}}(X)(\xi_{cat}^{(\ell)})^m$

对每个 $r\in\{0,\ldots,K_{hid}^{(\ell)}-1\}$ 和 $j\in\{0,\ldots,d_h^{(\ell)}-1\}$，逐列拼接约束固定为：

$$C_{concat}^{(\ell,r,j)}(X)=P_{Q_N}(X)\Big(P_{H_{cat}^{(\ell)}[:,r d_h^{(\ell)}+j]}^{can}(X)-P_{H_{agg}^{(\ell,r)}[:,j]}^{can}(X)\Big).$$

其中
$P_{A_{cat,m}^{(\ell)}}(X)=\sum_{i=0}^{n_N-1}H_{cat,i,m}^{(\ell)}L_i^{(N)}(X)$

该拼接压缩等式原生定义在 $\mathbb H_N$，登记为 $\mathcal R_{bind,N}$ 的成员并只进入 $\mathcal C_{bind,N}$。$\mathbb H_{cat}$ 商仅承载输出投影共享维的 $a,b,Acc$ 递推；row-major coefficient commitment $P_{H_{cat}}$ 不得作为 $\mathbb H_{cat}$ evaluation trace 打开。

对每个输出头 $s\in\{0,1,\ldots,K_{out}-1\}$，输出投影的共享维绑定显式定义为：


$C_{outproj,0}^{(s)}(X)=First_{cat}(X)\cdot P_{Acc^{proj(out,s)}}(X)$


$C_{outproj,1}^{(s)}(X)=\big(1-Last_{cat}(X)\big)\Big(P_{Acc^{proj(out,s)}}(\omega_{cat}X)-P_{Acc^{proj(out,s)}}(X)-P_{a^{proj(out,s)}}(X)P_{b^{proj(out,s)}}(X)\Big)$


输出投影累加器的终点与 $P_{Y'^{(s)}}(y_{proj}^{(out,s)})$ 通过 $\pi_{bind}^{out,s}$ 的 KZG 同点平移归一化关系绑定，不引入裸 $\mu_{proj}^{(out,s)}$。

#### 3.3.12 输出层类别共享维域约束

对每个输出头 $s$，源 / 目标注意力共享维递推定义在 $\mathbb H_C$；随后列出的输出矩阵压缩关系定义在 $\mathbb H_N$ 并进入 $\mathcal C_{bind,N}$。具体定义如下：

##### （一）输出层源注意力绑定


$C_{outsrc,0}^{(s)}(X)=First_C(X)\cdot P_{Acc^{src(out,s)}}(X)$


$C_{outsrc,1}^{(s)}(X)=\big(1-Last_C(X)\big)\Big(P_{Acc^{src(out,s)}}(\omega_CX)-P_{Acc^{src(out,s)}}(X)-P_{a^{src(out,s)}}(X)P_{b^{src(out,s)}}(X)\Big)$


输出源注意力累加器的终点与 $P_{E_{src}^{(out,s)}}(y_{src}^{(out,s)})$ 通过 KZG 同点平移归一化关系绑定，不引入裸 $\mu_{src}^{(out,s)}$。

##### （二）输出层目标注意力绑定


$C_{outdst,0}^{(s)}(X)=First_C(X)\cdot P_{Acc^{dst(out,s)}}(X)$


$C_{outdst,1}^{(s)}(X)=\big(1-Last_C(X)\big)\Big(P_{Acc^{dst(out,s)}}(\omega_CX)-P_{Acc^{dst(out,s)}}(X)-P_{a^{dst(out,s)}}(X)P_{b^{dst(out,s)}}(X)\Big)$


输出目标注意力累加器的终点与 $P_{E_{dst}^{(out,s)}}(y_{dst}^{(out,s)})$ 通过 KZG 同点平移归一化关系绑定，不引入裸 $\mu_{dst}^{(out,s)}$。

##### （三）尺度对齐后聚合前输出压缩绑定

输出投影与有限域聚合前输出的类别压缩先分别定义为：

$$C_{outY'}^{(s)}(X)=P_{Y'^{\star(s)}}(X)-\sum_{c=0}^{C-1}P_{Y'^{(s)}[:,c]}^{can}(X)\xi_{out}^c,$$

$$C_{outYpre}^{(s)}(X)=P_{Y^{pre,\star(s)}}(X)-\sum_{c=0}^{C-1}P_{Y^{pre(s)}[:,c]}^{can}(X)\xi_{out}^c.$$

两式原生定义在 $\mathbb H_N$，登记到 $\mathcal R_{bind,N}$。其中 $Y'^{\star,edge(s)}$ 与 $Y^{pre,\star,edge(s)}$ 分别通过 source / target indexed-view 绑定到对应节点压缩列。


先对每个类别索引 $c$ 定义节点域列多项式：$P_{A_{\widetilde Y,c}^{(s)}}(X)=\sum_{i=0}^{n_N-1}\widetilde Y_{i,c}^{pre(s)}L_i^{(N)}(X)$

则有：$C_{out\widetilde Y}^{(s)}(X)=P_{\widetilde Y^{pre,\star(s)}}(X)-\sum_{c=0}^{C-1}P_{A_{\widetilde Y,c}^{(s)}}(X)\xi_{out}^c$

##### （四）输出头压缩绑定

先对每个类别索引 $c$ 定义节点域列多项式：$P_{A_{Y,c}^{(s)}}(X)=\sum_{i=0}^{n_N-1}Y_{i,c}^{(s)}L_i^{(N)}(X)$

则有：$C_{outY}^{(s)}(X)=P_{Y^{\star(s)}}(X)-\sum_{c=0}^{C-1}P_{A_{Y,c}^{(s)}}(X)\xi_{out}^c$

##### （五）最终平均输出压缩绑定

先对每个类别索引 $c$ 定义：$P_{A_{Y,c}}(X)=\sum_{i=0}^{n_N-1}Y_{i,c}L_i^{(N)}(X)$

则有：$C_{outY}^{avg}(X)=P_{Y^{\star}}(X)-\sum_{c=0}^{C-1}P_{A_{Y,c}}(X)\xi_{out}^c$

$C_{outY'}^{(s)},C_{outYpre}^{(s)},C_{out\widetilde Y}^{(s)},C_{outY}^{(s)},C_{outY}^{avg}$ 的全部多项式都原生定义在 $\mathbb H_N$，因此统一登记为 $\mathcal R_{bind,N}$ 的成员并只进入 $\mathcal C_{bind,N}$。$\mathbb H_C$ 商只承载输出注意力向量点积的 $a,b,Acc$ 递推。

#### 3.3.13 输出层在边域上的全部约束

在边域 $\mathbb H_{edge}$ 上，对每个输出头 $s$，显式定义如下约束。

##### （一）源路由边端约束


$C_{out,src,edge,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_{src}^{edge(out,s)}}(X)$


$C_{out,src,edge,1}^{(s)}(X)=\big(1-Last_{edge}(X)\big)\Big(\big(P_{R_{src}^{edge(out,s)}}(\omega_{edge}X)-P_{R_{src}^{edge(out,s)}}(X)\big)\big(P_{Query^{src(out,s)}}(X)+\beta_{src}^{(out,s)}\big)-P_{Q_{edge}^{valid}}(X)\Big)$


输出头源路由终点关系为

$$P_{R_{src}^{node(out,s)}}(z_{last,N})-P_{R_{src}^{edge(out,s)}}(z_{last,edge})=0,$$

并由 $\mathbf S_{route}$ 中对应的 KZG 同点平移归一化关系验证。

##### （二）LeakyReLU 约束


$C_{out,L,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_L^{(out,s)}}(X)$


$$\begin{aligned}
C_{out,L,1}^{(s)}(X)={}&\big(1-Last_{edge}(X)\big)\Big[\big(P_{R_L^{(out,s)}}(\omega_{edge}X)-P_{R_L^{(out,s)}}(X)\big)
\big(P_{Table^{L(out,s)}}(X)+\beta_L^{(out,s)}\big)
\big(P_{Query^{L(out,s)}}(X)+\beta_L^{(out,s)}\big)\\
&-P_{Q_{tbl}^{L(out,s)}}(X)P_{m_L^{(out,s)}}(X)\big(P_{Query^{L(out,s)}}(X)+\beta_L^{(out,s)}\big)\\
&+P_{Q_{qry}^{L(out,s)}}(X)\big(P_{Table^{L(out,s)}}(X)+\beta_L^{(out,s)}\big)
\Big]\end{aligned}$$


$C_{out,L,2}^{(s)}(X)=Last_{edge}(X)\cdot P_{R_L^{(out,s)}}(X)$


$C_{out,L,tbl}^{(s)}(X)=P_{Table^{L(out,s)}}(X)-\big(P_{T_{LReLU},x}(X)+\eta_L^{(out,s)}P_{T_{LReLU},y}(X)\big)$


$C_{out,L,qry}^{(s)}(X)=P_{Query^{L(out,s)}}(X)-\big(P_{S^{(out,s)}}(X)+\eta_L^{(out,s)}P_{Z^{(out,s)}}(X)\big)$

$$C_{out,score}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{S^{(out,s)}}(X)-P_{E_{src}^{edge(out,s)}}(X)-P_{E_{dst}^{edge(out,s)}}(X)\Big).$$

##### （三）最大值唯一性约束


$C_{out,max,bin}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(out,s)}}(X)\big(P_{s_{max}^{(out,s)}}(X)-1\big)$


$C_{out,max,zero}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(out,s)}}(X)P_{\Delta^{+(out,s)}}(X)$


$C_{out,max,0}^{(s)}(X)=First_{edge}(X)\cdot\big(P_{C_{max}^{(out,s)}}(X)-P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(out,s)}}(X)\big)$


$$\begin{aligned}
C_{out,max,1}^{(s)}(X)={}&\big(1-Last_{edge}(X)\big)\Big[P_{C_{max}^{(out,s)}}(\omega_{edge}X)\\
&-\Big((1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{C_{max}^{(out,s)}}(X)\\
&\qquad+P_{Q_{edge}^{valid}}(\omega_{edge}X)\big(P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_{s_{max}^{(out,s)}}(\omega_{edge}X)\\
&\qquad\qquad+(1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X))(P_{C_{max}^{(out,s)}}(X)+P_{s_{max}^{(out,s)}}(\omega_{edge}X))\big)\Big)
\Big]\end{aligned}$$


$C_{out,max,end}^{(s)}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{C_{max}^{(out,s)}}(X)-1\big)$

$$C_{out,delta}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{\Delta^{+(out,s)}}(X)-P_{M^{edge(out,s)}}(X)+P_{Z^{(out,s)}}(X)\Big).$$

##### （四）范围检查约束


$C_{out,R,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_R^{(out,s)}}(X)$


$$\begin{aligned}
C_{out,R,1}^{(s)}(X)={}&\big(1-Last_{edge}(X)\big)\Big[\big(P_{R_R^{(out,s)}}(\omega_{edge}X)-P_{R_R^{(out,s)}}(X)\big)
\big(P_{Table^{R(out,s)}}(X)+\beta_R^{(out,s)}\big)
\big(P_{Query^{R(out,s)}}(X)+\beta_R^{(out,s)}\big)\\
&-P_{Q_{tbl}^{R(out,s)}}(X)P_{m_R^{(out,s)}}(X)\big(P_{Query^{R(out,s)}}(X)+\beta_R^{(out,s)}\big)\\
&+P_{Q_{qry}^{R(out,s)}}(X)\big(P_{Table^{R(out,s)}}(X)+\beta_R^{(out,s)}\big)
\Big]\end{aligned}$$


$C_{out,R,2}^{(s)}(X)=Last_{edge}(X)\cdot P_{R_R^{(out,s)}}(X)$


$C_{out,R,tbl}^{(s)}(X)=P_{Table^{R(out,s)}}(X)-P_{T_{range}}(X)$


$C_{out,R,qry}^{(s)}(X)=P_{Query^{R(out,s)}}(X)-P_{\Delta^{+(out,s)}}(X)$

##### （五）指数查表约束


$C_{out,exp,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_{exp}^{(out,s)}}(X)$


$$\begin{aligned}
C_{out,exp,1}^{(s)}(X)={}&\big(1-Last_{edge}(X)\big)\Big[\big(P_{R_{exp}^{(out,s)}}(\omega_{edge}X)-P_{R_{exp}^{(out,s)}}(X)\big)
\big(P_{Table^{exp(out,s)}}(X)+\beta_{exp}^{(out,s)}\big)
\big(P_{Query^{exp(out,s)}}(X)+\beta_{exp}^{(out,s)}\big)\\
&-P_{Q_{tbl}^{exp(out,s)}}(X)P_{m_{exp}^{(out,s)}}(X)\big(P_{Query^{exp(out,s)}}(X)+\beta_{exp}^{(out,s)}\big)\\
&+P_{Q_{qry}^{exp(out,s)}}(X)\big(P_{Table^{exp(out,s)}}(X)+\beta_{exp}^{(out,s)}\big)
\Big]\end{aligned}$$


$C_{out,exp,2}^{(s)}(X)=Last_{edge}(X)\cdot P_{R_{exp}^{(out,s)}}(X)$


$C_{out,exp,tbl}^{(s)}(X)=P_{Table^{exp(out,s)}}(X)-\big(P_{T_{exp},x}(X)+\eta_{exp}^{(out,s)}P_{T_{exp},y}(X)\big)$


$C_{out,exp,qry}^{(s)}(X)=P_{Query^{exp(out,s)}}(X)-\big(P_{\Delta^{+(out,s)}}(X)+\eta_{exp}^{(out,s)}P_{U^{(out,s)}}(X)\big)$

**有限域归一化权重约束**

$$C_{out,alpha}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{\alpha^{(out,s)}}(X)-P_{U^{(out,s)}}(X)P_{inv^{edge(out,s)}}(X)\Big).$$

$$C_{out,psq,mul}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{\widehat y^{\star(s)}}(X)-P_{\alpha^{(out,s)}}(X)P_{Y'^{\star,edge(s)}}(X)\Big),$$

$$C_{out,psq,w}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{w_{out}^{(s)}}(X)-P_{U^{(out,s)}}(X)-\lambda_{out}^{(s)}P_{\widehat y^{\star(s)}}(X)\Big),$$

$$C_{out,psq,Tedge}^{(s)}(X)=P_{Q_{edge}^{valid}}(X)\Big(P_{T_{out}^{pre,edge(s)}}(X)-P_{Sum^{edge(out,s)}}(X)-\lambda_{out}^{(s)}P_{Y^{pre,\star,edge(s)}}(X)\Big).$$

##### （六）聚合前输出 PSQ 约束


$C_{out,psq,0}^{(s)}(X)=First_{edge}(X)\cdot\big(P_{PSQ^{(out,s)}}(X)-P_{Q_{edge}^{valid}}(X)P_{w_{out}^{(s)}}(X)\big)$


$$\begin{aligned}
C_{out,psq,1}^{(s)}(X)={}&\big(1-Last_{edge}(X)\big)\Big[P_{PSQ^{(out,s)}}(\omega_{edge}X)\\
&-\Big((1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{PSQ^{(out,s)}}(X)\\
&\qquad+P_{Q_{edge}^{valid}}(\omega_{edge}X)\big(P_{Q_{grp\_new}^{edge}}(\omega_{edge}X)P_{w_{out}^{(s)}}(\omega_{edge}X)\\
&\qquad\qquad+(1-P_{Q_{grp\_new}^{edge}}(\omega_{edge}X))(P_{PSQ^{(out,s)}}(X)+P_{w_{out}^{(s)}}(\omega_{edge}X))\big)\Big)
\Big]\end{aligned}$$


$C_{out,psq,end}^{(s)}(X)=P_{Q_{grp\_end}^{edge}}(X)\cdot\big(P_{PSQ^{(out,s)}}(X)-P_{T_{out}^{pre,edge(s)}}(X)\big)$

##### （七）目标路由边端约束


$C_{out,dst,edge,0}^{(s)}(X)=First_{edge}(X)\cdot P_{R_{dst}^{edge(out,s)}}(X)$


$C_{out,dst,edge,1}^{(s)}(X)=\big(1-Last_{edge}(X)\big)\Big(\big(P_{R_{dst}^{edge(out,s)}}(\omega_{edge}X)-P_{R_{dst}^{edge(out,s)}}(X)\big)\big(P_{Query^{dst(out,s)}}(X)+\beta_{dst}^{(out,s)}\big)-P_{Q_{edge}^{valid}}(X)\Big)$


输出头目标路由终点关系为

$$P_{R_{dst}^{node(out,s)}}(z_{last,N})-P_{R_{dst}^{edge(out,s)}}(z_{last,edge})=0,$$

并由 $\mathbf S_{route}$ 中对应的 KZG 同点平移归一化关系验证。

##### （八）边域绑定约束


$C_{out,src,qry}^{(s)}(X)=P_{Query^{src(out,s)}}(X)-P_{Q_{edge}^{valid}}(X)\Big(P_{src}(X)+\eta_{src}^{(out,s)}P_{E_{src}^{edge(out,s)}}(X)+(\eta_{src}^{(out,s)})^2P_{Y'^{\star,edge(s)}}(X)\Big)$

$$\begin{aligned}
C_{out,dst,qry}^{(s)}(X) ={} & P_{Query^{dst(out,s)}}(X) - P_{Q_{edge}^{valid}}(X)\Big( P_{dst}(X) + \eta_{dst}^{(out,s)}P_{E_{dst}^{edge(out,s)}}(X) \\
& + (\eta_{dst}^{(out,s)})^2 P_{M^{edge(out,s)}}(X) + (\eta_{dst}^{(out,s)})^3 P_{Sum^{edge(out,s)}}(X) \\
& + (\eta_{dst}^{(out,s)})^4 P_{inv^{edge(out,s)}}(X) + (\eta_{dst}^{(out,s)})^5 P_{Y^{pre,\star,edge(s)}}(X) + (\eta_{dst}^{(out,s)})^6 P_{Y^{\star,edge(s)}}(X) \Big)
\end{aligned}$$


**输出聚合回缩、尺度对齐与偏置约束**

定义：

$$C_{out,num}^{(s)}(X)=P_{Q_{NC}^{valid}}(X)\big(P_{Y_{agg,num,flat}^{(s)}}(X)-P_{Sum_{flat}^{(out,s)}}(X)P_{Y_{pre,flat}^{(s)}}(X)\big).$$

聚合回缩除法实例 $\nu=outq,s$ 取：

$$P_{x_\nu}=S_{out,pre}P_{Y_{agg,num,flat}^{(s)}},\qquad
P_{d_\nu}=S_{Y'}^{(s)}P_{Sum_{flat}^{(out,s)}},\qquad
P_{y_\nu}=P_{Y_{q,flat}^{(s)}}.$$

尺度对齐除法实例 $\nu=align,s$ 取：

$$P_{x_\nu}=S_{out,head}P_{Y_{q,flat}^{(s)}},\qquad
P_{d_\nu}=S_{out,pre},\qquad
P_{y_\nu}=P_{\widetilde Y_{flat}^{pre(s)}}.$$

分别使用 $\mathcal C_{div,outq}^{(s)}$ 与 $\mathcal C_{div,align}^{(s)}$。私有偏置关系为：

$$C_{out,bias}^{(s)}(X)=P_{Q_{NC}^{valid}}(X)\big(P_{Y_{flat}^{(s)}}(X)-P_{\widetilde Y_{flat}^{pre(s)}}(X)-P_{b_{flat}^{(out,s)}}(X)\big).$$

$P_{b_{flat}^{(out,s)}}$ 与 proof-local 偏置 canonical commitment 的 indexed-view 关系由 $\pi_{bind}^{out,s}$ 验证；proof-local commitment 与 $VK_{model}$ 锚点的可除关系在同一子证明中验证。

**最终多头整数平均约束**

定义：

$$C_{avg,sum}(X)=P_{Q_{NC}^{valid}}(X)\left(P_{Y_{sum,flat}}(X)-\sum_{s=0}^{K_{out}-1}P_{Y_{flat}^{(s)}}(X)\right).$$

平均除法实例 $\nu=avg$ 取：

$$P_{x_\nu}=P_{Y_{sum,flat}},\qquad P_{d_\nu}=K_{out},\qquad P_{y_\nu}=P_{Y_{flat}},$$

并使用第 3.3.6 节后给出的 $\mathcal C_{div,avg}$。$P_{Y_{flat}}$ 与公开输出 $P_Y$ 的 basis link 由 $\pi_{bind}^{out,avg}$ 验证。

#### 3.3.14 输出层在节点域上的全部约束

在节点域 $\mathbb H_N$ 上，对每个输出头 $s$ 显式定义：

##### （一）源路由节点端约束


$C_{out,src,node,0}^{(s)}(X)=First_N(X)\cdot P_{R_{src}^{node(out,s)}}(X)$


$C_{out,src,node,1}^{(s)}(X)=\big(1-Last_N(X)\big)\Big(\big(P_{R_{src}^{node(out,s)}}(\omega_NX)-P_{R_{src}^{node(out,s)}}(X)\big)\big(P_{Table^{src(out,s)}}(X)+\beta_{src}^{(out,s)}\big)-P_{Q_N}(X)P_{m_{src}^{(out,s)}}(X)\Big)$


输出头源路由节点端与边端的终点相等关系由第 3.5.3 节的 KZG 同点平移归一化关系验证，不加入节点域 quotient。

##### （二）目标路由节点端约束


$C_{out,dst,node,0}^{(s)}(X)=First_N(X)\cdot P_{R_{dst}^{node(out,s)}}(X)$


$C_{out,dst,node,1}^{(s)}(X)=\big(1-Last_N(X)\big)\Big(\big(P_{R_{dst}^{node(out,s)}}(\omega_NX)-P_{R_{dst}^{node(out,s)}}(X)\big)\big(P_{Table^{dst(out,s)}}(X)+\beta_{dst}^{(out,s)}\big)-P_{Q_N}(X)P_{m_{dst}^{(out,s)}}(X)\Big)$


输出头目标路由节点端与边端的终点相等关系由第 3.5.3 节的 KZG 同点平移归一化关系验证，不加入节点域 quotient。

##### （三）逆元约束


$C_{out,inv}^{(s)}(X)=P_{Q_N}(X)\cdot\big(P_{Sum^{(out,s)}}(X)P_{inv^{(out,s)}}(X)-1\big)$

##### （四）表端绑定约束


$C_{out,src,tbl}^{(s)}(X)=P_{Table^{src(out,s)}}(X)-P_{Q_N}(X)\Big(P_{Idx_N}(X)+\eta_{src}^{(out,s)}P_{E_{src}^{(out,s)}}(X)+(\eta_{src}^{(out,s)})^2P_{Y'^{\star(s)}}(X)\Big)$


$$\begin{aligned}
C_{out,dst,tbl}^{(s)}(X)={}&P_{Table^{dst(out,s)}}(X)-P_{Q_N}(X)\Big(
P_{Idx_N}(X)+\eta_{dst}^{(out,s)}P_{E_{dst}^{(out,s)}}(X)
+(\eta_{dst}^{(out,s)})^2P_{M^{(out,s)}}(X)\\
&\qquad+(\eta_{dst}^{(out,s)})^3P_{Sum^{(out,s)}}(X)
+(\eta_{dst}^{(out,s)})^4P_{inv^{(out,s)}}(X)
+(\eta_{dst}^{(out,s)})^5P_{Y^{pre,\star(s)}}(X)
+(\eta_{dst}^{(out,s)})^6P_{Y^{\star(s)}}(X)\Big)
\end{aligned}$$

**PSQ 节点端目标值绑定**

$$C_{out,psq,Tnode}^{(s)}(X)=P_{Q_N}(X)\Big(P_{T_{out}^{pre(s)}}(X)-P_{Sum^{(out,s)}}(X)-\lambda_{out}^{(s)}P_{Y^{pre,\star(s)}}(X)\Big).$$

##### （五）最终平均输出一致性约束

最终平均的数值关系在边域的 $SignedRoundDiv$ 约束中证明。节点域只保留最终输出压缩对象的 canonical trace；$P_Y$、$P_{Y_{flat}}$ 与 $P_{Y^{\star}}$ 的一致性由 $\pi_{bind}^{out,avg}$ 的 basis link、indexed-view 和压缩关系验证，不使用有限域等式 $K_{out}P_{Y^{\star}}-\sum_sP_{Y^{\star(s)}}=0$ 代替整数舍入平均。

**LogUp 与 route 分母逆元约束登记**

`ProofSchema` 为每个工作域固定有序分母实例登记表 $\mathfrak I_{\mathcal D}$。其内容至少包括：$\mathbb H_{FH}$ 上的特征检索；$\mathbb H_{edge}$ 上的 LeakyReLU、范围、指数、ELU、全部 `SignedRoundDiv` limb 范围查询和 route 查询端；$\mathbb H_N$ 上的 route 表端；以及 $\Pi_{bind}$ 在各原生工作域上的 indexed-view 表端与查询端。

对每个实例 $\iota$，按第 3.3.1--3.3.3 节生成其 $C_{inv},C_{pad}$。定义

$$\mathcal C_{denom,\mathcal D}(X)=\sum_{\iota\in\mathfrak I_{\mathcal D}}\sum_{u\in\{tblinv,tblpad,qryinv,qrypad,nodeinv,nodepad,edgeinv,edgepad\}}\alpha_{quot}^{e_{\iota,u}}C_{\iota,u}(X),$$

其中不适用于该实例的项不进入登记表。指数 $e_{\iota,u}$ 由全局 relation registry 唯一固定。任何 $J$ commitment、evaluation 或约束缺失都会导致 $\mathcal C_{denom,\mathcal D}$ 不完整并被 schema gate 拒绝。

对工作域 $\mathbb H_{\mathcal D}$ 中登记为 `ZeroPad` 的全部 trace，定义

$$\mathcal C_{padding,\mathcal D}(X)=\sum_{P\in\mathfrak P_{\mathcal D}^{zero}}\alpha_{quot}^{e_{pad,P}}\big(1-P_{Q_P}(X)\big)P_P(X).$$

$\mathfrak P_{\mathcal D}^{zero}$、选择器和指数由 `ProofSchema` 固定；状态机和 coefficient-tail 对象不得重复进入该集合。

#### 3.3.15 七个工作域的商多项式

七个工作域的商多项式在多层 / 多头场景下按**层序 + 头 序 + avg 对象**的固定顺序聚合。具体写法如下。

##### （一）边域商多项式

$$\begin{aligned}
t_{edge}(X)=\frac{1}{Z_{edge}(X)}\Bigg[
&\sum_{\ell=1}^{L-1}\sum_{r=0}^{K_{hid}^{(\ell)}-1}\Big(
\alpha_{quot}^{e_{src,edge,0}^{(\ell,r)}}C_{src,edge,0}^{(\ell,r)}
+\alpha_{quot}^{e_{src,edge,1}^{(\ell,r)}}C_{src,edge,1}^{(\ell,r)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{score}^{(\ell,r)}}C_{score}^{(\ell,r)}
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
+\alpha_{quot}^{e_{max,end}^{(\ell,r)}}C_{max,end}^{(\ell,r)}
+\alpha_{quot}^{e_{delta}^{(\ell,r)}}C_{delta}^{(\ell,r)}\\
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
+\alpha_{quot}^{e_{alpha}^{(\ell,r)}}C_{alpha}^{(\ell,r)}
+\alpha_{quot}^{e_{psq,mul}^{(\ell,r)}}C_{psq,mul}^{(\ell,r)}
+\alpha_{quot}^{e_{psq,w}^{(\ell,r)}}C_{psq,w}^{(\ell,r)}
+\alpha_{quot}^{e_{psq,Tedge}^{(\ell,r)}}C_{psq,Tedge}^{(\ell,r)}
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
+\alpha_{quot}^{e_{src,qry}^{(\ell,r)}}C_{src,qry}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,qry}^{(\ell,r)}}C_{dst,qry}^{(\ell,r)}
\Big)\\
&+\sum_{s=0}^{K_{out}-1}\Big(
\alpha_{quot}^{e_{out,src,edge,0}^{(s)}}C_{out,src,edge,0}^{(s)}
+\alpha_{quot}^{e_{out,src,edge,1}^{(s)}}C_{out,src,edge,1}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{out,score}^{(s)}}C_{out,score}^{(s)}
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
+\alpha_{quot}^{e_{out,max,end}^{(s)}}C_{out,max,end}^{(s)}
+\alpha_{quot}^{e_{out,delta}^{(s)}}C_{out,delta}^{(s)}\\
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
+\alpha_{quot}^{e_{out,alpha}^{(s)}}C_{out,alpha}^{(s)}
+\alpha_{quot}^{e_{out,psq,mul}^{(s)}}C_{out,psq,mul}^{(s)}
+\alpha_{quot}^{e_{out,psq,w}^{(s)}}C_{out,psq,w}^{(s)}
+\alpha_{quot}^{e_{out,psq,Tedge}^{(s)}}C_{out,psq,Tedge}^{(s)}
+\alpha_{quot}^{e_{out,psq,0}^{(s)}}C_{out,psq,0}^{(s)}
+\alpha_{quot}^{e_{out,psq,1}^{(s)}}C_{out,psq,1}^{(s)}
+\alpha_{quot}^{e_{out,psq,end}^{(s)}}C_{out,psq,end}^{(s)}
+\alpha_{quot}^{e_{out,dst,edge,0}^{(s)}}C_{out,dst,edge,0}^{(s)}\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{out,dst,edge,1}^{(s)}}C_{out,dst,edge,1}^{(s)}
+\alpha_{quot}^{e_{out,src,qry}^{(s)}}C_{out,src,qry}^{(s)}
+\alpha_{quot}^{e_{out,dst,qry}^{(s)}}C_{out,dst,qry}^{(s)}
\Big)
+\sum_{\ell=1}^{L-1}\sum_{r=0}^{K_{hid}^{(\ell)}-1}\left(
\alpha_{quot}^{e_{agg,num}^{(\ell,r)}}C_{agg,num}^{(\ell,r)}
+\mathcal C_{div,agg}^{(\ell,r)}
\right)\\
&+\sum_{s=0}^{K_{out}-1}\left(
\alpha_{quot}^{e_{out,num}^{(s)}}C_{out,num}^{(s)}
+\mathcal C_{div,outq}^{(s)}
+\mathcal C_{div,align}^{(s)}
+\alpha_{quot}^{e_{out,bias}^{(s)}}C_{out,bias}^{(s)}
\right)\\
&+\alpha_{quot}^{e_{avg,sum}}C_{avg,sum}
+\mathcal C_{div,avg}
+\mathcal C_{bind,edge}(X)
+\mathcal C_{denom,edge}(X)+\mathcal C_{padding,edge}(X)
\Bigg]
\end{aligned}$$

##### （二）拼接共享维域商多项式


$$t_{cat}(X)=\frac{1}{Z_{cat}(X)}\Bigg[
\sum_{s=0}^{K_{out}-1}\Big(
\alpha_{quot}^{e_{outproj,0}^{(s)}}C_{outproj,0}^{(s)}
+\alpha_{quot}^{e_{outproj,1}^{(s)}}C_{outproj,1}^{(s)}
\Big)
+\mathcal C_{bind,cat}(X)
+\mathcal C_{denom,cat}(X)+\mathcal C_{padding,cat}(X)
\Bigg]$$

##### （三）输出层类别共享维域商多项式


$$\begin{aligned}
t_C(X)=\frac{1}{Z_C(X)}\Bigg[
&\sum_{s=0}^{K_{out}-1}\Big(
\alpha_{quot}^{e_{outsrc,0}^{(s)}}C_{outsrc,0}^{(s)}
+\alpha_{quot}^{e_{outsrc,1}^{(s)}}C_{outsrc,1}^{(s)}
\\
&\qquad\qquad\qquad
+\alpha_{quot}^{e_{outdst,0}^{(s)}}C_{outdst,0}^{(s)}
+\alpha_{quot}^{e_{outdst,1}^{(s)}}C_{outdst,1}^{(s)}
\Big)
+\mathcal C_{bind,C}(X)
+\mathcal C_{denom,C}(X)+\mathcal C_{padding,C}(X)
\Bigg]
\end{aligned}$$

##### （四）节点域商多项式

$$\begin{aligned}
t_N(X)=\frac{1}{Z_N(X)}\Bigg[&
\sum_{\ell=1}^{L-1}\sum_{r=0}^{K_{hid}^{(\ell)}-1}\Big(
\alpha_{quot}^{e_{src,node,0}^{(\ell,r)}}C_{src,node,0}^{(\ell,r)}
+\alpha_{quot}^{e_{src,node,1}^{(\ell,r)}}C_{src,node,1}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,node,0}^{(\ell,r)}}C_{dst,node,0}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,node,1}^{(\ell,r)}}C_{dst,node,1}^{(\ell,r)}
+\alpha_{quot}^{e_{inv}^{(\ell,r)}}C_{inv}^{(\ell,r)}
+\alpha_{quot}^{e_{src,tbl}^{(\ell,r)}}C_{src,tbl}^{(\ell,r)}
+\alpha_{quot}^{e_{dst,tbl}^{(\ell,r)}}C_{dst,tbl}^{(\ell,r)}
+\alpha_{quot}^{e_{psq,Tnode}^{(\ell,r)}}C_{psq,Tnode}^{(\ell,r)}
\Big)\\
&+\sum_{s=0}^{K_{out}-1}\Big(
\alpha_{quot}^{e_{out,src,node,0}^{(s)}}C_{out,src,node,0}^{(s)}
+\alpha_{quot}^{e_{out,src,node,1}^{(s)}}C_{out,src,node,1}^{(s)}
+\alpha_{quot}^{e_{out,dst,node,0}^{(s)}}C_{out,dst,node,0}^{(s)}
+\alpha_{quot}^{e_{out,dst,node,1}^{(s)}}C_{out,dst,node,1}^{(s)}
+\alpha_{quot}^{e_{out,inv}^{(s)}}C_{out,inv}^{(s)}
+\alpha_{quot}^{e_{out,src,tbl}^{(s)}}C_{out,src,tbl}^{(s)}
+\alpha_{quot}^{e_{out,dst,tbl}^{(s)}}C_{out,dst,tbl}^{(s)}
+\alpha_{quot}^{e_{out,psq,Tnode}^{(s)}}C_{out,psq,Tnode}^{(s)}
\Big)\\
&+\mathcal C_{bind,N}(X)
+\mathcal C_{denom,N}(X)+\mathcal C_{padding,N}(X)
\Bigg]
\end{aligned}$$

输入共享维域和隐藏层单头共享维域的商分别定义为：

$$t_{in}(X)=\frac{\displaystyle
\sum_{\ell=1}^{L-1}\sum_{r=0}^{K_{hid}^{(\ell)}-1}
\Big(\alpha_{quot}^{e_{proj,0}^{(\ell,r)}}C_{proj,0}^{(\ell,r)}
+\alpha_{quot}^{e_{proj,1}^{(\ell,r)}}C_{proj,1}^{(\ell,r)}\Big)
+\mathcal C_{bind,in}(X)+\mathcal C_{denom,in}(X)+\mathcal C_{padding,in}(X)}{Z_{in}(X)},$$

$$\begin{aligned}
t_{d_h}(X)=\frac{1}{Z_{d_h}(X)}\Bigg[&
\sum_{\ell=1}^{L-1}\sum_{r=0}^{K_{hid}^{(\ell)}-1}\Big(
\alpha_{quot}^{e_{srcbind,0}^{(\ell,r)}}C_{srcbind,0}^{(\ell,r)}
+\alpha_{quot}^{e_{srcbind,1}^{(\ell,r)}}C_{srcbind,1}^{(\ell,r)}\\
&+\alpha_{quot}^{e_{dstbind,0}^{(\ell,r)}}C_{dstbind,0}^{(\ell,r)}
+\alpha_{quot}^{e_{dstbind,1}^{(\ell,r)}}C_{dstbind,1}^{(\ell,r)}
+\alpha_{quot}^{e_{starbind,0}^{(\ell,r)}}C_{starbind,0}^{(\ell,r)}
+\alpha_{quot}^{e_{starbind,1}^{(\ell,r)}}C_{starbind,1}^{(\ell,r)}\Big)
+\mathcal C_{bind,d_h}(X)+\mathcal C_{denom,d_h}(X)+\mathcal C_{padding,d_h}(X)\Bigg]
\end{aligned}$$

$t_{FH}$、$t_{in}$ 和 $t_{d_h}$ 的对象与点评值均按 `ProofSchema` 固定顺序进入 transcript 与 serializer。


**隐藏约束分子、校正商与商掩码**

对每个工作域 $\mathbb H_{\mathcal D}$，令第 3.3.7 至 3.3.15 节完全展开的语义约束分子为：

$$N_{\mathcal D}(X)=Z_{\mathcal D}(X)t_{\mathcal D}(X).$$

实际约束分子 $N_{\mathcal D}^{zk}(X)$ 由第 3.3.7 至 3.3.15 节的完整公式构造，其中每个私有工作域多项式使用实际提交的隐藏对象 $P^{zk}=P+Z_{native(P)}R_P$，每个 relation view 使用第 0.7.8 节绑定的隐藏 commitment。`ProofSchema` 必须保证进入某个工作域 quotient 的 private trace 使用该工作域上的 canonical trace 或已完成 indexed-view / basis-link 的 relation view；未经绑定的跨域多项式不得直接进入 $N_{\mathcal D}^{zk}$。

由于隐藏对象在相应约束工作域的有效位置保持相同语义值，存在校正多项式：

$$B_{\mathcal D}(X)=\frac{N_{\mathcal D}^{zk}(X)-N_{\mathcal D}(X)}{Z_{\mathcal D}(X)}.$$

定义校正商：

$$t_{\mathcal D}^{corr}(X)=t_{\mathcal D}(X)+B_{\mathcal D}(X)=\frac{N_{\mathcal D}^{zk}(X)}{Z_{\mathcal D}(X)}.$$

为隐藏校正商，独立采样 $R_{t,\mathcal D}$，并定义实际商 commitment 对象：

$$t_{\mathcal D}^{zk}(X)=t_{\mathcal D}^{corr}(X)+Z_{\mathcal D}(X)R_{t,\mathcal D}(X).$$

证明者提交 $[t_{\mathcal D}^{zk}]$ 与 $[R_{t,\mathcal D}]$，不提交 $t_{\mathcal D}$、$B_{\mathcal D}$ 或 $t_{\mathcal D}^{corr}$。验证者在挑战点 $z_{\mathcal D}$ 检查：

$$N_{\mathcal D}^{zk}(z_{\mathcal D})
-Z_{\mathcal D}(z_{\mathcal D})t_{\mathcal D}^{zk}(z_{\mathcal D})
+Z_{\mathcal D}(z_{\mathcal D})^2R_{t,\mathcal D}(z_{\mathcal D})=0.$$

该等式中的 private evaluations 均为 masked evaluation。$R_{t,\mathcal D}$ 在不同工作域和 proof 之间独立采样；其次数、$t_{\mathcal D}^{zk}$ 的次数和相应 opening witness 次数必须登记在 $\mathcal R_{deg}$ 并由 $D_{max}$ 覆盖。

独立商掩码用于隐藏 quotient commitment 并防止跨 proof 关联；由于验证者同时打开 $t_{\mathcal D}^{zk}(z_{\mathcal D})$ 与 $R_{t,\mathcal D}(z_{\mathcal D})$，它本身不单独隐藏校正商在该点的线性组合。该点的 witness 隐私由进入 $N_{\mathcal D}^{zk}$ 的各私有 trace 的独立 domain-vanishing blinders、预先登记的最大独立观察数以及整组 masked openings 共同保证。`privacy_policy` 必须在 setup 前为 $R_{t,\mathcal D}$ 和所有参与该等式的私有对象固定独立观察预算与最低 blinder 次数。proof 字段超出公开观察预算时由 prover 与 verifier 拒绝；私有随机系数不足或跨 proof 复用由诚实证明者生成器、运行时审计和安全测试拒绝，不得声称验证者能从单份 proof 反推出该事实。

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

全部外点评值按第 3.4.2 节的不同点分组规则分别批量开放；不存在跨点统一 $\rho_{ext}$。



#### 3.4.1 masked 外点评值绑定的主实验固定规则

由于 $P^{zk}(X)=P(X)+Z_{\mathcal D}(X)R_P(X)$ 在工作域外通常不等于 $P(X)$，外点评值绑定使用 masked evaluation 与第 0.7.8 节的 KZG 方程。表中的 `ZKSubproofMode` 专指第 0.7.8 节定义的 KZG-only 构造。

| 关系族 | 固定实现方式 | 说明 |
|---|---|---|
| CRPC 投影绑定 | `ZKSubproofMode` | 不输出裸折叠向量和裸 $\mu$ |
| 源注意力 / 目标注意力绑定 | `ZKSubproofMode` | 只输出承诺、masked evaluation 和子证明 |
| 压缩特征绑定 | `ZKSubproofMode` | 只输出承诺、masked evaluation 和子证明 |
| 拼接绑定 | `DomainQuotientMode` | 在工作域内用商多项式约束 |
| lookup / LogUp 累加器 | `DomainQuotientMode` | 使用域内商多项式约束 |
| route 节点端与边端递推 | `DomainQuotientMode` | 分别使用节点域和边域商多项式约束 |
| route 跨域终点相等 | `ZKSubproofMode` | 点互异时使用 `KZGShiftRelationProof`；点合并后登记普通零值 claim |
| PSQ 状态机 | `DomainQuotientMode` | 使用域内商多项式约束 |
| 最大值唯一性 / 范围 / 逆元 / 归一化 | `DomainQuotientMode` | 使用域内商多项式约束 |
| 输出平均绑定 | `DomainQuotientMode` | 使用域内商多项式约束 |
| 最终 logits 绑定 | `PublicOutputMode` | 主实验公开 logits 并绑定到 proof |

实现要求：

1. `ProofSchema` 必须为表中的每个关系族登记唯一 `relation_mode`；`DomainQuotientMode`、`ZKSubproofMode` 和 `PublicOutputMode` 的对象集合不得重叠或由证明者临时选择。
2. 每个 $\Pi_{bind}$ 子证明必须携带 `binding_mode = ZKSubproofMode`、关系子类型和域分离标签；只由域内商约束的关系不得伪装成 $\Pi_{bind}$ 字段。
3. Verifier 必须拒绝缺失模式、关系族与模式不一致、同一关系被重复编码或未按 `ProofSchema` 登记的 proof。
4. Private binding 不得输出裸折叠向量、裸模型参数评值、裸中间激活或裸 $\mu$。
5. 子证明必须使用独立 domain-separation tag，例如 `bind:hidden:<layer>:<head>:proj`。
6. 每个登记的绑定关系都必须具有完整的 commitments、挑战、工作域约束、KZG witnesses 和 verifier 方程；任一部分缺失时验证者必须拒绝。

公开多项式的 claimed value 采用唯一规则：验证者从公共输入或公开输出本地计算该值；`ProofSchema` 必须在该对象所属的 `Eval_ext` 或 `Eval_dom` 固定位置把它序列化且仅序列化一次，字段类型固定为 `PublicValue`。验证者必须逐字节与本地计算值相等后才将该值吸入 transcript。协议不得接受由 prover 自行声明且验证者无法本地重算的 `PublicValue`，opening block 也不得重复编码该值。

对 $P_H,P_H^{feat},P_Y,P_{Y^\star},P_{Y^{\star,edge}}$ 及其他由公共输入唯一确定的动态公开多项式，commitment 必须在相应随机挑战产生前固定；随后在至少一个承诺后随机点以及该对象被后续约束实际使用的全部协议点上，以验证者本地计算的公开评值执行 KZG opening。若恶意 prover 提交了与公开多项式不同、但仍处于全局 KZG 次数界 $D_{max}$ 内的多项式，其通过这些承诺后随机点检查的概率使用 `RandomIdentityLedger` 的最坏次数 $d_c^{max}$ 计入定理 2；不得使用无法由 verifier 强制的对象级实际次数。

#### 3.4.2 KZG opening 的正式算法接口与验证方程

主实验唯一允许 `point_grouped_batch_opening`。所有九个顶层 block 及其子块产生的普通 KZG claims 必须先汇总到唯一全局 `StandardPointGroupRegistry`，再按规范点编码排序并分组；每个不同标量点独立生成一个同点 batch witness，不使用跨点 quotient folding，也不允许按 proof block、子证明或关系类型分别生成同点 witness。

每个 registry entry 固定包含 $(point,canonical\_claim\_list,owner\_block,group\_id)$。claim 的全局顺序按 `ProofSchema` 分配的 $(claim\_id,object\_id,commitment\_reference,evaluation\_kind,claim\_role,value\_reference\_id)$ canonical bytes 排序。group 的 `owner_block` 只决定唯一 witness 的物理归属，不参与 claim 顺序。重复的同一 $([P],z)$ 只能保留一次且 claimed value 必须一致；若来自多个字段，保留字典序最小的 canonical `claim_id` 作为唯一条目，其余位置只能引用该条目；值不一致时 parser 立即拒绝。14 个工作域 opening 点 $z_{\mathcal D},z_{\mathcal D}\omega_{\mathcal D}$ 按 $FH,edge,in,d_h,cat,C,N$ 的固定顺序生成，并通过只依赖公开先前点的 counter 排除条件保证两两不同；这些排除条件进入 `RandomIdentityLedger` 与 `SecurityBudgetLedger`。

若 registry point 等于上述任一工作域 opening 点，其唯一 `OpeningWitness` 归属 $Open_{dom}$；其余点的唯一 `OpeningWitness` 归属 $W_{ext}$。$S_{route}$ 与 $Pi_{bind}$ 只保存 `MetadataReference(group_id)`，不得再次序列化曲线 witness。

对点 $z$ 上的有序 claims $([P_0],v_0),\ldots,([P_{m-1}],v_{m-1})$，所有 `PublicValue` 必须先与验证者本地重建值逐字节比较，所有 `MaskedPrivateValue` 必须已经在其唯一字段位置固定。随后按下式生成非零挑战：

$$\rho_z=H_{FS}\Big(\text{transcript},\operatorname{ASCII}(\texttt{"zkgat:kzg:point-group"}),enc_{\mathbb F_p}(z),\operatorname{CanonicalClaimList}_z\Big).$$

$\operatorname{CanonicalClaimList}_z$ 的每个条目必须依次吸入 `claim_id`、`object_id`、commitment reference 与规范曲线点字节、`evaluation_kind`、`claim_role`、visibility tag、value type tag、`value_reference_id` 以及解析后的 claimed-value canonical bytes。`value_reference_id` 不能替代 claimed value 本身；引用值必须在生成 $\rho_z$ 前解析并吸入同一规范域元素字节。禁止只吸入 value tag 而在挑战生成后再选择或修改 $v_i$。

$$P_{\rho_z}(X)=\sum_{i=0}^{m-1}\rho_z^iP_i(X),\qquad
v_{\rho_z}=\sum_{i=0}^{m-1}\rho_z^iv_i,$$

$$W_z(X)=\frac{P_{\rho_z}(X)-v_{\rho_z}}{X-z}.$$

proof 对该点只编码 $[W_z]$。验证者计算

$$C_{\rho_z}=\sum_{i=0}^{m-1}\rho_z^i[P_i]$$

并检查

$$e(C_{\rho_z}-v_{\rho_z}G_1,G_2)=e([W_z],[\tau-z]_2).$$

14 个工作域 opening points 已按上述公开排除规则两两不同，因此每个工作域固定拥有两个 registry entries；外点评值、route 关系和平移关系按其实际标量点登记，若命中某个工作域点则合并到该既有 entry。重复的同一 $([P],z)$ 只能出现一次，parser 必须在生成挑战前合并并检查 claimed value 一致。

字段值按来源分为 `PublicValue` 与 `MaskedPrivateValue`。同一个平移检查值 $u_{rel,j}$ 被两个 claims 引用时只序列化一次，第二处使用 `MetadataReference`；不得把相同私有值复制为两个字段。零关系 claim 的 value 固定为公开常数 $0$。

对每个点都必须计算 $[\tau-z]_2=[\tau]_2-zG_2$。若该元素为单位元，必须立即返回 `SRS_COMPROMISED` 并撤销该 SRS；不得重新派生点后继续使用同一 SRS。构造 witness 时不得使用 $\tau$。`proof_size_bytes` 固定为最终 $\pi_{GAT}$ 九个顶层 block 的规范序列化字节总数，包含 $M_{pub}$ 中完整 $Y_{public}$、全部 tags、长度前缀、commitments、`PublicValue`、`MaskedPrivateValue`、opening witnesses 和 `MetadataReference` 编码；不统计 SRS、manifest、外部注册证明、日志或路径字符串。

#### 3.4.3 reference forward 一致性要求

每个数据集必须保存以下 reference forward 摘要：

1. 明文 GAT forward 的最终 logits hash；
2. 量化后 GAT forward 的最终 logits hash；
3. 按 `output_digest` 计算的 M_pub 完整 $Y_{public}$ 摘要；
4. `model_arch`、`model_params`、`quant_config`、`static_tables`；
5. self-loop、edge sort 和 multi-graph batch 配置。

proof 中的完整公开 logits 必须与量化 reference logits 完全一致。reference forward 时间不计入 proving time，但其摘要必须保存在实验 artifact 中。

### 3.5 最终证明对象

最终证明对象：$\pi_{GAT}$，按如下**固定顺序**序列化为有序元组：

$\pi_{GAT} = \Big( M_{pub},\  \mathbf{Com}_{dyn}^{zk},\  \mathbf{S}_{route},\  \mathbf{Eval}_{ext}^{zk},\  \mathbf{Eval}_{dom}^{zk},\  \mathbf{Com}_{quot}^{zk},\  \mathbf{Open}_{dom}^{zk},\  W_{ext}^{zk},\  \Pi_{bind}^{zk} \Big)$

其中各大块的内部顺序必须完全固定为：

#### 3.5.1 元数据块

$M_{pub}=\big(protocol\_name,proof\_nonce,proof\_task,srs\_id,protocol\_schema\_id,instance\_schema\_hash,srs\_type,srs\_degree\_bound,kzg\_assumption\_profile,privacy\_policy,input\_privacy,output\_mode,opening\_mode,binding\_mode,model\_arch,model\_params,static\_tables,quant\_config,domain\_cfg,dim\_cfg,encoding,padding\_rule,degree\_bound,security\_budget,public\_output\big)$，其中

$$public\_output=(N,C,Y_{public},output\_digest).$$

$Y_{public}$ 按行优先编码恰好 $NC$ 个 canonical field elements；$output\_digest$ 必须按第 0.8.5 节 `output_digest` 重算。验证者本地计算 $P_Y$ 在全部协议挑战点的公开评值。$[P_Y]$ 必须在 $y_{avg}$ 生成前提交，并通过这些公开评值的 KZG opening 与完整数组绑定；验证者无需持有完整 G1 proving key。

#### 3.5.2 动态承诺块

$\mathbf{Com}_{dyn}$ 按以下顺序拼接：

1. 特征检索承诺块 $\mathbf{Com}_{feat}$，其内部顺序严格按第 2.1.7 节中承诺出现顺序：

   $\mathbf{Com}_{feat}=([P_H],[P_H^{feat}],[P_{Table^{feat}}],[P_{Query^{feat}}],[P_{m_{feat}}],[P_{J_{tbl}^{feat}}],[P_{J_{qry}^{feat}}],[P_{R_{feat}}])$

2. 隐藏层 family：对每个隐藏层 $\ell=1,\ldots,L-1$，依次写入

   $\mathbf{Com}_{hidden}^{(\ell,0)},\mathbf{Com}_{hidden}^{(\ell,1)},\ldots,\mathbf{Com}_{hidden}^{(\ell,K_{hid}^{(\ell)}-1)}$

   其中每个 $\mathbf{Com}_{hidden}^{(\ell,r)}$ 的内部顺序，**唯一且完全**由第 3.5.2.1 节的显式枚举定义；实现时不得再使用“按第 2.2 节自动平铺”的口径。

3. 隐藏层拼接块 family：对每个隐藏层 $\ell=1,\ldots,L-1$，在该层全部 头 承诺块之后立即写入

   $\mathbf{Com}_{cat}^{(\ell)}=([P_{H_{cat}}^{(\ell)}],[P_{H_{cat}^{\star}}^{(\ell)}])$

   不得把某层的拼接块移动到下一层 hidden family 之后。

4. 输出层承诺块 family：

   $\mathbf{Com}_{out}^{(0)},\mathbf{Com}_{out}^{(1)},\ldots,\mathbf{Com}_{out}^{(K_{out}-1)}$

   其中每个 $\mathbf{Com}_{out}^{(s)}$ 的内部顺序，**唯一且完全**由第 3.5.2.2 节的显式枚举定义；不同输出头的对象不得交叉混排。

5. 最终平均输出承诺块：$\mathbf{Com}_{out}^{avg}=([P_Y],[P_{Y^{\star}}],[P_{Y^{\star,edge}}],[P_{Y_{sum,flat}}],[P_{Y_{flat}}],\mathbf{Com}_{div}^{avg})$


因此：$\mathbf{Com}_{dyn}=\big(\mathbf{Com}_{feat},\ \{\mathbf{Com}_{hidden}^{(\ell,r)}\}_{\ell,r\ \text{按层序与头序}},\ \{\mathbf{Com}_{cat}^{(\ell)}\}_{\ell=1}^{L-1},\ \mathbf{Com}_{out}^{(0)},\ldots,\mathbf{Com}_{out}^{(K_{out}-1)},\ \mathbf{Com}_{out}^{avg}\big)$


##### 3.5.2.1 单个隐藏头承诺块 $\mathbf{Com}_{hidden}^{(\ell,r)}$ 的完全枚举

对任意隐藏层 $\ell$、任意 hidden 头 $r$，正式固定：


$$\begin{aligned}
\mathbf{Com}_{hidden}^{(\ell,r)}=\big(
&[P_{H'^{(\ell,r)}}],[P_{a^{proj(\ell,r)}}],[P_{b^{proj(\ell,r)}}],[P_{Acc^{proj(\ell,r)}}],\\
&[P_{E_{src}^{(\ell,r)}}],[P_{a^{src(\ell,r)}}],[P_{b^{src(\ell,r)}}],[P_{Acc^{src(\ell,r)}}],\\
&[P_{E_{dst}^{(\ell,r)}}],[P_{a^{dst(\ell,r)}}],[P_{b^{dst(\ell,r)}}],[P_{Acc^{dst(\ell,r)}}],\\
&[P_{H^{\star(\ell,r)}}],[P_{a^{\star(\ell,r)}}],[P_{b^{\star(\ell,r)}}],[P_{Acc^{\star(\ell,r)}}],\\
&[P_{E_{src}^{edge(\ell,r)}}],[P_{H_{src}^{\star,edge(\ell,r)}}],[P_{Table^{src(\ell,r)}}],[P_{Query^{src(\ell,r)}}],[P_{m_{src}^{(\ell,r)}}],[P_{J_{src}^{node(\ell,r)}}],[P_{J_{src}^{edge(\ell,r)}}],[P_{R_{src}^{node(\ell,r)}}],[P_{R_{src}^{edge(\ell,r)}}],\\
&[P_{E_{dst}^{edge(\ell,r)}}],[P_{S^{(\ell,r)}}],[P_{Z^{(\ell,r)}}],[P_{M^{(\ell,r)}}],[P_{M^{edge(\ell,r)}}],[P_{\Delta^{+(\ell,r)}}],[P_{s_{max}^{(\ell,r)}}],[P_{C_{max}^{(\ell,r)}}],\\
&[P_{Table^{L(\ell,r)}}],[P_{Query^{L(\ell,r)}}],[P_{m_L^{(\ell,r)}}],[P_{J_{tbl}^{L(\ell,r)}}],[P_{J_{qry}^{L(\ell,r)}}],[P_{R_L^{(\ell,r)}}],\\
&[P_{Table^{R(\ell,r)}}],[P_{Query^{R(\ell,r)}}],[P_{m_R^{(\ell,r)}}],[P_{J_{tbl}^{R(\ell,r)}}],[P_{J_{qry}^{R(\ell,r)}}],[P_{R_R^{(\ell,r)}}],\\
&[P_{U^{(\ell,r)}}],[P_{Sum^{(\ell,r)}}],[P_{Sum^{edge(\ell,r)}}],[P_{inv^{(\ell,r)}}],[P_{inv^{edge(\ell,r)}}],[P_{\alpha^{(\ell,r)}}],\\
&[P_{Table^{exp(\ell,r)}}],[P_{Query^{exp(\ell,r)}}],[P_{m_{exp}^{(\ell,r)}}],[P_{J_{tbl}^{exp(\ell,r)}}],[P_{J_{qry}^{exp(\ell,r)}}],[P_{R_{exp}^{(\ell,r)}}],\\
&[P_{H_{agg,pre}^{(\ell,r)}}],[P_{H_{agg,pre}^{\star(\ell,r)}}],[P_{H_{agg,pre}^{\star,edge(\ell,r)}}],[P_{\widehat v_{pre}^{\star(\ell,r)}}],[P_{w_{psq}^{(\ell,r)}}],[P_{T_{psq}^{(\ell,r)}}],[P_{T_{psq}^{edge(\ell,r)}}],[P_{PSQ^{(\ell,r)}}],\\
&[P_{H_{agg}^{(\ell,r)}}],[P_{H_{agg}^{\star(\ell,r)}}],[P_{H_{agg}^{\star,edge(\ell,r)}}],[P_{Table^{ELU(\ell,r)}}],[P_{Query^{ELU(\ell,r)}}],[P_{m_{ELU}^{(\ell,r)}}],[P_{J_{tbl}^{ELU(\ell,r)}}],\ [P_{J_{qry}^{ELU(\ell,r)}}],\ [P_{R_{ELU}^{(\ell,r)}}],\\
&[P_{Table^{dst(\ell,r)}}],[P_{Query^{dst(\ell,r)}}],[P_{m_{dst}^{(\ell,r)}}],[P_{J_{dst}^{node(\ell,r)}}],[P_{J_{dst}^{edge(\ell,r)}}],[P_{R_{dst}^{node(\ell,r)}}],[P_{R_{dst}^{edge(\ell,r)}}],\\
&[P_{Sum_{flat}^{(\ell,r)}}],[P_{H_{agg,pre,flat}^{(\ell,r)}}],[P_{H_{agg,num,flat}^{(\ell,r)}}],[P_{H_{ELU,in,flat}^{(\ell,r)}}],\mathbf{Com}_{div}^{agg,(\ell,r)},\\
&[P_{H_{agg,flat}^{(\ell,r)}}]
\big).
\end{aligned}$$

实现时不得再使用“按第 2.2 节出现顺序自动平铺”的模糊口径；serializer / parser 必须以上述枚举为准。

##### 3.5.2.2 单个输出头承诺块 $\mathbf{Com}_{out}^{(s)}$ 的完全枚举

对任意输出头 $s$，正式固定：


$$\begin{aligned}
\mathbf{Com}_{out}^{(s)}=\big(
&[P_{Y'^{(s)}}],[P_{Y'^{\star(s)}}],[P_{a^{proj(out,s)}}],[P_{b^{proj(out,s)}}],[P_{Acc^{proj(out,s)}}],\\
&[P_{E_{src}^{(out,s)}}],[P_{a^{src(out,s)}}],[P_{b^{src(out,s)}}],[P_{Acc^{src(out,s)}}],\\
&[P_{E_{dst}^{(out,s)}}],[P_{a^{dst(out,s)}}],[P_{b^{dst(out,s)}}],[P_{Acc^{dst(out,s)}}],\\
&[P_{E_{src}^{edge(out,s)}}],[P_{Y'^{\star,edge(s)}}],[P_{Table^{src(out,s)}}],[P_{Query^{src(out,s)}}],[P_{m_{src}^{(out,s)}}],[P_{J_{src}^{node(out,s)}}],[P_{J_{src}^{edge(out,s)}}],[P_{R_{src}^{node(out,s)}}],[P_{R_{src}^{edge(out,s)}}],\\
&[P_{E_{dst}^{edge(out,s)}}],[P_{S^{(out,s)}}],[P_{Z^{(out,s)}}],[P_{M^{(out,s)}}],[P_{M^{edge(out,s)}}],[P_{\Delta^{+(out,s)}}],[P_{s_{max}^{(out,s)}}],[P_{C_{max}^{(out,s)}}],\\
&[P_{Table^{L(out,s)}}],[P_{Query^{L(out,s)}}],[P_{m_L^{(out,s)}}],[P_{J_{tbl}^{L(out,s)}}],[P_{J_{qry}^{L(out,s)}}],[P_{R_L^{(out,s)}}],\\
&[P_{Table^{R(out,s)}}],[P_{Query^{R(out,s)}}],[P_{m_R^{(out,s)}}],[P_{J_{tbl}^{R(out,s)}}],[P_{J_{qry}^{R(out,s)}}],[P_{R_R^{(out,s)}}],\\
&[P_{Table^{exp(out,s)}}],[P_{Query^{exp(out,s)}}],[P_{m_{exp}^{(out,s)}}],[P_{J_{tbl}^{exp(out,s)}}],[P_{J_{qry}^{exp(out,s)}}],[P_{R_{exp}^{(out,s)}}],\\
&[P_{U^{(out,s)}}],[P_{Sum^{(out,s)}}],[P_{Sum^{edge(out,s)}}],[P_{inv^{(out,s)}}],[P_{inv^{edge(out,s)}}],[P_{\alpha^{(out,s)}}],\\
&[P_{Y^{pre(s)}}],[P_{Y^{pre,\star(s)}}],[P_{Y^{pre,\star,edge(s)}}],[P_{\widehat y^{\star(s)}}],[P_{w_{out}^{(s)}}],[P_{T_{out}^{pre(s)}}],[P_{T_{out}^{pre,edge(s)}}],[P_{PSQ^{(out,s)}}],\\
&[P_{\widetilde Y^{pre(s)}}],[P_{\widetilde Y^{pre,\star(s)}}],[P_{Y^{(s)}}],[P_{Y^{\star(s)}}],[P_{Y^{\star,edge(s)}}],\\
&[P_{Table^{dst(out,s)}}],[P_{Query^{dst(out,s)}}],[P_{m_{dst}^{(out,s)}}],[P_{J_{dst}^{node(out,s)}}],[P_{J_{dst}^{edge(out,s)}}],[P_{R_{dst}^{node(out,s)}}],[P_{R_{dst}^{edge(out,s)}}],\\
&[P_{Sum_{flat}^{(out,s)}}],[P_{Y_{pre,flat}^{(s)}}],[P_{Y_{agg,num,flat}^{(s)}}],[P_{Y_{q,flat}^{(s)}}],\mathbf{Com}_{div}^{outq,s},\\
&[P_{\widetilde Y_{flat}^{pre(s)}}],[P_{b_{flat}^{(out,s)}}],[P_{Y_{flat}^{(s)}}],\mathbf{Com}_{div}^{align,s}
\big).
\end{aligned}$$

最终平均输出承诺块定义为：


$\mathbf{Com}_{out}^{avg}=([P_Y],[P_{Y^{\star}}],[P_{Y^{\star,edge}}],[P_{Y_{sum,flat}}],[P_{Y_{flat}}],\mathbf{Com}_{div}^{avg})$


**定点除法辅助承诺族**

对任一除法实例 $\nu$，固定：

$$\mathbf{Com}_{div}^{\nu}=\big(
[P_{\sigma_\nu}],[P_{z_\nu}],[P_{a_\nu}],[P_{a_\nu^{-1}}],[P_{d_\nu^{-1}}],[P_{q_\nu}],[P_{r_\nu}],[P_{\delta_\nu}],
\{\mathbf{Com}_{range}^{\nu,V}\}_{V\in\mathcal V_{div}}
\big),$$

其中

$$\mathcal V_{div}=(a,d-1,q,r,\delta),$$

且每个范围子块按公开 limb 顺序编码：

$$\mathbf{Com}_{range}^{\nu,V}=\Big(\{[P_{V^{[h]}}],[P_{Query_R^{\nu,V,h}}],[P_{m_R^{\nu,V,h}}],[P_{J_{tbl}^{R,\nu,V,h}}],[P_{J_{qry}^{R,\nu,V,h}}],[P_{R_R^{\nu,V,h}}]\}_{h=0}^{\ell_{\nu,V}-1}\Big).$$

范围表端 $P_{T_{range}}$、表端有效区选择器和查询选择器 $P_{Q_\nu}$ 由验证者本地重建，不在动态承诺块中重复编码。实例顺序固定为隐藏聚合实例 $agg,(\ell,r)$、输出聚合实例 $outq,s$、输出尺度对齐实例 $align,s$、最终平均实例 $avg$；各实例内部严格使用上述顺序。

#### 3.5.3 路由公开总和块

$\mathbf S_{route}$ 是第三个顶层 block。该块只承载 route 节点端和边端终点相等关系的规范 metadata、平移 commitment 与 opening 引用，不编码裸 $S_{src},S_{dst}$ 或未掩蔽终点。

对每个路由方向，`RelationMeta` 固定关系

$$P_{R^{node}}(z_{last,N})-P_{R^{edge}}(z_{last,edge})=0.$$

若 $z_{last,N}=z_{last,edge}$，令

$$F(X)=P_{R^{node}}(X)-P_{R^{edge}}(X),$$

固定结构为

$$RouteEq_{same}=\big(relation\_tag,relation\_mode,MetadataReference(group\_id_{route})\big),$$

该零值 claim 与同一点的全部普通 claims 合并到全局 `StandardPointGroupRegistry`。若该点是工作域 opening 点，唯一 witness 位于 $Open_{dom}$；否则位于 $W_{ext}$。不生成 route 专用 witness。

若两个终点不同，规范化点序、枢轴和符号由第 0.7.8 节唯一确定。设规范关系为 $F_1(z_1)+F_2(z_2)=0$，$z_\star=z_1$，$\Delta_2=z_2-z_\star$，并提交

$$[\widetilde F_2],\qquad \widetilde F_2(X)=F_2(X+\Delta_2).$$

随后生成该 route 关系的 $y_{rel}$，登记

$$([\widetilde F_2],y_{rel},u_{rel,2}),\qquad([F_2],y_{rel}+\Delta_2,u_{rel,2}),$$

以及

$$([F_1+\widetilde F_2],z_\star,0).$$

$u_{rel,2}$ 作为 `MaskedPrivateValue` 只在 $\mathbf{Eval}_{ext}$ 序列化一次；三个 claims 统一进入全局 `StandardPointGroupRegistry`，其 witness 根据点是否为工作域规范点分别归属 $Open_{dom}$ 或 $W_{ext}$。固定结构为

$$RouteEq_{multi}=\big(relation\_tag,relation\_mode,RelationMeta,[\widetilde F_2],MetadataReference(\mathcal G_{route}^{std})\big),$$

其中 $\mathcal G_{route}^{std}$ 是三个 claims 对应的规范 `group_id` 引用集合。

$\mathbf S_{route}$ 不编码任一终点值或重复 KZG witness。内部顺序先按隐藏层序和头序写入 source / destination route，再按输出头序写入 source / destination route。分支由两个公开终点是否相等唯一确定，proof 不得自行选择。

#### 3.5.4 外点评值块

$\mathbf{Eval}_{ext}$ 按如下顺序排列：

1. hidden 层 family：对每个隐藏层 $\ell=1,\ldots,L-1$、以及该层每个 头 $r=0,\ldots,K_{hid}^{(\ell)}-1$，依次写入与
   $y_{proj}^{(\ell,r)},y_{src}^{(\ell,r)},y_{dst}^{(\ell,r)},y_{\star}^{(\ell,r)},y_{agg,pre}^{(\ell,r)},y_{agg}^{(\ell,r)}$ 对应的外点评值。

2. hidden 层拼接阶段：对每个隐藏层 $\ell=1,\ldots,L-1$，写入$P_{H_{cat}^{\star(\ell)}}(y_{cat}^{(\ell)})$ 及其绑定所需外点评值。

3. 输出层 family：对每个 $s=0,\ldots,K_{out}-1$，依次写入与$y_{proj}^{(out,s)},y_{src}^{(out,s)},y_{dst}^{(out,s)},y_{out,pre}^{(s)},y_{bias}^{(out,s)},y_{out}^{(s)}$ 对应的外点评值。

4. 最终平均输出：写入 $(P_{Y^{\star}}(y_{avg}),P_Y(y_{avg}))$ 以及 $\pi_{bind}^{out,avg}$ 所需的外点评值。

5. route 平移检查值：按 $\mathbf S_{route}$ 的固定内部顺序，对每个 `RouteEq_multi` 写入唯一的 $u_{rel,2}^{route}$。该字段类型固定为 `MaskedPrivateValue`，两个平移检查 claims 通过 `MetadataReference` 引用同一字段；`RouteEq_same` 不产生该字段。

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

route 平移检查子向量固定为

$$\mathbf{Eval}_{ext}^{route}=\big(u_{rel,2}^{route}\big)_{RouteEq_{multi}\text{ 按 }\mathbf S_{route}\text{ 顺序}}.$$

本文正式证明对象不暴露任何额外的折叠向量点评值。第 3.5.4 节已经枚举的 $\mathbf{Eval}_{ext}$ 列表就是该顶层块的完整列表；$\Pi_{bind}$ 内部按第 3.5.9 节编码的 masked evaluations 属于绑定子证明自身，不得在 $\mathbf{Eval}_{ext}$ 中重复编码。serializer 与 parser 不得增加未登记的额外点评值。

#### 3.5.5 域内点评值块

对任意定义在工作域 $\mathbb H_{\mathcal D}$ 上并且进入 proof object 的已承诺多项式 $P$，统一记其域内点评值对为：$\operatorname{ev}_{\mathcal D}(P):=\big(P(z_{\mathcal D}),P(z_{\mathcal D}\omega_{\mathcal D})\big)$。本节显式枚举的 $\mathbf{Eval}_{dom}$ 只承载九个顶层 proof block 中 `Com_dyn` 与 `Com_quot` 所属对象；只在某个 $\Pi_{bind}$ 内出现的 canonical / view / indexed-view / basis / CRPC / 聚合辅助对象，其同样的域内评值对固定编码在该子证明的 $\mathbf{Eval}_{bind}^{zk}$ 中，不得同时复制到 $\mathbf{Eval}_{dom}$。两处合并后的对象集合必须与全局 quotient registry 精确相等。

若某条 quotient identity 还依赖验证者本地重建的公共多项式（例如公共 selector、公共索引列、公共拓扑多项式），则这些公共对象的域内点值**不**在 $\mathbf{Eval}_{dom}$ 中单独编码，而由验证者按与下文相同的对象顺序本地重建并插入检查。 所有 $Q_{tbl},Q_{qry},Q_{valid},Q_{grp}$、枚举列和拓扑列均适用该规则；任何同名动态 commitment 或序列化 evaluation 都必须拒绝。

$\mathbf{Eval}_{dom}$ 在本文正式协议中的唯一固定顺序定义为：

$\mathbf{Eval}_{dom}=(\mathbf{Eval}_{FH},\mathbf{Eval}_{edge},\mathbf{Eval}_{in},\mathbf{Eval}_{d_h},\mathbf{Eval}_{cat},\mathbf{Eval}_{C},\mathbf{Eval}_{N})$。

每个工作域块的最后两个对象固定为 $\operatorname{ev}_{\mathcal D}(t_{\mathcal D}^{zk})$ 与 $\operatorname{ev}_{\mathcal D}(R_{t,\mathcal D})$。虽然 quotient identity 只使用 $z_{\mathcal D}$ 处的值，为保持域块的唯一成对 opening 布局，二者仍在 $z_{\mathcal D}$ 与 $z_{\mathcal D}\omega_{\mathcal D}$ 两点同时编码和开放；相应两次观察必须计入 `privacy_policy`。

**定点除法域内点评值元组**

对任一除法实例 $\nu$，定义：

$$\operatorname{ev}_{div}(\nu)=\big(
\operatorname{ev}_{edge}(P_{\sigma_\nu}),\operatorname{ev}_{edge}(P_{z_\nu}),\operatorname{ev}_{edge}(P_{a_\nu}),\operatorname{ev}_{edge}(P_{a_\nu^{-1}}),\operatorname{ev}_{edge}(P_{d_\nu^{-1}}),\operatorname{ev}_{edge}(P_{q_\nu}),\operatorname{ev}_{edge}(P_{r_\nu}),\operatorname{ev}_{edge}(P_{\delta_\nu}),
\{\operatorname{ev}_{range}(\nu,V)\}_{V\in\mathcal V_{div}}
\big),$$

其中

$$\operatorname{ev}_{range}(\nu,V)=\Big(
\{\operatorname{ev}_{edge}(P_{V^{[h]}}),\operatorname{ev}_{edge}(P_{Query_R^{\nu,V,h}}),\operatorname{ev}_{edge}(P_{m_R^{\nu,V,h}}),\operatorname{ev}_{edge}(P_{J_{tbl}^{R,\nu,V,h}}),\operatorname{ev}_{edge}(P_{J_{qry}^{R,\nu,V,h}}),\operatorname{ev}_{edge}(P_{R_R^{\nu,V,h}})\}_{h=0}^{\ell_{\nu,V}-1}
\Big).$$

所有元组均在 $z_{edge}$ 与 $z_{edge}\omega_{edge}$ 两点按 `ProofSchema` 固定顺序编码。

##### 3.5.5.1 特征检索域块


$\mathbf{Eval}_{FH}=\big(
\operatorname{ev}_{FH}(P_H^{feat}),\
\operatorname{ev}_{FH}(P_{Table^{feat}}),\ \operatorname{ev}_{FH}(P_{Query^{feat}}),\
\operatorname{ev}_{FH}(P_{m_{feat}}),\
\operatorname{ev}_{FH}(P_{J_{tbl}^{feat}}),\ \operatorname{ev}_{FH}(P_{J_{qry}^{feat}}),\
\operatorname{ev}_{FH}(P_{R_{feat}}),\operatorname{ev}_{FH}(t_{FH}^{zk}),\operatorname{ev}_{FH}(R_{t,FH})
\big)$

##### 3.5.5.2 边域块

边域块固定为：$\mathbf{Eval}_{edge}=\big(
\{\mathbf{Eval}_{edge}^{hidden,(\ell,r)}\}_{\ell=1}^{L-1}\!{}_{r=0}^{K_{hid}^{(\ell)}-1},\ \{\mathbf{Eval}_{edge}^{out,s}\}_{s=0}^{K_{out}-1},\
\mathbf{Eval}_{edge}^{avg},\operatorname{ev}_{edge}(t_{edge}^{zk}),\operatorname{ev}_{edge}(R_{t,edge})
\big)$

其中，对任意隐藏层 $\ell$、任意 hidden 头 $r$，固定定义：


$\mathbf{Eval}_{edge}^{hidden,(\ell,r)}=\big(
\operatorname{ev}_{edge}(P_{E_{src}^{edge(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{H_{src}^{\star,edge(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Query^{src(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{J_{src}^{edge(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{R_{src}^{edge(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{E_{dst}^{edge(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{S^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Z^{(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{M^{edge(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{\Delta^{+(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{s_{max}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{C_{max}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Table^{L(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Query^{L(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{m_L^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{J_{tbl}^{L(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{J_{qry}^{L(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{R_L^{(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{Table^{R(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Query^{R(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{m_R^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{J_{tbl}^{R(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{J_{qry}^{R(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{R_R^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{U^{(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{Sum^{edge(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{inv^{edge(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{\alpha^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Table^{exp(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{Query^{exp(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{m_{exp}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{J_{tbl}^{exp(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{J_{qry}^{exp(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{R_{exp}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{H_{agg,pre}^{\star,edge(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{\widehat v_{pre}^{\star(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{w_{psq}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{T_{psq}^{edge(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{PSQ^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Table^{ELU(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Query^{ELU(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{m_{ELU}^{(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{J_{tbl}^{ELU(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{J_{qry}^{ELU(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{R_{ELU}^{(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{H_{agg}^{\star,edge(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Query^{dst(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{J_{dst}^{edge(\ell,r)}}),\ \operatorname{ev}_{edge}(P_{R_{dst}^{edge(\ell,r)}}),\
\operatorname{ev}_{edge}(P_{Sum_{flat}^{(\ell,r)}}),\operatorname{ev}_{edge}(P_{H_{agg,pre,flat}^{(\ell,r)}}),\operatorname{ev}_{edge}(P_{H_{agg,num,flat}^{(\ell,r)}}),\operatorname{ev}_{edge}(P_{H_{ELU,in,flat}^{(\ell,r)}}),\operatorname{ev}_{div}(agg,(\ell,r)),\operatorname{ev}_{edge}(P_{H_{agg,flat}^{(\ell,r)}})
\big)$

对任意输出头 $s$，固定定义：


$\mathbf{Eval}_{edge}^{out,s}=\big(
\operatorname{ev}_{edge}(P_{E_{src}^{edge(out,s)}}),\
\operatorname{ev}_{edge}(P_{Query^{src(out,s)}}),\
\operatorname{ev}_{edge}(P_{J_{src}^{edge(out,s)}}),\ \operatorname{ev}_{edge}(P_{R_{src}^{edge(out,s)}}),\ \operatorname{ev}_{edge}(P_{E_{dst}^{edge(out,s)}}),\ \operatorname{ev}_{edge}(P_{S^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Z^{(out,s)}}),\ \operatorname{ev}_{edge}(P_{M^{edge(out,s)}}),\
\operatorname{ev}_{edge}(P_{\Delta^{+(out,s)}}),\ \operatorname{ev}_{edge}(P_{s_{max}^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{C_{max}^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Table^{L(out,s)}}),\
\operatorname{ev}_{edge}(P_{Query^{L(out,s)}}),\
\operatorname{ev}_{edge}(P_{m_L^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{J_{tbl}^{L(out,s)}}),\ \operatorname{ev}_{edge}(P_{J_{qry}^{L(out,s)}}),\
\operatorname{ev}_{edge}(P_{R_L^{(out,s)}}),\ \operatorname{ev}_{edge}(P_{Table^{R(out,s)}}),\
\operatorname{ev}_{edge}(P_{Query^{R(out,s)}}),\ \operatorname{ev}_{edge}(P_{m_R^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{J_{tbl}^{R(out,s)}}),\ \operatorname{ev}_{edge}(P_{J_{qry}^{R(out,s)}}),\ \operatorname{ev}_{edge}(P_{R_R^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{U^{(out,s)}}),\ \operatorname{ev}_{edge}(P_{Sum^{edge(out,s)}}),\
\operatorname{ev}_{edge}(P_{inv^{edge(out,s)}}),\ \operatorname{ev}_{edge}(P_{\alpha^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Table^{exp(out,s)}}),\ \operatorname{ev}_{edge}(P_{Query^{exp(out,s)}}),\
\operatorname{ev}_{edge}(P_{m_{exp}^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{J_{tbl}^{exp(out,s)}}),\ \operatorname{ev}_{edge}(P_{J_{qry}^{exp(out,s)}}),\
\operatorname{ev}_{edge}(P_{R_{exp}^{(out,s)}}),\
\operatorname{ev}_{edge}(P_{Y'^{\star,edge(s)}}),\
\operatorname{ev}_{edge}(P_{\widehat y^{\star(s)}}),\
\operatorname{ev}_{edge}(P_{w_{out}^{(s)}}),\ \operatorname{ev}_{edge}(P_{T_{out}^{pre,edge(s)}}),\
\operatorname{ev}_{edge}(P_{PSQ^{(out,s)}}),\ \operatorname{ev}_{edge}(P_{Query^{dst(out,s)}}),\
\operatorname{ev}_{edge}(P_{J_{dst}^{edge(out,s)}}),\ \operatorname{ev}_{edge}(P_{R_{dst}^{edge(out,s)}}),\
\operatorname{ev}_{edge}(P_{Y^{pre,\star,edge(s)}}),\
\operatorname{ev}_{edge}(P_{Y^{\star,edge(s)}}),\
\operatorname{ev}_{edge}(P_{Sum_{flat}^{(out,s)}}),\operatorname{ev}_{edge}(P_{Y_{pre,flat}^{(s)}}),\operatorname{ev}_{edge}(P_{Y_{agg,num,flat}^{(s)}}),\operatorname{ev}_{edge}(P_{Y_{q,flat}^{(s)}}),\operatorname{ev}_{div}(outq,s),\
\operatorname{ev}_{edge}(P_{\widetilde Y_{flat}^{pre(s)}}),\operatorname{ev}_{edge}(P_{b_{flat}^{(out,s)}}),\operatorname{ev}_{edge}(P_{Y_{flat}^{(s)}}),\operatorname{ev}_{div}(align,s)
\big)$

最终平均输出在边域上的唯一块固定为：

$$\mathbf{Eval}_{edge}^{avg}=\big(\operatorname{ev}_{edge}(P_{Y^{\star,edge}}),\operatorname{ev}_{edge}(P_{Y_{sum,flat}}),\operatorname{ev}_{edge}(P_{Y_{flat}}),\operatorname{ev}_{div}(avg)\big).$$

##### 3.5.5.3 输入共享维域块

输入共享维域块固定为：$\mathbf{Eval}_{in}=\big(\{\mathbf{Eval}_{in}^{hidden,(\ell,r)}\}_{\ell=1}^{L-1}\!{}_{r=0}^{K_{hid}^{(\ell)}-1},\operatorname{ev}_{in}(t_{in}^{zk}),\operatorname{ev}_{in}(R_{t,in}) \big)$

其中：$\mathbf{Eval}_{in}^{hidden,(\ell,r)}=\big(
\operatorname{ev}_{in}(P_{a^{proj(\ell,r)}}),\
\operatorname{ev}_{in}(P_{b^{proj(\ell,r)}}),\
\operatorname{ev}_{in}(P_{Acc^{proj(\ell,r)}})
\big)$

输入共享维域不包含输出层 family，也不包含 avg 块。

##### 3.5.5.4 隐藏层单头共享维域块

隐藏层单头共享维域块固定为：$\mathbf{Eval}_{d_h}=\big(
\{\mathbf{Eval}_{d_h}^{hidden,(\ell,r)}\}_{\ell=1}^{L-1}\!{}_{r=0}^{K_{hid}^{(\ell)}-1},\operatorname{ev}_{d_h}(t_{d_h}^{zk}),\operatorname{ev}_{d_h}(R_{t,d_h})
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
\{\mathbf{Eval}_{cat}^{out,s}\}_{s=0}^{K_{out}-1},\operatorname{ev}_{cat}(t_{cat}^{zk}),\operatorname{ev}_{cat}(R_{t,cat})
\big).$

拼接矩阵 $P_{H_{cat}^{(\ell)}}$ 是 coefficient-basis relation view，不属于 $\mathbb H_{cat}$，因而不存在 `Eval_cat^{concat}` 子块；其 canonical 列、coefficient view 和压缩关系只在对应 $\pi_{bind}^{concat,(\ell)}$ 中开放。

对任意输出头 $s$：


$\mathbf{Eval}_{cat}^{out,s}=\big(
\operatorname{ev}_{cat}(P_{a^{proj(out,s)}}),\
\operatorname{ev}_{cat}(P_{b^{proj(out,s)}}),\
\operatorname{ev}_{cat}(P_{Acc^{proj(out,s)}})
\big).$

该域不包含 avg 块。

##### 3.5.5.6 输出类别共享维域块

输出类别共享维域块固定为：$\mathbf{Eval}_{C}=\big(\{\mathbf{Eval}_{C}^{out,s}\}_{s=0}^{K_{out}-1},\operatorname{ev}_{C}(t_C^{zk}),\operatorname{ev}_{C}(R_{t,C})\big)$

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
\mathbf{Eval}_{N}^{avg},\operatorname{ev}_{N}(t_N^{zk}),\operatorname{ev}_{N}(R_{t,N})
\big).$

其中，对任意隐藏层 $\ell$、任意 hidden 头 $r$：


$\mathbf{Eval}_{N}^{hidden,(\ell,r)}=\big(
\operatorname{ev}_{N}(P_{E_{src}^{(\ell,r)}}),\ \operatorname{ev}_{N}(P_{E_{dst}^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{H^{\star(\ell,r)}}),\ \operatorname{ev}_{N}(P_{Table^{src(\ell,r)}}),\
\operatorname{ev}_{N}(P_{m_{src}^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{J_{src}^{node(\ell,r)}}),\ \operatorname{ev}_{N}(P_{R_{src}^{node(\ell,r)}}),\ \operatorname{ev}_{N}(P_{M^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{Sum^{(\ell,r)}}),\ \operatorname{ev}_{N}(P_{inv^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{H_{agg,pre}^{\star(\ell,r)}}),\
\operatorname{ev}_{N}(P_{T_{psq}^{(\ell,r)}}),\ \operatorname{ev}_{N}(P_{H_{agg}^{\star(\ell,r)}}),\
\operatorname{ev}_{N}(P_{Table^{dst(\ell,r)}}),\ \operatorname{ev}_{N}(P_{m_{dst}^{(\ell,r)}}),\
\operatorname{ev}_{N}(P_{J_{dst}^{node(\ell,r)}}),\ \operatorname{ev}_{N}(P_{R_{dst}^{node(\ell,r)}})
\big).$

对任意隐藏层 $\ell$：


$\mathbf{Eval}_{N}^{concat,\ell}=\big(
\operatorname{ev}_{N}(P_{H_{cat}^{\star(\ell)}})
\big).$

对任意输出头 $s$：


$\mathbf{Eval}_{N}^{out,s}=\big(
\operatorname{ev}_{N}(P_{E_{src}^{(out,s)}}),\ \operatorname{ev}_{N}(P_{E_{dst}^{(out,s)}}),\
\operatorname{ev}_{N}(P_{Table^{src(out,s)}}),\ \operatorname{ev}_{N}(P_{m_{src}^{(out,s)}}),\
\operatorname{ev}_{N}(P_{J_{src}^{node(out,s)}}),\ \operatorname{ev}_{N}(P_{R_{src}^{node(out,s)}}),\
\operatorname{ev}_{N}(P_{M^{(out,s)}}),\
\operatorname{ev}_{N}(P_{Sum^{(out,s)}}),\
\operatorname{ev}_{N}(P_{inv^{(out,s)}}),\
\operatorname{ev}_{N}(P_{Y'^{\star(s)}}),\
\operatorname{ev}_{N}(P_{Y^{pre,\star(s)}}),\
\operatorname{ev}_{N}(P_{T_{out}^{pre(s)}}),\
\operatorname{ev}_{N}(P_{\widetilde Y^{pre,\star(s)}}),\ \operatorname{ev}_{N}(P_{Y^{\star(s)}}),\
\operatorname{ev}_{N}(P_{Table^{dst(out,s)}}),\ \operatorname{ev}_{N}(P_{m_{dst}^{(out,s)}}),\
\operatorname{ev}_{N}(P_{J_{dst}^{node(out,s)}}),\ \operatorname{ev}_{N}(P_{R_{dst}^{node(out,s)}})
\big).$

最终平均输出在节点域上的唯一块固定为：


$\mathbf{Eval}_{N}^{avg}=\big(
\operatorname{ev}_{N}(P_{Y^{\star}})
\big).$

上述 $\mathbf{Eval}_{FH},\mathbf{Eval}_{edge},\mathbf{Eval}_{in},\mathbf{Eval}_{d_h},\mathbf{Eval}_{cat},\mathbf{Eval}_{C},\mathbf{Eval}_{N}$ 的块数、块顺序与块内字段，构成本文协议对 $\mathbf{Eval}_{dom}$ 的**唯一完整枚举**。serializer、parser、prover 与 verifier 都必须严格按该枚举实现；不得增加、删除、换序或合并任何域内点评值块。
#### 3.5.6 商承诺块

$\mathbf{Com}_{quot}$ 按七个工作域固定为：

$$\mathbf{Com}_{quot}=\big(
([t_{FH}^{zk}],[R_{t,FH}]),
([t_{edge}^{zk}],[R_{t,edge}]),
([t_{in}^{zk}],[R_{t,in}]),
([t_{d_h}^{zk}],[R_{t,d_h}]),
([t_{cat}^{zk}],[R_{t,cat}]),
([t_C^{zk}],[R_{t,C}]),
([t_N^{zk}],[R_{t,N}])
\big).$$

每一对内部先写隐藏校正商 commitment，再写独立商掩码 commitment。未校正的语义商、校正项和未掩蔽校正商不进入 proof。

#### 3.5.7 域内 opening 块

$\mathbf{Open}_{dom}$ 按七个工作域固定为：

$\mathbf{Open}_{dom}=(Open_{FH},Open_{edge},Open_{in},Open_{d_h},Open_{cat},Open_C,Open_N)$，其中每个 $Open_{\mathcal D}=((group\_id_{\mathcal D,0},[W_{z_{\mathcal D}}]),(group\_id_{\mathcal D,1},[W_{z_{\mathcal D}\omega_{\mathcal D}}]))$

每个 $Open_{\mathcal D}$ 编码全局 `StandardPointGroupRegistry` 中由该工作域两个规范 opening 点拥有的唯一 point-grouped KZG witness commitments，不编码任何 evaluation。该点的 claims 可以来自 $Eval_{dom}$、$S_{route}$、$Eval_{ext}$ 或任意 $Pi_{bind}$ 子块，但必须在同一个全局 point group 内折叠。$t_{\mathcal D}^{zk}$、$R_{t,\mathcal D}$ 及其他对象的 claimed values 仍只在各自唯一的 evaluation 字段中编码。

各工作域 witness 的对象归属规则：

为避免实现时误把对象放入错误工作域，固定如下归属：

1. $Open_{FH}$：

   只包含特征检索子系统的 $\mathbb H_{FH}$ 对象，即 $P_H^{feat},P_{Table^{feat}},P_{Query^{feat}},P_{m_{feat}},P_{J_{tbl}^{feat}},P_{J_{qry}^{feat}},P_{R_{feat}},t_{FH}^{zk},R_{t,FH}$ 在规范点组上的 KZG witness commitments；claimed values 只在 $\mathbf{Eval}_{FH}$ 中编码。$P_H$ 是 coefficient-basis 公共对象，只在相应 coefficient 关系点开放；两个公开选择器由验证者本地评值。

2. $Open_{edge}$：

   包含全部边域对象，尤其是 hidden family 与 output family 中所有 source / target 广播列和 route 查询列、LeakyReLU / range / exp lookup 列、$E_{src}^{edge},E_{dst}^{edge},H_{src}^{\star,edge},H_{agg,pre}^{\star,edge},H_{agg}^{\star,edge},S,Z,M^{edge},\Delta^+,s_{max},C_{max},U,Sum^{edge},inv^{edge},\alpha$、PSQ 源值、状态机及其边域目标值，以及 $Y'^{\star,edge(s)},Y^{pre,\star,edge(s)},Y^{\star,edge(s)},Y^{\star,edge}$。同时包含第 3.5.5 节登记在边域的定点除法域内点评值元组中的 $Sum_{flat}$、有限域聚合矩阵、整数聚合分子、ELU/输出定点结果、尺度对齐、偏置广播、最终整数平均、全部 $SignedRoundDiv$ 辅助列和范围 LogUp 对象。属于 $\Pi_{bind}$ 的 canonical/view、basis、CRPC 与承诺后聚合关系对象，其 claimed values 由第 3.5.9 节的 $\mathbf{Eval}_{bind}^{zk}$ 编码；若 claim 点是边域规范 opening 点，其唯一 witness 仍由本 $Open_{edge}$ 的全局 point group 编码，$Pi_{bind}$ 只保存引用。验证者重建边域 quotient 时从相应绑定子证明读取已登记的 masked evaluations。

3. $Open_{in}$：

   包含全部定义在 $\mathbb H_{in}$ 上的 projection 折叠向量和共享维累加器；$P_{Q_{in}^{valid,(\ell)}}$ family 由验证者本地评值，不进入 opening。

4. $Open_{d_h}$：

   包含全部定义在 $\mathbb H_{d_h}$ 上的 hidden-head 绑定折叠向量和共享维累加器；$P_{Q_{d_h}^{valid,(\ell)}}$ family 由验证者本地评值，不进入 opening。

5. $Open_{cat}$：

   只包含输出投影共享维的 $a^{proj},b^{proj},Acc^{proj}$ 以及 $t_{cat}^{zk},R_{t,cat}$ 的 point-grouped witnesses；hidden concat 的 canonical 列和 coefficient relation view 属于对应 $\Pi_{bind}$，不得在本块重复开放。$P_{Q_{cat}^{valid,(\ell)}}$ family 与 $P_{Q_{cat}^{valid,(out)}}$ 由验证者本地评值，不进入 opening。

6. $Open_C$：

   包含全部定义在 $\mathbb H_C$ 上的输出层类别维绑定对象和最终 average 所需的类别维绑定接口对象；公共类别枚举对象由验证者本地评值。

7. $Open_N$：

   包含全部私有节点域对象，尤其是全部$E_{src},E_{dst},M,Sum,inv,H^{\star},H_{agg,pre}^{\star},H_{agg}^{\star},H_{cat}^{\star},Y'^{\star},Y^{pre,\star},\widetilde Y^{pre,\star},Y^{\star}$，各类 node-route 累加器，以及最终输出$P_{Y^{\star}}$。

对任意工作域 $\mathbb H_{\mathcal D}$，进入 $C_{bind,\mathcal D}$ 的关系对象若归属于 $\Pi_{bind}$，其 commitment 与 masked evaluation 按第 3.5.9 节编码；普通动态 trace 的 evaluation 按 $Eval_{dom}$ 编码。两类 claims 都必须进入同一个全局 point group，实际 KZG witness 统一由本节对应 $Open_{\mathcal D}$ 编码。`ProofSchema` 必须保证 evaluation 字段唯一、claim 去重且 witness owner 唯一。

$\mathbf{Eval}_{dom}$ 与 $\mathbf{Open}_{dom}$ 都按七域固定顺序排列；$Open_{\mathcal D}$ 内部先 $z_{\mathcal D}$、后 $z_{\mathcal D}\omega_{\mathcal D}$，但每个点组内部的 claim 顺序必须服从全局 registry，而不要求与某个单独 evaluation block 连续对应。任何实现都不得以“该对象可由别的域对象推出”为由省略已登记 claim。

#### 3.5.8 外点评值折叠见证

$W_{ext}$ 固定为按规范点序排列的全局普通 point-grouped witness 向量：

$$W_{ext}=\big((group\_id_{y_1},[W_{y_1}]),\ldots,(group\_id_{y_t},[W_{y_t}])\big).$$

$y_1,\ldots,y_t$ 是全局 `StandardPointGroupRegistry` 中不等于任何工作域规范 opening 点的全部唯一点，claims 可以来自 $\mathbf{Eval}_{ext}$、$\mathbf S_{route}$、$\mathbf{Eval}_{bind}^{zk}$ 或其他已登记字段。每个 witness 折叠该点跨所有 block 的 canonical claim list；不同点之间不存在统一 folding witness。$S_{route}$ 与 $Pi_{bind}$ 只保存 `MetadataReference(group_id)`，实际外点 witnesses 统一保存在本块。

#### 3.5.9 张量绑定子证明块

$\Pi_{bind}$ 的固定顶层顺序保持：

$$\Pi_{bind}= \big( \pi_{bind}^{feat},\ \{\pi_{bind}^{hidden,(\ell,r)}\}_{\ell,r},\ \{\pi_{bind}^{concat,(\ell)}\}_{\ell=1}^{L-1},\ \pi_{bind}^{out,0},\ldots,\pi_{bind}^{out,K_{out}-1},\ \pi_{bind}^{out,avg} \big).$$

每个子证明固定为

$$\pi_{bind}^{object}=\big(RelationMeta,\mathbf{Com}_{modelReblind},\mathbf{Com}_{view},\mathbf{Com}_{viewLogUp},\mathbf{Com}_{agg},\mathbf{Com}_{basis},\mathbf{Com}_{CRPC},\mathbf{Com}_{rel},\mathbf{Eval}_{bind}^{zk},\mathbf{OpenRef}_{bind}\big).$$

1. `RelationMeta` 编码关系标签、来源 / 目标对象、层号、头号、工作域、真实长度、coefficient 区域选择器摘要、chunk offset、规范点序、常数项、从原始 commitments 到各 $F_j$ 的公开线性聚合映射，以及规范枢轴 $z_\star$、平移量 $\Delta_j$、$y_{rel}$ 域分离标签和普通 opening claim 顺序。
2. $\mathbf{Com}_{modelReblind}$ 对每个使用的长期模型 canonical commitment 编码 $[P_{model}^{proof}]$ 与 $[S_{model}]$；工作域由 metadata 唯一确定。
3. $\mathbf{Com}_{view}$ 先按 `RelationMeta` 的矩阵对象顺序和列序编码尚未在其他块出现的动态 canonical 列 commitments $[P_{M[:,j]}^{can}]$；已经在 $\mathbf{Com}_{dyn}$、同一 $\Pi_{bind}$ 的更早字段或 $VK_{model}$ 中出现的 canonical commitment 只能编码 `MetadataReference`，不得重复序列化同一曲线点。随后编码 proof-local view value commitments、coefficient chunk commitments以及 $Q_{sem},Q_{gap},Q_{tail}$ 的公共引用。row-major coefficient view、canonical 列和 `flat` evaluation view 必须使用不同对象标签。
4. $\mathbf{Com}_{viewLogUp}$ 对每个 indexed-view 按顺序编码 $[Table],[Query],[m],[J_{tbl}],[J_{qry}],[R_{tbl}],[R_{qry}]$；其中 $m$ 的字段类型固定为公开拓扑重数 commitment，其 claimed values 由验证者本地重建；公开选择器不重复编码动态 commitment。
5. $\mathbf{Com}_{agg}$ 编码承诺后压缩、广播、边乘积和聚合状态机 commitments；相关约束只通过第 3.5.6 节的七域全局商验证，本字段不得包含局部 quotient commitments。
6. $\mathbf{Com}_{basis}$ 编码 $[\widehat P_M]$、coefficient traces、gap-zero 约束对象和各 chunk basis accumulators。
7. $\mathbf{Com}_{CRPC}$ 编码 $[\widehat P_C]$、按共享维排列的 $[\widehat A_t],[\widehat B_t]$、$[\widehat Q_{CRPC}]$ 与 $[R_Q^{coef}]$。
8. $\mathbf{Com}_{rel}$ 按规范化关系顺序分支编码：$m\ge2$ 时按点序编码 $\{[\widetilde F_j]\}_{j=2}^{m}$；$m=1$ 时该字段为空并只登记普通零值 claim 的 `MetadataReference`；$m=0$ 时不得出现关系字段。
9. $\mathbf{Eval}_{bind}^{zk}$ 按工作域顺序编码该子证明独占且进入 quotient registry 的 canonical / view / indexed-view / basis / CRPC / 聚合辅助多项式评值，并按关系顺序编码每个 $u_{rel,j}$。公开对象使用 `PublicValue`，私有对象和平移检查值使用 `MaskedPrivateValue`；同一 $u_{rel,j}$ 只编码一次，其第二个 claim 使用 `MetadataReference`。不得编码 $F_j(z_j)$ 或裸终点。
10. $\mathbf{OpenRef}_{bind}$ 按规范点序编码该子证明 claims 所属的 `MetadataReference(group_id)`，不得编码任何 `OpeningWitness` 曲线点。$m\ge2$ 的平移检查和关系零值 claim、$m=1$ 的零值 claim 均进入全局普通点组；$m=0$ 不产生引用。每个引用必须解析到 $Open_{dom}$ 或 $W_{ext}$ 中唯一存在的 registry entry。

hidden 子块包含 $H'$、$\alpha$、$H_{agg,pre}$ 的聚合绑定以及定点 view；output 子块包含 $Y'$、$\alpha^{out}$、$Y^{pre}$、定点回缩、proof-local 偏置 canonical / broadcast 关系；avg 子块包含各 $Y^{(s)}$、$Y_{sum,flat}$、$Y_{flat}$、公开 $Y_{public}$、$P_Y$ 和 $P_{Y^\star}$ 的一致性。

#### 3.5.10 Serializer / Parser 的固定字节级规则

为消除实现歧义，proof object 的字节级编码固定遵循下列规则；这些规则属于正式协议的一部分，而不是实现建议。

1. 所有整数统一按无符号 little-endian 编码。

2. 所有域元素统一编码成固定长度 `field_bytes`，其长度由 `encoding` 唯一确定；同一 proof 中不得混用不同长度的域元素编码。parser 必须把字节串解释为无符号整数并检查其严格小于 $p$；非规范代表、超模数编码和同一域元素的多种编码均必须拒绝。

3. 所有曲线点统一采用 compressed encoding，具体编码格式由 `encoding` 唯一确定；serializer 与 parser 不得一端使用 compressed、另一端使用 uncompressed。parser 必须在任何曲线点进入 transcript 前检查编码规范唯一、曲线上点和正确素数阶子群；除具体关系明确允许零 commitment / 零 witness 的字段外，不得以单位元替代必需非单位元字段。

4. 所有 variable-length 向量、family 块、子证明列表，都必须显式写入长度前缀；parser 不得通过“读到下一个 tag”来猜测边界。

5. 对所有 hidden family、concat family、output family，必须显式携带：
   - `layer_id`
   - `head_id`（若适用）
   即使协议顺序已经固定，也不得依赖“隐式位置下标”来恢复对象身份。

6. `M_pub`、`Com_dyn`、`S_route`、`Eval_ext`、`Eval_dom`、`Com_quot`、`Open_dom`、`W_ext`、`Pi_bind` 这九个顶层块，必须各自携带固定 tag；tag 的字节串取值由 `encoding` 唯一确定。

7. `ProofSchema` 的字段类型固定为 `Commitment`、`MetadataReference`、`PublicValue`、`MaskedPrivateValue` 和 `OpeningWitness`。曲线承诺只能使用 `Commitment`；已经在其他块或验证上下文中出现的对象只能使用 `MetadataReference`；$Y_{public}$ 及验证者可本地计算的公开评值只能使用 `PublicValue`；私有 polynomial opening 的 claimed value 只能使用 `MaskedPrivateValue`；KZG witness 只能使用 `OpeningWitness`。evaluation 只允许出现在 `S_route`、`Eval_ext`、`Eval_dom` 或 `Eval_bind`；`OpeningWitness` 曲线点只允许出现在 `Open_dom` 或 `W_ext`。`Pi_bind.OpenRef_bind` 与 `S_route` 只能使用 `MetadataReference(group_id)`。同一 commitment 不得在多个块中重复序列化，重复引用必须使用 `MetadataReference`。

8. parser 若遇到下列任一情形，必须直接拒绝该 proof，而不是“容错继续解析”：
   - 未声明字段；
   - 顶层块缺失；
   - family 长度与 `dim_cfg` 不一致；
   - 对象顺序与第 3.5 节固定顺序不一致；
   - 出现多余对象；
   - 出现重复 tag。

9. parser 必须先完成九个顶层块的结构校验、对象索引和全局 `StandardPointGroupRegistry` 重建，确认每个 group 只有一个 owner witness、所有 `MetadataReference(group_id)` 可解析且不存在跨块重复 claim；verifier 随后严格按第 3.2 节的逻辑阶段重放 transcript。九个顶层块的物理序列化顺序保持第 3.5 节的固定顺序，但物理块位置不等同于挑战吸入阶段；例如 $\Pi_{bind}$ 中在 $\alpha_{quot}$ 之前固定的 commitments，必须在对应逻辑阶段吸入，即使 $\Pi_{bind}$ 作为第九个顶层块序列化。每个逻辑阶段内部仍按该字段在所属子块中的固定字节顺序逐字节吸入，不得按实现容器或对象类型重新排序。

10. Serializer、parser、prover 与 verifier 必须逐项采用本文给出的固定顺序、唯一顺序和完全枚举，不得按实现内部容器顺序重新排列对象。

## 4. 验证

验证算法使用只读验证上下文：

$$\begin{aligned}
VerifierContext=(&VK_{KZG},VK_{static},VK_{model},srs\_manifest,srs\_id,srs\_type,srs\_degree\_bound,\\
&ProtocolCompilerArtifact,protocol\_schema\_id,instance\_schema\_hash,privacy\_policy,domain\_cfg,quant\_cfg,\\
&encoding,padding\_rule,degree\_bound).
\end{aligned}$$

`ProtocolCompilerArtifact` 必须是第 1.4 节定义的完整只读编译工件，或由验证者从本地认证注册表按 `protocol_schema_id` 与 `instance_schema_hash` 唯一加载；仅提供两个 hash 而缺少工件内容属于无效上下文。

$VK_{model}$ 只包含隐藏模型 commitments、形状、布局和摘要，不包含模型参数明文。$VK_{static}$ 与 $VK_{model}$ 都必须是验证者预先认证的注册工件：前者绑定公开静态表 canonical bytes，后者绑定模型身份、隐藏 commitments、公开范围和尺度证书。验证算法输入为：

- 验证上下文 $VerifierContext$
- 公共输入：$(I,src,dst,G_{batch},node\_ptr,edge\_ptr,N,E,N_{total},L,\{K_{hid}^{(\ell)}\}_{\ell=1}^{L-1},\{d_{in}^{(\ell)}\}_{\ell=1}^{L},\{d_h^{(\ell)}\}_{\ell=1}^{L-1},\{d_{cat}^{(\ell)}\}_{\ell=1}^{L-1},C,B,K_{out},Y_{public})$，其中 $Y_{public}\in\mathbb F_p^{N\times C}$ 为完整公开 logits
- 证明对象：$\pi_{GAT}$

验证者按下列步骤执行。

### 4.1 元数据与对象完整性检查

1. 首先对 `ProtocolCompilerArtifact` 执行 canonical 重编码，分别重算 `protocol_schema_id` 与 `instance_schema_hash`，并检查其 `proof_schema`、三份账本、constraint registry、全局 point-group registry template、degree registry、shift spec、公共列 padding registry、KZG 假设 profile 和 transcript 阶段表全部存在且交叉引用闭合；任一不匹配立即拒绝。随后解析 $M_{pub}$，逐项检查 $protocol\_name$、$proof\_task$、$srs\_id$、$protocol\_schema\_id$、$instance\_schema\_hash$、$srs\_type$、$srs\_degree\_bound$、$kzg\_assumption\_profile$、$privacy\_policy$、$input\_privacy$、$output\_mode$、$opening\_mode$、$binding\_mode$、$model\_arch$、$model\_params$、$static\_tables$、$quant\_config$、$domain\_cfg$、$dim\_cfg$、$encoding$、$padding\_rule$、$degree\_bound$、$security\_budget$、$public\_output$ 与本地期望完全一致；不得只检查其中子集。`proof_nonce` 必须是规范编码的 256 位值；验证者不得声称能够验证其是否由 CSPRNG 产生。$M_{pub}$ 中的 $Y_{public}$ 必须与验证算法外部输入的 $Y_{public}$ 逐元素、逐形状完全相等，且其摘要正确。验证者还必须从 `quant_config` 读取全部尺度链、每个范围对象的 limb 数与 `integer_bound_ledger`，校验公开输入范围、最大入度、静态表界、认证模型上界摘要、全部尺度等式以及无模回绕不等式。
2. 无条件检查 $N\ge1$、$G_{batch}\ge1$、$I$ 长度为 $N$、每个 $I_i\in[0,N_{total})$ 且 $I_0,\ldots,I_{N-1}$ 两两不同；检查 $E\ge1$、$N_{total}\ge N$，$src,dst$ 长度为 $E$ 且均位于 $[0,N)$；检查 $node\_ptr,edge\_ptr$ 的长度都恰为 $G_{batch}+1$，严格单调、首尾正确并与公开图块边界一致；检查公开边序列按二元组 $(edge\_gid(k),dst(k))$ 字典序非降序，$node\_gid(src(k))=node\_gid(dst(k))=edge\_gid(k)$，且每个有效节点至少有一条入边。稳定性是公开预处理对同键边保持输入相对顺序的确定性要求；验证者只对已进入 statement 的公开序列检查上述字典序和其 canonical 摘要，不从不可见的预处理前顺序推断稳定性。随后本地重建 $edge\_gid,node\_gid,Q_{graph\_new}^{edge},Q_{graph\_end}^{edge},Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$。当 $G_{batch}=1$ 时，这些 batch-aware 公共拓扑对象按单图退化公式唯一确定；验证者仍然必须显式重建，不得省略。验证者还必须从公开 $T_H$ 与 $I$ 建立 $P_H$、$P_H^{feat}$ 的公开评值器，从完整 $Y_{public}$ 建立 $P_Y$、$P_{Y^\star}$、$P_{Y^{\star,edge}}$ 的公开评值器，并本地重建全部公开选择器和索引列。上述动态公共多项式的 commitments 必须在相应挑战前提交，并在随机挑战点以验证者本地计算的 `PublicValue` 通过 KZG opening；验证者不得假设自己持有 $PK_{coef}$ 或据此声称直接重建 commitment。静态表 commitments 从经过认证的 $VK_{static}$ 加载。proof 中不得出现同名动态替代对象。验证者还必须从 $I,src,dst$ 和每个 indexed-view 的公开索引映射重建 $m_{feat},m_{src},m_{dst},m_{view}^{can}$；这些重数 commitments 在其全部域点评值和独立公开绑定点上的 claimed values 必须标记为 `PublicValue` 并与本地评值逐项相等。拓扑重数不得按 `PrivateWitness` 解析，也不得仅依赖 route / LogUp 终点相等间接约束。
3. 检查 $\mathbf{Com}_{dyn}$、$\mathbf{S}_{route}$、$\mathbf{Eval}_{ext}$、$\mathbf{Eval}_{dom}$、$\mathbf{Com}_{quot}$、$\mathbf{Open}_{dom}$、$W_{ext}$、$\Pi_{bind}$ 的对象数量与顺序是否与第 3.5 节完全一致；从全部 evaluation 字段和关系 metadata 重建全局 `StandardPointGroupRegistry`，确认每个点只有一个 folding challenge、一个 owner witness，且所有 $Pi_{bind}$ / route 引用都解析到该唯一 entry。
4. 若 parser 读到任何未声明字段、重复字段、缺失字段、长度前缀不一致、family 个数与 `dim_cfg` 不一致，或顶层块 tag 与第 3.5.10 节不一致，则验证必须立即拒绝，不得继续做 opening 验证。

### 4.2 重放 Fiat–Shamir transcript

验证者必须按第 3.2.1 节的固定顺序重放全部挑战，包括：

- 全部隐藏层 family 的挑战；
- 每个隐藏层各自的拼接阶段挑战；
- 全部输出头挑战；
- 最终平均输出压缩挑战 $y_{avg}$；
- 商聚合挑战、各工作域 opening 折叠挑战、外点评值折叠挑战。

验证者只对不泄露陷门的公开条件重放 counter：每个 $z_{\mathcal D}$ 满足 $Z_{\mathcal D}(z_{\mathcal D})\ne0$，并按固定域顺序检查 14 个 $z_{\mathcal D},z_{\mathcal D}\omega_{\mathcal D}$ 两两不同；全部字段压缩、随机线性组合、平移检查挑战和普通 folding challenges 满足各自公开排除条件。每个 KZG opening 点固定后都计算 $[\tau-z]_2$；若为单位元，立即返回 `SRS_COMPROMISED` 并撤销该 `srs_id`，不得通过 counter 改点继续验证。对规范化关系，$m=0$ 不得携带关系字段，$m=1$ 只允许普通零值 claim 引用，$m\ge2$ 只允许平移 commitments、单次编码的 masked 检查值和普通 KZG witness 引用；$y_{rel}$ 及其公开排除条件必须按 `RelationMeta` 重放。LogUp 分母非零只通过 $J_{tbl},J_{qry}$ 的商约束验证。

### 4.3 域内 opening 验证

对七个工作域 $FH,edge,in,d_h,cat,C,N$ 的全局 point groups 逐一验证；同一工作域点上的普通 trace、quotient、route 和 $Pi_{bind}$ claims 必须使用同一个 folding challenge 与同一个 witness，确保：

- 特征检索对象、全部隐藏层对象、各隐藏层拼接对象、每个输出头对象、最终平均输出对象，全部在对应工作域上正确开放；
- 对每个隐藏层 $\ell$，hidden family 的对象数量恰好为 $K_{hid}^{(\ell)}$；
- 输出头 family 的对象数量恰好为 $K_{out}$；
- 每个隐藏层的拼接对象都只出现一次，且位于该层全部 hidden-head family 之后；
- 最终平均输出对象仅出现一次，且位于全部输出头 family 之后。


### 4.4 外点评值折叠 opening 验证

对全局 `StandardPointGroupRegistry` 中所有非工作域点执行同点折叠 opening 验证；claims 可以来自 $\mathbf{Eval}_{ext}$、$\mathbf S_{route}$ 或任意 $Pi_{bind}$ 子块，确保：

- $\mathbf{Eval}_{ext}$ 的块数、块顺序与块内字段，必须与第 3.5.4 节给出的完整枚举完全一致；多一个、少一个、换序、插入未枚举字段，验证都必须立即拒绝；
- 每个隐藏层 hidden-head 外点评值块都与其对应的 $(\ell,r)$ 身份一致；
- 每个输出头的投影、源注意力、目标注意力、聚合前压缩输出、尺度对齐后聚合前压缩输出、加偏置后输出压缩外点评值均正确；
- 最终平均输出的 $P_{Y^{\star}}(y_{avg})$ 与 $P_Y(y_{avg})$ 正确；
- 每个 `RouteEq_multi` 的 $u_{rel,2}^{route}$ 只序列化一次，平移 commitment 在 $y_{rel}$、来源虚拟 commitment 在 $y_{rel}+\Delta_2$ 使用同一 claimed value，且关系零值 claim 在 $z_\star$ 正确开放。

### 4.5 quotient 身份检查

对每个工作域 $\mathbb H_{\mathcal D}$，验证者按第 3.3.7 至 3.3.15 节的完整固定顺序，使用 proof 中的 masked evaluations 和本地重建的公开多项式评值，重建

$$N_{\mathcal D}^{zk}(z_{\mathcal D}).$$

重建内容包括全部 lookup、route 递推、最大值、PSQ、共享维累加器、拼接、输出关系，第 3.3.6 节后给出的所有 `SignedRoundDiv` 算术约束和范围 LogUp 约束，以及第 3.3.6 节后给出的按工作域聚合的 $C_{bind,\mathcal D}$。$C_{bind,\mathcal D}$ 所需的绑定对象评值必须从对应 $\Pi_{bind}$ 子块的 $\mathbf{Eval}_{bind}^{zk}$ 读取，其 KZG claims 按全局 point group 在 $Open_{dom}$ 或 $W_{ext}$ 中验证；不得用未承诺的本地中间值重建。随后检查

$$N_{\mathcal D}^{zk}(z_{\mathcal D})
-Z_{\mathcal D}(z_{\mathcal D})t_{\mathcal D}^{zk}(z_{\mathcal D})
+Z_{\mathcal D}(z_{\mathcal D})^2R_{t,\mathcal D}(z_{\mathcal D})=0$$

对全部 $\mathcal D\in\{FH,edge,in,d_h,cat,C,N\}$ 成立。任何工作域缺失商掩码、evaluation、opening 或移位点评值时均拒绝。最终多头平均只接受第 3.3.13 节的整数 `SignedRoundDiv` 关系，不使用 $K_{out}^{-1}$ 的有限域乘法替代。

### 4.6 张量绑定子证明检查

验证者按 $\Pi_{bind}$ 固定顺序执行：

1. 检查每个 `RelationMeta`、对象身份、形状、层号、头号、工作域、coefficient 区域选择器、固定边域 chunk 描述、规范点序、$z_\star$ 与 $\Delta_j$ 是否与 `ProofSchema` 完全一致。
2. 对每个 proof-local 模型 commitment，令 $n=n_{\mathcal D}$，从 manifest 计算 $[\tau^n]_2=[Z_{\mathcal D}]_2+G_2$，并检查等价的 shifted-KZG 方程

   $$e([S_{model}],[\tau^n]_2)=e([P_{model}^{proof}]-[P_{model}^{vk}]+[S_{model}],G_2).$$

   验证者同时检查 `kzg_assumption_profile` 覆盖 $(D_{max},n)$、实例资源界满足 $\deg S_{model}\le D_{max}-n$ 的诚实生成配置，并确认后续全部关系只引用 $[P_{model}^{proof}]$。
3. 验证 indexed-view 的表 / 查询编码、分母逆元、起点和递推；跨域终点按第 0.7.8 节进入普通同点零值 claim或 `KZGShiftRelationProof`。有效分母逆元约束失败、关系字段与点数分支不一致或 claim 重复登记时拒绝。
4. 检查 coefficient object 的 $Q_{sem},Q_{gap},Q_{tail}$ 互斥覆盖、gap-zero 约束、semantic indexed-view、固定边域 chunk 映射和 basis link。
5. 对每个规范化线性关系执行唯一分支：

   - $m=0$：检查 $\lambda_0=0$，并确认不存在关系字段；
   - $m=1$：确认 $([\lambda_0+F_1],z_1,0)$ 已由该点唯一的普通 point-grouped KZG opening 验证；
   - $m\ge2$：重放 $z_\star,\Delta_j,y_{rel}$，按点序读取 $[\widetilde F_j]$ 与单次编码的 $u_{rel,j}$；确认两个平移检查 claims 分别在 $y_{rel}$ 和 $y_{rel}+\Delta_j$ 的普通 point groups 中验证，并确认 $([G_{rel}],z_\star,0)$ 已验证，其中

     $$[G_{rel}]=\lambda_0G_1+[F_1]+\sum_{j=2}^{m}[\widetilde F_j].$$

   proof 不得包含 $F_j(z_j)$、裸 route terminal 或关系专用重复 witness。
6. 验证 CRPC 同点评值 openings，并检查

   $$\widehat P_C(\gamma_{CRPC})-\sum_t\widehat A_t(\gamma_{CRPC})\widehat B_t(\gamma_{CRPC})-\gamma_{CRPC}^{L_C}\widehat Q_{CRPC}(\gamma_{CRPC})+\gamma_{CRPC}^{L_C+L_Q}R_Q^{coef}(\gamma_{CRPC})=0.$$
7. 验证承诺后聚合、压缩、拼接、偏置广播、定点 view 和最终平均关系；这些关系的 quotient 只能来自 $\mathbf{Com}_{quot}$ 的七域全局商，$\mathbf{Com}_{agg}$ 中出现局部 quotient commitment 时拒绝。
8. 对 $\mathbf S_{route}$ 的每个 `RouteEq` 检查公开终点分支。同一点分支确认普通零值 claim；多点分支确认平移 commitment、平移检查 claims 和关系零值 claim 均由普通 point-grouped openings 验证。不得接受 route 专用重复 witness。
9. 从 $M_{pub}$ 中的完整 $Y_{public}$ 计算 $P_Y$、$P_{Y^\star}$ 的公开挑战点评值；验证相应 KZG openings，并检查与输出 view 和最终平均关系一致。
10. 根据第 1.4 节从实际 proof 字段重建 `ObservationLedger`、`RandomIdentityLedger` 与 `SecurityBudgetLedger` 的 canonical 摘要，检查语义 / blinder 覆盖证书、字段依赖 DAG、point-grouped 去重、表达式 DAG、挑战阶段、公开排除多项式、$d_c^{max}<p-B_c$、精确总优势不超过 $2^{-\lambda_{target}}$ 和全局 $D_{max}$。验证者不在未知 $\tau$ 上计算形式观察矩阵秩，也不声称由 proof 检查私有随机系数的 CSPRNG 来源、实际最高非零次数或跨 proof 复用；这些属于安全证明中的 `BadCRS_obs` 界与诚实证明者合规测试。

任一对象缺失、重复、换序、长期模型 commitment 被直接 opening、观察字段超出登记上限、`RandomIdentityLedger` 的公开候选条件失败、账本摘要不一致或 KZG witness 失败时均拒绝。

### 4.7 验证结论

若以上检查全部通过，则接受证明；否则拒绝。

验证者还必须执行 schema-level proof leakage audit gate：任何私有对象出现在 `PublicValue`、未登记字段、调试字段或禁止的 payload 位置时直接拒绝，并强制 `private_plain_value_count==0`。验证者不能仅从一个 `MaskedPrivateValue` 的数值判断诚实证明者是否正确使用 blinder；masked slot 的生成正确性、CSPRNG 来源和随机性复用由证明者实现审计与第 6.2 节隐私测试保证。


## 5. 安全模型与证明目标

### 5.1 参与方与攻击者

证明者 $\mathcal P$ 是 GAT 推理服务端，持有模型参数、模型验证键对应的承诺开口信息以及一次推理中的所有中间 witness。验证者 $\mathcal V$ 是客户端或审计方，持有公开输入、$VK_{KZG}$、经过认证的 $VK_{static}$ 与 $VK_{model}$、公开元数据和最终输出声明。静态表注册与模型注册的真实性属于验证上下文建立阶段：恶意推理证明者不能替换这些注册工件；若注册者不受信任，则必须先验证第 1.5 节规定的一次性注册证明。

攻击者是恶意证明者，可以尝试：替换模型参数、篡改节点特征检索、跳过或篡改注意力 softmax、伪造 lookup、伪造路由、伪造 PSQ、篡改输出头平均、改变 public metadata、重排承诺块、复用其他证明的 opening 或构造与最终 logits 不一致的 proof。

### 5.2 公开输入与私有见证

公开输入包括：图拓扑与 batch 边界、节点索引 $I$、$src,dst,node\_ptr,edge\_ptr$、任务和结构元数据、量化配置、静态表标识、模型结构标识、SRS 标识以及完整最终 logits $Y$。$Y$ 固定为公开输出。

私有见证包括：模型参数开口、隐藏层和输出层全部投影结果、注意力打分、由私有查询分布决定的静态表 lookup multiplicity、lookup query、路由辅助列、Softmax 中间量、PSQ 状态、聚合前后激活、输出头中间输出以及所有 private polynomial 的 blinding randomness。由公开 $I,src,dst$ 或公开 indexed-view 索引映射唯一决定的 $m_{feat},m_{src},m_{dst},m_{view}^{can}$ 属于公开对象，不属于私有见证。

### 5.3 完备性

若证明者按本文定义的多层多头 GAT 前向、公开预处理、量化配置、lookup 表、路由规则、PSQ 状态机和 KZG SRS 生成证明，则验证者接受。

### 5.4 计算可靠性

计算可靠性依赖单生成元 KZG polynomial binding / opening soundness、七域 quotient 随机点检验、LogUp 分母逆元约束、CRPC、Fiat--Shamir 随机预言机模型、认证的静态表 / 模型注册边界和隐藏 SRS 陷门。跨点评值先通过承诺后随机平移检查归一化到同一点，再由普通 KZG 零值 opening 验证；协议不引入第二生成元、Pedersen、IPA、sumcheck 或独立顶层证明系统。

所有概率型代数检查必须出现在 `RandomIdentityLedger`。可靠性分析只使用 verifier 可强制的全局次数界 $D_{max}$：对账本条目 $c$，错误身份多项式的最坏次数为 $d_c^{max}$，公开排除多项式次数为 $B_c$，候选集合满足 $|\mathcal A_c|\ge p-B_c>d_c^{max}$。对象级 $b_O$ 不作为恶意证明者可靠性假设。

LogUp / indexed-view 使用的 tuple compression 也是随机多项式身份。对长度至多 $t$ 的两个不同元组，其压缩碰撞多项式次数至多为 $t-1$；每个实例、挑战复用次数和联合上界都由 `RandomIdentityLedger` 明确登记。平移一致性、公开多项式绑定、CRPC 和七域 quotient 同样按账本的表达式 DAG 和最坏次数计入，不允许用“其他随机检查”笼统省略。

proof-local 模型再盲化通过第 1.4 节的 `ShiftedKZG-Divisibility(D_{max},n_{\mathcal D})` 方程固定为与 $VK_{model}$ 相同的工作域取值，并同时强制存在次数至多 $D_{max}-n_{\mathcal D}$ 的 $S_{model}$。该 pairing 的失败优势按 `kzg_assumption_profile` 计入底层 KZG / 配对假设，不作为 Schwartz--Zippel 项重复计算。

### 5.5 零知识与泄露边界

动态私有 trace 使用独立 domain-vanishing blinding，coefficient object 使用独立 coefficient-tail blinding，长期模型 commitment 每份 proof 执行独立 proof-local 再盲化且不做私有 opening，七域校正商使用独立商掩码。KZG 平移关系只公开平移 commitment、一次编码的 `MaskedPrivateValue`、普通 KZG witness 和公开零值 claim，不公开原关系点私有评值。

零知识采用 CRS + 可编程随机预言机模型下的模拟定义。`Real_ZK` 使用真实 setup、诚实证明者和私有见证；`Sim_ZK` 的 setup 模拟器生成与真实分布相同的 SRS，并仅在安全证明内部保留 $\tau$。运行时 prover、verifier、文件和日志均不包含该陷门。证明模拟器只接收公开 statement：公开图与特征、metadata、认证的 $VK_{static},VK_{model}$ 和完整 $Y_{public}$。

`ObservationLedger` 使用第 1.4 节的形式不定元 $\mathsf T$ 坐标，并在域元素观察与群指数观察的直和空间中分别建立 $A_{sec}$ 与 $A_{blind}$。它不把未知 $P(\tau)$ 或 $W(\tau)$ 当成 verifier 可计算标量；其核心条件是每个 transcript 阶段均满足 $\operatorname{Im}(A_{sec})\subseteq\operatorname{Im}(A_{blind})$，而不是只检查 blinder 矩阵自身有秩。除 `BadCRS_obs` 外，该覆盖条件保证规范 payload 中所有登记私有语义方向均可由诚实 blinder 模拟。模拟器按公开表达式 DAG 联合采样 commitments、评值和 quotient 字段；对 $z\ne\tau$ 的折叠 commitment $C$ 与模拟值 $v$，使用

$$W=(\tau-z)^{-1}(C-vG_1)$$

生成满足标准 KZG pairing 的 witness。proof-local 模型可除关系、七域 quotient、CRPC、basis link和平移关系必须联合模拟，不能分别生成互相矛盾的局部 transcript。

存在两类必须显式计入的统计 / CRS 项：

1. **私有零分母 abort。** 对第 $\nu$ 个 LogUp / route / indexed-view 挑战，设有效私有分母候选数上界为 $q_{den,\nu}<p$。真实发布挑战是均匀挑战在未命中该私有集合条件下的分布，与无条件均匀分布的统计距离至多
   $$\Delta_{abort,\nu}\le\frac{q_{den,\nu}}p.$$
   因此
   $$\Delta_{abort}\le\sum_\nu\frac{q_{den,\nu}}p=:\varepsilon_{abort}.$$
   proof 不编码失败次数；恶意 nonce grinding 另计入 $Q_{FS}$。
2. **形式观察坏 CRS。** 设第 1.4 节各 RREF 枢轴、覆盖矩阵分母与覆盖等式证书关于 $\mathsf T$ 的去重根次数上界为 $d_{O,crs}$，则
   令 $N_\tau=|\mathbb F_p^\ast\setminus\bigcup_{(\mathcal D,n)}\mathbb H_{\mathcal D,n}|$，则
   $$\Delta_{crs}\le\Pr[BadCRS_{obs}]\le\sum_O\frac{d_{O,crs}}{N_\tau}=: \varepsilon_{crs}.$$
   该项覆盖平移 blinder 中 $Z_{\mathcal D}(\tau+\Delta)=0$ 等依赖隐藏陷门、无法由公开 counter 排除的退化。

只依赖公开挑战的排除条件由 verifier 重放，不泄露 witness。`proof_nonce` 是 transcript 域分离输入；应用层 anti-replay cache 不属于密码学接受语义。

proof 中允许出现的私有域元素仅为 `MaskedPrivateValue`。完整 $Y_{public}$ 是唯一动态公开输出。`private_plain_value_count==0`、长期模型私有 opening count 为零和未登记私有评值 count 为零是 verifier 的强制 gate。CSPRNG 来源、独立性和 fresh-proof 不复用由诚实证明者实现审计保证，不伪装成单份 proof 可验证条件。

### 5.6 SRS 假设

SRS 由独立 trusted setup 生成；证明键和验证键发布后必须销毁 $\tau$。安全性依赖 manifest 的 `kzg_assumption_profile`：所选曲线上的普通 KZG binding / opening soundness，以及第 1.4 节明确给出的 `ShiftedKZG-Divisibility(D_{max},n)` 代数知识假设；后者覆盖全部允许工作域大小并给出安全位数。安全性还依赖认证的 $VK_{static},VK_{model}$。manifest 必须同时绑定 `domain_size_catalog` 和 `domain_generator_catalog`；Lagrange SRS、零化 G2 commitment、生成元及行顺序必须使用同一目录项。

setup 保证 $\tau$ 不属于 manifest 中任何工作域。若运行时某公开 opening 点满足 $[\tau-z]_2=\mathcal O_{\mathbb G_2}$，则 $z=\tau$ 已公开识别，相关 SRS 必须立即撤销，不能通过 counter 改点继续使用。形式观察矩阵在实际 $\tau$ 上退化但不直接暴露 $\tau$ 的情形属于 $BadCRS_{obs}$，按第 5.5 节的 $\Delta_{crs}$ 计入安全界。

### 5.7 定理式安全声明

**定理 1（完备性）**：若公开输入、认证静态表、模型、量化前向和全部见证满足本文约束，公开挑战未命中公开排除集合，且该次未发布 proof 未发生私有 LogUp 零分母事件，则诚实证明被接受。发生私有零分母时，诚实 prover 按第 0.8.2 节丢弃未发布 proof 并使用新的 `proof_nonce` 与全部 proof-local randomness 从头生成。

**定理 2（计算可靠性）**：在 KZG binding / opening、SRS 一致性审计和认证注册边界成立的条件下，接受的 proof 固定了一组满足七域约束、CRPC、indexed-view、route、量化、KZG 平移归一化关系和公开输出绑定的全局次数不超过 $D_{max}$ 的多项式。若攻击者至多执行 `SecurityBudgetLedger` 登记的 $Q_{FS}^{max}$ 次随机预言机查询、公开 counter 尝试或 nonce grinding，则

$$
\operatorname{Adv}_{sound}
\le
2^{-\lambda_{KZG}}+2^{-\lambda_{register}}+2^{-\lambda_{RO}}
+\varepsilon_{alg}
+\operatorname{negl}(\lambda),
$$

其中 $\varepsilon_{alg}=Q_{FS}^{max}\sum_c multiplicity_c d_c^{max}/(p-B_c)$，且实例注册已验证该上界连同其他预算项满足 $\varepsilon_{total}\le2^{-\lambda_{target}}$。

该求和明确覆盖七域 quotient、平移一致性、公开多项式绑定、CRPC、tuple compression / LogUp collision、随机线性折叠及其他登记身份。底层 KZG opening 方程不重复计入多项式身份求和。

**定理 3（计算零知识）**：在 CRS 模拟器保留 setup 陷门、随机预言机可编程、诚实证明者按 `ObservationLedger` 的语义像空间覆盖条件和 $q_O^{max}$ 预算独立采样全部 blinder、且 payload 严格服从 `ProofSchema` 的条件下，存在多项式时间模拟器仅凭公开 statement 生成与真实 proof 计算不可区分的 transcript，并满足

$$
\operatorname{Adv}_{ZK}
\le
2^{-\lambda_{KZG}}+2^{-\lambda_{RO}}
+\varepsilon_{abort}+\varepsilon_{crs}
+\operatorname{negl}(\lambda)
\le 2^{-\lambda_{target}}+\operatorname{negl}(\lambda).
$$

该定理覆盖规范序列化 payload，不覆盖运行时间、内存访问、GPU 调度、日志或其他实现侧信道；任何新增字段必须先进入 `ProofSchema`、三份账本、次数编译器和模拟器定义。

**定理 4（公开输出一致性）**：验证接受时，$M_{pub}$ 中完整 $Y_{public}$ 定义的 $P_Y$ 与证明 commitment 的 KZG openings 一致，并与输出平均 trace、输出压缩关系及认证表驱动定点前向一致。错误替代通过的概率使用 `RandomIdentityLedger` 的对应 $d_c^{max}/(p-B_c)$ 项和底层 KZG 优势计入定理 2。

## 6. 实现与实验合规要求

### 6.1 主实验代码要求

1. 证明与验证路径不得包含 `seed -> tau`；
2. prover/verifier 数据结构不得保存 $\tau$；
3. KZG commitment 必须基于 SRS powers 或 Lagrange-basis SRS；`VK_{KZG}` 必须按 manifest 固定顺序加载并校验 $G_2,[\tau]_2$ 以及当前 `domain_cfg` 选中的七个 $[Z_{\mathcal D,n_{\mathcal D}}]_2$；
4. private polynomial 必须采用本文规定的 domain-vanishing 或 coefficient-tail blinding；
5. proof serializer 必须区分 `PublicValue` 与 `MaskedPrivateValue`；$\mathbf{Com}_{rel}$ 与 $\mathbf S_{route}$ 必须按 $m=0$、$m=1$、$m\ge2$ 的规范分支编码；$m\ge2$ 只允许平移 commitments、单次编码的 masked 检查值和普通 KZG witness 引用，$m=1$ 只允许普通零值 claim 引用；不得编码 $F_j(z_j)$ 或裸终点；
6. verifier 必须执行 metadata、schema、opening、quotient、binding、KZG-shift-relation、model-reblind、`ObservationLedger` / `RandomIdentityLedger` / `SecurityBudgetLedger` 摘要和 leakage audit gate；不得把随机源、实际最低 blinder 次数或跨 proof 随机性复用声称为可由单份 proof 验证；
7. benchmark 结果必须来自本文定义的完整证明与验证路径。

### 6.2 必需实验

节点分类实验覆盖 Cora、Citeseer、Pubmed、PPI 和 ogbn-arxiv。每个数据集至少报告 CPU full 与 GPU full 的 commitment time、prove time、verify time 和 proof size；结果使用满足本文 SRS、blinding、binding、quotient 和 verifier 规则的同一证明路径。计时口径固定为：`commitment_time` 统计单份 proof 中全部动态、绑定辅助、平移 commitments 和七域商 commitments 的 MSM / commitment 用时，是 `prove_time` 的组成部分；平移关系的 IFFT coefficient materialization 与 Taylor-shift 运算计入 `prove_time` 但不重复计入 `commitment_time`；`prove_time` 从动态见证已经生成完毕、开始构造首个 proof commitment 起，到规范 $\pi_{GAT}$ 序列化完成为止；`verify_time` 从解析首字节起，到返回最终接受 / 拒绝结果为止。SRS 生成、静态表注册、模型注册、公开预处理、reference forward、文件加载预热和结果落盘不计入上述三项，但必须分别记录。

安全实验包括 tamper rejection、proof leakage audit、SRS metadata mismatch rejection、SRS consistency audit rejection、proof-scope mismatch rejection、commitment order tamper、opening value tamper、challenge tamper、lookup inverse tamper、route relation tamper、PSQ state tamper、$\Pi_{bind}$ relation tamper、model reblinding tamper、quotient correction tamper 和 final logits tamper。

KZG 平移关系测试必须同时覆盖 `shift_algorithm_id`、coefficient materialization 缓存键、Taylor-shift coefficient 与直接多项式代入的一致性，以及跨 proof 复用私有 coefficient cache；并覆盖：篡改规范点序、来源聚合映射、$z_\star$、$\Delta_j$、$y_{rel}$、任一 $[\widetilde F_j]$、$u_{rel,j}$、平移检查 claim、关系零值 claim 或普通 folding challenge；在 $m=0$、$m=1$ 分支注入多点字段；将同一 $u_{rel,j}$ 复制为两个字段而不是使用 `MetadataReference`；删除任一平移 opening 或关系零值 opening；以及把未掩蔽 $F_j(z_j)$ 或裸 route terminal 写入 proof。

SRS 测试必须覆盖 coefficient powers、Lagrange SRS 与零化多项式 G2 commitments 不一致，并覆盖构造 $[\tau-z]_2=\mathcal O$ 的测试工件，确认实现返回 `SRS_COMPROMISED`、撤销工件且不重新派生 opening 点。还必须覆盖 `kzg_assumption_profile` 缺失或不覆盖某个 $(D_{max},n)$、篡改 $[S_{model}]$、篡改派生的 $C_{shift}$、以及试图用次数超过 $D_{max}-n$ 的 $S_{model}$ 伪造可除关系。隐私测试必须覆盖遗漏 `ObservationLedger` 依赖边、point-grouped witness 重复计数、平移依赖未回溯、形式 minor 恒零、`BadCRS_obs` 概率项遗漏、增加未登记 commitment / masked value / opening witness，以及诚实证明者 sampler 的随机系数不足、对象间复用和跨 proof 复用；后三类属于实现合规测试，不作为 verifier 从单份 proof 推断的拒绝理由。可靠性测试必须覆盖 `RandomIdentityLedger` 缺项、恶意对象使用全局 $D_{max}$ 高次多项式、$d_c^{max}\ge p-B_c$、tuple-compression 碰撞项遗漏和错误挑战复用次数。`SecurityBudgetLedger` 测试必须覆盖 $Q_{FS}^{max}$ 缺失、使用浮点计算预算、任一分项超界、总优势大于 $2^{-\lambda_{target}}$ 以及 manifest 安全位数与实例目标不一致。`ObservationLedger` 测试还必须包含 $o_1=s+r,o_2=2s+r$ 型反例，确认只检查 $A_{blind}$ 秩会被拒绝，只有 $\operatorname{Im}(A_{sec})\subseteq\operatorname{Im}(A_{blind})$ 才通过。还必须覆盖：任一 `SignedRoundDiv` 在 $a=0,z=1$ 时篡改 $a^{-1}\ne0$ 必须因 $C_{\nu,zinv}$ 拒绝；同一点跨 $Open_{dom}$、$W_{ext}$ 与多个 $Pi_{bind}$ 重复 witness 或 claimed value 冲突必须拒绝；公共 helper 使用非零 padding 必须拒绝；`output_digest` 的形状、长度、端序、域分离标签或元素编码任一变化必须拒绝，且 $N,C,NC$ 任一无法规范编码为 `u64` 或 64 位乘法发生溢出时必须在承诺前拒绝；缺失或篡改 `ProtocolCompilerArtifact` 必须拒绝；`srs_type` 取任何非 `trusted_srs` 值时必须拒绝。指数表测试必须确认 $ExpMap(0)$ 为最大输出、输入差增大时输出不增，并拒绝把 `ExpMap` 解释为正指数。平移测试必须确认私有最高非零系数、blinder 高位零或线性组合高位抵消不会改变 $d_{shift,j}^{cfg}$、$n_{shift,j}^{cfg}$、`instance_schema_hash`、MSM 逻辑长度或 transcript。point-group 测试必须确认 claimed-value canonical bytes 在 $\rho_z$ 前吸入，生成挑战后篡改 value 必须拒绝。最终平均测试必须确认 $y_{avg}$ 在完整 $\mathbf{Com}_{out}^{avg}$ 固定前不可生成。实例工件测试必须确认 `instance_schema_hash` 只绑定点组模板和导出规则，不绑定 proof-time 实际标量点；实际点或碰撞合并结果被篡改时必须因 transcript 或 registry 不一致拒绝。

LogUp 必须覆盖有效位置零分母、逆元列置零、无效区逆元非零、在 $\beta$ 前提交累加器、私有拒绝计数注入以及拓扑重数 commitment / `PublicValue` 篡改。前向关系必须覆盖源 / 目标广播替换、$S$ 与两个注意力广播列不一致、$\Delta^+\ne M^{edge}-Z$、$\widehat v$ 或 $\widehat y$ 乘积篡改、$w$ 或 $T^{edge}/T^{node}$ 自由列篡改、拼接列交换、跨层输入替换、输出压缩列替换和 route 分支错误。route 测试还必须在 $N\le i<n_N$ 与 $E\le k<n_{edge}$ 的填充位置分别篡改四组主 route 的 $Table/Query$，并确认 $Q_N/Q_{edge}^{valid}$ 门控和 `ZeroPad` 约束拒绝；不得通过填充区 $Idx_N,src,dst$ 的偶然取值绕过。公开 logits 必须覆盖数组截断、换序、hash 不匹配和 $P_Y$ 公开评值 opening 不匹配。上述任一测试均必须被拒绝。

对比实验给出 proof-scope 对齐表，说明各方法是否证明 full GAT forward、是否绑定 final logits、是否使用 full dataset、verifier 是否读取 witness、proof size 统计口径和 setup assumption。

### 6.3 正确性一致性测试

所有数据集执行 reference consistency test：

1. `reference_fp32_forward`：浮点参考模型前向；
2. `reference_quant_forward`：量化语义前向；
3. `secure_trace_forward`：证明系统 trace 前向；
4. `proof_public_output`：proof object 中公开输出。

验收条件：

- `reference_quant_forward_hash == secure_trace_forward_hash`；
- `secure_trace_forward_hash == proof_public_output_hash`；
- 所有 hash、配置标识和数据集标识进入 run manifest。

### 6.4 最终论文实验验收标准

论文结果只读取满足以下条件的 run：

1. `protocol_name == zkgat`；
2. `protocol_schema_id` 与正式协议模板一致；
3. `instance_schema_hash` 与当前数据集、模型结构、对象清单和次数登记一致；
4. `srs_type == trusted_srs`；
5. `srs_id` 与本地 SRS manifest 一致，当前七域大小均属于 manifest 目录；
6. `privacy_policy == model_and_trace_hidden`；
7. `input_privacy == public_graph_public_features`；
8. `output_mode == public_logits`；
9. `opening_mode == point_grouped_batch_opening`；
10. `binding_mode == zk_subproof_for_crpc`；
11. `proof_task == node_classification_full_gat`；
12. `private_plain_value_count == 0`；
13. `verifier_reads_witness == false`；
14. `tamper_tests_passed == true`；
15. `reference_consistency_passed == true`；
16. `security_budget_passed == true`，且实例化 `SecurityBudgetLedger` 的精确总优势不超过 manifest 声明的目标；
17. `proof_size_bytes` 必须等于第 3.5 节九个顶层 block 的规范序列化总字节数，并包含 $M_{pub}$ 中完整 $Y_{public}$；不统计 SRS、manifest、外部注册证明、日志、路径、时间戳和调试输出。

### 6.5 论文结果来源锁定

所有进入论文正文、附录和结果表的实验结果，必须由同一协议实现、同一 `protocol_schema_id`、同一 SRS manifest、同一证明路径和验证路径生成，并统一导出到 `results/paper_secure/`。同一数据集的 CPU / GPU 运行必须使用完全相同的 `instance_schema_hash`、模型 commitments、量化配置、公开输入和 $Y_{public}$；不同数据集分别登记各自的 `model_params`、`quant_config` 与 `instance_schema_hash`，不得错误要求跨数据集共享同一模型或量化参数。

每条结果至少记录 `protocol_name`、`protocol_schema_id`、`instance_schema_hash`、`srs_type`、`srs_id`、`kzg_assumption_profile`、`privacy_policy`、`input_privacy`、`output_mode`、`opening_mode`、`binding_mode`、`private_plain_value_count`、`security_budget_passed`、`reference_consistency_passed`、`tamper_tests_passed` 和 `verifier_reads_witness`。不满足任一验收条件的运行不进入论文正文、附录或结果表。
