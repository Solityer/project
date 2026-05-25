# GAT-zkGPT

## 0. 符号

### 0.1 模型结构

本协议证明一个可参数化的多层 GAT 前向关系。

1. 设注意力层总数为 $L$。其中前 $L-1$ 层是隐藏层，第 $L$ 层是最终输出层。

2. 对每个隐藏层 $\ell\in\{1,2,\ldots,L-1\}$，记该层隐藏注意力头数量为 $K_{hid}^{(\ell)}$，每个头的输出维度为 $d_h^{(\ell)}$。该层每个头完成注意力聚合后逐元素施加 ELU，随后把全部头的 ELU 后输出沿特征维拼接：

   $$d_{cat}^{(\ell)}=K_{hid}^{(\ell)}\cdot d_h^{(\ell)}.$$

3. 第 $\ell$ 个隐藏层的输入维度记为 $d_{in}^{(\ell)}$。其中 $d_{in}^{(1)}=d_{in}$，且对任意 $\ell\ge 2$，都有：

   $$d_{in}^{(\ell)}=d_{cat}^{(\ell-1)}.$$

4. 最终输出层第 $L$ 层仍由 $K_{out}$ 个输出注意力头组成。每个输出注意力头先完成注意力聚合，再加上该头自己的输出偏置向量。若 $K_{out}>1$，则全部输出头的后偏置输出按头维做算术平均，得到最终 logits。

5. 输出注意力头的激活函数固定为恒等映射，不施加 ELU。恒等映射的输入是已经加上该输出头偏置后的注意力聚合结果。

6. 对任意多层 GAT，证明者与验证者都必须按层序 $\ell=1,\ldots,L-1$ 逐层实例化隐藏层模板；第 $\ell-1$ 层的拼接输出 $H_{cat}^{(\ell-1)}$ 必须作为第 $\ell$ 层唯一输入表示。最终再实例化输出层模板。

7. 协议实例中的 $L,\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1},K_{out}$ 必须由公开元数据 $M_{pub}$ 唯一确定。

为避免公式中引入过多层指标，隐藏层模板统一采用如下局部简写：

- 隐藏层模板索引：$\ell\in\{1,2,\ldots,L-1\}$；
- 当前层第 $r$ 个隐藏头：$r\in\{0,1,\ldots,K_{hid}-1\}$；
- 当前隐藏层输入维度：$d_{in}:=d_{in}^{(\ell)}$；
- 当前隐藏层每头输出维度：$d_h:=d_h^{(\ell)}$；
- 当前隐藏层头数：$K_{hid}:=K_{hid}^{(\ell)}$；
- 当前隐藏层拼接维：$d_{cat}:=K_{hid}\cdot d_h$；
- 输出类别数：$C$；
- 输出层第 $s$ 个输出头：$s\in\{0,1,\ldots,K_{out}-1\}$。

凡正文中为了公式紧凑而省略的 $(\ell)$、$(\ell,r)$、$(out,s)$ 标记，只能在数学推导中省略；在协议接口、配置字段、承诺名、输入层布局、proof label、transcript tag 中都必须恢复完整指标。

### 0.2 有限域、多线性扩展、承诺、GKR、Lasso 与 Fiat--Shamir

记有限域为 $\mathbb F_p$，其中 $p$ 为大素数。所有证明内运算都在 $\mathbb F_p$ 上进行。

采用定点量化语义：若域元素 $x\in\mathbb F_p$ 的标准代表满足 $x>\frac{p-1}{2}$，则在实数语义下把它解释为负数 $x-p$。LeakyReLU、ELU、范围比较、指数查表、量化回缩均在该有符号域语义下定义。

本文采用如下证明后端：

1. 算术关系由 ML-friendly GKR 证明。算术关系包括加法、乘法、线性组合、状态机递推、逆元乘法校验、压缩绑定、矩阵乘法输出一致性之外的普通电路约束。

2. 线性层中的大规模矩阵乘法由专用矩阵乘法 sumcheck 证明。该协议使用多线性扩展，把 $C=A\cdot B$ 的随机点评值归约为 $A,B$ 的随机点评值。矩阵乘法证明的 batching、padding 与分组规则由 $matrix\_proof\_cfg\_id$ 固定。

3. 非算术关系由 Lasso lookup 证明。本文非算术关系包括：
   - $T_{LReLU}$ 查表；
   - $T_{ELU}$ 查表；
   - $T_{exp}$ 查表；
   - $T_{range}$ 范围关系；
   - 若 $quant\_cfg\_id$ 指定某个除法、重缩放或回缩为 advice-based rounding relation，则对应范围关系也进入 Lasso。

4. 中间变量、查表 query、rounding advice 与状态机 advice 统一视为输入层 witness 的组成部分，但不得一次性无条件承诺。本文采用分阶段输入层承诺：先承诺不依赖 Fiat--Shamir 压缩挑战的基础前向对象，再生成压缩挑战并承诺压缩对象，随后生成 PSQ 挑战并承诺 PSQ 状态机对象。GKR、Lasso 与矩阵乘法 sumcheck 最终共同归约到这些分段输入层承诺。

5. 随机预言机统一写作 $H_{FS}(\cdot)$。所有 Fiat--Shamir 挑战必须按本文固定顺序生成，禁止重排、回退、漏吸入或重复吸入。

#### 0.2.1 多线性扩展

对长度为 $2^m$ 的向量 $V:\{0,1\}^m\to\mathbb F_p$，其多线性扩展记为 $\widetilde V:\mathbb F_p^m\to\mathbb F_p$，满足：

$$\widetilde V(x)=\sum_{b\in\{0,1\}^m}\widetilde{eq}(x,b)V(b),$$

其中

$$\widetilde{eq}(x,b)=\prod_{t=1}^m\big((1-x_t)(1-b_t)+x_tb_t\big).$$

所有矩阵、张量、表端 query、lookup query 和电路输入层向量都先按本文展平规则展平，再 padding 到二次幂长度，并以多线性扩展形式参与承诺与 sumcheck。

#### 0.2.2 抽象承诺接口

本协议的承诺后端为 Hyrax-style MLE 承诺接口 $zkPC$。同一协议实例只使用该承诺后端。底层接口记为：

- $zkPC.KeyGen(1^\lambda)\to pp$；
- $zkPC.Commit(\widetilde V,pp,r)\to com_V$；
- $zkPC.Open(\widetilde V,r_V;z)\to (\widetilde V(z),\pi_{open})$；
- $zkPC.Verify(com_V,z,\widetilde V(z),\pi_{open},pp)\to\{0,1\}$。

普通 $Open/Verify$ 仅用于描述 Hyrax-style MLE commitment 的底层能力，不作为本协议对非公开 witness segment 的实际开放接口。本协议中所有非公开 witness segment 的实际 opening 均使用第 0.2.3 节定义的 $OpenMasked/VerifyMasked$ 接口；公开 $X_{pub}$ 由验证者本地评值，不需要承诺 opening。

模型参数承诺与静态特征表承诺在预处理阶段生成并跨证明复用；本次证明相关的基础中间变量、压缩中间变量、PSQ 状态机对象、lookup query 和 rounding advice 按第 2 章的分阶段顺序承诺。

#### 0.2.3 Zero-knowledge masking 规则

本文的证明目标是 zero-knowledge，而不只是 verifiable computation。所有非公开 segment，包括 $T_H,W,Aux_{base},Aux_{comp},Aux_{psq},L_{qry}$，都必须使用 hiding commitment randomness。所有由 GKR、Lasso、矩阵乘法 sumcheck 或 combine 协议导致的 witness MLE opening，都必须加入 masking polynomial / blinding evaluation。验证者只获得 masked evaluation 及其一致性证明，不获得未掩蔽的 witness 随机点评值。具体 masking 方式由 $proof\_backend\_id$ 与 $encoding\_id$ 固定。

本文在抽象接口中记：

- $zkPC.OpenMasked(\widetilde V,r_V;z)\to(\widehat V(z),\pi_{open}^{mask})$；
- $zkPC.VerifyMasked(com_V,z,\widehat V(z),\pi_{open}^{mask},pp)\to\{0,1\}$。

其中 $\widehat V(z)$ 表示按照 $proof\_backend\_id$ 固定 masking 规则得到的 masked evaluation，而不是未掩蔽的 $\widetilde V(z)$。后文凡涉及非公开 segment 的 opening，均默认使用 $OpenMasked/VerifyMasked$。

对于跨证明复用的静态承诺段 $T_H$ 与 $W$，$com_{T_H}$ 与 $com_W$ 可以复用，但每次证明中的 $OpenMasked$ 必须使用本次证明独立的 opening blinding / masking randomness。禁止在不同证明实例之间复用 masked opening 的随机性或可关联的 blinding 多项式。

### 0.3 图、局部子图、多图批处理与排序

- 全局节点总数：$N_{total}$；
- 当前证明实例中的总节点数：$N$；
- 当前证明实例中的总边数：$E$；
- 当前证明实例中的图个数：$G_{batch}$；
- 局部子图 / 批处理后的节点绝对编号序列：$I=(I_0,I_1,\ldots,I_{N-1})$；
- 边源索引序列：$src=(src(0),src(1),\ldots,src(E-1))$；
- 边目标索引序列：$dst=(dst(0),dst(1),\ldots,dst(E-1))$；
- 节点前缀指针：$node\_ptr=(\nu_0,\nu_1,\ldots,\nu_{G_{batch}})$，其中 $0=\nu_0<\nu_1<\cdots<\nu_{G_{batch}}=N$；
- 边前缀指针：$edge\_ptr=(\epsilon_0,\epsilon_1,\ldots,\epsilon_{G_{batch}})$，其中 $0=\epsilon_0<\epsilon_1<\cdots<\epsilon_{G_{batch}}=E$。

对节点 $i$，定义 $node\_gid(i)=g$，其中 $\nu_g\le i<\nu_{g+1}$。对边 $k$，定义 $edge\_gid(k)=g$，其中 $\epsilon_g\le k<\epsilon_{g+1}$。合法边必须满足：

$$src(k),dst(k)\in\{0,1,\ldots,N-1\},\quad node\_gid(src(k))=edge\_gid(k),\quad node\_gid(dst(k))=edge\_gid(k).$$

若 $G_{batch}=1$，公共输入仍必须显式写成 $node\_ptr=(0,N)$、$edge\_ptr=(0,E)$。

边序列必须满足两级稳定排序：

1. 先按 $edge\_gid(k)$ 非降序排列；
2. 在每个图块内部按 $dst(k)$ 稳定非降序排列。

等价地，若 $0\le k_1<k_2\le E-1$，则必须满足：

$$edge\_gid(k_1)<edge\_gid(k_2)\quad\text{或}\quad\big(edge\_gid(k_1)=edge\_gid(k_2)\ \text{且}\ dst(k_1)\le dst(k_2)\big).$$

若原始输入不满足该条件，则见证生成前先做图内节点重编号、局部子图抽取、按图分块稳定排序、每个图块内部按 $dst$ 稳定排序。验证者只接收已经按规范生成的公开序列。

#### 0.3.1 batch-aware 目标组选择器

由于公开边序列已按图分块并在图内按 $dst$ 排序，定义：

$$Q_{new}^{edge}[k]=\begin{cases}1,&k=0,\\1,&1\le k<E\ \text{且}\ dst(k)\ne dst(k-1),\\0,&\text{其他有效边位置},\\0,&k\ge E,\end{cases}$$

$$Q_{end}^{edge}[k]=\begin{cases}1,&k=E-1,\\1,&0\le k<E-1\ \text{且}\ dst(k+1)\ne dst(k),\\0,&\text{其他有效边位置},\\0,&k\ge E,\end{cases}$$

$$Q_{graph\_new}^{edge}[k]=\begin{cases}1,&k=0,\\1,&1\le k<E\ \text{且}\ edge\_gid(k)\ne edge\_gid(k-1),\\0,&\text{其他有效边位置},\\0,&k\ge E,\end{cases}$$

$$Q_{graph\_end}^{edge}[k]=\begin{cases}1,&k=E-1,\\1,&0\le k<E-1\ \text{且}\ edge\_gid(k+1)\ne edge\_gid(k),\\0,&\text{其他有效边位置},\\0,&k\ge E.\end{cases}$$

batch-aware 选择器定义为：

$$Q_{grp\_new}^{edge}=1-(1-Q_{graph\_new}^{edge})(1-Q_{new}^{edge}),$$

$$Q_{grp\_end}^{edge}=1-(1-Q_{graph\_end}^{edge})(1-Q_{end}^{edge}).$$

后文所有最大值计数状态机、PSQ 状态机和图内 softmax 分组约束都只使用 $Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$。即使 $G_{batch}=1$，协议仍使用上述 batch-aware 选择器。

### 0.4 静态表与量化尺度

静态表包括：

1. 全局特征表：$T_H\in\mathbb F_p^{N_{total}\times d_{in}^{(1)}}$；
2. LeakyReLU 表：$T_{LReLU}\subseteq\mathbb F_p\times\mathbb F_p$；
3. ELU 表：$T_{ELU}\subseteq\mathbb F_p\times\mathbb F_p$；
4. 指数表：$T_{exp}\subseteq\mathbb F_p\times\mathbb F_p$；
5. 范围表：$T_{range}=\{0,1,2,\ldots,2^B-1\}$。

所有范围表与区间检查均要求 $2^B<p$，并且范围语义按域元素的标准非负代表解释。对形如 $r-v$ 的 range query，$r-v$ 表示整数语义下的区间剩余量，再编码进 $\mathbb F_p$；不得依赖模 $p$ 环绕来满足范围关系。

为避免多值歧义，要求：

- 对任意 $x$，$T_{LReLU}$ 中至多存在一个 $y$ 使得 $(x,y)\in T_{LReLU}$；
- 对任意 $x$，$T_{ELU}$ 中至多存在一个 $y$ 使得 $(x,y)\in T_{ELU}$；
- 对任意 $u$，$T_{exp}$ 中至多存在一个 $v$ 使得 $(u,v)\in T_{exp}$。

与 softmax 链路有关的量化尺度全部视为公开参数：

- 差分尺度：$S_\Delta$；
- 指数输出尺度：$S_{exp}$；
- 逆元见证尺度：$S_{inv}$；
- 归一化权重尺度：$S_\alpha$；
- 隐藏层聚合前尺度：$S_{agg,pre}$；
- 隐藏层 ELU 后尺度：$S_{agg}$；
- 拼接后表示尺度：$S_{cat}$；
- 输出层聚合前尺度：$S_{out,pre}$；
- 输出头加偏置后尺度：$S_{out,head}$；
- 最终 logits 尺度：$S_{out}$。

舍入方式、截断方式、乘法后的 rescale 规则、域元素的有符号解释规则全部由 $quant\_cfg\_id$ 唯一确定。

输出层平均语义定义如下：每个输出头 $s$ 的聚合前输出先根据公开规则对齐到 $S_{out,head}$：

$$\widetilde Y^{pre(s)}=Rescale_{out}(Y^{pre(s)};S_{out,pre}\to S_{out,head}),$$

再加偏置：

$$Y^{(s)}=\widetilde Y^{pre(s)}+b^{(out,s)}.$$

最终多头平均固定在共同尺度 $S_{out,head}$ 上执行，且 $S_{out}=S_{out,head}$：

$$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)},\qquad Y_{i,c}=K_{out}^{-1}\cdot Y_{i,c}^{sum}.$$

禁止在最终平均后额外引入第二个平均后重缩放子步骤。

### 0.5 模型参数

#### 0.5.1 隐藏层参数族

对任意隐藏层 $\ell\in\{1,\ldots,L-1\}$ 与隐藏头 $r\in\{0,\ldots,K_{hid}^{(\ell)}-1\}$，定义：

- 投影矩阵 $W^{(\ell,r)}\in\mathbb F_p^{d_{in}^{(\ell)}\times d_h^{(\ell)}}$；
- 源方向注意力向量 $a_{src}^{(\ell,r)}\in\mathbb F_p^{d_h^{(\ell)}}$；
- 目标方向注意力向量 $a_{dst}^{(\ell,r)}\in\mathbb F_p^{d_h^{(\ell)}}$。

#### 0.5.2 输出层参数

对输出头 $s\in\{0,\ldots,K_{out}-1\}$，定义：

- 输出投影矩阵 $W^{(out,s)}\in\mathbb F_p^{d_{in}^{(L)}\times C}$；
- 输出层源方向注意力向量 $a_{src}^{(out,s)}\in\mathbb F_p^C$；
- 输出层目标方向注意力向量 $a_{dst}^{(out,s)}\in\mathbb F_p^C$；
- 输出层偏置向量 $b^{(out,s)}\in\mathbb F_p^C$。

其中 $d_{in}^{(L)}=d_{cat}^{(L-1)}$。注意力向量采用源/目标分拆形式：

$$\mathbf a^{(out,s)\top}[u\|v]=a_{src}^{(out,s)\top}u+a_{dst}^{(out,s)\top}v.$$

所有模型参数在多次证明中固定，统一进入模型承诺 $com_W$ 和模型验证键 $VK_{model}^{zkGPT}$。模型承诺按层序、头序和输出头序固定排列。

### 0.6 全部中间变量

#### 0.6.1 原始特征

对每个局部节点 $i\in\{0,\ldots,N-1\}$ 与输入维 $j\in\{0,\ldots,d_{in}^{(1)}-1\}$，定义：

$$H_{i,j}=T_H[I_i,j].$$

于是 $H\in\mathbb F_p^{N\times d_{in}^{(1)}}$。

#### 0.6.2 单个隐藏层模板中第 $r$ 个注意力头的节点域变量

以下对象对应固定隐藏层 $\ell$ 的单层模板。对每个 $r$，定义：

1. 投影结果：

   $$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}.$$

2. 节点域源注意力：

   $$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)}.$$

3. 节点域目标注意力：

   $$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}.$$

4. 节点域压缩特征。令 $\xi^{(\ell,r)}$ 为 transcript 生成的压缩挑战：

   $$H_i^{\star(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}(\xi^{(\ell,r)})^j.$$

5. 节点域组最大值：

   $$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}.$$

6. 节点域分母：

   $$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}.$$

7. 节点域逆元：

   $$inv_i^{(r)}=(Sum_i^{(r)})^{-1}.$$

8. 聚合前隐藏矩阵：

   $$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}.$$

9. 聚合前压缩特征：

   $$H_{agg,pre,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,pre,i,j}^{(r)}(\xi^{(\ell,r)})^j.$$

10. ELU 后隐藏矩阵：

    $$H_{agg,i,j}^{(r)}=ELU(H_{agg,pre,i,j}^{(r)}).$$

11. ELU 后压缩特征：

    $$H_{agg,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,i,j}^{(r)}(\xi^{(\ell,r)})^j.$$

#### 0.6.3 单个隐藏层模板中第 $r$ 个注意力头的边域变量

对每个边 $k$，定义：

1. 源注意力广播：

   $$E_{src,k}^{edge(r)}=E_{src,src(k)}^{(r)}.$$

2. 目标注意力广播：

   $$E_{dst,k}^{edge(r)}=E_{dst,dst(k)}^{(r)}.$$

3. 源压缩特征广播：

   $$H_{src,k}^{\star,edge(r)}=H_{src(k)}^{\star(r)}.$$

4. 聚合前压缩特征广播：

   $$H_{agg,pre,k}^{\star,edge(r)}=H_{agg,pre,dst(k)}^{\star(r)}.$$

5. 聚合后压缩特征广播：

   $$H_{agg,k}^{\star,edge(r)}=H_{agg,dst(k)}^{\star(r)}.$$

6. 线性打分：

   $$S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)}.$$

7. LeakyReLU 后打分：

   $$Z_k^{(r)}=LReLU(S_k^{(r)}).$$

8. 最大值广播：

   $$M_k^{edge(r)}=M_{dst(k)}^{(r)}.$$

9. 非负差分：

   $$\Delta_k^{+(r)}=M_k^{edge(r)}-Z_k^{(r)}.$$

10. 指数输出：

    $$U_k^{(r)}=ExpMap(\Delta_k^{+(r)}).$$

11. 分母广播：

    $$Sum_k^{edge(r)}=Sum_{dst(k)}^{(r)}.$$

12. 逆元广播：

    $$inv_k^{edge(r)}=inv_{dst(k)}^{(r)}.$$

13. 归一化权重：

    $$\alpha_k^{(r)}=U_k^{(r)}\cdot inv_k^{edge(r)}.$$

14. 聚合前压缩加权特征：

    $$\widehat v_{pre,k}^{\star(r)}=\alpha_k^{(r)}H_{src,k}^{\star,edge(r)}.$$

#### 0.6.4 单个隐藏层模板的拼接阶段变量

定义拼接结果 $H_{cat}\in\mathbb F_p^{N\times d_{cat}}$：

$$H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}.$$

令 $\xi_{cat}^{(\ell)}$ 为拼接压缩挑战，定义：

$$H_{cat,i}^{\star}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}(\xi_{cat}^{(\ell)})^m.$$

#### 0.6.5 输出层变量

令 $\xi_{out}$ 为输出层类别压缩挑战。对每个输出头 $s$，定义：

1. 输出投影：

   $$Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}^{(L-1)}-1}H_{cat,i,m}^{(L-1)}W_{m,c}^{(out,s)}.$$

2. 输出层源 / 目标注意力：

   $$E_{src,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{src,c}^{(out,s)},$$

   $$E_{dst,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{dst,c}^{(out,s)}.$$

3. 源 / 目标广播：

   $$E_{src,k}^{edge(out,s)}=E_{src,src(k)}^{(out,s)},\qquad E_{dst,k}^{edge(out,s)}=E_{dst,dst(k)}^{(out,s)}.$$

4. 输出层打分、LeakyReLU、最大值、差分、指数、分母、逆元、权重：

   $$S_k^{(out,s)}=E_{src,k}^{edge(out,s)}+E_{dst,k}^{edge(out,s)},$$

   $$Z_k^{(out,s)}=LReLU(S_k^{(out,s)}),$$

   $$M_i^{(out,s)}=\max\{Z_k^{(out,s)}\mid dst(k)=i\},$$

   $$M_k^{edge(out,s)}=M_{dst(k)}^{(out,s)},$$

   $$\Delta_k^{+(out,s)}=M_k^{edge(out,s)}-Z_k^{(out,s)},$$

   $$U_k^{(out,s)}=ExpMap(\Delta_k^{+(out,s)}),$$

   $$Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)},$$

   $$Sum_k^{edge(out,s)}=Sum_{dst(k)}^{(out,s)},$$

   $$inv_i^{(out,s)}=(Sum_i^{(out,s)})^{-1},\qquad inv_k^{edge(out,s)}=inv_{dst(k)}^{(out,s)},$$

   $$\alpha_k^{(out,s)}=U_k^{(out,s)}inv_k^{edge(out,s)}.$$

5. 输出投影类别压缩与源广播：

   $$Y_i'^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}\xi_{out}^c,$$

   $$Y_k'^{\star,edge(s)}=Y_{src(k)}'^{\star(s)}.$$

6. 压缩加权边特征：

   $$\widehat y_k^{\star(s)}=\alpha_k^{(out,s)}Y_k'^{\star,edge(s)}.$$

7. 输出层聚合前输出与压缩：

   $$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)},$$

   $$Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{pre(s)}\xi_{out}^c.$$

8. 尺度对齐、加偏置与头输出：

   $$\widetilde Y_{i,c}^{pre(s)}=Rescale_{out}(Y_{i,c}^{pre(s)};S_{out,pre}\to S_{out,head}),$$

   $$Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}.$$

9. 头输出压缩：

   $$Y_i^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{(s)}\xi_{out}^c.$$

10. 最终多头平均输出：

    $$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)},\qquad Y_{i,c}=K_{out}^{-1}Y_{i,c}^{sum}.$$

11. 最终输出压缩：

    $$Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c.$$

### 0.7 工作域、选择器、辅助公开列与展平规则

#### 0.7.1 逻辑工作域

本文采用七类逻辑工作域组织见证、约束与输入层布局：

1. 特征检索域：$\mathcal D_{FH}$；
2. 边域：$\mathcal D_{edge}$；
3. 输入共享维域：$\mathcal D_{in}$；
4. 隐藏层单头共享维域：$\mathcal D_{d_h}$；
5. 拼接共享维域：$\mathcal D_{cat}$；
6. 输出层类别共享维域：$\mathcal D_C$；
7. 节点域：$\mathcal D_N$。

这些域表示被展平、padding 后进入 MLE/GKR/Lasso 的逻辑数组。每个逻辑域的长度均取二次幂：

- $n_{FH}\ge\max\{N_{total}d_{in}^{(1)},Nd_{in}^{(1)}\}$；
- $n_{edge}\ge\max\{E,N,Nd_h^{max},NC\}$；
- $n_{in}\ge d_{in}^{max}$；
- $n_{d_h}\ge d_h^{max}$；
- $n_{cat}\ge d_{cat}^{max}$；
- $n_C\ge C$；
- $n_N\ge N$。

每个长度均 padding 到二次幂；对应 MLE 变量数分别为 $\log n_{FH},\log n_{edge},\log n_{in},\log n_{d_h},\log n_{cat},\log n_C,\log n_N$。

Lookup 表不并入上述七类逻辑工作域。本文单独定义 Lasso 表域与 query 域：$n_{LReLU}^{tbl}$、$n_{ELU}^{tbl}$、$n_{exp}^{tbl}$、$n_{range}^{struct}$ 以及各 lookup family 的 $n_{qry}^{(\mathcal L)}$。其中 $T_{range}$ 按 Lasso structured table 处理，不把 $2^B$ 作为 $\mathcal D_{edge}$ 的长度下界。

#### 0.7.2 公共有效区选择器

有效区选择器保留为公共输入层向量，并进入 GKR 算术约束：

$$Q_{edge}^{valid}[k]=\begin{cases}1,&0\le k<E,\\0,&E\le k<n_{edge},\end{cases}\qquad Q_N[i]=\begin{cases}1,&0\le i<N,\\0,&N\le i<n_N.\end{cases}$$

对每个隐藏层 $\ell$：

$$Q_{in}^{valid,(\ell)}[m]=\begin{cases}1,&0\le m<d_{in}^{(\ell)},\\0,&d_{in}^{(\ell)}\le m<n_{in},\end{cases}$$

$$Q_{d_h}^{valid,(\ell)}[j]=\begin{cases}1,&0\le j<d_h^{(\ell)},\\0,&d_h^{(\ell)}\le j<n_{d_h},\end{cases}$$

$$Q_{cat}^{valid,(\ell)}[m]=\begin{cases}1,&0\le m<d_{cat}^{(\ell)},\\0,&d_{cat}^{(\ell)}\le m<n_{cat}.\end{cases}$$

输出层输入共享维有效区选择器：

$$Q_{cat}^{valid,(out)}[m]=\begin{cases}1,&0\le m<d_{in}^{(L)},\\0,&d_{in}^{(L)}\le m<n_{cat}.\end{cases}$$

输出层类别有效区选择器：

$$Q_C^{valid}[c]=\begin{cases}1,&0\le c<C,\\0,&C\le c<n_C.\end{cases}$$

$Q_{cat}^{valid,(out)}$ 与 $Q_{cat}^{valid,(L-1)}$ 必须作为两个不同 tag 的公共对象处理，即使二者取值相同也不得在序列化和 transcript 中合并。

#### 0.7.3 公共枚举列与辅助列

定义公共枚举列：

$$Idx_N[i]=i,\quad Idx_{in}[m]=m,\quad Idx_{d_h}[j]=j,\quad Idx_{cat}[m]=m,\quad Idx_C[c]=c.$$

定义特征检索辅助列。对 $u=v\cdot d_{in}^{(1)}+j$：

$$Row_{feat}^{tbl}[u]=v,\qquad Col_{feat}^{tbl}[u]=j.$$

对 $q=i\cdot d_{in}^{(1)}+j$：

$$Row_{feat}^{qry}[q]=i,\qquad Col_{feat}^{qry}[q]=j,\qquad I_{feat}^{qry}[q]=I_i.$$

这些辅助列作为公共输入层向量进入 transcript 和 GKR 约束。

#### 0.7.4 公开路由 wiring 与稀疏一致性关系

本文不把源路由、目标路由或特征检索物化为稠密 $E\times N$ 或 $(Nd_{in})\times(N_{total}d_{in})$ 矩阵，也不把它们作为普通 dense matrix multiplication 进入矩阵乘法 sumcheck。由于 $src,dst,I$ 全部是公开输入，路由关系在 ML-friendly GKR 电路中以公开 wiring equality 的形式检查。

对任意节点向量 $V^{node}\in\mathbb F_p^N$，源广播对象 $V^{edge}_{src}\in\mathbb F_p^E$ 的正式约束为：

$$Q_{edge}^{valid}[k]\cdot\big(V^{edge}_{src}[k]-V^{node}[src(k)]\big)=0.$$

目标广播对象 $V^{edge}_{dst}\in\mathbb F_p^E$ 的正式约束为：

$$Q_{edge}^{valid}[k]\cdot\big(V^{edge}_{dst}[k]-V^{node}[dst(k)]\big)=0.$$

对特征检索，$T_H$ 作为静态承诺表进入 $VK_{static}^{zkGPT}$，其承诺记为 $com_{T_H}$。原始特征检索关系为：

$$Q_N[i]\cdot Q_{in}^{valid,(1)}[j]\cdot\big(H_{i,j}-T_H[I_i,j]\big)=0.$$

上述三类关系均作为 GKR arithmetic / wiring consistency 关系证明，且不进入 dense 矩阵乘法 sumcheck。公开 wiring 索引表由公共输入唯一重建，证明者不得提交可变 routing wiring 索引。

#### 0.7.5 展平规则

对任意矩阵 $M\in\mathbb F_p^{r\times c}$，采用行优先展平：

$$\operatorname{flat}_{r,c}(i,j)=i\cdot c+j.$$

特别地：

- $N\times d_{in}$ 矩阵使用 $i\cdot d_{in}+j$；
- $N\times d_h$ 矩阵使用 $i\cdot d_h+j$；
- $N\times d_{cat}$ 矩阵使用 $i\cdot d_{cat}+m$；
- $N\times C$ 矩阵使用 $i\cdot C+c$。

所有向量和矩阵展平后都 padding 到对应二次幂长度，padding 位置为零，并由有效区选择器屏蔽。

#### 0.7.6 矩阵乘法 sumcheck 关系

设 $A\in\mathbb F_p^{n\times m}$，$B\in\mathbb F_p^{m\times k}$，$C=A\cdot B\in\mathbb F_p^{n\times k}$。三者每个维度均 padding 到二次幂。令 $r\in\mathbb F_p^{\log n}$，$r'\in\mathbb F_p^{\log k}$，则：

$$\widetilde C(r\|r')=\sum_{c\in\{0,1\}^{\log m}}\widetilde A(r\|c)\widetilde B(c\|r').$$

证明者和验证者对上式运行 sumcheck，将 $\widetilde C$ 的随机点评值归约为 $\widetilde A,\widetilde B$ 的随机点评值。

对同形状或可统一 padding 的多组矩阵乘法 $C_t=A_tB_t$，$t\in\{0,\ldots,M-1\}$，将 $A_t,B_t,C_t$ 沿批维拼成三维张量 $A\in\mathbb F_p^{M\times n\times m}$、$B\in\mathbb F_p^{M\times m\times k}$、$C\in\mathbb F_p^{M\times n\times k}$，使用 batched 关系：

$$\widetilde C(u\|r\|r')=\sum_{c\in\{0,1\}^{\log m}}\widetilde A(u\|r\|c)\widetilde B(u\|c\|r').$$

这是本文对多个隐藏头、多个输出头、多个挑战压缩内积以及同形状线性层做 batched matrix multiplication / circuit squeeze 时使用的矩阵证明规则。源路由、目标路由和特征检索不属于这里的 dense 矩阵乘法对象。

#### 0.7.7 输入层布局、分阶段 advice 与 lookup query 绑定

本文把所有需要被证明系统访问的值合并为统一逻辑输入层，但该输入层不是一次性承诺的单一向量，而是由七个固定 segment 组成：

$$V_{in}=X_{pub}\|T_H\|W\|Aux_{base}\|Aux_{comp}\|Aux_{psq}\|L_{qry}.$$

其中：

- $X_{pub}$：公共输入向量，包括 $I,src,dst,node\_ptr,edge\_ptr$、选择器、公开拓扑辅助列、公开枚举列、padding 规则摘要与公开 routing wiring 索引；
- $T_H$：静态特征表 segment，由 $com_{T_H}$ 绑定；
- $W$：模型参数 segment，由 $com_W$ 绑定；
- $Aux_{base}$：不依赖 Fiat--Shamir 压缩挑战的基础前向对象，包括 $H,H',E_{src},E_{dst},S,Z,M,\Delta^+,U,Sum,inv,\alpha,H_{agg,pre},H_{agg},H_{cat},Y',Y^{pre},\widetilde Y^{pre},Y^{(s)}$ 等；
- $Aux_{comp}$：依赖 $\xi^{(\ell,r)},\xi_{cat}^{(\ell)},\xi_{out}$ 的压缩对象，包括 $H^\star,H_{agg,pre}^\star,H_{agg}^\star,H_{cat}^\star,Y'^\star,Y^{pre,\star},Y^\star$；
- $Aux_{psq}$：依赖 $\lambda_{psq}$ 的状态机对象，包括 $w_{psq},T_{psq},T_{psq}^{edge},PSQ,C_{max},s_{max}$；
- $L_{qry}$：lookup query segment。对二元查表，$L_{qry}$ 存储未压缩的 raw query columns $(x_q,y_q)$；对范围查表，$L_{qry}$ 存储 raw scalar range query。随机线性压缩后的 $Query_{\mathcal L}=x_q+\eta_{\mathcal L}y_q$ 不在 $com_{qry}$ 之前预先承诺，而是在 Lasso 阶段由已承诺 raw columns 和挑战 $\eta_{\mathcal L}$ 派生。

所有 segment、family、offset、长度、padding 长度、MLE 变量数和 serializer tag 必须由 $Layout_{in}$ 固定。GKR、Lasso 与矩阵乘法 sumcheck 最终只归约到上述 segment 的 masked opening；不得在证明过程中临时增加未声明 segment。

### 0.8 填充、零分母、Softmax 可行性与预处理对象

#### 0.8.1 Padding 规则

对任意逻辑数组，真实长度为 $L_{real}$、padding 后长度为 $n$。对所有 $t\in\{L_{real},\ldots,n-1\}$：

- 见证值补零；
- 表值补零；
- 查询值补零；
- advice 值补零；
- lookup query 补零；
- 有效区选择器为零；
- 状态机在填充区保持常值或由有效区选择器屏蔽，具体形式必须由 $padding\_rule\_id$ 固定。

#### 0.8.2 零分母冲突

对任意 Lasso lookup 子系统，查表挑战和压缩挑战必须在对应 witness commitment 固定后再生成。若采用随机线性压缩 query，则挑战不得落入导致分母或压缩碰撞的坏集合。transcript 必须在吸入 table id、query family id、query commitment 后再生成对应 lookup challenge。

#### 0.8.3 Softmax 可行性

每个有效节点至少有一条入边：

$$\#\{k\mid dst(k)=i\}\ge 1\quad\forall i\in\{0,\ldots,N-1\}.$$

若原始图不满足该条件，必须在公开预处理阶段显式加入自环后才允许进入协议。

同时要求：

1. 合法输入下实数域最大分母严格小于模数 $p$；
2. 量化后最小聚合分母至少为 $1$。

这样才能保证所有隐藏层 $Sum_i^{(r)}\ne0$ 与输出层 $Sum_i^{(out,s)}\ne0$。

#### 0.8.4 预处理对象

参数生成阶段形成以下预处理对象：

- $pp$ 与承诺验证参数；
- 模型参数承诺 $com_W$ 与 $VK_{model}^{zkGPT}$；
- 静态表描述、$com_{T_H}$ 与 Lasso 表预处理对象；
- $T_{range}$ 的 structured Lasso decomposition / preprocessing；
- 二元 lookup table 的随机线性压缩规则与 compressed-table preprocessing 规则；
- GKR 电路结构；
- batched matrix multiplication 证明结构；
- MLE 展平布局、七段 $Layout_{in}$ offset 表、padding 表；
- $\widetilde{eq}$ 评值生成计划；
- sumcheck 预处理参数；
- commitment 预处理参数。

单次证明与验证均引用由 $protocol\_id$ 和各配置标识绑定的预处理对象。

#### 0.8.5 公开元数据与版本元数据字段表

证明对象必须携带公开元数据 $M_{pub}$，字段固定为：

1. $protocol\_id$：本文协议版本与 transcript 规则版本；
2. $proof\_backend\_id$：固定为 zkGPT-style，即 ML-friendly GKR + Lasso + matrix-mul sumcheck + MLE commitment；
3. $model\_arch\_id$：绑定 $L,\{K_{hid}^{(\ell)},d_h^{(\ell)}\},K_{out}$，隐藏层 ELU、输出头加偏置、最终输出按头平均等结构语义；
4. $model\_param\_id$：绑定 $com_W$ 与模型参数承诺顺序；
5. $static\_table\_id$：绑定静态特征表承诺 $com_{T_H}$、$T_{LReLU},T_{ELU},T_{exp},T_{range}$ 的表编码与 Lasso table 配置；
6. $quant\_cfg\_id$：绑定全部量化尺度、舍入规则、重缩放规则与有符号解释规则；
7. $domain\_cfg$：固定为 $(n_{FH},n_{edge},n_{in},n_{d_h},n_{cat},n_C,n_N)$；
8. $dim\_cfg$：固定为 $(G_{batch},N,E,N_{total},node\_ptr,edge\_ptr,L,\{d_{in}^{(\ell)}\}_{\ell=1}^{L},\{d_h^{(\ell)}\}_{\ell=1}^{L-1},\{d_{cat}^{(\ell)}\}_{\ell=1}^{L-1},C,B,K_{out})$；
9. $encoding\_id$：域元素字节序、承诺编码、proof object 编码、transcript 吸入顺序；
10. $padding\_rule\_id$：padding 与选择器重建规则；
11. $circuit\_layout\_id$：输入层布局、$Aux_{base}/Aux_{comp}/Aux_{psq}/L_{qry}$ offset、GKR 分层布局、squeeze 规则；
12. $lasso\_cfg\_id$：Lasso 表、query family、range decomposition 参数；
13. $matrix\_proof\_cfg\_id$：矩阵乘法 sumcheck 与 batched matrix multiplication 配置。

验证者必须从 $\pi_{GAT}$ 中按固定顺序解析这些字段，并纳入 transcript 与一致性检查。

#### 0.8.6 大图稀疏执行与批处理假设

1. 注意力打分、softmax、PSQ、路由与 lookup 全部仅在公开边集 $E$ 上展开，禁止改写为全图稠密两两注意力。

2. 边域承载边相关 witness。边相关对象必须以边并行方式生成，并进入 $Aux_{base}$、$Aux_{comp}$、$Aux_{psq}$ 或 $L_{qry}$；节点域和共享维域对象只作为聚合、路由和绑定接口存在。

3. 排序、自环和局部子图抽取属于公开预处理，不作为额外证明关系写入协议。

4. 多图 batch 中，softmax 分组、最大值唯一性、PSQ 状态机与路由总和必须在各图内部成立；禁止跨图归一化。

5. 本协议不引入证明级分块语义。内部 witness 生成方式不得改变最终输入层承诺、proof object 布局或 transcript 语义。

## 1. 参数生成

### 1.1 输入

参数生成算法输入为：

- 安全参数 $\lambda$；
- 有限域模数 $p$；
- 局部子图规模上界 $(N,E)$；
- 全局节点数 $N_{total}$；
- 当前证明实例图个数 $G_{batch}$；
- 节点 / 边前缀指针 $node\_ptr,edge\_ptr$；
- 总层数 $L$；
- 各隐藏层结构 $\{K_{hid}^{(\ell)},d_h^{(\ell)}\}_{\ell=1}^{L-1}$；
- 各层输入维 $\{d_{in}^{(\ell)}\}_{\ell=1}^{L}$，其中 $d_{in}^{(1)}=d_{in}$ 且 $d_{in}^{(\ell+1)}=d_{cat}^{(\ell)}$；
- 输出类别数 $C$；
- 输出层头数 $K_{out}$；
- 范围检查位宽 $B$；
- 静态表 $T_H,T_{LReLU},T_{ELU},T_{exp},T_{range}$；
- 隐藏层参数族 $\{W^{(\ell,r)},a_{src}^{(\ell,r)},a_{dst}^{(\ell,r)}\}_{\ell,r}$；
- 输出层参数族 $\{W^{(out,s)},a_{src}^{(out,s)},a_{dst}^{(out,s)},b^{(out,s)}\}_s$；
- 量化配置 $quant\_cfg\_id$；
- GKR、Lasso、矩阵乘法 sumcheck 与承诺后端配置。

### 1.2 输出

参数生成算法输出：

- 公共参数 $pp$；
- 模型参数承诺 $com_W$；
- 模型验证键 $VK_{model}^{zkGPT}$；
- 静态表验证键 $VK_{static}^{zkGPT}$，其中包含静态特征表承诺 $com_{T_H}$ 以及 lookup 表描述；
- Lasso 表配置与预处理对象 $VK_{Lasso}$；
- GKR 电路描述 $C_{GAT}$；
- 矩阵乘法 sumcheck 描述 $C_{matmul}$；
- 输入层布局描述 $Layout_{in}$；
- 七类逻辑工作域长度 $(n_{FH},n_{edge},n_{in},n_{d_h},n_{cat},n_C,n_N)$；
- 公开拓扑对象重建规则；
- transcript 与 serializer 规则。

### 1.3 工作域

采用全局共享域定义：

$$d_{in}^{max}=\max\{d_{in}^{(1)},\ldots,d_{in}^{(L)}\},$$

$$d_h^{max}=\max\{d_h^{(1)},\ldots,d_h^{(L-1)}\},$$

$$d_{cat}^{max}=\max\{d_{cat}^{(1)},\ldots,d_{cat}^{(L-1)}\}.$$

在每个具体隐藏层 $\ell$：

- 输入共享维只占用 $\mathcal D_{in}$ 前 $d_{in}^{(\ell)}$ 个有效位置；
- 单头输出维只占用 $\mathcal D_{d_h}$ 前 $d_h^{(\ell)}$ 个有效位置；
- 拼接维只占用 $\mathcal D_{cat}$ 前 $d_{cat}^{(\ell)}$ 个有效位置；
- 输出层输入共享维只占用 $\mathcal D_{cat}$ 前 $d_{in}^{(L)}=d_{cat}^{(L-1)}$ 个位置；
- 其余位置按 padding 规则补零。

#### 1.3.1 特征检索域

取最小二次幂长度：

$$n_{FH}\ge \max\{N_{total}d_{in}^{(1)},Nd_{in}^{(1)}\}.$$

#### 1.3.2 边域

取最小二次幂长度：

$$n_{edge}\ge\max\{E,N,Nd_h^{max},NC\}.$$

$T_{LReLU},T_{ELU},T_{exp},T_{range}$ 使用 Lasso 表域和 lookup query 域，不参与 $n_{edge}$ 的取值。尤其是 $T_{range}$ 按 structured table 处理，不能把 $2^B$ 并入边域。

#### 1.3.3 输入共享维域

取最小二次幂长度 $n_{in}\ge d_{in}^{max}$。

#### 1.3.4 隐藏层单头共享维域

取最小二次幂长度 $n_{d_h}\ge d_h^{max}$。

#### 1.3.5 拼接共享维域

取最小二次幂长度 $n_{cat}\ge d_{cat}^{max}$。

#### 1.3.6 输出层类别共享维域

取最小二次幂长度 $n_C\ge C$。

#### 1.3.7 节点域

取最小二次幂长度 $n_N\ge N$。

### 1.4 zkGPT 后端初始化

运行：

$$pp\leftarrow zkPC.KeyGen(1^\lambda).$$

构造以下协议后端对象：

1. GKR 电路 $C_{GAT}$。其逻辑输入层为 $V_{in}=X_{pub}\|T_H\|W\|Aux_{base}\|Aux_{comp}\|Aux_{psq}\|L_{qry}$，电路输出为最终 logits $Y$ 及必要的公开 claim。普通算术约束、公开 routing wiring、特征检索 indexed access 与 lookup query consistency 全部进入该电路。

2. Lasso lookup 系统 $VK_{Lasso}$。包括：
   - range table $T_{range}$；
   - LeakyReLU table $T_{LReLU}$；
   - ELU table $T_{ELU}$；
   - exp table $T_{exp}$；
   - 若量化配置引入其他固定表，也必须由 $static\_table\_id$ 绑定。

3. 矩阵乘法 sumcheck 系统 $C_{matmul}$。包括所有隐藏层投影、输出层投影、注意力向量点积与挑战压缩内积等可批处理矩阵乘法。特征检索、源路由和目标路由固定作为 GKR public wiring / indexed access，不进入 $C_{matmul}$。

4. 输入层布局 $Layout_{in}$。该布局固定 $X_{pub},T_H,W,Aux_{base},Aux_{comp},Aux_{psq},L_{qry}$ 中每个 family 的 offset、长度、padding、tag 与 MLE 变量数。

5. Fiat--Shamir 域分离标签。必须至少包含：
   - `gat-zkgpt/meta`；
   - `gat-zkgpt/model-commit`；
   - `gat-zkgpt/static-table`；
   - `gat-zkgpt/input-commit`；
   - `gat-zkgpt/lasso`；
   - `gat-zkgpt/matmul`；
   - `gat-zkgpt/gkr`；
   - `gat-zkgpt/open`。

本协议只使用上述 GKR、Lasso、矩阵乘法 sumcheck 与 MLE commitment 对象。

### 1.5 静态表与模型承诺

#### 1.5.1 静态表编码与静态承诺

全局特征表按行优先展平：

$$Flat(T_H)[v\cdot d_{in}^{(1)}+j]=T_H[v,j].$$

将 $Flat(T_H)$ padding 到二次幂长度，定义 MLE $\widetilde T_H$，采样静态承诺随机性 $r_{T_H}$，生成：

$$com_{T_H}\leftarrow zkPC.Commit(\widetilde T_H,pp,r_{T_H}).$$

$com_{T_H}$ 写入 $VK_{static}^{zkGPT}$，并由 $static\_table\_id$ 绑定。验证者不需要持有完整 $T_H$ 明文，但在特征检索相关 claim 归约到 $T_H$ 时，证明者必须给出对 $com_{T_H}$ 的 masked opening。

LeakyReLU、ELU、ExpMap 表均编码为二元 lookup table：

$$T_{LReLU}^{pair}=\{(x,y):(x,y)\in T_{LReLU}\},$$

$$T_{ELU}^{pair}=\{(x,y):(x,y)\in T_{ELU}\},$$

$$T_{exp}^{pair}=\{(u,v):(u,v)\in T_{exp}\}.$$

范围表为结构化表：

$$T_{range}=\{0,1,\ldots,2^B-1\}.$$

本协议固定采用 $T_H$ 私有静态承诺模式：验证者只绑定 $com_{T_H}$ 与 $static\_table\_id$，不要求持有完整 $T_H$ 明文。

二元表的随机线性压缩规则、范围表的分解方式、Lasso 子表参数和 preprocessing 结果由 $VK_{static}^{zkGPT}$ 和 $VK_{Lasso}$ 绑定。

#### 1.5.2 模型参数承诺

将所有模型参数按固定顺序展平成一个模型向量 $W_{all}$：

1. 对 $\ell=1,\ldots,L-1$，按 $r=0,\ldots,K_{hid}^{(\ell)}-1$ 依次写入：

   $$W^{(\ell,r)},\ a_{src}^{(\ell,r)},\ a_{dst}^{(\ell,r)}.$$

2. 对 $s=0,\ldots,K_{out}-1$ 依次写入：

   $$W^{(out,s)},\ a_{src}^{(out,s)},\ a_{dst}^{(out,s)},\ b^{(out,s)}.$$

将 $W_{all}$ padding 到二次幂长度，定义 MLE $\widetilde W_{all}$，采样承诺随机性 $r_W$，生成：

$$com_W\leftarrow zkPC.Commit(\widetilde W_{all},pp,r_W).$$

$VK_{model}^{zkGPT}$ 至少包括 $com_W$、参数 family offset 表、参数 shape 表、模型参数版本号和编码规则。

### 1.6 公共拓扑对象与辅助公开对象

验证者根据公共输入本地重建：

1. $src,dst,I,node\_ptr,edge\_ptr$；
2. $node\_gid,edge\_gid$；
3. $Q_{edge}^{valid},Q_N,Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$；
4. $Q_{in}^{valid,(\ell)},Q_{d_h}^{valid,(\ell)},Q_{cat}^{valid,(\ell)},Q_{cat}^{valid,(out)},Q_C^{valid}$；
5. $Idx_N,Idx_{in},Idx_{d_h},Idx_{cat},Idx_C$；
6. 特征检索辅助列 $Row_{feat}^{tbl},Col_{feat}^{tbl},Row_{feat}^{qry},Col_{feat}^{qry},I_{feat}^{qry}$；
7. 公开路由 wiring 索引表 $src,dst,I$ 及其派生访问表；
8. 所有矩阵乘法 padding shape 与 batch shape。

这些对象不作为 prover 动态承诺提交；它们作为公共输入由验证者本地计算，并作为 ML-friendly GKR 的公共 wiring / 常量输入接入电路。

## 2. 见证生成与承诺

### 2.0 总体顺序与实现约束

见证生成与承诺按以下阶段执行。分阶段的目的，是避免把依赖 Fiat--Shamir 挑战的压缩对象或 PSQ 对象提前放入同一个基础承诺中。

1. 基础前向阶段：计算不依赖随机压缩挑战的 GAT 明文对象，包括 $H,H',E_{src},E_{dst},S,Z,M,\Delta^+,U,Sum,inv,\alpha,H_{agg,pre},H_{agg},H_{cat},Y',Y^{pre},\widetilde Y^{pre},Y^{(s)}$ 等基础对象，并计算最终公开输出 $Y$。$Y$ 只作为顶层公开输出进入证明对象，不写入 $Aux_{base}$。
2. 基础承诺阶段：构造 $Aux_{base}$ 并生成 $com_{base}$。$Aux_{base}$ 不得包含 $H^\star,H_{cat}^\star,Y'^\star,Y^\star,w_{psq},T_{psq},PSQ$ 等依赖后续挑战的对象。
3. 压缩挑战阶段：在吸入 $com_W,com_{T_H},com_{base}$ 后生成 $\xi^{(\ell,r)},\xi_{cat}^{(\ell)},\xi_{out}$。
4. 压缩对象阶段：根据上述挑战计算所有压缩对象，构造 $Aux_{comp}$ 并生成 $com_{comp}$。
5. PSQ 挑战阶段：在吸入 $com_{comp}$ 后生成所有 $\lambda_{psq}$。
6. PSQ / lookup 阶段：根据 $\lambda_{psq}$ 计算 $w_{psq},T_{psq},T_{psq}^{edge},PSQ$，同时构造 lookup query 向量 $L_{qry}$。随后生成 $com_{psq}$ 与 $com_{qry}$。
7. 证明阶段：在 $com_W,com_{T_H},com_{base},com_{comp},com_{psq},com_{qry}$ 全部固定后，进入第 3 章的 Lasso、矩阵乘法 sumcheck、GKR、combine 与 masked opening。

禁止把前向计算与证明消息生成耦合。所有预处理对象必须在参数生成阶段确定。

### 2.1 数据加载与特征检索

#### 2.1.1 原始特征

对每个局部节点 $i$ 与输入维 $j$，计算：

$$H_{i,j}=T_H[I_i,j].$$

$H$ 作为 witness 进入 $Aux_{base}$。本协议固定证明以下特征检索关系：

$$Q_N[i]\cdot Q_{in}^{valid,(1)}[j]\cdot(H_{i,j}-T_H[I_i,j])=0.$$

$T_H$ 由 $com_{T_H}$ 静态承诺绑定，验证者不接受把 $H$ 直接声明为公共输入并跳过特征检索证明。

#### 2.1.2 特征检索证明对象

生成以下对象：

- $Flat(H)$；
- 静态特征表承诺 $com_{T_H}$；
- 公开索引 $I_i$ 与输入维索引 $j$；
- GKR wiring consistency claim：$H_{i,j}=T_H[I_i,j]$。

该关系进入 GKR wiring / arithmetic consistency 电路。具体地，$T_H[I_i,j]$ 表示 $T_H$ segment 中 offset 为 $I_i\cdot d_{in}^{(1)}+j$ 的输入层位置；GKR 的公开 indexed wiring gate 将 $H_{i,j}$ 与该位置连接并检查二者相等。该 indexed wiring 是 instance-specific public wiring：验证者在吸入公开 $I$ 后本地重建连接表，连接表摘要进入 transcript；证明者不能提交、改变或替换该连接表。

### 2.2 单个隐藏层模板中第 $r$ 个注意力头的完整见证生成

本节对应固定隐藏层 $\ell$ 和隐藏头 $r$。多层 GAT 必须按层序实例化：完成第 $\ell$ 层所有头与拼接后，才能进入第 $\ell+1$ 层。

#### 2.2.1 投影

计算：

$$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}.$$

生成矩阵乘法 claim：

$$H'^{(r)}=H\cdot W^{(r)}.$$

将该 claim 放入 $MatMulClaims$。若同一层多个头形状相同，则可批处理为：

$$\{H'^{(r)}=H\cdot W^{(r)}\}_{r=0}^{K_{hid}^{(\ell)}-1}.$$

批处理不改变对象顺序。Proof object 中仍按 $r=0,1,\ldots,K_{hid}^{(\ell)}-1$ 固定顺序声明每个输出对象。

#### 2.2.2 源注意力、目标注意力与压缩特征

基础前向阶段先计算不依赖压缩挑战的源 / 目标注意力对象：

$$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)},$$

$$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}.$$

上述两个对象属于 $Aux_{base}$，必须在 $com_{base}$ 之前计算并承诺。对应关系作为矩阵-向量乘法或 GKR arithmetic relation 进入 base-claim：

$$E_{src}^{(r)}=H'^{(r)}a_{src}^{(r)},\qquad E_{dst}^{(r)}=H'^{(r)}a_{dst}^{(r)}.$$

在第 2.0 节的压缩挑战阶段获得 $\xi^{(\ell,r)}$ 后，再计算压缩特征：

$$H_i^{\star(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}(\xi^{(\ell,r)})^j.$$

$H^{\star(r)}$ 属于 $Aux_{comp}$，必须在 $com_{base}$ 固定且 $\xi^{(\ell,r)}$ 生成之后计算。该压缩内积关系进入 comp-claim：

$$H^{\star(r)}=H'^{(r)}\cdot (1,\xi^{(\ell,r)},\ldots,(\xi^{(\ell,r)})^{d_h-1})^\top.$$

#### 2.2.3 源路由

计算：

$$E_{src}^{edge(r)}=R_{src}E_{src}^{(r)},\qquad H_{src}^{\star,edge(r)}=R_{src}H^{\star(r)}.$$

上述关系作为公开 wiring equality 进入 $C_{GAT}$。$R_{src}$ 不物化为稠密矩阵，也不进入普通 dense matrix multiplication sumcheck。

#### 2.2.4 LeakyReLU、最大值、唯一性与范围检查

计算：

$$S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)},$$

$$Z_k^{(r)}=LReLU(S_k^{(r)}).$$

为 $Z=LReLU(S)$ 生成 Lasso query：

$$L_{LReLU}^{(\ell,r)}[k]=(S_k^{(r)},Z_k^{(r)}).$$

计算组最大值：

$$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\},\qquad M^{edge(r)}=R_{dst}M^{(r)}.$$

计算：

$$\Delta_k^{+(r)}=M_k^{edge(r)}-Z_k^{(r)}.$$

按 $lasso\_cfg\_id$ 为 $\Delta_k^{+(r)}\in[0,2^B-1]$ 生成范围 query。默认编码为：

$$q_{\Delta,L}^{(\ell,r)}[k]=\Delta_k^{+(r)},\qquad q_{\Delta,R}^{(\ell,r)}[k]=2^B-1-\Delta_k^{+(r)},$$

并把 $q_{\Delta,L}^{(\ell,r)},q_{\Delta,R}^{(\ell,r)}$ 写入 $L_{qry}$，由 Lasso 检查二者均属于 $T_{range}$。若配置只检查非负性，则必须由 $lasso\_cfg\_id$ 显式绑定对应的单侧 query family。

引入 $s_{max}^{(r)}[k]\in\{0,1\}$，满足：

$$Q_{edge}^{valid}[k]\cdot s_{max}^{(r)}[k](s_{max}^{(r)}[k]-1)=0,$$

$$Q_{edge}^{valid}[k]\cdot s_{max}^{(r)}[k]\Delta_k^{+(r)}=0.$$

定义计数状态机：

$$C_{max}^{(r)}[0]=Q_{edge}^{valid}[0]s_{max}^{(r)}[0],$$

对 $0\le k<n_{edge}-1$：

$$\begin{aligned}C_{max}^{(r)}[k+1]=&(1-Q_{edge}^{valid}[k+1])C_{max}^{(r)}[k] \\
&+Q_{edge}^{valid}[k+1]\Big(Q_{grp\_new}^{edge}[k+1]s_{max}^{(r)}[k+1]\\
&\qquad +(1-Q_{grp\_new}^{edge}[k+1])(C_{max}^{(r)}[k]+s_{max}^{(r)}[k+1])\Big).
\end{aligned}$$

组末约束：

$$Q_{grp\_end}^{edge}[k](C_{max}^{(r)}[k]-1)=0.$$

这些状态机和二值性约束进入 GKR arithmetic circuit。

#### 2.2.5 指数、分母、逆元与归一化权重

计算：

$$U_k^{(r)}=ExpMap(\Delta_k^{+(r)}).$$

生成 Lasso exp query：

$$L_{exp}^{(\ell,r)}[k]=(\Delta_k^{+(r)},U_k^{(r)}).$$

计算节点分母：

$$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}.$$

计算广播与逆元：

$$Sum^{edge(r)}=R_{dst}Sum^{(r)},\qquad inv_i^{(r)}=(Sum_i^{(r)})^{-1},\qquad inv^{edge(r)}=R_{dst}inv^{(r)}.$$

逆元关系进入 GKR：

$$Q_N[i](Sum_i^{(r)}inv_i^{(r)}-1)=0.$$

归一化权重：

$$\alpha_k^{(r)}=U_k^{(r)}inv_k^{edge(r)}.$$

本协议固定采用域内 softmax 权重语义：

$$\alpha_k^{(r)}=U_k^{(r)}inv_k^{edge(r)}.$$

该处不额外引入 $Rescale_\alpha$。

#### 2.2.6 聚合前隐藏矩阵、聚合前压缩特征与 PSQ

计算：

$$H_{agg,pre,i,j}^{(r)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}.$$

计算压缩特征：

$$H_{agg,pre,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,pre,i,j}^{(r)}(\xi^{(\ell,r)})^j.$$

计算边级压缩加权特征：

$$\widehat v_{pre,k}^{\star(r)}=\alpha_k^{(r)}H_{src,k}^{\star,edge(r)}.$$

广播：

$$H_{agg,pre}^{\star,edge(r)}=R_{dst}H_{agg,pre}^{\star(r)}.$$

在第 2.0 节的 PSQ 挑战阶段获得 $\lambda_{psq}^{(\ell,r)}$ 后，定义：

$$w_{psq}^{(r)}[k]=U_k^{(r)}+\lambda_{psq}^{(\ell,r)}\widehat v_{pre,k}^{\star(r)},$$

$$T_{psq}^{(r)}[i]=Sum_i^{(r)}+\lambda_{psq}^{(\ell,r)}H_{agg,pre,i}^{\star(r)},$$

$$T_{psq}^{edge(r)}[k]=Sum_k^{edge(r)}+\lambda_{psq}^{(\ell,r)}H_{agg,pre,k}^{\star,edge(r)}.$$

定义 PSQ 状态机：

$$PSQ^{(r)}[0]=Q_{edge}^{valid}[0]w_{psq}^{(r)}[0],$$

对 $0\le k<n_{edge}-1$：

$$\begin{aligned}PSQ^{(r)}[k+1]=&(1-Q_{edge}^{valid}[k+1])PSQ^{(r)}[k]\\
&+Q_{edge}^{valid}[k+1]\Big(Q_{grp\_new}^{edge}[k+1]w_{psq}^{(r)}[k+1]\\
&\qquad +(1-Q_{grp\_new}^{edge}[k+1])(PSQ^{(r)}[k]+w_{psq}^{(r)}[k+1])\Big).
\end{aligned}$$

组末约束：

$$Q_{grp\_end}^{edge}[k](PSQ^{(r)}[k]-T_{psq}^{edge(r)}[k])=0.$$

该约束同时强制：

$$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)},$$

$$H_{agg,pre,i}^{\star(r)}=\sum_{\{k\mid dst(k)=i\}}\widehat v_{pre,k}^{\star(r)}.$$

同时对 $H_{agg,pre}^{(r)}$ 与 $H_{agg,pre}^{\star(r)}$ 施加压缩绑定关系，进入 GKR arithmetic circuit。

#### 2.2.7 ELU

对每个 $i,j$：

$$H_{agg,i,j}^{(r)}=ELU(H_{agg,pre,i,j}^{(r)}).$$

展平 $H_{agg,pre}^{(r)}$ 与 $H_{agg}^{(r)}$，生成 Lasso query：

$$L_{ELU}^{(\ell,r)}[i\cdot d_h+j]=(H_{agg,pre,i,j}^{(r)},H_{agg,i,j}^{(r)}).$$

计算：

$$H_{agg,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,i,j}^{(r)}(\xi^{(\ell,r)})^j.$$

广播：

$$H_{agg}^{\star,edge(r)}=R_{dst}H_{agg}^{\star(r)}.$$

压缩绑定与广播关系进入 GKR / matrix claim。

#### 2.2.8 目标路由的延迟定稿

目标路由同时绑定：

- $E_{dst}^{(r)}$；
- $M^{(r)}$；
- $Sum^{(r)}$；
- $inv^{(r)}$；
- $H_{agg}^{\star(r)}$。

统一生成以下广播关系：

$$E_{dst}^{edge(r)}=R_{dst}E_{dst}^{(r)},$$

$$M^{edge(r)}=R_{dst}M^{(r)},$$

$$Sum^{edge(r)}=R_{dst}Sum^{(r)},$$

$$inv^{edge(r)}=R_{dst}inv^{(r)},$$

$$H_{agg}^{\star,edge(r)}=R_{dst}H_{agg}^{\star(r)}.$$

这些关系全部通过 GKR public wiring / linear consistency 证明。$R_{dst}$ 不物化为稠密矩阵，也不进入普通 dense matrix multiplication sumcheck。

### 2.3 隐藏层模板的拼接阶段

对固定隐藏层 $\ell$，定义：

$$H_{cat,i,r\cdot d_h+j}^{(\ell)}=H_{agg,i,j}^{(\ell,r)}.$$

在第 2.0 节的压缩挑战阶段获得 $\xi_{cat}^{(\ell)}$ 后，计算：

$$H_{cat,i}^{\star(\ell)}=\sum_{m=0}^{d_{cat}^{(\ell)}-1}H_{cat,i,m}^{(\ell)}(\xi_{cat}^{(\ell)})^m.$$

拼接关系进入 GKR arithmetic circuit：

1. 每个 $H_{cat}^{(\ell)}$ 分量必须等于相应头的 $H_{agg}^{(\ell,r)}$ 分量；
2. $H_{cat}^{\star(\ell)}$ 必须等于 $H_{cat}^{(\ell)}$ 的挑战压缩。

完成第 $\ell$ 层拼接后，若 $\ell<L-1$，则设置下一层输入：

$$H^{(\ell+1)}:=H_{cat}^{(\ell)}.$$

### 2.4 输出层的完整见证生成

输出层由 $K_{out}$ 个输出注意力头组成。对每个输出头 $s$ 先完成 projection / score / softmax / aggregation / bias-add，再对全部输出头求平均。

#### 2.4.1 第 $s$ 个输出头的输出投影

计算：

$$Y_{i,c}'^{(s)}=\sum_{m=0}^{d_{cat}^{(L-1)}-1}H_{cat,i,m}^{(L-1)}W_{m,c}^{(out,s)}.$$

生成矩阵乘法 claim：

$$Y'^{(s)}=H_{cat}^{(L-1)}W^{(out,s)}.$$

多个输出头可按 $s$ 批处理；proof object 中必须保持 $s=0,1,\ldots,K_{out}-1$ 的顺序。

#### 2.4.2 第 $s$ 个输出头的源 / 目标注意力绑定

基础前向阶段先计算不依赖输出类别压缩挑战的源 / 目标注意力对象：

$$E_{src,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{src,c}^{(out,s)},$$

$$E_{dst,i}^{(out,s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}a_{dst,c}^{(out,s)}.$$

上述两个对象属于 $Aux_{base}$，对应矩阵-向量关系进入 base-claim。

在第 2.0 节的压缩挑战阶段获得 $\xi_{out}$ 后，再计算：

$$Y_i'^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}'^{(s)}\xi_{out}^c.$$

$Y'^{\star(s)}$ 属于 $Aux_{comp}$，其类别压缩内积关系进入 comp-claim。上述 base-claim 与 comp-claim 均可由 $MatMulClaims$ 或 GKR arithmetic circuit 证明，但 proof label 必须区分输出所在 segment。

#### 2.4.3 第 $s$ 个输出头的源路由、LeakyReLU、最大值、范围检查、指数、分母、逆元与归一化权重

源路由：

$$E_{src}^{edge(out,s)}=R_{src}E_{src}^{(out,s)},\qquad Y'^{\star,edge(s)}=R_{src}Y'^{\star(s)}.$$

目标路由：

$$E_{dst}^{edge(out,s)}=R_{dst}E_{dst}^{(out,s)}.$$

打分与查表：

$$S_k^{(out,s)}=E_{src,k}^{edge(out,s)}+E_{dst,k}^{edge(out,s)},$$

$$Z_k^{(out,s)}=LReLU(S_k^{(out,s)}).$$

生成 Lasso query：

$$L_{LReLU}^{(out,s)}[k]=(S_k^{(out,s)},Z_k^{(out,s)}).$$

最大值、差分、范围、指数、分母、逆元、权重按隐藏层同构生成：

$$M_i^{(out,s)}=\max\{Z_k^{(out,s)}\mid dst(k)=i\},$$

$$M^{edge(out,s)}=R_{dst}M^{(out,s)},$$

$$\Delta_k^{+(out,s)}=M_k^{edge(out,s)}-Z_k^{(out,s)},$$

$$U_k^{(out,s)}=ExpMap(\Delta_k^{+(out,s)}),$$

$$Sum_i^{(out,s)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out,s)},$$

$$Sum^{edge(out,s)}=R_{dst}Sum^{(out,s)},$$

$$inv_i^{(out,s)}=(Sum_i^{(out,s)})^{-1},$$

$$inv^{edge(out,s)}=R_{dst}inv^{(out,s)},$$

$$\alpha_k^{(out,s)}=U_k^{(out,s)}inv_k^{edge(out,s)}.$$

按 $lasso\_cfg\_id$ 为 $\Delta_k^{+(out,s)}\in[0,2^B-1]$ 生成范围 query。默认写入：

$$q_{\Delta,L}^{(out,s)}[k]=\Delta_k^{+(out,s)},\qquad q_{\Delta,R}^{(out,s)}[k]=2^B-1-\Delta_k^{+(out,s)},$$

并由 Lasso 检查二者均属于 $T_{range}$；若配置只检查非负性，则必须由 $lasso\_cfg\_id$ 显式绑定。指数 query $L_{exp}^{(out,s)}$、最大值唯一性、PSQ 所需分组、逆元关系均进入相应证明子系统。

#### 2.4.4 第 $s$ 个输出头的聚合前输出、PSQ 与压缩绑定

计算：

$$\widehat y_k^{\star(s)}=\alpha_k^{(out,s)}Y_k'^{\star,edge(s)}.$$

计算：

$$Y_{i,c}^{pre(s)}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out,s)}Y_{src(k),c}'^{(s)}.$$

压缩：

$$Y_i^{pre,\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{pre(s)}\xi_{out}^c.$$

广播：

$$Y^{pre,\star,edge(s)}=R_{dst}Y^{pre,\star(s)}.$$

在第 2.0 节的 PSQ 挑战阶段获得输出头挑战 $\lambda_{psq}^{(out,s)}$ 后，定义：

$$w_{psq}^{(out,s)}[k]=U_k^{(out,s)}+\lambda_{psq}^{(out,s)}\widehat y_k^{\star(s)},$$

$$T_{psq}^{(out,s)}[i]=Sum_i^{(out,s)}+\lambda_{psq}^{(out,s)}Y_i^{pre,\star(s)},$$

$$T_{psq}^{edge(out,s)}[k]=Sum_k^{edge(out,s)}+\lambda_{psq}^{(out,s)}Y_k^{pre,\star,edge(s)}.$$

按隐藏层同构的 PSQ 状态机施加组内和约束。$Y^{pre(s)}$ 与 $Y^{pre,\star(s)}$ 的压缩绑定进入 GKR。

#### 2.4.5 第 $s$ 个输出头的输出偏置、加偏置后压缩绑定与目标路由

对每个 $i,c$：

$$\widetilde Y_{i,c}^{pre(s)}=Rescale_{out}(Y_{i,c}^{pre(s)};S_{out,pre}\to S_{out,head}).$$

若 $Rescale_{out}$ 非恒等，则 $\widetilde Y^{pre(s)}$ 是 advice，并按 $quant\_cfg\_id$ 生成 rounding/range query。若为恒等，则只施加：

$$\widetilde Y_{i,c}^{pre(s)}=Y_{i,c}^{pre(s)}.$$

加偏置：

$$Y_{i,c}^{(s)}=\widetilde Y_{i,c}^{pre(s)}+b_c^{(out,s)}.$$

压缩：

$$Y_i^{\star(s)}=\sum_{c=0}^{C-1}Y_{i,c}^{(s)}\xi_{out}^c.$$

广播：

$$Y^{\star,edge(s)}=R_{dst}Y^{\star(s)}.$$

这些关系进入 GKR public wiring / arithmetic consistency claim。

#### 2.4.6 最终多头平均输出

计算：

$$Y_{i,c}^{sum}=\sum_{s=0}^{K_{out}-1}Y_{i,c}^{(s)},\qquad Y_{i,c}=K_{out}^{-1}Y_{i,c}^{sum}.$$

计算最终压缩：

$$Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c.$$

广播：

$$Y^{\star,edge}=R_{dst}Y^{\star}.$$

最终平均与最终压缩全部进入 GKR arithmetic circuit。若 $K_{out}=1$，仍必须显式保留该平均约束的 tag，不能在 proof object 中删除。

#### 2.4.7 分阶段输入层承诺

完成基础前向对象后，证明者首先构造：

$$Aux_{base}=\operatorname{EncodeBase}(H,\{H',E_{src},E_{dst},S,Z,M,\Delta^+,U,Sum,inv,\alpha,H_{agg,pre},H_{agg},H_{cat},Y',Y^{pre},\widetilde Y^{pre},Y^{(s)}\}).$$

最终输出 $Y$ 作为顶层公开输出写入证明对象，不写入 $Aux_{base}$；GKR 输出 claim 必须直接绑定该公开 $Y$。

采样 $r_{base}$ 并提交：

$$com_{base}\leftarrow zkPC.Commit(\widetilde{Aux}_{base},pp,r_{base}).$$

随后 transcript 吸入 $com_W,com_{T_H},com_{base}$ 并生成 $\xi^{(\ell,r)},\xi_{cat}^{(\ell)},\xi_{out}$。证明者根据这些挑战构造：

$$Aux_{comp}=\operatorname{EncodeComp}(H^\star,H_{agg,pre}^\star,H_{agg}^\star,H_{cat}^\star,Y'^\star,Y^{pre,\star},Y^\star).$$

采样 $r_{comp}$ 并提交：

$$com_{comp}\leftarrow zkPC.Commit(\widetilde{Aux}_{comp},pp,r_{comp}).$$

接着 transcript 吸入 $com_{comp}$ 并生成所有 $\lambda_{psq}$。证明者根据这些挑战构造：

$$Aux_{psq}=\operatorname{EncodePSQ}(w_{psq},T_{psq},T_{psq}^{edge},PSQ,C_{max},s_{max}).$$

同时构造 lookup query 向量：

$$L_{qry}=\operatorname{EncodeLookupQueries}(L_{LReLU},L_{ELU},L_{exp},L_{range},L_{round}).$$

采样 $r_{psq},r_{qry}$ 并提交：

$$com_{psq}\leftarrow zkPC.Commit(\widetilde{Aux}_{psq},pp,r_{psq}),$$

$$com_{qry}\leftarrow zkPC.Commit(\widetilde{L}_{qry},pp,r_{qry}).$$

证明对象中的动态承诺块只包含按固定顺序排列的输入层相关承诺。

## 3. 证明生成

### 3.1 输入与输出

证明算法输入：

- 参数生成输出：$(pp,VK_{model}^{zkGPT},VK_{static}^{zkGPT},VK_{Lasso},C_{GAT},C_{matmul},Layout_{in})$；
- 公共输入：$(I,src,dst,G_{batch},node\_ptr,edge\_ptr,N,E,N_{total},L,\{K_{hid}^{(\ell)}\},\{d_{in}^{(\ell)}\},\{d_h^{(\ell)}\},\{d_{cat}^{(\ell)}\},C,B,K_{out})$；
- 模型参数 witness $W_{all}$ 及其承诺随机性 $r_W$；
- 本次证明 witness $Aux_{base},Aux_{comp},Aux_{psq},L_{qry}$ 及其承诺随机性；
- 最终输出 $Y$。

输出最终证明：

$$\pi_{GAT}^{zkGPT}=(M_{pub},Y,Com_{in},\Pi_{Lasso},\Pi_{MatMul},\Pi_{GKR},\Pi_{combine},Open_{in}).$$

### 3.2 Fiat--Shamir 挑战顺序

证明者和验证者必须严格按以下顺序重放 transcript。

#### 3.2.1 全局阶段

1. 吸入域分离标签 `gat-zkgpt/meta`。
2. 吸入 $M_{pub}$。
3. 吸入公共输入：$I,src,dst,node\_ptr,edge\_ptr,N,E,N_{total},G_{batch},L,\{d_{in}^{(\ell)}\},\{d_h^{(\ell)}\},\{d_{cat}^{(\ell)}\},C,B,K_{out}$。
4. 吸入验证者本地重建的公共拓扑对象摘要：$node\_gid,edge\_gid,Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$、公开 routing wiring 索引摘要以及 $T_H$ indexed access 布局摘要。
5. 吸入 $com_W,com_{T_H},VK_{static}^{zkGPT},VK_{Lasso},Layout_{in}$。
6. 吸入基础动态承诺 $com_{base}$ 与输出 $Y$。
7. 生成压缩挑战族：$\xi^{(\ell,r)}$、$\xi_{cat}^{(\ell)}$、$\xi_{out}$。挑战生成顺序必须与层序、头序一致。
8. 吸入压缩对象承诺 $com_{comp}$。
9. 生成所有 $\lambda_{psq}$，顺序按隐藏层层序、隐藏头序、输出头序排列。
10. 吸入 $com_{psq},com_{qry}$，随后进入 Lasso、矩阵乘法与 GKR 阶段。

#### 3.2.2 Lasso 阶段

1. 吸入所有 lookup family 的 tag、table id、raw query commitment、query 长度、padding 长度。
2. 对每个二元 lookup family 生成 pair-compression challenge $\eta_{\mathcal L}$，并由 $L_{qry}$ 中已承诺的 raw query columns 派生 compressed query。
3. 对每个 range lookup family 生成 range decomposition / structured-table 所需挑战。
4. 生成 Lasso 协议内部挑战。
5. 按固定顺序运行 lookup family：hidden range，按 $\ell,r$；output range，按 $s$；hidden LeakyReLU，按 $\ell,r$；output LeakyReLU，按 $s$；hidden ELU，按 $\ell,r$；hidden exp，按 $\ell,r$；output exp，按 $s$；rescale/rounding range，按 $Layout_{in}$ 中声明顺序。

#### 3.2.3 矩阵乘法阶段

1. 吸入所有 matrix claim 的 tag、shape、batch shape、输入 family id、输出 family id。
2. 对每个 batched matrix multiplication group 生成随机点评值。
3. 运行专用矩阵乘法 sumcheck，把输出矩阵 claim 归约到 $W,Aux_{base},Aux_{comp}$ 的输入层 masked-opening claim；若涉及公开输入 segment，则归约为验证者本地评值 claim。

#### 3.2.4 GKR 阶段

1. 吸入 GKR 电路 id、squeeze layout id、输入层承诺和输出 claim。
2. 验证者生成输出层随机点 $r^{(d)}$。
3. 从输出层到输入层运行 GKR sumcheck。每一层的 sumcheck 挑战由前一轮 transcript 生成。
4. 每一层允许连接到上一层和输入层的 ML-friendly gates。所有 intermediate advice 均位于输入层，因此多个相互独立的约束子电路可在同一组层内被 squeeze。

#### 3.2.5 输入层合并与 masked opening 阶段

1. GKR、Lasso 和矩阵乘法 proof 最终产生若干输入层 MLE claim。
2. 证明者和验证者运行 combine sumcheck，将所有 claim 先合并为对虚拟输入层 $V_{in}$ 的分段 claim，再按 $Layout_{in}$ 的段选择规则归约到 $\widetilde T_H$、$\widetilde W_{all}$、$\widetilde{Aux}_{base}$、$\widetilde{Aux}_{comp}$、$\widetilde{Aux}_{psq}$、$\widetilde{L}_{qry}$ 的 masked-opening claim，以及公开 $\widetilde X_{pub}$ 的本地评值 claim。combine 等式按 masking-adjusted claim 检查，证明对象中不得暴露未掩蔽的 witness MLE 随机点评值。
3. 吸入 opening 点，生成 opening challenge。
4. 证明者对 $T_H,W,Aux_{base},Aux_{comp},Aux_{psq},L_{qry}$ 输出 $zkPC.OpenMasked$ 证明；对公开 $X_{pub}$，验证者本地计算其 MLE 评值，不提交 opening。

### 3.3 约束族

本节列出正式约束族。所有约束分为三类：Lasso lookup relation、matrix multiplication relation、GKR arithmetic relation。

#### 3.3.1 Lasso lookup 关系

每个二元 lookup family 都先由 transcript 生成独立压缩挑战 $\eta_{\mathcal L}$。若表行为 $(x_t,y_t)$，查询行为 $(x_q,y_q)$，则定义：

$$Table_{\mathcal L}[t]=x_t+\eta_{\mathcal L}y_t,$$

$$Query_{\mathcal L}[q]=x_q+\eta_{\mathcal L}y_q.$$

Lasso 证明 $Query_{\mathcal L}$ 属于压缩后的 $Table_{\mathcal L}$。由于 $\eta_{\mathcal L}$ 在 $com_{qry}$ 之后生成，$L_{qry}$ 中不得预先只存储压缩后的单列 query；$L_{qry}$ 必须存储二元 lookup 的 raw query columns $(x_q,y_q)$，并由 GKR 检查这些 raw columns 与 $Aux_{base},Aux_{comp},Aux_{psq}$ 中基础变量一致。压缩列 $Query_{\mathcal L}$ 只作为 Lasso 阶段的派生线性组合参与证明。范围关系统一写为区间关系 $v\in[\ell,r]$，并转换为：

$$v-\ell\in T_{range},\qquad r-v\in T_{range}.$$

当区间为 $[0,2^B-1]$ 时，上式退化为 $v\in T_{range}$ 与 $2^B-1-v\in T_{range}$；若协议只需要非负性且已有公开上界，可只保留相应的必要 query family，但该选择必须由 $lasso\_cfg\_id$ 固定。所有区间上下界均要求小于 $p$，range query 的值按标准非负代表解释；若整数语义下 $v-\ell$ 或 $r-v$ 不在 $[0,2^B-1]$，则对应 query 不得通过 range lookup，不能借助模 $p$ 环绕通过验证。


1. LeakyReLU：

   $$\forall (\ell,r),k:\quad (S_k^{(\ell,r)},Z_k^{(\ell,r)})\in T_{LReLU}.$$

   $$\forall (out,s),k:\quad (S_k^{(out,s)},Z_k^{(out,s)})\in T_{LReLU}.$$

2. ELU：

   $$\forall (\ell,r),i,j:\quad (H_{agg,pre,i,j}^{(\ell,r)},H_{agg,i,j}^{(\ell,r)})\in T_{ELU}.$$

3. ExpMap：

   $$\forall (\ell,r),k:\quad (\Delta_k^{+(\ell,r)},U_k^{(\ell,r)})\in T_{exp},$$

   $$\forall (out,s),k:\quad (\Delta_k^{+(out,s)},U_k^{(out,s)})\in T_{exp}.$$

4. Range：

   对隐藏层与输出层的 $\Delta^+$，协议语义为 $\Delta^+\in[0,2^B-1]$。默认把该区间关系编码为 $q_L=\Delta^+$ 与 $q_R=2^B-1-\Delta^+$ 两个 raw range query，并检查：

   $$q_L\in T_{range},\qquad q_R\in T_{range}.$$

   若 $lasso\_cfg\_id$ 指定单侧非负性检查，则必须明确绑定对应 query family 与公开上界来源。

5. Rounding / rescale range：若 $quant\_cfg\_id$ 指定某个 $z=round(F(x))$ 为 advice，则根据公开规则构造等价范围不等式，再按 $v\in[\ell,r]\Rightarrow v-\ell,r-v\in T_{range}$ 的规则进入 Lasso range query。对除法型关系：

   $$z=round\left(\frac{C_1}{C_2}\cdot\frac{x}{y}\right)$$

   转换为：

   $$C_2y(2z-1)\le 2C_1x\le C_2y(2z+1)-1.$$

   该类不等式必须在整数语义下落入无模环绕区间；参与不等式的左右边界、误差项和乘法中间量的取值上界必须由 $quant\_cfg\_id$ 固定。若关系包含分母 $y$，则 $y$ 的正性、符号或非零性必须由 $quant\_cfg\_id$ 规定并通过 range / selector / inverse 约束证明；不得利用域内模 $p$ 环绕满足舍入不等式。

所有 lookup query 都必须绑定到 $L_{qry}$ 中的固定 offset，并通过 GKR 约束检查 raw query columns 与 $Aux_{base},Aux_{comp},Aux_{psq}$ 中基础变量一致。

#### 3.3.2 矩阵乘法关系

矩阵乘法关系包括：

1. 隐藏层投影：$H'^{(\ell,r)}=H^{(\ell)}W^{(\ell,r)}$。
2. 隐藏层源 / 目标注意力：$E_{src}^{(\ell,r)}=H'^{(\ell,r)}a_{src}^{(\ell,r)}$，$E_{dst}^{(\ell,r)}=H'^{(\ell,r)}a_{dst}^{(\ell,r)}$。
3. 压缩绑定可作为矩阵-向量关系：$H^{\star(\ell,r)}=H'^{(\ell,r)}v_{\xi}^{(\ell,r)}$。
4. 输出层投影：$Y'^{(s)}=H_{cat}^{(L-1)}W^{(out,s)}$。
5. 输出层注意力与类别压缩：$E_{src}^{(out,s)},E_{dst}^{(out,s)},Y'^{\star(s)}$。

特征检索、源路由与目标路由不属于本节矩阵乘法关系；它们作为第 3.3.3 节的 GKR wiring consistency 关系处理。

所有同形状矩阵乘法可进入 batched matrix multiplication sumcheck；不满足同形状的 group 必须分组，不得强行 padding 到会改变语义的布局。

矩阵乘法 claim 按输出所在 segment 分为 base-claim 与 comp-claim。base-claim 的输出位于 $Aux_{base}$，例如 $H'=HW$、$E_{src}=H'a_{src}$、$E_{dst}=H'a_{dst}$；comp-claim 的输出位于 $Aux_{comp}$，例如 $H^\star=H'v_\xi$、$Y'^\star=Y'v_{\xi_{out}}$。所有 comp-claim 必须在对应 transcript challenge 已生成且 $com_{comp}$ 已固定后参与证明，证明标签必须显式标记其输出 segment。

#### 3.3.3 GKR arithmetic 关系

GKR 电路检查全部普通算术关系，包括：

1. 打分：

   $$S=E_{src}^{edge}+E_{dst}^{edge}.$$

2. 最大值差分：

   $$\Delta^+=M^{edge}-Z.$$

3. 最大值唯一性：

   $$Q_{edge}^{valid}\cdot s_{max}(s_{max}-1)=0,\qquad Q_{edge}^{valid}\cdot s_{max}\Delta^+=0,$$

   以及 $C_{max}$ 状态机递推和组末 $C_{max}=1$。

4. 逆元：

   $$Q_N(Sum\cdot inv-1)=0.$$

5. 归一化权重：

   $$\alpha=U\cdot inv^{edge}$$

   本协议固定为纯域内乘法语义，不在该处引入额外 rescale/rounding 分支。

6. PSQ：$w_{psq},T_{psq},T_{psq}^{edge},PSQ$ 的定义、递推与组末约束。

7. 压缩绑定：$H^{\star}$、$H_{agg,pre}^{\star}$、$H_{agg}^{\star}$、$H_{cat}^{\star}$、$Y'^{\star}$、$Y^{pre,\star}$、$Y^{\star}$、最终 $Y^{\star}$。

8. 拼接：

   $$H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}.$$

9. 输出层 bias add 与平均：

   $$Y^{(s)}=\widetilde Y^{pre(s)}+b^{(out,s)},$$

   $$Y=K_{out}^{-1}\sum_sY^{(s)}.$$

10. lookup query consistency：$L_{qry}$ 中每个 raw query column 必须等于对应 $Aux_{base},Aux_{comp},Aux_{psq}$ 中的基础变量或由这些变量经公开算术关系得到的 range 差值。二元 lookup 的压缩 query $Query_{\mathcal L}=x_q+\eta_{\mathcal L}y_q$ 只由 raw columns 和 $\eta_{\mathcal L}$ 在线性组合中派生，不作为预承诺 witness。对区间关系 $v\in[\ell,r]$，$L_{qry}$ 存储 $q_L=v-\ell$ 与 $q_R=r-v$，GKR 检查 $q_L,q_R$ 与 $v$ 的一致性，Lasso 检查 $q_L,q_R\in T_{range}$。

#### 3.3.4 Circuit squeeze 规则

由于所有中间结果作为 advice 位于输入层，不同约束子电路之间可以不按神经网络前向顺序形成拓扑依赖。GKR 电路构造时允许：

1. 将多个独立 rounding/range 检查子电路压入相同电路层；
2. 将隐藏层不同头的同构约束压入相同电路层；
3. 将输出层不同输出头的同构约束压入相同电路层；
4. 将不同层中形状相同且只依赖输入层 advice 的一致性检查压入相同电路层。

限制：

- squeeze 只改变证明电路拓扑，不改变前向 witness 的定义；
- squeeze 不允许改变 $Aux$ 中对象顺序；
- squeeze 不允许合并不同语义 tag；
- squeeze 不允许把尚未由 transcript 生成的挑战提前用于 witness 计算；
- squeeze layout 必须由 $circuit\_layout\_id$ 固定。

#### 3.3.5 输入层一致性

所有 proof 子系统最终归约到输入层。逻辑输入层按七段排列：

$$V_{in}=X_{pub}\|T_H\|W\|Aux_{base}\|Aux_{comp}\|Aux_{psq}\|L_{qry}.$$

段选择采用 $Layout_{in}$ 固定的 one-hot / selector 多项式。若最终合并点为 $(r^-,\rho)$，其中 $r^-$ 是段内随机点，$\rho$ 是段选择随机点，则验证者检查：

$$\widetilde V_{in}(r^-,\rho)=\sum_{seg\in\mathcal S}Sel_{seg}(\rho)\cdot\widetilde V_{seg}(r^-),$$

其中：

$$\mathcal S=\{X_{pub},T_H,W,Aux_{base},Aux_{comp},Aux_{psq},L_{qry}\}.$$

$X_{pub}$ 由验证者本地评值；$T_H,W,Aux_{base},Aux_{comp},Aux_{psq},L_{qry}$ 分别由 $com_{T_H},com_W,com_{base},com_{comp},com_{psq},com_{qry}$ 的 masked opening 绑定。段编码公式、selector 顺序和 padding 长度必须由 $Layout_{in}$ 固定，并在 verifier 中逐字节重放。

### 3.4 Masked opening 与输入层 claim 合并

证明者收集来自以下来源的 claim：

1. Lasso 对 lookup raw query segment $L_{qry}$ 及其挑战压缩线性组合的 claim；
2. GKR 对 $T_H,W,Aux_{base},Aux_{comp},Aux_{psq},L_{qry},X_{pub}$ 的 claim；
3. 矩阵乘法 sumcheck 对 $W,Aux_{base},Aux_{comp},X_{pub}$ 的 claim；
4. 顶层公开输出 $Y$ 与最终压缩 $Y^\star$ 的 claim。

证明者与验证者运行合并 sumcheck，把这些 claim 合并为有限个 masked opening。合并过程中使用的等式均按 $proof\_backend\_id$ 固定的 masking-adjusted claim 形式检查，证明者不得把未掩蔽的 $\widetilde V(z)$ 作为 proof object 字段输出：

$$zkPC.OpenMasked(\widetilde T_H,r_{T_H};z_T),\quad zkPC.OpenMasked(\widetilde W,r_W;z_W),\quad zkPC.OpenMasked(\widetilde{Aux}_{base},r_{base};z_{base}),$$

$$zkPC.OpenMasked(\widetilde{Aux}_{comp},r_{comp};z_{comp}),\quad zkPC.OpenMasked(\widetilde{Aux}_{psq},r_{psq};z_{psq}),\quad zkPC.OpenMasked(\widetilde{L}_{qry},r_{qry};z_{qry}),$$

以及必要时的 $\widetilde X_{pub}$ 本地评值。$X_{pub}$ 完全公开，由验证者本地计算评值，不需要承诺 opening。

为满足 zero-knowledge，所有 witness segment 的 commitment 与 opening 必须使用 hiding randomness；GKR、Lasso、矩阵乘法 sumcheck 和 combine 阶段中暴露的 witness MLE evaluation 必须采用 masking polynomial 或等价 blinding 机制。验证等式在 masked evaluation 上检查，最终只泄露公共输入、公开输出 $Y$ 以及由协议规定可公开的元数据，不泄露 $T_H,W,Aux_{base},Aux_{comp},Aux_{psq},L_{qry}$ 的额外信息。

### 3.5 最终证明对象

最终证明对象固定为：

$$\pi_{GAT}^{zkGPT}=\big(M_{pub},Y,Com_{in},\Pi_{Lasso},\Pi_{MatMul},\Pi_{GKR},\Pi_{combine},Open_{in}\big).$$

#### 3.5.1 元数据块

$M_{pub}$ 按第 0.8.5 节字段顺序编码。不得追加未声明字段，不得重排字段。

#### 3.5.2 输入承诺块

$$Com_{in}=(com_{T_H},com_W,com_{base},com_{comp},com_{psq},com_{qry}).$$

若 $com_{T_H}$ 或 $com_W$ 已由验证者预先持有，证明对象仍必须包含其 id 或摘要，确保 transcript 绑定同一个静态特征表与模型参数承诺。

#### 3.5.3 Lasso 证明块

$\Pi_{Lasso}$ 包括所有 lookup family 的证明，顺序固定为：

1. hidden range，按 $\ell,r$；
2. output range，按 $s$；
3. hidden LeakyReLU，按 $\ell,r$；
4. output LeakyReLU，按 $s$；
5. hidden ELU，按 $\ell,r$；
6. hidden exp，按 $\ell,r$；
7. output exp，按 $s$；
8. rescale/rounding range，按 $Layout_{in}$ 中声明顺序。

#### 3.5.4 矩阵乘法证明块

$\Pi_{MatMul}$ 包括所有 matrix multiplication sumcheck 证明。顺序由 $C_{matmul}$ 的 claim table 固定。Claim table 必须列出：tag、shape、batch size、left input family、right input family、output family、padding shape。

#### 3.5.5 GKR 证明块

$\Pi_{GKR}$ 包括 ML-friendly GKR 的 layer-by-layer sumcheck transcript、每层 claimed evaluations、输入层连接 claim 和 squeeze layout id。

#### 3.5.6 合并证明块

$\Pi_{combine}$ 包括把 Lasso、MatMul、GKR 的输入层 claim 合并为 masked-opening claim 的 sumcheck 证明。

#### 3.5.7 Opening 块

$Open_{in}$ 包括对 $com_{T_H},com_W,com_{base},com_{comp},com_{psq},com_{qry}$ 的 masked opening 证明、opening 点、masked evaluation 值以及相应的 hiding / masking 证明消息。若 $X_{pub}$ 公开，则不提交 $X_{pub}$ opening；验证者本地计算其 MLE 评值。

#### 3.5.8 Serializer / Parser 固定规则

1. 所有顶层块必须带固定 tag。
2. 所有 family 必须带长度前缀、shape、padding 后长度、offset。
3. family 数量必须与 $dim\_cfg$ 和 $Layout_{in}$ 一致。
4. parser 遇到未声明字段、重复 tag、缺失 tag、长度不一致、对象顺序错误，必须立即拒绝。
5. verifier replay transcript 的吸入顺序必须与 serializer 输出顺序逐字节一致，不得按对象类型重新排序。

## 4. 验证

验证算法输入：

- 参数生成输出：$(pp,VK_{model}^{zkGPT},VK_{static}^{zkGPT},VK_{Lasso},C_{GAT},C_{matmul},Layout_{in})$；
- 公共输入：$(I,src,dst,G_{batch},node\_ptr,edge\_ptr,N,E,N_{total},L,\{K_{hid}^{(\ell)}\},\{d_{in}^{(\ell)}\},\{d_h^{(\ell)}\},\{d_{cat}^{(\ell)}\},C,B,K_{out})$；
- 证明对象 $\pi_{GAT}^{zkGPT}$。

验证者按下列步骤执行。

### 4.1 元数据与对象完整性检查

1. 解析 $M_{pub}$，逐项检查 $protocol\_id,proof\_backend\_id,model\_arch\_id,model\_param\_id,static\_table\_id,quant\_cfg\_id,domain\_cfg,dim\_cfg,encoding\_id,padding\_rule\_id,circuit\_layout\_id,lasso\_cfg\_id,matrix\_proof\_cfg\_id$ 与本地期望一致。

2. 本地重建 $node\_gid,edge\_gid,Q_{graph\_new}^{edge},Q_{graph\_end}^{edge},Q_{grp\_new}^{edge},Q_{grp\_end}^{edge}$，检查 $node\_ptr,edge\_ptr$ 与公开边界一致。

3. 本地重建公开 routing wiring 索引表，检查 $src,dst,I$ 是否命中合法索引，且边端 routing 不跨图连接。

4. 检查每个节点至少有一条入边。若不满足，直接拒绝。

5. 检查 $Com_{in},\Pi_{Lasso},\Pi_{MatMul},\Pi_{GKR},\Pi_{combine},Open_{in}$ 的对象数量与顺序是否与第 3.5 节一致。

6. 若 parser 读到未声明字段、重复字段、缺失字段、长度前缀不一致、family 个数与 `dim_cfg` 不一致，或顶层 tag 不一致，立即拒绝。

### 4.2 重放 Fiat--Shamir transcript

验证者按第 3.2 节顺序重放 transcript：

1. 吸入 $M_{pub}$、公共输入、公共拓扑对象摘要、公开 routing wiring 索引摘要以及 $T_H$ indexed access 布局摘要；
2. 吸入 $com_W,com_{T_H},VK_{static}^{zkGPT},VK_{Lasso},Layout_{in}$；
3. 吸入 $com_{base}$ 与输出 $Y$，重新生成 $\xi^{(\ell,r)},\xi_{cat}^{(\ell)},\xi_{out}$；
4. 吸入 $com_{comp}$ 并重新生成所有 $\lambda_{psq}$；
5. 吸入 $com_{psq},com_{qry}$；
6. 重放 Lasso 阶段挑战，包括每个二元 lookup family 的 $\eta_{\mathcal L}$、range structured-table 挑战和 Lasso 内部挑战；
7. 重放矩阵乘法阶段挑战；
8. 重放 GKR 阶段挑战；
9. 重放输入层合并与 masked opening 阶段挑战。

任一挑战与 proof 中隐含挑战不一致，验证拒绝。

### 4.3 Lasso 验证

验证者按第 3.2.2 节与第 3.5.3 节固定顺序，对每个 lookup family 执行 Lasso verification：

1. 对 hidden range，按 $\ell,r$ 检查 query 长度、padding 长度、table id 与 $Layout_{in}$ 一致，并验证其属于 $T_{range}$；
2. 对 output range，按 $s$ 执行同样检查；
3. 对 hidden LeakyReLU，按 $\ell,r$ 验证二元 raw query 经 $\eta_{\mathcal L}$ 压缩后属于压缩后的 $T_{LReLU}$；
4. 对 output LeakyReLU，按 $s$ 执行同样检查；
5. 对 hidden ELU，按 $\ell,r$ 验证二元 raw query 经 $\eta_{\mathcal L}$ 压缩后属于压缩后的 $T_{ELU}$；
6. 对 hidden exp，按 $\ell,r$ 验证二元 raw query 经 $\eta_{\mathcal L}$ 压缩后属于压缩后的 $T_{exp}$；
7. 对 output exp，按 $s$ 执行同样检查；
8. 对 rescale/rounding range，按 $Layout_{in}$ 中声明顺序验证其属于对应结构化范围表。

对所有 lookup family，验证者还必须检查 lookup table id、query family tag、raw query commitment、query 长度、padding 长度与 $VK_{Lasso}$ 和 $Layout_{in}$ 一致。Lasso 只证明 query 属于表；query 与基础变量的一致性由 GKR 的 lookup query consistency 约束检查。两者缺一不可。

### 4.4 矩阵乘法证明验证

验证者按 $C_{matmul}$ 的 claim table 验证 $\Pi_{MatMul}$：

1. 对每个矩阵乘法或 batched matrix multiplication group，检查 shape、padding shape、batch size 与 $Layout_{in}$ 一致。
2. 运行矩阵乘法 sumcheck verification，将输出 claim 归约为输入矩阵对应的输入层 claim；非公开输入矩阵最终只进入 masked-opening claim，不直接暴露裸 MLE evaluation。
3. 对 grouping/bookkeeping table 相关消息检查 transcript 顺序和随机点一致。
4. 将最终输入矩阵 claim 作为 masked-opening claim 送入第 4.6 节的输入层合并验证；公开输入相关 claim 由验证者本地评值。

若任一矩阵乘法 claim 验证失败，直接拒绝。

### 4.5 GKR 验证

验证者对 $\Pi_{GKR}$ 执行 ML-friendly GKR verification：

1. 检查 $C_{GAT}$ 和 $circuit\_layout\_id$ 是否与本地配置一致。
2. 检查输出 claim 是否绑定到证明对象中的 $Y$。
3. 从输出层随机点开始，逐层验证 sumcheck。
4. 对每层的加法门、乘法门、常量门、输入层连接门执行标准 GKR 检查。
5. 检查 circuit squeeze 后的层数、每层 gate 数和输入层连接表是否与 $Layout_{in}$ 一致。
6. 将最终输入层 claim 送入第 4.6 节。

GKR 验证覆盖：最大值状态机、PSQ 状态机、逆元乘法、拼接、压缩绑定、bias add、最终平均、lookup query consistency、普通 routing arithmetic consistency。

### 4.6 输入层合并与 masked opening 验证

验证者收集来自 Lasso、矩阵乘法 sumcheck、GKR 的全部输入层 claim，运行 $\Pi_{combine}$ 验证，把这些 claim 合并为对 $\widetilde T_H,\widetilde W,\widetilde{Aux}_{base},\widetilde{Aux}_{comp},\widetilde{Aux}_{psq},\widetilde{L}_{qry}$ 的 masked-opening claim，以及公开 $\widetilde X_{pub}$ 的本地评值 claim。

随后执行：

1. 验证 $zkPC.VerifyMasked(com_{T_H},z_T,\widehat T_H(z_T),\pi_T^{mask},pp)=1$；
2. 验证 $zkPC.VerifyMasked(com_W,z_W,\widehat W(z_W),\pi_W^{mask},pp)=1$；
3. 验证 $zkPC.VerifyMasked(com_{base},z_{base},\widehat{Aux}_{base}(z_{base}),\pi_{base}^{mask},pp)=1$；
4. 验证 $zkPC.VerifyMasked(com_{comp},z_{comp},\widehat{Aux}_{comp}(z_{comp}),\pi_{comp}^{mask},pp)=1$；
5. 验证 $zkPC.VerifyMasked(com_{psq},z_{psq},\widehat{Aux}_{psq}(z_{psq}),\pi_{psq}^{mask},pp)=1$；
6. 验证 $zkPC.VerifyMasked(com_{qry},z_{qry},\widehat{L}_{qry}(z_{qry}),\pi_{qry}^{mask},pp)=1$；
7. 对 $X_{pub}$，验证者本地计算 $\widetilde X_{pub}(z_X)$，并检查 $Layout_{in}$ 指定的分段合并等式成立。

若采用其他段编码，以 $Layout_{in}$ 中固定公式为准。

### 4.7 验证结论

验证者在且仅在以下条件全部满足时接受：

1. 元数据、公共输入、拓扑、选择器、公开 routing wiring 索引与特征表 indexed access 布局全部一致；
2. transcript 重放得到的所有挑战一致；
3. 所有 Lasso lookup 证明通过；
4. 所有矩阵乘法 sumcheck 证明通过；
5. GKR arithmetic circuit 证明通过；
6. 输入层 claim 合并证明通过；
7. $com_{T_H},com_W,com_{base},com_{comp},com_{psq},com_{qry}$ 的 masked opening 证明通过；
8. 输出 $Y$ 与 GKR 输出 claim 绑定一致；
9. proof object 未出现多余字段、缺失字段、重复 tag 或顺序错误。

否则输出拒绝。
