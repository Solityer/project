# GAT-ZKML（代码实现与论文撰写终稿，修订版）
Gpu终态

> - `zkMaP / batch-zkMaP`：用于矩阵乘、矩阵—向量乘以及外点评值批量证明；
> - `zkVC-CRPC / PSQ`：用于矩阵乘编码与分组前缀和约束压缩；
> - `LogUp / lookup`：用于特征检索、节点到边广播、LeakyReLU、指数映射与范围检查。

## 0. 符号

### 0.1 有限域、双线性群与承诺系统

- 记有限域为$\mathbb F_p$，其中 $p$ 为大素数。（所有运算结果都对 $p$ 取模，保证不溢出）

	> 定点数与符号约定：采用定点数量化。对于域元素 $x \in \mathbb F_p$，若其数值大于 $(p-1)/2$，在语义上视为负数 $x-p$。模型中的 LeakyReLU 激活函数及算术比较逻辑均基于此“大数即负数”的约定实现。

- 记双线性群为$(\mathbb G_1,\mathbb G_2,\mathbb G_T,e)$，其中$e: \mathbb G_1 \times \mathbb G_2 \rightarrow \mathbb G_T$ 是非退化双线性映射

- 记生成元为$G_1 \in \mathbb G_1, \quad G_2 \in \mathbb G_2$

- 使用 KZG 多项式承诺。对一个次数严格小于 $D$ 的多项式 $P(X)$，其承诺记为$[P] = P(\tau) G_1$$，其中 $\tau$ 为 setup 阶段采样的隐藏陷门

- 随机预言机写作$H_{FS}(\cdot)$

### 0.2 图、局部子图与索引

- 全局总节点数：$N_{total}$
- 当前局部子图节点数：$N$
- 当前局部子图边数：$E$
- 局部节点绝对编号序列记为$I = (I_0,I_1,\ldots,I_{N-1})$$，其中 $I_i$ 表示局部节点 $i$ 在全局图中的绝对编号
- 局部边源索引序列记为：$src = (src(0),src(1),\ldots,src(E-1))$
- 局部边目标索引序列记为：$dst = (dst(0),dst(1),\ldots,dst(E-1))$
- 对任意 $k\in\{0,1,\ldots,E-1\}$，都有$src(k) \in \{0,1,\ldots,N-1\}, \quad dst(k) \in \{0,1,\ldots,N-1\}$

**排序约定** 固定要求边序列按目标节点索引 $dst(k)$ 非降序排列。对任意满足 $0\leq k_1 < k_2 \leq E-1$ 的整数 $k_1,k_2$，都有

$$dst(k_1) \leq dst(k_2)$$

如果原始输入边序列没有满足这一条件，则在见证生成前先做一次确定性稳定排序。这个排序结果属于证明者的见证预处理过程；验证者只接收已经按此规范生成的公开索引序列。

> 为什么要强行排序：这是实现 PSQ的前置条件。GAT的核心是“每个目标节点聚合其所有邻居的信息”。 如果边是乱序的，在证明求和时就需要频繁随机寻址。按目标节点排好序后，指向同一个点的边在数组里就是连在一起的一段。只需要从头扫到尾，一段一段地处理，大大降低了ZK电路的开销。

### 0.3 模型维度与静态表

- 输入特征维度：

	$$d_{in}$$

- 中间隐藏维度：

	$$d$$

- 输出类别数：

	$$C$$

- 范围检查位宽：$B$ ，因此范围表覆盖：

	$$T_{range}=\{0,1,2,\ldots,2^B-1\}$$

	> 证明一个数是正数的最快方法是证明“这个数在 0 到 255 的预设表格里”。$B$ 就是这个表的位数（如 $B=8$ 代表 0~255）。

- 全局特征表：

	$$T_H \in \mathbb F_p^{N_{total}\times d_{in}}$$

- LeakyReLU 查表：

	$$T_{LReLU} \subseteq \mathbb F_p \times \mathbb F_p$$

- 指数查表：

	$$T_{exp} \subseteq \mathbb F_p \times \mathbb F_p$$

- 范围表：

	$$T_{range}=\{0,1,2,\ldots,2^B-1\}$$

为了避免 lookup 多值歧义，明确要求：

- 对任意输入 $x\in \mathbb F_p$，在 $T_{LReLU}$ 中至多存在一个输出 $y\in \mathbb F_p$ 使得

	$$(x,y)\in T_{LReLU}$$

- 对任意输入 $u\in \mathbb F_p$，在 $T_{exp}$ 中至多存在一个输出 $v\in \mathbb F_p$ 使得

	$$(u,v)\in T_{exp}$$

这个假设保证：当 lookup 证明“查询二元组属于表”时，输出值由输入值唯一确定

### 0.4 模型参数

模型参数固定，记为：

- 第一层投影矩阵：

	$$W \in \mathbb F_p^{d_{in}\times d}$$

- 源方向注意力向量：

	$$a_{src} \in \mathbb F_p^{d}$$

- 目标方向注意力向量：

	$$a_{dst} \in \mathbb F_p^{d}$$

- 输出层权重矩阵：

	$$W_{out} \in \mathbb F_p^{d\times C}$$

- 输出层偏置向量：

	$$b \in \mathbb F_p^{C}$$

默认这些参数在多次证明中固定，因此直接预处理进模型验证键 $VK_{model}$，而不是在每次证明中重复提交承诺。

### 0.5 中间变量

#### 0.5.1 节点域变量

- 局部原始特征矩阵：

	$$H \in \mathbb F_p^{N\times d_{in}}, \quad H_{i,j}=T_H[I_i,j]$$

- 第一层投影结果：

	$$H' \in \mathbb F_p^{N\times d}$$

- 节点域源注意力：

	$$E_{src} \in \mathbb F_p^{N}$$

- 节点域目标注意力：

	$$E_{dst} \in \mathbb F_p^{N}$$

- 节点域随机维压缩特征：

	$$H^{\star} \in \mathbb F_p^{N}$$

- 节点域组最大值：

	$$M \in \mathbb F_p^{N}$$

- 节点域 Softmax 分母：

	$$Sum \in \mathbb F_p^{N}$$

- 节点域分母逆元：

	$$inv \in \mathbb F_p^{N}, \quad inv_i = Sum_i^{-1}$$

	 （该定义依赖第 0.8.4 节中的非零分母前提。）

- 节点域聚合特征矩阵：

	$$H_{agg} \in \mathbb F_p^{N\times d}$$

- 节点域随机维压缩聚合特征：

	$$H_{agg}^{\star} \in \mathbb F_p^{N}$$

- 输出预测矩阵：

	$$Y \in \mathbb F_p^{N\times C}$$

#### 0.5.2 边域变量

- 边域源注意力广播：

	$$E_{src}^{edge} \in \mathbb F_p^{E}, \qquad E_{src,k}^{edge}=E_{src,src(k)}$$

- 边域目标注意力广播：

	$$E_{dst}^{edge} \in \mathbb F_p^{E}, \qquad E_{dst,k}^{edge}=E_{dst,dst(k)}$$

- 边域源随机维压缩特征广播：

	$$H_{src}^{\star,edge} \in \mathbb F_p^{E}, \qquad H_{src,k}^{\star,edge}=H_{src(k)}^{\star}$$

- 边域聚合压缩特征广播：

	$$H_{agg}^{\star,edge} \in \mathbb F_p^{E}, \qquad H_{agg,k}^{\star,edge}=H_{agg,dst(k)}^{\star}$$

- 边域分母广播：

	$$Sum^{edge} \in \mathbb F_p^{E}, \qquad Sum_k^{edge}=Sum_{dst(k)}$$

- 边域逆元广播：

	$$inv^{edge} \in \mathbb F_p^{E}, \qquad inv_k^{edge}=inv_{dst(k)}$$

- 边域线性打分：

	$$S \in \mathbb F_p^{E}, \qquad S_k = E_{src,k}^{edge}+E_{dst,k}^{edge}$$

- 边域激活后打分：

	$$Z \in \mathbb F_p^{E}, \qquad Z_k=LReLU(S_k)$$

- 边域最大值广播：

	$$M^{edge} \in \mathbb F_p^{E}, \qquad M_k^{edge}=M_{dst(k)}$$

- 边域非负差分：

	$$\Delta^+ \in \mathbb F_p^{E}, \qquad \Delta_k^+=M_k^{edge}-Z_k$$

- 边域指数映射输出：

	$$U \in \mathbb F_p^{E}, \qquad U_k=ExpMap(\Delta_k^+)$$

- 边域归一化注意力：

	$$\alpha \in \mathbb F_p^{E}, \qquad \alpha_k = U_k \cdot inv_k^{edge}$$

- 边域压缩加权特征：

	$$\widehat v^{\star} \in \mathbb F_p^{E}, \qquad \widehat v_k^{\star}=\alpha_k \cdot H_{src,k}^{\star,edge}$$

> 点到边（广播）：点身上的属性分给每一条出发的边。此时数据从节点域进入边域。
>
> 边上算分：算邻居间的注意力 $\alpha$。
>
> 边到点（聚合）：根据好感度把邻居的信息收回来。这是 GAT 的核心循环。

#### 0.5.3 压缩挑战与向量

令

$$\xi \in \mathbb F_p$$

为随机维压缩挑战，定义向量：

$$c_{\xi} = (1,\xi,\xi^2,\ldots,\xi^{d-1})^\top \in \mathbb F_p^{d}$$

于是有：

$$H_i^{\star} = \sum_{j=0}^{d-1} H'_{i,j}\xi^j$$

，以及：

$$H_{agg,i}^{\star} = \sum_{j=0}^{d-1} H_{agg,i,j}\xi^j$$

> 为什么要压缩：特征向量通常是 128 维。如果要证明每一维特征在边上是怎么传的，开销会很大。
>
> 方案：用随机数 $\xi$ 把 128 维压缩成一个数。只要这个压缩后的数在求和时对得上，原始那 128 维就极大概率都对。

#### 0.5.4 公开输入与动态见证

为避免证明者提交对象与验证者本地重建对象混淆，固定采用以下分层：

公开输入 / 公开派生对象：由验证者根据公共输入直接重建，不进入动态承诺集合 $\mathbf{Com}_{dyn}$，包括：

- 节点绝对编号多项式 $P_I$；
- 拓扑索引多项式 $P_{src},P_{dst}$；
- 组选择器与有效区选择器多项式：$P_{Q_{new}^{edge}},P_{Q_{end}^{edge}},P_{Q_{edge}^{valid}},P_{Q_N},P_{Q_{proj}^{valid}},P_{Q_d^{valid}}$，以及各 lookup 的 $P_{Q_{tbl}},P_{Q_{qry}}$；其中 route 节点端直接复用 $P_{Q_N}$，edge 查询端使用 $P_{Q_{qry}^{src}},P_{Q_{qry}^{dst}}$，并约定 $P_{Q_{qry}^{src}}=P_{Q_{qry}^{dst}}=P_{Q_{edge}^{valid}}$。
- 静态表承诺与固定模型参数承诺。

动态见证对象：由证明者生成并承诺，包括 $P_H,P_{H'},P_{E_{src}},P_{E_{dst}},P_M,P_{Sum},P_{inv},P_{H_{agg}},P_Y$ 以及所有 lookup / route / zkMaP / PSQ 所需的辅助多项式。

公开标量对象：由证明者给出、验证者直接使用、不经过 KZG 开放，包括双累加器 route 的公开总和 $S_{src},S_{dst}$。它们属于最终证明对象中的公开评值字段，而不是新的多项式承诺。

凡是出现“吸入 $P_I$”的地方，均指验证者先根据公共输入重建 $P_I$ 或其承诺，再将该公开对象按固定顺序吸入 transcript；它不是证明者额外提交的动态见证。

### 0.6 分组选择器

由于边序列按 $dst(k)$ 非降序排列，因此可以定义组起点选择器、组末尾选择器以及边域有效区选择器：

- 组起点选择器：对任意 $k\in\{0,1,\ldots,E-1\}$，定义

	$$Q_{new}^{edge}[k]= \begin{cases} 1, & k=0\\ 1, & 1\leq k\leq E-1 \text{ 且 } dst(k)\neq dst(k-1)\\ 0, & 1\leq k\leq E-1 \text{ 且 } dst(k)=dst(k-1) \end{cases}$$

- 组末尾选择器：对任意 $k\in\{0,1,\ldots,E-1\}$，定义

	$$Q_{end}^{edge}[k]= \begin{cases} 1, & k=E-1,\\ 1, & 0\leq k\leq E-2 \text{ 且 } dst(k+1)\neq dst(k),\\ 0, & 0\leq k\leq E-2 \text{ 且 } dst(k+1)=dst(k). \end{cases}$$

- 边域有效区选择器：对任意 $k\in\{0,1,\ldots,n_{edge}-1\}$，定义

	$$Q_{edge}^{valid}[k]=\begin{cases}1,&0\le k\le E-1\\0,&E\le k\le n_{edge}-1\end{cases}$$

定义其插值多项式：

$$P_{Q_{edge}^{valid}}(X)=\sum_{k=0}^{n_{edge}-1} Q_{edge}^{valid}[k] L_k^{(edge)}(X).$$

此外，节点域、$d_{in}$ 域与 $d$ 域的公共有效区选择器固定定义为：

$$Q_N[i]=\begin{cases}1,&0\le i\le N-1\\0,&N\le i\le n_N-1\end{cases}$$

$$Q_{proj}^{valid}[j]=\begin{cases}1,&0\le j\le d_{in}-1\\0,&d_{in}\le j\le n_{in}-1\end{cases}$$

$$Q_d^{valid}[j]=\begin{cases}1,&0\le j\le d-1\\0,&d\le j\le n_d-1\end{cases}$$

其插值多项式分别为：

$$P_{Q_N}(X)=\sum_{i=0}^{n_N-1} Q_N[i]L_i^{(N)}(X),$$

$$P_{Q_{proj}^{valid}}(X)=\sum_{j=0}^{n_{in}-1} Q_{proj}^{valid}[j]L_j^{(in)}(X),$$

$$P_{Q_d^{valid}}(X)=\sum_{j=0}^{n_d-1} Q_d^{valid}[j]L_j^{(d)}(X).$$

对于 route 子协议，不再单独引入新的 node 端表选择器：node 端统一复用 $Q_N$；edge 查询端统一定义

$$Q_{qry}^{src}[k]=Q_{qry}^{dst}[k]=Q_{edge}^{valid}[k],\qquad 0\le k\le n_{edge}-1,$$

从而

$$P_{Q_{qry}^{src}}(X)=P_{Q_{qry}^{dst}}(X)=P_{Q_{edge}^{valid}}(X).$$

> $Q_{new}$：相当于聚合的“起点灯”，亮了就代表这组数据开始算了。
>
> $Q_{end}$：相当于聚合的“结算灯”，亮了就把累加的结果上交给节点。
>
> $Q_{edge}^{valid}$：屏蔽边域 padding 区，使状态机在无效区只能保持常值，不能伪造额外转移。

### 0.7 展平规则

#### 0.7.1 矩阵行优先展平

对一个矩阵 $M\in \mathbb F_p^{r\times c}$，定义行优先展平索引：

$$\operatorname{flat}_{r,c}(i,j)=i\cdot c + j$$

，其中 $0\leq i\leq r-1$，$0\leq j\leq c-1$

#### 0.7.2 CRPC编码

> CRPC编码是把矩阵乘法变成多项式加法

对矩阵乘法

$$A\in \mathbb F_p^{m\times \ell}, \quad B\in \mathbb F_p^{\ell\times n}, \quad C=A\cdot B\in \mathbb F_p^{m\times n}$$

定义输出矩阵系数多项式：

$$P_C(X)=\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} C_{i,j} X^{i\cdot n + j}$$

对共享维 $t\in\{0,1,\ldots,\ell-1\}$，定义：

$$A_t^{\langle n\rangle}(X)=\sum_{i=0}^{m-1} A_{i,t} X^{i\cdot n} \quad B_t(X)=\sum_{j=0}^{n-1} B_{t,j} X^{j}$$

于是有：

$$P_C(X)=\sum_{t=0}^{\ell-1} A_t^{\langle n\rangle}(X)\cdot B_t(X)$$

#### 0.7.3  zkMaP张量绑定约束

为了防止恶意证明者在后续的 zkMaP 操作中，单独伪造折叠向量多项式（例如 $P_{a^{proj}}(X)$ 和 $P_{b^{proj}}(X)$）以满足点积约束，协议明确规定：

所有形如 $a_m = \sum_i M_{i,m}\,\chi_i(y)$ 或 $b_m = \sum_j W_{m,j}\,\psi_j(y)$ 的一维折叠向量，必须与其关联的二维矩阵承诺（如 $[P_H]$、$[P_{H'}]$、$[P_{H_{agg}}]$、$[V_W]$、$[V_{W_{out}}]$）在数学上强绑定。

其中 $\chi_i(y)$ 与 $\psi_j(y)$ 由具体编码方式决定：

- 对节点域 Lagrange 编码对象，使用节点域基函数权重 $L_i^{(N)}(y)$；
- 对 CRPC 系数编码对象，使用相应的幂权重 $y^{i\cdot n}$ 或 $y^j$。

本规范把该绑定作为最终证明对象中的显式子证明族：

$$\Pi_{bind}=\big(\pi_{bind}^{proj},\pi_{bind}^{src},\pi_{bind}^{dst},\pi_{bind}^{\star},\pi_{bind}^{agg},\pi_{bind}^{out}\big).$$

每个 $\pi_{bind}^{(\cdot)}$ 必须证明对应折叠向量确实来自已经承诺的二维矩阵 / 向量对象。后端可采用以下任一种等价实现：

1. 单变量 Sumcheck：证明折叠向量是底层张量在指定挑战点上的加权和；
2. 同态交叉项检查：按原始 CRPC / zkMaP 方案构造绑定商多项式；
3. 任何与上述两者安全等价、且在代数群模型下可验证的张量绑定后端。

所有绑定子证明都使用主 transcript 派生出的域分离标签，但不改变第 3.1 节主 Fiat–Shamir 挑战顺序。固定的一级标签为：

- $\mathrm{DST}_{bind}^{proj}=\texttt{"GAT-ZKML/BIND/PROJ"}$；
- $\mathrm{DST}_{bind}^{src}=\texttt{"GAT-ZKML/BIND/SRC"}$；
- $\mathrm{DST}_{bind}^{dst}=\texttt{"GAT-ZKML/BIND/DST"}$；
- $\mathrm{DST}_{bind}^{\star}=\texttt{"GAT-ZKML/BIND/STAR"}$；
- $\mathrm{DST}_{bind}^{agg}=\texttt{"GAT-ZKML/BIND/AGG"}$；
- $\mathrm{DST}_{bind}^{out}=\texttt{"GAT-ZKML/BIND/OUT"}$。

若某一组子证明的内部后端还需要区分具体来源对象（例如 $W$ 与 $W_{out}$，或同一组中的矩阵承诺与偏置向量承诺），则必须在对应一级标签后继续拼接二级标签，例如 $\texttt{"GAT-ZKML/BIND/OUT/W_OUT"}$、$\texttt{"GAT-ZKML/BIND/OUT/BIAS"}$。验证者在校验 $\Pi_{bind}$ 时，必须逐组使用完全相同的域分离标签初始化其内部 transcript。

### 0.8 实现规范与安全约束

#### 0.8.1 零分母冲突

对任意 LogUp 子系统，都会出现形如

$$\frac{1}{Table+\beta}, \quad \frac{1}{Query+\beta}$$

 的项。

为了使分母可逆，需要排除零分母集合。对任意 LogUp 子系统 $\mathcal L$，其零分母集合定义为

$$Bad_{\mathcal L} = \{-Table[t] \mid t\text{ 在表有效区}\} \cup \{-Query[t] \mid t\text{ 在查询有效区}\}$$

明确要求：对每个 lookup / route 子系统，Fiat–Shamir 挑战在基础值承诺固定之后、派生见证（$Table$ / $Query$ / $m$ / $R$）生成之前采样；安全性要求是：挑战采样时，所有决定 $Query$ / $Table$ 基础值的对象都已不可篡改。

并且验证者语义上隐含事件

$$\beta_{\mathcal L} \notin Bad_{\mathcal L}$$

，这是标准随机预言机下的高概率事件。

> 分母不能为零，否则计算就崩了。通过随机挑战 $\beta$ 确保永远不会撞上那个让分母为零的数字。

#### 0.8.2 二次幂填充规则

对每个工作域 $\mathbb H_{\mathcal D}$ 上的离散向量，若真实长度为 $L_{\mathcal D}$，而工作域长度为 $n_{\mathcal D}$，则对所有 $t\in\{L_{\mathcal D},L_{\mathcal D}+1,\ldots,n_{\mathcal D}-1\}$，统一定义：

- 见证值填零；
- 重数填零；
- 查询 / 表项的无效位置填零；
- 由有效区选择器 $Q_{tbl}$、$Q_{qry}$、$Q_{edge}^{valid}$ 屏蔽无效位置；
- 状态机连续性约束：对于所有累加器多项式（如 $P_R, P_{C_{max}}, P_{PSQ}$），在有效区结束后的填充位置，其值必须强制等于该有效区最后一个点的值，或者通过有效区选择器把填充区的递推项强制清零，以防证明者在无效区伪造状态。

对本文中两类边域状态机，进一步固定采用以下 padding 规则：

- 最大值唯一性状态机：对所有 $k\ge E$，规定 $s_{max}[k]=0$，并强制 $C_{max}[k]=C_{max}[E-1]$；
- 压缩 PSQ 状态机：对所有 $k\ge E$，规定 $w_{psq}[k]=0$、$T_{psq}^{edge}[k]=0$，并强制 $PSQ[k]=PSQ[E-1]$。

> $P(x)=P(\omega^{-1}x)$ 这里的 $\omega$ 指代的是具体对应工作域的生成元。
>
> ZK 里的多项式长度通常取 2 的次幂；如果真实数据较短，剩余位置全部由上述规则统一填充。
>
> 记号约定：若某一节为了突出真实语义，只把边域/节点域离散对象写到真实区末尾（例如写成 $\sum_{k=0}^{E-1}$ 或 $\sum_{i=0}^{N-1}$），则其精确定义仍然是“先按本节 padding 规则补到完整工作域，再在整个工作域上做插值”。实现代码必须生成完整长度为 $n_{\mathcal D}$ 的数组，不能只存真实区后直接跳过插值补零。
>
> 边界规则：当某个工作域的真实长度恰好满足 $L_{\mathcal D}=n_{\mathcal D}$ 时，padding 区为空集。此时实现层必须跳过所有“对 $t\in\{L_{\mathcal D},\ldots,n_{\mathcal D}-1\}$”形式的循环，不得访问不存在的填充区首元素，也不得额外生成任何 padding 评值。协议层对此视为空约束集。

#### 0.8.3 多项式次数约束

对每一个定义在工作域 $\mathbb H_{\mathcal D}$ 上的插值多项式 $P$，都要求：

$$\deg P < n_{\mathcal D}$$

#### 0.8.4 Softmax 的数学可行性

为了使第 2.6–2.8 步中的最大值、分组求和与逆元约束有定义，显式加入以下前提：

1. 对每个局部节点 $i\in\{0,1,\ldots,N-1\}$，局部子图中至少存在一条以 $i$ 为目标的边。也就是说

$$\#\{k\in\{0,1,\ldots,E-1\}\mid dst(k)=i\} \ge 1$$

。因而 $M_i=\max\{Z_k\mid dst(k)=i\}$ 在每个组上都有定义。

> 在工程数据预处理阶段，应确保为所有真实节点添加自环（Self-loop），从而静态保证所有有效节点的 $Sum_i \neq 0$。而对于因二次幂填充（Padding）产生的无效节点，相关变量直接填 0 即可，有效区选择器 $Q_N$ 会自动屏蔽其逆元约束检查

2. 对每个局部节点 $i$，其 Softmax 分母

	$$Sum_i=\sum_{\{k\mid dst(k)=i\}} U_k$$

	 在域 $\mathbb F_p$ 中非零，因此逆元

	$$inv_i=Sum_i^{-1}$$

	有定义。

3. 溢出安全边界：实现时必须静态保证所有合法输入产生的 $Sum_i$ 在实数域上的最大值小于有限域模数 $p$（即 $\max(\text{degree}(i)) \cdot \max(T_{exp,y}) < p$）。这确保了 $Sum_i$ 在 $\mathbb F_p$ 中的非零属性是由输入数值逻辑保证的，而非模运算偶然产生的回卷结果。

4. 量化极小值下限：量化方案必须保证有效节点的最小聚合值在量化后至少为 1（即 $Sum_i \ge 1$）。杜绝极端情况下浮点数极小导致在定点数 $\mathbb F_p$ 语义下被截断为 0，从而引发零分母崩溃。

第二条并不是由 lookup 本身自动推出的，而是量化表、域大小与输入范围共同满足的参数选择前提；实现时必须保证所有合法输入都不会使任一组的 $Sum_i$ 在 $\mathbb F_p$ 中回卷为零。

> 每个点必须至少有个邻居。指数和不能为零（不能除以零）。

#### 0.8.5 指数表与归一化链路的量化缩放因子

协议采用定点数量化，所有与 Softmax 相关的缩放因子都必须作为公共参数固定，并在静态表生成、见证生成与验证实现中保持一致。记：

- $S_{\Delta}$：差分 $\Delta_k^+$ 的反量化尺度；
- $S_{exp}$：指数表输出 $U_k$ 的量化尺度；
- $S_{inv}$：逆元 witness $inv_i$ 的量化尺度；
- $S_{\alpha}$：归一化权重 $\alpha_k$ 的量化尺度；
- $S_{agg}$：聚合特征 $H_{agg}$ 与 $H_{agg}^{\star}$ 的量化尺度。

静态表 $T_{exp}$ 必须按以下公开规则离线生成：若其第一列存放的是量化差分 $u$，则第二列必须存放

$$v=\operatorname{Round}\!\left(S_{exp}\cdot e^{-u/S_{\Delta}}\right),$$

并把二元组 $(u,v)$ 写入 $T_{exp}$。

随后的归一化链路必须使用固定的 rescale 约定：

1. $U_k$ 统一按尺度 $S_{exp}$ 解释；
2. $Sum_i=\sum_{k:dst(k)=i}U_k$ 与 $U_k$ 共享同一尺度；
3. $inv_i$ 必须按公开规则编码为对 $Sum_i$ 的定点逆元近似，其尺度为 $S_{inv}$；
4. $\alpha_k$ 必须由 $U_k$ 与 $inv_{dst(k)}$ 按固定 rescale 规则得到，并写成尺度 $S_{\alpha}$ 的量化值；
5. 边级加权特征 $\widehat v_k^{\star}=\alpha_k H_{src,k}^{\star,edge}$ 以及节点级聚合值 $H_{agg}^{\star}$ 必须在公开约定的尺度转换后再进入 PSQ 约束。

本协议的代数约束只验证“域元素层面的等式”，并不替代量化规范本身。因此 $S_{\Delta},S_{exp},S_{inv},S_{\alpha},S_{agg}$、舍入方式（向最近整数舍入 / 截断）以及乘法后的 rescale 规则，都必须写入实现配置并视为公开参数的一部分。

### 0.9 统一约定

#### 0.9.1 热路径对象

本协议中的“热路径对象”定义为：在证明者 / 验证者主循环中被高频访问、其实现方式会直接决定时延量级的对象。包括：

1. 有限域元素：

	$$H_{i,j},\ H'_{i,j},\ E_{src,i},\ E_{dst,i},\ Z_k,\ M_i,\ \Delta_k^+,\ U_k,\ Sum_i,\ inv_i,\ \alpha_k,\ H_{agg,i,j},\ Y_{i,j}$$

	以及所有挑战值：

	$$\eta_{feat},\eta_{src},\eta_{dst},\eta_L,\eta_{exp},\beta_{feat},\beta_{src},\beta_{dst},\beta_L,\beta_{exp},\beta_R,\xi,\lambda_{psq},\alpha_{quot},\rho_{ext},\{v_{\mathcal D}\},\{z_{\mathcal D}\}$$

2. 曲线群元素：

	$$[V_P]\in G_1, \quad [W_{\mathcal D}]\in G_1, \quad [W_{ext}]\in G_1, \quad [1]_2,[\tau]_2\in G_2$$

3. SRS、静态验证键与模型验证键：

	$$PK=\{[\tau^0]_1,[\tau^1]_1,\dots,[\tau^{D_{max}}]_1\}, \quad VK_{KZG}=\{[1]_2,[\tau]_2\}$$

	$$VK_{static}=\{[V_{T_H}],[V_{T_{LReLU},x}],[V_{T_{LReLU},y}],[V_{T_{exp},x}],[V_{T_{exp},y}],[V_{T_{range}}]\}$$

	$$VK_{model}=\{[V_W],[V_{a_{src}}],[V_{a_{dst}}],[V_{W_{out}}],[V_b]\}$$

#### 0.9.2 原生对象要求

所有热路径对象必须以原生类型保存：

- 所有有限域元素必须直接表示为底层原生域元素类型；

- 所有群元素必须直接表示为底层原生曲线点类型；

- 禁止在主循环中进行如下往返转换：

	$$\texttt{FieldElement}\to \texttt{string}\to \texttt{Fr}, \quad \texttt{CurvePoint}\to \texttt{bytes}\to \texttt{G1/G2}$$

若实现不满足这一点，则即使协议层已经压缩证明尺寸，实际运行时间仍会被字符串解析、对象重建、内存复制主导。

#### 0.9.3 静态缓存要求

在程序启动后、第一次 proving / verifying 之前，必须完成以下一次性初始化：

1. 反序列化并缓存 SRS：

	$$PK,\ VK_{KZG}$$

2. 反序列化并缓存静态表承诺：

	$$VK_{static}$$

3. 反序列化并缓存模型承诺：

	$$VK_{model}$$

4. 为各工作域构建 FFT / IFFT 计划：

	$$\mathbb H_{FH},\ \mathbb H_{edge},\ \mathbb H_{in},\ \mathbb H_d,\ \mathbb H_N$$

	 ，预计算缓存单位根表，并为商多项式计算额外构建 $4n_{FH}$ 和 $4n_{edge}$ 规模的扩展域 FFT / IFFT 计划，防止 LogUp 分子计算时的混叠效应。

	> 增加 $4n$ 规模的 FFT 计划缓存，支持 $3n$ 阶 LogUp 分子多项式的点对点除法，防止计算混叠

以上对象在单次证明 / 验证期间不得重复构造

## 1. 参数生成

**输入** 参数生成算法的输入为：

- 安全参数 $\lambda$
- 有限域大小 $p$
- 局部子图规模上界 $N,E$
- 全局节点数 $N_{total}$
- 模型维度 $d_{in},d,C$
- 范围检查位宽 $B$
- 静态表 $T_H,T_{LReLU},T_{exp},T_{range}$
- 固定模型参数 $W,a_{src},a_{dst},W_{out},b$

**输出** 参数生成算法输出：

- KZG 证明键 $PK$
- KZG 验证键 $VK_{KZG}$
- 静态表验证键 $VK_{static}$
- 模型验证键 $VK_{model}$
- 各工作域 $\mathbb H_{FH},\mathbb H_{edge},\mathbb H_{in},\mathbb H_d,\mathbb H_N$
- 与这些工作域相关的零化多项式

### 1.1 工作域

采用五类工作域，而不是把所有对象塞进一个巨大的统一域。

#### 1.1.1 特征检索域

全局特征表展平长度为 $N_{total} \cdot d_{in}$，局部特征查询长度为 $N \cdot d_{in}$。

取最小的二次幂长度 $n_{FH}$ 满足：

$$n_{FH} \geq \max\{N_{total} \cdot d_{in},N \cdot d_{in}\}+2$$

定义工作域：

$$\mathbb H_{FH} = \{1,\omega_{FH},\omega_{FH}^2,\ldots,\omega_{FH}^{n_{FH}-1}\}$$

零化多项式：

$$Z_{FH}(X)=X^{n_{FH}}-1$$

#### 1.1.2 边域

所有边级对象统一放在一个边域中。取最小的二次幂长度 $n_{edge}$ 满足：

$$n_{edge} \geq \max\{N,E,|T_{LReLU}|,|T_{exp}|,2^B\}+2$$

定义：

$$\mathbb H_{edge}=\{1,\omega_{edge},\omega_{edge}^2,\ldots,\omega_{edge}^{n_{edge}-1}\}$$

零化多项式：

$$Z_{edge}(X)=X^{n_{edge}}-1$$

为了便于表述，在不同子步骤中仍使用记号

$$\mathbb H_{src},\mathbb H_{dst},\mathbb H_{L},\mathbb H_{exp},\mathbb H_{R},\mathbb H_{PSQ}$$

，但它们全部等于 $\mathbb H_{edge}$

#### 1.1.3 $d_{in}$ 域

第一层矩阵乘的共享维长度是 $d_{in}$。取最小的二次幂长度 $n_{in}$ 满足：

$$n_{in} \geq d_{in}+2$$

定义：

$$\mathbb H_{in}=\{1,\omega_{in},\omega_{in}^2,\ldots,\omega_{in}^{n_{in}-1}\}$$

零化多项式：

$$Z_{in}(X)=X^{n_{in}}-1$$

#### 1.1.4  $d$ 域

注意力投影、压缩特征绑定、聚合压缩绑定、输出层矩阵乘的共享维长度都是 $d$。

取最小的二次幂长度 $n_d$ 满足：

$$n_d \geq d+2$$

定义：

$$\mathbb H_d=\{1,\omega_d,\omega_d^2,\ldots,\omega_d^{n_d-1}\}$$

零化多项式：

$$Z_d(X)=X^{n_d}-1$$

#### 1.1.5 节点域

节点级对象长度是 $N$。取最小的二次幂长度 $n_N$ 满足：

$$n_N \geq N+2$$

定义：

$$\mathbb H_N=\{1,\omega_N,\omega_N^2,\ldots,\omega_N^{n_N-1}\}$$

零化多项式：

$$Z_N(X)=X^{n_N}-1$$

> 在 ZK 证明中，最耗时的操作之一是 FFT（快速傅里叶变换），它的计算复杂度是 $O(n \log n)$。
>
> 如果不分域：所有的计算都要按照最大的数据量（比如全局特征表 $N_{total} \cdot d_{in}$，可能有几百万行）来开辟空间。即使只是在算一个 128 维的向量乘法，也要占用几百万个位置，这会产生巨大的计算浪费。
>
> 如果分域：节点相关的计算在小域做，边相关的在大域做。这让 FFT 的 $n$ 始终保持在“够用就好”的最小范围，速度提升非常显著。
>
> 在普通的编程里，如果有 $N$ 个节点，会用数组下标 `0, 1, 2, ..., N-1` 来找它们。但在 ZK（零知识证明）的多项式里，数组被转化成了多项式。$\mathbb H_N$ 里的每一个元素（$1, \omega_N, \omega_N^2, \ldots$）就是一个采样点：在点 $1$ 处，多项式的值代表第 0 个节点的数据、在点 $\omega_N$ 处，代表第 1 个节点的数据。

### 1.2 KZG初始化

采样隐藏陷门：

$$\tau \xleftarrow{\$} \mathbb F_p$$

设：

$$D_{max}=\max\{3n_{FH}+8, 3n_{edge}+8, 2n_{in}+8, 2n_d+8, 2n_N+8, Nd+C, NC+1\}$$

输出证明：

$$PK = \{G_1,\tau G_1,\tau^2 G_1,\ldots,\tau^{D_{max}} G_1\}$$

输出验证：

$$VK_{KZG}=\{[1]_2,[\tau]_2\}$$

> 在标准的 Plonk 算法中，如果商多项式次数太高，通常会被切割成几段，但这会增加证明者的计算压力（MSM 运算）。
>
> 为了性能，选择不切割高次项。由于像LeakyReLU这样的约束涉及到三个多项式连乘，LogUp 的状态转移方程涉及三个 $n$ 阶多项式的乘积，商多项式阶数约为 $2n$（分子 $3n$）。将次数上界提升至 $3n+8$ 是为了确保 LogUp 协议中的高阶分子多项式（涉及三个 $n$ 阶多项式连乘）无需进行多项式切分（Splitting），从而最大化证明者的 GPU 利用率并简化商多项式构造。额外加入 $+8$ 的安全缓冲，是为了防止复杂选择器多项式引入的少数额外次数导致最终商多项式溢出。

### 1.3 静态表多项式与模型承诺

> 在 ZK-GAT 证明开始之前，把不会随每一轮证明改变的数据（如全局数据库、激活函数表、模型权重）提前承诺
>
> 这样做的逻辑是：验证者只需要保存这些数据的承诺，而不需要在每次验证时都重新读取巨大的特征表或模型参数，极大提高了效率。

**全局特征表多项式** 定义展平后的全局特征表多项式：

$$P_{T_H}(X)=\sum_{v=0}^{N_{total}-1}\sum_{j=0}^{d_{in}-1} T_H[v,j] X^{v d_{in}+j}$$

**LeakyReLU 表多项式** 设 $T_{LReLU}$ 的第 $t$ 行是二元组 $(T_{LReLU}[t,0],T_{LReLU}[t,1])$。定义：

输入列表多项式：

$$P_{T_{LReLU},x}(X)=\sum_{t=0}^{|T_{LReLU}|-1} T_{LReLU}[t,0] L_t^{(L)}(X)$$

输出列表多项式：

$$P_{T_{LReLU},y}(X)=\sum_{t=0}^{|T_{LReLU}|-1} T_{LReLU}[t,1] L_t^{(L)}(X)$$

**指数表多项式** 设 $T_{exp}$ 的第 $t$ 行是二元组 $(T_{exp}[t,0],T_{exp}[t,1])$。

定义：

$$P_{T_{exp},x}(X)=\sum_{t=0}^{|T_{exp}|-1} T_{exp}[t,0] L_t^{(exp)}(X)$$

$$P_{T_{exp},y}(X)=\sum_{t=0}^{|T_{exp}|-1} T_{exp}[t,1] L_t^{(exp)}(X)$$

**范围表多项式** 定义：

$$P_{T_{range}}(X)=\sum_{t=0}^{2^B-1} t\,L_t^{(R)}(X)$$

**汇总**：以上静态多项式承诺全部进入 $VK_{static}$

**模型承诺** 对以下模型参数多项式做静态承诺：

- $W$ 的系数多项式承诺 $[V_W]$
- $a_{src}$ 的系数多项式承诺 $[V_{a_{src}}]$
- $a_{dst}$ 的系数多项式承诺 $[V_{a_{dst}}]$
- $W_{out}$ 的系数多项式承诺 $[V_{W_{out}}]$
- $b$ 的系数多项式承诺 $[V_b]$

这些承诺进入 $VK_{model}$，并在后文统一用 $[V_{\cdot}]$ 记号表示

### 1.4 公共拓扑多项式

在边域上插值多项式：

源索引多项式：

$$P_{src}(X)=\sum_{k=0}^{E-1} src(k) L_k^{(edge)}(X)$$

 （记录了每一条边的起点）

目标索引多项式：

$$P_{dst}(X)=\sum_{k=0}^{E-1} dst(k) L_k^{(edge)}(X)$$

 （记录了每一条边的终点）

组起点选择器：当某条边是一个新目标节点的第一个邻居时，它的值为 $1$，否则为 $0$。

定义：

$$P_{Q_{new}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1} Q_{new}^{edge}[k] L_k^{(edge)}(X)$$

组末尾选择器：当某条边是当前目标节点的最后一个邻居时，它的值为 $1$，否则为 $0$。

定义：

$$P_{Q_{end}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1} Q_{end}^{edge}[k] L_k^{(edge)}(X)$$

边域有效区选择器：用于在 padding 区屏蔽状态转移。

定义：

$$P_{Q_{edge}^{valid}}(X)=\sum_{k=0}^{n_{edge}-1} Q_{edge}^{valid}[k] L_k^{(edge)}(X)$$

## 2. 见证生成与承诺

### 2.0 转录顺序

除特别说明外，本节每个“本步骤承诺对象”都表示该步骤最终产出的承诺集合，不表示 Fiat–Shamir 的吸入顺序。

对所有依赖挑战的 lookup / route 子系统，统一采用以下无环标准表述：

1. Fiat–Shamir挑战在基础值承诺固定之后、派生见证（$Table$ / $Query$ / $m$ / $R$）生成之前采样
2. 安全性要求是：挑战采样时，所有决定 $Query$ / $Table$ 基础值的对象都已不可篡改。

虽然本节的步骤是按“计算 $\rightarrow$ 承诺 $\rightarrow$ 生成挑战”的协议依赖逻辑书写的，但在工程代码实现中，坚决不要把计算与承诺强行耦合在一起。

证明者的正确架构分为三阶段：

1. 前向计算：计算出全部的神经网络明文变量（$H', Z, M, U, Sum, inv, \alpha, H_{agg}, Y$），此阶段无任何密码学和多项式操作。
2. 挑战与Trace构建：根据严格的 Fiat-Shamir 顺序，生成诸如 $\xi, \eta, \beta$ 等挑战，并依赖这些挑战构造 $Table, Query, m, R$ 等派生见证轨迹列
3. 承诺与证明：集中调用并行的 MSM 与 FFT 生成承诺和最终的KZG开放证明

### 2.1 数据加载

**输入**：

- 全局特征表 $T_H$
- 局部节点绝对编号序列 $I$

#### 2.1.1 计算

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 和每个特征维 $j\in\{0,1,\ldots,d_{in}-1\}$，定义：

$$H_{i,j}=T_H[I_i,j]$$

#### 2.1.2 挑战生成与表项

根据 3.1 节规定的顺序，生成挑战 $\eta_{feat},\beta_{feat}$。

随后定义特征检索表端：对每个全局表端索引

$$u=v\cdot d_{in}+j,\qquad 0\le v\le N_{total}-1,\ 0\le j\le d_{in}-1,$$

定义

$$Table^{feat}[u]=v+\eta_{feat}\,j+\eta_{feat}^2 T_H[v,j].$$

定义局部查询端：对每个查询索引 $q=i d_{in}+j$，设：

$$Query^{feat}[q]=I_i+\eta_{feat}\,j+\eta_{feat}^2 H_{i,j}$$

> 为了证明数据的来源，需要把“位置”和“数值”绑定在一起。利用随机挑战 $\eta_{feat}$，将 节点ID、维度索引、特征值压缩成一个数字

#### 2.1.3 重数

对每个全局条目索引 $u=v \cdot d_{in}+j$，定义重数：

$$m_{feat}[u]=\#\{i\in\{0,1,\ldots,N-1\}\mid I_i=v\}$$

因为一个局部节点一旦命中全局节点 $v$，就会查询其全部 $d_{in}$ 维特征，所以同一节点 $v$ 的每个特征维都共享同一个命中次数。

#### 2.1.4 有效区选择器

表端有效区长度是 $N_{total} \cdot d_{in}$，定义：

$$Q_{tbl}^{feat}[t]= \begin{cases} 1, & 0\leq t \leq N_{total}d_{in}-1\\ 0, & N_{total}d_{in}\leq t \leq n_{FH}-1 \end{cases}$$

查询端有效区长度是 $N \cdot d_{in}$，定义：

$$Q_{qry}^{feat}[t]= \begin{cases} 1, & 0\leq t \leq Nd_{in}-1\\ 0, & Nd_{in}\leq t \leq n_{FH}-1 \end{cases}$$

> 因为多项式长度是固定的二次幂（$n_{FH}$），而实际数据可能没那么多，所以用 $Q$ 来标记哪些位置是真实数据，哪些是填充的零（Padding）。

#### 2.1.5 LogUp累加器

定义累加器离散状态：

$$R_{feat}[0]=0$$

对每个 $t\in\{0,1,\ldots,n_{FH}-2\}$，定义：

$$R_{feat}[t+1]=R_{feat}[t] + Q_{tbl}^{feat}[t]\cdot \frac{m_{feat}[t]}{Table^{feat}[t]+\beta_{feat}} - Q_{qry}^{feat}[t]\cdot \frac{1}{Query^{feat}[t]+\beta_{feat}}$$

> LogUp 在证什么： 在 GAT 中，我们要从全局库 $T_H$ 里调取节点特征。 证明者必须证明：他手里的 $H_{i,j}$ 确实来自于库里。 我们通过 $\eta$ 把 (ID, 维度, 数值) 打包成一个数字（指纹），然后通过累加器 $R_{feat}$ 证明“查询的指纹集合”确实包含在“库指纹集合”中。

#### 2.1.6 多项式编码

定义节点绝对编号多项式：

$$P_I(X)=\sum_{i=0}^{N-1} I_i L_i^{(N)}(X)$$

定义原始特征矩阵的系数多项式：

$$P_H(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{d_{in}-1} H_{i,j} X^{i d_{in}+j}$$

定义：

$$P_{Table^{feat}}(X)=\sum_{t=0}^{n_{FH}-1} Table^{feat}[t] L_t^{(FH)}(X)$$

$$P_{Query^{feat}}(X)=\sum_{t=0}^{n_{FH}-1} Query^{feat}[t] L_t^{(FH)}(X)$$

$$P_{m_{feat}}(X)=\sum_{t=0}^{n_{FH}-1} m_{feat}[t] L_t^{(FH)}(X)$$

$$P_{Q_{tbl}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1} Q_{tbl}^{feat}[t] L_t^{(FH)}(X)$$

$$P_{Q_{qry}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1} Q_{qry}^{feat}[t] L_t^{(FH)}(X)$$

$$P_{R_{feat}}(X)=\sum_{t=0}^{n_{FH}-1} R_{feat}[t] L_t^{(FH)}(X)$$

#### 2.1.7 承诺

证明者对下列动态多项式提交承诺：

$$[P_H], [P_{Table^{feat}}], [P_{Query^{feat}}], [P_{m_{feat}}], [P_{R_{feat}}]$$

其中 $[P_I]$ 由验证者根据公共输入本地重建，不属于动态承诺对象。

#### 2.1.8 输出

本步骤输出：

- 原始特征矩阵 $H$
- 原始特征多项式 $P_H$
- 特征检索 lookup / LogUp witness 多项式。

### 2.2 特征投影

**输入**：

- 原始特征矩阵 $H$
- 固定权重矩阵 $W$

#### 2.2.1 计算

对任意 $i\in\{0,1,\ldots,N-1\}$ 与 $j\in\{0,1,\ldots,d-1\}$，计算：

$$H'_{i,j}=\sum_{m=0}^{d_{in}-1} H_{i,m}W_{m,j}$$

#### 2.2.2 CRPC 编码

定义输出多项式：

$$P_{H'}(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{d-1} H'_{i,j} X^{i d + j}$$

对每个共享维索引 $m\in\{0,1,\ldots,d_{in}-1\}$，定义：

$$P_{H'}(X)=\sum_{m=0}^{d_{in}-1} A_m^{proj}(X)B_m^{proj}(X)$$

其中：

$$A_m^{proj}(X)=\sum_{i=0}^{N-1} H_{i,m} X^{i d} \text{(包含H的第m列)} \qquad B_m^{proj}(X)=\sum_{j=0}^{d-1} W_{m,j} X^j \text{(包含W的第m行)}$$

#### 2.2.3 zkMaP折叠

> 张量绑定说明： 根据第 0.7.3 节，此处的折叠向量 $a_m^{proj}$ 与 $b_m^{proj}$ 在实际电路中已被底层的 Tensor Binding 协议（如 Sumcheck）与承诺 $[P_H]$ 及 $[V_W]$ 强制绑定，防止证明者脱离原矩阵恶意捏造点积见证。

根据 3.1 节规定的顺序，生成挑战：

$$y_{proj}=H_{FS}(\text{transcript},[P_H],[P_{H'}],[V_W])$$

然后定义折叠向量多项式：

$$a_m^{proj}=A_m^{proj}(y_{proj})=\sum_{i=0}^{N-1} H_{i,m} y_{proj}^{i d}$$

  （将 $H$ 的每一列在 $y_{proj}$ 处压缩为标量）

$$b_m^{proj}=B_m^{proj}(y_{proj})=\sum_{j=0}^{d-1} W_{m,j} y_{proj}^{j}$$

  （将 $W$ 的每一行在 $y_{proj}$ 处压缩为标量）

定义折叠值：

$$\mu_{proj}=\sum_{m=0}^{d_{in}-1} a_m^{proj} b_m^{proj}$$

按照 CRPC 编码，必须满足：

$$P_{H'}(y_{proj})=\mu_{proj}$$

#### 2.2.4 共享维累加器

定义离散状态：

$$Acc_{proj}[0]=0$$

对每个 $m\in\{0,1,\ldots,d_{in}-1\}$，定义状态转移：

$$Acc_{proj}[m+1]=Acc_{proj}[m]+a_m^{proj} b_m^{proj}$$

对每个 $m\in\{d_{in}+1,d_{in}+2,\ldots,n_{in}-1\}$，定义：

$$Acc_{proj}[m]=Acc_{proj}[d_{in}]$$

同时对所有 $m\in\{d_{in},d_{in}+1,\ldots,n_{in}-1\}$，定义填充值：

$$a_m^{proj}=0, \quad b_m^{proj}=0$$

定义有效区选择器：

$$Q_{proj}^{valid}[m]= \begin{cases} 1, & 0\leq m \leq d_{in}-1\\ 0, & d_{in}\leq m \leq n_{in}-1 \end{cases}$$

对应插值多项式分别为：

$$P_{a^{proj}}(X)=\sum_{m=0}^{n_{in}-1} a_m^{proj} L_m^{(in)}(X)$$

$$P_{b^{proj}}(X)=\sum_{m=0}^{n_{in}-1} b_m^{proj} L_m^{(in)}(X)$$

$$P_{Acc^{proj}}(X)=\sum_{m=0}^{n_{in}-1} Acc_{proj}[m] L_m^{(in)}(X)$$

$$P_{Q_{proj}^{valid}}(X)=\sum_{m=0}^{n_{in}-1} Q_{proj}^{valid}[m] L_m^{(in)}(X)$$

#### 2.2.5 承诺

证明者对下列动态多项式提交承诺：

$$[P_{H'}], [P_{a^{proj}}], [P_{b^{proj}}], [P_{Acc^{proj}}]$$

#### 2.2.6 输出

本步骤输出：

- 投影矩阵 $H'$；
- 其系数多项式 $P_{H'}$；
- zkMaP见证 $P_{a^{proj}},P_{b^{proj}},P_{Acc^{proj}}$；
- 外点评值 $\mu_{proj}$。

> zkMaP 的“指纹抽查”逻辑：矩阵乘法 $H \cdot W$ 的计算量是 $O(N \cdot d_{in} \cdot d)$。 直接证明极其耗时。zkMaP 利用 CRPC 将矩阵变成多项式。 我们只需要在随机选的点 $y_{proj}$ 验证多项式的值是否匹配。

### 2.3 注意力向量压缩

**输入**：

- 第一层投影结果 $H'$
- 固定向量 $a_{src}$、$a_{dst}$
- 压缩挑战 $\xi$

**顺序说明**：

压缩挑战 $\xi$ 必须在所有依赖 $\xi$ 的见证生成之前确定。所以根据 3.1 节规定的顺序，生成压缩挑战：

$$\xi = H_{FS}(\text{transcript},[P_{H'}])$$

随后才能计算 $H^{\star}$、$H_{src}^{\star,edge}$、$H_{agg}^{\star}$ 等依赖 $\xi$ 的对象。

#### 2.3.1 计算

对任意节点 $i\in\{0,1,\ldots,N-1\}$，定义：

源注意力：

$$E_{src,i}=\sum_{j=0}^{d-1} H'_{i,j} a_{src,j}$$

目标注意力：

$$E_{dst,i}=\sum_{j=0}^{d-1} H'_{i,j} a_{dst,j}$$

压缩特征：

$$H_i^{\star}=\sum_{j=0}^{d-1} H'_{i,j} \xi^j$$

> 此处使用几何级数向量$c_{\xi} = (1, \xi, \xi^2, ...)$进行线性组合，将$d$维特征压缩为单一标量$H_i^{\star}$

#### 2.3.2 多项式编码

按照第 0.8.2 节的 padding 规则，把节点向量 $E_{src},E_{dst},H^{\star}$ 先补齐到长度 $n_N$，再在节点域 $\mathbb H_N$ 上做 Lagrange 插值：

$$P_{E_{src}}(X)=\sum_{i=0}^{n_N-1} E_{src}[i] L_i^{(N)}(X)$$

$$P_{E_{dst}}(X)=\sum_{i=0}^{n_N-1} E_{dst}[i] L_i^{(N)}(X)$$

$$P_{H^{\star}}(X)=\sum_{i=0}^{n_N-1} H^{\star}[i] L_i^{(N)}(X)$$

这里 $E_{src}[i]=E_{dst}[i]=H^{\star}[i]=0$ 对所有 $i\ge N$。这样定义后，后续 route 绑定约束、节点域开放与外点评值都引用同一组节点域多项式，不再混用“系数多项式”和“节点域插值多项式”。

#### 2.3.3 zkMaP向量

分别对 $H' a_{src}=E_{src}$、$H' a_{dst}=E_{dst}$、$H' c_{\xi}=H^{\star}$ 做三组完全独立的 zkMaP。

> 张量绑定说明：根据第 0.7.3 节，以下三组 zkMaP 的折叠向量都必须附带显式绑定子证明，证明其确实来自输入矩阵承诺 $[P_{H'}]$ 以及相应的静态权重承诺。

**源注意力组**

根据 3.1 节规定的顺序，生成挑战：

$$y_{src}=H_{FS}(\text{transcript},[P_{H'}],[P_{E_{src}}],[V_{a_{src}}])$$

定义折叠向量：

$$a_j^{src}=\sum_{i=0}^{N-1} H'_{i,j} L_i^{(N)}(y_{src})$$

定义权重向量：

$$b_j^{src}=a_{src,j}$$

定义外点评估值：

$$\mu_{src}=\sum_{j=0}^{d-1} a_j^{src} b_j^{src}$$

应满足：

$$P_{E_{src}}(y_{src})=\mu_{src}$$

**目标注意力组**

根据 3.1 节规定的顺序，生成挑战：

$$y_{dst}=H_{FS}(\text{transcript},[P_{H'}],[P_{E_{dst}}],[V_{a_{dst}}])$$

定义折叠向量：

$$a_j^{dst}=\sum_{i=0}^{N-1} H'_{i,j} L_i^{(N)}(y_{dst}),\quad b_j^{dst}=a_{dst,j},\quad \mu_{dst}=\sum_{j=0}^{d-1} a_j^{dst} b_j^{dst}$$

应满足：

$$P_{E_{dst}}(y_{dst})=\mu_{dst}$$

**源压缩特征组**

根据 3.1 节规定的顺序，生成挑战：

$$y_{\star}=H_{FS}(\text{transcript},[P_{H'}],[P_{H^{\star}}])$$

定义折叠向量：

$$a_j^{\star}=\sum_{i=0}^{N-1} H'_{i,j} L_i^{(N)}(y_{\star}),\quad b_j^{\star}=\xi^j,\quad \mu_{\star}=\sum_{j=0}^{d-1} a_j^{\star} b_j^{\star}$$

应满足：

$$P_{H^{\star}}(y_{\star})=\mu_{\star}$$

#### 2.3.4 共享维累加器

对任一符号 $\bullet\in\{src,dst,\star\}$，统一定义点积的中间前缀和：

$$Acc_{\bullet}[0]=0$$

对每个 $j\in\{0,1,\ldots,d-1\}$，定义前缀和的递推公式：

$$Acc_{\bullet}[j+1]=Acc_{\bullet}[j]+a_j^{\bullet} b_j^{\bullet}$$

对每个 $j\in\{d,d+1,d+2,\ldots,n_d-1\}$，定义填充规则和填充值：

$$Acc_{\bullet}[j]=Acc_{\bullet}[d] \quad a_j^{\bullet}=0, \quad b_j^{\bullet}=0$$

定义共享有效区选择器：

$$Q_d^{valid}[j]= \begin{cases} 1, & 0\leq j \leq d-1\\ 0, & d\leq j \leq n_d-1 \end{cases}$$

对应插值多项式分别为：

$$P_{a^{\bullet}}(X)=\sum_{j=0}^{n_d-1} a_j^{\bullet} L_j^{(d)}(X)$$

$$P_{b^{\bullet}}(X)=\sum_{j=0}^{n_d-1} b_j^{\bullet} L_j^{(d)}(X)$$

$$P_{Acc^{\bullet}}(X)=\sum_{j=0}^{n_d-1} Acc_{\bullet}[j] L_j^{(d)}(X)$$

#### 2.3.5 承诺

证明者对下列动态多项式提交承诺：

$$[P_{E_{src}}], [P_{E_{dst}}], [P_{H^{\star}}]$$

$$[P_{a^{src}}], [P_{b^{src}}], [P_{Acc^{src}}]$$

$$[P_{a^{dst}}], [P_{b^{dst}}], [P_{Acc^{dst}}]$$

$$[P_{a^{\star}}], [P_{b^{\star}}], [P_{Acc^{\star}}]$$

#### 2.3.6 输出

- 节点域源注意力 $E_{src}$
- 节点域目标注意力 $E_{dst}$
- 节点域压缩特征 $H^{\star}$
- 三组 zkMaP 见证
- 三个外点评值 $\mu_{src},\mu_{dst},\mu_{\star}$

### 2.4 源节点属性路由

**输入**：

- 节点域值 $E_{src}$
- 节点域值 $H^{\star}$
- 边源索引 $src(k)$

#### 2.4.1 计算

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，定义：（$src(k)$ 是公开的拓扑索引，表示边 $k$ 的源节点编号）

源注意力广播：

$$E_{src,k}^{edge}=E_{src,src(k)}$$

压缩特征广播：$H_{src,k}^{\star,edge}=H_{src(k)}^{\star}$

#### 2.4.2 挑战生成、元组与重数

根据 3.1 节规定的顺序，生成挑战 $\eta_{src},\beta_{src}$。

随后定义节点表端表项：对每个 $i\in\{0,1,\ldots,N-1\}$，设：

$$Table^{src}[i]=i+\eta_{src} E_{src,i}+\eta_{src}^2 H_i^{\star}$$

对每个节点 $i\in\{0,1,\ldots,N-1\}$，定义重数：

$$m_{src}[i]=\#\{k\in\{0,1,\ldots,E-1\}\mid src(k)=i\}$$

定义边查询端表项：对每个 $k\in\{0,1,\ldots,E-1\}$，设：$Query^{src}[k]=src(k)+\eta_{src} E_{src,k}^{edge}+\eta_{src}^2 H_{src,k}^{\star,edge}$

解释：节点 $i$ 在 $src$ 序列中出现的次数，表示该节点的属性被多少条边所引用

#### 2.4.3 双累加器与全局路由总和

证明者计算全局路由总和（公开透明放入 Transcript）：

$$S_{src} = \sum_{i=0}^{N-1} \frac{m_{src}[i]}{Table^{src}[i]+\beta_{src}} = \sum_{k=0}^{E-1} \frac{1}{Query^{src}[k]+\beta_{src}}$$

**节点域累加器** $R_{src}^{node}$：初始 $R_{src}^{node}[0]=0$。对 $t\in\{0,\dots,n_N-2\}$：

$$R_{src}^{node}[t+1] = R_{src}^{node}[t] + Q_N[t] \cdot \frac{m_{src}[t]}{Table^{src}[t]+\beta_{src}}$$

**边域累加器** $R_{src}^{edge}$：初始 $R_{src}^{edge}[0]=0$。对 $t\in\{0,\dots,n_{edge}-2\}$：

$$R_{src}^{edge}[t+1] = R_{src}^{edge}[t] + Q_{qry}^{src}[t] \cdot \frac{1}{Query^{src}[t]+\beta_{src}}$$

(注：$Q_{qry}^{src}$ 有效区为 $0 \le t \le E-1$)

#### 2.4.4 多项式编码与承诺

边域广播对象编码为：

$$P_{E_{src}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1} E_{src}^{edge}[k] L_k^{(edge)}(X)$$

$$P_{H_{src}^{\star,edge}}(X)=\sum_{k=0}^{n_{edge}-1} H_{src}^{\star,edge}[k] L_k^{(edge)}(X)$$

节点域对象编码为：

$$P_{Table^{src}}(X)=\sum_{t=0}^{n_N-1} Table^{src}[t] L_t^{(N)}(X)$$

$$P_{m_{src}}(X)=\sum_{t=0}^{n_N-1} m_{src}[t] L_t^{(N)}(X)$$

$$P_{R_{src}^{node}}(X)=\sum_{t=0}^{n_N-1} R_{src}^{node}[t] L_t^{(N)}(X)$$

边域对象编码为：

$$P_{Query^{src}}(X)=\sum_{t=0}^{n_{edge}-1} Query^{src}[t] L_t^{(edge)}(X)$$

$$P_{R_{src}^{edge}}(X)=\sum_{t=0}^{n_{edge}-1} R_{src}^{edge}[t] L_t^{(edge)}(X)$$

证明者提交承诺：

$$[P_{E_{src}^{edge}}], [P_{H_{src}^{\star,edge}}], [P_{Table^{src}}], [P_{m_{src}}], [P_{Query^{src}}], [P_{R_{src}^{node}}], [P_{R_{src}^{edge}}]$$

，并将标量 $S_{src}$ 加入最终证明对象中的公开评值列表。

#### 2.4.5 输出

- 边域广播值 $E_{src}^{edge}$；
- 边域广播值 $H_{src}^{\star,edge}$；
- 源 route 的 LogUp witness。

> 路由（Route）就是“搬运工”。节点上有特征，边上有源 ID。我们要把节点特征搬到每一条出发的边上。LogUp 证明了：边 $k$ 从其源点 $src(k)$ 拿到的数据确实没被调包。

### 2.5 目标节点属性路由（双累加器架构）

**输入**：

- 节点域目标注意力 $E_{dst}$；
- 节点域组最大值 $M$；
- 节点域 Softmax 分母 $Sum$；
- 节点域逆元 $inv$；
- 节点域压缩聚合特征 $H_{agg}^{\star}$；
- 边目标索引 $dst(k)$。

#### 2.5.1 计算

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，定义：

目标注意力广播：

$$E_{dst,k}^{edge}=E_{dst,dst(k)}$$

组最大值广播：

$$M_k^{edge}=M_{dst(k)}$$

分母广播：

$$Sum_k^{edge}=Sum_{dst(k)}$$

逆元广播：

$$inv_k^{edge}=inv_{dst(k)}$$

压缩聚合特征广播：

$$H_{agg,k}^{\star,edge}=H_{agg,dst(k)}^{\star}$$

这些广播多项式无条件立即提交承诺。

#### 2.5.2 延迟绑定与元组语义（安全性）

由于 $H_{agg}^{\star}$ 在第 2.8 步才最终生成，因此 dst-route 子协议必须在 $[P_{E_{dst}}],[P_M],[P_{Sum}],[P_{inv}],[P_{H_{agg}^{\star}}]$ **以及边域广播承诺** $[P_{E_{dst}^{edge}}], [P_{M^{edge}}], [P_{Sum^{edge}}], [P_{inv^{edge}}], [P_{H_{agg}^{\star,edge}}]$ 全部固定之后（详见第 3.1 节第 12 步），统一采样 $\eta_{dst}, \beta_{dst}$ 并 finalize。这补全了防止伪造查询依赖的 Fiat-Shamir 漏洞。

当 3.1 节第 12 步的挑战已经生成后，dst-route 的 tuple 语义定义为：

对每个节点 $i\in\{0,1,\ldots,N-1\}$，定义表端表项：

$$Table^{dst}[i]=i+\eta_{dst}E_{dst,i}+\eta_{dst}^2 M_i+\eta_{dst}^3 Sum_i+\eta_{dst}^4 inv_i+\eta_{dst}^5 H_{agg,i}^{\star}$$

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，定义查询端表项：

$$Query^{dst}[k]=dst(k)+\eta_{dst}E_{dst,k}^{edge}+\eta_{dst}^2 M_k^{edge}+\eta_{dst}^3 Sum_k^{edge}+\eta_{dst}^4 inv_k^{edge}+\eta_{dst}^5 H_{agg,k}^{\star,edge}$$

对每个节点 $i\in\{0,1,\ldots,N-1\}$，定义重数：

$$m_{dst}[i]=\#\{k\in\{0,1,\ldots,E-1\}\mid dst(k)=i\}$$

#### 2.5.3 双累加器与全局路由总和

计算公开的全局目标路由总和：

$$S_{dst} = \sum_{i=0}^{N-1} \frac{m_{dst}[i]}{Table^{dst}[i]+\beta_{dst}} = \sum_{k=0}^{E-1} \frac{1}{Query^{dst}[k]+\beta_{dst}}$$

**节点域累加器** $R_{dst}^{node}$：初始 $R_{dst}^{node}[0]=0$。对 $t\in\{0,\dots,n_N-2\}$：

$$R_{dst}^{node}[t+1] = R_{dst}^{node}[t] + Q_N[t] \cdot \frac{m_{dst}[t]}{Table^{dst}[t]+\beta_{dst}}$$

**边域累加器** $R_{dst}^{edge}$：初始 $R_{dst}^{edge}[0]=0$。对 $t\in\{0,\dots,n_{edge}-2\}$：

$$R_{dst}^{edge}[t+1] = R_{dst}^{edge}[t] + Q_{qry}^{dst}[t] \cdot \frac{1}{Query^{dst}[t]+\beta_{dst}}$$

#### 2.5.4 广播多项式编码与即时承诺

$$P_{E_{dst}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1} E_{dst,k}^{edge} L_k^{(edge)}(X)$$

$$P_{M^{edge}}(X)=\sum_{k=0}^{n_{edge}-1} M_k^{edge} L_k^{(edge)}(X)$$

$$P_{Sum^{edge}}(X)=\sum_{k=0}^{n_{edge}-1} Sum_k^{edge} L_k^{(edge)}(X)$$

$$P_{inv^{edge}}(X)=\sum_{k=0}^{n_{edge}-1} inv_k^{edge} L_k^{(edge)}(X)$$

$$P_{H_{agg}^{\star,edge}}(X)=\sum_{k=0}^{n_{edge}-1} H_{agg,k}^{\star,edge} L_k^{(edge)}(X)$$

这些基础广播多项式不依赖 $\eta_{dst}, \beta_{dst}$，因此证明者在本步骤立即提交承诺：

$$[P_{E_{dst}^{edge}}], [P_{M^{edge}}], [P_{Sum^{edge}}], [P_{inv^{edge}}], [P_{H_{agg}^{\star,edge}}]$$

#### 2.5.5 派生多项式编码与延迟承诺

当 3.1 节第 12 步已经生成 $\eta_{dst}, \beta_{dst}$ 后，定义：

节点域对象：

$$P_{Table^{dst}}(X)=\sum_{t=0}^{n_N-1} Table^{dst}[t] L_t^{(N)}(X)$$

$$P_{m_{dst}}(X)=\sum_{t=0}^{n_N-1} m_{dst}[t] L_t^{(N)}(X)$$

$$P_{R_{dst}^{node}}(X)=\sum_{t=0}^{n_N-1} R_{dst}^{node}[t] L_t^{(N)}(X)$$

边域对象：

$$P_{Query^{dst}}(X)=\sum_{t=0}^{n_{edge}-1} Query^{dst}[t] L_t^{(edge)}(X)$$

$$P_{R_{dst}^{edge}}(X)=\sum_{t=0}^{n_{edge}-1} R_{dst}^{edge}[t] L_t^{(edge)}(X)$$

随后证明者再提交承诺：

$$[P_{Table^{dst}}], [P_{m_{dst}}], [P_{Query^{dst}}], [P_{R_{dst}^{node}}], [P_{R_{dst}^{edge}}]$$

，并将 $S_{dst}$ 加入最终证明对象中的公开评值列表。

#### 2.5.6 输出

- 边域广播值 $E_{dst}^{edge},M^{edge},Sum^{edge},inv^{edge},H_{agg}^{\star,edge}$；
- 构造的目标路由的 LogUp见证

### 2.6 注意力分数

**输入**：

- 边域源注意力 $E_{src}^{edge}$；
- 边域目标注意力 $E_{dst}^{edge}$；
- LeakyReLU 表 $T_{LReLU}$；
- 范围表 $T_{range}$；
- 组选择器 $Q_{new}^{edge},Q_{end}^{edge}$。

#### 2.6.1 计算

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，计算分数：

$$S_k=E_{src,k}^{edge}+E_{dst,k}^{edge} \quad Z_k=LReLU(S_k)$$

#### 2.6.2 挑战生成与激活映射查找

根据 3.1 节规定的顺序，生成挑战 $\eta_L,\beta_L$。

对每个表索引 $t\in\{0,1,\ldots,|T_{LReLU}|-1\}$，定义表端表项：

$$Table^{L}[t]=T_{LReLU}[t,0]+\eta_L T_{LReLU}[t,1]$$

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，定义查询端表项：

$$Query^{L}[k]=S_k+\eta_L Z_k$$

对每个表索引 $t\in\{0,1,\ldots,|T_{LReLU}|-1\}$，定义重数：（记录

$$(S_k,Z_k)$$

对在静态表中的命中次数）

$$m_L[t]=\#\{k\in\{0,1,\ldots,E-1\}\mid (S_k,Z_k)=(T_{LReLU}[t,0],T_{LReLU}[t,1])\}$$

定义有效区选择器：

$$Q_{tbl}^{L}[t]= \begin{cases} 1, & 0\leq t \leq |T_{LReLU}|-1\\ 0, & |T_{LReLU}|\leq t \leq n_{edge}-1 \end{cases} \qquad Q_{qry}^{L}[t]= \begin{cases} 1, & 0\leq t \leq E-1\\ 0, & E\leq t \leq n_{edge}-1 \end{cases}$$

定义累加器初始值：

$$R_L[0]=0$$

对每个 $t\in\{0,1,\ldots,n_{edge}-2\}$，定义递推公式：

$$R_L[t+1]=R_L[t]+Q_{tbl}^{L}[t]\cdot \frac{m_L[t]}{Table^{L}[t]+\beta_L}-Q_{qry}^{L}[t]\cdot \frac{1}{Query^{L}[t]+\beta_L}$$

定义多项式：

$$P_S(X)=\sum_{k=0}^{E-1} S_k L_k^{(edge)}(X)$$

$$P_Z(X)=\sum_{k=0}^{E-1} Z_k L_k^{(edge)}(X)$$

$$P_{Table^{L}}(X)=\sum_{t=0}^{n_{edge}-1} Table^{L}[t] L_t^{(edge)}(X)$$

$$P_{Query^{L}}(X)=\sum_{t=0}^{n_{edge}-1} Query^{L}[t] L_t^{(edge)}(X)$$

$$P_{m_L}(X)=\sum_{t=0}^{n_{edge}-1} m_L[t] L_t^{(edge)}(X)$$

$$P_{R_L}(X)=\sum_{t=0}^{n_{edge}-1} R_L[t] L_t^{(edge)}(X)$$

提交承诺：

$$[P_S], [P_Z], [P_{Table^{L}}], [P_{Query^{L}}], [P_{m_L}], [P_{R_L}]$$

#### 2.6.3 组最大值提取与非负差分计算

对每个节点 $i\in\{0,1,\ldots,N-1\}$，计算节点域最大值：

$$M_i = \max\{Z_k \mid dst(k)=i\}$$

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，边域非负差分：

$$M_k^{edge}=M_{dst(k)} \quad \Delta_k^+=M_k^{edge}-Z_k$$

位宽充足性条件（约束前提）：为了使差分 $\Delta_k^+$ 能够被范围表证明，必须选择位宽 $B$ 使得对所有合法输入都成立：

$$0\leq \Delta_k^+ \leq 2^B-1$$

 。这个条件不是运行时约束，而是参数选择时的充分性条件。若实现中无法静态保证这一条件，则应增大 $B$ 或调整定点量化范围。

#### 2.6.4 最大值指示变量与零差分约束

引入指示变量：

$$s_{max}[k] \in \{0,1\}, \quad 0\leq k\leq E-1$$

 ，在同一目标节点组中，恰有一个边位置被选为“最大值见证位置”，该位置满足 $\Delta_k^+=0$。

#### 2.6.5 唯一性计数状态机

定义离散状态变量：

$$C_{max}[0]=s_{max}[0]$$

对每个 $k\in\{0,1,\ldots,E-2\}$，定义状态转移方程：

$$C_{max}[k+1]=Q_{new}^{edge}[k+1]\cdot s_{max}[k+1]+(1-Q_{new}^{edge}[k+1])\cdot (C_{max}[k]+s_{max}[k+1])$$

padding 区规则固定为：对所有 $k\in\{E,E+1,\ldots,n_{edge}-1\}$，令

$$s_{max}[k]=0,\qquad C_{max}[k]=C_{max}[E-1].$$

因此，在有效区之后状态机只允许保持常值，不允许继续增长。

组末检查条件是：对每个 $k\in\{0,1,\ldots,E-1\}$，都要求

$$Q_{end}^{edge}[k]\cdot (C_{max}[k]-1)=0$$

在第 $k$ 条边是组末尾时，这个约束强制该组中恰好累计到一个 $s_{max}=1$。

#### 2.6.6 并列最大值处理逻辑

最大值语义由三类条件共同保证：

1. 非负差分条件：

	$$\Delta_k^+=M_k^{edge}-Z_k$$

2. 范围检查条件：

	$$\Delta_k^+ \in \{0,1,2,\ldots,2^B-1\}$$

3. 唯一零差分条件：

	$$s_{max}[k]\cdot \Delta_k^+=0$$

	 ，并且每组恰有一个 $s_{max}=1$

由这三点可推出：在每个目标节点组中，$M_i$ 至少是该组所有 $Z_k$ 的上界；并且至少有一个位置使得 $M_i=Z_k$。再结合“每组恰有一个 $s_{max}=1$”可知：每组恰有一个被选中的零差分见证位置。这里证明的是“唯一被选中的最大值见证位置”，而不是“组内严格唯一最大值”；因此若同一组中存在多个边位置同时达到相同最大值，协议仍可接受，只要求其中恰有一个位置被 $s_{max}$ 选中。

#### 2.6.7 范围检查查找

根据 3.1 节规定的顺序，生成挑战 $\beta_R$。

定义表端：

$$Table^{R}[t]=t, \quad 0\leq t\leq 2^B-1$$

定义查询端：

$$Query^{R}[k]=\Delta_k^+, \quad 0\leq k\leq E-1$$

定义重数：

$$m_R[t]=\#\{k\in\{0,1,\ldots,E-1\}\mid \Delta_k^+=t\}$$

定义选择器：

$$Q_{tbl}^{R}[t]= \begin{cases} 1, & 0\leq t \leq 2^B-1\\ 0, & 2^B\leq t \leq n_{edge}-1 \end{cases} \qquad Q_{qry}^{R}[t]= \begin{cases} 1, & 0\leq t \leq E-1\\ 0, & E\leq t \leq n_{edge}-1 \end{cases}$$

定义累加器初始值：

$$R_R[0]=0$$

对每个 $t\in\{0,1,\ldots,n_{edge}-2\}$，定义递推方程：

$$R_R[t+1]=R_R[t]+Q_{tbl}^{R}[t]\cdot \frac{m_R[t]}{Table^{R}[t]+\beta_R}-Q_{qry}^{R}[t]\cdot \frac{1}{Query^{R}[t]+\beta_R}$$

#### 2.6.8 多项式编码与承诺

定义多项式：

$$P_M(X)=\sum_{i=0}^{N-1} M_i L_i^{(N)}(X)$$

$$P_{M^{edge}}(X)=\sum_{k=0}^{n_{edge}-1} M_k^{edge} L_k^{(edge)}(X)$$

$$P_{\Delta}(X)=\sum_{k=0}^{n_{edge}-1} \Delta_k^+ L_k^{(edge)}(X)$$

$$P_{s_{max}}(X)=\sum_{k=0}^{n_{edge}-1} s_{max}[k] L_k^{(edge)}(X)$$

$$P_{C_{max}}(X)=\sum_{k=0}^{n_{edge}-1} C_{max}[k] L_k^{(edge)}(X)$$

$$P_{Table^{R}}(X)=\sum_{t=0}^{n_{edge}-1} Table^{R}[t] L_t^{(edge)}(X)$$

$$P_{Query^{R}}(X)=\sum_{t=0}^{n_{edge}-1} Query^{R}[t] L_t^{(edge)}(X)$$

$$P_{m_R}(X)=\sum_{t=0}^{n_{edge}-1} m_R[t] L_t^{(edge)}(X)$$

$$P_{R_R}(X)=\sum_{t=0}^{n_{edge}-1} R_R[t] L_t^{(edge)}(X)$$

提交承诺：

$$[P_M], [P_{M^{edge}}], [P_{\Delta}], [P_{s_{max}}], [P_{C_{max}}], [P_{Table^{R}}], [P_{Query^{R}}], [P_{m_R}], [P_{R_R}]$$

#### 2.6.9 输出

- 边域打分 $S$、激活后打分 $Z$；
- 节点域最大值 $M$ 与边域广播 $M^{edge}$；
- 非负差分 $\Delta^+$；
- 最大值 witness $s_{max},C_{max}$；
- LeakyReLU 与 range 两套 LogUp witness。

### 2.7 掩码归一化

**输入**：

- 边域差分 $\Delta^+$；
- 指数表 $T_{exp}$；
- 边域广播 $Sum^{edge},inv^{edge}$ 将在本步骤中先作为 witness 使用，之后由 dst-route 保证其来源正确。

#### 2.7.1 指数映射查找

根据 3.1 节规定的顺序，生成挑战 $\eta_{exp},\beta_{exp}$。

随后对每个边索引 $k\in\{0,1,\ldots,E-1\}$，计算指数映射结果：

$$U_k=ExpMap(\Delta_k^+)$$

> 注意映射语义：由于输入是非负差分 $\Delta_k^+ = M_k - Z_k$，为了与标准 GAT 的 Softmax 防溢出逻辑（计算 $e^{Z_k - M_k}$）保持数学一致性，静态表 $T_{exp}$ 的底层映射逻辑必须是计算负指数，即 $T_{exp}[t, 1] = \text{Quantize}(e^{-T_{exp}[t, 0]})$。

定义表端表项：对每个 $t\in\{0,1,\ldots,|T_{exp}|-1\}$，构造二元组：

$$Table^{exp}[t]=T_{exp}[t,0]+\eta_{exp} T_{exp}[t,1]$$

定义查询端表项：对每个 $k\in\{0,1,\ldots,E-1\}$，构造查询：

$$Query^{exp}[k]=\Delta_k^++\eta_{exp} U_k$$

定义重数：对每个 $t\in\{0,1,\ldots,|T_{exp}|-1\}$，记录每个表项在查询中出现的次数：

$$m_{exp}[t]=\#\{k\in\{0,1,\ldots,E-1\}\mid (\Delta_k^+,U_k)=(T_{exp}[t,0],T_{exp}[t,1])\}$$

定义有效区选择器：

$$Q_{tbl}^{exp}[t]= \begin{cases} 1, & 0\leq t \leq |T_{exp}|-1\\ 0, & |T_{exp}|\leq t \leq n_{edge}-1 \end{cases} \qquad Q_{qry}^{exp}[t]= \begin{cases} 1, & 0\leq t \leq E-1\\ 0, & E\leq t \leq n_{edge}-1 \end{cases}$$

定义累加器初始值：

$$R_{exp}[0]=0$$

对每个 $t\in\{0,1,\ldots,n_{edge}-2\}$，定义递推方程：（证明 $\Delta_k^+$ 到 $U_k$ 的映射关系）

$$R_{exp}[t+1]=R_{exp}[t]+Q_{tbl}^{exp}[t]\cdot \frac{m_{exp}[t]}{Table^{exp}[t]+\beta_{exp}}-Q_{qry}^{exp}[t]\cdot \frac{1}{Query^{exp}[t]+\beta_{exp}}$$

#### 2.7.2 目标节点组分母聚合

对每个节点 $i\in\{0,1,\ldots,N-1\}$，计算Softmax 分母之和：

$$Sum_i=\sum_{\{k\mid dst(k)=i\}} U_k$$

根据第 0.8.4 节的前提，每个组至少包含一条边，因此上式在每个 $i$ 上都有定义。这不是验证者直接重新求和得到，而是作为见证给出，随后通过 PSQ 与路由约束统一证明。

#### 2.7.3 分母逆元计算与节点广播

对每个节点 $i\in\{0,1,\ldots,N-1\}$，进行逆元计算：

$$inv_i = Sum_i^{-1}$$

> 工程实现提示：根据 0.8.2 节的填充规则，当 $i \ge N$（即 Padding 区域）时，由于缺乏真实边聚合，$Sum_i$ 的默认填充值为 0。证明者在生成见证时，对无效节点的 $inv_i$ 应直接赋 0（或任意域元素），切勿执行实际的除法操作导致运行时 `DivisionByZero`。约束系统中的选择器 $Q_N$ 会自动放行这些无效位置。

根据第 0.8.4 节的前提，所有 $Sum_i$ 在 $\mathbb F_p$ 中均非零（量化方案已保证 $Sum_i \ge 1$），因此该逆元定义合法。

对每个边索引 $k\in{0,1,\ldots,E-1}$，定义属性广播，将节点域的属性赋值回边域，以便进行边级计算：

$$Sum_k^{edge} = Sum_{dst(k)} \quad inv_k^{edge}=inv_{dst(k)}$$

#### 2.7.4 边域归一化注意力权重

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，计算最终的归一化权重：

$$\alpha_k = U_k \cdot inv_k^{edge}$$

#### 2.7.5 多项式编码与承诺

$$P_U(X)=\sum_{k=0}^{n_{edge}-1} U_k L_k^{(edge)}(X)$$

$$P_{Sum}(X)=\sum_{i=0}^{N-1} Sum_i L_i^{(N)}(X)$$

$$P_{inv}(X)=\sum_{i=0}^{N-1} inv_i L_i^{(N)}(X)$$

$$P_{\alpha}(X)=\sum_{k=0}^{n_{edge}-1} \alpha_k L_k^{(edge)}(X)$$

$$P_{Table^{exp}}(X)=\sum_{t=0}^{n_{edge}-1} Table^{exp}[t] L_t^{(edge)}(X)$$

$$P_{Query^{exp}}(X)=\sum_{t=0}^{n_{edge}-1} Query^{exp}[t] L_t^{(edge)}(X)$$

$$P_{m_{exp}}(X)=\sum_{t=0}^{n_{edge}-1} m_{exp}[t] L_t^{(edge)}(X)$$

$$P_{R_{exp}}(X)=\sum_{t=0}^{n_{edge}-1} R_{exp}[t] L_t^{(edge)}(X)$$

提交承诺：

$$[P_U], [P_{Sum}], [P_{inv}], [P_{\alpha}], [P_{Table^{exp}}], [P_{Query^{exp}}], [P_{m_{exp}}], [P_{R_{exp}}]$$

#### 2.7.6 输出

本步骤输出：

- 指数映射结果 $U$；
- 节点域分母 $Sum$；
- 节点域逆元 $inv$；
- 归一化注意力 $\alpha$；
- exp lookup witness。

### 2.8 特征聚合

**输入**：

- 归一化注意力 $\alpha$；
- 第一层投影矩阵 $H'$；
- 边域广播值 $H_{src}^{\star,edge}$；
- 节点域分母 $Sum$；
- 压缩挑战 $\xi$；
- 目标路由广播值 $H_{agg}^{\star,edge}$。

#### 2.8.1 高维聚合矩阵

对任意节点 $i\in\{0,1,\ldots,N-1\}$ 和特征维 $j\in\{0,1,\ldots,d-1\}$，聚合计算：

$$H_{agg,i,j}=\sum_{\{k\mid dst(k)=i\}} \alpha_k H'_{src(k),j}$$

这是 GAT 聚合的真实高维矩阵。

#### 2.8.2 随机维压缩聚合特征

对任意节点 $i\in\{0,1,\ldots,N-1\}$，计算公式：

$$H_{agg,i}^{\star}=\sum_{j=0}^{d-1} H_{agg,i,j} \xi^j$$

聚合特征一致性绑定要求：为了防止证明者独立伪造 $H_{agg}$ 与 $H_{agg}^{\star}$，必须正式加入绑定关系：

$$H_{agg} c_{\xi} = H_{agg}^{\star}$$

这组绑定不能只做口头 soundness 说明，而必须进入正式证明系统。

#### 2.8.3 聚合特征一致性绑定

定义系数多项式：

$$P_{H_{agg}}(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{d-1} H_{agg,i,j} X^{i d + j}$$

定义节点域插值多项式：

$$P_{H_{agg}^{\star}}(X)=\sum_{i=0}^{n_N-1} H_{agg}^{\star}[i] L_i^{(N)}(X)$$

其中对所有 $i\ge N$，规定 $H_{agg}^{\star}[i]=0$。

> 张量绑定说明：根据第 0.7.3 节，此处的折叠向量 $a_j^{agg}$ 与 $b_j^{agg}$ 同样必须附带显式绑定子证明，证明它们确实来自输入矩阵承诺 $[P_{H_{agg}}]$。

根据 3.1 节规定的顺序，生成挑战：

$$y_{agg}=H_{FS}(\text{transcript},[P_{H_{agg}}],[P_{H_{agg}^{\star}}]).$$

定义向量折叠与外点评估。

折叠 $H_{agg}$ 的列：

$$a_j^{agg}=\sum_{i=0}^{N-1} H_{agg,i,j} L_i^{(N)}(y_{agg}),\qquad 0\le j\le d-1.$$

定义权重向量：

$$b_j^{agg}=\xi^j,\qquad 0\le j\le d-1.$$

计算折叠总和：

$$\mu_{agg}=\sum_{j=0}^{d-1} a_j^{agg} b_j^{agg}. $$

应满足一致性要求：

$$P_{H_{agg}^{\star}}(y_{agg})=\mu_{agg}. $$

定义共享维累加器：

$$Acc_{agg}[0]=0.$$

对每个 $j\in\{0,1,\ldots,d-1\}$，递推方程：

$$Acc_{agg}[j+1]=Acc_{agg}[j]+a_j^{agg}b_j^{agg}. $$

对所有 padding 位置 $j\in\{d,d+1,\ldots,n_d-1\}$，固定采用与其余 zkMaP 子协议完全一致的规则：

$$a_j^{agg}=0,\qquad b_j^{agg}=0,\qquad Acc_{agg}[j]=Acc_{agg}[d].$$

因此终点条件写为：

$$Acc_{agg}[d]=\mu_{agg}. $$

对应插值多项式定义为：

$$P_{a^{agg}}(X)=\sum_{j=0}^{n_d-1} a_j^{agg}L_j^{(d)}(X),$$

$$P_{b^{agg}}(X)=\sum_{j=0}^{n_d-1} b_j^{agg}L_j^{(d)}(X),$$

$$P_{Acc^{agg}}(X)=\sum_{j=0}^{n_d-1} Acc_{agg}[j]L_j^{(d)}(X).$$

证明者必须一并提交承诺：

$$[P_{H_{agg}}],\ [P_{H_{agg}^{\star}}],\ [P_{a^{agg}}],\ [P_{b^{agg}}],\ [P_{Acc^{agg}}].$$

其中 $[P_{H_{agg}}]$ 与 $[P_{H_{agg}^{\star}}]$ 是本步骤的主对象承诺，$[P_{a^{agg}}],[P_{b^{agg}}],[P_{Acc^{agg}}]$ 是后续 d 域 zkMaP 约束与 batch opening 所必需的辅助承诺。

#### 2.8.4 边级加权压缩特征

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，在边域计算每一条边对聚合的贡献（已压缩）：

$$\widehat v_k^{\star}=\alpha_k H_{src,k}^{\star,edge}$$

定义多项式：

$$P_{\widehat v^{\star}}(X)=\sum_{k=0}^{n_{edge}-1} \widehat v_k^{\star} L_k^{(edge)}(X)$$

提交承诺：

$$[P_{\widehat v^{\star}}]$$

#### 2.8.5 压缩 PSQ 挑战

根据 3.1 节规定的顺序，生成压缩PSQ挑战：

$$\lambda_{psq}=H_{FS}(\text{transcript},[P_U],[P_{\alpha}],[P_{H_{src}^{\star,edge}}],[P_{Sum}],[P_{H_{agg}^{\star}}],[P_{H_{agg}^{\star,edge}}],[P_{\widehat v^{\star}}])$$

注意：这里不能把 $[P_{w_{psq}}]$、$[P_{T_{psq}^{edge}}]$、$[P_{PSQ}]$ 吸进去，因为它们本身依赖 $\lambda_{psq}$。

#### 2.8.6 混合项与广播值构造

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，边域混合项：

$$w_{psq}[k]=U_k+\lambda_{psq} \widehat v_k^{\star}$$

对每个节点 $i\in\{0,1,\ldots,N-1\}$，节点域混合目标：

$$T_{psq}[i]=Sum_i+\lambda_{psq} H_{agg,i}^{\star}$$

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，计算其广播值：

$$T_{psq}^{edge}[k]=Sum_k^{edge}+\lambda_{psq} H_{agg,k}^{\star,edge}$$

#### 2.8.7 压缩 PSQ 状态机

定义离散状态：

$$PSQ[0]=w_{psq}[0]$$

对每个 $k\in\{1,2,\ldots,E-1\}$，递推方程：

$$PSQ[k]=Q_{new}^{edge}[k] w_{psq}[k] + (1-Q_{new}^{edge}[k]) (PSQ[k-1]+w_{psq}[k])$$

padding 区规则固定为：对所有 $k\in\{E,E+1,\ldots,n_{edge}-1\}$，令

$$w_{psq}[k]=0,\qquad T_{psq}^{edge}[k]=0,\qquad PSQ[k]=PSQ[E-1].$$

这个递推的含义是：

- 如果 $k$ 是新组起点，则以本组第一个边权 $w_{psq}[k]$ 作为本组前缀和起点；
- 如果 $k$ 在组内延续，则把当前 $w_{psq}[k]$ 加到前一个前缀和上；
- 如果 $k$ 处于 padding 区，则状态保持常值。

#### 2.8.8 压缩 PSQ 输出约束

对每个边索引 $k\in\{0,1,\ldots,E-1\}$，要求当且仅当 $k$ 是组末时，满足：

$$Q_{end}^{edge}[k] \cdot (PSQ[k]-T_{psq}^{edge}[k]) = 0$$

这个条件保证每个目标节点组的边级混合前缀和等于节点级混合目标值。

因此，压缩 PSQ 同时证明了两件事：

1. 分母关系：

	$$Sum_i = \sum_{\{k\mid dst(k)=i\}} U_k$$

2. 压缩聚合关系：

	$$H_{agg,i}^{\star} = \sum_{\{k\mid dst(k)=i\}} \widehat v_k^{\star}$$

#### 2.8.9 多项式编码与承诺

$$P_{w_{psq}}(X)=\sum_{k=0}^{n_{edge}-1} w_{psq}[k] L_k^{(edge)}(X)$$

$$P_{T_{psq}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1} T_{psq}^{edge}[k] L_k^{(edge)}(X)$$

$$P_{PSQ}(X)=\sum_{k=0}^{n_{edge}-1} PSQ[k] L_k^{(edge)}(X)$$

提交承诺：

$$[P_{w_{psq}}], [P_{T_{psq}^{edge}}], [P_{PSQ}]$$

#### 2.8.10 输出

本步骤输出：

- 高维聚合矩阵 $H_{agg}$；
- 压缩聚合特征 $H_{agg}^{\star}$；
- $H_{agg} c_{\xi}=H_{agg}^{\star}$ 的 zkMaP witness；
- 压缩边级加权特征 $\widehat v^{\star}$；
- 压缩 PSQ witness。

### 2.9 输出层

**输入**：

- 聚合矩阵 $H_{agg}$；
- 输出层权重 $W_{out}$；
- 输出层偏置 $b$。

#### 2.9.1 计算

线性部分计算，将聚合特征矩阵与输出层权重矩阵相乘：

$$Y_{i,j}^{lin}=\sum_{m=0}^{d-1} H_{agg,i,m} W_{out,m,j}$$

最终预测计算，在线性结果上叠加偏置向量：

$$Y_{i,j}=Y_{i,j}^{lin}+b_j$$

#### 2.9.2 CRPC 编码

线性部分编码多项式：

$$P_{Y^{lin}}(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{C-1} Y_{i,j}^{lin} X^{i C+j}$$

最终输出编码多项式：

$$P_Y(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{C-1} Y_{i,j} X^{i C+j}$$

由于 $b$ 是固定模型参数，不需要单独提交见证多项式。定义

先定义：对 $r\neq 1$，

$$\Gamma_N(r)=\dfrac{1-r^N}{1-r},$$

并约定

$$\Gamma_N(1)=N.$$

则输出层偏置的外点评值可由验证者确定性计算为

$$\mu_{bias}^{out}=\left(\sum_{j=0}^{C-1} b_j y_{out}^j\right)\cdot \Gamma_N\!\left(y_{out}^C\right).$$

这样无论 $y_{out}^C$ 是否等于 $1$，公式都有定义；协议不需要“重新采样 $y_{out}$”这一额外分支。验证者只需按上式在 $O(C+\log N)$ 时间内本地计算即可。

#### 2.9.3 zkMaP 折叠

> 张量绑定说明：根据第 0.7.3 节，此处的输出层折叠同样必须附带显式绑定子证明 $\pi_{bind}^{out}$。

根据 3.1 节规定的顺序，生成挑战：

$$y_{out}=H_{FS}(\text{transcript},[P_{H_{agg}}],[P_{Y^{lin}}],[P_Y],[V_{W_{out}}],[V_b])$$

定义

压缩特征：

$$a_m^{out}=\sum_{i=0}^{N-1} H_{agg,i,m} y_{out}^{i C}$$

压缩权重：

$$b_m^{out}=\sum_{j=0}^{C-1} W_{out,m,j} y_{out}^{j}$$

线性部分折叠值：

$$\mu_{acc}^{out}=\sum_{m=0}^{d-1} a_m^{out} b_m^{out}$$

偏置部分折叠值：

$$\mu_{bias}^{out}=\left(\sum_{j=0}^{C-1} b_j y_{out}^{j}\right)\cdot \Gamma_N\!\left(y_{out}^{C}\right)$$

总输出折叠值：

$$\mu_{out}=\mu_{acc}^{out}+\mu_{bias}^{out}$$

定义外点评值绑定：

$$\mu_{Y^{lin}}=\mu_{acc}^{out}$$

，并要求

$$P_{Y^{lin}}(y_{out})=\mu_{Y^{lin}},\qquad P_Y(y_{out})=P_{Y^{lin}}(y_{out})+\mu_{bias}^{out}=\mu_{out}$$

这里显式把线性部分多项式 $P_{Y^{lin}}$ 与最终输出多项式 $P_Y$ 在同一外点评值点上绑定，避免只在口头上使用 $Y^{lin}$ 而未对其对应多项式做承诺。

#### 2.9.4 共享维累加器

定义初始值：

$$Acc_{out}[0]=0$$

对每个 $m\in\{0,1,\ldots,d-1\}$，递推方程：

$$Acc_{out}[m+1]=Acc_{out}[m]+a_m^{out} b_m^{out}$$

对每个 $m\in\{d+1,d+2,\ldots,n_d-1\}$，填充：

$$Acc_{out}[m]=Acc_{out}[d]$$

并对所有 $m\in\{d,d+1,\ldots,n_d-1\}$，填充项清零：

$$a_m^{out}=0, \quad b_m^{out}=0$$

插值多项式：

$$P_{a^{out}}(X)=\sum_{m=0}^{n_d-1} a_m^{out} L_m^{(d)}(X),$$

$$P_{b^{out}}(X)=\sum_{m=0}^{n_d-1} b_m^{out} L_m^{(d)}(X),$$

$$P_{Acc^{out}}(X)=\sum_{m=0}^{n_d-1} Acc_{out}[m] L_m^{(d)}(X).$$

提交承诺：

$$[P_{Y^{lin}}], [P_Y], [P_{a^{out}}], [P_{b^{out}}], [P_{Acc^{out}}]$$

#### 2.9.5 输出

本步骤输出：

- 预测矩阵 $Y$；
- 输出多项式 $P_{Y^{lin}}$ 与 $P_Y$；
- 输出层 zkMaP witness；
- 输出层外点评值 $\mu_{Y^{lin}}$ 与 $\mu_{out}$。

## 3. 证明

**输入**：

证明算法输入：

- 参数生成输出 $(PK,VK_{KZG},VK_{static},VK_{model})$；
- 公共输入 $(I,src,dst,N,E,d_{in},d,C,B)$；
- witness 生成阶段得到的所有动态 witness

**输出**：

证明算法输出最终证明：

$$\pi_{GAT}$$

### 3.1 挑战顺序

为避免前后依赖冲突，并确保拓扑安全，挑战顺序固定如下：

> 安全性备注：第 1 步强制要求在生成任何随机挑战前，Transcript 必须包含所有决定计算逻辑的公共参数，以实现抗自适应攻击的安全性。

1. 吸入所有公共拓扑信息：包括 $N, E, d_{in}, d, C, B$ 以及所有公开拓扑重建对象 $[P_I], [P_{src}], [P_{dst}], [P_{Q_{new}^{edge}}], [P_{Q_{end}^{edge}}]$、基础动态承诺 $[P_H]$ 与静态特征表承诺，生成 $\eta_{feat},\beta_{feat}$；
2. 吸入 $[P_H],[P_{H'}],[V_W]$，生成 $y_{proj}$；
3. 吸入 $[P_{H'}]$，生成 $\xi$；
4. 吸入 $[P_{H'}],[P_{E_{src}}],[V_{a_{src}}]$，生成 $y_{src}$；
5. 吸入 $[P_{H'}],[P_{E_{dst}}],[V_{a_{dst}}]$，生成 $y_{dst}$；
6. 吸入 $[P_{H'}],[P_{H^{\star}}]$，生成 $y_{\star}$；
7. 吸入 $[P_{E_{src}}],[P_{H^{\star}}]$，生成 $\eta_{src},\beta_{src}$；
8. 吸入 $[P_S],[P_Z]$ 与 LeakyReLU 静态表承诺，生成 $\eta_L,\beta_L$；
9. 吸入 $[P_M],[P_{M^{edge}}],[P_{\Delta}]$ 与范围表承诺，生成 $\beta_R$；
10. 吸入 $[P_{\Delta}],[P_U]$ 与指数表承诺，生成 $\eta_{exp},\beta_{exp}$；
11. 吸入 $[P_{H_{agg}}],[P_{H_{agg}^{\star}}]$，生成 $y_{agg}$；
12. 吸入 $[P_{E_{dst}}],[P_M],[P_{Sum}],[P_{inv}],[P_{H_{agg}^{\star}}]$，以及边域广播多项式承诺 $[P_{E_{dst}^{edge}}], [P_{M^{edge}}], [P_{Sum^{edge}}], [P_{inv^{edge}}], [P_{H_{agg}^{\star,edge}}]$，生成 $\eta_{dst},\beta_{dst}$
13. 吸入 $[P_U],[P_{\alpha}],[P_{H_{src}^{\star,edge}}],[P_{Sum}],[P_{H_{agg}^{\star}}],[P_{H_{agg}^{\star,edge}}],[P_{\widehat v^{\star}}]$，生成 $\lambda_{psq}$；
14. 吸入 $[P_{H_{agg}}],[P_{Y^{lin}}],[P_Y],[V_{W_{out}}],[V_b]$，生成 $y_{out}$；
15. 吸入全部动态承诺（明确包含所有的基础见证多项式，以及所有的累加器 $P_R$、状态机 $P_{Acc}, P_{PSQ}$ 等全部派生见证多项式）与全部静态承诺，生成 quotient 聚合挑战 $\alpha_{quot}$；
16. 对每个工作域分别生成域内开放点 $z_{FH},z_{edge},z_{in},z_d,z_N$；
17. 对每个工作域分别生成域内 batch opening 折叠挑战 $v_{FH},v_{edge},v_{in},v_d,v_N$；
18. 生成外点评值批量折叠挑战 $\rho_{ext}$。

> 挑战顺序的安全性：在Fiat-Shamir变换中，顺序就是生命线。 证明者必须先提交承诺，然后才能获得挑战值。 如果顺序颠倒，证明者就可以根据题目去伪造数据。这个 18 步顺序确保了整个计算链条的因果律。

### 3.2 各工作域商多项式约束构造

#### 3.2.1 特征检索域约束

起点约束（累加器初始为 0）：

$$C_{feat,0}(X)=L_0^{(FH)}(X) P_{R_{feat}}(X)$$

状态转移约束（核心 LogUp 递推）：

$$\begin{aligned} C_{feat,1}(X)=&\big(P_{R_{feat}}(\omega_{FH}X)-P_{R_{feat}}(X)\big)\big(P_{Table^{feat}}(X)+\beta_{feat}\big)\big(P_{Query^{feat}}(X)+\beta_{feat}\big)\\ &-P_{Q_{tbl}^{feat}}(X)P_{m_{feat}}(X)\big(P_{Query^{feat}}(X)+\beta_{feat}\big)\\ &+P_{Q_{qry}^{feat}}(X)\big(P_{Table^{feat}}(X)+\beta_{feat}\big) \end{aligned}$$

终点约束（查表平衡，即 $R[n]=0$）：

$$C_{feat,2}(X)=L_{n_{FH}-1}^{(FH)}(X) P_{R_{feat}}(X)$$

定义特征检索商多项式：

$$t_{FH}(X)=\frac{\alpha_{quot}^0 C_{feat,0}(X)+\alpha_{quot}^1 C_{feat,1}(X)+\alpha_{quot}^2 C_{feat,2}(X)}{Z_{FH}(X)}$$

#### 3.2.2 源节点路由约束

**节点域端约束 (在** $\mathbb H_N$ **中)**：

$$C_{src\_node,0}(X)=L_0^{(N)}(X) P_{R_{src}^{node}}(X)$$

$$C_{src\_node,1}(X)=P_{Q_N}(X)\Big[\big(P_{R_{src}^{node}}(\omega_N X)-P_{R_{src}^{node}}(X)\big)\big(P_{Table^{src}}(X)+\beta_{src}\big)-P_{m_{src}}(X)\Big]$$

$$C_{src\_node,2}(X)=\big(1-P_{Q_N}(X)\big)\big(P_{R_{src}^{node}}(\omega_N X)-P_{R_{src}^{node}}(X)\big)$$

$$C_{src\_node,3}(X)=L_{n_N-1}^{(N)}(X)\big(P_{R_{src}^{node}}(X)-S_{src}\big)$$

$$C_{src\_node,bind}(X)=P_{Table^{src}}(X) - \big(P_I(X) + \eta_{src} P_{E_{src}}(X) + \eta_{src}^2 P_{H^{\star}}(X)\big)$$

**边域端约束 (在** $\mathbb H_{edge}$ **中)**：

$$C_{src\_edge,0}(X)=L_0^{(edge)}(X) P_{R_{src}^{edge}}(X)$$

$$C_{src\_edge,1}(X)=P_{Q_{qry}^{src}}(X)\Big[\big(P_{R_{src}^{edge}}(\omega_{edge} X)-P_{R_{src}^{edge}}(X)\big)\big(P_{Query^{src}}(X)+\beta_{src}\big)-1\Big]$$

$$C_{src\_edge,2}(X)=\big(1-P_{Q_{qry}^{src}}(X)\big)\big(P_{R_{src}^{edge}}(\omega_{edge} X)-P_{R_{src}^{edge}}(X)\big)$$

$$C_{src\_edge,3}(X)=L_{n_{edge}-1}^{(edge)}(X)\big(P_{R_{src}^{edge}}(X)-S_{src}\big)$$

$$C_{src\_edge,bind}(X)=P_{Query^{src}}(X) - \big(P_{src}(X) + \eta_{src} P_{E_{src}^{edge}}(X) + \eta_{src}^2 P_{H_{src}^{\star,edge}}(X)\big)$$

#### 3.2.3 目标节点路由约束（双累加器与结构绑定）

**节点域端约束（在** $\mathbb H_N$ **中）**：

$$C_{dst\_node,0}(X)=L_0^{(N)}(X) P_{R_{dst}^{node}}(X)$$

$$C_{dst\_node,1}(X)=P_{Q_N}(X)\Big[\big(P_{R_{dst}^{node}}(\omega_N X)-P_{R_{dst}^{node}}(X)\big)\big(P_{Table^{dst}}(X)+\beta_{dst}\big)-P_{m_{dst}}(X)\Big]$$

$$C_{dst\_node,2}(X)=\big(1-P_{Q_N}(X)\big)\big(P_{R_{dst}^{node}}(\omega_N X)-P_{R_{dst}^{node}}(X)\big)$$

$$C_{dst\_node,3}(X)=L_{n_N-1}^{(N)}(X)\big(P_{R_{dst}^{node}}(X)-S_{dst}\big)$$

$$C_{dst\_node,bind}(X)=P_{Table^{dst}}(X)-\big(P_I(X)+\eta_{dst}P_{E_{dst}}(X)+\eta_{dst}^2P_M(X)+\eta_{dst}^3P_{Sum}(X)+\eta_{dst}^4P_{inv}(X)+\eta_{dst}^5P_{H_{agg}^{\star}}(X)\big)$$

**边域端约束（在** $\mathbb H_{edge}$ **中）**：

$$C_{dst\_edge,0}(X)=L_0^{(edge)}(X) P_{R_{dst}^{edge}}(X)$$

$$C_{dst\_edge,1}(X)=P_{Q_{qry}^{dst}}(X)\Big[\big(P_{R_{dst}^{edge}}(\omega_{edge}X)-P_{R_{dst}^{edge}}(X)\big)\big(P_{Query^{dst}}(X)+\beta_{dst}\big)-1\Big]$$

$$C_{dst\_edge,2}(X)=\big(1-P_{Q_{qry}^{dst}}(X)\big)\big(P_{R_{dst}^{edge}}(\omega_{edge}X)-P_{R_{dst}^{edge}}(X)\big)$$

$$C_{dst\_edge,3}(X)=L_{n_{edge}-1}^{(edge)}(X)\big(P_{R_{dst}^{edge}}(X)-S_{dst}\big)$$

$$C_{dst\_edge,bind}(X)=P_{Query^{dst}}(X)-\big(P_{dst}(X)+\eta_{dst}P_{E_{dst}^{edge}}(X)+\eta_{dst}^2P_{M^{edge}}(X)+\eta_{dst}^3P_{Sum^{edge}}(X)+\eta_{dst}^4P_{inv^{edge}}(X)+\eta_{dst}^5P_{H_{agg}^{\star,edge}}(X)\big)$$

#### 3.2.4 LeakyReLU 查找约束

$$C_{L,0}(X)=L_0^{(edge)}(X) P_{R_L}(X)$$

$$\begin{aligned} C_{L,1}(X)=&\big(P_{R_L}(\omega_{edge}X)-P_{R_L}(X)\big)\big(P_{Table^{L}}(X)+\beta_L\big)\big(P_{Query^{L}}(X)+\beta_L\big)\\ &-P_{Q_{tbl}^{L}}(X)P_{m_L}(X)\big(P_{Query^{L}}(X)+\beta_L\big)\\ &+P_{Q_{qry}^{L}}(X)\big(P_{Table^{L}}(X)+\beta_L\big) \end{aligned}$$

$$C_{L,2}(X)=L_{n_{edge}-1}^{(edge)}(X) P_{R_L}(X)$$

$$C_{L,bind\_tbl}(X)=P_{Table^{L}}(X) - \big(P_{T_{LReLU},x}(X)+\eta_L P_{T_{LReLU},y}(X)\big)$$

$$C_{L,bind\_qry}(X)=P_{Query^{L}}(X) - \big(P_S(X)+\eta_L P_Z(X)\big)$$

#### 3.2.5 范围查找约束

$$C_{R,0}(X)=L_0^{(edge)}(X) P_{R_R}(X)$$

$$\begin{aligned} C_{R,1}(X)=&\big(P_{R_R}(\omega_{edge}X)-P_{R_R}(X)\big)\big(P_{Table^{R}}(X)+\beta_R\big)\big(P_{Query^{R}}(X)+\beta_R\big)\\ &-P_{Q_{tbl}^{R}}(X)P_{m_R}(X)\big(P_{Query^{R}}(X)+\beta_R\big)\\ &+P_{Q_{qry}^{R}}(X)\big(P_{Table^{R}}(X)+\beta_R\big). \end{aligned}$$

$$C_{R,2}(X)=L_{n_{edge}-1}^{(edge)}(X) P_{R_R}(X)$$

$$C_{R,bind\_tbl}(X)=P_{Table^{R}}(X) - P_{T_{range}}(X)$$

$$C_{R,bind\_qry}(X)=P_{Query^{R}}(X) - P_{\Delta}(X)$$

#### 3.2.6 指数查找约束

$$C_{exp,0}(X)=L_0^{(edge)}(X) P_{R_{exp}}(X)$$

$$\begin{aligned} C_{exp,1}(X)=&\big(P_{R_{exp}}(\omega_{edge}X)-P_{R_{exp}}(X)\big)\big(P_{Table^{exp}}(X)+\beta_{exp}\big)\big(P_{Query^{exp}}(X)+\beta_{exp}\big)\\ &-P_{Q_{tbl}^{exp}}(X)P_{m_{exp}}(X)\big(P_{Query^{exp}}(X)+\beta_{exp}\big)\\ &+P_{Q_{qry}^{exp}}(X)\big(P_{Table^{exp}}(X)+\beta_{exp}\big) \end{aligned}$$

$$C_{exp,2}(X)=L_{n_{edge}-1}^{(edge)}(X) P_{R_{exp}}(X)$$

$$C_{exp,bind\_tbl}(X)=P_{Table^{exp}}(X) - \big(P_{T_{exp},x}(X)+\eta_{exp} P_{T_{exp},y}(X)\big)$$

$$C_{exp,bind\_qry}(X)=P_{Query^{exp}}(X) - \big(P_{\Delta}(X)+\eta_{exp} P_U(X)\big)$$

#### 3.2.7 最大值与唯一性约束

定义差分一致性约束：

$$C_{max,0}(X)=P_{\Delta}(X)-P_{M^{edge}}(X)+P_Z(X)$$

定义布尔性约束（指示变量必须是 0 或 1）：

$$C_{max,1}(X)=P_{s_{max}}(X)\big(P_{s_{max}}(X)-1\big)$$

定义零差分约束（选中的点差分必须为 0）：

$$C_{max,2}(X)=P_{s_{max}}(X) P_{\Delta}(X)$$

定义计数状态机起点约束：

$$C_{max,3}(X)=L_0^{(edge)}(X)\big(P_{C_{max}}(X)-P_{s_{max}}(X)\big)$$

定义计数状态机转移约束（在组内累加指示变量）。对每个工作域点，把 $P_{C_{max}}(\omega_{edge}X)$ 看作下一个位置的计数值，因此定义：

$$\begin{aligned} C_{max,4}(X)=&P_{Q_{edge}^{valid}}(\omega_{edge}X) \cdot \Big[ P_{C_{max}}(\omega_{edge}X) - P_{Q_{new}^{edge}}(\omega_{edge}X)P_{s_{max}}(\omega_{edge}X) \\ & - \big(1-P_{Q_{new}^{edge}}(\omega_{edge}X)\big)\big(P_{C_{max}}(X)+P_{s_{max}}(\omega_{edge}X)\big) \Big] \\ & + \big(1-P_{Q_{edge}^{valid}}(\omega_{edge}X)\big) \cdot \big[ P_{C_{max}}(\omega_{edge}X) - P_{C_{max}}(X) \big] \end{aligned}$$

> 逻辑备注：此约束确保在每个目标节点组内，$C_{max}$ 严格累加 $s_{max}$ 的值；而在 padding 区，状态只能保持常值。结合组末约束 $Q_{end}^{edge}(C_{max}-1)=0$，强制实现“组内有且仅有一个最大值见证点”的语义。若组内存在多个数值相等的最大值，证明者需选择其中一个位置令 $s_{max}=1$，其余位置为 $0$。

定义组末检查约束（强制每组恰好有一个最大值见证）：

$$C_{max,5}(X)=P_{Q_{end}^{edge}}(X)\big(P_{C_{max}}(X)-1\big)$$

#### 3.2.8 分母逆元与边级归一化约束

定义节点有效区选择器：

$$Q_N[i]= \begin{cases} 1, & 0\leq i \leq N-1\\ 0, & N\leq i \leq n_N-1 \end{cases}$$

插值为：

$$P_{Q_N}(X)=\sum_{i=0}^{n_N-1} Q_N[i] L_i^{(N)}(X)$$

定义分母逆元约束：

$$C_{inv,0}(X)=P_{Q_N}(X)\big(P_{Sum}(X)P_{inv}(X)-1\big)$$

定义边级广播归一化约束：

$$C_{inv,1}(X)=P_{\alpha}(X)-P_U(X)P_{inv^{edge}}(X)$$

#### 3.2.9 聚合与PSQ约束

压缩加权特征乘法：

$$C_{\widehat{v}^{\star},0}(X)=P_{\widehat v^{\star}}(X)-P_{\alpha}(X)P_{H_{src}^{\star,edge}}(X)$$

定义 PSQ 聚合起点约束：

$$C_{psq,0}(X)=L_0^{(edge)}(X)\big(P_{PSQ}(X)-P_{w_{psq}}(X)\big)$$

定义 PSQ 聚合转移约束：

$$\begin{aligned} C_{psq,1}(X)=&P_{Q_{edge}^{valid}}(\omega_{edge}X)\cdot \Big[P_{PSQ}(\omega_{edge}X)-P_{Q_{new}^{edge}}(\omega_{edge}X)P_{w_{psq}}(\omega_{edge}X)\\&-\big(1-P_{Q_{new}^{edge}}(\omega_{edge}X)\big)\big(P_{PSQ}(X)+P_{w_{psq}}(\omega_{edge}X)\big)\Big]\\&+\big(1-P_{Q_{edge}^{valid}}(\omega_{edge}X)\big)\cdot\big(P_{PSQ}(\omega_{edge}X)-P_{PSQ}(X)\big) \end{aligned}$$

定义 PSQ 聚合组末输出约束：

$$C_{psq,2}(X)=P_{Q_{end}^{edge}}(X)\big(P_{PSQ}(X)-P_{T_{psq}^{edge}}(X)\big)$$

#### 3.2.10 共享维zkMaP约束

$d_{in}$ 域第一层投影：

$$C_{proj,0}(X)=L_0^{(in)}(X) P_{Acc^{proj}}(X),$$

$$C_{proj,1}(X)=P_{Q_{proj}^{valid}}(X)\big(P_{Acc^{proj}}(\omega_{in}X)-P_{Acc^{proj}}(X)-P_{a^{proj}}(X)P_{b^{proj}}(X)\big),$$

$$C_{proj,2}(X)=\big(1-P_{Q_{proj}^{valid}}(X)\big)\big(P_{Acc^{proj}}(\omega_{in}X)-P_{Acc^{proj}}(X)\big),$$

$$C_{proj,3}(X)=L_{d_{in}}^{(in)}(X)\big(P_{Acc^{proj}}(X)-\mu_{proj}\big).$$

$d$ 域的五组zkMaP域内约束：

对源注意力、目标注意力、源压缩特征、聚合压缩绑定、输出层五组 witness，分别定义完全相同模板的四条约束。为了避免歧义，下面逐组写出。

源注意力组

$$C_{srcmv,0}(X)=L_0^{(d)}(X) P_{Acc^{src}}(X),$$

$$C_{srcmv,1}(X)=P_{Q_d^{valid}}(X)\big(P_{Acc^{src}}(\omega_d X)-P_{Acc^{src}}(X)-P_{a^{src}}(X)P_{b^{src}}(X)\big),$$

$$C_{srcmv,2}(X)=\big(1-P_{Q_d^{valid}}(X)\big)\big(P_{Acc^{src}}(\omega_d X)-P_{Acc^{src}}(X)\big),$$

$$C_{srcmv,3}(X)=L_d^{(d)}(X)\big(P_{Acc^{src}}(X)-\mu_{src}\big).$$

目标注意力组

$$C_{dstmv,0}(X)=L_0^{(d)}(X) P_{Acc^{dst}}(X),$$

$$C_{dstmv,1}(X)=P_{Q_d^{valid}}(X)\big(P_{Acc^{dst}}(\omega_d X)-P_{Acc^{dst}}(X)-P_{a^{dst}}(X)P_{b^{dst}}(X)\big),$$

$$C_{dstmv,2}(X)=\big(1-P_{Q_d^{valid}}(X)\big)\big(P_{Acc^{dst}}(\omega_d X)-P_{Acc^{dst}}(X)\big),$$

$$C_{dstmv,3}(X)=L_d^{(d)}(X)\big(P_{Acc^{dst}}(X)-\mu_{dst}\big).$$

源压缩特征组

$$C_{star,0}(X)=L_0^{(d)}(X) P_{Acc^{\star}}(X),$$

$$C_{star,1}(X)=P_{Q_d^{valid}}(X)\big(P_{Acc^{\star}}(\omega_d X)-P_{Acc^{\star}}(X)-P_{a^{\star}}(X)P_{b^{\star}}(X)\big),$$

$$C_{star,2}(X)=\big(1-P_{Q_d^{valid}}(X)\big)\big(P_{Acc^{\star}}(\omega_d X)-P_{Acc^{\star}}(X)\big),$$

$$C_{star,3}(X)=L_d^{(d)}(X)\big(P_{Acc^{\star}}(X)-\mu_{\star}\big).$$

聚合压缩绑定组

$$C_{agg,0}(X)=L_0^{(d)}(X) P_{Acc^{agg}}(X),$$

$$C_{agg,1}(X)=P_{Q_d^{valid}}(X)\big(P_{Acc^{agg}}(\omega_d X)-P_{Acc^{agg}}(X)-P_{a^{agg}}(X)P_{b^{agg}}(X)\big),$$

$$C_{agg,2}(X)=\big(1-P_{Q_d^{valid}}(X)\big)\big(P_{Acc^{agg}}(\omega_d X)-P_{Acc^{agg}}(X)\big),$$

$$C_{agg,3}(X)=L_d^{(d)}(X)\big(P_{Acc^{agg}}(X)-\mu_{agg}\big).$$

输出层组

$$C_{out,0}(X)=L_0^{(d)}(X) P_{Acc^{out}}(X),$$

$$C_{out,1}(X)=P_{Q_d^{valid}}(X)\big(P_{Acc^{out}}(\omega_d X)-P_{Acc^{out}}(X)-P_{a^{out}}(X)P_{b^{out}}(X)\big),$$

$$C_{out,2}(X)=\big(1-P_{Q_d^{valid}}(X)\big)\big(P_{Acc^{out}}(\omega_d X)-P_{Acc^{out}}(X)\big),$$

$$C_{out,3}(X)=L_d^{(d)}(X)\big(P_{Acc^{out}}(X)-\mu_{acc}^{out}\big).$$

#### 3.2.11 各工作域商多项式

定义节点域总约束，这里不仅有逆元，还有双累加器的 Node 端：

$$\begin{aligned} N_N(X)= & \alpha_{quot}^{29} C_{inv,0}(X) \\ & + \sum_{r=0}^{3} \alpha_{quot}^{60+r} C_{src\_node,r}(X) + \alpha_{quot}^{64} C_{src\_node,bind}(X) \\ & + \sum_{r=0}^{3} \alpha_{quot}^{65+r} C_{dst\_node,r}(X) + \alpha_{quot}^{69} C_{dst\_node,bind}(X) \end{aligned}$$

定义节点域商多项式：

$$t_N(X)=\frac{N_N(X)}{Z_N(X)}$$

定义边域总约束多项式：

$$\begin{aligned} N_{edge}(X)=&\alpha_{quot}^3 C_{src\_edge,0}(X)+\alpha_{quot}^4 C_{src\_edge,1}(X)+\alpha_{quot}^5 C_{src\_edge,2}(X)\\ &+\alpha_{quot}^6 C_{dst\_edge,0}(X)+\alpha_{quot}^7 C_{dst\_edge,1}(X)+\alpha_{quot}^8 C_{dst\_edge,2}(X)\\ &+\alpha_{quot}^9 C_{L,0}(X)+\alpha_{quot}^{10} C_{L,1}(X)+\alpha_{quot}^{11} C_{L,2}(X)\\ &+\alpha_{quot}^{12} C_{R,0}(X)+\alpha_{quot}^{13} C_{R,1}(X)+\alpha_{quot}^{14} C_{R,2}(X)\\ &+\alpha_{quot}^{15} C_{exp,0}(X)+\alpha_{quot}^{16} C_{exp,1}(X)+\alpha_{quot}^{17} C_{exp,2}(X)\\ &+\alpha_{quot}^{18} C_{max,0}(X)+\alpha_{quot}^{19} C_{max,1}(X)+\alpha_{quot}^{20} C_{max,2}(X)\\ &+\alpha_{quot}^{21} C_{max,3}(X)+\alpha_{quot}^{22} C_{max,4}(X)+\alpha_{quot}^{23} C_{max,5}(X)\\ &+\alpha_{quot}^{24} C_{inv,1}(X)+\alpha_{quot}^{25} C_{\widehat{v}^{\star},0}(X)\\ &+\alpha_{quot}^{26} C_{psq,0}(X)+\alpha_{quot}^{27} C_{psq,1}(X)+\alpha_{quot}^{28} C_{psq,2}(X) \\ &+ \alpha_{quot}^{54} C_{L,bind\_tbl}(X) + \alpha_{quot}^{55} C_{L,bind\_qry}(X) \\ &+ \alpha_{quot}^{56} C_{R,bind\_tbl}(X) + \alpha_{quot}^{57} C_{R,bind\_qry}(X) \\ &+ \alpha_{quot}^{58} C_{exp,bind\_tbl}(X) + \alpha_{quot}^{59} C_{exp,bind\_qry}(X) \\ &+ \alpha_{quot}^{70} C_{src\_edge,3}(X) + \alpha_{quot}^{71} C_{src\_edge,bind}(X) \\ &+ \alpha_{quot}^{72} C_{dst\_edge,3}(X) + \alpha_{quot}^{73} C_{dst\_edge,bind}(X) \end{aligned}$$

定义边域商多项式：

$$t_{edge}(X)=\frac{N_{edge}(X)}{Z_{edge}(X)}$$

定义 $d_{in}$ 域商多项式：

$$t_{in}(X)=\frac{\alpha_{quot}^{30} C_{proj,0}(X)+\alpha_{quot}^{31} C_{proj,1}(X)+\alpha_{quot}^{32} C_{proj,2}(X)+\alpha_{quot}^{33} C_{proj,3}(X)}{Z_{in}(X)}$$

定义 $d$ 域商多项式

$$\begin{aligned} t_d(X)=\frac{1}{Z_d(X)}\Big(&\alpha_{quot}^{34} C_{srcmv,0}(X)+\alpha_{quot}^{35} C_{srcmv,1}(X)+\alpha_{quot}^{36} C_{srcmv,2}(X)+\alpha_{quot}^{37} C_{srcmv,3}(X)\\ &+\alpha_{quot}^{38} C_{dstmv,0}(X)+\alpha_{quot}^{39} C_{dstmv,1}(X)+\alpha_{quot}^{40} C_{dstmv,2}(X)+\alpha_{quot}^{41} C_{dstmv,3}(X)\\ &+\alpha_{quot}^{42} C_{star,0}(X)+\alpha_{quot}^{43} C_{star,1}(X)+\alpha_{quot}^{44} C_{star,2}(X)+\alpha_{quot}^{45} C_{star,3}(X)\\ &+\alpha_{quot}^{46} C_{agg,0}(X)+\alpha_{quot}^{47} C_{agg,1}(X)+\alpha_{quot}^{48} C_{agg,2}(X)+\alpha_{quot}^{49} C_{agg,3}(X)\\ &+\alpha_{quot}^{50} C_{out,0}(X)+\alpha_{quot}^{51} C_{out,1}(X)+\alpha_{quot}^{52} C_{out,2}(X)+\alpha_{quot}^{53} C_{out,3}(X)\Big) \end{aligned}$$

证明者提交商多项式承诺：

$$[t_{FH}], [t_{edge}], [t_{in}], [t_d], [t_N]$$

实现约定：虽然本节按工作域分组书写，导致 $\alpha_{quot}$ 的幂次在单个公式内部不一定按升序排列，但全局约束索引必须连续覆盖 $\{0,1,\ldots,73\}$，不存在空洞或复用。代码实现时应维护一个全局枚举表或宏定义，而不是手写散落的幂次常数。

### 3.3 域内多项式批量开放

对每个工作域，证明者在对应挑战点评开所有后续 quotient identity 实际用到的见证、多项式与商多项式。本文统一采用按“多项式—点评”对集合做的标准 multi-point KZG batch opening；当某个约束用到了 next-row 项 $P(\omega X)$ 时，必须把 $X$ 与 $\omega X$ 两个点评同时纳入开放集合。

各工作域的点评集合固定为：

- $S_{FH}=\{z_{FH},z_{FH}\omega_{FH}\}$；
- $S_{edge}=\{z_{edge},z_{edge}\omega_{edge}\}$；
- $S_{in}=\{z_{in},z_{in}\omega_{in}\}$；
- $S_d=\{z_d,z_d\omega_d\}$；
- $S_N=\{z_N,z_N\omega_N\}$。

各工作域的开放对象固定为：

1. 特征检索域 $\mathbb H_{FH}$：
	- 在 $z_{FH}$ 与 $z_{FH}\omega_{FH}$ 开 $P_{Table^{feat}},P_{Query^{feat}},P_{m_{feat}},P_{R_{feat}},t_{FH}$。
2. 边域 $\mathbb H_{edge}$：
	- src-route 相关：在 $z_{edge}$ 与 $z_{edge}\omega_{edge}$ 开 $P_{Query^{src}},P_{R_{src}^{edge}},P_{E_{src}^{edge}},P_{H_{src}^{\star,edge}}$；
	- dst-route 相关：在 $z_{edge}$ 与 $z_{edge}\omega_{edge}$ 开 $P_{Query^{dst}},P_{R_{dst}^{edge}},P_{E_{dst}^{edge}},P_{M^{edge}},P_{Sum^{edge}},P_{inv^{edge}},P_{H_{agg}^{\star,edge}}$；
	- LeakyReLU lookup：在 $z_{edge}$ 与 $z_{edge}\omega_{edge}$ 开 $P_{R_L},P_{Table^{L}},P_{Query^{L}},P_{m_L},P_S,P_Z$；
	- range lookup：在 $z_{edge}$ 与 $z_{edge}\omega_{edge}$ 开 $P_{R_R},P_{Table^{R}},P_{Query^{R}},P_{m_R},P_{\Delta}$；
	- exp lookup：在 $z_{edge}$ 与 $z_{edge}\omega_{edge}$ 开 $P_{R_{exp}},P_{Table^{exp}},P_{Query^{exp}},P_{m_{exp}},P_U$；
	- 最大值与唯一性：在 $z_{edge}$ 与 $z_{edge}\omega_{edge}$ 开 $P_{M^{edge}},P_{\Delta},P_{s_{max}},P_{C_{max}},P_Z$；
	- 归一化与聚合：在 $z_{edge}$ 与 $z_{edge}\omega_{edge}$ 开 $P_U,P_{inv^{edge}},P_{\alpha},P_{\widehat v^{\star}},P_{w_{psq}},P_{T_{psq}^{edge}},P_{PSQ},P_{H_{src}^{\star,edge}},P_{H_{agg}^{\star,edge}}$；
	- 商多项式：在 $z_{edge}$ 与 $z_{edge}\omega_{edge}$ 开 $t_{edge}$。
3. $d_{in}$ 域 $\mathbb H_{in}$：
	- 在 $z_{in}$ 与 $z_{in}\omega_{in}$ 开 $P_{a^{proj}},P_{b^{proj}},P_{Acc^{proj}},t_{in}$。
4. $d$ 域 $\mathbb H_d$：
	- 在 $z_d$ 与 $z_d\omega_d$ 开 $P_{a^{src}},P_{b^{src}},P_{Acc^{src}},P_{a^{dst}},P_{b^{dst}},P_{Acc^{dst}},P_{a^{\star}},P_{b^{\star}},P_{Acc^{\star}},P_{a^{agg}},P_{b^{agg}},P_{Acc^{agg}},P_{a^{out}},P_{b^{out}},P_{Acc^{out}},t_d$。
5. 节点域 $\mathbb H_N$：
	- src-route node 端：在 $z_N$ 与 $z_N\omega_N$ 开 $P_{E_{src}},P_{H^{\star}},P_{Table^{src}},P_{m_{src}},P_{R_{src}^{node}}$；
	- dst-route node 端：在 $z_N$ 与 $z_N\omega_N$ 开 $P_{E_{dst}},P_M,P_{Sum},P_{inv},P_{H_{agg}^{\star}},P_{Table^{dst}},P_{m_{dst}},P_{R_{dst}^{node}}$；
	- 商多项式：在 $z_N$ 与 $z_N\omega_N$ 开 $t_N$。

所有公开多项式（如 $P_I,P_{src},P_{dst},P_{Q_{new}^{edge}},P_{Q_{end}^{edge}},P_{Q_{edge}^{valid}},P_{Q_N},P_{Q_{proj}^{valid}},P_{Q_d^{valid}}$ 以及各 lookup / route 的选择器与静态表多项式）都由验证者本地重建并直接在相同点评计算，不进入证明者的域内 opening witness。

对每个工作域 $\mathcal D$，证明者使用独立随机折叠挑战 $v_{\mathcal D}$ 对该域内全部“多项式—点评”对做标准 multi-point KZG batch opening 折叠，并最终只输出一个该工作域的批量开放见证。

### 3.4 外点评值批量开放

需要做外点评值证明的对象是：

$$P_{H'}(y_{proj})=\mu_{proj}$$

$$P_{E_{src}}(y_{src})=\mu_{src}$$

$$P_{E_{dst}}(y_{dst})=\mu_{dst}$$

$$P_{H^{\star}}(y_{\star})=\mu_{\star}$$

$$P_{H_{agg}^{\star}}(y_{agg})=\mu_{agg}$$

$$P_{Y^{lin}}(y_{out})=\mu_{Y^{lin}}$$

$$P_Y(y_{out})=\mu_{out}$$

用批量折叠挑战 $\rho_{ext}$ 折叠成一个外部 opening witness：

$$W_{ext}(X)=\sum_{r=0}^{6} \rho_{ext}^{r} \cdot \frac{P_r(X)-\mu_r}{X-y_r}$$

其中

$$(P_0,\mu_0,y_0)=(P_{H'},\mu_{proj},y_{proj}),$$

$$(P_1,\mu_1,y_1)=(P_{E_{src}},\mu_{src},y_{src}),$$

$$(P_2,\mu_2,y_2)=(P_{E_{dst}},\mu_{dst},y_{dst}),$$

$$(P_3,\mu_3,y_3)=(P_{H^{\star}},\mu_{\star},y_{\star}),$$

$$(P_4,\mu_4,y_4)=(P_{H_{agg}^{\star}},\mu_{agg},y_{agg}),$$

$$(P_5,\mu_5,y_5)=(P_{Y^{lin}},\mu_{Y^{lin}},y_{out}),$$

$$(P_6,\mu_6,y_6)=(P_Y,\mu_{out},y_{out}).$$

提交承诺：

$$[W_{ext}]$$

### 3.5 最终证明对象

最终证明定义为

$$\pi_{GAT}=\Big(\mathbf{Com}_{dyn},\mathbf{Com}_{quot},\mathbf{Eval}_{pub},\mathbf{Eval}_{dom},\mathbf{Eval}_{ext},\mathbf{Open}_{dom},[W_{ext}],\Pi_{bind}\Big)$$

其中：

- $\mathbf{Com}_{dyn}$ 是全部动态 witness 承诺的固定顺序串联；
- $\mathbf{Com}_{quot}=\{[t_{FH}],[t_{edge}],[t_{in}],[t_d],[t_N]\}$；
- $\mathbf{Eval}_{pub}=\{S_{src},S_{dst}\}$；
- $\mathbf{Eval}_{dom}$ 是所有域内评值；
- $\mathbf{Eval}_{ext}=\{\mu_{proj},\mu_{src},\mu_{dst},\mu_{\star},\mu_{agg},\mu_{Y^{lin}},\mu_{out}\}$；
- $\mathbf{Open}_{dom}$ 是五个工作域各自一个的批量 KZG opening witness；
- $[W_{ext}]$ 是外点评值批量 opening witness；
- $\Pi_{bind}=\big(\pi_{bind}^{proj},\pi_{bind}^{src},\pi_{bind}^{dst},\pi_{bind}^{\star},\pi_{bind}^{agg},\pi_{bind}^{out}\big)$ 是六组 zkMaP / CRPC 折叠的显式张量绑定子证明。

## 4. 验证

**输入**：

验证算法输入：

- 公共输入 $(I,src,dst,N,E,d_{in},d,C,B)$；
- 静态验证键 $(VK_{KZG},VK_{static},VK_{model})$；
- 证明 $\pi_{GAT}$。

**输出**：

验证算法输出一个比特：接受或拒绝。

### 4.1 重新生成挑战

验证者严格按 3.1 中规定的顺序重建全部挑战：

$$\eta_{feat},\beta_{feat},y_{proj},\xi,y_{src},y_{dst},y_{\star},\eta_{src},\beta_{src},\eta_L,\beta_L,\beta_R,\eta_{exp},\beta_{exp},y_{agg},\eta_{dst},\beta_{dst},\lambda_{psq},y_{out},\alpha_{quot},z_{FH},z_{edge},z_{in},z_d,z_N,v_{FH},v_{edge},v_{in},v_d,v_N,\rho_{ext}.$$

若任何一步无法与证明中的承诺序列匹配，则立即拒绝。局部小节中若出现更简写的挑战公式或吸入对象描述，一律以本节的顺序与对象列表为唯一实现标准。

### 4.2 重建公开对象

验证者根据公共输入与静态键重建：

- $P_I$、$P_{src}$、$P_{dst}$；
- $P_{Q_{new}^{edge}}$、$P_{Q_{end}^{edge}}$、$P_{Q_{edge}^{valid}}$；
- 节点有效区选择器 $P_{Q_N}$；
- 共享维有效区选择器 $P_{Q_{proj}^{valid}}$ 与 $P_{Q_d^{valid}}$；
- 各 lookup / route 的表端有效区选择器与查询端有效区选择器；
- 静态表多项式承诺；
- 固定模型参数承诺；
- 证明中给出的公开标量 $S_{src},S_{dst}$。

输出层偏置折叠值利用等比数列快速公式本地计算。定义

先定义：对 $r\neq 1$，

$$\Gamma_N(r)=\dfrac{1-r^N}{1-r},$$

并约定

$$\Gamma_N(1)=N.$$

则

$$\mu_{bias}^{out}=\left( \sum_{j=0}^{C-1} b_j y_{out}^j \right) \cdot \Gamma_N\!\left(y_{out}^C\right).$$

该式在 $y_{out}^C=1$ 时仍然有定义，因此验证者不需要重新采样挑战。

验证者还必须本地求出所有在 quotient identity 中出现的公开多项式在各自查询点上的评值；特别是必须显式构造并使用 $P_I(z_N)$、$P_{src}(z_{edge})$、$P_{dst}(z_{edge})$、$P_{Q_{new}^{edge}}(z_{edge})$、$P_{Q_{end}^{edge}}(z_{edge})$、$P_{Q_{edge}^{valid}}(z_{edge})$、$P_{Q_N}(z_N)$、$P_{Q_{proj}^{valid}}(z_{in})$、$P_{Q_d^{valid}}(z_d)$ 以及各 lookup / route 选择器在 $z_{FH},z_{edge},z_N$ 与其必要相邻点上的评值。

拓扑索引承诺的处理规则保持不变：若图拓扑在多轮证明中固定，验证者可直接从 $VK_{static}$ 读取 $[P_I],[P_{src}],[P_{dst}]$；若拓扑为动态输入，验证者需本地执行 MSM 重建这些承诺。

### 4.3 外点评值检查

验证者用 batch KZG opening 检查以下七个外点评值条件：

第一层投影：

$$P_{H'}(y_{proj})=\mu_{proj}$$

源注意力：

$$P_{E_{src}}(y_{src})=\mu_{src}$$

目标注意力：

$$P_{E_{dst}}(y_{dst})=\mu_{dst}$$

压缩特征：

$$P_{H^{\star}}(y_{\star})=\mu_{\star}$$

聚合一致性：

$$P_{H_{agg}^{\star}}(y_{agg})=\mu_{agg}$$

输出层线性：

$$P_{Y^{lin}}(y_{out})=\mu_{Y^{lin}}$$

输出层最终值：

$$P_Y(y_{out})=\mu_{out}$$

其中对输出层的验证逻辑：验证者先本地定义

$$\mu_{acc}^{out}:=\mu_{Y^{lin}}$$

 ，然后检查

$$\mu_{out}=\mu_{acc}^{out}+\mu_{bias}^{out}$$

### 4.4 五大工作域代数等式检查

#### 4.4.1 特征检索域

验证者在点 $z_{FH}$ 以及必要的 $z_{FH}\omega_{FH}$ 处，代入全部特征检索域见证求值，检查

$$t_{FH}(z_{FH}) Z_{FH}(z_{FH}) = \alpha_{quot}^0 C_{feat,0}(z_{FH}) + \alpha_{quot}^1 C_{feat,1}(z_{FH}) + \alpha_{quot}^2 C_{feat,2}(z_{FH}).$$

在代入 $C_{feat,1}$ 时，必须显式使用 $z_{FH}\omega_{FH}$ 处的 next-row 评值，而不是把特征检索域退化为单点评检查。

#### 4.4.2 边域

验证者在点 $z_{edge}$ 以及必要的 $z_{edge}\omega_{edge}$ 处，代入全部边域见证求值，检查

$$\begin{aligned} t_{edge}(z_{edge}) Z_{edge}(z_{edge})= &\alpha_{quot}^3 C_{src_edge,0}(z_{edge})+\alpha_{quot}^4 C_{src_edge,1}(z_{edge})+\alpha_{quot}^5 C_{src_edge,2}(z_{edge})\ &+\alpha_{quot}^6 C_{dst_edge,0}(z_{edge})+\alpha_{quot}^7 C_{dst_edge,1}(z_{edge})+\alpha_{quot}^8 C_{dst_edge,2}(z_{edge})\ &+\alpha_{quot}^9 C_{L,0}(z_{edge})+\alpha_{quot}^{10} C_{L,1}(z_{edge})+\alpha_{quot}^{11} C_{L,2}(z_{edge})\ &+\alpha_{quot}^{12} C_{R,0}(z_{edge})+\alpha_{quot}^{13} C_{R,1}(z_{edge})+\alpha_{quot}^{14} C_{R,2}(z_{edge})\ &+\alpha_{quot}^{15} C_{exp,0}(z_{edge})+\alpha_{quot}^{16} C_{exp,1}(z_{edge})+\alpha_{quot}^{17} C_{exp,2}(z_{edge})\ &+\alpha_{quot}^{18} C_{max,0}(z_{edge})+\alpha_{quot}^{19} C_{max,1}(z_{edge})+\alpha_{quot}^{20} C_{max,2}(z_{edge})\ &+\alpha_{quot}^{21} C_{max,3}(z_{edge})+\alpha_{quot}^{22} C_{max,4}(z_{edge})+\alpha_{quot}^{23} C_{max,5}(z_{edge})\ &+\alpha_{quot}^{24} C_{inv,1}(z_{edge})+\alpha_{quot}^{25} C_{\widehat{v}^{\star},0}(z_{edge})\ &+\alpha_{quot}^{26} C_{psq,0}(z_{edge})+\alpha_{quot}^{27} C_{psq,1}(z_{edge})+\alpha_{quot}^{28} C_{psq,2}(z_{edge})\ &+\alpha_{quot}^{54} C_{L,bind_tbl}(z_{edge})+\alpha_{quot}^{55} C_{L,bind_qry}(z_{edge})\ &+\alpha_{quot}^{56} C_{R,bind_tbl}(z_{edge})+\alpha_{quot}^{57} C_{R,bind_qry}(z_{edge})\ &+\alpha_{quot}^{58} C_{exp,bind_tbl}(z_{edge})+\alpha_{quot}^{59} C_{exp,bind_qry}(z_{edge})\ &+\alpha_{quot}^{70} C_{src_edge,3}(z_{edge})+\alpha_{quot}^{71} C_{src_edge,bind}(z_{edge})\ &+\alpha_{quot}^{72} C_{dst_edge,3}(z_{edge})+\alpha_{quot}^{73} C_{dst_edge,bind}(z_{edge}). \end{aligned}$$

#### 4.4.3 节点域

验证者在点 $z_N$ 以及必要的 $z_N\omega_N$ 处，代入全部节点域见证求值，检查

$$\begin{aligned} t_N(z_N) Z_N(z_N)= &\alpha_{quot}^{29} C_{inv,0}(z_N)\ &+\alpha_{quot}^{60} C_{src_node,0}(z_N)+\alpha_{quot}^{61} C_{src_node,1}(z_N)+\alpha_{quot}^{62} C_{src_node,2}(z_N)+\alpha_{quot}^{63} C_{src_node,3}(z_N)+\alpha_{quot}^{64} C_{src_node,bind}(z_N)\ &+\alpha_{quot}^{65} C_{dst_node,0}(z_N)+\alpha_{quot}^{66} C_{dst_node,1}(z_N)+\alpha_{quot}^{67} C_{dst_node,2}(z_N)+\alpha_{quot}^{68} C_{dst_node,3}(z_N)+\alpha_{quot}^{69} C_{dst_node,bind}(z_N). \end{aligned}$$

在代入 $C_{src\_node,1},C_{src\_node,2},C_{dst\_node,1},C_{dst\_node,2}$ 时，必须显式使用 $z_N\omega_N$ 处的 next-row 评值，而不是把这些项退化为单点评检查。

#### 4.4.4 $d_{in}$ 域

验证者在点 $z_{in}$ 与必要的 $z_{in}\omega_{in}$ 处检查：

$$t_{in}(z_{in}) Z_{in}(z_{in}) = \alpha_{quot}^{30} C_{proj,0}(z_{in}) + \alpha_{quot}^{31} C_{proj,1}(z_{in}) + \alpha_{quot}^{32} C_{proj,2}(z_{in}) + \alpha_{quot}^{33} C_{proj,3}(z_{in})$$

#### 4.4.5 $d$ 域

验证者在点 $z_d$ 与必要的 $z_d\omega_d$ 处检查

$$\begin{aligned} t_d(z_d) Z_d(z_d)= &\alpha_{quot}^{34} C_{srcmv,0}(z_d)+\alpha_{quot}^{35} C_{srcmv,1}(z_d)+\alpha_{quot}^{36} C_{srcmv,2}(z_d)+\alpha_{quot}^{37} C_{srcmv,3}(z_d)\ &+\alpha_{quot}^{38} C_{dstmv,0}(z_d)+\alpha_{quot}^{39} C_{dstmv,1}(z_d)+\alpha_{quot}^{40} C_{dstmv,2}(z_d)+\alpha_{quot}^{41} C_{dstmv,3}(z_d)\ &+\alpha_{quot}^{42} C_{star,0}(z_d)+\alpha_{quot}^{43} C_{star,1}(z_d)+\alpha_{quot}^{44} C_{star,2}(z_d)+\alpha_{quot}^{45} C_{star,3}(z_d)\ &+\alpha_{quot}^{46} C_{agg,0}(z_d)+\alpha_{quot}^{47} C_{agg,1}(z_d)+\alpha_{quot}^{48} C_{agg,2}(z_d)+\alpha_{quot}^{49} C_{agg,3}(z_d)\ &+\alpha_{quot}^{50} C_{out,0}(z_d)+\alpha_{quot}^{51} C_{out,1}(z_d)+\alpha_{quot}^{52} C_{out,2}(z_d)+\alpha_{quot}^{53} C_{out,3}(z_d). \end{aligned}$$

### 4.5  KZG批量配对检查

对每个工作域，验证者都按第 3.3 节对应的点评集合做 KZG 批量开放配对检查：

- $\mathbb H_{FH}$：对 $\{z_{FH},z_{FH}\omega_{FH}\}$ 做 multi-point batch opening；
- $\mathbb H_{edge}$：对 $\{z_{edge},z_{edge}\omega_{edge}\}$ 做 multi-point batch opening；
- $\mathbb H_{in}$：对 $\{z_{in},z_{in}\omega_{in}\}$ 做 multi-point batch opening；
- $\mathbb H_d$：对 $\{z_d,z_d\omega_d\}$ 做 multi-point batch opening；
- $\mathbb H_N$：对 $\{z_N,z_N\omega_N\}$ 做 multi-point batch opening。

虽然每个工作域最终仍只保留一个批量 witness，但该 witness 一律来自标准多点评 KZG 折叠，而不是把不同点评误并为同一点。

### 4.6  接受条件

当且仅当以下条件同时成立时，验证者接受：

1. 全部 Fiat–Shamir 挑战按固定顺序可重建；
2. 七个外点评值批量 opening 全部通过；
3. 五个工作域的 quotient identity 全部成立；
4. 五个工作域按各自点评集合定义的 KZG batch opening 全部通过；
5. 六组张量绑定子证明 $\Pi_{bind}=\big(\pi_{bind}^{proj},\pi_{bind}^{src},\pi_{bind}^{dst},\pi_{bind}^{\star},\pi_{bind}^{agg},\pi_{bind}^{out}\big)$ 全部在其各自的域分离标签下验证通过；
6. 公开输入、静态表、模型键、量化缩放因子、padding 规则、次数界彼此一致。

若其中任一条件不成立，则验证者拒绝。


## 5. 实现接口与工程落地清单

本节不是新的协议层约束，而是把前文已经固定的对象、顺序与依赖关系整理成可以直接映射到代码仓库的数据结构与执行流程。实现时建议严格把“公共输入”“静态模型键”“动态见证”“公开评值”“KZG 证明对象”分层存放，避免在 witness builder、prover、verifier 三端混用记号。

### 5.1 建议的数据结构

建议在代码中把证明实例组织为以下五个结构体：

1. `PublicInstance`
   - 标量：`N_total, N, E, d_in, d, C, B`；
   - 局部节点绝对编号数组：`I[0..N-1]`；
   - 拓扑数组：`src[0..E-1], dst[0..E-1]`；
   - 由公共输入重建的选择器数组与多项式评值缓存。

2. `ModelVK`
   - 静态表承诺：`[T_H], [T_LReLU], [T_exp], [T_range]`；
   - 模型参数承诺：`[V_W], [V_a_src], [V_a_dst], [V_W_out], [V_b]`；
   - 工作域生成元与 vanishing polynomial 元数据。

3. `Witness`
   - 节点域原生对象：`H, H', E_src, E_dst, H_star, M, Sum, inv, H_agg, H_agg_star, Y`；
   - 边域原生对象：`E_src_edge, E_dst_edge, H_src_star_edge, H_agg_star_edge, Sum_edge, inv_edge, S, Z, M_edge, Delta, U, alpha, vhat_star`；
   - lookup / route / zkMaP / PSQ 所需全部辅助离散向量。

4. `CommitmentSet`
   - 所有动态 KZG 承诺；
   - 所有商多项式承诺；
   - 六组张量绑定子证明承诺/中间对象。

5. `Proof`
   - 域内 opening 评值表；
   - 域内 batch opening witness；
   - 外点评值与对应 batch opening witness；
   - 公开标量 `S_src, S_dst`；
   - `Pi_bind` 六组子证明。

### 5.2 建议的 witness builder 执行顺序

在工程中，最稳定的实现方式是把 witness 构造划分为如下顺序化流水线：

1. 读取公共输入并检查 `dst` 非降序；若不满足则执行稳定排序，并同步重排边相关辅助数组。
2. 根据 `I` 从全局特征表读取 `H`。
3. 计算第一层投影 `H' = H W`。
4. 计算 `E_src = H' a_src`、`E_dst = H' a_dst`、`H_star = H' c_xi`。
5. 完成 src-route 广播，得到 `E_src_edge, H_src_star_edge`。
6. 计算 `S = E_src_edge + E_dst_edge`，通过 LeakyReLU lookup 得到 `Z`。
7. 逐目标组扫描，得到 `M, M_edge, Delta, s_max, C_max`，并完成 range lookup witness。
8. 通过 exp lookup 得到 `U`。
9. 逐目标组扫描得到 `Sum`，再求逆得到 `inv`，并广播得到 `Sum_edge, inv_edge`。
10. 计算 `alpha = U * inv_edge`。
11. 先在明文中计算真实高维聚合矩阵 `H_agg`，再压缩得到 `H_agg_star`。
12. 计算 `vhat_star = alpha * H_src_star_edge`，构造 `w_psq, T_psq_edge, PSQ`。
13. 完成 dst-route 广播，得到 `E_dst_edge, M_edge, Sum_edge, inv_edge, H_agg_star_edge`，并在全部依赖对象承诺固定后 finalize dst-route。
14. 计算输出层 `Y = H_agg W_out + b`。
15. 依次构造全部 lookup、route、zkMaP、PSQ 辅助多项式与商多项式输入。

### 5.3 prover 侧必须固定的工程约束

1. 所有工作域数组必须分配到完整长度 `n_D`；禁止只保存真实区再在插值时临时猜测 padding。
2. 所有 transcript 吸入顺序必须与第 3.1 节完全一致；建议实现为不可变枚举表，禁止散落在各函数内部手写。
3. 所有 `Q_*` 选择器必须由公共输入确定后重建；不得把它们当作自由见证提交。
4. 所有 route 的公开总和 `S_src, S_dst` 必须在生成对应挑战后、提交最终 proof 前固定写入 proof 对象。
5. 所有 d 域 zkMaP 子协议都必须共享同一份 `Q_d^{valid}` 定义与 padding 规则。
6. 所有张量绑定子证明必须使用第 0.7.3 节规定的域分离标签；不能与主 transcript 的标签复用。

### 5.4 verifier 侧建议的校验顺序

验证器实现建议按如下顺序，以便尽早拒绝错误证明：

1. 解析 `PublicInstance`，检查维度、域大小与排序约定。
2. 重建所有公开多项式与公开选择器。
3. 重放第 3.1 节挑战顺序，检查 transcript 一致性。
4. 验证全部外点评值 opening。
5. 验证全部域内 batch opening。
6. 在五个工作域分别检查 quotient identity。
7. 验证六组张量绑定子证明。
8. 最后检查公开标量、量化配置、模型键版本号与静态表版本号一致性。
