# GAT-ZKML

## 0. 符号

### 0.1 模型结构

单层 GAT 的官方多头结构：

1. 隐藏层注意力头数量固定为：$K_{hid}=8$

2. 输出层注意力头数量固定为：$K_{out}=1$

3. 每个隐藏层注意力头完成聚合后，逐元素施加 ELU项。八个隐藏层注意力头的 ELU 后输出沿特征维拼接。

4. 输出层不是“仿射线性层 + 偏置”，而是单个输出注意力头。

5. 输出注意力头的激活函数固定为恒等映射，不再施加 ELU。

固定记号如下：

- 输入特征维度：$d_{in}$

- 第 $r$ 个隐藏层注意力头的编号：$r\in\{0,1,\ldots,7\}$

- 每个隐藏层注意力头的输出维度：$d_h$

- 拼接后的总维度：$d_{cat}=K_{hid}\cdot d_h=8d_h$

- 输出类别数：$C$

- 输出层的唯一注意力头记为：$out$


### 0.2 有限域、双线性群、KZG 与 Fiat–Shamir

记有限域为 $\mathbb F_p$，其中 $p$ 为大素数。所有域内运算都在 $\mathbb F_p$ 上进行。

采用定点数量化语义：若域元素 $x\in\mathbb F_p$ 的标准代表满足 $x>\frac{p-1}{2}$，则在实数语义下把它解释为负数$x-p$。

LeakyReLU、ELU、范围比较、指数查表、量化回缩，均在上述有符号域语义下定义。

记双线性群为$(\mathbb G_1,\mathbb G_2,\mathbb G_T,e)$，其中 $e:\mathbb G_1\times\mathbb G_2\to\mathbb G_T$ 是非退化双线性映射。生成元分别记为：$G_1\in\mathbb G_1,\quad G_2\in\mathbb G_2$ 

使用 KZG 多项式承诺，对次数严格小于 $D$ 的多项式$P(X)\in\mathbb F_p[X]$，其承诺记为：$[P]=P(\tau)G_1$，其中 $\tau\in\mathbb F_p$ 是 setup 阶段采样的隐藏陷门。

KZG 验证键记为：$VK_{KZG}=\{[1]_2,[\tau]_2\}$，其中 $[1]_2=G_2,\quad [\tau]_2=\tau G_2$ 

随机预言机统一写作$H_{FS}(\cdot)$ 

所有 Fiat–Shamir 挑战都必须按本文规定的固定顺序生成，禁止重排、回退、漏吸入或重复吸入。

### 0.3 图、局部子图与排序约定

- 全局节点总数：$N_{total}$

- 当前局部子图节点数：$N$

- 当前局部子图边数：$E$

- 局部节点绝对编号序列：$I=(I_0,I_1,\ldots,I_{N-1})$

- 局部边源索引序列：$src=(src(0),src(1),\ldots,src(E-1))$

- 局部边目标索引序列：$dst=(dst(0),dst(1),\ldots,dst(E-1))$


对任意边索引 $k\in\{0,1,\ldots,E-1\}$，都有：$src(k)\in\{0,1,\ldots,N-1\},\quad dst(k)\in\{0,1,\ldots,N-1\}$

固定要求边序列按目标节点索引非降序排列： 对任意满足 $0\le k_1<k_2\le E-1$ 的整数 $k_1,k_2$，都有$dst(k_1)\le dst(k_2)$

若输入边不满足这一条件，则在见证生成前做一次确定性稳定排序。 验证者只接收已经按此规范生成的公开边序列。

### 0.4 静态表与量化尺度

静态表包括：

1. 全局特征表：$T_H\in\mathbb F_p^{N_{total}\times d_{in}}$

2. LeakyReLU 查表：$T_{LReLU}\subseteq \mathbb F_p\times\mathbb F_p$

3. ELU 查表：$T_{ELU}\subseteq \mathbb F_p\times\mathbb F_p$

4. 指数查表：$T_{exp}\subseteq \mathbb F_p\times\mathbb F_p$

5. 范围表：$T_{range}=\{0,1,2,\ldots,2^B-1\}$，其中 $B$ 是范围检查位宽。


为避免多值歧义，要求：

- 对任意 $x\in\mathbb F_p$，在 $T_{LReLU}$ 中至多存在一个 $y\in\mathbb F_p$ 使得 $(x,y)\in T_{LReLU}$。
- 对任意 $x\in\mathbb F_p$，在 $T_{ELU}$ 中至多存在一个 $y\in\mathbb F_p$ 使得 $(x,y)\in T_{ELU}$。
- 对任意 $u\in\mathbb F_p$，在 $T_{exp}$ 中至多存在一个 $v\in\mathbb F_p$ 使得 $(u,v)\in T_{exp}$。

与 Softmax 链路有关的量化尺度全部视为公开参数：

- 差分反量化尺度：$S_{\Delta}$

- 指数输出尺度：$S_{exp}$

- 逆元见证尺度：$S_{inv}$

- 归一化权重尺度：$S_{\alpha}$

- 隐藏层聚合前尺度：$S_{agg,pre}$

- 隐藏层 ELU 后尺度：$S_{agg}$

- 拼接后表示尺度：$S_{cat}$

- 输出层中间量尺度：$S_{out}$


舍入方式（向最近整数舍入或截断）、乘法后的 rescale 规则也全部是公开参数。

### 0.5 模型参数

#### 0.5.1 隐藏层参数族

对每个隐藏层注意力头 $r\in\{0,1,\ldots,7\}$，定义：

- 投影矩阵$W^{(r)}\in\mathbb F_p^{d_{in}\times d_h}$

- 源方向注意力向量$a_{src}^{(r)}\in\mathbb F_p^{d_h}$

- 目标方向注意力向量$a_{dst}^{(r)}\in\mathbb F_p^{d_h}$


#### 0.5.2 输出层参数

定义：

- 输出投影矩阵$W^{(out)}\in\mathbb F_p^{d_{cat}\times C}$

- 输出层源方向注意力向量$a_{src}^{(out)}\in\mathbb F_p^C$

- 输出层目标方向注意力向量$a_{dst}^{(out)}\in\mathbb F_p^C$


正式协议中不再引入额外偏置向量 $b$。

所有模型参数在多次证明中固定，因此统一预处理进入模型验证键$VK_{model}$

### 0.6 全部中间变量

#### 0.6.1 原始特征

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 与每个输入维索引 $j\in\{0,1,\ldots,d_{in}-1\}$，定义$H_{i,j}=T_H[I_i,j]$

于是$H\in\mathbb F_p^{N\times d_{in}}$

#### 0.6.2 第 $r$ 个隐藏层注意力头的节点域变量

对每个 $r\in\{0,1,\ldots,7\}$，定义：

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


#### 0.6.3 第 $r$ 个隐藏层注意力头的边域变量

对每个 $r\in\{0,1,\ldots,7\}$，定义：

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


#### 0.6.4 拼接阶段变量

定义拼接结果$H_{cat}\in\mathbb F_p^{N\times d_{cat}}$，其中对任意 $i\in\{0,1,\ldots,N-1\}$、$r\in\{0,1,\ldots,7\}$、$j\in\{0,1,\ldots,d_h-1\}$，都有

$H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}$

定义拼接压缩挑战：$\xi_{cat}\in\mathbb F_p$

定义拼接压缩特征：$H_{cat}^{\star}\in\mathbb F_p^N \quad H_{cat,i}^{\star}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}\xi_{cat}^m$

#### 0.6.5 输出层变量

输出层有两类共享维：一类是输入共享维 $d_{cat}$，另一类是类别共享维 $C$。 因此我们不仅需要 $d_{cat}$ 域，还需要显式引入类别共享域。

定义输出层类别压缩挑战：$\xi_{out}\in\mathbb F_p$

定义：

1. 输出投影结果：$Y'\in\mathbb F_p^{N\times C}, \qquad Y'_{i,c}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out)}$

2. 输出层源注意力：$E_{src}^{(out)}\in\mathbb F_p^N, \qquad E_{src,i}^{(out)}=\sum_{c=0}^{C-1}Y'_{i,c}a_{src,c}^{(out)}$

3. 输出层目标注意力：$E_{dst}^{(out)}\in\mathbb F_p^N, \qquad E_{dst,i}^{(out)}=\sum_{c=0}^{C-1}Y'_{i,c}a_{dst,c}^{(out)}$

4. 输出层源注意力广播：$E_{src}^{edge(out)}\in\mathbb F_p^E, \qquad E_{src,k}^{edge(out)}=E_{src,src(k)}^{(out)}$

5. 输出层目标注意力广播：$E_{dst}^{edge(out)}\in\mathbb F_p^E, \qquad E_{dst,k}^{edge(out)}=E_{dst,dst(k)}^{(out)}$

6. 输出层线性打分：$S^{(out)}\in\mathbb F_p^E, \qquad S_k^{(out)}=E_{src,k}^{edge(out)}+E_{dst,k}^{edge(out)}$

7. 输出层 LeakyReLU 后打分：$Z^{(out)}\in\mathbb F_p^E, \qquad Z_k^{(out)}=LReLU(S_k^{(out)})$

8. 输出层组最大值：$M^{(out)}\in\mathbb F_p^N, \qquad M_i^{(out)}=\max\{Z_k^{(out)}\mid dst(k)=i\}$

9. 输出层最大值广播：$M^{edge(out)}\in\mathbb F_p^E, \qquad M_k^{edge(out)}=M_{dst(k)}^{(out)}$

10. 输出层非负差分：$\Delta^{+(out)}\in\mathbb F_p^E, \qquad \Delta_k^{+(out)}=M_k^{edge(out)}-Z_k^{(out)}$

11. 输出层指数输出：$U^{(out)}\in\mathbb F_p^E, \qquad U_k^{(out)}=ExpMap(\Delta_k^{+(out)})$

12. 输出层分母：$Sum^{(out)}\in\mathbb F_p^N, \qquad Sum_i^{(out)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out)}$

13. 输出层分母广播：$Sum^{edge(out)}\in\mathbb F_p^E, \qquad Sum_k^{edge(out)}=Sum_{dst(k)}^{(out)}$

14. 输出层逆元：$inv^{(out)}\in\mathbb F_p^N, \qquad inv_i^{(out)}=(Sum_i^{(out)})^{-1}$

15. 输出层逆元广播：$inv^{edge(out)}\in\mathbb F_p^E, \qquad inv_k^{edge(out)}=inv_{dst(k)}^{(out)}$

16. 输出层归一化权重：$\alpha^{(out)}\in\mathbb F_p^E, \qquad \alpha_k^{(out)}=U_k^{(out)}\cdot inv_k^{edge(out)}$

17. 输出层投影结果的类别压缩：$Y'^{\star}\in\mathbb F_p^N, \qquad Y_i'^{\star}=\sum_{c=0}^{C-1}Y'_{i,c}\xi_{out}^c$

18. 输出层投影结果的边域广播压缩：$Y'^{\star,edge}\in\mathbb F_p^E, \qquad Y_{k}'^{\star,edge}=Y_{src(k)}'^{\star}$

19. 输出层压缩加权边特征：$\widehat y^{\star}\in\mathbb F_p^E, \qquad \widehat y_k^{\star}=\alpha_k^{(out)}Y_{k}'^{\star,edge}$

20. 最终输出：$Y\in\mathbb F_p^{N\times C}, \qquad Y_{i,c}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out)}Y'_{src(k),c}$

21. 最终输出的类别压缩：$Y^{\star}\in\mathbb F_p^N, \qquad Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c$

22. 最终输出压缩广播：$Y^{\star,edge}\in\mathbb F_p^E, \qquad Y_k^{\star,edge}=Y_{dst(k)}^{\star}$


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

对任意工作域 $\mathbb H_{\mathcal D}$，记其拉格朗日基函数为：$L_0^{(\mathcal D)}(X),L_1^{(\mathcal D)}(X),\ldots,L_{n_{\mathcal D}-1}^{(\mathcal D)}(X)$

首点指示多项式记为：$First_{\mathcal D}(X)=L_0^{(\mathcal D)}(X)$

末点指示多项式记为：$Last_{\mathcal D}(X)=L_{n_{\mathcal D}-1}^{(\mathcal D)}(X)$

#### 0.7.3 公共有效区选择器

1. 边域有效区选择器：$Q_{edge}^{valid}[k]= \begin{cases} 1 &0\le k\le E-1,\\ 0,&E\le k\le n_{edge}-1 \end{cases}$

2. 节点域有效区选择器：$Q_N[i]= \begin{cases} 1 &0\le i\le N-1,\\ 0,&N\le i\le n_N-1 \end{cases}$

3. 输入共享维域有效区选择器：$Q_{in}^{valid}[m]= \begin{cases} 1 &0\le m\le d_{in}-1,\\ 0,&d_{in}\le m\le n_{in}-1 \end{cases}$

4. 隐藏层单头共享维域有效区选择器：$Q_{d_h}^{valid}[j]= \begin{cases} 1 &0\le j\le d_h-1,\\ 0,&d_h\le j\le n_{d_h}-1 \end{cases}$

5. 拼接共享维域有效区选择器：$Q_{cat}^{valid}[m]= \begin{cases} 1 &0\le m\le d_{cat}-1,\\ 0,&d_{cat}\le m\le n_{cat}-1 \end{cases}$

6. 输出层类别共享维域有效区选择器：$Q_C^{valid}[c]= \begin{cases} 1 &0\le c\le C-1,\\ 0,&C\le c\le n_C-1 \end{cases}$


#### 0.7.4 分组选择器

由于边按 $dst(k)$ 非降序排列，因此定义：

1. 组起点选择器：$Q_{new}^{edge}[k]= \begin{cases} 1 &k=0 \\ 1 &1\le k\le E-1\ \text{且}\ dst(k)\ne dst(k-1) \\ 0 &1\le k\le E-1\ \text{且}\ dst(k)=dst(k-1) \\ 0 &k\ge E  \end{cases}$

2. 组末尾选择器：$Q_{end}^{edge}[k]= \begin{cases} 1 &k=E-1 \\ 1 &0\le k\le E-2\ \text{且}\ dst(k+1)\ne dst(k) \\ 0 &0\le k\le E-2\ \text{且}\ dst(k+1)=dst(k) \\ 0 &k\ge E  \end{cases}$


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

于是：$P_C(X)=\sum_{t=0}^{\ell-1}A_t^{\langle n\rangle}(X)B_t(X)$

#### 0.7.8 张量绑定子证明

所有形如：$a_t=\sum_i M_{i,t}\chi_i(y)$ 或 $b_t=\sum_j W_{t,j}\psi_j(y)$ 的一维折叠向量，都必须与其来源矩阵承诺强绑定。

最终绑定子证明族记为：$\Pi_{bind}= \big( \pi_{bind}^{feat}, \pi_{bind}^{hidden,0},\ldots,\pi_{bind}^{hidden,7}, \pi_{bind}^{concat}, \pi_{bind}^{out} \big)$ 

每个子证明都必须使用独立的域分离标签初始化内部 transcript。

### 0.8 Padding、零分母、Softmax 可行性与缓存要求

#### 0.8.1 Padding 规则

对任意工作域 $\mathbb H_{\mathcal D}$，若真实长度为 $L_{\mathcal D}$、工作域长度为 $n_{\mathcal D}$，则所有离散列必须补到长度 $n_{\mathcal D}$。

对所有 $t\in\{L_{\mathcal D},L_{\mathcal D}+1,\ldots,n_{\mathcal D}-1\}$，统一规定：

- 见证值补零；
- 表值补零；
- 查询值补零；
- 重数补零；
- 由有效区选择器屏蔽无效位置；
- 所有状态机在 padding 区保持常值，或者等价地把 padding 区递推项强制清零。

#### 0.8.2 零分母冲突

对任意 LogUp 子系统 $\mathcal L$，零分母集合定义为：$Bad_{\mathcal L} = \{-Table[t]\mid t\text{ 在表有效区}\} \cup \{-Query[t]\mid t\text{ 在查询有效区}\}$

要求：每个 lookup / 路由子系统的挑战必须在决定其基础值的承诺固定之后再生成，并且语义上必须满足：$\beta_{\mathcal L}\notin Bad_{\mathcal L}$

#### 0.8.3 Softmax 可行性

要求每个有效节点至少有一条入边。 工程上应通过加入自环保证：$\#\{k\mid dst(k)=i\}\ge 1 \quad \forall i\in\{0,1,\ldots,N-1\}$

同时还必须满足：

1. 所有合法输入下的实数域最大分母严格小于模数 $p$；
2. 量化后最小聚合值至少为 $1$。

这样才能保证：$Sum_i^{(r)}\ne 0 \quad Sum_i^{(out)}\ne 0$ 在域内成立。

#### 0.8.4 热路径对象与缓存要求

热路径对象包括：所有域元素、所有挑战值、所有曲线点、SRS、静态表承诺、模型承诺、全部 FFT / IFFT 计划以及 quotient 计算的扩展域计划。

程序启动后必须一次性缓存：

- $PK$；
- $VK_{KZG}$；
- $VK_{static}$；
- $VK_{model}$；
- $\mathbb H_{FH},\mathbb H_{edge},\mathbb H_{in},\mathbb H_{d_h},\mathbb H_{cat},\mathbb H_C,\mathbb H_N$ 的 FFT / IFFT 计划；
- quotient 计算所需的扩展域计划。

单次证明或验证期间不得重复构造这些对象。

#### 0.8.5 公开元数据与版本元数据字段表

为保证证明对象、静态表、模型参数与量化配置之间的一致性，证明对象中必须携带一份**公开元数据 / 版本元数据**，其字段集合固定为：

1. 协议标识字段：$protocol\_id$

	该字段固定标识本文协议版本与 transcript 规则版本。

2. 模型结构版本字段：$model\_arch\_id$

	该字段固定绑定：隐藏层注意力头数量 $K_{hid}=8$、输出层注意力头数量 $K_{out}=1$、隐藏层使用 ELU、输出层使用恒等激活、拼接维定义 $d_{cat}=8d_h$。

3. 模型参数版本字段：$model\_param\_id$

	该字段唯一标识 $VK_{model}$ 所对应的参数承诺集合。

4. 静态表版本字段：$static\_table\_id$

	该字段唯一标识 $T_H,T_{LReLU},T_{ELU},T_{exp},T_{range}$ 以及 $VK_{static}$ 所对应的静态承诺集合。

5. 量化配置版本字段：$quant\_cfg\_id$

	该字段唯一标识全部量化尺度、舍入规则、rescale 规则与符号解释规则。

6. 工作域配置字段：$domain\_cfg$

	其内容必须至少包括：$(n_{FH},n_{edge},n_{in},n_{d_h},n_{cat},n_C,n_N)$ 以及对应生成元选择规则。

7. 维度配置字段：$dim\_cfg$

	其内容必须至少包括$(N,E,N_{total},d_{in},d_h,d_{cat},C,B)$ 

8. 编码与序列化字段：$encoding\_id$

	该字段固定标识域元素字节序、曲线点编码方式、承诺序列化顺序以及 transcript 吸入顺序编码方式。

9. Padding 与选择器规则字段：$padding\_rule\_id$

	该字段固定标识第 0.8.1 节规定的 padding 语义以及各有效区选择器的公共重建规则。

10. 次数界配置字段：$degree\_bound\_id$

	该字段固定标识 $D_{max}$ 与各工作域 quotient / opening 所采用的次数界策略。

上述字段是证明对象内部携带的公共字段。验证者必须从 $\pi_{GAT}$ 中按固定顺序解析这些字段，并把它们纳入 transcript 与一致性检查。这些字段不是建议性注释，而是参与一致性检查的正式公共对象。

## 1. 参数生成

### 1.1 输入

参数生成算法输入为：

- 安全参数 $\lambda$；
- 有限域模数 $p$；
- 局部子图规模上界 $(N,E)$；
- 全局节点数 $N_{total}$；
- 模型维度 $(d_{in},d_h,C)$；
- 范围检查位宽 $B$；
- 静态表 $T_H,T_{LReLU},T_{ELU},T_{exp},T_{range}$；
- 固定参数族 $\{W^{(r)},a_{src}^{(r)},a_{dst}^{(r)}\}_{r=0}^{7}$；
- 输出层参数 $W^{(out)},a_{src}^{(out)},a_{dst}^{(out)}$。

### 1.2 输出

参数生成算法输出：

- KZG 证明键

	$PK.$

- KZG 验证键

	$VK_{KZG}.$

- 静态表验证键

	$VK_{static}.$

- 模型验证键

	$VK_{model}.$

- 各工作域

	$\mathbb H_{FH},\mathbb H_{edge},\mathbb H_{in},\mathbb H_{d_h},\mathbb H_{cat},\mathbb H_C,\mathbb H_N.$

- 各工作域零化多项式

	$Z_{FH},Z_{edge},Z_{in},Z_{d_h},Z_{cat},Z_C,Z_N.$

### 1.3 工作域

#### 1.3.1 特征检索域

取最小二次幂长度 $n_{FH}$ 满足

$n_{FH}\ge \max\{N_{total}d_{in},Nd_{in}\}+2.$

定义

$\mathbb H_{FH}=\{1,\omega_{FH},\omega_{FH}^2,\ldots,\omega_{FH}^{n_{FH}-1}\}, \qquad Z_{FH}(X)=X^{n_{FH}}-1.$

#### 1.3.2 边域

取最小二次幂长度 $n_{edge}$ 满足

$n_{edge}\ge \max\{N,E,Nd_h,|T_{LReLU}|,|T_{ELU}|,|T_{exp}|,2^B\}+2.$

定义

$\mathbb H_{edge}=\{1,\omega_{edge},\omega_{edge}^2,\ldots,\omega_{edge}^{n_{edge}-1}\}, \qquad Z_{edge}(X)=X^{n_{edge}}-1.$

#### 1.3.3 输入共享维域

取最小二次幂长度 $n_{in}$ 满足

$n_{in}\ge d_{in}+2.$

定义

$\mathbb H_{in}=\{1,\omega_{in},\omega_{in}^2,\ldots,\omega_{in}^{n_{in}-1}\}, \qquad Z_{in}(X)=X^{n_{in}}-1.$

#### 1.3.4 隐藏层单头共享维域

取最小二次幂长度 $n_{d_h}$ 满足

$n_{d_h}\ge d_h+2.$

定义

$\mathbb H_{d_h}=\{1,\omega_{d_h},\omega_{d_h}^2,\ldots,\omega_{d_h}^{n_{d_h}-1}\}, \qquad Z_{d_h}(X)=X^{n_{d_h}}-1.$

#### 1.3.5 拼接共享维域

取最小二次幂长度 $n_{cat}$ 满足

$n_{cat}\ge d_{cat}+2.$

定义

$\mathbb H_{cat}=\{1,\omega_{cat},\omega_{cat}^2,\ldots,\omega_{cat}^{n_{cat}-1}\}, \qquad Z_{cat}(X)=X^{n_{cat}}-1.$

#### 1.3.6 输出层类别共享维域

取最小二次幂长度 $n_C$ 满足

$n_C\ge C+2.$

定义

$\mathbb H_C=\{1,\omega_C,\omega_C^2,\ldots,\omega_C^{n_C-1}\}, \qquad Z_C(X)=X^{n_C}-1.$

#### 1.3.7 节点域

取最小二次幂长度 $n_N$ 满足

$n_N\ge N+2.$

定义

$\mathbb H_N=\{1,\omega_N,\omega_N^2,\ldots,\omega_N^{n_N-1}\}, \qquad Z_N(X)=X^{n_N}-1.$

### 1.4 KZG 初始化

采样隐藏陷门

$\tau\xleftarrow{\$}\mathbb F_p.$

取次数上界

$D_{max}= \max\{ 3n_{FH}+8,\, 3n_{edge}+8,\, 2n_{in}+8,\, 2n_{d_h}+8,\, 2n_{cat}+8,\, 2n_C+8,\, 2n_N+8,\, Nd_{cat}+C,\, NC+1 \}.$

输出证明键

$PK=\{G_1,\tau G_1,\tau^2G_1,\ldots,\tau^{D_{max}}G_1\}.$

输出验证键

$VK_{KZG}=\{[1]_2,[\tau]_2\}.$

### 1.5 静态表与模型承诺

#### 1.5.1 全局特征表多项式

定义

$P_{T_H}(X)=\sum_{v=0}^{N_{total}-1}\sum_{j=0}^{d_{in}-1}T_H[v,j]X^{v\cdot d_{in}+j}.$

#### 1.5.2 LeakyReLU 表多项式

设 $T_{LReLU}$ 的第 $t$ 行是 $(T_{LReLU}[t,0],T_{LReLU}[t,1])$，定义

$P_{T_{LReLU},x}(X)=\sum_{t=0}^{|T_{LReLU}|-1}T_{LReLU}[t,0]L_t^{(edge)}(X),$

$P_{T_{LReLU},y}(X)=\sum_{t=0}^{|T_{LReLU}|-1}T_{LReLU}[t,1]L_t^{(edge)}(X).$

#### 1.5.3 ELU 表多项式

设 $T_{ELU}$ 的第 $t$ 行是 $(T_{ELU}[t,0],T_{ELU}[t,1])$，定义

$P_{T_{ELU},x}(X)=\sum_{t=0}^{|T_{ELU}|-1}T_{ELU}[t,0]L_t^{(edge)}(X),$

$P_{T_{ELU},y}(X)=\sum_{t=0}^{|T_{ELU}|-1}T_{ELU}[t,1]L_t^{(edge)}(X).$

#### 1.5.4 指数表多项式

设 $T_{exp}$ 的第 $t$ 行是 $(T_{exp}[t,0],T_{exp}[t,1])$，定义

$P_{T_{exp},x}(X)=\sum_{t=0}^{|T_{exp}|-1}T_{exp}[t,0]L_t^{(edge)}(X),$

$P_{T_{exp},y}(X)=\sum_{t=0}^{|T_{exp}|-1}T_{exp}[t,1]L_t^{(edge)}(X).$

#### 1.5.5 范围表多项式

定义

$P_{T_{range}}(X)=\sum_{t=0}^{2^B-1}t\,L_t^{(edge)}(X).$

#### 1.5.6 静态表验证键

定义

$VK_{static} = \{ [V_{T_H}], [V_{T_{LReLU},x}], [V_{T_{LReLU},y}], [V_{T_{ELU},x}], [V_{T_{ELU},y}], [V_{T_{exp},x}], [V_{T_{exp},y}], [V_{T_{range}}] \}.$

#### 1.5.7 模型承诺与模型验证键

对每个 $r\in\{0,1,\ldots,7\}$，承诺

$[V_{W^{(r)}}],\quad [V_{a_{src}^{(r)}}],\quad [V_{a_{dst}^{(r)}}].$

对输出层承诺

$[V_{W^{(out)}}],\quad [V_{a_{src}^{(out)}}],\quad [V_{a_{dst}^{(out)}}].$

因此

$VK_{model} = \Big( \{[V_{W^{(r)}}],[V_{a_{src}^{(r)}}],[V_{a_{dst}^{(r)}}]\}_{r=0}^{7}, [V_{W^{(out)}}],[V_{a_{src}^{(out)}}],[V_{a_{dst}^{(out)}}] \Big).$

### 1.6 公共拓扑多项式与辅助公开多项式

定义：

1. 源索引多项式

	$P_{src}(X)=\sum_{k=0}^{E-1}src(k)L_k^{(edge)}(X).$

2. 目标索引多项式

	$P_{dst}(X)=\sum_{k=0}^{E-1}dst(k)L_k^{(edge)}(X).$

3. 组起点选择器多项式

	$P_{Q_{new}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{new}^{edge}[k]L_k^{(edge)}(X).$

4. 组末尾选择器多项式

	$P_{Q_{end}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{end}^{edge}[k]L_k^{(edge)}(X).$

5. 边域有效区选择器多项式

	$P_{Q_{edge}^{valid}}(X)=\sum_{k=0}^{n_{edge}-1}Q_{edge}^{valid}[k]L_k^{(edge)}(X).$

6. 节点域有效区选择器多项式

	$P_{Q_N}(X)=\sum_{i=0}^{n_N-1}Q_N[i]L_i^{(N)}(X).$

7. 输入共享维有效区选择器多项式

	$P_{Q_{in}^{valid}}(X)=\sum_{m=0}^{n_{in}-1}Q_{in}^{valid}[m]L_m^{(in)}(X).$

8. 隐藏层共享维有效区选择器多项式

	$P_{Q_{d_h}^{valid}}(X)=\sum_{j=0}^{n_{d_h}-1}Q_{d_h}^{valid}[j]L_j^{(d_h)}(X).$

9. 拼接共享维有效区选择器多项式

	$P_{Q_{cat}^{valid}}(X)=\sum_{m=0}^{n_{cat}-1}Q_{cat}^{valid}[m]L_m^{(cat)}(X).$

10. 输出层类别共享维有效区选择器多项式

	$P_{Q_C^{valid}}(X)=\sum_{c=0}^{n_C-1}Q_C^{valid}[c]L_c^{(C)}(X).$

11. 节点域枚举多项式

	$P_{Idx_N}(X)=\sum_{i=0}^{n_N-1}Idx_N[i]L_i^{(N)}(X).$

12. 输入共享维枚举多项式

	$P_{Idx_{in}}(X)=\sum_{m=0}^{n_{in}-1}Idx_{in}[m]L_m^{(in)}(X).$

13. 隐藏层共享维枚举多项式

	$P_{Idx_{d_h}}(X)=\sum_{j=0}^{n_{d_h}-1}Idx_{d_h}[j]L_j^{(d_h)}(X).$

14. 拼接共享维枚举多项式

	$P_{Idx_{cat}}(X)=\sum_{m=0}^{n_{cat}-1}Idx_{cat}[m]L_m^{(cat)}(X).$

15. 输出类别枚举多项式

	$P_{Idx_C}(X)=\sum_{c=0}^{n_C-1}Idx_C[c]L_c^{(C)}(X).$

16. 特征检索表端行列辅助多项式

	$P_{Row_{feat}^{tbl}}(X)=\sum_{u=0}^{n_{FH}-1}Row_{feat}^{tbl}[u]L_u^{(FH)}(X),$

	$P_{Col_{feat}^{tbl}}(X)=\sum_{u=0}^{n_{FH}-1}Col_{feat}^{tbl}[u]L_u^{(FH)}(X).$

17. 特征检索查询端局部行列辅助多项式

	$P_{Row_{feat}^{qry}}(X)=\sum_{q=0}^{n_{FH}-1}Row_{feat}^{qry}[q]L_q^{(FH)}(X),$

	$P_{Col_{feat}^{qry}}(X)=\sum_{q=0}^{n_{FH}-1}Col_{feat}^{qry}[q]L_q^{(FH)}(X).$

18. 特征检索查询端绝对节点编号辅助多项式。对每个

$q=i\cdot d_{in}+j$

定义

$I_{feat}^{qry}[q]=I_i,$

然后插值得到

$P_{I_{feat}^{qry}}(X)=\sum_{q=0}^{n_{FH}-1}I_{feat}^{qry}[q]L_q^{(FH)}(X).$

这些对象全部由验证者根据公共输入本地重建，不属于动态承诺对象。

## 2. 见证生成与承诺

### 2.0 总体顺序与实现约束

按“前向计算 → 挑战与轨迹列生成 → 多项式编码 → 提交承诺”的顺序书写。 工程实现必须严格分为三阶段：

1. 前向阶段：只计算神经网络明文对象；
2. 挑战与轨迹阶段：按固定 Fiat–Shamir 顺序生成挑战，并据此构造表列、查询列、重数列、累加器列、状态机列；
3. 承诺与证明阶段：集中进行 FFT / MSM，生成全部承诺与全部开放证明。

禁止把前向计算与承诺生成耦合在一起。

### 2.1 数据加载与特征检索

#### 2.1.1 原始特征

对每个局部节点 $i\in\{0,1,\ldots,N-1\}$ 与每个输入维索引 $j\in\{0,1,\ldots,d_{in}-1\}$，定义

$H_{i,j}=T_H[I_i,j].$

#### 2.1.2 特征检索表端与查询端

生成挑战

$\eta_{feat},\beta_{feat}.$

对每个全局表端索引

$u=v\cdot d_{in}+j,\qquad 0\le v\le N_{total}-1,\quad 0\le j\le d_{in}-1,$

定义表端离散列

$Table^{feat}[u]=v+\eta_{feat}j+\eta_{feat}^2T_H[v,j].$

对每个查询索引

$q=i\cdot d_{in}+j,\qquad 0\le i\le N-1,\quad 0\le j\le d_{in}-1,$

定义查询端离散列

$Query^{feat}[q]=I_i+\eta_{feat}j+\eta_{feat}^2H_{i,j}.$

#### 2.1.3 重数列

对每个全局条目索引 $u=v\cdot d_{in}+j$，定义

$m_{feat}[u]=\#\{i\in\{0,1,\ldots,N-1\}\mid I_i=v\}.$

#### 2.1.4 表端与查询端有效区选择器

定义表端有效区选择器

$Q_{tbl}^{feat}[t]= \begin{cases} 1,&0\le t\le N_{total}d_{in}-1,\\ 0,&N_{total}d_{in}\le t\le n_{FH}-1. \end{cases}$

定义查询端有效区选择器

$Q_{qry}^{feat}[t]= \begin{cases} 1,&0\le t\le Nd_{in}-1,\\ 0,&Nd_{in}\le t\le n_{FH}-1. \end{cases}$

#### 2.1.5 特征检索累加器

定义累加器离散列

$R_{feat}[0]=0.$

对所有 $t\in\{0,1,\ldots,n_{FH}-2\}$，定义

$R_{feat}[t+1] = R_{feat}[t] + Q_{tbl}^{feat}[t]\cdot\frac{m_{feat}[t]}{Table^{feat}[t]+\beta_{feat}} - Q_{qry}^{feat}[t]\cdot\frac{1}{Query^{feat}[t]+\beta_{feat}}.$

padding 区保持常值。

#### 2.1.6 多项式编码

定义：

1. 节点绝对编号多项式

	$P_I(X)=\sum_{i=0}^{N-1}I_iL_i^{(N)}(X).$

2. 原始特征系数多项式

	$P_H(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{d_{in}-1}H_{i,j}X^{i\cdot d_{in}+j}.$

3. 表端多项式

	$P_{Table^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Table^{feat}[t]L_t^{(FH)}(X).$

4. 查询端多项式

	$P_{Query^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Query^{feat}[t]L_t^{(FH)}(X).$

5. 重数多项式

	$P_{m_{feat}}(X)=\sum_{t=0}^{n_{FH}-1}m_{feat}[t]L_t^{(FH)}(X).$

6. 有效区选择器多项式

	$P_{Q_{tbl}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Q_{tbl}^{feat}[t]L_t^{(FH)}(X),$

	$P_{Q_{qry}^{feat}}(X)=\sum_{t=0}^{n_{FH}-1}Q_{qry}^{feat}[t]L_t^{(FH)}(X).$

7. 累加器多项式

	$P_{R_{feat}}(X)=\sum_{t=0}^{n_{FH}-1}R_{feat}[t]L_t^{(FH)}(X).$

#### 2.1.7 承诺

证明者提交承诺：

$[P_H],\ [P_{Table^{feat}}],\ [P_{Query^{feat}}],\ [P_{m_{feat}}],\ [P_{Q_{tbl}^{feat}}],\ [P_{Q_{qry}^{feat}}],\ [P_{R_{feat}}].$

其中$[P_I]$不作为动态承诺对象由证明者提交。验证者根据公共输入本地重建$[P_I]$，并在第 3.2.1 节规定的位置将其吸入 transcript。

### 2.2 第 $r$ 个隐藏层注意力头的完整见证生成

以下步骤对每个 $r\in\{0,1,\ldots,7\}$ 独立执行。

#### 2.2.1 投影

对每个节点 $i$ 与每个隐藏维索引 $j\in\{0,1,\ldots,d_h-1\}$，定义

$H_{i,j}'^{(r)}=\sum_{m=0}^{d_{in}-1}H_{i,m}W_{m,j}^{(r)}.$

定义输出系数多项式

$P_{H'^{(r)}}(X)=\sum_{i=0}^{N-1}\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}X^{i\cdot d_h+j}.$

对每个共享维索引 $m\in\{0,1,\ldots,d_{in}-1\}$，定义

$A_m^{proj(r)}(X)=\sum_{i=0}^{N-1}H_{i,m}X^{i\cdot d_h},$

$B_m^{proj(r)}(X)=\sum_{j=0}^{d_h-1}W_{m,j}^{(r)}X^j.$

于是

$P_{H'^{(r)}}(X)=\sum_{m=0}^{d_{in}-1}A_m^{proj(r)}(X)B_m^{proj(r)}(X).$

生成挑战

$y_{proj}^{(r)}=H_{FS}(\text{transcript},[P_H],[P_{H'^{(r)}}],[V_{W^{(r)}}]).$

定义折叠向量

$a_m^{proj(r)}=A_m^{proj(r)}(y_{proj}^{(r)})=\sum_{i=0}^{N-1}H_{i,m}(y_{proj}^{(r)})^{i\cdot d_h},$

$b_m^{proj(r)}=B_m^{proj(r)}(y_{proj}^{(r)})=\sum_{j=0}^{d_h-1}W_{m,j}^{(r)}(y_{proj}^{(r)})^j.$

定义外点评值

$\mu_{proj}^{(r)}=\sum_{m=0}^{d_{in}-1}a_m^{proj(r)}b_m^{proj(r)}.$

要求

$P_{H'^{(r)}}(y_{proj}^{(r)})=\mu_{proj}^{(r)}.$

定义共享维累加器

$Acc_{proj}^{(r)}[0]=0,$

$Acc_{proj}^{(r)}[m+1]=Acc_{proj}^{(r)}[m]+a_m^{proj(r)}b_m^{proj(r)}, \qquad 0\le m\le d_{in}-1.$

对 $m\ge d_{in}$ 的 padding 区，统一规定

$a_m^{proj(r)}=0,\qquad b_m^{proj(r)}=0,\qquad Acc_{proj}^{(r)}[m+1]=Acc_{proj}^{(r)}[m].$

定义多项式

$P_{a^{proj(r)}}(X)=\sum_{m=0}^{n_{in}-1}a_m^{proj(r)}L_m^{(in)}(X),$

$P_{b^{proj(r)}}(X)=\sum_{m=0}^{n_{in}-1}b_m^{proj(r)}L_m^{(in)}(X),$

$P_{Acc^{proj(r)}}(X)=\sum_{m=0}^{n_{in}-1}Acc_{proj}^{(r)}[m]L_m^{(in)}(X).$

提交承诺

$[P_{H'^{(r)}}],\ [P_{a^{proj(r)}}],\ [P_{b^{proj(r)}}],\ [P_{Acc^{proj(r)}}].$

#### 2.2.2 源注意力绑定

生成压缩挑战

$\xi^{(r)}=H_{FS}(\text{transcript},[P_{H'^{(r)}}]).$

对每个节点 $i$，定义

$E_{src,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{src,j}^{(r)}.$

把 $E_{src}^{(r)}$ padding 到长度 $n_N$，插值成

$P_{E_{src}^{(r)}}(X)=\sum_{i=0}^{n_N-1}E_{src}^{(r)}[i]L_i^{(N)}(X).$

生成挑战

$y_{src}^{(r)}=H_{FS}(\text{transcript},[P_{H'^{(r)}}],[P_{E_{src}^{(r)}}],[V_{a_{src}^{(r)}}]).$

定义折叠向量

$a_j^{src(r)}=\sum_{i=0}^{N-1}H_{i,j}'^{(r)}L_i^{(N)}(y_{src}^{(r)}),$

$b_j^{src(r)}=a_{src,j}^{(r)}.$

定义外点评值

$\mu_{src}^{(r)}=\sum_{j=0}^{d_h-1}a_j^{src(r)}b_j^{src(r)}.$

要求

$P_{E_{src}^{(r)}}(y_{src}^{(r)})=\mu_{src}^{(r)}.$

定义共享维累加器

$Acc_{src}^{(r)}[0]=0,$

$Acc_{src}^{(r)}[j+1]=Acc_{src}^{(r)}[j]+a_j^{src(r)}b_j^{src(r)}, \qquad 0\le j\le d_h-1.$

对 padding 区统一规定

$a_j^{src(r)}=0,\qquad b_j^{src(r)}=0,\qquad Acc_{src}^{(r)}[j+1]=Acc_{src}^{(r)}[j].$

定义

$P_{a^{src(r)}}(X),\quad P_{b^{src(r)}}(X),\quad P_{Acc^{src(r)}}(X)$

分别为其在 $\mathbb H_{d_h}$ 上插值多项式。

提交承诺

$[P_{E_{src}^{(r)}}],\ [P_{a^{src(r)}}],\ [P_{b^{src(r)}}],\ [P_{Acc^{src(r)}}].$

#### 2.2.3 目标注意力绑定

对每个节点 $i$，定义

$E_{dst,i}^{(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}a_{dst,j}^{(r)}.$

把 $E_{dst}^{(r)}$ padding 到长度 $n_N$，插值成

$P_{E_{dst}^{(r)}}(X)=\sum_{i=0}^{n_N-1}E_{dst}^{(r)}[i]L_i^{(N)}(X).$

生成挑战

$y_{dst}^{(r)}=H_{FS}(\text{transcript},[P_{H'^{(r)}}],[P_{E_{dst}^{(r)}}],[V_{a_{dst}^{(r)}}]).$

定义折叠向量

$a_j^{dst(r)}=\sum_{i=0}^{N-1}H_{i,j}'^{(r)}L_i^{(N)}(y_{dst}^{(r)}),$

$b_j^{dst(r)}=a_{dst,j}^{(r)}.$

定义外点评值

$\mu_{dst}^{(r)}=\sum_{j=0}^{d_h-1}a_j^{dst(r)}b_j^{dst(r)}.$

要求

$P_{E_{dst}^{(r)}}(y_{dst}^{(r)})=\mu_{dst}^{(r)}.$

定义共享维累加器

$Acc_{dst}^{(r)}[0]=0,$

$Acc_{dst}^{(r)}[j+1]=Acc_{dst}^{(r)}[j]+a_j^{dst(r)}b_j^{dst(r)}, \qquad 0\le j\le d_h-1.$

定义

$P_{a^{dst(r)}}(X),\quad P_{b^{dst(r)}}(X),\quad P_{Acc^{dst(r)}}(X)$

分别为其在 $\mathbb H_{d_h}$ 上插值多项式。

提交承诺

$[P_{E_{dst}^{(r)}}],\ [P_{a^{dst(r)}}],\ [P_{b^{dst(r)}}],\ [P_{Acc^{dst(r)}}].$

#### 2.2.4 压缩特征绑定

对每个节点 $i$，定义

$H_i^{\star(r)}=\sum_{j=0}^{d_h-1}H_{i,j}'^{(r)}(\xi^{(r)})^j.$

插值成

$P_{H^{\star(r)}}(X)=\sum_{i=0}^{n_N-1}H^{\star(r)}[i]L_i^{(N)}(X).$

生成挑战

$y_{\star}^{(r)}=H_{FS}(\text{transcript},[P_{H'^{(r)}}],[P_{H^{\star(r)}}]).$

定义折叠向量

$a_j^{\star(r)}=\sum_{i=0}^{N-1}H_{i,j}'^{(r)}L_i^{(N)}(y_{\star}^{(r)}),$

$b_j^{\star(r)}=(\xi^{(r)})^j.$

定义外点评值

$\mu_{\star}^{(r)}=\sum_{j=0}^{d_h-1}a_j^{\star(r)}b_j^{\star(r)}.$

要求

$P_{H^{\star(r)}}(y_{\star}^{(r)})=\mu_{\star}^{(r)}.$

定义共享维累加器

$Acc_{\star}^{(r)}[0]=0,$

$Acc_{\star}^{(r)}[j+1]=Acc_{\star}^{(r)}[j]+a_j^{\star(r)}b_j^{\star(r)}.$

定义

$P_{a^{\star(r)}}(X),\quad P_{b^{\star(r)}}(X),\quad P_{Acc^{\star(r)}}(X)$

为其在 $\mathbb H_{d_h}$ 上插值多项式。

提交承诺

$[P_{H^{\star(r)}}],\ [P_{a^{\star(r)}}],\ [P_{b^{\star(r)}}],\ [P_{Acc^{\star(r)}}].$

#### 2.2.5 源路由

对每个边索引 $k$，定义

$E_{src,k}^{edge(r)}=E_{src,src(k)}^{(r)}, \qquad H_{src,k}^{\star,edge(r)}=H_{src(k)}^{\star(r)}.$

生成挑战

$\eta_{src}^{(r)},\beta_{src}^{(r)}.$

对每个节点 $i$，定义表端

$Table^{src(r)}[i] = i+\eta_{src}^{(r)}E_{src,i}^{(r)}+(\eta_{src}^{(r)})^2H_i^{\star(r)}.$

定义节点重数

$m_{src}^{(r)}[i]=\#\{k\mid src(k)=i\}.$

对每个边索引 $k$，定义查询端

$Query^{src(r)}[k] = src(k)+\eta_{src}^{(r)}E_{src,k}^{edge(r)}+(\eta_{src}^{(r)})^2H_{src,k}^{\star,edge(r)}.$

定义公开总和

$S_{src}^{(r)} = \sum_{i=0}^{N-1}\frac{m_{src}^{(r)}[i]}{Table^{src(r)}[i]+\beta_{src}^{(r)}} = \sum_{k=0}^{E-1}\frac{1}{Query^{src(r)}[k]+\beta_{src}^{(r)}}.$

定义节点端累加器

$R_{src}^{node(r)}[0]=0,$

$R_{src}^{node(r)}[i+1] = R_{src}^{node(r)}[i] + Q_N[i]\cdot\frac{m_{src}^{(r)}[i]}{Table^{src(r)}[i]+\beta_{src}^{(r)}}, \qquad 0\le i\le n_N-2.$

定义边端累加器

$R_{src}^{edge(r)}[0]=0,$

$R_{src}^{edge(r)}[k+1] = R_{src}^{edge(r)}[k] + Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{src(r)}[k]+\beta_{src}^{(r)}}, \qquad 0\le k\le n_{edge}-2.$

插值得到

$P_{Table^{src(r)}},\ P_{Query^{src(r)}},\ P_{m_{src}^{(r)}},\ P_{R_{src}^{node(r)}},\ P_{R_{src}^{edge(r)}}.$

提交承诺

$[P_{Table^{src(r)}}],\ [P_{Query^{src(r)}}],\ [P_{m_{src}^{(r)}}],\ [P_{R_{src}^{node(r)}}],\ [P_{R_{src}^{edge(r)}}].$

#### 2.2.6 LeakyReLU、最大值、唯一性与范围检查

对每个边索引 $k$，定义

$S_k^{(r)}=E_{src,k}^{edge(r)}+E_{dst,k}^{edge(r)}, \qquad Z_k^{(r)}=LReLU(S_k^{(r)}).$

生成挑战

$\eta_L^{(r)},\beta_L^{(r)}.$

定义 LeakyReLU 表端

$Table^{L(r)}[t]=T_{LReLU}[t,0]+\eta_L^{(r)}T_{LReLU}[t,1],$

查询端

$Query^{L(r)}[k]=S_k^{(r)}+\eta_L^{(r)}Z_k^{(r)}.$

定义表端有效区选择器

$Q_{tbl}^{L(r)}[t]= \begin{cases} 1,&0\le t\le |T_{LReLU}|-1,\\ 0,&|T_{LReLU}|\le t\le n_{edge}-1, \end{cases}$

查询端有效区选择器

$Q_{qry}^{L(r)}[k]=Q_{edge}^{valid}[k].$

对每个表端索引 $t$，定义重数

$m_L^{(r)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid (S_k^{(r)},Z_k^{(r)})=(T_{LReLU}[t,0],T_{LReLU}[t,1])\}.$

定义累加器

$R_L^{(r)}[0]=0,$

$R_L^{(r)}[k+1] = R_L^{(r)}[k] + Q_{tbl}^{L(r)}[k]\cdot\frac{m_L^{(r)}[k]}{Table^{L(r)}[k]+\beta_L^{(r)}} - Q_{qry}^{L(r)}[k]\cdot\frac{1}{Query^{L(r)}[k]+\beta_L^{(r)}}, \qquad 0\le k\le n_{edge}-2.$

对 padding 区保持常值。

插值得到

$P_{Table^{L(r)}},\ P_{Query^{L(r)}},\ P_{m_L^{(r)}},\ P_{Q_{tbl}^{L(r)}},\ P_{Q_{qry}^{L(r)}},\ P_{R_L^{(r)}}.$

定义组最大值

$M_i^{(r)}=\max\{Z_k^{(r)}\mid dst(k)=i\}.$

定义最大值广播与非负差分

$M_k^{edge(r)}=M_{dst(k)}^{(r)}, \qquad \Delta_k^{+(r)}=M_k^{edge(r)}-Z_k^{(r)}.$

引入指示变量

$s_{max}^{(r)}[k]\in\{0,1\}.$

要求：

1. 二值性

	$s_{max}^{(r)}[k]\big(s_{max}^{(r)}[k]-1\big)=0.$

2. 被选中的位置必须零差分

	$s_{max}^{(r)}[k]\cdot \Delta_k^{+(r)}=0.$

定义组内计数状态机

$C_{max}^{(r)}[0]=Q_{edge}^{valid}[0]\cdot s_{max}^{(r)}[0].$

对 $0\le k\le n_{edge}-2$，定义

$C_{max}^{(r)}[k+1] = (1-Q_{edge}^{valid}[k+1])C_{max}^{(r)}[k] + Q_{edge}^{valid}[k+1] \Big( Q_{new}^{edge}[k+1]s_{max}^{(r)}[k+1] + (1-Q_{new}^{edge}[k+1])\big(C_{max}^{(r)}[k]+s_{max}^{(r)}[k+1]\big) \Big).$

对每个边索引 $k$，要求组末约束

$Q_{end}^{edge}[k]\cdot\big(C_{max}^{(r)}[k]-1\big)=0.$

生成范围检查挑战

$\beta_R^{(r)}.$

定义范围表端

$Table^{R(r)}[t]=t,$

范围查询端

$Query^{R(r)}[k]=\Delta_k^{+(r)}.$

定义表端有效区选择器

$Q_{tbl}^{R(r)}[t]= \begin{cases} 1,&0\le t\le 2^B-1,\\ 0,&2^B\le t\le n_{edge}-1, \end{cases}$

查询端有效区选择器

$Q_{qry}^{R(r)}[k]=Q_{edge}^{valid}[k].$

对每个表端索引 $t$，定义重数

$m_R^{(r)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid \Delta_k^{+(r)}=t\}.$

定义累加器

$R_R^{(r)}[0]=0,$

$R_R^{(r)}[k+1] = R_R^{(r)}[k] + Q_{tbl}^{R(r)}[k]\cdot\frac{m_R^{(r)}[k]}{Table^{R(r)}[k]+\beta_R^{(r)}} - Q_{qry}^{R(r)}[k]\cdot\frac{1}{Query^{R(r)}[k]+\beta_R^{(r)}}, \qquad 0\le k\le n_{edge}-2.$

对 padding 区保持常值。

插值得到

$P_{Table^{R(r)}},\ P_{Query^{R(r)}},\ P_{m_R^{(r)}},\ P_{Q_{tbl}^{R(r)}},\ P_{Q_{qry}^{R(r)}},\ P_{R_R^{(r)}}.$

把

$S^{(r)},\ Z^{(r)},\ M^{(r)},\ M^{edge(r)},\ \Delta^{+(r)},\ s_{max}^{(r)},\ C_{max}^{(r)}$

分别插值成

$P_{S^{(r)}},\ P_{Z^{(r)}},\ P_{M^{(r)}},\ P_{M^{edge(r)}},\ P_{\Delta^{+(r)}},\ P_{s_{max}^{(r)}},\ P_{C_{max}^{(r)}}.$

提交承诺

$[P_{S^{(r)}}],\ [P_{Z^{(r)}}],\ [P_{M^{(r)}}],\ [P_{M^{edge(r)}}],\ [P_{\Delta^{+(r)}}],\ [P_{s_{max}^{(r)}}],\ [P_{C_{max}^{(r)}}],\ [P_{Table^{L(r)}}],\ [P_{Query^{L(r)}}],\ [P_{m_L^{(r)}}],\ [P_{Q_{tbl}^{L(r)}}],\ [P_{Q_{qry}^{L(r)}}],\ [P_{R_L^{(r)}}],\ [P_{Table^{R(r)}}],\ [P_{Query^{R(r)}}],\ [P_{m_R^{(r)}}],\ [P_{Q_{tbl}^{R(r)}}],\ [P_{Q_{qry}^{R(r)}}],\ [P_{R_R^{(r)}}].$

#### 2.2.7 指数、分母、逆元与归一化权重

生成挑战

$\eta_{exp}^{(r)},\beta_{exp}^{(r)}.$

对每个边索引 $k$，定义

$U_k^{(r)}=ExpMap(\Delta_k^{+(r)}).$

定义指数表端

$Table^{exp(r)}[t]=T_{exp}[t,0]+\eta_{exp}^{(r)}T_{exp}[t,1],$

查询端

$Query^{exp(r)}[k]=\Delta_k^{+(r)}+\eta_{exp}^{(r)}U_k^{(r)}.$

定义表端有效区选择器

$Q_{tbl}^{exp(r)}[t]= \begin{cases} 1,&0\le t\le |T_{exp}|-1,\\ 0,&|T_{exp}|\le t\le n_{edge}-1, \end{cases}$

查询端有效区选择器

$Q_{qry}^{exp(r)}[k]=Q_{edge}^{valid}[k].$

对每个表端索引 $t$，定义重数

$m_{exp}^{(r)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid (\Delta_k^{+(r)},U_k^{(r)})=(T_{exp}[t,0],T_{exp}[t,1])\}.$

定义累加器

$R_{exp}^{(r)}[0]=0,$

$R_{exp}^{(r)}[k+1] = R_{exp}^{(r)}[k] + Q_{tbl}^{exp(r)}[k]\cdot\frac{m_{exp}^{(r)}[k]}{Table^{exp(r)}[k]+\beta_{exp}^{(r)}} - Q_{qry}^{exp(r)}[k]\cdot\frac{1}{Query^{exp(r)}[k]+\beta_{exp}^{(r)}}, \qquad 0\le k\le n_{edge}-2.$

对 padding 区保持常值。

插值得到

$P_{Table^{exp(r)}},\ P_{Query^{exp(r)}},\ P_{m_{exp}^{(r)}},\ P_{Q_{tbl}^{exp(r)}},\ P_{Q_{qry}^{exp(r)}},\ P_{R_{exp}^{(r)}}.$

对每个节点 $i$，定义

$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)}, \qquad inv_i^{(r)}=(Sum_i^{(r)})^{-1}.$

对每个边索引 $k$，定义广播

$Sum_k^{edge(r)}=Sum_{dst(k)}^{(r)}, \qquad inv_k^{edge(r)}=inv_{dst(k)}^{(r)}.$

定义归一化权重

$\alpha_k^{(r)}=U_k^{(r)}\cdot inv_k^{edge(r)}.$

把

$U^{(r)},\ Sum^{(r)},\ Sum^{edge(r)},\ inv^{(r)},\ inv^{edge(r)},\alpha^{(r)}$

分别插值成

$P_{U^{(r)}},\ P_{Sum^{(r)}},\ P_{Sum^{edge(r)}},\ P_{inv^{(r)}},\ P_{inv^{edge(r)}},\ P_{\alpha^{(r)}}.$

提交承诺

$[P_{U^{(r)}}],\ [P_{Sum^{(r)}}],\ [P_{Sum^{edge(r)}}],\ [P_{inv^{(r)}}],\ [P_{inv^{edge(r)}}],\ [P_{\alpha^{(r)}}],\ [P_{Table^{exp(r)}}],\ [P_{Query^{exp(r)}}],\ [P_{m_{exp}^{(r)}}],\ [P_{Q_{tbl}^{exp(r)}}],\ [P_{Q_{qry}^{exp(r)}}],\ [P_{R_{exp}^{(r)}}].$

#### 2.2.8 聚合前隐藏矩阵、聚合前压缩特征与 PSQ

对每个节点 $i$ 与每个隐藏维索引 $j$，定义

$H_{agg,pre,i,j}^{(r)} = \sum_{\{k\mid dst(k)=i\}}\alpha_k^{(r)}H_{src(k),j}'^{(r)}.$

定义压缩特征

$H_{agg,pre,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,pre,i,j}^{(r)}(\xi^{(r)})^j.$

定义边级压缩加权特征

$\widehat v_{pre,k}^{\star(r)}=\alpha_k^{(r)}H_{src,k}^{\star,edge(r)}.$

定义广播

$H_{agg,pre,k}^{\star,edge(r)}=H_{agg,pre,dst(k)}^{\star(r)}$ 

在决定$U^{(r)},\ Sum^{(r)},\ Sum^{edge(r)},\ \widehat v_{pre}^{\star(r)},\ H_{agg,pre}^{\star(r)}$这些基础值的相关承诺对象全部固定之后，生成挑战$\lambda_{psq}^{(r)}$。

定义

$w_{psq}^{(r)}[k]=U_k^{(r)}+\lambda_{psq}^{(r)}\widehat v_{pre,k}^{\star(r)}.$

定义节点端目标值

$T_{psq}^{(r)}[i]=Sum_i^{(r)}+\lambda_{psq}^{(r)}H_{agg,pre,i}^{\star(r)}.$

定义边端广播目标值

$T_{psq}^{edge(r)}[k]=Sum_k^{edge(r)}+\lambda_{psq}^{(r)}H_{agg,pre,k}^{\star,edge(r)}.$



定义 PSQ 状态机

$PSQ^{(r)}[0]=Q_{edge}^{valid}[0]\cdot w_{psq}^{(r)}[0].$

对 $0\le k\le n_{edge}-2$，定义

$PSQ^{(r)}[k+1] = (1-Q_{edge}^{valid}[k+1])PSQ^{(r)}[k] + Q_{edge}^{valid}[k+1] \Big( Q_{new}^{edge}[k+1]w_{psq}^{(r)}[k+1] + (1-Q_{new}^{edge}[k+1])\big(PSQ^{(r)}[k]+w_{psq}^{(r)}[k+1]\big) \Big).$

对每个边索引 $k$，施加组末约束

$Q_{end}^{edge}[k]\cdot\big(PSQ^{(r)}[k]-T_{psq}^{edge(r)}[k]\big)=0.$

组末约束等价地同时强制了

$Sum_i^{(r)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(r)},$

$H_{agg,pre,i}^{\star(r)}=\sum_{\{k\mid dst(k)=i\}}\widehat v_{pre,k}^{\star(r)}.$

插值得到

$P_{H_{agg,pre}^{(r)}},\ P_{H_{agg,pre}^{\star(r)}},\ P_{\widehat v_{pre}^{\star(r)}},\ P_{w_{psq}^{(r)}},\ P_{T_{psq}^{(r)}},\ P_{T_{psq}^{edge(r)}},\ P_{PSQ^{(r)}}.$

生成聚合前压缩绑定挑战

$y_{agg,pre}^{(r)}=H_{FS}(\text{transcript},[P_{H_{agg,pre}^{(r)}}],[P_{H_{agg,pre}^{\star(r)}}]).$

要求

$P_{H_{agg,pre}^{\star(r)}}(y_{agg,pre}^{(r)}) = \sum_{j=0}^{d_h-1} \Big( \sum_{i=0}^{N-1}H_{agg,pre,i,j}^{(r)}L_i^{(N)}(y_{agg,pre}^{(r)}) \Big) (\xi^{(r)})^j.$

提交承诺

$[P_{H_{agg,pre}^{(r)}}],\ [P_{H_{agg,pre}^{\star(r)}}],\ [P_{\widehat v_{pre}^{\star(r)}}],\ [P_{w_{psq}^{(r)}}],\ [P_{T_{psq}^{(r)}}],\ [P_{T_{psq}^{edge(r)}}],\ [P_{PSQ^{(r)}}].$

#### 2.2.9 ELU

本节 ELU 查表复用边域$\mathbb H_{edge}$作为工作域；下文“ELU 工作域”均指$\mathbb H_{edge}$。

对每个节点 $i$ 与每个隐藏维索引 $j$，定义

$H_{agg,i,j}^{(r)}=ELU(H_{agg,pre,i,j}^{(r)}).$

将 $H_{agg,pre}^{(r)}$ 与 $H_{agg}^{(r)}$ 展平到 ELU 工作域，生成挑战

$\eta_{ELU}^{(r)},\beta_{ELU}^{(r)}.$

定义 ELU 表端

$Table^{ELU(r)}[t]=T_{ELU}[t,0]+\eta_{ELU}^{(r)}T_{ELU}[t,1].$

定义 ELU 查询端

$Query^{ELU(r)}[q]=H_{agg,pre}^{(r)}[q]+\eta_{ELU}^{(r)}H_{agg}^{(r)}[q].$

定义表端有效区选择器

$Q_{tbl}^{ELU(r)}[t]= \begin{cases} 1,&0\le t\le |T_{ELU}|-1,\\ 0,&|T_{ELU}|\le t\le n_{edge}-1, \end{cases}$

查询端有效区选择器

$Q_{qry}^{ELU(r)}[q]= \begin{cases} 1,&0\le q\le Nd_h-1,\\ 0,&Nd_h\le q\le n_{edge}-1. \end{cases}$

对每个表端索引 $t$，定义重数

$m_{ELU}^{(r)}[t] = \#\{q\in\{0,1,\ldots,Nd_h-1\}\mid (H_{agg,pre}^{(r)}[q],H_{agg}^{(r)}[q])=(T_{ELU}[t,0],T_{ELU}[t,1])\}.$

定义累加器

$R_{ELU}^{(r)}[0]=0,$

$R_{ELU}^{(r)}[q+1] = R_{ELU}^{(r)}[q] + Q_{tbl}^{ELU(r)}[q]\cdot\frac{m_{ELU}^{(r)}[q]}{Table^{ELU(r)}[q]+\beta_{ELU}^{(r)}} - Q_{qry}^{ELU(r)}[q]\cdot\frac{1}{Query^{ELU(r)}[q]+\beta_{ELU}^{(r)}}, \qquad 0\le q\le n_{edge}-2.$

插值得到

$P_{Table^{ELU(r)}},\ P_{Query^{ELU(r)}},\ P_{m_{ELU}^{(r)}},\ P_{Q_{tbl}^{ELU(r)}},\ P_{Q_{qry}^{ELU(r)}},\ P_{R_{ELU}^{(r)}}.$

定义聚合后压缩特征

$H_{agg,i}^{\star(r)}=\sum_{j=0}^{d_h-1}H_{agg,i,j}^{(r)}(\xi^{(r)})^j.$

定义边域广播

$H_{agg,k}^{\star,edge(r)}=H_{agg,dst(k)}^{\star(r)}.$

生成聚合后压缩绑定挑战

$y_{agg}^{(r)}=H_{FS}(\text{transcript},[P_{H_{agg}^{(r)}}],[P_{H_{agg}^{\star(r)}}]).$

要求

$P_{H_{agg}^{\star(r)}}(y_{agg}^{(r)}) = \sum_{j=0}^{d_h-1} \Big( \sum_{i=0}^{N-1}H_{agg,i,j}^{(r)}L_i^{(N)}(y_{agg}^{(r)}) \Big) (\xi^{(r)})^j.$

提交

$[P_{H_{agg}^{(r)}}],\ [P_{H_{agg}^{\star(r)}}],\ [P_{Table^{ELU(r)}}],\ [P_{Query^{ELU(r)}}],\ [P_{m_{ELU}^{(r)}}],\ [P_{R_{ELU}^{(r)}}].$

#### 2.2.10 目标路由的延迟定稿

目标路由需要同时绑定：

- 目标注意力；
- 组最大值；
- 分母；
- 逆元；
- ELU 后压缩聚合特征。

因此在

$P_{E_{dst}^{(r)}},\ P_{M^{(r)}},\ P_{Sum^{(r)}},\ P_{inv^{(r)}},\ P_{H_{agg}^{\star(r)}}$

及其边域广播对象全部固定之后，统一生成挑战

$\eta_{dst}^{(r)},\beta_{dst}^{(r)}.$

对每个节点 $i$，定义表端

$Table^{dst(r)}[i] = i +\eta_{dst}^{(r)}E_{dst,i}^{(r)} +(\eta_{dst}^{(r)})^2M_i^{(r)} +(\eta_{dst}^{(r)})^3Sum_i^{(r)} +(\eta_{dst}^{(r)})^4inv_i^{(r)} +(\eta_{dst}^{(r)})^5H_{agg,i}^{\star(r)}.$

对每个边索引 $k$，定义查询端

$Query^{dst(r)}[k] = dst(k) +\eta_{dst}^{(r)}E_{dst,k}^{edge(r)} +(\eta_{dst}^{(r)})^2M_k^{edge(r)} +(\eta_{dst}^{(r)})^3Sum_k^{edge(r)} +(\eta_{dst}^{(r)})^4inv_k^{edge(r)} +(\eta_{dst}^{(r)})^5H_{agg,k}^{\star,edge(r)}.$

定义节点重数

$m_{dst}^{(r)}[i]=\#\{k\mid dst(k)=i\}.$

定义公开总和

$S_{dst}^{(r)} = \sum_{i=0}^{N-1}\frac{m_{dst}^{(r)}[i]}{Table^{dst(r)}[i]+\beta_{dst}^{(r)}} = \sum_{k=0}^{E-1}\frac{1}{Query^{dst(r)}[k]+\beta_{dst}^{(r)}}.$

定义节点端累加器

$R_{dst}^{node(r)}[0]=0,$

$R_{dst}^{node(r)}[i+1] = R_{dst}^{node(r)}[i] + Q_N[i]\cdot\frac{m_{dst}^{(r)}[i]}{Table^{dst(r)}[i]+\beta_{dst}^{(r)}}.$

定义边端累加器

$R_{dst}^{edge(r)}[0]=0,$

$R_{dst}^{edge(r)}[k+1] = R_{dst}^{edge(r)}[k] + Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{dst(r)}[k]+\beta_{dst}^{(r)}}.$

插值得到

$P_{Table^{dst(r)}},\ P_{Query^{dst(r)}},\ P_{m_{dst}^{(r)}},\ P_{R_{dst}^{node(r)}},\ P_{R_{dst}^{edge(r)}}.$

提交承诺

$[P_{Table^{dst(r)}}],\ [P_{Query^{dst(r)}}],\ [P_{m_{dst}^{(r)}}],\ [P_{R_{dst}^{node(r)}}],\ [P_{R_{dst}^{edge(r)}}].$

### 2.3 拼接阶段

对每个节点 $i$、每个隐藏层注意力头 $r$、每个局部维索引 $j$，定义

$H_{cat,i,r\cdot d_h+j}=H_{agg,i,j}^{(r)}.$

定义拼接系数多项式

$P_{H_{cat}}(X)=\sum_{i=0}^{N-1}\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}X^{i\cdot d_{cat}+m}.$

生成拼接压缩挑战

$\xi_{cat}=H_{FS}(\text{transcript},[P_{H_{cat}}]).$

对每个节点 $i$，定义

$H_{cat,i}^{\star}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}\xi_{cat}^m.$

插值得到

$P_{H_{cat}^{\star}}(X)=\sum_{i=0}^{n_N-1}H_{cat}^{\star}[i]L_i^{(N)}(X).$

生成拼接绑定挑战

$y_{cat}=H_{FS}(\text{transcript},[P_{H_{cat}}],[P_{H_{cat}^{\star}}]).$

要求

$P_{H_{cat}^{\star}}(y_{cat}) = \sum_{m=0}^{d_{cat}-1} \Big( \sum_{i=0}^{N-1}H_{cat,i,m}L_i^{(N)}(y_{cat}) \Big)\xi_{cat}^m.$

提交

$[P_{H_{cat}}],\ [P_{H_{cat}^{\star}}].$

### 2.4 输出层的完整见证生成

#### 2.4.1 输出投影

对每个节点 $i$ 与类别索引 $c$，定义

$Y'_{i,c}=\sum_{m=0}^{d_{cat}-1}H_{cat,i,m}W_{m,c}^{(out)}.$

定义系数多项式

$P_{Y'}(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}Y'_{i,c}X^{i\cdot C+c}.$

按 CRPC 定义

$A_m^{proj(out)}(X)=\sum_{i=0}^{N-1}H_{cat,i,m}X^{i\cdot C},$

$B_m^{proj(out)}(X)=\sum_{c=0}^{C-1}W_{m,c}^{(out)}X^c.$

于是

$P_{Y'}(X)=\sum_{m=0}^{d_{cat}-1}A_m^{proj(out)}(X)B_m^{proj(out)}(X).$

生成挑战

$y_{proj}^{(out)}=H_{FS}(\text{transcript},[P_{H_{cat}}],[P_{Y'}],[V_{W^{(out)}}]).$

定义折叠向量

$a_m^{proj(out)}=A_m^{proj(out)}(y_{proj}^{(out)}), \qquad b_m^{proj(out)}=B_m^{proj(out)}(y_{proj}^{(out)}).$

定义外点评值

$\mu_{proj}^{(out)}=\sum_{m=0}^{d_{cat}-1}a_m^{proj(out)}b_m^{proj(out)}.$

要求

$P_{Y'}(y_{proj}^{(out)})=\mu_{proj}^{(out)}.$

定义输入共享维累加器

$Acc^{proj(out)}[0]=0,$

$Acc^{proj(out)}[m+1]=Acc^{proj(out)}[m]+a_m^{proj(out)}b_m^{proj(out)}, \qquad 0\le m\le d_{cat}-1.$

对 padding 区统一规定

$a_m^{proj(out)}=0,\qquad b_m^{proj(out)}=0,\qquad Acc^{proj(out)}[m+1]=Acc^{proj(out)}[m], \qquad d_{cat}\le m\le n_{cat}-1.$

定义

$P_{a^{proj(out)}}(X)=\sum_{m=0}^{n_{cat}-1}a_m^{proj(out)}L_m^{(cat)}(X),$

$P_{b^{proj(out)}}(X)=\sum_{m=0}^{n_{cat}-1}b_m^{proj(out)}L_m^{(cat)}(X),$

$P_{Acc^{proj(out)}}(X)=\sum_{m=0}^{n_{cat}-1}Acc^{proj(out)}[m]L_m^{(cat)}(X).$

提交承诺

$[P_{Y'}],\ [P_{a^{proj(out)}}],\ [P_{b^{proj(out)}}],\ [P_{Acc^{proj(out)}}].$

#### 2.4.2 输出层源注意力绑定

对每个节点 $i$，定义

$E_{src,i}^{(out)}=\sum_{c=0}^{C-1}Y'_{i,c}a_{src,c}^{(out)}.$

插值成

$P_{E_{src}^{(out)}}(X)=\sum_{i=0}^{n_N-1}E_{src}^{(out)}[i]L_i^{(N)}(X).$

生成挑战

$y_{src}^{(out)}=H_{FS}(\text{transcript},[P_{Y'}],[P_{E_{src}^{(out)}}],[V_{a_{src}^{(out)}}]).$

定义类别共享维折叠向量

$a_c^{src(out)}=\sum_{i=0}^{N-1}Y'_{i,c}L_i^{(N)}(y_{src}^{(out)}), \qquad b_c^{src(out)}=a_{src,c}^{(out)}.$

定义外点评值

$\mu_{src}^{(out)}=\sum_{c=0}^{C-1}a_c^{src(out)}b_c^{src(out)}.$

要求

$P_{E_{src}^{(out)}}(y_{src}^{(out)})=\mu_{src}^{(out)}.$

定义类别共享维累加器

$Acc_{src}^{(out)}[0]=0,$

$Acc_{src}^{(out)}[c+1] = Acc_{src}^{(out)}[c]+a_c^{src(out)}b_c^{src(out)}, \qquad 0\le c\le C-1.$

对 padding 区统一规定

$a_c^{src(out)}=0,\qquad b_c^{src(out)}=0,\qquad Acc_{src}^{(out)}[c+1]=Acc_{src}^{(out)}[c], \qquad C\le c\le n_C-1.$

定义

$P_{a^{src(out)}}(X)=\sum_{c=0}^{n_C-1}a_c^{src(out)}L_c^{(C)}(X),$

$P_{b^{src(out)}}(X)=\sum_{c=0}^{n_C-1}b_c^{src(out)}L_c^{(C)}(X),$

$P_{Acc^{src(out)}}(X)=\sum_{c=0}^{n_C-1}Acc_{src}^{(out)}[c]L_c^{(C)}(X).$

提交承诺

$[P_{E_{src}^{(out)}}],\ [P_{a^{src(out)}}],\ [P_{b^{src(out)}}],\ [P_{Acc^{src(out)}}].$

#### 2.4.3 输出层目标注意力绑定

对每个节点 $i$，定义

$E_{dst,i}^{(out)}=\sum_{c=0}^{C-1}Y'_{i,c}a_{dst,c}^{(out)}.$

插值成

$P_{E_{dst}^{(out)}}(X)=\sum_{i=0}^{n_N-1}E_{dst}^{(out)}[i]L_i^{(N)}(X).$

生成挑战

$y_{dst}^{(out)}=H_{FS}(\text{transcript},[P_{Y'}],[P_{E_{dst}^{(out)}}],[V_{a_{dst}^{(out)}}]).$

定义折叠向量

$a_c^{dst(out)}=\sum_{i=0}^{N-1}Y'_{i,c}L_i^{(N)}(y_{dst}^{(out)}), \qquad b_c^{dst(out)}=a_{dst,c}^{(out)}.$

定义外点评值

$\mu_{dst}^{(out)}=\sum_{c=0}^{C-1}a_c^{dst(out)}b_c^{dst(out)}.$

要求

$P_{E_{dst}^{(out)}}(y_{dst}^{(out)})=\mu_{dst}^{(out)}.$

定义类别共享维累加器

$Acc_{dst}^{(out)}[0]=0,$

$Acc_{dst}^{(out)}[c+1] = Acc_{dst}^{(out)}[c]+a_c^{dst(out)}b_c^{dst(out)}, \qquad 0\le c\le C-1.$

对 padding 区统一规定

$a_c^{dst(out)}=0,\qquad b_c^{dst(out)}=0,\qquad Acc_{dst}^{(out)}[c+1]=Acc_{dst}^{(out)}[c], \qquad C\le c\le n_C-1.$

定义

$P_{a^{dst(out)}}(X)=\sum_{c=0}^{n_C-1}a_c^{dst(out)}L_c^{(C)}(X),$

$P_{b^{dst(out)}}(X)=\sum_{c=0}^{n_C-1}b_c^{dst(out)}L_c^{(C)}(X),$

$P_{Acc^{dst(out)}}(X)=\sum_{c=0}^{n_C-1}Acc_{dst}^{(out)}[c]L_c^{(C)}(X).$

提交承诺

$[P_{E_{dst}^{(out)}}],\ [P_{a^{dst(out)}}],\ [P_{b^{dst(out)}}],\ [P_{Acc^{dst(out)}}].$

#### 2.4.4 输出层源路由

对每个边索引 $k$，定义

$E_{src,k}^{edge(out)}=E_{src,src(k)}^{(out)}.$

生成挑战

$\eta_{src}^{(out)},\beta_{src}^{(out)}.$

对每个节点 $i$，定义表端

$Table^{src(out)}[i]=i+\eta_{src}^{(out)}E_{src,i}^{(out)}.$

定义重数

$m_{src}^{(out)}[i]=\#\{k\mid src(k)=i\}.$

对每个边索引 $k$，定义查询端

$Query^{src(out)}[k]=src(k)+\eta_{src}^{(out)}E_{src,k}^{edge(out)}.$

定义公开总和

$S_{src}^{(out)} = \sum_{i=0}^{N-1}\frac{m_{src}^{(out)}[i]}{Table^{src(out)}[i]+\beta_{src}^{(out)}} = \sum_{k=0}^{E-1}\frac{1}{Query^{src(out)}[k]+\beta_{src}^{(out)}}.$

定义节点端累加器

$R_{src}^{node(out)}[0]=0,$

$R_{src}^{node(out)}[i+1] = R_{src}^{node(out)}[i] + Q_N[i]\cdot\frac{m_{src}^{(out)}[i]}{Table^{src(out)}[i]+\beta_{src}^{(out)}}, \qquad 0\le i\le n_N-2.$

定义边端累加器

$R_{src}^{edge(out)}[0]=0,$

$R_{src}^{edge(out)}[k+1] = R_{src}^{edge(out)}[k] + Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{src(out)}[k]+\beta_{src}^{(out)}}, \qquad 0\le k\le n_{edge}-2.$

插值得到

$P_{Table^{src(out)}},\ P_{Query^{src(out)}},\ P_{m_{src}^{(out)}},\ P_{R_{src}^{node(out)}},\ P_{R_{src}^{edge(out)}}.$

提交承诺

$[P_{Table^{src(out)}}],\ [P_{Query^{src(out)}}],\ [P_{m_{src}^{(out)}}],\ [P_{R_{src}^{node(out)}}],\ [P_{R_{src}^{edge(out)}}].$

#### 2.4.5 输出层 LeakyReLU、最大值、范围检查、指数、分母、逆元与归一化权重

对每个边索引 $k$，定义

$E_{dst,k}^{edge(out)}=E_{dst,dst(k)}^{(out)},$

$S_k^{(out)}=E_{src,k}^{edge(out)}+E_{dst,k}^{edge(out)},$

$Z_k^{(out)}=LReLU(S_k^{(out)}).$

生成挑战

$\eta_L^{(out)},\beta_L^{(out)}.$

定义 LeakyReLU 表端与查询端

$Table^{L(out)}[t]=T_{LReLU}[t,0]+\eta_L^{(out)}T_{LReLU}[t,1],$

$Query^{L(out)}[k]=S_k^{(out)}+\eta_L^{(out)}Z_k^{(out)}.$

定义表端有效区选择器

$Q_{tbl}^{L(out)}[t]= \begin{cases} 1,&0\le t\le |T_{LReLU}|-1,\\ 0,&|T_{LReLU}|\le t\le n_{edge}-1, \end{cases}$

查询端有效区选择器

$Q_{qry}^{L(out)}[k]=Q_{edge}^{valid}[k].$

对每个表端索引 $t$，定义重数

$m_L^{(out)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid (S_k^{(out)},Z_k^{(out)})=(T_{LReLU}[t,0],T_{LReLU}[t,1])\}.$

定义累加器

$R_L^{(out)}[0]=0,$

$R_L^{(out)}[k+1] = R_L^{(out)}[k] + Q_{tbl}^{L(out)}[k]\cdot\frac{m_L^{(out)}[k]}{Table^{L(out)}[k]+\beta_L^{(out)}} - Q_{qry}^{L(out)}[k]\cdot\frac{1}{Query^{L(out)}[k]+\beta_L^{(out)}}, \qquad 0\le k\le n_{edge}-2.$

插值得到

$P_{Table^{L(out)}},\ P_{Query^{L(out)}},\ P_{m_L^{(out)}},\ P_{Q_{tbl}^{L(out)}},\ P_{Q_{qry}^{L(out)}},\ P_{R_L^{(out)}}.$

定义组最大值与差分

$M_i^{(out)}=\max\{Z_k^{(out)}\mid dst(k)=i\},$

$M_k^{edge(out)}=M_{dst(k)}^{(out)},$

$\Delta_k^{+(out)}=M_k^{edge(out)}-Z_k^{(out)}.$

引入

$s_{max}^{(out)}[k]\in\{0,1\}, \qquad C_{max}^{(out)}[k]$

并使用与隐藏层相同的二值性、零差分、组内计数与组末唯一性约束。

生成范围检查挑战

$\beta_R^{(out)}.$

定义范围表端与查询端

$Table^{R(out)}[t]=t, \qquad Query^{R(out)}[k]=\Delta_k^{+(out)}.$

定义表端有效区选择器

$Q_{tbl}^{R(out)}[t]= \begin{cases} 1,&0\le t\le 2^B-1,\\ 0,&2^B\le t\le n_{edge}-1, \end{cases}$

查询端有效区选择器

$Q_{qry}^{R(out)}[k]=Q_{edge}^{valid}[k].$

对每个表端索引 $t$，定义重数

$m_R^{(out)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid \Delta_k^{+(out)}=t\}.$

定义累加器

$R_R^{(out)}[0]=0,$

$R_R^{(out)}[k+1] = R_R^{(out)}[k] + Q_{tbl}^{R(out)}[k]\cdot\frac{m_R^{(out)}[k]}{Table^{R(out)}[k]+\beta_R^{(out)}} - Q_{qry}^{R(out)}[k]\cdot\frac{1}{Query^{R(out)}[k]+\beta_R^{(out)}}, \qquad 0\le k\le n_{edge}-2.$

插值得到

$P_{Table^{R(out)}},\ P_{Query^{R(out)}},\ P_{m_R^{(out)}},\ P_{Q_{tbl}^{R(out)}},\ P_{Q_{qry}^{R(out)}},\ P_{R_R^{(out)}}.$

生成指数挑战

$\eta_{exp}^{(out)},\beta_{exp}^{(out)}.$

定义

$U_k^{(out)}=ExpMap(\Delta_k^{+(out)}),$

$Table^{exp(out)}[t]=T_{exp}[t,0]+\eta_{exp}^{(out)}T_{exp}[t,1],$

$Query^{exp(out)}[k]=\Delta_k^{+(out)}+\eta_{exp}^{(out)}U_k^{(out)}.$

定义表端有效区选择器

$Q_{tbl}^{exp(out)}[t]= \begin{cases} 1,&0\le t\le |T_{exp}|-1,\\ 0,&|T_{exp}|\le t\le n_{edge}-1, \end{cases}$

查询端有效区选择器

$Q_{qry}^{exp(out)}[k]=Q_{edge}^{valid}[k].$

对每个表端索引 $t$，定义重数

$m_{exp}^{(out)}[t] = \#\{k\in\{0,1,\ldots,E-1\}\mid (\Delta_k^{+(out)},U_k^{(out)})=(T_{exp}[t,0],T_{exp}[t,1])\}.$

定义累加器

$R_{exp}^{(out)}[0]=0,$

$R_{exp}^{(out)}[k+1] = R_{exp}^{(out)}[k] + Q_{tbl}^{exp(out)}[k]\cdot\frac{m_{exp}^{(out)}[k]}{Table^{exp(out)}[k]+\beta_{exp}^{(out)}} - Q_{qry}^{exp(out)}[k]\cdot\frac{1}{Query^{exp(out)}[k]+\beta_{exp}^{(out)}}, \qquad 0\le k\le n_{edge}-2.$

插值得到

$P_{Table^{exp(out)}},\ P_{Query^{exp(out)}},\ P_{m_{exp}^{(out)}},\ P_{Q_{tbl}^{exp(out)}},\ P_{Q_{qry}^{exp(out)}},\ P_{R_{exp}^{(out)}}.$

对每个节点 $i$，定义

$Sum_i^{(out)}=\sum_{\{k\mid dst(k)=i\}}U_k^{(out)}, \qquad inv_i^{(out)}=(Sum_i^{(out)})^{-1}.$

对每个边索引 $k$，定义

$Sum_k^{edge(out)}=Sum_{dst(k)}^{(out)}, \qquad inv_k^{edge(out)}=inv_{dst(k)}^{(out)}, \qquad \alpha_k^{(out)}=U_k^{(out)}\cdot inv_k^{edge(out)}.$

把

$S^{(out)},\ Z^{(out)},\ M^{(out)},\ M^{edge(out)},\ \Delta^{+(out)},\ U^{(out)},\ Sum^{(out)},\ Sum^{edge(out)},\ inv^{(out)},\ inv^{edge(out)},\alpha^{(out)}$

分别插值成

$P_{S^{(out)}},\ P_{Z^{(out)}},\ P_{M^{(out)}},\ P_{M^{edge(out)}},\ P_{\Delta^{+(out)}},\ P_{U^{(out)}},\ P_{Sum^{(out)}},\ P_{Sum^{edge(out)}},\ P_{inv^{(out)}},\ P_{inv^{edge(out)}},\ P_{\alpha^{(out)}}.$

同时把

$s_{max}^{(out)},\ C_{max}^{(out)},\ Table^{L(out)},\ Query^{L(out)},\ m_L^{(out)},\ Q_{tbl}^{L(out)},\ Q_{qry}^{L(out)},\ R_L^{(out)},$

$Table^{R(out)},\ Query^{R(out)},\ m_R^{(out)},\ Q_{tbl}^{R(out)},\ Q_{qry}^{R(out)},\ R_R^{(out)},$

$Table^{exp(out)},\ Query^{exp(out)},\ m_{exp}^{(out)},\ Q_{tbl}^{exp(out)},\ Q_{qry}^{exp(out)},\ R_{exp}^{(out)}$

分别插值成对应多项式。

提交承诺

$[P_{S^{(out)}}],\ [P_{Z^{(out)}}],\ [P_{M^{(out)}}],\ [P_{M^{edge(out)}}],\ [P_{\Delta^{+(out)}}],\ [P_{U^{(out)}}],\ [P_{Sum^{(out)}}],\ [P_{Sum^{edge(out)}}],\ [P_{inv^{(out)}}],\ [P_{inv^{edge(out)}}],\ [P_{\alpha^{(out)}}],$

$[P_{s_{max}^{(out)}}],\ [P_{C_{max}^{(out)}}],\ [P_{Table^{L(out)}}],\ [P_{Query^{L(out)}}],\ [P_{m_L^{(out)}}],\ [P_{Q_{tbl}^{L(out)}}],\ [P_{Q_{qry}^{L(out)}}],\ [P_{R_L^{(out)}}],$

$[P_{Table^{R(out)}}],\ [P_{Query^{R(out)}}],\ [P_{m_R^{(out)}}],\ [P_{Q_{tbl}^{R(out)}}],\ [P_{Q_{qry}^{R(out)}}],\ [P_{R_R^{(out)}}],$

$[P_{Table^{exp(out)}}],\ [P_{Query^{exp(out)}}],\ [P_{m_{exp}^{(out)}}],\ [P_{Q_{tbl}^{exp(out)}}],\ [P_{Q_{qry}^{exp(out)}}],\ [P_{R_{exp}^{(out)}}].$

#### 2.4.6 输出层最终聚合、压缩输出与 PSQ

生成输出类别压缩挑战

$\xi_{out}=H_{FS}(\text{transcript},[P_{Y'}]).$

对每个节点 $i$，定义

$Y_i'^{\star}=\sum_{c=0}^{C-1}Y'_{i,c}\xi_{out}^c.$

对每个边索引 $k$，定义

$Y_k'^{\star,edge}=Y_{src(k)}'^{\star}.$

对每个边索引 $k$，定义压缩加权边特征

$\widehat y_k^{\star}=\alpha_k^{(out)}Y_k'^{\star,edge}.$

对每个节点 $i$ 与类别索引 $c$，定义最终输出

$Y_{i,c}=\sum_{\{k\mid dst(k)=i\}}\alpha_k^{(out)}Y'_{src(k),c}.$

于是压缩输出满足

$Y_i^{\star}=\sum_{c=0}^{C-1}Y_{i,c}\xi_{out}^c.$

对每个边索引 $k$，定义广播

$Y_k^{\star,edge}=Y_{dst(k)}^{\star}.$

生成 PSQ 挑战

$\lambda_{out}.$

定义边级权值

$w_{out}[k]=U_k^{(out)}+\lambda_{out}\widehat y_k^{\star}.$

定义节点目标值

$T_{out}[i]=Sum_i^{(out)}+\lambda_{out}Y_i^{\star}.$

定义边域广播目标值

$T_{out}^{edge}[k]=Sum_k^{edge(out)}+\lambda_{out}Y_k^{\star,edge}.$

定义输出聚合状态机

$PSQ^{(out)}[0]=Q_{edge}^{valid}[0]\cdot w_{out}[0].$

对 $0\le k\le n_{edge}-2$，定义

$PSQ^{(out)}[k+1] = (1-Q_{edge}^{valid}[k+1])PSQ^{(out)}[k] + Q_{edge}^{valid}[k+1] \Big( Q_{new}^{edge}[k+1]w_{out}[k+1] + (1-Q_{new}^{edge}[k+1])\big(PSQ^{(out)}[k]+w_{out}[k+1]\big) \Big).$

对每个边索引 $k$，组末约束为

$Q_{end}^{edge}[k]\cdot\big(PSQ^{(out)}[k]-T_{out}^{edge}[k]\big)=0.$

这样强制了

$Y_i^{\star}=\sum_{\{k\mid dst(k)=i\}}\widehat y_k^{\star}.$

插值得到

$P_{Y'^{\star}}(X)=\sum_{i=0}^{n_N-1}Y'^{\star}[i]L_i^{(N)}(X),$

$P_{Y'^{\star,edge}}(X)=\sum_{k=0}^{n_{edge}-1}Y'^{\star,edge}[k]L_k^{(edge)}(X),$

$P_{\widehat y^{\star}}(X)=\sum_{k=0}^{n_{edge}-1}\widehat y^{\star}[k]L_k^{(edge)}(X),$

$P_{w_{out}}(X)=\sum_{k=0}^{n_{edge}-1}w_{out}[k]L_k^{(edge)}(X),$

$P_{T_{out}}(X)=\sum_{i=0}^{n_N-1}T_{out}[i]L_i^{(N)}(X),$

$P_{T_{out}^{edge}}(X)=\sum_{k=0}^{n_{edge}-1}T_{out}^{edge}[k]L_k^{(edge)}(X),$

$P_{PSQ^{(out)}}(X)=\sum_{k=0}^{n_{edge}-1}PSQ^{(out)}[k]L_k^{(edge)}(X).$

定义最终输出系数多项式

$P_Y(X)=\sum_{i=0}^{N-1}\sum_{c=0}^{C-1}Y_{i,c}X^{i\cdot C+c}.$

定义压缩输出多项式

$P_{Y^{\star}}(X)=\sum_{i=0}^{n_N-1}Y^{\star}[i]L_i^{(N)}(X).$

定义压缩输出广播多项式

$P_{Y^{\star,edge}}(X)=\sum_{k=0}^{n_{edge}-1}Y^{\star,edge}[k]L_k^{(edge)}(X).$

生成输出压缩绑定挑战

$y_{out}^{\star}=H_{FS}(\text{transcript},[P_Y],[P_{Y^{\star}}]).$

要求

$P_{Y^{\star}}(y_{out}^{\star}) = \sum_{c=0}^{C-1} \Big( \sum_{i=0}^{N-1}Y_{i,c}L_i^{(N)}(y_{out}^{\star}) \Big)\xi_{out}^c$ 

提交输出层的全部动态承诺：

$[P_{Y'}],\ [P_{E_{src}^{(out)}}],\ [P_{E_{dst}^{(out)}}],\ [P_{a^{src(out)}}],\ [P_{b^{src(out)}}],\ [P_{Acc^{src(out)}}],\ [P_{a^{dst(out)}}],\ [P_{b^{dst(out)}}],\ [P_{Acc^{dst(out)}}],$

$[P_{Table^{src(out)}}],\ [P_{Query^{src(out)}}],\ [P_{m_{src}^{(out)}}],\ [P_{R_{src}^{node(out)}}],\ [P_{R_{src}^{edge(out)}}],$

$[P_{S^{(out)}}],\ [P_{Z^{(out)}}],\ [P_{M^{(out)}}],\ [P_{M^{edge(out)}}],\ [P_{\Delta^{+(out)}}],\ [P_{s_{max}^{(out)}}],\ [P_{C_{max}^{(out)}}],$

$[P_{Table^{L(out)}}],\ [P_{Query^{L(out)}}],\ [P_{m_L^{(out)}}],\ [P_{Q_{tbl}^{L(out)}}],\ [P_{Q_{qry}^{L(out)}}],\ [P_{R_L^{(out)}}],$

$[P_{Table^{R(out)}}],\ [P_{Query^{R(out)}}],\ [P_{m_R^{(out)}}],\ [P_{Q_{tbl}^{R(out)}}],\ [P_{Q_{qry}^{R(out)}}],\ [P_{R_R^{(out)}}],$

$[P_{U^{(out)}}],\ [P_{Sum^{(out)}}],\ [P_{Sum^{edge(out)}}],\ [P_{inv^{(out)}}],\ [P_{inv^{edge(out)}}],\ [P_{\alpha^{(out)}}],$

$[P_{Table^{exp(out)}}],\ [P_{Query^{exp(out)}}],\ [P_{m_{exp}^{(out)}}],\ [P_{Q_{tbl}^{exp(out)}}],\ [P_{Q_{qry}^{exp(out)}}],\ [P_{R_{exp}^{(out)}}],$

$[P_{Y'^{\star}}],\ [P_{Y'^{\star,edge}}],\ [P_{\widehat y^{\star}}],\ [P_Y],\ [P_{Y^{\star}}],\ [P_{Y^{\star,edge}}],\ [P_{w_{out}}],\ [P_{T_{out}}],\ [P_{T_{out}^{edge}}],\ [P_{PSQ^{(out)}}].$

#### 2.4.7 输出层目标路由

在输出层中，目标路由需要绑定：

- 目标注意力；
- 组最大值；
- 分母；
- 逆元；
- 压缩最终输出。

因此在

$P_{E_{dst}^{(out)}},\ P_{M^{(out)}},\ P_{Sum^{(out)}},\ P_{inv^{(out)}},\ P_{Y^{\star}}$

及其边域广播对象全部固定之后，统一生成挑战

$\eta_{dst}^{(out)},\beta_{dst}^{(out)}.$

对每个节点 $i$，定义表端

$Table^{dst(out)}[i] = i +\eta_{dst}^{(out)}E_{dst,i}^{(out)} +(\eta_{dst}^{(out)})^2M_i^{(out)} +(\eta_{dst}^{(out)})^3Sum_i^{(out)} +(\eta_{dst}^{(out)})^4inv_i^{(out)} +(\eta_{dst}^{(out)})^5Y_i^{\star}.$

对每个边索引 $k$，定义查询端

$Query^{dst(out)}[k] = dst(k) +\eta_{dst}^{(out)}E_{dst,k}^{edge(out)} +(\eta_{dst}^{(out)})^2M_k^{edge(out)} +(\eta_{dst}^{(out)})^3Sum_k^{edge(out)} +(\eta_{dst}^{(out)})^4inv_k^{edge(out)} +(\eta_{dst}^{(out)})^5Y_k^{\star,edge}.$

定义重数

$m_{dst}^{(out)}[i]=\#\{k\mid dst(k)=i\}.$

定义公开总和

$S_{dst}^{(out)} = \sum_{i=0}^{N-1}\frac{m_{dst}^{(out)}[i]}{Table^{dst(out)}[i]+\beta_{dst}^{(out)}} = \sum_{k=0}^{E-1}\frac{1}{Query^{dst(out)}[k]+\beta_{dst}^{(out)}}.$

定义节点端累加器

$R_{dst}^{node(out)}[0]=0,$

$R_{dst}^{node(out)}[i+1] = R_{dst}^{node(out)}[i] + Q_N[i]\cdot\frac{m_{dst}^{(out)}[i]}{Table^{dst(out)}[i]+\beta_{dst}^{(out)}}, \qquad 0\le i\le n_N-2.$

定义边端累加器

$R_{dst}^{edge(out)}[0]=0,$

$R_{dst}^{edge(out)}[k+1] = R_{dst}^{edge(out)}[k] + Q_{edge}^{valid}[k]\cdot\frac{1}{Query^{dst(out)}[k]+\beta_{dst}^{(out)}}, \qquad 0\le k\le n_{edge}-2.$

插值得到

$P_{Table^{dst(out)}},\ P_{Query^{dst(out)}},\ P_{m_{dst}^{(out)}},\ P_{R_{dst}^{node(out)}},\ P_{R_{dst}^{edge(out)}}$

提交承诺

$[P_{Table^{dst(out)}}],\ [P_{Query^{dst(out)}}],\ [P_{m_{dst}^{(out)}}],\ [P_{R_{dst}^{node(out)}}],\ [P_{R_{dst}^{edge(out)}}]$

在上述输出层目标路由相关对象全部固定之后，再生成输出外点挑战

$y_{out}=H_{FS}(\text{transcript},[P_{Y'}],[P_Y],[P_{Y^{\star}}],[P_{{Table}^{dst(out)}}],[P_{Query^{dst(out)}}])$ 

## 3. 证明生成

### 3.1 输入与输出

证明算法输入是：

- 参数生成输出

	$(PK,VK_{KZG},VK_{static},VK_{model}).$

- 公共输入

	$(I,src,dst,N,E,d_{in},d_h,C,B).$

- 全部动态见证对象。

输出最终证明

$\pi_{GAT}.$

### 3.2 Fiat–Shamir 挑战顺序

验证者与证明者必须严格按以下顺序重放。

#### 3.2.1 全局阶段

1. 先从证明对象

	$\pi_{GAT}$

	中按第 3.5 节规定的固定顺序解析元数据块

	$M_{pub},$

	并把

	$M_{pub}$

	与全部公开量化参数一起吸入 transcript。随后再吸入

	$N,E,d_{in},d_h,d_{cat},C,B$

	以及

	$[P_I],[P_{src}],[P_{dst}],[P_{Q_{new}^{edge}}],[P_{Q_{end}^{edge}}],$

	再吸入基础动态承诺 $[P_H]$ 与静态特征表承诺，生成

	$\eta_{feat},\beta_{feat}.$

	这里先吸入 $M_{pub}$，后再次显式吸入

	$N,E,d_{in},d_h,d_{cat},C,B$

	是协议定义中的**有意双重绑定**，而不是可删的书写重复。实现时必须严格按这一顺序重放 transcript，禁止把这些显式公共输入仅仅视为 $M_{pub}.dim\_cfg$ 的冗余展开而省略。

2. 对每个隐藏层注意力头 $r=0,1,\ldots,7$，按顺序生成：

	$y_{proj}^{(r)},$

	$\xi^{(r)},$

	$y_{src}^{(r)},$

	$y_{dst}^{(r)},$

	$y_{\star}^{(r)},$

	$\eta_{src}^{(r)},\beta_{src}^{(r)},$

	$\eta_L^{(r)},\beta_L^{(r)},$

	$\beta_R^{(r)},$

	$\eta_{exp}^{(r)},\beta_{exp}^{(r)},$

	$\lambda_{psq}^{(r)},$

	$y_{agg,pre}^{(r)},$

	$\eta_{ELU}^{(r)},\beta_{ELU}^{(r)},$

	$y_{agg}^{(r)},$

	$\eta_{dst}^{(r)},\beta_{dst}^{(r)}.$

3. 拼接阶段严格分两步执行：

	(a) 先吸入 $[P_{H_{cat}}]$，生成

	$\xi_{cat}.$

	(b) 再吸入 $[P_{H_{cat}^{\star}}]$，生成

	$y_{cat}.$

4. 输出层阶段依次生成：

	$y_{proj}^{(out)},$

	$y_{src}^{(out)},$

	$y_{dst}^{(out)},$

	$\eta_{src}^{(out)},\beta_{src}^{(out)},$

	$\eta_L^{(out)},\beta_L^{(out)},$

	$\beta_R^{(out)},$

	$\eta_{exp}^{(out)},\beta_{exp}^{(out)},$

	$\xi_{out},$

	$\lambda_{out},$

	$y_{out}^{\star},$

	$\eta_{dst}^{(out)},\beta_{dst}^{(out)},$

	$y_{out}.$

5. 吸入全部动态承诺与全部静态承诺，生成 quotient 聚合挑战

	$\alpha_{quot}.$

6. 生成工作域开放点

	$z_{FH},z_{edge},z_{in},z_{d_h},z_{cat},z_C,z_N.$

7. 生成各工作域 batch opening 折叠挑战

	$v_{FH},v_{edge},v_{in},v_{d_h},v_{cat},v_C,v_N.$

8. 生成外点评值批量折叠挑战

	$\rho_{ext}.$

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

1. 起点约束

	$C_{lookup,0}(X)=First_{\mathcal D}(X)\cdot P_R(X).$

2. 递推约束

	$\begin{aligned} C_{lookup,1}(X) ={}& \big(P_R(\omega_{\mathcal D}X)-P_R(X)\big) \big(P_{Table}(X)+\beta\big) \big(P_{Query}(X)+\beta\big)\\ &- P_{Q_{tbl}}(X)P_m(X)\big(P_{Query}(X)+\beta\big)\\ &+ P_{Q_{qry}}(X)\big(P_{Table}(X)+\beta\big). \end{aligned}$

3. 终点约束

	$C_{lookup,2}(X)=Last_{\mathcal D}(X)\cdot P_R(X).$

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

1. 起点约束

	$C_{route,node,0}(X)=First_N(X)\cdot P_{R^{node}}(X).$

2. 递推约束

	$C_{route,node,1}(X) = \big(P_{R^{node}}(\omega_NX)-P_{R^{node}}(X)\big)\big(P_{Table}(X)+\beta\big) - P_{Q_N}(X)P_m(X).$

3. 终点约束

	$C_{route,node,2}(X)=Last_N(X)\cdot\big(P_{R^{node}}(X)-S\big).$

#### 3.3.3 通用边路由模板

设某个边端路由子系统定义在 $\mathbb H_{edge}$ 上，含有：

- 查询多项式 $P_{Query}(X)$；
- 边端累加器 $P_{R^{edge}}(X)$；
- 边域有效区选择器 $P_{Q_{edge}^{valid}}(X)$；
- 公共总和 $S$；
- 挑战 $\beta$。

则三条主约束为：

1. 起点约束

	$C_{route,edge,0}(X)=First_{edge}(X)\cdot P_{R^{edge}}(X).$

2. 递推约束

	$C_{route,edge,1}(X) = \big(P_{R^{edge}}(\omega_{edge}X)-P_{R^{edge}}(X)\big)\big(P_{Query}(X)+\beta\big) - P_{Q_{edge}^{valid}}(X).$

3. 终点约束

	$C_{route,edge,2}(X)=Last_{edge}(X)\cdot\big(P_{R^{edge}}(X)-S\big).$

#### 3.3.4 最大值唯一性约束

设包含：

- 二值指示多项式 $P_s(X)$；
- 差分多项式 $P_{\Delta}(X)$；
- 计数状态机 $P_C(X)$；
- 组起点选择器 $P_{Q_{new}^{edge}}(X)$；
- 组末尾选择器 $P_{Q_{end}^{edge}}(X)$；
- 边域有效区选择器 $P_{Q_{edge}^{valid}}(X)$。

则约束为：

1. 二值性

	$C_{max,bin}(X)=P_{Q_{edge}^{valid}}(X)\cdot P_s(X)\big(P_s(X)-1\big).$

2. 被选中位置零差分

	$C_{max,zero}(X)=P_{Q_{edge}^{valid}}(X)\cdot P_s(X)\cdot P_{\Delta}(X).$

3. 起点计数

	$C_{max,0}(X)=First_{edge}(X)\cdot\big(P_C(X)-P_{Q_{edge}^{valid}}(X)P_s(X)\big).$

4. 递推

	$\begin{aligned} C_{max,1}(X) ={}& P_C(\omega_{edge}X) - \Big( \big(1-P_{Q_{edge}^{valid}}(\omega_{edge}X)\big)P_C(X)\\ &\qquad\qquad+ P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{new}^{edge}}(\omega_{edge}X)P_s(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ \big(1-P_{Q_{new}^{edge}}(\omega_{edge}X)\big)\big(P_C(X)+P_s(\omega_{edge}X)\big) \big) \Big). \end{aligned}$

5. 组末唯一性

	$C_{max,end}(X)=P_{Q_{end}^{edge}}(X)\cdot\big(P_C(X)-1\big).$

#### 3.3.5 PSQ 状态机约束

设包含：

- 边级权值多项式 $P_w(X)$；
- 边级目标值多项式 $P_{T^{edge}}(X)$；
- 状态机多项式 $P_{PSQ}(X)$；
- 组起点选择器 $P_{Q_{new}^{edge}}(X)$；
- 组末尾选择器 $P_{Q_{end}^{edge}}(X)$；
- 边域有效区选择器 $P_{Q_{edge}^{valid}}(X)$。

则约束为：

1. 起点约束

	$C_{psq,0}(X)=First_{edge}(X)\cdot\big(P_{PSQ}(X)-P_{Q_{edge}^{valid}}(X)P_w(X)\big).$

2. 递推约束

	$\begin{aligned} C_{psq,1}(X) ={}& P_{PSQ}(\omega_{edge}X) - \Big( \big(1-P_{Q_{edge}^{valid}}(\omega_{edge}X)\big)P_{PSQ}(X)\\ &\qquad\qquad+ P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{new}^{edge}}(\omega_{edge}X)P_w(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ \big(1-P_{Q_{new}^{edge}}(\omega_{edge}X)\big)\big(P_{PSQ}(X)+P_w(\omega_{edge}X)\big) \big) \Big). \end{aligned}$

3. 组末约束

	$C_{psq,end}(X)=P_{Q_{end}^{edge}}(X)\cdot\big(P_{PSQ}(X)-P_{T^{edge}}(X)\big).$

#### 3.3.6 逆元约束

设节点域分母、逆元分别为 $P_{Sum}(X)$、$P_{inv}(X)$，则节点域逆元约束为

$C_{inv}(X)=P_{Q_N}(X)\cdot\big(P_{Sum}(X)P_{inv}(X)-1\big).$

本文正式协议**不**采用复合多项式替换

$P_{Sum}(P_{dst}(X)),\qquad P_{inv}(P_{dst}(X))$

来表达边域广播一致性。

本文唯一采用的正式口径是：

1. 先把边域广播列 $Sum^{edge}$、$inv^{edge}$ 显式作为独立见证列提交；
2. 再通过目标路由查询端绑定、PSQ 目标值绑定以及后续对应的边域 / 节点域约束，强制这些广播列与对应节点域列保持一致。

因此，在 quotient 约束族中，$3.3.8$、$3.3.9$、$3.3.13$、$3.3.14$ 已经给出的目标路由与输出聚合约束，就是本文用于保证广播一致性的**唯一正式约束来源**；本节不再额外引入第二套复合多项式广播约束。

#### 3.3.7 特征检索域商多项式

特征检索域上的全部约束定义为：

1. lookup 起点约束

	$C_{feat,0}(X)=First_{FH}(X)\cdot P_{R_{feat}}(X).$

2. lookup 递推约束

	$\begin{aligned} C_{feat,1}(X) ={}& \big(P_{R_{feat}}(\omega_{FH}X)-P_{R_{feat}}(X)\big) \big(P_{Table^{feat}}(X)+\beta_{feat}\big) \big(P_{Query^{feat}}(X)+\beta_{feat}\big)\\ &- P_{Q_{tbl}^{feat}}(X)P_{m_{feat}}(X)\big(P_{Query^{feat}}(X)+\beta_{feat}\big)\\ &+ P_{Q_{qry}^{feat}}(X)\big(P_{Table^{feat}}(X)+\beta_{feat}\big). \end{aligned}$

3. lookup 终点约束

	$C_{feat,2}(X)=Last_{FH}(X)\cdot P_{R_{feat}}(X).$

4. 表端绑定约束

	$C_{feat,tbl}(X)=P_{Table^{feat}}(X)-\Big(P_{Row_{feat}^{tbl}}(X)+\eta_{feat}P_{Col_{feat}^{tbl}}(X)+\eta_{feat}^2P_{T_H}(X)\Big).$

5. 查询端绑定约束

	$C_{feat,qry}(X)=P_{Query^{feat}}(X)-\Big(P_{I_{feat}^{qry}}(X)+\eta_{feat}P_{Col_{feat}^{qry}}(X)+\eta_{feat}^2P_H(X)\Big).$

于是特征检索域商多项式完全展开为

$t_{FH}(X) = \frac{ \alpha_{quot}^{e_{feat,0}}C_{feat,0}(X) + \alpha_{quot}^{e_{feat,1}}C_{feat,1}(X) + \alpha_{quot}^{e_{feat,2}}C_{feat,2}(X) + \alpha_{quot}^{e_{feat,tbl}}C_{feat,tbl}(X) + \alpha_{quot}^{e_{feat,qry}}C_{feat,qry}(X) }{ Z_{FH}(X) }.$

#### 3.3.8 第 $r$ 个隐藏层注意力头在边域上的全部约束

对每个 $r\in\{0,1,\ldots,7\}$，在边域 $\mathbb H_{edge}$ 上定义以下约束。

##### （一）源路由边端约束

$C_{src,edge,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{src}^{edge(r)}}(X),$

$C_{src,edge,1}^{(r)}(X) = \big(P_{R_{src}^{edge(r)}}(\omega_{edge}X)-P_{R_{src}^{edge(r)}}(X)\big)\big(P_{Query^{src(r)}}(X)+\beta_{src}^{(r)}\big)-P_{Q_{edge}^{valid}}(X),$

$C_{src,edge,2}^{(r)}(X)=Last_{edge}(X)\cdot\big(P_{R_{src}^{edge(r)}}(X)-S_{src}^{(r)}\big).$

##### （二）LeakyReLU 约束

$C_{L,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_L^{(r)}}(X),$

$\begin{aligned} C_{L,1}^{(r)}(X) ={}& \big(P_{R_L^{(r)}}(\omega_{edge}X)-P_{R_L^{(r)}}(X)\big) \big(P_{Table^{L(r)}}(X)+\beta_L^{(r)}\big) \big(P_{Query^{L(r)}}(X)+\beta_L^{(r)}\big)\\ &- P_{Q_{tbl}^{L(r)}}(X)P_{m_L^{(r)}}(X)\big(P_{Query^{L(r)}}(X)+\beta_L^{(r)}\big)\\ &+ P_{Q_{qry}^{L(r)}}(X)\big(P_{Table^{L(r)}}(X)+\beta_L^{(r)}\big), \end{aligned}$

$C_{L,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_L^{(r)}}(X),$

$C_{L,tbl}^{(r)}(X)=P_{Table^{L(r)}}(X)-\big(P_{T_{LReLU},x}(X)+\eta_L^{(r)}P_{T_{LReLU},y}(X)\big),$

$C_{L,qry}^{(r)}(X)=P_{Query^{L(r)}}(X)-\big(P_{S^{(r)}}(X)+\eta_L^{(r)}P_{Z^{(r)}}(X)\big).$

##### （三）最大值唯一性约束

$C_{max,bin}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(r)}}(X)\big(P_{s_{max}^{(r)}}(X)-1\big),$

$C_{max,zero}^{(r)}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(r)}}(X)P_{\Delta^{+(r)}}(X),$

$C_{max,0}^{(r)}(X)=First_{edge}(X)\cdot\big(P_{C_{max}^{(r)}}(X)-P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(r)}}(X)\big),$

$\begin{aligned} C_{max,1}^{(r)}(X) ={}& P_{C_{max}^{(r)}}(\omega_{edge}X)\\ &- \Big( (1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{C_{max}^{(r)}}(X) + P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{new}^{edge}}(\omega_{edge}X)P_{s_{max}^{(r)}}(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ (1-P_{Q_{new}^{edge}}(\omega_{edge}X))(P_{C_{max}^{(r)}}(X)+P_{s_{max}^{(r)}}(\omega_{edge}X)) \big) \Big), \end{aligned}$

$C_{max,end}^{(r)}(X)=P_{Q_{end}^{edge}}(X)\cdot\big(P_{C_{max}^{(r)}}(X)-1\big).$

##### （四）范围检查约束

$C_{R,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_R^{(r)}}(X),$

$\begin{aligned} C_{R,1}^{(r)}(X) ={}& \big(P_{R_R^{(r)}}(\omega_{edge}X)-P_{R_R^{(r)}}(X)\big) \big(P_{Table^{R(r)}}(X)+\beta_R^{(r)}\big) \big(P_{Query^{R(r)}}(X)+\beta_R^{(r)}\big)\\ &- P_{Q_{tbl}^{R(r)}}(X)P_{m_R^{(r)}}(X)\big(P_{Query^{R(r)}}(X)+\beta_R^{(r)}\big)\\ &+ P_{Q_{qry}^{R(r)}}(X)\big(P_{Table^{R(r)}}(X)+\beta_R^{(r)}\big), \end{aligned}$

$C_{R,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_R^{(r)}}(X),$

$C_{R,tbl}^{(r)}(X)=P_{Table^{R(r)}}(X)-P_{T_{range}}(X),$

$C_{R,qry}^{(r)}(X)=P_{Query^{R(r)}}(X)-P_{\Delta^{+(r)}}(X).$

##### （五）指数查表约束

$C_{exp,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{exp}^{(r)}}(X),$

$\begin{aligned} C_{exp,1}^{(r)}(X) ={}& \big(P_{R_{exp}^{(r)}}(\omega_{edge}X)-P_{R_{exp}^{(r)}}(X)\big) \big(P_{Table^{exp(r)}}(X)+\beta_{exp}^{(r)}\big) \big(P_{Query^{exp(r)}}(X)+\beta_{exp}^{(r)}\big)\\ &- P_{Q_{tbl}^{exp(r)}}(X)P_{m_{exp}^{(r)}}(X)\big(P_{Query^{exp(r)}}(X)+\beta_{exp}^{(r)}\big)\\ &+ P_{Q_{qry}^{exp(r)}}(X)\big(P_{Table^{exp(r)}}(X)+\beta_{exp}^{(r)}\big), \end{aligned}$

$C_{exp,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_{exp}^{(r)}}(X),$

$C_{exp,tbl}^{(r)}(X)=P_{Table^{exp(r)}}(X)-\big(P_{T_{exp},x}(X)+\eta_{exp}^{(r)}P_{T_{exp},y}(X)\big),$

$C_{exp,qry}^{(r)}(X)=P_{Query^{exp(r)}}(X)-\big(P_{\Delta^{+(r)}}(X)+\eta_{exp}^{(r)}P_{U^{(r)}}(X)\big).$

##### （六）聚合前 PSQ 约束

$C_{psq,0}^{(r)}(X)=First_{edge}(X)\cdot\big(P_{PSQ^{(r)}}(X)-P_{Q_{edge}^{valid}}(X)P_{w_{psq}^{(r)}}(X)\big),$

$\begin{aligned} C_{psq,1}^{(r)}(X) ={}& P_{PSQ^{(r)}}(\omega_{edge}X)\\ &- \Big( (1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{PSQ^{(r)}}(X) + P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{new}^{edge}}(\omega_{edge}X)P_{w_{psq}^{(r)}}(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ (1-P_{Q_{new}^{edge}}(\omega_{edge}X))(P_{PSQ^{(r)}}(X)+P_{w_{psq}^{(r)}}(\omega_{edge}X)) \big) \Big), \end{aligned}$

$C_{psq,end}^{(r)}(X)=P_{Q_{end}^{edge}}(X)\cdot\big(P_{PSQ^{(r)}}(X)-P_{T_{psq}^{edge(r)}}(X)\big).$

##### （七）ELU 约束

$C_{ELU,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{ELU}^{(r)}}(X),$

$\begin{aligned} C_{ELU,1}^{(r)}(X) ={}& \big(P_{R_{ELU}^{(r)}}(\omega_{edge}X)-P_{R_{ELU}^{(r)}}(X)\big) \big(P_{Table^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big) \big(P_{Query^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big)\\ &- P_{Q_{tbl}^{ELU(r)}}(X)P_{m_{ELU}^{(r)}}(X)\big(P_{Query^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big)\\ &+ P_{Q_{qry}^{ELU(r)}}(X)\big(P_{Table^{ELU(r)}}(X)+\beta_{ELU}^{(r)}\big), \end{aligned}$

$C_{ELU,2}^{(r)}(X)=Last_{edge}(X)\cdot P_{R_{ELU}^{(r)}}(X),$

$C_{ELU,tbl}^{(r)}(X)=P_{Table^{ELU(r)}}(X)-\big(P_{T_{ELU},x}(X)+\eta_{ELU}^{(r)}P_{T_{ELU},y}(X)\big),$

$C_{ELU,qry}^{(r)}(X)=P_{Query^{ELU(r)}}(X)-\big(P_{H_{agg,pre}^{(r)}}(X)+\eta_{ELU}^{(r)}P_{H_{agg}^{(r)}}(X)\big).$

##### （八）目标路由边端约束

$C_{dst,edge,0}^{(r)}(X)=First_{edge}(X)\cdot P_{R_{dst}^{edge(r)}}(X),$

$C_{dst,edge,1}^{(r)}(X) = \big(P_{R_{dst}^{edge(r)}}(\omega_{edge}X)-P_{R_{dst}^{edge(r)}}(X)\big)\big(P_{Query^{dst(r)}}(X)+\beta_{dst}^{(r)}\big)-P_{Q_{edge}^{valid}}(X),$

$C_{dst,edge,2}^{(r)}(X)=Last_{edge}(X)\cdot\big(P_{R_{dst}^{edge(r)}}(X)-S_{dst}^{(r)}\big).$

##### （九）边域绑定约束

$C_{src,qry}^{(r)}(X)=P_{Query^{src(r)}}(X)-\Big(P_{src}(X)+\eta_{src}^{(r)}P_{E_{src}^{edge(r)}}(X)+(\eta_{src}^{(r)})^2P_{H_{src}^{\star,edge(r)}}(X)\Big),$

$C_{dst,qry}^{(r)}(X)=P_{Query^{dst(r)}}(X)-\Big( P_{dst}(X)+\eta_{dst}^{(r)}P_{E_{dst}^{edge(r)}}(X)+(\eta_{dst}^{(r)})^2P_{M^{edge(r)}}(X)+(\eta_{dst}^{(r)})^3P_{Sum^{edge(r)}}(X)+(\eta_{dst}^{(r)})^4P_{inv^{edge(r)}}(X)+(\eta_{dst}^{(r)})^5P_{H_{agg}^{\star,edge(r)}}(X) \Big).$

#### 3.3.9 第 $r$ 个隐藏层注意力头在节点域上的全部约束

对每个 $r$，在节点域 $\mathbb H_N$ 上定义：

1. 源路由节点端三条约束

	$C_{src,node,0}^{(r)}(X)=First_N(X)\cdot P_{R_{src}^{node(r)}}(X),$

	$C_{src,node,1}^{(r)}(X)=\big(P_{R_{src}^{node(r)}}(\omega_NX)-P_{R_{src}^{node(r)}}(X)\big)\big(P_{Table^{src(r)}}(X)+\beta_{src}^{(r)}\big)-P_{Q_N}(X)P_{m_{src}^{(r)}}(X),$

	$C_{src,node,2}^{(r)}(X)=Last_N(X)\cdot\big(P_{R_{src}^{node(r)}}(X)-S_{src}^{(r)}\big).$

2. 目标路由节点端三条约束

	$C_{dst,node,0}^{(r)}(X)=First_N(X)\cdot P_{R_{dst}^{node(r)}}(X),$

	$C_{dst,node,1}^{(r)}(X)=\big(P_{R_{dst}^{node(r)}}(\omega_NX)-P_{R_{dst}^{node(r)}}(X)\big)\big(P_{Table^{dst(r)}}(X)+\beta_{dst}^{(r)}\big)-P_{Q_N}(X)P_{m_{dst}^{(r)}}(X),$

	$C_{dst,node,2}^{(r)}(X)=Last_N(X)\cdot\big(P_{R_{dst}^{node(r)}}(X)-S_{dst}^{(r)}\big).$

3. 分母逆元约束

	$C_{inv}^{(r)}(X)=P_{Q_N}(X)\cdot\big(P_{Sum^{(r)}}(X)P_{inv^{(r)}}(X)-1\big).$

4. 源路由表端绑定

	$C_{src,tbl}^{(r)}(X)=P_{Table^{src(r)}}(X)-\Big(P_{Idx_N}(X)+\eta_{src}^{(r)}P_{E_{src}^{(r)}}(X)+(\eta_{src}^{(r)})^2P_{H^{\star(r)}}(X)\Big).$

5. 目标路由表端绑定

	$\begin{aligned} C_{dst,tbl}^{(r)}(X) ={}& P_{Table^{dst(r)}}(X) - \Big( P_{Idx_N}(X) + \eta_{dst}^{(r)}P_{E_{dst}^{(r)}}(X)\\ &\qquad +(\eta_{dst}^{(r)})^2P_{M^{(r)}}(X) +(\eta_{dst}^{(r)})^3P_{Sum^{(r)}}(X) +(\eta_{dst}^{(r)})^4P_{inv^{(r)}}(X) +(\eta_{dst}^{(r)})^5P_{H_{agg}^{\star(r)}}(X) \Big). \end{aligned}$

#### 3.3.10 第 $r$ 个隐藏层注意力头在共享维域上的全部绑定约束

##### （一）输入共享维域 $\mathbb H_{in}$

$C_{proj,0}^{(r)}(X)=First_{in}(X)\cdot P_{Acc^{proj(r)}}(X),$

$C_{proj,1}^{(r)}(X)=P_{Acc^{proj(r)}}(\omega_{in}X)-P_{Acc^{proj(r)}}(X)-P_{a^{proj(r)}}(X)P_{b^{proj(r)}}(X),$

$C_{proj,2}^{(r)}(X)=Last_{in}(X)\cdot\big(P_{Acc^{proj(r)}}(X)-\mu_{proj}^{(r)}\big).$

##### （二）隐藏层单头共享维域 $\mathbb H_{d_h}$

源注意力绑定：

$C_{srcbind,0}^{(r)}(X)=First_{d_h}(X)\cdot P_{Acc^{src(r)}}(X),$

$C_{srcbind,1}^{(r)}(X)=P_{Acc^{src(r)}}(\omega_{d_h}X)-P_{Acc^{src(r)}}(X)-P_{a^{src(r)}}(X)P_{b^{src(r)}}(X),$

$C_{srcbind,2}^{(r)}(X)=Last_{d_h}(X)\cdot\big(P_{Acc^{src(r)}}(X)-\mu_{src}^{(r)}\big).$

目标注意力绑定：

$C_{dstbind,0}^{(r)}(X)=First_{d_h}(X)\cdot P_{Acc^{dst(r)}}(X),$

$C_{dstbind,1}^{(r)}(X)=P_{Acc^{dst(r)}}(\omega_{d_h}X)-P_{Acc^{dst(r)}}(X)-P_{a^{dst(r)}}(X)P_{b^{dst(r)}}(X),$

$C_{dstbind,2}^{(r)}(X)=Last_{d_h}(X)\cdot\big(P_{Acc^{dst(r)}}(X)-\mu_{dst}^{(r)}\big).$

压缩特征绑定：

$C_{starbind,0}^{(r)}(X)=First_{d_h}(X)\cdot P_{Acc^{\star(r)}}(X),$

$C_{starbind,1}^{(r)}(X)=P_{Acc^{\star(r)}}(\omega_{d_h}X)-P_{Acc^{\star(r)}}(X)-P_{a^{\star(r)}}(X)P_{b^{\star(r)}}(X),$

$C_{starbind,2}^{(r)}(X)=Last_{d_h}(X)\cdot\big(P_{Acc^{\star(r)}}(X)-\mu_{\star}^{(r)}\big).$

先对每个隐藏维索引 $j\in\{0,1,\ldots,d_h-1\}$ 定义节点域列多项式

$P_{A_{agg,pre,j}^{(r)}}(X)=\sum_{i=0}^{n_N-1}H_{agg,pre,i,j}^{(r)}L_i^{(N)}(X),$

$P_{A_{agg,j}^{(r)}}(X)=\sum_{i=0}^{n_N-1}H_{agg,i,j}^{(r)}L_i^{(N)}(X).$

聚合前压缩绑定：

$C_{aggpre}^{(r)}(X)=P_{H_{agg,pre}^{\star(r)}}(X)-\sum_{j=0}^{d_h-1}P_{A_{agg,pre,j}^{(r)}}(X)(\xi^{(r)})^j.$

聚合后压缩绑定：

$C_{agg}^{(r)}(X)=P_{H_{agg}^{\star(r)}}(X)-\sum_{j=0}^{d_h-1}P_{A_{agg,j}^{(r)}}(X)(\xi^{(r)})^j.$

#### 3.3.11 拼接共享维域约束

在 $\mathbb H_{cat}$ 上，先定义参考拼接多项式

$P_{H_{cat}}^{ref}(X) = \sum_{i=0}^{N-1}\sum_{r=0}^{7}\sum_{j=0}^{d_h-1} H_{agg,i,j}^{(r)}X^{i\cdot d_{cat}+r\cdot d_h+j}.$

再对每个拼接维索引 $m\in\{0,1,\ldots,d_{cat}-1\}$ 定义节点域列多项式

$P_{A_{cat,m}}(X)=\sum_{i=0}^{n_N-1}H_{cat,i,m}L_i^{(N)}(X).$

于是定义：

1. 拼接线性一致性约束

	$C_{cat,lin}(X)=P_{H_{cat}}(X)-P_{H_{cat}}^{ref}(X).$

2. 拼接压缩绑定约束

	$C_{cat,bind}(X)=P_{H_{cat}^{\star}}(X)-\sum_{m=0}^{d_{cat}-1}P_{A_{cat,m}}(X)\xi_{cat}^m.$

3. 输出投影输入共享维绑定约束

	$C_{outproj,0}(X)=First_{cat}(X)\cdot P_{Acc^{proj(out)}}(X),$

	$C_{outproj,1}(X)=P_{Acc^{proj(out)}}(\omega_{cat}X)-P_{Acc^{proj(out)}}(X)-P_{a^{proj(out)}}(X)P_{b^{proj(out)}}(X),$

	$C_{outproj,2}(X)=Last_{cat}(X)\cdot\big(P_{Acc^{proj(out)}}(X)-\mu_{proj}^{(out)}\big).$

#### 3.3.12 输出层类别共享维域约束

在 $\mathbb H_C$ 上，定义：

1. 输出层源注意力绑定

	$C_{outsrc,0}(X)=First_C(X)\cdot P_{Acc^{src(out)}}(X),$

	$C_{outsrc,1}(X)=P_{Acc^{src(out)}}(\omega_CX)-P_{Acc^{src(out)}}(X)-P_{a^{src(out)}}(X)P_{b^{src(out)}}(X),$

	$C_{outsrc,2}(X)=Last_C(X)\cdot\big(P_{Acc^{src(out)}}(X)-\mu_{src}^{(out)}\big).$

2. 输出层目标注意力绑定

	$C_{outdst,0}(X)=First_C(X)\cdot P_{Acc^{dst(out)}}(X),$

	$C_{outdst,1}(X)=P_{Acc^{dst(out)}}(\omega_CX)-P_{Acc^{dst(out)}}(X)-P_{a^{dst(out)}}(X)P_{b^{dst(out)}}(X),$

	$C_{outdst,2}(X)=Last_C(X)\cdot\big(P_{Acc^{dst(out)}}(X)-\mu_{dst}^{(out)}\big).$

3. 先对每个类别索引 $c\in\{0,1,\ldots,C-1\}$ 定义节点域列多项式

	$P_{A_{Y,c}}(X)=\sum_{i=0}^{n_N-1}Y_{i,c}L_i^{(N)}(X).$

	输出压缩绑定约束

	$C_{outY}(X)=P_{Y^{\star}}(X)-\sum_{c=0}^{C-1}P_{A_{Y,c}}(X)\xi_{out}^c.$

#### 3.3.13 输出层在边域上的全部约束

在边域 $\mathbb H_{edge}$ 上，输出层的全部约束显式写为：

##### （一）输出层源路由边端约束

$C_{out,src,edge,0}(X)=First_{edge}(X)\cdot P_{R_{src}^{edge(out)}}(X),$

$C_{out,src,edge,1}(X) = \big(P_{R_{src}^{edge(out)}}(\omega_{edge}X)-P_{R_{src}^{edge(out)}}(X)\big)\big(P_{Query^{src(out)}}(X)+\beta_{src}^{(out)}\big)-P_{Q_{edge}^{valid}}(X),$

$C_{out,src,edge,2}(X)=Last_{edge}(X)\cdot\big(P_{R_{src}^{edge(out)}}(X)-S_{src}^{(out)}\big).$

##### （二）输出层 LeakyReLU 约束

$C_{out,L,0}(X)=First_{edge}(X)\cdot P_{R_L^{(out)}}(X),$

$\begin{aligned} C_{out,L,1}(X) ={}& \big(P_{R_L^{(out)}}(\omega_{edge}X)-P_{R_L^{(out)}}(X)\big) \big(P_{Table^{L(out)}}(X)+\beta_L^{(out)}\big) \big(P_{Query^{L(out)}}(X)+\beta_L^{(out)}\big)\\ &- P_{Q_{tbl}^{L(out)}}(X)P_{m_L^{(out)}}(X)\big(P_{Query^{L(out)}}(X)+\beta_L^{(out)}\big)\\ &+ P_{Q_{qry}^{L(out)}}(X)\big(P_{Table^{L(out)}}(X)+\beta_L^{(out)}\big), \end{aligned}$

$C_{out,L,2}(X)=Last_{edge}(X)\cdot P_{R_L^{(out)}}(X),$

$C_{out,L,tbl}(X)=P_{Table^{L(out)}}(X)-\big(P_{T_{LReLU},x}(X)+\eta_L^{(out)}P_{T_{LReLU},y}(X)\big),$

$C_{out,L,qry}(X)=P_{Query^{L(out)}}(X)-\big(P_{S^{(out)}}(X)+\eta_L^{(out)}P_{Z^{(out)}}(X)\big).$

##### （三）输出层最大值唯一性约束

$C_{out,max,bin}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(out)}}(X)\big(P_{s_{max}^{(out)}}(X)-1\big),$

$C_{out,max,zero}(X)=P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(out)}}(X)P_{\Delta^{+(out)}}(X),$

$C_{out,max,0}(X)=First_{edge}(X)\cdot\big(P_{C_{max}^{(out)}}(X)-P_{Q_{edge}^{valid}}(X)P_{s_{max}^{(out)}}(X)\big),$

$\begin{aligned} C_{out,max,1}(X) ={}& P_{C_{max}^{(out)}}(\omega_{edge}X)\\ &- \Big( (1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{C_{max}^{(out)}}(X) + P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{new}^{edge}}(\omega_{edge}X)P_{s_{max}^{(out)}}(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ (1-P_{Q_{new}^{edge}}(\omega_{edge}X))(P_{C_{max}^{(out)}}(X)+P_{s_{max}^{(out)}}(\omega_{edge}X)) \big) \Big), \end{aligned}$

$C_{out,max,end}(X)=P_{Q_{end}^{edge}}(X)\cdot\big(P_{C_{max}^{(out)}}(X)-1\big).$

##### （四）输出层范围检查约束

$C_{out,R,0}(X)=First_{edge}(X)\cdot P_{R_R^{(out)}}(X),$

$\begin{aligned} C_{out,R,1}(X) ={}& \big(P_{R_R^{(out)}}(\omega_{edge}X)-P_{R_R^{(out)}}(X)\big) \big(P_{Table^{R(out)}}(X)+\beta_R^{(out)}\big) \big(P_{Query^{R(out)}}(X)+\beta_R^{(out)}\big)\\ &- P_{Q_{tbl}^{R(out)}}(X)P_{m_R^{(out)}}(X)\big(P_{Query^{R(out)}}(X)+\beta_R^{(out)}\big)\\ &+ P_{Q_{qry}^{R(out)}}(X)\big(P_{Table^{R(out)}}(X)+\beta_R^{(out)}\big), \end{aligned}$

$C_{out,R,2}(X)=Last_{edge}(X)\cdot P_{R_R^{(out)}}(X),$

$C_{out,R,tbl}(X)=P_{Table^{R(out)}}(X)-P_{T_{range}}(X),$

$C_{out,R,qry}(X)=P_{Query^{R(out)}}(X)-P_{\Delta^{+(out)}}(X).$

##### （五）输出层指数查表约束

$C_{out,exp,0}(X)=First_{edge}(X)\cdot P_{R_{exp}^{(out)}}(X),$

$\begin{aligned} C_{out,exp,1}(X) ={}& \big(P_{R_{exp}^{(out)}}(\omega_{edge}X)-P_{R_{exp}^{(out)}}(X)\big) \big(P_{Table^{exp(out)}}(X)+\beta_{exp}^{(out)}\big) \big(P_{Query^{exp(out)}}(X)+\beta_{exp}^{(out)}\big)\\ &- P_{Q_{tbl}^{exp(out)}}(X)P_{m_{exp}^{(out)}}(X)\big(P_{Query^{exp(out)}}(X)+\beta_{exp}^{(out)}\big)\\ &+ P_{Q_{qry}^{exp(out)}}(X)\big(P_{Table^{exp(out)}}(X)+\beta_{exp}^{(out)}\big), \end{aligned}$

$C_{out,exp,2}(X)=Last_{edge}(X)\cdot P_{R_{exp}^{(out)}}(X),$

$C_{out,exp,tbl}(X)=P_{Table^{exp(out)}}(X)-\big(P_{T_{exp},x}(X)+\eta_{exp}^{(out)}P_{T_{exp},y}(X)\big),$

$C_{out,exp,qry}(X)=P_{Query^{exp(out)}}(X)-\big(P_{\Delta^{+(out)}}(X)+\eta_{exp}^{(out)}P_{U^{(out)}}(X)\big).$

##### （六）输出层聚合 PSQ 约束

$C_{out,psq,0}(X)=First_{edge}(X)\cdot\big(P_{PSQ^{(out)}}(X)-P_{Q_{edge}^{valid}}(X)P_{w_{out}}(X)\big),$

$\begin{aligned} C_{out,psq,1}(X) ={}& P_{PSQ^{(out)}}(\omega_{edge}X)\\ &- \Big( (1-P_{Q_{edge}^{valid}}(\omega_{edge}X))P_{PSQ^{(out)}}(X) + P_{Q_{edge}^{valid}}(\omega_{edge}X) \big( P_{Q_{new}^{edge}}(\omega_{edge}X)P_{w_{out}}(\omega_{edge}X)\\ &\qquad\qquad\qquad\qquad+ (1-P_{Q_{new}^{edge}}(\omega_{edge}X))(P_{PSQ^{(out)}}(X)+P_{w_{out}}(\omega_{edge}X)) \big) \Big), \end{aligned}$

$C_{out,psq,end}(X)=P_{Q_{end}^{edge}}(X)\cdot\big(P_{PSQ^{(out)}}(X)-P_{T_{out}^{edge}}(X)\big).$

##### （七）输出层目标路由边端约束

$C_{out,dst,edge,0}(X)=First_{edge}(X)\cdot P_{R_{dst}^{edge(out)}}(X),$

$C_{out,dst,edge,1}(X) = \big(P_{R_{dst}^{edge(out)}}(\omega_{edge}X)-P_{R_{dst}^{edge(out)}}(X)\big)\big(P_{Query^{dst(out)}}(X)+\beta_{dst}^{(out)}\big)-P_{Q_{edge}^{valid}}(X),$

$C_{out,dst,edge,2}(X)=Last_{edge}(X)\cdot\big(P_{R_{dst}^{edge(out)}}(X)-S_{dst}^{(out)}\big).$

#### 3.3.14 输出层在节点域上的全部约束

在节点域 $\mathbb H_N$ 上，显式定义：

1. 输出层源路由节点端起点约束

	$C_{out,src,node,0}(X)=First_N(X)\cdot P_{R_{src}^{node(out)}}(X).$

2. 输出层源路由节点端递推约束

	$C_{out,src,node,1}(X) = \big(P_{R_{src}^{node(out)}}(\omega_NX)-P_{R_{src}^{node(out)}}(X)\big)\big(P_{Table^{src(out)}}(X)+\beta_{src}^{(out)}\big)-P_{Q_N}(X)P_{m_{src}^{(out)}}(X).$

3. 输出层源路由节点端终点约束

	$C_{out,src,node,2}(X)=Last_N(X)\cdot\big(P_{R_{src}^{node(out)}}(X)-S_{src}^{(out)}\big).$

4. 输出层目标路由节点端起点约束

	$C_{out,dst,node,0}(X)=First_N(X)\cdot P_{R_{dst}^{node(out)}}(X).$

5. 输出层目标路由节点端递推约束

	$C_{out,dst,node,1}(X) = \big(P_{R_{dst}^{node(out)}}(\omega_NX)-P_{R_{dst}^{node(out)}}(X)\big)\big(P_{Table^{dst(out)}}(X)+\beta_{dst}^{(out)}\big)-P_{Q_N}(X)P_{m_{dst}^{(out)}}(X).$

6. 输出层目标路由节点端终点约束

	$C_{out,dst,node,2}(X)=Last_N(X)\cdot\big(P_{R_{dst}^{node(out)}}(X)-S_{dst}^{(out)}\big).$

7. 输出层逆元约束

	$C_{out,inv}(X)=P_{Q_N}(X)\cdot\big(P_{Sum^{(out)}}(X)P_{inv^{(out)}}(X)-1\big).$

8. 输出层源路由表端绑定

	$C_{out,src,tbl}(X)=P_{Table^{src(out)}}(X)-\big(P_{Idx_N}(X)+\eta_{src}^{(out)}P_{E_{src}^{(out)}}(X)\big).$

9. 输出层目标路由表端绑定

	$\begin{aligned} C_{out,dst,tbl}(X) ={}& P_{Table^{dst(out)}}(X) - \Big( P_{Idx_N}(X) +\eta_{dst}^{(out)}P_{E_{dst}^{(out)}}(X) +(\eta_{dst}^{(out)})^2P_{M^{(out)}}(X)\\ &\qquad +(\eta_{dst}^{(out)})^3P_{Sum^{(out)}}(X) +(\eta_{dst}^{(out)})^4P_{inv^{(out)}}(X) +(\eta_{dst}^{(out)})^5P_{Y^{\star}}(X) \Big). \end{aligned}$

#### 3.3.15 七个工作域的商多项式完全展开

为避免任何省略，七个工作域的商多项式分别写作：

1. 特征检索域

	$t_{FH}(X) = \frac{ \alpha_{quot}^{e_{feat,0}}C_{feat,0}(X) + \alpha_{quot}^{e_{feat,1}}C_{feat,1}(X) + \alpha_{quot}^{e_{feat,2}}C_{feat,2}(X) + \alpha_{quot}^{e_{feat,tbl}}C_{feat,tbl}(X) + \alpha_{quot}^{e_{feat,qry}}C_{feat,qry}(X) }{Z_{FH}(X)}.$

2. 边域

	$\begin{aligned} t_{edge}(X) ={}& \frac{1}{Z_{edge}(X)} \Bigg[ \sum_{r=0}^{7} \Big( \alpha_{quot}^{e_{src,edge,0,r}}C_{src,edge,0}^{(r)}(X) + \alpha_{quot}^{e_{src,edge,1,r}}C_{src,edge,1}^{(r)}(X) + \alpha_{quot}^{e_{src,edge,2,r}}C_{src,edge,2}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{L,0,r}}C_{L,0}^{(r)}(X) + \alpha_{quot}^{e_{L,1,r}}C_{L,1}^{(r)}(X) + \alpha_{quot}^{e_{L,2,r}}C_{L,2}^{(r)}(X) + \alpha_{quot}^{e_{L,tbl,r}}C_{L,tbl}^{(r)}(X) + \alpha_{quot}^{e_{L,qry,r}}C_{L,qry}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{max,bin,r}}C_{max,bin}^{(r)}(X) + \alpha_{quot}^{e_{max,zero,r}}C_{max,zero}^{(r)}(X) + \alpha_{quot}^{e_{max,0,r}}C_{max,0}^{(r)}(X) + \alpha_{quot}^{e_{max,1,r}}C_{max,1}^{(r)}(X) + \alpha_{quot}^{e_{max,end,r}}C_{max,end}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{R,0,r}}C_{R,0}^{(r)}(X) + \alpha_{quot}^{e_{R,1,r}}C_{R,1}^{(r)}(X) + \alpha_{quot}^{e_{R,2,r}}C_{R,2}^{(r)}(X) + \alpha_{quot}^{e_{R,tbl,r}}C_{R,tbl}^{(r)}(X) + \alpha_{quot}^{e_{R,qry,r}}C_{R,qry}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{exp,0,r}}C_{exp,0}^{(r)}(X) + \alpha_{quot}^{e_{exp,1,r}}C_{exp,1}^{(r)}(X) + \alpha_{quot}^{e_{exp,2,r}}C_{exp,2}^{(r)}(X) + \alpha_{quot}^{e_{exp,tbl,r}}C_{exp,tbl}^{(r)}(X) + \alpha_{quot}^{e_{exp,qry,r}}C_{exp,qry}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{psq,0,r}}C_{psq,0}^{(r)}(X) + \alpha_{quot}^{e_{psq,1,r}}C_{psq,1}^{(r)}(X) + \alpha_{quot}^{e_{psq,end,r}}C_{psq,end}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{ELU,0,r}}C_{ELU,0}^{(r)}(X) + \alpha_{quot}^{e_{ELU,1,r}}C_{ELU,1}^{(r)}(X) + \alpha_{quot}^{e_{ELU,2,r}}C_{ELU,2}^{(r)}(X) + \alpha_{quot}^{e_{ELU,tbl,r}}C_{ELU,tbl}^{(r)}(X) + \alpha_{quot}^{e_{ELU,qry,r}}C_{ELU,qry}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{dst,edge,0,r}}C_{dst,edge,0}^{(r)}(X) + \alpha_{quot}^{e_{dst,edge,1,r}}C_{dst,edge,1}^{(r)}(X) + \alpha_{quot}^{e_{dst,edge,2,r}}C_{dst,edge,2}^{(r)}(X) + \alpha_{quot}^{e_{src,qry,r}}C_{src,qry}^{(r)}(X) + \alpha_{quot}^{e_{dst,qry,r}}C_{dst,qry}^{(r)}(X) \Big)\\ &\qquad + \alpha_{quot}^{e_{out,src,edge,0}}C_{out,src,edge,0}(X) + \alpha_{quot}^{e_{out,src,edge,1}}C_{out,src,edge,1}(X) + \alpha_{quot}^{e_{out,src,edge,2}}C_{out,src,edge,2}(X)\\ &\qquad + \alpha_{quot}^{e_{out,L,0}}C_{out,L,0}(X) + \alpha_{quot}^{e_{out,L,1}}C_{out,L,1}(X) + \alpha_{quot}^{e_{out,L,2}}C_{out,L,2}(X) + \alpha_{quot}^{e_{out,L,tbl}}C_{out,L,tbl}(X) + \alpha_{quot}^{e_{out,L,qry}}C_{out,L,qry}(X)\\ &\qquad + \alpha_{quot}^{e_{out,max,bin}}C_{out,max,bin}(X) + \alpha_{quot}^{e_{out,max,zero}}C_{out,max,zero}(X) + \alpha_{quot}^{e_{out,max,0}}C_{out,max,0}(X) + \alpha_{quot}^{e_{out,max,1}}C_{out,max,1}(X) + \alpha_{quot}^{e_{out,max,end}}C_{out,max,end}(X)\\ &\qquad + \alpha_{quot}^{e_{out,R,0}}C_{out,R,0}(X) + \alpha_{quot}^{e_{out,R,1}}C_{out,R,1}(X) + \alpha_{quot}^{e_{out,R,2}}C_{out,R,2}(X) + \alpha_{quot}^{e_{out,R,tbl}}C_{out,R,tbl}(X) + \alpha_{quot}^{e_{out,R,qry}}C_{out,R,qry}(X)\\ &\qquad + \alpha_{quot}^{e_{out,exp,0}}C_{out,exp,0}(X) + \alpha_{quot}^{e_{out,exp,1}}C_{out,exp,1}(X) + \alpha_{quot}^{e_{out,exp,2}}C_{out,exp,2}(X) + \alpha_{quot}^{e_{out,exp,tbl}}C_{out,exp,tbl}(X) + \alpha_{quot}^{e_{out,exp,qry}}C_{out,exp,qry}(X)\\ &\qquad + \alpha_{quot}^{e_{out,psq,0}}C_{out,psq,0}(X) + \alpha_{quot}^{e_{out,psq,1}}C_{out,psq,1}(X) + \alpha_{quot}^{e_{out,psq,end}}C_{out,psq,end}(X) + \alpha_{quot}^{e_{out,dst,edge,0}}C_{out,dst,edge,0}(X) + \alpha_{quot}^{e_{out,dst,edge,1}}C_{out,dst,edge,1}(X) + \alpha_{quot}^{e_{out,dst,edge,2}}C_{out,dst,edge,2}(X) \Bigg]. \end{aligned}$

3. 输入共享维域

	$t_{in}(X)= \frac{ \sum_{r=0}^{7} \big( \alpha_{quot}^{e_{proj,0,r}}C_{proj,0}^{(r)}(X) + \alpha_{quot}^{e_{proj,1,r}}C_{proj,1}^{(r)}(X) + \alpha_{quot}^{e_{proj,2,r}}C_{proj,2}^{(r)}(X) \big) }{Z_{in}(X)}.$

4. 隐藏层单头共享维域

	$\begin{aligned} t_{d_h}(X) ={}& \frac{1}{Z_{d_h}(X)} \sum_{r=0}^{7} \Big( \alpha_{quot}^{e_{srcbind,0,r}}C_{srcbind,0}^{(r)}(X) + \alpha_{quot}^{e_{srcbind,1,r}}C_{srcbind,1}^{(r)}(X) + \alpha_{quot}^{e_{srcbind,2,r}}C_{srcbind,2}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{dstbind,0,r}}C_{dstbind,0}^{(r)}(X) + \alpha_{quot}^{e_{dstbind,1,r}}C_{dstbind,1}^{(r)}(X) + \alpha_{quot}^{e_{dstbind,2,r}}C_{dstbind,2}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{starbind,0,r}}C_{starbind,0}^{(r)}(X) + \alpha_{quot}^{e_{starbind,1,r}}C_{starbind,1}^{(r)}(X) + \alpha_{quot}^{e_{starbind,2,r}}C_{starbind,2}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{aggpre,r}}C_{aggpre}^{(r)}(X) + \alpha_{quot}^{e_{agg,r}}C_{agg}^{(r)}(X) \Big). \end{aligned}$

5. 拼接共享维域

	$t_{cat}(X) = \frac{ \alpha_{quot}^{e_{cat,lin}}C_{cat,lin}(X) + \alpha_{quot}^{e_{cat,bind}}C_{cat,bind}(X) + \alpha_{quot}^{e_{outproj,0}}C_{outproj,0}(X) + \alpha_{quot}^{e_{outproj,1}}C_{outproj,1}(X) + \alpha_{quot}^{e_{outproj,2}}C_{outproj,2}(X) }{Z_{cat}(X)}.$

6. 输出层类别共享维域

	$t_C(X) = \frac{ \alpha_{quot}^{e_{outsrc,0}}C_{outsrc,0}(X) + \alpha_{quot}^{e_{outsrc,1}}C_{outsrc,1}(X) + \alpha_{quot}^{e_{outsrc,2}}C_{outsrc,2}(X) + \alpha_{quot}^{e_{outdst,0}}C_{outdst,0}(X) + \alpha_{quot}^{e_{outdst,1}}C_{outdst,1}(X) + \alpha_{quot}^{e_{outdst,2}}C_{outdst,2}(X) + \alpha_{quot}^{e_{outY}}C_{outY}(X) }{Z_C(X)}.$

7. 节点域

	$\begin{aligned} t_N(X) ={}& \frac{1}{Z_N(X)} \Bigg[ \sum_{r=0}^{7} \Big( \alpha_{quot}^{e_{src,node,0,r}}C_{src,node,0}^{(r)}(X) + \alpha_{quot}^{e_{src,node,1,r}}C_{src,node,1}^{(r)}(X) + \alpha_{quot}^{e_{src,node,2,r}}C_{src,node,2}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{dst,node,0,r}}C_{dst,node,0}^{(r)}(X) + \alpha_{quot}^{e_{dst,node,1,r}}C_{dst,node,1}^{(r)}(X) + \alpha_{quot}^{e_{dst,node,2,r}}C_{dst,node,2}^{(r)}(X) + \alpha_{quot}^{e_{inv,r}}C_{inv}^{(r)}(X)\\ &\qquad + \alpha_{quot}^{e_{src,tbl,r}}C_{src,tbl}^{(r)}(X) + \alpha_{quot}^{e_{dst,tbl,r}}C_{dst,tbl}^{(r)}(X) \Big)\\ &\qquad + \alpha_{quot}^{e_{out,src,node,0}}C_{out,src,node,0}(X) + \alpha_{quot}^{e_{out,src,node,1}}C_{out,src,node,1}(X) + \alpha_{quot}^{e_{out,src,node,2}}C_{out,src,node,2}(X)\\ &\qquad + \alpha_{quot}^{e_{out,dst,node,0}}C_{out,dst,node,0}(X) + \alpha_{quot}^{e_{out,dst,node,1}}C_{out,dst,node,1}(X) + \alpha_{quot}^{e_{out,dst,node,2}}C_{out,dst,node,2}(X) + \alpha_{quot}^{e_{out,inv}}C_{out,inv}(X)\\ &\qquad + \alpha_{quot}^{e_{out,src,tbl}}C_{out,src,tbl}(X) + \alpha_{quot}^{e_{out,dst,tbl}}C_{out,dst,tbl}(X) \Bigg] \end{aligned}$

### 3.4 外点评值、域内点评值与开放

外点评值集合必须至少包括：

1. 特征检索相关外点评值；

2. 对每个隐藏层注意力头 $r$：

	$P_{H'^{(r)}}(y_{proj}^{(r)}),$

	$P_{E_{src}^{(r)}}(y_{src}^{(r)}),$

	$P_{E_{dst}^{(r)}}(y_{dst}^{(r)}),$

	$P_{H^{\star(r)}}(y_{\star}^{(r)}),$

	$P_{H_{agg,pre}^{\star(r)}}(y_{agg,pre}^{(r)}),$

	$P_{H_{agg}^{\star(r)}}(y_{agg}^{(r)}).$

3. 拼接阶段：

	$P_{H_{cat}^{\star}}(y_{cat}).$

4. 输出层：

	$P_{Y'}(y_{proj}^{(out)}),$

	$P_{E_{src}^{(out)}}(y_{src}^{(out)}),$

	$P_{E_{dst}^{(out)}}(y_{dst}^{(out)}),$

	$P_{Y^{\star}}(y_{out}^{\star}),$

	$P_Y(y_{out}).$

对每个工作域

$\mathcal D\in\{FH,edge,in,d_h,cat,C,N\},$

都在点评集合

$\{z_{\mathcal D},z_{\mathcal D}\omega_{\mathcal D}\}$

上执行多点评 batch opening。

对全部外点评值统一用 $\rho_{ext}$ 做折叠，形成唯一的外点评值批量开放。

### 3.5 最终证明对象

最终证明对象

$\pi_{GAT}$

必须按**固定顺序**序列化为下列有序元组：

$\pi_{GAT} = \Big( M_{pub},\  \mathbf{Com}_{dyn},\  \mathbf{S}_{route},\  \mathbf{Eval}_{ext},\  \mathbf{Eval}_{dom},\  \mathbf{Com}_{quot},\  \mathbf{Open}_{dom},\  W_{ext},\  \Pi_{bind} \Big).$

其中各字段定义如下。

1. 元数据块

	$M_{pub}$

	必须严格按第 0.8.5 节字段表的顺序写成

	$M_{pub} = \big( protocol\_id,\ model\_arch\_id,\ model\_param\_id,\ static\_table\_id,\ quant\_cfg\_id,\ domain\_cfg,\ dim\_cfg,\ encoding\_id,\ padding\_rule\_id,\ degree\_bound\_id \big).$

2. 动态承诺块

	$\mathbf{Com}_{dyn}$

	必须严格按下列顺序拼接：

	$\mathbf{Com}_{dyn} = \big( \mathbf{Com}_{feat},\ \mathbf{Com}_{hidden}^{(0)},\ldots,\mathbf{Com}_{hidden}^{(7)},\ \mathbf{Com}_{cat},\ \mathbf{Com}_{out} \big).$

	其中

	$\mathbf{Com}_{feat} = \big( [P_H],\ [P_{Table^{feat}}],\ [P_{Query^{feat}}],\ [P_{m_{feat}}],\ [P_{Q_{tbl}^{feat}}],\ [P_{Q_{qry}^{feat}}],\ [P_{R_{feat}}] \big),$

	$\mathbf{Com}_{hidden}^{(r)}$

	定义为第 $r$ 个隐藏层注意力头在第 2.2.1 至第 2.2.10 节中**按出现顺序**提交的全部承诺的有序拼接，

	$\mathbf{Com}_{cat} = \big( [P_{H_{cat}}],\ [P_{H_{cat}^{\star}}] \big),$

	$\mathbf{Com}_{out}$

	定义为第 2.4.1 至第 2.4.7 节中**按出现顺序**提交的全部承诺的有序拼接。

3. 路由公开总和块

	$\mathbf{S}_{route} = \big( S_{src}^{(0)},S_{dst}^{(0)},\ldots,S_{src}^{(7)},S_{dst}^{(7)},S_{src}^{(out)},S_{dst}^{(out)} \big).$

4. 外点评值块

	$\mathbf{Eval}_{ext}$

	必须严格按下列顺序排列：

	$\mathbf{Eval}_{ext} = \Big( \{P_{H'^{(r)}}(y_{proj}^{(r)}),\ P_{E_{src}^{(r)}}(y_{src}^{(r)}),\ P_{E_{dst}^{(r)}}(y_{dst}^{(r)}),\ P_{H^{\star(r)}}(y_{\star}^{(r)}),\ P_{H_{agg,pre}^{\star(r)}}(y_{agg,pre}^{(r)}),\ P_{H_{agg}^{\star(r)}}(y_{agg}^{(r)})\}_{r=0}^{7},$

	$P_{H_{cat}^{\star}}(y_{cat}), \ P_{Y'}(y_{proj}^{(out)}),\ P_{E_{src}^{(out)}}(y_{src}^{(out)}),\ P_{E_{dst}^{(out)}}(y_{dst}^{(out)}),\ P_{Y^{\star}}(y_{out}^{\star}),\ P_Y(y_{out}) \Big).$

5. 域内点评值块

	$\mathbf{Eval}_{dom}$

	是七个工作域域内 opening 所需全部点评值的有序拼接：

	$\mathbf{Eval}_{dom} = \big( \mathbf{Eval}_{FH},\mathbf{Eval}_{edge},\mathbf{Eval}_{in},\mathbf{Eval}_{d_h},\mathbf{Eval}_{cat},\mathbf{Eval}_{C},\mathbf{Eval}_{N} \big).$

	对每个工作域 $\mathcal D\in\{FH,edge,in,d_h,cat,C,N\}$，

	$\mathbf{Eval}_{\mathcal D}$

	必须按“先 $z_{\mathcal D}$，后 $z_{\mathcal D}\omega_{\mathcal D}$”的顺序记录；在每个点评处，又必须按该工作域 batch opening 折叠时所使用的多项式顺序记录全部取值。该多项式顺序固定由第 3.3.15 节各工作域商多项式分子中实际出现的对象顺序与验证端域内 batch opening 检查顺序共同决定，禁止实现时重新排序。

6. 商多项式承诺块

	$\mathbf{Com}_{quot} = \big( [t_{FH}],[t_{edge}],[t_{in}],[t_{d_h}],[t_{cat}],[t_C],[t_N] \big).$

7. 域内 batch opening witness 块

	$\mathbf{Open}_{dom} = \big( W_{FH},W_{edge},W_{in},W_{d_h},W_{cat},W_C,W_N \big).$

8. 外点评值批量 opening witness

	$W_{ext}$

	是由第 3.4 节定义的唯一外部批量开放见证。

9. 张量绑定子证明块

	$\Pi_{bind} = \big( \pi_{bind}^{feat}, \pi_{bind}^{hidden,0},\ldots,\pi_{bind}^{hidden,7}, \pi_{bind}^{concat}, \pi_{bind}^{out} \big).$

除上述固定顺序外，证明对象中不允许再插入任何未声明字段。若工程实现为了传输方便使用字节级封装格式，则其解码结果必须与上述有序元组完全一致。

## 4. 验证

### 4.1 输入

验证算法输入为：

- 公共输入

	$(I,src,dst,N,E,d_{in},d_h,C,B);$

- 静态验证键

	$(VK_{KZG},VK_{static},VK_{model});$

- 完整证明对象

	$\pi_{GAT};$

	其中其首字段必须是第 3.5 节定义的元数据块

	$M_{pub};$

- 全部公开量化参数。

正式口径固定为：第 0.8.5 节定义的公开元数据 / 版本元数据**作为证明对象内部字段随** $\pi_{GAT}$ **一并传输**，而不是作为独立于 $\pi_{GAT}$ 之外的额外输入对象单独传递。

### 4.2 公开对象重建

验证者必须先从

$\pi_{GAT}$

中按第 3.5 节规定的固定顺序解析出

$M_{pub}, \mathbf{Com}_{dyn}, \mathbf{S}_{route}, \mathbf{Eval}_{ext}, \mathbf{Eval}_{dom}, \mathbf{Com}_{quot}, \mathbf{Open}_{dom}, W_{ext}, \Pi_{bind}.$

随后验证者必须本地重建：

1. $P_I$；
2. $P_{src},P_{dst}$；
3. $P_{Q_{new}^{edge}},P_{Q_{end}^{edge}},P_{Q_{edge}^{valid}},P_{Q_N},P_{Q_{in}^{valid}},P_{Q_{d_h}^{valid}},P_{Q_{cat}^{valid}},P_{Q_C^{valid}}$；
4. 所有辅助索引多项式；
5. 所有静态表多项式与静态表承诺；
6. 所有模型承诺引用；
7. 第 0.8.5 节定义的全部元数据字段的语义值。

验证者随后必须检查：解析出的

$M_{pub}$

是否与公共输入、静态验证键、模型验证键以及全部公开量化参数一致。特别地，验证者必须先检查

$M_{pub}.dim\_cfg$

与外部公共输入中的

$(N,E,N_{total},d_{in},d_h,d_{cat},C,B)$

完全一致，并检查

$M_{pub}.domain\_cfg$

与参数生成输出中的各工作域长度配置完全一致；若不一致，则无需继续后续检查，直接拒绝。

### 4.3 重新生成全部挑战

验证者必须严格按第 3.2 节的顺序重放 transcript，重新生成：

- $\eta_{feat},\beta_{feat}$；
- 对每个隐藏层注意力头的全部挑战；
- 拼接阶段全部挑战；
- 输出层全部挑战；
- $\alpha_{quot}$；
- 各工作域开放点；
- 各工作域 batch opening 折叠挑战；
- $\rho_{ext}$。

任何顺序不一致、重复吸入、漏吸入或字段缺失都必须直接拒绝。

### 4.4 外点评值检查

验证者检查：

1. 所有外点评值是否与相应承诺、挑战和折叠等式一致；
2. 每个隐藏层注意力头的投影、源注意力、目标注意力、压缩特征、聚合前压缩、ELU 后压缩外点评值是否成立；
3. 拼接阶段压缩外点评值是否成立；
4. 输出层投影、源注意力、目标注意力、压缩输出、最终输出外点评值是否成立。

### 4.5 七个工作域的 quotient identity 检查

验证者不允许使用“分子和”这种占位记号，而必须把第 3.3.15 节右侧的分子逐项代入检查。具体地，在相应开放点上分别检查：

1. 在 $\mathbb H_{FH}$ 上，检查

	$t_{FH}(z_{FH})Z_{FH}(z_{FH}) = \alpha_{quot}^{e_{feat,0}}C_{feat,0}(z_{FH}) + \alpha_{quot}^{e_{feat,1}}C_{feat,1}(z_{FH}) + \alpha_{quot}^{e_{feat,2}}C_{feat,2}(z_{FH}) + \alpha_{quot}^{e_{feat,tbl}}C_{feat,tbl}(z_{FH}) + \alpha_{quot}^{e_{feat,qry}}C_{feat,qry}(z_{FH}).$

2. 在 $\mathbb H_{edge}$ 上，检查

	$\begin{aligned} t_{edge}(z_{edge})Z_{edge}(z_{edge}) ={}& \sum_{r=0}^{7} \Big( \alpha_{quot}^{e_{src,edge,0,r}}C_{src,edge,0}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{src,edge,1,r}}C_{src,edge,1}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{src,edge,2,r}}C_{src,edge,2}^{(r)}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{L,0,r}}C_{L,0}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{L,1,r}}C_{L,1}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{L,2,r}}C_{L,2}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{L,tbl,r}}C_{L,tbl}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{L,qry,r}}C_{L,qry}^{(r)}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{max,bin,r}}C_{max,bin}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{max,zero,r}}C_{max,zero}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{max,0,r}}C_{max,0}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{max,1,r}}C_{max,1}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{max,end,r}}C_{max,end}^{(r)}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{R,0,r}}C_{R,0}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{R,1,r}}C_{R,1}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{R,2,r}}C_{R,2}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{R,tbl,r}}C_{R,tbl}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{R,qry,r}}C_{R,qry}^{(r)}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{exp,0,r}}C_{exp,0}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{exp,1,r}}C_{exp,1}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{exp,2,r}}C_{exp,2}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{exp,tbl,r}}C_{exp,tbl}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{exp,qry,r}}C_{exp,qry}^{(r)}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{psq,0,r}}C_{psq,0}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{psq,1,r}}C_{psq,1}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{psq,end,r}}C_{psq,end}^{(r)}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{ELU,0,r}}C_{ELU,0}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{ELU,1,r}}C_{ELU,1}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{ELU,2,r}}C_{ELU,2}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{ELU,tbl,r}}C_{ELU,tbl}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{ELU,qry,r}}C_{ELU,qry}^{(r)}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{dst,edge,0,r}}C_{dst,edge,0}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{dst,edge,1,r}}C_{dst,edge,1}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{dst,edge,2,r}}C_{dst,edge,2}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{src,qry,r}}C_{src,qry}^{(r)}(z_{edge}) +\alpha_{quot}^{e_{dst,qry,r}}C_{dst,qry}^{(r)}(z_{edge}) \Big) \\ &\quad +\alpha_{quot}^{e_{out,src,edge,0}}C_{out,src,edge,0}(z_{edge}) +\alpha_{quot}^{e_{out,src,edge,1}}C_{out,src,edge,1}(z_{edge}) +\alpha_{quot}^{e_{out,src,edge,2}}C_{out,src,edge,2}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{out,L,0}}C_{out,L,0}(z_{edge}) +\alpha_{quot}^{e_{out,L,1}}C_{out,L,1}(z_{edge}) +\alpha_{quot}^{e_{out,L,2}}C_{out,L,2}(z_{edge}) +\alpha_{quot}^{e_{out,L,tbl}}C_{out,L,tbl}(z_{edge}) +\alpha_{quot}^{e_{out,L,qry}}C_{out,L,qry}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{out,max,bin}}C_{out,max,bin}(z_{edge}) +\alpha_{quot}^{e_{out,max,zero}}C_{out,max,zero}(z_{edge}) +\alpha_{quot}^{e_{out,max,0}}C_{out,max,0}(z_{edge}) +\alpha_{quot}^{e_{out,max,1}}C_{out,max,1}(z_{edge}) +\alpha_{quot}^{e_{out,max,end}}C_{out,max,end}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{out,R,0}}C_{out,R,0}(z_{edge}) +\alpha_{quot}^{e_{out,R,1}}C_{out,R,1}(z_{edge}) +\alpha_{quot}^{e_{out,R,2}}C_{out,R,2}(z_{edge}) +\alpha_{quot}^{e_{out,R,tbl}}C_{out,R,tbl}(z_{edge}) +\alpha_{quot}^{e_{out,R,qry}}C_{out,R,qry}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{out,exp,0}}C_{out,exp,0}(z_{edge}) +\alpha_{quot}^{e_{out,exp,1}}C_{out,exp,1}(z_{edge}) +\alpha_{quot}^{e_{out,exp,2}}C_{out,exp,2}(z_{edge}) +\alpha_{quot}^{e_{out,exp,tbl}}C_{out,exp,tbl}(z_{edge}) +\alpha_{quot}^{e_{out,exp,qry}}C_{out,exp,qry}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{out,psq,0}}C_{out,psq,0}(z_{edge}) +\alpha_{quot}^{e_{out,psq,1}}C_{out,psq,1}(z_{edge}) +\alpha_{quot}^{e_{out,psq,end}}C_{out,psq,end}(z_{edge}) \\ &\quad +\alpha_{quot}^{e_{out,dst,edge,0}}C_{out,dst,edge,0}(z_{edge}) +\alpha_{quot}^{e_{out,dst,edge,1}}C_{out,dst,edge,1}(z_{edge}) +\alpha_{quot}^{e_{out,dst,edge,2}}C_{out,dst,edge,2}(z_{edge}) \end{aligned}$    



3. 在$\mathbb H_{in}$上，检查

	$t_{in}(z_{in})Z_{in}(z_{in})
	=
	\sum_{r=0}^{7}
	\big(
	\alpha_{quot}^{e_{proj,0,r}}C_{proj,0}^{(r)}(z_{in})
	+
	\alpha_{quot}^{e_{proj,1,r}}C_{proj,1}^{(r)}(z_{in})
	+
	\alpha_{quot}^{e_{proj,2,r}}C_{proj,2}^{(r)}(z_{in})
	\big)$ 

4. 在 \($\mathbb H_{d_h}$\) 上，检查

	$\begin{aligned}
	t_{d_h}(z_{d_h})Z_{d_h}(z_{d_h})
	={}&
	\sum_{r=0}^{7}
	\Big(
	\alpha_{quot}^{e_{srcbind,0,r}}C_{srcbind,0}^{(r)}(z_{d_h})
	+
	\alpha_{quot}^{e_{srcbind,1,r}}C_{srcbind,1}^{(r)}(z_{d_h})
	+
	\alpha_{quot}^{e_{srcbind,2,r}}C_{srcbind,2}^{(r)}(z_{d_h})\\
	&\qquad
	+
	\alpha_{quot}^{e_{dstbind,0,r}}C_{dstbind,0}^{(r)}(z_{d_h})
	+
	\alpha_{quot}^{e_{dstbind,1,r}}C_{dstbind,1}^{(r)}(z_{d_h})
	+
	\alpha_{quot}^{e_{dstbind,2,r}}C_{dstbind,2}^{(r)}(z_{d_h})\\
	&\qquad
	+
	\alpha_{quot}^{e_{starbind,0,r}}C_{starbind,0}^{(r)}(z_{d_h})
	+
	\alpha_{quot}^{e_{starbind,1,r}}C_{starbind,1}^{(r)}(z_{d_h})
	+
	\alpha_{quot}^{e_{starbind,2,r}}C_{starbind,2}^{(r)}(z_{d_h})\\
	&\qquad
	+
	\alpha_{quot}^{e_{aggpre,r}}C_{aggpre}^{(r)}(z_{d_h})
	+
	\alpha_{quot}^{e_{agg,r}}C_{agg}^{(r)}(z_{d_h})
	\Big)
	\end{aligned}$      

5. 在 $\mathbb H_{cat}$上，检查

	$t_{cat}(z_{cat})Z_{cat}(z_{cat})
	=
	\alpha_{quot}^{e_{cat,lin}}C_{cat,lin}(z_{cat})
	+
	\alpha_{quot}^{e_{cat,bind}}C_{cat,bind}(z_{cat})
	+
	\alpha_{quot}^{e_{outproj,0}}C_{outproj,0}(z_{cat})
	+
	\alpha_{quot}^{e_{outproj,1}}C_{outproj,1}(z_{cat})
	+
	\alpha_{quot}^{e_{outproj,2}}C_{outproj,2}(z_{cat})$    

6. 在 $\mathbb H_C$上，检查

	$t_C(z_C)Z_C(z_C)
	=
	\alpha_{quot}^{e_{outsrc,0}}C_{outsrc,0}(z_C)
	+
	\alpha_{quot}^{e_{outsrc,1}}C_{outsrc,1}(z_C)
	+
	\alpha_{quot}^{e_{outsrc,2}}C_{outsrc,2}(z_C)
	+
	\alpha_{quot}^{e_{outdst,0}}C_{outdst,0}(z_C)
	+
	\alpha_{quot}^{e_{outdst,1}}C_{outdst,1}(z_C)
	+
	\alpha_{quot}^{e_{outdst,2}}C_{outdst,2}(z_C)
	+
	\alpha_{quot}^{e_{outY}}C_{outY}(z_C)$   

7. 在 $\mathbb H_N$上，检查

	$\begin{aligned} t_N(z_N)Z_N(z_N) ={}& \sum_{r=0}^{7}\bigg( \alpha_{\text{quot}}^{e_{\text{src,node},0,r}}C_{\text{src,node},0}^{(r)}(z_N) +\alpha_{\text{quot}}^{e_{\text{src,node},1,r}}C_{\text{src,node},1}^{(r)}(z_N) +\alpha_{\text{quot}}^{e_{\text{src,node},2,r}}C_{\text{src,node},2}^{(r)}(z_N) \\ &\quad +\alpha_{\text{quot}}^{e_{\text{dst,node},0,r}}C_{\text{dst,node},0}^{(r)}(z_N) +\alpha_{\text{quot}}^{e_{\text{dst,node},1,r}}C_{\text{dst,node},1}^{(r)}(z_N) +\alpha_{\text{quot}}^{e_{\text{dst,node},2,r}}C_{\text{dst,node},2}^{(r)}(z_N) +\alpha_{\text{quot}}^{e_{\text{inv},r}}C_{\text{inv}}^{(r)}(z_N) \\ &\quad +\alpha_{\text{quot}}^{e_{\text{src,tbl},r}}C_{\text{src,tbl}}^{(r)}(z_N) +\alpha_{\text{quot}}^{e_{\text{dst,tbl},r}}C_{\text{dst,tbl}}^{(r)}(z_N)\bigg) \\ &\quad +\alpha_{\text{quot}}^{e_{\text{out,src,node},0}}C_{\text{out,src,node},0}(z_N) +\alpha_{\text{quot}}^{e_{\text{out,src,node},1}}C_{\text{out,src,node},1}(z_N) +\alpha_{\text{quot}}^{e_{\text{out,src,node},2}}C_{\text{out,src,node},2}(z_N) \\ &\quad +\alpha_{\text{quot}}^{e_{\text{out,dst,node},0}}C_{\text{out,dst,node},0}(z_N) +\alpha_{\text{quot}}^{e_{\text{out,dst,node},1}}C_{\text{out,dst,node},1}(z_N) +\alpha_{\text{quot}}^{e_{\text{out,dst,node},2}}C_{\text{out,dst,node},2}(z_N) +\alpha_{\text{quot}}^{e_{\text{out,inv}}}C_{\text{out,inv}}(z_N) \\ &\quad +\alpha_{\text{quot}}^{e_{\text{out,src,tbl}}}C_{\text{out,src,tbl}}(z_N) +\alpha_{\text{quot}}^{e_{\text{out,dst,tbl}}}C_{\text{out,dst,tbl}}(z_N) \end{aligned}$       

对每个工作域，都必须在相应点评集合上完成 KZG batch opening 配对检查。

### 4.6 张量绑定子证明检查

验证者必须逐组检查$\Pi_{bind}$中的全部子证明，至少包括：

1. 特征投影绑定；
2. 八个隐藏层注意力头的：
	- 投影绑定；
	- 源注意力绑定；
	- 目标注意力绑定；
	- 压缩特征绑定；
	- 聚合前压缩绑定；
	- 聚合后压缩绑定；
3. 拼接绑定；
4. 输出层：
	- 输出投影绑定；
	- 输出源注意力绑定；
	- 输出目标注意力绑定；
	- 压缩输出绑定。

每个子证明都必须在其独立域分离标签下验证通过。

### 4.7 接受条件

当且仅当以下条件全部成立时，验证者接受：

1. 全部挑战可按固定顺序重建；
2. 全部外点评值批量 opening 通过；
3. 七个工作域的 quotient identity 全部成立；
4. 七个工作域的 KZG batch opening 全部通过；
5. 全部张量绑定子证明验证通过；
6. 公开输入、静态表、模型键、量化尺度、padding 规则、次数界以及证明对象内部携带的元数据块$M_{pub}$与第 0.8.5 节定义的字段语义彼此一致。

若任一条件不成立，则验证者拒绝。

### 4.8 实现建议

1. 验证实现应先检查公共输入与维度，再重放 transcript，然后先做外点评值检查，再做域内 batch opening，最后做 quotient 与张量绑定检查。
2. 所有工作域数组都必须按完整长度 \(n_{\mathcal D}\) 处理，不能只处理真实区。
3. 所有选择器 \(Q_*\) 都必须由公共输入重建，禁止从证明对象中直接读取。
4. 所有路由公开总和都必须写入证明对象并纳入 transcript。
5. 所有对象名都必须带显式层次和下标，禁止复用单头口径名字。
6. 若实现层为了加速而引入额外缓存列、辅助列或预计算值，只要它们不改变本文约束语义，就可以加入工程代码；但正式证明对象中必须与本文定义的一一对应关系保持一致。
7. 证明对象的序列化与反序列化必须严格遵循第 3.5 节给出的字段顺序，禁止在工程实现中改用“哈希表式”“按名称无序收集式”或依赖语言运行时容器遍历顺序的编码方式。

## 5. 需要特别提醒的实验实现点

1. **不要把“输出层最终聚合 \(Y\)”只写成前向公式而不证明。**
	本文已经显式引入了$\xi_{out},\ Y'^{\star},\ \widehat y^{\star},\ Y^{\star},\ \lambda_{out},\ PSQ^{(out)}$来完成输出聚合证明。实现时不得省掉。
2. **输出层类别共享维域 ($\mathbb H_C$) 不能省。**
	输出层的源注意力、目标注意力和最终输出压缩绑定都依赖类别维 \(C\)，因此不能错误复用 ($\mathbb H_{d_h}$)。
3. **目标路由必须延迟 finalize。**
	因为目标路由绑定的对象要等最大值、分母、逆元、聚合后压缩特征全部生成后才能固定。
4. **拼接阶段不是简单搬运，而是正式约束。**
	必须显式生成 \($P_{H_{cat}}$\) 和 \($P_{H_{cat}^{\star}}$\)，并验证拼接一致性与压缩一致性。
5. **所有“广播对象”都必须有明确定义。**
	如$H_{src}^{\star,edge(r)},\ 
	H_{agg,pre}^{\star,edge(r)},\ 
	H_{agg}^{\star,edge(r)},\ 
	Y'^{\star,edge},\ 
	Y^{\star,edge}$等，都不能只在后文首次使用时临时引入。







