# GPT-2 基于论文 HD-R1CS 的证明包

## Claim

令 $G$ 是一个采用确定性整数语义的 GPT-2 规格。$G$ 固定 Transformer block 数、Attention 布局、LayerNorm、MLP、参数布局与 sharing、量化 scale、rounding、saturation、causal mask 和输出语义；令 $\theta$ 表示 token/position embeddings、projection matrices、LayerNorm affine values 和 MLP weights 等 learned parameters。记其整数推理函数为

$$
F_G(\theta,x)=y.
$$

令编译器 $\mathsf{Compile}$ 按 `latex/4-hdr1cs.tex` 的定义，把 $(G,\theta)$ 编译为一个矩阵电路及其 HD-R1CS 私有对象

$$
(A_L,A_R,A_O,\{B^{(t)}\}_{t=0}^{T},W),
$$

并生成论文原生 HD-R1CS instance

$$
I_{G,\theta}=
(\mathbb F,n,M,\mathbb T,
C_L,C_R,C_O,\{C^{(t)}\}_{t=0}^{T},C_{\mathrm{io}}).
$$

其中 $C_L,C_R,C_O$ 分别承诺 $A_L,A_R,A_O$，$C^{(t)}$ 承诺 $B^{(t)}$，$C_{\mathrm{io}}$ 承诺由 $B^{(0)}$ 选择出的模型参数、输入和输出块。prover 在论文 Algorithm `alg:hdr1cs` 的 online commitment phase 对完整 witness $W$ 生成 $C_w$。

这里的 $W$ 是 HD-R1CS 的**逻辑 padded witness**。物理 prover 不构造它的 padding entries，而是按 `latex/batchpcs` 把 heterogeneous true-size blocks 组织为 bucketed vector assembly（BVA），并使用 bucket-wise even--odd（EO）folding 和 BatchPCS 生成同一个逻辑 evaluation 的 commitment/opening。

**定理（条件正确性与结构隐私）。** 如果：

1. $I_{G,\theta}$ 中的 commitments 确实由 $\mathsf{Compile}(G,\theta)$ 的输出生成；
2. zkSMART verifier 接受 $I_{G,\theta}$、$C_w$ 和 proof $\pi$；并且
3. 下述 assumptions 成立，

那么除组合 soundness error 外，存在一组与全部 commitments 一致的参数 $\theta$ 和全局 witness $W$，使得被 $C_{\mathrm{io}}$ 绑定的输出满足

$$
y=F_G(\theta,x).
$$

此外，在 composed zkSNARK 的零知识性和 PCS hiding 性下，proof transcript 不泄露 $A_L,A_R,A_O$、$\{B^{(t)}\}$、参数或中间值；它仍然公开论文 instance 中的 $\mathbb F,n,M,\mathbb T$ 和 commitments。若 $x$ 或 $y$ 被声明为 public I/O，它们也属于显式泄露。

本定理不引入独立 $C_\theta$、模型注册协议或可复用架构合法性证明。论文的 $C_L,C_R,C_O,\{C^{(t)}\},C_{\mathrm{io}}$ 就是模型与结构的 binding boundary。

## Status

**PROVABLE AFTER WEAKENING / EXTRA ASSUMPTION**

“证明普通浮点 GPT-2”这一不加限定的命题目前不能成立。论文证明的是有限域上的 HD-R1CS；`latex/8-floating-point.tex` 也明确把目标限定为固定量化语义。因此可证明的命题是：接受的 proof 证明了由已承诺矩阵电路定义的、bit-exact 的整数 GPT-2 推理。

这是一份协议级条件证明，还不是对当前 Rust 代码已经完成 GPT-2 支持的证明。当前全局实现尚无完整 GPT-2 compiler，而且 normalization prototype 使用 transparent lookup，不能满足本证明的完整零知识结论。

## Assumptions

1. **论文 HD-R1CS relation。** $A_L,A_R,A_O\in\{0,1\}^{M\times M}$ 是论文定义的 row-sparse wiring matrices；$B^{(t)}\in\{0,1\}^{M\times M}$ 是按类型分块的 selectors；$B^{(0)}$ 选择 I/O gates，$\{B^{(t)}\}_{t=1}^{T}$ 选择并完备划分 operation gates。
2. **编译器忠实性。** $\mathsf{Compile}$ 为 GPT-2 的每个参数、常量、中间量和输出分配唯一 global block ID，并为每个语义步骤生成且只生成一个等价 primitive gate。所有 edge 都写入 $A_L,A_R,A_O$，所有 gate type 都写入对应的 $B^{(t)}$。
3. **异构 padding 与 BatchPCS 正确性。** 每个真实 $r\times c$ 矩阵在 HD-R1CS 语义中以固定布局嵌入 $n\times n$ block；block 外元素由 virtual-zero 语义固定为零。物理表示只保存 true entries，并按 `latex/batchpcs/5-bva.tex` 的 BVA layout 记录 bucket sizes 和映射 $\varphi$。commit、open、projection、inner product、cross-inner-product 和 folding 都按 nonzero bucket 执行：不分配 padding entries，也不对它们执行 field multiplication/addition。
4. **确定性整数语义。** 每个 scale、zero point、rounding mode、truncation、clamp、LayerNorm epsilon、Softmax clipping 和 causal-mask convention 都由 $G$ 唯一确定。
5. **无 field wrap-around。** 所有整数输入、中间累加器和辅助 witness 都有经证明的范围，使其到 $\mathbb F$ 的编码是单射；因此 field equality 可以解释为目标整数 equality。
6. **非线性 lookup 正确性。** GELU、exponential、reciprocal、inverse square root、range 和必要的 requantization tables 是确定性、带 domain separation tag 的认证表；lookup proof 同时证明 key 和 value 的 membership。
7. **原子协议 soundness。** matrix multiplication、addition、Hadamard、lookup、sumcheck、Kronecker reduction、diagonal-block reduction、blockwise multiplication，以及 BatchPCS 的 BVA/TBV/EO-folding argument 均对相应关系 sound。
8. **PCS security。** 对 $A_L,A_R,A_O,B^{(t)},C_{\mathrm{io}}$ 所承诺的 I/O blocks 和 $W$ 使用的 PCS 是 knowledge-sound、binding 且 hiding 的。
9. **组合与 transcript。** 论文 `latex/5-ssm.tex` 的 PIOP+PCS composition theorem 适用；Fiat--Shamir challenges 在相关 commitment 之后生成，并对不同 equation 和 subprotocol 做 domain separation。
10. **公开 I/O binding。** 如果 verifier 需要检查给定明文输入 $x$ 或输出 $y$，I/O equation 还证明 $C_{\mathrm{io}}$ 中相应 block 等于该公开值。若没有这一步，定理只证明 committed input 到 committed output 的计算。

## Notation

- $L$：Transformer block 数。
- $S$：sequence length。
- $d$：hidden width。
- $h$：attention head 数，$d_h=d/h$。
- $d_{ff}$：MLP hidden width。
- $W$：论文中的一个全局 witness matrix，包含参数、输入、输出、中间量和辅助 witness blocks。
- $W_i$：global block ID 为 $i$ 的 $n\times n$ padded witness block。
- $N_i$：第 $i$ 个 heterogeneous block 的 true physical entry 数。
- $q$：组成全局 heterogeneous witness 的 true-size vectors 数量。
- $N=\sum_iN_i$：全部实际保存的 witness entries 数。
- $K$：BVA 的逻辑长度；BatchPCS layout 保证 $K<8N$。
- $\varphi$：从 nonzero/true-size vectors 到 BVA buckets 的位置映射；zero bucket 的逆映射为 $\mathsf{null}$。
- $D_j$：EO-LiteBullet 第 $j$ 轮实际存储的 partial map `bucket_id -> nonzero vector`；不在 $\operatorname{dom}(D_j)$ 中的 bucket 在逻辑上为零。
- $A_L,A_R,A_O$：论文中的三张 wiring matrices。
- $B^{(t)}$：论文中选择 type-$t$ gates 的矩阵。
- $W^{(t)}_{L/R/O}$：经论文 Eq. `hdr1csrotate` routing 后的 type-$t$ operands/results。
- $\mathbb T_{\mathrm{GPT2}}$：公开 primitive operation set。
- $\Phi$：一个带 tag 的确定性 lookup relation。
- $\operatorname{RQ}$：由 $G$ 固定的 integer rescale、rounding 和 clamp 关系。
- $\epsilon_{\mathrm{HD}}$：所有 HD-R1CS operation PIOP 的总 soundness error。
- $\epsilon_{\mathrm{PCS}}$：全部 PCS openings 的总 extraction/soundness error。
- $\epsilon_{\mathrm{FS}}$：Fiat--Shamir 与随机多项式检测的总 error。

为了直接复用论文的 operation-wise batching，取

$$
\mathbb T_{\mathrm{GPT2}}
=\{\mathsf{mul},\mathsf{add},\mathsf{had},\mathsf{lookup}\},
\qquad T=4.
$$

因此 GPT-2 proof 仍严格包含论文定义的 $T+1=5$ 个全局 HD-R1CS equations：一个 I/O equation 和四个 operation-specific equations。Attention、MLP、LayerNorm 或 Softmax 不是额外的顶层 proof；它们被编译到这四种 primitive types。

## Proof Strategy

采用“编译正确性 + relation soundness + 拓扑归纳”的直接证明：

1. 先证明任意 GPT-2 tensor shape 都能用论文的逻辑 $n\times n$ block 表示，同时用 BVA/TBV 与 EO-folding 保证 padding zeros 从不 materialize。
2. 把 GPT-2 每个语义算子分解到 $\mathbb T_{\mathrm{GPT2}}$。
3. 用论文的 $A_L,A_R,A_O$ 和 $B^{(t)}$ 证明所有 primitive gates 从同一个 $W$ 读取并写回同一个 global block namespace。
4. 由四个 operation equations 推出每个 primitive gate 正确。
5. 按 GPT-2 DAG 的拓扑顺序归纳，推出最终输出正确。
6. 最后调用论文 PIOP+PCS composition theorem 得到知识可靠性、succinctness 和零知识性。

## Dependency Map

1. 主定理依赖 Lemma 1（异构 block 嵌入与非物化 padding）、Lemma 2（GPT-2 primitive decomposition）、Lemma 3（HD-R1CS routing）、Lemma 4（per-type equation soundness）和 Lemma 5（全局拓扑组合）。
2. LayerNorm 正确性依赖 reduction、Hadamard square、inverse-square-root lookup、affine transform 和 requantization。
3. Attention 正确性依赖 Q/K/V matmul、causal mask、maximum/range certificate、exponential/reciprocal lookup、Hadamard scaling 和 value matmul。
4. MLP 正确性依赖两个 affine matmul、GELU lookup、requantization 和 residual addition。
5. 模型结构 hiding 依赖 PCS hiding 与 composed proof 的 zero knowledge，但允许泄露 $(\mathbb F,n,M,\mathbb T)$。
6. “这是指定的 GPT-2”依赖 Assumption 2，即 commitments 来自 $\mathsf{Compile}(G)$；论文协议本身只证明 committed HD-R1CS instance 的 satisfiability。

## Proof

### Step 1: 异构 block 嵌入与 padding 非物化

令

$$
n\ge \max\{S,d,d_h,d_{ff},|V|\text{ 的分块宽度}\}
$$

或令实现使用逻辑上等价的分桶 blocks。对任意 $X\in\mathbb F^{r\times c}$，$r,c\le n$，定义

$$
\iota_{r,c}(X)=
\begin{bmatrix}
X & 0\\
0 & 0
\end{bmatrix}
\in\mathbb F^{n\times n}.
$$

若 $A\in\mathbb F^{r\times k}$、$B\in\mathbb F^{k\times c}$，则由普通 block multiplication，

$$
\iota_{r,k}(A)\,\iota_{k,c}(B)=\iota_{r,c}(AB).
$$

等式成立是因为所有涉及 padding row/column 的求和项至少含一个零因子。由 Assumption 3，padding 在逻辑 relation 中恒为零。因此论文对 $n\times n$ blocks 的 multiplication equation 同时证明 GPT-2 中的矩形 Q/K/V、attention、MLP 和 LM-head products。

接下来证明这些逻辑零不需要进入物理 witness。将每个 block 的 true entries 表示为向量 $a_i\in\mathbb F^{N_i}$，总 true size 为

$$
N=\sum_{i=1}^{q}N_i.
$$

按 `latex/batchpcs/5-bva.tex`，先把每个 $N_i$ 向上对齐到 power-of-two bucket，再插入满足 TBV alignment 的概念性 zero buckets。所得 assembly 记为

$$
\mathcal A=
\left(K,q,I,\{(N_i,a_i)\}_{i=1}^{q},
\{k_j\}_{j=1}^{I},\varphi\right),
$$

且满足

$$
K<8N,
\qquad
I\le q+\log_2N+1.
$$

对一个 padding bucket $j$，layout 只记录

$$
\varphi^{-1}(j)=\mathsf{null};
$$

内存中没有对应 field-element array。若 $\zeta$ 是 routing reduction 产生的 matching TBV，则 BatchPCS Eq. `ipbvatbv` 给出

$$
\langle\mathcal A,\zeta\rangle
=\sum_{i=1}^{q}
\eta_{\varphi(i)}
\cdot\widetilde a_i(z_{\varphi(i)}).
$$

右侧只遍历 $q$ 个 true-size vectors；zero buckets 不出现在求和中。`latex/batchpcs/app-pcs.tex` 的 bucket-wise commitment/opening 同样只遍历 nonzero buckets，并根据 $\varphi(i)$ 计算每个 true entry 对应的逻辑 generator/evaluation weight。因此生成的 commitment 和 opening 与显式展开长度 $K$ 的 padded vector 相同，但 prover 不读取、写入或分配任何 padding element。

更强的不变量是：在 EO-LiteBullet 的每一轮 $j$，prover 只持有 partial map $D_j$。对 matching TBV $\zeta_j$，inner product 的实现定义为

$$
\mathsf{IP}(D_j,\zeta_j)
=
\sum_{b\in\operatorname{dom}(D_j)}
\eta_b\,\widetilde{D_j[b]}(z_b).
$$

这个循环的 index set 是 $\operatorname{dom}(D_j)$，而不是全部 logical buckets。因而它不会计算 $0\cdot\eta_b$，不会读取 zero bucket，也不会为 zero bucket 调用 projection/folding。EO-LiteBullet 每轮需要的两个 cross terms 也使用同一规则，只遍历 split 后实际存在的 nonzero buckets。

令 $\bot$ 表示不存在的 bucket，而不是一个包含零元素的 vector。fold transition 用以下四个分支实现：

$$
\begin{aligned}
\mathsf{Fold}_y(\bot,\bot)&=\bot,\\
\mathsf{Fold}_y(u,\bot)&=u,\\
\mathsf{Fold}_y(\bot,v)&=y v,\\
\mathsf{Fold}_y(u,v)&=u+yv.
\end{aligned}
$$

前两个分支不执行任何与零有关的 field operation；第三个分支只缩放实际存在的 $v$；第四个分支只处理两个实际 vectors。commitment accumulator 同理从群单位元开始，只遍历 $\operatorname{dom}(D_j)$，不会计算零 bucket 对应的 scalar multiplication。

因此，对所有 padding coordinates，物理 prover 执行的 witness-field乘法次数和加法次数均为

$$
\#\mathsf{Mul}_{\mathrm{padding}}
=\#\mathsf{Add}_{\mathrm{padding}}=0.
$$

padding 只产生 bucket length、offset 和 `null` 标记等 metadata 操作。`latex/batchpcs/algorithm/alg-bvaop.tex` 中 singleton 分支写出的 `[0]` 必须作为 $\bot$ 的数学记号实现，不能分配成一元素 field vector。

这里必须使用 BatchPCS 的 EO-folding，而不是直接在原 layout 上使用传统 LR-folding。LR-folding把相距半个向量的元素配对，会反复混合 zero 与 nonzero regions；EO-folding只配对相邻元素，使长度至少为 $2$ 的 zero bucket folding 后仍是 zero bucket。`latex/batchpcs/4-tbv.tex` 的 Proposition `eolr` 证明

$$
\mathsf{EOFold}(a)
=\Gamma_{\mu-1}
\left(\mathsf{LRFold}(\Gamma_\mu(a))\right),
$$

其中 $a\in\mathbb F^{2^\mu}$，$\Gamma_\mu$ 是 $\mu$-bit reversal permutation。因此 EO protocol 与在置换向量上运行 LR protocol 具有相同 completeness 和 soundness，但前者允许 prover 跳过 zero buckets。其物理 RAM 为

$$
O(N+q\log N),
$$

并且不会分配长度 $K$ 的 padded-coordinate buffer，更不会分配 homogeneous 展开后的 $Mn^2$ entries。padding 只贡献 $O(I)$ layout metadata，不贡献 witness IP/folding arithmetic。这证明了“HD-R1CS 逻辑上 padding、BatchPCS 物理上既不 materialize padding，也不显式计算 padding”同时成立。

### Step 2: GPT-2 前向计算的 primitive decomposition

对 token IDs $x=(x_1,\ldots,x_S)$，embedding 输出为

$$
H^{(0)}=S_xE_{\mathrm{tok}}+S_{\mathrm{pos}}E_{\mathrm{pos}},
$$

其中 $S_x$ 和 $S_{\mathrm{pos}}$ 是 row-selection matrices。公开 token IDs 时 $S_x$ 由公开输入唯一决定；私有 token IDs 时它属于 committed I/O 并需额外证明 one-hot 与 token binding。两次乘法属于 $\mathsf{mul}$，求和属于 $\mathsf{add}$。

对第 $\ell\in\{0,\ldots,L-1\}$ 个 pre-LayerNorm GPT-2 block，定义

$$
U^{(\ell)}=\operatorname{LN}_1(H^{(\ell)}),
$$

$$
Q_h=U^{(\ell)}W_{Q,h},\quad
K_h=U^{(\ell)}W_{K,h},\quad
V_h=U^{(\ell)}W_{V,h},
$$

$$
Z_h=\operatorname{Softmax}
\left(\operatorname{RQ}(Q_hK_h^\top)+\mathsf{Mask}\right)V_h,
$$

$$
\widetilde H^{(\ell)}
=H^{(\ell)}+
\operatorname{Concat}(Z_1,\ldots,Z_h)W_O+b_O,
$$

$$
R^{(\ell)}=\operatorname{LN}_2(\widetilde H^{(\ell)}),
$$

$$
H^{(\ell+1)}=
\widetilde H^{(\ell)}+
\operatorname{GELU}(R^{(\ell)}W_1+b_1)W_2+b_2.
$$

所有 dense products 属于 $\mathsf{mul}$；bias、mask 和 residual 属于 $\mathsf{add}$；逐元素乘法、square 和 selector products 属于 $\mathsf{had}$；GELU、exponential、reciprocal、inverse square root 及 range/requantization 的非线性部分属于带 domain tag 的 $\mathsf{lookup}$。head split、transpose、concat 和 reshape 使用固定 block layout 或固定 permutation matrix multiplication，不引入新的 operation type。

最后，

$$
H_{\mathrm{out}}=\operatorname{LN}_f(H^{(L)}),
\qquad
Y=H_{\mathrm{out}}E_{\mathrm{tok}}^\top,
$$

其中标准 GPT-2 的 LM head 与 token embedding weight tying 由同一个 global block ID 表示，而不是复制一份参数。至此，GPT-2 的完整前向图被分解到 $\mathbb T_{\mathrm{GPT2}}$。

### Step 3: LayerNorm 可由四种 primitive types 精确表示

对一行 $z=(z_1,\ldots,z_d)$，避免在 field 中直接除以 $d$，引入

$$
s=\sum_{i=1}^{d}z_i,
\qquad
c_i=d z_i-s,
$$

$$
q_i=c_i^2,
\qquad
v=\sum_{i=1}^{d}q_i+\epsilon_G.
$$

sum 和 broadcast 是乘以固定 all-ones/selection matrix 后的 $\mathsf{mul}$，减法是带负常数的 $\mathsf{add}$，square 是 $\mathsf{had}$。认证 lookup 证明

$$
(\tau_{\mathrm{isqrt}},v,r)\in\mathcal T_{\Phi}.
$$

随后 $c_i r$、乘以 $\gamma_i$ 属于 $\mathsf{had}$，加 $\beta_i$ 属于 $\mathsf{add}$，$\operatorname{RQ}$ 由 Assumptions 4--6 的量化约束证明。因此所有 LayerNorm outputs 等于 $G$ 定义的整数 LayerNorm。这里没有新增 `LayerNorm` HD-R1CS equation。

### Step 4: Softmax 可由四种 primitive types 精确表示

对一个 causal attention row 的 eligible scores $s_i$，令 $m$ 为 row maximum，并引入 one-hot witness $e_i$：

$$
\sum_i e_i=1,
\qquad
e_i(e_i-1)=0,
\qquad
e_i(m-s_i)=0.
$$

range lookup 证明所有 $m-s_i$ 非负。于是至少有一个 $s_i=m$ 且没有 $s_i>m$，所以 $m=\max_i s_i$。causal mask 保证 masked positions 不进入 eligible set。

接着证明

$$
(\tau_{\exp},s_i-m,u_i)\in\mathcal T_{\Phi},
\qquad
z=\sum_i u_i,
$$

$$
(\tau_{\mathrm{recip}},z,r)\in\mathcal T_{\Phi},
\qquad
\bar p_i=\operatorname{RQ}(u_i r).
$$

令 $S=2^q$ 为 manifest 固定的整数概率单位。逐元素 requantization
通常会因取整损失质量，因此还必须证明

$$
\delta=S-\sum_i\bar p_i,
\qquad
(\tau_{\mathrm{range}},\delta,\delta)\in\mathcal T_{\Phi},
$$

并复用上面已经受约束的 argmax one-hot witness：

$$
c_i=e_i\delta,
\qquad
p_i=\bar p_i+c_i,
\qquad
\boxed{\sum_i p_i=S}.
$$

因此 correction 只能进入一个已证明为 row maximum 的位置，同时最终定点
Softmax 行严格加总为 $S$（把整数解释成 Q$q$ 后即为 1）。linear sums 和
shifts 使用 $\mathsf{mul}/\mathsf{add}$，booleanity、correction 和其它 products
使用 $\mathsf{had}$，exp/reciprocal/range 使用 $\mathsf{lookup}$。因此 $p$ 等于
$G$ 固定的、带确定性 argmax residual correction 的整数 Softmax，而不是实数
无限精度 Softmax。这个结论同样不需要新增 `Softmax` equation。

### Step 5: 论文 routing equations 绑定全局数据流

对每个 $t\in[T]$，论文 Eq. `hdr1csrotate` 定义

$$
W_L^{(t)}=[(B^{(t)}A_L)\otimes I_n]W,
$$

$$
W_R^{(t)}=[(B^{(t)}A_R)\otimes I_n]W,
\qquad
W_O^{(t)}=[(B^{(t)}A_O)\otimes I_n]W.
$$

由 Assumption 1，每个 active operation row 在 $A_L$ 和 $A_O$ 中有唯一选择；binary gate 在 $A_R$ 中也有唯一选择。乘以 $A_{L/R/O}\otimes I_n$ 因而把同一个 $W$ 中被选择的 source/output blocks 移到相应 gate row。再乘 $B^{(t)}\otimes I_n$ 删除其它类型的 rows。

所以任一 Transformer layer 的输出若在下一层被读取，两次引用都指向同一个 $W_i$。不存在逐层独立 witness，也不需要 layer-boundary equality proof。这正是一个 global HD-R1CS instance 的跨层一致性。

### Step 6: 五个全局 HD-R1CS equations 推出全部 local gates 正确

I/O equation 证明

$$
C_{\mathrm{io}}
\text{ opens to }
(B^{(0)}\otimes I_n)W.
$$

因此参数、输入和最终输出与全局 $W$ 中的对应 blocks 相同，不需要独立 $C_\theta$。

四个 operation equations 分别为：

$$
\mathsf{Diag}_n(W_L^{(\mathsf{mul})})
\mathsf{Diag}_n(W_R^{(\mathsf{mul})})
=\mathsf{Diag}_n(W_O^{(\mathsf{mul})}),
$$

$$
W_L^{(\mathsf{add})}+W_R^{(\mathsf{add})}
=W_O^{(\mathsf{add})},
$$

$$
W_L^{(\mathsf{had})}\circ W_R^{(\mathsf{had})}
=W_O^{(\mathsf{had})},
$$

以及对所有 selected nonzero entries 的 tagged membership：

$$
(\tau_j,W_{L,j}^{(\mathsf{lookup})},W_{O,j}^{(\mathsf{lookup})})
\in\mathcal T_{\Phi}.
$$

每个等式在其它 type rows 上都是 $0=0$，在 $B^{(t)}$ 选择的 rows 上正好是相应 local gate relation。Assumptions 6 和 7 保证接受相应 PIOP 意味着所有 selected local relations 同时成立，除其 soundness error 外。

量化 multiplication 按论文 `latex/8-floating-point.tex` 的方式，在 multiplication result 上施加 committed scaling vector 和

$$
\operatorname{Clamp}_k
\left(\operatorname{Trunc}_{2k}
(\theta\cdot(A B))\right).
$$

Assumptions 4--6 使该 relation 等于 $G$ 指定的 integer rescaling，而不是 field division。

### Step 7: 每个 GPT-2 macro operation 正确

Step 3 与 Step 4 已证明 LayerNorm 和 Softmax 的 primitive decomposition 正确。由 Step 6：

- embedding selection 与 addition 正确；
- 每个 Q/K/V projection、$QK^\top$、probability-value product 和 output projection 正确；
- causal mask、residual 和 bias additions 正确；
- 两个 MLP affine maps 与 GELU lookup 正确；
- final LayerNorm 和 tied LM-head multiplication 正确。

每次 requantization 都受相同的 scale/range/rounding constraints 约束；由 Assumption 5，不存在通过 field wrap-around 满足等式但违反整数语义的 assignment。

### Step 8: 按全局 GPT-2 DAG 进行归纳

按依赖关系拓扑排序全部 global block IDs。I/O equation 绑定参数、输入、常量，并指定被承诺的 output block。参数、输入和常量构成归纳的 base blocks；output block 是归纳最终需要到达的目标，不作为 base 假设。

归纳假设某一 operation gate 的全部 predecessor blocks 已等于 $F_G$ 在该位置的确定性值。由 Step 5，该 gate 的 $W_{L/R}$ 正是这些 predecessor blocks；由 Step 6 和 Step 7，其 $W_O$ 等于对应 primitive 或 macro semantic function 的唯一输出。因此该 output block 也等于 $F_G$ 的确定性值。

对拓扑序中的所有 gates 应用有限归纳，得到 final LayerNorm 和 LM head 的 output block 等于 $F_G(\theta,x)$。I/O equation 又把该 block 与 $C_{\mathrm{io}}$ 中的 output 绑定；在 Assumption 10 的 public-output 模式下，它进一步等于 verifier 提供的 $y$。所以

$$
y=F_G(\theta,x).
$$

### Step 9: 从 PIOP acceptance 得到 knowledge soundness

论文的 recursive sumcheck、blockwise multiplication、Kronecker reduction 和 diagonal-block reduction，把上述五个 equations 约化为 $A_L,A_R,A_O,B^{(t)}$ 和逻辑 $W$ 的 MLE evaluation claims。Step 1 的 BVA/TBV reduction 再把 heterogeneous witness claims 合并成只访问 true entries 的 BatchPCS opening。由 Assumption 8，接受全部 PCS openings 时存在与 commitments 一致的 extracted matrices；由 Assumption 7，这些 matrices 满足五个 HD-R1CS equations。

对所有 subprotocol failure events 使用 union bound，整体错误至多为

$$
\epsilon_{\mathrm{HD}}
+\epsilon_{\mathrm{PCS}}
+\epsilon_{\mathrm{FS}}.
$$

结合 Step 8，即得条件正确性结论。

### Step 10: 结构隐私

$A_L,A_R,A_O$ 的非零位置编码 GPT-2 wiring，$B^{(t)}$ 编码每个 gate 的 operation type；这些对象不出现在 public instance 中，verifier 只接收 $C_L,C_R,C_O,\{C^{(t)}\}$。参数和中间值分别只通过 $C_{\mathrm{io}}$、$C_w$ 及其零知识 openings 出现。

由 Assumption 8 的 hiding 和 Assumption 9 的 zkSNARK composition theorem，存在 simulator 在不知道这些私有 matrices 的情况下生成与真实 transcript 不可区分的视图。因此 transcript 不泄露 committed wiring、type partition、参数和 intermediate values。

公开的 $n$、总 gate count $M$、operation set $\mathbb T_{\mathrm{GPT2}}$、commitment 数量以及被明确公开的输入输出不被隐藏。故隐私结论严格是“隐藏论文 commitments 的内容”，而不是隐藏 HD-R1CS instance 的公开尺寸。

由 Steps 1--10，定理成立。$\square$

## Corrections or Missing Assumptions

- 原始“对 GPT-2 的证明”必须解释为固定量化/整数 GPT-2，而不能解释为 PyTorch 浮点执行或模型精度证明。
- 论文把 $C_L,C_R,C_O,\{C^{(t)}\},C_{\mathrm{io}}$ 作为 instance 输入；本证明沿用该结构，没有增加独立 $C_\theta$ 或架构合法性 proof。
- 若 commitments 不是由受信 compiler 对指定 $(G,\theta)$ 生成，zkSMART 只能证明某个 committed HD-R1CS instance 可满足，不能自行证明该 instance 就是声称的 GPT-2。
- 论文 homogeneous $n\times n$ relation 延伸到 heterogeneous true shapes 需要 Assumption 3 的逻辑 zero-padding lemma；物理 commitment/opening/IP 必须使用 BVA/TBV 和 bucket-wise EO-folding，不能先构造 padded vector，也不能把伪代码中的 `[0]` 实现为 field element。
- tagged union lookup 是对论文单一 $\phi$ lookup 的 domain-separated 实例化；其 table encoding 必须进入 compiler specification。
- 当前 Rust normalization prototype 的 transparent lookup 会公开 lookup rows，因此尚不满足 Step 10 的完整零知识结论。

## Open Risks

- 完整 GPT-2 compiler 尚未实现，Assumption 2 目前没有端到端代码证据。
- LayerNorm、Softmax、requantization 和 range gadgets 的 bit-exact reference matching 仍需逐算子与端到端测试。
- BVA 的 dimension commitments、bucket count 和 proof serialization 是否泄露 true shapes 需要单独审计；“不 materialize padding”是 RAM 性质，不自动等价于 shape hiding。
- full-context attention 的 witness 是 $O(LS^2)$ 规模；其峰值 RSS 需要在不改变 global HD-R1CS statement 的条件下用 streaming/replay 控制。
- 若 lookup tables、tag 列表或每类 proof 消息数量按实际模型变化，operation distribution 会从 transcript side channel 泄露；它们必须由公开 $\mathbb T$ 和容量配置固定。
