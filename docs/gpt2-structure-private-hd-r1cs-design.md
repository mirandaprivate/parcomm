# zkSMART 复杂模型结构隐藏设计

## Target

为 GPT-2 一类由异构矩阵、Attention、MLP、归一化和查表组成的复杂模型，设计一个与当前全局 HD-R1CS 兼容的结构隐私方案。目标不是只隐藏参数，而是使验证者无法从公开实例和证明转录中恢复真实层数、真实矩阵尺寸、算子分布、路由和参数共享方式，同时继续使用一个全局 HD-R1CS 实例、一个全局 witness 命名空间和一个在线 `C_w`。

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

当前实现能够绑定公开结构，但还不能隐藏结构：`GlobalNormalizationInstance` 公开 manifest、输入输出、lookup tag 列表和全局 shape；架构矩阵使用零 hiding 的确定性 commitment，验证者从公开 manifest 重建并逐项比较；transparent lookup 还直接携带 tag、输入和输出行。因此，当前路径至多隐藏 witness commitment 中的值，不能声称具有模型结构隐私。

真正的结构隐私要求把公开的真实 manifest $M$ 改成公开的容量配置 $U$，把真实 shape、路由和 operator selector 作为本次证明的私有输入。HD-R1CS 后续需要打开 $A_L,A_R,A_O$ 和 $B^{(t)}$ 的 MLE，因此 prover 仍在 proof transcript 中对这些矩阵生成带新鲜随机数的 hiding commitment，但不建立离线 $C_{\mathrm{arch}}$、独立 $C_\theta$、模型注册表或可复用的架构证明。

## Invariant Object

结构隐藏不能改变已经接受的两个系统不变量：

1. 所有层仍属于同一个 **Global HD-R1CS instance**，不能退化为逐层证明。
2. 所有输入相关的中间值仍由同一个在线 **Witness commitment** $C_w$ 绑定，所有 routed witness claims 最终同步约化到一次 $C_w$ opening。

本次证明中的结构 commitment 只是一组临时协议消息：

$$
C_A=\mathsf{Com}(A_L,A_R,A_O,\{B^{(t)}\};\rho_A),
\qquad
C_w=\mathsf{Com}(\theta,w;\rho_w).
$$

模型参数直接属于全局 witness，由同一个 $C_w$ 绑定，不再建立独立 $C_\theta$。routing、selector 和 shape 的必要合法性约束与推理方程放在同一个 proof 中检查，而不是产生可复用的 $\pi_{\mathrm{arch}}$。

## Assumptions

- PCS/commitment 对本次证明中的架构矩阵和全局 witness 同时具有 binding 与 hiding 性质。
- 公开容量配置 $U$ 固定 commitment domain、证明消息数量和每种 operation family 的容量。
- 全部 operation family 每次都进入 transcript，即使某一种在真实模型中的使用量为零。
- 同一次证明约束 routing、selector、active mask、shape descriptor、I/O binding 和 parameter view，而不只证明算术方程。
- lookup argument 是零知识的；当前 transparent tagged lookup 不满足这个条件。
- 若要求对远程 prover 主机也隐藏模型，则还需要 oblivious storage/access 或 TEE。零知识证明自身只能向 verifier 隐藏 witness，不能向执行 prover 代码的主机隐藏内存访问和文件布局。
- 输出本身和多次查询可能支持黑盒模型提取；这属于显式接受的功能泄露，不由 proof transcript 的零知识性消除。

## Notation

- $U$：公开的模型容量桶和固定 instruction set。
- $M$：私有模型 manifest，包括真实拓扑、shape、operation selector、路由和共享关系。
- $K$：公开的最大 gate slot 数。
- $a_j\in\{0,1\}$：第 $j$ 个 slot 是否为真实 gate。
- $z_{j,t}\in\{0,1\}$：第 $j$ 个 slot 是否使用 operation family $t$。
- $A_L,A_R,A_O$：被承诺的全局 routing 矩阵。
- $B^{(t)}$：被承诺的 operation selector 矩阵。
- $C_A$：本次 proof transcript 中对 routing 和 selector 矩阵的临时 hiding commitments。
- $C_w$：同时绑定参数和执行中间值的全局 witness commitment。
- $\mathcal L(M)$：协议明确允许泄露的信息。

## Derivation Strategy

先定义允许泄露的等价类，再修改 statement，而不是从“commitment 看起来是随机的”反推结构已经隐藏。然后用固定容量 slot、私有 active mask、私有 operation selector 和私有 routing 将真实模型嵌入统一程序。所有结构 commitment、结构一致性约束和推理约束都在一次 proof 中完成。

## Derivation Map

1. 定义 verifier 的结构隐私边界和显式 leakage。
2. 把公开 `manifest` 重构为公开 `profile` 与私有 `HiddenArchitectureWitness`。
3. 用固定 slot 与 no-op padding 隐藏真实 gate 数和层数。
4. 用 selector 与固定 operation family 隐藏每种算子的真实数量。
5. 用 shape mask 和容量桶隐藏真实矩阵尺寸，同时保持 virtual zero 和 RAM 可控。
6. 在同一次 proof 中排除任意 routing 和非法 shape。
7. 用固定 transcript schema 和隐藏 lookup 限制 proof-level leakage。

## Main Derivation

### Step 1: 先定义结构隐私，而不是只定义 commitment

对同一公开配置 $U$ 中的两个模型 $M_0,M_1$，如果它们具有相同的显式 leakage，期望验证者视图不可区分：

$$
\mathsf{View}_V(U,C_{A,0},C_{w,0},\pi_0)
\approx_c
\mathsf{View}_V(U,C_{A,1},C_{w,1},\pi_1).
$$

这里比较的不是 commitment 是否包含明文，而是整个观察面。建议把允许泄露函数写成：

$$
\mathcal L(M)=
(\text{profile id},\text{input/output specification},
\text{fixed operation basis},\text{capacity bounds}).
$$

真实 gate 数、各类型 gate 数、真实 shape、routing、参数 sharing、proof 长度和 proof 消息布局都不应超出 $\mathcal L(M)$。prover 时间、RSS 和网络时延是否属于攻击面必须单独声明；若 verifier 能观察它们，就需要做 bucket 化或 padding。

### Step 2: 重构被证明的 statement

当前公开 manifest 版本证明的是：

$$
\exists\theta,w:\quad y=F_M(\theta,x),
$$

其中 $M$ 公开。结构隐藏版本应改成：

$$
\begin{aligned}
\exists M,\theta,w,\rho_A,\rho_w:\quad
&C_A=\mathsf{Com}(\mathsf{Enc}(M);\rho_A)\land
C_w=\mathsf{Com}(\theta,w;\rho_w)\\
&\land\ \mathsf{WellFormed}_U(M)
\land\ y=\mathsf{Exec}_U(M,\theta,x).
\end{aligned}
$$

proof 开始前的公开 statement 是

$$
(U,x,y),
$$

随后 $C_A$ 和 $C_w$ 作为 prover 的首轮消息进入 Fiat--Shamir transcript。这个 statement 有意证明“存在一个容量 $U$ 内的合法隐藏模型产生 $y$”，不声明它是某个预先注册的模型。参数属于 $C_w$，因此不需要单独的 $C_\theta$。

### Step 3: 用固定 slot 隐藏层数和 gate 数

公开 profile 给出 $K$ 个逻辑 gate slot，真实程序通过私有 active bit $a_j$ 占用其中一部分：

$$
a_j(a_j-1)=0.
$$

未使用 slot 是受约束的 no-op/zero slot，而不是在 transcript 中省略：

$$
(1-a_j)\,w_{O,j}=0.
$$

这样真实层数和 gate 数由 $\{a_j\}$ 决定，但证明接口始终表现为 $K$ 个 slot。inactive slot 可以由 virtual zero 表示，不需要物理分配整块 padding。

### Step 4: 用私有 selector 隐藏算子分布

对固定 operation basis $\mathbb T$，每个 slot 使用 one-hot selector：

$$
z_{j,t}(z_{j,t}-1)=0,
\qquad
\sum_{t\in\mathbb T}z_{j,t}=a_j.
$$

在 HD-R1CS 表达中，继续使用论文的 $B^{(t)}$，但所有 $B^{(t)}$ 维度由 $U$ 固定，其非零位置和真实计数被本次证明的 hiding commitment 隐藏。每一种类型的全局方程都必须出现；不能根据 $M^{(t)}=0$ 省掉一次子协议，否则 proof layout 会泄露算子分布。

为了降低 $T$ 并增强隐私，GPT-2 建议公开的不是 `Attention`、`MLP`、`Softmax`、`LayerNorm` 等语义算子，而是少量证明 IR family：

1. structured linear；
2. structured bilinear；
3. hiding tagged lookup；
4. I/O binding。

Softmax 和 LayerNorm 的语义被编译成上述固定 family。lookup tag 也必须是隐藏 witness，且证明其属于认证表；当前 proof 中公开的 `lookup_tags` 和 `RevealedLookupBlock` 必须被替换。

### Step 5: 证明 routing 的合法性

$A_L,A_R,A_O$ 的元素需满足 booleanity。对 active gate，每个需要的输入和输出行恰有一个来源：

$$
\sum_k A_L[j,k]=a_j,
\qquad
\sum_k A_O[j,k]=a_j,
$$

$$
\sum_k A_R[j,k]=a_j\cdot \mathsf{isBinary}_j.
$$

用公开 slot 顺序约束 $A_{L/R}[j,k]=0$ for $k\ge j$，即可在不公开真实边的情况下证明无环。还需要约束每个 output block 的写入唯一性、公共输入只能来自 I/O 区、公共输出确实由输出 selector 选中。仅仅对 $A_L,A_R,A_O$ 做 PCS opening，不能自动推出这些 well-formedness 性质。

### Step 6: 隐藏异构 true shape

profile 为每类 block 给出公开容量 $(R_{\max},C_{\max})$，真实 shape 使用私有 prefix mask $r_{j,p},c_{j,q}$：

$$
r_{j,p+1}(1-r_{j,p})=0,
\qquad
c_{j,q+1}(1-c_{j,q})=0.
$$

这保证 mask 形成连续前缀。容量之外和真实 shape 之外的值必须是 virtual zero：

$$
(1-r_{j,p}c_{j,q})W_j[p,q]=0.
$$

相连 edge 的 shape compatibility 不能由 verifier 从公开 metadata 检查，而要在本次 proof 内检查。实用实现可以给 block 加私有 shape class，并约束 routing 只连接兼容 class；这比为每条边逐坐标比较 shape 更便宜。

物理存储仍只保存 true entries，unused slots 和 padding 均用 virtual zero，因此 verifier privacy 不要求实际分配整个公开容量。代价是 PCS domain 和 transcript 必须按容量固定，proof 生成时间可能仍随非零数量变化。

### Step 7: 在一次 proof 中约束结构与推理

证明接口改为：

$$
(U,M,\theta,x)\mapsto (y,C_A,C_w,\pi).
$$

其中 $M$、$\theta$ 和执行 witness 都是 prover 的私有输入。$C_A$ 和 $C_w$ 使用本次证明的新鲜 hiding randomness。证明 $\pi$ 同时检查：

- active bit 和 operation selector 是合法 partition；
- routing 稀疏、无环且读写规则正确；
- shape mask 合法且每条 edge shape-compatible；
- parameter descriptor、slice/view 和 weight tying 正确；
- I/O selector 与 profile 的公共接口一致；
- 所有隐藏 tag 属于固定 instruction/table registry；
- 无 inactive slot 可以影响公共输出。

然后运行一个全局 HD-R1CS，并把所有 routed witness claims 合并到一次 $C_w$ opening。架构矩阵仍各自需要论文协议规定的 PCS openings；“一次 opening”继续只指全局 witness $C_w$ 的最终 opening。

### Step 8: GPT-2 的推荐公开桶

完全按一个巨大 universal circuit 补齐所有 GPT-2 变体会与 RAM 目标冲突。建议定义少量公开 bucket，例如：

$$
U=(L_{\max},d_{\max},h_{\max},d_{ff,\max},S_{\max},K_t^{\max}).
$$

bucket ID 明确泄露，但 bucket 内隐藏：

- 真实 transformer block 数；
- hidden width、head 数、head width 和 MLP expansion；
- Q/K/V fused 或 separate；
- exact routing 和计算顺序；
- pruning/sparsity 结构；
- parameter tying 和 view；
- 各 primitive family 的真实使用数量。

如果协议只服务标准 GPT-2 Small，而且 `12/768/12/3072` 都公开，那么“结构隐私”的剩余价值很小，重点应放在参数和中间激活隐私。只有需要在同一 verifier 接口下容纳多种规模或私有剪枝/稀疏结构时，隐藏 manifest 才值得增加的合法性证明成本。

## RAM-Controlled Implementation Shape

为了同时满足结构隐私和峰值 RAM 可控，建议采用以下折中：

1. **逻辑域按 bucket 固定，物理存储按 true size。** commitment 与 Fiat--Shamir domain 使用 $U$；prover 只流式读取真实 entries。具体采用 `latex/batchpcs` 的 BVA/TBV 与 bucket-wise EO-folding：padding bucket 仅由 layout 中的 $\varphi^{-1}(i)=\mathsf{null}$ 表示，commit/open/IP/cross-IP/fold 都只遍历 nonzero-bucket map。不能执行 $0\cdot z$ 或给 singleton zero 分配 `[0]`；fold 使用 `None/Some` 分支做恒等化简。传统 LR-folding 会混合 zero/nonzero regions；EO-folding 通过 bit-reversal permutation 与 LR-folding 等价，同时保持 zero buckets 可跳过。
2. **每个 family 固定一个聚合协议入口。** 公开的是 $K_t^{\max}$，不是实际 $M^{(t)}$；空余行由私有 selector 关闭。
3. **结构约束与推理一起流式执行。** 不缓存离线证明，也不物化 inactive slots；必要的 routing/shape 检查会成为每次证明的固定成本。
4. **按 bucket 做 resource padding。** 如果 verifier 可测量远程 proving 时延，只保证落在公开的时间/RSS bucket；没有必要追求逐指令常数时间。
5. **若 prover 主机不可信，另加系统防护。** true-size 文件、内存分配和访问轨迹会直接泄露结构，必须用 TEE、oblivious streaming/ORAM 或把证明放在模型所有者控制的机器上。

## Implementation Plan

### Phase 0: 明确隐私模式

保留现有公开结构路径，并新增显式模式，避免把两种安全语义混在一起：

```rust
enum ArchitectureVisibility {
    PublicManifest,
    PrivateProfile { profile_id: ProfileId },
}
```

公开模式继续由 verifier 从 manifest 重建 zero-hiding architecture commitment；私有模式禁止 verifier 获得 manifest，也禁止调用 `public_architecture_matches`。

### Phase 1: 新增私有架构数据边界

建议新增：

```rust
struct PrivateArchitectureProfile { /* public capacities and fixed families */ }
struct HiddenArchitectureWitness { /* routing, selectors, masks, descriptors */ }
```

私有模式的 architecture artifacts 在每次 `prove` 中使用新鲜随机 hiding commitment，不能为了 verifier 可重建性设为零。参数块直接放入全局 witness，由 $C_w$ 绑定。

### Phase 2: 固定 transcript schema

- 为每个公开 family 总是生成同构 proof section。
- proof 序列化长度只依赖 profile。
- 不公开 actual operation counts、`lookup_tags` 或 revealed lookup rows。
- 不因空 family、inactive slot、短 shape 而省略 transcript 消息。

### Phase 3: 把结构一致性约束加入主证明

按 active/selector、routing、shape、parameter descriptor、I/O 的顺序加入同一个全局 proof，并首先覆盖已有 heterogeneous MLP。GPT-2 compiler 只有在这一层稳定后再接入，避免把“compiler 生成了合法结构”当成 cryptographic proof。

### Phase 4: 替换 transparent lookup

当前 normalization proof 会直接公开 lookup tag、输入和输出，因此应替换成 hiding tagged lookup 或 committed table membership proof。该阶段完成前，不应把 normalization 路径纳入结构私有声明。

### Phase 5: GPT-2 bucket 与实验

先选两个同 bucket、结构不同但 I/O interface 相同的模型，例如不同层数或不同 MLP expansion。验证：

- public instance 字段集合和序列化长度相同；
- proof 长度和每类消息数量相同；
- verifier 只使用 profile、proof 内 commitments、输入和输出；
- 两个合法模型都能证明并验证；
- RSS 随 physical entries 而不是 logical padded entries 增长；
- timing/RSS 的残余差异被记录为显式 side-channel leakage。

## Required Tests

1. 修改 proof 中的 architecture commitment、routing、selector、active bit、shape mask 或 tag 后验证失败。
2. 将一份 proof 与另一个隐藏拓扑或另一组参数拼接后验证失败。
3. public instance 和 API 中不存在 model registry、独立 $C_\theta$ 或离线 $\pi_{\mathrm{arch}}$。
4. inactive slot 的任意非零输出被拒绝。
5. 非法回边、多个来源、shape mismatch 和错误 parameter view 被拒绝。
6. 同一 profile 内两个不同结构的 public instance/proof 序列化长度相同。
7. 某 operation family 实际计数为零时，其 transcript section 仍存在且可验证。
8. proof 中不出现真实维度、真实 gate count、lookup tag 列表或 revealed lookup rows。
9. 同一模型的两次证明使用不同 hiding randomness，但 proof 长度保持相同。
10. 全部 routed online claims 仍只由同一个 `C_w` 和一次最终 witness opening 绑定。
11. instrumentation 必须报告 `materialized_padding_entries = 0`、`padding_ip_field_ops = 0` 和 `padding_fold_field_ops = 0`；测试同时覆盖中间 zero bucket、trailing zero bucket 和 singleton zero/nonzero merge。

## Boundaries

- 这是“同一公开 bucket 内”的结构隐私，不是隐藏 bucket 自身。
- 隐藏结构不等于隐藏模型行为；公开输出和反复查询仍可能泄露模型信息。
- 零知识不隐藏 prover 主机观察到的磁盘布局、allocation、访问轨迹和运行时间。
- 结构 commitment 的 hiding 不能替代主证明中的结构一致性约束。
- 该 statement 只证明“某个合法隐藏模型”完成了推理，不证明它对应一个预先注册或命名的模型；这是移除离线步骤后的有意边界。
- 对标准、完全公开规格的 GPT-2 Small，隐藏拓扑带来的安全收益可能小于工程成本，应先确认威胁模型。

## Open Risks

1. 结构一致性约束可能成为每次证明的主要 prover 成本，需要 sparse/virtual 协议和流式调度。
2. 隐藏 true shape 后，跨 edge shape compatibility 的通用约束可能过重，可能需要有限 shape class 或 typed slot。
3. hiding lookup 的证明大小与 RAM 尚未实验确认。
4. 固定 proof schema 容易做到，固定远程 timing/RSS 只能做到 bucket 级近似。
5. 若公开输出 vocab size 或序列长度，本身可能反推出模型 family；这些字段必须明确列入 $\mathcal L(M)$。
