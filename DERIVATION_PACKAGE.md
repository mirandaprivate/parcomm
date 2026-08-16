# Derivation Package: Minimal HD-R1CS equations for GPT-2 normalization

## Target

Determine whether GPT-2 Softmax and LayerNorm can be merged while minimizing
the number of global HD-R1CS equation families.

The target is an exact compiler decomposition for a manifest-defined integer
GPT-2 execution. It is not an approximation that replaces Softmax by
LayerNorm, or vice versa.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

Softmax and LayerNorm cannot be merged into one mathematical function. They
can, however, be compiled into the same three global HD-R1CS operation-equation
families, which are the recommended practical minimum:

1. a structured linear equation;
2. a structured bilinear equation; and
3. one tagged lookup equation.

Together with the separate I/O-binding equation, the proposed GPT-2 relation
therefore needs four top-level equation groups. This count assumes that range,
comparison, selection, and requantization are decomposed into linear and
bilinear constraints, and that element-wise products are represented as
structured diagonal views in the bilinear equation. The normalization prototype
now implements these capabilities for one LayerNorm row and one Softmax row. It
uses separate backend reductions for ordinary matrix products and
virtual-diagonal Hadamard products while treating both as one algebraic
bilinear family. Extending the compiler over the complete GPT-2 graph remains
future work.

An aggressive algebraic encoding can absorb the structured linear equation
into the bilinear equation using homogeneous or concatenated blocks. That
leaves two operation equations (universal bilinear and tagged lookup) plus I/O.
It is not the recommended RAM-controlled design because it increases augmented
blocks, routing, and projection work merely to remove one inexpensive linear
equation.

## Invariant Object

The invariant object is the global residual family over one routed witness:

$$
R_{\mathrm{lin}}(w)=0,\qquad
R_{\mathrm{bil}}(w)=0,\qquad
R_{\mathrm{lut}}(w)=0.
$$

Semantic operator names do not determine the equation grouping. Every
intermediate value has a global block ID, and every local transition is routed
to one of the three residual families.

## Assumptions

- The manifest fixes an integer execution semantics, including all scales,
  rounding, saturation, epsilon values, masks, and lookup-table domains.
- All intermediate signed values have proved bounds that prevent field
  wrap-around.
- Routing and true shapes are public or committed, and a global block ID refers
  to the same witness value in all equation families.
- The linear equation supports public structured maps, including reductions,
  broadcasts, permutations, and bit recomposition.
- The bilinear equation supports matrix multiplication and virtual diagonal
  views without materializing dense diagonal matrices.
- A tagged lookup argument proves membership in one authenticated disjoint
  union of nonlinear tables.
- The lookup key contains the operation tag, so an exponential row cannot be
  reused as a reciprocal or inverse-square-root row.
- Fiat--Shamir batching and the final witness opening are sound.

## Notation

- $x_i$: one input row element.
- $d$: true LayerNorm row width.
- $s$: LayerNorm row sum.
- $c_i$: centered LayerNorm numerator.
- $v$: LayerNorm variance statistic in the manifest's integer scale.
- $m$: maximum eligible Softmax score in one row.
- $e_i$: integer exponential lookup output.
- $z$: Softmax denominator.
- $r$: reciprocal or inverse-square-root lookup output.
- $\operatorname{RQ}_M$: manifest-defined integer requantization.
- $\tau$: authenticated lookup tag.
- $b_j$: a bit witness constrained by $b_j(b_j-1)=0$.

## Derivation Strategy

First derive exact integer dataflows for LayerNorm and Softmax. Then classify
each transition as linear, bilinear, lookup, or range/control. Finally compile
range/control constraints into the linear and bilinear families and combine all
nonlinear tables with tagged lookup keys.

The equation count is minimized at the proof-IR level, not by pretending that
the two semantic operators are identical.

## Derivation Map

1. LayerNorm reduces to sums, centering, squares, inverse-square-root lookup,
   affine products, and requantization.
2. Softmax reduces to maximum certification, shifting, exponential lookup,
   sum, reciprocal lookup, products, and requantization.
3. Sums, shifts, broadcasts, routing, and bit recomposition enter the global
   linear equation.
4. Squares, products, booleanity, and selector products enter the global
   bilinear equation.
5. Exponential, reciprocal, and inverse square root enter one tagged lookup
   equation.
6. Hence neither a `Softmax` nor a `LayerNorm` HD-R1CS operation type is needed.

## Main Derivation

### Step 1: LayerNorm as primitive relations

Avoid field division by $d$. For a row of true width $d$, define

$$
s=\sum_{i=1}^{d}x_i,
\qquad
c_i=d x_i-s.
$$

These are exact linear identities. Next define

$$
q_i=c_i^2,
\qquad
v=\sum_{i=1}^{d}q_i+\epsilon_M,
$$

where $\epsilon_M$ includes the manifest's required integer rescaling. The
squares are bilinear; the reduction and epsilon addition are linear.

Use a tagged lookup

$$
(\tau_{\mathrm{isqrt}},v,r)
\in T_{\mathrm{norm}}
$$

to bind $r$ to the manifest-defined inverse-square-root approximation. For the
affine output, introduce $p_i$ and an integer requantization witness:

$$
p_i=\gamma_i c_i,
\qquad
y_i=\operatorname{RQ}_M(p_i r+\beta_i).
$$

The two products are bilinear after introducing an intermediate for $p_i r$;
the bias is linear. The quotient, remainder, rounding, and clamp relations of
$\operatorname{RQ}_M$ are handled in Step 3.

### Step 2: Softmax as primitive relations

For every eligible position in one causally masked row, define

$$
u_i=x_i-m.
$$

To prove that $m$ is a maximum, prove $m-x_i\ge 0$ for all eligible $i$ and
prove that at least one eligible input equals $m$. One exact encoding uses
one-hot witnesses $h_i$:

$$
\sum_i h_i=1,
\qquad
h_i(h_i-1)=0,
\qquad
h_i(m-x_i)=0.
$$

The inequalities are range decompositions. The first relation and bit
recomposition are linear; booleanity and the zero-selector relation are
bilinear.

Bind the exponential and reciprocal values through the same tagged table:

$$
(\tau_{\exp},u_i,e_i)\in T_{\mathrm{norm}},
\qquad
z=\sum_i e_i,
\qquad
(\tau_{\mathrm{recip}},z,r)\in T_{\mathrm{norm}}.
$$

Finally,

$$
y_i=\operatorname{RQ}_M(e_i r).
$$

The two lookup tags distinguish Softmax rows from LayerNorm's inverse-square-
root rows even though all are proved by one lookup equation.

### Step 3: Eliminate a separate range/control equation

For a bounded unsigned value $a$, introduce bits $b_j$ and prove

$$
a=\sum_{j=0}^{k-1}2^j b_j,
\qquad
b_j(b_j-1)=0.
$$

The first constraint joins the linear family and the second joins the bilinear
family. Signed values use a manifest-fixed offset or signed decomposition.

Requantization is encoded by

$$
t=2^r q+e,
\qquad
0\le e<2^r,
$$

plus the manifest's rounding and saturation selectors. The equality and bit
recomposition are linear. Selector products and booleanity are bilinear.
Comparison and clamp use the same gadgets. Therefore range/control does not
require a fourth operation-specific HD-R1CS equation family.

### Step 4: Three shared operation equations

Let $b^{\mathrm{lin}}$, $b^{\mathrm{bil}}$, and $b^{\mathrm{lut}}$ select all
primitive gates of their respective types across the complete GPT-2 graph.
The proposed global relation contains:

1. **Linear**

   $$
   R_{\mathrm{lin}}(w)
   =L_Ow-L_1w-L_2w-c=0,
   $$

   where the public structured maps encode addition, reduction, broadcast,
   permutation, routing, and bit recomposition.

2. **Bilinear**

   $$
   R_{\mathrm{bil}}(w)
   =\mathcal B(L_Lw,L_Rw)-L_Ow=0,
   $$

   where $\mathcal B$ uses ordinary block multiplication for dense matmuls and
   structured diagonal views for point-wise products.

3. **Tagged lookup**

   $$
   R_{\mathrm{lut}}(w)=0
   \quad\Longleftrightarrow\quad
   (\tau_j,u_j,v_j)\in T_{\mathrm{norm}}
   \text{ for every selected lookup row }j.
   $$

All LayerNorm and Softmax instances, at every block and token position, are
batched into these three equations. Cross-step consistency follows from shared
global block IDs and the final witness-opening batch, not from per-operator
proof composition.

### Step 5: Why a single `Normalization` black-box equation is not preferable

A mode bit $\mu$ could define one conditional formula whose $\mu=0$ branch is
LayerNorm and whose $\mu=1$ branch is Softmax. This only hides multiple
residuals behind selector products:

$$
(1-\mu)R_{\mathrm{LN}}=0,
\qquad
\mu R_{\mathrm{SM}}=0,
\qquad
\mu(\mu-1)=0.
$$

It does not reduce the number of arithmetic facts, lookup rows, or auxiliary
witnesses. It also adds mode-selection constraints. Decomposition into the
three shared primitive families yields the same or fewer proof obligations and
keeps the global operation-type interpretation of HD-R1CS.

### Step 6: Aggressive two-operation lower bound

Any linear relation can be embedded into matrix multiplication. For example,

$$
X+Y=Z
\quad\Longleftrightarrow\quad
\begin{bmatrix}X&Y\end{bmatrix}
\begin{bmatrix}I\\I\end{bmatrix}=Z.
$$

Reductions and affine maps admit analogous public structured right operands.
Consequently all linear, bilinear, range-bit, comparison, Softmax, and
LayerNorm arithmetic can be placed in one universal bilinear equation, while
exp, reciprocal, inverse square root, and GELU use one tagged lookup equation.
The total becomes:

1. I/O binding;
2. universal bilinear arithmetic; and
3. tagged lookup.

This is an exact algebraic reduction, not an approximation. However, it trades
one cheap equation for wider augmented operands and extra virtual views. Unless
benchmarks show those views are essentially free in the heterogeneous PCS
reduction, the three-operation design in Step 4 is preferable.

## Remarks and Interpretation

- “Merge” should mean shared equation families, selectors, transcript batching,
  and final commitment opening. It should not mean identical semantics.
- The same tagged lookup equation can also include GELU, provided the tag is
  included in the authenticated key.
- Q/K/V and MLP matrix products enter the same bilinear family. Thus the whole
  GPT-2 model still has three operation equation families, rather than three
  extra equations just for normalization.
- If implementation simplicity is more important than equation count, a
  separate range equation is reasonable. That gives four operation equations
  plus I/O, but may be easier to optimize and audit.
- The equation-count choices are therefore: aggressive `2 + I/O`, recommended
  `3 + I/O`, or implementation-simple `4 + I/O` with a specialized range
  family.

## Boundaries and Non-Claims

- `global_mlp/normalization.rs` now supports this relation for a single
  normalization slice. It does not yet compile every LayerNorm/Softmax row of a
  full GPT-2 execution.
- The current tagged lookup is transparent: lookup inputs and outputs are
  included in the proof, checked against public tables, and routed back to the
  global witness. It does not hide those rows.
- The derivation does not prove floating-point PyTorch Softmax or LayerNorm. It
  proves only the integer tables and rounding semantics fixed by the manifest.
- Reducing the number of top-level equations does not automatically reduce
  witness size. Softmax still needs exponentials, a denominator, maximum/range
  witnesses, and output products; LayerNorm still needs centered values,
  squares, a variance statistic, and affine products.
- A tagged table reduces equation families, not the number of lookup rows.

## Open Risks

- A universal bilinear equation requires virtual diagonal views; materializing
  diagonal matrices would violate the RAM objective.
- Very large range decompositions may dominate the witness even though they do
  not add an equation family.
- A one-hot maximum certificate adds one bit per eligible position. A reduction
  tree or a specialized maximum argument may be cheaper for long sequences.
- Lookup-table domain design must prevent collisions across tags, scales, and
  signed encodings.
- The proposed three-family design needs focused benchmarks against a four-
  family design with a specialized range equation before choosing the final
  implementation.
