//! Convert a collection of matrix evaluation claims into a single
//! BVA--TBV inner product, following Section 6 of the paper
//! ("ParComm: Collection Commitment and Opening").
//!
//! Each input matrix `M_j` of shape `(mu_j, nu_j)` with left/right evaluation
//! points `(x_l, x_r)` is recursively split into aligned power-of-two
//! submatrices (Lemma "Recursive dyadic splitting"). Every submatrix is
//! vectorized in column-major order and becomes one BVA bucket. The fixed
//! coordinates of the split path contribute a scalar factor, and the remaining
//! coordinates form the local TBV point. After the verifier samples `omega`,
//! the compiled buckets define a TBV `zeta` such that
//!
//! ```text
//!   sum_j omega^{j-1} * (l_j M_j r_j^T) = <a, zeta>,
//! ```
//!
//! where `a` is the BVA holding the concatenated submatrices. This mirrors the
//! single-vector compilation in `BatchPCS::gen_tbv_from_values_and_points`,
//! generalized from one bucket per vector to many buckets per matrix.

use ark_ff::PrimeField;
use rayon::prelude::*;

use crate::bva::BVA;
use crate::tbv::{Bucket, TBV};
use crate::MyInt;

use mat::xi::xi_get_at_position;

/// A single matrix evaluation claim `v_j = l_j M_j r_j^T`.
///
/// The matrix is stored in column-major order: `cols[c][r]` is the entry at
/// row `r`, column `c`. This matches `DenseMatCM<MyInt, F>::data`. The zero
/// extension of `M_j` to power-of-two dimensions is implicit; only the
/// `mu_j * nu_j` real entries are stored.
pub struct MatEvalClaim<F: PrimeField> {
    /// Column-major columns; `cols.len() == nu`, each column has `mu` entries.
    pub cols: Vec<Vec<MyInt>>,
    /// Real shape `(mu, nu)`.
    pub shape: (usize, usize),
    /// Left evaluation point, length `ceil(log2(mu))` (big-endian: `x_l[0]` is
    /// the most significant row bit, matching `DenseMatCM::proj_lr`).
    pub x_l: Vec<F>,
    /// Right evaluation point, length `ceil(log2(nu))`.
    pub x_r: Vec<F>,
}

impl<F: PrimeField> MatEvalClaim<F> {
    /// Build a claim from column-major data, checking the point lengths against
    /// the shape.
    pub fn new(cols: Vec<Vec<MyInt>>, shape: (usize, usize), x_l: Vec<F>, x_r: Vec<F>) -> Self {
        let (mu, nu) = shape;
        assert!(mu > 0 && nu > 0, "matrix dimensions must be non-zero");
        assert_eq!(cols.len(), nu, "column count does not match nu");
        assert!(
            cols.iter().all(|col| col.len() == mu),
            "every column must have mu entries"
        );
        assert_eq!(
            x_l.len(),
            ceil_log2(mu),
            "x_l length must be ceil(log2(mu))"
        );
        assert_eq!(
            x_r.len(),
            ceil_log2(nu),
            "x_r length must be ceil(log2(nu))"
        );
        Self {
            cols,
            shape,
            x_l,
            x_r,
        }
    }
}

/// Per-bucket compilation metadata, indexed by source submatrix index.
///
/// `eta_base` is the product of the fixed-coordinate factors `F_l * F_r`; the
/// `omega^{j-1}` weight is multiplied in only once `omega` is known.
#[derive(Clone, Debug)]
struct BucketMeta<F: PrimeField> {
    /// Index `j` of the originating matrix (drives the `omega^{j-1}` weight).
    mat_index: usize,
    /// Product of the fixed left and right coordinate factors.
    eta_base: F,
    /// Local TBV point in the reversed (LSB-first) convention used by
    /// `xi_rev`, length `log2(bucket size)`.
    z: Vec<F>,
    /// Provenance of every point coordinate this bucket consumes, so the §6.2
    /// lookup layer can bind the compiled `Y`/`Z` entries to the committed raw
    /// points. Indices are into the originating matrix's `x_l` / `x_r`.
    prov: BucketProvenance,
}

/// Point-coordinate provenance of one bucket, in the codebase's tensor basis
/// (`xi = prod x^bit`): fixed split factors are `x_l`/`x_r` coordinates on the
/// set high bits (no `1 - x` terms), and the local point `z` is the low
/// coordinates in reversed order.
#[derive(Clone, Debug, Default)]
pub struct BucketProvenance {
    /// Originating matrix index `j`.
    pub mat_index: usize,
    /// `x_l` coordinate indices contributing to `F_l` (set high row bits).
    pub xfac_l: Vec<usize>,
    /// `x_r` coordinate indices contributing to `F_r` (set high col bits).
    pub xfac_r: Vec<usize>,
    /// `x_l` coordinate indices forming the local row point `z`, in `z` order.
    pub z_l: Vec<usize>,
    /// `x_r` coordinate indices forming the local col point `z`, in `z` order.
    pub z_r: Vec<usize>,
}

/// Compiled collection: the BVA holding all submatrices plus the metadata that
/// turns it into a TBV once `omega` is sampled.
pub struct MatToBva<F: PrimeField> {
    /// BVA whose source vectors are the column-major submatrices.
    pub bva: BVA,
    /// Number of input matrices `t`.
    pub num_mats: usize,
    /// Bucket metadata, indexed by source submatrix index (`bva.vecs` index).
    metas: Vec<BucketMeta<F>>,
}

impl<F: PrimeField> MatToBva<F> {
    /// Recursively split every matrix, assemble the BVA, and record the
    /// per-bucket TBV compilation metadata.
    pub fn build(claims: &[MatEvalClaim<F>]) -> Self {
        assert!(!claims.is_empty(), "at least one matrix claim is required");

        let mut vecs: Vec<Vec<MyInt>> = Vec::new();
        let mut metas: Vec<BucketMeta<F>> = Vec::new();

        for (mat_index, claim) in claims.iter().enumerate() {
            split_matrix_into_buckets(claim, mat_index, &mut vecs, &mut metas);
        }

        let bva = BVA::new(vecs);
        Self {
            bva,
            num_mats: claims.len(),
            metas,
        }
    }

    /// Compile the buckets into a TBV given the batching challenge `omega`.
    ///
    /// The returned TBV `zeta` satisfies `<a, zeta> = sum_j omega^{j-1} v_j`,
    /// where `a` is `self.bva`. Buckets are emitted in BVA layout order so the
    /// TBV bucket sizes match the committed BVA bucket by bucket.
    pub fn gen_tbv(&self, omega: F) -> TBV<F> {
        let powers = omega_powers(omega, self.num_mats);
        let layout = &self.bva.layout;

        let buckets: Vec<Bucket<F>> = layout
            .phi_inv
            .iter()
            .enumerate()
            .map(|(bucket_idx, &src_idx_opt)| {
                let k = layout.bucket_sizes[bucket_idx];
                match src_idx_opt {
                    Some(src_idx) => {
                        let meta = &self.metas[src_idx];
                        Bucket {
                            k,
                            eta: powers[meta.mat_index] * meta.eta_base,
                            z: meta.z.clone(),
                        }
                    }
                    // BvaLayout::new never leaves a real bucket empty, but keep
                    // a zero bucket for safety so the TBV stays well formed.
                    None => Bucket {
                        k,
                        eta: F::zero(),
                        z: vec![F::zero(); k.ilog2() as usize],
                    },
                }
            })
            .collect();

        TBV::new(layout.k_total, buckets)
    }

    /// Per-source-bucket compilation metadata, in BVA source order (i.e.
    /// indexed by `bva.vecs` / `layout.phi` source index): `(mat_index,
    /// eta_base, z_i)`. `eta_base = F_l * F_r` is the omega-independent scale;
    /// the full bucket scale is `omega^{mat_index} * eta_base`. Consumed by
    /// `ParComm` when building the `Y` and `Z` compilation matrices.
    pub fn bucket_metadata(&self) -> Vec<(usize, F, Vec<F>)> {
        self.metas
            .iter()
            .map(|m| (m.mat_index, m.eta_base, m.z.clone()))
            .collect()
    }

    /// Per-source-bucket point-coordinate provenance, in BVA source order.
    /// Used by the §6.2 lookup layer to build the compiled `Y`/`Z` and index
    /// matrices and bind them to the committed raw points.
    pub fn bucket_provenance(&self) -> Vec<BucketProvenance> {
        self.metas.iter().map(|m| m.prov.clone()).collect()
    }

    /// Per-matrix evaluation values `v_j = l_j M_j r_j^T`, reconstructed from
    /// the split. Useful for cross-checking against a direct dense projection.
    pub fn per_matrix_values(&self) -> Vec<F> {
        let mut values = vec![F::zero(); self.num_mats];
        for (src_idx, meta) in self.metas.iter().enumerate() {
            let vec = &self.bva.vecs[src_idx];
            let local = crate::xi_rev::xi_rev_ip_myint_from_challenges_and_vector(vec, &meta.z);
            values[meta.mat_index] += meta.eta_base * local;
        }
        values
    }

    /// Batched value `c = sum_j omega^{j-1} v_j`.
    pub fn batched_value(&self, omega: F) -> F {
        let powers = omega_powers(omega, self.num_mats);
        self.per_matrix_values()
            .iter()
            .zip(powers.iter())
            .map(|(v, p)| *v * *p)
            .sum()
    }
}

/// `ceil(log2(n))` for `n >= 1`; returns `0` for `n == 1`.
fn ceil_log2(n: usize) -> usize {
    assert!(n > 0, "log2 of zero is undefined");
    n.next_power_of_two().trailing_zeros() as usize
}

/// Powers `[1, omega, ..., omega^{count-1}]`.
fn omega_powers<F: PrimeField>(omega: F, count: usize) -> Vec<F> {
    std::iter::successors(Some(F::one()), |&x| Some(x * omega))
        .take(count)
        .collect()
}

/// Partition `[0, n)` into aligned power-of-two intervals, greedily taking the
/// largest block that both fits the remaining length and is aligned to the
/// current offset. Returns `(offset, size)` pairs from left to right.
fn aligned_dyadic_intervals(n: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    while offset < n {
        let remaining = n - offset;
        let fit = prev_power_of_two(remaining);
        // A zero offset divides every block size, so the fit is the only bound.
        let align = if offset == 0 {
            fit
        } else {
            1usize << offset.trailing_zeros()
        };
        let size = align.min(fit);
        out.push((offset, size));
        offset += size;
    }
    out
}

/// Largest power of two `<= n` for `n >= 1`.
fn prev_power_of_two(n: usize) -> usize {
    debug_assert!(n >= 1);
    1usize << (usize::BITS - 1 - n.leading_zeros())
}

/// Split one matrix into aligned power-of-two submatrices, appending each
/// column-major submatrix to `vecs` and its compilation metadata to `metas`.
fn split_matrix_into_buckets<F: PrimeField>(
    claim: &MatEvalClaim<F>,
    mat_index: usize,
    vecs: &mut Vec<Vec<MyInt>>,
    metas: &mut Vec<BucketMeta<F>>,
) {
    let (mu, nu) = claim.shape;
    let a = ceil_log2(mu); // number of row bits
    let b = ceil_log2(nu); // number of column bits

    let row_intervals = aligned_dyadic_intervals(mu);
    let col_intervals = aligned_dyadic_intervals(nu);

    // Emit submatrices in (matrix, row, column) order. BvaLayout::new later
    // re-sorts by non-increasing size with this index as the tie-breaker, so
    // the row/column order here fixes ties deterministically.
    for &(row_off, row_len) in &row_intervals {
        let alpha = row_len.trailing_zeros() as usize; // local row bits
        let high_r = row_off >> alpha; // fixed high row bits
                                       // Fixed-row factor: product over the first (a - alpha) challenges of
                                       // x_l selected by the bits of high_r, in the big-endian convention.
        let f_l = xi_get_at_position(high_r, &claim.x_l[..a - alpha]);
        // x_l coordinate indices whose product forms F_l (set high row bits).
        let xfac_l: Vec<usize> = (0..a - alpha)
            .filter(|&i| (high_r >> i) & 1 == 1)
            .map(|i| (a - alpha) - 1 - i)
            .collect();
        // Local row point: the last alpha challenges of x_l, reversed into the
        // LSB-first order consumed by xi_rev.
        let z_l: Vec<F> = claim.x_l[a - alpha..].iter().rev().copied().collect();
        let z_l_cols: Vec<usize> = (0..alpha).map(|t| a - 1 - t).collect();

        for &(col_off, col_len) in &col_intervals {
            let beta = col_len.trailing_zeros() as usize; // local column bits
            let high_c = col_off >> beta;
            let f_r = xi_get_at_position(high_c, &claim.x_r[..b - beta]);
            let xfac_r: Vec<usize> = (0..b - beta)
                .filter(|&i| (high_c >> i) & 1 == 1)
                .map(|i| (b - beta) - 1 - i)
                .collect();
            let z_r: Vec<F> = claim.x_r[b - beta..].iter().rev().copied().collect();
            let z_r_cols: Vec<usize> = (0..beta).map(|t| b - 1 - t).collect();

            // Column-major vectorization of the submatrix: index = c' * 2^alpha
            // + r'. The row bits (z_l) occupy the low positions, the column
            // bits (z_r) the high positions, matching xi_rev(z)[idx].
            let mut vec = Vec::with_capacity(row_len * col_len);
            for c in 0..col_len {
                let column = &claim.cols[col_off + c];
                vec.extend_from_slice(&column[row_off..row_off + row_len]);
            }

            let mut z = z_l.clone();
            z.extend_from_slice(&z_r);

            let prov = BucketProvenance {
                mat_index,
                xfac_l: xfac_l.clone(),
                xfac_r,
                z_l: z_l_cols.clone(),
                z_r: z_r_cols,
            };

            vecs.push(vec);
            metas.push(BucketMeta {
                mat_index,
                eta_base: f_l * f_r,
                z,
                prov,
            });
        }
    }
}

/// Reference dense evaluation `l M r^T` in the big-endian point convention,
/// summing only over the real `mu * nu` entries.
///
/// Kept `pub(crate)` for tests and callers that want a direct cross-check
/// without going through the `mat` crate's matrix types.
pub fn dense_matrix_eval<F: PrimeField>(
    cols: &[Vec<MyInt>],
    shape: (usize, usize),
    x_l: &[F],
    x_r: &[F],
) -> F {
    let (mu, nu) = shape;
    let l = mat::xi::xi_from_challenges(&x_l.to_vec());
    let r = mat::xi::xi_from_challenges(&x_r.to_vec());
    let to_f = |x: MyInt| {
        if x >= 0 {
            F::from(x as u64)
        } else {
            -F::from((-(x as i32)) as u64)
        }
    };
    (0..nu)
        .into_par_iter()
        .map(|c| {
            let rc = r[c];
            let column = &cols[c];
            let col_sum: F = (0..mu).map(|row| l[row] * to_f(column[row])).sum();
            col_sum * rc
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bva::BvaOps;
    use ark_bls12_381::Fr;
    use ark_ff::One;
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};

    fn rand_challenges(rng: &mut StdRng, len: usize) -> Vec<Fr> {
        (0..len).map(|_| Fr::from(rng.gen::<u64>())).collect()
    }

    /// Build a random column-major MyInt matrix of the given shape.
    fn rand_matrix(rng: &mut StdRng, mu: usize, nu: usize) -> Vec<Vec<MyInt>> {
        (0..nu)
            .map(|_| (0..mu).map(|_| rng.gen_range(-8..8) as MyInt).collect())
            .collect()
    }

    fn make_claim(rng: &mut StdRng, mu: usize, nu: usize) -> MatEvalClaim<Fr> {
        let cols = rand_matrix(rng, mu, nu);
        let x_l = rand_challenges(rng, ceil_log2(mu));
        let x_r = rand_challenges(rng, ceil_log2(nu));
        MatEvalClaim::new(cols, (mu, nu), x_l, x_r)
    }

    #[test]
    fn dyadic_intervals_cover_and_align() {
        for n in 1..=40usize {
            let intervals = aligned_dyadic_intervals(n);
            let mut covered = 0;
            for &(off, size) in &intervals {
                assert!(size.is_power_of_two());
                assert_eq!(off % size, 0, "interval {off} not aligned to {size}");
                assert_eq!(off, covered, "intervals not contiguous");
                covered += size;
            }
            assert_eq!(covered, n, "intervals do not cover [0, {n})");
        }
    }

    #[test]
    fn single_matrix_value_matches_dense() {
        let mut rng = StdRng::seed_from_u64(1);
        // Non-power-of-two shape to exercise multi-bucket splitting.
        let claim = make_claim(&mut rng, 25, 3);
        let expected = dense_matrix_eval(&claim.cols, claim.shape, &claim.x_l, &claim.x_r);

        let compiled = MatToBva::build(std::slice::from_ref(&claim));
        let values = compiled.per_matrix_values();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], expected, "reconstructed value must equal l M r");
    }

    #[test]
    fn bva_tbv_inner_product_matches_batched_value() {
        let mut rng = StdRng::seed_from_u64(7);
        let shapes = [(25, 3), (16, 16), (1, 7), (5, 1), (8, 4)];
        let claims: Vec<MatEvalClaim<Fr>> = shapes
            .iter()
            .map(|&(mu, nu)| make_claim(&mut rng, mu, nu))
            .collect();

        // Independent ground-truth per-matrix values from the dense projection.
        let dense_values: Vec<Fr> = claims
            .iter()
            .map(|c| dense_matrix_eval(&c.cols, c.shape, &c.x_l, &c.x_r))
            .collect();

        let compiled = MatToBva::build(&claims);
        assert_eq!(compiled.per_matrix_values(), dense_values);

        let omega = Fr::from(0x9e3779b97f4a7c15u64);
        let tbv = compiled.gen_tbv(omega);

        // TBV bucket sizes must match the BVA layout bucket by bucket.
        assert!(compiled.bva.check_consistency_with_tbv(&tbv));

        // Core identity: <a, zeta> = sum_j omega^{j-1} v_j.
        let ip = compiled.bva.ip_with_tbv(&tbv);
        let expected = compiled.batched_value(omega);
        assert_eq!(ip, expected);

        // Cross-check the batched value against the independent dense values.
        let powers = omega_powers(omega, claims.len());
        let dense_batched: Fr = dense_values
            .iter()
            .zip(powers.iter())
            .map(|(v, p)| *v * *p)
            .sum();
        assert_eq!(expected, dense_batched);
    }

    #[test]
    fn dense_ip_consistency_via_expand() {
        // The bucketed inner product must agree with the fully materialized
        // dense inner product between the BVA and the expanded TBV.
        let mut rng = StdRng::seed_from_u64(11);
        let claims = vec![make_claim(&mut rng, 12, 6), make_claim(&mut rng, 4, 4)];
        let compiled = MatToBva::build(&claims);
        let omega = Fr::from(5u64);
        let tbv = compiled.gen_tbv(omega);

        assert!(compiled.bva.ip_consistency_check(&tbv));
    }

    #[test]
    fn power_of_two_single_bucket_matches_vector_convention() {
        // A k-by-1 matrix is a single bucket whose local point is the reversed
        // left point, matching BatchPCS's single-vector compilation.
        let mut rng = StdRng::seed_from_u64(3);
        let claim = make_claim(&mut rng, 8, 1);
        let compiled = MatToBva::build(std::slice::from_ref(&claim));
        assert_eq!(compiled.bva.layout.bucket_sizes, vec![8]);
        assert_eq!(compiled.metas.len(), 1);
        let expected_z: Vec<Fr> = claim.x_l.iter().rev().copied().collect();
        assert_eq!(compiled.metas[0].z, expected_z);
        assert_eq!(compiled.metas[0].eta_base, Fr::one());
    }
}
