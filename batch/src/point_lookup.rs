//! Point-compilation lookup layer (paper Section 6.2, "Compiling Matrix Points
//! into a TBV"), adapted to this codebase's tensor basis.
//!
//! ParComm hides the recursive split by committing only the *raw* per-matrix
//! evaluation points (matrix `X` / `P` here), and proving that every compiled
//! scale/point entry is an *indexed read* of a virtual source table built from
//! those points. This module builds that compilation:
//!
//! * `P` — the raw points, one row per matrix: `[x_r | x_l]` (each side padded
//!   to `d = log2 K`). This is what gets committed (`C_X`); it never reveals the
//!   split or the expanded `l_j`, `r_j`.
//! * `S` — the virtual source table `[P | Omega | 1 | 0]`, where column `2d`
//!   holds `Omega_omega[j] = omega^{j}`, column `2d+1` is the constant `1`, and
//!   column `2d+2` is the constant `0`. Only the `P` block is committed; the
//!   `Omega`/`1`/`0` columns are public (closed form), matching the source-MLE
//!   reduction of Eq. (source-mle-reduction).
//! * `Y` — one row per bucket; row product is the TBV scale
//!   `eta_i = omega^{mat_index} * F_l * F_r`. In the tensor basis each fixed
//!   split factor is a single `x_l`/`x_r` coordinate (set high bits only), so
//!   `Y`'s entries are reads of the `Omega`, `P`, and `1` columns of `S`.
//! * `Z` — one row per bucket; the first `log2 k_i` entries are the local point
//!   `z_i`, the rest are zeros. Entries are reads of the `P` and `0` columns.
//!
//! Each compiled entry carries a committed `(row, col)` index into `S`; the
//! indexed-read relation `Y[i][h] = S[row][col]` (and likewise `Z`) is what a
//! LogUp argument proves against `C_X`. This module builds the matrices, the
//! index/value data, and the read multiplicities, and (in tests) checks that
//! the reads are valid, that `prod Y = eta`, that `Z` recovers `z_i`, and that
//! the value-carrying LogUp identity holds. The succinct fractional-sum
//! reduction is layered on top separately.

use ark_ff::PrimeField;

use crate::mat_to_bva::{MatEvalClaim, MatToBva};

/// Column index in `S` of the `Omega_omega` block, given the per-side width `d`.
fn omega_col(d: usize) -> usize {
    2 * d
}
/// Column index in `S` of the constant-one cell.
fn one_col(d: usize) -> usize {
    2 * d + 1
}
/// Column index in `S` of the constant-zero cell.
fn zero_col(d: usize) -> usize {
    2 * d + 2
}

/// Committed/looked-up indexed reads for one compiled matrix (`Y` or `Z`):
/// per bucket row, the `S` row/column each entry reads and the value read.
#[derive(Clone, Debug)]
pub struct IndexedReads<F: PrimeField> {
    /// `row[i][h]`: source-table row read by compiled entry `(i, h)`.
    pub row: Vec<Vec<usize>>,
    /// `col[i][h]`: source-table column read by compiled entry `(i, h)`.
    pub col: Vec<Vec<usize>>,
    /// `val[i][h]`: the compiled value (equal to `S[row][col]` for valid reads).
    pub val: Vec<Vec<F>>,
}

impl<F: PrimeField> IndexedReads<F> {
    fn new(rows: usize) -> Self {
        Self {
            row: vec![Vec::new(); rows],
            col: vec![Vec::new(); rows],
            val: vec![Vec::new(); rows],
        }
    }
    fn push(&mut self, i: usize, r: usize, c: usize, v: F) {
        self.row[i].push(r);
        self.col[i].push(c);
        self.val[i].push(v);
    }
}

/// The full point compilation: source table plus the `Y`/`Z` indexed reads.
pub struct PointCompilation<F: PrimeField> {
    /// Number of matrices `t`.
    pub num_mats: usize,
    /// Per-side coordinate width `d = log2 K`.
    pub d: usize,
    /// Number of buckets `I` (rows of `Y`/`Z`).
    pub num_buckets: usize,
    /// Width of `Y` (holds `omega`, all fixed factors, and one-padding).
    pub width_y: usize,
    /// Width of `Z` (holds the local point and zero-padding).
    pub width_z: usize,
    /// Source table `S` in row-major order: `S[j][c]`, `t` real rows.
    pub source: Vec<Vec<F>>,
    /// Number of columns of `S` (`2d + 3`).
    pub source_cols: usize,
    /// Indexed reads defining `Y`.
    pub y_reads: IndexedReads<F>,
    /// Indexed reads defining `Z`.
    pub z_reads: IndexedReads<F>,
}

impl<F: PrimeField> PointCompilation<F> {
    /// Build the compilation for a collection of matrix claims and the batching
    /// challenge `omega`. `compiled` supplies the bucket order and provenance.
    pub fn build(claims: &[MatEvalClaim<F>], compiled: &MatToBva<F>, omega: F) -> Self {
        let num_mats = claims.len();
        let d = compiled.bva.layout.k_total.ilog2() as usize;
        let prov = compiled.bucket_provenance();
        let num_buckets = prov.len();
        let source_cols = 2 * d + 3;

        // Powers omega^{0..t}.
        let powers: Vec<F> = std::iter::successors(Some(F::one()), |&x| Some(x * omega))
            .take(num_mats)
            .collect();

        // --- Source table S = [ x_r (d) | x_l (d) | Omega | 1 | 0 ], one row per matrix.
        let mut source = vec![vec![F::zero(); source_cols]; num_mats];
        for (j, claim) in claims.iter().enumerate() {
            for (c, &v) in claim.x_r.iter().enumerate() {
                source[j][c] = v;
            }
            for (c, &v) in claim.x_l.iter().enumerate() {
                source[j][d + c] = v;
            }
            source[j][omega_col(d)] = powers[j];
            source[j][one_col(d)] = F::one();
            // zero_col stays zero.
        }

        // Column of `S` for an x_l / x_r coordinate.
        let col_xl = |coord: usize| d + coord;
        let col_xr = |coord: usize| coord;

        // Widths: Y must fit omega + all fixed factors; Z must fit the local point.
        let max_y = prov
            .iter()
            .map(|p| 1 + p.xfac_l.len() + p.xfac_r.len())
            .max()
            .unwrap_or(1);
        let max_z = prov
            .iter()
            .map(|p| p.z_l.len() + p.z_r.len())
            .max()
            .unwrap_or(0);
        let width_y = max_y.max(1);
        let width_z = max_z.max(1);

        let mut y_reads = IndexedReads::new(num_buckets);
        let mut z_reads = IndexedReads::new(num_buckets);

        for (i, p) in prov.iter().enumerate() {
            let j = p.mat_index;
            let claim = &claims[j];

            // --- Y row: omega^{j}, then x_l / x_r fixed factors, then ones.
            y_reads.push(i, j, omega_col(d), powers[j]);
            for &coord in &p.xfac_l {
                y_reads.push(i, j, col_xl(coord), claim.x_l[coord]);
            }
            for &coord in &p.xfac_r {
                y_reads.push(i, j, col_xr(coord), claim.x_r[coord]);
            }
            while y_reads.row[i].len() < width_y {
                y_reads.push(i, 0, one_col(d), F::one());
            }

            // --- Z row: local point coordinates, then zeros.
            for &coord in &p.z_l {
                z_reads.push(i, j, col_xl(coord), claim.x_l[coord]);
            }
            for &coord in &p.z_r {
                z_reads.push(i, j, col_xr(coord), claim.x_r[coord]);
            }
            while z_reads.row[i].len() < width_z {
                z_reads.push(i, 0, zero_col(d), F::zero());
            }
        }

        Self {
            num_mats,
            d,
            num_buckets,
            width_y,
            width_z,
            source,
            source_cols,
            y_reads,
            z_reads,
        }
    }

    /// Source-table value at `(row, col)` (public closed form for the derived
    /// columns; the `P` block is what `C_X` commits).
    pub fn source_at(&self, row: usize, col: usize) -> F {
        self.source[row][col]
    }

    /// Row-product of `Y` row `i` (the reconstructed TBV scale `eta_i`).
    pub fn y_row_product(&self, i: usize) -> F {
        self.y_reads.val[i].iter().copied().product()
    }

    /// The local point `z_i` recovered from `Z` row `i` (leading `len` entries).
    pub fn z_row_prefix(&self, i: usize, len: usize) -> Vec<F> {
        self.z_reads.val[i][..len].to_vec()
    }

    /// Read-multiplicity of every source cell across all `Y` and `Z` reads:
    /// `mult[j][c]` = number of reads hitting `S[j][c]`.
    pub fn multiplicities(&self) -> Vec<Vec<usize>> {
        let mut mult = vec![vec![0usize; self.source_cols]; self.num_mats];
        for reads in [&self.y_reads, &self.z_reads] {
            for i in 0..self.num_buckets {
                for h in 0..reads.row[i].len() {
                    mult[reads.row[i][h]][reads.col[i][h]] += 1;
                }
            }
        }
        mult
    }

    /// Value-carrying LogUp key of a source cell:
    /// `row + alpha*col + alpha^2 * value`.
    fn key(&self, row: usize, col: usize, value: F, alpha: F) -> F {
        F::from(row as u64) + alpha * F::from(col as u64) + alpha * alpha * value
    }

    /// Evaluate the LogUp identity residual
    /// `sum_reads 1/(beta + key) - sum_S mult/(beta + key)`.
    /// It is zero iff every read is a valid indexed read of `S`.
    pub fn logup_residual(&self, alpha: F, beta: F) -> F {
        let mut acc = F::zero();
        // Read side: + 1/(beta + key(read)).
        for reads in [&self.y_reads, &self.z_reads] {
            for i in 0..self.num_buckets {
                for h in 0..reads.row[i].len() {
                    let k = self.key(reads.row[i][h], reads.col[i][h], reads.val[i][h], alpha);
                    acc += (beta + k).inverse().expect("denominator must be non-zero");
                }
            }
        }
        // Table side: - mult/(beta + key(cell)).
        let mult = self.multiplicities();
        for j in 0..self.num_mats {
            for c in 0..self.source_cols {
                let m = mult[j][c];
                if m == 0 {
                    continue;
                }
                let k = self.key(j, c, self.source[j][c], alpha);
                acc -=
                    F::from(m as u64) * (beta + k).inverse().expect("denominator must be non-zero");
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{One, Zero};
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
    use poly_commit::MyInt;

    fn ceil_log2(n: usize) -> usize {
        n.next_power_of_two().trailing_zeros() as usize
    }

    fn make_claim(rng: &mut StdRng, mu: usize, nu: usize) -> MatEvalClaim<Fr> {
        let cols: Vec<Vec<MyInt>> = (0..nu)
            .map(|_| (0..mu).map(|_| rng.gen_range(-8..8) as MyInt).collect())
            .collect();
        let x_l: Vec<Fr> = (0..ceil_log2(mu))
            .map(|_| Fr::from(rng.gen::<u64>()))
            .collect();
        let x_r: Vec<Fr> = (0..ceil_log2(nu))
            .map(|_| Fr::from(rng.gen::<u64>()))
            .collect();
        MatEvalClaim::new(cols, (mu, nu), x_l, x_r)
    }

    fn setup(seed: u64) -> (Vec<MatEvalClaim<Fr>>, MatToBva<Fr>, Fr) {
        let mut rng = StdRng::seed_from_u64(seed);
        let shapes = [(4usize, 4usize), (8, 2), (2, 2), (3, 3), (5, 7)];
        let claims: Vec<MatEvalClaim<Fr>> = shapes
            .iter()
            .map(|&(mu, nu)| make_claim(&mut rng, mu, nu))
            .collect();
        let compiled = MatToBva::build(&claims);
        let omega = Fr::from(0x1234_5678_9abc_def1u64);
        (claims, compiled, omega)
    }

    #[test]
    fn reads_are_valid_indexed_reads_of_source() {
        let (claims, compiled, omega) = setup(1);
        let pc = PointCompilation::build(&claims, &compiled, omega);

        for reads in [&pc.y_reads, &pc.z_reads] {
            for i in 0..pc.num_buckets {
                for h in 0..reads.row[i].len() {
                    let (r, c, v) = (reads.row[i][h], reads.col[i][h], reads.val[i][h]);
                    assert_eq!(v, pc.source_at(r, c), "invalid read at bucket {i}, pos {h}");
                }
            }
        }
    }

    #[test]
    fn y_row_product_matches_eta_and_z_recovers_local_point() {
        let (claims, compiled, omega) = setup(2);
        let pc = PointCompilation::build(&claims, &compiled, omega);

        // Reference scales/points from the TBV.
        let tbv = compiled.gen_tbv(omega);
        let layout = &compiled.bva.layout;

        for i in 0..pc.num_buckets {
            // Bucket i (source order) maps to TBV bucket phi[i].
            let bucket = &tbv.buckets[layout.phi[i]];
            assert_eq!(pc.y_row_product(i), bucket.eta, "prod(Y row {i}) != eta");
            let z = pc.z_row_prefix(i, bucket.z.len());
            assert_eq!(z, bucket.z, "Z row {i} does not recover z_i");
        }
    }

    #[test]
    fn logup_identity_holds_for_valid_reads() {
        let (claims, compiled, omega) = setup(3);
        let pc = PointCompilation::build(&claims, &compiled, omega);
        let alpha = Fr::from(9_876_543_210u64);
        let beta = Fr::from(1_111_222_333u64);
        assert_eq!(pc.logup_residual(alpha, beta), Fr::zero());
    }

    #[test]
    fn logup_identity_breaks_under_tampering() {
        let (claims, compiled, omega) = setup(4);
        let mut pc = PointCompilation::build(&claims, &compiled, omega);
        // Corrupt one compiled Y value so it no longer matches the source cell.
        pc.y_reads.val[0][0] += Fr::one();
        let alpha = Fr::from(42u64);
        let beta = Fr::from(7u64);
        assert_ne!(pc.logup_residual(alpha, beta), Fr::zero());
    }
}
