use crate::MyInt;
use ark_ff::PrimeField;
use rayon::prelude::*;

/// Layout information for bucketed vector assembly (BVA).
#[derive(Debug, Clone, Default)]
pub struct BvaLayout {
    pub k_total: usize,
    pub bucket_sizes: Vec<usize>,
    pub offsets: Vec<usize>, // global start offset for each original vector (by original index)
    pub phi: Vec<usize>,     // maps each original vector index j to its bucket index
    pub phi_inv: Vec<Option<usize>>, // maps each bucket index to optional source vector index
}

pub trait BvaOps<F: PrimeField> {
    /// Check that bucket sizes match TBV buckets (same count and per-bucket k).
    fn check_consistency_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> bool;
    /// EO-fold each bucket with scalar y, returning a field buffer holding folded buckets.
    fn eo_fold(&self, y: F) -> BVAFieldBufs<F>;
    /// EO-fold each bucket with scalar y, returning a field buffer holding folded buckets.
    fn eo_fold_in_place(&mut self, y: F);
    /// Bucket-wise inner product with TBV (skipping zero buckets).
    fn ip_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> F;
    /// Bucket-wise odd of BVA and even of TBV (folding each pair of length-1 bucket).
    fn ip_oe_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> F;
    /// Bucket-wise even of BVA and odd of TBV (folding each pair of length-1 bucket).
    fn ip_eo_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> F;
    /// Expand into a dense field vector of length layout.k_total.
    fn expand_to_field_vec(&self) -> Vec<F>;
    /// Inner product with a tensor-structured vector generated from challenges.
    fn ip_with_xi_rev_from_challenges(&self, challenges: &[F]) -> F;
    /// Compare bucket-wise IP with dense IP against expanded TBV.
    fn ip_consistency_check(&self, tbv: &crate::tbv::TBV<F>) -> bool {
        let ip_bucket = self.ip_with_tbv(tbv);
        let dense = self.expand_to_field_vec();
        let tbv_dense = tbv.expand();
        if dense.len() != tbv_dense.len() {
            return false;
        }
        let ip_dense: F = dense
            .iter()
            .zip(tbv_dense.iter())
            .map(|(a, b)| *a * *b)
            .sum();
        ip_bucket == ip_dense
    }
}

/// Owned witness vectors (MyInt) bundled with layout.
#[derive(Debug, Clone, Default)]
pub struct BVA {
    pub vecs: Vec<Vec<MyInt>>, // Indexed by source vector index (src_idx)
    pub layout: BvaLayout,
}

impl BVA {
    /// Construct a BVA from owned witness vectors, building the layout from their lengths.
    pub fn new(vecs: Vec<Vec<MyInt>>) -> Self {
        let shapes: Vec<usize> = vecs.iter().map(|v| v.len()).collect();
        let layout = BvaLayout::new(&shapes);
        BVA { vecs, layout }
    }

    pub fn total_len(&self) -> usize {
        self.vecs.iter().map(|v| v.len()).sum()
    }

    pub fn expand_to_myint_vec(&self) -> Vec<MyInt> {
        let mut out = vec![0 as MyInt; self.layout.k_total];
        let mut current_offset = 0;

        for (bucket_idx, &src_idx_opt) in self.layout.phi_inv.iter().enumerate() {
            let k = self.layout.bucket_sizes[bucket_idx];
            if let Some(src_idx) = src_idx_opt {
                let v = &self.vecs[src_idx];
                let len = v.len().min(k);
                out[current_offset..current_offset + len].copy_from_slice(&v[..len]);
            }
            current_offset += k;
        }
        out
    }
}

/// Owned field buffers with attached BVA layout.
#[derive(Debug, Clone)]
pub struct BVAFieldBufs<F: PrimeField> {
    pub vecs: Vec<Vec<F>>, // Indexed by source vector index (src_idx)
    pub layout: BvaLayout,
}

impl<F: PrimeField> BVAFieldBufs<F> {
    pub fn new(vecs: Vec<Vec<F>>, layout: BvaLayout) -> Self {
        BVAFieldBufs { vecs, layout }
    }

    pub fn total_len(&self) -> usize {
        self.vecs.iter().map(|v| v.len()).sum()
    }
}

impl<F: PrimeField> BvaOps<F> for BVA {
    fn check_consistency_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> bool {
        if self.layout.bucket_sizes.len() != tbv.buckets.len() {
            return false;
        }
        self.layout
            .bucket_sizes
            .iter()
            .zip(tbv.buckets.iter())
            .all(|(k, b)| *k == b.k)
    }

    fn eo_fold(&self, y: F) -> BVAFieldBufs<F> {
        let folded_layout = self.layout.eo_fold_layout();

        // Precompute folded results for each source vector, stored by src_idx
        let to_f = |v: MyInt| {
            if v >= 0 {
                F::from(v as u64)
            } else {
                -F::from((-(v as i32)) as u64)
            }
        };
        let mut precomputed: Vec<Vec<F>> = vec![Vec::new(); self.vecs.len()];

        // The first fold converts i8 witnesses into field elements. Each real
        // bucket performs its pairwise arithmetic with Rayon.
        for (old_idx, &old_k) in self.layout.bucket_sizes.iter().enumerate() {
            if let Some(src_idx) = self.layout.phi_inv[old_idx] {
                if !self.vecs[src_idx].is_empty() {
                    if old_k > 1 {
                        let new_k = old_k / 2;
                        let mut out = vec![F::zero(); new_k];
                        crate::utils::eo_fold_myint_to_field_with_par(
                            &self.vecs[src_idx],
                            y,
                            &mut out,
                            true,
                        );
                        precomputed[src_idx] = out;
                    } else {
                        let v = self.vecs[src_idx].get(0).copied().unwrap_or(0);
                        precomputed[src_idx] = vec![to_f(v)];
                    }
                }
            }
        }

        // Assemble sequentially into new layout (preserve original index logic)
        let mut new_vecs = vec![Vec::new(); self.vecs.len()];

        let mut old_idx = 0usize;
        let old_bucket_sizes = &self.layout.bucket_sizes;
        let old_phi_inv = &self.layout.phi_inv;

        for (new_b_idx, &_new_k) in folded_layout.bucket_sizes.iter().enumerate() {
            let old_k = old_bucket_sizes[old_idx];
            if old_k == 1
                && old_idx + 1 < old_bucket_sizes.len()
                && old_bucket_sizes[old_idx + 1] == 1
            {
                // Merged case: pair (idx, idx+1) into one bucket of size 1
                let left_src = old_phi_inv[old_idx];
                let right_src = old_phi_inv[old_idx + 1];
                let l_val = left_src
                    .and_then(|s| precomputed[s].get(0).copied())
                    .unwrap_or_else(F::zero);
                let r_val = right_src
                    .and_then(|s| precomputed[s].get(0).copied())
                    .unwrap_or_else(F::zero);
                let folded = l_val + r_val * y;

                if let Some(tgt) = folded_layout.phi_inv[new_b_idx] {
                    new_vecs[tgt] = vec![folded];
                }
                old_idx += 2;
            } else if old_k == 1 {
                // Unpaired case
                let src = old_phi_inv[old_idx];
                if let Some(src) = src {
                    if !precomputed[src].is_empty() {
                        if let Some(tgt) = folded_layout.phi_inv[new_b_idx] {
                            new_vecs[tgt] = std::mem::take(&mut precomputed[src]);
                        }
                    }
                }
                old_idx += 1;
            } else {
                // Halving case: k > 1 becomes k/2
                if let Some(src_idx) = old_phi_inv[old_idx] {
                    if let Some(tgt) = folded_layout.phi_inv[new_b_idx] {
                        new_vecs[tgt] = std::mem::take(&mut precomputed[src_idx]);
                    }
                }
                old_idx += 1;
            }
        }

        BVAFieldBufs {
            vecs: new_vecs,
            layout: folded_layout,
        }
    }

    fn eo_fold_in_place(&mut self, _y: F) {
        // BVA doesn't support in-place folding at MyInt level due to type constraints
        // The folding operation requires field arithmetic (multiplication by y)
        // We must convert to BVAFieldBufs for proper folding
        panic!("BVA does not support eo_fold_in_place - use eo_fold() to convert to BVAFieldBufs");
    }

    fn ip_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> F {
        let threshold = crate::BUCKET_PAR_THRESHOLD;
        let max_k = self.layout.bucket_sizes.last().copied().unwrap_or(0);

        if max_k > threshold {
            // Intra-bucket parallelism
            let mut acc = F::zero();
            for (idx, b) in tbv.buckets.iter().enumerate() {
                if b.eta.is_zero() {
                    continue;
                }
                if let Some(src_idx) = self.layout.phi_inv[idx] {
                    if !self.vecs[src_idx].is_empty() {
                        let src = &self.vecs[src_idx];
                        let dense: Vec<F> = src
                            .iter()
                            .map(|&x| {
                                if x >= 0 {
                                    F::from(x as u64)
                                } else {
                                    -F::from((-(x as i32)) as u64)
                                }
                            })
                            .collect();
                        let folded = crate::xi_rev::xi_rev_ip_from_challenges_and_vector_with_par(
                            &dense, &b.z, true,
                        );
                        acc += b.eta * folded;
                    }
                }
            }
            acc
        } else {
            // Inter-bucket parallelism
            tbv.buckets
                .par_iter()
                .enumerate()
                .filter(|(_, b)| !b.eta.is_zero())
                .map(|(idx, b)| {
                    if let Some(src_idx) = self.layout.phi_inv[idx] {
                        if !self.vecs[src_idx].is_empty() {
                            let src = &self.vecs[src_idx];
                            let dense: Vec<F> = src
                                .iter()
                                .map(|&x| {
                                    if x >= 0 {
                                        F::from(x as u64)
                                    } else {
                                        -F::from((-(x as i32)) as u64)
                                    }
                                })
                                .collect();
                            let folded =
                                crate::xi_rev::xi_rev_ip_from_challenges_and_vector_with_par(
                                    &dense, &b.z, false,
                                );
                            return b.eta * folded;
                        }
                    }
                    F::zero()
                })
                .reduce(|| F::zero(), |a, b| a + b)
        }
    }

    fn ip_oe_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> F {
        let to_f = |x: MyInt| {
            if x >= 0 {
                F::from(x as u64)
            } else {
                -F::from((-(x as i32)) as u64)
            }
        };
        let threshold = crate::BUCKET_PAR_THRESHOLD;
        let max_k = self.layout.bucket_sizes.last().copied().unwrap_or(0);

        let mut acc = if max_k > threshold {
            // Intra-bucket parallelism (within each bucket)
            tbv.buckets
                .iter()
                .enumerate()
                .filter(|(_, b)| b.k > 1 && !b.eta.is_zero())
                .map(|(idx, b)| {
                    if let Some(src_idx) = self.layout.phi_inv[idx] {
                        if !self.vecs[src_idx].is_empty() {
                            let src = &self.vecs[src_idx];
                            let res = crate::xi_rev::xi_ip_oe_myint_with_xi_rev_par::<F>(src, &b.z);
                            return b.eta * res;
                        }
                    }
                    F::zero()
                })
                .sum::<F>()
        } else {
            // Inter-bucket parallelism (between buckets), intra-bucket is serial
            tbv.buckets
                .par_iter()
                .enumerate()
                .filter(|(_, b)| b.k > 1 && !b.eta.is_zero())
                .map(|(idx, b)| {
                    if let Some(src_idx) = self.layout.phi_inv[idx] {
                        if !self.vecs[src_idx].is_empty() {
                            let src = &self.vecs[src_idx];
                            let res = crate::xi_rev::ip_oe_myint_with_xi_rev_with_par::<F>(
                                src, &b.z, false,
                            );
                            return b.eta * res;
                        }
                    }
                    F::zero()
                })
                .reduce(|| F::zero(), |a, b| a + b)
        };

        // Handle k=1 buckets in pairs (no vector allocation)
        let n = tbv.buckets.len();
        let mut i = 0;
        while i < n {
            if tbv.buckets[i].k == 1 {
                if i + 1 < n && tbv.buckets[i + 1].k == 1 {
                    let even_eta = tbv.buckets[i].eta;
                    if let Some(src_idx_odd) = self.layout.phi_inv[i + 1] {
                        if !self.vecs[src_idx_odd].is_empty() {
                            let val_raw = self.vecs[src_idx_odd][0];
                            acc += to_f(val_raw) * even_eta;
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        acc
    }

    fn ip_eo_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> F {
        let to_f = |x: MyInt| {
            if x >= 0 {
                F::from(x as u64)
            } else {
                -F::from((-(x as i32)) as u64)
            }
        };
        let threshold = crate::BUCKET_PAR_THRESHOLD;
        let max_k = self.layout.bucket_sizes.last().copied().unwrap_or(0);

        let mut acc = if max_k > threshold {
            // Intra-bucket parallelism
            tbv.buckets
                .iter()
                .enumerate()
                .filter(|(_, b)| b.k > 1 && !b.eta.is_zero())
                .map(|(idx, b)| {
                    if let Some(src_idx) = self.layout.phi_inv[idx] {
                        if !self.vecs[src_idx].is_empty() {
                            let src = &self.vecs[src_idx];
                            let res = crate::xi_rev::xi_ip_eo_myint_with_xi_rev_par::<F>(src, &b.z);
                            return b.eta * res;
                        }
                    }
                    F::zero()
                })
                .sum::<F>()
        } else {
            // Inter-bucket parallelism
            tbv.buckets
                .par_iter()
                .enumerate()
                .filter(|(_, b)| b.k > 1 && !b.eta.is_zero())
                .map(|(idx, b)| {
                    if let Some(src_idx) = self.layout.phi_inv[idx] {
                        if !self.vecs[src_idx].is_empty() {
                            let src = &self.vecs[src_idx];
                            let res = crate::xi_rev::ip_eo_myint_with_xi_rev_with_par::<F>(
                                src, &b.z, false,
                            );
                            return b.eta * res;
                        }
                    }
                    F::zero()
                })
                .reduce(|| F::zero(), |a, b| a + b)
        };

        // Handle k=1 buckets in pairs (no vector allocation)
        let n = tbv.buckets.len();
        let mut i = 0;
        while i < n {
            if tbv.buckets[i].k == 1 {
                if i + 1 < n && tbv.buckets[i + 1].k == 1 {
                    let odd_eta = tbv.buckets[i + 1].eta;
                    if let Some(src_idx_even) = self.layout.phi_inv[i] {
                        if !self.vecs[src_idx_even].is_empty() {
                            let val_raw = self.vecs[src_idx_even][0];
                            acc += to_f(val_raw) * odd_eta;
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        acc
    }

    fn expand_to_field_vec(&self) -> Vec<F> {
        let mut out = vec![F::zero(); self.layout.k_total];
        let mut current_offset = 0;

        let to_f = |x: MyInt| {
            if x >= 0 {
                F::from(x as u64)
            } else {
                -F::from((-(x as i32)) as u64)
            }
        };

        for (bucket_idx, &src_idx_opt) in self.layout.phi_inv.iter().enumerate() {
            let k = self.layout.bucket_sizes[bucket_idx];
            if let Some(src_idx) = src_idx_opt {
                if !self.vecs[src_idx].is_empty() {
                    let v = &self.vecs[src_idx];
                    let len = v.len().min(k);
                    for i in 0..len {
                        out[current_offset + i] = to_f(v[i]);
                    }
                }
            }
            current_offset += k;
        }
        out
    }

    fn ip_with_xi_rev_from_challenges(&self, challenges: &[F]) -> F {
        let tbv = crate::tbv::TBV::new_from_challenges(&self.layout.bucket_sizes, challenges);
        self.ip_with_tbv(&tbv)
    }
}

impl<F: PrimeField> BvaOps<F> for BVAFieldBufs<F> {
    fn check_consistency_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> bool {
        self.layout.bucket_sizes.len() == tbv.buckets.len()
            && self
                .layout
                .bucket_sizes
                .iter()
                .zip(tbv.buckets.iter())
                .all(|(k, b)| *k == b.k)
    }

    fn eo_fold(&self, y: F) -> BVAFieldBufs<F> {
        let mut next = self.clone();
        next.eo_fold_in_place(y);
        next
    }

    fn eo_fold_in_place(&mut self, y: F) {
        let folded_layout = self.layout.eo_fold_layout();

        let old_bucket_sizes = &self.layout.bucket_sizes;
        let old_phi_inv = &self.layout.phi_inv;

        // Fold each field bucket in its existing allocation. Pair arithmetic
        // is parallel within the bucket; compaction happens after all pairs
        // are complete, so no read races with an overlapping destination.
        for old_idx in 0..old_bucket_sizes.len() {
            if old_bucket_sizes[old_idx] > 1 {
                if let Some(src_idx) = old_phi_inv[old_idx] {
                    let values = &mut self.vecs[src_idx];
                    if values.is_empty() {
                        continue;
                    }
                    values.par_chunks_mut(2).for_each(|pair| {
                        if pair.len() == 2 {
                            pair[0] += y * pair[1];
                        }
                    });
                    let half = values.len().div_ceil(2);
                    for idx in 1..half {
                        values[idx] = values[2 * idx];
                    }
                    values.truncate(half);
                }
            }
        }

        // Second pass: create new vector layout
        let mut new_vecs = vec![Vec::new(); self.vecs.len()];

        let mut old_idx = 0usize;
        for (new_b_idx, &_new_k) in folded_layout.bucket_sizes.iter().enumerate() {
            let old_k = old_bucket_sizes[old_idx];

            if old_k == 1
                && old_idx + 1 < old_bucket_sizes.len()
                && old_bucket_sizes[old_idx + 1] == 1
            {
                // Merged case: create new merged vector
                let left_src = old_phi_inv[old_idx];
                let right_src = old_phi_inv[old_idx + 1];
                let l_val = left_src
                    .and_then(|s| self.vecs[s].get(0).copied())
                    .unwrap_or_else(F::zero);
                let r_val = right_src
                    .and_then(|s| self.vecs[s].get(0).copied())
                    .unwrap_or_else(F::zero);
                let folded = l_val + r_val * y;

                if let Some(tgt) = folded_layout.phi_inv[new_b_idx] {
                    new_vecs[tgt] = vec![folded];
                }
                old_idx += 2;
            } else if old_k == 1 {
                // Unpaired case
                if let Some(src) = old_phi_inv[old_idx] {
                    if let Some(tgt) = folded_layout.phi_inv[new_b_idx] {
                        new_vecs[tgt] = std::mem::take(&mut self.vecs[src]);
                    }
                }
                old_idx += 1;
            } else {
                // Halving case
                if let Some(src_idx) = old_phi_inv[old_idx] {
                    if let Some(tgt) = folded_layout.phi_inv[new_b_idx] {
                        new_vecs[tgt] = std::mem::take(&mut self.vecs[src_idx]);
                    }
                }
                old_idx += 1;
            }
        }

        self.vecs = new_vecs;
        self.layout = folded_layout;
    }

    fn ip_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> F {
        let threshold = crate::BUCKET_PAR_THRESHOLD;
        let max_k = self.layout.bucket_sizes.last().copied().unwrap_or(0);

        if max_k > threshold {
            // Intra-bucket parallelism
            let mut acc = F::zero();
            for (bucket_idx, &src_idx_opt) in self.layout.phi_inv.iter().enumerate() {
                if let Some(src_idx) = src_idx_opt {
                    if !self.vecs[src_idx].is_empty() {
                        let b = &tbv.buckets[bucket_idx];
                        if b.eta.is_zero() {
                            continue;
                        }
                        let folded = crate::xi_rev::xi_rev_ip_from_challenges_and_vector_with_par(
                            &self.vecs[src_idx],
                            &b.z,
                            true,
                        );
                        acc += b.eta * folded;
                    }
                }
            }
            acc
        } else {
            // Inter-bucket parallelism
            self.layout
                .phi_inv
                .par_iter()
                .enumerate()
                .map(|(bucket_idx, &src_idx_opt)| {
                    if let Some(src_idx) = src_idx_opt {
                        if !self.vecs[src_idx].is_empty() {
                            let b = &tbv.buckets[bucket_idx];
                            if b.eta.is_zero() {
                                return F::zero();
                            }
                            let folded =
                                crate::xi_rev::xi_rev_ip_from_challenges_and_vector_with_par(
                                    &self.vecs[src_idx],
                                    &b.z,
                                    false,
                                );
                            return b.eta * folded;
                        }
                    }
                    F::zero()
                })
                .reduce(|| F::zero(), |a, b| a + b)
        }
    }

    fn ip_oe_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> F {
        let threshold = crate::BUCKET_PAR_THRESHOLD;
        let max_k = self.layout.bucket_sizes.last().copied().unwrap_or(0);

        let mut acc = if max_k > threshold {
            // Intra-bucket parallelism
            tbv.buckets
                .iter()
                .enumerate()
                .filter(|(_, b)| b.k > 1 && !b.eta.is_zero())
                .map(|(idx, b)| {
                    if let Some(src_idx) = self.layout.phi_inv[idx] {
                        if !self.vecs[src_idx].is_empty() {
                            let src = &self.vecs[src_idx];
                            let res =
                                crate::xi_rev::ip_oe_with_xi_rev_with_par::<F>(src, &b.z, true);
                            return b.eta * res;
                        }
                    }
                    F::zero()
                })
                .sum::<F>()
        } else {
            // Inter-bucket parallelism
            tbv.buckets
                .par_iter()
                .enumerate()
                .filter(|(_, b)| b.k > 1 && !b.eta.is_zero())
                .map(|(idx, b)| {
                    if let Some(src_idx) = self.layout.phi_inv[idx] {
                        if !self.vecs[src_idx].is_empty() {
                            let src = &self.vecs[src_idx];
                            let res =
                                crate::xi_rev::ip_oe_with_xi_rev_with_par::<F>(src, &b.z, false);
                            return b.eta * res;
                        }
                    }
                    F::zero()
                })
                .reduce(|| F::zero(), |a, b| a + b)
        };

        // Handle k=1 buckets in pairs (no vector allocation)
        let n = tbv.buckets.len();
        let mut i = 0;
        while i < n {
            if tbv.buckets[i].k == 1 {
                if i + 1 < n && tbv.buckets[i + 1].k == 1 {
                    let even_eta = tbv.buckets[i].eta;
                    if let Some(src_idx_odd) = self.layout.phi_inv[i + 1] {
                        if !self.vecs[src_idx_odd].is_empty() {
                            let val_raw = self.vecs[src_idx_odd][0];
                            acc += val_raw * even_eta;
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        acc
    }

    fn ip_eo_with_tbv(&self, tbv: &crate::tbv::TBV<F>) -> F {
        let threshold = crate::BUCKET_PAR_THRESHOLD;
        let max_k = self.layout.bucket_sizes.last().copied().unwrap_or(0);

        let mut acc = if max_k > threshold {
            // Intra-bucket parallelism
            tbv.buckets
                .iter()
                .enumerate()
                .filter(|(_, b)| b.k > 1 && !b.eta.is_zero())
                .map(|(idx, b)| {
                    if let Some(src_idx) = self.layout.phi_inv[idx] {
                        if !self.vecs[src_idx].is_empty() {
                            let src = &self.vecs[src_idx];
                            let res =
                                crate::xi_rev::ip_eo_with_xi_rev_with_par::<F>(src, &b.z, true);
                            return b.eta * res;
                        }
                    }
                    F::zero()
                })
                .sum::<F>()
        } else {
            // Inter-bucket parallelism
            tbv.buckets
                .par_iter()
                .enumerate()
                .filter(|(_, b)| b.k > 1 && !b.eta.is_zero())
                .map(|(idx, b)| {
                    if let Some(src_idx) = self.layout.phi_inv[idx] {
                        if !self.vecs[src_idx].is_empty() {
                            let src = &self.vecs[src_idx];
                            let res =
                                crate::xi_rev::ip_eo_with_xi_rev_with_par::<F>(src, &b.z, false);
                            return b.eta * res;
                        }
                    }
                    F::zero()
                })
                .reduce(|| F::zero(), |a, b| a + b)
        };

        // Handle k=1 buckets in pairs (no vector allocation)
        let n = tbv.buckets.len();
        let mut i = 0;
        while i < n {
            if tbv.buckets[i].k == 1 {
                if i + 1 < n && tbv.buckets[i + 1].k == 1 {
                    let odd_eta = tbv.buckets[i + 1].eta;
                    if let Some(src_idx_even) = self.layout.phi_inv[i] {
                        if !self.vecs[src_idx_even].is_empty() {
                            let val_raw = self.vecs[src_idx_even][0];
                            acc += val_raw * odd_eta;
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        acc
    }

    fn expand_to_field_vec(&self) -> Vec<F> {
        let mut out = vec![F::zero(); self.layout.k_total];
        let mut current_offset = 0;

        for (bucket_idx, &src_idx_opt) in self.layout.phi_inv.iter().enumerate() {
            let k = self.layout.bucket_sizes[bucket_idx];
            if let Some(src_idx) = src_idx_opt {
                if !self.vecs[src_idx].is_empty() {
                    let v = &self.vecs[src_idx];
                    let len = v.len().min(k);
                    out[current_offset..current_offset + len].copy_from_slice(&v[..len]);
                }
            }
            current_offset += k;
        }
        out
    }

    fn ip_with_xi_rev_from_challenges(&self, challenges: &[F]) -> F {
        let tbv = crate::tbv::TBV::new_from_challenges(&self.layout.bucket_sizes, challenges);
        self.ip_with_tbv(&tbv)
    }
}

impl BvaLayout {
    /// Build a BVA layout given vector lengths (non-powers-of-two are padded to next power of two).
    /// Buckets are ordered by decreasing size, so every bucket is naturally
    /// aligned after all larger buckets. The final power-of-two suffix is
    /// logical zero padding and is not represented as a bucket.
    pub fn new(shapes: &[usize]) -> BvaLayout {
        assert!(!shapes.is_empty(), "shapes must be non-empty");
        assert!(
            shapes.iter().all(|&shape| shape != 0),
            "shapes must be non-zero"
        );

        let mut pairs: Vec<(usize, usize)> = shapes
            .iter()
            .enumerate()
            .map(|(idx, &n)| {
                let padded = n
                    .checked_next_power_of_two()
                    .expect("shape padding overflow");
                (idx, padded)
            })
            .collect();

        pairs.sort_by(|(left_idx, left_size), (right_idx, right_size)| {
            right_size
                .cmp(left_size)
                .then_with(|| left_idx.cmp(right_idx))
        });

        let mut k_total = 0usize;
        let mut bucket_sizes = Vec::with_capacity(shapes.len());
        let mut offsets = vec![0usize; shapes.len()];
        let mut phi = vec![0usize; shapes.len()];
        let mut phi_inv: Vec<Option<usize>> = Vec::with_capacity(shapes.len());

        for &(orig_idx, n) in &pairs {
            offsets[orig_idx] = k_total;
            bucket_sizes.push(n);
            phi[orig_idx] = bucket_sizes.len() - 1;
            phi_inv.push(Some(orig_idx));
            k_total = k_total.checked_add(n).expect("layout length overflow");
        }

        let k_total = k_total
            .checked_next_power_of_two()
            .expect("logical layout length overflow");

        BvaLayout {
            k_total,
            bucket_sizes,
            offsets,
            phi,
            phi_inv,
        }
    }
}

impl BvaLayout {
    /// Compute the layout after one EO-fold (length halves, each bucket size halves).
    /// Buckets of size 1 remain size 1. `phi_flat` is cleared; reconstruct if needed.
    pub fn eo_fold_layout(&self) -> BvaLayout {
        assert!(self.k_total > 1, "cannot fold a length-one layout");
        let n_sources = self.phi.len();
        let mut bucket_sizes: Vec<usize> = Vec::with_capacity(self.bucket_sizes.len() / 2 + 1); // Rough estimate
        let mut phi_new: Vec<usize> = vec![usize::MAX; n_sources];
        let mut phi_inv: Vec<Option<usize>> = Vec::with_capacity(self.bucket_sizes.len() / 2 + 1); // Rough estimate

        let mut idx = 0usize;
        while idx < self.bucket_sizes.len() {
            let k = self.bucket_sizes[idx];
            if k == 1 && idx + 1 < self.bucket_sizes.len() && self.bucket_sizes[idx + 1] == 1 {
                // pair (idx, idx+1) into one bucket of size 1
                let left_src = self.phi_inv[idx];
                let right_src = self.phi_inv[idx + 1];
                let b_idx_new = bucket_sizes.len();
                if let Some(s) = left_src {
                    phi_new[s] = b_idx_new;
                }
                if let Some(s) = right_src {
                    phi_new[s] = b_idx_new;
                }
                phi_inv.push(left_src.or(right_src));
                bucket_sizes.push(1);
                idx += 2;
            } else if k == 1 {
                // unpaired tail size-1 bucket
                let src = self.phi_inv[idx];
                let b_idx_new = bucket_sizes.len();
                if let Some(s) = src {
                    phi_new[s] = b_idx_new;
                }
                phi_inv.push(src);
                bucket_sizes.push(1);
                idx += 1;
            } else {
                // k > 1
                let b_idx_new = bucket_sizes.len();
                if let Some(src) = self.phi_inv[idx] {
                    phi_new[src] = b_idx_new;
                }
                phi_inv.push(self.phi_inv[idx]);
                bucket_sizes.push(k / 2);
                idx += 1;
            }
        }

        // Reconstruct offsets from phi_new mapping
        let mut offsets = vec![0usize; n_sources];
        let mut bucket_offsets = Vec::with_capacity(bucket_sizes.len());
        let mut current_offset = 0usize;
        for &k in &bucket_sizes {
            bucket_offsets.push(current_offset);
            current_offset += k;
        }

        for src_idx in 0..n_sources {
            let b_idx = phi_new[src_idx];
            if b_idx != usize::MAX {
                offsets[src_idx] = bucket_offsets[b_idx];
            }
        }

        debug_assert!(bucket_sizes.iter().sum::<usize>() <= self.k_total / 2);
        debug_assert!(phi_inv.iter().all(Option::is_some));

        BvaLayout {
            k_total: self.k_total / 2,
            bucket_sizes,
            offsets,
            phi: phi_new,
            phi_inv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tbv::TBV;
    use ark_bls12_381::Fr;

    fn fold_dense<F: PrimeField>(v: &[F], y: F) -> Vec<F> {
        let mut out = Vec::with_capacity((v.len() + 1) / 2);
        let mut i = 0;
        while i < v.len() {
            let l = v[i];
            let r = v.get(i + 1).copied().unwrap_or_else(F::zero);
            out.push(l + r * y);
            i += 2;
        }
        out
    }

    #[test]
    fn test_layout_no_padding() {
        let shapes = vec![4, 8];
        let layout = BvaLayout::new(&shapes);
        assert_eq!(layout.k_total, 16);
        assert_eq!(layout.bucket_sizes, vec![8, 4]);
        assert_eq!(layout.phi, vec![1, 0]);
        assert_eq!(layout.offsets, vec![8, 0]);
    }

    #[test]
    fn decreasing_layout_has_one_real_bucket_per_vector() {
        let layout = BvaLayout::new(&[3, 5, 2]);

        assert_eq!(layout.k_total, 16);
        assert_eq!(layout.bucket_sizes, vec![8, 4, 2]);
        assert_eq!(layout.phi, vec![1, 0, 2]);
        assert_eq!(layout.phi_inv, vec![Some(1), Some(0), Some(2)]);
        assert_eq!(layout.offsets, vec![8, 0, 12]);
        assert_eq!(layout.bucket_sizes.len(), 3);
        assert!(layout.phi_inv.iter().all(Option::is_some));
    }

    #[test]
    fn test_layout_with_padding() {
        let shapes = vec![16, 4];
        let layout = BvaLayout::new(&shapes);
        assert_eq!(layout.bucket_sizes, vec![16, 4]);
        assert_eq!(layout.phi, vec![0, 1]);
        assert_eq!(layout.offsets[0], 0);
        assert_eq!(layout.offsets[1], 16);
        assert_eq!(layout.k_total, 32);
        assert!(layout.phi_inv.iter().all(Option::is_some));
    }

    #[test]
    fn test_eo_fold_dense_consistency() {
        let bva = BVA::new(vec![vec![1], vec![2], vec![3, 4], vec![5, 6, 7, 8]]);
        let y = Fr::from(9u64);

        let dense0 = bva.expand_to_field_vec();
        let dense_fold = fold_dense(&dense0, y);

        let buf = bva.eo_fold(y);
        let dense_buf = buf.expand_to_field_vec();
        assert_eq!(dense_fold, dense_buf);

        let mut buf2 = buf.clone();
        buf2.eo_fold_in_place(y);
        let dense_buf2 = buf2.expand_to_field_vec();
        let dense_fold2 = fold_dense(&dense_buf, y);
        assert_eq!(dense_fold2, dense_buf2);
    }

    #[test]
    fn test_eo_fold_size1_tail() {
        let bva = BVA::new(vec![vec![1], vec![2], vec![3]]);
        let y = Fr::from(5u64);

        let dense0 = bva.expand_to_field_vec();
        let dense_fold = fold_dense(&dense0, y);

        let buf = bva.eo_fold(y);
        let dense_buf = buf.expand_to_field_vec();
        assert_eq!(dense_fold, dense_buf);

        let mut buf2 = buf.clone();
        buf2.eo_fold_in_place(y);
        let dense_buf2 = buf2.expand_to_field_vec();
        let dense_fold2 = fold_dense(&dense_buf, y);
        assert_eq!(dense_fold2, dense_buf2);
    }

    #[test]
    fn test_layout_with_non_power_of_two_shapes() {
        // shapes will be padded to next power-of-two: 3->4, 5->8
        let shapes = vec![3, 5];
        let layout = BvaLayout::new(&shapes);
        assert_eq!(layout.bucket_sizes, vec![8, 4]);
        assert_eq!(layout.k_total, 16);
        assert_eq!(layout.offsets, vec![8, 0]);
    }

    #[test]
    fn test_bucket_sizes_are_valid_tbv() {
        let shapes = vec![4, 4, 16];
        let layout = BvaLayout::new(&shapes);
        // Check TBV conditions: each size is a power of two, sizes decrease,
        // and every real bucket is aligned at its prefix offset.
        assert!(layout.k_total.is_power_of_two(), "K not power of two");

        let mut prev = 0usize;
        let mut prefix = 0usize;
        for window in layout.bucket_sizes.windows(2).chain(std::iter::once(
            &layout.bucket_sizes[layout.bucket_sizes.len() - 1..],
        )) {
            let k = window[0];
            assert!(k.is_power_of_two(), "bucket size {} not power of two", k);
            if prev != 0 {
                assert!(k <= prev, "bucket sizes not non-increasing");
            }
            prefix += k;
            if window.len() == 2 {
                let k_next = window[1];
                assert!(
                    prefix % k_next == 0,
                    "alignment failed: k_next={} does not divide prefix={}",
                    k_next,
                    prefix
                );
            }
            prev = k;
        }

        // Construct a dummy TBV to ensure TBV::new accepts these sizes.
        let buckets = layout
            .bucket_sizes
            .iter()
            .map(|&k| crate::tbv::Bucket {
                k,
                eta: ark_bls12_381::Fr::from(1u64),
                z: vec![ark_bls12_381::Fr::from(1u64); k.trailing_zeros() as usize],
            })
            .collect::<Vec<_>>();
        let tbv = TBV::new(layout.k_total, buckets);
        assert_eq!(tbv.k_total, layout.k_total);
    }

    #[test]
    fn test_ip_consistency_expand_vs_bucket() {
        // Build a simple TBV: k=[4,4], eta=1/3, z length matches log2(k)
        let tbv = TBV::new(
            8,
            vec![
                crate::tbv::Bucket {
                    k: 4,
                    eta: Fr::from(1u64),
                    z: vec![Fr::from(2u64), Fr::from(2u64)],
                },
                crate::tbv::Bucket {
                    k: 4,
                    eta: Fr::from(3u64),
                    z: vec![Fr::from(5u64), Fr::from(5u64)],
                },
            ],
        );
        // Witness: two vectors of len 4
        let w = BVA::new(vec![vec![1i8, 2, 3, 4], vec![5i8, 6, 7, 8]]);
        assert!(w.check_consistency_with_tbv(&tbv));

        // Bucket-wise IP
        let ip_bucket = w.ip_with_tbv(&tbv);

        // Dense expand and IP
        let dense_w: Vec<Fr> = w.expand_to_field_vec();
        let dense_tbv = tbv.expand();
        let ip_dense: Fr = dense_w
            .iter()
            .zip(dense_tbv.iter())
            .map(|(a, b)| *a * *b)
            .sum();

        assert_eq!(ip_bucket, ip_dense);
    }

    #[test]
    fn test_ip_consistency_odd_and_heterogeneous() {
        // Shapes: [3, 5] -> padded to [4, 8], plus a padding bucket to align
        let layout = BvaLayout::new(&[3, 5]);
        // Build TBV matching layout.bucket_sizes (e.g., [4,4,8])
        let buckets = layout
            .bucket_sizes
            .iter()
            .map(|&k| crate::tbv::Bucket {
                k,
                eta: Fr::from(2u64), // uniform eta for simplicity
                z: vec![Fr::from(3u64); k.trailing_zeros() as usize],
            })
            .collect::<Vec<_>>();
        let tbv = TBV::new(layout.k_total, buckets);

        // Witness vectors: original lengths 3 and 5; padding buckets will be zeros
        let w = BVA::new(vec![
            vec![1i8, -2, 3],       // len 3
            vec![4i8, 5, -6, 7, 8], // len 5
        ]);

        // Bucket-wise IP
        let ip_bucket = w.ip_with_tbv(&tbv);

        // Dense IP
        let dense_w: Vec<Fr> = w.expand_to_field_vec();
        let dense_tbv = tbv.expand();
        let ip_dense: Fr = dense_w
            .iter()
            .zip(dense_tbv.iter())
            .map(|(a, b)| *a * *b)
            .sum();

        assert_eq!(ip_bucket, ip_dense);
    }

    #[test]
    fn test_ip_eo_matches_dense_even_bva_vs_odd_tbv() {
        // Single bucket k=4, no k=1 edge cases.
        let tbv = TBV::new(
            4,
            vec![crate::tbv::Bucket {
                k: 4,
                eta: Fr::from(3u64),
                z: vec![Fr::from(2u64), Fr::from(5u64)],
            }],
        );
        let w = BVA::new(vec![vec![1i8, -2, 3, -4]]);
        assert!(w.check_consistency_with_tbv(&tbv));

        let ip_eo = w.ip_eo_with_tbv(&tbv);

        let dense_w = w.expand_to_field_vec();
        let dense_tbv = tbv.expand();
        let bva_even: Vec<Fr> = dense_w
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, v)| *v)
            .collect();
        let tbv_odd: Vec<Fr> = dense_tbv
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, v)| *v)
            .collect();

        let expected: Fr = bva_even
            .iter()
            .zip(tbv_odd.iter())
            .map(|(a, b)| *a * *b)
            .sum();

        assert_eq!(ip_eo, expected);
    }

    #[test]
    fn test_ip_oe_matches_dense_odd_bva_vs_even_tbv() {
        // Single bucket k=4, no k=1 edge cases.
        let tbv = TBV::new(
            4,
            vec![crate::tbv::Bucket {
                k: 4,
                eta: Fr::from(7u64),
                z: vec![Fr::from(3u64), Fr::from(11u64)],
            }],
        );
        let w = BVA::new(vec![vec![2i8, 4, -6, 8]]);
        assert!(w.check_consistency_with_tbv(&tbv));

        let ip_oe = w.ip_oe_with_tbv(&tbv);

        let dense_w = w.expand_to_field_vec();
        let dense_tbv = tbv.expand();
        let bva_odd: Vec<Fr> = dense_w
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, v)| *v)
            .collect();
        let tbv_even: Vec<Fr> = dense_tbv
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, v)| *v)
            .collect();

        let expected: Fr = bva_odd
            .iter()
            .zip(tbv_even.iter())
            .map(|(a, b)| *a * *b)
            .sum();

        assert_eq!(ip_oe, expected);
    }
}
