//! Authenticated linear folding of committed codewords with identical parameters.
//! The caller binds roots/weights/folded root before deriving the query seed.
use super::*;

impl<B: ShortBaseField> ShortQaFoldPC<B> {
    fn batch_shape(&self, pp: &ShortQaFoldParameters<B>) -> Result<(usize, usize), Error> {
        validate_parameters(self, pp)?;
        Ok((
            pp.sequence.top_code()?.codeword_len()?,
            pp.sequence.base_columns()?,
        ))
    }

    pub(crate) fn fold_codewords(
        &self,
        pp: &ShortQaFoldParameters<B>,
        caches: &[&ShortQaFoldCommitmentCache<B>],
        weights: &[B],
    ) -> Result<(ShortQaFoldCommitment, ShortQaFoldCommitmentCache<B>), Error> {
        let (rows, cols) = self.batch_shape(pp)?;
        if !(2..=128).contains(&caches.len())
            || !caches.len().is_power_of_two()
            || caches.len() != weights.len()
            || caches.iter().any(|cache| {
                cache.n_ext_rows != rows
                    || cache.n_cols != cols
                    || cache.encoded_entries.len() != rows * cols
            })
        {
            return Err(Error::InvalidCommitment);
        }
        const TILE: usize = 1024;
        let mut encoded_entries = vec![B::zero(); rows * cols];
        let fold = |(tile, output): (usize, &mut [B])| {
            for (cache, &weight) in caches.iter().zip(weights) {
                B::mul_add_scalar_slice(output, &cache.encoded_entries[tile * TILE..tile * TILE + output.len()], weight);
            }
        };
        #[cfg(feature = "parallel")]
        encoded_entries.par_chunks_mut(TILE).enumerate().for_each(fold);
        #[cfg(not(feature = "parallel"))]
        encoded_entries.chunks_mut(TILE).enumerate().for_each(fold);
        let tree = build_base_tree::<B>(&encoded_entries, rows, cols)?;
        let root = ShortQaFoldCommitment { root: tree.root() };
        Ok((
            root,
            ShortQaFoldCommitmentCache {
                encoded_entries,
                n_ext_rows: rows,
                n_cols: cols,
                tree,
                commit_timings: [0.0; 4],
                #[cfg(feature = "profile-encoding")]
                encoding_columns: Vec::new(),
            },
        ))
    }

    fn fold_selection(
        &self,
        pp: &ShortQaFoldParameters<B>,
        seed: [u8; 32],
    ) -> Result<RowSelection, Error> {
        let rows = pp.sequence.top_code()?.codeword_len()?;
        let count = pp.sequence.top_code()?.num_queries(
            pp.security_bits,
            pp.sequence.proximity_rf_layers(),
            rows,
        )?;
        Ok(if count == rows {
            RowSelection::All(rows)
        } else {
            RowSelection::Partial(sample_without_replacement(count, rows, seed)?)
        })
    }

    pub(crate) fn prove_codeword_fold(
        &self,
        pp: &ShortQaFoldParameters<B>,
        caches: &[&ShortQaFoldCommitmentCache<B>],
        folded: &ShortQaFoldCommitmentCache<B>,
        seed: [u8; 32],
    ) -> Result<Vec<u8>, Error> {
        let (rows, cols) = self.batch_shape(pp)?;
        let selection = self.fold_selection(pp, seed)?;
        let mut bytes = b"QAFOLDB1".to_vec();
        write_len(&mut bytes, caches.len())?;
        for cache in caches.iter().copied().chain(std::iter::once(folded)) {
            if cache.n_ext_rows != rows
                || cache.n_cols != cols
                || cache.encoded_entries.len() != rows * cols
            {
                return Err(Error::InvalidCommitment);
            }
            let payload = if selection.is_all() {
                ShortQaBaseRows::AllRows {
                    encoded_entries: cache.encoded_entries.clone(),
                    n_rows: rows,
                    n_cols: cols,
                }
            } else {
                let queries = selection
                    .iter()
                    .map(|index| {
                        Ok(ShortQaBaseRowQuery {
                            row: base_encoded_row(&cache.encoded_entries, rows, cols, index),
                            path: cache.tree.path(index)?,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                ShortQaBaseRows::Partial(queries)
            };
            write_base_rows(&mut bytes, &payload)?;
        }
        Ok(bytes)
    }

    pub(crate) fn prove_codeword_fold_recomputed<T>(
        &self, pp: &ShortQaFoldParameters<B>, roots: &[ShortQaFoldCommitment],
        folded: &ShortQaFoldCommitmentCache<B>, seed: [u8; 32], mut rebuild: T,
    ) -> Result<Vec<u8>, Error>
    where T: FnMut(usize, &mut ShortQaFoldCommitWorkspace<B>) -> Result<ShortQaFoldCommitmentCache<B>, Error> {
        let (rows, cols) = self.batch_shape(pp)?;
        if !(2..=128).contains(&roots.len()) || !roots.len().is_power_of_two() {
            return Err(Error::InvalidCommitment);
        }
        let selection = self.fold_selection(pp, seed)?;
        let serialize = |cache: &ShortQaFoldCommitmentCache<B>| -> Result<Vec<u8>, Error> {
            if cache.n_ext_rows != rows || cache.n_cols != cols || cache.encoded_entries.len() != rows * cols {
                return Err(Error::InvalidCommitment);
            }
            let payload = if selection.is_all() {
                ShortQaBaseRows::AllRows { encoded_entries: cache.encoded_entries.clone(), n_rows: rows, n_cols: cols }
            } else {
                ShortQaBaseRows::Partial(selection.iter().map(|index| Ok(ShortQaBaseRowQuery {
                    row: base_encoded_row(&cache.encoded_entries, rows, cols, index), path: cache.tree.path(index)?,
                })).collect::<Result<Vec<_>, Error>>()?)
            };
            let mut bytes = Vec::new(); write_base_rows(&mut bytes, &payload)?; Ok(bytes)
        };
        let mut bytes = b"QAFOLDB1".to_vec();
        write_len(&mut bytes, roots.len())?;
        let mut previous = None;
        let mut payload = Vec::new();
        let mut workspace = ShortQaFoldCommitWorkspace::default();
        for (block, root) in roots.iter().enumerate() {
            // Equal consecutive roots authenticate identical payloads, notably the zero suffix.
            if previous != Some(root.root) {
                let cache = rebuild(block, &mut workspace)?;
                if cache.tree.root() != root.root { return Err(Error::InvalidCommitment); }
                payload = serialize(&cache)?;
                workspace.recycle(cache);
                previous = Some(root.root);
            }
            bytes.extend_from_slice(&payload);
            if bytes.len() > 256 * 1024 * 1024 { return Err(Error::EncodingError); }
        }
        bytes.extend_from_slice(&serialize(folded)?);
        Ok(bytes)
    }

    pub(crate) fn verify_codeword_fold(
        &self,
        pp: &ShortQaFoldParameters<B>,
        roots: &[ShortQaFoldCommitment],
        weights: &[B],
        folded: &ShortQaFoldCommitment,
        seed: [u8; 32],
        bytes: &[u8],
    ) -> Result<bool, Error> {
        let (rows, cols) = self.batch_shape(pp)?;
        if !(2..=128).contains(&roots.len())
            || !roots.len().is_power_of_two()
            || roots.len() != weights.len()
            || bytes.len() > 256 * 1024 * 1024
        {
            return Ok(false);
        }
        let selection = self.fold_selection(pp, seed)?;
        let mut reader = ByteReader::new(bytes);
        if reader.take(8)? != b"QAFOLDB1" || reader.read_len()? != roots.len() {
            return Ok(false);
        }
        let mut payloads = Vec::with_capacity(roots.len() + 1);
        for root in roots.iter().chain(std::iter::once(folded)) {
            let payload = reader.read_base_rows::<B>()?;
            let authenticated = match &payload {
                ShortQaBaseRows::Partial(queries) => {
                    !selection.is_all()
                        && queries.len() == selection.len()
                        && queries.iter().zip(selection.iter()).all(|(query, index)| {
                            query.row.len() == cols
                                && hash_base_row::<B>(&query.row).is_ok_and(|leaf| {
                                    verify_merkle_path(leaf, index, rows, &query.path, &root.root)
                                })
                        })
                }
                ShortQaBaseRows::AllRows {
                    encoded_entries,
                    n_rows,
                    n_cols,
                } => {
                    selection.is_all()
                        && *n_rows == rows
                        && *n_cols == cols
                        && encoded_entries.len() == rows * cols
                        && build_base_tree::<B>(encoded_entries, rows, cols)?.root() == root.root
                }
            };
            if !authenticated {
                return Ok(false);
            }
            payloads.push(payload);
        }
        if reader.remaining() != 0 {
            return Ok(false);
        }
        let entry =
            |payload: &ShortQaBaseRows<B>, query: usize, index: usize, column: usize| match payload
            {
                ShortQaBaseRows::Partial(queries) => queries[query].row[column],
                ShortQaBaseRows::AllRows {
                    encoded_entries, ..
                } => encoded_entries[column * rows + index],
            };
        for (query, index) in selection.iter().enumerate() {
            for column in 0..cols {
                let expected = payloads
                    .iter()
                    .zip(weights)
                    .fold(B::zero(), |sum, (payload, &weight)| {
                        sum + weight * entry(payload, query, index, column)
                    });
                if entry(&payloads[roots.len()], query, index, column) != expected {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}
