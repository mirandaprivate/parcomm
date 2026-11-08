use crate::{Result, check};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bucket {
    pub matrix: usize,
    pub row_offset: usize,
    pub col_offset: usize,
    pub rows: usize,
    pub cols: usize,
    pub physical_rows: usize,
    pub physical_cols: usize,
    pub offset: usize,
}
impl Bucket {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    pub shapes: Vec<Shape>,
    pub buckets: Vec<Bucket>,
    pub active: usize,
    pub logical: usize,
}

/// Same four-interval cap and last-interval rounding as modelconvert.py.
pub fn intervals(length: usize, budget: usize) -> Vec<(usize, usize)> {
    let mut sizes = Vec::new();
    let mut remaining = length;
    while remaining > 0 && sizes.len() < budget {
        let size = 1usize << remaining.ilog2();
        sizes.push(size);
        remaining -= size;
    }
    if remaining > 0 {
        *sizes.last_mut().unwrap() *= 2;
    }
    let mut offset = 0;
    sizes
        .into_iter()
        .map(|size| {
            let current = offset;
            offset += size;
            (current, size)
        })
        .collect()
}
impl Layout {
    pub fn new(shapes: &[Shape]) -> Result<Self> {
        check(
            !shapes.is_empty() && shapes.len() <= 65536,
            "invalid matrix count",
        )?;
        let mut buckets = Vec::new();
        for (matrix, shape) in shapes.iter().enumerate() {
            check(
                shape.rows > 0
                    && shape.cols > 0
                    && shape.rows <= (1 << 30)
                    && shape.cols <= (1 << 30),
                "invalid matrix shape",
            )?;
            for (row_offset, rows) in intervals(shape.rows, 4) {
                for (col_offset, cols) in intervals(shape.cols, 4) {
                    let len = rows.checked_mul(cols).ok_or("matrix size overflow")?;
                    check(len <= 1usize << 37, "matrix exceeds supported capacity")?;
                    buckets.push(Bucket {
                        matrix,
                        row_offset,
                        col_offset,
                        rows,
                        cols,
                        physical_rows: rows.min(shape.rows - row_offset),
                        physical_cols: cols.min(shape.cols - col_offset),
                        offset: 0,
                    });
                }
            }
        }
        buckets.sort_by_key(|bucket| {
            (
                std::cmp::Reverse(bucket.len()),
                bucket.matrix,
                bucket.row_offset,
                bucket.col_offset,
            )
        });
        let mut active = 0usize;
        for bucket in &mut buckets {
            bucket.offset = active;
            active = active
                .checked_add(bucket.len())
                .ok_or("BVA size overflow")?;
            check(
                active <= 1usize << 37,
                "collection exceeds supported capacity",
            )?;
            check(bucket.offset % bucket.len() == 0, "unaligned BVA bucket")?;
        }
        let logical = active.max(1024).next_power_of_two();
        Ok(Self {
            shapes: shapes.to_vec(),
            buckets,
            active,
            logical,
        })
    }
    pub fn encode(&self, matrices: &[Vec<i8>]) -> Result<Vec<i8>> {
        check(matrices.len() == self.shapes.len(), "matrix count mismatch")?;
        for (values, shape) in matrices.iter().zip(&self.shapes) {
            check(
                values.len() == shape.rows * shape.cols,
                "matrix length mismatch",
            )?;
        }
        let mut values = vec![0; self.active];
        for bucket in &self.buckets {
            for row in 0..bucket.physical_rows {
                let source =
                    (bucket.row_offset + row) * self.shapes[bucket.matrix].cols + bucket.col_offset;
                let destination = bucket.offset + row * bucket.cols;
                values[destination..destination + bucket.physical_cols].copy_from_slice(
                    &matrices[bucket.matrix][source..source + bucket.physical_cols],
                );
            }
        }
        Ok(values)
    }
    pub fn validate_values(&self, values: &[i8]) -> Result<()> {
        check(values.len() == self.active, "active BVA length mismatch")?;
        for bucket in &self.buckets {
            for row in 0..bucket.rows {
                let valid = if row < bucket.physical_rows {
                    bucket.physical_cols
                } else {
                    0
                };
                check(
                    values[bucket.offset + row * bucket.cols + valid
                        ..bucket.offset + (row + 1) * bucket.cols]
                        .iter()
                        .all(|&v| v == 0),
                    "nonzero boundary padding",
                )?;
            }
        }
        Ok(())
    }
}
