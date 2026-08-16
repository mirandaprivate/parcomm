use sha2::{Digest, Sha256};
use std::convert::TryFrom;

use crate::hdr1cs::mlp::MlpLayer;
use crate::utils::heterogeneous::{MatrixBlock, MatrixDims};

pub const FAIR_MLP_FORMAT: &str = "zksmart-pr1cs-fair-relu-mlp-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FairMlpSpec {
    pub layers: usize,
    pub width: usize,
}

impl FairMlpSpec {
    pub fn validate(self) -> Result<(), FairMlpError> {
        if self.layers == 0 {
            return Err(FairMlpError::ZeroLayers);
        }
        if self.width < 4 || !self.width.is_power_of_two() {
            return Err(FairMlpError::InvalidWidth(self.width));
        }
        self.model_parameters().ok_or(FairMlpError::SizeOverflow)?;
        Ok(())
    }

    pub fn weight_parameters(self) -> Option<u128> {
        (self.layers as u128)
            .checked_mul(self.width as u128)?
            .checked_mul(self.width as u128)
    }

    pub fn bias_parameters(self) -> Option<u128> {
        (self.layers as u128).checked_mul(self.width as u128)
    }

    pub fn model_parameters(self) -> Option<u128> {
        self.weight_parameters()?
            .checked_add(self.bias_parameters()?)
    }

    pub fn public_input(self) -> Vec<i32> {
        vec![1; self.width]
    }

    pub fn expected_output(self) -> Vec<i32> {
        (0..self.width)
            .map(|row| if row % 2 == 0 { 1 } else { 0 })
            .collect()
    }

    pub fn fixture_digest(self) -> Result<[u8; 32], FairMlpError> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(FAIR_MLP_FORMAT.as_bytes());
        hasher.update((self.layers as u64).to_le_bytes());
        hasher.update((self.width as u64).to_le_bytes());

        let weight_row = (0..self.width)
            .map(|column| weight_at(column) as i8 as u8)
            .collect::<Vec<_>>();
        let bias = (0..self.width)
            .map(|row| bias_at(row) as i8 as u8)
            .collect::<Vec<_>>();
        for _ in 0..self.layers {
            for _ in 0..self.width {
                hasher.update(&weight_row);
            }
            hasher.update(&bias);
        }
        hasher.update(vec![1_u8; self.width]);
        hasher.update(
            self.expected_output()
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<_>>(),
        );
        Ok(hasher.finalize().into())
    }

    pub fn reference_inference(self) -> Result<Vec<i32>, FairMlpError> {
        self.validate()?;
        let mut values = self.public_input();
        for _ in 0..self.layers {
            let mut next = vec![0_i32; self.width];
            for (row, output) in next.iter_mut().enumerate() {
                let mut accumulator = 0_i64;
                for (column, value) in values.iter().enumerate() {
                    accumulator += i64::from(weight_at(column)) * i64::from(*value);
                }
                accumulator += i64::from(bias_at(row));
                let activated = accumulator.max(0);
                *output = i32::try_from(activated).map_err(|_| FairMlpError::ValueOverflow)?;
            }
            values = next;
        }
        Ok(values)
    }

    pub fn layers(self) -> FairMlpLayers {
        FairMlpLayers {
            spec: self,
            next_layer: 0,
        }
    }
}

pub struct FairMlpLayers {
    spec: FairMlpSpec,
    next_layer: usize,
}

impl Iterator for FairMlpLayers {
    type Item = MlpLayer;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_layer == self.spec.layers {
            return None;
        }
        self.next_layer += 1;
        let columns = (0..self.spec.width)
            .map(|column| vec![weight_at(column); self.spec.width])
            .collect::<Vec<_>>();
        let weight = MatrixBlock::from_columns(
            MatrixDims::new(self.spec.width, self.spec.width).unwrap(),
            columns,
        )
        .unwrap();
        let bias = (0..self.spec.width).map(bias_at).collect();
        Some(MlpLayer { weight, bias })
    }
}

pub fn weight_at(column: usize) -> i32 {
    match column % 4 {
        0 | 1 => 1,
        2 | 3 => -1,
        _ => unreachable!(),
    }
}

pub fn bias_at(row: usize) -> i32 {
    if row % 2 == 0 {
        1
    } else {
        -1
    }
}

pub fn hex_digest(digest: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FairMlpError {
    ZeroLayers,
    InvalidWidth(usize),
    SizeOverflow,
    ValueOverflow,
}

impl std::fmt::Display for FairMlpError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroLayers => write!(formatter, "MLP must contain at least one layer"),
            Self::InvalidWidth(width) => {
                write!(
                    formatter,
                    "width must be a power of two and at least four, got {width}"
                )
            }
            Self::SizeOverflow => write!(formatter, "MLP size overflows its counter"),
            Self::ValueOverflow => write!(formatter, "reference inference overflows i32"),
        }
    }
}

impl std::error::Error for FairMlpError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_fixture_is_stable_across_layers() {
        for layers in [1, 2, 4, 32] {
            let spec = FairMlpSpec { layers, width: 32 };
            assert_eq!(spec.reference_inference().unwrap(), spec.expected_output());
            assert_eq!(spec.reference_inference().unwrap().iter().sum::<i32>(), 16);
        }
    }

    #[test]
    fn parameter_count_matches_the_frozen_full_workload() {
        let spec = FairMlpSpec {
            layers: 1024,
            width: 1024,
        };
        assert_eq!(spec.weight_parameters(), Some(1_073_741_824));
        assert_eq!(spec.bias_parameters(), Some(1_048_576));
        assert_eq!(spec.model_parameters(), Some(1_074_790_400));
    }

    #[test]
    fn fixture_digest_binds_layers_and_width() {
        let base = FairMlpSpec {
            layers: 4,
            width: 4,
        };
        assert_ne!(
            base.fixture_digest().unwrap(),
            FairMlpSpec {
                layers: 5,
                width: 4
            }
            .fixture_digest()
            .unwrap()
        );
        assert_eq!(
            base.fixture_digest().unwrap(),
            base.fixture_digest().unwrap()
        );
    }
}
