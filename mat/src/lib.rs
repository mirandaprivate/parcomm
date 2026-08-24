/// Matrix utilities shared by ParComm protocols.
///
#[allow(unused)]
#[macro_use]
extern crate derivative;
#[macro_use]
extern crate ark_std;

pub mod utils;

pub use crate::utils::linear;
pub use crate::utils::matdef::{DenseMatCM, DenseMatFieldCM, ShortInt};
pub use crate::utils::xi;

pub type MyInt = poly_commit::MyInt;
pub type MyShortInt = MyInt;

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use super::{DenseMatCM, ShortInt};

    fn assert_short_int<Integer: ShortInt>() {}

    #[test]
    fn test_basic() {
        assert_eq!(1 + 1, 2);
    }

    #[test]
    fn short_int_accepts_i8_and_i16() {
        assert_short_int::<i8>();
        assert_short_int::<i16>();

        let mut matrix = DenseMatCM::<i16, Fr>::new(2, 2);
        matrix.from_vec(&vec![i16::MIN, 30_000, -1_024, 17_000]);
        assert_eq!(matrix.data[0], vec![i16::MIN, 30_000]);
        assert_eq!(matrix.data[1], vec![-1_024, 17_000]);
    }
}
