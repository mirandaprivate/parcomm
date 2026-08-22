/// Matrix utilities shared by ParComm protocols.
///
#[allow(unused)]
#[macro_use]
extern crate derivative;
#[macro_use]
extern crate ark_std;


pub mod utils;


pub use crate::utils::linear;
pub use crate::utils::matdef::{DenseMatCM,DenseMatFieldCM,ShortInt};
pub use crate::utils::xi;

pub type MyInt = poly_commit::MyInt;
pub type MyShortInt = MyInt;

#[cfg(test)]
mod tests {
    #[test]
    fn test_basic() {
        assert_eq!(1 + 1, 2);
    }
}
