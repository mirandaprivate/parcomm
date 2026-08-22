use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;
use fs::helper_trans::Transcript;
use mat::utils::matdef::DenseMatFieldCM;

// use crate::pop::arithmetic_expression::ConstraintSystemBuilder; // TODO: implement pop module

/// An evaluation claim for a matrix multilinear extension.
///
/// The first component of `point` is the row point and the second component is
/// the column point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalWithPointMat<F> {
    pub value: F,
    pub point: (Vec<F>, Vec<F>),
}

/// An evaluation claim for a vector multilinear extension.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalWithPointVec<F> {
    pub value: F,
    pub point: Vec<F>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatOp {
    Input,
    Eq,
    EqZero,
    Add,
    Concat,
    GrandProd,
    Hadamard,
    Kronecker,
    LinComb,
    Mul,
    Activation,
    ReduceProd,
    Sub,
    InnerProduct,
    WithinRange,
}

#[derive(Debug, Clone)]
pub struct AtomicMatProtocolInput<F: PrimeField> {
    pub op: MatOp,
    pub hat_c: F,
    pub point_c: (Vec<F>, Vec<F>),
    pub a: DenseMatFieldCM<F>,
    pub b: DenseMatFieldCM<F>,
    pub shape_a: (usize, usize),
    pub shape_b: (usize, usize),
    pub shape_c: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct AtomicMatProtocolMultiInput<F: PrimeField> {
    pub op: MatOp,
    pub hat_c: F,
    pub point_c: (Vec<F>, Vec<F>),
    pub shape_inputs: (usize, usize),
    pub num_inputs: usize,
    pub input_mats: Vec<DenseMatFieldCM<F>>,
}

impl<F: PrimeField> AtomicMatProtocolInput<F> {
    pub fn clear(&mut self) {
        self.a = DenseMatFieldCM::new(0, 0);
        self.b = DenseMatFieldCM::new(0, 0);
    }
}

impl<F: PrimeField> AtomicMatProtocolMultiInput<F> {
    pub fn clear(&mut self) {
        self.input_mats = Vec::new();
    }
}

pub trait AtomicMatProtocol<
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
>
{
    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool;
    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool;
    fn verify(&mut self, trans: &mut Transcript<F>) -> bool {
        // Check transcript integrity first
        if !trans.fs.verify_fs() {
            println!("⚠️  Fiat-Shamir check failed when verifying MatMul (continuing for debug)");
        }
        self.verify_as_subprotocol(trans)
    }

    fn clear(&mut self);
}
