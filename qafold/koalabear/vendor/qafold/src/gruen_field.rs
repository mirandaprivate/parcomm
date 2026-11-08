//! Scalar prime fields and recursively typed quadratic challenge fields.
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

pub trait Field:
    Copy
    + Send
    + Sync
    + Debug
    + Eq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const ORDER: u128;
    fn from_u64(x: u64) -> Self;
    fn random(rng: &mut Rng) -> Self;
    fn inverse(self) -> Option<Self>;
    /// Highest quadratic generator; zero for prime fields.
    fn generator() -> Self;
    /// Multiply by the highest generator without a general field multiplication.
    fn mul_generator(self) -> Self;
    fn validate();
    fn pow(self, mut n: u128) -> Self {
        let mut a = self;
        let mut out = Self::ONE;
        while n != 0 {
            if n & 1 != 0 {
                out = out * a;
            }
            a = a * a;
            n >>= 1;
        }
        out
    }
}

pub trait Prime: Field {
    const MODULUS: u64;
    const NAME: &'static str;
    const TWO_ADICITY: usize;
    fn value(self) -> u64;
    fn root(log: usize) -> Option<Self> {
        if log > Self::TWO_ADICITY {
            return None;
        }
        if log == 0 {
            return Some(Self::ONE);
        }
        // Find a generator of the two-primary subgroup, not necessarily of F_p^*.
        for candidate in 2..1000 {
            let x = Self::from_u64(candidate).pow((Self::MODULUS as u128 - 1) >> log);
            if x.pow(1 << (log - 1)) != Self::ONE {
                return Some(x);
            }
        }
        None
    }
}

/// An algebra over the trace's base field, with a cheap base-scalar multiplication.
pub trait Extension<B: Prime>: Field {
    const DEGREE: usize;
    fn lift(x: B) -> Self;
    fn scale(self, x: B) -> Self;
}

macro_rules! prime {
    ($name:ident, $word:ty, $p:expr, $adicity:expr) => {
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $name($word);
        impl Add for $name {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                let sum = self.0 as u128 + rhs.0 as u128;
                Self(
                    (if sum >= $p as u128 {
                        sum - $p as u128
                    } else {
                        sum
                    }) as $word,
                )
            }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                let (a, b) = (self.0 as u64, rhs.0 as u64);
                Self((if a >= b { a - b } else { $p - (b - a) }) as $word)
            }
        }
        impl Neg for $name {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self::ZERO - self
            }
        }
        impl Mul for $name {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                // For a <=32-bit prime the product fits u64. A constant divisor
                // permits native multiply-high reduction without wide division.
                if $p as u128 <= u32::MAX as u128 {
                    Self(((self.0 as u64 * rhs.0 as u64) % ($p as u64)) as $word)
                } else {
                    Self(((self.0 as u128 * rhs.0 as u128) % $p as u128) as $word)
                }
            }
        }
        impl Field for $name {
            const ZERO: Self = Self(0);
            const ONE: Self = Self(1);
            const ORDER: u128 = $p as u128;
            #[inline]
            fn from_u64(x: u64) -> Self {
                Self((x % $p) as $word)
            }
            fn random(rng: &mut Rng) -> Self {
                // Rejection sampling, including for the almost-2^64 Goldilocks modulus.
                let threshold = ($p as u64).wrapping_neg() % $p;
                loop {
                    let x = rng.next_u64();
                    if x >= threshold {
                        return Self::from_u64(x);
                    }
                }
            }
            fn inverse(self) -> Option<Self> {
                (self != Self::ZERO).then(|| self.pow($p as u128 - 2))
            }
            fn generator() -> Self {
                Self::ZERO
            }
            #[inline]
            fn mul_generator(self) -> Self {
                Self::ZERO
            }
            fn validate() {}
        }
        impl Prime for $name {
            const MODULUS: u64 = $p;
            const NAME: &'static str = stringify!($name);
            const TWO_ADICITY: usize = $adicity;
            fn value(self) -> u64 {
                self.0 as u64
            }
        }
        impl Extension<Self> for $name {
            const DEGREE: usize = 1;
            fn lift(x: Self) -> Self {
                x
            }
            fn scale(self, x: Self) -> Self {
                self * x
            }
        }
    };
}
prime!(F251, u8, 251, 1);
prime!(F65521, u16, 65521, 4);
prime!(Mersenne31, u32, 2147483647, 1);
prime!(KoalaBear, u32, 2130706433, 24);
prime!(Goldilocks, u64, 18446744069414584321, 32);

/// `F[u]/(u^2-(F::generator()+OFFSET))`; coefficients use native base word sizes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Quad<F: Field, const OFFSET: u64>(pub F, pub F);

impl<F: Field, const O: u64> Quad<F, O> {
    #[inline]
    fn mul_nu(x: F) -> F {
        let offset = if O == 0 {
            F::ZERO
        } else if O == 1 {
            x
        } else if O == 2 {
            x + x
        } else {
            x * F::from_u64(O)
        };
        x.mul_generator() + offset
    }
}
impl<F: Field, const O: u64> Add for Quad<F, O> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl<F: Field, const O: u64> Sub for Quad<F, O> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}
impl<F: Field, const O: u64> Neg for Quad<F, O> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }
}
impl<F: Field, const O: u64> Mul for Quad<F, O> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let ac = self.0 * rhs.0;
        let bd = self.1 * rhs.1;
        Self(
            ac + Self::mul_nu(bd),
            (self.0 + self.1) * (rhs.0 + rhs.1) - ac - bd,
        )
    }
}
impl<F: Field, const O: u64> Field for Quad<F, O> {
    const ZERO: Self = Self(F::ZERO, F::ZERO);
    const ONE: Self = Self(F::ONE, F::ZERO);
    const ORDER: u128 = F::ORDER * F::ORDER;
    fn from_u64(x: u64) -> Self {
        Self(F::from_u64(x), F::ZERO)
    }
    fn random(rng: &mut Rng) -> Self {
        Self(F::random(rng), F::random(rng))
    }
    fn inverse(self) -> Option<Self> {
        let norm_inv = (self.0 * self.0 - Self::mul_nu(self.1 * self.1)).inverse()?;
        Some(Self(self.0 * norm_inv, -self.1 * norm_inv))
    }
    fn generator() -> Self {
        Self(F::ZERO, F::ONE)
    }
    #[inline]
    fn mul_generator(self) -> Self {
        Self(Self::mul_nu(self.1), self.0)
    }
    fn validate() {
        F::validate();
        assert_eq!(
            (F::generator() + F::from_u64(O)).pow((F::ORDER - 1) / 2),
            -F::ONE,
            "quadratic nonresidue must satisfy Euler's criterion"
        );
    }
}
impl<B: Prime, F: Extension<B>, const O: u64> Extension<B> for Quad<F, O> {
    const DEGREE: usize = 2 * F::DEGREE;
    #[inline]
    fn lift(x: B) -> Self {
        Self(F::lift(x), F::ZERO)
    }
    #[inline]
    fn scale(self, x: B) -> Self {
        Self(self.0.scale(x), self.1.scale(x))
    }
}

pub type E251 = Quad<Quad<Quad<Quad<F251, 250>, 1>, 0>, 0>;
pub type E65521 = Quad<Quad<Quad<F65521, 17>, 0>, 0>;
pub type EMersenne31 = Quad<Quad<Mersenne31, 2147483646>, 2>;
pub type EGoldilocks = Quad<Goldilocks, 7>;
/// v^4=3, with u=v^2. Nested storage order is [1,v^2,v,v^3].
/// Same binomial extension as Plonky3's X^4-3, with an explicit basis permutation.
pub type EKoalaBear = Quad<Quad<KoalaBear, 3>, 0>;

/// Deterministic experiment RNG. Not a Fiat-Shamir transcript or cryptographic RNG.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}
