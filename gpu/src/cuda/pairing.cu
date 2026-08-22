// Batched BLS12-381 Miller loops on the device.
//
// `inner_pairing_product` is a product of pairings, which is one multi-Miller
// loop followed by a single final exponentiation. Only the Miller loops grow
// with the input, so this file computes them on the device and reduces them to
// one Fp12; the caller runs the lone final exponentiation on the host, where
// arkworks already has a tuned implementation.
//
// The field tower is written out here rather than taken from ICICLE, whose
// device-side arithmetic only compiles inside its own CUDA backend. Everything
// stays in Montgomery form so the limbs match `ark_ff` and can cross the
// boundary without conversion.

#include <cstddef>
#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// BLS12-381 base field, Montgomery form
// ---------------------------------------------------------------------------

constexpr int kBaseLimbs = 6;

struct Fq {
  std::uint64_t limbs[kBaseLimbs];
};

static_assert(sizeof(Fq) == 48, "unexpected BLS12-381 Fq layout");

__constant__ std::uint64_t kBaseModulus[kBaseLimbs] = {
  0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
  0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL,
};

// R mod p, which is the Montgomery form of one.
__constant__ std::uint64_t kBaseOne[kBaseLimbs] = {
  0x760900000002fffdULL, 0xebf4000bc40c0002ULL, 0x5f48985753c758baULL,
  0x77ce585370525745ULL, 0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL,
};

// -p^{-1} mod 2^64, the CIOS reduction constant.
constexpr std::uint64_t kBaseModulusInverse = 0x89f3fffcfffcfffdULL;

__device__ __forceinline__ bool base_is_less_than_modulus(const std::uint64_t* value)
{
#pragma unroll
  for (int limb = kBaseLimbs - 1; limb >= 0; --limb) {
    if (value[limb] != kBaseModulus[limb]) { return value[limb] < kBaseModulus[limb]; }
  }
  return false;
}

__device__ __forceinline__ void base_subtract_modulus(std::uint64_t* value)
{
  std::uint64_t borrow = 0;
#pragma unroll
  for (int limb = 0; limb < kBaseLimbs; ++limb) {
    const std::uint64_t subtrahend = kBaseModulus[limb];
    const std::uint64_t intermediate = value[limb] - subtrahend;
    const std::uint64_t next_borrow =
      (value[limb] < subtrahend) || (borrow == 1 && intermediate == 0);
    value[limb] = intermediate - borrow;
    borrow = next_borrow;
  }
}

__device__ __forceinline__ Fq fq_zero()
{
  Fq result;
#pragma unroll
  for (int limb = 0; limb < kBaseLimbs; ++limb) { result.limbs[limb] = 0; }
  return result;
}

__device__ __forceinline__ Fq fq_one()
{
  Fq result;
#pragma unroll
  for (int limb = 0; limb < kBaseLimbs; ++limb) { result.limbs[limb] = kBaseOne[limb]; }
  return result;
}

__device__ __forceinline__ bool fq_is_zero(const Fq& value)
{
  std::uint64_t accumulated = 0;
#pragma unroll
  for (int limb = 0; limb < kBaseLimbs; ++limb) { accumulated |= value.limbs[limb]; }
  return accumulated == 0;
}

__device__ __forceinline__ Fq fq_add(const Fq& left, const Fq& right)
{
  Fq result;
  std::uint64_t carry = 0;
#pragma unroll
  for (int limb = 0; limb < kBaseLimbs; ++limb) {
    const std::uint64_t sum = left.limbs[limb] + right.limbs[limb];
    const std::uint64_t carried = sum + carry;
    result.limbs[limb] = carried;
    carry = (sum < left.limbs[limb]) || (carry == 1 && carried == 0);
  }
  // The modulus is under 2^383, so at most one conditional subtraction runs.
  if (carry != 0 || !base_is_less_than_modulus(result.limbs)) {
    base_subtract_modulus(result.limbs);
  }
  return result;
}

__device__ __forceinline__ Fq fq_negate(const Fq& value)
{
  if (fq_is_zero(value)) { return value; }

  Fq result;
  std::uint64_t borrow = 0;
#pragma unroll
  for (int limb = 0; limb < kBaseLimbs; ++limb) {
    const std::uint64_t minuend = kBaseModulus[limb];
    const std::uint64_t intermediate = minuend - value.limbs[limb];
    const std::uint64_t next_borrow =
      (minuend < value.limbs[limb]) || (borrow == 1 && intermediate == 0);
    result.limbs[limb] = intermediate - borrow;
    borrow = next_borrow;
  }
  return result;
}

__device__ __forceinline__ Fq fq_sub(const Fq& left, const Fq& right)
{
  return fq_add(left, fq_negate(right));
}

/// CIOS Montgomery multiplication.
///
/// Both inputs and the result are Montgomery forms, so this matches `ark_ff`
/// multiplication limb for limb.
__device__ __forceinline__ Fq fq_mul(const Fq& left, const Fq& right)
{
  std::uint64_t accumulator[kBaseLimbs + 1] = {0, 0, 0, 0, 0, 0, 0};

#pragma unroll
  for (int i = 0; i < kBaseLimbs; ++i) {
    // accumulator += left * right.limbs[i]
    std::uint64_t carry = 0;
#pragma unroll
    for (int j = 0; j < kBaseLimbs; ++j) {
      const std::uint64_t low = left.limbs[j] * right.limbs[i];
      const std::uint64_t high = __umul64hi(left.limbs[j], right.limbs[i]);
      std::uint64_t sum = accumulator[j] + low;
      std::uint64_t next_carry = high + (sum < low);
      sum += carry;
      next_carry += (sum < carry);
      accumulator[j] = sum;
      carry = next_carry;
    }
    accumulator[kBaseLimbs] = accumulator[kBaseLimbs] + carry;
    const std::uint64_t overflow = (accumulator[kBaseLimbs] < carry) ? 1ULL : 0ULL;

    // accumulator = (accumulator + modulus * m) / 2^64
    const std::uint64_t m = accumulator[0] * kBaseModulusInverse;
    std::uint64_t reduction_carry = 0;
    {
      const std::uint64_t low = kBaseModulus[0] * m;
      const std::uint64_t high = __umul64hi(kBaseModulus[0], m);
      const std::uint64_t sum = accumulator[0] + low;
      reduction_carry = high + (sum < low);
    }
#pragma unroll
    for (int j = 1; j < kBaseLimbs; ++j) {
      const std::uint64_t low = kBaseModulus[j] * m;
      const std::uint64_t high = __umul64hi(kBaseModulus[j], m);
      std::uint64_t sum = accumulator[j] + low;
      std::uint64_t next_carry = high + (sum < low);
      sum += reduction_carry;
      next_carry += (sum < reduction_carry);
      accumulator[j - 1] = sum;
      reduction_carry = next_carry;
    }
    const std::uint64_t tail = accumulator[kBaseLimbs] + reduction_carry;
    accumulator[kBaseLimbs - 1] = tail;
    accumulator[kBaseLimbs] = overflow + (tail < reduction_carry);
  }

  Fq result;
#pragma unroll
  for (int limb = 0; limb < kBaseLimbs; ++limb) { result.limbs[limb] = accumulator[limb]; }
  if (accumulator[kBaseLimbs] != 0 || !base_is_less_than_modulus(result.limbs)) {
    base_subtract_modulus(result.limbs);
  }
  return result;
}

/// p - 2, the exponent that inverts a non-zero element by Fermat's theorem.
__constant__ std::uint64_t kBaseModulusMinusTwo[kBaseLimbs] = {
  0xb9feffffffffaaa9ULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
  0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL,
};

constexpr int kBaseModulusBits = 381;

/// Inverts by exponentiation.
///
/// The folds normalize one point per thread, and a single inversion costs a few
/// percent of the surrounding scalar multiplication, so this avoids the
/// cross-thread coordination a batch inversion would need.
__device__ Fq fq_inverse(const Fq& value)
{
  Fq result = fq_one();
#pragma unroll 1
  for (int bit = kBaseModulusBits - 1; bit >= 0; --bit) {
    result = fq_mul(result, result);
    if ((kBaseModulusMinusTwo[bit / 64] >> (bit % 64)) & 1ULL) {
      result = fq_mul(result, value);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Fp2 = Fq[u] / (u^2 + 1)
// ---------------------------------------------------------------------------

struct Fp2 {
  Fq c0;
  Fq c1;
};

__device__ __forceinline__ Fp2 fp2_zero() { return Fp2{fq_zero(), fq_zero()}; }
__device__ __forceinline__ Fp2 fp2_one() { return Fp2{fq_one(), fq_zero()}; }

__device__ __forceinline__ Fp2 fp2_add(const Fp2& left, const Fp2& right)
{
  return Fp2{fq_add(left.c0, right.c0), fq_add(left.c1, right.c1)};
}

__device__ __forceinline__ Fp2 fp2_sub(const Fp2& left, const Fp2& right)
{
  return Fp2{fq_sub(left.c0, right.c0), fq_sub(left.c1, right.c1)};
}

__device__ __forceinline__ Fp2 fp2_negate(const Fp2& value)
{
  return Fp2{fq_negate(value.c0), fq_negate(value.c1)};
}

/// Multiplies both components by a base field element.
__device__ __forceinline__ Fp2 fp2_scale(const Fp2& value, const Fq& scalar)
{
  return Fp2{fq_mul(value.c0, scalar), fq_mul(value.c1, scalar)};
}

__device__ __forceinline__ Fp2 fp2_mul(const Fp2& left, const Fp2& right)
{
  const Fq v0 = fq_mul(left.c0, right.c0);
  const Fq v1 = fq_mul(left.c1, right.c1);
  const Fq cross = fq_mul(fq_add(left.c0, left.c1), fq_add(right.c0, right.c1));
  return Fp2{fq_sub(v0, v1), fq_sub(cross, fq_add(v0, v1))};
}

__device__ __forceinline__ Fp2 fp2_sqr(const Fp2& value)
{
  // (c0 + c1 u)^2 = (c0 + c1)(c0 - c1) + 2 c0 c1 u
  const Fq sum = fq_add(value.c0, value.c1);
  const Fq difference = fq_sub(value.c0, value.c1);
  const Fq product = fq_mul(value.c0, value.c1);
  return Fp2{fq_mul(sum, difference), fq_add(product, product)};
}

__device__ __forceinline__ bool fp2_is_zero(const Fp2& value)
{
  return fq_is_zero(value.c0) && fq_is_zero(value.c1);
}

/// `(c0 + c1 u)^-1 = (c0 - c1 u) / (c0^2 + c1^2)`.
__device__ Fp2 fp2_inverse(const Fp2& value)
{
  const Fq norm = fq_add(fq_mul(value.c0, value.c0), fq_mul(value.c1, value.c1));
  const Fq norm_inverse = fq_inverse(norm);
  return Fp2{fq_mul(value.c0, norm_inverse), fq_negate(fq_mul(value.c1, norm_inverse))};
}

/// Multiplies by `u + 1`, the non-residue generating Fp6 over Fp2.
__device__ __forceinline__ void mul_fp2_by_nonresidue(Fp2& value)
{
  const Fq previous = value.c0;
  value.c0 = fq_sub(value.c0, value.c1);
  value.c1 = fq_add(value.c1, previous);
}

// ---------------------------------------------------------------------------
// Fp6 = Fp2[v] / (v^3 - (u + 1))
// ---------------------------------------------------------------------------

struct Fp6 {
  Fp2 c0;
  Fp2 c1;
  Fp2 c2;
};

__device__ __forceinline__ Fp6 fp6_zero() { return Fp6{fp2_zero(), fp2_zero(), fp2_zero()}; }

__device__ __forceinline__ Fp6 fp6_add(const Fp6& left, const Fp6& right)
{
  return Fp6{
    fp2_add(left.c0, right.c0), fp2_add(left.c1, right.c1), fp2_add(left.c2, right.c2)};
}

__device__ __forceinline__ Fp6 fp6_sub(const Fp6& left, const Fp6& right)
{
  return Fp6{
    fp2_sub(left.c0, right.c0), fp2_sub(left.c1, right.c1), fp2_sub(left.c2, right.c2)};
}

__device__ __forceinline__ Fp6 fp6_negate(const Fp6& value)
{
  return Fp6{fp2_negate(value.c0), fp2_negate(value.c1), fp2_negate(value.c2)};
}

__device__ Fp6 fp6_mul(const Fp6& left, const Fp6& right)
{
  // Karatsuba across the three coefficients.
  const Fp2 v0 = fp2_mul(left.c0, right.c0);
  const Fp2 v1 = fp2_mul(left.c1, right.c1);
  const Fp2 v2 = fp2_mul(left.c2, right.c2);

  Fp2 t0 = fp2_sub(
    fp2_mul(fp2_add(left.c1, left.c2), fp2_add(right.c1, right.c2)), fp2_add(v1, v2));
  mul_fp2_by_nonresidue(t0);
  t0 = fp2_add(t0, v0);

  Fp2 shifted = v2;
  mul_fp2_by_nonresidue(shifted);
  const Fp2 t1 = fp2_add(
    fp2_sub(fp2_mul(fp2_add(left.c0, left.c1), fp2_add(right.c0, right.c1)), fp2_add(v0, v1)),
    shifted);

  const Fp2 t2 = fp2_add(
    fp2_sub(fp2_mul(fp2_add(left.c0, left.c2), fp2_add(right.c0, right.c2)), fp2_add(v0, v2)),
    v1);

  return Fp6{t0, t1, t2};
}

/// Multiplies by `v`, the non-residue generating Fp12 over Fp6.
__device__ __forceinline__ void mul_fp6_by_nonresidue(Fp6& value)
{
  const Fp2 previous = value.c1;
  value.c1 = value.c0;
  value.c0 = value.c2;
  mul_fp2_by_nonresidue(value.c0);
  value.c2 = previous;
}

// ---------------------------------------------------------------------------
// Fp12 = Fp6[w] / (w^2 - v)
// ---------------------------------------------------------------------------

struct Fp12 {
  Fp6 c0;
  Fp6 c1;
};

__device__ __forceinline__ Fp12 fp12_one()
{
  return Fp12{Fp6{fp2_one(), fp2_zero(), fp2_zero()}, fp6_zero()};
}

__device__ Fp12 fp12_mul(const Fp12& left, const Fp12& right)
{
  const Fp6 a = fp6_mul(left.c0, right.c0);
  const Fp6 b = fp6_mul(left.c1, right.c1);
  Fp6 shifted = b;
  mul_fp6_by_nonresidue(shifted);
  const Fp6 cross = fp6_sub(
    fp6_mul(fp6_add(left.c0, left.c1), fp6_add(right.c0, right.c1)), fp6_add(a, b));
  return Fp12{fp6_add(a, shifted), cross};
}

__device__ Fp12 fp12_sqr(const Fp12& value)
{
  // Complex squaring: (a + b w)^2 = (a + b)(a + v b) - a b - v a b + 2 a b w.
  const Fp6 product = fp6_mul(value.c0, value.c1);
  Fp6 shifted = value.c1;
  mul_fp6_by_nonresidue(shifted);
  Fp6 lead = fp6_sub(
    fp6_mul(fp6_add(value.c0, value.c1), fp6_add(value.c0, shifted)), product);
  Fp6 shifted_product = product;
  mul_fp6_by_nonresidue(shifted_product);
  lead = fp6_sub(lead, shifted_product);
  return Fp12{lead, fp6_add(product, product)};
}

// ---------------------------------------------------------------------------
// Miller loop
// ---------------------------------------------------------------------------

struct G1Affine {
  Fq x;
  Fq y;
};

struct G2Affine {
  Fp2 x;
  Fp2 y;
};

// BLS12-381 has a negative, low-Hamming-weight parameter; the loop runs over its
// 64 bits and the sign is applied by conjugating the result.
__constant__ std::uint32_t kLoopParameter[2] = {0x00010000, 0xd2010000};
constexpr int kLoopBits = 64;

__device__ __forceinline__ bool loop_bit(int index)
{
  return (kLoopParameter[index / 32] >> (index % 32)) & 1u;
}

/// The twisted curve coefficient `b = 4 + 4u`, built from the field's own one so
/// that it lands in Montgomery form.
__device__ __forceinline__ Fp2 weierstrass_b_g2()
{
  const Fq one = fq_one();
  const Fq two = fq_add(one, one);
  const Fq four = fq_add(two, two);
  return Fp2{four, four};
}

/// Doubles the running G2 point and returns the line coefficients.
///
/// `r` carries the accumulator as `(x, y, z)` in projective coordinates, which
/// is why it is typed as an Fp6 rather than a curve point.
__device__ Fp6 double_in_place(Fp6& r, const Fq& two_inv)
{
  Fp2& x = r.c0;
  Fp2& y = r.c1;
  Fp2& z = r.c2;

  Fp2 a = fp2_scale(fp2_mul(x, y), two_inv);
  const Fp2 b = fp2_sqr(y);
  const Fp2 c = fp2_sqr(z);
  const Fp2 e = fp2_mul(weierstrass_b_g2(), fp2_add(fp2_add(c, c), c));
  const Fp2 f = fp2_add(fp2_add(e, e), e);
  const Fp2 g = fp2_scale(fp2_add(b, f), two_inv);
  const Fp2 h = fp2_sub(fp2_sqr(fp2_add(y, z)), fp2_add(b, c));
  const Fp2 i = fp2_sub(e, b);
  const Fp2 j = fp2_sqr(x);
  const Fp2 e_square = fp2_sqr(e);

  x = fp2_mul(a, fp2_sub(b, f));
  y = fp2_sub(fp2_sqr(g), fp2_add(fp2_add(e_square, e_square), e_square));
  z = fp2_mul(b, h);

  // BLS12-381 uses an M-twist.
  return Fp6{i, fp2_add(fp2_add(j, j), j), fp2_negate(h)};
}

/// Adds the fixed G2 point to the running one and returns the line coefficients.
__device__ Fp6 add_in_place(Fp6& r, const G2Affine& q)
{
  Fp2& x = r.c0;
  Fp2& y = r.c1;
  Fp2& z = r.c2;

  const Fp2 theta = fp2_sub(y, fp2_mul(q.y, z));
  const Fp2 lambda = fp2_sub(x, fp2_mul(q.x, z));
  const Fp2 c = fp2_sqr(theta);
  const Fp2 d = fp2_sqr(lambda);
  const Fp2 e = fp2_mul(lambda, d);
  const Fp2 f = fp2_mul(z, c);
  const Fp2 g = fp2_mul(x, d);
  const Fp2 h = fp2_sub(fp2_add(e, f), fp2_add(g, g));

  x = fp2_mul(lambda, h);
  y = fp2_sub(fp2_mul(theta, fp2_sub(g, h)), fp2_mul(e, y));
  z = fp2_mul(z, e);

  const Fp2 j = fp2_sub(fp2_mul(theta, q.x), fp2_mul(lambda, q.y));
  return Fp6{j, fp2_negate(theta), lambda};
}

__device__ void mul_by_1(Fp6& r, const Fp2& c1)
{
  const Fp2 b_b = fp2_mul(r.c1, c1);

  Fp2 t1 = fp2_sub(fp2_mul(c1, fp2_add(r.c1, r.c2)), b_b);
  mul_fp2_by_nonresidue(t1);

  const Fp2 t2 = fp2_sub(fp2_mul(c1, fp2_add(r.c0, r.c1)), b_b);

  r.c0 = t1;
  r.c1 = t2;
  r.c2 = b_b;
}

__device__ void mul_by_01(Fp6& r, const Fp2& c0, const Fp2& c1)
{
  const Fp2 a_a = fp2_mul(r.c0, c0);
  const Fp2 b_b = fp2_mul(r.c1, c1);

  Fp2 t1 = fp2_sub(fp2_mul(c1, fp2_add(r.c1, r.c2)), b_b);
  mul_fp2_by_nonresidue(t1);
  t1 = fp2_add(t1, a_a);

  const Fp2 t3 = fp2_add(fp2_sub(fp2_mul(c0, fp2_add(r.c0, r.c2)), a_a), b_b);

  const Fp2 t2 =
    fp2_sub(fp2_mul(fp2_add(c0, c1), fp2_add(r.c0, r.c1)), fp2_add(a_a, b_b));

  r.c0 = t1;
  r.c1 = t2;
  r.c2 = t3;
}

/// Sparse Fp12 multiplication by a line whose only non-zero coefficients sit at
/// positions 0, 1 and 4.
__device__ void mul_by_014(Fp12& f, const Fp2& c0, const Fp2& c1, const Fp2& c4)
{
  Fp6 aa = f.c0;
  mul_by_01(aa, c0, c1);

  Fp6 bb = f.c1;
  mul_by_1(bb, c4);

  const Fp2 o = fp2_add(c1, c4);

  f.c1 = fp6_add(f.c1, f.c0);
  mul_by_01(f.c1, c0, o);
  f.c1 = fp6_sub(fp6_sub(f.c1, aa), bb);

  f.c0 = bb;
  const Fp2 previous_c1 = f.c0.c1;
  f.c0.c1 = f.c0.c0;
  f.c0.c0 = f.c0.c2;
  mul_fp2_by_nonresidue(f.c0.c0);
  f.c0.c2 = previous_c1;
  f.c0 = fp6_add(f.c0, aa);
}

/// Evaluates a line at the G1 point and accumulates it into `f`.
__device__ __forceinline__ void ell(Fp12& f, const Fp6& coeffs, const G1Affine& p)
{
  const Fp2 c0 = coeffs.c0;
  const Fp2 c1 = fp2_scale(coeffs.c1, p.x);
  const Fp2 c2 = fp2_scale(coeffs.c2, p.y);
  mul_by_014(f, c0, c1, c2);
}

/// One Miller loop, with the line coefficients produced as they are consumed.
///
/// Precomputing them the way host implementations do would need about twenty
/// kilobytes per pair, so the running G2 point stays in local storage and each
/// line is used immediately.
__device__ Fp12 miller_loop(const G1Affine& p, const G2Affine& q, const Fq& two_inv)
{
  Fp12 f = fp12_one();
  Fp6 r = Fp6{q.x, q.y, fp2_one()};

  for (int j = kLoopBits - 1; j > 0; j--) {
    f = fp12_sqr(f);
    const Fp6 line = double_in_place(r, two_inv);
    ell(f, line, p);
    if (loop_bit(j - 1)) {
      const Fp6 addition = add_in_place(r, q);
      ell(f, addition, p);
    }
  }

  // The loop parameter is negative, so the result is conjugated.
  f.c1 = fp6_negate(f.c1);
  return f;
}

// ---------------------------------------------------------------------------
// Curve arithmetic for the in-place folds
// ---------------------------------------------------------------------------
//
// BLS12-381 has `a = 0`, so the Jacobian formulas below are the usual
// short-Weierstrass ones specialized to that case. Points live on the device
// in affine form between rounds, which is what the Miller loop consumes, so
// each fold normalizes its result before storing it.

/// Reduction challenges are BLS12-381 scalars, which fit in 255 bits.
constexpr int kScalarBits = 255;

struct G1Jacobian {
  Fq x;
  Fq y;
  Fq z;
};

__device__ __forceinline__ G1Jacobian g1_identity()
{
  return G1Jacobian{fq_one(), fq_one(), fq_zero()};
}

__device__ __forceinline__ bool g1_is_identity(const G1Jacobian& point)
{
  return fq_is_zero(point.z);
}

__device__ G1Jacobian g1_double(const G1Jacobian& point)
{
  if (g1_is_identity(point)) { return point; }

  const Fq a = fq_mul(point.x, point.x);
  const Fq b = fq_mul(point.y, point.y);
  const Fq c = fq_mul(b, b);
  Fq d = fq_sub(fq_mul(fq_add(point.x, b), fq_add(point.x, b)), fq_add(a, c));
  d = fq_add(d, d);
  const Fq e = fq_add(fq_add(a, a), a);
  const Fq f = fq_mul(e, e);

  const Fq x = fq_sub(f, fq_add(d, d));
  Fq eight_c = fq_add(c, c);
  eight_c = fq_add(eight_c, eight_c);
  eight_c = fq_add(eight_c, eight_c);
  const Fq y = fq_sub(fq_mul(e, fq_sub(d, x)), eight_c);
  const Fq yz = fq_mul(point.y, point.z);
  return G1Jacobian{x, y, fq_add(yz, yz)};
}

/// Adds an affine point to a Jacobian one.
__device__ G1Jacobian g1_add_affine(const G1Jacobian& point, const Fq& qx, const Fq& qy)
{
  if (g1_is_identity(point)) { return G1Jacobian{qx, qy, fq_one()}; }

  const Fq zz = fq_mul(point.z, point.z);
  const Fq u2 = fq_mul(qx, zz);
  const Fq s2 = fq_mul(fq_mul(qy, point.z), zz);

  if (fq_is_zero(fq_sub(u2, point.x))) {
    // Equal x means either a doubling or a pair of opposite points.
    if (fq_is_zero(fq_sub(s2, point.y))) { return g1_double(point); }
    return g1_identity();
  }

  const Fq h = fq_sub(u2, point.x);
  const Fq hh = fq_mul(h, h);
  Fq i = fq_add(hh, hh);
  i = fq_add(i, i);
  const Fq j = fq_mul(h, i);
  Fq r = fq_sub(s2, point.y);
  r = fq_add(r, r);
  const Fq v = fq_mul(point.x, i);

  const Fq x = fq_sub(fq_sub(fq_mul(r, r), j), fq_add(v, v));
  Fq y_scaled = fq_mul(point.y, j);
  y_scaled = fq_add(y_scaled, y_scaled);
  const Fq y = fq_sub(fq_mul(r, fq_sub(v, x)), y_scaled);
  const Fq z = fq_sub(fq_sub(fq_mul(fq_add(point.z, h), fq_add(point.z, h)), zz), hh);
  return G1Jacobian{x, y, z};
}

__device__ G1Affine g1_to_affine(const G1Jacobian& point)
{
  if (g1_is_identity(point)) { return G1Affine{fq_zero(), fq_zero()}; }
  const Fq z_inverse = fq_inverse(point.z);
  const Fq z_inverse_squared = fq_mul(z_inverse, z_inverse);
  return G1Affine{
    fq_mul(point.x, z_inverse_squared),
    fq_mul(point.y, fq_mul(z_inverse_squared, z_inverse)),
  };
}

struct G2Jacobian {
  Fp2 x;
  Fp2 y;
  Fp2 z;
};

__device__ __forceinline__ G2Jacobian g2_identity()
{
  return G2Jacobian{fp2_one(), fp2_one(), fp2_zero()};
}

__device__ __forceinline__ bool g2_is_identity(const G2Jacobian& point)
{
  return fp2_is_zero(point.z);
}

__device__ G2Jacobian g2_double(const G2Jacobian& point)
{
  if (g2_is_identity(point)) { return point; }

  const Fp2 a = fp2_sqr(point.x);
  const Fp2 b = fp2_sqr(point.y);
  const Fp2 c = fp2_sqr(b);
  Fp2 d = fp2_sub(fp2_sqr(fp2_add(point.x, b)), fp2_add(a, c));
  d = fp2_add(d, d);
  const Fp2 e = fp2_add(fp2_add(a, a), a);
  const Fp2 f = fp2_sqr(e);

  const Fp2 x = fp2_sub(f, fp2_add(d, d));
  Fp2 eight_c = fp2_add(c, c);
  eight_c = fp2_add(eight_c, eight_c);
  eight_c = fp2_add(eight_c, eight_c);
  const Fp2 y = fp2_sub(fp2_mul(e, fp2_sub(d, x)), eight_c);
  const Fp2 yz = fp2_mul(point.y, point.z);
  return G2Jacobian{x, y, fp2_add(yz, yz)};
}

__device__ G2Jacobian g2_add_affine(const G2Jacobian& point, const Fp2& qx, const Fp2& qy)
{
  if (g2_is_identity(point)) { return G2Jacobian{qx, qy, fp2_one()}; }

  const Fp2 zz = fp2_sqr(point.z);
  const Fp2 u2 = fp2_mul(qx, zz);
  const Fp2 s2 = fp2_mul(fp2_mul(qy, point.z), zz);

  if (fp2_is_zero(fp2_sub(u2, point.x))) {
    if (fp2_is_zero(fp2_sub(s2, point.y))) { return g2_double(point); }
    return g2_identity();
  }

  const Fp2 h = fp2_sub(u2, point.x);
  const Fp2 hh = fp2_sqr(h);
  Fp2 i = fp2_add(hh, hh);
  i = fp2_add(i, i);
  const Fp2 j = fp2_mul(h, i);
  Fp2 r = fp2_sub(s2, point.y);
  r = fp2_add(r, r);
  const Fp2 v = fp2_mul(point.x, i);

  const Fp2 x = fp2_sub(fp2_sub(fp2_sqr(r), j), fp2_add(v, v));
  Fp2 y_scaled = fp2_mul(point.y, j);
  y_scaled = fp2_add(y_scaled, y_scaled);
  const Fp2 y = fp2_sub(fp2_mul(r, fp2_sub(v, x)), y_scaled);
  const Fp2 z = fp2_sub(fp2_sub(fp2_sqr(fp2_add(point.z, h)), zz), hh);
  return G2Jacobian{x, y, z};
}

__device__ G2Affine g2_to_affine(const G2Jacobian& point)
{
  if (g2_is_identity(point)) { return G2Affine{fp2_zero(), fp2_zero()}; }
  const Fp2 z_inverse = fp2_inverse(point.z);
  const Fp2 z_inverse_squared = fp2_sqr(z_inverse);
  return G2Affine{
    fp2_mul(point.x, z_inverse_squared),
    fp2_mul(point.y, fp2_mul(z_inverse_squared, z_inverse)),
  };
}

__device__ __forceinline__ bool scalar_bit(const std::uint64_t* scalar, int index)
{
  return (scalar[index / 64] >> (index % 64)) & 1ULL;
}

/// `values[i] = values[i] + scalar * values[i + half]`, in place and affine.
///
/// The reduction consumes the lower half in the next round, so the upper half
/// is simply left behind rather than compacted.
__global__ void g1_fold_kernel(
  G1Affine* values,
  std::size_t half,
  std::uint64_t scalar0,
  std::uint64_t scalar1,
  std::uint64_t scalar2,
  std::uint64_t scalar3)
{
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= half) { return; }

  const std::uint64_t scalar[4] = {scalar0, scalar1, scalar2, scalar3};
  const G1Affine upper = values[index + half];

  G1Jacobian accumulator = g1_identity();
#pragma unroll 1
  for (int bit = kScalarBits - 1; bit >= 0; --bit) {
    accumulator = g1_double(accumulator);
    if (scalar_bit(scalar, bit)) {
      accumulator = g1_add_affine(accumulator, upper.x, upper.y);
    }
  }

  const G1Affine lower = values[index];
  accumulator = g1_add_affine(accumulator, lower.x, lower.y);
  values[index] = g1_to_affine(accumulator);
}

__global__ void g2_fold_kernel(
  G2Affine* values,
  std::size_t half,
  std::uint64_t scalar0,
  std::uint64_t scalar1,
  std::uint64_t scalar2,
  std::uint64_t scalar3)
{
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= half) { return; }

  const std::uint64_t scalar[4] = {scalar0, scalar1, scalar2, scalar3};
  const G2Affine upper = values[index + half];

  G2Jacobian accumulator = g2_identity();
#pragma unroll 1
  for (int bit = kScalarBits - 1; bit >= 0; --bit) {
    accumulator = g2_double(accumulator);
    if (scalar_bit(scalar, bit)) {
      accumulator = g2_add_affine(accumulator, upper.x, upper.y);
    }
  }

  const G2Affine lower = values[index];
  accumulator = g2_add_affine(accumulator, lower.x, lower.y);
  values[index] = g2_to_affine(accumulator);
}

__global__ void miller_loop_kernel(
  const G1Affine* left,
  const G2Affine* right,
  std::size_t left_offset,
  std::size_t right_offset,
  std::size_t count,
  Fq two_inv,
  Fp12* output)
{
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= count) { return; }
  output[index] =
    miller_loop(left[left_offset + index], right[right_offset + index], two_inv);
}

/// Multiplies `values[index]` by its partner `half` slots away.
///
/// Halving the live range each launch keeps the reduction in device memory and
/// leaves one Fp12 for the host to finish.
__global__ void fp12_reduce_kernel(Fp12* values, std::size_t half, std::size_t count)
{
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= half) { return; }
  const std::size_t partner = index + half;
  if (partner < count) { values[index] = fp12_mul(values[index], values[partner]); }
}

/// Reports a launch failure without waiting for the kernel to finish.
///
/// Kernels queued on the same stream already run in order, so the only places
/// that need to block are the copies back to the host.
int launch_status()
{
  return static_cast<int>(cudaGetLastError());
}


} // namespace

namespace {

/// Runs the Miller loops for one slice pair and reduces them to a single Fp12.
int miller_loop_slice(
  const G1Affine* left,
  const G2Affine* right,
  std::size_t left_offset,
  std::size_t right_offset,
  std::size_t count,
  const Fq& two_inv,
  Fp12* values,
  void* output)
{
  constexpr int threads = 64;
  const std::size_t blocks = (count + threads - 1) / threads;
  miller_loop_kernel<<<static_cast<int>(blocks), threads>>>(
    left, right, left_offset, right_offset, count, two_inv, values);
  int status = launch_status();
  if (status != 0) { return status; }

  std::size_t live = count;
  while (live > 1) {
    const std::size_t half = (live + 1) / 2;
    constexpr int reduce_threads = 128;
    const std::size_t reduce_blocks = (half + reduce_threads - 1) / reduce_threads;
    fp12_reduce_kernel<<<static_cast<int>(reduce_blocks), reduce_threads>>>(values, half, live);
    status = launch_status();
    if (status != 0) { return status; }
    live = half;
  }

  return static_cast<int>(cudaMemcpy(output, values, sizeof(Fp12), cudaMemcpyDeviceToHost));
}

} // namespace

extern "C" int parcomm_multi_miller_loop(
  const void* left,
  const void* right,
  std::size_t count,
  const void* two_inv,
  void* scratch,
  void* output)
{
  if (left == nullptr || right == nullptr || two_inv == nullptr || scratch == nullptr ||
      output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (count == 0) { return static_cast<int>(cudaErrorInvalidValue); }

  return miller_loop_slice(
    static_cast<const G1Affine*>(left), static_cast<const G2Affine*>(right), 0, 0, count,
    *static_cast<const Fq*>(two_inv), static_cast<Fp12*>(scratch), output);
}

/// Both cross pairings of one reduction round, over the resident vectors.
///
/// The round pairs the lower half of one vector with the upper half of the
/// other, so neither side moves across the bus.
extern "C" int parcomm_pairing_cross(
  const void* tier_one,
  const void* basis,
  std::size_t half,
  const void* two_inv,
  void* scratch,
  void* lower_output,
  void* upper_output)
{
  if (tier_one == nullptr || basis == nullptr || two_inv == nullptr || scratch == nullptr ||
      lower_output == nullptr || upper_output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (half == 0) { return static_cast<int>(cudaErrorInvalidValue); }

  const auto* left = static_cast<const G1Affine*>(tier_one);
  const auto* right = static_cast<const G2Affine*>(basis);
  const Fq& inverse_of_two = *static_cast<const Fq*>(two_inv);
  auto* values = static_cast<Fp12*>(scratch);

  int status = miller_loop_slice(left, right, 0, half, half, inverse_of_two, values, lower_output);
  if (status != 0) { return status; }
  return miller_loop_slice(left, right, half, 0, half, inverse_of_two, values, upper_output);
}

/// Folds both resident vectors in place, halving their live length.
extern "C" int parcomm_pairing_fold(
  void* tier_one,
  void* basis,
  std::size_t half,
  const std::uint64_t* tier_one_scalar,
  const std::uint64_t* basis_scalar)
{
  if (tier_one == nullptr || basis == nullptr || tier_one_scalar == nullptr ||
      basis_scalar == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (half == 0) { return static_cast<int>(cudaErrorInvalidValue); }

  constexpr int threads = 128;
  const std::size_t blocks = (half + threads - 1) / threads;
  g1_fold_kernel<<<static_cast<int>(blocks), threads>>>(
    static_cast<G1Affine*>(tier_one), half, tier_one_scalar[0], tier_one_scalar[1],
    tier_one_scalar[2], tier_one_scalar[3]);
  int status = launch_status();
  if (status != 0) { return status; }

  g2_fold_kernel<<<static_cast<int>(blocks), threads>>>(
    static_cast<G2Affine*>(basis), half, basis_scalar[0], basis_scalar[1], basis_scalar[2],
    basis_scalar[3]);
  // The next round reads these vectors from another launch, so the fold only
  // has to be ordered, not awaited.
  return launch_status();
}

/// Copies one G2 point back to the host.
extern "C" int parcomm_pairing_read_basis(const void* basis, std::size_t index, void* output)
{
  if (basis == nullptr || output == nullptr) { return static_cast<int>(cudaErrorInvalidValue); }
  return static_cast<int>(cudaMemcpy(
    output, static_cast<const G2Affine*>(basis) + index, sizeof(G2Affine),
    cudaMemcpyDeviceToHost));
}
