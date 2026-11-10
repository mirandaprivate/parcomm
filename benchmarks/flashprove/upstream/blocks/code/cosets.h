// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Evaluation points and basis factors of the Reed-Solomon code modulo p = 2^64 - 2^32 + 1, used by
// blocks/code/code.h. No function here checks log_n or a position (an index into a codeword);
// code_ok() in that file checks 1 <= log_n <= GL64_MOST_LOG_N.
//
// A code of length 2^log_n uses 2^(log_n - s) cosets of the subgroup H of order 2^s, where
// s = min(log_n, GL64_SUBGROUP_LOG). Outside tests GL64_SUBGROUP_LOG is 32: one coset up to
// log_n = 32 and several beyond, because no subgroup of two-power order has more than 2^32 points.
// With 7, which generates the multiplicative group, and w = GL64_ROOT_32, of order 2^32, position I
// has the point
//     z_I = 7^(1 + c) * prod_{i < s, bit i of I is 1} w^(2^(31 - i)),   c = I >> s.
// Here c picks the coset 7^(1 + c) H. The low s bits of I are the exponent of a generator of H in
// reversed bit order, so positions 2 m and 2 m + 1 have the points z and -z. The cosets are
// distinct because 7^(c - c') is in H only if 2^32 - 1 divides c - c'. No point is 0.
// The factors at z are f_i = z^(2^i); basis function j, the product of the f_i over the set bits i
// of j, is z^j. For i < s, f_i depends only on bits i and up of I, and bit i flips its sign.
#pragma once
#include "field/gl64.h"
#include <vector>

// For tests only: a value below 32 puts short codes on several cosets, which
// blocks/code/tests/cosets_test.cpp tests with the value 4. It changes a code's points, but nothing
// written or hashed records it. So it stops blocks/iop/context.h (ProverContext, VerifierContext)
// and a production build (SERVING_BUILD) from compiling, and puts the string that begins
// "BLK-NOT-FOR-SERVING" (blocks/fence.h) into other programs.
#ifndef BLK_GL64_SUBGROUP_LOG
#define BLK_GL64_SUBGROUP_LOG 32
#endif
#if BLK_GL64_SUBGROUP_LOG != 32
#define BLK_CODE_FOR_TESTS_ONLY 1
BLK_MARK_NOT_FOR_SERVING();  // blocks/fence.h misses a #define that comes after its first include
#endif

namespace blk {
// 7 generates the multiplicative group, of order 2^32 (2^32 - 1), and GL64_ROOT_32 its subgroup of
// order 2^32. This root is chosen because its powers of orders 64, 32, 16, 8, 4 are 2^3, 2^6, 2^12,
// 2^24, 2^48, so multiplying by them is a shift, as in prover/field/transform_groups_gl64.cuh.
// blocks/code/tests/cosets_test.cpp checks that 7 is a generator, the root's value and the powers.
constexpr u64 GL64_GENERATOR = 7;
constexpr u64 GL64_ROOT_32 = 0xE2AF70635744FD26ull; // 7^(5 (2^32 - 1)), of order 2^32
constexpr unsigned GL64_SUBGROUP_LOG = BLK_GL64_SUBGROUP_LOG;
constexpr unsigned GL64_MOST_LOG_N = 60; // up to 2^28 cosets of 2^32 points; there are 2^32 - 1 cosets
static_assert(GL64_SUBGROUP_LOG >= 1 && GL64_SUBGROUP_LOG <= 32, "the two-power subgroups have at most 2^32 points");

BLK_HD inline unsigned gl64_subgroup_log(unsigned log_n) {
  return log_n < GL64_SUBGROUP_LOG ? log_n : GL64_SUBGROUP_LOG;
}
BLK_HD inline u64 gl64_pow(u64 a, u64 e) {
  u64 r = 1;
  for (; e; e >>= 1, a = GL64::mul(a, a))
    if (e & 1) r = GL64::mul(r, a);
  return r;
}
// The point z_I of position I.
BLK_HD inline u64 gl64_point(unsigned log_n, u64 I) {
  const unsigned s = gl64_subgroup_log(log_n);
  u64 z = gl64_pow(GL64_GENERATOR, 1 + (I >> s)), root = GL64_ROOT_32; // root has the order 2^(i + 1) in step i
  for (unsigned i = 32; i-- > 0; root = GL64::mul(root, root))
    if (i < s && ((I >> i) & 1)) z = GL64::mul(z, root);
  return z;
}
// f[i] = f_i at position I, for i < count.
BLK_HD inline void gl64_factors(unsigned log_n, u64 I, unsigned count, u64 *f) {
  if (count > 0) f[0] = gl64_point(log_n, I);
  for (unsigned i = 1; i < count; i++) f[i] = GL64::mul(f[i - 1], f[i - 1]);
}
// The table of the fast transform that encodes a row (`Encoder`, blocks/code/code.h). With
// at = 2^(log_n - 1 - L), tw[at + h] = f_L at the positions I with I >> L = 2 h, for L < layers and
// h < at. Other entries are 0. Requires layers <= s (not checked), because bit L flips the sign of
// f_L only if L < s.
inline void gl64_table(unsigned log_n, unsigned layers, std::vector<u64> &tw) {
  const size_t n = (size_t)1 << log_n;
  const unsigned s = gl64_subgroup_log(log_n);
  tw.assign(n, 0);
  if (layers == 0) return;
  // The points at the even positions 2 m: bit b of m is bit b + 1 of the position.
  u64 *z = tw.data() + n / 2;
  z[0] = GL64_GENERATOR;
  for (unsigned b = 0; b + 1 < log_n; b++) {
    const unsigned i = b + 1;
    const u64 step = i < s ? gl64_pow(GL64_ROOT_32, 1ull << (31 - i)) : gl64_pow(GL64_GENERATOR, 1ull << (i - s));
    for (size_t m = 0; m < ((size_t)1 << b); m++) z[m + ((size_t)1 << b)] = GL64::mul(z[m], step);
  }
  for (unsigned L = 1; L < layers; L++) {
    const size_t at = n >> (L + 1);
    for (size_t h = 0; h < at; h++) tw[at + h] = GL64::mul(tw[2 * at + 2 * h], tw[2 * at + 2 * h]);
  }
}
} // namespace blk
