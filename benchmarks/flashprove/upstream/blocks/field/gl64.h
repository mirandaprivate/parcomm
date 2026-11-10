// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Arithmetic modulo the "Goldilocks" prime p = 2^64 - 2^32 + 1 on bare u64 words, for host and
// device code (`BLK_HD`, blocks/base.h). blocks/field/field.h takes struct GL64
// as `BaseField` and builds on it the types F = Z/p and E, the cubic extension of F.
// Unless its comment says otherwise, a function assumes that its arguments are canonical words
// (below p), does not check this, and returns a canonical word.
//
// The reduction. With e = 2^32 - 1 (`EPS`) we have 2^64 = e, 2^96 = -1 and 2^128 = -2^32 (mod p).
// A 128-bit number with the 32-bit words w0 .. w3 is therefore congruent to
// (w0 + 2^32 w1) - w3 + e w2, and e w2 = 2^32 w2 - w2 needs no multiplication. A carry out of
// 64 bits is corrected by adding e, a borrow by subtracting e.
// These are the steps of the function reduce128 in the plonky2 library (Polygon Zero, field/src/goldilocks_field.rs,
// https://github.com/0xPolygonZero/plonky2, under the MIT or the Apache-2.0 licence); here a last step makes the result
// canonical.
// Plonky3's crate p3-goldilocks (src/goldilocks.rs; MIT or Apache-2.0) has the same steps.
// This is true of the crate's version 0.8.0 (https://github.com/Plonky3/Plonky3).
// See THIRD_PARTY.md, "The fields' arithmetic".
#pragma once
#include "base.h"

namespace blk {
struct GL64 {
  static constexpr u64 P = 0xFFFFFFFF00000001ull;
  static constexpr u64 EPS = 0xFFFFFFFFull; // 2^64 mod p
  static constexpr unsigned BITS = 64;      // the length of p in bits
  // E is F[X] / (X^3 - NR). p - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537, so 3 divides it only once, and
  // 2 has order 192 = 64 * 3 (2^96 = -1, 2^64 != 1), which therefore does not divide (p - 1) / 3.
  // So 2^((p - 1) / 3) != 1: 2 is not a cube and X^3 - 2 is irreducible.
  // (5 is a cube, so X^3 - 5 is not.) blocks/field/tests/field_test.cpp checks 2 and 5 this way.
  static constexpr u64 NR = 2;
  static constexpr double LOG2_P = 64.0 - 3.3592e-10; // log2(p) = 64 - 3.35903615e-10, rounded down
  static constexpr double LOG2_REFUSAL = -32.0;       // `from_word` rejects 2^32 - 1 of 2^64 words

  static BLK_HD u64 canonical(u64 x) { // any word to the canonical word of its class
    return x >= P ? x - P : x;
  }
  // a + b = a - (p - b); after a borrow p is added back, which is the same as subtracting e.
  static BLK_HD u64 add(u64 a, u64 b) {
    const u64 nb = P - b, d = a - nb;
    return a < nb ? d - EPS : d;
  }
  static BLK_HD u64 sub(u64 a, u64 b) {
    const u64 d = a - b;
    return a < b ? d - EPS : d;
  }
  static BLK_HD u64 neg(u64 a) {
    return a ? P - a : 0;
  }
  // The product of any two words as hi 2^64 + lo.
  static BLK_HD void mul_wide(u64 a, u64 b, u64 &hi, u64 &lo) {
#ifdef __CUDA_ARCH__
    lo = a * b;
    hi = __umul64hi(a, b);
#else
    const u128 t = (u128)a * b;
    lo = (u64)t;
    hi = (u64)(t >> 64);
#endif
  }
  // hi 2^64 + lo modulo p, for any two words.
  // reduce128() corrects a borrow or a carry under a mask, not under an "if". On the host
  // (g++ 13.3, -O2, one thread, random operands) a product in E was faster with masks than with
  // branches, in a loop whose products do not wait for each other and in one where each waits for
  // the one before, by our timing; so was a verifier's run.
  //
  // reduce128() and dot3() are not forced inline, because the timings disagree. Forcing them made
  // g++ stop inlining E's product in the two loops, which were then slower. The verifier called
  // that product as a function either way; forcing made its run shorter on a quiet machine, less
  // or not at all on a busy one. Timed without the always_inline that blocks/field/field.h puts on
  // E's product.
  static BLK_HD u64 reduce128(u64 hi, u64 lo) {
    const u64 hh = hi >> 32, hl = hi & EPS;
    u64 t = lo - hh;
    t -= EPS & (0 - (u64)(lo < hh)); // if there was a borrow: t is then at least 2^64 - e, so this does not wrap
    const u64 m = (hl << 32) - hl;   // e hl, at most e^2 = 2^64 - 2^33 + 1
    u64 r = t + m;
    r += EPS & (0 - (u64)(r < m)); // if there was a carry: r is then at most t - 2^33 + 1, so this does not wrap
    return canonical(r);
  }
  static BLK_HD u64 mul(u64 a, u64 b) {
    u64 hi, lo;
    mul_wide(a, b, hi, lo);
    return reduce128(hi, lo);
  }
  // a0 b0 + a1 b1 + a2 b2. The sum of three 128-bit products is c 2^128 + hi 2^64 + lo with c <= 2, and 2^128 = -2^32.
  // The sum is reduced once: lazy reduction, whose public sources stand at the product of two E in field/field.h.
  static BLK_HD u64 dot3(u64 a0, u64 b0, u64 a1, u64 b1, u64 a2, u64 b2) {
    u64 hi, lo, h, l, c = 0;
    mul_wide(a0, b0, hi, lo);
    mul_wide(a1, b1, h, l);
    lo += l;
    h += (lo < l); // the high word of a product is at most 2^64 - 2, so this does not wrap
    hi += h;
    c += (hi < h);
    mul_wide(a2, b2, h, l);
    lo += l;
    h += (lo < l);
    hi += h;
    c += (hi < h);
    return sub(reduce128(hi, lo), c << 32);
  }
  // NR a, computed as a + a, which is right only while NR is 2. Nothing in this file checks that;
  // blocks/field/tests/field_test.cpp compares the result with the product NR a.
  static BLK_HD u64 times_nr(u64 a) {
    return add(a, a);
  }
  // Any word modulo p.
  static BLK_HD u64 from_u64(u64 v) {
    return canonical(v);
  }
  // One try at turning a uniform word into a uniform element: the word itself, rejected (false, and
  // `out` is 0) if it is not below p. Given that it is not rejected, the element is exactly
  // uniform. A word is never reduced modulo p here: that would give the residues below 2^32 - 1
  // twice the weight of the others.
  static BLK_HD bool from_word(u64 w, u64 &out) {
    out = w < P ? w : 0;
    return w < P;
  }
};
} // namespace blk
