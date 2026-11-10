// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The base field F = Z/p and its cubic extension E = F[X] / (X^3 - NR), for host and device code
// (`BLK_HD`, blocks/base.h). NR is a cubic non-residue (not a cube in F), so E is a field.
// The prime is p = 2^64 - 2^32 + 1 (Goldilocks), with NR = 2. Struct GL64 in blocks/field/gl64.h
// has the arithmetic on words modulo p; this file calls it `BaseField`.
// An F stores the canonical word of its residue class, the one below p. The functions here assume
// this of their arguments without checking it, and preserve it. Turn outside data into an F with a
// function below that tests or reduces it; F{w} is only for a word known to be below p. In bytes,
// an F is its word, 8 little-endian bytes; an E is its coordinates of 1, X, X^2 in order, 24 bytes.
#pragma once
#include "base.h"
#include "field/gl64.h"

namespace blk {
typedef GL64 BaseField;

constexpr u64 P = BaseField::P;
constexpr u64 NR = BaseField::NR;                              // X^3 = NR in E
constexpr unsigned F_BITS = BaseField::BITS;                   // the length of p in bits
constexpr double LOG2_F = BaseField::LOG2_P;                   // log2 |F|
constexpr double LOG2_E = 3 * BaseField::LOG2_P;               // log2 |E|
constexpr double LOG2_DRAW_REFUSAL = BaseField::LOG2_REFUSAL;  // >= log2 Pr[`F_from_word` rejects]
constexpr size_t F_BYTES = 8, E_BYTES = 24;
constexpr unsigned E_DEGREE = 3;

struct F {
  u64 v; // canonical
  static BLK_HD F zero() {
    return F{0};
  }
  static BLK_HD F one() {
    return F{1};
  }
};
BLK_HD inline F operator+(F a, F b) {
  return F{BaseField::add(a.v, b.v)};
}
BLK_HD inline F operator-(F a, F b) {
  return F{BaseField::sub(a.v, b.v)};
}
BLK_HD inline F operator-(F a) {
  return F{BaseField::neg(a.v)};
}
BLK_HD inline F operator*(F a, F b) {
  return F{BaseField::mul(a.v, b.v)};
}
BLK_HD inline bool operator==(F a, F b) {
  return a.v == b.v;
}
BLK_HD inline bool operator!=(F a, F b) {
  return a.v != b.v;
}
BLK_HD inline F &operator+=(F &a, F b) {
  return a = a + b;
}
BLK_HD inline F &operator-=(F &a, F b) {
  return a = a - b;
}
BLK_HD inline F &operator*=(F &a, F b) {
  return a = a * b;
}
BLK_HD inline bool is_zero(F a) {
  return a.v == 0;
}
// from_u64(), from_i64(): the residue of v modulo p, for any v, negative or not.
BLK_HD inline F from_u64(u64 v) {
  return F{BaseField::from_u64(v)};
}
BLK_HD inline F from_i64(i64 v) {
  return v < 0 ? -from_u64(0 - (u64)v) : from_u64((u64)v);
}

// The verifier draws its field-element challenges from E: ChainT::draw_E(), blocks/chain/chain.h.
struct E {
  F c[3]; // c[0] + c[1] X + c[2] X^2
  static BLK_HD E zero() {
    return E{{F{0}, F{0}, F{0}}};
  }
  static BLK_HD E one() {
    return E{{F{1}, F{0}, F{0}}};
  }
  static BLK_HD E from(F a) {
    return E{{a, F{0}, F{0}}};
  }
  static BLK_HD E from(F a, F b, F c) {
    return E{{a, b, c}};
  }
  // The basis element X^i for i = 0, 1, 2; zero for a larger i.
  static BLK_HD E theta(unsigned i) {
    return E{{F{(u64)(i == 0)}, F{(u64)(i == 1)}, F{(u64)(i == 2)}}};
  }
};
static_assert(sizeof(F) == F_BYTES && sizeof(E) == E_BYTES, "F and E have no padding");

BLK_HD inline E operator+(const E &a, const E &b) {
  return E{{a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2]}};
}
BLK_HD inline E operator-(const E &a, const E &b) {
  return E{{a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2]}};
}
BLK_HD inline E operator-(const E &a) {
  return E{{-a.c[0], -a.c[1], -a.c[2]}};
}
// (a b)_0 = a0 b0 + NR (a1 b2 + a2 b1), (a b)_1 = a0 b1 + a1 b0 + NR a2 b2,
// (a b)_2 = a0 b2 + a1 b1 + a2 b0.
// Lazy reduction: the double-width products are summed unreduced, three to a coordinate, and
// BaseField::dot3() reduces only the sums.
// Public source: lazy reduction in extension fields: D. F. Aranha, K. Karabina, P. Longa,
// C. H. Gebotys, J. Lopez, "Faster Explicit Formulas for Computing Pairings over Ordinary Curves",
// EUROCRYPT 2011, IACR ePrint 2010/526, section 3.1 (which gives k reductions for a product in
// F_p[x] / (x^k - w) as a result of Lim and Hwang), https://eprint.iacr.org/2010/526.
// This is a public idea: lazy reduction in extension fields: P. Longa, "Efficient Algorithms for
// Large Prime Characteristic Fields and Their Application to Bilinear Pairings", IACR Transactions
// on Cryptographic Hardware and Embedded Systems, Vol. 2023, No. 3, pp. 445-472, section 1 (its
// account of lazy reduction, not the method that the paper itself proposes),
// https://tches.iacr.org/index.php/TCHES/article/download/10970/10277.
// This is a public idea: precomputed multiples in modular arithmetic: D. J. Bernstein, P. Schwabe,
// "NEON Crypto", CHES 2012, LNCS 7428, pp. 320-339, paragraph
// "Multiplication mod 2^130 - 5 on NEON",
// https://link.springer.com/content/pdf/10.1007/978-3-642-33027-8_19.pdf. There the multiples are
// of an integer's limbs. Here they are of an element's coordinates, and BaseField::times_nr()
// computes the two, NR b1 and NR b2, in each call and reduces them.
// Forced inline: left to itself, g++ 13.3 at -O2 calls this product as a function, and those calls
// were a large part of a verifier's run.
__attribute__((always_inline)) BLK_HD inline E operator*(const E &a, const E &b) {
  const u64 a0 = a.c[0].v, a1 = a.c[1].v, a2 = a.c[2].v, b0 = b.c[0].v, b1 = b.c[1].v, b2 = b.c[2].v;
  const u64 n1 = BaseField::times_nr(b1), n2 = BaseField::times_nr(b2);
  return E{{F{BaseField::dot3(a0, b0, a1, n2, a2, n1)}, F{BaseField::dot3(a0, b1, a1, b0, a2, n2)},
            F{BaseField::dot3(a0, b2, a1, b1, a2, b0)}}};
}
BLK_HD inline E operator*(const E &a, F b) {
  return E{{a.c[0] * b, a.c[1] * b, a.c[2] * b}};
}
BLK_HD inline E operator*(F b, const E &a) {
  return a * b;
}
BLK_HD inline E operator+(const E &a, F b) {
  return E{{a.c[0] + b, a.c[1], a.c[2]}};
}
BLK_HD inline E operator-(const E &a, F b) {
  return E{{a.c[0] - b, a.c[1], a.c[2]}};
}
BLK_HD inline bool operator==(const E &a, const E &b) {
  return a.c[0] == b.c[0] && a.c[1] == b.c[1] && a.c[2] == b.c[2];
}
BLK_HD inline bool operator!=(const E &a, const E &b) {
  return !(a == b);
}
BLK_HD inline E &operator+=(E &a, const E &b) {
  return a = a + b;
}
BLK_HD inline E &operator-=(E &a, const E &b) {
  return a = a - b;
}
BLK_HD inline E &operator*=(E &a, const E &b) {
  return a = a * b;
}
BLK_HD inline E &operator*=(E &a, F b) {
  return a = a * b;
}
BLK_HD inline bool is_zero(const E &a) {
  return (a.c[0].v | a.c[1].v | a.c[2].v) == 0;
}

// x^e, with x^0 = 1 also for x = 0.
BLK_HD inline F pow(F x, u64 e) {
  F r = F::one();
  for (; e; e >>= 1, x = x * x)
    if (e & 1) r = r * x;
  return r;
}
BLK_HD inline E pow(E x, u64 e) {
  E r = E::one();
  for (; e; e >>= 1, x = x * x)
    if (e & 1) r = r * x;
  return r;
}
// 1 / x. Both overloads return 0 for x = 0: a caller that can meet 0 must test for it first.
BLK_HD inline F inv(F x) {
  return pow(x, P - 2);
}
// With x = a + b X + c X^2 put A = a^2 - NR b c, B = NR c^2 - a b, C = b^2 - a c. Then
// x (A + B X + C X^2) is the norm N = a A + NR (b C + c B), which is in F and is not 0 for x != 0.
BLK_HD inline E inv(const E &x) {
  const F a = x.c[0], b = x.c[1], c = x.c[2], nr = F{NR};
  const F A = a * a - nr * (b * c), B = nr * (c * c) - a * b, C = b * b - a * c;
  const F n = inv(a * A + nr * (b * C + c * B));
  return E{{A * n, B * n, C * n}};
}

// Bytes. F_from_bytes() and E_from_bytes() return false and set `out` to zero if a word is not
// below p, so each element has exactly one valid encoding. A caller must test the result: zero is
// itself a valid element.
BLK_HD inline u64 word_from_bytes(const u8 *b) {
  u64 w = 0;
  for (int i = 7; i >= 0; i--) w = (w << 8) | b[i];
  return w;
}
BLK_HD inline void word_to_bytes(u64 w, u8 *b) {
  for (int i = 0; i < 8; i++) b[i] = (u8)(w >> (8 * i));
}
// Source for accepting only the canonical encoding: RFC 9496 (de Valence, Grigg, Hamburg, Lovecruft, Tankersley and
// Valsorda, "The ristretto255 and decaf448 Groups", 2023), section 3 and section 4.3.1, step 1; there for the
// encodings of group elements.
BLK_HD inline bool F_from_bytes(const u8 *b, F &out) {
  const u64 w = word_from_bytes(b);
  const bool ok = w < P;
  out = F{ok ? w : 0};
  return ok;
}
BLK_HD inline void F_to_bytes(F a, u8 *b) {
  word_to_bytes(a.v, b);
}
BLK_HD inline bool E_from_bytes(const u8 *b, E &out) {
  bool ok = true;
  for (int i = 0; i < 3; i++) ok = F_from_bytes(b + 8 * i, out.c[i]) && ok;
  if (!ok) out = E::zero();
  return ok;
}
BLK_HD inline void E_to_bytes(const E &a, u8 *b) {
  for (int i = 0; i < 3; i++) F_to_bytes(a.c[i], b + 8 * i);
}

// One step of rejection sampling. Takes `word` itself as the element; if it is not below p, returns
// false and sets `out` to zero. If `word` is uniform, an accepted element is exactly uniform in F,
// and the chance of rejection is just under 2^LOG2_DRAW_REFUSAL = 2^-32. After a rejection, try the
// next word. For an E, draw three F one after the other.
// Public source: Rejection sampling, as in: FIPS 204 (National Institute of Standards and
// Technology, "Module-Lattice-Based Digital Signature Standard", 2024), Algorithms 14 and 30.
BLK_HD inline bool F_from_word(u64 word, F &out) {
  return BaseField::from_word(word, out.v);
}
} // namespace blk
