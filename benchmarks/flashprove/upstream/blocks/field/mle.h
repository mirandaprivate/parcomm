// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Vectors of length 2^k read as multilinear polynomials in k variables, and small polynomial
// helpers. F is the base field and E its cubic extension (blocks/field/field.h).
// Throughout, variable t goes with bit t of the index, and bit 0 is the least significant.
// eq(j, z) = prod_t (z[t] if bit t of j is 1, else 1 - z[t]): the multilinear equality polynomial.
// The value at the point z of a vector v's multilinear extension (`mle`) is sum_j eq(j, z) v[j].
// The arrays z, a, b and f need at least k entries; this is not checked. A failed BLK_REQUIRE ends
// the process; BLK_HD marks functions that also compile as CUDA device code (both: blocks/base.h).
#pragma once
#include "field/field.h"
#include <vector>

namespace blk {
// out[j] = eq(j, z) for all j < 2^k.
inline void eq_table(const E *z, unsigned k, E *out) {
  out[0] = E::one();
  for (unsigned t = 0; t < k; t++) {
    const size_t half = (size_t)1 << t;
    for (size_t j = 0; j < half; j++) {
      const E hi = out[j] * z[t];
      out[j + half] = hi;
      out[j] = out[j] - hi;
    }
  }
}
// eq(j, z) for one index; bits of j from bit k up are ignored.
BLK_HD inline E eq_index(u64 j, const E *z, unsigned k) {
  E r = E::one();
  for (unsigned t = 0; t < k; t++) r = r * (((j >> t) & 1) ? z[t] : E::one() - z[t]);
  return r;
}
// prod_t (a[t] b[t] + (1 - a[t]) (1 - b[t])): the equality polynomial at two points.
BLK_HD inline E eq_eval(const E *a, const E *b, unsigned k) {
  E r = E::one();
  for (unsigned t = 0; t < k; t++) {
    const E ab = a[t] * b[t];
    r = r * (ab + ab + E::one() - a[t] - b[t]);
  }
  return r;
}
// coeff[0] + coeff[1] x + .. + coeff[d - 1] x^(d - 1); 0 for d = 0.
BLK_HD inline E poly_eval(const E *coeff, size_t d, E x) {
  E r = E::zero();
  for (size_t i = d; i-- > 0;) r = r * x + coeff[i];
  return r;
}
// (1 - g) a + g b: one step of fixing a variable at g.
BLK_HD inline E fold_pair(F a, F b, const E &g) {
  return g * (b - a) + a;
}
BLK_HD inline E fold_pair(const E &a, const E &b, const E &g) {
  return a + g * (b - a);
}
// sum_{j < len} eq(j, z) v[j] for a vector over F or E with len <= 2^k; entries from len on count
// as zero. Computed with about len + k products.
template <class T> inline E mle_eval(const T *v, size_t len, const E *z, unsigned k) {
  BLK_REQUIRE(k < 64 && len <= ((size_t)1 << k), "mle_eval: the vector is longer than the cube");
  if (len == 0) return E::zero();
  if (k == 0) return E::zero() + v[0];
  std::vector<E> a((len + 1) / 2);
  for (size_t j = 0; j < a.size(); j++) a[j] = 2 * j + 1 < len ? fold_pair(v[2 * j], v[2 * j + 1], z[0]) : (E::one() - z[0]) * v[2 * j];
  size_t m = a.size();
  for (unsigned t = 1; t < k; t++) {
    const size_t h = (m + 1) / 2;
    for (size_t j = 0; j < h; j++) a[j] = 2 * j + 1 < m ? fold_pair(a[2 * j], a[2 * j + 1], z[t]) : (E::one() - z[t]) * a[2 * j];
    m = h;
  }
  return a[0];
}

// Product forms. A vector of length 2^k is a product form with the factors f[0 .. k) if its entry
// j is the product of the f[t] over the bits t of j that are 1. Used for the rows of a code's
// generator matrix (blocks/code/code.h).
// product_table() writes the first len entries of the product form to out.
inline void product_table(const F *f, unsigned k, size_t len, F *out) {
  BLK_REQUIRE(k < 64 && len <= ((size_t)1 << k), "product_table: more entries than the factors give");
  if (len == 0) return;
  out[0] = F::one();
  for (unsigned t = 0; t < k; t++) {
    const size_t half = (size_t)1 << t;
    for (size_t j = 0; j < half && j + half < len; j++) out[j + half] = out[j] * f[t];
  }
}
// Entry j of the product form; bits of j from bit k up are ignored.
BLK_HD inline F product_index(const F *f, unsigned k, u64 j) {
  F r = F::one();
  for (unsigned t = 0; t < k; t++)
    if ((j >> t) & 1) r = r * f[t];
  return r;
}
// The inner product of a product form with eq(., z): prod_t (1 - z[t] + z[t] f[t]).
BLK_HD inline E product_mle(const F *f, const E *z, unsigned k) {
  E r = E::one();
  for (unsigned t = 0; t < k; t++) r = r * (z[t] * (f[t] - F::one()) + F::one());
  return r;
}
// sum_{j < len} v[j] * (entry j of the product form), for a vector over F or E.
template <class T> inline T product_dot(const T *v, size_t len, const F *f, unsigned k) {
  BLK_REQUIRE(k < 64 && len <= ((size_t)1 << k), "product_dot: the vector is longer than the form");
  if (len == 0) return T::zero();
  std::vector<T> a(v, v + len);
  size_t m = len;
  for (unsigned t = 0; t < k && m > 1; t++) {
    const size_t h = (m + 1) / 2;
    for (size_t j = 0; j < h; j++) a[j] = 2 * j + 1 < m ? a[2 * j] + a[2 * j + 1] * f[t] : a[2 * j];
    m = h;
  }
  return a[0];
}
} // namespace blk
