// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Field arithmetic for FOLD, which reduces a linear claim on a group of 2^k main rows and 2^h mask
// rows (random values under the mask code) to a claim on three single rows (fold_rounds.h). The
// functions fold a table of rows, and store the coefficients of FOLD's hiding polynomial (hp.h:
// random, added to a sumcheck's summand) in rows that can be folded. The field type K (as in hp.h)
// is a template parameter so that tests can enumerate these coefficients over tiny fields.
//
//  * As in tensor.h, a table of 2^a rows is one vector: entry c of row b is at index b + 2^a c. The
//    row number is the low a bits of the index; its lowest bit belongs to the first variable.
//  * Folding the first variable at g halves the number of rows: new row b = (1 - g) * row 2b
//    + g * row (2b + 1). On the vector: v'[i] = (1 - g) v[2i] + g v[2i + 1], for any row length.
//    Messages, random tails (the random coefficients of the rows' encodings), opened columns and
//    coefficient vectors are folded by this rule. Encoding is linear and acts row by row, so it
//    commutes with folding.
//
// Follows the definition of the interleaved fold, and the packed table of the universal hiding polynomial, of
// the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <cstddef>
#include <vector>

#include "base.h"
#include "iop/hp.h"

namespace blk {

// Folds the first variable in place. The check sees only the vector's length, not its row count.
// A failed BLK_REQUIRE (blocks/base.h) ends the process.
template <class K>
void fold_low_once(std::vector<K> &v, const K &g) {
  BLK_REQUIRE(v.size() % 2 == 0, "fold_low_once: the table has an odd number of rows");
  for (size_t i = 0; i < v.size() / 2; i++) v[i] = v[2 * i] + g * (v[2 * i + 1] - v[2 * i]);
  v.resize(v.size() / 2);
}

// Folds the first count variables at g[0..count).
template <class K>
std::vector<K> fold_low(std::vector<K> v, const K *g, unsigned count) {
  for (unsigned j = 0; j < count; j++) fold_low_once(v, g[j]);
  return v;
}

// The three coefficients, lowest first, of q(X) = < a folded at X, m folded at X >. With rows a_b
// and m_b, q(X) = sum_b < a_{2b} + X (a_{2b+1} - a_{2b}), m_{2b} + X (m_{2b+1} - m_{2b}) >.
// One round's polynomial, before masking, in a sumcheck on the inner product of two tables.
template <class K>
std::vector<K> pair_products(const std::vector<K> &a, const std::vector<K> &m) {
  BLK_REQUIRE(a.size() == m.size() && a.size() % 2 == 0, "pair_products: the two tables differ in size, or have an odd number of rows");
  std::vector<K> q(3, K::zero());
  for (size_t i = 0; i < a.size(); i += 2) {
    const K a0 = a[i], a1 = a[i + 1] - a0, m0 = m[i], m1 = m[i + 1] - m0;
    q[0] = q[0] + a0 * m0;
    q[1] = q[1] + a0 * m1 + a1 * m0;
    q[2] = q[2] + a1 * m1;
  }
  return q;
}

template <class K>
K inner(const std::vector<K> &a, const std::vector<K> &b) {
  BLK_REQUIRE(a.size() == b.size(), "inner: the two vectors differ in length");
  K s = K::zero();
  for (size_t i = 0; i < a.size(); i++) s = s + a[i] * b[i];
  return s;
}

// One `Weight` (hp.h: how one variable is summed) per variable of FOLD's hiding polynomial
// U(X_1..X_k) = sum_l w_l(X_l) * prod_{j>l} X_j, summed over {0,1}^k.
// w_l(X) = sum_{t<=2} w_{l,t} X^t, with w_{l,0} = 0 for l < k: 2k + 1 random coefficients.
template <class K>
std::vector<Weight<K>> fold_mask_variables(unsigned k) {
  return std::vector<Weight<K>>(k, Weight<K>::plain(3));
}

// Columns (entries per row) that fold_mask_table() fills: the mask code's minimum message length.
// Not checked: h <= k.
inline size_t fold_mask_columns(unsigned k, unsigned h) { return 2 * size_t(k) + 1 - h; }

// Stores the coefficients w (order: hp_place()) in 2^h rows of length len, sent as FOLD's fresh
// mask rows. FOLD folds them, like the group's 2^h mask rows, at its first h challenges g_1..g_h;
// U(g) is the folded row's inner product with fold_mask_weights(). Entries are multilinear in the
// bits X_1..X_h of the row number b; folding sets X_l = g_l.
// ones(l, b) = X_l * .. * X_h: 1 if bits l..h of b are all 1, else 0.
//   column 0:          w_{h+1,1} (w_{k,0} if h = k) + sum_{l<=h} w_{l,1} ones(l, b)
//   column l = 1..h:   w_{l,2} ones(l, b)
//   then one column, equal in all rows, for each w_{l,t}, (l,t) = (h+1,2), (h+2,1), (h+2,2), ..,
//                      (k,1), (k,2), and last (k,0) if h < k
//   then zeros.
template <class K>
std::vector<K> fold_mask_table(unsigned k, unsigned h, size_t len, const std::vector<K> &w) {
  BLK_REQUIRE(k >= 1 && h <= k, "fold_mask_table: no row variable, or more mask rows than main rows");
  BLK_REQUIRE(len >= fold_mask_columns(k, h), "fold_mask_table: the rows are too short for the coefficients");
  const std::vector<Weight<K>> vars = fold_mask_variables<K>(k);
  BLK_REQUIRE(w.size() == hp_entries(vars), "fold_mask_table: the number of coefficients is not 2k + 1");
  auto coeff = [&](unsigned l, unsigned t) { return w[hp_place(vars, l - 1, t)]; };  // l counted from 1
  const size_t rows = size_t(1) << h;
  std::vector<K> table(rows * len, K::zero());
  for (size_t b = 0; b < rows; b++) {
    auto at = [&](size_t col) -> K & { return table[b + rows * col]; };
    at(0) = h < k ? coeff(h + 1, 1) : coeff(k, 0);
    for (unsigned l = h; l >= 1 && ((b >> (l - 1)) & 1); l--) {  // ends at the first 0 bit from bit h downwards
      at(0) = at(0) + coeff(l, 1);
      at(l) = coeff(l, 2);
    }
    size_t col = h + 1;
    for (unsigned l = h + 1; l <= k; l++)
      for (unsigned t = (l == h + 1 ? 2 : 1); t <= 2; t++) at(col++) = coeff(l, t);
    if (h < k) at(col++) = coeff(k, 0);
  }
  return table;
}

// The public weights u(g), in the table's column order: U(g) = < u(g), table folded at g_1..g_h >.
// blocks/iop/tests/fold_rounds_test.cpp checks this at random w, g for h <= k <= 5.
template <class K>
std::vector<K> fold_mask_weights(unsigned k, unsigned h, size_t len, const std::vector<K> &g) {
  BLK_REQUIRE(k >= 1 && h <= k && g.size() == k, "fold_mask_weights: no row variable, more mask rows than main rows, or not one challenge for each variable");
  BLK_REQUIRE(len >= fold_mask_columns(k, h), "fold_mask_weights: the rows are too short for the coefficients");
  std::vector<K> after(k + 1, K::one());  // after[l] = prod_{j > l} g_j with variables counted from 1
  for (unsigned l = k; l-- > 0;) after[l] = after[l + 1] * g[l];
  std::vector<K> u(len, K::zero());
  u[0] = after[h];
  for (unsigned l = 1; l <= h; l++) u[l] = g[l - 1] * after[h];
  size_t col = h + 1;
  for (unsigned l = h + 1; l <= k; l++)
    for (unsigned t = (l == h + 1 ? 2 : 1); t <= 2; t++) u[col++] = (t == 1 ? g[l - 1] : g[l - 1] * g[l - 1]) * after[l];
  if (h < k) u[col++] = K::one();
  return u;
}

}  // namespace blk
