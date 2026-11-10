// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// HP: the hiding polynomial of a sumcheck, and `Weight`: how one variable is summed.
//
// A sumcheck proves a claim (phi_1 x .. x phi_k)(G) = value, where G is a polynomial in k
// variables and each phi_j is a linear functional on polynomials in one variable: a way to "sum
// over" variable j. In round i the prover sends the polynomial in X_i that is left when the earlier
// variables are set to the verifier's challenges and the later variables' functionals have been
// applied. These round polynomials depend on the witness. To hide it, the callers
// (blocks/iop/psc.h, blocks/iop/fold_rounds.h) run the sumcheck on U + eps * G: U is the random
// polynomial defined below, and eps is a challenge chosen after the prover has sent the sum of U.
//
// This file is arithmetic only: the caller supplies the coefficients of U and sends the messages.
// A failed BLK_REQUIRE (blocks/base.h) ends the process. Everything here is a template over the
// field type K, which needs a default constructor, + - * ==, K::zero(), K::one() and inv(K).
// Variables are numbered from 0 in the code: X_1 is variable 0.
//
// Follows the definition of the universal hiding polynomial, and the lemma that its sumcheck
// messages are uniform, of the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <cstddef>
#include <utility>
#include <vector>

#include "base.h"

namespace blk {

// ---- Polynomials in one variable: vectors of coefficients, lowest degree first ----

template <class K>
K poly_at(const std::vector<K>& g, const K& x) {
  K r = K::zero();
  for (size_t t = g.size(); t-- > 0;) r = r * x + g[t];
  return r;
}

// The integer n as an element of K. Takes n additions: meant for small n.
template <class K>
K small_int(unsigned n) {
  K r = K::zero();
  for (unsigned i = 0; i < n; ++i) r = r + K::one();
  return r;
}

// The polynomial of degree < values.size() that takes values[i] at the node i = 0, 1, 2, ..
// (Newton's divided differences). Not checked: the nodes must be distinct in K (characteristic >=
// values.size()).
template <class K>
std::vector<K> poly_from_values_at_0_1_2(std::vector<K> values) {
  const size_t n = values.size();
  for (size_t s = 1; s < n; ++s) {  // values[i] becomes the divided difference on the nodes i-s .. i
    const K step = inv(small_int<K>(unsigned(s)));
    for (size_t i = n; i-- > s;) values[i] = (values[i] - values[i - 1]) * step;
  }
  std::vector<K> g(n, K::zero());  // Horner on the Newton form: g = values[i] + (X - i) g
  for (size_t i = n; i-- > 0;) {
    const K node = small_int<K>(unsigned(i));
    for (size_t t = n; t-- > 1;) g[t] = g[t - 1] - node * g[t];
    g[0] = values[i] - node * g[0];
  }
  return g;
}

// ---- One variable of a sumcheck ----

// How one variable is summed: a linear functional phi on polynomials with d coefficients, and a
// polynomial psi with phi(psi) = 1. Build one with the static functions below, which check this.
// The verifier's check on a round polynomial q is phi(q) = running claim: the claimed sum, then
// the previous q at its challenge. The check determines one coefficient of q from the others, so
// that coefficient is not sent; `fixed` is its index. It is the linear coefficient for plain()
// (phi(X) = 1: no division) and the constant one otherwise (divided by phi(1), which is 1 except
// for domain(), where it is the number of points).
template <class K>
struct Weight {
  unsigned d = 0;         // polynomials in this variable have d >= 2 coefficients (degree < d)
  std::vector<K> moment;  // moment[t] = phi(X^t), t < d
  std::vector<K> psi;     // coefficients of psi, at most d
  unsigned fixed = 0;     // the coefficient that the check fixes
  K fixed_inv;            // 1 / moment[fixed]

  K apply(const std::vector<K>& g) const {  // phi(g)
    BLK_REQUIRE(g.size() <= d, "Weight::apply: the polynomial has too many coefficients");
    K r = K::zero();
    for (size_t t = 0; t < g.size(); ++t) r = r + moment[t] * g[t];
    return r;
  }
  K of_one() const { return moment[0]; }  // phi(1)

  // Sets g[fixed] to the one value for which phi(g) = claim. The old g[fixed] is ignored.
  void complete(std::vector<K>& g, const K& claim) const {
    BLK_REQUIRE(g.size() == d, "Weight::complete: the polynomial must have d coefficients");
    K rest = K::zero();
    for (unsigned t = 0; t < d; ++t)
      if (t != fixed) rest = rest + moment[t] * g[t];
    g[fixed] = (claim - rest) * fixed_inv;
  }

  // phi(g) = g(0) + g(1): a plain sum over {0,1}. psi = X.
  static Weight plain(unsigned d) {
    Weight w = shell(d, 1);
    w.moment[0] = K::one() + K::one();
    for (unsigned t = 1; t < d; ++t) w.moment[t] = K::one();
    w.psi = {K::zero(), K::one()};
    return w.checked();
  }
  // phi(g) = (1 - z) g(0) + z g(1): a sum over x in {0,1} weighted by the equality polynomial's
  // factor (1 - z)(1 - x) + z x, which the summand then does not include. psi = 1.
  static Weight equality(const K& z, unsigned d) {
    Weight w = shell(d, 0);
    w.moment[0] = K::one();
    for (unsigned t = 1; t < d; ++t) w.moment[t] = z;
    w.psi = {K::one()};
    return w.checked();
  }
  // phi(g) = g(0). Used for a further variable that does not change the sum (Y in psc.h). psi = 1.
  static Weight at_zero(unsigned d) { return equality(K::zero(), d); }
  // phi(g) = sum of g over the points of a set D. psi = 1/|D|, so |D| must not vanish in K.
  static Weight domain(const std::vector<K>& points, unsigned d) {
    return combination(points, std::vector<K>(points.size(), K::one()), d);
  }
  // phi(g) = sum_x Lag_x(z) g(x) over the points x of a set D, with Lag_x the Lagrange
  // polynomials of D: the value at z of the polynomial of degree < |D| that agrees with g on D.
  // psi = 1. Not checked: the points must be distinct.
  static Weight lagrange(const std::vector<K>& points, const K& z, unsigned d) {
    std::vector<K> c(points.size(), K::one());
    for (size_t i = 0; i < points.size(); ++i)
      for (size_t j = 0; j < points.size(); ++j)
        if (j != i) c[i] = c[i] * (z - points[j]) * inv(points[i] - points[j]);
    return combination(points, c, d);
  }

 private:
  static Weight shell(unsigned d, unsigned fixed) {
    BLK_REQUIRE(d >= 2, "Weight: at least two coefficients");
    Weight w;
    w.d = d;
    w.moment.assign(d, K::zero());
    w.fixed = fixed;
    return w;
  }
  // phi(g) = sum_i c[i] g(points[i]), psi = the constant 1 / sum_i c[i].
  static Weight combination(const std::vector<K>& points, const std::vector<K>& c, unsigned d) {
    Weight w = shell(d, 0);
    for (size_t i = 0; i < points.size(); ++i) {
      K power = c[i];
      for (unsigned t = 0; t < d; ++t, power = power * points[i]) w.moment[t] = w.moment[t] + power;
    }
    BLK_REQUIRE(!(w.moment[0] == K::zero()), "Weight: the weights of the points sum to zero");
    w.psi = {inv(w.moment[0])};
    return w.checked();
  }
  Weight checked() {
    BLK_REQUIRE(!(moment[fixed] == K::zero()), "Weight: the check does not fix this coefficient");
    fixed_inv = inv(moment[fixed]);
    BLK_REQUIRE(psi.size() <= d && apply(psi) == K::one(), "Weight: phi(psi) must be 1");
    return *this;
  }
};

// ---- The random coefficients ----

// The hiding polynomial is
//
//     U(X_1..X_k) = sum_l  w_l(X_l) * prod_{j>l} psi_j(X_j),       phi_j(psi_j) = 1 for every j,
//
// where w_l is a polynomial of degree < d_l (`Weight::d` for X_l) with random coefficients.
// Two facts about U:
//   1. U is cheap: its sum and all its round polynomials together cost O(sum of the degrees) field
//      operations.
//   2. For every choice of challenges, the map from the random coefficients to (sum, round
//      polynomials) is onto the set of all tuples that pass the verifier's checks. So if the
//      coefficients are uniform, fresh and enter nothing else that is shown, the messages of the
//      sumcheck of U are uniform among the tuples that pass, and the messages for
//      U + (anything of degree < d_j in each X_j) are uniform among the tuples that pass for the
//      shifted claim: they reveal nothing but the claim.
//      What is and is not shown about hiding for a whole run: docs/limits.md#weight-hiding.
// A constant term of w_l, l < k, can be moved into w_{l+1} (as that constant times psi_{l+1}), so
// these k - 1 constants are fixed to zero; the last variable keeps its constant. The map of fact 2
// is then one to one as well. blocks/iop/tests/hp_test.cpp checks this for some small shapes, by
// enumeration over the fields of 3, 5 and 7 elements.

// Number of random coefficients of U (`entries` in identifiers): 1 + sum_j (d_j - 1).
template <class K>
size_t hp_entries(const std::vector<Weight<K>>& vars) {
  size_t n = 1;
  for (const Weight<K>& v : vars) {
    BLK_REQUIRE(v.d >= 2, "hp_entries: a variable that no constructor of Weight has made");
    n += v.d - 1;
  }
  return n;
}

// The index of the coefficient of X_l^t in the vector of random coefficients. They are ordered by
// variable, then by rising degree: 0 .. d - 1 for the last variable, 1 .. d - 1 for every other.
template <class K>
size_t hp_place(const std::vector<Weight<K>>& vars, unsigned l, unsigned t) {
  const bool last = l + 1 == vars.size();
  BLK_REQUIRE(l < vars.size() && t < vars[l].d && (last || t >= 1), "hp_place: no such coefficient");
  size_t at = 0;
  for (unsigned j = 0; j < l; ++j) at += vars[j].d - 1;
  return at + (last ? t : t - 1);
}

// U at the challenges gamma, as a linear form in the random coefficients:
// out[hp_place(l, t)] = gamma_l^t prod_{j>l} psi_j(gamma_j). The form depends only on `vars` and
// gamma, so a verifier computes it without the coefficients (psc_rows() in psc.h).
template <class K>
std::vector<K> hp_form(const std::vector<Weight<K>>& vars, const std::vector<K>& gamma) {
  BLK_REQUIRE(!vars.empty() && gamma.size() == vars.size(), "hp_form: one challenge for each variable");
  std::vector<K> out(hp_entries(vars));
  K tail = K::one();  // prod_{j>l} psi_j(gamma_j)
  for (unsigned l = unsigned(vars.size()); l-- > 0;) {
    const bool last = l + 1 == vars.size();
    K power = last ? tail : tail * gamma[l];
    for (unsigned t = last ? 0 : 1; t < vars[l].d; ++t, power = power * gamma[l]) out[hp_place(vars, l, t)] = power;
    tail = tail * poly_at(vars[l].psi, gamma[l]);
  }
  return out;
}

// ---- The prover's side ----

// The sumcheck of U: call sum(), then round() and bind(challenge) for each variable, then value().
template <class K>
class HidingPolynomial {
 public:
  // entries: hp_entries(vars) coefficients in the order of hp_place().
  HidingPolynomial(const std::vector<Weight<K>>& vars, const std::vector<K>& entries) : vars_(vars) {
    const unsigned k = unsigned(vars.size());
    BLK_REQUIRE(k >= 1 && entries.size() == hp_entries(vars), "HidingPolynomial: wrong number of coefficients");
    w_.resize(k);
    for (unsigned l = 0; l < k; ++l) {
      w_[l].assign(vars[l].d, K::zero());
      for (unsigned t = (l + 1 == k ? 0 : 1); t < vars[l].d; ++t) w_[l][t] = entries[hp_place(vars, l, t)];
    }
    // after_[i]: what the terms of the variables after i contribute to round i, a constant:
    // sum_{l>i} phi_l(w_l) prod_{i<j<l} phi_j(1). One more step of the same recursion gives the
    // sum of U.
    after_.assign(k, K::zero());
    for (unsigned i = k - 1; i-- > 0;) after_[i] = vars[i + 1].apply(w_[i + 1]) + vars[i + 1].of_one() * after_[i + 1];
    sum_ = vars[0].apply(w_[0]) + vars[0].of_one() * after_[0];
  }

  K sum() const { return sum_; }  // (phi_1 x .. x phi_k)(U)

  // Round polynomial of the next variable i, d_i coefficients: w_i + before_ * psi_i + after_[i].
  std::vector<K> round() const {
    BLK_REQUIRE(i_ < vars_.size(), "HidingPolynomial::round: all variables are bound");
    std::vector<K> q = w_[i_];
    for (size_t t = 0; t < vars_[i_].psi.size(); ++t) q[t] = q[t] + before_ * vars_[i_].psi[t];
    q[0] = q[0] + after_[i_];
    return q;
  }
  void bind(const K& gamma) {
    BLK_REQUIRE(i_ < vars_.size(), "HidingPolynomial::bind: all variables are bound");
    before_ = before_ * poly_at(vars_[i_].psi, gamma) + poly_at(w_[i_], gamma);
    ++i_;
  }
  K value() const {  // U at the challenges, once all variables are bound
    BLK_REQUIRE(i_ == vars_.size(), "HidingPolynomial::value: bind all variables first");
    return before_;
  }

 private:
  std::vector<Weight<K>> vars_;
  std::vector<std::vector<K>> w_;
  std::vector<K> after_;
  K sum_;
  K before_ = K::zero();  // sum_{l<i} w_l(gamma_l) prod_{l<j<i} psi_j(gamma_j)
  unsigned i_ = 0;
};

}  // namespace blk
