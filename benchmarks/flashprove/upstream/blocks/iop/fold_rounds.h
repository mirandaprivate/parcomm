// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The rounds of the sumcheck FOLD as arithmetic over a field: the values in the prover's messages
// and the claims that follow from them. fold.h adds oracle messages, hashing and bytes. The field K is a template
// parameter, as in hp.h, so that tests can run the prover's code exhaustively over a tiny field.
//
// FOLD starts from one linear claim on a group of 2^k main rows m, 2^h mask rows mu (rows of
// random values; h <= k) and kept rows m_j, which are not among the 2^k and are not folded (see
// fold_kept_factor()). With coefficient vectors c, d and c_j, the claim is
//       < c, m > + < d, mu > + V = target,        V = sum_j < c_j, m_j >.
// Its summand in the k row variables, with a challenge eps and a hiding polynomial U (hp.h), is
//       H(X) = U(X) + eps * ( < c(X), m(X) > + < d(X_1..X_h), mu(X_1..X_h) > * X_{h+1} * .. * X_k
//                             + (1 - X_1) * .. * (1 - X_k) * V ),
// where c(X), m(X), .. are the rows folded at X (fold_algebra.h). A fresh mask of 2^h rows holds
// U's coefficients. FOLD ends with one new linear claim on the kept rows and on three single rows:
// m folded at the challenges g_1..g_k, and mu and the fresh mask, each folded at g_1..g_h.
// Order that soundness needs and the caller must keep (nothing here checks it): the fresh mask and
// the sum of U are sent before eps is known; each round's polynomial before that round's challenge.
//
// Follows the construction of the zero-knowledge sumcheck IOR, with the two-coefficient rounds of the
// complete zero-knowledge IOPP, of the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <utility>
#include <vector>

#include "base.h"
#include "iop/fold_algebra.h"
#include "iop/hp.h"

namespace blk {

// The hiding polynomial's 2k + 1 coefficients: each is one of the private random values `coins`.
// One value under two coefficients is a hiding fault that no verifier rejects. With a generator
// that numbers its draws, the prover's self-check ends such a run (fold_mask_as_drawn(), fold.h):
// the second coefficient is then not its own draw. This project's build with that fault is not part
// of this repository.
template <class K>
std::vector<K> fold_mask_from_coins(unsigned k, std::vector<K> coins) {
  BLK_REQUIRE(coins.size() == 2 * size_t(k) + 1, "fold_mask_from_coins: the number of coins is not 2k + 1");
  return coins;
}

// What is left of the group's (old) mask rows and of the fresh mask when FOLD ends.
// A row's message: what is encoded, without the random tail.
template <class K>
struct FoldMaskEnd {
  std::vector<K> d;      // coefficients on `mu`: eps * g_{h+1} * .. * g_k * d(g_1..g_h)
  std::vector<K> mu;     // the folded old mask row's message
  std::vector<K> fresh;  // the folded fresh mask row's message
};

// The prover's side. It does not hold the main rows, which are large: round() takes their unmasked
// polynomial, pair_products() (fold_algebra.h) of c and m as folded so far, as an argument.
// Calls: start(eps); for each of the k variables round(), then bind(g); end().
template <class K>
class FoldProverRounds {
 public:
  // w: the hiding polynomial's coefficients; d, mu: the mask rows' coefficients and messages, each
  // 2^h rows of length len_zk in one vector (layout: fold_algebra.h); kept: V. The fresh mask needs
  // len_zk >= fold_mask_columns(k, h).
  FoldProverRounds(unsigned k, unsigned h, size_t len_zk, const std::vector<K> &w, std::vector<K> d, std::vector<K> mu,
                   const K &kept = K::zero())
      : k_(k), h_(h), hiding_(fold_mask_variables<K>(k), w), fresh_(fold_mask_table(k, h, len_zk, w)),
        d_(std::move(d)), mu_(std::move(mu)), kept_(kept) {
    BLK_REQUIRE(d_.size() == (size_t(1) << h) * len_zk && mu_.size() == d_.size(), "FOLD: the mask rows' coefficients or messages are not tables of 2^h rows");
  }
  K mask_sum() const { return hiding_.sum(); }
  void start(const K &eps) {
    BLK_REQUIRE(!started_, "FOLD: eps was given twice");
    eps_ = eps, started_ = true;
  }
  // All three coefficients of this round's polynomial, lowest first.
  std::vector<K> round(const std::vector<K> &main_part) const {
    BLK_REQUIRE(started_ && j_ < k_, "FOLD: a round before eps or after the last variable");
    BLK_REQUIRE(main_part.size() == 3, "FOLD: the main rows' polynomial has not three coefficients");
    std::vector<K> q = hiding_.round();
    std::vector<K> mask(3, K::zero());
    if (j_ < h_)
      mask = pair_products(d_, mu_);
    else {
      if (!folded_inner_known_) folded_inner_ = inner(d_, mu_), folded_inner_known_ = true;  // the two rows no longer change after round h: computed once
      mask[1] = folded_inner_ * later_;  // times X; the product of the later X_i sums to 1
    }
    for (unsigned t = 0; t < 3; t++) q[t] = q[t] + eps_ * (main_part[t] + mask[t]);
    q[0] = q[0] + eps_ * kept_;  // kept_ * (1 - X); the product of the later (1 - X_i) sums to 1
    q[1] = q[1] - eps_ * kept_;
    return q;
  }
  void bind(const K &g) {
    BLK_REQUIRE(started_ && j_ < k_, "FOLD: a challenge before eps or after the last variable");
    hiding_.bind(g);
    if (j_ < h_) {
      fold_low_once(d_, g);
      fold_low_once(mu_, g);
      fold_low_once(fresh_, g);
    } else {
      later_ = later_ * g;
    }
    kept_ = kept_ * (K::one() - g);
    j_++;
  }
  // with_d = false leaves d empty, for a caller that computes it with fold_mask_form().
  FoldMaskEnd<K> end(bool with_d = true) const {
    BLK_REQUIRE(j_ == k_, "FOLD: rounds are missing");
    FoldMaskEnd<K> e{with_d ? d_ : std::vector<K>(), mu_, fresh_};
    const K s = eps_ * later_;
    for (K &x : e.d) x = x * s;
    return e;
  }

 private:
  unsigned k_, h_, j_ = 0;
  HidingPolynomial<K> hiding_;
  std::vector<K> fresh_, d_, mu_;
  K kept_;                                // V times (1 - g) for the challenges so far
  K eps_ = K::zero(), later_ = K::one();  // later_ = g_{h+1} * .. * g_j after j challenges
  bool started_ = false;
  mutable K folded_inner_ = K::zero();  // < d, mu > of the two rows folded at g_1..g_h
  mutable bool folded_inner_known_ = false;
};

// The verifier's side: the running claim. It starts at mask_sum + eps * target, the claimed sum of
// H over {0,1}^k: mask_sum is the sent sum of U, and the products of variables in H make < d, mu >
// and V count once. A round's message is the constant and the quadratic coefficient; the linear one
// is the value for which q(0) + q(1) equals the running claim. That check therefore holds by
// construction, and a false polynomial shows only as a false claim at the end. After k calls of
// round(), which are not counted, claim() is the new claim's target.
template <class K>
class FoldVerifierRounds {
 public:
  FoldVerifierRounds(const K &mask_sum, const K &eps, const K &target) : claim_(mask_sum + eps * target) {}
  // For a prover that does not know `target`: starts at q(0) + q(1) of its first round's q.
  // Not for a verifier: the running claim would not depend on `target`.
  static FoldVerifierRounds from_first_round(const std::vector<K> &q) {
    BLK_REQUIRE(q.size() == 3, "FOLD: a round's polynomial has not three coefficients");
    return FoldVerifierRounds(q[0] + q[0] + q[1] + q[2], K::zero(), K::zero());
  }
  void round(const K &c0, const K &c2, const K &g) {
    std::vector<K> q{c0, K::zero(), c2};
    Weight<K>::plain(3).complete(q, claim_);
    claim_ = poly_at(q, g);
  }
  const K &claim() const { return claim_; }

 private:
  K claim_;
};

// FoldMaskEnd::d, computed from the claim's d (2^h rows in one vector) and the k challenges g.
template <class K>
std::vector<K> fold_mask_form(unsigned k, unsigned h, std::vector<K> d, const K &eps, const std::vector<K> &g) {
  BLK_REQUIRE(g.size() == k && h <= k, "fold_mask_form: more mask rows than main rows, or not one challenge for each variable");
  d = fold_low(std::move(d), g.data(), h);
  K s = eps;
  for (unsigned t = h; t < k; t++) s = s * g[t];
  for (K &x : d) x = x * s;
  return d;
}

// The factor s = eps * (1 - g_1) .. (1 - g_k) of the kept rows' coefficients in the new claim.
// Kept rows are main rows that a later part of the protocol reads as they are. They are treated as
// if every main row were a bundle of several rows side by side, with a copy of kept row j as part j
// of every bundle and c_j on the copy in bundle 0 only. Folding equal copies gives the row back,
// and folding (c_j, 0, .., 0) gives (1 - X_1) .. (1 - X_k) c_j. So nothing is copied or folded: H
// gets its term in V, and the new claim has the coefficients s * c_j on kept row j itself.
// blocks/iop/tests/fold_rounds_test.cpp compares this with the bundles written out.
template <class K>
K fold_kept_factor(const K &eps, const std::vector<K> &g) {
  K s = eps;
  for (const K &x : g) s = s * (K::one() - x);
  return s;
}

}  // namespace blk
