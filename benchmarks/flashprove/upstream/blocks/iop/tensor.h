// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Coefficient vectors v of linear claims, in structured form. A linear claim (`Row` in
// blocks/iop/relation.h) says that <v, committed messages> has a public value. A `Tensor` is a
// product over the bits of the index, optionally times a slice of a written-out vector; a `Form` is
// a sum of Tensors and single entries. Fixing the low variables at a point, which is what a fold
// does to a coefficient vector, changes scalars, shortens slices and drops those variables, so a
// verifier need not write a long vector out.
//
// A prover's Tensor or Form may refer to a vector that only its engine holds. An engine (`Engine`,
// blocks/engine/engine.h) does the parts of a prover's work that grow with the size of its tables,
// on a CPU or a GPU. A function here that would need an entry of such a vector ends the process, as
// does a failed BLK_REQUIRE (blocks/base.h).
//
// Index convention of blocks/: an index j < 2^a has bits j_0 (least significant) .. j_{a-1}, and
// variable t belongs to bit t. A table of 2^k rows of length l is the vector with entry c of row b
// at index b + 2^k c, so the variables 0 .. k-1 give the row's number.
//
// Coefficient vectors in structured form, and the fixing of their low variables, follow the succinct linear forms of
// the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "base.h"
#include "field/field.h"

namespace blk {

using TableId = u32;  // a table that a prover's engine holds (blocks/engine/engine.h)

// Whether an element of the extension field E is in the base field F. Many factors of a coefficient
// vector are (a code's generator rows, powers of a position). E times such an element takes three
// multiplications in F instead of nine and gives the same element.
inline bool lies_in_F(const E &a) { return (a.c[1].v | a.c[2].v) == 0; }
inline E times(const E &a, const E &b) { return lies_in_F(b) ? a * b.c[0] : a * b; }  // a b
inline void add_product(E &sum, const E &a, F b) {                                     // sum += a b
  for (unsigned i = 0; i < E_DEGREE; i++) sum.c[i] = sum.c[i] + a.c[i] * b;
}

// A vector of 2^a entries: v[j] = scalar * prod_t (bit t of j ? hi[t] : lo[t]), t < a. It covers
//   the equality vector eq(., z):            lo[t] = 1 - z[t],  hi[t] = z[t];
//   a code's generator row at a position:    lo[t] = 1,         hi[t] = f_t (defined below);
//   the choice of an aligned block of rows:  (lo, hi) = (1, 0) or (0, 1) on the bits above it.
// Here f_t is the code's factor t at that position (factors() in blocks/code/code.h): the code's
// basis function j at a position is the product of the f_t over the set bits t of j.
// An optional further factor, the dense factor, is a slice of a written-out vector, indexed by some
// of the variables (the dense variables). So coefficients v[(i, c)] = A[i] * B[c] on a matrix, with
// A a product and B written out, are one Tensor on each aligned block in which laid.h places the
// matrix. Many Tensors may share one B.
//
// Storage. A variable with the pair (1, 0) or (0, 1) is a choice variable (`picks()`): it fixes one
// bit of the index, and all of them are stored as two bit masks. The pairs of the other variables,
// the real variables, are in a list that Tensors share and that never changes once it is shared; a
// Tensor points to the list and keeps the index of its first remaining pair. So copying a Tensor
// without a dense factor allocates nothing, insert_choice() moves bits and fix_low() steps over
// pairs. At most 63 variables. A moved-from Tensor has lost its list and must not be used.
// own_pairs() relies on use_count(), which is exact only while no other thread copies or destroys a
// Tensor that shares the list: until the last push() on any of the Tensors that share a list, only
// one thread may use them.
struct Tensor {
  E scalar = E::one();
  // The dense factor, if dense_vars is not empty: (*values)[first + d], where bit i of d is bit
  // dense_vars[i] of the index. A caller that fills dense_vars gives these variables the pair
  // (1, 1) and calls set_values() last, which checks this.
  std::vector<unsigned> dense_vars;
  std::shared_ptr<const std::vector<E>> values;
  u64 first = 0;
  // Instead of `values`, the dense factor may be in a table of the prover's engine: entry first + d
  // of table `held`, read as one vector of held_length entries over E. This saves copying a long
  // vector that the engine computed to the host and back. set_values() and set_held() end the
  // process rather than leave `values` and `held` both set.
  // - Only a prover's Tensor has such a factor: Statement::add_row() (relation.h) ends the process
  //   if a claim added to a verifier's statement has one. The claims on out-of-domain samples,
  //   which send.h builds itself, do not go through add_row() and contain no Tensor of a caller.
  // - Only an engine can read it. set_held() sees only the stated length, so the engine must
  //   compare held_length and the slice with the table's actual length: EngineCpu::readable() does
  //   so through with_table(), EngineGpu::factor_of() directly.
  // - Every function here that needs an entry of it ends the process: fix_low() over a dense
  //   variable, folded_slice(), add_to(), mle(), and at() except where a choice bit differs or the
  //   other factors give 0.
  // - It may have no dense variable: it is then the single entry `first` (in laid.h, a block that
  //   is one element wide).
  static constexpr TableId NOT_HELD = ~TableId(0);
  TableId held = NOT_HELD;
  u64 held_length = 0;
  bool is_held() const { return held != NOT_HELD; }

  unsigned vars() const { return vars_; }
  bool has_dense() const { return !dense_vars.empty() || is_held(); }
  // The pair of variable t.
  E lo(unsigned t) const { return picks(t) ? (((pick_value_ >> t) & 1) ? E::zero() : E::one()) : (*pairs_)[2 * real_at(t)]; }
  E hi(unsigned t) const { return picks(t) ? (((pick_value_ >> t) & 1) ? E::one() : E::zero()) : (*pairs_)[2 * real_at(t) + 1]; }
  std::vector<E> los() const {  // all of them, written out
    std::vector<E> v;
    for (unsigned t = 0; t < vars_; t++) v.push_back(lo(t));
    return v;
  }
  std::vector<E> his() const {
    std::vector<E> v;
    for (unsigned t = 0; t < vars_; t++) v.push_back(hi(t));
    return v;
  }
  bool picks(unsigned t) const { return (pick_mask_ >> t) & 1; }
  u64 pick_mask() const { return pick_mask_; }
  u64 pick_value() const { return pick_value_; }
  // For an engine that goes through thousands of Tensors: a pointer to the pairs (lo, hi) of the
  // real variables, lowest variable first: 2 (vars() - the number of choice variables) elements.
  // With no real variable it means nothing (nullptr, or past the list's end). Valid until this
  // Tensor is changed or destroyed. Copies of a Tensor share the list, so while both live, an equal
  // pointer with an equal number of real variables means equal pairs; the converse does not hold.
  const E *real_pairs() const { return pairs_ ? pairs_->data() + 2 * from_ : nullptr; }

  static Tensor eq(const E *z, unsigned a, E scalar = E::one()) {
    Tensor v;
    v.scalar = scalar;
    for (unsigned t = 0; t < a; t++) v.push(E::one() - z[t], z[t]);
    return v;
  }
  // lo = 1, hi = f[t]: the vector (scalar * prod over the set bits t of j of f[t])_j.
  static Tensor product(const F *f, unsigned a, E scalar = E::one()) {
    Tensor v;
    v.scalar = scalar;
    for (unsigned t = 0; t < a; t++) v.push(E::one(), E::from(f[t]));
    return v;
  }
  // Appends a variable, the new highest, with the pair (l, h).
  void push(const E &l, const E &h) {
    BLK_REQUIRE(vars_ < 63, "tensor: more than 63 variables");
    const bool l0 = is_zero(l), h0 = is_zero(h);
    if ((l0 && h == E::one()) || (h0 && l == E::one())) {
      pick_mask_ |= u64(1) << vars_;
      if (l0) pick_value_ |= u64(1) << vars_;
    } else {
      own_pairs();
      std::vector<E> &p = const_cast<std::vector<E> &>(*pairs_);  // ours alone (own_pairs)
      p.push_back(l), p.push_back(h);
    }
    vars_++;
  }
  // Replaces the pair of variable t.
  void set_pair(unsigned t, const E &l, const E &h) {
    std::vector<E> los, his;
    for (unsigned i = 0; i < vars_; i++) los.push_back(i == t ? l : lo(i)), his.push_back(i == t ? h : hi(i));
    rebuild(los, his);
  }
  // Variable t becomes variable (t + vars() - by) mod vars(): the first `by` variables move after
  // the others. dense_vars is not updated: that is left to the caller.
  void rotate(unsigned by) {
    std::vector<E> los, his;
    for (unsigned i = 0; i < vars_; i++) los.push_back(lo((i + by) % vars_)), his.push_back(hi((i + by) % vars_));
    rebuild(los, his);
  }
  // Appends count variables that index the dense factor. They become the next higher bits of the
  // factor's index.
  void push_dense(unsigned count) {
    BLK_REQUIRE(!values && !is_held(), "tensor: a dense variable is added after the slice was fixed");
    for (unsigned i = 0; i < count; i++) dense_vars.push_back(vars()), push(E::one(), E::one());
  }
  // Sets the dense factor to the slice [first_, first_ + 2^(number of dense variables)) of values_.
  void set_values(std::shared_ptr<const std::vector<E>> values_, u64 first_) {
    BLK_REQUIRE(!dense_vars.empty() && values_ && !is_held(), "tensor: a slice for a tensor without dense variables, no vector, or a second factor");
    dense_check(first_, values_->size());
    values = std::move(values_), first = first_;
  }
  // The same for a vector of `length` entries in the table `id` of the prover's engine.
  void set_held(TableId id, u64 first_, u64 length) {
    BLK_REQUIRE(id != NOT_HELD && !values && !is_held(), "tensor: no table, or a second factor");
    dense_check(first_, length);
    held = id, held_length = length, first = first_;
  }
  // For an engine: the same Tensor with the dense factor as a host vector. `table` has all entries
  // of the table `held`; its length is compared with held_length, and the slice with the table.
  Tensor with_table(std::shared_ptr<const std::vector<E>> table) const {
    BLK_REQUIRE(is_held() && table, "tensor: no factor in the engine's hands, or no table");
    BLK_REQUIRE(table->size() == held_length, "tensor: the engine's table has another length than the factor says");
    dense_check(first, table->size());
    Tensor v = *this;
    v.held = NOT_HELD, v.held_length = 0;
    if (dense_vars.empty()) v.scalar = v.scalar * (*table)[first], v.first = 0;
    else v.values = std::move(table);
    return v;
  }
  // Inserts, before variable `at`, count choice variables that fix their index bits to `value`
  // (least significant bit first). Not checked: value < 2^count; higher bits of `value` are ignored
  // (Form::insert_choice() ends the process on them).
  void insert_choice(unsigned at, unsigned count, u64 value) {
    BLK_REQUIRE(at <= vars(), "tensor: no such variable");
    BLK_REQUIRE(count <= 63 && vars_ + count <= 63, "tensor: more than 63 variables");  // count alone too: a huge count (a subtraction that wrapped) could wrap this sum
    if (!count) return;
    const u64 low = (u64(1) << at) - 1, ones = (u64(1) << count) - 1;
    pick_mask_ = (pick_mask_ & low) | ((pick_mask_ & ~low) << count) | (ones << at);
    pick_value_ = (pick_value_ & low) | ((pick_value_ & ~low) << count) | ((value & ones) << at);
    vars_ += count;
    for (unsigned &d : dense_vars)
      if (d >= at) d += count;
  }
  // The entry at index j.
  E at(u64 j) const {
    factor_check();
    if ((j & pick_mask_) != pick_value_) return E::zero();
    E r = scalar;
    size_t pair = from_;
    for (unsigned t = 0; t < vars_; t++)
      if (!picks(t)) r = r * (*pairs_)[2 * pair++ + ((j >> t) & 1)];
    if (!has_dense() || is_zero(r)) return r;
    on_host("tensor: an entry is asked of a factor that only the engine holds");
    return r * (*values)[first + dense_index(j)];
  }
  // The multilinear extension at a point x of vars() coordinates: sum_j eq(j, x) v[j].
  E mle(const E *x) const {
    on_host("tensor: the value at a point is asked of a factor that only the engine holds");  // also for a factor with no dense variable, which `fix_low` would let through
    Tensor v = *this;
    v.fix_low(x, vars());
    return v.scalar;
  }
  // A cache for fix_low(): the product of the factors that the choice variables among the first
  // count variables get at g. Many Tensors of a Form have the same choice bits, so Form::fix_low()
  // passes the last product from one Tensor to the next. Valid for one g only.
  struct Picked {
    u64 mask = 0, value = 0;
    E product = E::one();
  };
  struct Slice {  // see folded_slice()
    std::shared_ptr<const std::vector<E>> values;
    E factor = E::one();
  };
  // Fixes the first count variables at g: entry j becomes sum_i eq(i, g) v[i + 2^count j]. Dense
  // variables among them are folded out of the slice.
  void fix_low(const E *g, unsigned count) {
    Picked last;
    fix_low(g, count, last);
  }
  // last: see Picked. ready: the folded slice, if the caller has it (folded_slice() of a Tensor
  // with the same slice and dense variables).
  void fix_low(const E *g, unsigned count, Picked &last, const Slice *ready = nullptr) {
    BLK_REQUIRE(count <= vars(), "tensor: more variables fixed than there are");
    factor_check();
    if (slice_is_cut(count)) on_host("tensor: a variable is fixed in a factor that only the engine holds");
    const u64 low = count < 64 ? (u64(1) << count) - 1 : ~u64(0), mask = pick_mask_ & low, value = pick_value_ & low;
    if (mask != last.mask || value != last.value) {
      last.mask = mask, last.value = value, last.product = E::one();
      for (unsigned t = 0; t < count; t++)
        if ((mask >> t) & 1) last.product = last.product * (((value >> t) & 1) ? g[t] : E::one() - g[t]);
    }
    if (mask) scalar = scalar * last.product;
    for (unsigned t = 0; t < count; t++) {
      if ((mask >> t) & 1) continue;
      const E &l = (*pairs_)[2 * from_], &h = (*pairs_)[2 * from_ + 1];
      from_++;
      if (l == E::one() && h == E::one()) continue;  // a dense variable, or no weight on this variable
      scalar = scalar * (l + g[t] * (h - l));
    }
    pick_mask_ = count < 64 ? pick_mask_ >> count : 0, pick_value_ = count < 64 ? pick_value_ >> count : 0;
    vars_ -= count;
    if (!has_dense()) return;
    if (slice_is_cut(count)) {
      const Slice s = ready ? *ready : folded_slice(g, count);
      dense_vars.erase(std::remove_if(dense_vars.begin(), dense_vars.end(), [&](unsigned d) { return d < count; }), dense_vars.end());
      if (dense_vars.empty()) scalar = scalar * s.factor, values.reset();
      else values = s.values;
      first = 0;
    }
    for (unsigned &d : dense_vars) d -= count;
  }
  // Whether one of the first count variables is dense.
  bool slice_is_cut(unsigned count) const {
    for (unsigned d : dense_vars)
      if (d < count) return true;
    return false;
  }
  // The dense factor after the dense variables below count are fixed at g: a shorter slice, from
  // entry 0 on, or, if no dense variable is left, a single factor. It depends only on the slice,
  // dense_vars, g and count, so Form::fix_low() folds each distinct slice once.
  // room: a scratch buffer that a caller passes from call to call; it only grows, and its contents
  // mean nothing between calls. The folds work in `room`, and the result is a new vector of the
  // folded length: a copy of the whole slice, halved by resize(), would keep the whole length
  // allocated for as long as the result lives.
  Slice folded_slice(const E *g, unsigned count, std::vector<E> &room) const {
    on_host("tensor: a variable is fixed in a factor that only the engine holds");
    size_t n = size_t(1) << dense_vars.size();
    BLK_REQUIRE(values, "tensor: dense variables without a vector");
    BLK_REQUIRE(first <= values->size() && n <= values->size() - first, "tensor: a written-out slice reaches beyond its vector");
    const E *from = values->data() + first;
    for (size_t i = dense_vars.size(); i-- > 0;) {  // from the top bit down, so that the lower bits keep their places
      if (dense_vars[i] >= count) continue;
      const E x = g[dense_vars[i]];  // a copy, which the loop's writes cannot change
      const size_t low_half = size_t(1) << i;
      if (room.size() < n / 2) room.resize(n / 2);
      E *to = room.data();  // in place from the second fold on: the entry written is at or before the two just read, and every later step reads further on
      for (size_t b = 0; b < n / (2 * low_half); b++)
        for (size_t a = 0; a < low_half; a++) {
          const E zero = from[a + 2 * low_half * b], one = from[a + low_half + 2 * low_half * b];
          to[a + low_half * b] = zero + (one - zero) * x;  // x on the right: what E's product computes from its right factor alone can then be hoisted out of the loop
        }
      from = to, n /= 2;
    }
    Slice out;
    if (n == 1) out.factor = from[0];
    out.values = std::make_shared<const std::vector<E>>(from, from + n);
    return out;
  }
  Slice folded_slice(const E *g, unsigned count) const {
    std::vector<E> room;
    return folded_slice(g, count, room);
  }
  // Adds all 2^vars() entries to out. Only the indices that the choice variables allow are visited.
  // room: a scratch buffer, as in folded_slice().
  // Saves multiplications where the pairs allow: none for the pair (1, 1) or by lo = 1; one for
  // both halves where lo = 1 - hi (an equality vector: v lo = v - v hi); a product by an element of
  // F where a pair is in F. The field elements are the same in every case.
  void add_to(E *out, std::vector<E> &room) const {
    factor_check();
    on_host("tensor: a factor that only the engine holds is to be written out");
    const size_t real = reals(), n = size_t(1) << real;
    if (room.size() < n) room.resize(n);
    E *v = room.data();  // entry j: bit r of j belongs to the r-th real variable, the lowest first
    v[0] = scalar;
    for (size_t r = 0; r < real; r++) {
      const size_t half = size_t(1) << r;
      const E l = (*pairs_)[2 * (from_ + r)], h = (*pairs_)[2 * (from_ + r) + 1];  // copies, which the loops' writes cannot change
      if (l == E::one() && h == E::one()) {
        for (size_t j = 0; j < half; j++) v[j + half] = v[j];
      } else if (lies_in_F(l) && lies_in_F(h)) {
        spread(v, half, l.c[0], h.c[0], l == E::one(), l + h == E::one());
      } else {
        spread(v, half, l, h, l == E::one(), l + h == E::one());
      }
    }
    // Entry j of v belongs to the index that has j's bits at the real variables' bits (`places`)
    // and pick_value_ at the others. (at - places) & places steps through the subsets of `places`.
    const u64 places = ~pick_mask_ & (vars_ < 64 ? (u64(1) << vars_) - 1 : ~u64(0));
    u64 at = 0;
    for (size_t j = 0; j < n; j++, at = (at - places) & places) {
      const u64 index = at | pick_value_;
      out[index] = out[index] + (has_dense() ? v[j] * (*values)[first + dense_index(index)] : v[j]);
    }
  }
  void add_to(E *out) const {
    std::vector<E> room;
    add_to(out, room);
  }
  // The factor on the variables from .. from + count - 1, written out: 2^count products of their
  // pairs, without the scalar and the dense factor. Computed as in add_to(); a choice variable
  // moves or clears entries and multiplies nothing.
  std::vector<E> part(unsigned from, unsigned count) const {
    BLK_REQUIRE(from <= vars_ && count <= vars_ - from, "tensor: a part beyond the variables");
    std::vector<E> w(size_t(1) << count);
    w[0] = E::one();
    for (unsigned t = 0; t < count; t++) {
      const size_t half = size_t(1) << t;
      if (picks(from + t)) {
        const bool one = (pick_value_ >> (from + t)) & 1;
        for (size_t j = 0; j < half; j++) w[j + half] = one ? w[j] : E::zero();
        for (size_t j = 0; one && j < half; j++) w[j] = E::zero();
        continue;
      }
      const E l = lo(from + t), h = hi(from + t);
      if (l == E::one() && h == E::one()) {
        for (size_t j = 0; j < half; j++) w[j + half] = w[j];
      } else if (lies_in_F(l) && lies_in_F(h)) {
        spread(w.data(), half, l.c[0], h.c[0], l == E::one(), l + h == E::one());
      } else {
        spread(w.data(), half, l, h, l == E::one(), l + h == E::one());
      }
    }
    return w;
  }

 private:
  // Adds one real variable: v[0 .. half) times h goes to v[half .. 2 half), and v[0 .. half) is
  // multiplied by l. M: E, or F for a pair in F.
  template <class M> static void spread(E *v, size_t half, const M l, const M h, bool l_is_one, bool l_is_one_less_h) {
    if (l_is_one) {
      for (size_t j = 0; j < half; j++) v[j + half] = v[j] * h;
    } else if (l_is_one_less_h) {
      for (size_t j = 0; j < half; j++) v[j + half] = v[j] * h, v[j] = v[j] - v[j + half];
    } else {
      for (size_t j = 0; j < half; j++) v[j + half] = v[j] * h, v[j] = v[j] * l;
    }
  }
  void factor_check() const { BLK_REQUIRE(dense_vars.empty() || values || is_held(), "tensor: dense variables without a vector"); }
  void on_host(const char *who) const { BLK_REQUIRE(!is_held(), who); }
  void dense_check(u64 first_, u64 length) const {
    BLK_REQUIRE(dense_vars.size() < 64 && first_ <= length && (u64(1) << dense_vars.size()) <= length - first_, "tensor: the slice does not lie in the vector");
    for (size_t i = 0; i < dense_vars.size(); i++) {
      BLK_REQUIRE(dense_vars[i] < vars() && lo(dense_vars[i]) == E::one() && hi(dense_vars[i]) == E::one(),
                  "tensor: a dense variable does not exist, or has another pair than (1, 1)");
      for (size_t j = 0; j < i; j++) BLK_REQUIRE(dense_vars[i] != dense_vars[j], "tensor: a variable is dense twice");
    }
  }
  u64 dense_index(u64 j) const {
    u64 d = 0;
    for (size_t i = 0; i < dense_vars.size(); i++) d |= ((j >> dense_vars[i]) & 1) << i;
    return d;
  }
  // The index, in the list, of the pair of the real variable t.
  size_t real_at(unsigned t) const { return from_ + (size_t)__builtin_popcountll(~pick_mask_ & ((u64(1) << t) - 1)); }
  size_t reals() const { return vars_ - (size_t)__builtin_popcountll(pick_mask_); }
  // Afterwards the list belongs to this Tensor only and ends with its last pair, so that a pair can
  // be appended.
  void own_pairs() {
    if (!pairs_) {
      pairs_ = std::make_shared<std::vector<E>>();
      from_ = 0;
    } else if (pairs_.use_count() > 1 || pairs_->size() != 2 * (from_ + reals())) {
      pairs_ = std::make_shared<std::vector<E>>(pairs_->begin() + 2 * from_, pairs_->begin() + 2 * (from_ + reals()));
      from_ = 0;
    }
  }
  void rebuild(const std::vector<E> &los, const std::vector<E> &his) {
    pairs_.reset(), from_ = 0, vars_ = 0, pick_mask_ = pick_value_ = 0;
    for (size_t i = 0; i < los.size(); i++) push(los[i], his[i]);
  }
  std::shared_ptr<const std::vector<E>> pairs_;  // lo, hi of the real variables, in their order
  size_t from_ = 0;                              // the first pair that is still there
  unsigned vars_ = 0;
  u64 pick_mask_ = 0, pick_value_ = 0;
};

// A hint for a prover's engine, which does not change the Form's vector: the tensors [from, to) and
// the entries [entries_from, entries_to) of the Form add up to scale * w, where w is written out in
// the engine's table `vector`. If choice variables were inserted after the hint was made, w has
// fewer variables than the Form: the sum is 0 at every index j with
// (j & choice_mask) != choice_value, and elsewhere it is scale * w at the index formed by the other
// bits of j, in their order. An engine may read w instead of computing those parts again. A
// verifier's Forms have no hints.
struct Held {
  TableId vector = ~TableId(0);
  E scale = E::one();
  size_t from = 0, to = 0, entries_from = 0, entries_to = 0;
  u64 choice_mask = 0, choice_value = 0;
};

// A term that only the prover's engine has: scale * w, with w written out in the engine's table
// `vector`, under choice variables as in Held. Unlike a hint it is part of the Form's vector. It
// saves a prover from building the structured form, which only a verifier needs, of a long claim
// that has been carried through many folds (see FrozenPart).
// - Only a prover's Form has one (a verifier has no engine).
// - Only an engine can read it. Form's mle(), fix_low(), add_to() and `freeze` end the process, and
//   so does at(j) unless (j & choice_mask) != choice_value, where the term is 0.
// - An engine that cannot use a term must end the process too, never skip the term.
struct EngineTerm {
  TableId vector = ~TableId(0);
  E scale = E::one();
  u64 choice_mask = 0, choice_value = 0;
};

// A long part that is stored once: scale * f under choice variables (as in Held), where f is a Form
// on `vars` variables that no longer changes and that many Forms may share. It stands for the same
// vector as its parts copied out (Form::flattened()) and exists only for speed. A claim carried
// through a chain of folds grows by thousands of Tensors at each fold, and between two folds it is
// only copied, scaled, put into one row of a larger table (choice variables) and fixed at those
// same variables: each of these takes one step here, whatever the length of f.
struct Form;
struct FrozenPart {
  std::shared_ptr<const Form> form;
  unsigned vars = 0;  // of *form; checked against it once, by `Form::freeze`
  E scale = E::one();
  u64 choice_mask = 0, choice_value = 0;
};

// A coefficient vector: the sum of the tensors, which must all have the same number of variables
// (fits() checks this, add() does not), of the single entries (index, coefficient), where an index
// may occur more than once, of the frozen parts and, for a prover, of the engine's terms. `held`:
// the hints.
struct Form {
  std::vector<Tensor> tensors;
  std::vector<std::pair<u64, E>> entries;
  std::vector<FrozenPart> frozen;
  std::vector<EngineTerm> engine_terms;
  std::vector<Held> held;

  bool empty() const { return tensors.empty() && entries.empty() && frozen.empty() && engine_terms.empty(); }
  // Turns this whole Form, a vector on vars variables, into one frozen part. Hints are dropped:
  // they refer to parts by their indices in `tensors` and `entries`.
  void freeze(unsigned vars) {
    require_written("form: a form with a term that only the engine knows is frozen");
    BLK_REQUIRE(!has_held_factor(), "form: a form with a factor that only the engine holds is frozen");
    BLK_REQUIRE(fits(vars), "form: a part does not fit a vector of this many variables");
    held.clear();
    auto all = std::make_shared<Form>();
    all->tensors.swap(tensors), all->entries.swap(entries), all->frozen.swap(frozen);
    frozen.push_back(FrozenPart{std::move(all), vars, E::one(), 0, 0});
  }
  // The same vector with every frozen part copied out, at all depths.
  Form flattened() const {
    Form f = *this;
    f.thaw();
    return f;
  }
  // flattened(), in place.
  void thaw() {
    std::vector<FrozenPart> parts;
    parts.swap(frozen);
    for (const FrozenPart &p : parts) {
      Form f = p.form->flattened();
      for (unsigned t = 0; t < 64; t++)
        if ((p.choice_mask >> t) & 1) f.insert_choice(t, 1, (p.choice_value >> t) & 1);
      for (Tensor &v : f.tensors) v.scalar = v.scalar * p.scale, tensors.push_back(std::move(v));
      for (const std::pair<u64, E> &e : f.entries) entries.emplace_back(e.first, e.second * p.scale);
    }
  }
  void require_written(const char *who) const { BLK_REQUIRE(engine_terms.empty(), who); }
  // Whether a tensor's dense factor is in an engine's table (Tensor::held). Frozen parts are not
  // searched: `freeze` ends the process on a Form with such a tensor.
  bool has_held_factor() const {
    for (const Tensor &v : tensors)
      if (v.is_held()) return true;
    return false;
  }
  void add(Tensor v) { tensors.push_back(std::move(v)); }
  void add_entry(u64 index, const E &coefficient) { entries.emplace_back(index, coefficient); }
  // this += x * other.
  void add(const Form &other, const E &x) {
    BLK_REQUIRE(&other != this, "form: a form is added to itself");
    const size_t t0 = tensors.size(), e0 = entries.size();
    for (Tensor v : other.tensors) v.scalar = v.scalar * x, tensors.push_back(std::move(v));
    for (const std::pair<u64, E> &e : other.entries) entries.emplace_back(e.first, e.second * x);
    for (FrozenPart p : other.frozen) p.scale = p.scale * x, frozen.push_back(std::move(p));
    for (EngineTerm t : other.engine_terms) t.scale = t.scale * x, engine_terms.push_back(t);
    for (Held h : other.held) {
      h.scale = h.scale * x;
      h.from += t0, h.to += t0, h.entries_from += e0, h.entries_to += e0;
      held.push_back(h);
    }
  }
  // The same for a form that is not needed afterwards: its tensors move.
  void add(Form &&other, const E &x) {
    BLK_REQUIRE(&other != this, "form: a form is added to itself");
    const size_t t0 = tensors.size(), e0 = entries.size();
    if (tensors.empty() && tensors.capacity() < other.tensors.size()) tensors = std::move(other.tensors);  // (capacity that a caller has reserved is kept)
    else tensors.insert(tensors.end(), std::make_move_iterator(other.tensors.begin()), std::make_move_iterator(other.tensors.end()));
    for (size_t i = t0; i < tensors.size(); i++) tensors[i].scalar = tensors[i].scalar * x;
    if (entries.empty()) entries = std::move(other.entries);
    else entries.insert(entries.end(), other.entries.begin(), other.entries.end());
    for (size_t i = e0; i < entries.size(); i++) entries[i].second = entries[i].second * x;
    for (FrozenPart &p : other.frozen) p.scale = p.scale * x, frozen.push_back(std::move(p));
    for (EngineTerm t : other.engine_terms) t.scale = t.scale * x, engine_terms.push_back(t);
    for (Held h : other.held) {
      h.scale = h.scale * x;
      h.from += t0, h.to += t0, h.entries_from += e0, h.entries_to += e0;
      held.push_back(h);
    }
  }
  void scale(const E &x) {
    for (Tensor &v : tensors) v.scalar = v.scalar * x;
    for (std::pair<u64, E> &e : entries) e.second = e.second * x;
    for (FrozenPart &p : frozen) p.scale = p.scale * x;
    for (EngineTerm &t : engine_terms) t.scale = t.scale * x;
    for (Held &h : held) h.scale = h.scale * x;
  }
  E at(u64 j) const {
    for (const EngineTerm &t : engine_terms)  // where a choice bit differs the term's entry is 0, whatever its vector contains; any other entry only the engine knows
      BLK_REQUIRE((j & t.choice_mask) != t.choice_value, "form: an entry is asked of a form with a term that only the engine knows");
    E r = E::zero();
    for (const Tensor &v : tensors) r += v.at(j);
    for (const std::pair<u64, E> &e : entries)
      if (e.first == j) r += e.second;
    for (const FrozenPart &p : frozen)
      if ((j & p.choice_mask) == p.choice_value) r += p.scale * p.form->at(without_bits(j, p.choice_mask));
    return r;
  }
  // The index formed by the bits of j outside mask, in their order.
  static u64 without_bits(u64 j, u64 mask) {
    u64 inner = 0;
    for (unsigned t = 0, i = 0; t < 64; t++)
      if (!((mask >> t) & 1)) inner |= ((j >> t) & 1) << i++;
    return inner;
  }
  // The multilinear extension at a point x of vars coordinates.
  E mle(const E *x, unsigned vars) const {
    BLK_REQUIRE(fits(vars), "form: a part does not fit a vector of this many variables");
    Form f = *this;
    f.fix_low(x, vars);
    return f.at(0);
  }
  // Fixes the first count variables at g, then multiplies everything by `times`. An entry moves to
  // the index formed by its higher bits, with the weight that g gives to its low bits. Hints are
  // dropped.
  void fix_low(const E *g, unsigned count, const E &times = E::one()) {
    require_written("form: variables are fixed in a form with a term that only the engine knows");
    const bool scaled = !(times == E::one());
    {  // frozen parts: a factor each if only choice variables are fixed in all; else `thaw` all
      const u64 low = count < 64 ? (u64(1) << count) - 1 : ~u64(0);
      bool all_choices = true;
      for (const FrozenPart &p : frozen) all_choices = all_choices && (p.choice_mask & low) == low;
      if (!all_choices) thaw();
      for (FrozenPart &p : frozen) {
        for (unsigned t = 0; t < count; t++) p.scale = p.scale * (((p.choice_value >> t) & 1) ? g[t] : E::one() - g[t]);
        if (scaled) p.scale = p.scale * times;
        p.choice_mask = count < 64 ? p.choice_mask >> count : 0, p.choice_value = count < 64 ? p.choice_value >> count : 0;
      }
    }
    // Each distinct slice that is cut is folded once (Tensor::folded_slice()).
    constexpr size_t NO_SLICE = ~size_t(0);
    std::map<std::tuple<const void *, u64, std::vector<unsigned>>, size_t> number;
    std::vector<size_t> slice_of(tensors.size(), NO_SLICE), first_with;
    for (size_t i = 0; i < tensors.size(); i++) {
      const Tensor &v = tensors[i];
      if (!v.has_dense() || !v.slice_is_cut(count)) continue;
      BLK_REQUIRE(!v.is_held(), "tensor: a variable is fixed in a factor that only the engine holds");
      BLK_REQUIRE(v.values, "tensor: dense variables without a vector");
      const auto at = number.emplace(std::make_tuple((const void *)v.values.get(), v.first, v.dense_vars), first_with.size());
      if (at.second) first_with.push_back(i);
      slice_of[i] = at.first->second;
    }
    std::vector<Tensor::Slice> folded(first_with.size());
    std::vector<E> room;
    for (size_t s = 0; s < first_with.size(); s++) folded[s] = tensors[first_with[s]].folded_slice(g, count, room);
    Tensor::Picked last;
    for (size_t i = 0; i < tensors.size(); i++) {
      Tensor &v = tensors[i];
      if (scaled) v.scalar = v.scalar * times;
      v.fix_low(g, count, last, slice_of[i] == NO_SLICE ? nullptr : &folded[slice_of[i]]);
    }
    if (count <= 12 && entries.size() > (size_t(1) << count)) {  // one product per entry: the weights of all values of the low bits are computed once
      std::vector<E> weight(size_t(1) << count, E::one());
      for (unsigned t = 0; t < count; t++)
        for (size_t j = 0; j < (size_t(1) << t); j++) weight[j | (size_t(1) << t)] = weight[j] * g[t], weight[j] = weight[j] * (E::one() - g[t]);
      for (E &w : weight) w = w * times;
      for (std::pair<u64, E> &e : entries) e.second = e.second * weight[e.first & ((u64(1) << count) - 1)], e.first >>= count;
    } else {
      for (std::pair<u64, E> &e : entries) {
        for (unsigned t = 0; t < count; t++) e.second = e.second * (((e.first >> t) & 1) ? g[t] : E::one() - g[t]);
        e.second = e.second * times;
        e.first = count < 64 ? e.first >> count : 0;
      }
    }
    held.clear();
    merge_entries();
  }
  // Merges entries that have the same index and sorts by index. Does nothing if there are hints,
  // which refer to entries by their indices in `entries`.
  void merge_entries() {
    if (!held.empty() || entries.size() < 2) return;
    if (entries.size() >= 4096) {
      // Many entries, often on few indices: they are first added up in a hash table with open
      // addressing, so that only the distinct ones are sorted. Sums in the field do not depend on
      // the order, so the result is the same list.
      size_t room = 1024;
      std::vector<std::pair<u64, E>> slot;
      std::vector<bool> used;
      size_t distinct = 0;
      auto place = [&](u64 index) {
        // The hash multiplies the index by 2^64 / the golden ratio, rounded down: the same constant
        // as the increment of SplitMix64 (splitmix64.c by Sebastiano Vigna, public domain).
        // The multiplier is that of Fibonacci hashing (D. E. Knuth, "The Art of Computer Programming", volume 3).
        u64 z = index * 0x9E3779B97F4A7C15ull;
        size_t at = (size_t)(z >> 20) & (room - 1);
        while (used[at] && slot[at].first != index) at = (at + 1) & (room - 1);
        return at;
      };
      auto grow = [&]() {
        std::vector<std::pair<u64, E>> old = std::move(slot);
        std::vector<bool> was = std::move(used);
        room *= 4;
        slot.assign(room, {0, E::zero()}), used.assign(room, false);
        for (size_t i = 0; i < old.size(); i++)
          if (was[i]) {
            const size_t at = place(old[i].first);
            slot[at] = old[i], used[at] = true;
          }
      };
      slot.assign(room, {0, E::zero()}), used.assign(room, false);
      for (const std::pair<u64, E> &e : entries) {
        size_t at = place(e.first);
        if (used[at]) {
          slot[at].second += e.second;
          continue;
        }
        if (2 * (distinct + 1) > room) grow(), at = place(e.first);
        slot[at] = e, used[at] = true, distinct++;
      }
      entries.clear();
      for (size_t i = 0; i < room; i++)
        if (used[i]) entries.push_back(slot[i]);
    }
    std::sort(entries.begin(), entries.end(), [](const std::pair<u64, E> &a, const std::pair<u64, E> &b) { return a.first < b.first; });
    size_t n = 0;
    for (size_t i = 0; i < entries.size(); i++) {
      if (n && entries[n - 1].first == entries[i].first) entries[n - 1].second += entries[i].second;
      else entries[n++] = entries[i];
    }
    entries.resize(n);
  }
  // Inserts, before variable `at`, count choice variables that fix their index bits to `value`.
  void insert_choice(unsigned at, unsigned count, u64 value) {
    BLK_REQUIRE(at + count < 64 && value >> count == 0, "form: the value has more bits than are inserted, or an index would pass 63 bits");
    const u64 low = (u64(1) << at) - 1;
    auto widen = [&](u64 j) { return (j & low) | ((j & ~low) << count); };
    for (Tensor &v : tensors) v.insert_choice(at, count, value);
    for (std::pair<u64, E> &e : entries) e.first = widen(e.first) | (value << at);
    for (FrozenPart &p : frozen) {
      p.choice_mask = widen(p.choice_mask) | (((u64(1) << count) - 1) << at);
      p.choice_value = widen(p.choice_value) | (value << at);
    }
    for (EngineTerm &t : engine_terms) {
      t.choice_mask = widen(t.choice_mask) | (((u64(1) << count) - 1) << at);
      t.choice_value = widen(t.choice_value) | (value << at);
    }
    for (Held &h : held) {
      h.choice_mask = widen(h.choice_mask) | (((u64(1) << count) - 1) << at);
      h.choice_value = widen(h.choice_value) | (value << at);
    }
  }
  // Adds all 2^vars entries to out.
  void add_to(E *out, unsigned vars) const {
    require_written("form: a form with a term that only the engine knows is to be written out");
    add_written_to(out, vars);
  }
  // The same without the engine's terms: for an engine, which adds those itself (engine_term_at()).
  void add_written_to(E *out, unsigned vars) const {
    std::vector<E> room;
    for (const Tensor &v : tensors) {
      BLK_REQUIRE(v.vars() == vars, "form: a tensor with another number of variables");
      v.add_to(out, room);
    }
    for (const std::pair<u64, E> &e : entries) {
      BLK_REQUIRE(e.first >> vars == 0, "form: an entry beyond the vector's length");
      out[e.first] += e.second;
    }
    for (const FrozenPart &p : frozen) {
      BLK_REQUIRE(frozen_fits(p, vars), "form: a frozen part with another number of variables");
      std::vector<E> inner(size_t(1) << p.vars, E::zero());
      p.form->add_to(inner.data(), p.vars);
      for (u64 j = 0; j < u64(1) << vars; j++)
        if ((j & p.choice_mask) == p.choice_value) out[j] += p.scale * inner[without_bits(j, p.choice_mask)];
    }
  }
  static bool frozen_fits(const FrozenPart &p, unsigned vars) {
    return vars < 64 && p.choice_mask >> vars == 0 && p.vars + (unsigned)__builtin_popcountll(p.choice_mask) == vars;
  }
  // Whether every part fits a vector of 2^vars entries. Of an engine's term only the choice mask
  // can be checked here.
  bool fits(unsigned vars) const {
    for (const FrozenPart &p : frozen)
      if (!frozen_fits(p, vars)) return false;
    for (const EngineTerm &t : engine_terms)
      if (vars < 64 && t.choice_mask >> vars != 0) return false;
    for (const Tensor &v : tensors)
      if (v.vars() != vars) return false;
    for (const std::pair<u64, E> &e : entries)
      if (vars < 64 && e.first >> vars != 0) return false;
    return true;
  }
};

// Entry j of scale * w under choice variables (see Held), given w written out.
inline E under_choices_at(const E &scale, u64 choice_mask, u64 choice_value, const std::vector<E> &w, u64 j, unsigned vars) {
  if ((j & choice_mask) != choice_value) return E::zero();
  u64 inner = 0;
  for (unsigned t = 0, i = 0; t < vars; t++)
    if (!((choice_mask >> t) & 1)) inner |= ((j >> t) & 1) << i++;
  BLK_REQUIRE(inner < w.size(), "form: a vector that the engine holds is shorter than the form says");
  return scale * w[inner];
}
// Entry j of the vector that a hint stands for, and of an engine's term, given w written out. For
// engines and tests.
inline E held_at(const Held &h, const std::vector<E> &w, u64 j, unsigned vars) { return under_choices_at(h.scale, h.choice_mask, h.choice_value, w, j, vars); }
inline E engine_term_at(const EngineTerm &t, const std::vector<E> &w, u64 j, unsigned vars) {
  return under_choices_at(t.scale, t.choice_mask, t.choice_value, w, j, vars);
}

} // namespace blk
