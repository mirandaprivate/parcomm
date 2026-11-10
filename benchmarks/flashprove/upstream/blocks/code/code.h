// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The linear codes that encode rows, with their randomized (zero-knowledge) encoding: Encoder for
// whole codewords on the CPU, symbol_at() and generator_row() for one position.
//
// A row is a vector encoded as one codeword of n = 2^log_n symbols, one at each position x. It has
// ell message entries and r random entries, the random tail (`tail`), which are to hide the message
// when up to hiding_positions() positions are opened. The code is plain Reed-Solomon on cosets
// modulo 2^64 - 2^32 + 1 (blocks/code/cosets.h), an evaluation code with a basis of product form:
// at position x there are factors f_0, f_1, .. (factors()), basis function j is the product of the
// f_i over the bits i of j that are 1, and symbol(x) = sum_j coefficient_j * basis_j(x). Position x
// has a point z, f_i = z^(2^i) and basis_j = z^j. Message and tail each fill one run of
// coefficients (from message_at() and tail_at() on), so that
//
//   symbol(x) = message_scale(x) * sum_{j < ell} message[j] * prod_{bit i of j} f_i
//             + tail_scale(x)    * sum_{j < r}   tail[j]    * prod_{bit i of j} f_i.
//
// A caller that uses message_scale() and tail_scale() need not know where the two runs start.
// A row over E, the extension of the prime field F (blocks/field/field.h), is encoded coordinate by
// coordinate: the same formula, over E.
//
// The plain Reed-Solomon encoding (tail, then message) follows the zero-knowledge encoding of Reed-Solomon codes of
// the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include "base.h"
#include "code/short_position.h"
#include "field/field.h"
#include "field/mle.h"
#include "code/cosets.h"
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace blk {
// The largest log_n that code_ok() accepts.
constexpr unsigned CODE_MOST_LOG_N = GL64_MOST_LOG_N;

// Call code_ok() first on numbers that come from outside the program: in this file only Encoder's
// constructor does, and shifts and sums can otherwise overflow. Positions must be below n, which is
// not checked. BLK_HD (blocks/base.h) marks the functions that GPU code can call too.
struct Code {
  unsigned log_n; // the length is 2^log_n
  size_t ell;     // message entries of a row
  size_t r;       // random entries of a row

  BLK_HD u64 n() const {
    return (u64)1 << log_n;
  }
  // The index of the first coefficient of the message, and of the tail: the tail has the
  // coefficients 0 .. r - 1 and message entry j the coefficient r + j, so message_scale = z^r and
  // tail_scale = 1.
  BLK_HD u64 message_at() const {
    return r;
  }
  BLK_HD u64 tail_at() const {
    return 0;
  }
  // The dimension of the evaluation code that contains all codewords: one more than the index of
  // the highest coefficient in use. Rate and distance are those of this dimension.
  BLK_HD u64 dim() const {
    return (u64)ell + r;
  }
  BLK_HD unsigned log_dim() const {
    return ceil_log2(dim());
  }
  BLK_HD unsigned log_ell() const { // the number of factors that the message's sum uses
    return ceil_log2(ell);
  }
  BLK_HD unsigned log_r() const { // the number of factors that the tail's sum uses
    return ceil_log2(r);
  }
  double rate() const {
    return (double)dim() / (double)n();
  }
};
BLK_HD inline bool operator==(const Code &a, const Code &b) {
  return a.log_n == b.log_n && a.ell == b.ell && a.r == b.r;
}

// Whether the three numbers describe a code. The dimension must also fit in one coset (2^s points,
// s = gl64_subgroup_log(log_n)), so that log_dim() <= s: the fast transform (Encoder::evaluate())
// uses the first log_dim() factors and needs what factors() states of the factors below s.
BLK_HD inline bool code_ok(const Code &c) {
  if (c.log_n < 1 || c.log_n > CODE_MOST_LOG_N || c.ell < 1) return false;
  if (c.ell > c.n() || c.r > c.n() || c.dim() > c.n()) return false;
  if (c.dim() > ((u64)1 << gl64_subgroup_log(c.log_n))) return false;
  return true;
}

// Two different codewords differ in at least this many positions.
// A polynomial of degree below dim has at most dim - 1 zeros.
BLK_HD inline u64 min_distance(const Code &c) {
  return c.n() - c.dim() + 1;
}
// With a uniform tail, the symbols of a row at any set of this many distinct positions are uniform
// and independent of the message. (A position that is opened twice shows the same symbol twice and
// counts once.)
// The tail's part of the symbol at x is tail_scale(x), which is never 0, times sum_j tail[j]
// basis_j(x), j < r. The basis is 1, z, .., z^(r-1), and evaluation at r distinct points is a
// bijection.
BLK_HD inline u64 hiding_positions(const Code &c) {
  return c.r;
}
// The least r for which hiding_positions() is at least t.
BLK_HD inline u64 tail_for_positions(u64 t) {
  return t;
}

// The main code encodes an oracle message's main rows, a mask code its mask rows (defined in
// blocks/iop/relation.h, which calls the main code the base code). A mask code is a code like any
// other, no longer than the main code (mask_fits()). Where a main row is read at x, a mask row is
// read at mask_position(): x modulo the mask code's length.
BLK_HD inline bool mask_fits(const Code &main, const Code &mask) {
  return mask.log_n <= main.log_n;
}
BLK_HD inline u64 mask_position(const Code &main, const Code &mask, u64 x) {
  return short_position(main.log_n, mask.log_n, x);
}

// Writes f[0 .. count) at position x; a count above 64 is cut to 64. log_dim() factors suffice for
// every sum in this file.
// For i < gl64_subgroup_log(log_n), f[i] depends only on bits >= i of x, and bit i negates it.
BLK_HD inline void factors(const Code &c, u64 x, unsigned count, F *f) {
  u64 w[64];
  if (count > 64) count = 64;
  gl64_factors(c.log_n, x, count, w);
  for (unsigned i = 0; i < count; i++) f[i] = F{w[i]};
}
// basis_j at the position whose factors are f, for any j < 2^count.
BLK_HD inline F basis_at(const F *f, unsigned count, u64 j) {
  return product_index(f, count, j);
}
// The formula's two scales, from a position's log_dim() factors. With r = 0, ignore tail_scale().
BLK_HD inline F message_scale(const Code &c, const F *f) {
  return basis_at(f, c.log_dim(), c.message_at());
}
BLK_HD inline F tail_scale(const Code &c, const F *f) {
  return basis_at(f, c.log_dim(), c.tail_at());
}

// What the formula at the top of this file needs at one position.
struct GeneratorRow {
  F message_scale, tail_scale;
  F f[64]; // log_dim() factors; the message's sum uses the first log_ell(), the tail's the first log_r()
};
BLK_HD inline GeneratorRow generator_at(const Code &c, u64 x) {
  GeneratorRow g;
  for (unsigned i = 0; i < 64; i++) g.f[i] = F::zero();
  factors(c, x, c.log_dim(), g.f);
  g.message_scale = message_scale(c, g.f);
  g.tail_scale = tail_scale(c, g.f);
  return g;
}
// Row x of the generator matrix: the coefficients of the ell message entries and of the r tail
// entries in the symbol at x. Either pointer may be null.
inline void generator_row(const Code &c, u64 x, F *on_message, F *on_tail) {
  const GeneratorRow g = generator_at(c, x);
  if (on_message) {
    product_table(g.f, c.log_ell(), c.ell, on_message);
    for (size_t j = 0; j < c.ell; j++) on_message[j] *= g.message_scale;
  }
  if (on_tail) {
    product_table(g.f, c.log_r(), c.r, on_tail);
    for (size_t j = 0; j < c.r; j++) on_tail[j] *= g.tail_scale;
  }
}
// The symbol at x of the row with this message and tail, over F or E: about ell + r products.
template <class T> inline T symbol_at(const Code &c, u64 x, const T *message, const T *tail) {
  const GeneratorRow g = generator_at(c, x);
  return product_dot(message, c.ell, g.f, c.log_ell()) * g.message_scale +
         product_dot(tail, c.r, g.f, c.log_r()) * g.tail_scale;
}

// The fast transform's table for one length, built at the first call and shared by later calls
// with the same log_n: n entries, kept until the program exits. Layer L of Encoder::evaluate()
// reads tw[2^(log_n - 1 - L) + h]: factor L at the positions whose bits above L spell h and whose
// bit L is 0. The entries of layers L >= gl64_subgroup_log(log_n) are 0.
// Safe to call from several threads. log_n is not checked here.
inline std::shared_ptr<const std::vector<F>> transform_table(unsigned log_n) {
  static std::mutex lock;
  static std::map<unsigned, std::shared_ptr<const std::vector<F>>> made;
  std::lock_guard<std::mutex> hold(lock);
  auto &slot = made[log_n];
  if (!slot) {
    std::vector<u64> tw;
    gl64_table(log_n, gl64_subgroup_log(log_n), tw);
    auto t = std::make_shared<std::vector<F>>(tw.size());
    for (size_t i = 0; i < tw.size(); i++) (*t)[i] = F{tw[i]};
    slot = t;
  }
  return slot;
}

// Encodes rows of one code. It holds the transform table: reuse one Encoder for many rows.
// The constructor ends the process (BLK_REQUIRE) if code_ok(c) is false.
class Encoder {
public:
  explicit Encoder(const Code &c) : code_(c) {
    BLK_REQUIRE(code_ok(c), "encoder: these numbers are not a code");
    tw_ = transform_table(c.log_n);
  }
  const Code &code() const {
    return code_;
  }
  // word[0 .. n) from message[0 .. ell) and tail[0 .. r), over F or E.
  template <class T> void encode(const T *message, const T *tail, T *word) const {
    const size_t block = (size_t)1 << code_.log_dim();
    for (size_t j = 0; j < block; j++) word[j] = T::zero();
    for (size_t j = 0; j < code_.ell; j++) word[code_.message_at() + j] = message[j];
    for (size_t j = 0; j < code_.r; j++) word[code_.tail_at() + j] = tail[j];
    evaluate(word);
  }
  // In place: word[0 .. 2^log_dim()) holds coefficients on entry, word[0 .. n) the evaluations on
  // return. Layer L turns (a, b), 2^L apart, into (a + t b, a - t b), t = factor L at the position
  // of a. From layer log_dim() up, b is 0 and the step is a copy: the first loop makes these
  // copies.
  template <class T> void evaluate(T *word) const {
    const size_t n = (size_t)code_.n();
    const unsigned ld = code_.log_dim();
    for (size_t at = (size_t)1 << ld; at < n; at <<= 1)
      for (size_t j = 0; j < at; j++) word[at + j] = word[j];
    const F *tw = tw_->data();
    for (unsigned L = ld; L-- > 0;) {
      const size_t st = (size_t)1 << L;
      const F *t = tw + (n >> (L + 1));
      for (size_t h = 0; h < (n >> (L + 1)); h++) {
        T *a = word + 2 * st * h, *b = a + st;
        for (size_t j = 0; j < st; j++) {
          const T u = a[j], v = b[j] * t[h];
          a[j] = u + v;
          b[j] = u - v;
        }
      }
    }
  }
  // The inverse of encode(); only tests call it. Returns false if word[0 .. n) is not a codeword;
  // message and tail are written in either case.
  template <class T> bool decode(const T *word, T *message, T *tail) const {
    const size_t n = (size_t)code_.n();
    const unsigned top = gl64_subgroup_log(code_.log_n);
    const size_t block = (size_t)1 << top; // the first coset, or the whole word
    std::vector<T> a(word, word + block);
    const F half = inv(F::one() + F::one());
    for (unsigned L = 0; L < top; L++) {
      const size_t st = (size_t)1 << L;
      const F *t = tw_->data() + (n >> (L + 1));
      for (size_t h = 0; h < (block >> (L + 1)); h++) {
        const F back = half * inv(t[h]);
        T *p = a.data() + 2 * st * h, *q = p + st;
        for (size_t j = 0; j < st; j++) {
          const T u = p[j], v = q[j];
          p[j] = (u + v) * half;
          q[j] = (u - v) * back;
        }
      }
    }
    for (size_t j = 0; j < code_.ell; j++) message[j] = a[code_.message_at() + j];
    for (size_t j = 0; j < code_.r; j++) tail[j] = a[code_.tail_at() + j];
    std::vector<T> again(n);
    encode(message, tail, again.data());
    for (size_t i = 0; i < n; i++)
      if (again[i] != word[i]) return false;
    return true;
  }

private:
  Code code_;
  std::shared_ptr<const std::vector<F>> tw_;
};

// Encodes one row, for callers that encode few.
inline void encode(const Code &c, const F *message, const F *tail, F *word) {
  Encoder(c).encode(message, tail, word);
}
inline void encode(const Code &c, const E *message, const E *tail, E *word) {
  Encoder(c).encode(message, tail, word);
}
} // namespace blk
