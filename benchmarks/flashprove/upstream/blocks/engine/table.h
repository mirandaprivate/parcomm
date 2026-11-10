// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// `Table`: field elements in CPU memory, as passed to and from an `Engine`
// (blocks/engine/engine.h). `Coins`: the prover's source of private randomness.
// A failed BLK_REQUIRE (blocks/base.h) ends the process.
#pragma once
#include <vector>

#include "base.h"
#include "field/field.h"

namespace blk {

// `rows` rows of `len` entries in the field F (deg = 1) or its cubic extension E (deg = 3).
// Read as a vector (see entry()) it has 2^log_rows rows; those past `rows` are zero and not
// stored, but entries() counts them.
struct Table {
  unsigned deg = 1;
  unsigned log_rows = 0;
  size_t rows = 0, len = 0;
  std::vector<F> a; // a[(row * len + place) * deg + coordinate]

  Table() = default;
  Table(unsigned deg_, size_t rows_, size_t len_) : deg(deg_), rows(rows_), len(len_), a(rows_ * len_ * deg_) {
    BLK_REQUIRE(deg == 1 || deg == 3, "table: entries are in F or in E");
    while ((size_t(1) << log_rows) < rows) log_rows++;
  }
  E at(size_t row, size_t place) const {
    if (row >= rows) return E::zero();
    const F *p = &a[(row * len + place) * deg];
    if (deg == 1) return E::from(p[0]);
    E x;
    x.c[0] = p[0], x.c[1] = p[1], x.c[2] = p[2];
    return x;
  }
  void set(size_t row, size_t place, const E &x) {
    BLK_REQUIRE(row < rows && place < len, "table: no such entry (rows beyond the real ones are zero and cannot be set)");
    F *p = &a[(row * len + place) * deg];
    if (deg == 1) {
      BLK_REQUIRE(x.c[1] == F::zero() && x.c[2] == F::zero(), "table: an element outside F in a table over F");
      p[0] = x.c[0];
    } else {
      p[0] = x.c[0], p[1] = x.c[1], p[2] = x.c[2];
    }
  }
  // Entry j of the table read as a vector: row = the low log_rows bits of j, place = the rest.
  // at() and entry() do not check that `place` is below `len`.
  E entry(u64 j) const { return at(j & ((u64(1) << log_rows) - 1), j >> log_rows); }
  void set_entry(u64 j, const E &x) { set(j & ((u64(1) << log_rows) - 1), j >> log_rows, x); }
  u64 entries() const { return (u64)len << log_rows; }
};

// The prover's private randomness. The calling program supplies the generator. Every call to
// fill() or bytes() must give fresh, independent, uniform values; nothing here checks that.
class Coins {
public:
  virtual ~Coins() = default;
  virtual void fill(F *out, size_t n) = 0;
  virtual void bytes(u8 *out, size_t n) = 0;
  // last_draw() and not_the_coins() serve not_as_drawn() (blocks/iop/send.h), which counts stored
  // values that no longer equal the random values drawn for them. A generator that can recompute
  // a draw, such as `StreamCoinsT` (blocks/coins/coins.h), overrides both. Any other generator
  // keeps NO_DRAW; not_as_drawn() then compares nothing and returns 0.
  //
  // last_draw(), called right after fill(), returns the number of that draw. After
  // bytes() it may return that call's number instead.
  static constexpr u32 NO_DRAW = ~u32(0);
  virtual u32 last_draw() const { return NO_DRAW; }
  // How many of the n elements of draw number `draw`, as stored at `values`, differ from the
  // values drawn. `left_out`: indices that are not compared, strictly ascending. Draws nothing.
  // The default returns n ("all differ"), so that a generator that overrides only last_draw()
  // shows up as a mismatch.
  virtual u64 not_the_coins(u32 draw, const F *values, u64 n, const std::vector<u64> &left_out) const { return (void)draw, (void)values, (void)left_out, n; }
  F f() {
    F x;
    fill(&x, 1);
    return x;
  }
  E e() {
    E x;
    fill(x.c, 3);
    return x;
  }
};

} // namespace blk
