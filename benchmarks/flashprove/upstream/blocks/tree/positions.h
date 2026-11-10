// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// PositionList: the positions (codeword indices) at which Merkle trees are opened: those of one
// group of oracle messages (encoded rows) read by one proximity test, or a committed table's tree.
//
// The protocol's verifier draws each position uniformly and independently from [0, 2^log_n), so a
// position can be drawn twice. The trees are opened at the set of drawn positions, in increasing
// order: every non-empty list comes from from_draws(), which sorts and removes duplicates. A
// soundness error depends on the number of draws, not on the size of the set; draws() returns them.
//
// A list drawn by PositionsDraw::positions() in blocks/chain/rounds.h carries a mark: a pointer
// that identifies its chain (the Fiat-Shamir hash chain). short_list() and part() copy the mark.
// receive_opening() in blocks/tree/opening.h ends the process unless the verifier's chain, or one
// that it follows (an earlier chain on whose seal it builds), drew the list (Rounds::drew());
// IndexKey::read() (blocks/iop/foldeq.h) opens a committed table's tree without this check. Once
// the proof is rejected (VerifierRounds::ok() is false), positions() returns the unmarked list {0}
// and receive_opening() skips the check. Only this class and its friends can access the mark.
#pragma once
#include <algorithm>
#include <memory>
#include <vector>

#include "base.h"
#include "code/short_position.h"

namespace blk {

class Rounds;
class PositionsDraw;

class PositionList {
public:
  PositionList() = default;

  // draws: at least one, each below 2^log_n. A failed BLK_REQUIRE (blocks/base.h) ends the process.
  static PositionList from_draws(std::vector<u64> draws, unsigned log_n) {
    BLK_REQUIRE(log_n < 64, "positions: the range must be 2^k with k below 64");
    BLK_REQUIRE(!draws.empty(), "positions: a list needs at least one draw");
    PositionList l;
    l.draws_ = draws;
    std::sort(draws.begin(), draws.end());
    draws.erase(std::unique(draws.begin(), draws.end()), draws.end());
    BLK_REQUIRE((draws.back() >> log_n) == 0, "positions: a draw outside the range");
    l.x_ = std::move(draws);
    l.log_n_ = log_n;
    return l;
  }

  // size(), empty(), operator[] and all() refer to the sorted set, not to the draws.
  unsigned log_n() const { return log_n_; }
  size_t size() const { return x_.size(); }
  bool empty() const { return x_.empty(); }
  u64 operator[](size_t i) const { return x_[i]; }
  const std::vector<u64> &all() const { return x_; }
  // In the order drawn, repeats included: for steps that run once per draw.
  const std::vector<u64> &draws() const { return draws_; }
  // Compares the range and the set only: not the draws' order or repeats, and not the mark.
  bool operator==(const PositionList &o) const { return log_n_ == o.log_n_ && x_ == o.x_; }
  bool operator!=(const PositionList &o) const { return !(*this == o); }

  // Where rows of the shorter length 2^log_short are read along with this list: short_position()
  // (blocks/code/short_position.h: x mod 2^log_short) of each position, sorted, without duplicates.
  std::vector<u64> short_positions(unsigned log_short) const {
    BLK_REQUIRE(log_short <= log_n_, "positions: rows longer than the list's range");
    std::vector<u64> y(x_.size());
    for (size_t i = 0; i < x_.size(); i++) y[i] = short_position(log_n_, log_short, x_[i]);
    std::sort(y.begin(), y.end());
    y.erase(std::unique(y.begin(), y.end()), y.end());
    return y;
  }

  // short_positions(log_short) as a PositionList, for a tree over rows of length 2^log_short. Its
  // draws are short_position() of this list's draws, in the same order.
  PositionList short_list(unsigned log_short) const {
    BLK_REQUIRE(log_short <= log_n_, "positions: rows longer than the list's range");
    std::vector<u64> y(draws_.size());
    for (size_t i = 0; i < draws_.size(); i++) y[i] = short_position(log_n_, log_short, draws_[i]);
    PositionList l = from_draws(std::move(y), log_short);
    l.drawn_by_ = drawn_by_;
    return l;
  }
  // Only draws first .. first + count - 1: a verifier's message can hold several lists' draws.
  PositionList part(size_t first, size_t count) const {
    BLK_REQUIRE(count >= 1 && first <= draws_.size() && count <= draws_.size() - first, "positions: a part beyond the list's draws");
    PositionList l = from_draws(std::vector<u64>(draws_.begin() + (std::ptrdiff_t)first, draws_.begin() + (std::ptrdiff_t)(first + count)), log_n_);
    l.drawn_by_ = drawn_by_;
    return l;
  }

  // The index of x in all().
  size_t index_of(u64 x) const { return index_in(x_, x); }
  // The index of y in a sorted vector, such as short_positions() returns; the order is not checked.
  static size_t index_in(const std::vector<u64> &sorted, u64 y) {
    auto it = std::lower_bound(sorted.begin(), sorted.end(), y);
    BLK_REQUIRE(it != sorted.end() && *it == y, "positions: asked for a position that is not in the list");
    return (size_t)(it - sorted.begin());
  }

private:
  friend class Rounds;
  friend class PositionsDraw;
  std::vector<u64> x_, draws_;
  unsigned log_n_ = 0;
  std::shared_ptr<const char> drawn_by_; // the mark of the chain that drew the list; none for a list made by from_draws()
};

} // namespace blk
