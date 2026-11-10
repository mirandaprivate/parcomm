// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A Merkle tree over digests: a perfect binary tree with 2^log_n leaves, opened at a list of
// positions (`PositionList` in blocks/tree/positions.h: distinct leaf numbers, increasing). Tree is
// the prover's side on a CPU, tree_check() the verifier's. What a leaf contains, its salt and its
// hash are in blocks/tree/opening.h. The format is specified in
// docs/format/FORMAT.md, R7.6 to R7.10, R7.14 and R7.16.
//
// The vertex at depth d and place i has the children (d + 1, 2 i) and (d + 1, 2 i + 1). The root
// is (0, 0); the leaf of position x is (log_n, x). An inner vertex's value is the hash of its two
// children's values (vertex() in blocks/hash/hash.h). That hash includes neither the tree's
// number, nor the vertex's place, nor the protocol run that made the tree. What ties a leaf to a
// run, a tree and a position is in the leaf's digest, which the caller computes.
#pragma once
#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "base.h"
#include "hash/hash.h"
#include "tree/positions.h"

namespace blk {

// A tree's depth and the parameters of its inner vertices' hash. `start` enters that hash as
// BLAKE2b's salt and personalization (blocks/hash/blake2b_f.h). By R7.7 of the format it is the
// configuration digest (a hash of the public configuration files), so trees under one
// configuration share it. Only tests set `binding`: blocks/iop/lin.h hashes the binding tree's
// vertices itself, in lin_binding_vertex().
struct TreeShape {
  unsigned log_n = 0; // the depth: 2^log_n leaves
  Digest start{};     // what the hash of every inner vertex starts from; never zeros
  bool binding = false; // if true: the domain number of the binding tree, over a batch's requests

  static TreeShape plain(unsigned log_n, const Digest &start, bool binding = false) { return TreeShape{log_n, start, binding}; }
  // The tag (domain number and start) that vertex() takes. Make it once per tree.
  Tag vertex_tag() const { return Tag(binding ? Domain::BINDING_NODE : Domain::TREE_NODE, 0, start); }
  bool valid() const { return log_n <= 48 && start != Digest{}; }
};

// The vertex (depth, place) as one number. LeafHash in opening.h puts the number of a leaf's
// vertex into the leaf's hash.
inline u64 tree_index(unsigned depth, u64 place) { return ((u64)1 << depth) + place; }

// The record of the vertex values, computed or given (read from a proof), that a verifier has seen
// under one root, across openings. A tree may be opened at several lists. With this record,
// tree_check() returns false if a vertex gets two different values.
class TreeValues {
public:
  // Records a value; a vertex keeps its first. Returns false once any vertex has been offered a
  // second, different value (a clash).
  bool put(unsigned depth, u64 place, const Digest &value) {
    entered_++;
    const auto at = seen_.emplace(tree_index(depth, place), value);
    if (at.second && on_trial_) fresh_.push_back(at.first->first);
    if (!at.second && at.first->second != value) clashed_ = true;
    return !clashed_;
  }
  bool clashed() const { return clashed_; }
  // A check calls begin(), then put(), then keep() if it accepts the opening and otherwise
  // take_back(), which removes the vertices added since begin() and a clash that arose since then.
  // A record may outlive the hash chain (the Fiat-Shamir transcript of one protocol) that reads an
  // opening, so a rejected opening must leave nothing that a later, valid one would clash with.
  void begin() {
    BLK_REQUIRE(!on_trial_, "tree: two checks at once on one record");
    on_trial_ = true, clashed_before_ = clashed_;
    fresh_.clear();
  }
  void keep() { on_trial_ = false, fresh_.clear(); }
  void take_back() {
    for (u64 vertex : fresh_) seen_.erase(vertex);
    clashed_ = clashed_before_;
    keep();
  }
  // For tests: the number of put() calls, those taken back included, and of vertices recorded.
  u64 entered() const { return entered_; }
  size_t vertices() const { return seen_.size(); }

private:
  std::map<u64, Digest> seen_;
  std::vector<u64> fresh_; // the vertices entered since begin()
  u64 entered_ = 0;
  bool clashed_ = false, clashed_before_ = false, on_trial_ = false;
};

// The walk from the opened leaves to the root, which prover and verifier share. It determines
// which digests an opening contains: the values of the siblings that do not follow from the opened
// leaves, level by level from the leaves up, from left to right within a level. The choice and the
// order depend only on the depth and the list.
// Side is the prover's or the verifier's view of the tree's values:
//   typename Side::Value
//   Value leaf(size_t i)                            the opened leaf at the i-th listed position
//   bool  given(unsigned d, u64 place, Value &v)    the next sibling's value: that of (d, place)
//   Value node(unsigned d, u64 place, left, right)  the vertex (d, place) from its children
// Returns false, with root unset, as soon as given() does. Ends the process unless s is valid() and
// list is a non-empty list over 2^log_n positions.
template <class Side> bool tree_walk(const TreeShape &s, const PositionList &list, Side &side, typename Side::Value &root) {
  using Value = typename Side::Value;
  BLK_REQUIRE(s.valid(), "tree: not a shape");
  BLK_REQUIRE(!list.empty() && list.log_n() == s.log_n, "tree: the list does not range over the tree's positions");
  std::vector<std::pair<u64, Value>> now(list.size()), next; // opened paths at one depth, by place
  for (size_t i = 0; i < list.size(); i++) now[i] = {list[i], side.leaf(i)};
  for (unsigned d = s.log_n; d > 0; d--) {
    next.clear();
    for (size_t v = 0; v < now.size(); v++) {
      const u64 place = now[v].first, parent = place >> 1;
      if ((place & 1) == 0 && v + 1 < now.size() && now[v + 1].first == place + 1) {
        next.push_back({parent, side.node(d - 1, parent, now[v].second, now[v + 1].second)});
        v++;
      } else {
        Value other;
        if (!side.given(d, place ^ 1, other)) return false;
        next.push_back({parent, (place & 1) ? side.node(d - 1, parent, other, now[v].second)
                                            : side.node(d - 1, parent, now[v].second, other)});
      }
    }
    now.swap(next);
  }
  BLK_REQUIRE(now.size() == 1, "tree: a fault in tree_walk: it did not end in one vertex");
  root = now[0].second;
  return true;
}

// The number of digests in an opening, not counting the leaves' contents.
inline size_t tree_given_count(const TreeShape &s, const PositionList &list) {
  struct Count {
    struct Value {};
    size_t n = 0;
    Value leaf(size_t) { return {}; }
    bool given(unsigned, u64, Value &) { return ++n, true; }
    Value node(unsigned, u64, const Value &, const Value &) { return {}; }
  } side;
  Count::Value root;
  tree_walk(s, list, side, root);
  return side.n;
}

// The verifier's check of one opening. leaves: the digests of the opened leaves, in the list's
// order. given: the digests from the proof. Returns true if and only if leaves has one digest for
// each listed position, the walk uses exactly the given digests and ends in root, and no vertex
// gets a value other than the one that seen has for it. The check takes no index, count or
// direction from the proof: s and list must be the verifier's own, and meet tree_walk()'s
// conditions. seen, unless null, must be the record of this root, which is not checked; a clash
// already in it makes the result false; it keeps the walk's values if the result is true, and after
// a false result it is as it was before, except for entered(). Without seen, openings are not
// compared with one another. *clash, unless null, reports whether seen holds a clash at the end of
// the walk: a vertex got two values, in this check or before it.
[[nodiscard]] inline bool tree_check(const Digest &root, const TreeShape &s, const PositionList &list, const std::vector<Digest> &leaves,
                                     const std::vector<Digest> &given, TreeValues *seen = nullptr, bool *clash = nullptr) {
  if (clash) *clash = false;
  if (leaves.size() != list.size()) return false;
  struct Check {
    using Value = Digest;
    const TreeShape &shape;
    const Tag tag;
    const PositionList &list;
    const std::vector<Digest> &leaves;
    const std::vector<Digest> &from_proof;
    TreeValues *seen;
    size_t used;
    const Value &note(unsigned d, u64 p, const Value &v) {
      if (seen && d > 0) seen->put(d, p, v); // the computed root is compared with `root` instead
      return v;
    }
    Value leaf(size_t i) { return note(shape.log_n, list[i], leaves[i]); }
    bool given(unsigned d, u64 p, Value &v) {
      if (used == from_proof.size()) return false;
      v = note(d, p, from_proof[used++]);
      return true;
    }
    Value node(unsigned d, u64 p, const Value &l, const Value &r) { return note(d, p, vertex(tag, l, r)); }
  } side{s, s.vertex_tag(), list, leaves, given, seen, 0};
  Digest computed;
  if (seen) seen->begin();
  const bool led = tree_walk(s, list, side, computed) && side.used == given.size() && computed == root;
  const bool clashed = seen && seen->clashed();
  if (clash) *clash = clashed;
  if (seen) led && !clashed ? seen->keep() : seen->take_back();
  return led && !clashed;
}

// The prover's tree, entirely in memory: 2^(log_n + 1) - 1 digests.
class Tree {
public:
  // leaves[x]: the digest of the leaf of position x.
  Tree(const TreeShape &s, std::vector<Digest> leaves) : s_(s) {
    BLK_REQUIRE(s_.valid() && leaves.size() == (size_t)1 << s_.log_n, "tree: not a shape, or not 2^depth leaves");
    const Tag tag = s_.vertex_tag();
    value_.resize(s_.log_n + 1);
    value_[s_.log_n] = std::move(leaves);
    for (unsigned d = s_.log_n; d > 0; d--) {
      std::vector<Digest> &v = value_[d - 1];
      v.resize((size_t)1 << (d - 1));
      for (size_t i = 0; i < v.size(); i++) v[i] = vertex(tag, value_[d][2 * i], value_[d][2 * i + 1]);
    }
  }

  const TreeShape &shape() const { return s_; }
  const Digest &root() const { return value_[0][0]; }

  // The digests that an opening contains, in the walk's order.
  std::vector<Digest> open(const PositionList &list) const {
    struct Fetch {
      struct Value {};
      const Tree &t;
      std::vector<Digest> out;
      Value leaf(size_t) { return {}; }
      bool given(unsigned d, u64 place, Value &) {
        out.push_back(t.value_[d][place]);
        return true;
      }
      Value node(unsigned, u64, const Value &, const Value &) { return {}; }
    } side{*this, {}};
    Fetch::Value root;
    tree_walk(s_, list, side, root);
    return std::move(side.out);
  }

private:
  TreeShape s_;
  std::vector<std::vector<Digest>> value_; // value_[d][place]
};

} // namespace blk
