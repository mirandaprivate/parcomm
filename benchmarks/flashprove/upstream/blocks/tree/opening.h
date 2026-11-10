// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A Merkle tree over codewords (rows) sent in one prover message: a leaf's content, an opening's
// bytes, the verifier's check, and the CPU prover's tree. blocks/tree/tree.h has the vertices above
// the leaves. The verifier gets the root first and later reads the rows at a list of positions,
// from an opening. On a bad opening, read_opening(), check_opening() and receive_opening() return
// false; BLK_REQUIRE (blocks/base.h) is for the program's own mistakes and ends the process.
//   - Main rows have length n = 2^log_n and are read at a position x. Mask rows, of random values,
//     have length n_zk = 2^log_n_zk <= n and are read at the short position of x, x mod n_zk:
//     short_position() in blocks/code/short_position.h.
//   - The leaf of position x holds the column at x, then a salt, and is opened in full. The column
//     is the symbols at x of all main rows, part by part (OracleTreeShape), row by row, an element
//     of the extension field as its coordinates in order; then, in the same order, all mask rows'
//     symbols at the short position of x. So a mask row's symbol is copied into n / n_zk leaves.
//   - A tree of mask rows only has log_n = log_n_zk: its column at x is their symbols at x.
//   - Nothing here makes the copies of a mask symbol agree: two opened positions over one short
//     position each show their own.
#pragma once
#include <memory>
#include <vector>

#include "base.h"
#include "engine/table.h"
#include "field/field.h"
#include "hash/hash.h"
#include "iop/message.h"
#include "tree/positions.h"
#include "tree/tree.h"

namespace blk {

// The rows of a tree come in parts: oracle messages (each a set of main rows encoded under one code
// and of mask rows under one mask code), or components of them, that are sent together and read at
// the same lists of positions. The caller chooses which parts share a tree and numbers its trees:
// `tree_number` goes into each leaf's hash.
// A verifier must take no field of a shape from a proof, `stored` least of all: it takes `run` from
// its own channel (`VerifierRounds` in blocks/chain/rounds.h, which computes it) and the other
// fields from the protocol's configuration.
struct OracleTreeShape {
  unsigned log_n = 0, log_n_zk = 0;              // a tree of mask rows only: log_n == log_n_zk
  std::vector<size_t> main_symbols, mask_symbols; // elements of F that each part has in a column: of its main rows, of its mask rows
  u32 tree_number = 0;
  size_t salt_bytes = SALT_FLOOR_BYTES; // at least that, a multiple of 8, at most SALT_MOST_BYTES
  // Each leaf has its own salt of this length; even the leaves over one short position share none.
  // The salt is meant to keep an unopened leaf's digest from revealing anything about its symbols.
  // `run`: what the hash of every leaf starts from, to bind the leaves to one run. It is the run's
  // value, Rounds::run(): a hash of the label, parameters' digest and random bytes of the hash
  // chain (one protocol's Fiat-Shamir transcript) that sends the root. Later chains that read the
  // tree keep it.
  // `parameters`: the same for inner vertices: the configuration digest (a hash of the public
  // files), Rounds::configuration().
  // `stored`: the tree outlives every run: it is made once, for a committed table (an "index" in
  // identifiers such as `IndexShape` and index_name()), and read by many runs. Its leaves are
  // hashed under their own `Domain` (a number for each use of the hash: blocks/hash/domains.h) and
  // start from `parameters` too; `run` is not used. index_name() in blocks/iop/foldeq.h puts the
  // root and the numbers that fix this shape into the table's name.
  bool stored = false;
  Digest run{}, parameters{};

  static size_t sum(const std::vector<size_t> &v) {
    size_t s = 0;
    for (size_t x : v) s += x;
    return s;
  }
  size_t main_width() const { return sum(main_symbols); }
  size_t mask_width() const { return sum(mask_symbols); }
  // Whether the tree has main rows, or mask rows, follows from the widths only.
  bool has_main() const { return main_width() > 0; }
  bool has_mask() const { return mask_width() > 0; }
  size_t width() const { return main_width() + mask_width(); } // of a column
  bool valid() const {
    if (main_symbols.size() != mask_symbols.size()) return false; // an entry of each for every part
    if (!has_main() && !has_mask()) return false;
    if (has_mask() && log_n_zk > log_n) return false;
    if (!has_main() && log_n != log_n_zk) return false;
    if (!salt_length_fits(salt_bytes) || width() >> 24) return false; // both go into the leaf tag
    if (parameters == Digest{} || (!stored && run == Digest{})) return false; // a start left unset
    return tree_number < (1u << 16) && tree().valid();
  }
  TreeShape tree() const { return TreeShape::plain(log_n, parameters); }
};

// Hashes a leaf: gives leaf() (blocks/hash/hash.h) the tag, the vertex number 2^log_n + x, the
// column and the salt. The tag is the same for all leaves of a tree: `Domain`, `tree_number`,
// column width, salt length, and the digest that the hash starts from.
struct LeafHash {
  Tag tag;
  unsigned log_n;
  size_t salt_bytes;
  explicit LeafHash(const OracleTreeShape &s)
      : tag(s.stored ? Domain::LEAF_STORED_MAIN : Domain::LEAF_MAIN, s.tree_number, s.stored ? s.parameters : s.run, s.width(), s.salt_bytes), log_n(s.log_n),
        salt_bytes(s.salt_bytes) {}
  Digest operator()(u64 x, const F *column, size_t count, const u8 *salt) const {
    return leaf(tag, tree_index(log_n, x), column, count, salt, salt_bytes);
  }
};
inline Digest oracle_leaf(const OracleTreeShape &s, u64 x, const F *column, size_t count, const u8 *salt) {
  return LeafHash(s)(x, column, count, salt);
}

// An opening at a list of positions. Its bytes, with no length field (write_opening()): each listed
// position's column and salt, then the digests that tree_walk() in tree.h asks for, which are the
// siblings that the opened leaves do not determine.
struct Opening {
  std::vector<F> columns;    // one after the other, in the list's order
  std::vector<u8> salts;
  std::vector<Digest> given; // digests that `tree_walk` cannot compute, in its order
  // The shape's widths, set together with the columns: column() needs them.
  size_t main_width = 0, mask_width = 0;

  const F *column(size_t i) const { return columns.data() + i * (main_width + mask_width); }
  // The two parts of the column of the i-th listed position.
  const F *main_column(size_t i) const { return column(i); }
  const F *mask_column(size_t i) const { return column(i) + main_width; }
};

// How many digests an opening at that list holds. An invalid shape, an empty list or a list that
// does not range over the tree's positions ends the process.
inline size_t opening_given(const OracleTreeShape &s, const PositionList &list) {
  BLK_REQUIRE(s.valid(), "opening: not a shape");
  BLK_REQUIRE(!list.empty() && list.log_n() == s.log_n, "opening: the list does not range over the tree's positions");
  return tree_given_count(s.tree(), list);
}
inline size_t opening_bytes(const OracleTreeShape &s, const PositionList &list) {
  return list.size() * (s.width() * F_BYTES + s.salt_bytes) + opening_given(s, list) * DIGEST_BYTES;
}

inline void write_opening(Writer &w, const OracleTreeShape &s, const PositionList &list, const Opening &o) {
  BLK_REQUIRE(o.given.size() == opening_given(s, list) && o.columns.size() == list.size() * s.width() && o.salts.size() == list.size() * s.salt_bytes,
              "opening: its pieces have not the sizes that shape and list ask for");
  for (size_t i = 0; i < list.size(); i++) {
    for (size_t j = 0; j < s.width(); j++) w.put(o.columns[i * s.width() + j]);
    w.put_bytes(o.salts.data() + i * s.salt_bytes, s.salt_bytes);
  }
  for (const Digest &d : o.given) w.put(d);
}

// False if bytes are missing or a symbol is not in canonical form (a number below the field's
// modulus), and at once if the Reader has failed before. In every case o has the full sizes that
// shape and list ask for, with zeros where nothing was read. Shape and list must fit each other, or
// the process ends (opening_given()).
[[nodiscard]] inline bool read_opening(Reader &r, const OracleTreeShape &s, const PositionList &list, Opening &o) {
  o = Opening();
  o.given.assign(opening_given(s, list), Digest{});
  o.main_width = s.main_width(), o.mask_width = s.mask_width();
  o.columns.assign(list.size() * s.width(), F::zero());
  o.salts.assign(list.size() * s.salt_bytes, 0);
  if (!r.ok()) return false;
  for (size_t i = 0; i < list.size(); i++) {
    for (size_t j = 0; j < s.width(); j++)
      if (!r.get(o.columns[i * s.width() + j])) return false;
    const u8 *q = r.take(s.salt_bytes);
    if (!q) return false;
    std::copy(q, q + s.salt_bytes, o.salts.begin() + i * s.salt_bytes);
  }
  for (Digest &d : o.given)
    if (!r.get(d)) return false;
  return true;
}

// Hashes each opened column with its salt; tree_check() in tree.h recomputes the root and compares.
// It is read_opening(), not this check, that rejects a symbol that is not in canonical form.
// seen, unless null: the vertex values seen under this root so far (`TreeValues`), for a tree
// opened at several lists. A vertex that gets a different value fails the check and sets *clash.
[[nodiscard]] inline bool check_opening(const Digest &root, const OracleTreeShape &s, const PositionList &list, const Opening &o, TreeValues *seen = nullptr,
                                        bool *clash = nullptr) {
  if (clash) *clash = false;
  (void)opening_given(s, list); // shape and list fit each other, or the process ends
  if (o.columns.size() != list.size() * s.width() || o.salts.size() != list.size() * s.salt_bytes) return false;
  const LeafHash leaf(s);
  std::vector<Digest> leaves(list.size());
  for (size_t i = 0; i < list.size(); i++) leaves[i] = leaf(list[i], o.columns.data() + i * s.width(), s.width(), o.salts.data() + i * s.salt_bytes);
  return tree_check(root, s.tree(), list, leaves, o.given, seen, clash);
}

// The verifier's entire step: reads an opening from the channel's proof and checks it. It reports a
// failure to the channel, so the failure lasts even if the caller drops the result. Ch is
// VerifierRounds, Root is SentRoot (blocks/chain/rounds.h). The values seen under this root, in any
// chain of the session (the chains of one verdict), are kept in VerifierSession::record()
// (blocks/chain/session.h): no caller carries them.
// The BLK_REQUIREs compare what the shape repeats (`salt_bytes`, `parameters`, `run`) with the
// channel's plan (`Planned` in blocks/numbers/plan.h: the chain's settings and list of rounds,
// fixed before the run) and configuration and with `sent`. They also require that this chain, or an
// earlier chain of the session that it follows (Binder::follows() in blocks/chain/public.h), drew
// the list and received the root. A failure there is the program's mistake, not a proof's. Three
// are skipped once the channel has rejected the proof, when `sent` may be empty. One of the three
// ends the process on a shape with `stored` set: for a stored tree use read_opening() and
// check_opening().
template <class Ch, class Root>
[[nodiscard]] bool receive_opening(Ch &ch, const Root &sent, const OracleTreeShape &s, const PositionList &list, Opening &o) {
  const Digest &root = sent.digest();
  TreeValues &seen = ch.session().record(root);
  BLK_REQUIRE(s.salt_bytes == ch.plan()->set.salt_bytes, "opening: the shape's salt has not the length that the chain's plan says");
  BLK_REQUIRE(s.parameters == ch.configuration(), "opening: the shape's vertices do not start from the chain's configuration");
  BLK_REQUIRE(!ch.ok() || ch.drew(list), "opening: a list of positions that neither this chain nor one that it follows has drawn");
  BLK_REQUIRE(!ch.ok() || ch.received(sent), "opening: a root that neither this chain nor one that it follows has received");
  BLK_REQUIRE(!ch.ok() || (!s.stored && s.run == sent.run() && s.parameters == sent.configuration()),
              "opening: the shape's hashes do not start from the values of the chain that received the root");
  if (!read_opening(ch.file_only(), s, list, o)) {
    ch.refuse_what_was_read("an opening is cut short or holds a symbol that is not in canonical form");
    return false;
  }
  bool clash = false;
  if (!check_opening(root, s, list, o, &seen, &clash)) { // the record is then as it was before
    ch.refuse(clash ? "a vertex of a tree gets two values under one root" : "an opening does not lead to its root");
    return false;
  }
  return true;
}

// The prover's tree on the CPU. Call order: set_leaf() at every position, finish(), then open().
// The constructor draws the salts from `coins`, the prover's private randomness (`Coins` in
// blocks/engine/table.h). The tree keeps salts and digests; open() fetches the columns again.
class OracleTree {
public:
  OracleTree(const OracleTreeShape &s, Coins &coins) : s_(s), leaf_(s) {
    BLK_REQUIRE(s_.valid(), "oracle tree: not a shape");
    const size_t n = (size_t)1 << s_.log_n;
    salt_.resize(n * s_.salt_bytes); // a salt for each leaf
    coins.bytes(salt_.data(), salt_.size());
    // No verifier can see whether each leaf has a salt of its own, or whether the salts are zeros: a
    // test must look at the prover's own salts and `coins`, as blocks/tests/hiding/salts_test.cpp does.
    leaves_.resize(n);
    set_.assign(n, false);
  }
  // column: the main rows' symbols at x, then the mask rows' at the short position of x.
  void set_leaf(u64 x, const F *column, size_t count) {
    BLK_REQUIRE(!tree_, "oracle tree: a leaf is set after the tree was finished");
    BLK_REQUIRE((x >> s_.log_n) == 0 && count == s_.width(), "oracle tree: not a leaf of this tree");
    leaves_[x] = leaf_(x, column, count, salt_.data() + x * s_.salt_bytes);
    set_[x] = true;
  }
  // Returns the root.
  Digest finish() {
    BLK_REQUIRE(!tree_, "oracle tree: finished twice");
    BLK_REQUIRE(std::find(set_.begin(), set_.end(), false) == set_.end(), "oracle tree: finished before every leaf was set");
    tree_.reset(new Tree(s_.tree(), std::move(leaves_)));
    return tree_->root();
  }
  const OracleTreeShape &shape() const { return s_; }

  // fetch(u64 x, std::vector<F> &column) must give the column set at x: only its size is checked.
  // Where one of the three switches below changes the first opened column, check_opening() must
  // fail: the opened leaves no longer lead to the root.
  template <class Fetch> void open(const PositionList &list, Opening &o, Fetch fetch) const {
    BLK_REQUIRE(tree_ != nullptr, "oracle tree: opened before it is finished");
    (void)opening_given(s_, list);
    o = Opening();
    o.main_width = s_.main_width(), o.mask_width = s_.mask_width();
    std::vector<F> column;
    for (size_t i = 0; i < list.size(); i++) {
      fetch(list[i], column);
      BLK_REQUIRE(column.size() == s_.width(), "oracle tree: a fetched column has not the leaf's size");
      o.columns.insert(o.columns.end(), column.begin(), column.end());
      o.salts.insert(o.salts.end(), salt_.begin() + list[i] * s_.salt_bytes, salt_.begin() + (list[i] + 1) * s_.salt_bytes);
    }
    o.given = tree_->open(list);
  }

private:
  OracleTreeShape s_;
  LeafHash leaf_;
  std::vector<u8> salt_;
  std::vector<bool> set_;
  std::vector<Digest> leaves_;
  std::unique_ptr<Tree> tree_;
};

} // namespace blk
