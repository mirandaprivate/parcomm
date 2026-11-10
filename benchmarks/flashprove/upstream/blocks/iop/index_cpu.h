// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// IndexCpu keeps a committed table (`Index` in identifiers) entirely in memory. It implements the
// prover's interface to such tables, `IndexHolder` (blocks/iop/foldeq.h). For toy sizes, and as the
// reference for GPU implementations (compared in prover/hash_commitment/tests/hf_lin_side_test.cu).
#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "base.h"
#include "code/code.h"
#include "engine/table.h"
#include "field/field.h"
#include "field/mle.h"
#include "iop/foldeq.h"
#include "tree/opening.h"

namespace blk {

// Stand-in for the configuration digest (the hash of the public files) in runs without them. It is
// part of the table's name(). Hash chains (Fiat-Shamir transcripts) that read the table must use
// the same digest.
inline Digest toy_configuration() {
  Digest d{};
  d.w[0] = 0x00796f74;
  return d;
}

class IndexCpu : public IndexHolder {
 public:
  // rows: s.rows() = 2^s.v rows of s.plain() entries over F. Draws from `coins`, in this order:
  // random tails (the random entries encoded with each row's message), pads (they mask the answers
  // of record(), the committer's answers to out-of-domain samples of each row), the tree's salts.
  IndexCpu(const IndexShape &s, const Table &rows, Coins &coins, const Digest &configuration = toy_configuration())
      : shape_(s) {
    side_.push_back(encode_side(rows, coins));
    commit(coins, configuration);
  }

  const IndexShape &shape() const override { return shape_; }
  std::vector<u8> commitment() const override { return digest_bytes(root_); }
  std::vector<u8> name() const override { return index_name(root_, shape_, tree_->shape().parameters); }
  Digest configuration() const override { return tree_->shape().parameters; }
  const Digest &root() const { return root_; }
  // For tests: entry p of row r's message, pads included.
  F message_entry(u64 r, size_t p) const { return side_[0].message[r * shape_.code.ell + p]; }
  // What a verifier has of this table. This key counts as untested. That is enough for the table's
  // one-time test (the proximity test of the committed word) and for a LINK proof in which this is
  // the further table: the proof ties it to a tested one. A later test on the table, and a LINK
  // proof in which this is the first table, need a `TestedTable` (foldeq.h) from check_key(),
  // check_entry() or, for tests, on_a_tests_word(). Given this key instead, no such run is
  // accepted: foldeq_columns_verify() ends the process, unless the proof was rejected before.
  IndexKey key() { return IndexKey(shape_, tree_->shape().parameters, root_, record()); }

  IndexRecord record() override { return record_at(record_seeds(shape_, name())); }

  // Counts the columns that proofs open, up to the limit hiding_positions(shape_.code), as
  // IndexHolder::begin() (foldeq.h) asks; false: these columns would pass the limit. The count is
  // kept in this object only, not durably. open() and fold() do not check that begin() was called.
  bool begin(u64 columns) override {
    if (columns > hiding_positions(shape_.code) - spent_) return false;
    spent_ += columns;
    return true;
  }
  u64 columns_left() const { return hiding_positions(shape_.code) - spent_; }

  void fold(const E *gamma, std::vector<E> &message, std::vector<E> &tail, BlockSums *blocks) override {
    const Side &s = side_[speaker_];
    std::vector<E> w(shape_.rows());
    eq_table(gamma, shape_.v, w.data());
    message.assign(shape_.code.ell, E::zero());
    tail.assign(shape_.code.r, E::zero());
    for (u64 r = 0; r < shape_.rows(); r++) {
      for (size_t j = 0; j < shape_.code.ell; j++) message[j] += w[r] * s.message[r * shape_.code.ell + j];
      for (size_t j = 0; j < shape_.code.r; j++) tail[j] += w[r] * s.tail[r * shape_.code.r + j];
    }
    if (blocks) block_sums(message, *blocks);
  }

  void open(const PositionList &at, Writer &file, std::vector<F> &columns) override {
    Opening o;
    tree_->open(at, o, [this](u64 x, std::vector<F> &column) { column = column_at(x); });
    write_opening(file, tree_->shape(), at, o);
    columns = std::move(o.columns);  // the table has no mask rows: a column is its main symbols
  }

 private:
  IndexRecord record_at(std::vector<SampleSeed> seed) const {
    IndexRecord rec;
    for (u64 r = 0; r < shape_.rows(); r++)
      for (unsigned i = 0; i < shape_.answers; i++) rec.y.push_back(record_value(shape_, seed[i], i, &side_[0].message[r * shape_.code.ell]));
    return rec;
  }
  struct Side {
    std::vector<F> message, tail, word;  // row by row
  };
  Side encode_side(const Table &rows, Coins &coins) const {
    const Code &c = shape_.code;
    BLK_REQUIRE(!index_fault(shape_), "table: not a shape");
    BLK_REQUIRE(rows.deg == 1 && rows.rows == shape_.rows() && rows.len == shape_.plain(), "table: the rows do not fit the shape");
    Side s;
    s.message.resize(shape_.rows() * c.ell), s.tail.resize(shape_.rows() * c.r), s.word.resize(shape_.rows() * c.n());
    coins.fill(s.tail.data(), s.tail.size());
    const Encoder enc(c);
    for (u64 r = 0; r < shape_.rows(); r++) {
      F *m = &s.message[r * c.ell];
      std::copy(&rows.a[r * rows.len], &rows.a[r * rows.len] + rows.len, m);
      coins.fill(m + shape_.plain(), shape_.pads());
      enc.encode(m, s.tail.data() + r * c.r, &s.word[r * c.n()]);
    }
    return s;
  }
  std::vector<F> column_at(u64 x) const {
    const Side &s = side_[0];
    std::vector<F> column(shape_.rows());
    for (u64 r = 0; r < shape_.rows(); r++) column[r] = s.word[r * shape_.code.n() + x];
    return column;
  }
  void commit(Coins &coins, const Digest &configuration) {
    tree_.reset(new OracleTree(index_tree(shape_, configuration), coins));
    for (u64 x = 0; x < shape_.code.n(); x++) {
      const std::vector<F> column = column_at(x);
      tree_->set_leaf(x, column.data(), column.size());
    }
    root_ = tree_->finish();
  }

  IndexShape shape_;
  std::vector<Side> side_;
  size_t speaker_ = 0;
  std::unique_ptr<OracleTree> tree_;
  Digest root_{};
  u64 spent_ = 0;
};

}  // namespace blk
