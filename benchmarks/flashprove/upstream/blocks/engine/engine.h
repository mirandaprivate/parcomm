// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Engine is the interface to a prover's heavy steps: those whose cost grows with the size of the
// main rows (storing and encoding tables, Merkle trees, inner products, folds) and the nonce
// search. The prover functions of blocks/iop/ decide on the host what is sent and in which order,
// do the small arithmetic, and call an Engine for the rest. EngineCpu (blocks/engine/engine_cpu.h)
// is the reference implementation; EngineGpu (prover/engine/engine_gpu.cuh) has the same calls.
//
// An oracle message (defined in blocks/iop/relation.h) has main rows under one base code and mask
// rows under one mask code. A row's random tail is Code::r entries of private randomness (`Coins`)
// encoded with its message. The engine stores main rows in main tables and draws and keeps their
// tails. Mask rows are short and stay on the host: the caller passes in their messages and tails.
//
// Lifetimes: an engine need not copy what it reads. A main table must outlive every tree committed
// over it; a tree must outlive every fold over its rows; a table that a fold's coefficient vector
// refers to must outlive that fold. EngineCpu ends the process on an earlier drop() or close().
// "Ends the process" means a failed BLK_REQUIRE (blocks/base.h): a line on stderr and no return.
#pragma once
#include <utility>
#include <vector>

#include "base.h"
#include "code/code.h"
#include "engine/table.h"
#include "field/field.h"
#include "hash/hash.h"
#include "iop/tensor.h"
#include "tree/opening.h"

namespace blk {

// TableId (blocks/iop/tensor.h): a main table that the engine stores.
using CommitId = u32; // a tree and the codewords under it
using FoldId = u32;   // a fold in progress
constexpr u32 NO_ID = ~u32(0);

// One oracle message as it goes into a tree. Either component may be absent (main == NO_ID;
// mask.rows == 0). F is the field and E its cubic extension (blocks/field/field.h).
struct OraclePart {
  TableId main = NO_ID; // rows of code.ell entries, over F or E
  Code code;
  Table mask;      // rows of mask_code.ell entries, over E
  Table mask_tail; // as many rows of mask_code.r entries, over E: their encoding randomness
  Code mask_code;
};

// The main rows of one oracle message in the table m of fold_begin(): a block of 2^log_rows rows
// (Table::log_rows) from first_row on, zero beyond the real rows (Table::rows).
struct FoldPart {
  CommitId commit = NO_ID;
  unsigned part = 0; // which part of that tree
  u64 first_row = 0; // a multiple of the block's size
};

class Engine {
public:
  virtual ~Engine() = default;

  // ---- main tables ----
  virtual TableId put(const Table &t) = 0;
  virtual Table get(TableId t) = 0; // a copy on the host: for short tables and for tests
  virtual void drop(TableId t) = 0;
  // A new table of 2^log_rows rows with the same entry as t at every index (see entries()).
  virtual TableId reshape(TableId t, unsigned log_rows) = 0;
  // The entries at the indices `at`, in that order. As in Table::entry(), entry c of row b has
  // index b + 2^log_rows c. A row beyond the real ones reads as zero.
  virtual std::vector<E> entries(TableId t, const std::vector<u64> &at) = 0;
  // A new table, whose lifetime does not depend on t's: t with the entries in `changed` replaced.
  // The process ends if an index is in no real row or occurs twice, or if a table over F gets a
  // value outside F. An engine that stores runs of neighbouring rows in cells narrower than a field
  // element may store, in the new table, each entire run that has a changed entry in full words,
  // which costs memory.
  virtual TableId with_entries(TableId t, const std::vector<std::pair<u64, E>> &changed) = 0;
  // The same in place. The process also ends if a tree that is not closed was committed over t, or
  // if a fold that has not ended reads t: what has been committed to does not change. An engine
  // that stores rows in narrow cells ends the process if a changed entry is in such a row.
  virtual void set_entries(TableId t, const std::vector<std::pair<u64, E>> &changed) = 0;
  // Cuts a folded row into rows of the next oracle message. Returns a new table over E of `rows`
  // rows of `len` entries, whose lifetime does not depend on t's. It has the first `count` entries
  // of t (one row over E) in the 2^bits rows from first_row on and is zero elsewhere: entry j of t
  // becomes entry j div 2^bits of row first_row + (j mod 2^bits).
  virtual TableId laid(TableId t, u64 count, u64 first_row, unsigned bits, size_t rows, size_t len) = 0;
  // sum_j v[j] * t[j], with t indexed as in entries(). v is a coefficient vector in structured form
  // (`Tensor`, `Form`: blocks/iop/tensor.h); it has log_rows + log2(len) variables, the bits of j.
  virtual E evaluate(TableId t, const Tensor &v) = 0;
  virtual E evaluate(TableId t, const Form &v) = 0;

  // ---- oracles ----
  // Encodes every row of every part, main rows with tails drawn from `coins`, and builds one
  // perfect binary Merkle tree, each leaf with its own fresh salt. The leaf of position x contains
  // the symbols at x of all main rows of all parts, then the symbols of all mask rows at x modulo
  // their length (short_position()), part by part and row by row; a row over E has three symbols at
  // a position, one for each coordinate. The main rows' codewords have one length in all parts; so
  // have the mask rows'. The caller chooses which oracle messages share a tree; `shape` describes
  // the tree in public terms (lengths, symbols of each part in a leaf, the tree's number), and the
  // process ends if `parts` does not fit it. Order of the draws from `coins`: for each part in turn
  // the tails of its main rows, row by row; then the leaves' salts, by position. Engines that keep
  // this order build equal trees from equal `coins`.
  virtual CommitId commit(const std::vector<OraclePart> &parts, const OracleTreeShape &shape, Coins &coins, Digest &root) = 0;
  // Opens the leaves at the positions `at`: symbols, salts and the digests their Merkle paths need.
  virtual void open(CommitId c, const PositionList &at, Opening &out) = 0;
  // The tails of one part's main rows: code.r entries for each row, over the rows' field.
  virtual Table tail(CommitId c, unsigned part) = 0;
  virtual void close(CommitId c) = 0; // codewords, tree and randomness are no longer needed

  // ---- the fold over the k row variables of a group ----
  // A group (blocks/iop/relation.h) is a set of oracle messages under one base code. m is its main
  // table: 2^k rows, filled according to `parts` and zero where no part covers them. c is a
  // coefficient vector on m, indexed as in entries(). Where c has a hint
  // (`Held`, blocks/iop/tensor.h), the engine may read the hint's table instead of computing the
  // parts of c that the hint covers.
  virtual FoldId fold_begin(const std::vector<FoldPart> &parts, unsigned k, const Form &c) = 0;
  // q[0] + q[1] X + q[2] X^2 is the sum over b of the inner product of c_lo + X (c_hi - c_lo) with
  // m_lo + X (m_hi - m_lo), where lo = 2b and hi = 2b + 1 are rows of the current c and m.
  virtual void fold_round(FoldId f, E q[3]) = 0;
  // For m, c and the tails: the new row b is (1 - gamma) * row 2b + gamma * row (2b + 1).
  virtual void fold_bind(FoldId f, const E &gamma) = 0;
  // Ends the fold, after k calls of fold_bind(). Returns the folded m, a new table of one row over
  // E, and writes the folded tail (code.r entries over E) to `tail`. If c_folded is not null it
  // receives the folded c as another such table, which a later `Form` can refer to in a hint.
  virtual TableId fold_end(FoldId f, Table &tail, TableId *c_folded = nullptr) = 0;

  // ---- work ----
  // Proof of work: the smallest nonce that passes chain_work_passes(prefix, nonce, bits)
  // (blocks/hash/hash.h).
  virtual u64 search_nonce(const ChainPrefix &prefix, unsigned bits) = 0;

  // ---- one thread at a time ----
  // An engine has no lock. At most one thread holds it, at first the thread that constructed it;
  // every call by a thread that does not hold it, except take_over(), ends the process. To pass the
  // engine on, the holder calls let_go(), which waits for the engine's unfinished work; then the
  // next thread calls take_over(), which ends the process if a thread still holds the engine. The
  // threads must order the two calls themselves. `Coins` objects are passed on separately, by the
  // let_go() and take_over() of blocks/coins/coins.h. By default these two calls and the three
  // under "loans" end the process: an engine that wraps another for a test or a measurement need
  // not override them.
  virtual void let_go() { BLK_REQUIRE(false, "engine: this engine does not go from hand to hand"); }
  virtual void take_over() { BLK_REQUIRE(false, "engine: this engine does not go from hand to hand"); }

  // ---- loans ----
  // borrow_table() and borrow_tree() give a table or a tree of another engine of the same
  // implementation an id in this engine, without copying it. The id is for reading: get(),
  // entries(), evaluate(), reshape() and with_entries() (which copy), open(), tail(), and a fold
  // over its rows together with rows of this engine; not set_entries(). With a tree come the main
  // tables of its parts: borrowed_tables() returns their ids in this engine, NO_ID for a part
  // without main rows. close() of a borrowed tree, or drop() of a table that was borrowed by
  // itself, ends the loan and frees nothing; the borrower's destruction ends its loans too. During
  // a loan the process ends on the lender's drop(), set_entries() or close() of what is lent, on a
  // second loan of it, and on the lender's destruction. It also ends if, at the borrowing or at a
  // use, a thread other than the calling one holds the lender.
  virtual TableId borrow_table(Engine &lender, TableId theirs) {
    (void)lender, (void)theirs;
    BLK_REQUIRE(false, "engine: this engine borrows nothing");
    return NO_ID;
  }
  virtual CommitId borrow_tree(Engine &lender, CommitId theirs) {
    (void)lender, (void)theirs;
    BLK_REQUIRE(false, "engine: this engine borrows nothing");
    return NO_ID;
  }
  virtual std::vector<TableId> borrowed_tables(CommitId tree) {
    (void)tree;
    BLK_REQUIRE(false, "engine: this engine borrows nothing");
    return {};
  }
};

} // namespace blk
