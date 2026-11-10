// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Gives out the entries of an oracle message's mask rows so that each is used once.
//
// An oracle message (relation.h) is a set of encoded main rows and mask rows sent in one prover
// message. Its mask rows hold uniform elements of the challenge field (exceptions: see
// `MaskLedger`, and fold_mask_table() in fold_algebra.h). A protocol block that sends a scalar
// which depends on the witness adds one such entry to it. Hiding rests on three rules:
//   1. An entry masks only one sent scalar.
//   2. An entry is used only after its oracle message's out-of-domain sample has been answered.
//   3. An entry is used before the first proximity test on a group containing its oracle message
//      (a group: oracle messages tested together).
// The types here make the straightforward ways to break a rule fail to compile or stop the program.
// Those under "Not caught" at the classes below are not stopped: avoiding them is the caller's
// duty. No honest run reveals such a fault: both sides make the same calls and every linear claim
// (`Row` in relation.h) stays true; what is lost is hiding.
#pragma once
#include <utility>
#include <vector>

#include "base.h"

namespace blk {

// Identifies mask entry number index() of oracle message oracle(). It cannot be copied (rule 1).
// On a moved-from slot, oracle() and index() stop the program. Row::add_mask() (relation.h) uses a
// slot up.
// Not caught: reading a slot's numbers does not use it up, so a prover's code can add one entry to
// two scalars. A linear claim can refer to a mask entry by its number, without a slot. A slot that
// was taken in time can still be used after MaskLedger::close().
class MaskSlot {
 public:
  MaskSlot(MaskSlot&& o) noexcept : oracle_(o.oracle_), index_(o.index_), full_(o.full_) { o.full_ = false; }
  MaskSlot& operator=(MaskSlot&& o) noexcept {
    oracle_ = o.oracle_, index_ = o.index_, full_ = o.full_;
    o.full_ = false;
    return *this;
  }
  MaskSlot(const MaskSlot&) = delete;
  MaskSlot& operator=(const MaskSlot&) = delete;
  u32 oracle() const {
    BLK_REQUIRE(full_, "MaskSlot: this slot has been moved away");
    return oracle_;
  }
  u32 index() const {
    BLK_REQUIRE(full_, "MaskSlot: this slot has been moved away");
    return index_;
  }

 private:
  friend class MaskLedger;
  MaskSlot(u32 oracle, u32 index) : oracle_(oracle), index_(index) {}
  u32 oracle_, index_;
  bool full_ = true;
};

// coeff * (the entry). K is the challenge field.
template <class K>
struct MaskTerm {
  MaskSlot slot;
  K coeff;
};

// States that the sample of oracle message `oracle` has been answered and that `count` entries of
// its mask rows, from entry `first` on, are free for masks. The sample's pads, which mask its
// answers, are not among them. The entry at index c of row b has the number b + 2^log_rows * c, as
// in blocks/iop/tensor.h. If `rows` is not a power of two, numbers with b >= rows belong to no
// entry and are skipped.
// Rule 2: a `MaskLedger` is built only from a receipt, and only `SendAccess` creates receipts.
// SEND, the block that takes the sample, creates one after the answers have been sent or read. A
// receipt cannot be copied, and building the ledger empties it: one receipt, one ledger.
// Not caught: a program that does not include blocks/iop/send.h can define `SendAccess` itself, as
// blocks/iop/tests/toy_support.h does for tests without oracle messages.
struct SendAccess;  // defined in send.h
class SampleReceipt {
 public:
  SampleReceipt(SampleReceipt&& o) noexcept
      : oracle_(o.oracle_), first_(o.first_), count_(o.count_), rows_(o.rows_), log_rows_(o.log_rows_), full_(o.full_) {
    o.full_ = false;
  }
  SampleReceipt(const SampleReceipt&) = delete;
  SampleReceipt& operator=(const SampleReceipt&) = delete;
  u32 oracle() const { return oracle_; }
  u32 first() const { return first_; }
  u32 count() const { return count_; }
  u32 rows() const { return rows_; }
  unsigned log_rows() const { return log_rows_; }

 private:
  friend struct SendAccess;
  SampleReceipt(u32 oracle, u32 first, u32 count, u32 rows = 1, unsigned log_rows = 0)
      : oracle_(oracle), first_(first), count_(count), rows_(rows), log_rows_(log_rows) {}
  friend class MaskLedger;
  u32 oracle_, first_, count_, rows_;
  unsigned log_rows_;
  bool full_ = true;
};

// Gives out the free mask entries of one oracle message, each once, in increasing order of their
// numbers. The order is public: prover and verifier make the same calls. Statement::begin_run() in
// relation.h, which starts a group's proximity test, closes the ledgers of the group's oracle
// messages (rule 3). After close(), take() stops the program and left() is 0.
// Not caught: a test that another `Statement` runs on the same message (sampled_elsewhere()) leaves
// this ledger open.
// Not caught: SEND cannot tell mask rows that hold fresh private randomness from mask rows that
// hold data: CS, the code switch, can store the old mask rows' messages and the old rows' random
// tails (their encoding randomness) there. Either way the ledger covers all entries below the first
// pad. Anyone who sends such a message must close its ledger at once, as switch_admit() in cs.h
// does; otherwise an entry that other claims read could be given out as an unused mask.
class MaskLedger {
 public:
  explicit MaskLedger(SampleReceipt&& r)
      : oracle_(r.oracle()), next_(r.first()), left_(r.count()), rows_(r.rows()), log_rows_(r.log_rows()) {
    BLK_REQUIRE(r.full_, "MaskLedger: this receipt has been moved away or has made its ledger");
    r.full_ = false;
  }
  // Neither copied nor moved: a copy would give out the same entries again, and a ledger moved out
  // of the `Statement` that owns it would miss close().
  MaskLedger(const MaskLedger&) = delete;
  MaskLedger& operator=(const MaskLedger&) = delete;

  MaskSlot take() {
    BLK_REQUIRE(!closed_, "MaskLedger: a proximity test on this oracle message has begun");
    BLK_REQUIRE(left_ > 0, "MaskLedger: the mask rows have no free entry left");
    while ((next_ & ((u32(1) << log_rows_) - 1)) >= rows_) ++next_;
    --left_;
    return MaskSlot(oracle_, next_++);
  }
  std::vector<MaskSlot> take(size_t n) {
    std::vector<MaskSlot> s;
    s.reserve(n);
    for (size_t i = 0; i < n; ++i) s.push_back(take());
    return s;
  }
  u32 left() const { return closed_ ? 0 : left_; }
  void close() { closed_ = true; }

 private:
  u32 oracle_, next_, left_, rows_;
  unsigned log_rows_;
  bool closed_ = false;
};

}  // namespace blk
