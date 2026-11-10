// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The running state (`Statement`) that the protocol's blocks (send.h, bat.h, fold.h and so on) pass
// along. Here "statement" means this state, not the glossary's statement (the part of a first query
// that says what is proved). Prover and verifier each keep a Statement and make the same calls on
// it, so the two copies agree at every step. It holds public data only: the oracle messages sent so
// far (encoded rows: OracleShape), the groups in which they are tested for proximity to their code
// (Group), and the open linear claims on them (`Row`; "row" in error messages).
//
// The interface enforces three rules, by its types or by a BLK_REQUIRE, which ends the process:
//   1. An oracle message can join a group, and a claim added to a group can name it, only after its
//      out-of-domain sample (a random linear form of it, taken by the block SEND) is answered.
//   2. A group's proximity test gets the sample claims ("form(message) = answer") of all the
//      group's members together with the open claims (RunInput). That is what makes several tests
//      on one oracle message be about the same message.
//   3. From begin_run() on, a group takes no new claim and no new member, and the mask entries of
//      its members (random elements, each added to one sent value: mask.h) are not given out.
// Not enforced: a caller can fill a GroupRow by hand; require_all_tested() does not see a claim
// that was never added to a group; sampled_elsewhere() leaves three conditions to its caller.
//
// A group of oracle messages follows the definition of a code complex, and a row with a public value the main
// relation, of the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "base.h"
#include "code/code.h"
#include "field/field.h"
#include "iop/mask.h"
#include "iop/tensor.h"

namespace blk {

using OracleId = u32;  // oracle messages are numbered in the order of sending
using GroupId = u32;
using CodeId = u32;
constexpr u32 NONE_ID = ~u32(0);

// The two tables of an oracle message: its main rows and its mask rows. Indices follow tensor.h:
// entry c of row b, in a table of 2^k rows, has index b + 2^k c. So the row bits are the low bits;
// the bits of c are the `place` bits.
enum class Part : u8 { MAIN, MASK };

// All codes of one statement; a CodeId is an index into `base` or into `mask`. Groups that share an
// oracle message with mask rows have the same mask code. The later stages of a proximity test have
// their own codes. Message lengths (`Code::ell`) must be powers of two.
struct Codes {
  std::vector<Code> base;
  std::vector<Code> mask;
};

// An oracle message is a set of main rows under one base code and of mask rows under one mask code,
// sent in one prover message. It is sampled at most once and tested only as a whole. This struct is
// what is public about it before it is sent. F is the field and E its extension of degree 3
// (blocks/field/field.h).
struct OracleShape {
  std::string name;
  CodeId code = NONE_ID;  // its base code, if it has main rows
  size_t main_rows = 0;   // real main rows; either number of rows may be 0, not both
  unsigned main_deg = 1;  // the main rows are all over F (1) or all over E (3)
  size_t mask_rows = 0;   // real mask rows; these are always over E
  CodeId mask_code = 0;   // their code
  unsigned answers = 1;   // how many out-of-domain answers it gets; 0: it is not sampled
};

// ---------------------------------------------------------------------------------------------------------------------
// Linear claims
// ---------------------------------------------------------------------------------------------------------------------

// A structured coefficient vector on one table of one oracle message. The table counts as having
// 2^ceil_log2(rows) rows (those beyond the real ones are zero), so v has ceil_log2(rows) +
// log2(message length) variables; add_row() checks that number.
struct Term {
  OracleId oracle;
  Part part;
  Tensor v;
};

// A coefficient on a single entry of such a table.
struct Unit {
  OracleId oracle;
  Part part;
  u64 index;
  E coeff;
};

// Coefficients, written out, on the entries first, first + step, first + 2 step, ... of a table.
// With step 1 the run follows the table read as a vector; with step 2^(row bits) it follows one
// row. The coefficients are either the run's own, over E, or a range of a shared vector over F that
// no one changes. Example: the GENERATOR claims (RowKind) that cs.h makes at one position, on rows
// under the base code, have the same coefficients on the rows' random tails (the random entries
// encoded with each row). Both forms are private and read only through size() and at(), so that no
// reader overlooks one of them.
class Run {
 public:
  OracleId oracle;
  Part part;
  u64 first, step;
  Run(OracleId o, Part p, u64 first_, u64 step_, std::vector<E> own) : oracle(o), part(p), first(first_), step(step_), own_(std::move(own)) {}
  Run(OracleId o, Part p, u64 first_, u64 step_, std::shared_ptr<const std::vector<F>> of_many, size_t from, size_t count)
      : oracle(o), part(p), first(first_), step(step_), of_many_(std::move(of_many)), from_(from), count_(count) {
    BLK_REQUIRE(of_many_ && from_ <= of_many_->size() && count_ <= of_many_->size() - from_, "run: the stretch does not lie in its vector");
  }
  size_t size() const { return of_many_ ? count_ : own_.size(); }
  E at(size_t j) const { return of_many_ ? E::from((*of_many_)[from_ + j]) : own_[j]; }
  // True if both runs cover the same entries of the same table.
  bool lies_as(const Run &o) const { return oracle == o.oracle && part == o.part && first == o.first && step == o.step && size() == o.size(); }
  // x * r, as a run that owns its coefficients.
  static Run scaled(const Run &r, const E &x) {
    Run s(r.oracle, r.part, r.first, r.step, std::vector<E>(r.size(), E::zero()));
    s.add_scaled(r, x);
    return s;
  }
  // this += x * other, entry by entry. Afterwards this run owns its coefficients.
  void add_scaled(const Run &other, const E &x) {
    BLK_REQUIRE(&other != this && lies_as(other), "run: a run is added to itself, or the two runs lie on different entries");
    if (of_many_) {  // a shared vector is not written to: copy it first
      own_.resize(count_);
      for (size_t j = 0; j < count_; j++) own_[j] = E::from((*of_many_)[from_ + j]);
      of_many_.reset(), from_ = count_ = 0;
    }
    if (other.of_many_) {
      const F *c = other.of_many_->data() + other.from_;
      for (size_t j = 0; j < own_.size(); j++) add_product(own_[j], x, c[j]);
    } else {
      for (size_t j = 0; j < own_.size(); j++) {
        const E &c = other.own_[j];
        if (lies_in_F(c))
          add_product(own_[j], x, c.c[0]);
        else
          own_[j] += x * c;
      }
    }
  }

 private:
  std::vector<E> own_;
  std::shared_ptr<const std::vector<F>> of_many_;
  size_t from_ = 0, count_ = 0;
};

// A whole coefficient vector on the main table of an oracle message. A Form (blocks/iop/tensor.h)
// can hold what Terms and Units cannot: shared frozen parts and, on the prover's side, hints and
// terms that only the compute engine reads (blocks/engine/engine.h: the prover's heavy steps).
struct MainForm {
  OracleId oracle;
  Form c;
};

// A claim's class in the fixed order in which BAT (bat.h) batches a group's claims: see RunInput.
// SAMPLE: SEND's "form(message) = answer". GENERATOR: "the codeword of this message has this symbol
// at an opened position"; its coefficients are the code's generator row at that position.
enum class RowKind : u8 { PLAIN, SAMPLE, GENERATOR };

// A linear claim: (sum of the terms, units, runs and forms) applied to the messages that the oracle
// messages encode = target. The target is always public.
struct Row {
  std::string name;
  RowKind kind = RowKind::PLAIN;
  std::vector<Term> terms;
  std::vector<Unit> units;
  std::vector<Run> runs;
  std::vector<MainForm> forms;
  E target = E::zero();

  void add(OracleId o, Part part, Tensor v) { terms.push_back(Term{o, part, std::move(v)}); }
  void add_form(OracleId o, Form c) { forms.push_back(MainForm{o, std::move(c)}); }
  void add_run(OracleId o, Part part, u64 first, u64 step, std::vector<E> coeff) {
    runs.push_back(Run(o, part, first, step, std::move(coeff)));
  }
  void add_run(OracleId o, Part part, u64 first, u64 step, std::shared_ptr<const std::vector<F>> of_many, size_t from, size_t count) {
    runs.push_back(Run(o, part, first, step, std::move(of_many), from, count));
  }
  void add_unit(OracleId o, Part part, u64 index, const E &coeff) { units.push_back(Unit{o, part, index, coeff}); }
  // Uses up a mask entry: the call empties `slot`, and a `MaskLedger` gives each entry out once.
  // Not enforced: add_unit() and add_run() can name any entry of a mask table without a `MaskSlot`.
  // They are for claims that carry an entry on or read it again; code that masks a value before
  // sending it takes a `MaskSlot`.
  void add_mask(MaskSlot &&slot, const E &coeff) {
    const MaskSlot s = std::move(slot);
    add_unit(s.oracle(), Part::MASK, s.index(), coeff);
  }
  void add_mask(MaskTerm<E> &&t) { add_mask(std::move(t.slot), t.coeff); }

  // The claim's total coefficient on one entry.
  E coefficient(OracleId o, Part part, u64 index) const {
    E sum = E::zero();
    for (const Term &t : terms)
      if (t.oracle == o && t.part == part) sum += t.v.at(index);
    for (const Unit &u : units)
      if (u.oracle == o && u.part == part && u.index == index) sum += u.coeff;
    for (const Run &r : runs)
      if (r.oracle == o && r.part == part && index >= r.first && (index - r.first) % r.step == 0 &&
          (index - r.first) / r.step < r.size())
        sum += r.at((index - r.first) / r.step);
    for (const MainForm &f : forms)
      if (f.oracle == o && part == Part::MAIN) sum += f.c.at(index);
    return sum;
  }

  // this += x * (the runs of other, and nothing else of it). A run of `other` that covers the same
  // entries as a run of this claim is added into that run (why: see RowsAside).
  void add_runs_scaled(const Row &other, const E &x) {
    for (const Run &r : other.runs) {
      Run *same = nullptr;
      for (Run &mine : runs)
        if (mine.lies_as(r)) {
          same = &mine;
          break;
        }
      if (same)
        same->add_scaled(r, x);
      else
        runs.push_back(Run::scaled(r, x));
    }
  }
  // this += x * other (coefficients and target).
  void add_scaled(const Row &other, const E &x) {
    for (Term t : other.terms) {
      t.v.scalar = t.v.scalar * x;
      terms.push_back(std::move(t));
    }
    for (Unit u : other.units) {
      u.coeff = u.coeff * x;
      units.push_back(u);
    }
    add_runs_scaled(other, x);
    for (MainForm f : other.forms) {
      f.c.scale(x);
      forms.push_back(std::move(f));
    }
    target = target + x * other.target;
  }
};

// The coefficients of a group's claim on the main rows of one kept member (see GroupLayout), on
// that member's own main table.
struct KeptForm {
  OracleId oracle = NONE_ID;
  Form c;
};
// One claim on a whole group, as a proximity test takes it. c is the coefficient vector on the
// group's main table (2^k rows); d is written out, on the group's mask table (2^h rows), which is
// short. Code that uses such a claim must read all three parts: c, kept and d.
struct GroupRow {
  GroupId group = NONE_ID;
  Form c;
  std::vector<KeptForm> kept;  // one per kept member that the claim reads, in the order of sending
  std::vector<E> d;
  E target = E::zero();
};

// Where each member's rows are in a group's main table (2^k rows) and mask table (2^h rows). A
// member with K real rows takes an aligned block of 2^ceil_log2(K) rows. Blocks are placed from row
// 0 by decreasing size, ties in the order of sending; the remaining rows are zero. h is the
// smallest number for which the mask blocks fit, k the smallest with k >= max(h, 1) for which the
// main blocks fit.
// A member may be kept: its main rows are read at the group's positions and tested with the group,
// but they are not folded with the other rows, because a later part of the protocol reads them
// again. They have no block among the 2^k rows, and the member's `main_first` is NONE.
// log_ell, log_ell_zk: log2 of the message length of the base code and of the mask code.
struct GroupLayout {
  unsigned k = 0, h = 0;
  unsigned log_ell = 0, log_ell_zk = 0;
  std::vector<OracleId> members;           // in the order of sending
  std::vector<bool> kept;                  // for each member
  std::vector<u64> main_first, mask_first;  // first row of each member's block; NONE if it has none
  static constexpr u64 NONE = ~u64(0);

  size_t find(OracleId o) const {
    for (size_t i = 0; i < members.size(); i++)
      if (members[i] == o) return i;
    BLK_REQUIRE(false, "group: this oracle message is not a member");
    return 0;
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// The statement
// ---------------------------------------------------------------------------------------------------------------------

// What a Statement keeps about one oracle message. sampled() and sampled_elsewhere() set `sampled`.
struct Oracle {
  OracleShape shape;
  bool sampled = false;
  std::vector<Row> sample_rows;        // the sample claims, one per answer; empty until sampled
  std::vector<std::pair<Part, u64>> pad;  // entries masking the answers; no other claim reads them
  std::vector<GroupId> groups;         // the groups that it belongs to
  std::unique_ptr<MaskLedger> ledger;  // its free mask entries, if sampled here and with mask rows
  bool spent_as_mask = false;          // `spend_as_mask` was called: it masks a fold or a last step
};

// A group is a set of whole oracle messages with one base code, which one proximity test reads at
// one list of positions. "Run" in `in_run`, begin_run() and RunInput means that test; the class
// `Run` is something else, and so is the glossary's run.
struct Group {
  std::string name;
  CodeId code = NONE_ID;
  CodeId mask_code = 0;
  std::vector<OracleId> members;  // in the order of sending
  std::vector<bool> kept;         // for each member: are its main rows kept (GroupLayout)?
  std::vector<Row> rows;          // open claims, in the order of creation
  bool in_run = false;
};

// What begin_run() gives to BAT: the group's layout and its claims, in BAT's fixed order: the PLAIN
// claims as created; then the sample claims, member by member in the order of sending; then the
// GENERATOR claims as created. Only a Statement makes one, and a caller can read its claims but not
// change them; `layout` is not protected in this way. (`RunInputCheat` is defined only in a test,
// blocks/iop/tests/cs_test.cpp.)
class Bat;
class RunInput {
 public:
  GroupLayout layout;
  const std::vector<Row> &rows() const { return rows_; }

 private:
  friend class Statement;
  friend class Bat;  // `Bat::combine` batches the claims once and moves them out: `take_rows`
  friend struct RunInputCheat;
  RunInput() = default;
  std::vector<Row> take_rows() { return std::move(rows_); }
  std::vector<Row> rows_;
};

class Statement {
 public:
  explicit Statement(Codes codes) : codes_(std::move(codes)) {
    for (const Code &c : codes_.base) BLK_REQUIRE(is_power_of_two(c.ell), "statement: a message length is no power of two");
    for (const Code &c : codes_.mask) BLK_REQUIRE(is_power_of_two(c.ell), "statement: a message length is no power of two");
  }
  Statement(Statement &&) = default;
  Statement &operator=(Statement &&) = default;

  // Marks a verifier's statement. A verifier has no compute engine, so add_row() then ends the
  // process if the claim has a part that only an engine can read (tensor.h: EngineTerm,
  // Tensor::held). VerifierContext's constructor sets the mark; nothing clears it, and a move into
  // the context of a later hash chain (a chain is one Fiat-Shamir transcript) keeps it.
  void has_no_engine() { no_engine_ = true; }
  const Codes &codes() const { return codes_; }
  size_t oracles() const { return oracles_.size(); }
  size_t groups() const { return groups_.size(); }
  const Oracle &oracle(OracleId o) const {
    BLK_REQUIRE(o < oracles_.size(), "statement: no such oracle message");
    return oracles_[o];
  }
  const Group &group(GroupId g) const {
    BLK_REQUIRE(g < groups_.size(), "statement: no such group");
    return groups_[g];
  }

  // Numbers of row bits, of place bits and of all index bits of a table of o; 0 for a table that o
  // does not have.
  unsigned row_bits(OracleId o, Part part) const {
    const OracleShape &s = oracle(o).shape;
    return ceil_log2(part == Part::MAIN ? s.main_rows : s.mask_rows);
  }
  unsigned place_bits(OracleId o, Part part) const {
    const OracleShape &s = oracle(o).shape;
    if (!has(o, part)) return 0;
    return part == Part::MAIN ? codes_.base[s.code].log_ell() : codes_.mask[s.mask_code].log_ell();
  }
  unsigned table_bits(OracleId o, Part part) const { return row_bits(o, part) + place_bits(o, part); }
  bool has(OracleId o, Part part) const {
    const OracleShape &s = oracle(o).shape;
    return (part == Part::MAIN ? s.main_rows : s.mask_rows) > 0;
  }

  GroupId new_group(std::string name, CodeId code, CodeId mask_code = 0) {
    BLK_REQUIRE(code < codes_.base.size() && mask_code < codes_.mask.size(), "statement: no such code");
    BLK_REQUIRE(mask_fits(codes_.base[code], codes_.mask[mask_code]), "statement: the mask code is longer than the group's");
    groups_.push_back(Group{std::move(name), code, mask_code, {}, {}, {}, false});
    return (GroupId)(groups_.size() - 1);
  }

  // Records that an oracle message has been sent. Until it is sampled no group takes it, so no
  // claim on a group can name it. (A mask for spend_as_mask() stays in this state.)
  OracleId sent(OracleShape shape) {
    BLK_REQUIRE(shape.main_rows + shape.mask_rows > 0, "statement: an oracle message without rows");
    BLK_REQUIRE(shape.main_rows == 0 || shape.code < codes_.base.size(), "statement: no such base code");
    BLK_REQUIRE(shape.mask_rows == 0 || shape.mask_code < codes_.mask.size(), "statement: no such mask code");
    BLK_REQUIRE(shape.main_deg == 1 || shape.main_deg == 3, "statement: main rows are over F or over E");
    oracles_.push_back(Oracle{std::move(shape), false, {}, {}, {}, nullptr, false});
    return (OracleId)(oracles_.size() - 1);
  }

  // Marks an oracle message that is never sampled (answers == 0) as the mask of one fold, or of the
  // last step of one proximity test. It can serve only once.
  void spend_as_mask(OracleId o) {
    Oracle &x = at(o);
    BLK_REQUIRE(!x.sampled && x.shape.answers == 0, "statement: a masking oracle is never sampled");
    BLK_REQUIRE(!x.spent_as_mask, "statement: a masking oracle serves one block only");
    x.spent_as_mask = true;
  }

  // Records that the answers to o's sample are in the prover's message. In a program that includes
  // send.h only SEND can call this: `SendAccess` is a pass-key, a type that only the class `Send`
  // can construct. A program without send.h can define its own `SendAccess`, as a test header does.
  void sampled(const SendAccess &, OracleId o, std::vector<Row> sample_rows, std::vector<std::pair<Part, u64>> pad,
               SampleReceipt &&receipt) {
    Oracle &x = at(o);
    BLK_REQUIRE(!x.sampled, "statement: an oracle message is sampled once");
    BLK_REQUIRE(sample_rows.size() == x.shape.answers && x.shape.answers >= 1, "statement: one sample row for each answer");
    BLK_REQUIRE(receipt.oracle() == o, "statement: the receipt is another oracle message's");
    x.sampled = true;
    x.sample_rows = std::move(sample_rows);
    x.pad = std::move(pad);
    for (Row &r : x.sample_rows) r.kind = RowKind::SAMPLE;
    if (x.shape.mask_rows > 0) x.ledger = std::make_unique<MaskLedger>(std::move(receipt));
  }

  // Declares that oracle message `mine` of this statement is oracle message `theirs` of the
  // statement `from`, where SEND has sampled it. One oracle message can be tested in several
  // statements, each with its own hash chain: the block LIN's Omega in 1 + N of them for a batch of
  // N requests (lin.h). Every test includes the one sample.
  // The sample claims are copied; this statement gives out no mask entries of the message.
  // Not checked here; the caller must see to it:
  //   - that the two are one oracle message, with the same tree and root: only shapes and codes are
  //     compared;
  //   - for soundness: that its root, its sample's seed and its answers are fixed before the first
  //     challenge of this statement's present hash chain, that is, by that chain's first query (the
  //     hash input made before the prover's first message). A Statement knows nothing of chains;
  //   - for hiding: a test that begins here does not close the `MaskLedger` in the other statement;
  //     and the positions at which the tests of all these statements open the oracle message have
  //     to be added up. Nothing here does that: two provers' contexts (context.h) share one count
  //     only if the caller also passes the tree on with TreeBook::adopt().
  void sampled_elsewhere(OracleId mine, const Statement &from, OracleId theirs) {
    BLK_REQUIRE(&from != this, "statement: the other statement is this one");
    Oracle &x = at(mine);
    const Oracle &y = from.oracle(theirs);
    const OracleShape &a = x.shape, &b = y.shape;
    BLK_REQUIRE(y.sampled, "statement: the oracle message has not been sampled in the other statement");
    BLK_REQUIRE(!x.sampled, "statement: an oracle message is sampled once");
    BLK_REQUIRE(a.main_rows == b.main_rows && a.main_deg == b.main_deg && a.mask_rows == b.mask_rows && a.answers == b.answers,
                "statement: the two oracle messages differ in shape");
    BLK_REQUIRE(!a.main_rows || codes_.base[a.code] == from.codes_.base[b.code], "statement: the two oracle messages differ in code");
    BLK_REQUIRE(!a.mask_rows || codes_.mask[a.mask_code] == from.codes_.mask[b.mask_code],
                "statement: the two oracle messages differ in mask code");
    x.sampled = true;
    x.sample_rows = y.sample_rows;
    x.pad = y.pad;
    for (Row &r : x.sample_rows) {
      for (Term &t : r.terms) t.oracle = mine;
      for (Unit &u : r.units) u.oracle = mine;
      for (Run &w : r.runs) w.oracle = mine;
      for (MainForm &f : r.forms) f.oracle = mine;
    }
  }

  // Makes o, as a whole, a member of g. kept: see GroupLayout. A message without main rows may join
  // a group of any base code.
  void join(GroupId g, OracleId o, bool kept = false) {
    Group &G = grp(g);
    Oracle &x = at(o);
    BLK_REQUIRE(x.sampled, "statement: only a sampled oracle message can join a group");
    BLK_REQUIRE(!G.in_run, "statement: the proximity test on this group has begun");
    BLK_REQUIRE(x.shape.main_rows == 0 || x.shape.code == G.code, "statement: a group has one base code");
    BLK_REQUIRE(x.shape.mask_rows == 0 || x.shape.mask_code == G.mask_code, "statement: a group has one mask code");
    BLK_REQUIRE(std::find(G.members.begin(), G.members.end(), o) == G.members.end(), "statement: already a member");
    BLK_REQUIRE(!kept || x.shape.main_rows > 0, "statement: only main rows can be kept");
    const size_t at_place = (size_t)(std::upper_bound(G.members.begin(), G.members.end(), o) - G.members.begin());
    G.members.insert(G.members.begin() + at_place, o);
    G.kept.insert(G.kept.begin() + at_place, kept);
    x.groups.push_back(g);
  }

  // o's `MaskLedger`, which gives out each free mask entry once, and none after begin_run() on a
  // group of o (rule 3 at the top of this file). Ends the process if o has none (Oracle::ledger).
  MaskLedger &masks(OracleId o) {
    Oracle &x = at(o);
    BLK_REQUIRE(x.ledger != nullptr, "statement: this oracle message has no mask entries to give");
    return *x.ledger;
  }

  // Adds an open claim to g. It may name members of g only, and it must have coefficient 0 on every
  // pad (Oracle::pad): a claim has a public target, and a pad that a second public value depends on
  // no longer hides the answer that it masks.
  void add_row(GroupId g, Row row) {
    Group &G = grp(g);
    BLK_REQUIRE(!G.in_run, "statement: the rows of a proximity test are fixed before its first challenge");
    BLK_REQUIRE(row.kind != RowKind::SAMPLE, "statement: sample rows come from SEND alone");
    if (no_engine_) {  // ends the process here, where the fault enters, not later inside a fold
      for (const Term &t : row.terms) BLK_REQUIRE(!t.v.is_held(), "statement: a verifier's row has a part that only a prover's engine holds");
      for (const MainForm &f : row.forms)
        BLK_REQUIRE(f.c.engine_terms.empty() && !f.c.has_held_factor(), "statement: a verifier's row has a part that only a prover's engine holds");
    }
    for (const Term &t : row.terms) check_place(G, t.oracle, t.part, t.v.vars(), 0);
    for (const Unit &u : row.units) check_place(G, u.oracle, u.part, table_bits(u.oracle, u.part), u.index);
    for (const Run &r : row.runs) {
      BLK_REQUIRE(r.size() != 0 && r.step >= 1, "statement: a run of coefficients is empty or has step 0");
      const u64 entries = u64(1) << table_bits(r.oracle, r.part);  // compared so that nothing can wrap around
      BLK_REQUIRE(r.first < entries && (r.size() - 1) <= (entries - 1 - r.first) / r.step, "statement: no such entry");
      check_place(G, r.oracle, r.part, table_bits(r.oracle, r.part), r.first + r.step * (r.size() - 1));
    }
    for (const MainForm &f : row.forms) {
      check_place(G, f.oracle, Part::MAIN, table_bits(f.oracle, Part::MAIN), 0);
      BLK_REQUIRE(f.c.fits(table_bits(f.oracle, Part::MAIN)), "statement: a part of a coefficient vector does not fit its table");
    }
    for (OracleId o : G.members)
      for (const std::pair<Part, u64> &p : oracle(o).pad)
        BLK_REQUIRE(is_zero(row.coefficient(o, p.first, p.second)), "statement: a row reads the pad of a sample");
    G.rows.push_back(std::move(row));
  }

  // The layout of g with its present members. It can change until begin_run() is called.
  GroupLayout layout(GroupId g) const {
    const Group &G = group(g);
    GroupLayout L;
    L.members = G.members;
    L.kept = G.kept;
    L.log_ell = codes_.base[G.code].log_ell();
    L.log_ell_zk = codes_.mask[G.mask_code].log_ell();
    u64 mask_used = lay(G, Part::MASK, L.mask_first);
    u64 main_used = lay(G, Part::MAIN, L.main_first);
    L.h = ceil_log2(mask_used);
    L.k = std::max(std::max(ceil_log2(main_used), L.h), 1u);
    return L;
  }

  // Begins the one proximity test on g (rules 2 and 3 at the top of this file). Mask entries of its
  // members that are still free are lost.
  RunInput begin_run(GroupId g) {
    Group &G = grp(g);
    BLK_REQUIRE(!G.in_run, "statement: one proximity test for each group");
    BLK_REQUIRE(!G.members.empty(), "statement: a group without members");
    G.in_run = true;
    RunInput in;
    in.layout = layout(g);
    for (Row &r : G.rows)
      if (r.kind == RowKind::PLAIN) in.rows_.push_back(std::move(r));  // G.rows is cleared below
    for (OracleId o : G.members) {
      Oracle &x = at(o);
      for (const Row &r : x.sample_rows) in.rows_.push_back(r);
      if (x.ledger) x.ledger->close();
    }
    for (Row &r : G.rows)
      if (r.kind == RowKind::GENERATOR) in.rows_.push_back(std::move(r));
    G.rows.clear();
    return in;
  }

  // Requires that every oracle message that the claim names has been sampled: what the claim says
  // is then about definite messages.
  void require_sampled(const Row &row) const {
    auto check = [&](OracleId o) { BLK_REQUIRE(oracle(o).sampled, "statement: a row names an oracle message that has not been sampled"); };
    for (const Term &t : row.terms) check(t.oracle);
    for (const Unit &u : row.units) check(u.oracle);
    for (const Run &r : row.runs) check(r.oracle);
    for (const MainForm &f : row.forms) check(f.oracle);
  }

  // For the end of a protocol. Requires that begin_run() has been called on every group and that
  // every sampled oracle message is in a group: an open claim or a sampled oracle message that no
  // proximity test includes would never be checked. Not seen: a claim that was never added to a
  // group (one that waits to be batched or split), an oracle message that was never sampled, and
  // whether a test that has begun was carried through.
  void require_all_tested() const {
    for (const Group &G : groups_) BLK_REQUIRE(G.in_run, "statement: the protocol ends with a group that has not been tested");
    for (const Oracle &x : oracles_)
      BLK_REQUIRE(!x.sampled || !x.groups.empty(), "statement: the protocol ends with an oracle message that is in no group");
  }

 private:
  static bool is_power_of_two(u64 x) { return x != 0 && (x & (x - 1)) == 0; }
  Oracle &at(OracleId o) {
    BLK_REQUIRE(o < oracles_.size(), "statement: no such oracle message");
    return oracles_[o];
  }
  Group &grp(GroupId g) {
    BLK_REQUIRE(g < groups_.size(), "statement: no such group");
    return groups_[g];
  }
  void check_place(const Group &G, OracleId o, Part part, unsigned vars, u64 index) const {
    BLK_REQUIRE(std::find(G.members.begin(), G.members.end(), o) != G.members.end(),
                "statement: a row on a group names members of that group only");
    BLK_REQUIRE(has(o, part), "statement: a row names a table that the oracle message does not have");
    BLK_REQUIRE(vars == table_bits(o, part), "statement: a term does not fit its table");
    BLK_REQUIRE(index >> table_bits(o, part) == 0, "statement: no such entry");
  }
  // Assigns the blocks of the members' main rows or mask rows (GroupLayout); returns the number of
  // rows used.
  u64 lay(const Group &G, Part part, std::vector<u64> &first) const {
    first.assign(G.members.size(), GroupLayout::NONE);
    std::vector<size_t> order;
    for (size_t i = 0; i < G.members.size(); i++)
      if (has(G.members[i], part) && !(part == Part::MAIN && G.kept[i])) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return row_bits(G.members[a], part) > row_bits(G.members[b], part);
    });
    u64 next = 0;
    for (size_t i : order) {
      first[i] = next;  // aligned, because the blocks before it are at least as large
      next += u64(1) << row_bits(G.members[i], part);
    }
    return next;
  }

  Codes codes_;
  std::vector<Oracle> oracles_;
  std::vector<Group> groups_;
  bool no_engine_ = false;
};

// ---------------------------------------------------------------------------------------------------------------------
// From claims on oracle messages to a claim on the group
// ---------------------------------------------------------------------------------------------------------------------

// The index, in a group's table with 2^group_row_bits rows, of entry `index` of a member's table
// with 2^row_bits rows whose block begins at row `first`.
inline u64 group_index(u64 index, unsigned row_bits, u64 first, unsigned group_row_bits) {
  const u64 row = index & ((u64(1) << row_bits) - 1), place = index >> row_bits;
  return (first + row) + (place << group_row_bits);
}

// Accumulator for a batching of many claims: their runs, summed by the entries that they cover, and
// their units, summed by entry, to be carried to the group's tables once at the end. Claims of one
// origin (GENERATOR claims at many positions) name the same entries again and again, so thousands
// of claims leave a few runs and a few units.
struct RowsAside {
  Row sums;  // runs and units only
  void add_scaled(const Row &row, const E &x) {
    sums.add_runs_scaled(row, x);
    for (const Unit &u : row.units) {
      const auto at = unit_at_.try_emplace(std::make_tuple(u.oracle, u.part, u.index), sums.units.size());  // allocates nothing if the entry is already there
      if (at.second)
        sums.units.push_back(Unit{u.oracle, u.part, u.index, times(x, u.coeff)});
      else
        sums.units[at.first->second].coeff += times(x, u.coeff);
    }
  }
  std::vector<E> room;  // work space for Tensor::add_to, reused from claim to claim

 private:
  std::map<std::tuple<OracleId, Part, u64>, size_t> unit_at_;  // an entry's index in sums.units
};

// acc += x * row, with every coefficient vector carried from its member's table to the group's.
// row comes by value, so that its tensors and forms can move into acc.
// aside, if given: the claim's runs and units are not carried now but added, times x, to
// aside->sums. A caller that adds up many claims passes the same object each time and carries
// aside->sums once at the end (with x = 1 and no aside).
inline void add_to_group_row(const Statement &S, const GroupLayout &L, Row row, const E &x, GroupRow &acc, RowsAside *aside = nullptr) {
  if (acc.d.empty()) acc.d.assign(size_t(1) << (L.h + L.log_ell_zk), E::zero());
  const bool plain = x == E::one();
  auto scaled = [&](const E &c) { return plain ? c : times(x, c); };
  std::vector<E> own_room;
  std::vector<E> &room = aside ? aside->room : own_room;
  auto kept_form = [&](OracleId o) -> KeptForm & {
    size_t at = 0;
    while (at < acc.kept.size() && acc.kept[at].oracle < o) at++;
    if (at == acc.kept.size() || acc.kept[at].oracle != o) acc.kept.insert(acc.kept.begin() + at, KeptForm{o, {}});
    return acc.kept[at];
  };
  auto carry = [&](OracleId o, Part part, Tensor v) {
    const size_t i = L.find(o);
    if (part == Part::MAIN && L.kept[i]) {
      v.scalar = v.scalar * x;
      kept_form(o).c.add(std::move(v));
      return;
    }
    const unsigned a = S.row_bits(o, part);
    const u64 first = part == Part::MAIN ? L.main_first[i] : L.mask_first[i];
    const unsigned bits = part == Part::MAIN ? L.k : L.h;
    v.scalar = v.scalar * x;
    v.insert_choice(a, bits - a, first >> a);  // the bits above the block pick the block
    if (part == Part::MAIN)
      acc.c.add(std::move(v));
    else
      v.add_to(acc.d.data(), room);
  };
  // One written-out coefficient, on entry "index" of a member's table.
  auto carry_entry = [&](size_t i, OracleId o, Part part, unsigned a, u64 index, const E &coeff) {
    if (part == Part::MASK)
      acc.d[group_index(index, a, L.mask_first[i], L.h)] += scaled(coeff);
    else if (L.kept[i])
      kept_form(o).c.add_entry(index, scaled(coeff));
    else
      acc.c.add_entry(group_index(index, a, L.main_first[i], L.k), scaled(coeff));
  };
  for (Term &t : row.terms) carry(t.oracle, t.part, std::move(t.v));
  if (aside) {
    aside->add_scaled(row, x);
  } else {
    for (const Unit &u : row.units) carry_entry(L.find(u.oracle), u.oracle, u.part, S.row_bits(u.oracle, u.part), u.index, u.coeff);
    for (const Run &r : row.runs) {
      const size_t i = L.find(r.oracle);
      const unsigned a = S.row_bits(r.oracle, r.part);
      for (size_t j = 0; j < r.size(); j++) carry_entry(i, r.oracle, r.part, a, r.first + r.step * j, r.at(j));
    }
  }
  for (MainForm &f : row.forms) {
    const size_t i = L.find(f.oracle);
    if (L.kept[i]) {
      kept_form(f.oracle).c.add(std::move(f.c), x);
      continue;
    }
    const unsigned a = S.row_bits(f.oracle, Part::MAIN);
    f.c.insert_choice(a, L.k - a, L.main_first[i] >> a);  // the bits above the block pick the block
    acc.c.add(std::move(f.c), x);
  }
  acc.target = acc.target + x * row.target;
}

}  // namespace blk
