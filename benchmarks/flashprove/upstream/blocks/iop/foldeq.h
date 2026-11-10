// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// FOLDEQ and the change of code that follows it (not the block CS, blocks/iop/cs.h): linear claims
// (type `Row`) on the fold of a committed table become linear claims on G_u, one fresh oracle
// message (encoded rows under a Merkle root).
// A committed table (`Index` in identifiers) is 2^v rows over the field F, encoded under one code,
// with a Merkle tree over the columns. It outlives every proof and is no proof's oracle message.
// For a fold point gamma (v challenges) the folded row u is sum_r eq(r, gamma) * row r. Encoding is
// linear, so the same sum of the codewords is the codeword, over the extension field E, of the
// folded message and random tail (the random entries encoded with a row's message).
// Messages: SEND (blocks/iop/send.h) sends the root of G_u, which contains the whole folded row,
// and takes an out-of-domain sample of it (a seed, then the prover's answer); the verifier draws
// positions of the table; the prover opens the columns there.
// Checks: the opened columns against the table's root. All else becomes claims with public values
// (`target` in `Row`) on G_u: SEND's sample claims, one claim per draw and, for the table's record
// (`IndexRecord`), IndexShape::answers more, each one answer folded over all rows. They are left to
// the proximity test on G_u's group (the oracle messages tested together).
// Callers: LIN (blocks/iop/lin.h), a batch's claims on one table; LINK (blocks/iop/link.h), a proof
// that ties a leaf table, a further table of a commitment, to the anchor, the first one. What this
// file leaves to its caller is stated at foldeq_prove() and foldeq_verify().
// Hiding: see record_value() (the pads), IndexHolder::record() (one record for a set of pads),
// IndexHolder::begin() (the count of opened columns) and `FoldeqShape` (what G_u contains).
//
// The table (rows that are codewords, columns hashed into a Merkle tree, a test by a random
// combination of the rows) is the commitment of Ligero: Ames, Hazay, Ishai, Venkitasubramaniam,
// "Ligero: Lightweight Sublinear Arguments Without a Trusted Setup", Cryptology ePrint Archive
// 2022/1608, sections 4.1 and 5.1; and of Brakedown, section 4.1 (cited in the next sentence).
// Reading the table at a point that folds its rows, and the split into a test, which may be made
// once, and evaluations, are Brakedown's: Golovnev, Lee, Setty, Thaler, Wahby, "Brakedown:
// Linear-time and field-agnostic SNARKs for R1CS", Cryptology ePrint Archive 2021/1043, section 4.1
// (there the combination of the rows is sent; here it is committed; that one test serves many
// evaluations is implied there and not said outright).
// The record (answers to out-of-domain samples of the stored rows, kept with the table's root) and
// the bound on the columns that may be opened of a table, with their count, follow further work
// that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "chain/rounds.h"
#include "code/code.h"
#include "engine/table.h"
#include "field/field.h"
#include "field/mle.h"
#include "iop/context.h"
#include "iop/relation.h"
#include "iop/send.h"
#include "iop/tensor.h"
#include "numbers/round_names.h"
#include "tree/opening.h"

namespace blk {

// ---------------------------------------------------------------------------------------------------------------------
// Structured vectors: a cut, and how a vector is placed in a table
// ---------------------------------------------------------------------------------------------------------------------

// Appends to out structured vectors (`Tensor`, tensor.h: a product of one factor per index bit)
// whose sum is v with an entry set to zero wherever bits from .. from + count - 1 of its index,
// read as a number, are >= len. A number is below len if and only if len has a 1 at the highest bit
// where the two differ: at most one summand for each 1 bit of len, and at most one in all if these
// bits index v's dense (written-out) factor, which is then copied and cut.
inline void tensor_below(const Tensor &v, unsigned from, unsigned count, u64 len, std::vector<Tensor> &out) {
  BLK_REQUIRE(from + count <= v.vars() && count < 64 && len <= (u64(1) << count), "tensor_below: the cut does not fit the vector");
  if (len == (u64(1) << count)) {
    out.push_back(v);
    return;
  }
  {  // Bits fixed by a factor 0 may put all nonzero entries of v below the cut, or none of them.
    u64 least = 0, most = 0;  // range of the number at a nonzero entry; dense bits are free
    for (unsigned t = 0; t < count; t++) {
      const bool written = std::find(v.dense_vars.begin(), v.dense_vars.end(), from + t) != v.dense_vars.end();
      if (!written && is_zero(v.lo(from + t))) least |= u64(1) << t;
      if (written || !is_zero(v.hi(from + t))) most |= u64(1) << t;
    }
    if (most < len) {
      out.push_back(v);
      return;
    }
    if (least >= len) return;
  }
  unsigned dense = 0;
  for (unsigned d : v.dense_vars) dense += d >= from && d < from + count;
  if (dense > 0) {  // the written-out factor is indexed by these bits: it is cut itself
    BLK_REQUIRE(dense == count, "tensor_below: the cut runs over written-out and product variables at once");
    BLK_REQUIRE(!v.is_held(), "tensor_below: a factor that only the engine holds is to be cut");
    auto values = std::make_shared<std::vector<E>>(v.values->begin() + v.first, v.values->begin() + v.first + (u64(1) << v.dense_vars.size()));
    for (u64 d = 0; d < values->size(); d++) {
      u64 number = 0;
      for (size_t i = 0; i < v.dense_vars.size(); i++)
        if (v.dense_vars[i] >= from && v.dense_vars[i] < from + count) number |= ((d >> i) & 1) << (v.dense_vars[i] - from);
      if (number >= len) (*values)[d] = E::zero();
    }
    Tensor w = v;
    w.set_values(std::move(values), 0);
    out.push_back(std::move(w));
    return;
  }
  for (unsigned q = count; q-- > 0;) {
    if (!((len >> q) & 1)) continue;
    Tensor w = v;
    for (unsigned t = q + 1; t < count; t++) ((len >> t) & 1) ? w.set_pair(from + t, E::zero(), w.hi(from + t)) : w.set_pair(from + t, w.lo(from + t), E::zero());
    w.set_pair(from + q, w.lo(from + q), E::zero());
    out.push_back(std::move(w));
  }
}

// How a vector is placed in the main table of an oracle message with 2^row_bits rows of
// 2^place_bits places: OracleShape::main_rows real rows; the rest, up to a power of two, virtual.
// index(j): the table's index of entry j; entry_at(): the inverse.
//   rows_high = false: entry j is entry j of the table read as a vector (tensor.h): row
//                      j mod 2^row_bits. All rows are in use from the first entries on, so the
//                      number of real rows must be a power of two.
//   rows_high = true:  a row contains 2^place_bits consecutive entries: row j div 2^place_bits. A
//                      vector of any length then takes just the rows that it needs.
// Either way the bits of j are the table's index bits in another order, so a structured vector
// stays one.
struct VectorLay {
  unsigned row_bits = 0, place_bits = 0;
  bool rows_high = false;

  unsigned bits() const { return row_bits + place_bits; }
  u64 index(u64 j) const {
    BLK_REQUIRE((j >> bits()) == 0, "lay: no such entry");
    return rows_high ? (j >> place_bits) | ((j & ((u64(1) << place_bits) - 1)) << row_bits) : j;
  }
  u64 entry_at(u64 index) const {
    return rows_high ? (index >> row_bits) | ((index & ((u64(1) << row_bits) - 1)) << place_bits) : index;
  }
  // v as a structured vector on the table: v is indexed by the low v.vars() bits of j, and the
  // result is 0 wherever a higher bit of j is 1.
  Tensor on_table(Tensor v) const {
    BLK_REQUIRE(v.vars() <= bits(), "lay: the vector is longer than the table");
    v.insert_choice(v.vars(), bits() - v.vars(), 0);
    if (rows_high) {
      v.rotate(place_bits);
      for (unsigned &d : v.dense_vars) d = d >= place_bits ? d - place_bits : d + row_bits;
    }
    return v;
  }
};

inline VectorLay vector_lay(const Codes &codes, const OracleShape &s, bool rows_high) {
  return VectorLay{ceil_log2(s.main_rows), codes.base[s.code].log_ell(), rows_high};
}
// How many leading entries of a vector are in real rows and before every main-table place of the
// pad of the oracle message's sample; 0 if the shape does not allow this placement or the table's
// index has more than 60 bits.
inline u64 vector_room(const Codes &codes, const OracleShape &s, bool rows_high) {
  const VectorLay lay = vector_lay(codes, s, rows_high);
  if (!rows_high && s.main_rows != (size_t(1) << lay.row_bits)) return 0;
  if (lay.bits() > 60) return 0;
  u64 room = (u64)s.main_rows << lay.place_bits;
  if (s.answers > 0)
    for (const PadPlace &p : Send::pad(codes, s))
      if (p.part == Part::MAIN) room = std::min(room, lay.entry_at(p.index));
  return room;
}

// ---------------------------------------------------------------------------------------------------------------------
// The committed table: what is public
// ---------------------------------------------------------------------------------------------------------------------

struct IndexShape {
  Code code{};           // of a row; rows are over F
  unsigned v = 0;        // 2^v rows
  unsigned answers = 0;  // answers of the record for each row
  // Bytes of salt in a leaf of the tree over the columns. The tree's number is not a member:
  // index_tree() sets it to 0 for every table, and tables are distinguished by their roots.
  size_t salt_bytes = SALT_FLOOR_BYTES;
  // The distances at which the tests on the table are sized: the one-time test (the committer's
  // own) and later tests (a batch's). Not bound to any hash chain (a Fiat-Shamir transcript,
  // blocks/chain/chain.h) or name: public_bytes() and index_name() leave this member out. The error
  // bounds (`Priced` in identifiers) of a later test and of a LINK proof are true only of a table
  // whose one-time test ran at the same distances, in another run; this member carries them there.
  // A run's shape has what the run is sized at, and the plan (the list of rounds with their error
  // bounds, blocks/numbers/plan.h) must have the same (priced_table()). The shape of a reader
  // (`IndexReader`) has what its creator put there; check_key() and check_entry() (blocks/frame/)
  // return their own run's shape inside a `TestedTable` (below), which ordinary code cannot alter.
  // foldeq_columns_verify() compares the two.
  TableDistances distances;

  u64 rows() const { return u64(1) << v; }
  size_t pads() const { return size_t(E_DEGREE) * answers; }  // entries at the end of a row's message
  size_t plain() const { return code.ell - pads(); }          // entries before them
  u64 pad_place(unsigned answer, unsigned t) const { return plain() + size_t(E_DEGREE) * answer + t; }
  unsigned cube_bits() const { return code.log_ell(); }
};
// What a plan's error bounds use of a table, as this run has it, for require_table()
// (blocks/chain/rounds.h), which compares it with the plan's. The numbers are public inputs twice:
// the plan bounds a drawn column's error from the code's rate, the rows and the record that the
// planner was given; the run reads the shape bound in its chain's first query (the hash input that
// fixes the statement before the prover speaks).
// test: which test on the table runs (`TableTest`, blocks/numbers/plan.h). No shape records this,
// so the caller passes it.
inline PricedTable priced_table(const IndexShape &s, TableTest test) {
  PricedTable t;
  t.code = PricedCode{s.code.log_n, s.code.ell, s.code.r}, t.v = s.v, t.answers = s.answers, t.distances = s.distances, t.test = test, t.said = true;
  return t;
}
// Null, or what is wrong. For numbers from a file.
inline const char *index_fault(const IndexShape &s) {
  if (!code_ok(s.code)) return "the table's numbers are not a code";
  if (s.v > 40) return "the table has too many rows";
  if (s.pads() >= s.code.ell) return "a row has no room beside its pads";
  if (!salt_length_fits(s.salt_bytes)) return "the leaves of the table's tree have no admitted salt";
  if (s.answers < 1) return "the table's number of answers is below 1";
  return nullptr;
}

// A table's record: the committer's answers to out-of-domain samples of each row, kept with the
// table's root. y[r * answers + i]: by row, then by answer. The seeds, one
// for each answer and shared by all rows, are a function of the table's name and shape only
// (record_seeds()): each side computes them, and they are stored nowhere.
// Error bounds and their theorems: blocks/numbers/blocks.h, plan_lin_shared() and plan_record().
//   Beyond half the distance a word can be close to several tables, and a prover
//   could argue from another one in each proof. "The table" is then the close table that agrees
//   with the record. The record's equations are linear in the rows, so at gamma each answer folds
//   into one claim on G_u.
// The answers must come before every gamma: a prover that knows gamma first fits them to any close
// table. So every chain
// that draws a gamma must have the record in its first query (`IndexName`).
struct IndexRecord {
  std::vector<E> y;
};
inline std::vector<u8> record_bytes(const IndexRecord &rec) {
  Writer w;
  for (const E &y : rec.y) w.put(y);
  return w.bytes();
}

// The seeds of the record of the table with this name (IndexReader::name()).
inline std::vector<SampleSeed> record_seeds(const IndexShape &s, const std::vector<u8> &name) {
  return record_seeds(name.data(), name.size(), s.cube_bits(), s.answers);
}
// Null, or what is wrong. For a record from a file.
inline const char *record_fault(const IndexShape &s, const IndexRecord &rec) {
  return rec.y.size() != s.rows() * s.answers ? "the record has not one answer for each row and seed" : nullptr;
}

inline void public_bytes(Writer &w, const SampleSeed &s) { public_bytes(w, s.z), public_bytes(w, s.beta); }
inline void public_bytes(Writer &w, const IndexShape &s) {
  public_bytes(w, s.code), public_bytes(w, s.v), public_bytes(w, s.answers), public_bytes(w, s.salt_bytes);
}
inline void public_bytes(Writer &w, const IndexRecord &r) { public_bytes(w, r.y); }
// Faults that each of these values shows by itself (blocks/chain/public.h). record_fault() checks a
// record against a shape.
inline const char *public_fault(const SampleSeed &s) { return is_zero(s.beta) ? "a sample's seed has the factor 0 on its pad" : nullptr; }
inline const char *public_fault(const IndexShape &s) { return index_fault(s); }
inline const char *public_fault(const IndexRecord &) { return nullptr; }  // any list of elements

// A table's item in the statement of a chain's first query: its commitment and its record's
// answers. This binds a proof to one table by value.
struct IndexName {
  std::vector<u8> commitment;
  IndexRecord record;
};
inline void public_bytes(Writer &w, const IndexName &n) { public_bytes(w, n.commitment), public_bytes(w, n.record); }
inline const char *public_fault(const IndexName &n) { return n.commitment.empty() ? "a table without a commitment" : public_fault(n.record); }

// What a verifier has of a committed table. The form of the tree and of an opening is left to the
// implementation: a long-lived table may be hashed differently from a proof's oracle messages.
class IndexReader {
 public:
  virtual ~IndexReader() = default;
  virtual const IndexShape &shape() const = 0;
  virtual const IndexRecord &record() const = 0;
  // The bytes that commit to the table (its root, say): for the first query of every chain on it.
  virtual std::vector<u8> commitment() const = 0;
  // The table's name: all that fixes which words are committed to and how a column is read. It is
  // derived from the commitment and from what the verifier has of its own, and is in no first
  // query. record_seeds() takes it, and published_id() (blocks/frame/heads.h) hashes the anchor's
  // into the commitment ID.
  virtual std::vector<u8> name() const { return commitment(); }
  // Null, or how the reader contradicts itself. Callers include foldeq_require() and LINK's
  // link_key_fault().
  virtual const char *fault() const { return nullptr; }
  // The configuration digest (a hash of the public configuration) from which the hashes of the
  // table's tree start: the verifier's own, never a key file's, so a table created under another
  // one fails every opening. LIN and LINK require it to equal their chain's. The default, zeros, is
  // meant to equal no chain's.
  virtual Digest configuration() const { return Digest{}; }
  // Reads from file the opening at the list `at` (the drawn positions, sorted, without repeats) and
  // checks it. columns[i * 2^v + r]: the symbol of row r at the i-th listed position. session
  // (blocks/chain/session.h): `IndexKey` enters what an accepted opening shows into the session's
  // one store of values under the table's root, so that a table read at several lists or in several
  // chains of one verdict has one value at each vertex.
  virtual bool read(Reader &file, const PositionList &at, std::vector<F> &columns, VerifierSession &session) const = 0;
};

// The name of a table hashed as index_tree() says: the root, the configuration digest and six
// numbers of the shape. Without the shape, one root could be opened as columns of several shapes.
// The name is the same in every role (anchor, leaf table, either table of a LINK proof), so a table
// has one record whatever its role.
inline std::vector<u8> index_name(const Digest &root, const IndexShape &s, const Digest &configuration) {
  Writer w;
  w.put(root);
  w.put(configuration);  // the configuration digest, from which the tree's hashes start
  for (u64 x : {(u64)s.code.log_n, (u64)s.code.ell, (u64)s.code.r, (u64)s.v, (u64)s.answers, (u64)s.salt_bytes}) w.put_u64(x);
  return w.bytes();
}
// The tree over the columns: one part, 2^v symbols of F in a leaf. It outlives every run, so its
// leaves start, as all its vertices do, from the configuration digest and from no chain's value.
inline OracleTreeShape index_tree(const IndexShape &s, const Digest &configuration) {
  OracleTreeShape t;
  t.log_n = s.code.log_n;
  t.main_symbols.push_back(s.rows()), t.mask_symbols.push_back(0);
  t.tree_number = 0, t.salt_bytes = s.salt_bytes;
  t.stored = true, t.parameters = configuration;
  return t;
}

inline std::vector<u8> digest_bytes(const Digest &d) {
  Writer w;
  w.put(d);
  return w.bytes();
}

// A table's key: the `IndexReader` of a table whose tree has the shape index_tree().
struct IndexKey : IndexReader {
  IndexShape table;
  Digest made_under{};  // the configuration's digest: the verifier's own
  Digest root{};        // root and rec: from the key file or a leaf file (`entry` in `check_entry`)
  IndexRecord rec;

  IndexKey() = default;
  IndexKey(IndexShape s, const Digest &configuration, const Digest &r, IndexRecord y) : table(s), made_under(configuration), root(r), rec(std::move(y)) {}
  const IndexShape &shape() const override { return table; }
  const IndexRecord &record() const override { return rec; }
  std::vector<u8> commitment() const override { return digest_bytes(root); }
  std::vector<u8> name() const override { return index_name(root, table, made_under); }
  Digest configuration() const override { return made_under; }
  const char *fault() const override;
  bool read(Reader &file, const PositionList &at, std::vector<F> &columns, VerifierSession &session) const override {
    Opening o;
    const OracleTreeShape tree = index_tree(table, made_under);
    if (!read_opening(file, tree, at, o) || !check_opening(root, tree, at, o, &session.record(root))) return false;
    columns = std::move(o.columns);  // the table has no mask rows: a column is its main symbols
    return true;
  }
};
// Null, or what is wrong. For a key from a file.
inline const char *index_key_fault(const IndexKey &key) {
  const IndexShape &s = key.table;
  if (const char *f = index_fault(s)) return f;
  if (key.made_under == Digest{}) return "the key names no configuration";
  return record_fault(s, key.rec);
}
inline const char *IndexKey::fault() const { return index_key_fault(*this); }

// ---------------------------------------------------------------------------------------------------------------------
// A table that has passed its test
// ---------------------------------------------------------------------------------------------------------------------

// The error bound of a later test on a committed table (a batch's test; the first table of a LINK
// proof) is valid only if the table has already passed its one-time test, or a LINK proof that ties
// it to such a table, at the same distances and under the same setting (`Setting`,
// blocks/numbers/setting.h, with the target in bits, proof-of-work bits and reading of the
// proximity bound). One type, `TestedTable`, carries that fact from the earlier check to the later
// one, inside one process. Two more types go with it.
//   TestedTable    A table's reader, the setting under which its test passed, and the other public
//                  values under which its key was checked, such as the ranges of the entries and
//                  the matrix's rows (key_public_bytes(), blocks/frame/key_fixed.h). Apart from
//                  tests (on_a_tests_word()) and UnprovedTable's stand-in (as_if_tested(), below),
//                  only check_key_under() (blocks/frame/check_key.h) and check_entry_under()
//                  (blocks/frame/check_entry.h) create one that counts as tested, as their last
//                  step, after every chain of the file has been accepted. It has no public data
//                  member and no setter. A copy of a tested table is tested.
//   UnprovedTable  A table's reader taken from the first bytes of a key file or leaf file, without
//                  the proofs that follow them, with the setting and the key's public values of the
//                  caller's own configuration. For a program that checks key file, leaf file and
//                  proof in separate processes, as cp_verify does. Where a tested table is needed,
//                  it serves two functions only, check_entry_if_tested() and
//                  check_proof_if_tested(). Their result type, `IfTested`, has no plain yes and
//                  converts to neither bool nor `ProofVerdict`. Inside them it stands in for a
//                  `TestedTable` though no test has passed in this process, so the comparisons
//                  below then have the caller's own values on both sides. The stand-in reaches the
//                  library's own checkers and no caller. UnprovedTable::reader() is public and
//                  counts as untested everywhere else.
//   TableRead      The argument type of the functions that read a table's columns: any
//                  `IndexReader`, which counts as untested, or a `TestedTable`.
// foldeq_columns_verify() ends the process if a read whose bound needs an earlier test gets an
// untested reader, or a table tested at other distances or under another setting.
// check_entry_under() and check_proof() (blocks/frame/check.h) end it if the table's key was
// checked under other public values than their own. An `EntryUnder` that a test constructs
// (`BLK_ENTRY_UNDER_FOR_A_TEST`) has no such values: check_entry_under() then skips the comparison.
// Limits: these types stop mistakes and ordinary code, not a caller who sets out to defeat them.
// They say nothing across processes or files: that the three stages of cp_verify checked the same
// bytes is that program's own logic. The whole contract, with the ways around the types that C++
// leaves open: blocks/chain/closed.h, above run_verifier().
enum class IfTested { NO, YES_IF_THE_TABLE_IS_TESTED, YES_WITH_WORK_IF_THE_TABLE_IS_TESTED };
// What the two functions that return `IfTested` leave in `why`, and in `place` or `at`, unless
// they reject (`refuse` in identifiers; printed "refused").
inline constexpr const char *IF_TESTED_WHY = "not refused; worth a yes only if the checks of the table's own files pass";

class Configured;   // blocks/frame/configure.h
class KeyUnder;     // blocks/frame/check_key.h
class EntryUnder;   // blocks/frame/check_entry.h
struct RequestPublic;  // blocks/frame/check.h, as the next two
struct LinksResult;
struct ProofFacts;
class UnprovedTable;

class TestedTable final {
 public:
  // A default-constructed table counts as untested, and its reader names no configuration, so no
  // check passes with it: LIN ends the process; LINK, check_entry_under() and check_proof() reject.
  TestedTable() = default;  // none yet
  const IndexReader &reader() const { return key_; }
  const Setting *tested_under() const { return tested_ ? &under_ : nullptr; }  // null: none yet
  // What key_public_bytes() gave at the key's check; a leaf table has the bytes of the key to which
  // it is linked.
  const std::vector<u8> &key_public() const { return key_public_; }
#ifdef BLK_TESTED_TABLE_FOR_A_TEST
  // For a test or a toy tool that checks a leaf file or a proof without a key file: the caller
  // vouches for the table.
  static TestedTable on_a_tests_word(IndexKey key, const Setting &under, std::vector<u8> key_public) { return TestedTable(std::move(key), under, std::move(key_public)); }
#endif

 private:
  TestedTable(IndexKey key, const Setting &under, std::vector<u8> key_public) : key_(std::move(key)), under_(under), key_public_(std::move(key_public)), tested_(true) {}
  friend class UnprovedTable;
  friend bool check_key_under(const KeyUnder &, Reader &, TestedTable &, const char *&, std::vector<std::pair<std::string, E>> *, const char **);
  friend bool check_entry_under(const EntryUnder &, const TestedTable &, Reader &, TestedTable &, const char *&, std::vector<std::pair<std::string, E>> *, const char **);
  IndexKey key_;
  Setting under_;
  std::vector<u8> key_public_;
  bool tested_ = false;
};
// Stops a production build (`SERVING` in macro names) also where the switch is defined after the
// first include of blocks/base.h, which the list in blocks/fence.h does not see.
#ifdef BLK_TESTED_TABLE_FOR_A_TEST
BLK_MARK_NOT_FOR_SERVING();
#endif

class UnprovedTable final {
 public:
  UnprovedTable() = default;  // none yet: the two functions that take it reject
  const IndexReader &reader() const { return key_; }  // for the table's name and the commitment ID

 private:
  UnprovedTable(IndexKey key, const Setting &under, std::vector<u8> key_public) : key_(std::move(key)), under_(under), key_public_(std::move(key_public)), read_(true) {}
  // An unfilled table (`!read_`), as after a failed read that the caller ignored, gives a
  // default-constructed `TestedTable`.
  TestedTable as_if_tested() const { return read_ ? TestedTable(key_, under_, key_public_) : TestedTable(); }
  friend bool read_key_table_unproved(const Configured &, Reader &, UnprovedTable &, const char *&);
  friend bool read_entry_table_unproved(const Configured &, Reader &, UnprovedTable &, const char *&);
  friend IfTested check_entry_if_tested(const Configured &, const UnprovedTable &, Reader &, const char *&, std::vector<std::pair<std::string, E>> *, const char **);
  friend IfTested check_proof_if_tested(const Configured &, const UnprovedTable &, u32, const std::vector<u8> &, const std::vector<u8> &, const E &, Reader &,
                                        const std::function<bool(VerifierContext &, const RequestPublic &, LinksResult &)> &, const std::function<void(Binder &)> &, const char *&, ProofFacts *,
                                        const char **);
  IndexKey key_;
  Setting under_;
  std::vector<u8> key_public_;
  bool read_ = false;
};

class TableRead {  // it holds references: an argument's type, not to be kept
 public:
  TableRead(const IndexReader &untested) : reader_(untested) {}
  TableRead(const TestedTable &tested) : reader_(tested.reader()), under_(tested.tested_under()) {}
  const IndexReader &reader() const { return reader_; }
  const Setting *tested_under() const { return under_; }  // null: no test has passed
#ifdef BLK_TESTED_TABLE_FOR_A_TEST
  // As TestedTable::on_a_tests_word(), for any reader. Both arguments must outlive the result.
  static TableRead on_a_tests_word(const IndexReader &reader, const Setting &under) {
    TableRead t(reader);
    t.under_ = &under;
    return t;
  }
#endif

 private:
  const IndexReader &reader_;
  const Setting *under_ = nullptr;
};
// Whether a read's error bound needs a table that has passed an earlier test.
enum class TableStanding { UNDER_TEST, ALREADY_TESTED };

// One answer of the record, from one row's message (entries over F or, for a folded row, over E):
// the multilinear extension of the entries before the pads, at seed.z, plus seed.beta times the
// answer's pad. A pad is one uniform element of E that no other answer reads: E_DEGREE entries over
// F at the end of the row's message, the coefficients of 1, X, X^2. z has no weight on any pad.
template <class T> inline E record_value(const IndexShape &s, const SampleSeed &seed, unsigned answer, const T *message) {
  E sum = mle_eval(message, s.code.ell, seed.z.data(), s.cube_bits());
  for (size_t p = s.plain(); p < s.code.ell; p++) sum -= eq_index(p, seed.z.data(), s.cube_bits()) * message[p];
  for (unsigned t = 0; t < E_DEGREE; t++) sum += seed.beta * E::theta(t) * message[s.pad_place(answer, t)];
  return sum;
}

// ---------------------------------------------------------------------------------------------------------------------
// The committed table: what its holder does for a prover
// ---------------------------------------------------------------------------------------------------------------------

// Sums over blocks of consecutive entries of a folded message:
// sums[y] = sum_{j < 2^k_lo} eq(j, low) message[j + 2^k_lo y], y < count.
struct BlockSums {
  const E *low = nullptr;  // k_lo coordinates
  unsigned k_lo = 0;
  u64 count = 0;
  std::vector<E> sums;
};
inline void block_sums(const std::vector<E> &message, BlockSums &b) {
  BLK_REQUIRE((b.count << b.k_lo) <= message.size(), "block sums: the blocks run beyond the message");
  std::vector<E> weight(size_t(1) << b.k_lo);
  eq_table(b.low, b.k_lo, weight.data());
  b.sums.assign(b.count, E::zero());
  for (u64 y = 0; y < b.count; y++)
    for (size_t j = 0; j < weight.size(); j++) b.sums[y] += weight[j] * message[j + (y << b.k_lo)];
}

// The form in which fold_table() returns the folded row, a table over E: `rows` rows of `len`
// consecutive entries; the folded message from entry 0, the folded tail from entry tail_first,
// zeros elsewhere, also where SEND later writes the pad of G_u's own sample
// (Engine::set_entries(), blocks/engine/engine.h). An engine that lets the holder fill this table
// again must write that pad again.
struct FoldedWant {
  size_t rows = 0, len = 0;
  u64 tail_first = 0;
};

// The prover's interface to a committed table. The implementation, the table's holder ("the keeper"
// in a message below), has the rows, their pads and random tails, the tree, and the count of opened
// columns. It need not store the codewords: fold() is one pass over the rows, and open() may be a
// second that computes the drawn columns again.
class IndexHolder {
 public:
  virtual ~IndexHolder() = default;
  virtual const IndexShape &shape() const = 0;
  virtual std::vector<u8> commitment() const = 0;  // as IndexReader's
  virtual std::vector<u8> name() const { return commitment(); }  // as IndexReader's
  virtual Digest configuration() const { return Digest{}; }  // as IndexReader's

  // The answers at the seeds that record_seeds() gives for the table's name. No call answers at
  // other seeds: a set of pads serves one record only, since two records give exact linear
  // functionals of the rows. Left to the committer: whenever a table is created, its pads, random
  // tails and tree salts all come from one secret used for that table only, or are all freshly
  // drawn, so that the same pads are never under two commitments.
  virtual IndexRecord record() = 0;

  // A proof that will draw this number of columns begins. A row's random tail hides only a fixed
  // number of columns over the table's whole life; the holder counts them here. False: the tails
  // cannot cover these columns as well. Then nothing of the table may be sent, and a fresh table is
  // needed. An abandoned proof has spent its columns all the same. So whoever keeps the count (the
  // holder, or a program whose counter it calls) must write it, where every later reading finds it,
  // before this call returns. The same party must make it durable (a loss of power does not undo
  // it) before fold() or fold_table() is called, because the columns are drawn after that call.
  virtual bool begin(u64 columns) = 0;

  // message[j] = sum_r eq(r, gamma) * (message entry j of row r), pads included; tail likewise.
  // blocks (may be null): its sums are filled in too. Hidden values, all of them.
  virtual void fold(const E *gamma, std::vector<E> &message, std::vector<E> &tail, BlockSums *blocks) = 0;

  // The same fold as G_u's main table (`want`), created where the prover's engine (`Engine`) reads
  // it without a copy: at full size the folded row is too large for host vectors. NO_ID: not
  // offered; the caller then calls fold(). The table must live until the engine has dropped it.
  virtual TableId fold_table(const E *gamma, const FoldedWant &want, BlockSums *blocks) {
    return (void)gamma, (void)want, (void)blocks, NO_ID;
  }

  // Writes the opening at the list, as IndexReader::read() reads it, and returns the symbols in the
  // same order. A holder that computes the columns again checks them against its tree before it
  // writes a byte; if that check fails, nothing that it writes depends on the table.
  virtual void open(const PositionList &at, Writer &file, std::vector<F> &columns) = 0;

  // The columns are opened: the holder may release what it has of this fold, except a table that
  // fold_table() returned.
  virtual void forget() {}
};

// ---------------------------------------------------------------------------------------------------------------------
// The folded row inside G_u, and the claims on it
// ---------------------------------------------------------------------------------------------------------------------

// FOLDEQ's public shape: `index` is the table's shape, `u` G_u's. G_u's message, as a
// vector: the folded message from entry 0, pads included, since the folded record reads them; the
// folded tail from tail_first(), the next multiple of the least power of two that holds the tail,
// so that an index inside the tail is the low bits of the entry's number; the pad of G_u's own
// sample where SEND puts it; zeros elsewhere. rows_high: as in `VectorLay`. leaf_class: G_u shares
// a tree only with oracle messages of its prover message that have the same number.
struct FoldeqShape {
  IndexShape index;
  OracleShape u;
  bool rows_high = false;
  unsigned leaf_class = 0;
  u64 columns = 0;  // how many positions of the table a proof draws; the plan has the same number

  u64 tail_first() const {
    const u64 step = u64(1) << index.code.log_r();
    return (index.code.ell + step - 1) / step * step;
  }
  u64 entries() const { return tail_first() + index.code.r; }
};
inline void public_bytes(Writer &w, const FoldeqShape &S) {
  public_bytes(w, S.index), public_bytes(w, S.u), public_bytes(w, S.rows_high), public_bytes(w, S.leaf_class), public_bytes(w, S.columns);
}
// Faults of the shape by itself.
inline const char *public_fault(const FoldeqShape &S) {
  if (const char *f = index_fault(S.index)) return f;
  if (const char *f = public_fault(S.u)) return f;
  if (S.u.main_rows == 0 || S.u.main_deg != E_DEGREE) return "the folded row's oracle message has no main rows over E";
  if (S.u.answers < 1) return "the folded row's oracle message has no sample (answers = 0)";
  if (S.columns < 1) return "no column is drawn";
  return nullptr;
}
// The same, and faults against the codes that the shape names.
inline const char *foldeq_fault(const Codes &codes, const FoldeqShape &S) {
  if (const char *f = public_fault(S)) return f;
  if (S.u.code >= codes.base.size()) return "the folded row's oracle message names a code that is not there";
  if (S.entries() > vector_room(codes, S.u, S.rows_high)) return "the folded row does not fit beside the pad of its oracle message";
  return nullptr;
}

// G_u's number, and how a vector is placed in its main table.
struct FoldedRow {
  OracleId oracle = NONE_ID;
  VectorLay lay;
};

// The value of the fold at a position, from the column there.
inline E foldeq_symbol(const std::vector<E> &eq_gamma, const F *column) {
  E sum = E::zero();
  for (size_t r = 0; r < eq_gamma.size(); r++) sum += eq_gamma[r] * column[r];
  return sum;
}

// Answer i of the record, folded: the claim that record_value()'s form on the folded message has
// the value sum_r eq(r, gamma) y[r * answers + i]. It reads every entry before the pads, also
// entries that no layout uses, and answer i's own pad.
inline Row foldeq_record_row(const FoldeqShape &S, const SampleSeed &seed, const IndexRecord &rec, unsigned i, const std::vector<E> &eq_gamma, const FoldedRow &at) {
  const IndexShape &s = S.index;
  BLK_REQUIRE(seed.z.size() == s.cube_bits() && rec.y.size() == s.rows() * s.answers, "record: not a record of this table");
  Row row;
  row.name = "the record, folded";
  const Tensor under = Tensor::eq(seed.z.data(), s.cube_bits());
  std::vector<Tensor> cut;
  tensor_below(under, 0, s.cube_bits(), s.code.ell, cut);
  for (Tensor &t : cut) row.add(at.oracle, Part::MAIN, at.lay.on_table(std::move(t)));
  for (unsigned a = 0; a < s.answers; a++)
    for (unsigned t = 0; t < E_DEGREE; t++) {
      const u64 p = s.pad_place(a, t);
      row.add_unit(at.oracle, Part::MAIN, at.lay.index(p), (a == i ? seed.beta * E::theta(t) : E::zero()) - under.at(p));
    }
  for (u64 r = 0; r < s.rows(); r++) row.target += eq_gamma[r] * rec.y[r * s.answers + i];
  return row;
}

// The claim on G_u that the folded row's codeword has the value y at the table's position x.
inline Row foldeq_column_row(const FoldeqShape &S, u64 x, const E &y, const FoldedRow &at) {
  const Code &c = S.index.code;
  const GeneratorRow g = generator_at(c, x);
  Row row;
  row.name = "a column of the table";
  row.kind = RowKind::GENERATOR;
  std::vector<Tensor> cut;
  tensor_below(Tensor::product(g.f, c.log_ell(), E::from(g.message_scale)), 0, c.log_ell(), c.ell, cut);
  for (Tensor &t : cut) row.add(at.oracle, Part::MAIN, at.lay.on_table(std::move(t)));
  if (c.r > 0) {
    cut.clear();
    tensor_below(Tensor::product(g.f, c.log_r(), E::from(g.tail_scale)), 0, c.log_r(), c.r, cut);
    for (Tensor &t : cut) {
      t.insert_choice(c.log_r(), at.lay.bits() - c.log_r(), S.tail_first() >> c.log_r());
      row.add(at.oracle, Part::MAIN, at.lay.on_table(std::move(t)));
    }
  }
  row.target = y;
  return row;
}

// G_u's main table on the host.
inline Table foldeq_table(const Codes &codes, const FoldeqShape &S, const std::vector<E> &message, const std::vector<E> &tail) {
  BLK_REQUIRE(message.size() == S.index.code.ell && tail.size() == S.index.code.r, "foldeq: not a folded row of this table");
  const VectorLay lay = vector_lay(codes, S.u, S.rows_high);
  Table t(E_DEGREE, S.u.main_rows, codes.base[S.u.code].ell);
  for (size_t j = 0; j < message.size(); j++) t.set_entry(lay.index(j), message[j]);
  for (size_t j = 0; j < tail.size(); j++) t.set_entry(lay.index(S.tail_first() + j), tail[j]);
  return t;
}

// ---------------------------------------------------------------------------------------------------------------------
// The block
// ---------------------------------------------------------------------------------------------------------------------

// The block in parts, for a caller that sends several folded rows in one prover message or draws
// the columns of several tables in one verifier message (link.h). foldeq_prove() and
// foldeq_verify() run them in order for one table.

// eq_gamma[r] = eq(r, gamma) for each row r.
inline std::vector<E> foldeq_weights(const FoldeqShape &S, const E *gamma) {
  std::vector<E> eq_gamma(S.index.rows());
  eq_table(gamma, S.index.v, eq_gamma.data());
  return eq_gamma;
}

// Prover, first part: the fold of the table at gamma as G_u's draft, not yet sent. blocks (may be
// null): sums over the folded message for the caller's next steps. record: not read by any code
// in this repository; a test build that is not part of it reads it.
inline OracleDraft foldeq_draft(ProverContext &P, IndexHolder &index, [[maybe_unused]] const IndexRecord &record, const FoldeqShape &S, const E *gamma,
                                BlockSums *blocks) {
  const char *fault = foldeq_fault(P.st.codes(), S);
  BLK_REQUIRE(!fault, fault ? fault : "");
  OracleDraft draft;
  draft.shape = S.u, draft.leaf_class = S.leaf_class;
  if (S.u.mask_rows) draft.mask = Send::fresh_mask(P.st.codes(), S.u, P.coins);
  if (S.rows_high) draft.main = index.fold_table(gamma, FoldedWant{S.u.main_rows, P.st.codes().base[S.u.code].ell, S.tail_first()}, blocks);
  if (draft.main != NO_ID) return draft;
  std::vector<E> u, tail;
  index.fold(gamma, u, tail, blocks);
  draft.main = P.engine.put(foldeq_table(P.st.codes(), S, u, tail));
  return draft;
}

// Both sides, once G_u is sent and its sample answered: G_u joins g, a group of its code, with the
// record's claims. name: the table's, for the record's seeds.
inline FoldedRow foldeq_joined(Statement &st, const FoldeqShape &S, const std::vector<u8> &name, const IndexRecord &rec, const std::vector<E> &eq_gamma, OracleId oracle,
                               GroupId g) {
  FoldedRow at{oracle, vector_lay(st.codes(), S.u, S.rows_high)};
  st.join(g, at.oracle);
  const std::vector<SampleSeed> seeds = record_seeds(S.index, name);
  for (unsigned i = 0; i < S.index.answers; i++) st.add_row(g, foldeq_record_row(S, seeds[i], rec, i, eq_gamma, at));
  return at;
}

// Opens the table's columns at the list and adds one claim on G_u for each draw, repeats included.
// The list must be drawn after G_u's root, seed and answer.
// foldeq_columns_verify() reads and checks the opening instead and adds the same claims.
inline void foldeq_columns_prove(ProverContext &P, IndexHolder &index, const FoldeqShape &S, const std::vector<E> &eq_gamma, const PositionList &list,
                                 const FoldedRow &at, GroupId g) {
  BLK_REQUIRE(list.draws().size() == S.columns && list.log_n() == S.index.code.log_n, "foldeq: the plan draws other columns than the shape says");
  std::vector<F> columns;
  index.open(list, P.ch.file_only(), columns);
  index.forget();
  BLK_REQUIRE(columns.size() == list.size() * S.index.rows(), "foldeq: the keeper hands back other columns than were asked for");
  for (u64 x : list.draws()) P.st.add_row(g, foldeq_column_row(S, x, foldeq_symbol(eq_gamma, &columns[list.index_of(x) * S.index.rows()]), at));
}
inline bool foldeq_columns_verify(VerifierContext &V, const TableRead &table, TableTest test, TableStanding standing, const FoldeqShape &S, const std::vector<E> &eq_gamma,
                                  const PositionList &list, const FoldedRow &at, GroupId g) {
  const IndexReader &key = table.reader();
  // The plan's error bound for the round closed last (in LIN and LINK, the round that drew these
  // positions) must have been computed for this table and the caller's `test`: otherwise, unless
  // the proof is already rejected, the process ends. The one-time test reads a table under test, a
  // later test a tested one, and a LINK proof one of each: for `TableTest::LINK` the caller's
  // `standing` alone says which, and nothing here checks it. Without the check of tested_under(), a
  // second holder's reader, built at the batch's distances, got two false claims accepted (test
  // harness, toy size, a target of 24 bits).
  V.ch.require_table(priced_table(S.index, test));
  BLK_REQUIRE(test == TableTest::LINK || (standing == TableStanding::ALREADY_TESTED) == (test == TableTest::LATER), "foldeq: the caller says that a one-time test reads a tested table, or a later test a table under test");
  if (standing == TableStanding::ALREADY_TESTED) {
    BLK_REQUIRE(table.tested_under(), "foldeq: the error bound of this read needs a table that has passed its one-time test, and this reader has passed none: hand in what check_key or check_entry gave");
    BLK_REQUIRE(Plan::same_setting(*table.tested_under(), V.ch.plan()->set),
                "foldeq: the table was tested under another setting (target, work bits) than this run's: key, entry and proof must be checked under one Configured");
  }
  // The reader's distances must be the run's (IndexShape::distances), or the process ends.
  // The check is here, where a reader is used, so that LIN
  // and both tables of LINK pass it, and before this function's ok(), because neither value comes
  // from the proof. LIN and LINK arrive after messages that they have read and an ok() of their
  // own, so the proof's bytes decide whether a wrongly sized reader first meets a rejection or this
  // end of the process; it never meets an acceptance, since every acceptance passes here. In the
  // programs cp_commit, cp_prove and cp_verify both values are copies of one `Configured`'s numbers
  // (blocks/frame/configure.h), so no file can trigger the check there: it guards a caller that
  // computes the sizes more than once.
  // What these checks do not cover: blocks/chain/closed.h, the contract.
  if (!(key.shape().distances == S.index.distances))
    std::fprintf(stderr, "foldeq: the reader says %s; the run has %s\n", text_of(key.shape().distances).c_str(), text_of(S.index.distances).c_str());
  BLK_REQUIRE(key.shape().distances == S.index.distances, "foldeq: the table's reader names other distances than this run is sized at: key, entry and batch must come from one sized whole");
  if (!V.ch.ok()) return false;
  BLK_REQUIRE(list.draws().size() == S.columns && list.log_n() == S.index.code.log_n, "foldeq: the plan draws other columns than the shape says");
  std::vector<F> columns;
  if (!key.read(V.ch.file_only(), list, columns, V.ch.session()) || columns.size() != list.size() * S.index.rows())
    return V.ch.refuse("the opened columns of the table do not lead to its root"), false;
  for (u64 x : list.draws()) V.st.add_row(g, foldeq_column_row(S, x, foldeq_symbol(eq_gamma, &columns[list.index_of(x) * S.index.rows()]), at));
  if (V.notes) {  // for comparing verifiers block by block; the notes change nothing
    const Row first = foldeq_column_row(S, list.draws()[0], foldeq_symbol(eq_gamma, &columns[list.index_of(list.draws()[0]) * S.index.rows()]), at);
    V.note("FOLDEQ: (Col_1)'s target", first.target);
    V.note("FOLDEQ: (Col_1) on G_u<0>", first.coefficient(at.oracle, Part::MAIN, at.lay.index(0)));
    if (S.index.code.r) V.note("FOLDEQ: (Col_1) on G_u<B>", first.coefficient(at.oracle, Part::MAIN, at.lay.index(S.tail_first())));
  }
  return true;
}

// Verifier, before anything is read. Shape and key are no parts of a proof, so a fault ends the
// process: a caller that reads them from a file checks them first.
inline void foldeq_require(const Codes &codes, const IndexReader &key, const FoldeqShape &S) {
  const char *fault = foldeq_fault(codes, S);
  BLK_REQUIRE(!fault, fault ? fault : "");
  fault = key.fault();
  BLK_REQUIRE(!fault, fault ? fault : "");
  fault = record_fault(S.index, key.record());
  // The reader's shape must equal the bound one in every bound member (all but `distances`), so
  // that the reader reads the table as the first query describes it.
  BLK_REQUIRE(!fault && key.shape().code == S.index.code && key.shape().v == S.index.v && key.shape().answers == S.index.answers &&
                  key.shape().salt_bytes == S.index.salt_bytes,
              fault ? fault : "foldeq: another table's key");
}

// The block for one table, prover's side. Not checked here: gamma was drawn on a chain whose first
// query contains the table's commitment and record, and index.begin(S.columns) has returned true.
// test: as in priced_table(). g: a group of G_u's code, which G_u and the claims join. blocks: as
// in foldeq_draft(). with: further oracle messages for G_u's prover message; with_ids (may be null)
// receives their numbers.
inline FoldedRow foldeq_prove(ProverContext &P, IndexHolder &index, const IndexRecord &record, const FoldeqShape &S, TableTest test, const E *gamma, GroupId g,
                              BlockSums *blocks, std::vector<OracleDraft> with = {}, std::vector<OracleId> *with_ids = nullptr) {
  const std::vector<E> eq_gamma = foldeq_weights(S, gamma);
  std::vector<OracleDraft> drafts;
  drafts.push_back(foldeq_draft(P, index, record, S, gamma, blocks));
  const IndexRecord &rec = record;
  for (OracleDraft &d : with) drafts.push_back(std::move(d));
  const std::vector<OracleId> ids = send_prove(P, std::move(drafts));
  if (with_ids) with_ids->assign(ids.begin() + 1, ids.end());
  const FoldedRow at = foldeq_joined(P.st, S, index.name(), rec, eq_gamma, ids[0], g);
  auto from = P.ch.close_positions(R_LOC);
  from.require_table(priced_table(S.index, test));
  foldeq_columns_prove(P, index, S, eq_gamma, from.positions(S.index.code.log_n), at, g);
  return at;
}

// The verifier's side. Not checked here, and required of the caller: gamma was drawn on V's chain,
// whose first query contains S (but for IndexShape::distances, which no first query contains) and
// the commitment and record (`IndexName`) of the table that `table` reads, and the reader's
// configuration() is that chain's. `IndexRecord` gives the reason for the record. LIN's
// lin_shared_verify() (blocks/iop/lin.h) checks or arranges all of these.
// False: the proof is rejected. True: the claims on G_u, now in g, are still to be tested.
// at: receives G_u's number and placement. table: for a later test a `TestedTable`; otherwise the
// process ends where the columns are read, unless the proof is rejected earlier.
// Contract: see blocks/chain/closed.h, above run_verifier().
inline bool foldeq_verify(VerifierContext &V, const TableRead &table, const FoldeqShape &S, TableTest test, const E *gamma, GroupId g, FoldedRow &at,
                          const std::vector<OracleNotice> &with = {}, std::vector<OracleId> *with_ids = nullptr) {
  BLK_REQUIRE(test != TableTest::LINK, "foldeq: a link reads two tables: call link_verify() (iop/link.h)");
  const IndexReader &key = table.reader();
  foldeq_require(V.st.codes(), key, S);
  const std::vector<E> eq_gamma = foldeq_weights(S, gamma);
  std::vector<OracleNotice> notices{OracleNotice{S.u, S.leaf_class}};
  notices.insert(notices.end(), with.begin(), with.end());
  std::vector<OracleId> ids;
  if (!send_verify(V, notices, ids)) return false;
  if (with_ids) with_ids->assign(ids.begin() + 1, ids.end());
  at = foldeq_joined(V.st, S, key.name(), key.record(), eq_gamma, ids[0], g);
  V.note("FOLDEQ: e_0", eq_gamma[0]);
  if (V.notes) V.note("FOLDEQ: (Rec_0)'s target", foldeq_record_row(S, record_seeds(S.index, key.name())[0], key.record(), 0, eq_gamma, at).target);
  auto from = V.ch.close_positions(R_LOC);
  from.require_table(priced_table(S.index, test));  // must be the plan's table: see `priced_table`
  return foldeq_columns_verify(V, table, test, test == TableTest::LATER ? TableStanding::ALREADY_TESTED : TableStanding::UNDER_TEST, S, eq_gamma, from.positions(S.index.code.log_n), at, g);
}

}  // namespace blk
