// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// SEND, a block of the protocol. The prover sends oracle messages: main rows (data) and mask rows
// (random values), each row encoded as a codeword; the proof has the roots of their Merkle trees.
// The verifier replies with the seed (a point z and a scalar beta) of an out-of-domain sample. The
// prover begins its next message with one answer for each oracle message: the value of a linear
// form, which the seed fixes, on the entire message that the oracle message encodes. This is to pin
// one message for each oracle message (explained at `Send`). The coordinates of z, beta and the
// answers are in E, the cubic extension of the field F. An oracle message whose
// OracleShape::answers (blocks/iop/relation.h) is s gets s answers, under s seeds; one with 0 is
// sent but not sampled. A call in which every oracle message has 0 answers ends the process
// ("send: nothing to sample"): such messages are sent alone, with send_oracles() and
// receive_oracles().
// Callers use send_prove() and send_verify(). Code that fills a table for send_prove() leaves 0 in
// the pad's entries (`place` in identifiers and messages, but for two local variables of that name,
// which hold the position inside the row); the pad is the random element that masks an answer
// (Send::pad()).
//
// Errors. SEND has two rounds (one, if the caller closes the answers' round: `AnswersRound`).
//    The error of the seed's round is the sum over the sampled oracle
//    messages of (L^2 / 2) * e. L bounds the number of codewords within the test's distance of any
//    word (blocks/numbers/codes.h); a is the dimension of the message's cube (`SampleCube`). The
//    factor L^2 / 2, a union over pairs, is Lemma 9.1 of Chiesa, Fenzi, Weissenberg, "Zero-knowledge IOPPs
//    for constrained interleaved codes" (ePrint 2026/391, version of February 25, 2026), proved there.
//    e = max(a / |E|, 1 / (|E| - 1)) bounds the chance that the form is 0 on a fixed non-zero
//    difference of two messages, by a short argument, which this file's writer wrote down itself:
//    if they differ in the answer's pad, at most one beta fits each z;
//    else, if they differ outside the pads, that part of
//    the form is a non-zero multilinear polynomial in z.
//    (The form is a variant of Lemma 9.3 there (ePrint 2026/391, version of February 25, 2026), which gives the same bound by the same two cases, for one answer.) With s
//    answers, which have s independent seeds and pads, e is taken to the power s: if they differ in pads
//    alone, the form of an answer whose pad differs is beta times the difference, never 0. The round of the answers,
//    after which the verifier's message is empty: 0 (with AnswersRound::OWN; else the caller closes
//    that round, with a message and an error of its own).
//    Computed by plan_send(), blocks/numbers/blocks.h.
//
// Follows the lemma on out-of-domain samples and, for the pad under beta and the bound e, the lemma
// on private zero-evaders and the complete zero-knowledge IOPP, of
// the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "chain/rounds.h"
#include "engine/table.h"
#include "field/field.h"
#include "iop/relation.h"
#include "iop/row_value.h"
#include "numbers/blocks.h"

namespace blk {

// Why a sample. Beyond half the minimum distance a committed word can be close to several
// codewords, and a prover could argue from one of them in one place and from another elsewhere. For
// all but a small fraction of the seeds, two different messages near one word give different
// answers, and the seed is drawn after the word is fixed. So from the answer on, "the message of
// this oracle message" is one definite thing: that of the nearest codeword, within the distance of
// its later proximity test, that agrees with the answer.
// SEND itself does not check that an answer is right. It keeps the equation form(message) = answer
// as a linear claim (`Row` in identifiers), the sample claim, in the `Statement`, the public state
// that both sides keep (blocks/iop/relation.h). Statement::begin_run() passes it to every proximity
// test on the oracle message.
//
// Orders that must hold, and what keeps them (tested by blocks/iop/tests/send_orders_test.cpp):
//   - Oracle message, then seed, then answer: send_prove() and send_verify() do the three steps in
//     one call.
//   - The seed shares its verifier's message with nothing: no positions, and no point at which
//     something will be folded and compared with the answer. A prover that knows such a point
//     before it answers can fit the answer to any nearby codeword. The channel is the object
//     through which a block sends and receives (blocks/chain/rounds.h). The handle that its
//     close_seed() returns can draw only seeds: a program that asks it for more does not compile.
//   - Nothing is drawn between seed and answers: the channel accepts only the answers at the start
//     of the next message. An answer that is kept out of the hash chain (the Fiat-Shamir
//     transcript), to be chosen later, is rejected (the test of this, on synthetic data with a
//     malicious prover, is not part of this repository).
//   - Before its answers are sent, an oracle message joins no group (the oracle messages that one
//     proximity test reads), no claim on it enters a group, and none of its mask entries is given
//     out. All three need the mark "sampled", which only `Send` can set (`SendAccess`), and it does
//     so last. Statement::sampled_elsewhere() copies a mark that `Send` has set in another
//     statement; it compares only the two shapes and codes, not the roots.
//
// Further tests in blocks/iop/tests/: send_form_test.cpp (the form, entry by entry, against its
// definition); send_bat_split_test.cpp (honest prover against verifier). Malicious provers that
// give a wrong answer, or a wrong answer and then pads altered to fit it, belong to tests that are
// not part of this repository. Another such test commits to a word close to two codewords: the
// sample pins one.
class Send;

// A pass-key: an object that only `Send` can construct. Statement::sampled() asks for one as an
// argument, and a SampleReceipt is made only through it. This stops mistakes, not intent: the
// functions of `Send` are templates, so a program can pass them a channel of its own that discards
// the answers, or specialize them. The library's channel (blocks/chain/rounds.h) ends the process
// if a round is closed before all answers are sent.
struct SendAccess {
 private:
  friend class Send;
  SendAccess() {}  // written out: with "= default" the struct would be an aggregate, and SendAccess{} would be open to all
  static SampleReceipt receipt(u32 oracle, u32 first, u32 count, u32 rows, unsigned log_rows) {
    return SampleReceipt(oracle, first, count, rows, log_rows);
  }
};

// Who closes the round of the answers. OWN: SEND, with an empty verifier's message (answer_round in
// plan_send()). THE_CALLERS: the block that calls SEND, with a verifier's message of its own.
enum class AnswersRound { OWN, THE_CALLERS };

// How the form indexes the entries of one oracle message: as the points of a cube {0, 1}^a. The
// main table has 2^b rows of length l (row bits low; rows beyond the real ones are zero), so
// A_m = b + log2(l) index bits; the mask table likewise has A_z. With one table only, a is that
// table's number of bits, and entry j has the coefficient eq(j, z), its weight under z. With both,
// a = max(A_m, A_z) + 1: the top bit is 0 for the main table and 1 for the mask table, each table
// is in the low bits, and the bits between are 0.
struct SampleCube {
  bool has_main = false, has_mask = false;
  unsigned main_row_bits = 0, main_bits = 0;  // b and A_m
  unsigned mask_row_bits = 0, mask_bits = 0;  // the same for the mask table
  unsigned bits = 0;                          // a: the form reads the first a coordinates of z
};

// The pad of an answer: one element of E in the message, set apart to mask that answer. It has
// weight 0 under z and the coefficient beta instead.
//   - Every entry outside the pads has its weight under z, whether or not anything reads it. An
//     entry without a weight could differ between two messages that no seed tells apart.
//   - The pad has no weight under z: else, for some seeds, the two weights cancel and the answer is
//     not uniform.
//   - beta is drawn at random and is not 0: with 0 the pad would drop out, the answer would not be
//     masked, and two messages that differ in the pad alone would not be told apart.
//   - No other claim reads the pad (Statement::add_row() checks this): claims have public values.
//   - In a table over E the pad is one entry. In a table over F it is three entries, read through
//     the basis (1, X, X^2) of E over F. (Three entries of E under that basis would not do: the pad
//     (X, -1, 0) would count for nothing, whatever the seed.)
//
// One entry of a pad, at `index` of the table `part`: in the form of answer number `answer` it has
// the coefficient beta * basis.
struct PadPlace {
  Part part;
  u64 index;
  unsigned answer;
  E basis;
};

class Send {
 public:
  static SampleCube cube(const Codes &codes, const OracleShape &s) {
    SampleCube c;
    c.has_main = s.main_rows > 0;
    c.has_mask = s.mask_rows > 0;
    if (c.has_main) c.main_row_bits = ceil_log2(s.main_rows), c.main_bits = c.main_row_bits + codes.base[s.code].log_ell();
    if (c.has_mask) c.mask_row_bits = ceil_log2(s.mask_rows), c.mask_bits = c.mask_row_bits + codes.mask[s.mask_code].log_ell();
    c.bits = c.has_main && c.has_mask ? std::max(c.main_bits, c.mask_bits) + 1 : c.main_bits + c.mask_bits;
    return c;
  }

  // The entries of the pads, by increasing index: the last s.answers entries of the last real mask
  // row or, without mask rows, of the last real main row (three times as many if the main rows are
  // over F).
  static std::vector<PadPlace> pad(const Codes &codes, const OracleShape &s) {
    const SampleCube c = cube(codes, s);
    const Part part = c.has_mask ? Part::MASK : Part::MAIN;
    const size_t rows = c.has_mask ? s.mask_rows : s.main_rows;
    const unsigned row_bits = c.has_mask ? c.mask_row_bits : c.main_row_bits;
    const size_t len = c.has_mask ? codes.mask[s.mask_code].ell : codes.base[s.code].ell;
    const unsigned each = (c.has_mask || s.main_deg == E_DEGREE) ? 1 : E_DEGREE;  // entries that make one element of E
    BLK_REQUIRE(len >= size_t(each) * s.answers, "send: the row is too short for the pad");
    std::vector<PadPlace> places;
    for (unsigned i = 0; i < s.answers; i++)
      for (unsigned t = 0; t < each; t++) {
        const u64 place = len - size_t(each) * s.answers + size_t(each) * i + t;
        places.push_back(PadPlace{part, (rows - 1) + (place << row_bits), i, E::theta(t)});
      }
    return places;
  }

  // How many mask entries a sampled message's `MaskLedger` (blocks/iop/mask.h) gives out: all with
  // an index below the pad's first entry. In an index the row's number is low, so every row gives
  // the ell_zk - answers entries before the pad's, and every row but the last gives one more. To
  // size a message, or to ask whether it has room, call these two functions: do not repeat the
  // rule.
  static u64 free_mask_entries(size_t mask_rows, size_t ell_zk, unsigned answers) {
    if (mask_rows == 0 || ell_zk < answers) return 0;
    return (u64)mask_rows * (ell_zk - answers) + (mask_rows - 1);
  }
  // The smallest number of mask rows that give out `entries` mask entries; 0 if 4,096 are too few.
  static size_t mask_rows_for(u64 entries, size_t ell_zk, unsigned answers) {
    for (size_t rows = 1; rows <= 4096; rows++)
      if (free_mask_entries(rows, ell_zk, answers) >= entries) return rows;
    return 0;
  }

  // Uniform values for an entire mask table, but 0 in the pad's entries, which send_prove() fills.
  static Table fresh_mask(const Codes &codes, const OracleShape &s, Coins &coins) {
    Table t(E_DEGREE, s.mask_rows, codes.mask[s.mask_code].ell);
    coins.fill(t.a.data(), t.a.size());
    if (s.answers > 0)
      for (const PadPlace &p : pad(codes, s)) t.set_entry(p.index, E::zero());
    return t;
  }

  // The linear form of o's sample claim for one answer; the caller sets the value (`target`). Its
  // coefficient is eq(., z) on every entry outside the pads, beta * basis on the pad of this
  // answer, and 0 on the pads of the other answers.
  static Row form(const Statement &S, OracleId o, const SampleSeed &seed, unsigned answer) {
    const OracleShape &s = S.oracle(o).shape;
    const SampleCube c = cube(S.codes(), s);
    BLK_REQUIRE(seed.z.size() >= c.bits, "send: the seed is too short");
    BLK_REQUIRE(!is_zero(seed.beta), "send: beta is 0");
    const E *z = seed.z.data();
    Row row;
    row.name = "sample of " + s.name;
    row.kind = RowKind::SAMPLE;
    Tensor on_main, on_mask;
    if (c.has_main) on_main = Tensor::eq(z, c.main_bits, front(z, c, c.main_bits, false));
    if (c.has_mask) on_mask = Tensor::eq(z, c.mask_bits, front(z, c, c.mask_bits, true));
    for (const PadPlace &p : pad(S.codes(), s)) {
      const E under_z = (p.part == Part::MAIN ? on_main : on_mask).at(p.index);
      row.add_unit(o, p.part, p.index, (p.answer == answer ? seed.beta * p.basis : E::zero()) - under_z);
    }
    if (c.has_main) row.add(o, Part::MAIN, std::move(on_main));
    if (c.has_mask) row.add(o, Part::MASK, std::move(on_mask));
    return row;
  }

  // Seed and answers, for the prover. The oracle messages `ids` are in the prover's open message;
  // those with answers > 0 are sampled. value(row) evaluates a claim's linear form on the prover's
  // messages. The answers go by oracle message, then by number of the answer.
  template <class Channel, class Value>
  static void prove(Channel &ch, Statement &S, const std::vector<OracleId> &ids, Value &&value, AnswersRound round = AnswersRound::OWN) {
    const std::vector<OracleId> todo = to_sample(S, ids);
    const std::vector<SampleSeed> seeds = draw(ch, S, todo);
    for (OracleId o : todo) {
      std::vector<Row> rows;
      for (unsigned i = 0; i < S.oracle(o).shape.answers; i++) {
        Row row = form(S, o, seeds[i], i);
        row.target = value(row);
        ch.send_answer(row.target);
        rows.push_back(std::move(row));
      }
      admit(S, o, std::move(rows));
    }
    if (round == AnswersRound::OWN) ch.close_empty(R_SEND_ANSWER);
  }

  // Seed and answers, for the verifier. False if an answer is missing from the proof or not in
  // canonical form, or if the channel had rejected before. The statement advances all the same (a
  // channel that has rejected gives zeros, and 1 for beta), so that a caller that goes on after a
  // rejection fails no check on the order of calls, which would end the process.
  template <class Channel>
  [[nodiscard]] static bool verify(Channel &ch, Statement &S, const std::vector<OracleId> &ids, AnswersRound round = AnswersRound::OWN) {
    const std::vector<OracleId> todo = to_sample(S, ids);
    const std::vector<SampleSeed> seeds = draw(ch, S, todo);
    bool good = true;
    for (OracleId o : todo) {
      std::vector<Row> rows;
      for (unsigned i = 0; i < S.oracle(o).shape.answers; i++) {
        Row row = form(S, o, seeds[i], i);
        good = ch.recv_answer(row.target) && good;
        rows.push_back(std::move(row));
      }
      admit(S, o, std::move(rows));
    }
    if (round == AnswersRound::OWN) ch.close_empty(R_SEND_ANSWER);
    return good;
  }

 private:
  // The factor that multiplies eq(., z_1 .. z_table_bits) for one of the two tables: it fixes the
  // cube's bits above the table's.
  static E front(const E *z, const SampleCube &c, unsigned table_bits, bool mask) {
    if (!(c.has_main && c.has_mask)) return E::one();
    E f = mask ? z[c.bits - 1] : E::one() - z[c.bits - 1];
    for (unsigned t = table_bits; t + 1 < c.bits; t++) f *= E::one() - z[t];
    return f;
  }

  // Those of `ids` that get answers.
  static std::vector<OracleId> to_sample(const Statement &S, const std::vector<OracleId> &ids) {
    std::vector<OracleId> todo;
    for (OracleId o : ids)
      if (S.oracle(o).shape.answers > 0) {
        BLK_REQUIRE(!S.oracle(o).sampled, "send: an oracle message is sampled once");
        BLK_REQUIRE(todo.empty() || todo.back() < o, "send: oracle messages come in the order of sending");
        todo.push_back(o);
      }
    BLK_REQUIRE(!todo.empty(), "send: nothing to sample");
    return todo;
  }

  // The verifier's message: for each answer number, the coordinates of z and then beta != 0. (Every
  // example configuration has one answer: the line `answers 1` of `policy.txt`.) The oracle
  // messages share these seeds: z has the largest dimension of their cubes, and each message uses
  // the first coordinates of z. The channel records how many answers the prover must send next.
  template <class Channel>
  static std::vector<SampleSeed> draw(Channel &ch, const Statement &S, const std::vector<OracleId> &todo) {
    unsigned bits = 0, answers = 0;
    for (OracleId o : todo) {
      bits = std::max(bits, cube(S.codes(), S.oracle(o).shape).bits);
      answers = std::max(answers, S.oracle(o).shape.answers);
    }
    SeedDraw from = ch.close_seed(R_SEND_SEED);
    std::vector<SampleSeed> seeds;
    for (unsigned i = 0; i < answers; i++) {
      unsigned users = 0;
      for (OracleId o : todo) users += S.oracle(o).shape.answers > i;
      seeds.push_back(from.seed(bits, users));
    }
    return seeds;
  }

  // Called when o's answers are sent or read: from now on claims, groups and masks may name o. With
  // several answers and several mask rows, the `MaskLedger` never gives out a few entries beside
  // the pad; they have their weight under z like all others.
  static void admit(Statement &S, OracleId o, std::vector<Row> rows) {
    const OracleShape &s = S.oracle(o).shape;
    const u32 free_entries = s.mask_rows > 0 ? (u32)free_mask_entries(s.mask_rows, S.codes().mask[s.mask_code].ell, s.answers) : 0;
    std::vector<std::pair<Part, u64>> places;
    for (const PadPlace &p : pad(S.codes(), s)) places.emplace_back(p.part, p.index);
    S.sampled(SendAccess(), o, std::move(rows), std::move(places),
              SendAccess::receipt(o, 0, free_entries, (u32)std::max<size_t>(s.mask_rows, 1), ceil_log2(s.mask_rows)));
  }
};

// For tests of any code that sends a sampled message. one, other: the pads recorded through
// ProverContext::pads (blocks/iop/context.h; Look = PadLook) in two runs on the same data with
// different private randomness (`coins` in identifiers). Returns null, or what is wrong: every
// entry of a pad must differ from 0, and no value may occur at two entries, of one run or of both.
// So no pad is constant, none is left undrawn, and none serves two tables or two runs. No such test
// can see that a draw is exactly uniform: for that, the code has only been read.
template <class Look> inline const char *pads_fault(const std::vector<Look> &one, const std::vector<Look> &other) {
  if (one.empty() || one.size() != other.size()) return "no pad was noted, or the two runs noted different numbers of places";
  std::vector<E> all;
  for (size_t i = 0; i < one.size(); i++) {
    if (one[i].oracle != other[i].oracle || one[i].in_mask_rows != other[i].in_mask_rows || one[i].place != other[i].place) return "the two runs noted different places";
    all.push_back(one[i].value), all.push_back(other[i].value);
  }
  for (size_t i = 0; i < all.size(); i++) {
    if (is_zero(all[i])) return "a pad's place holds 0";
    for (size_t j = 0; j < i; j++)
      if (all[i] == all[j]) return "one value stands at two pads' places";
  }
  return nullptr;
}

// SEND for a prover's context pr (blocks/iop/context.h): draws the pads, then runs the three steps.
// Returns the numbers of the oracle messages, in the order given.
//
// Hiding. If the pad is uniform and enters nothing else that is shown, the answer is uniform,
// whatever the message, because pad -> beta * pad is a bijection of E. But the pad is an entry of
// its row's message, so it also enters:
//   - the row's symbols at opened positions, which the row's random tail must cover (below);
//   - the prover's messages in each proximity test on the row, which that test's masks must cover
//     (blocks/iop/fold.h, blocks/iop/base_case.h);
//   - the new oracle message of a code switch: switch_drafts() (blocks/iop/cs.h) copies into it the
//     pads of the oracle messages that it sends again. The new message's pad and random tails must
//     cover them.
// Only this function draws the pad: it writes fresh private random values into the pad's entries,
// in the mask table or, through the engine (`pr.engine`, which stores the main tables), in the main
// table. It ends the process if one of these entries is not 0, so that a pad overwrites no data.
// After the sending it reads every pad back; a difference ends the process (own-test-pad-kept,
// in the body).
// Send::prove() draws no pad: after a bare send_oracles() (blocks/iop/context.h) it would answer
// with the pads as they were.
// A pad that is never drawn is 0; the answer, which is in the proof, is then an unmasked linear
// form of the message; and no verifier and no test on a proof can notice. Only own-test-pad-kept
// does, and only inside send_prove(). So tests read the entries in the prover's memory
// (pads_fault()). This project's forms of blocks/iop/tests/send_pad_test.cpp for a sender that draws
// no pad, and for one that draws one value and writes it to every entry of every pad, pass only if
// send_verify(), the only verifier in that test, accepts; they are not part of this repository.
// The random tail of every row (random coefficients that are encoded with its entries) must cover
// all positions at which all tests together open it. open_oracles() (blocks/iop/context.h) counts
// them for each tree and ends the process if there are more. Two contexts share a tree's count only
// if one took the tree over by TreeBook::adopt().
// What this achieves for a whole run: docs/limits.md#weight-hiding.
template <class ProverCtx, class Draft>
inline std::vector<OracleId> send_prove(ProverCtx &pr, std::vector<Draft> drafts, AnswersRound round = AnswersRound::OWN) {
  const Codes &codes = pr.st.codes();
  struct Laid {  // [own-test-pad-kept] each message's pads as drawn, in the order of `Send::pad`
    std::vector<std::vector<E>> of;
    std::vector<std::vector<u32>> draw;  // each pad's draw number (`Coins::last_draw`), or NO_DRAW
    ~Laid() {
      for (std::vector<E> &pads : of) wipe(pads.data(), pads.size() * sizeof(E));
    }
  } laid{std::vector<std::vector<E>>(drafts.size()), std::vector<std::vector<u32>>(drafts.size())};
  for (Draft &d : drafts) {
    if (d.shape.answers == 0) continue;  // it is not sampled: no pad
    std::vector<E> &mine = laid.of[size_t(&d - drafts.data())];
    const std::vector<PadPlace> places = Send::pad(codes, d.shape);
    mine.reserve(places.size());  // (no growth: a vector that grows frees a buffer with pads in it, unwiped)
    [[maybe_unused]] std::vector<u32> &numbers = laid.draw[size_t(&d - drafts.data())];
    [[maybe_unused]] auto number = [&pr] {  // the number of the draw that `fresh` has just made
      return pr.coins.last_draw();
    };
    const char *empty = "send: a pad's place holds something: whoever makes a table leaves these places empty";
    [[maybe_unused]] auto fresh = [&pr](bool in_F) {
      return in_F ? E::from(pr.coins.f()) : pr.coins.e();
    };
    if (d.shape.mask_rows > 0) {
      BLK_REQUIRE(d.mask.deg == E_DEGREE && d.mask.rows == d.shape.mask_rows && d.mask.len == codes.mask[d.shape.mask_code].ell,
                  "send: the mask table does not fit the shape");
      for (const PadPlace &p : places) BLK_REQUIRE_NAMED(is_zero(d.mask.entry(p.index)), "own-test-pad-place", empty);
      for (const PadPlace &p : places) mine.push_back(fresh(false)), numbers.push_back(number()), d.mask.set_entry(p.index, mine.back());
    } else {
      std::vector<u64> at;
      for (const PadPlace &p : places) at.push_back(p.index);
      for (const E &x : pr.engine.entries(d.main, at)) BLK_REQUIRE_NAMED(is_zero(x), "own-test-pad-place", empty);
      std::vector<std::pair<u64, E>> drawn;  // over F an element of E is three entries (PadPlace::basis): each is uniform in F
      drawn.reserve(at.size());
      for (u64 index : at) mine.push_back(fresh(d.shape.main_deg == 1)), numbers.push_back(number()), drawn.emplace_back(index, mine.back());
      pr.engine.set_entries(d.main, drawn);
      wipe(drawn.data(), drawn.size() * sizeof drawn[0]);
    }
  }
  const std::vector<OracleId> ids = send_oracles(pr, std::move(drafts));
  // [own-test-pad-kept] A self-check, in every build.
  // Each pad is read back from the tables as the
  // context keeps them (`pr.behind`) and compared with the value that was drawn: with a copy kept
  // at the draw (`laid.of`) and, if the generator numbers its draws, with the draw computed a
  // second time from its number. A pad that was lost, or written to a neighbouring entry, leaves an
  // answer of the message unmasked in a proof that verifies. The pads' indices are computed again,
  // from the shape in the statement, so a sampled message for which no pad was drawn fails too. The
  // GPU engine's own check after a write (write_entries(), prover/engine/engine_gpu.cuh) reads at
  // the address to which it has written, so it cannot see a wrong address. This check draws nothing
  // and sends nothing.
  // It comes after send_oracles(), not right after set_entries(), so that it also sees a pad that
  // was lost while the message was encoded and committed: it reads the tables from which every
  // later answer is computed. Nothing has left the process at either point.
  // Not seen:
  //   - a pad that is overwritten after this check and before the answer is computed (the answer is
  //     then unmasked, but the proof does not verify);
  //   - with a generator that does not number its draws, a value lost right after its draw, before
  //     the copy is taken: the copy and the table then hold the same value;
  //   - a fault in the formula of Send::pad(), which gives the index both for the write and for the
  //     reading back: only the engine's address is computed twice. (This does little harm:
  //     Send::form() takes the index from the same function, for prover and verifier, and an index
  //     that pointed at non-zero data would fail own-test-pad-place.)
  //   - a value that is drawn once and used again (this project's test build of such a sender,
  //     which records no draw's number and is not part of this repository, passes);
  //   - the copies that the engine has encoded by now;
  //   - entries that Engine::with_entries() writes later (the GPU engine does not read these back),
  //     or that the GPU engine writes again when a table is moved (only write_entries() reads those
  //     back).
  // This function wipes the copies of the pads that it makes for this check after use, and
  // `laid.of` also on every return or exception. Engine::entries() clears its own working copies:
  // the GPU engine zeroes its list of cells in GPU memory and the copy on the host; the CPU engine
  // reserves its list in full, so no buffer is freed while the list grows. Not wiped: anything if
  // another failed check ends the process inside this function.
  // Tested by blocks/iop/tests/send_pad_test.cpp (`Astray`), only for pads in main rows; the
  // comparison with the draw also there (`Recalling`, a generator of the test's own), for pads in
  // main and in mask rows.
  bool kept = ids.size() == laid.of.size();  // one oracle message for each draft, in the drafts' order
  for (size_t i = 0; i < laid.of.size() && i < ids.size(); i++) {
    const OracleShape &s = pr.st.oracle(ids[i]).shape;  // not the draft's shape: see above
    std::vector<u64> at;
    bool in_mask_rows = false;
    if (s.answers > 0)
      for (const PadPlace &p : Send::pad(codes, s)) at.push_back(p.index), in_mask_rows = p.part == Part::MASK;
    const std::vector<E> &drawn = laid.of[i];
    kept = kept && drawn.size() == at.size() && laid.draw[i].size() == at.size();
    if (at.empty()) continue;
    std::vector<E> stand;
    stand.reserve(at.size());
    if (in_mask_rows)
      for (u64 index : at) stand.push_back(pr.behind[ids[i]].mask.entry(index));
    else
      stand = pr.engine.entries(pr.behind[ids[i]].main, at);
    kept = kept && stand.size() == at.size();
    for (size_t k = 0; k < stand.size() && k < drawn.size(); k++) kept = (stand[k] == drawn[k]) && kept;  // all of them: no early end
    // The second comparison: with the draw itself, computed again from its number. It sees a value
    // lost right after its draw, which the copy in `laid.of`, taken from the drawing function's
    // return value, shares with the table. `EngineCoins` (prover/hiding/engine_coins.cuh), the
    // generator of cp_prove and cp_commit, computes the draw by a second implementation of the rule
    // that derives the random values. `StreamCoinsT` (blocks/coins/coins.h) uses the code that
    // drew: it sees a value lost or overwritten after its draw, not a fault in how that code
    // applies the rule.
    const size_t elements = !in_mask_rows && s.main_deg == 1 ? 1 : E_DEGREE;  // over F a pad's entry is one element of F
    for (size_t k = 0; k < stand.size() && k < laid.draw[i].size(); k++)
      if (laid.draw[i][k] != Coins::NO_DRAW) kept = pr.coins.not_the_coins(laid.draw[i][k], stand[k].c, elements, {}) == 0 && kept;
    wipe(stand.data(), stand.size() * sizeof(E));
  }
  for (std::vector<E> &pads : laid.of) wipe(pads.data(), pads.size() * sizeof(E));  // now: a failed check runs no destructor
  BLK_REQUIRE_NAMED(kept, "own-test-pad-kept", "internal error (own-test-pad-kept): a one-time pad of an oracle message does not stand at its place as it was drawn. The program or the machine has a fault; nothing of this run may be used");
  if (pr.pads)  // for tests (PadLook, context.h): the pads' entries, read back from where the prover keeps its tables
    for (OracleId o : ids) {
      const OracleShape &s = pr.st.oracle(o).shape;
      if (s.answers == 0) continue;
      for (const PadPlace &p : Send::pad(codes, s))
        pr.pads->push_back({o, p.part == Part::MASK, p.index, p.part == Part::MASK ? pr.behind[o].mask.entry(p.index) : pr.engine.entries(pr.behind[o].main, {p.index})[0]});
    }
  Send::prove(pr.ch, pr.st, ids, [&pr](const Row &row) { return row_value(pr, row); }, round);
  return ids;
}

// [own-test-index-mask, own-test-ranges-mask, own-test-base-mask, own-test-lin-mask,
// own-test-fold-mask, own-test-mask-rows] A self-check
// ("the look behind the sending" in messages) on a message's mask rows, after it is sent. Returns
// how many elements of F in oracle message o's mask table and random tail, as the prover's context
// keeps them, are not the private random values that were drawn for them. The blocks later compute
// from these two tables (values of claims, mask entries given out, messages of proximity tests),
// and a value that was lost between the draw and the sending gives a proof, or a key file, that
// verifies and hides less.
// Not seen: a value lost in the copies that send_oracles() passes to the engine, which are made
// before this check, or in their encoding, from which the symbols at opened positions are taken.
// Not compared: the pad's entries, which send_prove() fills with draws of their own and checks
// itself (own-test-pad-kept).
// open_oracles() (blocks/iop/context.h) calls this check on every message with mask rows that it
// opens (own-test-mask-rows), so an opened message's tail is compared at least there, except in a
// context that did not send the message (lin_second_prove_on_shape(), blocks/iop/lin.h).
// The result is 0 also where nothing is compared, and nothing reports that. The mask table is
// compared only if the code that drew it recorded the draw's number in OracleDraft::mask_draw: only
// make_key_under() (blocks/frame/check_key.h), wf_rows_prove() (blocks/iop/wf.h), lin_draft()
// (blocks/iop/lin.h) and base_mask_make() with base_mask_draft() (blocks/iop/base_case.h) do; the
// callers of Send::fresh_mask() in blocks/iop/foldeq.h, blocks/iop/req.h and blocks/iop/rng.h do
// not: of their messages only the tail is compared, when they are opened. In cp_commit and cp_prove
// no sent message keeps a table that one of these three has drawn: the folded row has no mask rows
// (blocks/frame/lin_frame.h, blocks/iop/link.h), rng_prove() has no caller, and cp_prove puts a row
// of its own in place of the table that ReqProver::first() draws (prover/prover/chain_run.cuh). If
// a program comes to send a message that keeps such a table, the code that drew it must record the
// draw's number, and ReqProver::first() must clear it where it replaces the table. Where this check
// is called on a fold mask's message, only the tail is compared: the table is made from several
// draws, which FOLD compares itself (fold_mask_as_drawn(), blocks/iop/fold.h).
// send_oracles() records the number of the tail's draw. Neither table is compared with a generator
// that gives its draws no number (Coins::NO_DRAW). A program that relies on this check must itself
// require a generator that numbers its draws.
// The committer program cp_commit does so after make_key(), before a byte of the key file leaves
// the process (prover/hash_commitment/hf_commit_once.cuh:
// "the key file's coins are not the program's own generator"), and in the same way for a leaf file
// (`entry` in identifiers and messages; prover/hash_commitment/hf_entry_make.cuh). The prover
// program cp_prove does so for a request's generator at the draw of the request's index-mask
// message (prover/prover/chain_frame.cuh), which it compares in its own code
// (prover/prover/chain_run.cuh). It also does so for a request's and the batch's generators before
// bytes of a proof leave the process (coins_hold(), prover/prover/chain_coins.cuh). Without these
// tests a result of 0 shows nothing.
// It draws nothing. The caller decides what a result other than 0 means.
template <class ProverCtx> inline u64 not_as_drawn(const ProverCtx &pr, OracleId o) {
  const auto &mine = pr.behind[o];
  const OracleShape &s = pr.st.oracle(o).shape;
  BLK_REQUIRE(s.mask_rows > 0, "send: the look behind the sending is for messages with mask rows");
  BLK_REQUIRE(mine.mask.rows == s.mask_rows && mine.mask_tail.rows == s.mask_rows, "send: the look behind the sending comes too late: the context has dropped the message's tables");  // (else nothing would be compared, and 0 returned)
  std::vector<u64> left_out;  // the pad's entries, as indices into the draw
  if (s.answers > 0)
    for (const PadPlace &p : Send::pad(pr.st.codes(), s)) {
      if (p.part != Part::MASK) continue;  // (cannot happen: `Send::pad` uses mask rows if any)
      const u64 row = p.index & ((u64(1) << mine.mask.log_rows) - 1), place = p.index >> mine.mask.log_rows;  // as in Table::entry()
      for (unsigned k = 0; k < E_DEGREE; k++) left_out.push_back((row * mine.mask.len + place) * mine.mask.deg + k);
    }
  std::sort(left_out.begin(), left_out.end());
  return (mine.mask_draw == Coins::NO_DRAW ? 0 : pr.coins.not_the_coins(mine.mask_draw, mine.mask.a.data(), mine.mask.a.size(), left_out)) +
         (mine.tail_draw == Coins::NO_DRAW ? 0 : pr.coins.not_the_coins(mine.tail_draw, mine.mask_tail.a.data(), mine.mask_tail.a.size(), {}));
}

// SEND for a verifier's context vf. If a root is missing, or the channel had rejected before, it
// returns false at once: the oracle messages are then in the statement but not marked as sampled.
// A caller that went on to use them would fail a check for the mark, which ends the process.
template <class VerifierCtx, class Notice>
[[nodiscard]] inline bool send_verify(VerifierCtx &vf, const std::vector<Notice> &notices, std::vector<OracleId> &ids,
                                      AnswersRound round = AnswersRound::OWN) {
  if (!receive_oracles(vf, notices, ids)) return false;
  return Send::verify(vf.ch, vf.st, ids, round);
}

}  // namespace blk
