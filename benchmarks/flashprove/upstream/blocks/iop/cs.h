// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// CS, the code switch ("the switch" below): re-encodes single rows under new codes.
// From: `SingleRows` (single_rows.h), to which FOLD (fold.h) reduces the rows of a group (oracle
//       messages tested together): main row, any kept rows (see KeptTarget), old and fresh mask
//       row, with one linear claim on their messages (`SingleRows::c`, `kept`, `d`, `target`).
// To:   a new oracle message g; open claims on g and the re-sent kept rows, none on the old rows.
//       An open claim's type is `Row`; error messages too say "row" for a claim ("the sample row").
// g (layout: SwitchTarget) and the re-sent kept rows contain the old rows' messages and random
// tails (the random coefficients of their encodings). So an old row's symbol at a position x is a
// public linear form of the new messages: the row of the old code's generator matrix at x (for a
// mask row: of the mask code's, at mask_position() of x). The
// verifier reads the old rows at t random positions; each symbol read becomes a claim on that form.
// If the old rows differ from the encoding of what was sent at more than the old group's distance
// (a fraction of the positions), some claim is false except with probability (1 - distance)^t.
// CS^sigma (cs_sigma_prove(), cs_sigma_verify()): g joins a group of the caller's; the claims stay
// open. CS (cs_prove(), cs_verify()) also batches that group's claims: its proximity test starts.
// The verifier tests no claim here: a false one is for that group's proximity test to catch.
// Messages, errors, hiding: switch_prove(). Why a re-sent message is the old one: switch_rows().
//
// The code switch follows the construction of the zero-knowledge code-switching IOR of
// the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "code/code.h"
#include "iop/bat.h"
#include "iop/context.h"
#include "iop/relation.h"
#include "iop/send.h"
#include "iop/single_rows.h"
#include "numbers/round_names.h"

namespace blk {

// ---------------------------------------------------------------------------------------------------------------------
// Where the new oracle message stores the old rows' messages and random tails
// ---------------------------------------------------------------------------------------------------------------------

// An aligned block of 2^bits rows of one table of g (`part`: the messages of its main rows, or of
// its mask rows). A vector fills it row index first: entry j is in row first_row + j mod 2^bits, at
// offset j div 2^bits.
struct Cut {
  Part part = Part::MAIN;
  unsigned bits = 0;
  u64 first_row = 0;  // a multiple of 2^bits
};

// Kept rows. The old group may have kept members: oracle messages whose main rows FOLD leaves as
// they were sent, because a later part of the protocol reads them again (GroupLayout, relation.h).
// The switch commits to each kept member a second time: in the prover message that contains g, but
// as a separate oracle message with the new group's code, new random tails and its own
// out-of-domain sample (a random linear form of its message, which the prover answers: SEND,
// send.h). Its table is the old one except at the pad, the entries that mask the sample's answer:
// the new pad is at the old pad's indices, and the old pad's values go into g, as do the old random
// tails.
// KeptTarget: the oracle message that re-sends one kept member. It has the old one's number of
// rows, field, message length and number of answers. leaf_class: oracle messages of one prover
// message with the same class share tree leaves (context.h).
struct KeptTarget {
  OracleShape shape;
  unsigned leaf_class = 0;
  bool kept_again = false;  // in the new group its rows are kept rows once more; else ordinary rows
};

// The shape of g, and where g stores the old rows' data. Two vectors go into g's tables, each into
// a Cut:
//   the main row's message;
//   "the rest": the mask rows' messages | the random tails of the main row, then of the kept rows |
//   the random tails of the mask rows | the old pads of the kept members; each in the order of the
//   rows.
// The pad of g's own sample is where SEND puts it; the rest fills the other entries of its Cut, in
// their order (SwitchPlaces). Entries that a vector leaves unused in its Cut are part of g's
// message (the sample gives them a weight); no other claim reads them. The honest prover fills them
// with zeros in CS^sigma and with uniform values in CS, because the argument for hiding that CS
// rests on counts these entries as uniform. The verifier's checks are the same for both: none tells
// which filling a prover used.
struct SwitchTarget {
  OracleShape shape;  // of g
  unsigned leaf_class = 0;
  Cut main;                      // takes the main row's message
  Cut rest;                      // takes the rest
  std::vector<KeptTarget> kept;  // one for each kept member of the old group, in the order of sending
};

// The target of a stage of a proximity test (iopp.h): 2^k main rows, which the main row's message
// fills exactly (iopp_fault() checks that, nothing here), and 2^h mask rows for the rest.
inline SwitchTarget switch_target_stage(std::string name, CodeId code, CodeId mask_code, unsigned k, unsigned h, unsigned answers) {
  SwitchTarget T;
  T.shape.name = std::move(name);
  T.shape.code = code, T.shape.main_rows = size_t(1) << k, T.shape.main_deg = E_DEGREE;
  T.shape.mask_code = mask_code, T.shape.mask_rows = size_t(1) << h;
  T.shape.answers = answers;
  T.main = Cut{Part::MAIN, k, 0};
  T.rest = Cut{Part::MASK, h, 0};
  return T;
}

// The number of row bits and the length of a row of a table of g.
inline unsigned switch_row_bits(const SwitchTarget &T, Part part) {
  return ceil_log2(part == Part::MAIN ? T.shape.main_rows : T.shape.mask_rows);
}
inline size_t switch_row_length(const Codes &codes, const SwitchTarget &T, Part part) {
  return part == Part::MAIN ? codes.base[T.shape.code].ell : codes.mask[T.shape.mask_code].ell;
}
// The index, in its table, of entry j of a vector that fills a Cut. In a table's index the low
// table_row_bits bits are the row, and the bits above them the offset in the row.
inline u64 cut_index(const Cut &cut, unsigned table_row_bits, u64 j) {
  return (cut.first_row + (j & ((u64(1) << cut.bits) - 1))) + ((j >> cut.bits) << table_row_bits);
}

// The kept members of the old group, in the order of sending.
inline std::vector<OracleId> switch_kept(const SingleRows &in) {
  std::vector<OracleId> kept;
  for (size_t i = 0; i < in.layout.members.size(); i++)
    if (in.layout.kept[i]) kept.push_back(in.layout.members[i]);
  return kept;
}

// Where the parts of the rest begin within it, and its length.
struct SwitchParts {
  std::vector<size_t> mask_at;       // the message of mask row j
  size_t main_tail_at = 0;           // the random tail of the main row
  std::vector<size_t> kept_tail_at;  // that of each kept row, by member and row
  std::vector<size_t> mask_tail_at;  // that of mask row j
  std::vector<size_t> old_pad_at;    // the old pad of each kept member
  size_t length = 0;
};
inline SwitchParts switch_parts(const Statement &S, const SingleRows &in) {
  const Codes &codes = S.codes();
  SwitchParts p;
  for (unsigned j = 0; j < in.masks; j++) p.mask_at.push_back(p.length), p.length += codes.mask[in.mask_code].ell;
  p.main_tail_at = p.length, p.length += codes.base[in.code].r;
  for (size_t j = 0; j < in.kept_rows; j++) p.kept_tail_at.push_back(p.length), p.length += codes.base[in.code].r;
  for (unsigned j = 0; j < in.masks; j++) p.mask_tail_at.push_back(p.length), p.length += codes.mask[in.mask_code].r;
  for (OracleId o : switch_kept(in)) p.old_pad_at.push_back(p.length), p.length += Send::pad(codes, S.oracle(o).shape).size();
  return p;
}

// The table indices of the entries that store the rest, in order: those of the rest's Cut that are
// not pad entries. The list may be as long as all 2^bits rows of the Cut, so it is not written out:
// operator[] skips the pad entries that come before entry i.
class SwitchPlaces {
 public:
  SwitchPlaces(const Codes &codes, const SwitchTarget &T) : cut_(T.rest), row_bits_(switch_row_bits(T, T.rest.part)) {
    const u64 room = (u64)switch_row_length(codes, T, T.rest.part) << cut_.bits;
    for (const PadPlace &p : Send::pad(codes, T.shape)) {  // table indices; renumbered below
      const u64 row = p.index & ((u64(1) << row_bits_) - 1), place = p.index >> row_bits_;
      if (p.part == cut_.part && row >= cut_.first_row && row < cut_.first_row + (u64(1) << cut_.bits)) pad_.push_back((row - cut_.first_row) + (place << cut_.bits));
    }
    std::sort(pad_.begin(), pad_.end());
    size_ = (size_t)(room - pad_.size());
  }
  size_t size() const { return size_; }
  u64 operator[](size_t i) const {
    u64 j = i;
    for (u64 p : pad_)
      if (p <= j) j++;
    return cut_index(cut_, row_bits_, j);
  }

 private:
  Cut cut_;
  unsigned row_bits_;
  std::vector<u64> pad_;  // the pad's entries, numbered in the Cut's fill order, increasing
  size_t size_ = 0;
};
inline SwitchPlaces switch_rest_places(const Codes &codes, const SwitchTarget &T) { return SwitchPlaces(codes, T); }

// Ends the process on a mistake in the public configuration: the sizes are fixed long before any
// proof is read.
inline void switch_target_check(const Statement &S, const SingleRows &in, const SwitchTarget &T) {
  const Codes &codes = S.codes();
  const OracleShape &s = T.shape;
  BLK_REQUIRE(in.masks == 2, "CS: the single rows are not those that a fold leaves: a main row, the old mask row and the fresh one");
  BLK_REQUIRE(in.code < codes.base.size() && in.mask_code < codes.mask.size(), "CS: no such code");
  BLK_REQUIRE(s.code < codes.base.size() && s.main_rows > 0 && s.main_deg == E_DEGREE, "CS: the new oracle message needs a code, main rows, and entries in E");
  BLK_REQUIRE(s.answers >= 1, "CS: the new oracle message has no answer to its sample");
  BLK_REQUIRE(T.main.part == Part::MAIN, "CS: the main row's message goes into main rows");
  BLK_REQUIRE((codes.base[in.code].ell & (codes.base[in.code].ell - 1)) == 0, "CS: the old main row's message length is no power of two");
  const u64 main_block = u64(1) << T.main.bits, rest_block = u64(1) << T.rest.bits;
  BLK_REQUIRE(T.main.first_row % main_block == 0 && T.rest.first_row % rest_block == 0, "CS: a block of rows is not aligned");
  // The two blocks are all the rows of g.
  if (T.rest.part == Part::MASK) {
    BLK_REQUIRE(s.mask_code < codes.mask.size(), "CS: no such code");
    BLK_REQUIRE(T.main.first_row == 0 && main_block == s.main_rows && T.rest.first_row == 0 && rest_block == s.mask_rows,
                "CS: the two blocks are not all the rows of the new oracle message");
  } else {
    BLK_REQUIRE(s.mask_rows == 0 && T.main.first_row == 0 && T.rest.first_row == main_block && main_block + rest_block == s.main_rows,
                "CS: the two blocks are not all the rows of the new oracle message");
  }
  BLK_REQUIRE(codes.base[in.code].ell >= main_block && codes.base[in.code].ell <= (u64)codes.base[s.code].ell << T.main.bits,
              "CS: the main row's message does not fit its block");
  BLK_REQUIRE(switch_parts(S, in).length <= switch_rest_places(codes, T).size(), "CS: the rest does not fit its block");
  const std::vector<OracleId> kept = switch_kept(in);
  BLK_REQUIRE(T.kept.size() == kept.size(), "CS: one new oracle message for each kept member of the old group");
  for (size_t i = 0; i < kept.size(); i++) {
    const OracleShape &o = S.oracle(kept[i]).shape, &n = T.kept[i].shape;
    BLK_REQUIRE(o.mask_rows == 0 && n.mask_rows == 0, "CS: a kept oracle message has no mask rows");
    BLK_REQUIRE(n.code < codes.base.size() && n.main_rows == o.main_rows && n.main_deg == o.main_deg && n.answers == o.answers &&
                    codes.base[n.code].ell == codes.base[o.code].ell,
                "CS: a kept oracle message is sent anew in the same shape");
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The claims on the new oracle message
// ---------------------------------------------------------------------------------------------------------------------

// A coefficient vector (`Form`, tensor.h) on the main row's message (2^vars entries), carried to
// g's main table. insert_choice() adds choice variables: index bits that it fixes to a given value,
// so the vector is 0 at all other indices.
inline Form switch_lift(Form c, unsigned vars, const Cut &cut, unsigned table_row_bits, unsigned table_bits) {
  BLK_REQUIRE(vars >= cut.bits, "CS: the vector is shorter than one entry for each row of its block");
  const unsigned above = table_row_bits - cut.bits;
  c.insert_choice(cut.bits, above, cut.first_row >> cut.bits);     // the row bits above the block pick it
  c.insert_choice(vars + above, table_bits - (vars + above), 0);  // places beyond the vector: no weight
  return c;
}

// A stretch of the rest (entries first .. first + count - 1), cut into pieces whose entries are
// equally spaced in g's table; a pad entry in between ends a piece. The pieces follow from the
// public sizes alone, and the claims of every position read use the same stretches, so
// switch_rows() computes them once per switch, not once per claim.
struct SwitchPiece {
  size_t from, to;  // the entries from, .., to - 1 of the stretch
  u64 first, step;  // the table index of entry `from`; the spacing of the indices
};
inline std::vector<SwitchPiece> switch_pieces(const SwitchPlaces &places, size_t first, size_t count) {
  std::vector<SwitchPiece> pieces;
  for (size_t i = 0; i < count;) {
    const u64 begin = places[first + i];
    const u64 step = i + 1 < count ? places[first + i + 1] - begin : 1;
    size_t end = i + 1;
    for (u64 last = begin; end < count && places[first + end] == last + step; end++) last += step;
    pieces.push_back(SwitchPiece{i, end, begin, step});
    i = end;
  }
  return pieces;
}
// Adds explicit coefficients on a stretch of the rest: one `Run` (relation.h) for each piece. Over
// E, the degree-3 extension of the field F: each Run gets its own copy.
inline void switch_add_rest(Row &row, OracleId g, Part part, const std::vector<SwitchPiece> &pieces, const std::vector<E> &coeff) {
  BLK_REQUIRE(coeff.size() == (pieces.empty() ? 0 : pieces.back().to), "CS: not one coefficient for each entry of the stretch");
  for (const SwitchPiece &p : pieces) row.add_run(g, part, p.first, p.step, std::vector<E>(coeff.begin() + p.from, coeff.begin() + p.to));
}
// The same over F, as the entries of a generator matrix are: the Runs share one vector, which is
// never changed. At one position the main row and all kept rows have the same coefficients on their
// random tails, each on its own stretch, so the vector is made once per position.
using SwitchCoefficients = std::shared_ptr<const std::vector<F>>;
inline void switch_add_rest(Row &row, OracleId g, Part part, const std::vector<SwitchPiece> &pieces, const SwitchCoefficients &coeff) {
  BLK_REQUIRE(coeff && coeff->size() == (pieces.empty() ? 0 : pieces.back().to), "CS: not one coefficient for each entry of the stretch");
  for (const SwitchPiece &p : pieces) row.add_run(g, part, p.first, p.step, coeff, p.from, p.to - p.from);
}

// Whether entries first .. first + 2^vars - 1 of the rest form an aligned box in their table's
// index: spaced 2^shift apart, from a start whose bits shift .. shift + vars - 1 are 0. A product
// form on these entries (a coefficient vector that is a product of one factor for each bit of the
// index) is then one `Tensor` (tensor.h) on the table's variables (its index bits).
struct SwitchBox {
  bool is = false;
  unsigned shift = 0;
  u64 start = 0;
};
template <class Places> inline SwitchBox switch_box(const Places &places, size_t first, unsigned vars) {
  SwitchBox b;
  const size_t count = size_t(1) << vars;
  if (vars == 0 || first + count > places.size()) return b;
  const u64 step = places[first + 1] - places[first];
  if (step == 0 || (step & (step - 1))) return b;
  for (size_t i = 1; i < count; i++)
    if (places[first + i] != places[first] + i * step) return b;
  b.shift = ceil_log2(step), b.start = places[first];
  b.is = ((b.start >> b.shift) & (count - 1)) == 0;
  return b;
}
// Extends v to the table's variables. v: on vars variables, the bits of an entry's number within
// the box.
inline Tensor switch_on_box(const SwitchBox &b, Tensor v, unsigned vars, unsigned table_bits) {
  v.insert_choice(0, b.shift, b.start & ((u64(1) << b.shift) - 1));
  v.insert_choice(b.shift + vars, table_bits - b.shift - vars, b.start >> (b.shift + vars));
  return v;
}

// Carries a form c on the main table of an old kept member over to the claim `row`: the same
// coefficients on the same entries of `anew`, except at the pad's entries, where `anew` has its own
// pad; what c has there goes to the old pad's entries in g.
inline void switch_add_kept(Row &row, const std::vector<PadPlace> &pad, OracleId anew, OracleId g, Part rest_part,
                            const SwitchPlaces &places, size_t old_pad_at, const Form &c) {
  row.add_form(anew, c);
  row.units.reserve(row.units.size() + 2 * pad.size());
  for (size_t i = 0; i < pad.size(); i++) {
    const E there = c.at(pad[i].index);
    row.add_unit(anew, Part::MAIN, pad[i].index, -there);
    row.add_unit(g, rest_part, places[old_pad_at + i], there);
  }
}

// A claim that reads only the main table of oracle message o, as a form on that table. Ends the
// process if the claim reads anything else.
inline Form switch_form_on(OracleId o, const Row &row) {
  Form c;
  auto here = [&](OracleId oracle, Part part) { BLK_REQUIRE(oracle == o && part == Part::MAIN, "CS: the sample row of a kept oracle message reads something else than its main table"); };
  for (const Term &t : row.terms) here(t.oracle, t.part), c.add(t.v);
  for (const Unit &u : row.units) here(u.oracle, u.part), c.add_entry(u.index, u.coeff);
  for (const Run &r : row.runs) {
    here(r.oracle, r.part);
    for (size_t j = 0; j < r.size(); j++) c.add_entry(r.first + r.step * j, r.at(j));
  }
  for (const MainForm &f : row.forms) here(f.oracle, Part::MAIN), c.add(f.c, E::one());
  return c;
}

// Why a re-sent kept message is the one that was sent before (the claims named here are listed
// after the argument). Let w be a kept member's old word (what its old root commits to), and K the
// message on which its claims were tested so far: that of the codeword near w that fits the old
// sample's answer. "w's list" is the set of codewords within the group's distance of w.
//   1. The root over w was sent before the old seed (the challenge that fixes the old sample's
//      form): w's list was fixed when that seed was drawn. So, except with the error of the old
//      seed's round, no two codewords of w's list with different messages have the same old sample.
//   2. The prover chooses the new oracle message and g knowing the old seed, the old answer and
//      every challenge under which the old claim was batched and folded. As tests on a word that it
//      is free to choose, the carried claim and the old sample claim are therefore worth nothing:
//      two linear conditions known in advance. (It can even solve the second for the old pad in g,
//      whose coefficient is not 0.)
//   3. But the new oracle message and g are sent, sampled and answered before the positions are
//      drawn. Except with the error of the new seed's round, each is pinned to one message. So
//      the old code's encoding of (the new message, with the old pad from g; the old random tail
//      from g) is one fixed word before the positions are known.
//   4. The kept rows' generator claims compare that word with w at the t positions. All single rows
//      are read at one list of positions: if any of them is farther than the distance from the
//      encoding of what was re-sent, so are all of them read jointly, and the one event "no draw
//      meets a difference" has probability at most (1 - distance)^t, with no factor for a list size
//      and no sum over the kept rows. So the fixed word is in w's list. (Where kept rows are over
//      F, their old pad and random tails are still entries of E in g. That leaves no freedom: the
//      parts of the fixed word outside F would be codewords that vanish wherever it agrees with w,
//      at more positions than the code's dimension, so they are 0.)
//   5. On w's list the old sample claim is the old sample: by step 1 it holds for K only. So the
//      new message is K outside the pad, and the old pad in g is K's. The old seed need not be
//      fresh for the new word: it is fresh for w's list, which is older.
// The errors used: the old seed's round; the positions' round on the old group; the new seed's
// round; the new group's batching, which counts all of these claims (blocks/iop/req.h compares
// their number with that in the plan, the list of all rounds and their errors). The argument needs
// one distance and one list size for the old and the new group: the frame of a request's proof (its
// chain of segments with one proximity test at the end), in that file, has that. A message kept
// over several switches is tied step by step, each time to the word and the sample of the step
// before.
// The old sample claim is also inside the carried claim, batched and folded. That copy would tie
// the message too, with error factors that the plan includes anyway, but only through the count in
// a fold round's error (how many challenges can make far words look close). Beyond a third of the
// minimum distance that count rests on the theorem on codes (taken as given; no proof of it
// verified by us). As a claim of its own, the old sample claim lets
// step 5 rest on the list bound alone (how many codewords are near a word), which needs only the
// code's minimum distance. This argument is our own; no one has refereed it.
//
// The claims that a switch makes, in this order; the same function on both sides:
//   - the old claim, carried over to where g stores its rows' messages;
//   - the old sample claims of the kept members, carried over in the same way;
//   - for each of the t draws in their order, one generator claim (a row of the old code's
//     generator matrix against the symbol read; for a mask row, of the mask code's) for the
//     main row, each kept row, the old mask row,
//     the fresh mask row. A position drawn twice is opened once and gives its claims twice: their
//     number is fixed.
// The sample claim of g is not among them: SEND leaves it in the statement. For batching,
// Statement::begin_run() (blocks/iop/relation.h) puts a group's claims in this order: plain claims,
// its members' sample claims, generator claims (`RowKind::GENERATOR`). anew: the oracle messages
// that re-send the kept members. y: the symbols read, ordered as by single_rows_symbols().
inline std::vector<Row> switch_rows(const Statement &S, const SingleRows &in, const SwitchTarget &T, OracleId g,
                                    const std::vector<OracleId> &anew, const PositionList &at, const std::vector<E> &y) {
  const Codes &codes = S.codes();
  BLK_REQUIRE(y.size() == at.size() * in.rows(), "CS: one symbol for each row at each listed position");
  const SwitchParts parts = switch_parts(S, in);
  const std::vector<OracleId> kept = switch_kept(in);
  BLK_REQUIRE(anew.size() == kept.size(), "CS: one new oracle message for each kept member of the old group");
  std::vector<std::vector<PadPlace>> pad;
  for (OracleId o : kept) pad.push_back(Send::pad(codes, S.oracle(o).shape));
  const SwitchPlaces places(codes, T);
  const unsigned row_bits = switch_row_bits(T, Part::MAIN);
  const unsigned table_bits = row_bits + codes.base[T.shape.code].log_ell();
  const Code &code = codes.base[in.code];
  std::vector<Row> rows;
  // A row of the mask code's generator matrix is a product form on the mask row's message. Where g
  // stores that message in a box (SwitchBox), the claim takes the form as one Tensor; otherwise
  // coefficient by coefficient.
  const Code &zk = codes.mask[in.mask_code];
  const bool zk_is_a_cube = in.masks > 0 && (zk.ell & (zk.ell - 1)) == 0;
  const unsigned rest_bits = S.table_bits(g, T.rest.part);
  std::vector<SwitchBox> box(in.masks);
  for (unsigned j = 0; zk_is_a_cube && j < in.masks; j++) box[j] = switch_box(places, parts.mask_at[j], ceil_log2(zk.ell));
  // The stretches of the rest that the claims below read, each cut into its pieces once.
  const std::vector<SwitchPiece> main_tail = switch_pieces(places, parts.main_tail_at, code.r);
  std::vector<std::vector<SwitchPiece>> kept_tail, mask_message, mask_tail;
  for (size_t at_rest : parts.kept_tail_at) kept_tail.push_back(switch_pieces(places, at_rest, code.r));
  for (unsigned j = 0; j < in.masks; j++) {
    mask_message.push_back(box[j].is ? std::vector<SwitchPiece>() : switch_pieces(places, parts.mask_at[j], zk.ell));  // used only where the message is in no box
    mask_tail.push_back(switch_pieces(places, parts.mask_tail_at[j], zk.r));
  }
  size_t sample_rows = 0;
  for (OracleId o : kept) sample_rows += S.oracle(o).sample_rows.size();
  rows.reserve(1 + sample_rows + at.draws().size() * in.rows());

  {
    Row old;  // made even if all its forms are empty: "0 = target" is a claim
    old.name = "claim carried through a code switch";
    old.add_form(g, switch_lift(in.c, code.log_ell(), T.main, row_bits, table_bits));
    for (const KeptForm &f : in.kept) {
      const size_t i = (size_t)(std::find(kept.begin(), kept.end(), f.oracle) - kept.begin());
      BLK_REQUIRE(i < kept.size(), "CS: the claim reads a kept member that the group does not have");
      switch_add_kept(old, pad[i], anew[i], g, T.rest.part, places, parts.old_pad_at[i], f.c);
    }
    for (unsigned j = 0; j < in.masks; j++) switch_add_rest(old, g, T.rest.part, switch_pieces(places, parts.mask_at[j], in.d[j].size()), in.d[j]);
    old.target = in.target;
    rows.push_back(std::move(old));
  }
  // The old sample claims of the kept members, each as a claim of its own, in the order of sending:
  // step 5 above switch_rows().
  for (size_t i = 0; i < kept.size(); i++)
    for (const Row &sample : S.oracle(kept[i]).sample_rows) {
      Row old;
      old.name = "old sample row of a kept oracle message";
      switch_add_kept(old, pad[i], anew[i], g, T.rest.part, places, parts.old_pad_at[i], switch_form_on(kept[i], sample));
      old.target = sample.target;
      rows.push_back(std::move(old));
    }
  for (u64 x : at.draws()) {
    const E *symbols = &y[at.index_of(x) * in.rows()];
    auto finish = [&](Row &gen, size_t row) {
      gen.name = "generator row";
      gen.kind = RowKind::GENERATOR;
      gen.target = symbols[row];
      rows.push_back(std::move(gen));
    };
    // The main row and the kept rows are under one code and are read at x itself (the mask rows
    // below: at mask_position() of x).
    const GeneratorRow at_x = generator_at(code, x);
    Form on_message;
    on_message.add(Tensor::product(at_x.f, code.log_ell(), E::from(at_x.message_scale)));
    const auto tail = std::make_shared<std::vector<F>>(code.r);
    generator_row(code, x, nullptr, tail->data());
    const SwitchCoefficients on_tail = tail;
    {
      Row gen;
      gen.add_form(g, switch_lift(on_message, code.log_ell(), T.main, row_bits, table_bits));
      switch_add_rest(gen, g, T.rest.part, main_tail, on_tail);
      finish(gen, 0);
    }
    size_t kept_row = 0;
    for (size_t i = 0; i < kept.size(); i++)
      for (size_t b = 0; b < S.oracle(kept[i]).shape.main_rows; b++, kept_row++) {
        Row gen;
        Form v = on_message;
        v.insert_choice(0, S.row_bits(kept[i], Part::MAIN), b);  // row b of the member's table
        switch_add_kept(gen, pad[i], anew[i], g, T.rest.part, places, parts.old_pad_at[i], v);
        switch_add_rest(gen, g, T.rest.part, kept_tail[kept_row], on_tail);
        finish(gen, 1 + kept_row);
      }
    for (unsigned j = 0; j < in.masks; j++) {
      const u64 x_zk = mask_position(codes.base[in.code], zk, x);
      const auto zk_message = std::make_shared<std::vector<F>>(box[j].is ? 0 : zk.ell), zk_tail = std::make_shared<std::vector<F>>(zk.r);
      generator_row(zk, x_zk, box[j].is ? nullptr : zk_message->data(), zk_tail->data());
      Row gen;
      if (box[j].is) {
        const GeneratorRow at_zk = generator_at(zk, x_zk);
        gen.add(g, T.rest.part, switch_on_box(box[j], Tensor::product(at_zk.f, ceil_log2(zk.ell), E::from(at_zk.message_scale)), ceil_log2(zk.ell), rest_bits));
      } else {
        switch_add_rest(gen, g, T.rest.part, mask_message[j], SwitchCoefficients(zk_message));
      }
      switch_add_rest(gen, g, T.rest.part, mask_tail[j], SwitchCoefficients(zk_tail));
      finish(gen, in.mask_row(j));
    }
  }
  return rows;
}

// What a switch returns, on both sides.
struct Switched {
  OracleId g = NONE_ID;
  std::vector<OracleId> kept;  // the oracle messages that re-send the kept members
  std::vector<OracleId> with;  // the other oracle messages of g's prover message
  PositionList at;             // where the old rows were read
};

// Splits the oracle messages of the switch's prover message, given in the order of sending: g,
// those that re-send kept members, the others.
inline void switch_sort(const std::vector<OracleId> &ids, const SwitchTarget &T, Switched &out) {
  out.g = ids[0];
  out.kept.assign(ids.begin() + 1, ids.begin() + 1 + T.kept.size());
  out.with.assign(ids.begin() + 1 + T.kept.size(), ids.end());
}

// Pass-key for switch_admit(). Its constructor is private, and the only friends are switch_prove()
// and switch_verify(), which make one after they have run SEND and then drawn the positions.
class SwitchAccess {
  SwitchAccess() {}  // written out: with "= default" the class would be an aggregate under C++17, and SwitchAccess{} would be open to all
  friend Switched switch_prove(ProverContext &, const SingleRows &, const SingleRowsWitness &, const SwitchTarget &, GroupId,
                               std::vector<OracleDraft>, E *);
  friend bool switch_verify(VerifierContext &, const SingleRows &, const SwitchTarget &, GroupId, const std::vector<OracleNotice> &, E *,
                            Switched &);
};

// g and the re-sent kept members join the group `to`, and the claims become its open claims. The
// same calls on both sides.
inline void switch_admit(const SwitchAccess &, Statement &S, const SingleRows &in, const SwitchTarget &T, GroupId to, const Switched &out,
                         const std::vector<E> &y) {
  if (T.shape.mask_rows > 0) S.masks(out.g).close();  // g's mask rows store the rest, not masks
  S.join(to, out.g);
  for (size_t i = 0; i < out.kept.size(); i++) S.join(to, out.kept[i], T.kept[i].kept_again);
  for (Row &row : switch_rows(S, in, T, out.g, out.kept, out.at, y)) S.add_row(to, std::move(row));
}

// ---------------------------------------------------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------------------------------------------------

// The tables to send (`OracleDraft`, context.h) for g and, after it, for the oracle messages that
// re-send the kept members, from the prover's witness for the old rows. uniform_room: unused
// entries get uniform values, not zeros (SwitchTarget).
inline std::vector<OracleDraft> switch_drafts(ProverContext &P, const SingleRows &in, const SingleRowsWitness &W, const SwitchTarget &T,
                                              bool uniform_room) {
  const Codes &codes = P.st.codes();
  const SwitchParts parts = switch_parts(P.st, in);
  const SwitchPlaces places(codes, T);
  const size_t ell = codes.base[in.code].ell;
  BLK_REQUIRE(W.mask.size() == in.masks && W.mask_tail.size() == in.masks && W.main_tail.len == codes.base[in.code].r,
              "CS: the witness does not fit the rows");

  const std::vector<OracleId> kept = switch_kept(in);
  std::vector<OracleDraft> anew;
  std::vector<E> rest, old_pads;
  rest.reserve(uniform_room ? places.size() : parts.length);
  for (const std::vector<E> &m : W.mask) rest.insert(rest.end(), m.begin(), m.end());
  for (size_t i = 0; i < W.main_tail.len; i++) rest.push_back(W.main_tail.at(0, i));
  for (size_t i = 0; i < kept.size(); i++) {
    const TreeBook::Place place = P.book.place(kept[i]);
    const Table tail = P.engine.tail(P.commit[place.tree], place.part);
    for (size_t b = 0; b < tail.rows; b++)
      for (size_t j = 0; j < tail.len; j++) rest.push_back(tail.at(b, j));
    // The kept table stays with the engine (`Engine`, blocks/engine/engine.h, stores the prover's
    // main tables). Its old pad is read, and what is sent is a copy with zeros at the pad's
    // entries: send_prove() draws the new pad there and ends the process unless all are 0.
    std::vector<u64> pad_at;
    for (const PadPlace &p : Send::pad(codes, P.st.oracle(kept[i]).shape)) pad_at.push_back(p.index);
    for (const E &x : P.engine.entries(P.behind[kept[i]].main, pad_at)) old_pads.push_back(x);
    OracleDraft d;
    d.shape = T.kept[i].shape, d.leaf_class = T.kept[i].leaf_class;
    std::vector<std::pair<u64, E>> emptied;
    for (u64 at : pad_at) emptied.emplace_back(at, E::zero());
    d.main = P.engine.with_entries(P.behind[kept[i]].main, emptied);
    anew.push_back(std::move(d));
  }
  for (const std::vector<E> &t : W.mask_tail) rest.insert(rest.end(), t.begin(), t.end());
  rest.insert(rest.end(), old_pads.begin(), old_pads.end());
  BLK_REQUIRE(rest.size() == parts.length, "CS: the witness does not fit the codes");
  // The unused entries after the rest: uniform where the caller asks for that, otherwise 0. Zeros
  // are not written out, and no private random value (`coins` in identifiers) is drawn for them: a
  // new Table is all zeros, and the table that the engine makes below is zero except for the row
  // placed in it (its contract in blocks/engine/engine.h). Between two segments of a request (a
  // segment is the part of its proof for one sub-proof; blocks/iop/req.h) the rest's Cut is an
  // entire main row, 2^22 entries at full size (prover/examples/target_full_f2), and the rest has
  // about 20,000; writing the zeros entry by entry took more than half of the time of a full-size
  // request's first chain (its own hash chain, before the batch's shared chain): one run with that
  // step, one without, under the same private random values, in builds with timing prints; the same
  // proof, byte for byte.
  BLK_REQUIRE(rest.size() <= places.size(), "CS: the rest does not fit its rows");
  if (uniform_room)
    while (rest.size() < places.size()) rest.push_back(P.coins.e());

  OracleDraft d;
  d.shape = T.shape, d.leaf_class = T.leaf_class;
  // whole: the rest is in mask rows, and the main row's message fills g's main rows exactly.
  bool whole = T.rest.part == Part::MASK && ell == (u64)codes.base[T.shape.code].ell << T.main.bits;
  if (T.rest.part == Part::MASK) {
    d.mask = Table(E_DEGREE, T.shape.mask_rows, codes.mask[T.shape.mask_code].ell);
    for (size_t i = 0; i < rest.size(); i++) d.mask.set_entry(places[i], rest[i]);  // SEND draws the pad
  }
  if (whole) {
    d.main = P.engine.reshape(W.main, T.main.bits);  // the same vector, cut anew: the engine's step
    anew.insert(anew.begin(), std::move(d));
    return anew;
  }
  // The general case. The engine places the main row's message; `beside` lists the other entries
  // to write: the unused ones after that message, if uniform; then the rest, if it is in main rows.
  // None is a pad entry (SwitchPlaces skips the pad, which is in the rest's rows): those stay 0,
  // for SEND to draw.
  const unsigned a = switch_row_bits(T, Part::MAIN);
  const size_t len = codes.base[T.shape.code].ell;
  const u64 room = (u64)len << T.main.bits;
  std::vector<std::pair<u64, E>> beside;
  beside.reserve((uniform_room ? (size_t)(room - ell) : 0) + (T.rest.part == Part::MAIN ? rest.size() : 0));
  if (uniform_room)
    for (u64 j = ell; j < room; j++) beside.emplace_back(cut_index(T.main, a, j), P.coins.e());
  if (T.rest.part == Part::MAIN) {
    for (size_t i = 0; i < rest.size(); i++) beside.emplace_back(places[i], rest[i]);
  }
  // No two of these name one entry (the engine would end the process): the unused entries are in
  // the main row's block of rows, the rest in its own (switch_target_check()), and within each the
  // indices differ. Nothing else refers yet to the table that the engine has just made, so the
  // entries are written into it, not into a copy.
  d.main = P.engine.laid(W.main, ell, T.main.first_row, T.main.bits, T.shape.main_rows, len);
  if (!beside.empty()) P.engine.set_entries(d.main, beside);
  anew.insert(anew.begin(), std::move(d));
  return anew;
}

// Both forms; theta is null for CS^sigma. Messages (P: prover, V: verifier):
//   1. P  g, then the oracle messages that re-send kept members, then `with`
//   2. V  the seed of their out-of-domain samples   } with 1: SEND (blocks/iop/send.h), which is
//   3. P  the answers                               } to pin one message for each sampled message
//   4. V  t positions of the old rows; in CS, after them in the same message, theta in E
// Then the prover opens the old rows at the positions. The verifier checks the openings as
// receive_opening() does (blocks/tree/opening.h: their form, their roots, one value a vertex) and
// itself tests no claim on the symbols (they enter the generator claims, which the new group's
// proximity test checks); its side also runs send_verify() and the channel's checks, and both
// rounds may carry proof of work (below).
// Order. SEND keeps messages 1 to 3 in this order. The positions come after the answers: near the
// word that was sent for g there may be several codewords, and a prover that knew the positions
// before it answered could pick the one that fits them. So the function that runs SEND also draws
// the positions (SwitchAccess). theta may share the positions' message because the claims that it
// batches are fixed by what was sent before and by the positions.
// Errors (plan_send(), plan_loc(), plan_cs() in blocks/numbers/blocks.h). The seed's round: SEND's
// error (two codewords near one sent word, with different messages, get the same sample), added up
// over the sampled oracle messages of message 1. The positions' round (`R_LOC` in CS^sigma, `R_CS`
// in CS): (1 - distance)^t, once for all rows that are read there; in CS add
// (number of claims - 1) / |E| for each codeword in the list of the new group's word (the codewords
// within the group's distance of it). Both rounds may carry proof of work.
// Hiding rests on three things. The random tails that encode g are new, uniform, and cover every
// position at which g is opened later. The pad is uniform. The old rows were encoded with random
// tails long enough for the t positions that are read here.
// What is and is not shown about hiding for a whole run: docs/limits.md#weight-hiding.
inline Switched switch_prove(ProverContext &P, const SingleRows &in, const SingleRowsWitness &W, const SwitchTarget &T, GroupId to,
                             std::vector<OracleDraft> with, E *theta) {
  switch_target_check(P.st, in, T);
  std::vector<OracleDraft> drafts = switch_drafts(P, in, W, T, theta != nullptr);
  for (OracleDraft &d : with) drafts.push_back(std::move(d));
  Switched out;
  switch_sort(send_prove(P, std::move(drafts), AnswersRound::THE_CALLERS), T, out);  // the answers are closed by the positions' round

  PositionsDraw from = P.ch.close_positions(theta ? R_CS : R_LOC);
  P.ch.require_codes(group_as_run(P.st.codes(), in.code, in.mask_code, in.masks > 0, 0, 0, 1));  // the plan computes the number of positions and the round's error from two codes: they must be those of the rows that are read here
  out.at = from.positions(P.st.codes().base[in.code].log_n);
  if (theta) *theta = from.batching();
  const std::vector<E> y = single_rows_open(P, in, out.at);

  switch_admit(SwitchAccess(), P.st, in, T, to, out, y);
  return out;
}

// CS^sigma. g joins the group `to`, and the claims become open claims of that group.
// with: the other oracle messages of g's prover message, sampled or not; they are sent last.
inline Switched cs_sigma_prove(ProverContext &P, const SingleRows &in, const SingleRowsWitness &W, const SwitchTarget &T, GroupId to,
                               std::vector<OracleDraft> with = {}) {
  return switch_prove(P, in, W, T, to, std::move(with), nullptr);
}

// CS. Every claim of the group `to` is then batched with the powers of theta: this is BAT (bat.h),
// the first step of a proximity test on the group, with its challenge drawn early. The process ends
// if the number of claims is not the one that the plan counts (require_rows()). row: the resulting
// claim; layout: the group's.
inline Switched cs_prove(ProverContext &P, const SingleRows &in, const SingleRowsWitness &W, const SwitchTarget &T, GroupId to,
                         std::vector<OracleDraft> with, GroupRow &row, GroupLayout &layout) {
  E theta;
  Switched out = switch_prove(P, in, W, T, to, std::move(with), &theta);
  RunInput run = P.st.begin_run(to);
  P.ch.require_rows(run.rows().size());
  layout = run.layout;
  row = Bat::combine(P.st, to, std::move(run), theta);
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------------------------------------------------

// The verifier's side of switch_prove(): the same messages. Returns false, before switch_admit(),
// if send_verify(), the channel or single_rows_read() rejects.
inline bool switch_verify(VerifierContext &V, const SingleRows &in, const SwitchTarget &T, GroupId to, const std::vector<OracleNotice> &with,
                          E *theta, Switched &out) {
  switch_target_check(V.st, in, T);
  std::vector<OracleNotice> notices;
  notices.push_back(OracleNotice{T.shape, T.leaf_class});
  for (const KeptTarget &k : T.kept) notices.push_back(OracleNotice{k.shape, k.leaf_class});
  notices.insert(notices.end(), with.begin(), with.end());
  std::vector<OracleId> ids;
  if (!send_verify(V, notices, ids, AnswersRound::THE_CALLERS)) return false;
  switch_sort(ids, T, out);

  PositionsDraw from = V.ch.close_positions(theta ? R_CS : R_LOC);
  V.ch.require_codes(group_as_run(V.st.codes(), in.code, in.mask_code, in.masks > 0, 0, 0, 1));  // as the prover's side
  out.at = from.positions(V.st.codes().base[in.code].log_n);
  if (theta) *theta = from.batching();
  if (!V.ch.ok()) return false;
  std::vector<E> y;
  if (!single_rows_read(V, in, out.at, y)) return false;
  if (V.notes) {  // the folded symbols at the first draw
    const E *first = &y[out.at.index_of(out.at.draws().front()) * in.rows()];
    V.note("CS: y_main at the first draw", first[0]);
    for (unsigned j = 0; j < in.masks; j++) V.note(j ? "CS: y_1 at the first draw" : "CS: y_0 at the first draw", first[in.mask_row(j)]);
  }

  switch_admit(SwitchAccess(), V.st, in, T, to, out, y);
  return true;
}

// CS^sigma, the verifier's side. The caller's contract: see run_verifier() in
// blocks/chain/closed.h.
inline bool cs_sigma_verify(VerifierContext &V, const SingleRows &in, const SwitchTarget &T, GroupId to, const std::vector<OracleNotice> &with,
                            Switched &out) {
  return switch_verify(V, in, T, to, with, nullptr, out);
}

// CS, the verifier's side; row, layout: as in cs_prove(). The caller's contract: see run_verifier()
// in blocks/chain/closed.h.
inline bool cs_verify(VerifierContext &V, const SingleRows &in, const SwitchTarget &T, GroupId to, const std::vector<OracleNotice> &with,
                      Switched &out, GroupRow &row, GroupLayout &layout) {
  E theta;
  if (!switch_verify(V, in, T, to, with, &theta, out)) return false;
  RunInput run = V.st.begin_run(to);
  V.ch.require_rows(run.rows().size());
  layout = run.layout;
  row = Bat::combine(V.st, to, std::move(run), theta);
  return true;
}

}  // namespace blk
