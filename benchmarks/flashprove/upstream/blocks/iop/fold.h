// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// FOLD, a sumcheck with a hiding polynomial, reduces one linear claim (`GroupRow`, relation.h) on
// the rows of a group to one on three single rows. A group is the oracle messages (its members:
// encoded rows, sent as Merkle roots) that one proximity test reads at one list of positions.
// From: 2^k main rows (messages m) under one code, 2^h mask rows (short, for masks; messages mu)
//       under the mask code, h <= k; unfilled rows are zero. Claim: < c, m > + < d, mu > = target.
// To:   `SingleRows` (single_rows.h): the main rows folded at g_1..g_k, the mask rows and a fresh
//       mask (`FoldMask`) folded at g_1..g_h; one linear claim on their messages.
// Kept rows, the main rows of a member that `GroupLayout` marks as kept, are not among the 2^k
// rows and are not folded; both claims may have terms on them (`KeptForm`): see fold_output().
// Messages (P: prover, V: verifier; the arithmetic is in fold_rounds.h):
//   P  the fresh mask, unless it was sent before; the sum over {0,1}^k of its hiding polynomial U
//   V  eps, one element: the sumcheck runs on U + eps * (the claim's summand)
//   k times:  P  the constant and the quadratic coefficient of the round's polynomial;   V  g_j
// The verifier tests no equation here. A prover that departs from the committed rows is left, up to
// the rounds' errors, with a claim that is false for them; the blocks after FOLD are to reject it.
//
// Follows the construction of the zero-knowledge sumcheck IOR and, for two-coefficient rounds and a mask sent early,
// the complete zero-knowledge IOPP, of the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "base.h"
#include "iop/context.h"
#include "iop/fold_rounds.h"
#include "iop/relation.h"
#include "iop/send.h"
#include "iop/single_rows.h"
#include "numbers/round_names.h"

namespace blk {

// The public parameters of one FOLD, and three options of the prover.
struct FoldShape {
  CodeId code = NONE_ID;       // of the main rows, in Codes::base
  CodeId mask_code = NONE_ID;  // of the mask rows, old and fresh, in Codes::mask
  GroupLayout layout;          // k, h, the members and where their rows are
  unsigned mask_leaf_class = 0;  // of the fresh mask: see FoldMask
  // Prover only. The engine (`Engine`, blocks/engine/engine.h: it holds the main tables and
  // computes on them) keeps the folded coefficient vector as a table, and the new claim refers to
  // it in a hint (`Held`, blocks/iop/tensor.h) for a later FOLD of a group that contains the folded
  // row. The caller must Engine::drop() the table once it passes no more `Form` (a coefficient
  // vector, same file) with the hint to the engine.
  bool keep_folded_coefficients = false;
  // Prover only; needs keep_folded_coefficients. The new claim's coefficient vector is only that
  // table (`EngineTerm`, tensor.h): the prover skips the structured form (a sum of tensors), which
  // grows with every FOLD that a claim passes through. A later FOLD without this flag fetches its
  // folded vector from the engine and writes it out. That must happen before a block that reads
  // the vector outside the engine (BASE, the proximity test's last block); it is short by then.
  bool folded_coefficients_alone = false;
  // Prover only. The caller does not know the claim's target; GroupRow::target may hold anything.
  // FOLD starts its running claim at q(0) + q(1) of its first round's polynomial q: q does not
  // depend on the target, and for a true claim that sum is where the verifier starts. This saves
  // the caller an evaluation of the claim on the committed tables. The messages sent are the same.
  // If the verifier's target is not what the prover's tables give, the running claims differ, and
  // the blocks after FOLD are to reject the proof.
  bool target_from_the_rounds = false;
};
// The table that the engine keeps for c, the coefficient vector of a claim that FOLD produced under
// keep_folded_coefficients.
inline TableId fold_kept_coefficients(const Form &c) {
  BLK_REQUIRE(c.held.size() + c.engine_terms.size() == 1, "FOLD: the claim does not name exactly one kept coefficient vector");
  return c.held.empty() ? c.engine_terms[0].vector : c.held[0].vector;
}

// Ends the process unless the shape fits the codes and each member's rows, taken alone, lie among
// the group's 2^k main and 2^h mask rows. Not checked: that two members' rows do not overlap (where
// they do, single_rows_symbols() reads only the later member's), that a block is aligned, and that
// first row + number of rows does not wrap (NONE can pass for a member that has rows).
// Statement::layout() (relation.h) makes layouts that meet all three.
inline void fold_shape_check(const Statement &st, const FoldShape &S) {
  const Codes &codes = st.codes();
  const GroupLayout &L = S.layout;
  BLK_REQUIRE(S.code < codes.base.size() && S.mask_code < codes.mask.size(), "FOLD: the shape names a code that does not exist");
  BLK_REQUIRE(L.k >= 1 && L.h <= L.k, "FOLD: no row variable, or more mask rows than main rows");
  BLK_REQUIRE(mask_fits(codes.base[S.code], codes.mask[S.mask_code]), "FOLD: the mask code is longer than the main code");
  BLK_REQUIRE(codes.mask[S.mask_code].ell >= fold_mask_columns(L.k, L.h), "FOLD: the mask rows are too short for the hiding polynomial");
  BLK_REQUIRE(L.kept.size() == L.members.size() && L.main_first.size() == L.members.size() && L.mask_first.size() == L.members.size(),
              "FOLD: the layout's lists do not have one entry for each member");
  bool folds_something = false;
  for (size_t i = 0; i < L.members.size(); i++) {
    const OracleShape &o = st.oracle(L.members[i]).shape;
    const bool folded = o.main_rows && !L.kept[i];
    folds_something = folds_something || folded;
    BLK_REQUIRE(!o.main_rows || o.code == S.code, "FOLD: a member's main rows are under another code than the group's");
    BLK_REQUIRE(!o.mask_rows || o.mask_code == S.mask_code, "FOLD: a member's mask rows are under another code than the group's");
    BLK_REQUIRE(folded ? L.main_first[i] + o.main_rows <= u64(1) << L.k : L.main_first[i] == GroupLayout::NONE,
                "FOLD: a member's main rows do not lie among the group's 2^k rows, or rows that are not folded have a place there");
    BLK_REQUIRE(o.mask_rows ? L.mask_first[i] + o.mask_rows <= u64(1) << L.h : L.mask_first[i] == GroupLayout::NONE,
                "FOLD: a member's mask rows do not lie among the group's 2^h rows, or a member without mask rows has a place there");
  }
  BLK_REQUIRE(folds_something, "FOLD: no member of the group has a main row to fold");
}

inline void fold_row_check(const Statement &st, const FoldShape &S, const GroupRow &row) {
  const Codes &codes = st.codes();
  BLK_REQUIRE(row.c.fits(S.layout.k + ceil_log2(codes.base[S.code].ell)), "FOLD: the claim's coefficients on the main rows have not the main table's length");
  BLK_REQUIRE(row.d.size() == codes.mask[S.mask_code].ell << S.layout.h, "FOLD: the claim's coefficients on the mask rows have not the mask table's length");
  for (const KeptForm &f : row.kept) {
    BLK_REQUIRE(S.layout.kept[S.layout.find(f.oracle)], "FOLD: the claim has separate coefficients on a member that is not kept");
    BLK_REQUIRE(f.c.fits(st.table_bits(f.oracle, Part::MAIN)), "FOLD: the claim's coefficients on a kept member have not its table's length");
  }
}

// The fresh mask of one FOLD: an oracle message of 2^h mask rows whose messages are built from the
// coefficients of the hiding polynomial U (fold_mask_table()). The coefficients come only from
// private randomness (`Coins`, blocks/engine/table.h); if the generator numbers its draws, the
// prover compares them with their draws before it sends the sum and the rounds computed from them
// (fold_mask_as_drawn(), which says what it misses); the root of the table's tree has been sent by
// then. On the verifier's side a FoldMask is only the oracle message's id.
// The mask gets no out-of-domain sample and is never part of a sampled oracle message. A sample is
// a claim on all entries of its oracle message; BAT, the batching before FOLD, takes that claim in;
// FOLD's input would then read FOLD's own mask.
// A caller may send the mask ahead, in a prover message before FOLD: make it with fold_mask_make(),
// add fold_mask_draft() (the verifier: fold_mask_notice()) to that message's oracle messages, and
// pass the id that it gets to sent_as(). Otherwise FOLD sends it in its first message.
// The mask shares tree leaves with oracle messages of its prover message that have its leaf class
// (lay_trees(), blocks/iop/context.h). It is read only at the list of positions of the block that
// follows this FOLD, so its class (FoldShape::mask_leaf_class) must be one that is read at that
// list and no other.
// One mask serves one FOLD: a FoldMask cannot be copied, and Statement::spend_as_mask() ends the
// process if its oracle message is named as a mask a second time.
class FoldMask {
 public:
  FoldMask() = default;
  FoldMask(FoldMask &&) = default;
  FoldMask &operator=(FoldMask &&) = default;
  FoldMask(const FoldMask &) = delete;
  FoldMask &operator=(const FoldMask &) = delete;

  bool sent() const { return id_ != NONE_ID; }
  OracleId id() const { return id_; }
  void sent_as(OracleId id) {
    BLK_REQUIRE(!sent(), "FOLD: a fresh mask was told its number twice");
    id_ = id;
  }
  const std::vector<E> &coefficients() const { return w_; }

 private:
  friend FoldMask fold_mask_make(const FoldShape &, Coins &);
  std::vector<E> w_;  // the prover's: the hiding polynomial's coefficients
  OracleId id_ = NONE_ID;
};

inline FoldMask fold_mask_make(const FoldShape &S, Coins &coins) {
  std::vector<E> c(2 * size_t(S.layout.k) + 1);
  for (E &x : c) x = coins.e();
  FoldMask m;
  m.w_ = fold_mask_from_coins(S.layout.k, std::move(c));
  return m;
}
// The same for a prover's context, and what every prover calls: draws from `P.coins` and records
// the number of each coefficient's draw, for fold_mask_as_drawn(). Coefficient i is draw i.
// The numbers are kept in the context and not in the FoldMask, a class that verifiers use too:
// `P.fold_draws`, under the address of the coefficients' buffer. A move of the FoldMask passes that
// buffer on, so the address stays; a FoldMask cannot be copied. A new mask overwrites an entry left
// under its address; fold_rounds_prove() erases its mask's entry.
inline FoldMask fold_mask_make(const FoldShape &S, ProverContext &P) {
  struct Noting : Coins {  // passes every call on and notes the number of each draw of elements
    Coins &from;
    std::vector<u32> draws;
    explicit Noting(Coins &c) : from(c) {}
    void fill(F *out, size_t n) override { from.fill(out, n), draws.push_back(from.last_draw()); }
    void bytes(u8 *out, size_t n) override { from.bytes(out, n); }
  } noting(P.coins);
  FoldMask m = fold_mask_make(S, noting);
  P.fold_draws[m.coefficients().data()] = std::move(noting.draws);
  return m;
}

inline OracleNotice fold_mask_notice(const FoldShape &S) {
  OracleNotice n;
  n.shape.name = "fold mask";
  n.shape.mask_rows = size_t(1) << S.layout.h;
  n.shape.mask_code = S.mask_code;
  n.shape.answers = 0;
  n.leaf_class = S.mask_leaf_class;
  return n;
}
inline OracleDraft fold_mask_draft(const Codes &codes, const FoldShape &S, const FoldMask &mask) {
  const size_t len = codes.mask[S.mask_code].ell;
  const std::vector<E> t = fold_mask_table(S.layout.k, S.layout.h, len, mask.coefficients());
  const OracleNotice n = fold_mask_notice(S);
  OracleDraft d;
  d.shape = n.shape, d.leaf_class = n.leaf_class;
  d.mask = Table(E_DEGREE, n.shape.mask_rows, len);
  for (u64 j = 0; j < t.size(); j++) d.mask.set_entry(j, t[j]);
  return d;
}

// FoldMask::sent_as() accepts any id, so both sides check the oracle named as the fresh mask here.
inline void fold_mask_check(const Statement &st, const FoldShape &S, OracleId fresh) {
  const Oracle &o = st.oracle(fresh);
  BLK_REQUIRE(o.shape.main_rows == 0 && o.shape.mask_rows == size_t(1) << S.layout.h && o.shape.mask_code == S.mask_code,
              "FOLD: the oracle named as the fresh mask has not just 2^h mask rows under the group's mask code");
  BLK_REQUIRE(!o.sampled && o.shape.answers == 0, "FOLD: the oracle named as the fresh mask has an out-of-domain sample");
  for (OracleId m : S.layout.members) BLK_REQUIRE(m != fresh, "FOLD: the oracle named as the fresh mask is a member of the group");
}

// FOLD after its first message: mask and sum have been sent, eps is still to come.
// eps must come after both: a prover that knows eps first can fit the sum to a false target and be
// honest from there on. So the rounds take eps only from a FoldStart, only fold_epsilon() sets it,
// and only fold_start_prove() and fold_start_verify() make one: they send or receive the sum, and
// before it the mask unless that went ahead.
class FoldStart {
 public:
  FoldStart(FoldStart &&) = default;
  FoldStart(const FoldStart &) = delete;

 private:
  FoldStart() = default;
  friend FoldStart fold_start_prove(ProverContext &, const FoldShape &, FoldMask);
  friend bool fold_start_verify(VerifierContext &, const FoldShape &, FoldMask, std::unique_ptr<FoldStart> &);
  template <class Ch> friend void fold_epsilon(Ch &, FoldStart &);
  friend void fold_rounds_prove(ProverContext &, const FoldShape &, const GroupRow &, FoldStart, SingleRows &, SingleRowsWitness &);
  friend bool fold_rounds_verify(VerifierContext &, const FoldShape &, GroupRow, FoldStart, SingleRows &);
  void same_shape(const FoldShape &S) const {
    BLK_REQUIRE(S.code == code_ && S.mask_code == mask_code_ && S.layout.k == k_ && S.layout.h == h_ && S.layout.members == members_,
                "FOLD: the rounds are run for another shape than the first message");
  }
  void remember(const Statement &st, const FoldShape &S) {
    code_ = S.code, mask_code_ = S.mask_code, k_ = S.layout.k, h_ = S.layout.h, members_ = S.layout.members;
    u64 kept_rows = 0;  // counted as if folded side by side with each main row (fold_rounds.h)
    for (size_t i = 0; i < S.layout.members.size(); i++)
      if (S.layout.kept[i]) kept_rows += st.oracle(S.layout.members[i]).shape.main_rows;
    group_ = group_as_run(st.codes(), S.code, S.mask_code, true, k_, h_, u64(1) << ceil_log2(1 + kept_rows));
  }
  FoldMask mask_;
  E sum_ = E::zero(), eps_ = E::zero();
  bool has_eps_ = false;
  CodeId code_ = NONE_ID, mask_code_ = NONE_ID;
  unsigned k_ = 0, h_ = 0;
  std::vector<OracleId> members_;
  PricedGroup group_;  // codes and numbers of rows under which this fold runs
};

// Draws eps, the only element of its verifier message; both sides call this.
// The plan (blocks/numbers/plan.h: the list of all rounds and their errors) records at this round
// the group from which it computed the errors of this stage (one FOLD with its change of code).
// require_group() ends the process if the plan records none there, or one whose codes or numbers
// of rows differ from `group_`; after a rejection it compares nothing.
template <class Ch>
void fold_epsilon(Ch &ch, FoldStart &s) {
  BLK_REQUIRE(!s.has_eps_, "FOLD: eps was drawn twice");
  s.eps_ = ch.close_epsilon(R_FOLD_EPSILON);
  ch.require_group(s.group_);  // the plan's errors for this stage must be computed from this group
  s.has_eps_ = true;
}

// FOLD's output; both sides compute it with this function. `gamma`: g_1..g_k. `claim`: the running
// claim after the last round, which becomes the new target. The coefficients on kept rows are
// multiplied by fold_kept_factor() = eps * (1 - g_1) * .. * (1 - g_k), a public value.
// kept_alone (prover only): the engine's table with the folded coefficients, which then is the new
// claim's coefficient vector, times eps.
inline SingleRows fold_output(const Statement &st, const FoldShape &S, GroupRow row, OracleId fresh, const E &eps,
                              const std::vector<E> &gamma, const E &claim, const TableId *kept_alone = nullptr) {
  const Codes &codes = st.codes();
  const unsigned k = S.layout.k, h = S.layout.h;
  SingleRows out;
  out.code = S.code, out.mask_code = S.mask_code, out.layout = S.layout, out.gamma = gamma;
  out.masks = 2, out.fresh = fresh;
  if (kept_alone) {
    out.c.engine_terms.push_back(EngineTerm{*kept_alone, eps, 0, 0});
  } else {
    out.c = std::move(row.c);  // `row` is by value: the verifier moves it in, the prover a copy
    out.c.fix_low(gamma.data(), k, eps);  // fixed and scaled in one visit
  }
  out.kept = std::move(row.kept);
  for (KeptForm &f : out.kept) f.c.scale(fold_kept_factor(eps, gamma));
  for (size_t i = 0; i < S.layout.members.size(); i++)
    if (S.layout.kept[i]) out.kept_rows += st.oracle(S.layout.members[i]).shape.main_rows;
  out.d.push_back(fold_mask_form(k, h, row.d, eps, gamma));
  out.d.push_back(fold_mask_weights(k, h, codes.mask[S.mask_code].ell, gamma));
  out.target = claim;
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------------------------------------------------

// The hiding polynomial's coefficients with which the prover runs the sumcheck: the fresh mask's.
inline std::vector<E> fold_rounds_coefficients(const FoldMask &mask) {
  std::vector<E> w = mask.coefficients();
  return w;
}

// One of the group's two small tables as one vector of 2^h rows: the mask rows' messages or, if
// `tail`, their random tails (the random coefficients of their encodings).
inline std::vector<E> fold_gather_mask(const ProverContext &P, const GroupLayout &L, bool tail, size_t len) {
  std::vector<E> t((size_t(1) << L.h) * len, E::zero());
  for (size_t i = 0; i < L.members.size(); i++) {
    if (L.mask_first[i] == GroupLayout::NONE) continue;
    const Table &own = tail ? P.behind[L.members[i]].mask_tail : P.behind[L.members[i]].mask;
    BLK_REQUIRE(own.len == len, "FOLD: a member's mask rows have not the mask code's lengths");
    for (size_t b = 0; b < own.rows; b++)
      for (size_t c = 0; c < len; c++) t[(L.mask_first[i] + b) + (c << L.h)] = own.at(b, c);
  }
  return t;
}

// [own-test-fold-mask] A self-check of the prover on a fresh mask that has been sent. It ends the
// process unless two things equal the private random values drawn for them, which `P.coins`
// regenerates from the draws' numbers:
//   the hiding polynomial's coefficients, one draw each, compared here by the numbers that
//   fold_mask_make(S, P) has recorded in `P.fold_draws`;
//   the mask rows' random tail (the random entries encoded with each row) as the context keeps it,
//   compared by not_as_drawn() (blocks/iop/send.h). That function skips this message's table,
//   which is made from the coefficients and not by one draw.
// The committed table, the sum and the rounds are all computed from the coefficients. If one is
// lost (say zeroed by a stray write) before the table is made, the three agree, every verifier
// accepts, and an element of the rounds' messages, or a public combination of them, goes out
// without its mask. A tail value lost before the rows are encoded also gives a proof that verifies
// and hides less.
// FOLD calls it twice: in fold_start_prove() before the sum (the table's root has gone to the
// channel by then, in this step or in an earlier message), and in fold_rounds_prove() before the
// first round, after the last read of the coefficients and of the tail.
// A mask for which the context has no numbers (made by fold_mask_make(S, coins)) fails the check
// if the generator numbers its draws.
// Not seen: a fault in both computations of a draw; a generator whose values are not random, which
// regenerates the same values; a draw's number overwritten with Coins::NO_DRAW (that coefficient is
// then not compared); a loss in a copy: the table and tail that the engine has encoded, the copy
// from which the sum is computed, the coefficients and the tail that the rounds hold. Such a copy
// disagrees with the rest, so the blocks after FOLD are to reject the proof, but an element in it
// may lack its mask: a program must verify a proof before it releases it. The context's copy of the
// table is compared with nothing; nothing reads it after the sending. The engine's part of each
// round's polynomial is not checked here.
// With a generator that does not number its draws, nothing is compared and nothing reports that:
// not_as_drawn()'s comment says what a program that relies on such a check must require.
// The check excludes no value that can be drawn, draws nothing and sends nothing.
inline void fold_mask_as_drawn(const ProverContext &P, const FoldMask &mask) {
  u64 differ = not_as_drawn(P, mask.id());
  const std::vector<E> &w = mask.coefficients();
  const auto noted = P.fold_draws.find(w.data());
  if (noted == P.fold_draws.end() || noted->second.size() != w.size())
    differ += P.coins.last_draw() != Coins::NO_DRAW;  // a mask without numbers, though the generator gives them
  else
    for (size_t i = 0; i < w.size(); i++)  // all of them: no early end
      if (noted->second[i] != Coins::NO_DRAW) differ += P.coins.not_the_coins(noted->second[i], w[i].c, E_DEGREE, {});
  BLK_REQUIRE_NAMED(differ == 0, "own-test-fold-mask",
                    "internal error (own-test-fold-mask): coefficients of a fold's hiding polynomial, or values of its mask rows' random tail, are not the random values that were drawn for them. The program or the machine has a fault; nothing of this run may be used");
}

inline FoldStart fold_start_prove(ProverContext &P, const FoldShape &S, FoldMask mask) {
  fold_shape_check(P.st, S);
  BLK_REQUIRE(mask.coefficients().size() == 2 * size_t(S.layout.k) + 1, "FOLD: this mask was made for another shape");
  if (!mask.sent()) mask.sent_as(send_oracles(P, {fold_mask_draft(P.st.codes(), S, mask)})[0]);
  fold_mask_check(P.st, S, mask.id());
  P.st.spend_as_mask(mask.id());
  FoldStart s;
  s.remember(P.st, S);
  s.sum_ = HidingPolynomial<E>(fold_mask_variables<E>(S.layout.k), fold_rounds_coefficients(mask)).sum();
  fold_mask_as_drawn(P, mask);
  s.mask_ = std::move(mask);
  P.ch.send_hiding_sum(s.sum_);
  return s;
}

inline void fold_rounds_prove(ProverContext &P, const FoldShape &S, const GroupRow &row, FoldStart start, SingleRows &out,
                              SingleRowsWitness &witness) {
  BLK_REQUIRE(start.has_eps_, "FOLD: eps has not been drawn");
  start.same_shape(S);
  fold_row_check(P.st, S, row);
  const GroupLayout &L = S.layout;
  const Code &zk = P.st.codes().mask[S.mask_code];
  const OracleId fresh = start.mask_.id();
  FoldProverRounds<E> rounds(L.k, L.h, zk.ell, fold_rounds_coefficients(start.mask_), row.d, fold_gather_mask(P, L, false, zk.ell),
                             kept_value(P, row.kept));
  std::vector<E> fresh_tail((size_t(1) << L.h) * zk.r);
  for (u64 j = 0; j < fresh_tail.size(); j++) fresh_tail[j] = P.behind[fresh].mask_tail.entry(j);
  fold_mask_as_drawn(P, start.mask_);  // nothing below reads the coefficients or this mask's tail in `P`
  P.fold_draws.erase(start.mask_.coefficients().data());
  rounds.start(start.eps_);
  FoldVerifierRounds<E> claim(start.sum_, start.eps_, row.target);
  const FoldId f = P.engine.fold_begin(fold_parts(P, L), L.k, row.c);
  std::vector<E> gamma;
  for (unsigned j = 0; j < L.k; j++) {
    E main[3];
    P.engine.fold_round(f, main);
    std::vector<E> main_part(main, main + 3);
    std::vector<E> q = rounds.round(main_part);
    if (j == 0 && S.target_from_the_rounds) claim = FoldVerifierRounds<E>::from_first_round(q);
    P.ch.send(q[0]);
    P.ch.send(q[2]);
    const E g = P.ch.close_algebraic(R_FOLD_ROUND).draw();
    P.ch.require_group(start.group_);  // each round's proximity term is computed from this group
    claim.round(q[0], q[2], g);
    P.engine.fold_bind(f, g);
    rounds.bind(g);
    gamma.push_back(g);
  }
  // A coefficient vector with a term that only the engine knows can be folded by the engine alone.
  const bool by_engine = !row.c.engine_terms.empty();
  TableId c_folded = NO_ID;
  witness.main = P.engine.fold_end(f, witness.main_tail, S.keep_folded_coefficients || by_engine ? &c_folded : nullptr);
  FoldMaskEnd<E> end = rounds.end(false);
  witness.mask = {std::move(end.mu), std::move(end.fresh)};
  witness.mask_tail = {fold_low(fold_gather_mask(P, L, true, zk.r), gamma.data(), L.h),
                       fold_low(std::move(fresh_tail), gamma.data(), L.h)};
  BLK_REQUIRE(S.keep_folded_coefficients || !S.folded_coefficients_alone, "FOLD: the folded coefficients cannot stand alone if they are not kept");
  // A copy of `row` for fold_output(). The coefficient vector on the main rows (thousands of
  // tensors in a large group) is copied only if the new claim is computed from it.
  const bool from_the_engine = S.folded_coefficients_alone || by_engine;
  GroupRow rest;
  rest.group = row.group, rest.kept = row.kept, rest.d = row.d, rest.target = row.target;
  if (!from_the_engine) rest.c = row.c;
  out = fold_output(P.st, S, std::move(rest), fresh, start.eps_, gamma, claim.claim(), from_the_engine ? &c_folded : nullptr);
  if (by_engine && !S.folded_coefficients_alone) {  // fetch the folded vector and write it out
    const Table folded = P.engine.get(c_folded);
    auto values = std::make_shared<std::vector<E>>(folded.entries());
    for (u64 j = 0; j < folded.entries(); j++) (*values)[j] = folded.entry(j);
    Tensor written;
    written.scalar = start.eps_;
    written.push_dense(L.log_ell);
    written.set_values(std::move(values), 0);
    out.c = Form();
    out.c.add(std::move(written));
    if (!S.keep_folded_coefficients) P.engine.drop(c_folded);
  }
  if (S.keep_folded_coefficients && !S.folded_coefficients_alone)
    out.c.held.push_back(Held{c_folded, start.eps_, 0, out.c.tensors.size(), 0, out.c.entries.size(), 0, 0});
}

// All of FOLD, the prover's side. `mask`: from fold_mask_make(), already sent or not. `out`: the
// new claim. `witness`: the messages and the random tails of the three single rows.
inline void fold_prove(ProverContext &P, const FoldShape &S, const GroupRow &row, FoldMask mask, SingleRows &out,
                       SingleRowsWitness &witness) {
  FoldStart s = fold_start_prove(P, S, std::move(mask));
  fold_epsilon(P.ch, s);
  fold_rounds_prove(P, S, row, std::move(s), out, witness);
}

// ---------------------------------------------------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------------------------------------------------

// The caller's contract: see run_verifier() in blocks/chain/closed.h.
inline bool fold_start_verify(VerifierContext &V, const FoldShape &S, FoldMask mask, std::unique_ptr<FoldStart> &start) {
  fold_shape_check(V.st, S);
  if (!mask.sent()) {
    std::vector<OracleId> ids;
    if (!receive_oracles(V, {fold_mask_notice(S)}, ids)) return false;
    mask.sent_as(ids[0]);
  }
  fold_mask_check(V.st, S, mask.id());
  V.st.spend_as_mask(mask.id());
  start.reset(new FoldStart());
  start->remember(V.st, S);
  start->mask_ = std::move(mask);
  return V.ch.recv_hiding_sum(start->sum_);
}

// Derives each round's linear coefficient from the running claim (`FoldVerifierRounds`,
// blocks/iop/fold_rounds.h): q(0) + q(1) equals that claim by construction, so no round's equation
// is tested. The caller's contract: see run_verifier() in blocks/chain/closed.h.
inline bool fold_rounds_verify(VerifierContext &V, const FoldShape &S, GroupRow row, FoldStart start, SingleRows &out) {
  BLK_REQUIRE(start.has_eps_, "FOLD: eps has not been drawn");
  start.same_shape(S);
  fold_row_check(V.st, S, row);
  FoldVerifierRounds<E> claim(start.sum_, start.eps_, row.target);
  std::vector<E> gamma;
  for (unsigned j = 0; j < S.layout.k; j++) {
    E c0, c2;
    if (!V.ch.recv(c0) || !V.ch.recv(c2)) return false;
    const E g = V.ch.close_algebraic(R_FOLD_ROUND).draw();
    V.ch.require_group(start.group_);  // as the prover's side
    claim.round(c0, c2, g);
    gamma.push_back(g);
  }
  out = fold_output(V.st, S, std::move(row), start.mask_.id(), start.eps_, gamma, claim.claim());
  // Later blocks copy and scale the claim's coefficient vector and place it into a row of a later
  // group: the call below stores it once, and the copies share it.
  out.c.freeze(ceil_log2(V.st.codes().base[S.code].ell));
  return V.ch.ok();
}

// All of FOLD, the verifier's side. `mask`: after sent_as() if the fresh mask came before FOLD,
// otherwise a new FoldMask. True means that the channel has rejected nothing so far, not that the
// claim holds: later blocks must check `out`. The caller's contract: see run_verifier() in
// blocks/chain/closed.h.
inline bool fold_verify(VerifierContext &V, const FoldShape &S, GroupRow row, FoldMask mask, SingleRows &out) {
  std::unique_ptr<FoldStart> s;
  if (!fold_start_verify(V, S, std::move(mask), s)) return false;
  fold_epsilon(V.ch, *s);
  return fold_rounds_verify(V, S, std::move(row), std::move(*s), out);
}

}  // namespace blk
