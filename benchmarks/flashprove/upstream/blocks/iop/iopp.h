// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The proximity test (`iopp` in identifiers) on a group: oracle messages (rows sent as Merkle
// roots) under one main and one mask code, read at one list of positions. It is meant to show that
// the rows are close to codewords that meet the group's open linear claims ("this linear form of
// the encoded messages has this public value"). IOPP: interactive oracle proof of proximity.
//
//   BAT     batches all open claims on the group and the claims on the out-of-domain samples
//           (random linear forms) of all its oracle messages into one claim (`GroupRow`)
//   for each stage but the last (stage 1 is the caller's group):
//     FOLD  folds 2^k main and 2^h mask rows into three single rows (the third from a fresh mask)
//     CS    code switch (`switch` in identifiers): re-encodes the three rows as one new oracle
//           message, which is the next stage
//   BASE    the last stage, one main row and one mask row, is sent, masked; the verifier checks the
//           claim and, at a list of positions, the encoding
// The plan (`Plan`, blocks/numbers/plan.h: each round with its error) must be made for the stages
// and codes that are run: see iopp_sized_fault().
//
// Follows the zero-knowledge proximity test (IOPP) for constrained codes, the sizes of its stages too, and, where masks go out with a
// stage's oracle message, its complete form, of the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "iop/base_case.h"
#include "iop/bat.h"
#include "iop/context.h"
#include "iop/cs.h"
#include "iop/fold.h"
#include "iop/single_rows.h"
#include "numbers/blocks.h"

namespace blk {

// 2^k main rows under `code` and 2^h mask rows (rows of random values) under `mask_code`.
struct IoppStage {
  CodeId code = NONE_ID;       // in Codes::base
  CodeId mask_code = NONE_ID;  // in Codes::mask
  unsigned k = 0, h = 0;
};

// FOLD's fresh mask and BASE's masking oracle are made only from private randomness (`coins` in
// identifiers), so when they are sent is a choice. Each is sent in FOLD's or BASE's first message
// or, with `masks_ride`, in the prover message of the CS that makes its stage. There it can share
// the Merkle tree of the stage's oracle message, because both are read at one list of positions and
// at no other.
// The number of answers to a sample is not a choice: it is Setting::samples, bound with the plan in
// the hash chain's first query (its input before the prover speaks).
struct IoppChoice {
  bool masks_ride = false;  // from stage 2 on, each mask is sent with the CS that makes its stage
};

inline void public_bytes(Writer &w, const IoppStage &s) {
  for (u64 x : {(u64)s.code, (u64)s.mask_code, (u64)s.k, (u64)s.h}) w.put_u64(x);
}
inline void public_bytes(Writer &w, const IoppChoice &c) { w.put_u64(c.masks_ride); }
// Checks one stage, or the choice, by itself. iopp_fault() checks the stages against each other and
// the codes.
inline const char *public_fault(const IoppStage &s) {
  return s.k > CODE_MOST_LOG_N || s.h > s.k ? "too many row variables, or more mask rows than main rows" : nullptr;
}
inline const char *public_fault(const IoppChoice &) { return nullptr; }

// Returns nullptr if the stages fit each other and the codes, else what is wrong, with *stage set
// to the failing stage's index (from 0). It never ends the process: safe on numbers from a file.
// A stage's main code has length n, message length l and r random entries (a row's random tail,
// `tail` in identifiers); its mask code has n_zk, l_zk, r_zk, and n_zk divides n. Primes mark the
// next stage. The main conditions:
//   l = 2^k' l'            the folded main message fills the next main rows exactly;
//   2^h' l_zk' >= 2 l_zk + r + 2 r_zk + answers
//                          the next mask rows take the two folded mask rows' messages, the random
//                          tails of all three folded rows and, for each answer to the new
//                          message's sample, the random pad that masks it; cs_prove() fills the
//                          rest with random values;
//   l_zk >= 2 k + 1 - h    room for the coefficients of FOLD's hiding polynomial;
//   k >= 1, h <= k         at every stage but the last, where k = h = 0.
// Not checked here: the plan sets how many positions are drawn at a stage, and its rows' random
// tails must hide that many; otherwise the prover's open_oracles() (context.h) ends the process.
inline const char *iopp_fault(const Codes &codes, const std::vector<IoppStage> &st, unsigned answers, size_t *stage = nullptr) {
  size_t at = 0;
  if (!stage) stage = &at;
  *stage = 0;
  if (st.size() < 2) return "a proximity test has fewer than two stages";
  if (answers < 1) return "the oracle message of a switch has no answer to its sample";
  auto power_of_two = [](u64 x) { return x != 0 && (x & (x - 1)) == 0; };
  for (size_t i = 0; i < st.size(); i++) {
    const IoppStage &s = st[i];
    *stage = i;
    if (s.code >= codes.base.size() || s.mask_code >= codes.mask.size()) return "the stage names a code that is not there";
    const Code &c = codes.base[s.code], &zk = codes.mask[s.mask_code];
    if (!code_ok(c) || !code_ok(zk)) return "the stage's numbers are not a code";
    if (!power_of_two(c.ell) || !power_of_two(zk.ell)) return "a message length is no power of two";
    if (!mask_fits(c, zk)) return "the mask code is longer than the main code";
    if (s.k > CODE_MOST_LOG_N || s.h > s.k) return "too many row variables, or more mask rows than main rows";
    if (i + 1 < st.size()) {
      if (s.k < 1) return "a stage but the last has no row variable";
      if (zk.ell < fold_mask_columns(s.k, s.h)) return "the mask rows are too short for the hiding polynomial of the fold";
    } else if (s.k != 0) {
      return "the last stage has more than one main row";
    }
    if (i == 0) continue;
    // Lengths are at most 2^CODE_MOST_LOG_N (2^60), so the sums fit in 64 bits. A shift that
    // overflows gives 0, because message lengths are powers of two, and 0 fails its comparison.
    const Code &pc = codes.base[st[i - 1].code], &pzk = codes.mask[st[i - 1].mask_code];
    const u64 room = (u64)zk.ell << s.h, carried = 2 * (u64)pzk.ell + pc.r + 2 * (u64)pzk.r;
    if ((u64)c.ell << s.k != pc.ell) return "the folded main message does not fill the main rows exactly";
    if (room < carried + answers) return "the mask rows have no room for masks, randomness and pad";
    if (answers > zk.ell) return "a mask row is shorter than the pad";
  }
  return nullptr;
}

// Turns solved sizes (`Stage`, blocks/numbers/blocks.h) into stages, so that no number is written
// twice. Appends to `codes` each code that it does not contain yet.
inline std::vector<IoppStage> iopp_stages(const std::vector<Stage> &solved, Codes &codes) {
  auto place = [](std::vector<Code> &list, const CodeSize &c) {
    const Code code{c.log_n, (size_t)c.ell, (size_t)c.r};
    for (size_t i = 0; i < list.size(); i++)
      if (list[i] == code) return (CodeId)i;
    list.push_back(code);
    return (CodeId)(list.size() - 1);
  };
  std::vector<IoppStage> st;
  for (const Stage &s : solved) st.push_back(IoppStage{place(codes.base, s.g.main), place(codes.mask, s.g.mask), s.g.k, s.g.h});
  return st;
}
// Returns nullptr if `solved`, from which plan_iopp() (blocks/numbers/blocks.h) computes the error
// of every round of the test, describes these stages under these codes; else what differs. Call it
// before every plan_iopp(): a plan sized for other codes than the run's states the errors of a
// protocol that is not run (a longer random tail raises a code's rate, so its rows need more
// positions).
// Compared: the number of stages, k, h, each code's log_n, ell and r; no other field of `Stage`.
// Not caught: a caller that plans with one list of codes and binds another in the run. There
// require_group() (blocks/chain/rounds.h) ends the process where a stage begins.
inline const char *iopp_sized_fault(const Codes &codes, const std::vector<IoppStage> &stages, const std::vector<Stage> &solved) {
  if (stages.size() != solved.size()) return "the proximity test is sized for another number of stages than it has";
  auto same = [](const Code &run, const CodeSize &priced) { return run.log_n == priced.log_n && run.ell == priced.ell && run.r == priced.r; };
  for (size_t i = 0; i < stages.size(); i++) {
    if (stages[i].code >= codes.base.size() || stages[i].mask_code >= codes.mask.size()) return "a stage of the proximity test names a code that is not there";
    if (stages[i].k != solved[i].g.k || stages[i].h != solved[i].g.h) return "a stage of the proximity test is sized for another number of rows than it has";
    if (!same(codes.base[stages[i].code], solved[i].g.main)) return "a stage of the proximity test is sized for another main code than it names";
    if (!same(codes.mask[stages[i].mask_code], solved[i].g.mask)) return "a stage of the proximity test is sized for another mask code than it names";
  }
  return nullptr;
}

// The shape of a stage's FOLD. `layout` may be empty: its members are not known before the stage's
// oracle message is sent, and a fresh mask can be made and sent without them.
inline FoldShape iopp_fold_shape(const IoppStage &s, GroupLayout layout) {
  FoldShape F;
  F.code = s.code, F.mask_code = s.mask_code, F.layout = std::move(layout);
  F.layout.k = s.k, F.layout.h = s.h;
  return F;
}

inline void iopp_first_stage_check(const Statement &S, GroupId g, const GroupLayout &L, const IoppStage &first) {
  BLK_REQUIRE(S.group(g).code == first.code && S.group(g).mask_code == first.mask_code && L.k == first.k && L.h == first.h,
              "IOPP: the first stage is not the group");
}

// The last stage as single rows. A group's layout has at least one row variable. With one main row,
// the second row is zeros that were never sent, and fixing the variable at 0 selects the first row.
inline SingleRows iopp_last_rows(const IoppStage &s, GroupLayout L, const GroupRow &row) {
  BLK_REQUIRE(L.k == 1 && L.h == 0 && L.members.size() == 1, "IOPP: the last stage is one oracle message of one main and one mask row");
  const E zero = E::zero();
  SingleRows out;
  out.code = s.code, out.mask_code = s.mask_code;
  out.masks = 1;
  out.c = row.c;
  out.c.fix_low(&zero, 1);
  out.d.push_back(row.d);
  out.target = row.target;
  L.k = 0;
  out.layout = std::move(L);
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------------------------------------------------

// Both sides end the process if iopp_fault() rejects the stages, if stage 1 is not group g, or if g
// was tested before (Statement::begin_run(), relation.h).
// first_mask: the first FOLD's fresh mask, if the caller has already sent it and called its
// sent_as(); otherwise FOLD sends a new one.
inline void iopp_prove(ProverContext &P, GroupId g, const std::vector<IoppStage> &st, const IoppChoice &choice, FoldMask first_mask = FoldMask()) {
  const unsigned answers = P.ch.plan()->set.samples;
  const char *fault = iopp_fault(P.st.codes(), st, answers);
  BLK_REQUIRE(!fault, fault ? fault : "");
  GroupLayout L;
  GroupRow row = bat_prove(P, g, &L);
  iopp_first_stage_check(P.st, g, L, st[0]);

  FoldMask fold_mask = std::move(first_mask);
  BaseMask base_mask;
  OracleId last = NONE_ID;
  TableId coefficients = NO_ID;  // folded coefficients that the engine keeps for the next FOLD
  for (size_t i = 0; i + 1 < st.size(); i++) {
    FoldShape F = iopp_fold_shape(st[i], L);
    F.keep_folded_coefficients = true;
    F.folded_coefficients_alone = i + 2 < st.size();  // not at the last FOLD: BASE reads the vector
    if (!fold_mask.sent()) fold_mask = fold_mask_make(F, P);
    SingleRows rows;
    SingleRowsWitness W;
    fold_prove(P, F, row, std::move(fold_mask), rows, W);
    fold_mask = FoldMask();
    if (coefficients != NO_ID) P.engine.drop(coefficients);
    coefficients = fold_kept_coefficients(rows.c);

    const IoppStage &next = st[i + 1];
    const bool to_last = i + 2 == st.size();
    const std::string name = "stage " + std::to_string(i + 2);
    const GroupId to = P.st.new_group(name, next.code, next.mask_code);
    std::vector<OracleDraft> with;
    if (choice.masks_ride && !to_last) {
      const FoldShape N = iopp_fold_shape(next, GroupLayout());
      fold_mask = fold_mask_make(N, P);
      with.push_back(fold_mask_draft(P.st.codes(), N, fold_mask));
    }
    if (to_last) base_mask = base_mask_make(P.st.codes(), next.code, next.mask_code, 1, P.engine, P.coins);
    if (choice.masks_ride && to_last) with.push_back(base_mask_draft(next.code, next.mask_code, 1, 0, base_mask));
    const Switched s = cs_prove(P, rows, W, switch_target_stage(name, next.code, next.mask_code, next.k, next.h, answers), to,
                                std::move(with), row, L);
    if (choice.masks_ride) to_last ? base_mask.sent_as(s.with[0]) : fold_mask.sent_as(s.with[0]);
    P.engine.drop(W.main);
    if (i > 0) release_oracles(P, single_rows_oracles(rows));  // stage 1 is the caller's to release
    last = s.g;
  }

  const SingleRows rows = iopp_last_rows(st.back(), L, row);
  if (coefficients != NO_ID) P.engine.drop(coefficients);  // BASE reads the claim on the host
  SingleRowsWitness W;  // folding over 0 variables returns the row and its random tail
  W.main = P.engine.fold_end(P.engine.fold_begin(fold_parts(P, rows.layout), 0, Form()), W.main_tail);
  W.mask.push_back(table_row(P.behind[last].mask, 0));
  W.mask_tail.push_back(table_row(P.behind[last].mask_tail, 0));
  base_prove(P, rows, W, std::move(base_mask), 0);
  P.engine.drop(W.main);
}

// ---------------------------------------------------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------------------------------------------------

// Returns false if the proof fails a check. Call iopp_fault() first on stages read from a file:
// a fault here ends the process.
// first_mask: as for iopp_prove(); here it is only the oracle message's id.
// True does not mean that the whole proof was read: the caller must end with its channel's
// finish(), which rejects left-over bytes.
// The plan that V's channel runs must come from one of the library's composite planners: see the
// contract at run_verifier() in blocks/chain/closed.h.
inline bool iopp_verify(VerifierContext &V, GroupId g, const std::vector<IoppStage> &st, const IoppChoice &choice, FoldMask first_mask = FoldMask()) {
  const unsigned answers = V.ch.plan()->set.samples;
  const char *fault = iopp_fault(V.st.codes(), st, answers);
  BLK_REQUIRE(!fault, fault ? fault : "");  // stages are configuration, not part of a proof
  GroupLayout L;
  GroupRow row;
  if (!bat_verify(V, g, row, &L)) return false;
  V.note("BAT: target", row.target);  // for comparison with a second verifier; no decision reads it
  iopp_first_stage_check(V.st, g, L, st[0]);

  FoldMask fold_mask = std::move(first_mask);
  BaseMask base_mask;
  for (size_t i = 0; i + 1 < st.size(); i++) {
    SingleRows rows;
    if (!fold_verify(V, iopp_fold_shape(st[i], L), std::move(row), std::move(fold_mask), rows)) return false;  // cs_verify below makes the next row
    V.note("FOLD: target", rows.target);
    fold_mask = FoldMask();

    const IoppStage &next = st[i + 1];
    const bool to_last = i + 2 == st.size();
    const std::string name = "stage " + std::to_string(i + 2);
    const GroupId to = V.st.new_group(name, next.code, next.mask_code);
    std::vector<OracleNotice> with;
    if (choice.masks_ride)
      with.push_back(to_last ? base_mask_notice(next.code, next.mask_code, 1, 0) : fold_mask_notice(iopp_fold_shape(next, GroupLayout())));
    Switched s;
    if (!cs_verify(V, rows, switch_target_stage(name, next.code, next.mask_code, next.k, next.h, answers), to, with, s, row, L))
      return false;
    V.note("CS: target", row.target);
    if (V.notes) V.note("CS: c'[0]", row.c.at(0)), V.note("CS: d'[0]", row.d.front()), V.note("CS: d's last entry", row.d.back());
    if (choice.masks_ride) to_last ? base_mask.sent_as(s.with[0]) : fold_mask.sent_as(s.with[0]);
  }
  return base_verify(V, iopp_last_rows(st.back(), L, row), std::move(base_mask), 0);
}

// The entry points: stages and choice are bound in the chain's first query (blocks/chain/public.h).
// The overloads above are for a caller that passes parts of a bound value read from its channel.
inline void iopp_prove(ProverContext &P, GroupId g, const Public<std::vector<IoppStage>> &stages, const Public<IoppChoice> &choice, FoldMask first_mask = {}) {
  iopp_prove(P, g, P.ch.get(stages), P.ch.get(choice), std::move(first_mask));
}
// Contract: as for the overload above.
[[nodiscard]] inline bool iopp_verify(VerifierContext &V, GroupId g, const Public<std::vector<IoppStage>> &stages, const Public<IoppChoice> &choice,
                                      FoldMask first_mask = {}) {
  return iopp_verify(V, g, V.ch.get(stages), V.ch.get(choice), std::move(first_mask));
}

}  // namespace blk
