// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The statement between the steps of a proximity test: a few single rows and one linear claim on
// the messages that they encode. A proximity test (iopp.h) is meant to show that a group's rows are
// close to codewords; a group's members are oracle messages: encoded main and mask rows, sent as
// Merkle roots. FOLD (fold.h) reduces the members' rows to these rows. CS, the code switch (cs.h),
// or BASE, the last step (base_case.h), takes them from there. Their order:
//     main         one row under the main code, read at position x
//     kept rows    any number: the main rows of kept members, which FOLD leaves as they were sent
//                  since a later part of the protocol reads them again (`GroupLayout`, relation.h)
//     old mask     the group's mask rows, folded; read at x modulo the mask code's length
//     fresh mask   the rows of FOLD's own mask message (`fresh`), folded; read in the same way
// The main row and the mask rows were not sent as such. Each is the fold, at the verifier's
// challenges, of up to 2^k main or 2^h mask rows that were sent, so its symbol at a position is the
// fold of their symbols there. With k = h = 0 (iopp_last_rows(), iopp.h) nothing is folded.
// A failed BLK_REQUIRE ends the process.
//
// Follows the output relation of the zero-knowledge sumcheck IOR of
// the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <utility>
#include <vector>

#include "base.h"
#include "iop/context.h"
#include "iop/fold_algebra.h"
#include "iop/relation.h"
#include "iop/tensor.h"

namespace blk {

// Public. Prover and verifier hold the same values; a `Form` (a structured coefficient vector,
// tensor.h) may be represented differently.
struct SingleRows {
  CodeId code = NONE_ID;       // of the main row, in Codes::base
  CodeId mask_code = NONE_ID;  // of the mask rows, in Codes::mask
  // How a position is read.
  GroupLayout layout;        // the members; where their rows start among 2^k main, 2^h mask rows
  std::vector<E> gamma;      // layout.k challenges; main rows use all, mask rows the first h
  unsigned masks = 0;        // 0: no mask row; 1: the old mask row; 2: the old and the fresh one
  OracleId fresh = NONE_ID;  // the oracle message with the fresh mask's 2^h rows, if masks = 2
  // The claim, with the public value `target`:
  //   < c, main message > + sum over f in kept of < f.c, messages of f.oracle's main rows >
  //     + sum_j < d[j], message of mask row j > = target.
  // c stays a `Form`, as the main row may be long; a verifier writes c out only in BASE.
  Form c;                         // on the main row's message
  std::vector<KeptForm> kept;     // on the main tables of the kept members that the claim reads
  std::vector<std::vector<E>> d;  // written out, one for each mask row
  E target = E::zero();

  size_t kept_rows = 0;  // number of main rows of all kept members of `layout`
  size_t rows() const { return 1 + kept_rows + masks; }
  size_t mask_row(unsigned j) const { return 1 + kept_rows + j; }  // index of mask row j
};

// What the prover knows of the main row and the mask rows: messages and encoding randomness (random
// tails), all in E, the extension field. For the kept rows: ProverContext::behind (messages),
// Engine::tail() (tails; blocks/engine/engine.h).
struct SingleRowsWitness {
  TableId main = NO_ID;                         // the main row's message: one row, in the engine
  Table main_tail;                              // its encoding randomness: one row
  std::vector<std::vector<E>> mask, mask_tail;  // message and encoding randomness of each mask row
};

// The code sizes and row counts on which FOLD, CS or BASE runs. require_group() and require_codes()
// (blocks/chain/rounds.h) compare them with those from which the plan (the list of rounds,
// blocks/numbers/plan.h) computed its round errors; a difference ends the process. `codes` must be
// the statement's (st.codes()), which are hashed into the hash chain; this is not checked here.
// with_mask = false: the mask code's three sizes stay 0, as in a plan's group without a mask row.
// `slots`: rows counted as if folded side by side (blocks/iop/fold_rounds.h).
inline PricedGroup group_as_run(const Codes &codes, CodeId code, CodeId mask_code, bool with_mask, unsigned k, unsigned h, u64 slots) {
  BLK_REQUIRE(code < codes.base.size() && (!with_mask || mask_code < codes.mask.size()), "a stage names a code that does not exist");
  auto of = [](const Code &c) { return PricedCode{c.log_n, (u64)c.ell, (u64)c.r}; };
  PricedGroup g;
  g.main = of(codes.base[code]);
  if (with_mask) g.mask = of(codes.mask[mask_code]);
  g.k = k, g.h = h, g.slots = slots, g.said = true;
  return g;
}

// The oracle messages that must be opened to read the single rows at a position.
inline std::vector<OracleId> single_rows_oracles(const SingleRows &s) {
  std::vector<OracleId> ids = s.layout.members;
  if (s.masks == 2) ids.push_back(s.fresh);
  return ids;
}

// The rows' symbols: y[i * s.rows() + j] is that of row j (main, kept rows by member and row, old
// mask, fresh mask) at the i-th listed position. Precondition (only the number of oracle messages
// is checked): opened.of begins with the opened columns of single_rows_oracles(s), at least
// `positions` for each.
inline std::vector<E> single_rows_symbols(const SingleRows &s, const Opened &opened, size_t positions) {
  const GroupLayout &L = s.layout;
  BLK_REQUIRE(s.gamma.size() == L.k && L.h <= L.k, "single rows: not one challenge for each row variable, or more mask rows than main rows");
  BLK_REQUIRE(L.kept.size() == L.members.size() && L.main_first.size() == L.members.size() && L.mask_first.size() == L.members.size(),
              "single rows: the layout's lists do not have one entry for each member");
  BLK_REQUIRE(opened.of.size() >= single_rows_oracles(s).size(), "single rows: opened columns of some oracle are missing");
  std::vector<E> y;
  for (size_t i = 0; i < positions; i++) {
    std::vector<E> main(size_t(1) << L.k, E::zero()), mask(size_t(1) << L.h, E::zero());  // rows that nobody sent are 0
    std::vector<E> kept;
    for (size_t w = 0; w < L.members.size(); w++) {
      const Opened::Column &col = opened.of[w][i];
      BLK_REQUIRE((L.kept[w] || col.main.empty() || L.main_first[w] + col.main.size() <= main.size()) &&
                      (col.mask.empty() || L.mask_first[w] + col.mask.size() <= mask.size()),
                  "single rows: a member's rows do not lie among the group's rows");
      for (size_t b = 0; b < col.main.size(); b++)
        if (L.kept[w])
          kept.push_back(col.main[b]);
        else
          main[L.main_first[w] + b] = col.main[b];
      for (size_t b = 0; b < col.mask.size(); b++) mask[L.mask_first[w] + b] = col.mask[b];
    }
    BLK_REQUIRE(kept.size() == s.kept_rows, "single rows: the number of kept rows is not the layout's");
    y.push_back(fold_low(std::move(main), s.gamma.data(), L.k)[0]);
    y.insert(y.end(), kept.begin(), kept.end());
    if (s.masks >= 1) y.push_back(fold_low(std::move(mask), s.gamma.data(), L.h)[0]);
    if (s.masks == 2) {
      const std::vector<E> &fresh = opened.of[L.members.size()][i].mask;
      BLK_REQUIRE(fresh.size() == size_t(1) << L.h, "single rows: the fresh mask has not as many rows as the group has mask rows");
      y.push_back(fold_low(fresh, s.gamma.data(), L.h)[0]);
    }
  }
  return y;
}

// The rows' symbols at positions `at`, ordered as by single_rows_symbols(). The prover writes the
// opening into the proof; the verifier checks it and, on a rejection, returns false, y unchanged.
inline std::vector<E> single_rows_open(ProverContext &P, const SingleRows &s, const PositionList &at) {
  Opened opened;
  open_oracles(P, single_rows_oracles(s), at, opened);
  return single_rows_symbols(s, opened, at.size());
}
inline bool single_rows_read(VerifierContext &V, const SingleRows &s, const PositionList &at, std::vector<E> &y) {
  if (!V.ch.ok()) return false;  // after a rejection the channel's list is a placeholder
  Opened opened;
  if (!read_opened(V, single_rows_oracles(s), at, opened)) return false;
  y = single_rows_symbols(s, opened, at.size());
  return true;
}

// `c` written out: ell entries. ell must be a power of two (not checked): Form::add_to() writes
// 2^ceil_log2(ell) entries.
inline std::vector<E> single_rows_main_form(const SingleRows &s, size_t ell) {
  std::vector<E> c(ell, E::zero());
  s.c.add_to(c.data(), ceil_log2(ell));
  return c;
}

inline std::vector<E> table_row(const Table &t, size_t row) {
  std::vector<E> v(t.len);
  for (size_t i = 0; i < t.len; i++) v[i] = t.at(row, i);
  return v;
}

// The value of the kept rows' part of a claim.
inline E kept_value(ProverContext &P, const std::vector<KeptForm> &kept) {
  E sum = E::zero();
  for (const KeptForm &f : kept) sum += P.engine.evaluate(P.behind[f.oracle].main, f.c);
  return sum;
}

// Whether the claim holds for this witness. For the prover's own checks and for tests.
inline bool single_rows_hold(ProverContext &P, const SingleRows &s, const SingleRowsWitness &w) {
  E sum = P.engine.evaluate(w.main, s.c) + kept_value(P, s.kept);
  for (unsigned j = 0; j < s.masks; j++) sum += inner(s.d[j], w.mask[j]);
  return sum == s.target;
}

}  // namespace blk
