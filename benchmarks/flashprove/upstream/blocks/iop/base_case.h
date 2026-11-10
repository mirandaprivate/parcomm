// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// BASE ends a proximity test (a test that rows are close to codewords): the last rows are sent,
// masked, and compared; no claim is left. Input: S (`SingleRows`, single_rows.h; "the complex" in
// messages): a main row and up to two mask rows Enc(M, R) (message M, random tail R), no kept rows
// (main rows that FOLD, fold.h, leaves as they were sent), and one linear claim < c, M > = target
// (c: `S.c`, written out by single_rows_main_form(); `S.d[j]` for mask row j). Messages:
//   P  masking oracle g*: a row Enc(M*, R*) for each input row, M*, R* uniform; target* = < c, M* >
//   V  lambda in the extension field E; g* and target* must precede it
//   P  M' = M* + lambda M and R' = R* + lambda R, not encoded: every row's M', then every row's R'
//   V  a list of main-row positions; M', R' must precede it; a mask row is read at mask_position()
// Hiding: if M* and R* are uniform and serve this BASE only, M' and R' are uniform whatever M and R
// are. The input rows' symbols at the listed positions are covered by those rows' own random tails,
// which must be uniform and hide that many positions (open_oracles() checks the count). The rest
// shows nothing more: target* follows from M', lambda and target, g*'s symbols from (M', R') and
// the input rows' symbols. What this achieves for a whole run: docs/limits.md#weight-hiding.
// The prover's own checks: the main row's M* is drawn on the host, passed to the engine (`Engine`,
// the CPU or GPU back end) and fetched back. Were it lost (zeroed, say) before g* is committed, g*,
// target* and M' would agree, no verifier could tell, and lambda M would be sent unmasked. So
// base_prove() requires the fetched M* to equal the drawn row and, if the generator of private
// randomness (`Coins`) numbers its draws, every M* and the mask rows' R* to equal their draws,
// recomputed. Not caught: a faulty generator, whose values are not random; the cases listed at the
// checks. Not checked at all: the main row's R*, which the engine draws and alone stores. No host
// copy of M* is erased: that is left to the calling program.
// The masking oracle gets no out-of-domain sample. It serves one BASE: a second spend_as_mask()
// (`Statement`, relation.h) on it ends the process. It may go out with a preceding prover message:
// base_mask_make(), then base_mask_draft() (verifier: base_mask_notice()) in that message's call,
// then sent_as(); else BASE sends it first. It is opened at BASE's list of positions only. Oracle
// messages of one `leaf_class` can share tree leaves and must be opened together, so a message that
// is sent in the same call and opened at another list needs another `leaf_class`.
//
// Follows the construction of the non-succinct zero-knowledge protocol for constrained codes of
// the paper that this protocol builds on (see THIRD_PARTY.md).
#pragma once
#include <utility>
#include <vector>

#include "base.h"
#include "iop/context.h"
#include "iop/send.h"
#include "iop/single_rows.h"
#include "numbers/round_names.h"

namespace blk {

// One BASE's masking oracle: move-only, and BASE consumes it. The verifier's side sets only `id_`.
class BaseMask {
 public:
  BaseMask() = default;
  BaseMask(BaseMask &&) = default;
  BaseMask &operator=(BaseMask &&) = default;
  BaseMask(const BaseMask &) = delete;
  BaseMask &operator=(const BaseMask &) = delete;

  bool sent() const { return id_ != NONE_ID; }
  OracleId id() const { return id_; }
  void sent_as(OracleId id) {
    BLK_REQUIRE(!sent(), "BASE: a masking oracle was told its number twice");
    id_ = id;
  }

 private:
  friend BaseMask base_mask_make(const Codes &, CodeId, CodeId, unsigned, Engine &, Coins &);
  friend OracleDraft base_mask_draft(CodeId, CodeId, unsigned, unsigned, const BaseMask &);
  TableId main_ = NO_ID;  // the prover's: the main row's M*; the mask rows' are in `mask_`
  std::vector<E> main_drawn_;  // the main row's M* as drawn, for the comparison in `base_prove`
  friend void base_prove(ProverContext &, const SingleRows &, const SingleRowsWitness &, BaseMask, unsigned);
  Table mask_;
  u32 main_draw_ = Coins::NO_DRAW, mask_draw_ = Coins::NO_DRAW;  // numbers of the two draws of M*
  OracleId id_ = NONE_ID;
};

// Draws M* for every row. code, mask_code, masks: those of the `SingleRows` that BASE will get.
inline BaseMask base_mask_make(const Codes &codes, CodeId code, CodeId mask_code, unsigned masks, Engine &engine, Coins &coins) {
  BaseMask m;
  Table main(E_DEGREE, 1, codes.base[code].ell);
  coins.fill(main.a.data(), main.a.size());
  m.main_draw_ = coins.last_draw();
  m.main_drawn_ = table_row(main, 0);
  m.main_ = engine.put(main);
  if (masks) {
    m.mask_ = Table(E_DEGREE, masks, codes.mask[mask_code].ell);
    coins.fill(m.mask_.a.data(), m.mask_.a.size());
    m.mask_draw_ = coins.last_draw();
  }
  return m;
}
inline OracleNotice base_mask_notice(CodeId code, CodeId mask_code, unsigned masks, unsigned leaf_class) {
  OracleNotice n;
  n.shape.name = "base mask";
  n.shape.code = code, n.shape.main_rows = 1, n.shape.main_deg = E_DEGREE;
  n.shape.mask_code = mask_code, n.shape.mask_rows = masks;
  n.shape.answers = 0;
  n.leaf_class = leaf_class;
  return n;
}
inline OracleDraft base_mask_draft(CodeId code, CodeId mask_code, unsigned masks, unsigned leaf_class, const BaseMask &m) {
  const OracleNotice n = base_mask_notice(code, mask_code, masks, leaf_class);
  OracleDraft d;
  d.shape = n.shape, d.leaf_class = n.leaf_class;
  d.main = m.main_, d.mask = m.mask_, d.mask_draw = m.mask_draw_;
  return d;
}

// Ends the process unless BASE can take S (no kept rows, mask code no longer than main code) and
// `star`, named as the masking oracle: a row over E per row of S, same codes, not in S, no sample.
inline void base_shape_check(const Statement &st, const SingleRows &S, OracleId star) {
  const Oracle &o = st.oracle(star);
  BLK_REQUIRE(S.kept_rows == 0 && S.kept.empty(), "BASE: the complex has kept rows");
  BLK_REQUIRE(S.masks == 0 || mask_fits(st.codes().base[S.code], st.codes().mask[S.mask_code]), "BASE: the mask code is longer than the main code");
  BLK_REQUIRE(o.shape.main_rows == 1 && o.shape.main_deg == E_DEGREE && o.shape.code == S.code && o.shape.mask_rows == S.masks &&
                  (S.masks == 0 || o.shape.mask_code == S.mask_code),
              "BASE: the oracle named as the masking oracle has not one row for each row of the complex, under the same codes");
  BLK_REQUIRE(!o.sampled && o.shape.answers == 0, "BASE: the oracle named as the masking oracle has an out-of-domain sample");
  for (OracleId m : single_rows_oracles(S)) BLK_REQUIRE(m != star, "BASE: the oracle named as the masking oracle is one of the oracles that it is to mask");
}

// Host copies of the message and the random tail of o's main row (row 0), fetched from the engine.
inline void base_fetch_main_row(ProverContext &P, OracleId o, std::vector<E> &message, std::vector<E> &tail) {
  const TreeBook::Place at = P.book.place(o);
  message = table_row(P.engine.get(P.behind[o].main), 0);
  tail = table_row(P.engine.tail(P.commit[at.tree], at.part), 0);
}

// mask: from base_mask_make(). leaf_class: the masking oracle's, used only if BASE sends it itself.
inline void base_prove(ProverContext &P, const SingleRows &S, const SingleRowsWitness &W, BaseMask mask, unsigned leaf_class) {
  const Code &code = P.st.codes().base[S.code];
  if (!mask.sent()) mask.sent_as(send_oracles(P, {base_mask_draft(S.code, S.mask_code, S.masks, leaf_class, mask)})[0]);
  const OracleId star = mask.id();
  base_shape_check(P.st, S, star);
  P.st.spend_as_mask(star);

  // Row by row: messages and random tails of the masking oracle, then of the input.
  std::vector<std::vector<E>> m_star(S.rows()), r_star(S.rows()), m(S.rows()), r(S.rows());
  base_fetch_main_row(P, star, m_star[0], r_star[0]);
  BLK_REQUIRE_NAMED(m_star[0] == mask.main_drawn_, "own-test-base-mask", "BASE: the masking word that the engine gives back is not the one that was drawn");
  // `main_drawn_` is copied after the draw. If the row is lost between the two, the copy and the
  // engine's table agree, the check above passes, and the main row's M' below is lambda M,
  // unmasked, in a proof that verifies. So what is used below must also equal the draws, which
  // `P.coins` recomputes from their numbers; base_mask_make() must have drawn from `P.coins`. The
  // main row's M* ("the masking word" in messages) is compared as fetched: a one-row table over E
  // is stored in the order of its draw. The mask rows and their random tails are compared as
  // `P.behind[star]` stores them (not_as_drawn(), send.h); they are read from there below.
  // Not compared, and nothing reports that: any draw, if the generator does not number its draws; a
  // draw whose number was not recorded or not passed on (it is then Coins::NO_DRAW).
  // Not caught: a fault in both computations of a draw; values that are not random, which the
  // generator recomputes the same; a value overwritten after these checks; a loss in the copies of
  // these rows that the engine has encoded by then. The main row's R* has no check: a test "no
  // entry is 0" would reject a valid value and catch one type of loss only.
  // Not compared here: the prover's other draws on the host (the coefficients of FOLD's fresh mask,
  // which FOLD compares itself: fold_mask_as_drawn(), fold.h; the drawn tables and random tails of
  // other oracle messages).
  // The checks exclude no value of M* or R*, draw nothing and send nothing.
  BLK_REQUIRE_NAMED(mask.main_draw_ == Coins::NO_DRAW || P.coins.not_the_coins(mask.main_draw_, m_star[0].data()->c, m_star[0].size() * E_DEGREE, {}) == 0, "own-test-base-mask",
                    "internal error (own-test-base-mask): the masking word of a proximity test's last step is not the random values that were drawn for it. The program or the machine has a fault; nothing of this run may be used");
  BLK_REQUIRE_NAMED(S.masks == 0 || not_as_drawn(P, star) == 0, "own-test-base-mask",
                    "internal error (own-test-base-mask): the mask rows of a proximity test's last step, or their random tail, are not the random values that were drawn for them. The program or the machine has a fault; nothing of this run may be used");
  m[0] = table_row(P.engine.get(W.main), 0), r[0] = table_row(W.main_tail, 0);
  for (unsigned j = 0; j < S.masks; j++) {
    m_star[1 + j] = table_row(P.behind[star].mask, j), r_star[1 + j] = table_row(P.behind[star].mask_tail, j);
    m[1 + j] = W.mask[j], r[1 + j] = W.mask_tail[j];
  }
  const std::vector<E> c = single_rows_main_form(S, code.ell);
  E target_star = inner(c, m_star[0]);
  for (unsigned j = 0; j < S.masks; j++) target_star += inner(S.d[j], m_star[1 + j]);
  P.ch.send(target_star);
  const E lambda = P.ch.close_algebraic(R_BASE_LAMBDA).draw();
  // The plan (`Plan`: the list of all rounds with their errors) must have computed the last stage's
  // errors for this group: these codes' sizes, with k = h = 0 and slots = 1 (nothing is folded).
  P.ch.require_group(group_as_run(P.st.codes(), S.code, S.mask_code, S.masks != 0, 0, 0, 1));  // this round begins the proximity test's last stage

  for (unsigned j = 0; j < S.rows(); j++) {
    BLK_REQUIRE(m[j].size() == m_star[j].size() && r[j].size() == r_star[j].size(), "BASE: the witness has not the codes' lengths");
    for (size_t i = 0; i < m[j].size(); i++) m[j][i] = m_star[j][i] + lambda * m[j][i];
    for (size_t i = 0; i < r[j].size(); i++) r[j][i] = r_star[j][i] + lambda * r[j][i];
  }
  for (const std::vector<E> &row : m)
    for (const E &x : row) P.ch.send(x);
  for (const std::vector<E> &row : r)
    for (const E &x : row) P.ch.send(x);
  const PositionList at = P.ch.close_positions(R_BASE).positions(code.log_n);
  P.ch.require_codes(group_as_run(P.st.codes(), S.code, S.mask_code, S.masks != 0, 0, 0, 1));  // the plan must have computed this round's error for these codes

  std::vector<OracleId> ids = single_rows_oracles(S);
  ids.push_back(star);
  Opened opened;
  open_oracles(P, ids, at, opened);
}

// Checks: < c, M' > = target* + lambda target; at every listed position and for every row, the
// symbol of Enc(M', R') equals g*'s symbol + lambda * the row's symbol (single_rows_symbols()).
// Caller's contract (where plan and shapes must come from): run_verifier(), blocks/chain/closed.h.
inline bool base_verify(VerifierContext &V, const SingleRows &S, BaseMask mask, unsigned leaf_class) {
  const Code &code = V.st.codes().base[S.code];
  if (!mask.sent()) {
    std::vector<OracleId> got;
    if (!receive_oracles(V, {base_mask_notice(S.code, S.mask_code, S.masks, leaf_class)}, got)) return false;
    mask.sent_as(got[0]);
  }
  base_shape_check(V.st, S, mask.id());
  V.st.spend_as_mask(mask.id());
  E target_star;
  if (!V.ch.recv(target_star)) return false;
  const E lambda = V.ch.close_algebraic(R_BASE_LAMBDA).draw();
  V.ch.require_group(group_as_run(V.st.codes(), S.code, S.mask_code, S.masks != 0, 0, 0, 1));  // as in `base_prove`

  std::vector<std::vector<E>> m(S.rows()), r(S.rows());
  for (unsigned j = 0; j < S.rows(); j++) {
    const Code &cj = j == 0 ? code : V.st.codes().mask[S.mask_code];
    m[j].resize(cj.ell), r[j].resize(cj.r);
  }
  for (std::vector<E> &row : m)
    for (E &x : row)
      if (!V.ch.recv(x)) return false;
  for (std::vector<E> &row : r)
    for (E &x : row)
      if (!V.ch.recv(x)) return false;
  E left = inner(single_rows_main_form(S, code.ell), m[0]);
  for (unsigned j = 0; j < S.masks; j++) left += inner(S.d[j], m[1 + j]);
  V.note("BASE: left side of the linear check", left);
  V.note("BASE: right side of the linear check", target_star + lambda * S.target);
  if (left != target_star + lambda * S.target) return V.ch.refuse("BASE: the masked message in the proof does not satisfy the linear claim"), false;

  const PositionList at = V.ch.close_positions(R_BASE).positions(code.log_n);
  V.ch.require_codes(group_as_run(V.st.codes(), S.code, S.mask_code, S.masks != 0, 0, 0, 1));  // as in `base_prove`
  if (!V.ch.ok()) return false;

  std::vector<OracleId> ids = single_rows_oracles(S);
  ids.push_back(mask.id());
  Opened opened;
  if (!read_opened(V, ids, at, opened)) return false;
  const std::vector<E> y = single_rows_symbols(S, opened, at.size());
  for (size_t i = 0; i < at.size(); i++) {
    const Opened::Column &star = opened.of.back()[i];
    for (unsigned j = 0; j < S.rows(); j++) {
      const Code &cj = j == 0 ? code : V.st.codes().mask[S.mask_code];
      const u64 x = j == 0 ? at[i] : mask_position(code, cj, at[i]);
      const E masked = (j == 0 ? star.main[0] : star.mask[j - 1]) + lambda * y[i * S.rows() + j];
      if (symbol_at(cj, x, m[j].data(), r[j].data()) != masked)
        return V.ch.refuse("BASE: the encoding of the masked message differs from the opened symbols at a listed position"), false;
    }
  }
  return true;
}

}  // namespace blk
