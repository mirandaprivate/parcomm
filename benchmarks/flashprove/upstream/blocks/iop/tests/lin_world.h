// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test fixture (`world` in identifiers) for the protocol blocks FOLDEQ, which folds the rows of a
// committed table (`Index` in identifiers), and LIN (blocks/iop/lin.h): one linear claim per
// request of a batch on that table. lin_session() proves a batch at toy size, with real hash
// chains (Fiat-Shamir transcripts), trees, CPU engine and proximity tests (`iopp` in identifiers).
// lin_verify_view() checks one request from its view: its two chains, the shared chain, and its
// path in the binding tree (the Merkle tree over the seals of the requests' first chains).
//
// Two limits, hence the two macros required below: the hand-written plans (each chain's list of
// rounds) give every round the error bound 0 (`Priced` in identifiers), and lin_verify_view()
// itself marks a batch's table as having passed the committer's one-time test.
//
// In place of the protocol before LIN, a request sends one sampled oracle message of mask rows only
// (LinCase::index_mask_rows, by default 1), then its claim's target (the public value), masked by
// one entry of that message. SEND (blocks/iop/send.h) sends and samples oracle messages; a pad
// masks each answer.
#pragma once
#ifndef BLK_PLANS_BY_HAND_FOR_A_TEST
#error "this world lays plans by hand: say #define BLK_PLANS_BY_HAND_FOR_A_TEST before the first include (numbers/plan.h)"
#endif
#ifndef BLK_TESTED_TABLE_FOR_A_TEST
#error "this world runs later tests on tables that it vouches for: say #define BLK_TESTED_TABLE_FOR_A_TEST before the first include (iop/foldeq.h)"
#endif
#include <functional>
#include <string>
#include <vector>

#include "iop/index_cpu.h"
#include "iop/lin.h"
#include "iop/tests/iopp_support.h"

namespace blk {

// The sizes of one case. The matrix M has matrix_rows rows and 2^(k_lo + v) columns; each of the
// table's 2^v rows stores 2^k_lo of the columns. Record: stored answers to samples of the table's
// rows, each with a pad in the row's message. LIN's three fresh oracle messages: the folded row,
// Omega (one entry per row of M) and W (one per column); with rows_high, each of their rows holds
// consecutive entries (VectorLay, blocks/iop/foldeq.h). The binding tree has 2^log_leaves leaves.
struct LinCase {
  const char *name;
  unsigned v, k_lo;
  u64 matrix_rows;
  size_t spare;       // message entries of a table's row beyond the matrix and the pads
  unsigned answers;   // of the record, for each row
  unsigned columns;   // of the table, drawn by a batch
  unsigned batches;   // the number of batches that the table's random tails cover
  unsigned log_rate;  // the table's code has at most the rate 2^-log_rate
  size_t ell_u, ell_omega, ell_w;  // message lengths of the three fresh codes; equal lengths mean one code
  bool rows_high;
  unsigned requests, log_leaves;
  unsigned draws;     // at every stage of every proximity test
  bool ahead;         // the tests' fresh masks go out with earlier prover messages where allowed
  // If true, each of the three codes has a random tail for just the lists that open it: one for the
  // folded row and for W, 1 + requests for Omega; equal message lengths then need not mean one
  // code. If false, all three have Omega's tail. The hiding tests need tails with nothing spare.
  bool tight_tails = false;
  unsigned index_mask_rows = 1;  // the number of mask rows of the request's mask message, which in a key file is the index-mask message
};

// The public set-up of a case.
struct LinWorld {
  Codes codes;
  LinShape S;
  LinBindingShape binding;
  Digest configuration = toy_configuration();  // toy value; chains, binding tree and table share it
  OracleNotice input;  // the request's mask message
  TableTest test;  // the test on the table that both sides declare they run; no default
  Planned first, shared, second;
  explicit LinWorld(TableTest says) : test(says) {}
};

inline size_t lin_rows_for(u64 entries, size_t ell, bool rows_high) {
  const size_t rows = (size_t)((entries + ell - 1) / ell);
  return rows_high ? rows : (size_t)pow2_from(rows);
}
inline unsigned lin_trees(const Codes &codes, const std::vector<OracleNotice> &sent) { return (unsigned)lay_trees(codes, sent).shapes.size(); }
inline void lin_plan_send(Plan &p, const Codes &codes, const std::vector<OracleNotice> &sent) {
  for (unsigned i = 0; i < lin_trees(codes, sent); i++) p.send_root("root");
  std::vector<OracleShape> shapes;
  for (const OracleNotice &n : sent)
    if (n.shape.answers) shapes.push_back(n.shape);
  toy_plan_sample(p, codes, shapes, 1);
}
inline void lin_plan_seal(Plan &p) { p.close(R_SEAL, RoundKind::SEAL, ErrorKind::ZERO, Bound((u64)0), Priced::NOTHING); }

// The three stages of a proximity test on a group of these oracle messages.
inline std::vector<IoppStage> lin_stages(Codes &codes, CodeId code, CodeId mask_code, const std::vector<OracleShape> &members, unsigned draws) {
  u64 main = 0, mask = 0;
  for (const OracleShape &m : members) main += m.main_rows ? pow2_from(m.main_rows) : 0, mask += m.mask_rows ? pow2_from(m.mask_rows) : 0;
  const unsigned h = ceil_log2(mask), k = std::max(std::max(ceil_log2(main), h), 1u);
  std::vector<IoppStage> st{IoppStage{code, mask_code, k, h}};
  toy_later_stages(codes, st, {Later{1, 0}, Later{0, 0}}, draws, 1);
  return st;
}

// test: TableTest::ONE_TIME is the committer's own test on its table (wf_world.h); a batch's is
// TableTest::LATER.
inline LinWorld lin_world(const LinCase &t, TableTest test) {
  LinWorld w(test);
  LinShape &S = w.S;
  IndexShape &I = S.fold.index;
  I.v = t.v, I.answers = t.answers;
  I.code.ell = (size_t)(t.matrix_rows << t.k_lo) + t.spare + I.pads();
  I.code.r = tail_for_positions((u64)t.columns * t.batches);
  I.code.log_n = I.code.log_dim() + t.log_rate;
  S.k_lo = t.k_lo, S.matrix_rows = t.matrix_rows;
  S.fold.columns = t.columns, S.fold.rows_high = S.omega_rows_high = S.w_rows_high = t.rows_high;

  // The mask codes' tails cover two lists, because two tests open a split's mask message. With
  // `ahead`, the requests' groups use a second mask code.
  w.codes.mask.push_back(toy_code(16, tail_for_positions(2 * t.draws), 0));
  if (t.ahead) w.codes.mask.push_back(toy_code(32, tail_for_positions(2 * t.draws), 0)), S.own_mask_code = 1;
  S.masks_ahead = S.choice.masks_ride = t.ahead;
  // Omega is opened by the shared test and by every request's; the other two by one test each.
  auto base = [&](size_t ell, u64 lists) {
    const Code c = toy_code(ell, tail_for_positions((t.tight_tails ? lists : (u64)(1 + t.requests)) * t.draws), w.codes.mask.back().log_n);
    for (size_t i = 0; i < w.codes.base.size(); i++)
      if (w.codes.base[i] == c) return (CodeId)i;
    w.codes.base.push_back(c);
    return (CodeId)(w.codes.base.size() - 1);
  };
  auto main_message = [&](const char *name, size_t ell, u64 entries, u64 lists) {
    OracleShape s;
    s.name = name, s.code = base(ell, lists), s.main_deg = E_DEGREE, s.main_rows = lin_rows_for(entries + 1, ell, t.rows_high);
    return s;
  };
  auto mask_message = [&](const char *name, CodeId mask_code) {
    OracleShape s;
    s.name = name, s.mask_rows = 1, s.mask_code = mask_code;
    return s;
  };
  S.fold.u = main_message("the folded row", t.ell_u, S.fold.entries(), 1);
  S.omega = main_message("Omega", t.ell_omega, t.matrix_rows, 1 + t.requests);
  S.w = main_message("W", t.ell_w, u64(1) << S.k_col(), 1);
  S.shared_split = mask_message("the shared split's mask", S.shared_mask_code);
  S.own_split = mask_message("a request's split's mask", S.own_mask_code);
  w.input = OracleNotice{mask_message("a request's masks", S.own_mask_code), 0};
  w.input.shape.mask_rows = t.index_mask_rows;  // before the stages below are sized from this shape
  w.binding.log_leaves = t.log_leaves;

  if (S.shared_split_needed()) {
    S.test_u = lin_stages(w.codes, S.fold.u.code, S.shared_mask_code, {S.fold.u, S.shared_split}, t.draws);
    S.test_omega = lin_stages(w.codes, S.omega.code, S.shared_mask_code, {S.omega, S.shared_split}, t.draws);
  } else {
    S.test_u = lin_stages(w.codes, S.fold.u.code, S.shared_mask_code, {S.fold.u, S.omega}, t.draws);
  }
  if (S.own_split_needed()) {
    S.test_own = lin_stages(w.codes, S.w.code, S.own_mask_code, {w.input.shape, S.w, S.own_split}, t.draws);
    S.test_own_omega = lin_stages(w.codes, S.omega.code, S.own_mask_code, {S.omega, S.own_split}, t.draws);
  } else {
    S.test_own = lin_stages(w.codes, S.w.code, S.own_mask_code, {w.input.shape, S.w, S.omega}, t.draws);
  }

  // The three chains' plans.
  Plan first = toy_empty_plan(), shared = toy_empty_plan(), second = toy_empty_plan();
  lin_plan_send(first, w.codes, {w.input});
  first.send_scalars("the claim's target", 1);
  lin_plan_send(first, w.codes, lin_message_w(S));
  lin_plan_seal(first);

  // Unless the proof is already rejected, require_table() and require_sharing()
  // (blocks/chain/rounds.h) end the process if the plan does not name the run's table, its test and
  // how many requests can use LIN's shared point, drawn here.
  shared.close(R_LIN_POINT, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = S.k_col();
  shared.rounds_.back().table = priced_table(I, w.test);  // the shared point folds the table's rows
  shared.rounds_.back().sharing = u64(1) << w.binding.log_leaves;  // the binding tree's leaves
  lin_plan_send(shared, w.codes, lin_message_u(S));
  toy_plan_positions(shared, R_LOC, t.columns, I.code.log_n, false);
  shared.rounds_.back().table = priced_table(I, w.test);  // the drawn columns are this table's
  lin_plan_send(shared, w.codes, lin_message_omega(S));
  shared.close(R_LIN_ZETA, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = S.k_row();
  if (S.shared_split_needed()) shared.send_scalars("share", 1), shared.close_none(R_SPLIT);
  // Linear claims (`Row`) that each test batches: the record's, the drawn columns', LIN's claim
  // (Fold) or its share, and one sample claim per sampled oracle message. (Fold) ties Omega to the
  // folded row, (Eval) W to the target, (Dot) W to Omega.
  toy_plan_iopp(shared, w.codes, S.test_u, t.draws, S.choice, S.masks_ahead, I.answers + t.columns + 1 + 2);
  if (S.shared_split_needed()) toy_plan_iopp(shared, w.codes, S.test_omega, t.draws, S.choice, false, 1 + 2);
  lin_plan_seal(shared);

  if (S.own_split_needed()) second.send_scalars("share", 1), second.close_none(R_SPLIT);
  toy_plan_iopp(second, w.codes, S.test_own, t.draws, S.choice, S.masks_ahead, 2 + 3);  // (Eval), (Dot) or its share, three members
  if (S.own_split_needed()) toy_plan_iopp(second, w.codes, S.test_own_omega, t.draws, S.choice, false, 1 + 2);
  w.first = first, w.shared = shared, w.second = second;
  return w;
}

// ---- the matrix and the claims ----

struct LinMatrix {
  u64 rows, columns;
  std::vector<F> a;  // a[y * columns + c]
  F at(u64 y, u64 c) const { return a[y * columns + c]; }
};
inline LinMatrix lin_matrix(const LinShape &S, u64 seed) {
  LinMatrix M{S.matrix_rows, u64(1) << S.k_col(), {}};
  M.a.resize(M.rows * M.columns);
  TestCoins(seed).fill(M.a.data(), M.a.size());
  return M;
}
// The rows of the committed table that stores M, without the record's pads: IndexCpu's input.
inline Table lin_table_rows(const LinShape &S, const LinMatrix &M) {
  Table t(1, S.fold.index.rows(), S.fold.index.plain());
  for (u64 r = 0; r < t.rows; r++)
    for (u64 y = 0; y < M.rows; y++)
      for (u64 j = 0; j < (u64(1) << S.k_lo); j++) t.set(r, j + (y << S.k_lo), E::from(M.at(y, j + (r << S.k_lo))));
  return t;
}

// Request c's claim without its masks and target: the row weights g and the point psi. Both sides
// derive them from the seed.
inline LinClaim lin_claim_form(const LinShape &S, u64 seed, unsigned c) {
  TestCoins data(seed * 1000 + c);
  LinClaim claim;
  std::vector<E> point(S.k_row());
  for (E &x : point) x = data.e();
  claim.g.push_back(Tensor::eq(point.data(), S.k_row(), data.e()));
  Tensor one_row;  // all weight on one row of the matrix
  one_row.scalar = data.e();
  one_row.insert_choice(0, S.k_row(), c % S.matrix_rows);
  claim.g.push_back(one_row);
  if (S.k_row() > 0) {  // weights written out; those beyond the matrix's rows do not count
    auto values = std::make_shared<std::vector<E>>(size_t(1) << S.k_row());
    for (E &x : *values) x = data.e();
    Tensor written;
    written.push_dense(S.k_row());
    written.set_values(std::move(values), 0);
    claim.g.push_back(std::move(written));
  }
  if (S.k_row() > 1) {  // weights written out for the rows with top index bit 0: all in the matrix
    auto values = std::make_shared<std::vector<E>>(size_t(1) << (S.k_row() - 1));
    for (E &x : *values) x = data.e();
    Tensor block;
    block.push_dense(S.k_row() - 1);
    block.set_values(std::move(values), 0);
    block.insert_choice(S.k_row() - 1, 1, 0);
    claim.g.push_back(std::move(block));
  }
  claim.psi.resize(S.k_col());
  for (E &x : claim.psi) x = data.e();
  return claim;
}
// The row weights g, cut to the matrix's rows. lin_w(): W = g^T M.
inline std::vector<E> lin_weights(const LinShape &S, const LinClaim &claim) {
  std::vector<E> g(size_t(1) << S.k_row(), E::zero());
  for (const Tensor &t : claim.g) t.add_to(g.data());
  g.resize(S.matrix_rows);
  return g;
}
inline std::vector<E> lin_w(const LinShape &S, const LinClaim &claim, const LinMatrix &M) {
  const std::vector<E> g = lin_weights(S, claim);
  std::vector<E> w(M.columns, E::zero());
  for (u64 y = 0; y < M.rows; y++)
    for (u64 c = 0; c < M.columns; c++) w[c] += g[y] * M.at(y, c);
  return w;
}

// ---- a session ----

struct LinView {
  E target;                                  // the claim's, public
  std::vector<u8> first, path, shared, second;
};

inline const LinShape &lin_shared_chains_shape(const LinWorld &w) { return w.S; }

// The public inputs of a request's run, as items of its first chain's first query.
struct LinFirstBound {
  LinPublic lin;
  Public<OracleNotice> input;
  Public<LinBindingShape> binding;
  Public<u64> seed;  // of the claims' forms
  Public<unsigned> request;
};
inline LinFirstBound lin_bind_first(Binder &b, const LinWorld &w, const IndexName &table, u64 seed, unsigned c) {
  LinFirstBound x;
  b.configuration(w.configuration);
  lin_bind_parameters(b, w.codes, w.S, x.lin);
  x.input = b.parameter("a request's mask message", w.input), x.binding = b.parameter("the binding tree", w.binding);
  lin_bind_table(b, table, x.lin);
  x.seed = b.statement("the claims' seed", seed), x.request = b.statement("request", c);
  return x;
}

struct LinSessionOptions {
  E off_target = E::zero();  // added to the true target of request 0
  u64 tests_coin_seed = 0;   // not 0: seeds other TestCoins for the shared part's proximity tests
  std::vector<E> add_to_w;   // added to the first entries of request 0's W
  std::vector<PadLook> *pads = nullptr;  // SEND adds each pad, for `pads_fault` (blocks/iop/send.h)
  // For hiding tests over whole sessions. `coins` replaces TestCoins(coin_seed); the chains' salts
  // (`toy_random_bytes`) still derive from coin_seed. With `look`, every Writer keeps a list of its
  // items and is passed to it when complete, as "first", "shared", "second" or "path", with the
  // request's number (0 for the shared chain).
  Coins *coins = nullptr;
  std::function<void(const char *chain, unsigned request, const Writer &)> look;
  bool not_counted = false;  // misuse: the shared part gets no `LinBegun`, so the process ends
  // For a hiding test that fixes the verifier's messages. The record
  // varies with the committer's randomness (`Coins`) and the matrix, and the first queries of the
  // first and shared chains contain it. If set, they contain *query_record instead (any value that
  // the test keeps the same in all runs), while the record's claims use the real record (`rec`).
  // lin_verify_view() rejects such a session, and ends the process if it is given a query_record
  // itself. The record is public: the test adds it to its view.
  const IndexRecord *query_record = nullptr;
};

// A source of private randomness that a test can redirect in the middle of a run.
class RelayCoins : public Coins {
 public:
  explicit RelayCoins(Coins &from) : from_(&from) {}
  void take_from(Coins &from) { from_ = &from; }
  void fill(F *out, size_t n) override { from_->fill(out, n); }
  void bytes(u8 *out, size_t n) override { from_->bytes(out, n); }

 private:
  Coins *from_;
};

// served: false if the table's random tails allow no further batch. column_rows: claims on drawn
// columns; different_column_rows: their distinct targets.
struct LinSession {
  bool served = false;
  std::vector<LinView> views;
  size_t column_rows = 0, different_column_rows = 0;
};

// Proves a batch of claims about M on the table in `index`, whose record is `rec`. seed: of the
// public claims; coin_seed: of the prover's private randomness. M may differ from the committed
// matrix, but no caller in this repository passes another. lin_begin() first counts the batch
// against the table and gives the `LinBegun` that the shared part requires.
inline LinSession lin_session(const LinCase &t, const LinWorld &w, IndexHolder &index, const IndexRecord &rec, const LinMatrix &M, u64 seed,
                              u64 coin_seed, const LinSessionOptions &opt = LinSessionOptions()) {
  const LinShape &S = w.S;
  LinSession out;
  TestCoins own_coins(coin_seed);
  Coins &coins = opt.coins ? *opt.coins : own_coins;
  EngineCpu engine;
  struct Request {
    Writer first, second;
    std::unique_ptr<ProverRounds> ch;
    std::unique_ptr<ProverContext> P;
    LinFirstBound bound;
    LinFirst lin;
    Sealed seal;
  };
  const IndexName table{index.commitment(), opt.query_record ? *opt.query_record : rec};
  std::vector<Request> req(t.requests);
  LinBinding binding(w.binding, w.configuration, coins);
  std::unique_ptr<LinBegun> begun;
  if (!lin_begin(index, S, begun)) return out;
  out.views.resize(t.requests);

  for (unsigned c = 0; c < t.requests; c++) {
    Request &r = req[c];
    Binder b(1, w.first);
    r.bound = lin_bind_first(b, w, table, seed, c);
    if (opt.look) r.first.keep_list(), r.second.keep_list();
    r.ch.reset(new ProverRounds(b, toy_random_bytes(coin_seed + 100 + c), r.first));
    r.P.reset(new ProverContext(r.bound.lin.codes, *r.ch, engine, coins));
    r.P->pads = opt.pads;
    std::vector<OracleDraft> drafts;
    drafts.push_back(lin_draft(*r.P, r.ch->get(r.bound.input).shape, false, r.ch->get(r.bound.input).leaf_class, {}));
    const OracleId input = send_prove(*r.P, std::move(drafts))[0];
    LinClaim claim = lin_claim_form(S, r.ch->get(r.bound.seed), r.ch->get(r.bound.request));
    std::vector<E> wc = lin_w(S, claim, M);
    MaskSlot slot = r.P->st.masks(input).take();
    claim.target = mle_eval(wc.data(), wc.size(), claim.psi.data(), S.k_col()) + r.P->mask_value(slot);
    if (c == 0) claim.target += opt.off_target;
    for (size_t j = 0; c == 0 && j < opt.add_to_w.size(); j++) wc[j] += opt.add_to_w[j];
    claim.add_mask(std::move(slot), E::one());
    out.views[c].target = claim.target;
    r.ch->send(claim.target);  // the claim's form comes from the first query, its target from this message
    r.lin = lin_first_prove(*r.P, r.bound.lin, std::move(claim), wc);
    BLK_REQUIRE(r.ch->seal(R_SEAL, r.seal), "a prover's channel does not refuse");
    r.ch->finish();  // otherwise `finish` on a chain that follows this one ends the process
    if (opt.look) opt.look("first", c, r.first);
    binding.put(c, r.seal.digest());
  }

  Writer shared_bytes;
  if (opt.look) shared_bytes.keep_list();
  Binder shared_b(LIN_LABEL_SHARED, w.shared);
  shared_b.configuration(w.configuration);
  const LinPublic shared_x = lin_bind_shared(shared_b, w.codes, lin_shared_chains_shape(w), table, binding.root());
  ProverRounds shared_ch(shared_b, Digest{}, shared_bytes);
  TestCoins other_coins(opt.tests_coin_seed);
  RelayCoins shared_coins(coins);
  ProverContext shared_P(shared_x.codes, shared_ch, engine, shared_coins);
  shared_P.pads = opt.pads;
  LinShared shared;
  if (opt.not_counted) begun.reset();
  if (opt.query_record) {  // not `lin_shared_prove`, which takes the record from the first query
    shared.bound = shared_x;
    lin_shared_prove_on_shape(shared_P, shared_ch.get(shared_x.shape), shared_x.places, w.test, index, rec, std::move(begun), shared);
  } else {
    lin_shared_prove(shared_P, shared_x, w.test, index, std::move(begun), shared);
  }
  out.column_rows = out.different_column_rows = 0;  // one row per draw, also of a repeated column
  std::vector<E> seen;
  for (const Row &row : shared_P.st.group(shared.group_u).rows) {
    if (row.kind != RowKind::GENERATOR) continue;
    out.column_rows++;
    if (std::find(seen.begin(), seen.end(), row.target) == seen.end()) seen.push_back(row.target), out.different_column_rows++;
  }
  if (opt.tests_coin_seed) shared_coins.take_from(other_coins);
  lin_shared_tests_prove(shared_P, shared_x, shared);
  Sealed shared_seal;
  BLK_REQUIRE(shared_ch.seal(R_SEAL, shared_seal), "a prover's channel does not refuse");
  shared_ch.finish();
  if (opt.look) opt.look("shared", 0, shared_bytes);
  shared_P.st.require_all_tested();

  for (unsigned c = 0; c < t.requests; c++) {
    Request &r = req[c];
    Binder b(LIN_LABEL_SECOND, w.second);
    b.configuration(w.configuration);
    lin_bind_second(b, r.seal, shared_seal);
    ProverRounds ch(b, toy_random_bytes(coin_seed + 200 + c), r.second);
    ProverContext P = lin_go_on(std::move(*r.P), ch);
    P.pads = opt.pads;
    const LinSecond second = lin_second_prove(P, r.bound.lin, r.lin, shared_P, shared);
    lin_second_tests_prove(P, r.bound.lin, r.lin, second);
    ch.finish();
    if (opt.look) opt.look("second", c, r.second);
    P.st.require_all_tested();
    Writer path;
    if (opt.look) path.keep_list();
    binding.write_path(path, c);
    if (opt.look) opt.look("path", c, path);
    out.views[c].first = r.first.bytes(), out.views[c].path = path.bytes(), out.views[c].shared = shared_bytes.bytes(), out.views[c].second = r.second.bytes();
  }
  lin_release(shared_P, shared);
  out.served = true;
  return out;
}

// What a malicious prover learns by running the verifier's steps on one of its own views. psi: the
// shared point. No caller in this repository passes one to lin_verify_view().
struct LinPeek {
  bool reached = false;
  std::vector<E> psi;
  E batching;
};
// The verifier of request c. It gets the table's key (`IndexKey`), the claims' seed and the
// request's view, and no other request's view. why: the step that rejected, if any.
// query_record: replaces the key's record in the first queries. That weakened verifier is the
// accepting control of a malicious prover that is not part of this repository. Here a non-null
// value ends the process (the first line of lin_verify_view()).
// other_shared_context: deliberate misuse: the second part gets the context of a shared chain whose
// seal its own chain's first query does not contain.
// handed: if set, replaces `read` below, to test that the process ends on an untested table or on
// one tested under another setting.
inline bool lin_verify_view(const LinWorld &w, const IndexKey &key, u64 seed, unsigned c, const LinView &view, const IndexRecord *query_record = nullptr,
                            const char **why = nullptr, LinPeek *peek = nullptr, bool other_shared_context = false, const TableRead *handed = nullptr) {
  BLK_REQUIRE(!query_record, "a verifier that puts another record than the key's into its first queries is no part of these tests");
  const IndexName table{key.commitment(), query_record ? *query_record : key.rec};
  const char *unused;
  const char *&reason = why ? *why : unused;
  reason = "";
  // foldeq_columns_verify() (blocks/iop/foldeq.h) ends the process if a later test reads a table
  // not marked as tested under the run's own `Setting`. Here no one-time test precedes a later
  // test, so this function marks the table itself.
  const TableRead read = handed ? *handed : w.test == TableTest::ONE_TIME ? TableRead(key) : TableRead::on_a_tests_word(key, w.shared->set);

  Reader first_bytes(view.first.data(), view.first.size());
  Binder first_b(1, w.first);
  const LinFirstBound x = lin_bind_first(first_b, w, table, seed, c);
  VerifierSession session;  // one verdict: one session for all its chains
  VerifierRounds first_ch(first_b, first_bytes, session);
  VerifierContext first_V(x.lin.codes, first_ch);
  std::vector<OracleId> ids;
  if (!send_verify(first_V, std::vector<OracleNotice>{first_ch.get(x.input)}, ids)) return reason = "first part: the input message", false;
  LinClaim claim = lin_claim_form(first_ch.get(x.lin.shape), first_ch.get(x.seed), first_ch.get(x.request));
  if (!first_ch.recv(claim.target)) return reason = "first part: the target", false;
  if (claim.target != view.target) return reason = "first part: the proof is about another target", false;
  claim.add_mask(first_V.st.masks(ids[0]).take(), E::one());
  LinFirst first;
  if (!lin_first_verify(first_V, x.lin, std::move(claim), first)) return reason = "first part", false;
  Sealed seal;
  if (!first_ch.seal(R_SEAL, seal) || !first_ch.finish()) return reason = "first part: its end", false;

  Reader path(view.path.data(), view.path.size());
  LinBindingRoot binding_root;
  if (!lin_binding_root(path, first_ch.get(x.binding), first_ch.configuration(), seal.digest(), binding_root) || !path.at_end()) return reason = "path", false;

  Reader shared_bytes(view.shared.data(), view.shared.size());
  Binder shared_b(LIN_LABEL_SHARED, w.shared);
  shared_b.configuration(w.configuration);
  const LinPublic shared_x = lin_bind_shared(shared_b, w.codes, lin_shared_chains_shape(w), table, binding_root);
  VerifierRounds shared_ch(shared_b, shared_bytes, session);
  VerifierContext shared_V(shared_x.codes, shared_ch);
  LinShared shared;
  if (!lin_shared_verify(shared_V, shared_x, w.test, read, shared)) return reason = "shared part", false;
  if (!lin_shared_tests_verify(shared_V, shared_x, shared)) return reason = "shared part: a proximity test", false;
  Sealed shared_seal;
  if (!shared_ch.seal(R_SEAL, shared_seal) || !shared_ch.finish()) return reason = "shared part: its end", false;
  shared_V.st.require_all_tested();

  Reader second_bytes(view.second.data(), view.second.size());
  Binder second_b(LIN_LABEL_SECOND, w.second);
  second_b.configuration(w.configuration);
  lin_bind_second(second_b, seal, shared_seal);
  VerifierRounds second_ch(second_b, second_bytes, session);
  VerifierContext V = lin_go_on(std::move(first_V), second_ch);
  LinSecond second;
  if (other_shared_context) {  // the same bytes read again, on a new channel (`VerifierRounds`)
    Reader again_bytes(view.shared.data(), view.shared.size());
    Binder again_b(LIN_LABEL_SHARED, w.shared);
    again_b.configuration(w.configuration);
    const LinPublic again_x = lin_bind_shared(again_b, w.codes, w.S, table, binding_root);
    VerifierRounds again_ch(again_b, again_bytes, session);
    VerifierContext again_V(again_x.codes, again_ch);
    LinShared again;
    if (!lin_shared_verify(again_V, again_x, w.test, read, again)) return reason = "shared part, read again", false;
    return lin_second_verify(V, x.lin, first, again_V, again, second);
  }
  if (!lin_second_verify(V, x.lin, first, shared_V, shared, second)) return reason = "second part", false;
  if (peek) {  // a malicious prover stops at the first challenge of the request's own test
    auto from = second_ch.close_algebraic(R_BAT);
    peek->batching = from.draw(), peek->psi = shared.psi, peek->reached = second_ch.ok();
    return reason = "looked at only", false;
  }
  if (!lin_second_tests_verify(V, x.lin, first, second)) return reason = "second part: a proximity test", false;
  if (!second_ch.finish()) return reason = "second part: its end", false;
  V.st.require_all_tested();
  // The session must contain the table's opening under its root: one value per vertex, whichever
  // chain reads the table.
  BLK_REQUIRE(session.record(key.root).vertices() > 0, "the world: the table's reader has entered nothing into the session's record");
  return session.accepted();
}

}  // namespace blk
