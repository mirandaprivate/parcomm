// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// LIN: proves one linear claim for each request of a batch on one committed table (`Index` in
// identifiers; blocks/iop/foldeq.h), with one fold of the table for the whole batch. It leaves no
// linear claim (`Row` in identifiers) open: it ends in proximity tests (`iopp` in identifiers). The
// table stores a matrix M (LinShape). Request c claims, for public g_c, psi_c, tau_c (LinClaim):
//     < g_c (x) eq(psi_c, .), M >  +  (mask entries of its own oracle messages)  =  tau_c.
// g_c weighs M's rows, the point psi_c (via eq) its columns; tau_c is LinClaim::target. (x) is the
// tensor product. An oracle message, such as W_c, G_u or Omega, is a set of encoded rows sent under
// a Merkle root; a mask entry is one random element of E, the extension field, in its mask rows.
// LIN has three parts, each on its own hash chain (a Fiat-Shamir transcript), so that a request's
// proof can be checked without the other requests' parts:
//   first, each request:   the prover sends W_c = g_c^T M;
//   shared, once a batch:  the verifier draws the shared point psi*; the prover sends G_u, the
//                          table's rows folded into one, and Omega = M eq(psi*, .);
//   second, each request:  < g_c, Omega > is tied to W_c at psi*.
// Callers use the functions that take a LinPublic. The prover's order: lin_begin(), all first
// parts, the binding tree (LinBinding: a Merkle tree over their seals; a seal is the digest that
// ends a chain), the shared part, all second parts, lin_release().
// Tests: blocks/iop/tests/lin_test.cpp (honest views, false targets, altered views, a caller's
// mistakes).
//
// Rows (Eval) and (Dot) have the form of two checks in: Nguyen, Seiler, "Greyhound: Fast Polynomial Commitments from
// Lattices", Cryptology ePrint Archive 2024/1293, section 1.2, "Simple proof of quadratic relations", equation (2),
// and section 3.1, Figure 1 (only this form is taken; there the combination of the columns is given; here it,
// Omega, is committed).
#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "chain/rounds.h"
#include "field/field.h"
#include "field/mle.h"
#include "iop/context.h"
#include "iop/foldeq.h"
#include "iop/iopp.h"
#include "iop/relation.h"
#include "iop/send.h"
#include "iop/split.h"
#include "iop/tensor.h"
#include "numbers/round_names.h"
#include "tree/tree.h"

namespace blk {

// A key file's two chain labels (ASCII "LIN", then 1 or 2); a proof's: blocks/frame/heads.h.
constexpr u64 LIN_LABEL_SHARED = 0x4c494e01, LIN_LABEL_SECOND = 0x4c494e02;

// ---------------------------------------------------------------------------------------------------------------------
// The public shape
// ---------------------------------------------------------------------------------------------------------------------

// M has matrix_rows rows and 2^k_col() columns: a row's index has k_row() bits, a column's k_col().
// Entry (y, j + 2^k_lo r) of M, j < 2^k_lo, is message entry j + 2^k_lo y of row r of the table,
// which has 2^v rows (`fold.index.v`).
// A proximity test reads one group of oracle messages, under one code. If G_u and Omega differ in
// code, the claim (Fold), defined under "The linear claims", reads two groups, and SPLIT (split.h)
// replaces it by one claim on each, with a mask entry from the mask-only message `shared_split`,
// sent with Omega. Likewise (Dot) and `own_split`, sent with W_c. LIN's splits never take a mask
// from a caller's message.
struct LinShape {
  FoldeqShape fold;     // the table and G_u
  unsigned k_lo = 0;    // a row of the table holds 2^k_lo columns of the matrix
  u64 matrix_rows = 0;
  OracleShape omega, w;  // main rows over E; layout: the flags below (`VectorLay`, foldeq.h)
  bool omega_rows_high = false, w_rows_high = false;
  OracleShape shared_split, own_split;  // mask rows only; sent only where a split is needed
  // Oracle messages of one prover message share tree leaves (one leaf class) only if they are
  // always read at the same lists of positions. Omega is read by all requests, a split's mask
  // message by two tests: each has its own class.
  unsigned omega_leaf_class = 1, shared_split_leaf_class = 2, w_leaf_class = 0, own_split_leaf_class = 3;
  CodeId shared_mask_code = 0, own_mask_code = 0;  // of the shared part's groups; of a request's
  // The first fold of a proximity test has a fresh mask, an oracle message that the fold (FOLD,
  // fold.h) sends first unless it was sent before. If true, it is sent ahead: with G_u for the test
  // on G_u's group, with W_c for the request's own test. Each is read at the same single list as
  // the message it is sent with, so it shares that message's tree.
  bool masks_ahead = false;
  // The stages of the four proximity tests: on G_u's group (with Omega in it, if the codes agree);
  // on Omega's group; on the request's own group (with Omega in it, if the codes agree); on the
  // request's group around Omega. The second and the fourth run only where a split is needed. All
  // four run under `choice` (iopp.h).
  std::vector<IoppStage> test_u, test_omega, test_own, test_own_omega;
  IoppChoice choice;

  unsigned k_col() const { return k_lo + fold.index.v; }
  unsigned k_row() const { return ceil_log2(matrix_rows); }
  bool shared_split_needed() const { return fold.u.code != omega.code; }
  bool own_split_needed() const { return w.code != omega.code; }
};

// Null, or what is wrong with the shape by itself; Binder calls it (blocks/chain/public.h).
inline const char *public_fault(const LinShape &S) {
  if (const char *f = public_fault(S.fold)) return f;
  if (S.matrix_rows < 1 || S.k_lo > 30 || S.k_row() > 30 || S.k_col() > 60) return "the matrix has no rows, or too many rows or columns";
  if ((S.matrix_rows << S.k_lo) > S.fold.index.plain()) return "the matrix does not fit the table's rows beside their pads";
  for (const OracleShape *o : {&S.omega, &S.w}) {
    if (const char *f = public_fault(*o)) return f;
    if (o->main_rows == 0 || o->main_deg != E_DEGREE) return "an oracle message of the test has no main rows over E";
    if (o->answers < 1) return "an oracle message of the test is not sampled";
  }
  if (S.fold.u.mask_rows && S.fold.u.mask_code != S.shared_mask_code) return "the folded row's mask rows have another code than its group";
  if (S.w.mask_rows && S.w.mask_code != S.own_mask_code) return "W's mask rows have another code than its group";
  if (S.omega.mask_rows) return "Omega has mask rows; it is a member of groups with different mask codes and may have main rows only";
  if (S.shared_split_needed() && S.omega_leaf_class == S.shared_split_leaf_class) return "Omega and the shared split's mask message have one leaf class; they are read at different lists";
  if (S.own_split_needed() && S.w_leaf_class == S.own_split_leaf_class) return "W and a request's split's mask message have one leaf class; they are read at different lists";
  for (int own = 0; own < 2; own++) {
    if (!(own ? S.own_split_needed() : S.shared_split_needed())) continue;
    const OracleShape &o = own ? S.own_split : S.shared_split;
    if (o.main_rows != 0 || o.mask_rows == 0 || o.answers < 1 || o.mask_code != (own ? S.own_mask_code : S.shared_mask_code))
      return "a split needs a sampled message of mask rows alone, under its groups' mask code";
  }
  return nullptr;
}
// The same, with the codes that the shape names. The first and shared parts end the process on a
// fault: check a shape read from a file with this first.
inline const char *lin_fault(const Codes &codes, const LinShape &S) {
  if (const char *f = public_fault(S)) return f;
  if (const char *f = foldeq_fault(codes, S.fold)) return f;
  if (S.omega.code >= codes.base.size() || S.w.code >= codes.base.size()) return "an oracle message of the test names a code that is not there";
  if (S.shared_mask_code >= codes.mask.size() || S.own_mask_code >= codes.mask.size()) return "a group of the test names a mask code that is not there";
  if (S.matrix_rows > vector_room(codes, S.omega, S.omega_rows_high)) return "Omega's layout runs into its pad";
  if ((u64(1) << S.k_col()) > vector_room(codes, S.w, S.w_rows_high)) return "W's layout runs into its pad";
  return nullptr;
}

// The shape of the first fold of a test, as far as its fresh mask needs it.
inline FoldShape lin_fold_shape(const std::vector<IoppStage> &test, unsigned leaf_class) {
  BLK_REQUIRE(!test.empty(), "LIN: a test without stages");
  FoldShape F = iopp_fold_shape(test[0], GroupLayout());
  F.mask_leaf_class = leaf_class;
  return F;
}
// The oracle messages of LIN's three prover messages that have any: W_c's, G_u's, Omega's. The
// verifier and the planning code (blocks/frame/lin_frame.h) take them from here; the prover's
// drafts follow the same order.
inline std::vector<OracleNotice> lin_message_w(const LinShape &S) {
  std::vector<OracleNotice> m{OracleNotice{S.w, S.w_leaf_class}};
  if (S.own_split_needed()) m.push_back(OracleNotice{S.own_split, S.own_split_leaf_class});
  if (S.masks_ahead) m.push_back(fold_mask_notice(lin_fold_shape(S.test_own, S.w_leaf_class)));
  return m;
}
inline std::vector<OracleNotice> lin_message_u(const LinShape &S) {
  std::vector<OracleNotice> m{OracleNotice{S.fold.u, S.fold.leaf_class}};
  if (S.masks_ahead) m.push_back(fold_mask_notice(lin_fold_shape(S.test_u, S.fold.leaf_class)));
  return m;
}
inline std::vector<OracleNotice> lin_message_omega(const LinShape &S) {
  std::vector<OracleNotice> m{OracleNotice{S.omega, S.omega_leaf_class}};
  if (S.shared_split_needed()) m.push_back(OracleNotice{S.shared_split, S.shared_split_leaf_class});
  return m;
}

inline void public_bytes(Writer &w, const LinShape &S) {
  public_bytes(w, S.fold), public_bytes(w, S.k_lo), public_bytes(w, S.matrix_rows), public_bytes(w, S.omega), public_bytes(w, S.w);
  public_bytes(w, S.omega_rows_high), public_bytes(w, S.w_rows_high), public_bytes(w, S.shared_split), public_bytes(w, S.own_split);
  for (unsigned c : {S.omega_leaf_class, S.shared_split_leaf_class, S.w_leaf_class, S.own_split_leaf_class}) public_bytes(w, c);
  public_bytes(w, S.shared_mask_code), public_bytes(w, S.own_mask_code), public_bytes(w, S.masks_ahead);
  for (const std::vector<IoppStage> *test : {&S.test_u, &S.test_omega, &S.test_own, &S.test_own_omega}) public_bytes(w, *test);
  public_bytes(w, S.choice);
}

// LIN's public inputs as handles into a chain's first query, the hash input that fixes parameters
// and statement before the prover speaks (blocks/chain/public.h).
struct LinPublic {
  Public<Codes> codes;
  Public<LinShape> shape;
  Public<IndexName> table;
  u64 places = 0;  // shared chain only: the binding tree's places (from `lin_bind_shared`); else 0
};
// Codes and shape are parameters, fixed before a table or a request is known. The table's
// commitment and record (IndexName, foldeq.h; a record: stored answers to samples of the table's
// rows) are statement: they come from a file that the committer wrote, and it never
// sets a parameter. So neither the parameters' digest nor the run's value, from which the leaves of
// a chain's trees start, depends on that file. Order of the items: configuration, parameters (LIN's
// and the caller's, in any order), table, the rest of the statement.
inline void lin_bind_parameters(Binder &b, const Codes &codes, const LinShape &S, LinPublic &x) {
  x.codes = b.parameter("codes", codes), x.shape = b.parameter("the test on the table", S);
}
// The first item of the statement. After it the Binder takes no further parameter.
inline void lin_bind_table(Binder &b, const IndexName &table, LinPublic &x) { x.table = b.statement("table", table); }
// Both calls, for a chain that adds no parameter after LIN's.
inline LinPublic lin_bind(Binder &b, const Codes &codes, const LinShape &S, const IndexName &table) {
  LinPublic x;
  lin_bind_parameters(b, codes, S, x);
  lin_bind_table(b, table, x);
  return x;
}

// One request's claim:  < g (x) eq(psi, .), M >  +  sum of coeff * (mask entry)  =  target.
// The row weights are the sum of the Tensors in `g` (tensor.h: structured coefficient vectors);
// their entries from matrix_rows on are ignored. `masks` names entries of mask rows of the
// request's own oracle messages, sent and sampled on its chain.
// The caller's duty, not checked here: every field, `target` and row weights included, is a
// function of what the request's first chain has absorbed before its seal: its first query, the
// prover's messages, the challenges. A prover that may choose a field later fits it to the
// challenges. The second part takes the claim from LinFirst.
struct LinClaim {
  std::vector<Tensor> g;    // the row weights, as a sum; each Tensor has k_row() variables
  std::vector<E> psi;       // k_col() coordinates
  std::vector<Unit> masks;  // the claim's mask entries, each with its coefficient
  E target = E::zero();

  // Consumes the mask entry.
  void add_mask(MaskSlot &&slot, const E &coeff) {
    const MaskSlot s = std::move(slot);
    masks.push_back(Unit{s.oracle(), Part::MASK, s.index(), coeff});
  }
};
// ---------------------------------------------------------------------------------------------------------------------
// The linear claims
// ---------------------------------------------------------------------------------------------------------------------

//   (Eval)  W_c at psi_c  +  the claim's mask entries  =  tau_c
//   (Fold)  Omega at zeta  =  < eq(psi_lo, .) (x) eq(zeta, .), u >,  u: the folded row in G_u
//   (Dot)   < g_c, Omega >  =  W_c at psi*
// "X at p" is the value at p of the multilinear extension of the vector X. psi* = (psi_lo, gamma)
// and zeta are the shared part's challenges. (Fold) and (Dot) sum over the rows y < matrix_rows of
// M only: a form that ran on would read entries that mean nothing.
// LIN itself checks nothing in a proof: FOLDEQ (foldeq.h: the fold of the table) checks the opened
// columns, the proximity tests the claims. All claims' values (`target` in identifiers) are 0,
// public, sent (sample answers, shares of splits), or computed from opened symbols of the table.

inline Row lin_eval_row(const LinShape &S, const LinClaim &claim, OracleId w, const VectorLay &w_lay) {
  BLK_REQUIRE(claim.psi.size() == S.k_col(), "LIN: a claim's point has one coordinate for each column bit");
  Row row;
  row.name = "LIN: the claim on W";
  row.units = claim.masks;
  row.target = claim.target;
  row.add(w, Part::MAIN, w_lay.on_table(Tensor::eq(claim.psi.data(), S.k_col())));
  return row;
}

inline Row lin_fold_row(const LinShape &S, const std::vector<E> &psi, const std::vector<E> &zeta, const FoldedRow &u, OracleId omega,
                        const VectorLay &omega_lay) {
  Row row;
  row.name = "LIN: Omega against the folded row";
  std::vector<Tensor> cut;
  tensor_below(Tensor::eq(zeta.data(), S.k_row()), 0, S.k_row(), S.matrix_rows, cut);
  for (Tensor &t : cut) row.add(omega, Part::MAIN, omega_lay.on_table(std::move(t)));
  Tensor on_u = Tensor::eq(psi.data(), S.k_lo, -E::one());
  for (unsigned t = 0; t < S.k_row(); t++) on_u.push(E::one() - zeta[t], zeta[t]);
  cut.clear();
  tensor_below(on_u, S.k_lo, S.k_row(), S.matrix_rows, cut);
  for (Tensor &t : cut) row.add(u.oracle, Part::MAIN, u.lay.on_table(std::move(t)));
  return row;
}

// weights: the claim's row weights (LinClaim::g), or the same vector as lin_weights_of_the_prover()
// returns it.
inline Row lin_dot_row(const LinShape &S, const std::vector<Tensor> &weights, const std::vector<E> &psi, OracleId omega, const VectorLay &omega_lay, OracleId w,
                       const VectorLay &w_lay) {
  Row row;
  row.name = "LIN: the row weights on Omega against W at the shared point";
  std::vector<Tensor> cut;
  for (const Tensor &g : weights) {
    BLK_REQUIRE(g.vars() == S.k_row(), "LIN: row weights have one variable for each row bit");
    tensor_below(g, 0, S.k_row(), S.matrix_rows, cut);
  }
  for (Tensor &t : cut) row.add(omega, Part::MAIN, omega_lay.on_table(std::move(t)));
  row.add(w, Part::MAIN, w_lay.on_table(Tensor::eq(psi.data(), S.k_col(), -E::one())));
  return row;
}

// The prover's own form of a claim's row weights. (Dot) reads only the sum of the claim's Tensors,
// and sums in the field are exact, so no byte of a proof depends on how the sum is represented. A
// claim can have thousands of Tensors (the committer's, in a key file: 11,401 at a model of 400
// billion weights; a request's: some hundreds), and the prover's engine (blocks/engine/engine.h:
// its heavy steps, on a CPU or a GPU) does work for each. From lin_written_out_from Tensors on, the
// prover adds them up once, on the CPU, into one Tensor that is dense over all row bits: 2^k_row()
// elements of memory. The weights are public. The verifier keeps the Tensors as they are.
// blocks/iop/tests/lin_test.cpp sets lin_written_out_from to 0 (always) and to the largest value
// (never) and requires the same proof bytes.
inline size_t lin_written_out_from = 1024;  // the prover's setting: a number of Tensors
inline std::vector<Tensor> lin_weights_of_the_prover(const LinShape &S, const std::vector<Tensor> &g) {
  if (g.size() < lin_written_out_from || S.k_row() == 0) return g;
  auto sum = std::make_shared<std::vector<E>>(size_t(1) << S.k_row(), E::zero());
  for (const Tensor &t : g) {
    BLK_REQUIRE(t.vars() == S.k_row(), "LIN: row weights have one variable for each row bit");
    t.add_to(sum->data());
  }
  Tensor one;
  one.push_dense(S.k_row());
  one.set_values(std::move(sum), 0);
  return {std::move(one)};
}

// A vector as the main table of an oracle message. All other entries are 0, the sample's pad (the
// random element that masks the sample's answer) too: send_prove() fills in every pad.
inline Table lin_table(const Codes &codes, const OracleShape &s, bool rows_high, const std::vector<E> &entries) {
  const VectorLay lay = vector_lay(codes, s, rows_high);
  Table t(E_DEGREE, s.main_rows, codes.base[s.code].ell);
  for (size_t j = 0; j < entries.size(); j++) t.set_entry(lay.index(j), entries[j]);
  return t;
}
inline OracleDraft lin_draft(ProverContext &P, const OracleShape &s, bool rows_high, unsigned leaf_class, const std::vector<E> &entries) {
  OracleDraft d;
  d.shape = s, d.leaf_class = leaf_class;
  if (s.main_rows) d.main = P.engine.put(lin_table(P.st.codes(), s, rows_high, entries));
  if (s.mask_rows) d.mask = Send::fresh_mask(P.st.codes(), s, P.coins), d.mask_draw = P.coins.last_draw();
  return d;
}
// [own-test-lin-mask] A self-check after each send_prove() in this file. For each sent message with
// mask rows, the mask table (but not the pad's entries) and the mask rows' random tail (the random
// coefficients encoded with each row), as the prover's context keeps them, must equal the private
// random values drawn for them, regenerated from the draws' numbers (not_as_drawn(), send.h).
// Later steps compute from these two tables, and a value lost between its draw and the sending
// gives a proof, or a key file, that verifies and hides less: if it is a mask entry of a split's
// message, the element that it masks goes out in the clear.
// Of a first fold's fresh mask sent with W_c only the tail is compared here: its table is made from
// several draws, which FOLD compares itself, with the tail once more (fold_mask_as_drawn(), fold.h).
// Not seen: the mask rows and tail of G_u, which foldeq_prove() sends for the shared part (a first
// fold's fresh mask sent with it is left to FOLD); a generator whose values are not random;
// a fault that hits both a draw and its regeneration; a value overwritten after this check; the
// copies of the two tables that the engine has encoded by then. With a generator that does not
// number its draws, nothing is compared. The check draws nothing and sends nothing.
inline const std::vector<OracleId> &lin_masks_as_drawn(ProverContext &P, const std::vector<OracleId> &sent) {
  u64 differ = 0;
  for (OracleId o : sent)
    if (P.st.oracle(o).shape.mask_rows) differ += not_as_drawn(P, o);  // all of them: no early end
  BLK_REQUIRE_NAMED(differ == 0, "own-test-lin-mask",
                    "internal error (own-test-lin-mask): values of the mask rows of a message of the test on the table, or of their random tail, are not the random values that were drawn for them. The program or the machine has a fault; nothing of this run may be used");
  return sent;
}

// ---------------------------------------------------------------------------------------------------------------------
// First part: a request by itself
// ---------------------------------------------------------------------------------------------------------------------

// On the request's first chain. One SEND (send.h: sends oracle messages and samples each out of
// domain) of W_c, one entry of E for each column of M. (Eval) is added to the request's own group:
// W_c, the oracle messages of the claim's masks and, where (Dot) is split, `own_split`.

// What the first part leaves for the second.
struct LinFirst {
  OracleId w = NONE_ID, split = NONE_ID;
  GroupId own = NONE_ID;  // the request's own group
  FoldMask mask;          // of the first fold of the test on that group, if sent with W_c
  LinClaim claim;
};

inline void lin_first_rows(Statement &st, const LinShape &S, LinClaim claim, const std::vector<OracleId> &ids, LinFirst &out) {
  out.w = ids[0];
  out.claim = std::move(claim);
  out.own = st.new_group("LIN: a request's own rows", S.w.code, S.own_mask_code);
  std::vector<OracleId> inputs;
  for (const Unit &u : out.claim.masks) inputs.push_back(u.oracle);
  std::sort(inputs.begin(), inputs.end());
  inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());
  for (OracleId o : inputs) st.join(out.own, o);
  st.join(out.own, out.w);
  if (S.own_split_needed()) st.join(out.own, out.split = ids[1]);
  if (S.masks_ahead) out.mask.sent_as(ids.back());
  st.add_row(out.own, lin_eval_row(S, out.claim, out.w, vector_lay(st.codes(), S.w, S.w_rows_high)));
}

// w: g^T M, one entry for each column. After this call the caller may add claims of its own to the
// group LinFirst::own; then it seals the chain.
inline LinFirst lin_first_prove_on_shape(ProverContext &P, const LinShape &S, LinClaim claim, std::vector<E> w) {
  const char *fault = lin_fault(P.st.codes(), S);
  BLK_REQUIRE(!fault, fault ? fault : "");
  BLK_REQUIRE(w.size() == (size_t(1) << S.k_col()), "LIN: W has one entry for each column");
  std::vector<OracleDraft> drafts;
  drafts.push_back(lin_draft(P, S.w, S.w_rows_high, S.w_leaf_class, w));
  if (S.own_split_needed()) drafts.push_back(lin_draft(P, S.own_split, false, S.own_split_leaf_class, {}));
  LinFirst out;
  if (S.masks_ahead) {
    const FoldShape F = lin_fold_shape(S.test_own, S.w_leaf_class);
    out.mask = fold_mask_make(F, P);
    drafts.push_back(fold_mask_draft(P.st.codes(), F, out.mask));
  }
  lin_first_rows(P.st, S, std::move(claim), lin_masks_as_drawn(P, send_prove(P, std::move(drafts))), out);
  return out;
}

inline bool lin_first_verify_on_shape(VerifierContext &V, const LinShape &S, LinClaim claim, LinFirst &out) {
  const char *fault = lin_fault(V.st.codes(), S);
  BLK_REQUIRE(!fault, fault ? fault : "");  // a bad shape is the caller's fault, not a proof's
  std::vector<OracleId> ids;
  if (!send_verify(V, lin_message_w(S), ids)) return false;
  lin_first_rows(V.st, S, std::move(claim), ids, out);
  if (V.notes) {  // values for comparing a second verifier block by block; no check reads them
    const VectorLay lay = vector_lay(V.st.codes(), S.w, S.w_rows_high);
    V.note("LIN: (Eval)'s target", out.claim.target);
    V.note("LIN: (Eval) on W<1>", lin_eval_row(S, out.claim, out.w, lay).coefficient(out.w, Part::MAIN, lay.index(1)));
  }
  return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Shared part: once for a batch. No claim is among the parameters.
// ---------------------------------------------------------------------------------------------------------------------

// Messages, on the batch's shared chain:
//   the verifier draws psi* = (psi_lo, gamma): k_lo + v elements of E, the table having 2^v rows;
//   FOLDEQ (blocks/iop/foldeq.h) at gamma: SEND of G_u, which holds u, the table's rows combined
//   with the weights eq(gamma, .); the prover opens columns of the table at drawn positions;
//   SEND of Omega, one entry for each row of M: Omega[y] is the inner product of eq(psi_lo, .) with
//   the 2^k_lo entries of u that belong to y;
//   the verifier draws zeta, k_row() elements, for (Fold);
//   proximity tests on G_u's group and on Omega's.
// Errors, beside those of SEND, FOLDEQ, SPLIT and the tests: psi*: N (k_lo + v) / |E| for N
// requests, since both sides of (Dot) are multilinear in psi* and W_c was fixed before; zeta:
// k_row() / |E|. Both by counting zeros. Formulas: plan_lin_shared(),
// blocks/numbers/blocks.h.
// Hiding, in all three parts: outside openings and the proximity tests the prover sends sample
// answers (each under its own pad), shares of splits (each under its own mask entry), and no other
// scalar. Omega is opened by 1 + N tests: its random tail must cover them all, or open_oracles()
// (blocks/iop/context.h) ends the process.

// What the shared part leaves for its tests and for the requests' second parts; `psi` is psi*.
struct LinShared {
  std::vector<E> psi, zeta;
  FoldedRow u;
  OracleId omega = NONE_ID, split = NONE_ID;
  GroupId group_u = NONE_ID, group_omega = NONE_ID;  // equal if G_u and Omega have one code
  FoldMask mask;  // of the first fold of the test on G_u's group, if sent with G_u
  LinPublic bound;  // the shared chain's inputs; set by `lin_shared_prove`, `lin_shared_verify`
};

template <class Channel> inline std::vector<E> lin_draw(Channel &ch, const char *round, unsigned count) {
  auto from = ch.close_algebraic(round);
  std::vector<E> x(count);
  for (E &e : x) e = from.draw();
  return x;
}
inline void lin_shared_groups(Statement &st, const LinShape &S, LinShared &out) {
  out.group_u = st.new_group("LIN: the folded row", S.fold.u.code, S.shared_mask_code);
  out.group_omega = S.shared_split_needed() ? st.new_group("LIN: Omega", S.omega.code, S.shared_mask_code) : out.group_u;
}
// What shares a leaf is read at exactly the same lists: an opening that leaves out an oracle
// message of a tree ends the process (TreeBook::whole_trees(), context.h). Omega's leaves are
// opened at 1 + N lists: the shared test's and every request's own. So Omega has its tree to
// itself. Above all, the fresh mask of the first fold of the shared test on Omega stays out: it is
// read at the shared list only, and in Omega's leaves every request's proof would show further
// symbols of it, a few requests together the whole mask. It is sent with G_u or as the fold's own
// first message. Both sides call this as soon as Omega is sent, and again in lin_adopt().
inline bool lin_alone_in_its_tree(const TreeBook &book, OracleId o) { return book.tree(book.place(o).tree).parts.size() == 1; }

inline void lin_shared_join(Statement &st, const TreeBook &book, const LinShape &S, const std::vector<OracleId> &ids, LinShared &out) {
  BLK_REQUIRE(lin_alone_in_its_tree(book, ids[0]), "LIN: something rides in Omega's leaves, which every request's test opens");
  st.join(out.group_omega, out.omega = ids[0]);
  if (!S.shared_split_needed()) return;
  st.join(out.group_omega, out.split = ids[1]);
  st.join(out.group_u, out.split);
}

// A pass-key (an object that only lin_begin() can construct): the batch's columns have been counted
// against what the table's random tails cover (IndexHolder::begin(), foldeq.h). The prover's shared
// part requires one. Call lin_begin() before any request's first part, whose chain already names
// the table in its first query. Not prevented: a caller that holds a LinBegun can copy it (the
// implicit copy constructor is public).
class LinBegun {
  friend bool lin_begin(IndexHolder &, const LinShape &, std::unique_ptr<LinBegun> &);
  LinBegun() {}  // written out: with "= default" the class would be an aggregate under C++17, and LinBegun{} would be open to all
};
// False if the random tails do not cover a further batch: the batch then needs another table. After
// true the columns stay counted, even if the batch is broken off (IndexHolder::begin()'s contract).
inline bool lin_begin(IndexHolder &index, const LinShape &S, std::unique_ptr<LinBegun> &out) {
  if (!index.begin(S.fold.columns)) return false;
  out.reset(new LinBegun());
  return true;
}

// Precondition: the chain was bound by lin_bind_shared().
// places: the number of places (leaves) of the binding tree: how many requests can share the point
//   drawn here. The plan (the list of rounds with their error bounds) multiplies the point's error
//   by its own number of requests, and require_sharing() (blocks/chain/rounds.h) ends the process
//   unless the two are equal. The functions on a LinPublic take it from no argument:
//   lin_bind_shared() reads it from the LinBindingRoot. No type protects it further: a caller could
//   overwrite the public field LinPublic::places after the binding, and only a scan of the sources
//   (its list of names is not part of this repository), not the compiler, keeps
//   made_up_without_a_tree() to tests.
// test: the committer's one-time test or a later one, a batch's (TableTest, blocks/numbers/plan.h).
//   Both run this code, but a later test's error bound for a drawn column is valid only for a table
//   that has passed the one-time test. So the caller states which test it runs, and require_table()
//   compares that with the test for which the plan's errors were computed, where the point and the
//   columns are drawn. No later test may be the first.
inline void lin_shared_prove_on_shape(ProverContext &P, const LinShape &S, u64 places, TableTest test, IndexHolder &index, const IndexRecord &rec, std::unique_ptr<LinBegun> begun,
                             LinShared &out) {
  const char *fault = lin_fault(P.st.codes(), S);
  BLK_REQUIRE(!fault, fault ? fault : "");
  BLK_REQUIRE(test != TableTest::LINK, "LIN: a test on the table is the one-time test or a later one");
  BLK_REQUIRE(begun != nullptr, "LIN: the batch has not been counted against the table's copy");
  out.psi = lin_draw(P.ch, R_LIN_POINT, S.k_col());
  P.ch.require_table(priced_table(S.fold.index, test)), P.ch.require_sharing(places);
  lin_shared_groups(P.st, S, out);
  BlockSums blocks{out.psi.data(), S.k_lo, S.matrix_rows, {}};
  std::vector<OracleDraft> with;
  std::vector<OracleId> with_ids;
  if (S.masks_ahead) {
    const FoldShape F = lin_fold_shape(S.test_u, S.fold.leaf_class);
    out.mask = fold_mask_make(F, P);
    with.push_back(fold_mask_draft(P.st.codes(), F, out.mask));
  }
  out.u = foldeq_prove(P, index, rec, S.fold, test, out.psi.data() + S.k_lo, out.group_u, &blocks, std::move(with), &with_ids);
  if (S.masks_ahead) out.mask.sent_as(with_ids[0]);

  std::vector<E> omega = std::move(blocks.sums);
  BLK_REQUIRE(omega.size() == S.matrix_rows, "LIN: the keeper has not summed the blocks");
  std::vector<OracleDraft> drafts;
  drafts.push_back(lin_draft(P, S.omega, S.omega_rows_high, S.omega_leaf_class, omega));
  if (S.shared_split_needed()) drafts.push_back(lin_draft(P, S.shared_split, false, S.shared_split_leaf_class, {}));
  lin_shared_join(P.st, P.book, S, lin_masks_as_drawn(P, send_prove(P, std::move(drafts))), out);

  out.zeta = lin_draw(P.ch, R_LIN_ZETA, S.k_row());
  Row row = lin_fold_row(S, out.psi, out.zeta, out.u, out.omega, vector_lay(P.st.codes(), S.omega, S.omega_rows_high));
  if (S.shared_split_needed())
    split_prove(P, row, out.group_omega, out.group_u, out.split);
  else
    P.st.add_row(out.group_u, std::move(row));
  P.behind[out.omega].held = true;  // the requests' tests read it too, from contexts of their own
}

// table: see lin_shared_verify().
inline bool lin_shared_verify_on_shape(VerifierContext &V, const LinShape &S, u64 places, TableTest test, const TableRead &table, LinShared &out) {
  const char *fault = lin_fault(V.st.codes(), S);
  BLK_REQUIRE(!fault, fault ? fault : "");
  BLK_REQUIRE(test != TableTest::LINK, "LIN: a test on the table is the one-time test or a later one");
  out.psi = lin_draw(V.ch, R_LIN_POINT, S.k_col());
  V.ch.require_table(priced_table(S.fold.index, test));  // see `lin_shared_prove_on_shape`
  V.ch.require_sharing(places);                          // see there
  lin_shared_groups(V.st, S, out);
  std::vector<OracleNotice> with = lin_message_u(S);
  with.erase(with.begin());
  std::vector<OracleId> with_ids;
  if (!foldeq_verify(V, table, S.fold, test, out.psi.data() + S.k_lo, out.group_u, out.u, with, &with_ids)) return false;
  if (S.masks_ahead) out.mask.sent_as(with_ids[0]);

  std::vector<OracleId> ids;
  if (!send_verify(V, lin_message_omega(S), ids)) return false;
  lin_shared_join(V.st, V.book, S, ids, out);

  out.zeta = lin_draw(V.ch, R_LIN_ZETA, S.k_row());
  Row row = lin_fold_row(S, out.psi, out.zeta, out.u, out.omega, vector_lay(V.st.codes(), S.omega, S.omega_rows_high));
  if (V.notes) {
    V.note("LIN: (Fold) on Omega<0>", row.coefficient(out.omega, Part::MAIN, vector_lay(V.st.codes(), S.omega, S.omega_rows_high).index(0)));
    V.note("LIN: (Fold) on G_u<1>", row.coefficient(out.u.oracle, Part::MAIN, out.u.lay.index(1)));
  }
  if (S.shared_split_needed()) return split_verify(V, row, out.group_omega, out.group_u, out.split);
  V.st.add_row(out.group_u, std::move(row));
  return V.ch.ok();
}

// The shared part's proximity tests. After them the caller seals the chain.
inline void lin_shared_tests_prove_on_shape(ProverContext &P, const LinShape &S, LinShared &x) {
  std::vector<OracleId> done{x.u.oracle};
  if (x.mask.sent()) done.push_back(x.mask.id());
  iopp_prove(P, x.group_u, S.test_u, S.choice, std::move(x.mask));
  release_oracles(P, done);  // the folded row is the largest object that a batch creates
  if (!S.shared_split_needed()) return;
  iopp_prove(P, x.group_omega, S.test_omega, S.choice);
  release_oracles(P, {x.split});
}
// Releases Omega; call it after the last request's second part and its tests.
inline void lin_release(ProverContext &shared_P, const LinShared &x) {
  shared_P.behind[x.omega].held = false;
  release_oracles(shared_P, {x.omega});
}
inline bool lin_shared_tests_verify_on_shape(VerifierContext &V, const LinShape &S, LinShared &x) {
  if (!iopp_verify(V, x.group_u, S.test_u, S.choice, std::move(x.mask))) return false;
  return !S.shared_split_needed() || iopp_verify(V, x.group_omega, S.test_omega, S.choice);
}

// ---------------------------------------------------------------------------------------------------------------------
// Second part: a request by itself, on a chain that follows its first chain and the shared chain
// ---------------------------------------------------------------------------------------------------------------------

// No oracle message is sent before the tests: (Dot), then a proximity test on the request's group:
// Omega, W_c, the oracle messages of the claim's masks. After a split there are two groups and two
// tests.

// Enters an oracle message that was sent and sampled on another chain of the batch into this
// context: same shape, root, tree and sample claims (a sample's form and its answer). It must have
// a tree to itself, or the process ends. What makes this sound is not checked here: the other
// chain's seal must be in this chain's first query (lin_second_prove() and lin_second_verify()
// check that).
inline OracleId lin_adopt(Statement &st, TreeBook &book, const Statement &from_st, const TreeBook &from_book, OracleId theirs, unsigned leaf_class) {
  const TreeBook::Tree &tree = from_book.tree(from_book.place(theirs).tree);
  BLK_REQUIRE(lin_alone_in_its_tree(from_book, theirs), "LIN: an oracle message that several chains read has a tree of its own");
  std::vector<OracleId> ids;
  const u32 mine = book.enter(st, std::vector<OracleNotice>{OracleNotice{from_st.oracle(theirs).shape, leaf_class}}, ids);
  book.adopt(mine, tree);  // keeps its sender's tree number and run's value; uses up a number here
  st.sampled_elsewhere(ids[0], from_st, theirs);
  return ids[0];
}

// What the second part leaves for its tests.
struct LinSecond {
  OracleId omega = NONE_ID;
  GroupId around_omega = NONE_ID;  // LinFirst::own, or a second group if W and Omega differ in code
};

inline Row lin_second_rows(Statement &st, const LinShape &S, const LinFirst &first, const std::vector<Tensor> &weights, const LinShared &shared, LinSecond &out) {
  out.around_omega = S.own_split_needed() ? st.new_group("LIN: a request's rows on Omega", S.omega.code, S.own_mask_code) : first.own;
  st.join(out.around_omega, out.omega);
  if (S.own_split_needed()) st.join(out.around_omega, first.split);
  return lin_dot_row(S, weights, shared.psi, out.omega, vector_lay(st.codes(), S.omega, S.omega_rows_high), first.w,
                     vector_lay(st.codes(), S.w, S.w_rows_high));
}

inline LinSecond lin_second_prove_on_shape(ProverContext &P, const LinShape &S, const LinFirst &first, const ProverContext &shared_P,
                                  const LinShared &shared) {
  LinSecond out;
  out.omega = lin_adopt(P.st, P.book, shared_P.st, shared_P.book, shared.omega, S.omega_leaf_class);
  P.behind.push_back(shared_P.behind[shared.omega]);
  // The draws' numbers are those of shared_P's generator, which need not be P.coins.
  P.behind.back().mask_draw = P.behind.back().tail_draw = Coins::NO_DRAW;
  P.commit.push_back(shared_P.commit[shared_P.book.place(shared.omega).tree]);
  Row row = lin_second_rows(P.st, S, first, lin_weights_of_the_prover(S, first.claim.g), shared, out);
  if (S.own_split_needed())
    split_prove(P, row, out.around_omega, first.own, first.split);
  else
    P.st.add_row(first.own, std::move(row));
  return out;
}
inline bool lin_second_verify_on_shape(VerifierContext &V, const LinShape &S, const LinFirst &first, const VerifierContext &shared_V,
                              const LinShared &shared, LinSecond &out) {
  out.omega = lin_adopt(V.st, V.book, shared_V.st, shared_V.book, shared.omega, S.omega_leaf_class);
  Row row = lin_second_rows(V.st, S, first, first.claim.g, shared, out);
  if (V.notes) {
    V.note("LIN: (Dot) on Omega<0>", row.coefficient(out.omega, Part::MAIN, vector_lay(V.st.codes(), S.omega, S.omega_rows_high).index(0)));
    V.note("LIN: (Dot) on W<1>", row.coefficient(first.w, Part::MAIN, vector_lay(V.st.codes(), S.w, S.w_rows_high).index(1)));
  }
  if (S.own_split_needed()) return split_verify(V, row, out.around_omega, first.own, first.split);
  V.st.add_row(first.own, std::move(row));
  return V.ch.ok();
}

// The positions at which Omega has been opened are counted at its tree, in one number shared by all
// TreeBooks that read the tree (TreeBook::Tree::positions, context.h; lin_adopt() passes it on). So
// the count of Omega's opened positions against its random tail, in open_oracles(), sees all 1 + N
// tests, in whatever order they run.
inline void lin_second_tests_prove_on_shape(ProverContext &P, const LinShape &S, LinFirst &first, const LinSecond &x) {
  std::vector<OracleId> done{first.w};
  if (first.mask.sent()) done.push_back(first.mask.id());
  iopp_prove(P, first.own, S.test_own, S.choice, std::move(first.mask));
  if (S.own_split_needed()) iopp_prove(P, x.around_omega, S.test_own_omega, S.choice);
  release_oracles(P, done);  // the caller releases the oracle messages of the claim's masks
  if (S.own_split_needed()) release_oracles(P, {first.split});
}
inline bool lin_second_tests_verify_on_shape(VerifierContext &V, const LinShape &S, LinFirst &first, const LinSecond &x) {
  if (!iopp_verify(V, first.own, S.test_own, S.choice, std::move(first.mask))) return false;
  return !S.own_split_needed() || iopp_verify(V, x.around_omega, S.test_own_omega, S.choice);
}

// ---------------------------------------------------------------------------------------------------------------------
// How the chains of a batch are tied together
// ---------------------------------------------------------------------------------------------------------------------

// One view for each request: a request's verifier gets no other request's first or second part. A
// view is the request's first part, its path in the binding tree, the shared part, the request's
// second part.
//   - The shared part's functions have no parameter through which a request's data could enter.
//   - A claim of request c on Omega goes into that request's own test, never into the shared one.
//   - A request's first chain ends in a seal. The seals are the leaves of a salted binding tree
//     whose root is in the shared chain's first query: psi* depends on every claim, W_c and sample
//     answer. A path holds the request's place and salt and sibling digests, no other seal.
//   - The shared chain's seal and the request's own seal are both in the first query of the
//     request's second chain.

// The functions below that take a LinPublic read codes, shape and table from the chain's own first
// query, so no part of LIN runs under another shape than its chain is bound to. The _on_shape
// functions above are their bodies; only a test that bypasses the binding on purpose calls them
// directly.

/* A binding tree's root with the number of its places. Two functions make one: `LinBinding::root`,
   for a tree that it built, and `lin_binding_root`, after reading a path up to the root. The
   latter's caller must pass the shape to which the request's first chain is bound: this is not
   checked here. The root fixes the depth: a leaf's hash takes in 2^depth + place (`tree_index`,
   blocks/tree/tree.h), and leaves and inner vertices are hashed under two domains, so a path of
   another depth does not lead to the same root (tested in blocks/iop/tests/lin_test.cpp). So the
   number of requests that can share a point reaches the shared chain from a bound shape, not as a
   number that a caller states. */
struct LinBindingShape;
class LinBindingRoot {
 public:
  LinBindingRoot() = default;  // none yet: 0 places, which fits no plan (`Rounds::require_sharing`)
  const Digest &digest() const { return digest_; }
  u64 places() const { return places_; }
  // The one exception, for a test or benchmark that runs the shared part alone, without requests or
  // a tree: any digest, with a stated number of places.
  static LinBindingRoot made_up_without_a_tree(const Digest &digest, u64 places) { return LinBindingRoot(digest, places); }

 private:
  friend class LinBinding;
  friend bool lin_binding_root(Reader &, const LinBindingShape &, const Digest &, const Digest &, LinBindingRoot &);
  LinBindingRoot(const Digest &digest, u64 places) : digest_(digest), places_(places) {}
  Digest digest_{};
  u64 places_ = 0;
};
// The shared chain: Binder(its label, plan). All views must derive the same chain, so it has
// no random bytes of its own: the binding tree's root takes their place. The root is fresh in every
// batch (salted leaves, uniform digests at free places), and every view leads to it.
inline LinPublic lin_bind_shared(Binder &b, const Codes &codes, const LinShape &S, const IndexName &table, const LinBindingRoot &binding_root) {
  LinPublic x = lin_bind(b, codes, S, table);
  b.random_bytes_are(b.statement("the binding tree's root", binding_root.digest()));
  x.places = binding_root.places();
  return x;
}
// A request's second chain: Binder(its label, plan). It is bound to all that its first chain
// and the shared chain are bound to.
inline void lin_bind_second(Binder &b, const Sealed &own, const Sealed &shared) {
  b.follows(own);
  b.follows(shared);
}

inline LinFirst lin_first_prove(ProverContext &P, const LinPublic &x, LinClaim claim, std::vector<E> w) {
  return lin_first_prove_on_shape(P, P.ch.get(x.shape), std::move(claim), std::move(w));
}
// Contract (the plan's origin; what is not checked): see run_verifier() in blocks/chain/closed.h.
[[nodiscard]] inline bool lin_first_verify(VerifierContext &V, const LinPublic &x, LinClaim claim, LinFirst &out) {
  return lin_first_verify_on_shape(V, V.ch.get(x.shape), std::move(claim), out);
}
// index: the table holder ("the keeper" in messages). It must hold the table that the chain's first
// query names, made under the chain's configuration digest (a hash of the public files), or the
// process ends.
inline void lin_shared_prove(ProverContext &P, const LinPublic &x, TableTest test, IndexHolder &index, std::unique_ptr<LinBegun> begun, LinShared &out) {
  const IndexName &name = P.ch.get(x.table);
  BLK_REQUIRE(index.commitment() == name.commitment, "LIN: the keeper holds another table than the chain's first query names");
  BLK_REQUIRE(index.configuration() == P.ch.configuration(), "LIN: the table's tree starts from another configuration's digest than the chain names");
  out.bound = x;
  lin_shared_prove_on_shape(P, P.ch.get(x.shape), x.places, test, index, name.record, std::move(begun), out);
}
// table: reads and checks the named table's columns (TableRead, foldeq.h). The one-time test takes any IndexReader. A
// later test needs a TestedTable made under the plan's Setting (target in bits, proof-of-work bits and so on): given a
// bare IndexReader or another Setting, foldeq_columns_verify() ends the process. No test need have
// passed in this process: the library makes a stand-in in check_proof_if_tested() and
// check_entry_if_tested() (for cp_verify); no caller can unless it defines
// BLK_TESTED_TABLE_FOR_A_TEST (tests, two toy tools).
// Contract: see run_verifier() in blocks/chain/closed.h.
[[nodiscard]] inline bool lin_shared_verify(VerifierContext &V, const LinPublic &x, TableTest test, const TableRead &table, LinShared &out) {
  const IndexReader &key = table.reader();
  const IndexName &name = V.ch.get(x.table);
  BLK_REQUIRE(key.commitment() == name.commitment && record_bytes(key.record()) == record_bytes(name.record),
              "LIN: the key is another table's than the chain's first query names");
  BLK_REQUIRE(key.configuration() == V.ch.configuration(), "LIN: the table's tree starts from another configuration's digest than the chain names");
  out.bound = x;
  return lin_shared_verify_on_shape(V, V.ch.get(x.shape), x.places, test, table, out);
}
inline void lin_shared_tests_prove(ProverContext &P, const LinPublic &x, LinShared &s) { lin_shared_tests_prove_on_shape(P, P.ch.get(x.shape), s); }
// Contract: see run_verifier() in blocks/chain/closed.h.
[[nodiscard]] inline bool lin_shared_tests_verify(VerifierContext &V, const LinPublic &x, LinShared &s) { return lin_shared_tests_verify_on_shape(V, V.ch.get(x.shape), s); }
// Codes, shape and table are in the first query of a request's first chain and of the shared chain:
// twice, since each chain must be bound to them by itself. The second chain follows both seals and
// can read both; here they are compared.
template <class Channel> inline bool lin_same_public(const Channel &ch, const LinPublic &a, const LinPublic &b) {
  Writer first, second;
  public_bytes(first, ch.get(a.codes)), public_bytes(first, ch.get(a.shape)), public_bytes(first, ch.get(a.table));
  public_bytes(second, ch.get(b.codes)), public_bytes(second, ch.get(b.shape)), public_bytes(second, ch.get(b.table));
  return first.bytes() == second.bytes();
}
// x: as the request's first chain is bound to it. The shared point, Omega's root and its sample
// claim are taken from the shared chain's context: this chain's first query must contain that
// chain's seal.
inline LinSecond lin_second_prove(ProverContext &P, const LinPublic &x, const LinFirst &first, const ProverContext &shared_P, const LinShared &shared) {
  BLK_REQUIRE(P.ch.stands_on(shared_P.ch), "LIN: the second chain's first query does not hold the seal of the shared chain that is handed in");
  BLK_REQUIRE(lin_same_public(P.ch, x, shared.bound), "LIN: the request's first chain and the shared chain hold different codes, shapes or tables");
  return lin_second_prove_on_shape(P, P.ch.get(x.shape), first, shared_P, shared);
}
// Contract: see run_verifier() in blocks/chain/closed.h.
[[nodiscard]] inline bool lin_second_verify(VerifierContext &V, const LinPublic &x, const LinFirst &first, const VerifierContext &shared_V, const LinShared &shared,
                                            LinSecond &out) {
  BLK_REQUIRE(V.ch.stands_on(shared_V.ch), "LIN: the second chain's first query does not hold the seal of the shared chain that is handed in");
  BLK_REQUIRE(lin_same_public(V.ch, x, shared.bound), "LIN: the request's first chain and the shared chain hold different codes, shapes or tables");
  return lin_second_verify_on_shape(V, V.ch.get(x.shape), first, shared_V, shared, out);
}
inline void lin_second_tests_prove(ProverContext &P, const LinPublic &x, LinFirst &first, const LinSecond &second) {
  lin_second_tests_prove_on_shape(P, P.ch.get(x.shape), first, second);
}
// Contract: see run_verifier() in blocks/chain/closed.h.
[[nodiscard]] inline bool lin_second_tests_verify(VerifierContext &V, const LinPublic &x, LinFirst &first, const LinSecond &second) {
  return lin_second_tests_verify_on_shape(V, V.ch.get(x.shape), first, second);
}

// Moves a context to a later chain, whose Binder has followed the earlier chain's seal (checked in
// context.h).
inline ProverContext lin_go_on(ProverContext &&old, ProverRounds &ch) { return ProverContext(std::move(old), ch); }
inline VerifierContext lin_go_on(VerifierContext &&old, VerifierRounds &ch) { return VerifierContext(std::move(old), ch); }

// The binding tree: 2^log_leaves leaves. A request's leaf is the hash of its seal and a salt. A
// leaf that no request takes is a uniform digest, so that a path shows neither how many requests
// share the batch nor anything of their seals. The caller chooses each request's place: to hide the
// order of arrival it must assign places at random.
// Source for shared challenges from the root of a Merkle tree over the verifiers' transcripts: Zhang, Xie, Hoang,
// Shi, Zhang, "Polynomial Commitment with a One-to-Many Prover and Applications", USENIX Security 2022, sections
// 1.2.1 ("Idea 2"), 2 ("Merkle Tree") and 3.2. The paper's own protocol uses the tree only where the verifiers'
// transcripts differ: there Merkle trees are inserted "in every round of Step 4 (i.e., the output layer)"; here one
// tree is built, over the seals in which the requests' first chains end.
// Source for the salted tree's form (index, salt, path) and for privacy within a batch as a defined notion:
// Aguilar-Melchor, Albrecht, Bailleux, Bindel, Howe, Hülsing, Joseph, Manzano, "Batch Signatures, Revisited",
// Cryptology ePrint Archive 2023/492, sections 3 and 4 (Algorithms 1 and 2). There the root is signed; here it starts
// the shared chain.
// That paper calls its scheme "essentially a refined version of" D. Benjamin, "Batch Signing for TLS", Internet-Draft
// draft-ietf-tls-batch-signing-00, 2020.
struct LinBindingShape {
  unsigned log_leaves = 0;
  size_t salt_bytes = 40;
};
inline void public_bytes(Writer &w, const LinBindingShape &s) { public_bytes(w, s.log_leaves), public_bytes(w, s.salt_bytes); }
inline const char *public_fault(const LinBindingShape &s) {
  if (s.log_leaves > 48) return "the binding tree is too deep";  // as every tree (tree/tree.h)
  if (!salt_length_fits(s.salt_bytes)) return "the binding tree's salts have no admitted length";
  return nullptr;
}
// A leaf is bound to its place; a vertex hashes its two children only, under the binding tree's own
// domain. configuration: the configuration digest, from which the tree's hashes start, since no
// run's value exists when the tree is made. Callers take it from their channel
// (ch.configuration()).
inline Digest lin_binding_vertex(const Digest &configuration, const Digest &left, const Digest &right) {
  return vertex(Tag(Domain::BINDING_NODE, 0, configuration), left, right);
}
inline Digest lin_binding_leaf(const LinBindingShape &s, const Digest &configuration, u64 place, const Digest &seal, const u8 *salt) {
  return leaf(Tag(Domain::LEAF_BINDING, 0, configuration, 4, s.salt_bytes), tree_index(s.log_leaves, place), seal, salt, s.salt_bytes);  // a leaf of one digest: four words
}

// The prover's binding tree. Order: put() once for each request, then root() or write_path(). These
// two close the tree: a later put() ends the process, as does a second put() at one place.
class LinBinding {
 public:
  LinBinding(const LinBindingShape &s, const Digest &configuration, Coins &coins)
      : s_(s), configuration_(configuration), level_(s.log_leaves + 1), salt_((size_t(1) << s.log_leaves) * s.salt_bytes) {
    coins.bytes(salt_.data(), salt_.size());
    level_[s.log_leaves].resize(size_t(1) << s.log_leaves);
    for (Digest &d : level_[s.log_leaves]) {
      u8 b[DIGEST_BYTES];
      coins.bytes(b, DIGEST_BYTES);
      digest_from_bytes(b, d);
    }
    taken_.assign(size_t(1) << s.log_leaves, false);
  }
  void put(u64 place, const Digest &seal) {
    BLK_REQUIRE(!closed_ && place < taken_.size() && !taken_[place], "binding tree: the place is taken, or the tree is closed");
    taken_[place] = true;
    level_[s_.log_leaves][place] = lin_binding_leaf(s_, configuration_, place, seal, &salt_[place * s_.salt_bytes]);
  }
  LinBindingRoot root() {
    if (!closed_)
      for (unsigned d = s_.log_leaves; d-- > 0;) {
        level_[d].resize(size_t(1) << d);
        for (size_t i = 0; i < level_[d].size(); i++) level_[d][i] = lin_binding_vertex(configuration_, level_[d + 1][2 * i], level_[d + 1][2 * i + 1]);
      }
    closed_ = true;
    return LinBindingRoot(level_[0][0], u64(1) << s_.log_leaves);
  }
  // A request's path: the place (u64), the salt, the siblings from the leaf upward.
  void write_path(Writer &w, u64 place) {
    root();
    BLK_REQUIRE(place < taken_.size() && taken_[place], "binding tree: no request stands there");
    w.put_u64(place);
    w.put_bytes(&salt_[place * s_.salt_bytes], s_.salt_bytes);
    for (unsigned d = s_.log_leaves; d > 0; d--) w.put(level_[d][(place >> (s_.log_leaves - d)) ^ 1]);
  }

 private:
  LinBindingShape s_;
  Digest configuration_;
  std::vector<std::vector<Digest>> level_;
  std::vector<u8> salt_;
  std::vector<bool> taken_;
  bool closed_ = false;
};

// The root that a request's seal and its path lead to. False if the place is out of range or bytes
// are missing. The root is compared with nothing here: it goes into the shared chain's first query
// (lin_bind_shared()), so a wrong path leads to another shared chain.
inline bool lin_binding_root(Reader &r, const LinBindingShape &s, const Digest &configuration, const Digest &seal, LinBindingRoot &out) {
  u64 place = 0;
  if (!r.get_u64(place) || (place >> s.log_leaves) != 0) return r.refuse(), false;
  const u8 *salt = r.take(s.salt_bytes);
  if (!salt) return false;
  Digest root = lin_binding_leaf(s, configuration, place, seal, salt);
  for (unsigned d = s.log_leaves; d > 0; d--, place >>= 1) {
    Digest sibling;
    if (!r.get(sibling)) return false;
    root = (place & 1) ? lin_binding_vertex(configuration, sibling, root) : lin_binding_vertex(configuration, root, sibling);
  }
  out = LinBindingRoot(root, u64(1) << s.log_leaves);
  return true;
}

}  // namespace blk
