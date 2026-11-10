// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// BAT batches J linear claims into one: the sum of x^i times claim number i, counted from 0, for a
// challenge x. A linear claim (`Row` in identifiers, "row" in error texts) states that a linear
// form of the messages encoded in some oracle messages (codeword rows sent as Merkle roots) has a
// public value (`target`). The prover sends nothing; the verifier sends x, one element of the
// extension field `E`. BAT has no check. Nothing is sent, so hiding needs nothing.
//
// Every claim, its value included, must be fixed before x is drawn (one exception: the first
// combine() below). Before it draws x, the channel, which derives challenges by hashing, absorbs
// what the prover has sent. The code that makes a claim must ensure, unchecked here, that its value
// depends only on the statement, on what was sent and on earlier challenges.
//
// Error: if some claim fails for a given tuple of messages, so does the sum, except for at most
// J - 1 values of x (a non-zero polynomial of degree below J). The bounds at run() and sigma() are
// elementary; the batched form is Definition 5.6 of Chiesa, Fenzi, Weissenberg, "Zero-knowledge
// IOPPs for constrained interleaved codes" (ePrint 2026/391, version of February 25, 2026), and the
// step is part of the proof of their code switch.
//
// The batched row follows the definition of the batched linear form of
// the paper that this protocol builds on
// (see THIRD_PARTY.md).
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "field/field.h"
#include "iop/relation.h"
#include "numbers/blocks.h"

namespace blk {

// Two forms: run() is the first step of a proximity test (blocks/iop/iopp.h), which is to show that
// the rows of a group (the oracle messages that one test reads together) are close to codewords;
// sigma() batches claims well before any test.
// Tests, in blocks/iop/tests/: `send_bat_split_test` (both forms, prover against verifier; the
// batched claims are evaluated on the prover's messages; begin_run()'s output is compared with the
// sample claims), `send_orders_test` (calls that break the rules of order must end the process) and
// `cs_test` (the first combine() against the same sum written out by hand).
class Bat {
 public:
  // Returns the sum of x^i times claim number i of `in`, coefficients and value alike, as one claim
  // on group g. For a caller that has drawn x itself. One such caller is, by design, the exception
  // to the rule at the top of this file: CS, the code switch inside a proximity test (cs_prove(),
  // cs_verify() in cs.h), gets x in one verifier message with its positions, after them, and only
  // then adds the claims that it makes from the opened positions. The Merkle roots sent before x
  // and the positions fix these claims.
  // combine() does not check when x was drawn or how many claims `in` has. Like run(), a caller
  // must call begin_run() before it draws x (CS: after) and pass the number of claims to
  // require_rows(). ReqProver::close() and ReqVerifier::close() in req.h do so.
  static GroupRow combine(const Statement &S, GroupId g, RunInput in, const E &x) {
    GroupRow out;
    out.group = g;
    E power = E::one();
    std::vector<Row> rows = in.take_rows();
    size_t tensors = 0;  // upper bound on the tensors that out.c receives: memory is reserved once
    for (const Row &r : rows) {
      tensors += r.terms.size();
      for (const MainForm &f : r.forms) tensors += f.c.tensors.size();
    }
    out.c.tensors.reserve(tensors);
    RowsAside aside;  // all claims' `runs` and `units`, summed and added to `out` once, at the end
    for (Row &r : rows) {
      add_to_group_row(S, in.layout, std::move(r), power, out, &aside);
      power *= x;
    }
    add_to_group_row(S, in.layout, std::move(aside.sums), E::one(), out);
    return out;
  }
  // The same sum for claims that stay claims on oracle messages.
  static Row combine(const std::vector<Row> &rows, const E &x, std::string name) {
    Row out;
    out.name = std::move(name);
    E power = E::one();
    for (const Row &r : rows) {
      out.add_scaled(r, power);
      power *= x;
    }
    return out;
  }

  // BAT, the first step of the proximity test on group g. Both sides make the same calls; Channel
  // is the prover's or the verifier's.
  // Before: all claims added to g (its open claims) and all sample claims of its oracle messages
  // (`RowKind::SAMPLE`: the answers to the out-of-domain samples by which SEND, blocks/iop/send.h,
  // is to pin one message for each oracle message). Each claim is about these pinned messages.
  // Statement::begin_run() in blocks/iop/relation.h numbers them: `PLAIN` claims as added; then
  // the sample claims, oracle message by oracle message as sent; then `GENERATOR` claims (a
  // codeword's symbol at an opened position) as added.
  // After: one claim on the whole group (`GroupRow`), of which the test shows only that some
  // codeword near the group's word meets it. The sample claims inside it tie that codeword to the
  // pinned messages, so callers do not supply them: begin_run() returns all of them with the rest.
  // In the controls of `send_two_codewords_test` and `split_two_masks_test`, a claim taken without
  // the sample claims holds for a second codeword near the word, one that disagrees with the
  // sample's answer. These two tests are not part of this repository.
  // begin_run() closes g: after it a new claim or oracle message for g, a second test on g, or
  // taking a mask entry of its oracle messages ends the process.
  // Error: L (J - 1) / |E|, computed by plan_bat() in blocks/numbers/blocks.h. The list factor L
  // bounds the number of codewords near the group's word, any of which the test could find. The
  // process ends if J is not the count given to plan_bat() for this round (require_rows()), unless
  // the proof is already rejected.
  template <class Channel>
  static GroupRow run(Channel &ch, Statement &S, GroupId g, GroupLayout *layout = nullptr) {
    RunInput in = S.begin_run(g);  // first: from here on nothing can be added to the group
    auto from = ch.close_algebraic(R_BAT);
    from.require_rows(in.rows().size());  // the round's printed error counts these rows
    if (layout) *layout = in.layout;
    return combine(S, g, std::move(in), from.draw());
  }
  // BAT-sigma batches claims on any sampled oracle messages into one such claim, well before any
  // proximity test (say, before SPLIT splits a claim on two groups). All are about pinned messages.
  // Error: (J - 1) / |E|, computed by plan_bat_sigma() in blocks/numbers/blocks.h. Only tests call
  // sigma(), through bat_sigma_prove() and bat_sigma_verify().
  template <class Channel>
  static Row sigma(Channel &ch, const Statement &S, const std::vector<Row> &rows, std::string name) {
    BLK_REQUIRE(!rows.empty(), "bat: no rows");
    for (const Row &r : rows) {
      BLK_REQUIRE(r.kind != RowKind::SAMPLE, "bat: a sample row stays with its oracle message until a proximity test takes it");
      S.require_sampled(r);  // the error has no list factor because the messages are pinned
    }
    auto from = ch.close_algebraic(R_BAT_SIGMA);
    from.require_rows(rows.size());
    return combine(rows, from.draw(), std::move(name));
  }
};

template <class ProverCtx>
inline GroupRow bat_prove(ProverCtx &pr, GroupId g, GroupLayout *layout = nullptr) {
  return Bat::run(pr.ch, pr.st, g, layout);
}
// False if the channel has rejected the proof's bytes, in this round (a missing or wrong proof of
// work) or earlier. The channel then gives 0 for x, and `out` means nothing.
template <class VerifierCtx>
[[nodiscard]] inline bool bat_verify(VerifierCtx &vf, GroupId g, GroupRow &out, GroupLayout *layout = nullptr) {
  out = Bat::run(vf.ch, vf.st, g, layout);
  return vf.ch.ok();
}
template <class ProverCtx>
inline Row bat_sigma_prove(ProverCtx &pr, const std::vector<Row> &rows, std::string name) {
  return Bat::sigma(pr.ch, pr.st, rows, std::move(name));
}
// False as for bat_verify().
template <class VerifierCtx>
[[nodiscard]] inline bool bat_sigma_verify(VerifierCtx &vf, const std::vector<Row> &rows, std::string name, Row &out) {
  out = Bat::sigma(vf.ch, vf.st, rows, std::move(name));
  return vf.ch.ok();
}

}  // namespace blk
