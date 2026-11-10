// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// SPLIT, one step of the proof system: replaces a linear claim (`Row` in identifiers) that reads
// oracle messages of two groups by one claim on each group. A proximity test reads one group, so a
// claim <a1, messages of G1> + <a2, messages of G2> = t can go into neither test. It becomes
//        on G1:  <a1, ..> + kappa = t'          on G2:  <a2, ..> - kappa = t - t',
// where kappa is a mask entry (a random element of the extension field E) and the prover sends
// t' (`share`). The verifier's message is empty, and SPLIT itself checks no equation.
//
// Soundness: one round, error 0, because the two new claims add up to the old one. This needs both
// tests to refer to the same kappa. The mask message (the oracle message that contains kappa) was
// sent earlier by SEND (send.h), which took an out-of-domain sample (a random linear form of it,
// chosen afterwards). So kappa is pinned before t' is sent, and both tests include the sample's
// claims. Without that, a word close to two codewords with different kappa would let a false claim
// pass as two true ones. Our own argument; it rests on SEND's error alone.
// Hiding: kappa is uniform and used for nothing else, so t' is uniform. Both tests open the mask
// message, so the random tail of its rows must be long enough for the positions opened by both.
#pragma once
#include <algorithm>
#include <utility>

#include "base.h"
#include "field/field.h"
#include "iop/relation.h"
#include "iop/row_value.h"
#include "numbers/round_names.h"

namespace blk {

// `mask_message`, a member of both g1 and g2, has mask rows only: a group has one base code, and
// g1's may differ from g2's; both have its mask code. Not checked: that the prover filled it with
// fresh random values, or that no other claim reads kappa (Row::add_unit() allows it).
// A split must come before the proximity test begins on any group that contains the mask message:
// the test closes the message's `MaskLedger` (blocks/iop/mask.h; gives out each mask entry once),
// and a later split, or one with no entry left, ends the process. So t' precedes both tests'
// challenges. The channel's `Plan` must list plan_split()'s round (blocks/numbers/blocks.h) next.
// Tests, in blocks/iop/tests/: send_bat_split_test.cpp (an honest run) and send_orders_test.cpp
// (order of calls). Not part of this repository: a test that sends a wrong t', and one that splits
// a false claim with a mask message close to two codewords: the test on g2 rejects. All four use
// short codes and no complete proximity test (substitute: blocks/iop/tests/send_world.h).
class Split {
 public:
  // value(r) returns the left side of claim r on the prover's messages; see row_value().
  template <class Channel, class Value>
  static void prove(Channel &ch, Statement &S, const Row &row, GroupId g1, GroupId g2, OracleId mask_message, Value &&value) {
    Halves h = halve(S, row, g1, g2, mask_message);
    E share = value(h.first);
    ch.send(share);
    ch.close_empty(R_SPLIT);
    finish(S, std::move(h), row.target, share, g1, g2);
  }
  // Returns false if t' cannot be read from the proof. The statement gets both claims all the same,
  // so a caller that continues trips no BLK_REQUIRE (blocks/base.h; ends the process) on the order
  // of calls.
  template <class Channel>
  [[nodiscard]] static bool verify(Channel &ch, Statement &S, const Row &row, GroupId g1, GroupId g2, OracleId mask_message) {
    Halves h = halve(S, row, g1, g2, mask_message);
    E share;
    const bool good = ch.recv(share);
    ch.close_empty(R_SPLIT);
    finish(S, std::move(h), row.target, share, g1, g2);
    return good;
  }

 private:
  struct Halves {
    Row first, second;
  };
  static bool member(const Statement &S, GroupId g, OracleId o) {
    const std::vector<OracleId> &m = S.group(g).members;
    return std::find(m.begin(), m.end(), o) != m.end();
  }
  // Sorts the claim's coefficients by group, then adds kappa to the first half and -kappa to the
  // second. The second half names the entry by its index: add_mask() uses up the `MaskSlot`.
  static Halves halve(Statement &S, const Row &row, GroupId g1, GroupId g2, OracleId mask_message) {
    BLK_REQUIRE(g1 != g2, "split: two different groups");
    BLK_REQUIRE(row.kind == RowKind::PLAIN, "split: only a plain row can be split, not a sample row or a generator row");
    BLK_REQUIRE(S.oracle(mask_message).shape.main_rows == 0, "split: the mask entry comes from a message with mask rows only");
    BLK_REQUIRE(member(S, g1, mask_message) && member(S, g2, mask_message), "split: the mask message is a member of both groups");
    Halves h;
    h.first.name = row.name + ", first share";
    h.second.name = row.name + ", second share";
    auto side = [&](OracleId o) -> Row & {
      const bool in1 = member(S, g1, o), in2 = member(S, g2, o);
      BLK_REQUIRE(in1 != in2, "split: each oracle message of the row is a member of exactly one of the two groups");
      return in1 ? h.first : h.second;
    };
    for (const Term &t : row.terms) side(t.oracle).terms.push_back(t);
    for (const Unit &u : row.units) side(u.oracle).units.push_back(u);
    for (const Run &r : row.runs) side(r.oracle).runs.push_back(r);
    for (const MainForm &f : row.forms) side(f.oracle).forms.push_back(f);
    MaskSlot kappa = S.masks(mask_message).take();
    h.second.add_unit(kappa.oracle(), Part::MASK, kappa.index(), -E::one());
    h.first.add_mask(std::move(kappa), E::one());
    return h;
  }
  static void finish(Statement &S, Halves h, const E &target, const E &share, GroupId g1, GroupId g2) {
    h.first.target = share;
    h.second.target = target - share;
    S.add_row(g1, std::move(h.first));
    S.add_row(g2, std::move(h.second));
  }
};

template <class ProverCtx>
inline void split_prove(ProverCtx &pr, const Row &row, GroupId g1, GroupId g2, OracleId mask_message) {
  Split::prove(pr.ch, pr.st, row, g1, g2, mask_message, [&pr](const Row &r) { return row_value(pr, r); });
}
template <class VerifierCtx>
[[nodiscard]] inline bool split_verify(VerifierCtx &vf, const Row &row, GroupId g1, GroupId g2, OracleId mask_message) {
  return Split::verify(vf.ch, vf.st, row, g1, g2, mask_message);
}

}  // namespace blk
