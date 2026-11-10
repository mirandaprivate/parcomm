// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Shared by tests of the code switch (CS), the proximity test and others: repeatable
// randomness (`coins` in identifiers), toy codes, stage sizes derived from the number of drawn
// positions (a second implementation of the formulas in solve_stages(), blocks/numbers/blocks.h),
// and a proximity test's plan (`Plan`, blocks/numbers/plan.h: the list of rounds, each with its
// error) written by hand, every round's error 0.
// The proximity test's steps (blocks/iop/iopp.h): BAT batches the claims into one, FOLD and CS
// reduce it stage by stage, BASE sends the last rows, masked.
#pragma once
#include <cstdio>
#include <cstring>

#include "engine/engine_cpu.h"
#include "iop/index_cpu.h"  // `toy_configuration`: a stand-in for the configuration digest
#include "iop/iopp.h"
#include "iop/row_value.h"

namespace blk {

[[maybe_unused]] static int failures = 0;
#define CHECK(cond, text) \
  do { \
    if (!(cond)) failures++, std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, text); \
  } while (0)

// Repeatable: test data and a toy prover's private randomness. Not for anything secret.
// SplitMix64 (Steele, Lea, Flood, OOPSLA 2014).
class TestCoins : public Coins {
 public:
  explicit TestCoins(u64 seed) : s_(seed) {}
  u64 word() {
    u64 z = (s_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  void fill(F *out, size_t n) override {
    for (size_t i = 0; i < n; i++)
      while (!F_from_word(word(), out[i])) {
      }
  }
  void bytes(u8 *out, size_t n) override {
    for (size_t i = 0; i < n; i++) out[i] = (u8)word();
  }

 private:
  u64 s_;
};

// 2^k main rows, 2^h mask rows.
struct Later {  // a stage after the first
  unsigned k, h;
};

inline u64 pow2_from(u64 x) { return u64(1) << ceil_log2(x); }

// ell message entries, a random tail of r entries, rate at most 1/4, length at least 2^least_log_n.
inline Code toy_code(size_t ell, size_t r, unsigned least_log_n) {
  Code c{0, ell, r};
  c.log_n = std::max(c.log_dim() + 2, least_log_n);
  return c;
}

// Appends `later` to st, whose last entry is the stage before them, and their codes to `codes`.
// `carried`: what the previous fold leaves: two mask rows' messages, three rows' random tails.
// `own_fold`: the mask rows' least length for the hiding polynomial of this stage's fold.
inline void toy_later_stages(Codes &codes, std::vector<IoppStage> &st, const std::vector<Later> &later, unsigned draws, unsigned answers) {
  const size_t r = tail_for_positions(draws);
  for (const Later &s : later) {
    const Code pc = codes.base[st.back().code], pzk = codes.mask[st.back().mask_code];
    const u64 carried = 2 * pzk.ell + pc.r + 2 * pzk.r;
    const u64 own_fold = s.k ? 2 * s.k + 1 - s.h : 1;
    const u64 ell_zk = std::max(pow2_from((carried + answers + (u64(1) << s.h) - 1) >> s.h), pow2_from(own_fold));
    codes.mask.push_back(toy_code(ell_zk, r, 0));
    codes.base.push_back(toy_code(pc.ell >> s.k, r, codes.mask.back().log_n));
    st.push_back(IoppStage{(CodeId)(codes.base.size() - 1), (CodeId)(codes.mask.size() - 1), s.k, s.h});
  }
}

// samples: answers to each out-of-domain sample.
inline Plan toy_empty_plan(unsigned samples = 1) {
  Setting set = setting_goldilocks();
  set.samples = samples;
  return Plan(set);
}

#ifdef BLK_PLANS_BY_HAND_FOR_A_TEST  // set before a test's first include (blocks/numbers/plan.h)
// SEND (blocks/iop/send.h) sends oracle messages and samples them out of domain. This plans the
// sample of `sampled`, once their roots are planned: the round of the seed, then `answers` answers
// for each. own_round: SEND closes the answers by a round with an empty verifier's message;
// otherwise the caller's next round does.
inline void toy_plan_sample(Plan &p, const Codes &codes, const std::vector<OracleShape> &sampled, unsigned answers, bool own_round = true) {
  unsigned cube = 0;
  for (const OracleShape &s : sampled) cube = std::max(cube, Send::cube(codes, s).bits);
  PlanRound &seed = p.close(R_SEND_SEED, RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING);
  seed.draw_E = (u64)answers * (cube + 1), seed.draw_nonzero = answers;  // for each answer a point and a pad factor, which is not 0
  p.send_answers("answers", (u64)answers * sampled.size());
  if (own_round) p.close_none(R_SEND_ANSWER);
}
// group: the stage's codes and numbers of rows, from group_as_run() (blocks/iop/single_rows.h).
// FOLD ends the process if it runs on another group than the plan records.
inline void toy_plan_fold(Plan &p, const PricedGroup &group, bool sends_mask) {
  if (sends_mask) p.send_root("fold mask");
  p.send_scalars("hiding sum", 1);
  PlanRound &eps = p.close(R_FOLD_EPSILON, RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING);
  eps.draw_E = 1, eps.group = group;
  for (unsigned j = 0; j < group.k; j++) {
    p.send_scalars("round polynomial", 2);
    PlanRound &round = p.close(R_FOLD_ROUND, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, Bound((u64)0), Priced::NOTHING);
    round.draw_E = 1, round.group = group;
  }
}
// rows: how many linear claims (`Row`) the element of E drawn after the positions batches; 0: no
// such element. group: the two codes of the rows read at the positions, from group_as_run() with
// k = h = 0, slots = 1; none where a committed table's columns are drawn.
inline void toy_plan_positions(Plan &p, const char *name, unsigned draws, unsigned log_n, u64 rows, const PricedGroup &group = PricedGroup()) {
  PlanRound &r = p.close(name, RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
  r.draw_positions = draws, r.log_n = log_n, r.draw_E = rows ? 1 : 0, r.rows = rows, r.group = group;
}
inline void toy_plan_bat(Plan &p, u64 rows) {
  PlanRound &r = p.close(R_BAT, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING);
  r.draw_E = 1, r.rows = rows;
}
// BAT, then FOLD and CS for each stage but the last, then BASE. rows: how many claims BAT combines.
inline void toy_plan_iopp(Plan &p, const Codes &codes, const std::vector<IoppStage> &st, unsigned draws, const IoppChoice &choice,
                          bool first_mask_sent, u64 rows) {
  toy_plan_bat(p, rows);
  for (size_t i = 0; i + 1 < st.size(); i++) {
    const IoppStage &s = st[i], &n = st[i + 1];
    toy_plan_fold(p, group_as_run(codes, s.code, s.mask_code, true, s.k, s.h, 1), i == 0 ? !first_mask_sent : !choice.masks_ride);
    p.send_root("switch");
    toy_plan_sample(p, codes, {switch_target_stage("", n.code, n.mask_code, n.k, n.h, p.set.samples).shape}, p.set.samples, false);
    toy_plan_positions(p, R_CS, draws, codes.base[s.code].log_n, 1 + p.set.samples + 3 * (u64)draws, group_as_run(codes, s.code, s.mask_code, true, 0, 0, 1));
  }
  const Code &c = codes.base[st.back().code], &zk = codes.mask[st.back().mask_code];
  if (!choice.masks_ride) p.send_root("base mask");
  p.send_scalars("masked target", 1);
  PlanRound &lambda = p.close(R_BASE_LAMBDA, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, Bound((u64)0), Priced::NOTHING);
  lambda.draw_E = 1, lambda.group = group_as_run(codes, st.back().code, st.back().mask_code, true, 0, 0, 1);
  p.send_scalars("masked word", c.ell + c.r + zk.ell + zk.r);
  toy_plan_positions(p, R_BASE, draws, c.log_n, 0, group_as_run(codes, st.back().code, st.back().mask_code, true, 0, 0, 1));
}
#endif

// A hash chain's 32-byte salt.
inline Digest toy_random_bytes(u64 seed) {
  TestCoins c(seed);
  u8 bytes[DIGEST_BYTES];
  c.bytes(bytes, sizeof bytes);
  Digest d;
  digest_from_bytes(bytes, d);
  return d;
}

// For the check on SEND's pads (each masks one sample answer): a toy prover must seed its `Coins`
// with toy_coins_seed() and set ProverContext::pads to toy_pads, where send_prove() records every
// pad, read back from the prover's memory.
inline u64 toy_coins_turn = 0;
inline std::vector<PadLook> *toy_pads = nullptr;
// The constant is no small multiple of TestCoins' step: otherwise the second run's stream could be
// the first's, shifted by a few words.
inline u64 toy_coins_seed(u64 seed) { return seed ^ (0x0123456789ABCDEFull * toy_coins_turn); }
// Calls prove() twice, with different private randomness; it must run a toy prover on the same
// data. Returns pads_fault() (blocks/iop/send.h) of the two records: null, or what is wrong, such
// as a pad that is 0 or one value at two pads, in one run (so a message sent again needs a new
// pad) or across runs.
template <class Prove> inline const char *toy_pads_fault(Prove prove) {
  std::vector<PadLook> one, other;
  toy_pads = &one, toy_coins_turn = 0;
  prove();
  toy_pads = &other, toy_coins_turn = 1;
  prove();
  toy_pads = nullptr, toy_coins_turn = 0;
  return pads_fault(one, other);
}

// Main rows from `data`, mask rows from `coins`.
inline OracleDraft toy_draft(const Codes &codes, const OracleNotice &n, Engine &engine, Coins &coins, Coins &data) {
  OracleDraft d;
  d.shape = n.shape, d.leaf_class = n.leaf_class;
  if (d.shape.main_rows) {
    Table m(d.shape.main_deg, d.shape.main_rows, codes.base[d.shape.code].ell);
    data.fill(m.a.data(), m.a.size());
    for (const PadPlace &p : Send::pad(codes, d.shape))  // left 0: `send_prove` draws every pad
      if (p.part == Part::MAIN) m.set_entry(p.index, E::zero());
    d.main = engine.put(m);
  }
  if (d.shape.mask_rows) d.mask = Send::fresh_mask(codes, d.shape, coins);
  return d;
}

// Three linear claims on the oracle messages `ids`, without their values (`target`). Together they
// have tensors, single entries, a run and mask entries. Prover and verifier make the same call: it
// takes mask entries from S.
inline std::vector<Row> toy_rows(Statement &S, const std::vector<OracleId> &ids, u64 seed) {
  TestCoins data(seed);
  std::vector<Row> rows(3);
  for (OracleId o : ids) {
    if (S.has(o, Part::MAIN)) {
      std::vector<E> z(S.table_bits(o, Part::MAIN));
      for (E &x : z) x = data.e();
      std::vector<F> f(z.size());
      data.fill(f.data(), f.size());
      const Tensor v[2] = {Tensor::eq(z.data(), (unsigned)z.size(), data.e()), Tensor::product(f.data(), (unsigned)f.size(), data.e())};
      for (unsigned i = 0; i < 2; i++) {
        for (const PadPlace &p : Send::pad(S.codes(), S.oracle(o).shape))  // no row reads the pad of a sample
          if (p.part == Part::MAIN) rows[i].add_unit(o, Part::MAIN, p.index, -v[i].at(p.index));
        rows[i].add(o, Part::MAIN, v[i]);
      }
      rows[2].add_unit(o, Part::MAIN, 1, data.e());
    }
    if (S.has(o, Part::MASK)) {
      rows[0].add_mask(S.masks(o).take(), data.e());
      rows[1].add_mask(S.masks(o).take(), data.e());
      rows[2].add_run(o, Part::MASK, 2, 1, {data.e(), data.e(), data.e()});
    }
  }
  return rows;
}

}  // namespace blk
