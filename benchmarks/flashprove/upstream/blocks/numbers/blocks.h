// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Planners: for each block (a step of the proof system) listed here, functions (`plan_send`, ...)
   that append its rounds, with error bounds (`Priced`, `priced` in identifiers), and messages,
   with sizes, to a Plan (plan.h). The error accounting (accounting.h) reads the finished plan, and
   prover and verifier go through it as they run. The caller passes all sizes: nothing here reads
   them from the objects on which a block runs, and a run checks only some (`MayLay`).
   An oracle message is rows sent as Merkle roots over their codewords; its out-of-domain sample is
   a random linear form of the whole message, chosen afterwards. A group (`GroupSize`) is rows that
   are tested together. A linear claim (`rows` in `plan_bat`, `batched_rows`, `old_rows`) says that
   a linear form of the encoded messages has a public value.
     SEND        an oracle message, the seed of its out-of-domain sample, the answer
     BAT         linear claims on a group into one, which is about some codeword within the distance
     BAT-sigma   linear claims into one, still about the decoded messages; BATT: then a sumcheck
     SPLIT       a claim on two groups into a claim on each
     PSC         masked sumcheck for a polynomial summand: hiding polynomial, variable Y
     ZERO, LINE, LKP  a zero-check's point; two evaluation claims into one; a lookup in a table
     FOLD        masked sumcheck over the row variables of a group
     LOC         a draw of positions of a group whose fold is re-encoded in a sampled oracle message
     CS, BASE    a proximity test's code switch; its last stage, which sends the last rows masked
     IOPP        a proximity test: BAT, then FOLD and CS stage by stage, then BASE
     RNG         the {-1, 0, 1} matrix for the approximate range check (ErrorKind::SMALL)
     LIN, LINK, RECORD  on committed tables: the test; the proof that ties two; a table's samples */
#pragma once
#include <atomic>
#include <cmath>
#include <string>
#include <vector>

#include "numbers/codes.h"
#include "numbers/plan.h"
#include "numbers/round_names.h"

namespace blk {

// ---------------------------------------------------------------------------------------------------------------------
// Groups

/* A group: 2^k main rows under one code (rows that were not sent count as zero rows) and 2^h mask
   rows (rows of random values) under the mask code. All are read at one list of positions (a mask
   row at x mod its length) and tested at one distance. slots > 1: each main row stands for `slots`
   rows folded side by side; rows that later parts of the protocol read are kept in this way. */
struct GroupSize {
  std::string name;
  CodeSize main, mask;  // mask.ell == 0: no mask rows
  unsigned k = 0, h = 0;
  u64 slots = 1;
  // The error analysis's two choices. `m` belongs to the theorem that codes.h cites under JOHNSON,
  // applied to the code of the two that has the larger rate (`mask_rules`).
  Radius rule = Radius::JOHNSON;  // the rule that fixes the distance
  u64 m = 0;  // JOHNSON: the theorem's integer m; AT: the distance, in positions of the main code
  bool has_mask() const { return mask.ell != 0; }
};
/* What `analysis` derives from a group and the Setting (setting.h): the distance and, at that
   distance, each code's count and list size (`Proximity`, codes.h). It is computed where it is
   needed and is a member of no struct, so that none of these is ever kept beside codes of which it
   is not true. */
struct Analysis {
  Proximity at;       // the main code at the group's distance
  Proximity mask_at;  // the mask code at the same distance; not read for a group without mask rows
  Big list = Big(1);  // list size (codewords within the distance) of main and mask rows together
  const char *fault = nullptr;  // of a bad analysis: `shape_fault`'s reason, if the shape is why
  bool with_mask = false;
  bool bad() const { return at.bad() || (with_mask && mask_at.bad()); }
};
inline Analysis analysis(const Setting &s, const GroupSize &g);
/* What a plan records about a group with FOLD's rounds and BASE's first round: the numbers that
   come from a public shape or code, and the two choices, `rule` and `m`. The blocks compare the
   numbers with the run's (`Rounds::require_group`, blocks/chain/rounds.h); a difference ends the
   process. A run has nothing to compare the choices with: instead, a finished plan in which rounds
   on the same two codes differ in a choice does not hold (plan.h, Cause::TWO_DISTANCES).
   The two overloads below bind every field of their struct to a name, so that a new field breaks
   the build here. One that comes from a public shape or code is then added to PricedCode or
   PricedGroup and to the line that fills it; a choice only to the names. */
inline PricedCode priced(const CodeSize &c) {
  const auto &[name, log_n, ell, r, over_E] = c;
  (void)name, (void)log_n, (void)ell, (void)r, (void)over_E;  // the names only count the fields
  return PricedCode{c.log_n, c.ell, c.r};  // left out: `name`, for prints; `over_E`, which sets a symbol's bytes and enters no error
}
inline PricedGroup priced(const GroupSize &g) {
  const auto &[name, main, mask, k, h, slots, rule, m] = g;
  (void)name, (void)main, (void)mask, (void)k, (void)h, (void)slots, (void)rule, (void)m;  // likewise
  PricedGroup x;  // left out: `name`, for prints
  x.main = priced(g.main), x.k = g.k, x.h = g.h, x.slots = g.slots, x.said = true;
  x.rule = (unsigned)g.rule, x.m = g.m;
  if (g.has_mask()) x.mask = priced(g.mask);
  return x;
}
// The same for a round that draws positions on a group's rows: the two codes, with k, h and slots
// as for single rows. Of a group's fields, the error agree^t of t positions (agree: 1 - distance)
// depends on the codes, `rule` and `m` only, and code that reads rows at positions knows their
// codes, not how many rows were folded into them (Rounds::require_codes()).
inline PricedGroup priced_codes(const GroupSize &g) {
  PricedGroup x = priced(g);
  x.k = x.h = 0, x.slots = 1;
  return x;
}

inline bool mask_rules(const Setting &s, const GroupSize &g);
/* Null, or why the blocks do not allow the shape. Mask rows are folded by the first h of the k row
   variables, so h <= k. A single mask row beside rows that are folded (h = 0 < k) has no proximity
   term (ErrorKind::CLOSENESS) in any round of the fold. Only the list size would then keep the
   distance below the mask code's Johnson radius, so the mask code's rate must be at most the main
   code's (`mask_rules`). */
inline const char *shape_fault(const Setting &s, const GroupSize &g) {
  if (!g.has_mask()) return g.h ? "mask rows without a mask code" : nullptr;
  if (g.mask.log_n > g.main.log_n) return "the mask code is longer than the main code";
  if (g.h > g.k) return "more row variables for the mask rows than for the main rows";
  if (g.k > 0 && g.h == 0 && mask_rules(s, g)) return "a single mask row beside rows that are folded has a code of a larger rate than theirs: lengthen the mask code";
  return nullptr;
}
/* The analysis of a group whose main code is at `at`. The mask code's proximity term is computed
   at the same distance; its count grows like rate^(-3/2), so a very low mask rate has a cost. */
inline Analysis analysis_at(const Setting &s, const GroupSize &g, const Proximity &at) {
  Analysis a;
  a.at = at, a.list = at.list, a.with_mask = g.has_mask();
  if ((a.fault = shape_fault(s, g))) a.at.count = Bound::failed();
  if (g.has_mask()) {
    /* The mask rows' symbols are stored as copies in the main rows' leaves, and nothing forces the
       copies to agree. So the mask code is analysed as its repetition, t = n / n_zk times (n_zk:
       the mask code's length), to the main length n.
       WITHIN A THIRD of the distance the count is that of a lemma of the paper that this protocol
       builds on (codes.h: THIRD; proved there) for any linear code, at the long length.
       Beyond a third the argument is this. The count, as codes.h defines it, and
       how long a list is, depend only on the code as a set of words, not on how its n positions
       are numbered. Under a numbering chosen for the proof the repeated words are polynomials in
       X^t on the subgroup of order n: a subcode of the Reed-Solomon code of degree at most
       t (d - 1) there (d: the mask code's dimension), whose reduced rate (d - 1) / n_zk is the
       mask code's own. So the term is the count of the theorem on codes, taken as given (codes.h:
       JOHNSON), at length n and that rate, plus one list size for the step to a subcode, which is
       added also where the two lengths are equal.
       Not claimed, and not true: that under the format's copy map and the main code's own
       numbering a repeated word is in a Reed-Solomon code on the main domain.
       The count requires of the format: one public copy map under which every short position has t
       copies; a coset as the short code's domain; n dividing p - 1; a distance below the mask
       code's own Johnson radius; mask symbols combined and compared with mask symbols only; symbols combined position by position only; a code that is linear over E.
       The term enters twice in each round j <= h of a fold, once in BASE's first round
       (`fold_round_error`, `base_error`), and in no other round.
       The group's list size goes by the smaller of the two minimum distances: it is the larger of
       the two codes' bounds, that of the code with the larger rate. */
    a.mask_at = at_distance(repeated(s, g.mask, g.main.log_n), at.agree);
    if (a.list < a.mask_at.list) a.list = a.mask_at.list;
    if (a.mask_at.source != CountSource::THIRD) a.mask_at.count = a.mask_at.count + Bound(a.mask_at.list);
  }
  return a;
}

// The error of FOLD's round j = 1..k.
// Follows the lemma on the round-by-round security of the zero-knowledge sumcheck IOR of
// the paper that this protocol builds on (see THIRD_PARTY.md).
inline Bound fold_round_error(const Setting &s, const GroupSize &g, const Analysis &a, unsigned j) {
  Bound e = mca_error(s, a.at) + Bound(a.list) * field_error(s, 2);
  if (g.has_mask() && j <= g.h) e = e + Bound(2) * mca_error(s, a.mask_at);
  return e;
}

// The error of BASE's first round, for a group of single rows.
// Follows the lemma on the round-by-round knowledge soundness of the non-succinct zero-knowledge
// protocol of the paper that this protocol builds on (see THIRD_PARTY.md).
inline Bound base_error(const Setting &s, const GroupSize &g, const Analysis &a) {
  Bound e = mca_error(s, a.at) + Bound(a.list) * field_error(s, 1);
  if (g.has_mask()) e = e + mca_error(s, a.mask_at);
  return e;
}

/* What `group_holds` assumes of the rounds outside FOLD and BASE whose error depends on a group's
   distance. cube_bits: the largest number of variables of a sample's point on one of its oracle
   messages. messages: how many of its oracle messages at most share one seed (their errors add).
   batched_rows: the largest number of linear claims that a BAT on it batches. A plan that exceeds
   these is not thereby accepted: the finished plan is checked on every round that was in fact
   planned. */
struct GroupLoad {
  unsigned cube_bits = 40;
  u64 messages = 4;
  u64 batched_rows = (u64)1 << 20;
};

// The largest error that meets the target with w bits of proof of work: 2^(w - target), divided in
// the scope REQUEST (`Scope`, setting.h) by Setting::instances. The programs leave that field at 1
// (setting_of(), blocks/frame/derive.h, does not set it): their bound on a REQUEST round counts one
// request alone, not all requests of a batch. Outside tests only cp_plan.cpp sets it, from its line
// `instances`.
inline Real goal_of(const Setting &s, ErrorKind why, Scope scope) {
  const Real g = Real::two_to((long)s.work_of(why, scope) - (long)s.target);
  return scope == Scope::REQUEST ? Real::div(g, Real(s.instances), Dir::DOWN) : g;
}

// True if the rounds whose error depends on the group's distance, except the draws of positions,
// meet the target under `load`.
inline bool group_holds(const Setting &s, Scope scope, const GroupSize &g, const GroupLoad &load) {
  const Analysis a = analysis(s, g);
  if (a.bad()) return false;
  const Bound worst = g.k ? fold_round_error(s, g, a, 1) : base_error(s, g, a);
  return worst.error() <= goal_of(s, ErrorKind::CLOSENESS, scope) &&
         (Bound(Big(load.messages)) * sample_error(s, a.list, load.cube_bits)).error() <= goal_of(s, ErrorKind::SAMPLE, scope) &&
         (Bound(a.list) * (powers_error(s, load.batched_rows) + field_error(s, 1))).error() <= goal_of(s, ErrorKind::FIELD, scope);  // with FOLD's first challenge
}

/* True if the mask code has the larger rate. The code with the larger rate determines the
   distance: one distance serves main and mask rows, and it must be within the range of the bounds
   for both codes. At the late stages of a proximity test the mask rows can be the longer
   messages. */
inline bool mask_rules(const Setting &s, const GroupSize &g) { return g.has_mask() && (u128)g.mask.k(s) * g.main.n() > (u128)g.main.k(s) * g.mask.n(); }
// The group under the rule JOHNSON with the integer m, for the code with the larger rate.
inline GroupSize group_of_m(const GroupSize &shape, u64 m) {
  GroupSize g = shape;
  g.m = m, g.rule = Radius::JOHNSON;
  return g;
}
inline Analysis analysis_of_m(const Setting &s, const GroupSize &g) {
  if (!mask_rules(s, g)) return analysis_at(s, g, johnson(s, g.main, g.m));
  Proximity z = johnson(repeated(s, g.mask, g.main.log_n), g.m);
  Analysis a = analysis_at(s, g, at_distance(s, g.main, z.agree));
  z.count = z.count + Bound(z.list);  // as in analysis_at
  a.mask_at = z;
  return a;
}
/* The group at the largest m that `largest_m` finds with `group_holds` true, that is at the
   smallest slack below the Johnson radius. If it finds none,
   the group has m = 0 and its analysis is bad. */
inline GroupSize size_group(const Setting &s, Scope scope, const GroupSize &shape, const GroupLoad &load) {
  u64 m = largest_m([&](u64 m) { return group_holds(s, scope, group_of_m(shape, m), load); });
  return group_of_m(shape, m);
}
// The group at a distance that the caller names, in whole positions of the main code's length (for
// the planning program cp_plan.cpp): an integer, so no interval (Bound) is passed.
inline GroupSize group_at(const GroupSize &shape, u64 positions) {
  GroupSize g = shape;
  g.m = positions, g.rule = Radius::AT;
  return g;
}
/* The analysis of a group under its rule. `rule` and `m` are choices: each gives the cited bounds
   as this file applies them to these codes, or a bad analysis. The bounds are not all proved
   (codes.h): the count that codes.h calls JOHNSON is assumed for these codes, and we have verified
   no proof of it; a mask code's count beyond a third of the distance is our own step
   (`analysis_at`). Neither choice shows that the rounds
   meet the target: for that the finished plan is checked. */
inline Analysis analysis(const Setting &s, const GroupSize &g) {
  switch (g.rule) {
    case Radius::JOHNSON: return analysis_of_m(s, g);
    case Radius::AT: return analysis_at(s, g, at_distance(s, g.main, Bound(1) - Bound::ratio(Big(g.m), Big(g.main.n()))));
  }
  return Analysis();
}
// For a planner: the group's analysis. A bad one rejects the plan, with `where` in the reason.
inline Analysis analysis(Plan &p, const std::string &where, const GroupSize &g) {
  const Analysis a = analysis(p.set, g);
  if (a.bad()) p.refuse(where + ": " + (a.fault ? a.fault : "the group's codes have no admissible distance under its rule"));
  return a;
}
inline u64 scope_positions(const Setting &s, Scope scope, const Bound &agree, const Bound &extra);
/* For an iteration from below on the numbers of positions and of random coordinates (`r`), as in
   `solve_stages`. On the way up a shape can break a rule that it keeps at the fixed point: with
   few random coordinates the mask code may still have the larger rate. The main code alone then
   gives a number of positions that is not too large (the mask code can only lower the distance),
   and the iteration can go on. 0: not even the main code alone meets the target. */
inline u64 positions_from_below(const Setting &s, Scope scope, GroupSize shape, const GroupLoad &load, const Bound &extra) {
  shape.mask = CodeSize(), shape.h = 0;
  const GroupSize g = size_group(s, scope, shape, load);
  return group_holds(s, scope, g, load) ? scope_positions(s, scope, analysis(s, g).at.agree, extra) : 0;
}
// On what the group's proximity terms rest: the weaker of its two codes' sources.
inline CountSource weakest_source(const Analysis &a) {
  return a.with_mask && (int)a.mask_at.source > (int)a.at.source ? a.mask_at.source : a.at.source;
}

// ---------------------------------------------------------------------------------------------------------------------
// Trees and openings

/* Marks a tree with an oracle message that may be read at one list of positions only: the fresh
   mask of a fold, BASE's masking codeword. An opening sends all of a leaf, so such a tree may be
   opened at that list only, and in no chain but the one that sent it or one that continues it.
   `plan_open` sees less than that. It rejects a second opening of the same TreeSize, or of a copy,
   within one Plan, and any opening of a marked one whose `from_another_chain` is set; the adopting
   chain's planner must set that field, and nothing here notices if it does not. At the run
   `TreeBook` (blocks/iop/context.h) has its own checks of the rule. The mark gives the tree a
   number, never given twice in a process, that copies of the TreeSize keep; a Plan remembers the
   numbers that it has opened. A TreeSize filled anew for the same tree gets a new number. */
struct OneList {
  bool yes = false;
  u64 number = 0;
  OneList &operator=(bool mark) {
    static std::atomic<u64> last{0};
    yes = mark;
    if (mark && !number) number = ++last;
    return *this;
  }
  explicit operator bool() const { return yes; }
};
/* The sizes of one Merkle tree, a perfect binary tree. A leaf has leaf_bytes of symbols and one
   salt: the symbols of the tree's rows at the leaf's position, then copies of the symbols of
   shorter rows in the same tree (mask rows), at that position modulo their length.
   Which oracle messages share a tree is the caller's choice: a function that sends them takes the
   number of roots, one that opens positions takes the trees. Of this file's functions only
   `plan_iopp` makes trees, for the oracle messages that are sent within the test. */
struct TreeSize {
  std::string name;
  unsigned log_n = 0;
  u64 leaf_bytes = 0;
  OneList one_list;                 // set by the code that fills this struct
  bool from_another_chain = false;  // set by the planner of a chain that adopts a tree which another chain has sent
};

// How many of 2^log_n positions do t independent uniform draws hit, on average?
inline double expected_distinct(unsigned log_n, u64 t) {
  const double n = std::ldexp(1.0, (int)log_n);
  return -n * std::expm1((double)t * std::log1p(-1.0 / n));
}
/* Sibling digests that a pruned opening of t positions needs. At the level with N vertices a
   digest is sent for each pair of siblings of which exactly one is above an opened leaf.
   `path_digests_most`, an upper bound: t distinct leaves need the most digests when their paths
   part as high in the tree as they can. With 2^a <= t < 2^(a+1), all vertices down to depth a are
   above an opened leaf, at depth a + 1 there are 2^(a+1) - t lone siblings, and below it no two
   paths share a vertex: t (log_n - a - 1) + 2^(a+1) - t. The number does not fall as t grows up to
   n / 2, and fewer distinct leaves than draws need no more. (blocks/numbers/tests/numbers_test.cpp
   compares the formula with a search over all splits, up to depth 7.)
   `path_digests_expected`, the average over t independent uniform draws: the sum over the levels
   of N ((1 - 1/N)^t - (1 - 2/N)^t). */
inline u64 path_digests_most(unsigned log_n, u64 t) {
  if (!t || !log_n) return 0;
  t = std::min<u64>(t, (u64)1 << (log_n - 1));
  unsigned a = 0;
  while (((u64)2 << a) <= t) ++a;
  return t * (log_n - a - 1) + ((u64)2 << a) - t;
}
inline double path_digests_expected(unsigned log_n, u64 t) {
  double d = 0;
  for (unsigned j = 1; j <= log_n; ++j) {
    const double N = std::ldexp(1.0, (int)j);
    d += N * (std::exp((double)t * std::log1p(-1.0 / N)) - (j == 1 ? 0.0 : std::exp((double)t * std::log1p(-2.0 / N))));
  }
  return d;
}

// ---------------------------------------------------------------------------------------------------------------------
// Which functions may add a round

class TestedIn;
struct SampleSize;
struct PscSize;
struct LookupSize;
struct TableSize;
struct LinSize;
/* Only the friends named here can add something to a Plan: Plan's member functions that do so are
   private (public for a test that defines BLK_PLANS_BY_HAND_FOR_A_TEST), this class is Plan's only
   friend, and all its members are private. This guards against a slip in a source that uses these
   headers as they are meant, not against a caller that sets out to get round it. A new function
   that calls a member of this class must be named in one of the two lists, which makes this the
   place to check a new error formula.
   The first list adds rounds with a verifier's message. Each function computes its round's error
   by a formula of this file, from its arguments and the plan's Setting, and states in a `Priced`
   (plan.h) from what. AS_SAID: from numbers that the plan records with the round and the block
   compares with its run. ON_THE_CALLERS_WORD: also from numbers that nothing checks at this round;
   a note on the line names them. A round's name is a constant of round_names.h; the code that runs
   the round passes the same one to the hash chain (the Fiat-Shamir transcript,
   blocks/chain/chain.h).
   The second list adds messages, items of a file and rounds with an empty verifier's message:
   sizes and names only, no error, no draw. `plan_chain`, the last, copies the rounds of a finished
   plan.
   READ ONCE marks where a listed function copies a size argument, so that a round's error and what
   is recorded with the round come from one object. The argument `where` is a label for prints and
   rejection reasons; no error depends on it. */
class MayLay final {
  friend void plan_send(Plan &, const std::string &, const std::vector<SampleSize> &, unsigned, bool);
  friend void plan_bat(Plan &, const std::string &, const TestedIn &, u64);
  friend void plan_bat_sigma(Plan &, const std::string &, u64);
  friend void plan_psc(Plan &, const std::string &, const PscSize &);
  friend void plan_zero(Plan &, const std::string &, unsigned);
  friend void plan_line(Plan &, const std::string &);
  friend void plan_shift(Plan &, const std::string &, u64);
  friend void plan_join(Plan &, const std::string &, unsigned);
  friend void plan_lkp(Plan &, const std::string &, unsigned);
  friend void plan_lookup(Plan &, const std::string &, const LookupSize &);
  friend void plan_rng(Plan &, const std::string &, u64);
  friend void plan_fold(Plan &, const std::string &, const GroupSize &, unsigned);
  friend void plan_fold_rounds(Plan &, const std::string &, const GroupSize &);
  friend u64 plan_loc(Plan &, const std::string &, const GroupSize &, const std::vector<TreeSize> &);
  friend u64 plan_cs(Plan &, const std::string &, const GroupSize &, const GroupSize &, unsigned, u64, unsigned, const std::vector<TreeSize> &, unsigned);
  friend u64 plan_base(Plan &, const std::string &, const GroupSize &, const std::vector<TreeSize> &, unsigned);
  friend void plan_record(Plan &, const std::string &, const TableSize &);
  friend void plan_seal(Plan &, const std::string &);
  friend u64 plan_lin_shared(Plan &, const std::string &, const TableSize &, const LinSize &, const TreeSize &);
  friend void plan_link(Plan &, const std::string &, const TableSize &, const SampleSize &, const TreeSize &, u64 &, u64 &);

  friend void plan_none(Plan &, const char *, const std::string &);
  friend void plan_scalars(Plan &, const char *, const std::string &, u64);
  friend void plan_roots(Plan &, const std::string &, unsigned);
  friend void plan_open(Plan &, const std::string &, const TreeSize &, u64);
  friend void plan_chain_start(Plan &, const std::string &, bool);
  friend void plan_file_item(Plan &, const char *, const std::string &, MessageKind, u64, double, u64);
  friend void plan_chain(Plan &, const Planned &);

  // All are noexcept: an exception inside one of these calls, from an allocation, ends the process.
  // close(): a round with a verifier's message. r: what the round draws and what the plan records
  // with it. Its planner fills r before the call, so that the plan never has a half-filled round.
  static void close(Plan &p, const char *name, const std::string &where, RoundKind kind, ErrorKind why, Bound error, Priced priced, PlanRound r = PlanRound()) noexcept {
    if (p.scope() == Scope::REQUEST && p.set.instances > 1) error = error * Bound(Big(p.set.instances));
    r.where = where;
    if (p.close(name, kind, why, error, priced, std::move(r)) && p.set.nonce_bytes) p.last_message_at(where);  // the nonce
  }
  static void enter(Plan &p, const PlanRound &r) noexcept { p.enter(r); }
  static void none(Plan &p, const char *name, const std::string &where) noexcept { p.close_none(name, where); }
  static void scalars(Plan &p, const char *what, const std::string &where, u64 n) noexcept {
    p.send_scalars(what, n);
    if (n) p.last_message_at(where);
  }
  static void answers(Plan &p, const std::string &where, u64 n) noexcept {
    p.send_answers("sample answers", n);
    if (n) p.last_message_at(where);
  }
  static void roots(Plan &p, const std::string &where, unsigned roots) noexcept {
    for (unsigned i = 0; i < roots; ++i) p.send_root("root"), p.last_message_at(where);
  }
  static void file(Plan &p, const char *what, const std::string &where, MessageKind kind, u64 bytes, double expected, u64 hash_calls) noexcept {
    p.file(what, kind, bytes, expected, hash_calls);
    if (bytes) p.last_message_at(where);
  }
  static void chain(Plan &p, const Planned &o) noexcept { p.append(o); }
  static bool first_opening(Plan &p, u64 tree) noexcept { return p.one_list_opened_.insert(tree).second; }  // of a tree that may be opened at one list only (`OneList`)
};

// The answers to one list of t positions in one tree: symbols, salts, sibling digests.
inline void plan_open(Plan &p, const std::string &where, const TreeSize &callers_tr, u64 t) {
  const TreeSize tr = callers_tr;  // READ ONCE (plan.h)
  const Setting &s = p.set;
  if (tr.one_list) {
    if (tr.from_another_chain) p.refuse(where + ": the tree " + tr.name + " holds the mask of one step and is read in another chain");
    if (!MayLay::first_opening(p, tr.one_list.number)) p.refuse(where + ": the tree " + tr.name + " holds the mask of one step and is opened at a second list");
  }
  const double main = expected_distinct(tr.log_n, t);
  // A leaf is hashed as label, symbols, salt. The salt never crosses a hash block's boundary: if
  // it would, zeros fill the block before it.
  auto calls = [&](u64 bytes) {
    const u64 before = s.label_bytes + bytes, block = s.hash_block_bytes;
    return before % block + s.salt_bytes <= block ? (before + s.salt_bytes + block - 1) / block : (before + block - 1) / block + (s.salt_bytes + block - 1) / block;
  };
  auto add = [&](const char *what, MessageKind k, u64 bytes, double expected, u64 hash_calls) { MayLay::file(p, what, where + ", " + tr.name, k, bytes, expected, hash_calls); };
  add("opened symbols", MessageKind::COLUMNS, t * tr.leaf_bytes, main * (double)tr.leaf_bytes, t * calls(tr.leaf_bytes));
  add("salts", MessageKind::SALTS, t * s.salt_bytes, main * s.salt_bytes, 0);
  add("paths", MessageKind::PATHS, path_digests_most(tr.log_n, t) * s.digest_bytes, path_digests_expected(tr.log_n, t) * s.digest_bytes, t * tr.log_n);
}

// ---------------------------------------------------------------------------------------------------------------------
// Small helpers

// A round with an empty verifier's message. If `name` is the format's name for a round of another
// kind, the process ends (Plan::require_kind()); a Merkle root in the prover's message rejects the
// plan (Plan::close_none()).
inline void plan_none(Plan &p, const char *name, const std::string &where) { MayLay::none(p, name, where); }
// In the prover's message: n elements of E (the challenge field, setting.h), named `what` in
// prints; `roots` Merkle roots.
inline void plan_scalars(Plan &p, const char *what, const std::string &where, u64 n) { MayLay::scalars(p, what, where, n); }
inline void plan_roots(Plan &p, const std::string &where, unsigned roots) { MayLay::roots(p, where, roots); }
// An item of a file that is in no chain's message: a header, a committed table's root, a place in
// a batch.
inline void plan_file_item(Plan &p, const char *what, const std::string &where, MessageKind kind, u64 bytes, double expected, u64 hash_calls) { MayLay::file(p, what, where, kind, bytes, expected, hash_calls); }
// Appends the rounds and messages of a finished chain's plan, under this plan's current part (a
// heading in prints, Plan::begin_part()): for prints and totals over several chains. If that plan
// was rejected, or its errors were computed under another Setting, this plan is rejected. In the
// middle of a prover's message the process ends.
inline void plan_chain(Plan &p, const Planned &callers_chain) {
  const Planned chain = callers_chain;  // READ ONCE (plan.h): a handle of its own
  MayLay::chain(p, chain);
}
// The smallest number of positions for a draw in `scope` (codes.h, positions()). 0: none reaches
// the target.
inline u64 scope_positions(const Setting &s, Scope scope, const Bound &agree, const Bound &extra) {
  const unsigned w = s.work_of(ErrorKind::QUERY, scope);
  const u64 t = positions(agree, s.target > w ? s.target - w : 0, extra, scope == Scope::REQUEST ? s.instances : 1);
  return t;
}
inline u64 scope_positions(const Setting &s, Scope scope, const Bound &agree) { return scope_positions(s, scope, agree, Bound((u64)0)); }
inline u64 plan_positions(const Plan &p, const Bound &agree, const Bound &extra = Bound((u64)0)) {
  return scope_positions(p.set, p.scope(), agree, extra);
}

// What a hash chain's part of a proof begins with: the chain's salt, as long as a digest, outside
// every message. A chain that hides nothing has none.
inline void plan_chain_start(Plan &p, const std::string &where, bool salted) {
  if (!salted) return;
  MayLay::file(p, "chain salt", where, MessageKind::SALTS, p.set.digest_bytes, p.set.digest_bytes, 0);
}

// ---------------------------------------------------------------------------------------------------------------------
// SEND, BAT, SPLIT

/* The groups in which some rows are tested. The error of a sample and of a BAT grows with the
   largest of their list sizes. The empty case, a list size of 1 that nothing checks, is made only
   by `no_sized_group`: `{}` does not compile. That name is for tests only; a search for it finds
   every such case. */
class TestedIn {
 public:
  TestedIn(const GroupSize &g) : groups_{g} {}
  TestedIn(const GroupSize &g, const GroupSize &h) : groups_{g, h} {}
  TestedIn(const TestedIn &) = default;  // a move is a copy, so std::move leaves none empty
  TestedIn &operator=(const TestedIn &) = default;
  std::vector<GroupSize>::const_iterator begin() const { return groups_.begin(); }
  std::vector<GroupSize>::const_iterator end() const { return groups_.end(); }

 private:
  TestedIn() {}
  friend TestedIn no_sized_group();
  std::vector<GroupSize> groups_;
};
inline TestedIn no_sized_group() { return TestedIn(); }
/* What a sample's error takes from one oracle message: the number of variables of the sample's
   point, and the groups in which the message's rows are tested (both, if they are in two). */
struct SampleSize {
  unsigned cube_bits;
  TestedIn groups;
};
inline PlanRound draws(u64 elements) {  // draws `elements` elements of E; records nothing else
  PlanRound r;
  r.draw_E = elements;
  return r;
}
inline Big list_of(Plan &p, const std::string &where, const TestedIn &groups) {
  Big list(1);
  for (const GroupSize &g : groups) list = std::max(list, analysis(p, where, g).list);
  return list;
}
/* SEND for all oracle messages of one prover message. Each gives Setting::samples answers. For
   each answer all share one seed: a point with the largest cube_bits coordinates, of which each
   message uses its first ones, and a factor for the pad, the random element of each message that
   masks this answer. The errors add. roots: the Merkle roots in the prover's message.
   answer_round false: the calling block closes the round of the answers with a verifier's message
   of its own. */
inline void plan_send(Plan &p, const std::string &where, const std::vector<SampleSize> &callers_o, unsigned roots, bool answer_round = true) {
  const std::vector<SampleSize> o = callers_o;  // READ ONCE (plan.h)
  plan_roots(p, where, roots);
  Bound e((u64)0);
  unsigned cube = 0;
  for (auto &x : o) e = e + sample_error(p.set, list_of(p, where, x.groups), x.cube_bits), cube = std::max(cube, x.cube_bits);
  {
    PlanRound r;
    r.draw_E = (u64)p.set.samples * (cube + 1), r.draw_nonzero = p.set.samples;  // for each answer a point and a pad factor, which is not 0
    MayLay::close(p, R_SEND_SEED, where, RoundKind::SEED, ErrorKind::SAMPLE, e, Priced::ON_THE_CALLERS_WORD, r);  // the groups whose list sizes count; the smaller `cube_bits`
  }
  MayLay::answers(p, where, (u64)p.set.samples * o.size());
  if (answer_round) plan_none(p, R_SEND_ANSWER, where);
}
// groups: as in a SampleSize, those in which the rows are tested. rows: how many linear claims.
inline void plan_bat(Plan &p, const std::string &where, const TestedIn &callers_groups, u64 rows) {
  const TestedIn groups = callers_groups;  // READ ONCE (plan.h)
  PlanRound r;
  r.draw_E = 1, r.rows = rows;
  MayLay::close(p, R_BAT, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(list_of(p, where, groups)) * powers_error(p.set, rows), Priced::ON_THE_CALLERS_WORD, r);  // the groups whose list sizes count; `rows` is recorded and compared
}
inline void plan_bat_sigma(Plan &p, const std::string &where, u64 rows) {
  PlanRound r;
  r.draw_E = 1, r.rows = rows;
  MayLay::close(p, R_BAT_SIGMA, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, powers_error(p.set, rows), Priced::AS_SAID, r);
}
inline void plan_split(Plan &p, const std::string &where) {
  plan_scalars(p, "split target", where, 1);
  plan_none(p, R_SPLIT, where);
}

// ---------------------------------------------------------------------------------------------------------------------
// PSC, ZERO, LINE, product trees, RNG

/* degree[j]: the round polynomial of variable j has degree below degree[j], which must be at least
   1 (not checked). degree_Y: the total degree of the summand in the maps that it evaluates.
   with_Y false: the summand is affine in all its maps; no further variable, and no values are sent.
   values: if with_Y is true, how many values at the last point the prover sends after Y's round.
   With Setting::saved_coefficient, the coefficient that the verifier's check fixes is not sent. */
struct PscSize {
  std::vector<unsigned> degree;
  unsigned degree_Y = 2;
  unsigned values = 2;
  bool with_Y = true;
};
inline void plan_psc(Plan &p, const std::string &where, const PscSize &callers_x) {
  const PscSize x = callers_x;  // READ ONCE (plan.h)
  const Setting &s = p.set;
  const unsigned saved = s.saved_coefficient ? 1 : 0;
  plan_scalars(p, "hiding sum", where, 1);
  /* Epsilon multiplies the summand before the hiding polynomial is added. The error is 1 / |E|,
     without the list size that is a factor in FOLD's epsilon round: the claim is about one fixed
     object. What the channel (`Rounds`) has absorbed fixes weights, summand and value; oracle
     messages whose samples are answered fix the maps; nothing is read at a distance from a
     codeword. A false claim survives at most one value of epsilon. */
  MayLay::close(p, R_PSC_EPSILON, where, RoundKind::EPSILON, ErrorKind::FIELD, field_error(s, 1), Priced::SETTING_ALONE, draws(1));
  for (size_t j = 0; j < x.degree.size();) {  // p.compact: a run of equal rounds is entered once
    size_t run = 1;
    if (p.compact)
      while (j + run < x.degree.size() && x.degree[j + run] == x.degree[j]) ++run;
    if (run > 1) p.push_repeat(run);
    plan_scalars(p, "round polynomial", where, x.degree[j] - saved);
    MayLay::close(p, R_PSC_ROUND, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, field_error(s, x.degree[j] - 1), Priced::THE_ROUND_ITSELF, draws(1));
    if (run > 1) p.pop_repeat();
    j += run;
  }
  if (!x.with_Y) return;
  plan_scalars(p, "round polynomial in Y", where, x.degree_Y + 1 - saved);
  {
    PlanRound r;
    r.draw_E = r.draw_nonzero = 1;
    MayLay::close(p, R_PSC_Y, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, field_error_nonzero(s, x.degree_Y), Priced::THE_ROUND_ITSELF, r);
  }
  plan_scalars(p, "values at the last point", where, x.values);
  plan_none(p, R_PSC_VALUES, where);
}
inline void plan_zero(Plan &p, const std::string &where, unsigned variables) {
  MayLay::close(p, R_ZERO, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, field_error(p.set, variables), Priced::THE_ROUND_ITSELF, draws(variables));
}
inline void plan_line(Plan &p, const std::string &where) {
  MayLay::close(p, R_LINE, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, field_error(p.set, 1), Priced::SETTING_ALONE, draws(1));
}
/* Joins several lookup tables into one by a random shift theta: key + j theta is looked up in the
   union of the tables T_j + j theta. A key outside its table is in the union for at most as many
   values of theta as the union has entries. */
inline void plan_shift(Plan &p, const std::string &where, u64 union_entries) {
  MayLay::close(p, R_LKP_SHIFT, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, field_error(p.set, union_entries), Priced::ON_THE_CALLERS_WORD, draws(1));  // the tables' entries
}
// Joins 2^variables claims of one shape into one, by equality weights at a fresh point: a false
// claim stays false but for variables / |E| of the points.
inline void plan_join(Plan &p, const std::string &where, unsigned variables) {
  MayLay::close(p, R_JOIN, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, field_error(p.set, variables), Priced::THE_ROUND_ITSELF, draws(variables));
}
// The challenge at which two products over 2^keys_log lookup keys are compared.
inline void plan_lkp(Plan &p, const std::string &where, unsigned keys_log) {
  MayLay::close(p, R_LKP, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound::ratio(Big::two_to(keys_log) - Big(1), p.set.card_E()), Priced::ON_THE_CALLERS_WORD, draws(1));  // the keys
}
/* The levels of a product tree from the root down. Level l = 0 .. levels - 1 is a PSC with l + more
   variables, for the product of two halves under equality weights, with two values; then LINE.
   more: variables that index the trees of a lookup forest (the product trees of several lookups)
   or the bits of a multiplicity. */
inline void plan_product_tree(Plan &p, const std::string &where, unsigned levels, unsigned more) {
  for (unsigned l = 0; l < levels; ++l) {
    PscSize x;
    x.degree.assign(l + more, 3);
    plan_psc(p, where, x);
    plan_line(p, where);
  }
}
/* BATT: batches `rows` linear claims on the committed table's matrix into one: a batching
   challenge, then a sumcheck over the k_col column variables. Its summand is affine in its maps,
   so it has no round in Y and no values at the last point. */
inline void plan_batt(Plan &p, const std::string &where, u64 rows, unsigned k_col) {
  plan_bat_sigma(p, where, rows);
  PscSize x;
  x.degree.assign(k_col, 3), x.with_Y = false;
  plan_psc(p, where, x);
}
/* One lookup (LKP): 2^v keys, never sent, in a public table of 2^d entries. Each multiplicity is
   written in 2^mu bits, which the caller has sent and sampled. shifted: several functions share the
   table (`plan_shift`). Three of the labels in `plan_lookup`:
     tie       a chain of sumchecks without variables, on values that are never sent: by squarings
               it ties the key side's root to the table side's roots, one for each bit position; a
               point then joins the claims on those roots
     leaves    the layer of the table side's leaves, affine in the bits
     bits      the check that every bit is 0 or 1 */
struct LookupSize {
  unsigned v = 0, d = 0, mu = 0;
  bool shifted = false;
  unsigned live = 0;            // how many of the 2^mu bit positions are in use; 0: all
  std::vector<unsigned> parts;  // log2 of each part's number of keys; empty: one part of 2^v keys
  LookupSize() = default;
  LookupSize(unsigned v_, unsigned d_, unsigned mu_, bool shifted_, unsigned live_ = 0, std::vector<unsigned> parts_ = {})
      : v(v_), d(d_), mu(mu_), shifted(shifted_), live(live_), parts(std::move(parts_)) {}
};
inline void plan_lookup(Plan &p, const std::string &where, const LookupSize &callers_x) {
  const LookupSize x = callers_x;  // READ ONCE (plan.h)
  /* A multiplicity of w bits can count all keys, and the sum of all multiplicities, read in F,
     cannot wrap around: (number of keys) < 2^w and 2^d (2^w - 1) < p. */
  const u64 slots = (u64)1 << std::min(x.mu, 6u), w = x.live ? x.live : slots;  // w: the bit positions in use
  const std::vector<unsigned> parts = x.parts.empty() ? std::vector<unsigned>{x.v} : x.parts;
  Big keys(0);
  for (unsigned v : parts) keys = keys + Big::two_to(v);
  if (x.mu < 1 || x.mu > 6 || w < 2 || w > slots || !(keys < Big::two_to(w)) || !(Big::two_to(x.d) * (Big::two_to(w) - Big(1)) < p.set.p))
    return p.refuse(where + ": the look-up's sizes are not admitted");
  if (x.shifted) plan_shift(p, where, (u64)1 << x.d);
  MayLay::close(p, R_LKP, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound::ratio(keys - Big(1), p.set.card_E()), Priced::ON_THE_CALLERS_WORD, draws(1));  // the keys: checked later, as the run follows the trees' rounds
  for (u64 j = 0; j + 1 < w;) {
    const u64 run = j && p.compact ? w - 1 - j : 1;  // all steps but the first are alike
    PscSize step;
    step.degree_Y = 3, step.values = j ? 2 : 3;
    if (run > 1) p.push_repeat(run);
    plan_psc(p, where + ", tie", step);
    if (run > 1) p.pop_repeat();
    j += run;
  }
  for (size_t i = 0; i + 1 < parts.size(); ++i) {  // from the product of the parts' roots to each root
    PscSize step;
    plan_psc(p, where + ", parts", step);
  }
  plan_join(p, where, x.mu);
  for (unsigned v : parts) plan_product_tree(p, where + ", keys", v, 0);
  plan_product_tree(p, where + ", table", x.d, x.mu);
  PscSize leaves, bits;
  leaves.degree.assign(x.d, 3), leaves.degree.insert(leaves.degree.end(), x.mu, 2), leaves.with_Y = false;  // variables are bound from the highest down
  plan_psc(p, where + ", leaves", leaves);
  plan_zero(p, where + ", bits", x.mu + x.d);
  bits.degree.assign(x.mu + x.d, 3), bits.degree_Y = 2, bits.values = 1;
  plan_psc(p, where + ", bits", bits);
}
/* The matrix of RNG, the approximate range check. Its entries are 0 with probability 1/2 and 1 or
   -1 with probability 1/4 each; the error is then 2^-combinations (uniform entries would give
   (5/9)^combinations). The number of combinations is derived: the target less the round's bits of
   proof of work, plus, in the scope REQUEST, ceil(log2(instances)), which is 0 in the programs
   (`goal_of`). */
inline unsigned rng_combinations(const Setting &s, Scope scope) {
  const unsigned w = s.work_of(ErrorKind::SMALL, scope);
  return (s.target > w ? s.target - w : 0) + (scope == Scope::REQUEST ? ceil_log2(s.instances) : 0);
}
inline unsigned rng_combinations(const Plan &p) { return rng_combinations(p.set, p.scope()); }
// columns: of all matrices of cells (committed small integers); N entries are drawn for each.
inline void plan_rng(Plan &p, const std::string &where, u64 columns) {
  const unsigned N = rng_combinations(p);
  if (N && columns > (~(u64)0 - 31) / N) return p.refuse(where + ": so many columns that the number of drawn entries does not fit 64 bits");
  PlanRound r;
  r.draw_trits = columns * N, r.combinations = N;
  MayLay::close(p, R_RNG, where, RoundKind::BITS, ErrorKind::SMALL, Bound(Real::two_to(-(long)N)), Priced::AS_SAID, r);
}

// ---------------------------------------------------------------------------------------------------------------------
// FOLD, LOC, CS and BASE; IOPP

inline void plan_fold_rounds(Plan &p, const std::string &where, const GroupSize &g);
// FOLD. The fresh mask: 2^h rows under the mask code with a hiding polynomial's coefficients.
// roots: 1 if the fresh mask's tree is sent with this message, 0 if the mask was sent before.
inline void plan_fold(Plan &p, const std::string &where, const GroupSize &callers_g, unsigned roots) {
  const GroupSize g = callers_g;  // READ ONCE (plan.h)
  const Analysis a = analysis(p, where, g);
  plan_roots(p, where, roots);
  plan_scalars(p, "hiding sum", where, 1);
  PlanRound r;
  r.draw_E = 1, r.group = priced(g);
  MayLay::close(p, R_FOLD_EPSILON, where, RoundKind::EPSILON, ErrorKind::FIELD, Bound(a.list) * field_error(p.set, 1), Priced::AS_SAID, r);
  plan_fold_rounds(p, where, g);
}
inline void plan_fold_rounds(Plan &p, const std::string &where, const GroupSize &callers_g) {
  const GroupSize g = callers_g;  // READ ONCE (plan.h)
  const Setting &s = p.set;
  const Analysis a = analysis(p, where, g);
  for (unsigned j = 1; j <= g.k;) {
    unsigned run = 1;  // past the mask's h variables all errors are equal
    if (p.compact && (!g.has_mask() || j > g.h)) run = g.k - j + 1;
    if (run > 1) p.push_repeat(run);
    plan_scalars(p, "round polynomial", where, s.saved_coefficient ? 2 : 3);
    PlanRound r;
    r.draw_E = 1, r.rests = g.has_mask() && j <= g.h ? weakest_source(a) : a.at.source, r.group = priced(g);
    MayLay::close(p, R_FOLD_ROUND, where, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, fold_round_error(s, g, a, j), Priced::AS_SAID, r);
    if (run > 1) p.pop_repeat();
    j += run;
  }
}
// LOC: t positions of a group, drawn after the answer to the sample of the oracle message that
// re-encodes its fold. Returns t.
inline u64 plan_loc(Plan &p, const std::string &where, const GroupSize &callers_g, const std::vector<TreeSize> &callers_trees) {
  const GroupSize g = callers_g;  // READ ONCE (plan.h)
  const std::vector<TreeSize> trees = callers_trees;
  const Analysis a = analysis(p, where, g);
  const u64 t = plan_positions(p, a.at.agree);
  if (!t) p.refuse(where + ": no number of positions reaches the target");
  PlanRound r;
  r.draw_positions = t, r.log_n = g.main.log_n, r.group = priced_codes(g);
  MayLay::close(p, R_LOC, where, RoundKind::POSITIONS, ErrorKind::QUERY, Bound::pow(a.at.agree, t), Priced::AS_SAID, r);
  for (auto &tr : trees) plan_open(p, where, tr, t);
  return t;
}
/* CS, inside a proximity test: the new oracle message and its sample; then, in one verifier's
   message, the positions of the old group and the challenge that batches old claims, sample claims
   and generator rows. A generator row claims that one opened symbol is the old code's linear form
   of the new message; there are `symbols` at each position. Returns the number of positions. */
// The errors follow the lemma on the round-by-round knowledge soundness of the zero-knowledge
// code-switching IOR of the paper that this protocol builds on (see THIRD_PARTY.md).
inline u64 plan_cs(Plan &p, const std::string &where, const GroupSize &callers_from, const GroupSize &callers_to, unsigned cube_bits, u64 old_rows,
                   unsigned symbols, const std::vector<TreeSize> &callers_trees, unsigned roots) {
  const GroupSize from = callers_from, to = callers_to;  // READ ONCE (plan.h)
  const std::vector<TreeSize> trees = callers_trees;
  const Analysis a = analysis(p, where, from), next_stage = analysis(p, where + ": the stage that follows", to);
  plan_send(p, where, {SampleSize{cube_bits, {to}}}, roots, false);
  /* The round's error is agree^t plus a batching term that grows with t. So t is iterated from 0:
     each step takes the smallest t that suffices with the batching term of the step before. The
     sequence never falls and never passes the smallest t that suffices, so it stops exactly
     there. */
  u64 t = 0;
  for (;;) {
    const u64 next = plan_positions(p, a.at.agree, Bound(next_stage.list) * powers_error(p.set, old_rows + p.set.samples + symbols * t));
    if (next == t || !next) {
      t = next;
      break;
    }
    t = next;
  }
  if (!t) p.refuse(where + ": no number of positions reaches the target");
  PlanRound r;
  r.draw_positions = t, r.log_n = from.main.log_n, r.draw_E = 1, r.rows = old_rows + p.set.samples + symbols * t, r.group = priced_codes(from);
  MayLay::close(p, R_CS, where, RoundKind::POSITIONS, ErrorKind::QUERY,
             Bound::pow(a.at.agree, t) + Bound(next_stage.list) * powers_error(p.set, old_rows + p.set.samples + symbols * t), Priced::ON_THE_CALLERS_WORD, r);  // the list size of the stage that follows
  for (auto &tr : trees) plan_open(p, where, tr, t);
  return t;
}
// BASE. g: one main row and, if it has a mask code, one mask row. roots: as in plan_fold(), for the
// tree of the masking codeword. Returns the number of positions.
inline u64 plan_base(Plan &p, const std::string &where, const GroupSize &callers_g, const std::vector<TreeSize> &callers_trees, unsigned roots) {
  const GroupSize g = callers_g;  // READ ONCE (plan.h)
  const std::vector<TreeSize> trees = callers_trees;
  const Setting &s = p.set;
  const Analysis a = analysis(p, where, g);
  plan_roots(p, where, roots);
  plan_scalars(p, "masked target", where, 1);
  {
    PlanRound r;
    r.draw_E = 1, r.rests = weakest_source(a), r.group = priced(g);
    MayLay::close(p, R_BASE_LAMBDA, where, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, base_error(s, g, a), Priced::AS_SAID, r);
  }
  plan_scalars(p, "masked message and randomness", where, g.main.coordinates() * g.slots + (g.has_mask() ? g.mask.coordinates() : 0));
  const u64 t = plan_positions(p, a.at.agree);
  if (!t) p.refuse(where + ": no number of positions reaches the target");
  PlanRound r;
  r.draw_positions = t, r.log_n = g.main.log_n, r.group = priced_codes(g);
  MayLay::close(p, R_BASE, where, RoundKind::POSITIONS, ErrorKind::QUERY, Bound::pow(a.at.agree, t), Priced::AS_SAID, r);
  for (auto &tr : trees) plan_open(p, where, tr, t);
  return t;
}

/* The stages of a proximity test. Stage 1 is the group under test. For each further stage the
   caller chooses the numbers of row variables k and h and the lengths of the two codes
   (StageChoice). The rest follows (primed: the new stage; _zk: the mask code):
     main message    ell' = slots ell / 2^k', which must be exact; no random coordinate goes into it
     mask message    2^h' ell_zk' = 2 ell_zk + slots r + 2 r_zk + r_ood: what the fold leaves of the
                     two mask rows (the old one, the fresh one) and of the random coordinates with
                     which the main row and these two were encoded, and r_ood >= samples entries of
                     which the last are the pad; ell_zk' is the smallest power of two with room,
                     and at least 2 k' + 1 - h' for the next fold's mask
     randomness      r' = r_zk' = the number of positions at which the stage is opened, which
                     depends on the rate, which depends on r': solved by iteration from below. */
// The sizes of the stages follow the zero-knowledge proximity test (IOPP) for constrained codes of
// the paper that this protocol builds on (see THIRD_PARTY.md).
struct StageChoice {
  unsigned k = 0, h = 0;
  unsigned log_n = 0, mask_log_n = 0;
};
struct Stage {
  GroupSize g;
  u64 t = 0;      // positions at which this stage's rows are opened
  u64 r_ood = 0;  // entries that fill the mask rows of this stage, the pad among them
};
inline u64 pow2_at_least(u64 x) { return (u64)1 << ceil_log2(x); }

inline unsigned stage_cube_bits(const GroupSize &g) {
  const unsigned a = g.k + ceil_log2(g.main.ell), b = g.has_mask() ? g.h + ceil_log2(g.mask.ell) : 0;
  return g.has_mask() ? std::max(a, b) + 1 : a;
}
/* first: stage 1, with its distance chosen; first_t: its number of positions. Adds nothing to p: it
   reads p's Setting and scope, and rejects
   p if a choice cannot be met; the result may then lack stages. A further stage's t leaves
   2^-(target + 30) (`margin`) for the batching term of CS's round; `plan_iopp` rejects the plan
   if that round needs more positions. */
inline std::vector<Stage> solve_stages(Plan &p, const GroupSize &first, u64 first_t, const std::vector<StageChoice> &choice) {
  const Setting &s = p.set;
  std::vector<Stage> st(1);
  st[0].g = first, st[0].t = first_t;
  if (choice.empty() || first.k == 0) p.refuse(first.name + ": a proximity test folds at its first stage and has a last stage");
  for (size_t i = 0; i < choice.size(); ++i) {
    const GroupSize &o = st.back().g;
    const StageChoice &c = choice[i];
    Stage n;
    n.g.name = first.name + ", stage " + std::to_string(i + 2);
    if (c.h > c.k || c.k > 32 || (i + 1 == choice.size() ? c.k != 0 : c.k == 0)) {
      p.refuse(n.g.name + ": a middle stage has 1 <= k and h <= k, the last stage k = h = 0");
      break;
    }
    n.g.k = c.k, n.g.h = c.h;
    n.g.main.name = n.g.name, n.g.main.log_n = c.log_n, n.g.main.over_E = true;
    n.g.mask.name = n.g.name + " mask", n.g.mask.log_n = c.mask_log_n, n.g.mask.over_E = true;
    if (o.main.ell * o.slots % ((u64)1 << c.k)) p.refuse(n.g.name + ": the folded row cannot be cut into 2^k rows");
    n.g.main.ell = o.main.ell * o.slots >> c.k;
    const u64 carried = 2 * o.mask.ell + o.main.r * o.slots + 2 * o.mask.r;
    const u64 own_fold = c.k ? 2 * c.k + 1 - c.h : 1;
    n.g.mask.ell = std::max(pow2_at_least((carried + s.samples + ((u64)1 << c.h) - 1) >> c.h), pow2_at_least(own_fold));
    n.r_ood = (n.g.mask.ell << c.h) - carried;
    u64 t = 0;
    for (int turn = 0;; ++turn) {
      if (turn == 32) {
        p.refuse(n.g.name + ": positions and random coordinates do not settle");
        break;
      }
      n.g.main.r = n.g.mask.r = tail_for(s, t);
      if (n.g.main.dim(s) > n.g.main.n() || n.g.mask.dim(s) > n.g.mask.n()) {
        p.refuse(n.g.name + ": message and randomness do not fit the code's length");
        break;
      }
      GroupLoad load;
      load.cube_bits = stage_cube_bits(n.g);
      load.messages = 1;
      load.batched_rows = 4 + 3 * std::max<u64>(o.main.r, 1024);  // assumed: CS into this stage batches three claims for each position of the stage before
      const Bound margin(Real::two_to(-(long)s.target - 30));
      const GroupSize shape = n.g;
      n.g = size_group(s, p.scope(), shape, load);
      const bool serves = group_holds(s, p.scope(), n.g, load);
      const Analysis a = analysis(s, n.g);
      const u64 t2 = serves ? plan_positions(p, a.at.agree, margin) : a.fault ? positions_from_below(s, p.scope(), shape, load, margin) : 0;
      if (!serves && t2 <= t) {  // `group_holds` fails, and no larger t is left to try
        p.refuse(n.g.name + ": " + (a.fault ? a.fault : "no distance under its rule holds the stage's rounds at the target"));
        break;
      }
      if (t2 <= t) break;
      t = t2;
    }
    n.t = t;
    st.push_back(n);
  }
  return st;
}

/* IOPP: a whole proximity test. rows: how many claims its BAT combines.
   first_trees: the trees that contain stage 1's rows. first_mask_roots: 1 if stage 1's FOLD sends
   its fresh mask, in a separate tree, which is then opened too; 0 if one of first_trees has it.
   masks_ride false: every later stage has two trees: that of CS's oracle message and that of the
   fresh mask of its FOLD (or of BASE's masking codeword). masks_ride true: those masks, which are
   made from randomness only, are sent in CS's prover message and share its tree's leaves; BASE's
   masking main row adds a symbol. */
inline void plan_iopp(Plan &p, const std::string &where, const std::vector<Stage> &st, const std::vector<TreeSize> &first_trees, u64 rows,
                      unsigned first_mask_roots, bool masks_ride = false) {
  const Setting &s = p.set;
  if (st.size() < 2) return p.refuse(where + ": a proximity test has at least two stages");
  /* The stages' random coordinates were sized for Stage::t positions. The numbers of positions are
     derived again here, each from its round's whole error; a larger one would open a row at more
     positions than the row hides. */
  auto within = [&](u64 t, const Stage &x) {
    if (t > x.t) p.refuse(where + ": a list is longer than the stage's random coordinates were sized for");
  };
  plan_bat(p, where, {st[0].g}, rows);
  std::vector<TreeSize> trees = first_trees;
  for (size_t i = 0; i + 1 < st.size(); ++i) {
    const GroupSize &g = st[i].g, &n = st[i + 1].g;
    const std::string w = where + ", stage " + std::to_string(i + 1);
    plan_fold(p, w, g, i ? (masks_ride ? 0 : 1) : first_mask_roots);
    if (g.has_mask() && (i ? !masks_ride : first_mask_roots != 0)) {
      TreeSize m;
      m.name = "fresh mask", m.log_n = g.mask.log_n, m.leaf_bytes = ((u64)s.e_bytes) << g.h, m.one_list = true;
      trees.push_back(m);
    }
    within(plan_cs(p, w, g, n, stage_cube_bits(n), 1, (unsigned)g.slots + (g.has_mask() ? 2 : 0), trees, 1), st[i]);
    TreeSize sw;
    sw.name = "switch oracle", sw.log_n = n.main.log_n;
    sw.leaf_bytes = (((u64)s.e_bytes) << n.k) + (((u64)s.e_bytes) << n.h) * (masks_ride ? 2 : 1);  // the stage's rows; copies of its mask rows and, with `masks_ride`, of its fold's fresh mask
    if (masks_ride && i + 2 == st.size()) sw.leaf_bytes += s.e_bytes;
    sw.one_list = masks_ride;
    trees.assign(1, sw);
  }
  const GroupSize &b = st.back().g;
  if (!masks_ride) {
    TreeSize m;
    m.name = "masking codeword", m.log_n = b.main.log_n, m.leaf_bytes = 2 * (u64)s.e_bytes, m.one_list = true;
    trees.push_back(m);
  }
  within(plan_base(p, where + ", last stage", b, trees, masks_ride ? 0 : 1), st.back());
}

// ---------------------------------------------------------------------------------------------------------------------
// The committed table: RECORD, LIN, LINK

/* The committed table: 2^v rows over F under one code. Its record: `answers` sample answers for
   each row, taken once, right after the table is committed. A one-time test at distance delta_M;
   every later test at a distance delta' <= delta_M, which may be beyond unique decoding.
   A record with more than one answer has an error bound here (a pair's error to the power
   `answers`), but this project's proofs cover a recorded table with one answer only, and the
   parameter file's line `answers` takes no other number. */
struct TableSize {
  CodeSize code;
  unsigned v = 0;           // 2^v rows
  unsigned answers = 1;     // answers for each row, each with a seed and pads of its own
  /* The caller's choice: the two distances, in positions or each by an integer m
     as under a group's rule JOHNSON (`TableDistances`: plan.h). `analysis` derives the rest from
     them, the fields above and the setting. */
  TableDistances distances;
};
struct TableAnalysis {  // a result, like Analysis: no struct stores it
  Proximity once;   // the code at delta_M
  Proximity later;  // the code at delta'
};
inline TableAnalysis analysis(const Setting &s, const TableSize &T) {
  TableAnalysis a;
  if (T.distances.m_once) {
    a.once = johnson(s, T.code, T.distances.m_once), a.later = johnson(s, T.code, T.distances.m_later);
    return a;
  }
  const Bound share = Bound::ratio(Big(T.distances.radius), Big(T.code.n()));
  a.once = at_distance(s, T.code, Bound(1) - share);
  a.later = at_distance(s, T.code, Bound(1) - Bound::ratio(Big(T.distances.later_radius), Big(T.code.n())));
  return a;
}
// What a plan records of the table at the rounds that fold its rows or draw its columns. It names
// all fields, as priced(GroupSize) does and for the same reason.
inline PricedTable priced(const TableSize &T, TableTest test) {
  const auto &[code, v, answers, distances] = T;
  (void)code, (void)v, (void)answers, (void)distances;  // likewise
  PricedTable x;
  x.code = priced(T.code), x.v = T.v, x.answers = T.answers, x.distances = T.distances, x.test = test, x.said = true;
  return x;
}
/* A condition on the distances, without which the formulas below do not hold: the error of the
   record's round uses the list size at delta_M, so no later test may go beyond it.
   It is not the only one. A further condition cannot be tested here: the bounds of a later test and of
   LINK are true only of a table that has passed its one-time test at these same distances, and
   that test is in another plan (`TableTest`, `Priced::ON_THE_CALLERS_WORD`:
   blocks/numbers/plan.h). At a run, the verifier compares the run's distances with those in the
   table's `IndexReader` and ends the process if they differ (`foldeq_columns_verify`,
   blocks/iop/foldeq.h). The reader's are a passed test's only if that test ran in the same
   process (`TestedTable`, `IndexShape::distances` there). */
inline bool table_admitted(const TableSize &T, const TableAnalysis &a) {
  if (a.once.bad()) return false;
  if (a.later.bad() || !T.answers) return false;
  return a.once.from_m && a.later.from_m ? a.later.m <= a.once.m : surely_le(a.later.delta, a.once.delta);
}
inline bool table_admitted(const Setting &s, const TableSize &T) { return table_admitted(T, analysis(s, T)); }
// The table's analysis; the plan is rejected unless table_admitted().
inline TableAnalysis analysis(Plan &p, const std::string &where, const TableSize &T) {
  const TableAnalysis a = analysis(p.set, T);
  if (!table_admitted(T, a)) p.refuse(where + ": the table's distances are not admitted");
  return a;
}
// The error of a fold of all 2^v rows by equality weights in one message: a proximity term for
// each variable, and (list size) v / |E|.
inline Bound table_fold_error(const Setting &s, const TableSize &T, const Proximity &at) {
  Bound e((u64)0);
  for (unsigned j = 1; j <= T.v; ++j) e = e + mca_error(s, at);
  return e + Bound(at.list) * field_error(s, T.v);
}
/* Derives the table's two distances, which may go up to the Johnson radius. Each comes from the
   largest m, found as in `size_group`, for which the fold point's round meets the target less its
   proof of work (with the proximity terms of two tables, as in a LINK proof). The fold point is the
   point at which the table's rows are folded (R_LIN_POINT, R_LINK_POINT). At the one-time distance
   the record's error must also be at most the bare target, 2^-target: the record's round has no
   proof of work, and the finished plan compares it with the bare target, whatever bits of proof of
   work the setting has for SAMPLE and ONCE. The later m is capped at the one-time m, so that a
   later test's distance is at most the one-time test's. requests: how many share a batch's point,
   which has point_variables coordinates. Returns false if no distance suffices. */
inline bool size_recorded_table(const Setting &s, TableSize &T, u64 requests, unsigned point_variables) {
  auto sized = [&](Scope scope, u64 sharing, bool with_record) {
    return largest_m([&](u64 m) {
      const Proximity x = johnson(s, T.code, m);
      if (x.bad()) return false;
      const Bound point = Bound(Big(sharing)) * field_error(s, point_variables) + Bound(2) * table_fold_error(s, T, x);
      return point.error() <= goal_of(s, ErrorKind::CLOSENESS, scope) &&
             (!with_record || sample_error(s, x.list, ceil_log2(T.code.ell), T.answers).error() <= Real::two_to(-(long)s.target));  // the target itself: the record's round has no proof of work (plan_record())
    });
  };
  const u64 m_once = sized(Scope::ONCE, 1, true), m_later = std::min(m_once, sized(Scope::SHARED, requests, false));
  if (!m_later) return false;
  T.distances.m_once = m_once, T.distances.m_later = m_later;
  return true;
}
/* RECORD: the table's record. Its seeds are a function of the table's name (root, configuration
   digest, shape: `index_name`), computed by `record_seeds` (both in blocks/iop/foldeq.h). They
   are drawn on a separate chain, without salt or proof of work: a name has one set of seeds, and
   so one record. With the root, the answers are in
   the first query (the hash input that starts a chain) of every later chain on the table. The
   round's error uses the list size at the largest distance at which the table is ever read
   (`TableAnalysis::once`): meeting the target there, without proof of work, bounds from below the
   slack under the Johnson radius. */
inline void plan_record(Plan &p, const std::string &where, const TableSize &callers_T) {
  const TableSize T = callers_T;  // READ ONCE (plan.h)
  const TableAnalysis A = analysis(p, where, T);
  if (!T.answers) return;  // the plan was rejected above; there is no message to plan
  const unsigned cube = ceil_log2(T.code.ell);
  PlanRound r;
  r.name = R_RECORD, r.where = where, r.kind = RoundKind::RECORD, r.why = ErrorKind::SAMPLE, r.scope = p.scope(), r.count = p.times();
  r.error = sample_error(p.set, A.once.list, cube, T.answers).error(), r.priced = Priced::ON_THE_CALLERS_WORD;  // no channel closes this round, so no block can check the plan's numbers there
  r.draw_E = (u64)T.answers * (cube + 1), r.draw_nonzero = T.answers;
  MayLay::enter(p, r);
  const u64 answers = ((u64)T.answers << T.v) * p.set.e_bytes;  // the seeds are in no file: they are recomputed from the table's name
  MayLay::file(p, "record", where, MessageKind::ANSWERS, answers, (double)answers, 0);
}
/* LIN (a linear claim of each request of a batch on the committed table) before its proximity
   tests. The programs run it on three hash chains, so that a request can be checked without the
   other requests' data:
     first    on the request's first chain: its oracle message W (the claim's row weights times the
              table's matrix) and W's sample (`plan_lin_first`), then the chain's seal;
     shared   on the batch's shared chain, which takes no request's claim: the shared point; the
              folded row re-encoded, its sample, the table's columns; Omega, the matrix times the
              equality vector of the shared point, its sample, the point at which the two are
              compared (`plan_lin_shared`; it returns the number of columns). This chain's first
              query has the root of a Merkle tree over the first chains' seals (`lin_bind_shared`,
              blocks/iop/lin.h), so that every W is fixed before the shared point is drawn: the
              bound on that point's round needs this order;
     then     the proximity tests, there and on the request's second chain, by `plan_split` and
              `plan_iopp`.
   w, u, omega: the sample sizes of W, the re-encoded folded row and Omega. instances: how many
   requests share the point. k_lo: the low variables of the shared point. rows_log: the variables of
   the point of comparison. one_time: the committer's own test, at delta_M. */
struct LinSize {
  SampleSize w, u, omega;  // set at construction: SampleSize has no default constructor
  u64 instances = 1;
  unsigned k_lo = 0, rows_log = 0;
  std::vector<SampleSize> beside_w = {}, beside_omega = {};  // further oracle messages in W's and in Omega's prover message; they share its seed
  bool one_time = false;
  unsigned roots_w = 1, roots_u = 1, roots_omega = 1;
};
// The query that ends a chain. Nothing of it is written into a proof.
inline void plan_seal(Plan &p, const std::string &where) {
  MayLay::close(p, R_SEAL, where, RoundKind::SEAL, ErrorKind::ZERO, Bound((u64)0), Priced::NOTHING);
}
inline std::vector<SampleSize> with_riders(const SampleSize &first, const std::vector<SampleSize> &beside) {
  std::vector<SampleSize> v(1, first);
  v.insert(v.end(), beside.begin(), beside.end());
  return v;
}
inline void plan_lin_first(Plan &p, const std::string &where, const LinSize &x) {
  plan_send(p, where + ", W", with_riders(x.w, x.beside_w), x.roots_w);
}
inline u64 plan_lin_shared(Plan &p, const std::string &where, const TableSize &callers_T, const LinSize &callers_x, const TreeSize &callers_tree) {
  const TableSize T = callers_T;  // READ ONCE (plan.h)
  const LinSize x = callers_x;
  const TreeSize tree = callers_tree;
  const Setting &s = p.set;
  const TableAnalysis A = analysis(p, where, T);
  const Proximity &at = x.one_time ? A.once : A.later;
  const Bound point = Bound(Big(x.instances)) * field_error(s, T.v + x.k_lo) + table_fold_error(s, T, at);
  {
    PlanRound r;
    r.draw_E = T.v + x.k_lo, r.rests = at.source, r.table = priced(T, x.one_time ? TableTest::ONE_TIME : TableTest::LATER), r.sharing = x.instances;
    MayLay::close(p, R_LIN_POINT, where, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, point, Priced::ON_THE_CALLERS_WORD, r);  // the table's distances
  }
  plan_send(p, where + ", folded row", {x.u}, x.roots_u);
  const Bound agree = at.agree;
  const u64 nu = plan_positions(p, agree);
  if (!nu) p.refuse(where + ": no number of columns reaches the target");
  PlanRound r;
  r.draw_positions = nu, r.log_n = T.code.log_n, r.table = priced(T, x.one_time ? TableTest::ONE_TIME : TableTest::LATER);
  MayLay::close(p, R_LOC, where + ", columns", RoundKind::POSITIONS, ErrorKind::QUERY, Bound::pow(agree, nu), Priced::ON_THE_CALLERS_WORD, r);  // the table's distances
  plan_open(p, where, tree, nu);
  plan_send(p, where + ", Omega", with_riders(x.omega, x.beside_omega), x.roots_omega);
  MayLay::close(p, R_LIN_ZETA, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, field_error(s, x.rows_log), Priced::THE_ROUND_ITSELF, draws(x.rows_log));
  return nu;
}
/* LINK ties a leaf table, a further committed table that is to have the same entries, to the
   anchor, the first one (`further` and `first` in names and output). Before its proximity test:
   the fold point; the two re-encoded folded rows and their samples; the columns of both tables,
   drawn in one message; the point of comparison. That message's round has the larger of its two
   lists' errors: which of the two folded rows is false is fixed before the draw, and that row's
   own list must miss the fault. */
inline void plan_link(Plan &p, const std::string &where, const TableSize &callers_T, const SampleSize &callers_u, const TreeSize &callers_tree, u64 &callers_nu_first, u64 &callers_nu_further) {
  const TableSize T = callers_T;  // READ ONCE (plan.h)
  const SampleSize u = callers_u;
  const TreeSize tree = callers_tree;
  const Setting &s = p.set;
  const TableAnalysis A = analysis(p, where, T);
  const Proximity &first = A.later;
  const Bound e = field_error(s, T.v) + table_fold_error(s, T, A.once) + table_fold_error(s, T, first);
  {
    PlanRound r;
    r.draw_E = T.v, r.rests = (int)first.source > (int)A.once.source ? first.source : A.once.source, r.table = priced(T, TableTest::LINK);
    MayLay::close(p, R_LINK_POINT, where, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, e, Priced::ON_THE_CALLERS_WORD, r);  // the table's distances
  }
  plan_roots(p, where, 1);
  {
    PlanRound r;
    r.draw_E = (u64)s.samples * (u.cube_bits + 1), r.draw_nonzero = s.samples;
    MayLay::close(p, R_SEND_SEED, where, RoundKind::SEED, ErrorKind::SAMPLE, Bound(2) * sample_error(s, list_of(p, where, u.groups), u.cube_bits), Priced::ON_THE_CALLERS_WORD, r);  // the group whose list size counts
  }
  MayLay::answers(p, where, 2 * s.samples);
  plan_none(p, R_SEND_ANSWER, where);
  const Bound a0 = first.agree, aj = A.once.agree;
  const u64 nu_first = plan_positions(p, a0), nu_further = plan_positions(p, aj);  // the caller's two numbers are written at the end and never read
  if (!nu_first || !nu_further) p.refuse(where + ": no number of columns reaches the target");
  const Bound e0 = Bound::pow(a0, nu_first), ej = Bound::pow(aj, nu_further);
  PlanRound r;
  r.draw_positions = nu_first + nu_further, r.log_n = T.code.log_n, r.first_list = nu_first, r.table = priced(T, TableTest::LINK);
  MayLay::close(p, R_LINK, where, RoundKind::POSITIONS, ErrorKind::QUERY, e0.hi < ej.hi ? ej : e0, Priced::ON_THE_CALLERS_WORD, r);  // the table's distances
  plan_open(p, where + ", first table", tree, nu_first);
  plan_open(p, where + ", further table", tree, nu_further);
  MayLay::close(p, R_LINK_PHI, where, RoundKind::ALGEBRAIC, ErrorKind::FIELD, field_error(s, ceil_log2(T.code.ell)), Priced::THE_ROUND_ITSELF, draws(ceil_log2(T.code.ell)));
  callers_nu_first = nu_first, callers_nu_further = nu_further;
}

}  // namespace blk
