// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The plan of one protocol run: every round with its error bound (`priced` in names and messages)
   and proof of work, and every message with its size, in protocol order.

   A planner (a `plan_..` function of blocks/numbers/blocks.h) appends the rounds and messages of
   one block (a step of the proof system, such as SEND or FOLD) from the block's sizes. Prover and
   verifier then step through the list with a `Cursor`: the channel (blocks/chain/rounds.h) takes
   each round's proof of work from its entry and ends the process if the round that runs differs from the next
   planned one in name, kind or counts (bytes sent, values drawn). A planning call (such as the verifier program
   `cp_verify` with `-plan`) plans by the same routines and runs nothing. What it prints joins the chains into
   one plan and, without `-fields`, may list equal rounds once, with a count: its lines are not a run's rounds.

   Whether an error bound was computed from the numbers on which a block really runs is checked only
   where the block compares them (`PricedGroup`, `PricedTable`, `PlanRound`); `Priced` marks the
   rounds whose bound rests on a number that nothing checks. The types here guard against mistakes,
   not against a caller who sets out to defeat them ("Two types", below). */
#pragma once
#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "numbers/real.h"
#include "numbers/round_names.h"
#include "numbers/setting.h"

namespace blk {

/* Gives the kind that belongs to a round's name. A hash chain is the Fiat-Shamir transcript of one
   protocol run (blocks/chain/chain.h); its first query is the hash input that fixes, before the
   prover speaks, the public inputs and, of the plan, each round's name, counts and proof of work
   (through a digest of the parameters). It does not cover a round's kind, so this table is part of
   the proof format. Returns false for a name outside the format: tests of the channel may invent
   names, but `configure` (blocks/frame/configure.h) rejects plans that contain one. */
inline bool kind_of_round(const std::string &name, RoundKind &kind) {
  static const struct {
    const char *name;
    RoundKind kind;
  } all[] = {{R_SEND_SEED, RoundKind::SEED},        {R_SEND_ANSWER, RoundKind::EMPTY},    {R_BAT, RoundKind::ALGEBRAIC},         {R_BAT_SIGMA, RoundKind::ALGEBRAIC},
             {R_SPLIT, RoundKind::EMPTY},           {R_PSC_EPSILON, RoundKind::EPSILON},  {R_PSC_ROUND, RoundKind::ALGEBRAIC},   {R_PSC_Y, RoundKind::ALGEBRAIC},
             {R_PSC_VALUES, RoundKind::EMPTY},      {R_ZERO, RoundKind::ALGEBRAIC},       {R_LINE, RoundKind::ALGEBRAIC},        {R_LKP_SHIFT, RoundKind::ALGEBRAIC},
             {R_JOIN, RoundKind::ALGEBRAIC},        {R_LKP, RoundKind::ALGEBRAIC},        {R_RNG, RoundKind::BITS},              {R_FOLD_EPSILON, RoundKind::EPSILON},
             {R_FOLD_ROUND, RoundKind::ALGEBRAIC},  {R_LOC, RoundKind::POSITIONS},        {R_CS, RoundKind::POSITIONS},          {R_BASE_LAMBDA, RoundKind::ALGEBRAIC},
             {R_BASE, RoundKind::POSITIONS},        {R_SEAL, RoundKind::SEAL},            {R_LAST, RoundKind::EMPTY},            {R_REPLY, RoundKind::EMPTY},
             {R_RECORD, RoundKind::RECORD},         {R_LIN_POINT, RoundKind::ALGEBRAIC},  {R_LIN_ZETA, RoundKind::ALGEBRAIC},    {R_LINK_POINT, RoundKind::ALGEBRAIC},
             {R_LINK, RoundKind::POSITIONS},        {R_LINK_PHI, RoundKind::ALGEBRAIC}};
  for (const auto &x : all)
    if (name == x.name) return kind = x.kind, true;
  return false;
}

/* The public numbers of a group (main rows under one code, mask rows under another, tested
   together) from which a plan computes the error bounds of a stage of a proximity test or of the
   closing of a segment (one sub-proof's part of a request's proof). The plan takes them from a
   `GroupSize` (blocks.h); a run takes them from public inputs of its own, a list of codes and a
   shape. So that the two cannot differ, the plan states them (`said`, `says` in names and messages)
   with the round that begins a stage (`R_FOLD_EPSILON`, `R_BASE_LAMBDA`) and with each round of a
   fold, and states the codes alone with every round that draws positions on the group (`R_LOC`,
   `R_CS`, `R_BASE`). Prover and verifier compare there (`Rounds::require_group`,
   `Rounds::require_codes`): other numbers, or a plan that states none, end the process. The stated
   numbers are in no chain's first query; codes and shapes are.
   A new public number of a code (`CodeSize`, codes.h) or a group (`GroupSize`) that an error bound
   uses must be added here and in `priced` (blocks.h). `priced` names every field of the two
   structs, so a new field stops the build there. */
struct PricedCode {
  unsigned log_n = 0;  // the length is 2^log_n
  u64 ell = 0, r = 0;  // message entries, random entries
  bool operator==(const PricedCode &o) const { return log_n == o.log_n && ell == o.ell && r == o.r; }
};
struct PricedGroup {
  PricedCode main, mask;  // a group without mask rows: the mask code's three numbers are 0
  unsigned k = 0, h = 0;  // row variables of the main and of the mask rows
  u64 slots = 1;          // rows side by side
  bool said = false;      // false: no group is stated here; `require_group` tests this flag too
  /* The parameters of the soundness analysis at which the bound was computed: `rule`, the rule that
     sets the group's distance (`Radius`, codes.h), and `m`, the rule's integer. Under JOHNSON, `m`
     is the integer m of the theorem cited in setting.h: the larger m, the smaller the slack between
     the distance and the Johnson radius of the code with the larger rate. Under AT, `m` is the
     distance, in positions of the main code. A run has no counterpart
     of the two, so no block can compare them. By this library's reading, the analysis fixes one
     distance for a stage, so the bounds of all its rounds must be computed at that one. A proximity
     term (in a round's error, the chance that a combination of words is close to the code although
     the words are not jointly close: `ErrorKind::CLOSENESS`, setting.h) is small at a near distance
     and few positions are needed at a far one: each bound is true alone, but the two together are
     not. The plan's verdict tests this (`Cause::TWO_DISTANCES`). */
  unsigned rule = 0;
  u64 m = 0;
  bool operator==(const PricedGroup &o) const { return main == o.main && mask == o.mask && k == o.k && h == o.h && slots == o.slots; }  // all but `said`, `rule` and `m`: what a run can be compared with
};

/* The distances at which a committed table (2^v encoded rows under a Merkle tree over their
   columns: `IndexShape`, blocks/iop/foldeq.h) is tested (see `TableTest`); like `rule` and `m`,
   parameters of the analysis. `radius` is delta_M, the one-time test's; `later_radius` is delta', a
   later test's. Both are public parameters, counted in positions out of the code's length. If
   `m_once` is not 0, `m_once` and `m_later` replace them, in this order: integers m as under the
   rule JOHNSON (`PricedGroup::m`), derived by `size_recorded_table` (blocks/numbers/blocks.h).
   See `IndexShape::distances` for what is checked of them and what is not. */
struct TableDistances {
  u64 radius = 0, later_radius = 0;
  u64 m_once = 0, m_later = 0;
  bool operator==(const TableDistances &o) const { return radius == o.radius && later_radius == o.later_radius && m_once == o.m_once && m_later == o.m_later; }
};
static_assert(sizeof(TableDistances) == 4 * sizeof(u64), "a further distance is also compared above and printed below");
inline std::string text_of(const TableDistances &d) {  // for stderr, before the process ends
  return "radius " + std::to_string(d.radius) + ", later radius " + std::to_string(d.later_radius) + ", m " + std::to_string(d.m_once) + ", later m " + std::to_string(d.m_later);
}
/* `PricedTable`: as `PricedGroup`, for the committed table (`TableSize`, blocks.h). The
   plan states it with the rounds that fold the table's rows or draw its columns: the point and
   columns of LIN (the test on the table), the point and two lists of LINK (which ties two tables).
   Prover and verifier compare there (`Rounds::require_table`). The distances are compared, but what
   is hashed of the table (`public_bytes`, `index_name`, blocks/iop/foldeq.h) leaves them out, so
   these rounds stay ON_THE_CALLERS_WORD (see `Priced`). */
/* Which test on the table a round belongs to; the bound for a drawn column depends on it.
     ONE_TIME  the committer's own test, which is to show that the committed word is within the
               one-time distance of the code; a column counts at that distance.
     LATER     a batch's test. Its bound is true only of a table that has passed the one-time test
               at the same distances, and that test is in another file's plan.
     LINK      treats the anchor (a commitment's first table) as a later test does and the leaf
               table (a further one) as the one-time test does.
   No shape says which test runs, and ONE_TIME and LATER run the same code: the caller names the
   test in an argument without a default (blocks/iop/lin.h), and it is compared with the plan's. */
enum class TableTest { LATER, ONE_TIME, LINK };
struct PricedTable {
  PricedCode code;
  unsigned v = 0;        // 2^v rows
  unsigned answers = 0;  // of its record, for each row
  TableDistances distances;
  bool said = false;     // as PricedGroup's
  TableTest test = TableTest::LATER;
  bool operator==(const PricedTable &o) const { return code == o.code && v == o.v && answers == o.answers && distances == o.distances && test == o.test; }  // all but said
};

/* What a round's error bound was computed from. `Plan::close` takes the value without a default,
   and the plan's verdict tests it against the round (`word_fault`).
     NOTHING              No formula: the bound is 0. The round may still state numbers, which the
                          blocks' comparisons need (hand-made plans in tests).
     SETTING_ALONE        The setting only: 1 / |E|, where E is the extension field from which
                          challenges are drawn (F is its base field).
     THE_ROUND_ITSELF     The setting and the round's own fields, which enter the chain's first
                          query and which the channel enforces: the elements drawn; the bytes of the
                          prover's message, which follow from a degree. Such a round, like one of
                          SETTING_ALONE, must state nothing.
     AS_SAID              What the plan states with the round (`says`), which a block compares at
                          the run, and for a group `rule` and `m`, which the verdict tests. Nothing
                          else.
     ON_THE_CALLERS_WORD  A number that a caller passed in and that nothing checks at this round,
                          whatever else the round states:
                          - the groups whose list sizes count at an out-of-domain sample or a
                            batching;
                          - the numbers of variables of the smaller oracle messages under one seed;
                          - a lookup's table and keys;
                          - the list size of the following stage, in the batching term of a code
                            switch (`R_CS`);
                          - the table's distances, at every round computed from them (LIN's point
                            and columns, LINK's point and two lists, a record): the bounds of a
                            later test and of LINK are true only of a table whose one-time test ran
                            at the same distances, and that test is in another file's plan
                            (`IndexShape::distances`, blocks/iop/foldeq.h).
                          Each such round is an open item; the value is there so that they can be
                          found and counted (`account`, blocks/numbers/accounting.h). */
enum class Priced { NOTHING, SETTING_ALONE, THE_ROUND_ITSELF, AS_SAID, ON_THE_CALLERS_WORD };

/* One round: a prover's message, then a verifier's message; either may be empty. Every round, even
   one of kind EMPTY, makes one query. It is a query of the hash chain, which absorbs what the
   prover has sent since the previous round; only a round of kind RECORD makes its query on a
   separate chain (`record_seeds`, blocks/chain/rounds.h).
   `why` (the argument that bounds the error) and `scope` (who pays for the proof of work, and how
   often) select `work_bits`; blocks/numbers/setting.h explains their values and those of `kind`.
   At the run a block compares what the plan states with a round (`rows`, `first_list`, `group`,
   `table`, `combinations`, `sharing`) with its own numbers, through the `Rounds::require_..`
   function of that name (for `group`, also `require_codes`). If no block does, the channel ends
   the process, except for the prover's `rows` in rounds that ran outside the channel
   (`ProverRounds::take_back`). Of these fields, only `rows` enters a chain's first query. */
// An error of its own for each round is Definition 3.11 of: Block, Garreta, Tiwari, Zając,
// "On Soundness Notions for Interactive Oracle Proofs", Cryptology ePrint Archive 2023/1256
// (version of March 5, 2024); by its footnote 1 implicit in earlier work.
struct PlanRound {
  std::string name;   // the block's name for this round, one of the constants of round_names.h; the Cursor checks it
  std::string where;  // which use of the block, for the print
  RoundKind kind = RoundKind::EMPTY;
  ErrorKind why = ErrorKind::ZERO;
  Scope scope = Scope::REQUEST;
  Real error;              // an upper bound on the round's error, work not counted
  CountSource rests = CountSource::JOHNSON;  // of a round with a proximity term: the weakest source of a count in it; the weakest of all unless its planner sets a better one
  unsigned work_bits = 0;  // bits of proof of work before the verifier's message; 0: no nonce in the proof
  u64 count = 1;           // the number of separate rounds like this one
  unsigned part = 0;       // index into Planned::Data::parts
  /* The prover's message, as the hash chain sees it: all bytes since the last round that enter the
     chain (answers, scalars, roots). Nonces, opened columns, salts and paths are not among them. */
  size_t message_bytes = 0;
  unsigned roots = 0;    // how many roots are in it
  unsigned answers = 0;  // how many sample answers (elements of E) are in it; they come first in it
  // The verifier's message.
  u64 draw_E = 0;          // elements of E
  u64 draw_nonzero = 0;    // of these: how many are drawn from E without 0
  u64 draw_positions = 0;  // positions, repetitions counted
  unsigned log_n = 0;      // they are drawn from [2^log_n]
  u64 draw_trits = 0;      // entries in {-1, 0, 1}
  u64 rows = 0;            // of a round that batches rows: how many, as its error counts them
  /* Of a message of positions that is cut into two lists, each with its own error bound (only
     LINK's is): the draws of the first list; the second has the rest. Compared by
     `Rounds::require_first_list`. 0: the plan states none (a list has at least one draw). It is in
     no chain's first query; the sum, `draw_positions`, is. */
  u64 first_list = 0;
  PricedGroup group;  // see above
  PricedTable table;
  u64 combinations = 0;  // of a round that draws a matrix of trits: how many combinations of each column, as its error counts them. 0: the plan states none. It is in no chain's first query; the product, draw_trits, is
  u64 sharing = 0;  // of a round whose challenge many requests share, each with a claim of its own: how many, as its error counts them (LIN's point). 0: the plan states none. It is in no chain's first query
  Priced priced = Priced::NOTHING;  // see above
  bool says() const { return rows || first_list || group.said || table.said || combinations || sharing; }
  // Returns null, or how `priced` contradicts the rest of the round.
  const char *word_fault() const {
    if ((draw_positions || draw_trits) && error.zero() && priced != Priced::ON_THE_CALLERS_WORD) return "a round that draws positions or a matrix is priced at 0, which only a round marked ON_THE_CALLERS_WORD may be";  // such a draw always has an error (no planner gives it 0; the tests' hand-made plans do, under the value that is counted)
    if (priced == Priced::NOTHING && !error.zero()) return "a round with an error is said to be priced from nothing";  // (the other way round a formula may give 0: one row to batch, a degree of 1)
    if ((priced == Priced::SETTING_ALONE || priced == Priced::THE_ROUND_ITSELF) && says()) return "a round that is priced from the setting or from its own fields says something beside them";
    if (priced == Priced::AS_SAID && !says()) return "a round that is priced from what the plan says beside it says nothing";
    return nullptr;
  }
};

// The most 64-bit words of the chain's answer that the round's verifier message can use.
// degree: of E over F.
inline u64 most_words(const PlanRound &r, unsigned degree = 3) {
  return DRAW_TRIES * degree * (r.draw_E + (NONZERO_TRIES - 1) * r.draw_nonzero) + r.draw_positions + (r.draw_trits + 31) / 32 + (r.kind == RoundKind::SEAL ? 4 : 0);
}

enum class MessageKind { SCALARS, ANSWERS, ROOT, NONCE, COLUMNS, SALTS, PATHS, HEAD };

// One run of bytes in a planned file: its size and what checking it costs the verifier.
// `where`, `count` and `part` are as in `PlanRound`.
struct PlanMessage {
  std::string name;
  std::string where;
  MessageKind kind = MessageKind::SCALARS;
  u64 bytes = 0;        // of one copy; for answers to a draw of positions the most that any list can need
  double expected = 0;  // of one copy, on average over the positions (equal positions are answered once, paths are pruned)
  u64 count = 1;
  unsigned part = 0;
  bool in_chain = false;  // absorbed by the hash chain before the next draw
  u64 hash_calls = 0;     // calls of the compression function by which the verifier checks one copy, at most
};

// The size of one part in bytes: `bytes` at most, `expected` on average (see `PlanMessage`).
struct PartSize {
  u64 bytes = 0;
  double expected = 0;
};

/* Two types. Planners write into a `Plan`; `Plan::finish` makes a read-only `Planned` of it;
   whatever runs, checks or prints a plan takes a `Planned`. The types guard against mistakes, not
   against a caller who sets out to defeat them: what binds a malicious prover is what `cp_verify`
   derives from the public files.

   In a source that uses these headers as they are meant, the compiler enforces that only the
   planners in the first friend list of `MayLay` (blocks/numbers/blocks.h) add a round with a
   verifier's message and compute its bound, that a finished plan does not change, and that rounds,
   setting and verdict stay together. (Through the second list anyone can add a prover's message, a
   round with an empty verifier's message or, by `plan_chain`, the rounds of a finished plan.)
   Tested by blocks/numbers/tests/plan_types_test.cpp and by sources in
   blocks/tests/must_not_build/, which the compiler must reject.

   In its own process a caller can get round this, and no C++ type stops it.
   Known ways:
   - a macro defined before these headers, or access checks switched off;
   - a cast or its equivalent on an object of these types (a union, memcpy, placement new, a
     destructor called by name);
   - a dangling reference: one kept beyond its object's lifetime, such as one into a std::variant
     whose alternative is then replaced. The C++ standard leaves its use undefined;
   - an explicit instantiation or specialisation of any template, the caller's own too, over a
     private name (template struct X<&Plan::enter>: access is not checked there);
   - a declaration in blk, std or at global scope before these headers: that overload is preferred
     inside a planner. prover/ and verifier/ declare names in namespace blk themselves; only code
     review guards that;
   - BLK_PLANS_BY_HAND_FOR_A_TEST (see `Plan`). It does not compile in a unit built with
     SERVING_BUILD, as `cp_verify` and the production build are. The plain targets `cp_prove` and
     `cp_commit` (the prover and committer programs) of prover/build.sh are not: there the script
     does not install a finished program that contains the marker string of blocks/fence.h;
   - code of the caller's that runs in the middle of a planner (a replaced global operator new, a
     second thread) and writes an object that the planner took by reference.
   Read once (`READ ONCE` in blocks/numbers/blocks.h): against the last way, because it is cheap,
   every listed planner copies each of its sizes once (not its label `where`, which enters no
   bound), so that a round's bound and what is stated with it come from one object, and a round
   enters the plan complete. A copy that allocates memory is not made in one step: a list is copied
   entry by entry, a struct member by member. So the caller's code can change the original in the
   middle of the copy, which then has the earlier parts unchanged and the later ones changed. That
   is an object that the caller could have passed in, and the bound and what is stated with it still
   come from the copy alone (docs/limits-all-rows.tsv, row T2.2).
   The copying does not cover the functions that run a proof, which take shapes, keys and readers by
   reference; the unlisted planners (`plan_iopp`, a `.._plan` function of blocks/frame/), which
   pass one object of their caller's to several listed ones; or the plan's scope, which is read
   separately for the factor `Setting::instances` in a round's bound, for its `scope` and for its
   `work_bits`. These are examples, not a complete list.
   blocks/numbers/tests/planners_read_once_test.cpp tests the copying in part;
   of the planners' 22 copies it cannot notice four, which its head names.

   Every caller chooses the setting, the planners' arguments and order, the part, scope and repeat,
   `compact`, `refuse` and when to `finish`; it can write on a copy of a `Plan`, and `plan_chain`
   puts a finished plan's rounds before, after or under a repeat of other rounds. These choices are
   checked only where a block compares at the run, and by the frames (the headers of blocks/frame/),
   which derive all of them from the public files except what a sub-proof's planner adds, which the
   program passes in. */

// Why a plan does not hold; only the first cause is kept. REFUSED is tested first; then, round by
// round, ABOVE_THE_TARGET before WORD; TWO_DISTANCES last.
enum class Cause { NONE, REFUSED, ABOVE_THE_TARGET, WORD, TWO_DISTANCES };

class Plan;

/* A handle to a finished plan's data: `q->set`, `q->rounds_`, `q->holds`. A source that uses
   these headers as they are meant can neither write nor destroy the data. Whatever outlives a call
   keeps its own copy of the handle (`Cursor`, `Binder` of blocks/chain/public.h, the channels), so
   assigning to the caller's variable does not change the plan that a channel steps through. A move
   is a copy, so no handle is empty. A default-constructed `Planned` (such as a struct's member
   before its planner has run) refers to a fixed, rejected plan (`refused` in names and messages)
   without rounds. A rejected plan keeps its rounds and messages: blocks/frame/hiding_conditions.h
   finishes a plan of its own only to read its messages. */
class Planned final {
 public:
  class Data final {
   public:
    const Setting set;
    const std::vector<PlanRound> rounds_;
    const std::vector<PlanMessage> messages_;
    const std::vector<std::string> parts;

    /* The verdict is stored: the constructor computes it from the members above, and neither can be
       written afterwards. */
    bool holds() const { return verdict_.first == Cause::NONE; }
    bool refused() const { return verdict_.first == Cause::REFUSED; }  // by refuse() (public: planners, Plan's own functions, any caller), by finish(), or nothing was planned
    Cause cause() const { return verdict_.first; }
    const std::string &why_not() const { return verdict_.second; }  // the first cause, in words for a log; empty: the plan holds

    // True if every round has one of the format's names (and so that name's kind:
    // Plan::require_kind).
    bool names_of_the_format() const {
      RoundKind kind;
      for (auto &r : rounds_)
        if (!kind_of_round(r.name, kind)) return false;
      return true;
    }
    u64 rounds() const {  // (this sum and size_of()'s are not guarded against a wrap: they are for prints)
      u64 k = 0;
      for (auto &r : rounds_) k += r.count;
      return k;
    }
    u64 queries() const { return rounds(); }  // every round makes one; a record's seed is a query too, of its own
    PartSize size_of(unsigned part) const {
      PartSize s;
      for (auto &m : messages_)
        if (m.part == part) s.bytes += m.bytes * m.count, s.expected += m.expected * (double)m.count;
      return s;
    }
    int part_index(const std::string &name) const {
      for (size_t i = 0; i < parts.size(); ++i)
        if (parts[i] == name) return (int)i;
      return -1;
    }
    /* An upper bound on the chance that the draw of one element of E reaches its cap. The chain
       makes an element of F from the next word of its answer that is below p, and takes 0 after
       DRAW_TRIES = 6 words that are not; a non-zero element of E is the first of NONZERO_TRIES = 2
       elements that is not 0, else 1 (blocks/base.h, blocks/chain/chain.h). So a verifier's message
       is a function of a fixed number of words (`most_words`), and each coordinate is uniform but
       for the chance that DRAW_TRIES words in a row are not below p. */
    Real capped() const {
      size_t bits = 0;
      while (Big::cmp(Big::two_to(bits), set.p) < 0) ++bits;  // a word's bits that are looked at
      return (Bound((u64)set.degree) * Bound::pow(Bound::ratio(Big::two_to(bits) - set.p, Big::two_to(bits)), DRAW_TRIES)).error();
    }
    // The round's error with the distance of its draws from uniform. unit: capped().
    Real whole_error(const PlanRound &r, const Real &unit) const {
      if (!r.draw_E) return r.error;
      Real e = Real::add(r.error, Real::mul(unit, Real(r.draw_E + (NONZERO_TRIES - 1) * r.draw_nonzero), Dir::UP), Dir::UP);
      if (r.draw_nonzero)  // the chance that every candidate is 0, so that the element is 1
        e = Real::add(e, Real::mul((Bound(1) / Bound::pow(Bound(set.card_E()), NONZERO_TRIES)).error(), Real(r.draw_nonzero), Dir::UP), Dir::UP);
      return e;
    }
    // Does the round hold at the setting's target, work counted? An exact comparison.
    bool holds(const PlanRound &r, const Real &unit) const { return whole_error(r, unit) <= Real::two_to((long)r.work_bits - (long)set.target); }
    bool holds(const PlanRound &r) const { return holds(r, capped()); }

    Data(const Data &) = delete;

   private:
    friend class Plan;
    friend class Planned;
    ~Data() = default;
    Data(const Setting &s, const std::vector<PlanRound> &rounds, const std::vector<PlanMessage> &messages, const std::vector<std::string> &names, const std::string &refused)
        : set(s), rounds_(rounds), messages_(messages), parts(names), verdict_(judged(refused)) {}
    static std::string at(size_t i, const PlanRound &r) { return "round " + std::to_string(i + 1) + ", " + r.name + (r.where.empty() ? "" : ", " + r.where) + ": "; }
    std::pair<Cause, std::string> judged(const std::string &refused) const {
      if (!refused.empty()) return {Cause::REFUSED, refused};
      const Real unit = capped();
      for (size_t i = 0; i < rounds_.size(); ++i) {
        if (!holds(rounds_[i], unit)) return {Cause::ABOVE_THE_TARGET, at(i, rounds_[i]) + "the error is above the target, work counted"};
        if (const char *fault = rounds_[i].word_fault()) return {Cause::WORD, at(i, rounds_[i]) + fault};
      }
      /* All rounds that state the same two codes must state the same `rule` and `m` (see
         `PricedGroup`). The round that begins a stage, its fold's rounds and the round that draws
         its positions all state the stage's two codes, and a run is compared with these: so no
         stage has its proximity terms computed at one distance and its positions at another.
         Not covered: a number that a round takes from another stage, and the list sizes that a
         stage's own sample and batching use; such rounds are ON_THE_CALLERS_WORD.
         Too strict: two stages of one plan that share both codes must share `rule` and `m`, though
         they may validly differ (hence `account` ignores this cause). The right key would be the
         stage. */
      std::map<std::array<u64, 6>, std::pair<unsigned, u64>> chosen;
      for (size_t i = 0; i < rounds_.size(); ++i) {
        const PricedGroup &g = rounds_[i].group;
        if (!g.said) continue;
        const std::pair<unsigned, u64> choice(g.rule, g.m);
        if (chosen.emplace(std::array<u64, 6>{g.main.log_n, g.main.ell, g.main.r, g.mask.log_n, g.mask.ell, g.mask.r}, choice).first->second != choice)
          return {Cause::TWO_DISTANCES, at(i, rounds_[i]) + "priced at another distance than an earlier round on the same two codes"};
      }
      return {Cause::NONE, std::string()};
    }
    const std::pair<Cause, std::string> verdict_;
  };

  Planned() : data_(nothing()) {}
  Planned(const Planned &) = default;
  Planned &operator=(const Planned &) = default;
#ifdef BLK_PLANS_BY_HAND_FOR_A_TEST
  Planned(const Plan &p);  // = p.finish(): see class Plan
#endif
  const Data *operator->() const { return data_.get(); }
  const Data &operator*() const { return *data_; }

 private:
  friend class Plan;
  static void end(const Data *d) { delete d; }
  explicit Planned(const Data *made) : data_(made, end) {}
  static const std::shared_ptr<const Data> &nothing() {
    static const std::shared_ptr<const Data> none(new Data(Setting(), {}, {}, {}, "nothing was planned"), end);
    return none;
  }
  std::shared_ptr<const Data> data_;  // never null
};

class Plan final {
 public:
  explicit Plan(const Setting &s, const std::string &first_part = "proof") : set(s) { parts_.push_back(first_part); }
  const Setting set;     // the error bounds read it. Const: a Plan can be copied but not assigned
  bool compact = false;  // a planning call at full size: blocks may enter equal rounds once, with a count

  // The only way from a Plan to a Planned. Rounds and messages are copied: nothing that can still
  // write this Plan reaches the Planned.
  Planned finish() const {
    std::string refused = refused_;
    if (refused.empty() && (open_bytes_ || open_roots_ || open_answers_)) refused = "the plan ends in the middle of a prover's message: no round takes it in";
    if (refused.empty() && !repeat_.empty()) refused = "the plan ends inside a repeat";
    return Planned(new Planned::Data(set, rounds_, messages_, parts_, refused));
  }

  // Rejects the plan: for parameters outside a formula's range, and for whatever else a caller
  // refuses (here: a repeat of 0 or of 2^64 times, a root before an empty verifier's message, too
  // many proof-of-work bits, an appended plan that was rejected or has another setting). The first
  // reason is kept. A rejected plan does not hold, whatever its rounds are like.
  void refuse(const std::string &why) {
    if (refused_.empty()) refused_ = why.empty() ? "refused, and the caller gave no reason" : why;  // an empty reason rejects too
  }
  bool refused() const { return !refused_.empty(); }
  const std::string &why() const { return refused_; }

  // Part, scope and repeat count for what planners add next: a caller's choice. These add nothing.
  // begin_part(): what follows belongs to this file or part of one ("proof", "key", "entry", ..).
  // push_repeat(n) .. pop_repeat(): what is added between them is entered once, with count n (at
  // full size a request has some 10^5 rounds, most of them equal); nested repeats multiply. Such a
  // plan serves for errors and sizes; a Cursor ends the process on it.
  void begin_part(const std::string &name) {
    for (part_ = 0; part_ < parts_.size(); ++part_)
      if (parts_[part_] == name) return;
    parts_.push_back(name);
  }
  void begin_scope(Scope s) { scope_ = s; }
  Scope scope() const { return scope_; }
  void push_repeat(u64 n) {
    if (!n) refuse("a repeat of 0 times"), n = 1;
    if (times_ > ~(u64)0 / n) refuse("repeats of 2^64 times or more"), n = 1;
    repeat_.push_back(n);
    times_ *= n;
  }
  void pop_repeat() {
    BLK_REQUIRE(!repeat_.empty(), "Plan: pop_repeat without push_repeat");
    times_ /= repeat_.back();
    repeat_.pop_back();
  }
  u64 times() const { return times_; }

  static bool same_setting(const Setting &a, const Setting &b) {  // as many names as a Setting has fields, so that a new field stops the build here
    const auto &[field_name, p, degree, circle, f_bytes, e_bytes, digest_bytes, salt_bytes, nonce_bytes, position_bytes, hash_block_bytes, label_bytes, chain_extra_bytes, chain_free_words, target,
                 instances, samples, saved_coefficient, work] = a;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 3; ++j)
        if (work[i][j] != b.work[i][j]) return false;
    return field_name == b.field_name && Big::cmp(p, b.p) == 0 && degree == b.degree && circle == b.circle && f_bytes == b.f_bytes && e_bytes == b.e_bytes && digest_bytes == b.digest_bytes &&
           salt_bytes == b.salt_bytes && nonce_bytes == b.nonce_bytes && position_bytes == b.position_bytes && hash_block_bytes == b.hash_block_bytes && label_bytes == b.label_bytes &&
           chain_extra_bytes == b.chain_extra_bytes && chain_free_words == b.chain_free_words && target == b.target &&
           instances == b.instances && samples == b.samples && saved_coefficient == b.saved_coefficient;
  }

  /* For tests that build or alter a plan by hand. Under BLK_PLANS_BY_HAND_FOR_A_TEST all that
     follows is public, a `Plan` can be made from a finished plan (from copies: no test writes a
     `Planned` either), and a `Plan` converts to a `Planned`. A test defines the macro in its own
     source before its first include; no header defines it. It does not compile in a unit built with
     SERVING_BUILD (blocks/fence.h), as `cp_verify` and the production build are. The plain targets
     `cp_prove` and `cp_commit` of prover/build.sh are not: there the script does not install a
     finished program that contains the marker string of blocks/fence.h. */
#ifdef BLK_PLANS_BY_HAND_FOR_A_TEST
  explicit Plan(const Planned &again) : set(again->set), rounds_(again->rounds_), messages_(again->messages_), parts_(again->parts), refused_(again->refused() ? again->why_not() : std::string()) {}
#else
 private:
#endif
  friend class MayLay;  // numbers/blocks.h, which this header includes at its end: no source sees a Plan without the one definition of MayLay and of the planners that it names

  // The prover sends n elements of E, n sample answers, a root.
  void send_scalars(const std::string &name, u64 n) {
    if (n) chain_message(name, MessageKind::SCALARS, n * set.e_bytes);
  }
  void send_answers(const std::string &name, u64 n) {
    if (!n) return;
    chain_message(name, MessageKind::ANSWERS, n * set.e_bytes);
    open_answers_ += (unsigned)n;
  }
  void send_root(const std::string &name) {
    chain_message(name, MessageKind::ROOT, set.digest_bytes);
    ++open_roots_;
  }
  // Bytes of the file that do not enter the chain: what answers a draw, a nonce, a file's items.
  void file(const std::string &name, MessageKind kind, u64 bytes, double expected, u64 hash_calls) {
    if (!bytes) return;
    PlanMessage m;
    m.name = name, m.kind = kind, m.bytes = bytes, m.expected = expected, m.count = times_, m.part = part_;
    m.hash_calls = hash_calls;
    messages_.push_back(m);
  }
  // Labels the message that was added last with its place in the protocol, for prints.
  void last_message_at(const std::string &where) { messages_.back().where = where; }
  // A round with an empty verifier's message. It still makes its query, which absorbs the prover's
  // message; a root in that message rejects the plan.
  PlanRound &close_none(const std::string &name, const std::string &where = std::string()) {
    PlanRound r;
    r.name = name, r.where = where, r.scope = scope_, r.count = times_, r.part = part_;
    if (open_roots_) refuse("round " + name + ": a root stands in the query of the next verifier's message that is not empty");
    r.message_bytes = open_bytes_, r.answers = open_answers_;
    open_bytes_ = 0, open_answers_ = 0;
    require_kind(r);
    rounds_.push_back(r);
    return rounds_.back();
  }
  // A round with a verifier's message. r: what the round draws and states, `rests` and `where`. The
  // other fields are set here (`work_bits` from the setting, by why and scope), and the round
  // enters the plan complete. Returns `work_bits`.
  unsigned close(const std::string &name, RoundKind kind, ErrorKind why, const Bound &error, Priced priced, PlanRound r) {
    BLK_REQUIRE(kind != RoundKind::EMPTY, "Plan: close() is for a verifier's message that is not empty");
    if (error.bad) refuse("round " + name + ": a formula was asked outside its range");
    r.name = name, r.kind = kind, r.why = why, r.scope = scope_, r.count = times_, r.part = part_;
    r.error = error.error(), r.priced = priced;
    r.work_bits = set.work_of(why, scope_);
    BLK_REQUIRE(kind != RoundKind::SEAL || !r.work_bits, "Plan: a seal bears no proof of work");
    // Six bits of room: an honest search then leaves the nonce's range with probability e^-64.
    if (r.work_bits >= set.target || (r.work_bits && r.work_bits + 6 > 8 * set.nonce_bytes))
      refuse("round " + name + ": more work bits than the target or the nonce has room for");
    r.message_bytes = open_bytes_, r.roots = open_roots_, r.answers = open_answers_;
    open_bytes_ = 0, open_roots_ = open_answers_ = 0;
    if (r.work_bits) file("nonce", MessageKind::NONCE, set.nonce_bytes, set.nonce_bytes, 0);
    require_kind(r);
    const unsigned work_bits = r.work_bits;
    rounds_.push_back(std::move(r));
    return work_bits;
  }
  // For a test that builds a plan by hand: adds an otherwise empty round, which the test fills in.
  PlanRound &close(const std::string &name, RoundKind kind, ErrorKind why, const Bound &error, Priced priced) {
    close(name, kind, why, error, priced, PlanRound());
    return rounds_.back();
  }
  // Adds a round that its planner filled itself: no query of this chain (a table's record).
  void enter(PlanRound r) {
    r.part = part_;
    require_kind(r);
    rounds_.push_back(r);
  }
  // Appends the rounds and messages of a finished plan, a chain of its own, under the present part
  // and repeat. For prints over several chains.
  void append(const Planned &o) {
    BLK_REQUIRE(!open_bytes_, "Plan: append in the middle of a prover's message");
    if (o->cause() == Cause::REFUSED) refuse(o->why_not());
    if (!same_setting(set, o->set)) refuse("rounds that were priced under another setting are appended");  // the verdict would read them under this plan's
    for (const PlanRound &r : o->rounds_)
      if (r.count > ~(u64)0 / times_) refuse("repeats of 2^64 times or more");
    for (const PlanMessage &m : o->messages_)
      if (m.count > ~(u64)0 / times_) refuse("repeats of 2^64 times or more");
    for (PlanRound r : o->rounds_) r.part = part_, r.count *= times_, rounds_.push_back(r);
    for (PlanMessage m : o->messages_) m.part = part_, m.count *= times_, messages_.push_back(m);
  }
  static void require_kind(const PlanRound &r) {
    RoundKind kind;
    BLK_REQUIRE(!kind_of_round(r.name, kind) || kind == r.kind, "Plan: a round of the format is planned with another kind than its name's");
  }
  void chain_message(const std::string &name, MessageKind kind, u64 bytes) {
    PlanMessage m;
    m.name = name, m.kind = kind, m.bytes = bytes, m.expected = (double)bytes, m.count = times_, m.part = part_;
    m.in_chain = true;
    messages_.push_back(m);
    open_bytes_ += bytes;
  }
  std::vector<PlanRound> rounds_;
  std::vector<PlanMessage> messages_;
  std::vector<std::string> parts_;
  std::set<u64> one_list_opened_;  // the running numbers of the trees that may be opened at one list alone and have been (blocks.h, OneList)
  std::string refused_;
  std::vector<u64> repeat_;
  u64 times_ = 1;
  unsigned part_ = 0;
  Scope scope_ = Scope::REQUEST;
  size_t open_bytes_ = 0;
  unsigned open_roots_ = 0, open_answers_ = 0;
};

#ifdef BLK_PLANS_BY_HAND_FOR_A_TEST
inline Planned::Planned(const Plan &p) : Planned(p.finish()) {}
BLK_MARK_NOT_FOR_SERVING();  // also where the macro was defined after the first include of base.h, which fence.h's list does not see
#endif

// For a function that returns its reason as a const char *: prints the plan's first cause to stderr
// and returns a fixed sentence, because the cause's own string is destroyed with the plan's data.
inline const char *does_not_hold(const Planned &p, const char *who) {
  std::fprintf(stderr, "%s: the parameters are not admitted: %s\n", who, p->why_not().c_str());
  return "the parameters are not admitted (the first cause is on stderr)";
}

// Steps through those rounds of a plan without repeats that are queries of the hash chain.
class Cursor {
 public:
  // Ends the process unless the plan holds (the exact comparison decides on every parameter set)
  // and was computed over this build's field.
  explicit Cursor(const Planned &p) : plan_(p) {
    const char *fault = p->holds() ? nullptr : does_not_hold(p, "Cursor");
    BLK_REQUIRE(!fault, fault ? fault : "");
    BLK_REQUIRE(p->set.circle == setting_of_this_build().circle && Big::cmp(p->set.p, setting_of_this_build().p) == 0 && p->set.degree == setting_of_this_build().degree,
                "Cursor: the plan's errors were reckoned over another field than this build's");
  }
  // The next round that is a query of the hash chain. The process ends if its name differs. A name
  // says which round of a block it is, not which use of the block (round_names.h).
  const PlanRound &next_round(const char *name) {
    skip();
    BLK_REQUIRE(at_ < plan_->rounds_.size(), "Cursor: a round is run after the last planned one");
    const PlanRound &r = plan_->rounds_[at_];
    BLK_REQUIRE(r.count == 1, "Cursor: a plan with repeats cannot be run");
    BLK_REQUIRE(r.name == name, "Cursor: the round that is run is not the next planned one");
    ++at_;  // after the checks: a round that was not asked for is not passed
    return r;
  }
  bool at_end() {
    skip();
    return at_ == plan_->rounds_.size();
  }

 private:
  void skip() {
    while (at_ < plan_->rounds_.size() && plan_->rounds_[at_].kind == RoundKind::RECORD) ++at_;  // its query is on a separate chain
  }
  const Planned plan_;  // a handle of its own
  size_t at_ = 0;
};

}  // namespace blk

#include "blocks.h"  // never one without the other: see «friend class MayLay» above. (By this spelling the file beside this one is taken, whatever the build line's -I say.)
