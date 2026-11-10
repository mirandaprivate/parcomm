// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The channel through which a protocol block's prover and verifier exchange messages over a hash
// chain (a Fiat-Shamir transcript, chain.h): ProverRounds writes a proof, VerifierRounds reads one,
// Rounds is what they share. A channel is made from a Binder (public.h), which supplies the chain's
// first query (the hash input that fixes the public inputs) and the plan (the list of its rounds).
//
// send..() writes an item into the proof and into the chain's next query. recv..() reads the item,
// rejects it if it is not in canonical form, and puts it into the verifier's own next query. A
// close_..() ends the prover's message, makes the round's query and gives access to the verifier's
// message, which is of one kind only:
//   close_algebraic()  elements of the field E  close_positions()  positions; an element if planned
//   close_epsilon()    one element of E         close_bits()       matrix entries in {-1, 0, 1}
//   close_seed()       seeds of samples         close_empty()      nothing
// seal() ends a chain with a digest, for a protocol of several chains. A mistake of the calling
// program (a call out of order, a round that differs from the plan) ends the process; whatever
// depends on a proof's bytes makes the verifier reject instead (see Rounds).
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "base.h"
#include "chain/chain.h"
#include "chain/public.h"
#include "chain/session.h"
#include "field/field.h"
#include "hash/hash.h"
#include "iop/message.h"
#include "numbers/plan.h"
#include "tree/positions.h"

namespace blk {

// The seed of an out-of-domain sample (the block SEND, blocks/iop/send.h): the point z at which the
// sampled message's multilinear extension is evaluated, and the factor beta with which the
// message's pad enters the answer. The pad is a random element that is sent along with the message
// and masks the answer (what is and is not shown about hiding: docs/limits.md#weight-hiding).
// Oracle messages (encoded rows, sent as Merkle roots) sampled together share one seed: each uses
// the first coordinates of z that it needs.
struct SampleSeed {
  std::vector<E> z;
  E beta = E::one(); // never 0
};

// The seeds of the sample whose answers are stored as the record of a committed table (`Index` in
// identifiers): one seed for each answer of a row. They depend on the arguments only (commitment:
// the table's root, or whatever bytes name it): no salt, proof of work (`work` in identifiers and
// messages), parameter or earlier chain enters. So a commitment has one record, however often an
// honest committer starts again (two records of one table would give away part of its content) and
// whatever the table is used for. Both sides compute the seeds themselves; a record (IndexRecord,
// blocks/iop/foldeq.h) contains only the answers.
inline std::vector<SampleSeed> record_seeds(const u8 *commitment, size_t bytes, unsigned variables, unsigned answers) {
  FirstQuery q;
  q.label = LABEL_RECORD;
  q.salted = false;
  q.statement.assign(commitment, commitment + bytes);
  Chain c(q);
  ChainAbsorb nothing = c.message();
  c.close(nothing, ChainKind::SEED);
  std::vector<SampleSeed> seeds(answers);
  for (SampleSeed &s : seeds) {
    s.z.resize(variables);
    for (E &x : s.z) x = c.draw_E();
    s.beta = c.draw_E_nonzero();
  }
  return seeds;
}

#ifdef BLK_TEST_PROVER_MESSAGES_OUTSIDE
BLK_MARK_NOT_FOR_SERVING();  // fence.h's list misses a macro defined after base.h's first include
inline bool prover_messages_outside = true; // read by ProverRounds alone (see there)
inline bool first_query_stands_still = false; // likewise (see there)
#endif

// Finds the nonce of a round's proof of work, in place of search_nonce() (chain.h).
using NonceSearch = std::function<u64(const ChainPrefix &, unsigned work_bits)>;

class Rounds;

// The base of the handles that a close_..() returns. A handle can be neither copied nor moved. It
// is valid until the next item is sent or received, the next round is closed, or the chain is
// sealed or lent out (ProverRounds::lend()).
class Draw {
public:
  Draw(const Draw &) = delete;
  Draw &operator=(const Draw &) = delete;

protected:
  explicit Draw(Rounds *r);
  Rounds &channel() const; // ends the process if the handle's verifier message is over
  Rounds *r_;
  u64 round_;
};

// Draws a round's elements of E: all by draw() or all by draw_nonzero() (never 0), as planned.
class AlgebraicDraw final : public Draw {
public:
  E draw();
  E draw_nonzero();
  // Forwards to Rounds::require_rows().
  void require_rows(u64 rows) const;

private:
  friend class Rounds;
  using Draw::Draw;
};

// Draws the seeds of out-of-domain samples only: no position and no single element. A prover that
// knew, before it answers a sample, where the sampled codeword will be read, or by which challenge
// it will be combined with others, could answer for another codeword. The next prover message must
// begin with all the answers that the seeds ask for (send_answer()): until they are in, nothing
// else can be sent and no round can be closed.
class SeedDraw final : public Draw {
public:
  // A seed whose point has `variables` coordinates, for samples with `answers` answers in all
  // (elements of E, counted over all oracle messages that use the seed). All seeds of a round have
  // the same, planned number of coordinates.
  SampleSeed seed(unsigned variables, unsigned answers);

private:
  friend class Rounds;
  using Draw::Draw;
};

class PositionsDraw final : public Draw {
public:
  // Draws the planned number of positions below 2^log_n, with repetition; log_n must be the planned
  // one. Once a round. The round's error bound (`priced` in identifiers and messages) counts the
  // draws; the list that is returned is their sorted set, so a position that was drawn twice is
  // opened once, and PositionList::draws() keeps the draws in order (blocks/tree/positions.h). The
  // verifier derives the positions itself; a proof does not list them.
  PositionList positions(unsigned log_n);
  // An element of E after the positions, where the round is planned with one: CS (the code switch,
  // blocks/iop/cs.h) batches its linear claims (`rows` in require_rows()) by it. A linear claim
  // states the value of a linear form of encoded messages.
  E batching();
  void require_rows(u64 rows) const; // as AlgebraicDraw's
  // These two forward to the functions of the same names in Rounds.
  void require_first_list(u64 draws) const;
  void require_table(const PricedTable &run) const;

private:
  friend class Rounds;
  using Draw::Draw;
};

class BitsDraw final : public Draw {
public:
  // n independent entries with Pr[0] = 1/2 and Pr[1] = Pr[-1] = 1/4, the distribution for which the
  // error bound of the approximate range check (RNG, blocks/iop/rng.h) is derived. Once a round,
  // with n as planned. Two bits of a 64-bit word make an entry, low bits first: the first bit says
  // whether the entry is nonzero, the second, if set, makes it -1.
  std::vector<signed char> trits(size_t n);

private:
  friend class Rounds;
  using Draw::Draw;
};

// What the two sides of a channel share.
//
// The plan (`Plan`, blocks/numbers/plan.h). Every close names its round; the channel follows the
// plan as it runs. It ends the process if the round is not the next planned one, if the prover's
// message has not the planned bytes, roots and answers, or if the verifier's message has not exactly the planned
// elements, positions and matrix entries. A round's bits of proof of work are the plan's too. So
// the rounds that run agree in name, kind, counts and work with the planned rounds, whose error
// bounds the plan lists. How far those bounds are tied to the numbers on which a block runs: see
// require_rows(). Not compared with the plan: the bytes written through file_only(), and how a
// round's answers are divided among its seeds.
//
// Rejection (`refuse` in identifiers and messages). Whatever depends on a proof's bytes makes the
// verifier's calls return false and never ends the process. The rejection is permanent: ok() stays
// false (a prover's ok() is always true), and the channel reads nothing more, leaves the chain
// alone and checks neither the plan nor the order of calls. Every recv..() then fails and gives
// zeros, and every draw gives a fixed value (0; 1 for an element that must not be 0; for positions
// one draw, the position 0). So a verifier's function may return at once or run on: finish() is
// false in either case. A function that runs on must take the number of positions from the list,
// never from a number of its own, so that lists, openings and loops still agree in size. A block
// reports a failed check of its own (an equation, an opening) through VerifierRounds::refuse().
class Rounds {
public:
  virtual ~Rounds() = default;
  Rounds(const Rounds &) = delete;
  Rounds &operator=(const Rounds &) = delete;

  AlgebraicDraw close_algebraic(const std::string &name) {
    close(name, RoundKind::ALGEBRAIC);
    return AlgebraicDraw(this);
  }
  // The one element of E that joins a hiding polynomial (a random polynomial added to a sumcheck's
  // summand) to a claim. The polynomial's sum must be the last item sent (send_hiding_sum()): a
  // prover that knew the element before it fixed the sum could fit the sum to a false claim.
  E close_epsilon(const std::string &name) {
    BLK_REQUIRE(!ok() || last_was_hiding_sum_, "rounds: the hiding polynomial's sum must be the last thing sent before this element");
    close(name, RoundKind::EPSILON);
    BLK_REQUIRE(!ok() || entry_->draw_E == 1, "rounds: the round is not planned with this element alone");
    return element(false);
  }
  // The verifier's message is empty. Every prover message must be closed, also one that the
  // verifier does not answer: the round makes its query, so that the message enters the chain.
  // Such a round has no proof of work, and its message must contain no root.
  void close_empty(const std::string &name) {
    BLK_REQUIRE(!ok() || roots_ == 0, "rounds: a root stands in the query of the next verifier's message that is not empty");
    close(name, RoundKind::EMPTY);
  }
  SeedDraw close_seed(const std::string &name) {
    close(name, RoundKind::SEED);
    return SeedDraw(this);
  }
  PositionsDraw close_positions(const std::string &name) {
    close(name, RoundKind::POSITIONS);
    return PositionsDraw(this);
  }
  BitsDraw close_bits(const std::string &name) {
    close(name, RoundKind::BITS);
    return BitsDraw(this);
  }
  // Ends the chain: one more query, with the open message in it, which must contain no root, and a
  // digest of everything that has entered the chain, taken from the answer. The digest goes into
  // another chain's first query (Binder::follows(out), public.h) and into no proof. Afterwards
  // nothing can be sent, closed or drawn. Returns false, with the digest of zeros, for a verifier
  // that has rejected. A malicious prover knows that digest beforehand and can make a proof for the
  // next chain on it, so `out` records the rejection: a verifier's channel on a chain that follows
  // this one starts rejected.
  [[nodiscard]] bool seal(const std::string &name, Sealed &out) {
    BLK_REQUIRE(!ok() || roots_ == 0, "rounds: a root stands in the query of a verifier's message that is drawn for the protocol, not of a seal");
    close(name, RoundKind::SEAL);
    Digest digest{};
    for (int i = 0; i < 4; i++) {
      const u64 w = word();
      digest.w[2 * i] = (u32)w, digest.w[2 * i + 1] = (u32)(w >> 32);
    }
    sealed_ = true;
    if (!ok()) fate_->refused = true;
    std::vector<std::shared_ptr<const ChainFate>> fates(1, fate_); // this chain's own, then those of the chains that it follows
    fates.insert(fates.end(), earlier_fates_.begin(), earlier_fates_.end());
    out = Sealed(marks_, digest, std::move(fates), configuration_);
    return ok();
  }

  // Reads a public input (Public<T> has no accessor). Ends the process unless the input is in the
  // first query of this chain or of a chain that it follows (public.h).
  template <class T> const T &get(const Public<T> &x) const {
    bool mine = false;
    for (const PublicMark &m : marks_) mine = mine || (x.mark_ != nullptr && x.mark_ == m);
    BLK_REQUIRE(mine && x.value_ != nullptr, "rounds: a public input that does not stand in this chain's first query");
    return *x.value_;
  }


  // The require_..() checks. A plan entry can record the numbers from which its round's error bound
  // was computed (PlanRound's rows, first_list, group, table, combinations, sharing:
  // blocks/numbers/plan.h). After the close and before the next item, the block that runs the round
  // passes in the numbers on which it runs. The process ends if the two differ or, except for rows,
  // if the entry records none (a planning function that forgot a number would otherwise go
  // unnoticed). settle() ends it if a recorded number was never compared; the one exemption is in
  // ProverRounds::take_back(). tell() first prints the round and both values; the check's own
  // message is fixed, because tests identify a check by it.
  // Limits: apart from a round's own counts, only these checks tie an error bound to the numbers of
  // a run, and they are worth something only for a plan made by this library's planning functions
  // (see run_verifier(), blocks/chain/closed.h).
  // The planned number is not returned for the caller to compare: after a rejection, when these
  // functions check nothing, such a comparison would end the process on a proof's bytes.

  // The round closed last batches `rows` linear claims (`rows` in this check's identifiers and
  // messages), as many as its planned error bound counts.
  void require_rows(u64 rows) const {
    const bool rows_fine = !ok() || (entry_ && entry_->rows == rows);
    if (!rows_fine) tell(entry_ ? std::to_string(entry_->rows) + " rows" : "", std::to_string(rows) + " rows");
    BLK_REQUIRE(rows_fine, "rounds: the round batches another number of rows than its planned error counts");
    asked_ |= ASKED_ROWS;
  }
  // The round closed last splits its positions into two lists, the first of `draws` draws, as its
  // planned error bound does (only the block LINK, which reads two committed tables, has such a
  // round).
  void require_first_list(u64 draws) const {
    const bool list_fine = !ok() || (entry_ && entry_->first_list != 0 && entry_->first_list == draws);
    if (!list_fine) tell(entry_ && entry_->first_list ? std::to_string(entry_->first_list) + " draws for the first list" : "nothing", std::to_string(draws) + " draws");
    BLK_REQUIRE(list_fine, "rounds: the round's first list has another number of draws than its planned error counts");
    asked_ |= ASKED_FIRST_LIST;
  }
  // The round closed last belongs to a stage of a proximity test (meant to show that the rows of
  // oracle messages are close to codewords): the epsilon and folding rounds of the block FOLD, the
  // lambda round of the block BASE (blocks/iop/fold.h, blocks/iop/base_case.h). The stage runs on
  // the group (the oracle messages that are tested together) from which the plan computed the
  // stage's error bounds: both codes' parameters and the numbers of rows (PricedGroup). run: taken
  // from the codes in the chain's first query and from the block's own shape.
  void require_group(const PricedGroup &run) const {
    const bool group_fine = !ok() || (entry_ && entry_->group.said && entry_->group == run);
    if (!group_fine) tell(entry_ ? text_of(entry_->group) : "", text_of(run, true));
    BLK_REQUIRE(group_fine, "rounds: the round begins a stage on another group than its planned errors are priced for, or the plan names none");
    asked_ |= ASKED_GROUP;
  }
  // The same comparison, under a message of its own, for a round that draws positions on a group's
  // rows (the rounds `LOC`, `CS`, `BASE`). run: the two codes only, again from the first query,
  // with k = h = 0 and slots = 1, which the error bound of a position does not use.
  void require_codes(const PricedGroup &run) const {
    const bool codes_fine = !ok() || (entry_ && entry_->group.said && entry_->group == run);
    if (!codes_fine) tell(entry_ ? text_of(entry_->group) : "", text_of(run, true));
    BLK_REQUIRE(codes_fine, "rounds: the round draws positions on rows of other codes than its planned error is priced for, or the plan names none");
    asked_ |= ASKED_GROUP;
  }
  // The same for a committed table, at a round whose error bound was computed from it: the rounds
  // of LIN (a batch's linear claims on the table) and of LINK that draw a point or the table's
  // columns (PricedTable). run: taken from the table's shape in the first query, except
  // IndexShape::distances, which no first query contains, and `test`, which the caller states.
  void require_table(const PricedTable &run) const {
    const bool table_fine = !ok() || (entry_ && entry_->table.said && entry_->table == run);
    if (!table_fine) tell(entry_ ? text_of(entry_->table) : "", text_of(run, true));
    BLK_REQUIRE(table_fine, "rounds: the round draws columns of another table than its planned error is priced for, or the plan names none");
    asked_ |= ASKED_TABLE;
  }
  // The same for the round that draws the matrix of an approximate range check (RNG): the number N
  // of combinations, from which the plan computed the round's error bound 2^-N. The channel's own
  // count covers only the number of the matrix's entries, columns x combinations.
  void require_combinations(u64 run) const {
    const bool combinations_fine = !ok() || (entry_ && entry_->combinations != 0 && entry_->combinations == run);
    if (!combinations_fine) tell(entry_ && entry_->combinations ? std::to_string(entry_->combinations) + " combinations" : "nothing", std::to_string(run) + " combinations");
    BLK_REQUIRE(combinations_fine, "rounds: the round draws a matrix of another number of combinations than its planned error is priced for, or the plan names none");
    asked_ |= ASKED_COMBINATIONS;
  }
  // The same for the round that draws a point shared by several requests (LIN's): the number of
  // requests that may share it, by which the plan multiplied the round's error bound. run: the
  // number of places (leaves) of the binding tree, the Merkle tree over the seals of the requests'
  // first chains.
  void require_sharing(u64 run) const {
    const bool sharing_fine = !ok() || (entry_ && entry_->sharing != 0 && entry_->sharing == run);
    if (!sharing_fine) tell(entry_ && entry_->sharing ? std::to_string(entry_->sharing) + " requests that share the point" : "nothing", std::to_string(run) + " requests");
    BLK_REQUIRE(sharing_fine, "rounds: the round draws a point for another number of requests than its planned error is priced for, or the plan names none");
    asked_ |= ASKED_SHARING;
  }

  // Whether this chain, or a chain that it follows, drew the list, or the list from which part() or
  // short_list() made it.
  bool drew(const PositionList &l) const {
    bool found = false;
    for (const PublicMark &m : marks_) found = found || (l.drawn_by_ && l.drawn_by_ == m);
    return found;
  }
  // Whether this chain follows `earlier`: its Binder called follows() on that chain's seal, or on
  // the seal of a chain that follows it.
  bool stands_on(const Rounds &earlier) const {
    bool found = false;
    for (size_t i = 1; i < marks_.size(); i++) found = found || (!earlier.marks_.empty() && marks_[i] == earlier.marks_[0]);
    return found;
  }
  const Planned &plan() const { return plan_; }
  // This chain's run's value, a hash of its label, parameters and salt (first_query_run(),
  // chain.h; messages call the salt "random bytes"): the leaves of the trees that the chain sends
  // are hashed starting from it. Each side takes it from its own channel, never from a proof.
  const Digest &run() const {
    BLK_REQUIRE(has_run_, "rounds: a chain without random bytes in its first query has no run's value");
    return run_;
  }
  // The configuration digest (a hash of the public configuration files), from the Binder; the same
  // in all chains that follow one another. The vertices of all trees, and the leaves of trees that
  // no chain's run has made (a committed table's, the binding tree), are hashed starting from it.
  const Digest &configuration() const { return configuration_; }
  // How many rounds have been closed; after a rejection a close still counts but makes no query.
  size_t queries() const { return (size_t)closes_; }
  virtual bool ok() const = 0;

protected:
  friend class Draw;
  friend class AlgebraicDraw;
  friend class SeedDraw;
  friend class PositionsDraw;
  friend class BitsDraw;
  explicit Rounds(const Planned &plan) : plan_(plan), cursor_(plan) {
    BLK_REQUIRE(plan->set.nonce_bytes >= 1 && plan->set.nonce_bytes <= 8 && plan->set.position_bytes == 0,
                "rounds: a nonce has 1 to 8 bytes, and a proof lists no position");
  }
  // The Binder's first query. The Binder takes no further input afterwards.
  static FirstQuery query_of(const Binder &b, const Digest &salt) { return b.first_query(salt); }
  void keep_marks(const Binder &b) {
    marks_.assign(1, b.mark_);
    marks_.insert(marks_.end(), b.earlier_.begin(), b.earlier_.end());
    earlier_fates_ = b.earlier_fates_;
  }
  // Whether a chain that this one follows has rejected, before its seal or after it.
  bool follows_a_refusal() const {
    bool refused = false;
    for (const auto &f : earlier_fates_) refused = refused || f->refused;
    return refused;
  }
  void start(const FirstQuery &q) {
    BLK_REQUIRE((q.label >> 32) == 0, "rounds: a protocol's label is below 2^32");
    chain_.reset(new Chain(q));
    has_run_ = q.salted;
    if (has_run_) run_ = first_query_run(q);
    if (has_run_ && run_ == Digest{}) { // chance 2^-256; no hash starts from the digest of zeros
      fail("a chain's run's value is the digest of zeros");
      run_.w[0] = 1; // a nonzero value for a function that runs on after the rejection
    }
  }

  // ---- the open prover message ----
  enum class Item { PLAIN, ROOT, ANSWER, HIDING_SUM };
  // Checks the rules of order for the next item; both sides call it before they write or read one.
  void item(Item what) {
    if (!ok()) return;
    BLK_REQUIRE(!sealed_, "rounds: the chain has been sealed");
    BLK_REQUIRE(!lent_, "rounds: the chain is lent out");
    settle();
    if (what == Item::ANSWER)
      BLK_REQUIRE(owed_ > 0, "rounds: an answer that no seed has asked for");
    else
      BLK_REQUIRE(owed_ == 0, "rounds: after a round of seeds the prover's message begins with the answers");
    if (what == Item::ANSWER) owed_--, answers_++;
    if (what == Item::ROOT) {
      BLK_REQUIRE(has_run_, "rounds: a chain that sends a tree has random bytes in its first query");
      roots_++;
    }
    last_was_hiding_sum_ = (what == Item::HIDING_SUM);
  }
  void absorb(const u8 *p, size_t n) { message_.insert(message_.end(), p, p + n); }
  void absorb(F x) {
    u8 b[F_BYTES];
    F_to_bytes(x, b);
    absorb(b, F_BYTES);
  }
  void absorb(const E &x) {
    u8 b[E_BYTES];
    E_to_bytes(x, b);
    absorb(b, E_BYTES);
  }
  void absorb(const Digest &d) {
    u8 b[DIGEST_BYTES];
    digest_to_bytes(d, b);
    absorb(b, DIGEST_BYTES);
  }
  void absorb_u64(u64 x) {
    u8 b[8];
    for (int i = 0; i < 8; i++) b[i] = (u8)(x >> (8 * i));
    absorb(b, 8);
  }

  // ---- what the two sides do differently ----
  virtual bool nonce(const ChainPrefix &p, unsigned work_bits, u64 &nonce) = 0; // search and write; or read
  virtual bool messages_stay_outside() const { return false; } // true only in a test's prover (ProverRounds)
  virtual void fail(const char *why) = 0; // the channel's own rejections

  static ChainKind number_of(RoundKind kind) {
    switch (kind) {
    case RoundKind::ALGEBRAIC: return ChainKind::ALGEBRAIC;
    case RoundKind::EPSILON: return ChainKind::EPSILON;
    case RoundKind::SEED: return ChainKind::SEED;
    case RoundKind::POSITIONS: return ChainKind::POSITIONS;
    case RoundKind::BITS: return ChainKind::BITS;
    case RoundKind::SEAL: return ChainKind::SEAL;
    case RoundKind::EMPTY: return ChainKind::EMPTY;
    default: break;
    }
    require_failed(__FILE__, __LINE__, "rounds: a round of this kind is no query of the chain");
  }

  // The verifier's message is over: everything in the plan entry must have been drawn.
  void settle() {
    if (!entry_) return;
    BLK_REQUIRE(drawn_E_ == entry_->draw_E && drawn_trits_ == entry_->draw_trits && positions_done_ == (entry_->draw_positions > 0),
                "rounds: less was drawn than the round is planned with");
    BLK_REQUIRE(drawn_nonzero_ == entry_->draw_nonzero, "rounds: another number of elements was drawn without 0 than the round is planned with");
    // Every number that the plan entry records must have been compared by a require_..() call.
    const unsigned said = (entry_->rows ? ASKED_ROWS : 0u) | (entry_->first_list ? ASKED_FIRST_LIST : 0u) | (entry_->group.said ? ASKED_GROUP : 0u) | (entry_->table.said ? ASKED_TABLE : 0u) | (entry_->combinations ? ASKED_COMBINATIONS : 0u) | (entry_->sharing ? ASKED_SHARING : 0u);
    const unsigned unasked = ok() ? said & ~asked_ : 0u;
    if (unasked)
      std::fprintf(stderr, "rounds: at the round %s%s%s: said by the plan and asked by nobody:%s%s%s%s%s%s\n", entry_->name.c_str(), entry_->where.empty() ? "" : ", ", entry_->where.c_str(),
                   unasked & ASKED_ROWS ? " the rows (require_rows)" : "", unasked & ASKED_FIRST_LIST ? " the first list (require_first_list)" : "", unasked & ASKED_GROUP ? " the group or its codes (require_group, require_codes)" : "",
                   unasked & ASKED_TABLE ? " the table (require_table)" : "", unasked & ASKED_COMBINATIONS ? " the combinations (require_combinations)" : "",
                   unasked & ASKED_SHARING ? " the requests that share the point (require_sharing)" : "");
    BLK_REQUIRE(!unasked, "rounds: the plan says beside the round what its error was priced from, and no block has asked for it");
    entry_ = nullptr;
  }
  // The next planned round, which must have this name and kind.
  const PlanRound &planned(const std::string &name, RoundKind kind) {
    BLK_REQUIRE(!sealed_, "rounds: the chain has been sealed");
    BLK_REQUIRE(owed_ == 0, "rounds: a round is closed while answers to a sample are owed");
    settle();
    const PlanRound &entry = cursor_.next_round(name.c_str());
    BLK_REQUIRE(entry.kind == kind, "rounds: the round is planned with another kind of verifier message");
    return entry;
  }
  void close(const std::string &name, RoundKind kind) {
    if (!ok()) {
      closes_++;
      entry_ = nullptr;
      return;
    }
    BLK_REQUIRE(!lent_, "rounds: the chain is lent out");
    const PlanRound &entry = planned(name, kind);
    BLK_REQUIRE(kind != RoundKind::EMPTY || entry.work_bits == 0, "rounds: a round in which the verifier says nothing has no work");
    // With this rule, draw_E and draw_nonzero tell which of a round's elements are drawn nonzero.
    BLK_REQUIRE(kind == RoundKind::SEED || entry.draw_nonzero == 0 || (kind == RoundKind::ALGEBRAIC && entry.draw_nonzero == entry.draw_E),
                "rounds: the elements of a round are all drawn with 0 or, in a round of elements alone, all without");
    ask(entry);
  }
  // Ends the open message and makes the round's query. With bits of proof of work in the plan
  // entry, the prover searches for a nonce and writes it into the proof; the verifier reads it, and
  // the work is tested before anything is drawn.
  void ask(const PlanRound &entry) {
    BLK_REQUIRE(message_.size() + outside_ == entry.message_bytes && roots_ == entry.roots && answers_ == entry.answers,
                "rounds: the prover's message has not the planned bytes, roots and answers");
    ChainAbsorb a = chain_->message();
    if (!messages_stay_outside()) a.push(message_.data(), message_.size());
    const ChainPrefix p = Chain::prefix(a, number_of(entry.kind), entry.work_bits);
    u64 n = 0;
    if (entry.work_bits > 0 && !nonce(p, entry.work_bits, n))
      fail("a nonce is missing");
    else if (!chain_->step(p, n, entry.work_bits))
      fail("a nonce does not give the proof of work that its round asks for");
    message_.clear();
    roots_ = answers_ = 0;
    outside_ = 0;
    last_was_hiding_sum_ = false;
    closes_++;
    entry_ = &entry;
    drawn_E_ = drawn_nonzero_ = drawn_trits_ = 0;
    positions_done_ = false;
    asked_ = 0;
  }

  // ---- the verifier's message; the handles call these ----
  E element(bool nonzero) {
    if (!ok()) return nonzero ? E::one() : E::zero();
    BLK_REQUIRE(++drawn_E_ <= entry_->draw_E, "rounds: more elements are drawn than the round is planned with");
    drawn_nonzero_ += nonzero ? 1 : 0;
    return nonzero ? chain_->draw_E_nonzero() : chain_->draw_E();
  }
  u64 word() { return ok() ? chain_->next_word() : 0; }

  // The end of the protocol: every planned round has been run completely, no answer is owed and no
  // prover message is open. A caller must end every chain of a proof, by finish() or
  // finish_inside_a_file(); the chains that this one follows must have been ended before it.
  void end() {
    for (const auto &f : earlier_fates_) BLK_REQUIRE(f->ended, "rounds: a chain that this one follows has not been brought to its end");
    BLK_REQUIRE(!lent_, "rounds: the chain is lent out");
    BLK_REQUIRE(owed_ == 0, "rounds: the protocol ends while answers to a sample are owed");
    settle();
    BLK_REQUIRE(cursor_.at_end(), "rounds: the protocol ends before the last planned round");
    BLK_REQUIRE(message_.empty(), "rounds: a prover's message is still open at the chain's end; its round is closed like every other");
  }

  const Planned plan_; // the channel's own copy of the handle to the plan
  Cursor cursor_;
  std::unique_ptr<Chain> chain_;
  std::vector<u8> message_;
  // Roots and answers in the open message; owed_: answers that seeds asked for and are still due.
  unsigned roots_ = 0, answers_ = 0, owed_ = 0;
  size_t outside_ = 0; // ask() adds it to the open message's length; always 0: no code raises it
  bool last_was_hiding_sum_ = false;
  std::vector<PublicMark> marks_; // the Binder's own, then those of the chains that it follows
  // Whether this chain has rejected or ended (ChainFate, public.h), shared through its seal with
  // the chains that follow it; then the same of the chains that it follows.
  std::shared_ptr<ChainFate> fate_ = std::make_shared<ChainFate>();
  std::vector<std::shared_ptr<const ChainFate>> earlier_fates_;
  bool has_run_ = false;
  Digest run_{}, configuration_{};
  bool sealed_ = false;
  bool lent_ = false;
  u64 closes_ = 0;
  const PlanRound *entry_ = nullptr; // the round closed last, until its verifier's message is over
  u64 drawn_E_ = 0, drawn_nonzero_ = 0, drawn_trits_ = 0;
  bool positions_done_ = false;
  static std::string text_of(const PricedCode &c) { return "2^" + std::to_string(c.log_n) + ", " + std::to_string(c.ell) + " + " + std::to_string(c.r); }
  static std::string text_of(const PricedGroup &g, bool of_the_run = false) {
    if (!g.said && !of_the_run) return "nothing";
    return "main code " + text_of(g.main) + "; mask code " + text_of(g.mask) + "; k " + std::to_string(g.k) + ", h " + std::to_string(g.h) + ", slots " + std::to_string(g.slots);
  }
  static std::string text_of(const PricedTable &t, bool of_the_run = false) {
    if (!t.said && !of_the_run) return "nothing";
    return "code " + text_of(t.code) + "; 2^" + std::to_string(t.v) + " rows; " + std::to_string(t.answers) + " answers; " + blk::text_of(t.distances) + "; " +
           (t.test == TableTest::ONE_TIME ? "the one-time test" : t.test == TableTest::LINK ? "a link" : "a later test");
  }
  void tell(const std::string &plan_says, const std::string &run_has) const {
    if (!entry_) return (void)std::fprintf(stderr, "rounds: no round is open: a block's question comes right behind the close of its round\n");
    std::fprintf(stderr, "rounds: at the round %s%s%s: the plan says %s; the run has %s\n", entry_->name.c_str(), entry_->where.empty() ? "" : ", ", entry_->where.c_str(), plan_says.c_str(), run_has.c_str());
  }
  static constexpr unsigned ASKED_ROWS = 1, ASKED_FIRST_LIST = 2, ASKED_GROUP = 4, ASKED_TABLE = 8, ASKED_COMBINATIONS = 16, ASKED_SHARING = 32;
  mutable unsigned asked_ = 0; // ASKED_ bits of the require_..() checks made since the last close
};

inline Draw::Draw(Rounds *r) : r_(r), round_(r->closes_) {}
inline Rounds &Draw::channel() const {
  BLK_REQUIRE(!r_->ok() || (r_->closes_ == round_ && r_->entry_ != nullptr && !r_->sealed_ && !r_->lent_),
              "rounds: this verifier message is over (an item was sent, a round closed, or the chain sealed or lent out)");
  return *r_;
}
inline E AlgebraicDraw::draw() { return channel().element(false); }
inline E AlgebraicDraw::draw_nonzero() { return channel().element(true); }
inline void AlgebraicDraw::require_rows(u64 rows) const { channel().require_rows(rows); }
inline void PositionsDraw::require_rows(u64 rows) const { channel().require_rows(rows); }
inline void PositionsDraw::require_first_list(u64 draws) const { channel().require_first_list(draws); }
inline void PositionsDraw::require_table(const PricedTable &run) const { channel().require_table(run); }
inline SampleSeed SeedDraw::seed(unsigned variables, unsigned answers) {
  BLK_REQUIRE(answers > 0, "rounds: a seed without an answer");
  Rounds &r = channel();
  BLK_REQUIRE(!r.ok() || (u64)(variables + 1) * r.entry_->draw_nonzero == r.entry_->draw_E, "rounds: the seeds of a round have all the planned number of coordinates");
  if (r.ok()) r.owed_ += answers;
  SampleSeed s;
  s.z.resize(variables);
  for (E &x : s.z) x = r.element(false);
  s.beta = r.element(true);
  return s;
}
inline PositionList PositionsDraw::positions(unsigned log_n) {
  Rounds &r = channel();
  if (!r.ok()) return PositionList::from_draws({0}, log_n);
  const PlanRound &entry = *r.entry_;
  BLK_REQUIRE(entry.log_n == log_n, "rounds: the round is planned with positions from another range");
  BLK_REQUIRE(!r.positions_done_, "rounds: the positions of a round are drawn once");
  BLK_REQUIRE(entry.draw_positions > 0, "rounds: the round is planned without positions");
  r.positions_done_ = true;
  std::vector<u64> x((size_t)entry.draw_positions, 0);
  for (u64 &v : x) v = r.chain_->draw_position(entry.log_n);
  PositionList l = PositionList::from_draws(std::move(x), entry.log_n);
  l.drawn_by_ = r.marks_[0];
  return l;
}
inline E PositionsDraw::batching() {
  Rounds &r = channel();
  BLK_REQUIRE(!r.ok() || r.positions_done_, "rounds: the batching element comes after the positions");
  return r.element(false);
}
inline std::vector<signed char> BitsDraw::trits(size_t n) {
  Rounds &r = channel();
  std::vector<signed char> t(n, 0);
  if (!r.ok()) return t;
  BLK_REQUIRE(r.drawn_trits_ == 0, "rounds: the entries of a round are drawn in one call");
  r.drawn_trits_ += n;
  BLK_REQUIRE(r.drawn_trits_ <= r.entry_->draw_trits, "rounds: more entries are drawn than the round is planned with");
  for (size_t i = 0; i < n; i += 32) {
    u64 w = r.word();
    // Entry = b0 - 2 (b0 & b1), b0 the lower bit: no branch on a drawn bit, which is unpredictable.
    for (size_t j = 0; j < 32 && i + j < n; j++, w >>= 2) t[i + j] = (signed char)((int)(w & 1) - (int)(w & (w << 1) & 2));
  }
  return t;
}

// The prover's side. Each send..() writes its item into the proof and adds it to the open message,
// which ask() puts into the round's query (except in a test's prover: see messages_stay_outside()).
class ProverRounds final : public Rounds {
public:
  // random_bytes: the chain's salt, from private randomness; not used if random_bytes_are() or
  // no_random_bytes() was called on the Binder.
  ProverRounds(const Binder &b, const Digest &random_bytes, Writer &proof, NonceSearch search = nullptr)
      : Rounds(b.plan()), w_(proof), search_(std::move(search)) {
    FirstQuery q = query_of(b, random_bytes);
#ifdef BLK_TEST_PROVER_MESSAGES_OUTSIDE
    // See messages_stay_outside() below.
    if (first_query_stands_still) q.statement.assign(q.statement.size(), 0), q.record.assign(q.record.size(), 0), q.salt = Digest{};
#endif
    if (q.salted && q.salt_in_proof) w_.put(q.salt);
    start(q);
    configuration_ = b.configuration_digest();
    keep_marks(b);
  }
  bool ok() const override { return true; }

  void send(F x) { item(Item::PLAIN), w_.put(x), absorb(x); }
  void send(const E &x) {
    item(Item::PLAIN), w_.put(x);
    absorb(x);
  }
  void send_u64(u64 x) { item(Item::PLAIN), w_.put_u64(x), absorb_u64(x); }
  void send_root(const Digest &root) {
    item(Item::ROOT), w_.put(root);
    absorb(root);
  }
  void send_answer(const E &y) {
    item(Item::ANSWER), w_.put(y);
    absorb(y);
  }
  void send_hiding_sum(const E &sum) {
    item(Item::HIDING_SUM), w_.put(sum);
    absorb(sum);
  }
  // For Merkle openings: columns, salts, path digests. These bytes enter no query of the chain.
  Writer &file_only() {
    BLK_REQUIRE(!lent_, "rounds: the chain is lent out");
    return w_;
  }

  // ---- lending the chain: rounds that run elsewhere ----
  // A sumcheck's rounds follow one another too fast for a trip from the GPU to the host and back
  // in each. lend() returns a copy of the chain and of the next query with the open message in it,
  // both plain data. A kernel continues with chain.h's calls (push what it sends, close the round,
  // draw) and keeps, in order, all bytes that it sent and all elements that it drew. Until
  // take_back(), any send, close, draw or file_only() on this channel ends the process.
  // take_back() runs the same rounds here from those bytes, writes them into the proof, checks each
  // round against the plan and returns the drawn elements. It ends the process unless the bytes are
  // field elements in canonical form, this chain draws the kernel's elements, and it reaches the
  // value and round count of the kernel's copy. Lent rounds have no proof of work and draw elements
  // of E or nothing; the kernel sends scalars only: no root, no answer to a sample. Bytes after the
  // last round's message stay open, as after send().
  // Not checked, because bytes do not show it: that an EPSILON round's message ends with the hiding
  // polynomial's sum, and (tail_is_hiding_sum) that the open bytes at the end do, for a
  // close_epsilon() that follows on the host. The verifier's channel applies both rules itself.
  struct Lent {
    Chain chain;
    ChainAbsorb message;
  };
  struct LentRound {
    const char *name;
    RoundKind kind;       // ALGEBRAIC, EPSILON or EMPTY (verifier's message empty: draws = 0)
    bool nonzero = false; // the elements were drawn with draw_E_nonzero()
    unsigned draws = 1;   // how many, one after the other
  };
  Lent lend() {
    BLK_REQUIRE(!lent_, "rounds: the chain is lent out");
    BLK_REQUIRE(!sealed_, "rounds: the chain has been sealed");
    BLK_REQUIRE(owed_ == 0, "rounds: the chain is lent out while answers to a sample are owed");
    settle();
    lent_ = true;
    Lent l{*chain_, chain_->message()};
    l.chain.shut(); // the verifier's message before the lent rounds is over for the borrower too
    l.message.push(message_.data(), message_.size());
    return l;
  }
  // drawn_elsewhere: the kernel's elements, as many as the rounds draw in all.
  std::vector<E> take_back(const std::vector<LentRound> &rounds, const u8 *sent, size_t bytes, const E *drawn_elsewhere, const Chain &end,
                           bool tail_is_hiding_sum = false) {
    return take_back(rounds, sent, bytes, drawn_elsewhere, end.value(), end.rounds(), tail_is_hiding_sum);
  }
  std::vector<E> take_back(const std::vector<LentRound> &rounds, const u8 *sent, size_t bytes, const E *drawn_elsewhere,
                           const Digest &end_value, u64 end_rounds, bool tail_is_hiding_sum = false) {
    BLK_REQUIRE(lent_, "rounds: the chain was not lent out");
    std::vector<E> drawn;
    size_t used = 0;
    auto plain = [&](size_t n) {
      BLK_REQUIRE(n <= bytes - used, "rounds: fewer bytes come back than the planned messages have");
      BLK_REQUIRE(n % F_BYTES == 0, "rounds: the bytes that come back are not whole field elements");
      for (size_t i = 0; i < n; i += F_BYTES) {
        F x;
        BLK_REQUIRE(F_from_bytes(sent + used + i, x), "rounds: the bytes that come back are not field elements in canonical form");
      }
      w_.put_bytes(sent + used, n);
      absorb(sent + used, n);
      used += n;
      if (n > 0) last_was_hiding_sum_ = false;
    };
    for (const LentRound &r : rounds) {
      BLK_REQUIRE(r.kind == RoundKind::ALGEBRAIC || r.kind == RoundKind::EPSILON || r.kind == RoundKind::EMPTY,
                  "rounds: only rounds that draw elements of E, or nothing, can be lent out");
      const PlanRound &entry = planned(r.name, r.kind);
      BLK_REQUIRE(entry.work_bits == 0, "rounds: a round with work cannot be lent out");
      BLK_REQUIRE(entry.draw_E == r.draws && (r.draws >= 1) == (r.kind != RoundKind::EMPTY), "rounds: a round that is lent out draws the planned number of elements");
      BLK_REQUIRE(entry.message_bytes >= message_.size(), "rounds: the prover's message has not the planned bytes, roots and answers");
      plain(entry.message_bytes - message_.size());
      ask(entry);
      // The one exemption from settle()'s rule: a lent round's `rows` (see require_rows()) are not
      // compared with the plan entry's. A kernel ran the round, and the host has no count of its
      // own here (that would take a second log beside the GPU's, kept in step by hand). A verdict
      // rests on the verifier's channel, which has no such exemption.
      asked_ |= ASKED_ROWS;
      for (unsigned i = 0; i < r.draws; i++) {
        drawn.push_back(element(r.nonzero));
        BLK_REQUIRE(drawn.back() == drawn_elsewhere[drawn.size() - 1], "rounds: elsewhere other elements were drawn than the messages give");
      }
    }
    settle();
    const size_t tail = bytes - used;
    plain(tail);
    BLK_REQUIRE(!tail_is_hiding_sum || tail >= E_BYTES, "rounds: no bytes are left open that could be a hiding polynomial's sum");
    last_was_hiding_sum_ = tail_is_hiding_sum;
    BLK_REQUIRE(chain_->rounds() == end_rounds && chain_->value() == end_value,
                "rounds: the chain that was run elsewhere has not the value that its messages give");
    lent_ = false;
    return drawn;
  }

#ifdef BLK_TEST_PROVER_MESSAGES_OUTSIDE
  // For a test's prover only. Nothing that it sends enters its queries, so its challenges depend
  // only on the first query and on each round's number, kind and work: the interactive protocol
  // under fixed challenges. Tests of hiding use it to see how what is sent varies with the prover's
  // private randomness. Verifiers are unchanged: they hash what they read and reject such a proof.
  // lend() does not fit such a build: a kernel hashes what it sends. For an ordinary proof from the
  // same program, a test sets prover_messages_outside to false meanwhile.
  // first_query_stands_still, set as well, is for a prover whose statement varies with its
  // randomness (a table's root, a binding tree's root): statement, record and salt become zeros of
  // the same lengths. So a salt in the proof is 32 zeros, and all runs have the same run's value.
  bool messages_stay_outside() const override { return prover_messages_outside; }
#endif
  void finish_inside_a_file() { end(), fate_->ended = true; } // the file goes on with another chain's bytes
  void finish() {
    end(), fate_->ended = true;
  }

protected:
  bool nonce(const ChainPrefix &p, unsigned work_bits, u64 &n) override {
    n = search_ ? search_(p, work_bits) : search_nonce(p, work_bits);
    const unsigned bytes = plan_->set.nonce_bytes;
    BLK_REQUIRE(bytes == 8 || (n >> (8 * bytes)) == 0, "rounds: the search left the range of a nonce");
    u8 b[8];
    for (int i = 0; i < 8; i++) b[i] = (u8)(n >> (8 * i));
    w_.put_bytes(b, bytes);
    return true;
  }
  void fail(const char *why) override {
    require_failed(__FILE__, __LINE__, why); // an honest prover's own nonce has the work
  }

private:
  Writer &w_;
  NonceSearch search_;
};

// A root as the verifier received it. With its 32 bytes it keeps the two values from which the
// tree's hashes start: the run's value (leaves) and the configuration digest (vertices) of the
// chain that received it. Only VerifierRounds::recv_root() pairs them, so no caller can give a root
// another chain's values. A later chain of the same VerifierSession that reads the tree gets the
// same object; receive_opening() (blocks/tree/opening.h) checks openings against it.
class SentRoot {
public:
  SentRoot() = default; // of no chain
  const Digest &digest() const { return digest_; }
  const Digest &run() const { return run_; }
  const Digest &configuration() const { return configuration_; }

private:
  friend class VerifierRounds;
  SentRoot(const Digest &digest, const Digest &run, const Digest &configuration, PublicMark sender)
      : digest_(digest), run_(run), configuration_(configuration), sender_(std::move(sender)) {}
  Digest digest_{}, run_{}, configuration_{};
  PublicMark sender_; // of the chain that received it
};

// The verifier's side. What its calls do once a proof is rejected: see Rounds, "Rejection".
class VerifierRounds final : public Rounds {
public:
  // Reads the chain's salt from `proof`, unless the Binder sets or omits it. `session` (session.h),
  // one for all chains of one file's check, must outlive the channel and contain the chains that
  // this one follows.
  VerifierRounds(const Binder &b, Reader &proof, VerifierSession &session) : Rounds(b.plan()), r_(proof), session_(session) {
    FirstQuery q = query_of(b, Digest{});
    if (q.salted && q.salt_in_proof && !r_.get(q.salt)) {
      q.salt = Digest{};
      fail("the chain's salt is missing");
    }
    start(q);
    configuration_ = b.configuration_digest();
    keep_marks(b);
    for (const auto &f : earlier_fates_) BLK_REQUIRE(session_.holds(f), "rounds: a chain follows a chain of another session");
    session_.enter(fate_, [this] { return ok(); });
    if (follows_a_refusal()) fail("an earlier chain of this proof was refused");
  }
  ~VerifierRounds() { // records the outcome for the session and for the chains that follow
    if (!ok()) fate_->refused = true;
    session_.leave(fate_);
  }
  VerifierRounds(const VerifierRounds &) = delete;
  VerifierRounds &operator=(const VerifierRounds &) = delete;
  VerifierSession &session() { return session_; }
  bool ok() const override { return why_ == nullptr && r_.ok() && !follows_a_refusal(); }
  // The first reason for a rejection, for a log; nullptr if there is none.
  const char *why() const {
    return why_ ? why_ : !r_.ok() ? unnamed() : follows_a_refusal() ? "an earlier chain of this proof was refused" : nullptr;
  }
  // Reports a failed check that the caller made itself: an equation, an opening that
  // does not lead to its root. why: a string literal (the pointer is kept).
  void refuse(const char *why) {
    fail(r_.ok() ? why : unnamed()); // a read through file_only() may have failed before, without a word to the channel
  }
  // For a caller whose own read through file_only() has just failed: it knows what it was reading.
  void refuse_what_was_read(const char *why) { fail(why); }

  [[nodiscard]] bool recv(F &x) { return item(Item::PLAIN), take(x); }
  [[nodiscard]] bool recv(E &x) { return item(Item::PLAIN), take(x); }
  [[nodiscard]] bool recv_u64(u64 &x) {
    item(Item::PLAIN);
    x = 0;
    if (!ok()) return false;
    if (!r_.get_u64(x)) {
      x = 0;
      fail("a value that the prover sends is missing");
      return false;
    }
    absorb_u64(x);
    return true;
  }
  // After a rejection, or if the root is missing, `root` is the default SentRoot, of no chain.
  [[nodiscard]] bool recv_root(SentRoot &root) {
    item(Item::ROOT);
    Digest d;
    root = SentRoot();
    if (!take(d)) return false;
    root = SentRoot(d, run(), configuration_, marks_[0]);
    return true;
  }
  // Whether this chain, or one that it follows, has received the root.
  bool received(const SentRoot &root) const {
    bool mine = false;
    for (const PublicMark &m : marks_) mine = mine || (root.sender_ && m == root.sender_);
    return mine;
  }
  [[nodiscard]] bool recv_answer(E &y) { return item(Item::ANSWER), take(y); }
  [[nodiscard]] bool recv_hiding_sum(E &sum) { return item(Item::HIDING_SUM), take(sum); }
  Reader &file_only() { return r_; }

  // True if and only if nothing was rejected and every byte of the proof was read. If nothing was
  // rejected before the call, end() ends the process unless the protocol is complete. The verdict
  // on a proof of several chains is VerifierSession::accepted() (session.h).
  [[nodiscard]] bool finish() {
    if (ok()) {
      end();
      if (!r_.at_end()) fail("bytes are left over at the end of the proof");
    }
    return ended();
  }
  // finish() for a chain after which the file continues with another chain's bytes: it does not
  // require that every byte was read. The file's last chain calls finish().
  [[nodiscard]] bool finish_inside_a_file() {
    if (ok()) end();
    return ended();
  }

protected:
  template <class T> bool take(T &x) {
    x = T{};
    if (!ok()) return false;
    if (!r_.get(x)) {
      x = T{};
      fail("a value that the prover sends is missing or not in canonical form");
      return false;
    }
    absorb(x);
    return true;
  }
  bool nonce(const ChainPrefix &, unsigned, u64 &n) override {
    const u8 *q = r_.take(plan_->set.nonce_bytes);
    if (!q) return false;
    n = 0;
    for (unsigned i = 0; i < plan_->set.nonce_bytes; i++) n |= (u64)q[i] << (8 * i);
    return true;
  }
  void fail(const char *why) override {
    fate_->refused = true;
    if (!why_) {
      why_ = why;
    }
    r_.refuse();
  }

private:
  // Records the outcome for the chains that follow. ok() also sees a failed read via file_only().
  bool ended() {
    if (!ok()) fate_->refused = true;
    fate_->ended = true;
    return ok();
  }
  static const char *unnamed() { return "an opening, or another answer to the verifier's queries, is cut short or not in canonical form"; }
  Reader &r_;
  VerifierSession &session_;
  const char *why_ = nullptr;
};
static_assert(std::is_trivially_copyable<ProverRounds::Lent>::value, "what is lent goes to a kernel as plain bytes");

} // namespace blk
