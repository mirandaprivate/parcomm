// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The Fiat-Shamir hash chain: hash outputs replace the verifier's messages of one interactive
// protocol. A round may require proof of work (`work_bits` bits): a nonce found by search.
//
// A query is an input to the hash; its answer is the output. A chain has one 32-byte value, rho.
// Its first value is the answer to the first query (`FirstQuery`), which fixes label, parameters,
// statement and any salt before the prover speaks. Round i makes one query,
//     (the chain's label, rho_{i-1}, i; the prover's message of the round, Merkle roots included;
//      the kind of the verifier's message and the round's `work_bits`; nonce_i, 0 without work),
// whose answer gives rho_i and the verifier's message. A round whose verifier's message is empty
// makes its query too, so every prover message enters a query. A query contains the entire previous
// value, so anyone who knows rho_i has made every query before it. A proof contains nothing of a
// chain but its salt (if `salt_in_proof`) and its nonces: no rho_i, no digest of a round.
//
// rounds.h, not this file, enforces which draws a round may make and in which order messages come.
#pragma once
#include <type_traits>
#include <vector>

#include "base.h"
#include "field/field.h"
#include "hash/hash.h"

namespace blk {

// Labels of first queries that this directory assigns itself. A protocol's own labels are below
// 2^32: Rounds::start() in rounds.h enforces that; `ChainT`'s constructor requires only a label
// below 2^48. 2^32 + 1 is reserved. A committed table is a matrix committed once and read by many
// runs; its record is its stored answers to out-of-domain samples (random linear forms chosen
// after the commitment).
constexpr u64 LABEL_RECORD = (u64)1 << 32; // the seeds of a committed table's record

// What a round's verifier's message consists of. The number is part of the round's query, so an
// answer found for one kind of round, or for another `work_bits`, answers a different query. A
// number must therefore never get a second meaning.
// ALGEBRAIC: elements of E, the cubic extension of the field F. EPSILON: the element that joins a
// sumcheck's hiding polynomial to a claim. SEED: seeds of out-of-domain samples. POSITIONS:
// indices into codewords, then in some rounds one element of E. BITS: entries of a matrix over
// {-1, 0, 1}. SEAL: a digest that ends a chain on which other chains build. EMPTY: nothing.
enum class ChainKind : u32 { ALGEBRAIC = 1, EPSILON = 2, SEED = 3, POSITIONS = 4, BITS = 5, SEAL = 6, EMPTY = 7 };

// What is fixed before the prover speaks. Only a test (blocks/chain/tests/rounds_test.cpp) fills
// `record`; Binder::first_query() in blocks/chain/public.h leaves it empty. Other code puts a
// table's record into an item of `statement` instead: `IndexName` in blocks/iop/foldeq.h.
// Source for hashing parameters and statement into every challenge (the strong Fiat-Shamir transformation): Dao,
// Miller, Wright, Grubbs, "Weak Fiat-Shamir Attacks on Modern Proof Systems", Cryptology ePrint Archive 2023/691
// (IEEE S&P 2023), section II-E, Definition 3. There the whole statement enters from the first challenge on; here a
// request's output token enters later, as a prover's message (verifier/src/cp_verify.cpp).
struct FirstQuery {
  u64 label = 0;             // which protocol the chain belongs to
  Digest parameters{};       // the hash of all parameters: field, hash, codes, numbers of positions, work bits, ..
  std::vector<u8> statement; // what is proved
  std::vector<u8> record;    // the record of the statement's committed table; only a test fills it
  // The chain's salt: 32 random bytes. The first query contains not the salt but the run's value, a
  // hash of label, parameters and salt. The leaves of the trees that a chain sends are hashed with
  // the run's value, so they can be hashed before the statement is known.
  //   salted, salt_in_proof   the prover draws the salt in private and writes it first among the
  //                           chain's bytes in the proof. A chain whose messages hide something is
  //                           of this kind.
  //   salted, not in proof    both sides supply the same 32 bytes, fresh in every run but known to
  //                           all who see the proof (such as the root of a tree of salted leaves).
  //                           They hide nothing. For a chain that several proofs must derive alike.
  //   not salted              no salt and no run's value. Such a chain can send no tree.
  // Source for a salt in the Fiat-Shamir query: Chiesa, Fenzi, "zkSNARKs in the ROM with Unconditional UC-Security",
  // Cryptology ePrint Archive 2024/724 (version of September 4, 2024), sections 2.4 and 2.5. There every round has a
  // fresh salt.
  // Source for a single salt at the start: Chiesa, Orrù, "A Fiat–Shamir Transformation From Duplex Sponges",
  // Cryptology ePrint Archive 2025/536 (version of March 27, 2026), section 2.5, "One salt suffices", and Theorem
  // 7.1. There for a sponge over an ideal permutation; for a hash chain only its Remark 2.2, without a proof.
  bool salted = true, salt_in_proof = true;
  Digest salt{};
};
// The run's value of a salted chain.
template <class Fn = HashFn> inline Digest first_query_run(const FirstQuery &q) { return run_value<Fn>(q.label, q.parameters, q.salt); }

// Draws. A verifier's message is drawn from its row of words: the 64-bit words of its answer blocks
// (see `ChainT`), four a block, in order. Each draw takes the next words.
//   element of F   the next word that `F_from_word` accepts (rejection sampling: a word is never
//                  reduced modulo p); after DRAW_TRIES rejected words the element is 0
//   element of E   three elements of F
//   nonzero E      the first of NONZERO_TRIES elements of E that is not 0, else 1
//   position       in [0, 2^k): the low k bits of the next word
// With these caps (both constants are in blocks/base.h) a verifier's message is a total function of
// a bounded number of words: at most 18 for an element of E, 36 for a nonzero one. For one element
// of E a cap is reached with probability below 3 * 2^-192 (p = 2^64 - 2^32 + 1); whole_error() in
// blocks/numbers/plan.h adds that, for each element drawn, to the round's error.
// These rules are for the verifier's messages only: other functions, with other rules, generate the
// prover's private randomness.
//
// The rules of a draw, over any source of the row's words: next() gives the next word. One text for every form of a round.
template <class Next> BLK_FN F draw_F_from(Next &next) {
  F x;
  for (unsigned t = 0; t < DRAW_TRIES; t++)
    if (F_from_word(next(), x)) return x;
  return F::zero();
}
template <class Next> BLK_FN E draw_E_from(Next &next) {
  E x;
  for (int i = 0; i < 3; i++) x.c[i] = draw_F_from(next);
  return x;
}
// Fast path for the first element of a verifier's message, from the first three words of its row.
// True, with the element in `out`, if no word is rejected and the element is not a 0 that `nonzero`
// forbids; `out` is then what the rules above give. False: nothing is consumed, and the caller
// draws by the rules from the row's first word. The fast path saves a kernel the word counter and
// the test for a further hash call.
BLK_FN bool draw_E_straight(u64 w0, u64 w1, u64 w2, bool nonzero, E &out) {
  E x;
  const bool taken = (int)F_from_word(w0, x.c[0]) & (int)F_from_word(w1, x.c[1]) & (int)F_from_word(w2, x.c[2]);
  if (!taken || (nonzero && x == E::zero())) return false;
  out = x;
  return true;
}
template <class Next> BLK_FN E draw_E_nonzero_from(Next &next) {
  for (unsigned t = 0; t < NONZERO_TRIES; t++) {
    const E x = draw_E_from(next);
    if (x != E::zero()) return x;
  }
  return E::one();
}

// BLK_RARE: a function for host and GPU that a kernel almost never reaches. BLK_FN
// (blocks/hash/tweak.h) forces inlining on the GPU; BLK_RARE asks the compiler not to inline
// (__noinline__: NVIDIA, "CUDA Programming Guide", "C/C++ Language Extensions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).
#if defined(__CUDACC__)
#define BLK_RARE __host__ __device__ __noinline__
#else
#define BLK_RARE inline
#endif

// `Fn` supplies the hash's compression function; `Chain`, below, has BLAKE2b's (`HashFn`). Except
// for the constructor from a `FirstQuery`, which is host-only, a chain is plain data whose
// functions run on host and GPU. A kernel can take a copy, run rounds without work on it and
// return it: ProverRounds::lend() and take_back() in blocks/chain/rounds.h. `ChainAbsorbT` (an open
// message), `ChainPrefixT` (a query without its nonce) and `ChainCall` (the 64 bytes of one hash
// call) are declared in blocks/hash/hash.h.
// A failed guard (BLK_REQUIRE in blocks/base.h, BLK_HASH_REQUIRE in blocks/hash/tweak.h) marks a
// caller's mistake and is fatal: it ends the host process or traps the kernel. A failed work test,
// which can depend on a proof's bytes, is not fatal: step() returns false.
//
// A round's answer is a sequence of 32-byte blocks B_0, B_1, .. Blocks 2 j and 2 j + 1 are the
// halves of hash call j, chain_call() in blocks/hash/hash.h: 64 bytes, on the query with j written
// into it. A call is made only when one of its blocks is needed.
//     without work:   B_0 = rho_i, the next value;           B_1, B_2, ..  the verifier's message
//     with work:      B_0 = the test block;   B_1 = rho_i;   B_2, B_3, ..  the verifier's message
// A round without work that sends up to four elements of E and draws one costs one compression,
// unless at least two of its first four words are rejected: rho_{i-1} goes into the hash's initial
// state, and 16 bytes of the hash's own label (domain number, the chain's label, i), 96 of message
// and 16 of kind, `work_bits` and nonce fill one 128-byte block.
template <class Fn> class ChainT {
public:
  ChainT() = default; // no chain yet, only a target of assignment: a round or a draw on it is fatal
  explicit ChainT(const FirstQuery &q) : rounds_(0), label_(q.label) {
    BLK_REQUIRE((q.label >> 48) == 0, "chain: a label is below 2^48");
    HasherT<Fn> h(TagT<Fn>(Domain::CHAIN_START), q.label);
    h.push(q.parameters);
    h.push_u64(q.statement.size()); // the length prefixes make the encoding unambiguous
    h.push(q.statement.data(), q.statement.size());
    h.push_u64(q.record.size());
    h.push(q.record.data(), q.record.size());
    h.push_u64(q.salted ? 1 : 0);
    if (q.salted) h.push(first_query_run<Fn>(q));
    rho_ = h.finish();
  }

  // Starts the next round's query; the caller pushes the prover's message, a multiple of 8 bytes
  // (otherwise prefix() is fatal).
  BLK_FN ChainAbsorbT<Fn> message() const { return ChainAbsorbT<Fn>(rho_, next_round(), label_); }
  // Ends the message and returns the entire query except its nonce. step() must be given the same
  // `work_bits`; nothing checks that (Rounds::ask() in blocks/chain/rounds.h takes both from one
  // entry of the plan, the list of all rounds: blocks/numbers/plan.h).
  BLK_FN static ChainPrefixT<Fn> prefix(ChainAbsorbT<Fn> &message, ChainKind kind, unsigned work_bits) {
    BLK_HASH_REQUIRE(work_bits < 256, "chain: a round's work bits have one byte of the query's last word");
    message.push_u64((u64)kind | (u64)work_bits << 8);
    return message.prefix();
  }
  // Makes the round's query. A round may require w = `work_bits` bits of proof of work: its answer
  // counts only if the low w bits of the test block's first word are 0. The prover tries nonces
  // until that holds (a try is hash call 0 only), and the nonce goes into the proof. A malicious
  // prover who hopes for a lucky verifier's message pays about 2^w queries for each try, so the
  // round's error is counted as 2^-w times what it is without work (docs/limits-all-rows.tsv, row
  // SO-32).
  // `work_bits` is at most 64: prefix() accepts up to 255, but above 64 the work test is fatal.
  // The work is tested first: if it fails, step() returns false, the chain is unchanged and nothing
  // is drawn from the answer.
  BLK_FN bool step(const ChainPrefixT<Fn> &p, u64 nonce, unsigned work_bits) {
    BLK_HASH_REQUIRE(work_bits > 0 || nonce == 0, "chain: a round without work has no nonce");
    const ChainCall first = chain_call(p, nonce, 0);
    if (!has_work(first, work_bits)) return false;
    take(p, nonce, work_bits, first);
    return true;
  }
  // The work test, on call 0 of a query.
  BLK_FN static bool has_work(const ChainCall &first, unsigned work_bits) { return work_test(first.w[0], work_bits); }
  // prefix() and step() in one, for a round without work.
  BLK_FN void close(ChainAbsorbT<Fn> &message, ChainKind kind) { step(prefix(message, kind, 0), 0, 0); }
  // An entire round without work whose message is exactly `words`.
  template <unsigned N> BLK_FN void round(const u64 (&words)[N], ChainKind kind) {
    ChainAbsorbT<Fn> m = message();
    BLK_UNROLL
    for (unsigned i = 0; i < N; i++) m.push_u64(words[i]);
    close(m, kind);
  }
  BLK_FN void round(ChainKind kind) { // with an empty message
    ChainAbsorbT<Fn> m = message();
    close(m, kind);
  }
  // For a kernel: round<N>() and the draws of all D elements of the verifier's message (nonzero
  // ones if `nonzero` is set) in one call, with the same query, answer and elements. The chain
  // stores no query: the message stays in the caller's registers for the answer's further hash
  // calls, which come only after rejected words or for a long verifier's message. So the verifier's
  // message is over when the call returns, as after shut().
  template <unsigned N, unsigned D> BLK_FN void round_and_draw(const u64 (&words)[N], ChainKind kind, E (&out)[D], bool nonzero = false) {
    CellsRow<N + 1> row;
    BLK_UNROLL
    for (unsigned i = 0; i < N; i++) row.cells[i] = words[i];
    row.cells[N] = (u64)kind; // and 0 bits of proof of work
    row.before = rho_, row.round = next_round(), row.label = label_;
    row.call = chain_call_cells<Fn>(row.before, row.round, row.cells, 0, 0, row.label);
    take_shut(chain_block(row.call, 0));
    if (D == 1 && draw_E_straight(row.call.w[4], row.call.w[5], row.call.w[6], nonzero, out[0])) return;
    const Drawn<D> drawn = draw_by_the_rules<N + 1, D>(row, nonzero);
    BLK_UNROLL
    for (unsigned i = 0; i < D; i++) out[i] = drawn.e[i];
  }
  // The same for a round whose verifier's message is empty or is not needed by this copy. Only a
  // test calls it (blocks/chain/tests/border_test.cpp).
  template <unsigned N> BLK_FN void round_and_shut(const u64 (&words)[N], ChainKind kind) {
    u64 cells[N + 1];
    BLK_UNROLL
    for (unsigned i = 0; i < N; i++) cells[i] = words[i];
    cells[N] = (u64)kind;
    take_shut(chain_block(chain_call_cells<Fn>(rho_, next_round(), cells, 0, 0, label_), 0));
  }
  // Advances the chain to `value`, unchecked, where the caller has made the round's hash call and
  // draws its verifier's message itself (round_and_draw(); chain_lanes.cuh). No verifier calls it.
  // A prover's copy that took in a wrong value is detected when it is returned:
  // ProverRounds::take_back() in rounds.h.
  BLK_FN void take_shut(const Digest &value) {
    rho_ = value;
    rounds_ = next_round();
    shut();
  }
  // Ends the current verifier's message: this copy draws again only after its next round.
  BLK_FN void shut() { next_ = 0, have_ = SHUT; }

  BLK_FN u64 rounds() const { return rounds_; }
  BLK_FN u64 label() const { return label_; } // see `FirstQuery`; it is part of every query
  // The chain's value rho, for comparing two copies of a chain.
  BLK_FN const Digest &value() const { return rho_; }

  BLK_FN u64 next_word() {
    BLK_HASH_REQUIRE(rounds_ > 0 && rounds_ != NO_CHAIN, "chain: a draw before the first round");
    BLK_HASH_REQUIRE(have_ != SHUT, "chain: this copy's verifier message is over; it draws again after its next round");
    if (next_ == have_) { // the next two blocks of the answer
      const ChainCall c = chain_call(query_, nonce_, call_++);
      BLK_UNROLL
      for (int i = 0; i < 8; i++) word_[i] = c.w[i];
      next_ = 0, have_ = 8;
    }
    return word_[next_++];
  }
  BLK_FN F draw_F() {
    Row row{this};
    return draw_F_from(row);
  }
  BLK_FN E draw_E() {
    Row row{this};
    return draw_E_from(row);
  }
  BLK_FN E draw_E_nonzero() {
    Row row{this};
    return draw_E_nonzero_from(row);
  }
  BLK_FN u64 draw_position(unsigned log_n) {
    BLK_HASH_REQUIRE(log_n < 64, "chain: positions lie below 2^63");
    return next_word() & (((u64)1 << log_n) - 1);
  }

  // Advances the chain by one round, given the round's query and its hash call 0. It does not check
  // that `first` belongs to this query, that the query is this chain's next, or the work. Only
  // step() calls it.
  BLK_FN void take(const ChainPrefixT<Fn> &p, u64 nonce, unsigned work_bits, const ChainCall &first) {
    rho_ = chain_block(first, work_bits > 0 ? 1 : 0);
    rounds_ = next_round();
    query_ = p, nonce_ = nonce, call_ = 1;
    next_ = 0, have_ = 0;
    if (work_bits == 0) { // hash call 0's second half is the first block of the verifier's message
      BLK_UNROLL
      for (int i = 0; i < 4; i++) word_[i] = first.w[4 + i];
      have_ = 4;
    }
  }

private:
  // The row of words of the current verifier's message, computed from the stored query.
  struct Row {
    ChainT *chain;
    BLK_FN u64 operator()() { return chain->next_word(); }
  };
  // The row of words of a round without work whose message is in `cells`: words 4 to 7 of hash call
  // 0, then all words of calls 1, 2, ..
  template <unsigned CELLS> struct CellsRow {
    u64 cells[CELLS];
    Digest before;
    u64 round, label, next_call = 1;
    ChainCall call;
    unsigned at = 4;
    BLK_FN u64 operator()() {
      if (at == 8) call = chain_call_cells<Fn>(before, round, cells, 0, next_call++, label), at = 0;
      return call.w[at++];
    }
  };
  // Draws all D elements of a round by the rules. Not inlined on a GPU: with a real hash a kernel
  // almost never gets here when D = 1, and inlined this would be most of a kernel's code, because
  // every word of every try may need a further hash call. The row goes in and the elements come out
  // by value: no address of a caller's variable is taken, so the fast path's values can stay in
  // registers.
  template <unsigned D> struct Drawn {
    E e[D];
  };
  template <unsigned CELLS, unsigned D> BLK_RARE static Drawn<D> draw_by_the_rules(CellsRow<CELLS> row, bool nonzero) {
    Drawn<D> out;
    for (unsigned i = 0; i < D; i++) out.e[i] = nonzero ? draw_E_nonzero_from(row) : draw_E_from(row);
    return out;
  }
  static constexpr u64 NO_CHAIN = ~(u64)0;
  static constexpr unsigned SHUT = 0xFF;
  BLK_FN u64 next_round() const {
    BLK_HASH_REQUIRE(rounds_ != NO_CHAIN, "chain: a chain that was made from no first query and is no copy of one");
    return rounds_ + 1;
  }

  // The chain's value; the rounds made so far (NO_CHAIN in a default-constructed chain); the label.
  Digest rho_{};
  u64 rounds_ = NO_CHAIN, label_ = 0;
  // The current round's query and nonce, kept for the further blocks of its answer. `call_`: the
  // next hash call's number. `word_`: computed words, of which next_ to have_ - 1 are not drawn
  // yet. have_ = SHUT: the verifier's message is over.
  ChainPrefixT<Fn> query_{};
  u64 nonce_ = 0, call_ = 0;
  u64 word_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  unsigned next_ = 0, have_ = 0;
};
using Chain = ChainT<HashFn>;
static_assert(std::is_trivially_copyable<Chain>::value && std::is_trivially_destructible<Chain>::value &&
                  std::is_trivially_copyable<ChainAbsorb>::value && std::is_trivially_copyable<ChainPrefix>::value,
              "a chain, an open message and a query go to a kernel as plain bytes");

// The prover's search on the CPU: the smallest nonce from `from` on whose answer passes the work
// test. It takes about 2^`work_bits` tries on average. A try costs one compression, or two if the
// nonce begins a new 128-byte block of the hash's input.
inline u64 search_nonce(const ChainPrefix &p, unsigned work_bits, u64 from = 0) {
  if (work_bits == 0) return 0;
  for (u64 nonce = from;; nonce++)
    if (Chain::has_work(chain_call(p, nonce, 0), work_bits)) return nonce;
}

} // namespace blk
