// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The prover's private randomness (`coin` in identifiers and messages): random tails of encoded
// rows, mask rows, pads, hiding polynomials, and the salts of tree leaves and hash chains. Keys are
// derived from a secret seed; under such a key StreamCoinsT implements `Coins`
// (blocks/engine/table.h). The only cryptographic primitive that the code here calls is f, the
// ChaCha20 block function (blocks/coins/chacha.h). The hash of chains and trees enters only through
// the digests that process_key() and making_key() take in: CoinInputs (blocks/coins/run_coins.h)
// makes them with that hash, under a domain-separation number of its own, Domain::COIN_INPUTS.
// No verifier and no proof sees how these values are made. Soundness assumes nothing of them;
// hiding depends on them, and no verifier can check them. Hiding requires:
//   1. The seed is uniform, secret and used for nothing else; keys derived from it are as secret.
//      A key's holder must wipe it after use: CoinKey.
//   2. Under one key no label (a value's name) is used twice with different inputs: process_key().
//   3. Under a run's process key no context (one proof or message) is used twice: CoinContext.
//   4. A committed table's lasting values come from a secret drawn at its creation: making_key().
//   5. f is a pseudo-random function: with derived values hiding is computational at best.
// Tests: blocks/coins/tests/coins_test.cpp (f itself on RFC 8439's vector, in rfc_vector(); keys,
// contexts, words and elements on values of blocks/coins/tests/coin_values.py, which derives them
// again in Python and cannot show a fault in f that it shares; rejected words, real and forced;
// ledger(); the operating system's source).
#pragma once
#include <atomic>
#include <cerrno>
#include <cstring>
#include <sys/random.h>
#include <thread>
#include <vector>

#include "coins/chacha.h"
#include "engine/table.h"
#include "field/field.h"
#include "hash/hash.h"

namespace blk {

// A ChaCha20 key. Its holder wipes it after use.
struct CoinKey {
  u32 w[8];
};
inline void wipe(CoinKey &k) { wipe(k.w, sizeof k.w); }  // `wipe` of n bytes: blocks/base.h

// key(K; a, b, c, d): words 0 to 7 of f(K; a, b, c, d). Host only: a kernel gets its context's key.
inline CoinKey coin_key(const CoinKey &K, u32 a, u32 b, u32 c, u32 d) {
  u32 block[16];
  chacha20_block(block, K.w, a, b, c, d);
  CoinKey out;
  for (int i = 0; i < 8; i++) out.w[i] = block[i];
  wipe(block, sizeof block);
  return out;
}
// K := key(K; four words) over n words, n a multiple of four.
// Public source: a chain of such calls, on inputs none of which begins another, is the cascade construction of:
// Bellare, Canetti and Krawczyk, "Pseudorandom functions revisited: the cascade construction and its concrete
// security", FOCS 1996, pages 514 to 523. That paper's argument has not been written out for this chain.
inline void coin_key_take(CoinKey &K, const u32 *words, size_t n) {
  BLK_REQUIRE(n % 4 == 0, "coins: a key takes words in by fours");
  for (size_t i = 0; i < n; i += 4) {
    CoinKey next = coin_key(K, words[i], words[i + 1], words[i + 2], words[i + 3]);
    K = next;
    wipe(next);
  }
}
inline CoinKey coin_key_from_bytes(const u8 b[32]) { // a key's 32 bytes are its eight words, lowest byte first
  CoinKey k;
  for (int i = 0; i < 8; i++) k.w[i] = b[4 * i] | (u32)b[4 * i + 1] << 8 | (u32)b[4 * i + 2] << 16 | (u32)b[4 * i + 3] << 24;
  return k;
}

// A child key of the seed: key(seed; "CPSD", kind, number, 0). A kind must have one use only; those
// in `cp_commit` and `cp_prove` are listed in docs/format/COIN-KEYS.md, "2. The schedule".
constexpr u32 COIN_SEED_WORD = 0x44535043; // "CPSD"
inline CoinKey seed_child(const CoinKey &seed, u32 kind, u32 number) { return coin_key(seed, COIN_SEED_WORD, kind, number, 0); }

// 32 fresh bytes from getrandom(); a call that fails with EINTR (a signal interrupted it) is
// repeated. If the call fails otherwise, gives fewer bytes or gives only zeros, as a dead source
// does, the process ends: no value is derived from such bytes.
// Source: a function with getrandom()'s arguments, so that a test can substitute a faulty one.
template <class Source> inline void fresh32(u8 out[32], Source source) {
  ssize_t given;
  do given = source(out, 32, 0);
  while (given < 0 && errno == EINTR);
  u8 any = 0;
  if (given == 32)
    for (int i = 0; i < 32; i++) any |= out[i];
  if (any) return;
  wipe(out, 32);
  BLK_REQUIRE(given != 32, "coins: the system's random generator (getrandom) gave 32 zero bytes, which a sound generator does not do");
  BLK_REQUIRE(false, "coins: the system's random generator (getrandom) failed or gave fewer than 32 bytes");
}
inline void fresh32(u8 out[32]) {
  fresh32(out, [](void *p, size_t n, unsigned flags) { return ::getrandom(p, n, flags); });
}

// The process key of one run: X = child, then X = key(X; four words) over the eight words of
// `fresh` and the eight of `inputs`. child: a child key for runs of this kind only. fresh: 32 bytes
// from fresh32(). inputs: a digest of everything that the run's messages depend on. The caller
// wipes child and fresh. `cp_commit` and `cp_prove` also compute a table's making key with this
// function; `fresh` is then what making_key() says of its own `fresh`.
// Condition 2 at the top of this file must hold for the key's whole life: retries, restarts, a
// second process, a restored or copied directory. The same values with the same inputs give the
// same proof, which is harmless; under other challenges they reveal masked values. A counter can be
// set back, hence `fresh`; if the operating system's source repeats, different inputs still get
// different values.
inline CoinKey process_key(const CoinKey &child, const u8 fresh[32], const Digest &inputs) {
  CoinKey X = child, f = coin_key_from_bytes(fresh);
  coin_key_take(X, f.w, 8);
  coin_key_take(X, inputs.w, 8);
  wipe(f);
  return X;
}
// A committed table's making key: the one secret from which its masks (its rows' random tails),
// its tree leaves' salts and its pads (each masks an out-of-domain
// sample's answer) all come. They outlive a run, so a run's process key cannot serve. Derived as in
// process_key(), with `table` for `inputs`: a digest of public facts, namely the table's
// configuration, its shape and which table it is. child: a child key for tables only.
// Hiding rests on a secret drawn fresh for this one creation of the table: the same values on other
// cells would reveal the cells' difference. For a leaf table (a further table of one commitment)
// the secret is `fresh`, which is kept with its secret files; for the anchor (the first table) it
// is the seed itself, and `fresh` is zeros.
// `table` covers nothing of the cells: it only guards against a source that repeats, since another
// configuration, shape or table still gets other values.
// The same three inputs on the same cells give the same encoded table, salts and root, so anyone
// who regenerates a table must carry on with the same count of its openings: a row's masks cover
// only a fixed number of opened columns (blocks/code/code.h, hiding_positions()).
// Only tests call making_key(): the programs `cp_commit` and `cp_prove` compute the same key with
// process_key() (prover/common/coin_keys.h, from()).
inline CoinKey making_key(const CoinKey &child, const u8 fresh[32], const Digest &table) {
  CoinKey X = child, f = coin_key_from_bytes(fresh);
  coin_key_take(X, f.w, 8);
  coin_key_take(X, table.w, 8);
  wipe(f);
  return X;
}

// A value's label is sixteen 32-bit words, which enter four at a time (K_proc: the process key):
//   K1 = key(K_proc; w4 .. w7),  K2 = key(K1; w8 .. w11),  K_ctx = key(K2; w12 .. w15),
//   block = f(K_ctx; w0, w1, w2, w3).
// w0 = call_lo, w1 = stream, w2 = retry, w3 = call_hi name the call. w4 .. w15 name the context,
// one proof or message with random values of its own: the fields below, all other words 0.
// `version`: the format's version word; `object`: the kind of object; `counter`, `member`: numbers
// of the caller's choice (in `cp_prove`, `member` is a request's slot in its batch).
// Under one process key a context serves one generator only: a second would start again at the
// first stream and repeat the values. That is the caller's duty: nothing here checks it.
// prover/common/coin_chacha.h uses the same labels: under a shared process key the two generators'
// `object` words must differ.
struct CoinContext {
  u64 counter = 0; // w4, w5
  u32 version = 0; // w10
  u32 object = 0;  // w11
  u32 member = 0;  // w12
};
inline CoinKey context_key(const CoinKey &process, const CoinContext &c) {
  CoinKey k1 = coin_key(process, (u32)c.counter, (u32)(c.counter >> 32), 0, 0), k2 = coin_key(k1, 0, 0, c.version, c.object);
  const CoinKey k = coin_key(k2, c.member, 0, 0, 0);
  wipe(k1), wipe(k2);
  return k;
}

// ---- from a context's key to a coin: host and GPU ----
// A call gives four 64-bit words, the lanes, from words 0 to 7 of its block. Word i of a stream is
// lane i mod 4 of call i div 4, at retry 0. Bytes (salts, which need only entropy) are these words,
// lowest byte first.
// A field element must be exactly uniform, so no word is reduced modulo p: element i is word i if
// F_from_word() (blocks/field/field.h) accepts it (`admit`, `refuse` in identifiers and messages);
// otherwise it is lane 0 of the same call at retry 8 k + lane, for the first k = 1, 2, .. that
// gives an accepted word. With p = 2^64 - 2^32 + 1 about one word in 2^32 is rejected, so large
// proofs do take this path.
// Any value can be computed alone from its key and label.
constexpr u32 COIN_FIRST_STREAM = 16384;
constexpr unsigned COIN_LANES = 4; // words of 64 bits that a call gives
// The block function as a type, so that a test can substitute another and force rejections.
struct ChaChaBlock {
  BLK_FN void operator()(u32 out[16], const u32 key[8], u32 a, u32 b, u32 c, u32 d) const { chacha20_block(out, key, a, b, c, d); }
};
// Lane `lane` of call `call` of a stream, at retry word `retry`.
template <class Block = ChaChaBlock>
BLK_FN u64 coin_lane(const CoinKey &context, u32 stream, u64 call, u32 retry, unsigned lane, Block f = Block()) {
  u32 o[16];
  f(o, context.w, (u32)call, stream, retry, (u32)(call >> 32));
  const u64 l0 = o[0] | (u64)o[1] << 32, l1 = o[2] | (u64)o[3] << 32, l2 = o[4] | (u64)o[5] << 32, l3 = o[6] | (u64)o[7] << 32;
  return lane == 0 ? l0 : lane == 1 ? l1 : lane == 2 ? l2 : l3;
}
template <class Block = ChaChaBlock> BLK_FN u64 coin_word(const CoinKey &context, u32 stream, u64 i, Block f = Block()) {
  return coin_lane(context, stream, i / COIN_LANES, 0, (unsigned)(i % COIN_LANES), f);
}
// Element i, given its first word. The word modulo p would not be uniform: a fault that harms
// hiding and that no verifier can see.
template <class Block = ChaChaBlock> BLK_FN F coin_admit(u64 word, const CoinKey &context, u32 stream, u64 i, Block f = Block()) {
  F x;
  for (u32 k = 1; !F_from_word(word, x); k++) {
    BLK_HASH_REQUIRE(!(k >> 29), "coins: no retry word is left");
    word = coin_lane(context, stream, i / COIN_LANES, 8 * k + (u32)(i % COIN_LANES), 0, f);
  }
  return x;
}
template <class Block = ChaChaBlock> BLK_FN F coin_element(const CoinKey &context, u32 stream, u64 i, Block f = Block()) {
  return coin_admit(coin_word(context, stream, i, f), context, stream, i, f);
}

// A program's own test of the generator, to be run at its start, before a seed or a key has reached
// it. It checks RFC 8439's block (2.3.2) and, under a key that is written here, a word that
// F_from_word() accepts and one whose retry is forced. A difference ends the process: a generator
// that is miscompiled or runs on faulty hardware would still give proofs that verify, and hide
// nothing.
inline void coins_own_test() {
  u32 key[8], out[16];
  for (u32 i = 0; i < 8; i++) key[i] = 0x03020100u + 0x04040404u * i;
  chacha20_block(out, key, 1, 0x09000000u, 0x4a000000u, 0);
  const u32 want[16] = {0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
                        0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9, 0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2};
  BLK_REQUIRE_NAMED(!std::memcmp(out, want, sizeof want), "own-test-coins", "coins: the generator's own test fails: ChaCha20's block is not RFC 8439's");
  CoinKey k;
  for (int i = 0; i < 8; i++) k.w[i] = key[i];
  struct FirstRefused { // every lane at retry 0 is 64 ones, which the field rejects
    BLK_FN void operator()(u32 o[16], const u32 K[8], u32 a, u32 b, u32 c, u32 d) const {
      chacha20_block(o, K, a, b, c, d);
      if (c == 0)
        for (int i = 0; i < 8; i++) o[i] = ~0u;
    }
  };
  F plain, retried;
  BLK_REQUIRE_NAMED(F_from_word(coin_lane(k, COIN_FIRST_STREAM, 0, 8 + 2, 0), retried) && F_from_word(coin_word(k, COIN_FIRST_STREAM, 2), plain), "own-test-coins",
              "coins: the generator's own test fails: its two words are not admitted");
  BLK_REQUIRE_NAMED(coin_element(k, COIN_FIRST_STREAM, 2) == plain && coin_element(k, COIN_FIRST_STREAM, 2, FirstRefused()) == retried && plain != retried, "own-test-coins",
              "coins: the generator's own test fails: an element is not its stream's word, or a refused word is not drawn again by the rule");
}

// ---- the coins of one context ----
struct CoinDraw { // an entry of `ledger`: one draw (a call of `open`) of `count` elements or bytes
  u32 stream;
  bool elements; // field elements, or bytes
  u64 count;
};
// The generator of one context. Every call of fill() or bytes() opens a new stream: streams are
// numbered from COIN_FIRST_STREAM in the order of these calls, so none is opened twice, and which
// stream serves what follows from the caller's order of draws. ledger() lists the draws. In
// `cp_prove` that order is meant to follow from public shapes, not from a request's values:
// prover/tests/coin_contexts_listed.py checks, on its own runs, that the requests' lists are equal.
// Base: `Coins`, or a class derived from it that adds calls, such as `CardCoins`
// (prover/engine/card_coins.cuh), whose calls write into GPU memory.
template <class Base = Coins, class Block = ChaChaBlock> class StreamCoinsT : public Base {
public:
  StreamCoinsT(const CoinKey &process, const CoinContext &context, Block f = Block())
      : f_(f), key_(context_key(process, context)), holder_(std::this_thread::get_id()) {}
  ~StreamCoinsT() override { wipe(key_); }
  StreamCoinsT(const StreamCoinsT &) = delete; // a copy would give the same coins a second time
  StreamCoinsT &operator=(const StreamCoinsT &) = delete;

  // One thread at a time. Streams are numbered by the length of `ledger_`, and last_stream() refers
  // to the latest draw: two threads drawing at once could get one number, that is the same values
  // for two objects, and no verifier would see it. So one thread holds the generator, at first the
  // one that constructed it; a call by another thread that draws, regenerates values or asks for
  // the last stream ends the process. ledger() checks nothing: read it after the last draw. The
  // generator is handed over, not taken, as an `Engine` is (blocks/engine/engine.h): the holder's
  // last call is let_go(), the next thread's first call take_over(). take_over() does not wait: it
  // ends the process unless no thread holds the generator, so the two threads must order these
  // calls themselves.
  void let_go() {
    held();
    holder_.store(std::thread::id());
  }
  void take_over() {
    std::thread::id nobody;
    BLK_REQUIRE(holder_.compare_exchange_strong(nobody, std::this_thread::get_id()), "coins: taken over while a thread holds the generator: its holder calls let_go() first");
  }

  void fill(F *out, size_t n) override { elements(open(true, n), 0, out, n); }
  void bytes(u8 *out, size_t n) override { raw(open(false, n), 0, out, n); }

  // The stream of the last fill() or bytes().
  u32 last_stream() const {
    held();
    BLK_REQUIRE(!ledger_.empty(), "coins: nothing was drawn yet");
    return ledger_.back().stream;
  }
  // The two queries of `Coins`, which serve not_as_drawn() (blocks/iop/send.h). not_the_coins()
  // returns how many of the n values differ from the stream's; it skips the indices in `left_out`,
  // which must be strictly ascending (not checked). It regenerates the draw with this same code, so
  // it detects a value that was lost or overwritten after its draw, not a fault in the generator
  // itself. EngineCoins (prover/hiding/engine_coins.cuh), the generator of `cp_commit` and
  // `cp_prove`, overrides it with a second implementation. That one's block function follows the
  // same public source as chacha20_block(), so it does not detect a fault taken from there.
  u32 last_draw() const override { return last_stream(); }
  u64 not_the_coins(u32 stream, const F *values, u64 n, const std::vector<u64> &left_out) const override {
    std::vector<F> anew(n);
    again(stream, 0, anew.data(), n);
    u64 differ = 0;
    for (u64 i = 0, skip = 0; i < n; i++) {
      if (skip < left_out.size() && left_out[skip] == i) skip++;
      else differ += !(anew[i] == values[i]);
    }
    wipe(anew.data(), n * sizeof(F));
    return differ;
  }
  // Regenerates values first .. first + n - 1 of an earlier stream, for a caller that notes
  // last_stream() and recomputes a random tail or a salt rather than keep it. In the second form
  // first and n count bytes. A range that was not drawn, or not as this type, ends the process.
  void again(u32 stream, u64 first, F *out, size_t n) const { elements(drawn(stream, true, first, n), first, out, n); }
  void again(u32 stream, u64 first, u8 *out, size_t n) const { raw(drawn(stream, false, first, n), first, out, n); }
  const std::vector<CoinDraw> &ledger() const { return ledger_; }

protected:
  u32 open(bool elements_, u64 count) {
    held();
    BLK_REQUIRE(ledger_.size() < ((size_t)1 << 30), "coins: a context has fewer than 2^30 streams");
    ledger_.push_back(CoinDraw{COIN_FIRST_STREAM + (u32)ledger_.size(), elements_, count});
    return ledger_.back().stream;
  }
  u32 drawn(u32 stream, bool elements_, u64 first, u64 n) const {
    held();
    BLK_REQUIRE(stream >= COIN_FIRST_STREAM && stream - COIN_FIRST_STREAM < ledger_.size(), "coins: this stream was never drawn");
    const CoinDraw &d = ledger_[stream - COIN_FIRST_STREAM];
    BLK_REQUIRE(d.elements == elements_ && first <= d.count && n <= d.count - first, "coins: these coins were never drawn");
    return stream;
  }
  const CoinKey &key() const { return key_; }

private:
  void held() const {
    BLK_REQUIRE(holder_.load() == std::this_thread::get_id(), "coins: a call by a thread that does not hold the generator: let_go() by its holder, then take_over()");
  }
  void elements(u32 stream, u64 first, F *out, size_t n) const {
    for (size_t j = 0; j < n;) { // a call's lanes together
      const u64 i = first + j, call = i / COIN_LANES;
      u32 o[16];
      f_(o, key_.w, (u32)call, stream, 0, (u32)(call >> 32));
      for (unsigned lane = (unsigned)(i % COIN_LANES); lane < COIN_LANES && j < n; lane++, j++)
        out[j] = coin_admit(o[2 * lane] | (u64)o[2 * lane + 1] << 32, key_, stream, first + j, f_);
      wipe(o, sizeof o);
    }
  }
  void raw(u32 stream, u64 first, u8 *out, size_t n) const { // first, n: in bytes
    u64 word = 0;
    for (size_t j = 0; j < n; j++) {
      const u64 at = first + j;
      if (j == 0 || at % 8 == 0) word = coin_word(key_, stream, at / 8, f_);
      out[j] = (u8)(word >> 8 * (at % 8));
    }
    wipe(&word, sizeof word);
  }
  Block f_;
  CoinKey key_;
  std::vector<CoinDraw> ledger_;
  std::atomic<std::thread::id> holder_;
};
using StreamCoins = StreamCoinsT<>;

} // namespace blk
