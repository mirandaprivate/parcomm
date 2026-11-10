// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The hash interface of blocks/, for host code and for GPU kernels. Every hash made here is BLAKE2b
// as specified (sequential, no key) on a whole string:
//     H(tag, index, start; bytes) = BLAKE2b(label, bytes) with start as salt and personalization,
// where label is 16 bytes, the tag's word (TagT::word) and then the index (below 2^63), and start
// is a digest. Whether a hash has a start depends only on the domain number in its tag (`Domain`,
// blocks/hash/domains.h); a hash without a start is BLAKE2b without salt and personalization.
// In Python (the tag word of a Merkle tree's leaf has more fields, and a call of a round of a
// Fiat-Shamir hash chain has digest_size=64):
//     hashlib.blake2b(struct.pack('<QQ', number + 65536 * tree, index) + data,
//                     digest_size=32, salt=start[:16], person=start[16:])
// BLAKE2b is specified in RFC 7693 (M-J. Saarinen, J-P. Aumasson, "The BLAKE2 Cryptographic Hash
// and Message Authentication Code (MAC)", November 2015, https://www.rfc-editor.org/rfc/rfc7693).
// blocks/hash/blake2b_f.h has the compression function and names the specification of salt and
// personalization.
// Byte-exact rule: docs/format/FORMAT.md, R2.1 to R2.3; the table of uses: R3.1 there.
// What is assumed of the hash, and why leaf, salt and round look as they do: blocks/hash/MODE.md.
// There the assumption rests on Luykx, Mennink, Neves, "Security Analysis of BLAKE2's Modes of
// Operation" (ePrint 2016/827).
// A check that fails (BLK_HASH_REQUIRE, blocks/hash/tweak.h) ends the process, or on a GPU the
// kernel: an argument that a function checks is never clamped to its range.
#pragma once
#include "base.h"
#include "field/field.h"
#include "hash/blake2b_f.h"
#include "hash/domains.h"

namespace blk {
// The build's hash: the default for the template parameter Fn, where it has one.
// Fn supplies the sizes of a chaining value and a block (Fn::CVW, Fn::BLKW, in 32-bit words), the
// first chaining value for a start, the counter and flag of each compression, and the compression.
// It is a parameter so that another hash could take BLAKE2b's place; none does.
#if BLK_HASH == 1
using HashFn = Blake2bF;
#else
#error "BLK_HASH must be 1 (BLAKE2b)"
#endif

// 256 bits. Where code works on 64-bit words, w[2 i] is the low half of word i.
constexpr size_t DIGEST_BYTES = 32;
struct Digest {
  u32 w[8];
};
BLK_FN bool operator==(const Digest &a, const Digest &b) {
  u32 d = 0;
  BLK_UNROLL
  for (int i = 0; i < 8; i++) d |= a.w[i] ^ b.w[i];
  return d == 0;
}
BLK_FN bool operator!=(const Digest &a, const Digest &b) { return !(a == b); }
// A word's lowest byte comes first. Every string of 32 bytes is a digest.
BLK_FN void digest_to_bytes(const Digest &d, u8 *out) {
  for (int i = 0; i < 32; i++) out[i] = (u8)(d.w[i / 4] >> 8 * (i % 4));
}
BLK_FN void digest_from_bytes(const u8 *in, Digest &d) {
  for (int i = 0; i < 8; i++)
    d.w[i] = in[4 * i] | (u32)in[4 * i + 1] << 8 | (u32)in[4 * i + 2] << 16 | (u32)in[4 * i + 3] << 24;
}

constexpr size_t LABEL_BYTES = 16, LABEL_WORDS = 2; // begins each hashed string: tag word, index
// Compressions in a hash of this many bytes of input; the label is added here. For BLAKE2b:
// ceil((16 + bytes) / 128).
template <class Fn = HashFn> BLK_HD constexpr u64 mode_calls(u64 bytes) { return (LABEL_BYTES + bytes + 4 * Fn::BLKW - 1) / (4 * Fn::BLKW); }

// Every leaf has a salt: part of the hashed bytes, not BLAKE2b's salt parameter. TagT's constructor
// for leaves enforces these bounds on the salt's length, whatever a parameter file or a test asks
// for. Specified in docs/format/FORMAT.md, R7.3.
constexpr size_t SALT_FLOOR_BYTES = 40, SALT_MOST_BYTES = 64;
BLK_HD constexpr bool salt_length_fits(u64 salt_bytes) { return salt_bytes >= SALT_FLOOR_BYTES && salt_bytes <= SALT_MOST_BYTES && salt_bytes % 8 == 0; }
// leaf_salt_at(): the offset of a leaf's salt in the hashed string. leaf_bytes(): the string's
// length. Both count the label. The salt is entirely within the last block: if it would cross a
// block boundary, zero padding ("fill" in messages) extends the string to that boundary first.
// Not checked here: salt_length_fits(salt_bytes).
template <class Fn = HashFn> BLK_HD constexpr u64 leaf_salt_at(u64 symbols, u64 salt_bytes) {
  return (LABEL_BYTES + 8 * symbols) % (4 * Fn::BLKW) + salt_bytes <= 4 * Fn::BLKW ? LABEL_BYTES + 8 * symbols
                                                                                   : ((LABEL_BYTES + 8 * symbols) / (4 * Fn::BLKW) + 1) * (4 * Fn::BLKW);
}
template <class Fn = HashFn> BLK_HD constexpr u64 leaf_bytes(u64 symbols, u64 salt_bytes) { return leaf_salt_at<Fn>(symbols, salt_bytes) + salt_bytes; }

// What the hashes of one use share: the label's first word and, for a tree's leaves or vertices,
// their start. In `word`, number is the Domain, and tree is the number of the tree that a leaf
// belongs to: 0 for vertices and outside trees, and under Domain::CHAIN_QUERY the chain's label
// instead (which protocol the chain runs). All leaves of a tree start from one digest. If the
// prover sends the tree's root in a round of a Fiat-Shamir hash chain, that digest is the chain's
// run_value(). If the root is instead one of a chain's public inputs (the stored tree of a
// committed table, the tree that binds a batch's requests), the leaves start from the configuration
// digest, a hash of the public configuration files. A domain whose start is not in the tag gets it
// in HasherT's second constructor.
// Plain data, so it can be passed to a kernel by value.
template <class Fn> struct TagT {
  u64 word; // number + 2^16 tree; in a leaf's tag also + 2^32 symbols + 2^56 (salt_bytes / 8)
  typename Fn::Start start;
  BLK_FN TagT(Domain d, u64 tree = 0) : word((u64)d | tree << 16), start{} {
    BLK_HASH_REQUIRE(domain_is_given_out(d) && tree < (d == Domain::CHAIN_QUERY ? (u64)1 << 48 : (u64)1 << 16),
                     "not a domain number of this format, a tree's number of 2^16 or more, or a chain's label of 2^48 or more");
    BLK_HASH_REQUIRE(!domain_start_in_tag(d), "the tag of a tree's hashes is made with their start");
  }
  // The tag of a tree's leaves. symbols: the count of 8-byte words in a leaf's content. With it and
  // the salt's length in the label, two leaves of different shapes never hash the same string.
  // Not checked: symbols >= 1 (required by docs/format/FORMAT.md, R2.3).
  BLK_FN TagT(Domain d, u32 tree, const Digest &from, u64 symbols, u64 salt_bytes)
      : word((u64)d | (u64)tree << 16 | symbols << 32 | (salt_bytes / 8) << 56) {
    BLK_HASH_REQUIRE(domain_is_leaf(d) && tree < (1u << 16), "only leaves have a shape in their tag; a tree's number is below 65536");
    BLK_HASH_REQUIRE(symbols < (1u << 24) && salt_length_fits(salt_bytes), "a leaf has fewer than 2^24 symbols and a salt of 40 to 64 bytes, a multiple of 8");
    BLK_HASH_REQUIRE(from != Digest{}, "a start of zeros: it was not set");
    BLK_UNROLL
    for (int i = 0; i < 8; i++) start.w[i] = from.w[i];
  }
  BLK_FN u64 symbols() const { return (word >> 32) & 0xFFFFFF; } // of a leaf's tag
  BLK_FN u64 salt_bytes() const { return 8 * (word >> 56); }
  // The tag of a tree's vertices.
  BLK_FN TagT(Domain d, u32 tree, const Digest &from) : word((u64)d | (u64)tree << 16) {
    BLK_HASH_REQUIRE(domain_start_in_tag(d) && !domain_is_leaf(d) && tree < (1u << 16), "a leaf's tag is made with the leaves' shape; only a tree's tags hold a start");
    BLK_HASH_REQUIRE(!domain_is_vertex(d) || tree == 0, "a vertex's hash holds nothing of its tree");
    BLK_HASH_REQUIRE(from != Digest{}, "a start of zeros: it was not set");
    BLK_UNROLL
    for (int i = 0; i < 8; i++) start.w[i] = from.w[i];
  }
  BLK_FN Domain number() const { return (Domain)(word & 0xFFFF); }
};
// As a parameter's type, TagArg<Fn> is not used to deduce Fn, which lets a Domain convert to it.
template <class T> struct AsWritten {
  typedef T type;
};
template <class Fn> using TagArg = const typename AsWritten<TagT<Fn>>::type &;

BLK_FN void index_in_bounds(u64 index) { BLK_HASH_REQUIRE(!(index >> 63), "a hash's index must be below 2^63"); }
// ---- a leaf's hash for kernels, on whole blocks of symbols ----
// For a kernel that keeps one chaining value per leaf: leaf_begin(), then leaf_step() on every
// block but the last, then leaf_last(). The caller builds the hashed string, and nothing here
// checks it: the first block begins with the label (tag.word, then the index), the salt is at
// leaf_salt_at(), and the last block is 0 after the string's end.
// cv: Fn::CVW / 2 64-bit words. block: Fn::BLKW / 2 64-bit words. step: the block's number, from 0.
// bytes: the whole string's length, the label included (leaf_bytes()).
template <class Fn = HashFn> BLK_FN void leaf_begin(u64 *cv, TagArg<Fn> tag) {
  BLK_HASH_REQUIRE(!domain_has_start(tag.number()) || domain_start_in_tag(tag.number()), "a hash under this number starts from a digest of its own");
  Fn::begin64(cv, tag.start, DIGEST_BYTES);
}
template <class Fn = HashFn> BLK_FN void leaf_step(u64 *cv, const u64 *block, u32 step) { Fn::compress64(cv, block, Fn::step(step)); }
template <class Fn = HashFn> BLK_FN Digest leaf_last(u64 *cv, const u64 *block, u32 bytes) {
  Fn::compress64(cv, block, Fn::last(bytes));
  Digest d;
  BLK_UNROLL
  for (int i = 0; i < 4; i++) d.w[2 * i] = (u32)cv[i], d.w[2 * i + 1] = (u32)(cv[i] >> 32);
  return d;
}

// The last word of a round's message, as "a round of a chain" below requires it: the constructor
// leaves bits 16 and up 0. chain_call_lanes() (blocks/hash/lanes.cuh) takes a RoundWord and does
// not check those bits; before_nonce() and chain_call_cells() here read a plain word and check it.
// In blocks/chain/ the two bytes are the round's kind and its bits of proof of work.
struct RoundWord {
  u32 v; // below 2^16
  BLK_FN constexpr RoundWord(u8 low, u8 high) : v(low | (u32)high << 8) {}
};
// The hash state that the calls of one round share: everything but the call's number and the
// nonce. Plain data, so it can be copied to a GPU.
template <class Fn> struct ChainPrefixT {
  u32 cv[Fn::CVW];     // the chaining value before 'block'
  u32 block[Fn::BLKW]; // the block that contains the message's last word; zeros after it
  u32 fill;            // the bytes in it: a multiple of 8, from 8 to a whole block
  u32 steps;           // the blocks before it
};
// The 64 bytes of one call, as words of 8 bytes: two 32-byte answer blocks of a round.
struct ChainCall {
  u64 w[8];
};
BLK_FN Digest chain_block(const ChainCall &c, int half) { // 0 or 1
  Digest d;
  BLK_UNROLL
  for (int i = 0; i < 4; i++) d.w[2 * i] = (u32)c.w[4 * half + i], d.w[2 * i + 1] = (u32)(c.w[4 * half + i] >> 32);
  return d;
}

// ---- the hash, fed piece by piece ----
// The text below of HasherT (the members h, t, f, buf and buflen of its state; load64, store32 and store64; the
// counter's increment; the last block's flag; the loop that loads a block's words; the walk of push() and finish())
// comes, by way of prover/second_writing/blake2b_plain.h, from the BLAKE2 reference implementation in C (blake2b-ref.c
// and its two headers, as the copies that librsync 2.3.4 bundles have them). The head of blake2b-ref.c, which its two
// headers have too:
//   BLAKE2 reference source code package - reference C implementations
//
//   Copyright 2012, Samuel Neves <sneves@dei.uc.pt>.  You may use this under the
//   terms of the CC0, the OpenSSL Licence, or the Apache Public License 2.0, at
//   your option.  The terms of these licenses can be found at:
//
//   - CC0 1.0 Universal : http://creativecommons.org/publicdomain/zero/1.0
//   - OpenSSL license   : https://www.openssl.org/source/license.html
//   - Apache 2.0        : http://www.apache.org/licenses/LICENSE-2.0
//
//   More information about the BLAKE2 hash function can be found at
//   https://blake2.net.
// Adapted from blake2b_update and blake2b_final, the state's type and the helper routines of blake2b-ref.c and its two
// headers, blake2.h and blake2-impl.h, of the BLAKE2 reference source code package (reference C implementations), as
// prover/second_writing/blake2b_plain.h has them; taken under CC0 1.0 Universal (SPDX: CC0-1.0).
// Our changes. The state is the object, so S-> is not written: h, t, f, buf and buflen are blake2b_state's members.
// push(in, inlen) is blake2b_update, and finish() is blake2b_final. load64, store32 and store64 keep their names;
// set_lastblock() and increment_counter() are blake2b_set_lastblock and blake2b_increment_counter.
// The compression is Fn's. Fn::compress64() counts its calls, and tests put another Fn in, so the class reaches
// BLAKE2b's compression function through Fn::compress64() alone, and Fn::begin64() takes the place of blake2b_init;
// init() sets t, buf, buflen and f to 0 one by one, where blake2b_init0 has a memset over the state. compress() keeps
// the first loop of blake2b_compress, which loads the block's words, and passes them on. The compression function's
// counter and flag travel in a Tweak: Fn::step() wants the block's number, which is t / 128 - 1 after the counter's
// increment, and Fn::last() wants the string's length, which is t after blake2b_final's increment.
// A string has at most 2^32 - 128 bytes, because Fn::step() and Fn::last() count in 32 bits. So t can be one word,
// without S->t[1] and its carry, and push() refuses the byte that would make the string longer.
// f is one u32 that holds 0 or 1, where blake2b_plain.h has u64 f[1] and stores a word of ones. The compression
// function does not read it: its flag comes from Fn::last(). f records that the object is finished: set_lastblock()
// refuses a second finishing, and push() refuses input after the first. blake2b_plain.h's text checks neither: the
// reference's blake2b_final refuses a second call, by a test that blake2b_plain.h's head lists as left out.
// The bytes from buf[buflen] on are 0 whenever a member returns, and no byte of a compressed block stays in the object;
// in blake2b_state the bytes from buf[buflen] on are stale. The receivers of before_nonce()'s block rely on it: they
// take the words behind its fill as zeros. So clear_buf(), which is ours, follows each compression of buf, and
// blake2b_final's memset, the padding, is left out: the zeros are there already. clear_buf() is ordinary stores and no
// wipe: a caller that must wipe an object has to do that itself. Either finishing ends with buflen = 0, which
// blake2b_final does not do: t has taken the buffer's bytes in by then, and bytes() adds t and buflen.
// memcpy is the loop copy(): the text is one for host code and GPU code, and this file does not include <cstring>.
// buflen is a u32, not a size_t: a type of fixed width, which together with f fills one word of 64 bits. The members'
// order is ours: leaf stands behind t, and buflen and f stand last. So the object has no padding.
// finish() returns a Digest, eight words of 32 bits, so the halves of h's first four words take the place of store64
// into out: the same 32 bytes. The digest length, 32 or 64, enters the first chaining value alone; blake2b_final's
// outlen and its temporary buffer are left out, as in blake2b_plain.h.
// push(), load64, store32 and store64 take pointers to u8, not to void, so the casts are left out. Loops with a fixed
// bound count in int and have BLK_UNROLL, and u8, u32 and u64 are written for the types of <cstdint>.
// The format's own, with no counterpart in the reference. The two constructors: the first takes the start from the tag
// and refuses a domain number whose start comes with each hash; the second takes the start as an argument and refuses
// every other domain number; both refuse an index of 2^63 or more, and both push the label as the string's first 16
// bytes, which bytes() does not count. push_u32() and push_u64(): store32 or store64, then push(). leaf, by which
// finish() holds a leaf's string to the length that its tag gives.
// before_nonce(), and what it requires of a round's message and returns. It finishes the object without a compression,
// which the receiver makes. Of blake2b_final it has two steps, in the reverse order: set_lastblock() first, the
// counter's increment last. That order is needed. After a finishing buflen is 0: the check on the length can then pass,
// and the check on the last word would read eight bytes before buf; set_lastblock() stops a finished object before
// both. And steps is read from t before t takes the last block's bytes in. After the hand-over it clears buf. It needs
// no step of its own to keep a full last block: blake2b_update never compresses a buffer that has just become full.
// F is the base field and E its cubic extension (blocks/field/field.h); push() feeds an E as its
// three coefficients, c[0] first.
template <class Fn> class HasherT {
public:
  static constexpr u32 BLOCK_BYTES = 4 * Fn::BLKW;
  // Begins the hash H(tag, index, start; bytes) whose start is the tag's own (zeros under a number that has no start).
  // The digest has 32 bytes. Refuses a domain number whose start comes with each hash, and an index of 2^63 or more.
  BLK_FN HasherT(const TagT<Fn> &tag, u64 index) {
    BLK_HASH_REQUIRE(!(domain_has_start(tag.number()) && !domain_start_in_tag(tag.number())),
                     "a hash under this number starts from a digest");
    init(tag, index, tag.start, (u32)DIGEST_BYTES);
  }
  // The same under a domain number whose start comes with each hash, as the argument start. Refuses every other domain
  // number. Under CHAIN_QUERY BLAKE2b's digest length is 64 bytes.
  BLK_FN HasherT(const TagT<Fn> &tag, u64 index, const Digest &start) {
    BLK_HASH_REQUIRE(domain_has_start(tag.number()) && !domain_start_in_tag(tag.number()),
                     "a hash under this number takes no start of its own");
    typename Fn::Start s;
    BLK_UNROLL
    for (int i = 0; i < 8; ++i) {
      s.w[i] = start.w[i];
    }
    init(tag, index, s, tag.number() == Domain::CHAIN_QUERY ? (u32)64 : (u32)DIGEST_BYTES);
  }

  // Appends x's four bytes, the lowest first.
  BLK_FN void push_u32(u32 x) {
    u8 in[4];
    store32(in, x);
    push(in, sizeof(in));
  }
  // Appends x's eight bytes, the lowest first.
  BLK_FN void push_u64(u64 x) {
    u8 in[8];
    store64(in, x);
    push(in, sizeof(in));
  }
  // Adapted from blake2b_update: appends inlen bytes to the string. in may be null if inlen is 0. A buffer that has
  // just become full is not compressed yet, because the string may end there, and its last block is compressed with the
  // last-block flag or handed over by before_nonce(). So after a call with inlen > 0, buflen is 1 to 128.
  BLK_FN void push(const u8 *in, size_t inlen) {
    if (inlen > 0) {
      BLK_HASH_REQUIRE(f == 0, "a hash that was finished takes no more input");
      BLK_HASH_REQUIRE(inlen <= MOST_BYTES - (t + buflen), "a hash takes at most 2^32 bytes less one block");
      size_t left = buflen;
      size_t fill = BLOCK_BYTES - left;
      if (inlen > fill) {
        buflen = 0;
        copy(buf + left, in, fill); // Fill buffer
        increment_counter(BLOCK_BYTES);
        compress(buf, Fn::step((u32)(t / BLOCK_BYTES) - 1)); // Compress
        clear_buf();
        in += fill;
        inlen -= fill;
        while (inlen > BLOCK_BYTES) {
          increment_counter(BLOCK_BYTES);
          compress(in, Fn::step((u32)(t / BLOCK_BYTES) - 1));
          in += BLOCK_BYTES;
          inlen -= BLOCK_BYTES;
        }
      }
      copy(buf + buflen, in, inlen);
      buflen += (u32)inlen;
    }
  }
  BLK_FN void push(const F &x) { push_u64(x.v); } // the 8 bytes of F_to_bytes
  BLK_FN void push(const E &x) {
    for (int i = 0; i < 3; i++) push_u64(x.c[i].v);
  }
  BLK_FN void push(const Digest &d) {
    for (int i = 0; i < 8; i++) push_u32(d.w[i]);
  }
  // The number of bytes pushed so far; the label is not counted.
  BLK_FN u64 bytes() const { return t + buflen - LABEL_BYTES; }

  // Adapted from blake2b_final: compresses the last block and returns the digest's first 32 bytes, the first four words
  // of h. The padding is there already: the bytes from buf[buflen] on are 0.
  BLK_FN Digest finish() {
    BLK_HASH_REQUIRE(leaf == 0 || t + buflen == leaf, "a leaf was not fed what its tag says: symbols, fill, salt");
    increment_counter(buflen);
    set_lastblock();
    compress(buf, Fn::last((u32)t));
    clear_buf();
    buflen = 0;
    Digest out;
    BLK_UNROLL
    for (int i = 0; i < (int)DIGEST_BYTES / 8; ++i) {
      out.w[2 * i] = (u32)h[i], out.w[2 * i + 1] = (u32)(h[i] >> 32);
    }
    return out;
  }
  // Finishes the object without a compression: hands over the chaining value and the block that holds the string's last
  // byte, for a receiver that completes this block and ends the hash itself. steps is the number of blocks compressed,
  // and fill the number of the string's bytes in the block. Of blake2b_final it has set_lastblock() and the counter's
  // increment, in the reverse order.
  BLK_FN ChainPrefixT<Fn> before_nonce() {
    // This call stays first. After a finishing buflen is 0: the check on the length can then pass, and the check on the
    // last word would read eight bytes before buf.
    set_lastblock();
    BLK_HASH_REQUIRE(bytes() > 0 && buflen % 8 == 0,
                     "a round's message is a multiple of 8 bytes and ends with the word that takes the call's number");
    BLK_HASH_REQUIRE(load64(buf + buflen - 8) >> 16 == 0,
                     "the last word of a round's message must leave its bits from 16 up to the call's number");
    ChainPrefixT<Fn> out;
    out.steps = (u32)(t / BLOCK_BYTES);
    out.fill = buflen;
    BLK_UNROLL
    for (int i = 0; i < 16; ++i) {
      const u64 w = load64(buf + i * sizeof(u64));
      out.block[2 * i] = (u32)w, out.block[2 * i + 1] = (u32)(w >> 32);
    }
    BLK_UNROLL
    for (int i = 0; i < 8; ++i) {
      out.cv[2 * i] = (u32)h[i], out.cv[2 * i + 1] = (u32)(h[i] >> 32);
    }
    increment_counter(buflen);
    clear_buf();
    buflen = 0;
    return out;
  }

private:
  static_assert(Fn::CVW == 16 && Fn::BLKW == 32, "h has eight words of 64 bits, and a block has sixteen");
  // The longest string, the label counted: Fn::step() and Fn::last() count in 32 bits.
  static constexpr u64 MOST_BYTES = ((u64)1 << 32) - BLOCK_BYTES;

  // Adapted from blake2b_state. h is the chaining value. t is the number of the string's bytes compressed or handed
  // over so far. f is 0 until the object is finished, then 1. buf[0 .. buflen) holds the bytes not yet compressed, and
  // the bytes from buf[buflen] on are 0. leaf is ours: the length that a leaf's tag prescribes for the string, or 0 if
  // the hash is no leaf's. In this order the members leave no padding.
  u64 h[8];
  u64 t;
  u64 leaf;
  u8 buf[BLOCK_BYTES];
  u32 buflen;
  u32 f;

  // Adapted from load64: reads eight bytes as a little-endian number.
  BLK_FN static u64 load64(const u8 *p) {
    return ((u64)(p[0]) << 0) |
           ((u64)(p[1]) << 8) |
           ((u64)(p[2]) << 16) |
           ((u64)(p[3]) << 24) |
           ((u64)(p[4]) << 32) |
           ((u64)(p[5]) << 40) |
           ((u64)(p[6]) << 48) |
           ((u64)(p[7]) << 56);
  }
  // Adapted from store32.
  BLK_FN static void store32(u8 *p, u32 w) {
    p[0] = (u8)(w >> 0);
    p[1] = (u8)(w >> 8);
    p[2] = (u8)(w >> 16);
    p[3] = (u8)(w >> 24);
  }
  // Adapted from store64.
  BLK_FN static void store64(u8 *p, u64 w) {
    p[0] = (u8)(w >> 0);
    p[1] = (u8)(w >> 8);
    p[2] = (u8)(w >> 16);
    p[3] = (u8)(w >> 24);
    p[4] = (u8)(w >> 32);
    p[5] = (u8)(w >> 40);
    p[6] = (u8)(w >> 48);
    p[7] = (u8)(w >> 56);
  }
  // In the place of memcpy.
  BLK_FN static void copy(u8 *dst, const u8 *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      dst[i] = src[i];
    }
  }
  // Ours, with no counterpart in the source: sets all of buf to 0. These are ordinary stores and no wipe: a caller that
  // must wipe an object has to do that itself.
  BLK_FN void clear_buf() {
    BLK_UNROLL
    for (int i = 0; i < (int)BLOCK_BYTES; ++i) {
      buf[i] = 0;
    }
  }
  // Adapted from blake2b_set_lastblock. The compression function has its flag from Fn::last(); f records that the
  // object is finished.
  BLK_FN void set_lastblock() {
    BLK_HASH_REQUIRE(f == 0, "a hash is finished once");
    f = 1;
  }
  // Adapted from blake2b_increment_counter.
  BLK_FN void increment_counter(const u64 inc) { t += inc; }
  // Adapted from the first loop of blake2b_compress, which loads the block's bytes; Fn::compress64() does the rest.
  BLK_FN void compress(const u8 *block, const Tweak &tweak) {
    u64 m[16];
    BLK_UNROLL
    for (int i = 0; i < 16; ++i) {
      m[i] = load64(block + i * sizeof(m[i]));
    }
    Fn::compress64(h, m, tweak);
  }
  // In the place of blake2b_init: an empty state, the first chaining value, and the label as the string's first 16
  // bytes.
  BLK_FN void init(const TagT<Fn> &tag, u64 index, const typename Fn::Start &start, u32 digest_bytes) {
    index_in_bounds(index);
    t = 0;
    leaf = domain_is_leaf(tag.number()) ? leaf_bytes<Fn>(tag.symbols(), tag.salt_bytes()) : 0;
    clear_buf();
    buflen = 0;
    f = 0;
    Fn::begin64(h, start, digest_bytes);
    push_u64(tag.word);
    push_u64(index);
  }
};

// ---- two digests ----
// two_digests(): the label and two digests, in one compression. node(): under a domain that is not
// for a tree's vertices, with an index; run_value() uses it. vertex(): under a domain for a tree's
// vertices, always with index 0.
template <class Fn> BLK_FN Digest two_digests(const TagT<Fn> &tag, u64 index, const Digest &left, const Digest &right) {
  static_assert(Fn::BLKW >= 20, "the label and two digests are one block");
  BLK_HASH_REQUIRE(!domain_has_start(tag.number()) || domain_start_in_tag(tag.number()), "a hash under this number starts from a digest of its own");
  index_in_bounds(index);
  u32 cv[Fn::CVW], m[Fn::BLKW];
  Fn::begin(cv, tag.start, DIGEST_BYTES);
  BLK_UNROLL
  for (int i = 0; i < Fn::BLKW; i++)
    m[i] = i == 0 ? (u32)tag.word : i == 1 ? (u32)(tag.word >> 32) : i == 2 ? (u32)index : i == 3 ? (u32)(index >> 32) : i < 12 ? left.w[(i - 4) & 7] : i < 20 ? right.w[(i - 12) & 7] : 0;
  Fn::compress(cv, m, Fn::last(LABEL_BYTES + 64));
  Digest d;
  BLK_UNROLL
  for (int i = 0; i < 8; i++) d.w[i] = cv[i];
  return d;
}
template <class Fn = HashFn> BLK_FN Digest node(TagArg<Fn> tag, u64 index, const Digest &left, const Digest &right) {
  BLK_HASH_REQUIRE(!domain_is_vertex(tag.number()), "a tree's vertex is made by vertex(), which takes no place");
  return two_digests<Fn>(tag, index, left, right);
}
// A vertex of a Merkle tree: H(TREE_NODE, tree 0, index 0, start = the configuration digest;
// left, right). The input does not depend on the tree, the vertex's position or the run: these
// enter a tree's hashes through its leaves only.
// The argument that this suffices is no theorem and covers perfect binary trees only:
// blocks/hash/MODE.md, "1. The rule in short".
// tag: Tag(Domain::TREE_NODE, 0, that digest), which a caller makes once; in the tree that binds a
// batch's requests, Domain::BINDING_NODE instead.
template <class Fn = HashFn> BLK_FN Digest vertex(TagArg<Fn> tag, const Digest &left, const Digest &right) {
  BLK_HASH_REQUIRE(domain_is_vertex(tag.number()), "a vertex is hashed under a number of vertices");
  return two_digests<Fn>(tag, 0, left, right);
}
template <class Fn = HashFn> BLK_FN Digest vertex(const Digest &left, const Digest &right, const Digest &start) {
  return two_digests<Fn>(TagT<Fn>(Domain::TREE_NODE, 0, start), 0, left, right);
}
// A run's value, from which the leaves of the trees made in one run of a chain start:
// H(RUN, index = the chain's label; the parameters' digest, the chain's 32 random bytes).
// The prover draws these bytes, or they are one of the chain's public inputs (an item of its first
// query): docs/format/FORMAT.md, R4.5.
// The parameters' digest is the hash of the chain's own list of public parameters (R4.4 there); the
// configuration digest is one item of that list.
template <class Fn = HashFn> BLK_FN Digest run_value(u64 label, const Digest &parameters, const Digest &random_bytes) {
  return node<Fn>(Domain::RUN, label, parameters, random_bytes);
}

// The end of both forms of leaf(). count and salt_bytes must be the tag's.
template <class Fn> BLK_FN Digest leaf_end(HasherT<Fn> &h, const TagT<Fn> &tag, size_t count, const u8 *salt, size_t salt_bytes) {
  BLK_HASH_REQUIRE(domain_is_leaf(tag.number()) && count == tag.symbols() && salt_bytes == tag.salt_bytes(), "a leaf of another shape than its tag says");
  for (u64 at = LABEL_BYTES + 8 * count; at < leaf_salt_at<Fn>(count, salt_bytes); at += 8) h.push_u64(0);
  h.push(salt, salt_bytes);
  return h.finish();
}
template <class Fn = HashFn>
BLK_FN Digest leaf(TagArg<Fn> tag, u64 index, const F *symbols, size_t count, const u8 *salt, size_t salt_bytes) {
  HasherT<Fn> h(tag, index);
  for (size_t i = 0; i < count; i++) h.push(symbols[i]);
  return leaf_end<Fn>(h, tag, count, salt, salt_bytes);
}
// A leaf whose content is one digest: four words.
template <class Fn = HashFn> BLK_FN Digest leaf(TagArg<Fn> tag, u64 index, const Digest &content, const u8 *salt, size_t salt_bytes) {
  HasherT<Fn> h(tag, index);
  h.push(content);
  return leaf_end<Fn>(h, tag, 4, salt, salt_bytes);
}

// ---- a round of a chain ----
// A round's answer is a sequence of 32-byte answer blocks. Blocks 2 j and 2 j + 1 are the halves of
// call j:
//     H(CHAIN_QUERY with the chain's label in the place of a tree's number, index = the round,
//       start = the chain's value before the round (rho_prev); the message, the nonce),  64 bytes.
// The message's last 8-byte word has j in bits 16 and up. The caller must leave those bits 0 and
// may use the lower 16 as it likes. So the calls of a round differ only in the last 16 bytes, and
// from a ChainPrefixT each costs one compression whatever the message's length, or two if that
// word ends a compression block. What each answer block means: blocks/chain/chain.h.
// Use: ChainAbsorb a(rho_prev, round); a.push(..); ..; then chain_call() on a.prefix().
template <class Fn> class ChainAbsorbT : public HasherT<Fn> {
public:
  // chain: the chain's label, below 2^48.
  BLK_FN explicit ChainAbsorbT(const Digest &rho_prev, u64 round = 0, u64 chain = 0) : HasherT<Fn>(TagT<Fn>(Domain::CHAIN_QUERY, chain), round, rho_prev) {}
  BLK_FN ChainPrefixT<Fn> prefix() { return this->before_nonce(); }
  Digest finish() = delete; // a call ends with its nonce
};
// Call j of a round. nonce: what a search for proof of work varies; a round without one uses 0.
// p: from ChainAbsorbT::prefix(); its fields are not checked here.
// The compiler hint is public: BLK_UNROLL on small per-thread arrays: M. Milakov (cited at
// chain_call_cells()).
template <class Fn> BLK_FN ChainCall chain_call(const ChainPrefixT<Fn> &p, u64 nonce, u64 j) {
  BLK_HASH_REQUIRE(!(j >> 48), "a call's number is below 2^48");
  const u32 at = p.fill / 4; // the nonce's first word, if it has room in the block
  u32 cv[Fn::CVW], m[Fn::BLKW];
  BLK_UNROLL
  for (int i = 0; i < Fn::CVW; i++) cv[i] = p.cv[i];
  BLK_UNROLL
  for (int i = 0; i < Fn::BLKW; i++) // no run-time index into m, so that it stays in registers
    m[i] = (u32)i == at       ? (u32)nonce
           : (u32)i == at + 1 ? (u32)(nonce >> 32)
           : (u32)i + 2 == at ? p.block[i] | (u32)(j << 16)
           : (u32)i + 1 == at ? p.block[i] | (u32)(j >> 16)
                              : p.block[i];
  const u32 bytes = p.steps * 4 * Fn::BLKW + p.fill + 8;
  if (at == (u32)Fn::BLKW) { // the block is full: the nonce begins the next
    Fn::compress(cv, m, Fn::step(p.steps));
    BLK_UNROLL
    for (int i = 0; i < Fn::BLKW; i++) m[i] = i == 0 ? (u32)nonce : i == 1 ? (u32)(nonce >> 32) : 0;
  }
  Fn::compress(cv, m, Fn::last(bytes));
  ChainCall c;
  BLK_UNROLL
  for (int i = 0; i < 8; i++) c.w[i] = cv[2 * i] | (u64)cv[2 * i + 1] << 32;
  return c;
}
// The same answer as ChainAbsorbT and chain_call() on a message of CELLS 8-byte words, with every
// array index fixed at compile time: for a chain that runs inside a kernel.
// The idea is public: complete unrolling by a template parameter (CELLS): M. Harris, "Optimizing
// Parallel Reduction in CUDA", NVIDIA Developer Technology, slides "Complete Unrolling", "Unrolling
// with Templates" and "Reduction #6: Completely Unrolled",
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf. There the parameter is the
// block size.
// The compiler hint is public: BLK_UNROLL (#pragma unroll in GPU code, tweak.h) on small per-thread
// arrays: M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog,
// 2015, "Static indexing",
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <class Fn = HashFn, int CELLS>
BLK_FN ChainCall chain_call_cells(const Digest &rho_prev, u64 round, const u64 (&cells)[CELLS], u64 nonce, u64 j, u64 chain = 0) {
  static_assert(CELLS >= 1, "the message ends with the word that takes the call's number");
  BLK_HASH_REQUIRE(!(cells[CELLS - 1] >> 16) && !(j >> 48), "the last word of a round's message must leave its bits from 16 up to the call's number, which is below 2^48");
  constexpr int PER = Fn::BLKW / 2, ALL = (int)LABEL_WORDS + CELLS + 1, STEPS = (ALL - 1) / PER;
  index_in_bounds(round);
  typename Fn::Start from;
  BLK_UNROLL
  for (int i = 0; i < 8; i++) from.w[i] = rho_prev.w[i];
  u64 h[Fn::CVW / 2], m[PER];
  Fn::begin64(h, from, 2 * DIGEST_BYTES);
  BLK_UNROLL
  for (int s = 0; s <= STEPS; s++) {
    BLK_UNROLL
    for (int i = 0; i < PER; i++) {
      const int cell = s * PER + i - (int)LABEL_WORDS;
      m[i] = cell == -2          ? TagT<Fn>(Domain::CHAIN_QUERY, chain).word
             : cell == -1        ? round
             : cell + 1 < CELLS  ? cells[cell >= 0 && cell + 1 < CELLS ? cell : 0]
             : cell + 1 == CELLS ? cells[CELLS - 1] | j << 16
             : cell == CELLS     ? nonce
                                 : 0;
    }
    if (s < STEPS) Fn::compress64(h, m, Fn::step((u32)s));
  }
  Fn::compress64(h, m, Fn::last(8 * ALL));
  ChainCall c;
  BLK_UNROLL
  for (int i = 0; i < 8; i++) c.w[i] = h[i];
  return c;
}
// Whether the low 'bits' bits of a word are 0. bits = 0: every word passes. At most 64.
BLK_FN bool work_test(u64 word, unsigned bits) {
  BLK_HASH_REQUIRE(bits <= 64, "the work test takes 0 to 64 bits");
  return bits == 0 || (word << (64 - bits)) == 0;
}

// The work test of a round, on one word of call 0: the first word of the round's first answer
// block, which a round with proof of work uses only for this test. Chain::has_work()
// (blocks/chain/chain.h) makes the same test with the word's index written as 0.
constexpr int CHAIN_WORK_WORD = 0; // index into ChainCall::w
template <class Fn> BLK_FN bool chain_work_passes(const ChainPrefixT<Fn> &p, u64 nonce, unsigned bits) {
  return work_test(chain_call(p, nonce, 0).w[CHAIN_WORK_WORD], bits);
}

// ---- randomness for tests and toy runs ----
// H(TEST_COINS, counter, start = a seed; nothing): 32 bytes for one compression.
template <class Fn = HashFn> BLK_FN Digest test_coins(const Digest &seed, u64 counter) {
  HasherT<Fn> h(TagT<Fn>(Domain::TEST_COINS), counter, seed);
  return h.finish();
}

// ---- the build's setting, under plain names ----
// A slab is as many 8-byte field elements as one compression takes in.
constexpr size_t HASH_BLOCK_BYTES = 4 * HashFn::BLKW; // what one compression takes in
constexpr size_t SLAB_SYMBOLS = HashFn::BLKW / 2;     // the same in field elements of 8 bytes
BLK_HD constexpr u64 hash_calls(u64 bytes) { return mode_calls<HashFn>(bytes); }
using Tag = TagT<HashFn>;
using Hasher = HasherT<HashFn>;
using ChainAbsorb = ChainAbsorbT<HashFn>;
using ChainPrefix = ChainPrefixT<HashFn>;
} // namespace blk
