// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// F, the compression function of BLAKE2b, for host code and GPU kernels; what hash.h needs to run
// BLAKE2b itself on it; and the call with a caller-chosen counter that the modes of older.h make.
//
// The text below of F and of its parts (the rotation in host code, the IV, G, a round, the compression, the first
// chaining values) comes from the BLAKE2 team's reference implementation in C. The head of its file blake2b-ref.c, which
// its two headers have too:
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
// Adapted from blake2b-ref.c and the two headers that it includes, blake2.h and blake2-impl.h, of the BLAKE2 reference
// source code package (reference C implementations); taken under CC0 1.0 Universal (SPDX: CC0-1.0).
// Our changes. F takes plain arguments where blake2b_compress reads a blake2b_state S and loads the block's bytes: h is
// S->h, m[i] is load64 of the block's i-th eight bytes, t_low is S->t[0] and t_high is S->t[1]. last takes the place of
// S->f[0]: true gives the word of 64 one bits that blake2b_set_lastblock stores, false gives 0. Nothing takes the place of
// S->f[1], which only blake2b_set_lastnode sets: it counts as 0. In the reference S->t[1] is the upper half of a count of
// bytes; here t_high is any word, and compress64 (below) puts a tweak's num + 2^32 l there.
// The macros G and ROUND are named BLK_B2_G and BLK_B2_ROUND, because G and ROUND are too short for macros in a header;
// both are undefined again after F.
// blake2b_IV is eight named constants, blake2b_IV0 for blake2b_IV[0] and so on; blake2b_sigma is constexpr and local to F;
// and F's two loops have BLK_UNROLL: so in GPU code every index is a constant and no table is read at run time. The loops
// count in int, not in size_t. u64 and u8 are written for uint64_t and uint8_t, and b2_right<c>(w) takes the place of
// rotr64(w, c). No blake2b_param is filled: param_word0, begin64 and first64 form the words of the parameter block that
// are not 0 and take them into the IV by xor, as blake2b_init_param does.
// RFC 7693 (M-J. Saarinen, J-P. Aumasson, "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)", November
// 2015, https://www.rfc-editor.org/rfc/rfc7693) specifies the function: section 2.1 (64 bits in a word, 12 rounds, the
// rotations 32, 24, 16, 63), section 2.6 (IV), section 2.7 (SIGMA), section 3.1 (G) and section 3.2 (F).
// Source for modelling F itself as a random function: Luykx, Mennink, Neves, "Security Analysis of BLAKE2's Modes of
// Operation" (ePrint 2016/827).
// prover/common/blake2b_f.h of this tree has the counterparts of F, compress and first: f2b::F, f2b::C and f2b::first_state.
// In GPU code the four rotations are written on the two 32-bit halves of a 64-bit word: by 32 the halves are swapped; by
// 24 and 16 whole bytes move (__byte_perm); by 63 is a rotation to the left by 1 (__funnelshift_l). Both functions:
// NVIDIA, CUDA Math API Reference Manual, "Integer Intrinsics" (https://docs.nvidia.com/cuda/cuda-math-api/).
// The compiler hint is public: BLK_UNROLL (tweak.h) is #pragma unroll in GPU code: NVIDIA, "CUDA Programming Guide", page
// "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html; for loops over small
// per-thread arrays: M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015, "Static
// indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
#pragma once
#include "hash/tweak.h"

namespace blk {
// x rotated to the right by N bits, N one of 32, 24, 16, 63.
// Public source: that a rotation of a 64-bit word by 32 bits is only a swap of its two 32-bit words on a 32-bit machine
// is stated in Aumasson and Bernstein, "SipHash: a fast short-input PRF" (2012, https://eprint.iacr.org/2012/351,
// section 4, "Choice of rotation counts"); here it serves BLAKE2b's rotation by 32 on the GPU's 32-bit registers.
template <int N> BLK_FN u64 b2_right(u64 x) {
  static_assert(N == 32 || N == 24 || N == 16 || N == 63, "one of the four rotation counts of RFC 7693, section 2.1");
#ifdef __CUDA_ARCH__
  const u32 lo = (u32)x, hi = (u32)(x >> 32);
  u32 l, h;
  if (N == 32)
    l = hi, h = lo;
  else if (N == 24)
    l = __byte_perm(lo, hi, 0x6543), h = __byte_perm(lo, hi, 0x2107);
  else if (N == 16)
    l = __byte_perm(lo, hi, 0x5432), h = __byte_perm(lo, hi, 0x1076);
  else
    l = __funnelshift_l(hi, lo, 1), h = __funnelshift_l(lo, hi, 1);
  return (u64)h << 32 | l;
#else
  return (x >> N) | (x << (64 - N)); // rotr64 of blake2-impl.h
#endif
}
// Adapted from blake2b_IV of blake2b-ref.c: blake2b_IV0 is blake2b_IV[0], and so on.
constexpr u64 blake2b_IV0 = 0x6a09e667f3bcc908ULL, blake2b_IV1 = 0xbb67ae8584caa73bULL;
constexpr u64 blake2b_IV2 = 0x3c6ef372fe94f82bULL, blake2b_IV3 = 0xa54ff53a5f1d36f1ULL;
constexpr u64 blake2b_IV4 = 0x510e527fade682d1ULL, blake2b_IV5 = 0x9b05688c2b3e6c1fULL;
constexpr u64 blake2b_IV6 = 0x1f83d9abfb41bd6bULL, blake2b_IV7 = 0x5be0cd19137e2179ULL;
// The mixing function G, adapted from the macro G of blake2b-ref.c. It is expanded only inside F, which has m and
// blake2b_sigma in scope. r and i are literals, so each index into m is a constant.
#define BLK_B2_G(r,i,a,b,c,d)               \
  do {                                      \
    a = a + b + m[blake2b_sigma[r][2*i+0]]; \
    d = b2_right<32>(d ^ a);                \
    c = c + d;                              \
    b = b2_right<24>(b ^ c);                \
    a = a + b + m[blake2b_sigma[r][2*i+1]]; \
    d = b2_right<16>(d ^ a);                \
    c = c + d;                              \
    b = b2_right<63>(b ^ c);                \
  } while(0)
// One round, adapted from the macro ROUND of blake2b-ref.c: G eight times on the work vector v. It is expanded only inside
// F, with a literal r from 0 to 11.
#define BLK_B2_ROUND(r)                    \
  do {                                     \
    BLK_B2_G(r,0,v[ 0],v[ 4],v[ 8],v[12]); \
    BLK_B2_G(r,1,v[ 1],v[ 5],v[ 9],v[13]); \
    BLK_B2_G(r,2,v[ 2],v[ 6],v[10],v[14]); \
    BLK_B2_G(r,3,v[ 3],v[ 7],v[11],v[15]); \
    BLK_B2_G(r,4,v[ 0],v[ 5],v[10],v[15]); \
    BLK_B2_G(r,5,v[ 1],v[ 6],v[11],v[12]); \
    BLK_B2_G(r,6,v[ 2],v[ 7],v[ 8],v[13]); \
    BLK_B2_G(r,7,v[ 3],v[ 4],v[ 9],v[14]); \
  } while(0)

struct Blake2bF {
  static constexpr int CVW = 16, BLKW = 32; // a chaining value and a block, in words of 32 bits
  static constexpr const char *NAME = "BLAKE2b";

  // h := F(h, m, t, f) with t = t_low + 2^64 t_high and f = last. Adapted from blake2b_compress of blake2b-ref.c.
  // t and f as in RFC 7693, section 3.2: the offset counter and the final-block flag.
  BLK_FN static void F(u64 h[8], const u64 m[16], u64 t_low, u64 t_high, bool last) {
    constexpr u8 blake2b_sigma[12][16] =
    {
      {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
      { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
      { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
      {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
      {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } ,
      {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
      { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } ,
      { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
      {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } ,
      { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 } ,
      {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
      { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
    };
    u64 v[16];
    int i;

    BLK_UNROLL
    for (i = 0; i < 8; ++i) {
      v[i] = h[i];
    }

    v[ 8] = blake2b_IV0;
    v[ 9] = blake2b_IV1;
    v[10] = blake2b_IV2;
    v[11] = blake2b_IV3;
    v[12] = blake2b_IV4 ^ t_low;
    v[13] = blake2b_IV5 ^ t_high;
    v[14] = blake2b_IV6 ^ (last ? (u64)-1 : (u64)0);
    v[15] = blake2b_IV7;

    BLK_B2_ROUND(0);
    BLK_B2_ROUND(1);
    BLK_B2_ROUND(2);
    BLK_B2_ROUND(3);
    BLK_B2_ROUND(4);
    BLK_B2_ROUND(5);
    BLK_B2_ROUND(6);
    BLK_B2_ROUND(7);
    BLK_B2_ROUND(8);
    BLK_B2_ROUND(9);
    BLK_B2_ROUND(10);
    BLK_B2_ROUND(11);

    BLK_UNROLL
    for (i = 0; i < 8; ++i) {
      h[i] = h[i] ^ v[i] ^ v[i + 8];
    }
  }
#undef BLK_B2_ROUND
#undef BLK_B2_G

  // F with counter t and flag f from a tweak: t mod 2^64 = c0 + 2^32 c1; t >> 64 = num + 2^32 l;
  // f = last. BLAKE2b itself, on inputs below 2^64 bytes, has t >> 64 = 0 (step(), last() below).
  // Each num defined in older.h is nonzero, so its calls are not BLAKE2b's. Not checked: num != 0.
  BLK_FN static void compress64(u64 h[8], const u64 m[16], const Tweak &t) {
    BLK_COUNT_CALL(t);
    F(h, m, t.c0 | (u64)t.c1 << 32, t.num | (u64)t.l << 32, t.last);
  }
  // The same on words of 32 bits: two make a word of F, the low word first. The output replaces cv.
  BLK_FN static void compress(u32 cv[CVW], const u32 block[BLKW], const Tweak &t) {
    u64 h[8], m[16];
    BLK_UNROLL
    for (int i = 0; i < 8; i++) h[i] = cv[2 * i] | (u64)cv[2 * i + 1] << 32;
    BLK_UNROLL
    for (int i = 0; i < 16; i++) m[i] = block[2 * i] | (u64)block[2 * i + 1] << 32;
    compress64(h, m, t);
    BLK_UNROLL
    for (int i = 0; i < 8; i++) cv[2 * i] = (u32)h[i], cv[2 * i + 1] = (u32)(h[i] >> 32);
  }
  // ---- BLAKE2b itself, sequential, without a key, with salt and personalization ----
  // The first chaining value is IV xor the parameter block (the BLAKE2 specification, https://www.blake2.net/blake2.pdf, section 2.8): byte 0 the
  // digest's length; byte 1, the key's length, 0; bytes 2 and 3, fanout and depth, 1; bytes 4 to 31 zero; bytes 32 to 47 the salt; bytes 48 to 63 the
  // personalization. Start: the 32 bytes of salt and personalization as eight words; zeros are what BLAKE2b has without them.
  struct Start {
    u32 w[8];
  };
  // The first word of 64 bits of the reference's parameter block (a blake2b_param of blake2.h), with the block's bytes as
  // blake2b_init of blake2b-ref.c sets them, read as load64 of blake2-impl.h reads eight bytes. Of load64's eight terms only
  // the four for bytes 0 to 3 are written: bytes 4 to 7 (leaf_length) are 0. The caller passes a digest_bytes from 1 to 64.
  BLK_FN static u64 param_word0(u32 digest_bytes) {
    return ((u64)digest_bytes << 0) | // byte 0: digest_length
           ((u64)0 << 8) |            // byte 1: key_length = 0
           ((u64)1 << 16) |           // byte 2: fanout = 1
           ((u64)1 << 24);            // byte 3: depth = 1
  }
  // Adapted from blake2b_init_param and blake2b_init0 of blake2b-ref.c: S->h[i] = blake2b_IV[i], then S->h[i] ^= load64 of the
  // parameter block's word i. Words 1 to 3 of the block are 0, and the xors with them are left out; words 4 and 5 are the
  // salt, words 6 and 7 the personalization.
  BLK_FN static void begin64(u64 h[8], const Start &start, u32 digest_bytes) {
    h[0] = blake2b_IV0 ^ param_word0(digest_bytes);
    h[1] = blake2b_IV1;
    h[2] = blake2b_IV2;
    h[3] = blake2b_IV3;
    h[4] = blake2b_IV4 ^ (start.w[0] | (u64)start.w[1] << 32);
    h[5] = blake2b_IV5 ^ (start.w[2] | (u64)start.w[3] << 32);
    h[6] = blake2b_IV6 ^ (start.w[4] | (u64)start.w[5] << 32);
    h[7] = blake2b_IV7 ^ (start.w[6] | (u64)start.w[7] << 32);
  }
  BLK_FN static void begin(u32 cv[CVW], const Start &start, u32 digest_bytes) {
    u64 h[8];
    begin64(h, start, digest_bytes);
    BLK_UNROLL
    for (int i = 0; i < 8; i++) cv[2 * i] = (u32)h[i], cv[2 * i + 1] = (u32)(h[i] >> 32);
  }
  // BLAKE2b's counter and flag: the bytes fed so far, this block's among them; the flag in the last call. An input has fewer than 2^32 bytes.
  BLK_FN static Tweak step(u32 s) { return Tweak{4 * BLKW * (s + 1), 0, 0, 0, false}; } // the call on block number s, which is not the last
  BLK_FN static Tweak last(u32 bytes) { return Tweak{bytes, 0, 0, 0, true}; }           // bytes: of the whole input

  // ---- for the modes of older.h ----
  // The first chaining value: the first state of BLAKE2b-512 (RFC 7693, section 3.3: IV with h[0] xor 0x01010040). This is what
  // blake2b_init(S, 64) of blake2b-ref.c leaves in S->h: param_word0(64) is 0x01010040, and the block's other seven words are 0.
  BLK_FN static void first64(u64 h[8]) {
    h[0] = blake2b_IV0 ^ param_word0(64);
    h[1] = blake2b_IV1;
    h[2] = blake2b_IV2;
    h[3] = blake2b_IV3;
    h[4] = blake2b_IV4;
    h[5] = blake2b_IV5;
    h[6] = blake2b_IV6;
    h[7] = blake2b_IV7;
  }
  BLK_FN static void first(u32 cv[CVW]) {
    u64 h[8];
    first64(h);
    BLK_UNROLL
    for (int i = 0; i < 8; i++) cv[2 * i] = (u32)h[i], cv[2 * i + 1] = (u32)(h[i] >> 32);
  }
};
} // namespace blk
