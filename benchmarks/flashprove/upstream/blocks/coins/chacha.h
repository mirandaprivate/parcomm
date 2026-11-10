// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The block function of ChaCha20, for host code and for GPU kernels.
// f(K; i12, i13, i14, i15) is the block function of RFC 8439, section 2.3 (Y. Nir, A. Langley,
// "ChaCha20 and Poly1305 for IETF Protocols", June 2018, https://www.rfc-editor.org/rfc/rfc8439).
// The state's words 0 to 3 are the RFC's constants and words 4 to 11 the key K. Word 12, where the
// RFC has its block counter, is i12; words 13 to 15, where it has its nonce, are i13 to i15. After
// twenty rounds the initial state is added. The output, a block, has sixteen words. The code is
// carried over from prover/common/coin_chacha.h, mostly under other names. It has no loop over
// rounds and indexes no array by a variable, so that in a kernel all words can stay in registers.
#pragma once
#include "hash/tweak.h" // BLK_FN

namespace blk {
BLK_FN u32 chacha_rotl(u32 x, int n) { // 0 < n < 32
#if defined(__CUDA_ARCH__)
  return __funnelshift_l(x, x, n); // NVIDIA, CUDA Math API Reference Manual, "Integer Intrinsics"
#else
  return (x << n) | (x >> (32 - n));
#endif
}
// The ChaCha quarter round: replaces the four words by its four results. The arguments must be four
// different objects. The words are secret, so only additions, exclusive ors and rotations by fixed
// counts touch them. Each rotation is a call of chacha_rotl(), the only rotation in this file.
// Adapted from the macro QUARTERROUND of D. J. Bernstein's ChaCha code (chacha-merged.c, version 20080118, public domain).
// RFC 8439, section 2.1, specifies the same function.
BLK_FN void chacha_quarter_round(u32 &a, u32 &b, u32 &c, u32 &d) {
  a = a + b, d = chacha_rotl(d ^ a, 16);
  c = c + d, b = chacha_rotl(b ^ c, 12);
  a = a + b, d = chacha_rotl(d ^ a, 8);
  c = c + d, b = chacha_rotl(b ^ c, 7);
}
// One column round and one diagonal round (RFC 8439, 2.3). The order of the eight quarter rounds is that of the round
// loop in D. J. Bernstein's chacha-merged.c.
#define BLK_CHACHA_DOUBLE_ROUND() \
  do { \
    ::blk::chacha_quarter_round(x0, x4, x8, x12); \
    ::blk::chacha_quarter_round(x1, x5, x9, x13); \
    ::blk::chacha_quarter_round(x2, x6, x10, x14); \
    ::blk::chacha_quarter_round(x3, x7, x11, x15); \
    ::blk::chacha_quarter_round(x0, x5, x10, x15); \
    ::blk::chacha_quarter_round(x1, x6, x11, x12); \
    ::blk::chacha_quarter_round(x2, x7, x8, x13); \
    ::blk::chacha_quarter_round(x3, x4, x9, x14); \
  } while (0)

// out = f(key; i12, i13, i14, i15). out must not overlap key.
BLK_FN void chacha20_block(u32 out[16], const u32 key[8], u32 i12, u32 i13, u32 i14, u32 i15) {
  const u32 c0 = 0x61707865u, c1 = 0x3320646eu, c2 = 0x79622d32u, c3 = 0x6b206574u;
  u32 x0 = c0, x1 = c1, x2 = c2, x3 = c3;
  u32 x4 = key[0], x5 = key[1], x6 = key[2], x7 = key[3];
  u32 x8 = key[4], x9 = key[5], x10 = key[6], x11 = key[7];
  u32 x12 = i12, x13 = i13, x14 = i14, x15 = i15;
  BLK_CHACHA_DOUBLE_ROUND(); //  1
  BLK_CHACHA_DOUBLE_ROUND(); //  2
  BLK_CHACHA_DOUBLE_ROUND(); //  3
  BLK_CHACHA_DOUBLE_ROUND(); //  4
  BLK_CHACHA_DOUBLE_ROUND(); //  5
  BLK_CHACHA_DOUBLE_ROUND(); //  6
  BLK_CHACHA_DOUBLE_ROUND(); //  7
  BLK_CHACHA_DOUBLE_ROUND(); //  8
  BLK_CHACHA_DOUBLE_ROUND(); //  9
  BLK_CHACHA_DOUBLE_ROUND(); // 10
  out[0] = x0 + c0, out[1] = x1 + c1, out[2] = x2 + c2, out[3] = x3 + c3;
  out[4] = x4 + key[0], out[5] = x5 + key[1], out[6] = x6 + key[2], out[7] = x7 + key[3];
  out[8] = x8 + key[4], out[9] = x9 + key[5], out[10] = x10 + key[6], out[11] = x11 + key[7];
  out[12] = x12 + i12, out[13] = x13 + i13, out[14] = x14 + i14, out[15] = x15 + i15;
}
#undef BLK_CHACHA_DOUBLE_ROUND
} // namespace blk
