// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The format's version word: four characters packed into a u32, the first in the low byte: 'H',
   then one digit each for the format, the field (BLK_FIELD) and the hash (BLK_HASH).
   "H821": format 8, Goldilocks (p = 2^64 - 2^32 + 1) with plain Reed-Solomon codes, BLAKE2b.
   There is no other field or hash.
   BLK_FIELD and BLK_HASH are compile-time settings (blocks/base.h), so a build accepts exactly
   one word, SERVED_VERSION. `read_parameters` (blocks/frame/parameters.h) compares it with the
   word on the parameter file's first line. `read_head` (blocks/frame/heads.h) compares it with
   the word in the header of a key, leaf or proof file. `Binder` (blocks/chain/public.h) compares
   nothing: it writes SERVED_VERSION first among the parameters whose hash goes into the first
   query of a hash chain (a Fiat-Shamir transcript). */
#pragma once
#include <string>

#include "base.h"

namespace blk {

inline constexpr char FORMAT_NUMBER = '8';

inline constexpr u32 version_word(unsigned field, unsigned hash) {
  return (u32)'H' | (u32)FORMAT_NUMBER << 8 | (u32)('0' + field) << 16 | (u32)('0' + hash) << 24;
}
inline constexpr u32 SERVED_VERSION = version_word(BLK_FIELD, BLK_HASH);

inline std::string version_text(u32 w) {
  std::string s(4, ' ');
  for (int i = 0; i < 4; ++i) s[(size_t)i] = (char)(w >> (8 * i));
  return s;
}
// Inverse of version_text(). If s is not four characters long, returns 0, which
// version_refusal() rejects.
inline u32 version_from_text(const std::string &s) {
  if (s.size() != 4) return 0;
  u32 w = 0;
  for (int i = 0; i < 4; ++i) w |= (u32)(unsigned char)s[(size_t)i] << (8 * i);
  return w;
}
// Returns nullptr if w is SERVED_VERSION, otherwise the rejection name.
inline const char *version_refusal(u32 w) { return w == SERVED_VERSION ? nullptr : "version-unknown"; }

}  // namespace blk
