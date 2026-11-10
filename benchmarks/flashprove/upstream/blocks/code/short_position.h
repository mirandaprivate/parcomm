// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The position at which a shorter codeword is read when a longer one is read at position x.
// A Merkle tree can contain main rows, codewords of 2^log_n symbols, and mask rows, codewords of
// 2^log_short <= 2^log_n symbols. The file format sets the short position to x mod 2^log_short
// (docs/format/FORMAT.md, R7.2), so that each short position goes with 2^(log_n - log_short)
// long ones. C++ and CUDA code should call this function, as the callers in blocks/ and
// prover/engine/ do, not repeat the formula.
// docs/format/format_rules.py has a copy in Python.
// Not checked: log_short <= log_n, log_short < 64 (at 64 the shift is undefined).
#pragma once
#include "base.h"

namespace blk {
// Follows the definition of the repetition transformation of
// the paper that this protocol builds on (see THIRD_PARTY.md).
BLK_HD inline u64 short_position(unsigned log_n, unsigned log_short, u64 x) {
  (void)log_n; // unused by this map
  return x & (((u64)1 << log_short) - 1);
}
} // namespace blk
