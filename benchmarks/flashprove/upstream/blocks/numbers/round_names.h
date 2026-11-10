// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The round names of a hash chain (a Fiat-Shamir transcript). A protocol block (blocks/iop/) passes
   the constant to the channel (blocks/chain/rounds.h) to close a round; `Cursor`
   (blocks/numbers/plan.h) stops the program unless the plan's next entry has that name. The strings
   are part of the proof format: they are hashed into a chain's first query, and `kind_of_round`
   (blocks/numbers/plan.h) gives each name's kind. The same name serves every use of a block.
   The verifier's message is empty in R_SEND_ANSWER, R_SPLIT, R_PSC_VALUES, R_LAST and R_REPLY
   (closed with `close_empty`). R_RECORD (the seeds of a committed table's stored samples, drawn on
   a separate chain) is a plan entry but no round of the plan's chain; `Cursor` skips it.
   No plan outside a test (blocks/chain/tests/rounds_test.cpp) has R_LAST.
   PSC: sumcheck with a hiding polynomial. LKP: lookup. RNG: approximate range check. */
#pragma once

namespace blk {

inline constexpr const char *R_SEND_SEED = "SEND seed";
inline constexpr const char *R_SEND_ANSWER = "SEND answer";
inline constexpr const char *R_BAT = "BAT";
inline constexpr const char *R_BAT_SIGMA = "BAT-sigma";
inline constexpr const char *R_SPLIT = "SPLIT";
inline constexpr const char *R_PSC_EPSILON = "PSC epsilon";
inline constexpr const char *R_PSC_ROUND = "PSC round";
inline constexpr const char *R_PSC_Y = "PSC Y";
inline constexpr const char *R_PSC_VALUES = "PSC values";
inline constexpr const char *R_ZERO = "ZERO";
inline constexpr const char *R_LINE = "LINE";
inline constexpr const char *R_LKP_SHIFT = "LKP shift";
inline constexpr const char *R_JOIN = "JOIN";
inline constexpr const char *R_LKP_PLANES = R_JOIN, *R_FOREST_POINT = R_JOIN;  // aliases of R_JOIN
inline constexpr const char *R_LKP = "LKP";
inline constexpr const char *R_RNG = "RNG";
inline constexpr const char *R_FOLD_EPSILON = "FOLD epsilon";
inline constexpr const char *R_FOLD_ROUND = "FOLD round";
inline constexpr const char *R_LOC = "LOC";
inline constexpr const char *R_CS = "CS";
inline constexpr const char *R_BASE_LAMBDA = "BASE lambda";
inline constexpr const char *R_BASE = "BASE";
inline constexpr const char *R_SEAL = "SEAL";
inline constexpr const char *R_LAST = "LAST";  // closes a chain's last prover message if still open
inline constexpr const char *R_REPLY = "REPLY";  // carries only the reply (next token), unmasked
inline constexpr const char *R_RECORD = "RECORD";
inline constexpr const char *R_LIN_POINT = "LIN point";
inline constexpr const char *R_LIN_ZETA = "LIN zeta";
inline constexpr const char *R_LINK_POINT = "LINK point";
inline constexpr const char *R_LINK = "LINK";
inline constexpr const char *R_LINK_PHI = "LINK phi";

}  // namespace blk
