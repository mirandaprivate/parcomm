// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Keeps macros (`switch` in messages) that are only for tests out of production builds (`SERVING`
// in macro names). Some test macros let a test disable checks, choose the private randomness
// (`COINS` in macro names) or call functions that only a test may call. Programs outside blocks/
// compile its headers too, and one such macro in a production build would make a prover that leaks
// or a verifier that is no verifier. So: (1) the list below leads to one macro,
// BLK_NOT_FOR_SERVING; (2) BLK_NOT_FOR_SERVING does not compile with SERVING_BUILD, which a
// production build must define, as prover/build.sh and verifier/build.sh do; (3) a program built
// under BLK_NOT_FOR_SERVING contains the string of BLK_MARK_NOT_FOR_SERVING(), which a build script
// can search for, as prover/build.sh does. Two of the three messages below still speak of a
// cheating prover: the code that makes the honest prover malicious, for tests, is not part of this
// repository.
#pragma once

// The list sees only macros defined on the command line or before this file is first included
// (blocks/base.h starts with it). Add each new macro here: stage `table` of tests/new/run_new.py
// reports a missing one only if named BLK_CHEAT_.., BLK_TEST_.., BLK_.._FOR_A_TEST,
// BLK_.._FOR_TESTS_ONLY or BLK_.._HIDE_NOTHING.

// blocks/code/cosets.h defines BLK_CODE_FOR_TESTS_ONLY only after this file, if
// BLK_GL64_SUBGROUP_LOG is not 32, so this list tests that value itself.
#if \
    defined(BLK_TEST_PROVER_MESSAGES_OUTSIDE) || defined(BLK_CODE_FOR_TESTS_ONLY) || (defined(BLK_GL64_SUBGROUP_LOG) && BLK_GL64_SUBGROUP_LOG != 32) || \
    defined(BLK_FOLD_PRINT) || \
    defined(BLK_KEY_UNDER_FOR_A_TEST) || defined(BLK_ENTRY_UNDER_FOR_A_TEST) || defined(BLK_LISTED_COINS_FOR_A_TEST) || defined(BLK_REPEATABLE_COINS_HIDE_NOTHING) || \
    defined(BLK_WF_ROWS_FOR_A_TEST) || defined(BLK_LEFT_BEHIND_LOOK_FOR_A_TEST) || defined(BLK_PLANS_BY_HAND_FOR_A_TEST) || \
    defined(BLK_TESTED_TABLE_FOR_A_TEST)
#define BLK_NOT_FOR_SERVING 1
#endif

#if defined(BLK_NOT_FOR_SERVING) && defined(SERVING_BUILD)
#error "a switch of a test or of a cheating prover on the build line of a program that serves"
#endif

// For a header that only tests may include, and for code under a test macro that a source can
// define after including this file, too late for the list: BLK_MARK_NOT_FOR_SERVING(); at file
// scope puts the string below into the program, and does not compile in a production build.
// About half of the headers that read a listed macro do not call it, and none of prover/engine/
// does: there, a macro defined that late compiles in a production build and leaves no string.
// __COUNTER__ gives each use a distinct variable name, so one translation unit may have several.
#define BLK_FENCE_JOIN2(a, b) a##b
#define BLK_FENCE_JOIN(a, b) BLK_FENCE_JOIN2(a, b)
#ifdef SERVING_BUILD
#define BLK_MARK_NOT_FOR_SERVING() static_assert(false, "a header for tests alone in a program that serves")
#else
#define BLK_MARK_NOT_FOR_SERVING() \
  __attribute__((used)) static const char BLK_FENCE_JOIN(blk_not_for_serving_mark_, __COUNTER__)[] = \
      "BLK-NOT-FOR-SERVING: built with a switch of a test or of a cheating prover"
#endif

#ifdef BLK_NOT_FOR_SERVING
BLK_MARK_NOT_FOR_SERVING();
#endif
