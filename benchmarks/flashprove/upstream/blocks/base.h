// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Shared definitions of the blocks/ library (field, hash, codes, the proof system's steps): integer
// types, the host/device marker BLK_HD, the build settings, and the guards BLK_REQUIRE and
// BLK_REQUIRE_NAMED. A guard ends the process: it is not a way to reject a proof.
#pragma once
#include "fence.h" // first of all: keeps macros that are only for tests out of production builds
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(__CUDACC__)
#define BLK_HD __host__ __device__
#else
#define BLK_HD
#endif

// GPU code must not be compiled by an nvcc older than 13.2.86 (CUDA 13.2 Update 2):
// prover/check_nvcc_version.sh gives the reason. prover/build.sh runs that script before it runs
// nvcc; this directive also stops an older nvcc that is started in another way.
#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ * 1000000 + __CUDACC_VER_MINOR__ * 10000 + __CUDACC_VER_BUILD__ < 13020086
#error "an nvcc older than 13.2.86 (CUDA 13.2 Update 2) cannot be used: see prover/check_nvcc_version.sh"
#endif

// Build settings. BLK_FIELD must be given and must be 2: p = 2^64 - 2^32 + 1 (Goldilocks) with
// plain Reed-Solomon codes; there is no other field. BLK_HASH: 1 = BLAKE2b, the only value
// (blocks/hash/hash.h rejects any other). The two numbers are the last two characters of the file
// format's version word, `H821` (blocks/frame/version.h).
#if !defined(BLK_FIELD) || BLK_FIELD != 2
#error "compile with -DBLK_FIELD=2 (p = 2^64 - 2^32 + 1): there is no other field"
#endif
#ifndef BLK_HASH
#define BLK_HASH 1
#endif

// `blk` is the namespace of the blocks/ library. The macros its headers define begin with BLK_.
namespace blk {
using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;
using u128 = unsigned __int128;

// Zeroes n bytes through a volatile pointer, so that the compiler cannot drop the writes. For
// secrets in memory that is about to be freed or to go out of scope.
inline void wipe(void *p, size_t n) {
  volatile u8 *v = static_cast<volatile u8 *>(p);
  for (size_t i = 0; i < n; i++) v[i] = 0;
}

// Caps on the rejection sampling of challenges from a hash chain, the Fiat-Shamir transcript
// (blocks/chain/chain.h): a field element is 0 after DRAW_TRIES rejected 64-bit words; an element
// of the extension field E that must not be 0 is 1 after NONZERO_TRIES zeros.
// They are defined here because blocks/numbers/plan.h (the plan: the list of rounds) needs them and
// does not include that header. It computes from them what the caps add to a round's error and the
// most words that a verifier's message can use. So the chain that runs and the computed error use
// the same caps.
constexpr unsigned DRAW_TRIES = 6, NONZERO_TRIES = 2;

// The smallest a with 2^a >= v; 0 for v <= 1.
BLK_HD inline unsigned ceil_log2(u64 v) {
  unsigned a = 0;
  while (a < 64 && (u64(1) << a) < v) a++;
  return a;
}

// Callback that a program can set, to run before a failed guard aborts the process. If it is null,
// as in the program `cp_verify`, a failed guard prints its line to stderr and calls abort().
// Two programs set it. The prover program `cp_prove` (arm_ends() in prover/prover/ends.h) sets one
// that calls stop(), which runs that program's wipe of its secrets, then exits. `cp_commit`, which
// commits to the weights (cp_commit_program() in prover/hash_commitment/hf_commit.cu), sets one
// that prints a line to stderr with `name` (see below), or "failed-check" if `name` is null, and
// returns.
// The callback runs on the failing thread, after the guard's line, outside any signal handler. If
// it returns, abort() follows. It may be entered from several threads at once, and must take no
// lock that a thread can hold at a guard. It cannot throw: the pointer's type is noexcept (tested
// by blocks/tests/must_not_build/a_guards_hook_that_may_throw.cpp), as are the two functions below,
// so that no exception leaves a failed guard for a caller that might catch it and continue.
// A translation unit that is linked with its symbols made local has its own copy of this variable,
// null until assigned. In `cp_prove`, the units that own the weight table get the pointer at set-up
// (hf_keeper_unit() in prover/hash_commitment/hf_keeper.cu); the verifier's unit
// (verifier/src/cp_verify.cpp) does not, so a guard that fails there calls abort()
// without the wipe.
// `name`: null, a guard's name (see BLK_REQUIRE_NAMED) or "cuda-call" (prover/engine/device.cuh);
// a string literal. Only the callback receives it: where this pointer is null, the name is unused.
// `cp_prove` prints it in its last line, because its production build redirects stderr to
// /dev/null (close() in prover/prover/closed_front.h). Being an argument, it reaches the program's
// callback from every unit whose own copy of this pointer has been assigned.
inline void (*at_a_failed_guard)(const char *name) noexcept = nullptr;
// Ends the process after a failed guard's line, and after the line that
// prover/engine/device.cuh prints when a CUDA call fails.
[[noreturn]] inline void end_at_a_failed_guard(const char *name = nullptr) noexcept {
  static thread_local bool failing = false;  // a guard that fails inside the call below, on the same thread, falls through to abort(): no loop
  void (*const call)(const char *) noexcept = failing ? nullptr : at_a_failed_guard;
  failing = true;
  if (call) call(name);
  std::abort();  // also if the call returns
}
[[noreturn]] inline void require_failed(const char *file, int line, const char *text, const char *name = nullptr) noexcept {
  std::fprintf(stderr, "%s:%d: %s\n", file, line, text);
  end_at_a_failed_guard(name);
}
} // namespace blk

// `BLK_REQUIRE` ends the process at a mistake of the program itself: a protocol block
// called out of order, a mask (a private random value) used twice, a size that does not fit its
// plan. No guard may depend on the bytes of a proof: for a bad proof a verifier returns false.
// Host code only.
// BLK_REQUIRE_NAMED(cond, name, text): the same with a name, for the few self-checks that protect
// private random values: a value compared with what was drawn or written, a place that must be
// empty before its draw, a generator's self-test. Their names begin "own-test-": after such a name
// `cp_prove` exits with status 9 (prover/prover/ends.h). The name tells an operator that one of
// these self-checks has failed: the machine or the program may lose or repeat private random
// values. Find the cause before restarting it.
// Some guards stop a prover that would give secrets away in proofs that every verifier accepts.
// Without such a guard a run could complete, the verifier could accept, and a proof could
// distinguish two witnesses.
// Some of these guards are in the verifier's program too: at a layout
// that would leak (the layout is public: which oracle messages share a tree's leaves, and at which
// lists of positions each is read), it stops and gives no verdict; without the guard it would go on
// and give one. If the rule above is kept, no guard depends on a proof's bytes, and wherever a
// verifier gives a verdict, it is the one that it would give without the guard.
#define BLK_REQUIRE_NAMED(cond, name, text) \
  do { \
    if (!(cond)) ::blk::require_failed(__FILE__, __LINE__, text, name); \
  } while (0)
#define BLK_REQUIRE(cond, text) BLK_REQUIRE_NAMED(cond, nullptr, text)
