// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Macros for code compiled for host and GPU; `Tweak`: what a call of F, BLAKE2b's compression
// function (blake2b_f.h), takes besides its chaining value and its block; and an optional call
// counter.
#pragma once
#include "base.h"
#if BLK_COUNT_HASH_CALLS
#include <atomic>
#endif

// BLK_FN: a small function for host and GPU, always inlined on the GPU.
#if defined(__CUDACC__)
#define BLK_FN __host__ __device__ __forceinline__
#else
#define BLK_FN inline
#endif
// BLK_UNROLL: a loop over a small per-thread array is unrolled on the GPU, so that the array stays
// in registers (the compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide",
// page "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// for loops over small per-thread arrays: M. Milakov,
// "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/).
// BLK_HASH_REQUIRE: a caller's mistake stops the run: the host process through BLK_REQUIRE
// (blocks/base.h), a kernel by __trap() (same page of the "CUDA Programming Guide",
// "Trap function"), after which every later CUDA call of the process reports an error. It is never
// compiled out: the bounds that it checks keep two different hash calls of the proof format from
// becoming the same call of F.
#if defined(__CUDA_ARCH__)
#define BLK_UNROLL _Pragma("unroll")
#define BLK_HASH_REQUIRE(cond, text) \
  do { \
    if (!(cond)) __trap(); \
  } while (0)
#else
#define BLK_UNROLL
#define BLK_HASH_REQUIRE(cond, text) BLK_REQUIRE(cond, text)
#endif

namespace blk {
// The tweak of a call: F's flag, and F's 128-bit counter as the four words c0, c1, num, l, lowest
// first. The struct declares l before num.
//   last    the flag: set only on the last call of a BLAKE2b hash or of Older::list_state();
//   c0, c1  the number of bytes fed so far; in Older::bare(), two words that its caller chooses;
//   num     0 in BLAKE2b; in older.h a domain-separation number; bits 16 up may name a tree;
//   l       0 in BLAKE2b and in Older::bare(); in Older::list_state() the list's length in words.
struct Tweak {
  u32 c0, c1, l, num;
  bool last;
};

// With -DBLK_COUNT_HASH_CALLS=1 the host's calls of Blake2bF::compress64() are counted; direct
// calls of F and GPU calls are not counted.
// BLK_COUNT_HASH_CALLS is not a test switch and is deliberately not listed in blocks/fence.h,
// whose switches do not compile with SERVING_BUILD: verifier/build.sh sets it with SERVING_BUILD,
// and the verifier `cp_verify` includes the count in the number of hash calls that it prints.
// The count follows from the public shapes and the words drawn from the hash chain, the Fiat-Shamir
// transcript (a rejected word is replaced by the next), so anyone who has the proof can compute it.
#if BLK_COUNT_HASH_CALLS
inline std::atomic<u64> hash_calls_made{0};
inline void count_hash_call(const Tweak &) { hash_calls_made.fetch_add(1, std::memory_order_relaxed); }
inline u64 hash_calls_counted() { return hash_calls_made.load(); }
#endif
#if BLK_COUNT_HASH_CALLS && !defined(__CUDA_ARCH__)
#define BLK_COUNT_CALL(t) ::blk::count_hash_call(t)
#else
#define BLK_COUNT_CALL(t) ((void)0)
#endif
} // namespace blk
