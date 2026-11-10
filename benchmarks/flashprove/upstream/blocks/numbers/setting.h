// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The inputs from which this directory derives parameters, round errors and sizes (`Setting`), and
   the enums by which a plan (blocks/numbers/plan.h: the list of a protocol's rounds) classifies its
   rounds. `setting_of` (blocks/frame/derive.h) makes a `Setting` from the parameter file
   (`policy.txt`). Nothing is checked here. */
#pragma once
#include <string>

#include "numbers/real.h"

namespace blk {

/* What the verifier's message of a round consists of. The channel (blocks/chain/rounds.h) lets a
   round draw only that. A round's query of the hash chain (the Fiat-Shamir transcript,
   blocks/chain/chain.h) hashes `ChainKind`'s number, not this enum's. SEAL, RECORD and EMPTY rounds
   have no proof of work (SEAL: Plan::close() ends the process otherwise; see `Setting::work`).
     ALGEBRAIC  elements of the challenge field E
     EPSILON    one element of E, which combines a claim with a hiding polynomial's sum; that sum
                must be the last thing sent before it
     SEED       seeds of out-of-domain samples (random linear forms of a sent message): for each, a
                point and a pad factor
     POSITIONS  positions of a code: uniform, independent, with repetition; in the code switch (CS),
                then one element of E
     BITS       a matrix with entries in {-1, 0, 1}, for the approximate range check (RNG)
     SEAL       nothing for the protocol: one more query, which ends a chain and binds all of it in
                one digest, for the first query of a chain that follows it
     RECORD     the seeds of a committed table's own samples, drawn on a separate chain whose
                statement is the table's name; not a round of the plan's chain
     EMPTY      nothing. The round still makes its query, which takes in the prover's message. */
enum class RoundKind { ALGEBRAIC = 1, EPSILON = 2, SEED = 3, POSITIONS = 4, BITS = 5, SEAL = 6, RECORD = 7, EMPTY = 0 };  // the numbers are fixed for good

/* The argument that bounds a round's error. With `Scope`, it selects the round's bits of proof of
   work.
     FIELD      a non-zero polynomial vanishes at the challenge: degree over |E|, times a list size
                where one applies
     SAMPLE     two codewords of a list have the same out-of-domain sample
     CLOSENESS  the proximity term: a combination of words is close to the code although the words
                are not jointly close (mutual correlated agreement); the only kind that can rest on
                the cited bound up to the Johnson radius (see `CountSource`)
     QUERY      a word that is far from what is claimed agrees with it at all drawn positions
     SMALL      the range check's error: a vector that is not small has small images under all drawn
                combinations
     ZERO       no error: SEAL and EMPTY rounds */
enum class ErrorKind { FIELD, SAMPLE, CLOSENESS, QUERY, SMALL, ZERO };

/* Which bound gives the count of exceptional challenges in a CLOSENESS error, the best-founded
   first. codes.h describes each. JOHNSON, the cited bound on mutual correlated agreement up to the
   Johnson radius, is taken as given. It is Theorem 4.6, taken with M = 1, of IACR ePrint report
   2025/2055 as posted on 6 November 2025 (a paper on proximity gaps of Reed-Solomon codes; authors
   Ben-Sasson, Carmon, Haböck, Kopparty and Saraf). It has an integer m >= 3 that ties the slack
   eta = 1 - sqrt(rho) - gamma to the size of the bound (gamma: the distance, delta in codes.h). The
   programs take m as the theorem prints it: m = max(ceil(sqrt(rho) / eta), 3). The same paper's
   Theorem 1.5 and page 24 have sqrt(rho) / (2 eta), and so has Theorem 4.12 of IACR ePrint report
   2026/680 (a survey of open problems on list decoding and correlated agreement; authors Arnon,
   Boneh and Fenzi). The printed form is the weaker reading: at a given slack its m, and so its
   bound, is never smaller.

   A combination of the rows of an interleaved word costs as much in a CLOSENESS error as a line
   through two words: an interleaved Reed-Solomon code is a Reed-Solomon code over a larger field on
   the same domain, and the cited bound counts exceptional challenges without regard to the field.
   That is our own short argument, unrefereed. Lemma 4.7 of ePrint 2026/680 has the number of rows
   times that; the programs do not use it. */
enum class CountSource { THIRD, JOHNSON };

// How often a round's proof of work is paid: per request, per batch, or once, by the committer.
enum class Scope { REQUEST, SHARED, ONCE };

struct Setting {
  std::string field_name;
  Big p;                  // the base field F has p elements
  unsigned degree = 3;    // E has p^degree elements
  bool circle = false;    // codes are circle codes (a subcode of codimension 1 of a Reed-Solomon code), not plain ones
  // bytes of one element of F, and of one of E
  unsigned f_bytes = 8, e_bytes = 24;
  unsigned digest_bytes = 32, salt_bytes = 40, nonce_bytes = 8;
  unsigned position_bytes = 0;      // always 0: a proof lists no drawn position. A plan's words hold the number (blocks/chain/public.h).
  unsigned hash_block_bytes = 128;  // what one call of the compression function takes in
  unsigned label_bytes = 16;        // every hashed string starts with a label: tag word, index word
  unsigned chain_extra_bytes = 32;  // in a query besides the prover's message: label, kind, nonce
  /* 8-byte words of a verifier's message that a query's first hash call yields; none if the round
     has proof of work. Each further call yields eight. */
  unsigned chain_free_words = 4;
  unsigned target = 128;            // every round's error times 2^-work is to be at most 2^-target
  // The programs leave `instances` at 1 (`setting_of` does not set it): their bound on a REQUEST
  // round is for one request alone, not jointly for the requests of a batch. Outside tests, only
  // the planning tool cp_plan.cpp sets it, from the line `instances` of its input file.
  u64 instances = 1;                // how many requests a REQUEST round's error is multiplied by
  unsigned samples = 1;             // answers to each out-of-domain sample
  /* true: a round polynomial is sent without the coefficient that the verifier's check fixes; the
     programs always do so. false only plans the cost of sending all coefficients; such a plan
     cannot be run. */
  bool saved_coefficient = true;
  // Row ZERO must stay 0: a SEAL round would take its bits (Plan::close() then ends the process).
  unsigned work[6][3] = {};         // bits of proof of work by ErrorKind and Scope

  Big card_E() const { return Big::pow(p, degree); }
  unsigned work_of(ErrorKind k, Scope s) const { return work[(int)k][(int)s]; }
  void set_work(ErrorKind k, Scope s, unsigned bits) { work[(int)k][(int)s] = bits; }
};

// p = 2^61 - 1, cubic extension, circle codes. Not the programs' field: only the tests in
// blocks/numbers/tests/ call this, and no program turns `circle` on.
inline Setting setting_m61() {
  Setting s;
  s.field_name = "2^61 - 1";
  s.p = Big::two_to(61) - Big(1);
  s.circle = true;
  return s;
}
// p = 2^64 - 2^32 + 1, cubic extension, plain Reed-Solomon codes.
inline Setting setting_goldilocks() {
  Setting s;
  s.field_name = "2^64 - 2^32 + 1";
  s.p = Big::two_to(64) - Big::two_to(32) + Big(1);
  s.circle = false;
  return s;
}

// The setting for the programs' field. `Cursor` (plan.h) runs only a plan over that field.
inline Setting setting_of_this_build() { return setting_goldilocks(); }

}  // namespace blk
