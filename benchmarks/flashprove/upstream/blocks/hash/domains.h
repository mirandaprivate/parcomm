// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Domain-separation numbers for H(tag, index, start; bytes) of blocks/hash/hash.h. Each use of the
// hash has a separate number: the tag's low 16 bits, so the first two bytes of the hashed string.
// The number alone decides if a hash has a start, a digest that sets its initial state
// (domain_has_start()).
// Another hash in this repository has numbers of its own (blocks/hash/older.h uses some): within
// 3 to 139, and 240 and 241; listed in verifier/HASHES.md, "The numbers that the sources pass".
// To keep one meaning per number across both hashes, stay in 160 to 239.
// No number is reused: 161, 163, 165, 167, 170 and 175 are retired. Add new numbers at the end.
#pragma once
#include "base.h"

namespace blk {
// A chain is a hash chain: the Fiat-Shamir transcript of one protocol (blocks/chain/chain.h); its
// label is an integer that says which protocol. Its first query is the string hashed under
// CHAIN_START. A run is one execution of a chain. The configuration digest is a hash of the public
// configuration files.
// The table of uses, with index, start and bytes: docs/format/FORMAT.md, R3.1.
// PUBLISHED_ID is computed by published_id() in blocks/frame/heads.h. A commitment holds the
// weights as one table and may store further copies of it. The anchor is the first table
// (`first_table_name` in published_id()); its record's answers are stored answers to samples of its
// rows at the record's seeds.
// For a proof, the inputs under COIN_INPUTS must cover all that the prover's messages depend on;
// `CoinInputs` (blocks/coins/run_coins.h) hashes whatever its caller adds. For the private coins of
// a commitment's table the inputs are public facts, nothing of the weights: lists::first_table()
// and lists::copy_table() in prover/common/coin_keys.h.
enum class Domain : u32 {
  CHAIN_START = 160,   // a chain's first value; index: the chain's label
  CHAIN_QUERY = 162,   // a hash call of a round: message, nonce; start: chain's value; index: round
  LEAF_MAIN = 164,     // a leaf: the symbols of a column, then its salt
  TREE_NODE = 166,     // a vertex from its two children; its tree, position and run are not hashed
  STATEMENT = 168,     // the digest of a statement
  PARAMETERS = 169,    // the digest of a parameter set
  TEST_COINS = 171,    // tests' private coins, toy programs' witnesses; start: seed; index: counter
  RUN = 172,           // run's value: parameters' digest and the chain's random bytes; index: label
  LEAF_BINDING = 173,  // a leaf of a tree whose root feeds a first query: no run's value exists yet
  LEAF_STORED_MAIN = 174, // as LEAF_MAIN, in a tree that is made once, stored and read by many runs
  COIN_INPUTS = 176,   // digest of named inputs (`CoinInputs`), which enters a key of private coins
  RESERVED = 177,      // never used, so a proof about the format has a string that is never hashed
  BINDING_NODE = 178,  // as TREE_NODE, in the tree of LEAF_BINDING
  PUBLISHED_ID = 179,  // commitment ID: configuration digest, anchor's name, its seeds and answers
};
// True for every number above but RESERVED. Constructing a `TagT` (blocks/hash/hash.h) with any
// other number ends the process or, on a GPU, the kernel: list a new number here.
BLK_HD constexpr bool domain_is_given_out(Domain d) {
  return d == Domain::CHAIN_START || d == Domain::CHAIN_QUERY || d == Domain::LEAF_MAIN || d == Domain::TREE_NODE || d == Domain::STATEMENT ||
         d == Domain::PARAMETERS || d == Domain::TEST_COINS || d == Domain::RUN || d == Domain::LEAF_BINDING || d == Domain::LEAF_STORED_MAIN ||
         d == Domain::COIN_INPUTS || d == Domain::BINDING_NODE || d == Domain::PUBLISHED_ID;
}
// True if the start is stored in the tag, not passed with each hash. A tree's leaves share a start:
// the run's value or, in a tree that no run made (LEAF_BINDING, LEAF_STORED_MAIN), the
// configuration digest. The vertices of every tree start from the configuration digest.
BLK_HD constexpr bool domain_start_in_tag(Domain d) {
  return d == Domain::LEAF_MAIN || d == Domain::TREE_NODE || d == Domain::LEAF_BINDING || d == Domain::LEAF_STORED_MAIN || d == Domain::BINDING_NODE;
}
// Source for leaves and vertices under different domain-separation numbers: RFC 6962 (Laurie, Langley, Kasper,
// "Certificate Transparency", 2013), section 2.1.
// The rule is kept in: RFC 9162 (Laurie, Messeri, Stradling, "Certificate Transparency Version 2.0", 2021), section
// 2.1.1, which obsoletes RFC 6962.
BLK_HD constexpr bool domain_is_leaf(Domain d) { return d == Domain::LEAF_MAIN || d == Domain::LEAF_BINDING || d == Domain::LEAF_STORED_MAIN; }
BLK_HD constexpr bool domain_is_vertex(Domain d) { return d == Domain::TREE_NODE || d == Domain::BINDING_NODE; }
BLK_HD constexpr bool domain_has_start(Domain d) {
  return domain_start_in_tag(d) || d == Domain::CHAIN_QUERY || d == Domain::TEST_COINS;
}
} // namespace blk
