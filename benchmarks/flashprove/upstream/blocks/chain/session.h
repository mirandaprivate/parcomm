// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The state that the hash chains of one checked file share on the verifier's side: a session.
//
// The check of a file ends in one verdict and may run several hash chains (blocks/chain/chain.h: a
// chain is the Fiat-Shamir transcript of one interactive protocol). A proof file has its request's
// first chain, the chain that its batch shares, and its second chain; a key file (`ivk/key.bin`)
// also has three; a leaf file (`ivk/leaf/<k>.bin`) has one. Each check (check_proof(),
// check_key_under() and check_entry_under() in blocks/frame/) constructs one VerifierSession and
// passes it to each of its chains' channels (VerifierRounds, blocks/chain/rounds.h: the object that
// runs one chain); the session must outlive these channels. It holds a record of digests for each
// Merkle root against which an opening is checked (record()), and each chain's outcome, from which
// accepted() gives the verdict. A chain that follows another (Binder::follows(),
// blocks/chain/public.h) must be in that chain's session, or the constructor of VerifierRounds ends
// the process.
//
// A session lives inside one process. `cp_verify` checks the three files in three processes, each
// with a session of its own: no record reaches from one file's check to another's. Stated in
// docs/format/FORMAT.md, "17.2 Where this page and the programs are known to differ", item 4.
#pragma once
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "base.h"
#include "chain/public.h"
#include "tree/tree.h"

namespace blk {

class VerifierSession {
public:
  VerifierSession() = default;
  VerifierSession(const VerifierSession &) = delete; // a copy would split the records
  VerifierSession &operator=(const VerifierSession &) = delete;

  // The digests, computed or given in the proof, at the vertices below `root`, from the openings
  // that tree_check() has accepted against this record, in any chain of the session (TreeValues,
  // blocks/tree/tree.h). Only the root's 32 bytes select the record: trees sent with equal roots
  // share one, and so do the session's chains that read one tree or one committed table (a matrix
  // stored once under a Merkle tree). tree_check() returns false for an opening that gives a vertex
  // a second, different digest, and leaves the record as it was; the proof is then rejected.
  TreeValues &record(const Digest &root) {
    std::array<u32, 8> key;
    for (int i = 0; i < 8; i++) key[(size_t)i] = root.w[i];
    return records_[key];
  }
  size_t roots() const { return records_.size(); }
  size_t chains() const { return chains_.size(); }

  // The verdict: true if no chain was rejected (`refused`) and every chain's channel has run
  // finish() or finish_inside_a_file(). After a rejection it returns false even if chains are
  // unfinished, so a verifier may return at the first rejection and a rejected proof does not end
  // the process here. Called with no rejection and an unfinished chain, or with no chain, it ends
  // the process: that is a programming error, and stopping keeps a chain's result from being
  // dropped.
  [[nodiscard]] bool accepted() const {
    BLK_REQUIRE(!chains_.empty(), "session: a verdict on no chain");
    for (const Entry &c : chains_)
      if (c.fate->refused || (c.still_good && !c.still_good())) return false; // asks a live channel
    for (const Entry &c : chains_) BLK_REQUIRE(c.fate->ended, "session: a chain that was started has not been brought to its end");
    return true;
  }

private:
  friend class VerifierRounds;
  // One chain's outcome (ChainFate, blocks/chain/public.h). The constructor of VerifierRounds adds
  // it with enter(); the destructor sets `refused` if its ok() is false, then calls leave(). Until
  // then a failed read of the proof's bytes may show only in ok().
  struct Entry {
    std::shared_ptr<const ChainFate> fate;
    std::function<bool()> still_good; // the live channel's `ok`; null after `leave`
  };
  void enter(std::shared_ptr<const ChainFate> fate, std::function<bool()> still_good) { chains_.push_back(Entry{std::move(fate), std::move(still_good)}); }
  void leave(const std::shared_ptr<const ChainFate> &fate) {
    for (Entry &c : chains_)
      if (c.fate == fate) c.still_good = nullptr;
  }
  bool holds(const std::shared_ptr<const ChainFate> &fate) const {
    bool found = false;
    for (const Entry &c : chains_) found = found || c.fate == fate;
    return found;
  }
  std::map<std::array<u32, 8>, TreeValues> records_;
  std::vector<Entry> chains_;
};

} // namespace blk
