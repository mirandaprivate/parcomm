// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The state of a protocol block's prover and verifier (`ProverContext`, `VerifierContext`), and the
// code that decides which Merkle tree an oracle message is stored in. A block is one step of the
// proof system, such as SEND (send.h); a hash chain is one protocol's Fiat-Shamir transcript.
// An oracle message (`OracleShape`, relation.h) is a set of encoded main rows and mask rows. One
// call of send_oracles() / receive_oracles() takes all those of a prover message; lay_trees()
// decides which share a tree. open_oracles() / read_opened() open whole trees at a position list.
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "chain/public.h"
#include "chain/rounds.h"
#include "code/short_position.h"
#include "engine/engine.h"
#include "iop/relation.h"
#include "numbers/blocks.h"
#include "tree/opening.h"

#ifdef BLK_CODE_FOR_TESTS_ONLY
#error "a code switched for tests (code/cosets.h): no proof is made or checked in such a build"
#endif

namespace blk {

// An oracle message that the prover is about to send. Oracle messages of one prover message with
// equal `leaf_class` can share leaves: they must always be opened together, at the same lists.
// `mask_draw` is for not_as_drawn() (blocks/iop/send.h), which counts the values of `mask` that
// differ from those drawn for it. A caller that wants `mask` compared sets `mask_draw` to
// Coins::last_draw() directly after the one draw that filled `mask` (e.g. in Send::fresh_mask()).
// Left at NO_DRAW, `mask` is not compared, and nothing reports that. This is so for the masks drawn
// in blocks/iop/foldeq.h, blocks/iop/req.h and blocks/iop/rng.h, for a table made from several
// draws (a fold mask's; blocks/iop/fold.h compares these draws itself), and for the masks that the
// prover program cp_prove draws for the messages at a request's start
// (prover/prover/chain_frame.cuh). cp_prove compares the first of these messages, the index-mask
// message (masks for the request's claims on the weight table), with its draw by a self-check of
// its own (prover/prover/chain_run.cuh), and any other of them with nothing. `mask_draw` is set in
// blocks/frame/check_key.h, blocks/iop/wf.h, blocks/iop/base_case.h and blocks/iop/lin.h.
struct OracleDraft {
  OracleShape shape;
  TableId main = NO_ID; // shape.main_rows rows, in the engine (the prover's CPU or GPU back end)
  Table mask;           // shape.mask_rows rows over E, the field's cubic extension
  unsigned leaf_class = 0;
  u32 mask_draw = Coins::NO_DRAW; // the number of the one draw that filled `mask`; see above
};
// The public half of a draft.
struct OracleNotice {
  OracleShape shape;
  unsigned leaf_class = 0;
};

// 32 bytes of private randomness (`coins` in identifiers), for a hash chain's salt.
inline Digest random_digest(Coins &coins) {
  u8 bytes[DIGEST_BYTES];
  coins.bytes(bytes, sizeof bytes);
  Digest d;
  digest_from_bytes(bytes, d);
  return d;
}
// What blocks/chain/public.h asks of a type bound into a chain: canonical bytes, a validity check.
inline void public_bytes(Writer &w, const Code &c) {
  for (u64 x : {(u64)c.log_n, (u64)c.ell, (u64)c.r}) w.put_u64(x);
}
inline void public_bytes(Writer &w, const Codes &c) {
  public_bytes(w, c.base);
  public_bytes(w, c.mask);
}
// Leaves out `name`, a label for output. One encoding per shape: without main rows, `code` is
// written as NONE_ID and `main_deg` as 1; without mask rows, `mask_code` as NONE_ID.
inline void public_bytes(Writer &w, const OracleShape &s) {
  for (u64 x : {(u64)(s.main_rows ? s.code : NONE_ID), (u64)(s.mask_rows ? s.mask_code : NONE_ID), (u64)s.main_rows,
                (u64)(s.main_rows ? s.main_deg : 1), (u64)s.mask_rows, (u64)s.answers})
    w.put_u64(x);
}
inline void public_bytes(Writer &w, const OracleNotice &n) {
  public_bytes(w, n.shape);
  w.put_u64(n.leaf_class);
}
inline const char *public_fault(const Code &c) { return code_ok(c) ? nullptr : "a code's numbers are no code of this field"; }
inline const char *public_fault(const Codes &c) {
  const char *fault = public_fault(c.base);
  return fault ? fault : public_fault(c.mask);
}
// Does not check that the codes that the shape names exist: Statement::sent() does.
inline const char *public_fault(const OracleShape &s) {
  if (!s.main_rows && !s.mask_rows) return "an oracle message without rows";
  if (s.main_rows && s.code == NONE_ID) return "an oracle message has main rows and names no code for them";
  if (s.mask_rows && s.mask_code == NONE_ID) return "an oracle message has mask rows and names no mask code";
  if (s.main_rows && s.main_deg != 1 && s.main_deg != E_DEGREE) return "an oracle message's main rows are over F or over E";
  return nullptr;
}
inline const char *public_fault(const OracleNotice &n) { return public_fault(n.shape); }

// Which oracle messages of one prover message share a tree: those of one `leaf_class` whose main
// rows have one length. An oracle message without main rows goes into the first tree of its class
// or, if there is none, into a tree of mask rows only. The leaf at position x contains the main
// rows' symbols at x, then copies of the mask rows' symbols at the short position of x (x modulo
// their length), and one salt. Trees are numbered in order of creation, those with main rows first.
// Both sides call lay_trees(); a plan uses tree_sizes(). Neither checks that `sent`'s codes exist.
struct TreeLay {
  std::vector<unsigned> tree, part;    // for each oracle message: its tree within the prover message, its part in that tree
  std::vector<OracleTreeShape> shapes; // for each tree (its number in the proof is still to be set)
};
inline TreeLay lay_trees(const Codes &codes, const std::vector<OracleNotice> &sent, size_t salt_bytes = 0 /* for counting trees */) {
  TreeLay lay;
  lay.tree.assign(sent.size(), 0), lay.part.assign(sent.size(), 0);
  std::vector<unsigned> leaf_class;
  for (int mask_only = 0; mask_only < 2; mask_only++)
    for (size_t i = 0; i < sent.size(); i++) {
      const OracleShape &s = sent[i].shape;
      if ((s.main_rows == 0) != (mask_only == 1)) continue;
      size_t t = 0;
      while (t < lay.shapes.size() &&
             !(leaf_class[t] == sent[i].leaf_class && (mask_only || lay.shapes[t].log_n == codes.base[s.code].log_n)))
        t++;
      if (t == lay.shapes.size()) {
        lay.shapes.emplace_back();
        leaf_class.push_back(sent[i].leaf_class);
        if (!mask_only) lay.shapes[t].log_n = codes.base[s.code].log_n;
      }
      lay.tree[i] = (unsigned)t;
    }
  for (size_t i = 0; i < sent.size(); i++) { // a tree's parts are in the order of sending
    const OracleShape &s = sent[i].shape;
    OracleTreeShape &t = lay.shapes[lay.tree[i]];
    if (s.mask_rows) {
      const unsigned log_n_zk = codes.mask[s.mask_code].log_n;
      BLK_REQUIRE(!t.has_mask() || t.log_n_zk == log_n_zk, "trees: mask rows of two lengths in one tree");
      t.log_n_zk = log_n_zk;
    }
    lay.part[i] = (unsigned)t.main_symbols.size();
    t.main_symbols.push_back(s.main_rows * s.main_deg);
    t.mask_symbols.push_back(s.mask_rows * E_DEGREE);
  }
  for (OracleTreeShape &t : lay.shapes) {
    t.salt_bytes = salt_bytes;
    if (!t.has_main()) t.log_n = t.log_n_zk;
    BLK_REQUIRE(!t.has_mask() || t.log_n_zk <= t.log_n, "trees: mask rows longer than the main rows above them");
  }
  return lay;
}
// The layout's sizes, for a plan (blocks/numbers/blocks.h): one `TreeSize` per tree, so per root.
inline std::vector<TreeSize> tree_sizes(const Codes &codes, const std::vector<OracleNotice> &sent) {
  std::vector<TreeSize> sizes;
  const TreeLay lay = lay_trees(codes, sent); // a plan adds the salts itself
  for (size_t i = 0; i < lay.shapes.size(); i++) {
    const OracleTreeShape &t = lay.shapes[i];
    TreeSize z;
    for (size_t o = 0; o < sent.size(); o++)
      if (lay.tree[o] == i) z.name += (z.name.empty() ? "tree of " : "; ") + sent[o].shape.name;
    z.log_n = t.log_n;
    z.leaf_bytes = (t.main_width() + t.mask_width()) * F_BYTES; // main symbols, then mask symbols
    for (size_t o = 0; o < sent.size(); o++)
      if (lay.tree[o] == i && sent[o].shape.answers == 0) z.one_list = true; // the plan rejects a second list on such a tree
    sizes.push_back(z);
  }
  return sizes;
}

// The symbols of some oracle messages at a list of positions: of[w][i] is for the w-th oracle
// message asked for and the i-th position x of the list; rows are read at x modulo their length.
// Each row gives one element of E, also a row over F.
struct Opened {
  struct Column {
    std::vector<E> main, mask;
  };
  std::vector<std::vector<Column>> of;
};

// The trees sent so far and each oracle message's place in them. Public; each side keeps its own.
class TreeBook {
public:
  struct Tree {
    OracleTreeShape shape;
    Digest root;
    // Verifier only: the root as its channel (`VerifierRounds`) received it, tied to the chain that
    // sent it. Openings are checked against `sent`, so a tree cannot be read under another chain's
    // values. The prover leaves it empty; receive_opening() ends the process on an empty one.
    SentRoot sent;
    std::vector<OracleId> parts;
    // True if a part has no out-of-domain sample (`answers` == 0). It masks one fold or a proximity
    // test's last step and is read at one list. A leaf is opened whole, so the whole tree is opened
    // once, in the chain that sent it or one that continues it: adopt() takes no such tree.
    bool one_list = false;
    // At how many lists the tree has been opened. Shared by each `TreeBook` reading it (adopt()).
    std::shared_ptr<unsigned> lists = std::make_shared<unsigned>(0);
    // Shared likewise: positions opened so far, over all lists and chains; repeats between lists
    // count. Prover only: open_oracles() checks it against hiding_positions() of each part's code.
    std::shared_ptr<u64> positions = std::make_shared<u64>(0);
    // (`VerifierSession` keeps the vertex values seen under each root; later openings must agree.)
  };
  struct Place {
    u32 tree;
    unsigned part;
  };
  // run: Rounds::run() of the chain that sends the roots; a new tree's leaf hashes start from it.
  // configuration: a hash of the public configuration files; inner vertices' hashes start from it.
  TreeBook(size_t salt_bytes, const Digest &run, const Digest &configuration)
      : salt_bytes_(salt_bytes), run_(run), configuration_(configuration) {}
  // Move-only, like its context: a copy would share the counts of what has been opened.
  TreeBook(const TreeBook &) = delete;
  TreeBook &operator=(const TreeBook &) = delete;
  TreeBook(TreeBook &&) = default;
  TreeBook &operator=(TreeBook &&) = default;
  const Tree &tree(u32 t) const { return trees_[t]; }
  void set_root(u32 t, const Digest &root) { trees_[t].root = root; }                                  // the prover's
  void set_root(u32 t, const SentRoot &sent) { trees_[t].root = sent.digest(), trees_[t].sent = sent; } // the verifier's
  // Continues on a later chain: existing trees keep their `run`, new ones take the later chain's.
  void goes_on(const Digest &run) { run_ = run; }
  size_t salt_bytes() const { return salt_bytes_; }
  // Makes tree t, just created by enter(), the tree `from` that another chain of this session has
  // sent: t takes `from`'s shape (with its number and `run`), root and shared counts.
  // A leaf is opened whole, so a tree is adopted with all its parts: a part unknown to this
  // `TreeBook` (a fold's mask, say) would be opened at lists that are not its own.
  void adopt(u32 t, const Tree &from) {
    BLK_REQUIRE(from.parts.size() == trees_[t].parts.size(), "trees: a tree is adopted with all the oracle messages that lie in it");
    BLK_REQUIRE(!from.one_list && !trees_[t].one_list, "trees: a tree that holds the mask of one step is read in no other chain");
    trees_[t].shape = from.shape, trees_[t].root = from.root, trees_[t].sent = from.sent, trees_[t].lists = from.lists, trees_[t].positions = from.positions;
  }
  // Both sides call this at every opening. The rule on sharing a leaf: a mask row's symbols are
  // only in leaves opened at exactly the lists that read it. All parts of a tree are opened
  // together (whole_trees()), so its sampled parts must be in the same groups (each group's test
  // reads at one list); a `one_list` tree is opened once, and its other parts belong to at most one
  // group. A violation is the program's fault (the layout is public), never a proof's.
  void count_opening(const Statement &st, u32 t) {
    Tree &tree = trees_[t];
    const std::vector<GroupId> *groups = nullptr;
    for (OracleId o : tree.parts) {
      if (st.oracle(o).shape.answers == 0) continue;
      const std::vector<GroupId> &mine = st.oracle(o).groups;
      BLK_REQUIRE(!groups || *groups == mine, "trees: oracle messages that share a leaf belong to different groups, so they are read at different lists");
      groups = &mine;
    }
    BLK_REQUIRE(!tree.one_list || !groups || groups->size() <= 1,
                "trees: the mask of one step shares a leaf with an oracle message that is read at several lists");
    BLK_REQUIRE(!tree.one_list || *tree.lists == 0, "trees: a tree that holds the mask of one step is opened at a second list");
    ++*tree.lists;
  }
  size_t trees() const { return trees_.size(); }
  Place place(OracleId o) const {
    BLK_REQUIRE(o < place_.size(), "trees: no such oracle message");
    return place_[o];
  }

  // Adds the oracle messages of one prover message to the statement and creates their trees, still
  // without roots (set_root()). Returns the number of the first new tree.
  u32 enter(Statement &st, const std::vector<OracleNotice> &sent, std::vector<OracleId> &ids) {
    const TreeLay lay = lay_trees(st.codes(), sent, salt_bytes_);
    const u32 first = (u32)trees_.size();
    for (const OracleTreeShape &s : lay.shapes) {
      trees_.emplace_back();
      trees_.back().shape = s;
      trees_.back().shape.tree_number = (u32)(trees_.size() - 1);
      trees_.back().shape.run = run_, trees_.back().shape.parameters = configuration_;
    }
    ids.clear();
    for (size_t i = 0; i < sent.size(); i++) {
      const OracleId o = st.sent(sent[i].shape);
      BLK_REQUIRE(o == place_.size(), "trees: every oracle message enters the statement through this call");
      place_.push_back({first + lay.tree[i], lay.part[i]});
      trees_[first + lay.tree[i]].parts.push_back(o);
      if (sent[i].shape.answers == 0) trees_[first + lay.tree[i]].one_list = true;
      ids.push_back(o);
    }
    return first;
  }

  // The trees that contain the oracle messages asked for, each once, in increasing order.
  std::vector<u32> trees_of(const std::vector<OracleId> &ids) const {
    std::vector<u32> t;
    for (OracleId o : ids) t.push_back(place(o).tree);
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    return t;
  }
  // The list for tree t when its group is read at `at`: `at`, or its short_list() if t is shorter.
  PositionList list_for(u32 t, const PositionList &at) const {
    const unsigned log_n = trees_[t].shape.log_n;
    if (log_n == at.log_n()) return at;
    return at.short_list(log_n); // it stays a list that this chain has drawn
  }
  // Extracts the rows of `ids` from their trees' opened columns. `trees`: trees_of(ids). opening[j]
  // belongs to trees[j] and was made or checked at list_for(trees[j], at).
  void sort_out(const Statement &st, const std::vector<OracleId> &ids, const PositionList &at, const std::vector<u32> &trees,
                const std::vector<Opening> &opening, Opened &out) const {
    out.of.assign(ids.size(), std::vector<Opened::Column>(at.size()));
    for (size_t w = 0; w < ids.size(); w++) {
      const Place p = place(ids[w]);
      const Tree &t = trees_[p.tree];
      const size_t j = (size_t)(std::lower_bound(trees.begin(), trees.end(), p.tree) - trees.begin());
      const PositionList own = list_for(p.tree, at);
      const OracleShape &s = st.oracle(ids[w]).shape;
      size_t main_skip = 0, mask_skip = 0;
      for (unsigned q = 0; q < p.part; q++) main_skip += t.shape.main_symbols[q], mask_skip += t.shape.mask_symbols[q];
      for (size_t i = 0; i < at.size(); i++) {
        const size_t here = PositionList::index_in(own.all(), short_position(at.log_n(), own.log_n(), at[i]));
        Opened::Column &col = out.of[w][i];
        if (s.main_rows) {
          const F *c = opening[j].main_column(here) + main_skip;
          for (size_t b = 0; b < s.main_rows; b++)
            col.main.push_back(s.main_deg == 1 ? E::from(c[b]) : E::from(c[3 * b], c[3 * b + 1], c[3 * b + 2]));
        }
        if (s.mask_rows) {
          const F *c = opening[j].mask_column(here) + mask_skip;
          for (size_t b = 0; b < s.mask_rows; b++) col.mask.push_back(E::from(c[3 * b], c[3 * b + 1], c[3 * b + 2]));
        }
      }
    }
  }

  // Whether `ids` has all parts of every tree that it touches. An opening shows whole leaves, so a
  // part that is left out would still be read, at a list that is not its own.
  bool whole_trees(const std::vector<OracleId> &ids) const {
    for (u32 t : trees_of(ids))
      for (OracleId o : trees_[t].parts)
        if (std::find(ids.begin(), ids.end(), o) == ids.end()) return false;
    return true;
  }

private:
  size_t salt_bytes_;
  Digest run_, configuration_;
  std::vector<Tree> trees_;
  std::vector<Place> place_;
};

// ---------------------------------------------------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------------------------------------------------

// The prover's private data for one oracle message. The main rows' random tails stay in the engine.
struct ProverOracle {
  TableId main = NO_ID;
  Table mask, mask_tail;
  u32 mask_draw = Coins::NO_DRAW, tail_draw = Coins::NO_DRAW; // for `not_as_drawn` (send.h)
  u64 opened = 0;    // the number of positions at which it has been opened
  unsigned lists = 0; // the number of lists
  bool held = false;  // a test on another chain of the session will still read it
};

// For tests: one pad entry, read back from the prover's tables. A pad masks a sample's answer.
struct PadLook {
  OracleId oracle;
  bool in_mask_rows;
  u64 place; // the entry's index in the table
  E value;
};

struct ProverContext {
  Statement st;
  ProverRounds &ch;
  Engine &engine;
  // For tests: no verifier and no test of proof bytes can detect a pad that was never drawn.
  // If set, send_prove() (send.h) appends every pad entry; pads_fault() compares two runs.
  // An honest run is no substitute: send_prove() ends the process if a pad differs from what it
  // recorded as drawn (own-test-pad-kept) or is missing, but not if one drawn value is used again.
  std::vector<PadLook> *pads = nullptr;
  Coins &coins;
  TreeBook book;
  std::vector<ProverOracle> behind; // by OracleId
  std::map<const E *, std::vector<u32>> fold_draws; // the draws' numbers of fold masks not yet folded: fold_mask_make(S, P), fold.h
  std::vector<CommitId> commit;     // by tree
  struct OpenedBytes {
    u32 tree;
    size_t positions, bytes;
  };
  std::vector<OpenedBytes> opened_bytes; // sizes of the openings written so far; nothing reads it

  ProverContext(const Public<Codes> &codes, ProverRounds &c, Engine &e, Coins &k)
      : st(Statement(c.get(codes))), ch(c), engine(e), coins(k), book(c.plan()->set.salt_bytes, c.run(), c.configuration()) {}
  // Continues on a later chain: statement, trees and tables move; `pads` is not carried over.
  ProverContext(ProverContext &&earlier, ProverRounds &later)
      : st(std::move(earlier.st)), ch(later), engine(earlier.engine), coins(earlier.coins), book(std::move(earlier.book)),
        behind(std::move(earlier.behind)), fold_draws(std::move(earlier.fold_draws)), commit(std::move(earlier.commit)), opened_bytes(std::move(earlier.opened_bytes)) {
    BLK_REQUIRE(later.stands_on(earlier.ch), "context: the later chain's first query does not hold the earlier chain's seal");
    BLK_REQUIRE(later.plan()->set.salt_bytes == book.salt_bytes(), "context: the two chains' plans differ in the length of a salt");
    book.goes_on(later.run());
  }

  // A mask entry's value. Reading does not use it up; the claim that takes the `MaskSlot` does.
  E mask_value(const MaskSlot &s) const { return behind[s.oracle()].mask.entry(s.index()); }
};

// Sends all oracle messages of the present prover message: encodes, builds trees, sends the roots.
inline std::vector<OracleId> send_oracles(ProverContext &P, std::vector<OracleDraft> drafts) {
  std::vector<OracleNotice> notices;
  for (const OracleDraft &d : drafts) notices.push_back({d.shape, d.leaf_class});
  std::vector<OracleId> ids;
  const u32 first = P.book.enter(P.st, notices, ids);
  std::vector<std::vector<OraclePart>> parts(P.book.trees() - first);
  for (size_t i = 0; i < drafts.size(); i++) {
    OracleDraft &d = drafts[i];
    const OracleShape &s = d.shape;
    BLK_REQUIRE((d.main != NO_ID) == (s.main_rows > 0) && d.mask.rows == s.mask_rows, "send: the tables do not fit the shape");
    OraclePart part;
    ProverOracle mine;
    if (s.main_rows) part.main = mine.main = d.main, part.code = P.st.codes().base[s.code];
    if (s.mask_rows) {
      part.mask_code = P.st.codes().mask[s.mask_code];
      mine.mask = std::move(d.mask), mine.mask_draw = d.mask_draw;
      mine.mask_tail = Table(E_DEGREE, s.mask_rows, part.mask_code.r);
      P.coins.fill(mine.mask_tail.a.data(), mine.mask_tail.a.size());
      mine.tail_draw = P.coins.last_draw();
      part.mask = mine.mask, part.mask_tail = mine.mask_tail;
    }
    parts[P.book.place(ids[i]).tree - first].push_back(std::move(part));
    P.behind.push_back(std::move(mine));
  }
  for (u32 t = first; t < P.book.trees(); t++) {
    Digest root;
    P.commit.push_back(P.engine.commit(parts[t - first], P.book.tree(t).shape, P.coins, root));
    P.book.set_root(t, root);
    P.ch.send_root(root);
  }
  return ids;
}

// A group's main rows as Engine::fold_begin() takes them. Kept rows (`GroupLayout`) are not folded.
inline std::vector<FoldPart> fold_parts(const ProverContext &P, const GroupLayout &L) {
  std::vector<FoldPart> parts;
  for (size_t i = 0; i < L.members.size(); i++) {
    if (L.main_first[i] == GroupLayout::NONE) continue;
    const TreeBook::Place p = P.book.place(L.members[i]);
    parts.push_back(FoldPart{P.commit[p.tree], p.part, L.main_first[i]});
  }
  return parts;
}

// Defined in blocks/iop/send.h, which this header must not include: a program without send.h may
// define its own `SendAccess` (Statement::sampled(), blocks/iop/relation.h). A program that calls
// open_oracles() includes send.h too, or it does not link.
template <class ProverCtx> u64 not_as_drawn(const ProverCtx &pr, OracleId o);

// Opens the oracle messages at the list: writes columns, salts and sibling digests into the proof
// (they enter no query of the hash chain) and returns the symbols. `ids`: whole trees only.
// Oracle messages are opened only here. The two checks on `TreeBook::Tree::positions` are for
// hiding: a row's symbols at more positions than its random tail has entries reveal linear
// functions of the row's entries. The count is raised first, then compared with hiding_positions()
// of each part's code, for main rows and mask rows separately; if it is larger, the process ends
// before any byte of that tree's opening is written. count_opening() counts the lists.
//
// [own-test-mask-rows] A third check for hiding, also before any byte of that tree's opening: the
// random tail of each message's mask rows, and the rows themselves where their draw's number was
// recorded, must equal the private random values that were drawn for them (not_as_drawn(),
// blocks/iop/send.h, which says what is compared). The tail covers the mask rows' symbols at the
// opened positions; a tail that is lost right after its draw gives a proof, or a key file, that
// verifies and hides less. Every tail is drawn in send_oracles() and a message can be opened only
// here, so no block that sends a message can leave the tail's comparison out. The check sees the
// tables that the context keeps, not the copies that the engine has encoded, from which the opened
// symbols come. With a generator that does not number its draws nothing is compared, and nothing
// reports that. Not compared: a message that another context sent
// (lin_second_prove_on_shape(), blocks/iop/lin.h).
inline void open_oracles(ProverContext &P, const std::vector<OracleId> &ids, const PositionList &at, Opened &out) {
  BLK_REQUIRE(P.book.whole_trees(ids), "open: an oracle message that shares a tree with those asked for is left out");
  const std::vector<u32> trees = P.book.trees_of(ids);
  std::vector<Opening> opening(trees.size());
  for (size_t j = 0; j < trees.size(); j++) {
    P.book.count_opening(P.st, trees[j]);
    const TreeBook::Tree &t = P.book.tree(trees[j]);
    const PositionList own = P.book.list_for(trees[j], at);
    *t.positions += own.size();
    for (OracleId o : t.parts) {
      const OracleShape &s = P.st.oracle(o).shape;
      P.behind[o].opened += own.size(); // in this context only; tests read it
      P.behind[o].lists++;
      BLK_REQUIRE(!s.main_rows || *t.positions <= hiding_positions(P.st.codes().base[s.code]),
                  "open: main rows are opened at more positions than their random tail hides");
      BLK_REQUIRE(!s.mask_rows || *t.positions <= hiding_positions(P.st.codes().mask[s.mask_code]),
                  "open: mask rows are opened at more positions than their random tail hides");
      BLK_REQUIRE_NAMED(!s.mask_rows || not_as_drawn(P, o) == 0, "own-test-mask-rows",
                        "internal error (own-test-mask-rows): values of the mask rows of a message that is about to be opened, or of their random tail, are not the random values that were drawn for them. The program or the machine has a fault; nothing of this run may be used");
    }
    P.engine.open(P.commit[trees[j]], own, opening[j]);
    const size_t before = P.ch.file_only().size();
    P.ch.file_only().mark("opening of tree " + std::to_string(trees[j]) + " at " + std::to_string(own.size()) + " positions");
    write_opening(P.ch.file_only(), t.shape, own, opening[j]);
    P.ch.file_only().mark("end of the opening");
    P.opened_bytes.push_back({trees[j], own.size(), P.ch.file_only().size() - before});
  }
  P.book.sort_out(P.st, ids, at, trees, opening, out);
}

// Call after a proximity test has read these oracle messages. Frees what no further test will read
// (trees, codewords, randomness, tables), so the prover's memory grows with one group, not the
// whole run. An oracle message is read once by the test of each of its groups; a mask of a fold or
// of a last step, in no group, once. A tree stays while any part is still to be read or is `held`.
inline void release_oracles(ProverContext &P, const std::vector<OracleId> &ids) {
  for (u32 t : P.book.trees_of(ids)) {
    if (P.commit[t] == NO_ID) continue;
    bool done = true;
    for (OracleId o : P.book.tree(t).parts)
      done = done && !P.behind[o].held && P.behind[o].lists >= std::max<size_t>(1, P.st.oracle(o).groups.size());
    if (!done) continue;
    P.engine.close(P.commit[t]);
    P.commit[t] = NO_ID;
    for (OracleId o : P.book.tree(t).parts) {
      if (P.behind[o].main != NO_ID) P.engine.drop(P.behind[o].main);
      P.behind[o].main = NO_ID, P.behind[o].mask = Table(), P.behind[o].mask_tail = Table();
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------------------------------------------------

struct VerifierContext {
  Statement st;
  VerifierRounds &ch;
  TreeBook book;
  // Intermediate values, such as a claim's value (`target`) after each block, for comparing a
  // second implementation of the verifier block by block: acceptance alone shows only that all
  // blocks together agree. Recorded only if `notes` is set; no decision reads them.
  std::vector<std::pair<std::string, E>> *notes = nullptr;
  void note(const char *what, const E &value) {
    if (notes) notes->emplace_back(what, value);
  }

  VerifierContext(const Public<Codes> &codes, VerifierRounds &c)
      : st(Statement(c.get(codes))), ch(c), book(c.plan()->set.salt_bytes, c.run(), c.configuration()) {
    st.has_no_engine();
  }
  VerifierContext(VerifierContext &&earlier, VerifierRounds &later) : st(std::move(earlier.st)), ch(later), book(std::move(earlier.book)) {
    BLK_REQUIRE(later.stands_on(earlier.ch), "context: the later chain's first query does not hold the earlier chain's seal");
    BLK_REQUIRE(later.plan()->set.salt_bytes == book.salt_bytes(), "context: the two chains' plans differ in the length of a salt");
    book.goes_on(later.run());
  }
};

// Reads the roots of all oracle messages of the present prover message.
inline bool receive_oracles(VerifierContext &V, const std::vector<OracleNotice> &sent, std::vector<OracleId> &ids) {
  const u32 first = V.book.enter(V.st, sent, ids);
  for (u32 t = first; t < V.book.trees(); t++) {
    SentRoot root;
    if (!V.ch.recv_root(root)) return false;
    V.book.set_root(t, root);
  }
  return true;
}

// Reads the trees' openings at the list, checks them against the roots, returns the symbols.
inline bool read_opened(VerifierContext &V, const std::vector<OracleId> &ids, const PositionList &at, Opened &out) {
  if (!V.ch.ok()) return false; // after a rejection the channel's lists are placeholders
  BLK_REQUIRE(V.book.whole_trees(ids), "open: an oracle message that shares a tree with those asked for is left out");
  const std::vector<u32> trees = V.book.trees_of(ids);
  std::vector<Opening> opening(trees.size());
  for (size_t j = 0; j < trees.size(); j++) {
    V.book.count_opening(V.st, trees[j]);
    const TreeBook::Tree &t = V.book.tree(trees[j]);
    const PositionList own = V.book.list_for(trees[j], at);
    if (!receive_opening(V.ch, t.sent, t.shape, own, opening[j])) return false;
  }
  V.book.sort_out(V.st, ids, at, trees, opening, out);
  return true;
}

} // namespace blk
