// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The first lines of the parameter file (`policy.txt`), from `version` to `table`, and the one
   strict parser with which prover and verifier read them.

   The order of the lines is fixed. A line is a name and its values, separated by single spaces and
   ended by a line feed. A value is a decimal number without sign or leading zero; the version is
   four characters. A blank line, a comment, a carriage return, or anything before the first line
   is rejected. Every byte of the file enters the configuration digest (`configuration_digest` in
   heads.h), which every hash chain's first query contains: two files that differ in a byte are two
   parameter sets.

   Only free choices have a line. The rest is derived, starting in derive.h: each group's slack,
   every number of positions, the length of every random tail (a row's random entries) but the
   committed table's, the later stages' message lengths, the number of combinations in the
   approximate range check (`smallness` in line names). So no file can ask for a list of positions
   that is too short for its own target, proof of work counted (the line `work positions` shortens
   the lists: see `ProofVerdict::YES_WITH_WORK` in configure.h); `configure` there rejects a file
   under which a round misses the target. */
#pragma once
#include <string>
#include <vector>

#include "frame/version.h"
#include "numbers/blocks.h"

namespace blk {

/* The lines `group <name> J n l nz lz` and `stages <name> k h n nz ..` of one group: the oracle
   messages (encoded rows, sent as Merkle roots) that one proximity test reads.
     J       always 1. The test's distance, at all stages, is the Johnson radius less a derived
             slack (`Radius` in blocks/numbers/codes.h, which states on what it rests)
     n, l    code length 2^n and message length 2^l of the main rows
     nz, lz  the same of the mask rows
   Each of the one to eight stages after the first has four numbers: 2^k main rows, 2^h mask rows,
   code lengths 2^n and 2^nz. Only the last stage has k = h = 0. */
struct GroupLine {
  unsigned log_n = 0, log_ell = 0, mask_log_n = 0, mask_log_ell = 0;
  std::vector<StageChoice> stages;
};

/* The lines, in their order.
     version W          the version word (blocks/frame/version.h)
     sec B              the target: every round's error times 2^-work is to be at most 2^-B
     reading R          always 1: the integer m of the proximity bound up to the Johnson radius is
                        read as its theorem prints it (`CountSource` in blocks/numbers/setting.h,
                        with the citation)
     rowcost C          always 1: combining the rows of an interleaved word puts no factor on that
                        bound (the same place says on what that rests)
     slots N            the largest number of requests in a batch, a power of two
     answers S          always 1: answers to each out-of-domain sample
     record M           always 1: the committed table has a record (stored answers to samples of
                        its rows)
     work <kind> A B C  five lines: field, sample, closeness (proximity), positions, smallness.
                        Bits of proof of work before a verifier's message whose round has an error
                        of this kind: A in a request's own rounds, B in a batch's shared rounds, C
                        in the committer's.
     salt Y             bytes of a tree leaf's salt: a multiple of 8, from 40 to 64
     listed P           always 0: a proof lists no drawn position
   Then `group` and `stages` (GroupLine) for eight groups:
     link         a segment of a request's proof (one sub-proof's part of the request's first hash
                  chain); the stages are those of the test on the message into which the last
                  segment is closed
     folded       in a batch's shared chain (the hash chain that all its proofs repeat): the folded
                  row of the committed table, re-encoded
     omega        in the same chain: Omega, the committed matrix times the equality vector of the
                  chain's shared point (blocks/iop/lin.h); same mask code as `folded`, because the
                  mask message that splits a claim on both groups belongs to both
     own          in a request's second chain: Omega together with the request's own vector W; same
                  main code as `omega`
     once-folded, once-omega, once-own   the same three in the committer's one-time test of the
                  table (key file); a leaf file's LINK proof, which ties its table to the key
                  file's, uses `once-folded`
     once-ranges  in the key file: the multiplicities of the lookups by which the committer proves
                  that the table's entries are in their ranges
   Then:
     ranges-part L      a part of such a lookup reads 2^L rows of the matrix (smaller parts: less
                        memory for the committer, a longer key file). blocks/frame/wf_frame.h
                        rejects an L that gives a lookup more than WF_MOST_PARTS parts or exceeds
                        the number of the matrix's row bits.
     table b n i m e w  the committed table: 2^b rows; code length 2^n, at most 2^32; i message
                        entries and m random ones in a row; e, always 0: the distances are
                        derived; w
                        coordinates of the shared point (`block_log`), of which b fold the rows
   `rest`: the lines after these, on the model and its sub-proofs, unread. */
struct Parameters {
  u32 version = 0;
  unsigned sec = 0;
  u64 slots = 0;
  unsigned answers = 0;
  unsigned work[5][3] = {};  // by ErrorKind (`FIELD` to `SMALL`, in the lines' order) and Scope
  unsigned salt = 0;
  GroupLine link, folded, omega, own, once_folded, once_omega, once_own, once_ranges;
  unsigned ranges_part = 0;
  unsigned table_rows_log = 0, table_log_n = 0, block_log = 0;
  u64 table_data = 0, table_random = 0;
  std::string rest;
};

// Reads a text, which must outlive the reader, line by line. Only the first failure is kept: why()
// gives its name and line number.
class StrictLines {
 public:
  explicit StrictLines(const std::string &text) : t_(text) {}
  bool ok() const { return why_.empty(); }
  const std::string &why() const { return why_; }
  void refuse(const std::string &why) {
    if (why_.empty()) why_ = why + " (line " + std::to_string(line_) + ")";
  }
  /* The next line must begin with `head` and go on with exactly `count` values (`count` < 0: one
     or more). A value of more than 15 digits is rejected: so every value is below 2^50, and no sum
     or small multiple of values wraps. After a failure the result is `count` zeros (one if
     `count` < 0), so a caller may index it. */
  std::vector<u64> numbers(const std::string &head, int count) {
    std::vector<u64> v;
    for (const std::string &w : words(head, count)) {
      u64 x = 0;
      if (w.empty() || w.size() > 15 || (w[0] == '0' && w.size() > 1)) refuse("parameter-number-spelling");
      for (char c : w) {
        if (c < '0' || c > '9') refuse("parameter-number-spelling");
        x = 10 * x + (u64)(c - '0');
      }
      v.push_back(x);
    }
    if (!ok()) v.assign(count < 0 ? 1 : (size_t)count, 0);
    return v;
  }
  // As numbers(), with the values as text; after a failure the result may have any length, even 0.
  std::vector<std::string> words(const std::string &head, int count) {
    ++line_;
    std::vector<std::string> v;
    const size_t end = t_.find('\n', at_);
    if (!ok() || end == std::string::npos) return refuse("parameter-line-missing: " + head), v;
    const std::string l = t_.substr(at_, end - at_);
    at_ = end + 1;
    if (l.compare(0, head.size() + 1, head + " ") != 0) return refuse("parameter-line-order: " + head), v;
    for (size_t i = head.size() + 1;;) {
      const size_t j = l.find(' ', i);
      v.push_back(l.substr(i, j == std::string::npos ? j : j - i));
      if (v.back().empty()) return refuse("parameter-line-blanks: " + head), v;
      if (j == std::string::npos) break;
      i = j + 1;
    }
    if (count >= 0 && v.size() != (size_t)count) refuse("parameter-line-values: " + head);
    return v;
  }
  std::string rest() const { return t_.substr(at_); }

 private:
  const std::string &t_;
  size_t at_ = 0;
  unsigned line_ = 0;
  std::string why_;
};

/* The largest n of a `group` or `stages` line. The proximity bound for a mask code repeated along
   the longer main domain needs that domain to be one coset of a subgroup, so its length must divide
   p - 1, where p = 2^64 - 2^32 + 1 is the field's modulus. Beyond 2^32 a domain is a union of
   cosets. */
inline constexpr u64 MOST_GROUP_LOG_N = 32;

inline void read_group(StrictLines &in, const std::string &name, GroupLine &g) {
  const auto a = in.numbers("group " + name, 5);
  g.log_n = (unsigned)a[1], g.log_ell = (unsigned)a[2], g.mask_log_n = (unsigned)a[3], g.mask_log_ell = (unsigned)a[4];
  if (a[0] != 1 || a[1] > MOST_GROUP_LOG_N || a[2] >= a[1] || a[3] > a[1] || a[4] >= a[3]) in.refuse("parameter-group-shape: " + name);
  const auto s = in.numbers("stages " + name, -1);
  if (s.size() % 4 || s.size() > 32) return in.refuse("parameter-stages-count: " + name);
  for (size_t i = 0; i < s.size(); i += 4) {
    const bool last = i + 4 == s.size();
    if (s[i] > 8 || s[i + 1] > s[i] || s[i + 2] > MOST_GROUP_LOG_N || s[i + 3] > s[i + 2] || (last ? s[i] != 0 : s[i] == 0))
      return in.refuse("parameter-stages-shape: " + name);
    g.stages.push_back(StageChoice{(unsigned)s[i], (unsigned)s[i + 1], (unsigned)s[i + 2], (unsigned)s[i + 3]});
  }
}

// Returns false, with the reason in `why`, if the text is rejected. Only syntax, ranges and shapes
// are checked here, not whether the parameters reach their target (derive.h, configure.h).
inline bool read_parameters(const std::string &text, Parameters &p, std::string &why) {
  p = Parameters();
  StrictLines in(text);
  const auto v = in.words("version", 1);
  p.version = v.empty() ? 0 : version_from_text(v[0]);
  if (in.ok() && version_refusal(p.version)) in.refuse(version_refusal(p.version));
  auto one = [&](const char *name, u64 lo, u64 hi) {
    const u64 x = in.numbers(name, 1)[0];
    if (x < lo || x > hi) in.refuse(std::string("parameter-range: ") + name);
    return x;
  };
  p.sec = (unsigned)one("sec", 1, 256);
  one("reading", 1, 1);
  one("rowcost", 1, 1);
  p.slots = one("slots", 1, (u64)1 << 20);
  if (p.slots & (p.slots - 1)) in.refuse("parameter-range: slots");  // the tree over a batch's seals has a power of two of places, and errors and random entries are sized for 'slots' requests
  p.answers = (unsigned)one("answers", 1, 1);
  one("record", 1, 1);
  static const char *const kind[5] = {"work field", "work sample", "work closeness", "work positions", "work smallness"};
  for (int k = 0; k < 5; ++k) {
    const auto w = in.numbers(kind[k], 3);
    for (int s = 0; s < 3; ++s) {
      p.work[k][s] = (unsigned)w[(size_t)s];
      if (w[(size_t)s] > 58 || w[(size_t)s] >= p.sec) in.refuse(std::string("parameter-range: ") + kind[k]);  // a nonce has 8 bytes
    }
  }
  p.salt = (unsigned)one("salt", 40, 64);  // the floor: what the leaves' hiding term needs as the protocol's text has it
  if (p.salt % 8) in.refuse("parameter-range: salt");
  one("listed", 0, 0);
  read_group(in, "link", p.link);
  read_group(in, "folded", p.folded);
  read_group(in, "omega", p.omega);
  read_group(in, "own", p.own);
  read_group(in, "once-folded", p.once_folded);
  read_group(in, "once-omega", p.once_omega);
  read_group(in, "once-own", p.once_own);
  read_group(in, "once-ranges", p.once_ranges);
  p.ranges_part = (unsigned)one("ranges-part", 1, 40);
  if (in.ok() && (p.own.log_n != p.omega.log_n || p.own.log_ell != p.omega.log_ell || p.once_own.log_n != p.once_omega.log_n ||
                  p.once_own.log_ell != p.once_omega.log_ell))
    in.refuse("parameter-own-code");
  if (in.ok() && (p.folded.mask_log_n != p.omega.mask_log_n || p.folded.mask_log_ell != p.omega.mask_log_ell ||
                  p.once_folded.mask_log_n != p.once_omega.mask_log_n || p.once_folded.mask_log_ell != p.once_omega.mask_log_ell))
    in.refuse("parameter-one-mask-code");
  const auto t = in.numbers("table", 6);
  p.table_rows_log = (unsigned)t[0], p.table_log_n = (unsigned)t[1], p.table_data = t[2], p.table_random = t[3];
  p.block_log = (unsigned)t[5];
  // n <= 32: the points of the table's code are one coset of a subgroup (blocks/code/cosets.h). A
  // longer code would lie on several cosets; key files, leaf files and proofs for one are untested.
  if (in.ok() && (t[0] > 20 || t[1] > 32 || t[2] == 0 || t[2] + t[3] >= (u64)1 << t[1] || t[4] != 0 || t[5] > 40 || t[5] < t[0]))
    in.refuse("parameter-table-shape");
  p.rest = in.rest();
  why = in.why();
  return in.ok();
}

}  // namespace blk
