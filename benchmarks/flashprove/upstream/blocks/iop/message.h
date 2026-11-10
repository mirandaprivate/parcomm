// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Byte encoding of proofs and hashed public data. A Writer appends typed items; a Reader reads them
// in the same order. An item carries no length or type: the reading code knows what follows (in a
// proof, from public data such as the plan of rounds, blocks/numbers/plan.h). Integers are
// little-endian. A read fails if it passes the end or meets an element not in canonical form
// (`F`, or a coefficient of `E`, not below the modulus `P`); later reads fail too. Only at_end()
// (no failure, no byte left) detects leftover bytes.
#pragma once
#include <cstring>
#include <string>
#include <vector>

#include "base.h"
#include "field/field.h"
#include "hash/hash.h"

namespace blk {

class Writer {
public:
  void put_bytes(const u8 *p, size_t n) { append('B', p, n); }
  void put_u64(u64 x) {
    u8 b[8];
    for (int i = 0; i < 8; i++) b[i] = (u8)(x >> (8 * i));
    append('W', b, 8);
  }
  void put(F x) {
    u8 b[F_BYTES];
    F_to_bytes(x, b);
    append('F', b, F_BYTES);
  }
  void put(const E &x) {
    u8 b[E_BYTES];
    E_to_bytes(x, b);
    append('E', b, E_BYTES);
  }
  void put(const Digest &d) {
    u8 b[DIGEST_BYTES];
    digest_to_bytes(d, b);
    append('D', b, DIGEST_BYTES);
  }
  size_t size() const { return buf_.size(); }
  const u8 *data() const { return buf_.data(); }
  const std::vector<u8> &bytes() const { return buf_; }
  void clear() { buf_.clear(), items_.clear(), marks_.clear(); }

  // List for programs that print or test a proof's fields: each item's offset, size and kind, and
  // each text set by mark(). It starts when keep_list() is called and does not change the bytes.
  struct Item {
    size_t at, bytes;
    char kind; // F, E: an element; D: a digest; W: a word of 64 bits; B: plain bytes
  };
  struct Mark {
    size_t at;
    std::string words;
  };
  void keep_list() { listed_ = true; }
  // Appends the bytes of `other`. If this Writer keeps its list, also appends the items and marks
  // of `other`, shifted; a non-empty `other` that keeps no list is listed as one item of kind B.
  void append(const Writer &other) {
    const size_t moved = buf_.size();
    if (listed_) {
      for (const Item &i : other.items_) items_.push_back(Item{i.at + moved, i.bytes, i.kind});
      for (const Mark &m : other.marks_) marks_.push_back(Mark{m.at + moved, m.words});
      if (!other.listed_ && other.size()) items_.push_back(Item{moved, other.size(), 'B'});
    }
    buf_.insert(buf_.end(), other.buf_.begin(), other.buf_.end());
  }
  void mark(const std::string &words) {
    if (listed_) marks_.push_back(Mark{buf_.size(), words});
  }
  const std::vector<Item> &items() const { return items_; }
  const std::vector<Mark> &marks() const { return marks_; }

private:
  void append(char kind, const u8 *p, size_t n) {
    if (listed_) items_.push_back(Item{buf_.size(), n, kind});
    buf_.insert(buf_.end(), p, p + n);
  }
  std::vector<u8> buf_;
  bool listed_ = false;
  std::vector<Item> items_;
  std::vector<Mark> marks_;
};

class Reader {
public:
  Reader(const u8 *p, size_t n) : p_(p), end_(p + n) {}
  bool ok() const { return ok_; }
  bool at_end() const { return ok_ && p_ == end_; }
  void refuse() { ok_ = false; }
  // Returns the next n bytes within the constructor's buffer (no copy), or nullptr on failure.
  const u8 *take(size_t n) {
    if (!ok_ || (size_t)(end_ - p_) < n) {
      ok_ = false;
      return nullptr;
    }
    const u8 *q = p_;
    p_ += n;
    return q;
  }
  bool get_u64(u64 &x) {
    const u8 *q = take(8);
    if (!q) return false;
    x = 0;
    for (int i = 0; i < 8; i++) x |= (u64)q[i] << (8 * i);
    return true;
  }
  bool get(F &x) {
    const u8 *q = take(F_BYTES);
    if (!q) return false;
    return ok_ = F_from_bytes(q, x);
  }
  bool get(E &x) {
    const u8 *q = take(E_BYTES);
    if (!q) return false;
    return ok_ = E_from_bytes(q, x);
  }
  bool get(Digest &d) {
    const u8 *q = take(DIGEST_BYTES);
    if (!q) return false;
    digest_from_bytes(q, d);
    return true;
  }

private:
  const u8 *p_, *end_;
  bool ok_ = true;
};

} // namespace blk
