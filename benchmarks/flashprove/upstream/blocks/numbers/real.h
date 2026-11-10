// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Arithmetic for deriving parameters and comparing a protocol round's error with a target
   such as 2^-128.

   Why not double: the number of codeword positions that a proximity test opens is the least
   integer, and the divisor in a code's slack (margin below the Johnson radius) the largest, at
   which an error stays within the target. One rounding error can move either by one, and prover and
   verifier must derive the same numbers. So:

     Big    a non-negative integer of any size, exact.
     Real   m * 2^e with m a Big. Every operation that cannot be exact takes a direction, DOWN or
            UP, and its result is on that side of the true value, cut to about REAL_BITS bits.
     Bound  a pair of Reals, lo <= true value <= hi: for quotients, square roots and errors.

   A decision must be an exact comparison of Reals at the safe end of a Bound: an error is compared
   at its hi. If the true value is too near the threshold for REAL_BITS bits to tell, the answer is
   the safe one: one more position, more slack, target not met. Only integer operations are used, so
   all machines get the same results; functions returning double are for printed figures only. */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "base.h"

namespace blk {

class Big {
 public:
  std::vector<u32> w;  // little endian; the top limb is not 0
  Big() {}
  Big(u64 x) {
    for (; x; x >>= 32) w.push_back((u32)x);
  }
  bool zero() const { return w.empty(); }
  size_t bits() const {
    if (w.empty()) return 0;
    size_t b = 32 * (w.size() - 1);
    for (u32 t = w.back(); t; t >>= 1) ++b;
    return b;
  }
  bool bit(size_t i) const { return i / 32 < w.size() && ((w[i / 32] >> (i % 32)) & 1); }
  // Is any of the lowest n bits set?
  bool any_below(size_t n) const {
    for (size_t i = 0; i < w.size() && 32 * i < n; ++i) {
      u32 x = w[i];
      if (n - 32 * i < 32) x &= ((u32)1 << (n - 32 * i)) - 1;
      if (x) return true;
    }
    return false;
  }
  bool fits_u64() const { return w.size() <= 2; }
  u64 to_u64() const {
    BLK_REQUIRE(fits_u64(), "Big::to_u64: the number does not fit");
    return (w.size() > 0 ? (u64)w[0] : 0) | (w.size() > 1 ? (u64)w[1] << 32 : 0);
  }
  static int cmp(const Big &a, const Big &b) {
    if (a.w.size() != b.w.size()) return a.w.size() < b.w.size() ? -1 : 1;
    for (size_t i = a.w.size(); i--;)
      if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
    return 0;
  }
  static Big two_to(size_t n) { return Big(1).shl(n); }
  Big shl(size_t n) const {
    if (w.empty() || !n) return *this;
    Big r;
    const size_t q = n / 32, s = n % 32;
    r.w.assign(w.size() + q + 1, 0);
    for (size_t i = 0; i < w.size(); ++i) {
      const u64 x = (u64)w[i] << s;
      r.w[i + q] |= (u32)x;
      r.w[i + q + 1] |= (u32)(x >> 32);
    }
    r.trim();
    return r;
  }
  Big shr(size_t n) const {
    const size_t q = n / 32, s = n % 32;
    Big r;
    if (q >= w.size()) return r;
    r.w.assign(w.size() - q, 0);
    for (size_t i = q; i < w.size(); ++i) {
      u64 x = w[i];
      if (i + 1 < w.size()) x |= (u64)w[i + 1] << 32;
      r.w[i - q] = (u32)(x >> s);
    }
    r.trim();
    return r;
  }
  friend Big operator+(const Big &a, const Big &b) {
    Big r;
    r.w.assign(std::max(a.w.size(), b.w.size()) + 1, 0);
    u64 c = 0;
    for (size_t i = 0; i < r.w.size(); ++i) {
      c += (i < a.w.size() ? a.w[i] : 0);
      c += (i < b.w.size() ? b.w[i] : 0);
      r.w[i] = (u32)c;
      c >>= 32;
    }
    r.trim();
    return r;
  }
  friend Big operator-(const Big &a, const Big &b) {
    BLK_REQUIRE(cmp(a, b) >= 0, "Big: a difference would be negative");
    Big r;
    r.w.assign(a.w.size(), 0);
    i64 c = 0;
    for (size_t i = 0; i < a.w.size(); ++i) {
      c += (i64)a.w[i] - (i < b.w.size() ? (i64)b.w[i] : 0);
      r.w[i] = (u32)c;
      c >>= 32;  // arithmetic shift: 0 or -1
    }
    r.trim();
    return r;
  }
  friend Big operator*(const Big &a, const Big &b) {
    Big r;
    if (a.zero() || b.zero()) return r;
    r.w.assign(a.w.size() + b.w.size(), 0);
    for (size_t i = 0; i < a.w.size(); ++i) {
      u64 c = 0;
      for (size_t j = 0; j < b.w.size(); ++j) {
        c += (u64)a.w[i] * b.w[j] + r.w[i + j];
        r.w[i + j] = (u32)c;
        c >>= 32;
      }
      r.w[i + b.w.size()] = (u32)c;
    }
    r.trim();
    return r;
  }
  friend bool operator==(const Big &a, const Big &b) { return cmp(a, b) == 0; }
  friend bool operator!=(const Big &a, const Big &b) { return cmp(a, b) != 0; }
  friend bool operator<(const Big &a, const Big &b) { return cmp(a, b) < 0; }
  friend bool operator<=(const Big &a, const Big &b) { return cmp(a, b) <= 0; }
  /* q = floor(a / b), r = a - q b; q must not be the same object as a or r. Schoolbook division on
     32-bit limbs (Knuth, The Art of Computer Programming, volume 2, 4.3.1, Algorithm D).
     A public rendering of that algorithm in C is divmnu64.c of H. S. Warren's "Hacker's Delight".
     PUBLISHED-WORKS.md, "Knuth's Algorithm D", cites
     both in full. */
  static void divmod(const Big &a, const Big &b, Big &q, Big &r) {
    BLK_REQUIRE(!b.zero(), "Big: division by zero");
    if (cmp(a, b) < 0) {
      q = Big();
      r = a;
      return;
    }
    if (b.w.size() == 1) {
      Big t;
      t.w.assign(a.w.size(), 0);
      u64 c = 0;
      for (size_t i = a.w.size(); i--;) {
        c = (c << 32) | a.w[i];
        t.w[i] = (u32)(c / b.w[0]);
        c %= b.w[0];
      }
      t.trim();
      q = t;
      r = Big(c);
      return;
    }
    size_t s = 0;
    for (u32 t = b.w.back(); !(t & 0x80000000u); t <<= 1) ++s;
    const Big v = b.shl(s);
    Big u = a.shl(s);
    const size_t n = v.w.size();
    u.w.resize(std::max(u.w.size(), a.w.size() + 1), 0);
    if (u.w.size() == a.w.size()) u.w.push_back(0);
    const size_t m = u.w.size() - n - 1;
    Big t;
    t.w.assign(m + 1, 0);
    for (size_t j = m + 1; j--;) {
      const u64 num = ((u64)u.w[j + n] << 32) | u.w[j + n - 1];
      u64 qh = num / v.w[n - 1], rh = num % v.w[n - 1];
      while (qh >> 32 || qh * v.w[n - 2] > ((rh << 32) | u.w[j + n - 2])) {
        --qh;
        rh += v.w[n - 1];
        if (rh >> 32) break;
      }
      i64 borrow = 0;
      u64 carry = 0;
      for (size_t i = 0; i < n; ++i) {
        carry += qh * v.w[i];
        const i64 d = (i64)u.w[i + j] - (i64)(u32)carry + borrow;
        u.w[i + j] = (u32)d;
        borrow = d >> 32;
        carry >>= 32;
      }
      const i64 d = (i64)u.w[j + n] - (i64)carry + borrow;
      u.w[j + n] = (u32)d;
      if (d < 0) {
        --qh;
        u64 c = 0;
        for (size_t i = 0; i < n; ++i) {
          c += (u64)u.w[i + j] + v.w[i];
          u.w[i + j] = (u32)c;
          c >>= 32;
        }
        u.w[j + n] += (u32)c;
      }
      t.w[j] = (u32)qh;
    }
    t.trim();
    u.trim();
    q = t;
    r = u.shr(s);
  }
  // floor of the square root, by Newton's iteration from above.
  static Big isqrt(const Big &a) {
    if (a.zero()) return a;
    Big x = two_to((a.bits() + 1) / 2);
    for (;;) {
      Big q, r;
      divmod(a, x, q, r);
      Big y = (x + q).shr(1);
      if (cmp(y, x) >= 0) return x;
      x = y;
    }
  }
  static Big pow(Big a, u64 n) {
    Big r(1);
    for (; n; n >>= 1, a = a * a)
      if (n & 1) r = r * a;
    return r;
  }
  std::string dec() const {
    if (zero()) return "0";
    std::string s;
    Big x = *this, q, r;
    const Big ten9(1000000000);
    while (!x.zero()) {
      divmod(x, ten9, q, r);
      u64 d = r.zero() ? 0 : r.to_u64();
      for (int i = 0; i < 9 && (!q.zero() || d); ++i, d /= 10) s.push_back((char)('0' + d % 10));
      x = q;
    }
    std::reverse(s.begin(), s.end());
    return s;
  }

 private:
  void trim() {
    while (!w.empty() && !w.back()) w.pop_back();
  }
};

enum class Dir { DOWN, UP };
static constexpr size_t REAL_BITS = 192;

// Not checked: overflow of e. The cost of add() and sub() grows with the exponents' difference.
struct Real {
  Big m;
  long e = 0;  // the value is m * 2^e
  Real() {}
  Real(u64 x) : m(x) {}
  Real(const Big &x) : m(x) {}
  Real(const Big &x, long ex) : m(x), e(ex) {}
  static Real two_to(long n) { return Real(Big(1), n); }
  bool zero() const { return m.zero(); }
  // The position of the top bit: 2^(top-1) <= value < 2^top, for a value that is not 0.
  long top() const { return (long)m.bits() + e; }
  Real cut(Dir d) const {
    const size_t b = m.bits();
    if (b <= REAL_BITS) return *this;
    const size_t s = b - REAL_BITS;
    Real r(m.shr(s), e + (long)s);
    if (d == Dir::UP && m.any_below(s)) r.m = r.m + Big(1);
    return r;
  }
  static int cmp(const Real &a, const Real &b) {
    if (a.zero() || b.zero()) return a.zero() ? (b.zero() ? 0 : -1) : 1;
    if (a.top() != b.top()) return a.top() < b.top() ? -1 : 1;
    return a.e >= b.e ? Big::cmp(a.m.shl((size_t)(a.e - b.e)), b.m) : Big::cmp(a.m, b.m.shl((size_t)(b.e - a.e)));
  }
  static Real add(const Real &a, const Real &b, Dir d) {
    if (a.zero()) return b;
    if (b.zero()) return a;
    const long e = std::min(a.e, b.e);
    return Real(a.m.shl((size_t)(a.e - e)) + b.m.shl((size_t)(b.e - e)), e).cut(d);
  }
  // a - b for a >= b.
  static Real sub(const Real &a, const Real &b, Dir d) {
    if (b.zero()) return a;
    const long e = std::min(a.e, b.e);
    return Real(a.m.shl((size_t)(a.e - e)) - b.m.shl((size_t)(b.e - e)), e).cut(d);
  }
  static Real mul(const Real &a, const Real &b, Dir d) { return Real(a.m * b.m, a.e + b.e).cut(d); }
  static Real div(const Real &a, const Real &b, Dir d) {
    BLK_REQUIRE(!b.zero(), "Real: division by zero");
    if (a.zero()) return a;
    const size_t s = REAL_BITS + 2 + b.m.bits();
    Big q, r;
    Big::divmod(a.m.shl(s), b.m, q, r);
    if (d == Dir::UP && !r.zero()) q = q + Big(1);
    return Real(q, a.e - (long)s - b.e).cut(d);
  }
  // The square root of a, rounded in direction d to REAL_BITS significant bits; a zero is returned
  // unchanged. The result's m has exactly REAL_BITS bits, or is 2^REAL_BITS if rounding up carries.
  // The root lies in [2^(root_top-1), 2^root_top), so its lowest kept bit is worth 2^unit. Counted
  // in that unit, the root is sqrt(y) with y = a / 2^(2 * unit). Its floor is Big::isqrt() of
  // floor(y), and it is a whole number only if y is a whole number and a perfect square.
  // The idea is public: the floor of a root is the floor of the root of the floor:
  // R. L. Graham, D. E. Knuth, O. Patashnik, "Concrete Mathematics", chapter 3
  static Real sqrt(const Real &a, Dir d) {
    if (a.zero()) return a;
    const long top = a.top();
    long root_top = top / 2;            // C++ rounds this towards zero,
    if (2 * root_top < top) ++root_top; // so only a positive odd top needs the step up
    const long unit = root_top - (long)REAL_BITS;
    const long shift = a.e - 2 * unit;  // y = a.m * 2^shift
    const Big whole = shift >= 0 ? a.m.shl((size_t)shift) : a.m.shr((size_t)-shift);  // floor(y)
    const bool fraction = shift < 0 && a.m.any_below((size_t)-shift);  // y is not a whole number
    Big root = Big::isqrt(whole);
    if (d == Dir::UP && (fraction || root * root != whole)) root = root + Big(1);
    return Real(root, unit);
  }
  static Real pow(Real a, u64 n, Dir d) {
    Real r(1);
    for (; n; n >>= 1) {
      if (n & 1) r = mul(r, a, d);
      if (n > 1) a = mul(a, a, d);
    }
    return r;
  }
  Big floor() const { return e >= 0 ? m.shl((size_t)e) : m.shr((size_t)-e); }
  Big ceil() const {
    if (e >= 0) return m.shl((size_t)e);
    Big r = m.shr((size_t)-e);
    return m.any_below((size_t)-e) ? r + Big(1) : r;
  }
  // For printed figures only.
  double log2() const {
    if (zero()) return -INFINITY;
    const size_t b = m.bits();
    const Big t = b > 64 ? m.shr(b - 64) : m.shl(64 - b);
    return (double)top() + std::log2((double)t.to_u64() / 18446744073709551616.0);
  }
  double value() const { return zero() ? 0.0 : std::exp2(log2()); }
};
inline bool operator<=(const Real &a, const Real &b) { return Real::cmp(a, b) <= 0; }
inline bool operator<(const Real &a, const Real &b) { return Real::cmp(a, b) < 0; }

/* lo <= true value <= hi, none negative. `bad`: a formula was used outside its range (a difference
   is not provably positive, a divisor may be 0). Big and Real end the process on misuse
   (BLK_REQUIRE, blocks/base.h); a Bound turns bad instead. A bad operand makes the result bad.
   A bad Bound's lo and hi mean nothing: a caller must treat it as a failure (error 1, parameters
   rejected). */
struct Bound {
  Real lo, hi;
  bool bad = false;
  Bound() {}
  Bound(u64 x) : lo(x), hi(x) {}
  Bound(const Big &x) : lo(Real(x).cut(Dir::DOWN)), hi(Real(x).cut(Dir::UP)) {}
  Bound(const Real &x) : lo(x), hi(x) {}
  Bound(const Real &l, const Real &h) : lo(l), hi(h), bad(h < l) {}  // two ends that have crossed are no interval
  static Bound ratio(const Big &a, const Big &b) { return Bound(a) / Bound(b); }
  static Bound failed() {
    Bound r((u64)1);
    r.bad = true;
    return r;
  }
  Bound inherit(const Bound &a, const Bound &b) {
    bad = bad || a.bad || b.bad;
    return *this;
  }
  friend Bound operator+(const Bound &a, const Bound &b) {
    return Bound(Real::add(a.lo, b.lo, Dir::DOWN), Real::add(a.hi, b.hi, Dir::UP)).inherit(a, b);
  }
  // a - b; bad unless the difference is provably positive, so a - a is bad.
  friend Bound operator-(const Bound &a, const Bound &b) {
    if (a.bad || b.bad || !(b.hi < a.lo)) return failed();
    return Bound(Real::sub(a.lo, b.hi, Dir::DOWN), Real::sub(a.hi, b.lo, Dir::UP));
  }
  friend Bound operator*(const Bound &a, const Bound &b) {
    return Bound(Real::mul(a.lo, b.lo, Dir::DOWN), Real::mul(a.hi, b.hi, Dir::UP)).inherit(a, b);
  }
  friend Bound operator/(const Bound &a, const Bound &b) {
    if (a.bad || b.bad || b.lo.zero()) return failed();
    return Bound(Real::div(a.lo, b.hi, Dir::DOWN), Real::div(a.hi, b.lo, Dir::UP));
  }
  static Bound sqrt(const Bound &a) {
    return Bound(Real::sqrt(a.lo, Dir::DOWN), Real::sqrt(a.hi, Dir::UP)).inherit(a, a);
  }
  static Bound pow(const Bound &a, u64 n) {
    return Bound(Real::pow(a.lo, n, Dir::DOWN), Real::pow(a.hi, n, Dir::UP)).inherit(a, a);
  }
  // An upper bound on the value, as an error probability; a bad Bound counts as 1.
  Real error() const { return bad ? Real(1) : hi; }
};
// Certainly a <= b, certainly a < b; false does not mean the opposite.
inline bool surely_le(const Bound &a, const Bound &b) { return !a.bad && !b.bad && a.hi <= b.lo; }
inline bool surely_lt(const Bound &a, const Bound &b) { return !a.bad && !b.bad && a.hi < b.lo; }

// For printed figures: minus log2 of an error; cut2() rounds down to two decimals (a weaker claim).
inline double bits_of(const Real &error) { return error.zero() ? INFINITY : -error.log2(); }
inline double cut2(double x) { return std::isfinite(x) ? std::floor(100.0 * x) / 100.0 : x; }

}  // namespace blk
