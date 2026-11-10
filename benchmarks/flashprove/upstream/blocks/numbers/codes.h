// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* For a code and a relative distance delta, computes three quantities that blocks.h needs for the
   round errors (one computed error for each round of challenges); also the sizes of a row's random
   tail (hidden_positions(), tail_for()), largest_m() and four more error terms, further down. E is
   the challenge field, an extension of the base field F (setting.h).

     list size      how many codewords can lie within the distance of one word
     count          how many z let u + z v agree with a codeword on a set of (1 - delta) n or more positions on which
                    u and v do not both agree with codewords: the error of mutual correlated agreement is count / |E|
     positions      the smallest t, or one more, with factor * ((1 - delta)^t + the round's other
                    error terms) <= 2^-bits; factor: the instances that share the round (else 1)

   Failure is reported by value: a `Bound` (real.h: an interval around the true value) or a
   `Proximity` that is bad, or 0 from `positions` and `largest_m`; johnson_list() reports through
   its parameter `bad` and returns 1, and Proximity::list has no bad state. The caller must check
   for it.

   A group of rows (`GroupSize`, blocks.h: rows tested together, at one distance) takes its distance
   by the rule JOHNSON (`Radius`), or as the caller gives it (AT). "The theorem on codes" is
   Theorem 4.6 of ePrint 2025/2055, the full version of a paper published at STOC 2026; the proceedings print only the plain form (their Theorem 1.5).
     JOHNSON  up to the Johnson radius, 1 - sqrt(rho), less a slack (`Proximity::eta`): list size
              and count grow as the slack shrinks; the count is that of the theorem on codes, taken as given.

   The rate in the formulas. The theorem on codes writes rho = k / n for the code of polynomials of
   degree at most k, which has dimension k + 1 and relative distance exactly 1 - rho. So a plain
   Reed-Solomon code of dimension d enters with k = d - 1. This follows the theorem on codes itself,
   ePrint 2025/2055, Theorem 4.6: "dimension k + 1. Denote rho = k / n, the slightly reduced rate of
   the code". The survey that setting.h cites for m, ePrint 2026/680, writes rho for dimension / n
   and restates the theorem with rho + 1 / n: a rate 2 / n larger than the source's. The formulas
   here follow the source.
   A circle code of dimension d (Haböck, Levit, Papini, "Circle STARKs", ePrint 2024/278) is a
   subcode of codimension 1 of a code that is equivalent to a Reed-Solomon code with k = d, and
   enters with k = d. Only this directory's tests use a Setting with circle codes
   (`Setting::circle`, setting.h). */
#pragma once
#include <string>

#include "numbers/real.h"
#include "numbers/setting.h"

namespace blk {

struct CodeSize {
  std::string name;
  unsigned log_n = 0;   // length n = 2^log_n
  u64 ell = 0;          // message length
  u64 r = 0;            // random coordinates of the zero-knowledge encoding
  bool over_E = false;  // symbols, messages and randomness are in E (physically: degree rows over F)
  u64 n() const { return (u64)1 << log_n; }
  u64 coordinates() const { return ell + r; }  // what a row consists of when it is sent in the clear
  /* The dimension of the code that contains all encodings. Plain Reed-Solomon: ell + r. Circle
     code: the random coordinates start at ell rounded up to a multiple of the least power of two
     >= r, with zeros between, and rate, distance, list size and positions are computed at that
     larger dimension. */
  u64 dim(const Setting &s) const {
    if (!s.circle || !r) return ell + r;
    const u64 step = (u64)1 << ceil_log2(r);
    return (ell + step - 1) / step * step + r;
  }
  // The formulas' k: see the top of this file.
  u64 k(const Setting &s) const { return s.circle ? dim(s) : dim(s) - 1; }
};

/* `hidden_positions`: at how many distinct positions the symbols of a row are exactly uniform if
   its random tail is.
   `tail_for`: how many random coordinates (the row's random tail) it takes to hide t positions.
   Plain Reed-Solomon: r and t. Circle code: an even number of random coordinates hides one position
   less. */
inline u64 hidden_positions(const Setting &s, const CodeSize &c) { return s.circle && c.r && c.r % 2 == 0 ? c.r - 1 : c.r; }
inline u64 tail_for(const Setting &s, u64 t) { return s.circle && t && t % 2 == 0 ? t + 1 : t; }

/* Where the count comes from (`CountSource`, setting.h).
     THIRD    3 floor(delta n) < d, d the minimum distance: count = floor(delta n) + 1, for every additive code and so for
              every interleaving. This follows the generic bound on mutual correlated agreement
              for additive codes, within a third of the minimum distance, in the paper that this
              protocol builds on (see THIRD_PARTY.md); the bound is proved there. The same count
              has sources in other papers.
              (1) For every linear code with 3 e < d: if a table of two rows f_1, f_2 is more than e
              columns from the interleaved code, at most e + 1 values of x bring the fold
              (1 - x) f_1 + x f_2 within e of the code. This is Roth and Zemor's, in the Ligero
              paper (ePrint 2022/1608, Appendix A), written out by Diamond and Posen
              (ePrint 2023/630, Theorem 2.1); for tables, and for all coordinates together at this
              value: Angeris, Evans, Roh (ePrint 2024/1399), with its proof in Diamond and Gruen
              (ePrint 2024/1351, Theorem 3.6). The citations in (1) are second-hand: not checked
              against the papers.
              (2) The survey ePrint 2026/680, p. 18, Theorem 4.8, credited to the Ligero paper's
              Appendix A: for any linear code and delta <= a third of its distance the error is at
              most (n delta + 1) / |F|, for mutual agreement too (its Lemma 4.6: below half the
              distance the two errors are equal, for every additive code).
     JOHNSON  delta < 1 - sqrt(rho): ePrint 2025/2055, Theorem 4.6 with M = 1:
              count = (2 (m + 1/2)^5 + 3 (m + 1/2) delta rho) / (3 rho^(3/2)) n
                      + (m + 1/2) / sqrt(rho),
              with the theorem's integer m >= 3. Every distance from a third of d up to the Johnson
              radius takes this count, in groups under the rules JOHNSON and AT and in the committed
              table (`TableSize`).
              We take this theorem as given: we have not verified
              a proof. The full version states it; its overview says that the authors "outline how
              our improvements impact the mutual correlated agreement theorem proven in" a note
              (p. 17). After the theorem comes one paragraph of four sentences (p. 29), which
              restricts to the separable case "for simplicity" and computes no bound. The full
              version says that a proof is discussed in that note, and its list of references marks
              it as a personal communication (pp. 28, 48). Read in the version dated November 6, 2025, and in the same paper's posting ECCC TR25-169 of November 7, 2025. */
// THIRD, one more footing in another paper, for every linear code at a radius below a third of its distance: Bordage,
// Chiesa, Guan, Manzur, "All Polynomial Generators Preserve Distance with Mutual Correlated Agreement", Cryptology
// ePrint Archive 2025/2051 (version of February 6, 2026), Theorem 6.1, taken for two words, and Corollary 8.6
// (Theorem 6.1 prints max(e, 1); the other sources have e + 1).
inline const char *name_of(CountSource s) {
  return s == CountSource::THIRD ? "within a third of the distance" : "Johnson";
}
// A count's source, as blocks/frame/prints.h prints it ("its closeness terms rest on ...").
inline const char *rests_on(CountSource s) {
  return s == CountSource::THIRD ? "a lemma of the paper that this protocol builds on (see THIRD_PARTY.md), proved there"
                                : "the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given";
}
// How a group of rows takes its distance: described at the top of this file.
enum class Radius { JOHNSON, AT };  // AT: no rule: a distance that a caller names, in whole positions (blocks.h: group_at)
inline const char *name_of(Radius r) {
  return r == Radius::JOHNSON ? "Johnson radius with slack" : "a distance of its own";
}

// A code at one relative distance `delta`, a fraction of the length n. Default-constructed it is
// bad(). A round error computed from a bad `agree`, `delta` or `count` is bad too, and
// Plan::close() then rejects the plan (plan.h: the list of all rounds). `list` has no bad state (it
// is 1 by default), so a caller that reads only `list` must check bad() itself.
struct Proximity {
  Bound agree = Bound::failed();  // 1 - delta
  Bound delta = Bound::failed();
  Bound eta;    // JOHNSON: 1 - sqrt(rho) - delta
  u64 m = 0;    // JOHNSON: the theorem's integer m (`CountSource` in setting.h)
  bool from_m = false;  // the distance was made from m, so sqrt(rho) / eta is the integer m exactly
  CountSource source = CountSource::JOHNSON;
  Bound count = Bound::failed();
  Big list = Big(1);  // an integer bound on the list size
  bool bad() const { return agree.bad || delta.bad || count.bad; }
};

/* A code as the formulas read it: a length n and the k of "The rate in the formulas" at the top of
   this file. Only `word_of` and `repeated` construct one, so the formulas cannot be given a k that
   does not come from a code's sizes.
   `repeated`: a code of length 2^b with every codeword written 2^(a-b) times, to the length 2^a;
   the same relative distance and the same k / n of the formulas (not dimension / length). It models
   a short row whose symbols are committed as copies in the leaves of a longer tree, where nothing
   forces the copies to agree. Use it only for distances, list sizes and counts. Where its count is
   not THIRD's, callers add one list size to it (`analysis_at`, blocks.h). */
class Word {
 public:
  u64 n() const { return (u64)1 << log_n_; }
  u64 k() const { return k_; }
  friend Word word_of(const Setting &s, const CodeSize &c);
  friend Word repeated(const Setting &s, const CodeSize &c, unsigned log_n);

 private:
  Word(unsigned log_n, u64 k) : log_n_(log_n), k_(k) {}
  unsigned log_n_;
  u64 k_;
};
inline Word word_of(const Setting &s, const CodeSize &c) { return Word(c.log_n, c.k(s)); }
inline Word repeated(const Setting &s, const CodeSize &c, unsigned log_n) {
  return log_n <= c.log_n ? word_of(s, c) : Word(log_n, c.k(s) << (log_n - c.log_n));
}

inline Bound rho_of(const Word &c) { return Bound::ratio(Big(c.k()), Big(c.n())); }
inline Bound rho_of(const Setting &s, const CodeSize &c) { return rho_of(word_of(s, c)); }

/* The counting form of the Johnson bound. With lambda = 1 - (the relative minimum distance), a word
   has at most (a - lambda) / (a^2 - lambda) codewords that agree with it on at least a fraction a
   of the positions, if a^2 > lambda. The bound uses only the minimum distance, so it holds for
   interleaved codes, and for a group's main code and mask code (the code of its rows of random
   values) read at the same positions, with the smaller of their relative distances. It is rounded
   down, because a list size is an integer. If a^2 > lambda is not certain, `bad` is set and 1 is
   returned. */
inline Big johnson_list(const Bound &agree, const Bound &lambda, bool &bad) {
  const Bound l = (agree - lambda) / (agree * agree - lambda);
  bad = bad || l.bad;
  return l.bad ? Big(1) : l.hi.floor();
}

// The count of JOHNSON above. M here is m + 1/2, not the theorem's M.
inline Bound johnson_count(const Word &c, const Bound &rho, const Bound &delta, u64 m) {
  const Bound M = Bound::ratio(Big(2) * Big(m) + Big(1), Big(2)), root = Bound::sqrt(rho);  // in `Big`, because 2 m + 1 can exceed 64 bits
  return (Bound(2) * Bound::pow(M, 5) + Bound(3) * M * delta * rho) / (Bound(3) * rho * root) * Bound(Big(c.n())) + M / root;
}

// The code at the distance that m gives: 1 - delta = sqrt(rho) (1 + 1/m).
inline Proximity johnson(const Word &c, u64 m) {
  Proximity x;
  const Bound rho = rho_of(c), root = Bound::sqrt(rho);
  x.m = m, x.from_m = true, x.source = CountSource::JOHNSON;
  x.eta = root / Bound(Big(m));
  x.agree = root + x.eta;
  x.delta = Bound(1) - x.agree;
  x.count = johnson_count(c, rho, x.delta, m);
  bool bad = m < 3 || c.k() >= c.n();
  x.list = johnson_list(x.agree, rho, bad);
  if (bad) x.count = Bound::failed();
  return x;
}
inline Proximity johnson(const Setting &s, const CodeSize &c, u64 m) { return johnson(word_of(s, c), m); }

// The code at the distance 1 - agree, with the count of the first source that applies: THIRD,
// JOHNSON. Bad at or beyond the Johnson radius.
inline Proximity at_distance(const Word &c, const Bound &agree) {
  Proximity x;
  if (agree.bad || c.k() >= c.n()) return x;  // x is bad; n - k below must not wrap
  const Bound n = Bound(Big(c.n())), d = Bound(Big(c.n() - c.k())), rho = rho_of(c);
  x.agree = agree;
  x.delta = Bound(1) - agree;
  if (x.delta.bad) return x.count = Bound::failed(), x;
  const Big errors = (x.delta * n).hi.floor();  // at most so many positions differ
  if (Big(3) * errors < Big(c.n() - c.k())) {
    x.source = CountSource::THIRD;
    x.count = Bound(errors + Big(1));
  } else {
    const Bound root = Bound::sqrt(rho);
    x.source = CountSource::JOHNSON;
    x.eta = agree - root;
    const Bound q = root / x.eta;
    if (q.bad || !q.hi.ceil().fits_u64()) return x.count = Bound::failed(), x;
    x.m = std::max<u64>(q.hi.ceil().to_u64(), 3);
    x.count = johnson_count(c, rho, x.delta, x.m);
    bool bad = false;
    if (!surely_lt(Bound(2) * x.delta * n, d)) x.list = johnson_list(agree, rho, bad);
    if (bad) x.count = Bound::failed();
  }
  return x;
}
inline Proximity at_distance(const Setting &s, const CodeSize &c, const Bound &agree) { return at_distance(word_of(s, c), agree); }

/* The error of mutual correlated agreement for a line through two words of the interleaved code.
   The number of rows has no factor in it: `CountSource` in setting.h says on what that rests.
   Where the count is that of the Reed-Solomon code in which a circle code lies, the circle code
   pays one more term for being a subcode: (list size) / |E|. */
inline Bound mca_error(const Setting &s, const Proximity &x) {
  Bound a = x.count;
  if (s.circle && x.source != CountSource::THIRD) a = a + Bound(x.list);
  return a / Bound(s.card_E());
}

/* The smallest t with factor (agree^t + extra) <= 2^-bits; 0 if there is none up to 2^32 or an
   argument is bad. extra is the rest of the round's error, factor the number of instances that
   share the round. Upper bounds are compared, so t is never too small; it can be one too large if
   the left side is within the power's rounding error (a few times t units in the last of real.h's
   REAL_BITS bits) of the right side. */
inline u64 positions(const Bound &agree, unsigned bits, const Bound &extra = Bound((u64)0), u64 factor = 1) {
  if (!factor) return 0;
  const Real goal = Real::div(Real::two_to(-(long)bits), Real(factor), Dir::DOWN);
  if (agree.bad || extra.bad || !(agree.hi < Real(1)) || !(extra.hi < goal)) return 0;
  auto ok = [&](u64 t) { return Real::add(Real::pow(agree.hi, t, Dir::UP), extra.hi, Dir::UP) <= goal; };
  u64 hi = 1;
  while (!ok(hi))
    if ((hi *= 2) > ((u64)1 << 32)) return 0;
  u64 lo = hi / 2;  // not ok, or 0
  while (hi - lo > 1) {
    const u64 mid = lo + (hi - lo) / 2;
    (ok(mid) ? hi : lo) = mid;
  }
  return hi;
}

/* An m in [3, limit] at which `good` is true, or 0 if it is false at every m from 3 to 66. It is
   the largest such m if `good` is true exactly on one interval of integers that begins below 67.
   (At a rate above 9/16 the smallest m give a distance below 0: the interval begins above 3.)
   Needs limit >= 67. */
template <class Good>
u64 largest_m(Good good, u64 limit = (u64)1 << 40) {
  u64 lo = 3;
  while (lo < 67 && !good(lo)) ++lo;
  if (lo == 67) return 0;
  u64 hi = lo + 1;
  while (hi < limit && good(hi)) lo = hi, hi *= 2;
  if (hi >= limit) {
    if (good(limit)) return limit;
    hi = limit;
  }
  while (hi - lo > 1) {
    const u64 mid = lo + (hi - lo) / 2;
    (good(mid) ? lo : hi) = mid;
  }
  return lo;
}

// Each function below bounds the probability of the stated event.
// powers_error(): `rows` values, not all 0, combined by the powers of a challenge from E give 0.
// field_error(): a nonzero polynomial of degree `degree` vanishes at a challenge from E;
// field_error_nonzero(): at a challenge from E without 0.
inline Bound powers_error(const Setting &s, u64 rows) { return Bound::ratio(Big(rows ? rows - 1 : 0), s.card_E()); }
inline Bound field_error(const Setting &s, u64 degree) { return Bound::ratio(Big(degree), s.card_E()); }
inline Bound field_error_nonzero(const Setting &s, u64 degree) { return Bound::ratio(Big(degree), s.card_E() - Big(1)); }
/* Two different messages of a list of size L have the same out-of-domain sample: a random linear
   form of the whole message, chosen after the message is sent (blocks/iop/send.h). The sample's
   seed is a point z of a cube with cube_bits variables and a factor beta != 0 for the pad, the one
   element of E in the message that is set apart to mask the answer. Two messages that differ in the
   pad agree for at most one beta at each z; two that differ elsewhere agree only for z on a
   hypersurface of degree cube_bits. That gives max(cube_bits / |E|, 1 / (|E| - 1)) for a pair and
   one answer. Further answers have independent seeds and pads of their own, so the pair's error is
   that value to the power `answers`. There are at most L^2 / 2 pairs. */
inline Bound sample_error(const Setting &s, const Big &list, unsigned cube_bits, unsigned answers) {
  return Bound(list * list) / Bound(2) * Bound::pow(cube_bits >= 2 ? field_error(s, cube_bits) : field_error_nonzero(s, 1), answers);
}
inline Bound sample_error(const Setting &s, const Big &list, unsigned cube_bits) { return sample_error(s, list, cube_bits, s.samples); }

}  // namespace blk
