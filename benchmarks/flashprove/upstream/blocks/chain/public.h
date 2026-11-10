// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Types that make a program bind the public inputs of a protocol run, that is, write them into the
// first query of its hash chain. The last paragraph of this comment says how far they reach. A hash
// chain (`ChainT`, blocks/chain/chain.h) is the Fiat-Shamir transcript of one interactive protocol;
// its first query is the hash input that starts it.
//
// The verifier's challenges are hashes of queries, and each query depends on all queries before it.
// A public input that is in no query (a table, a layout, a key, a claim's value) changes no
// challenge. A prover who can choose or influence such an input picks it after seeing the
// challenges, and a false claim is accepted with certainty.
//
// The rule for a function that runs on a channel (`Rounds` in blocks/chain/rounds.h, through which
// a side sends, receives and gets challenges): besides its context (`ProverContext`,
// `VerifierContext`) it takes only Public<T> values; what other functions have made on the same
// channel from bound values, received messages and challenges; and the prover's private data. It
// takes no plain number, struct, table or callback that its caller could have obtained elsewhere.
// The types enforce the rule only where a value enters a chain. A value read by Rounds::get() is
// plain, and the lower protocol steps in blocks/iop/ (send.h, bat.h, split.h, psc.h, fold.h, cs.h,
// base_case.h) take plain values: there the caller must keep the rule.
#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "base.h"
#include "chain/chain.h"
#include "field/field.h"
#include "frame/version.h"
#include "hash/hash.h"
#include "iop/message.h"
#include "numbers/plan.h"

namespace blk {

// Canonical bytes: a type T that is to be bound needs, next to T, an overload
//   void public_bytes(Writer &, const T &)
// which writes every field that changes what a verifier does, a list's length before the list, and
// no field that the other fields determine.
template <class T, std::enable_if_t<std::is_integral<T>::value || std::is_enum<T>::value, int> = 0>
inline void public_bytes(Writer &w, T x) {
  w.put_u64((u64)x);
}
inline void public_bytes(Writer &w, F x) { w.put(x); }
inline void public_bytes(Writer &w, const E &x) { w.put(x); }
inline void public_bytes(Writer &w, const Digest &x) { w.put(x); }
inline void public_bytes(Writer &w, const std::string &s) {
  w.put_u64(s.size());
  w.put_bytes(reinterpret_cast<const u8 *>(s.data()), s.size());
}
// A byte string: its length as one 64-bit word, then the raw bytes (other vectors of integers: one
// word for each entry).
inline void public_bytes(Writer &w, const std::vector<u8> &bytes) {
  w.put_u64(bytes.size());
  w.put_bytes(bytes.data(), bytes.size());
}
template <class A, class B> inline void public_bytes(Writer &w, const std::pair<A, B> &x);
template <class T> inline void public_bytes(Writer &w, const std::vector<T> &v) {
  w.put_u64(v.size());
  for (const auto &x : v) public_bytes(w, (const T &)x);
}
template <class A, class B> inline void public_bytes(Writer &w, const std::pair<A, B> &x) {
  public_bytes(w, x.first);
  public_bytes(w, x.second);
}

// Faults: T also needs an overload
//   const char *public_fault(const T &)
// which returns nullptr, or a message if the value contradicts itself or is no valid value of its
// type (a count that differs from its list's length, an index beyond its table). The Binder calls
// it on every item's value (binding a type without the overload does not compile) and ends the
// process on a fault, so a verifier that reads such a value from a file must call public_fault()
// first and reject the file if it reports a fault. What only several bound objects together can
// show (a proximity test's stages against the codes) is checked by the function that takes them.
template <class T, std::enable_if_t<std::is_integral<T>::value || std::is_enum<T>::value, int> = 0>
inline const char *public_fault(T) {
  return nullptr;
}
inline const char *public_fault(F) { return nullptr; }
inline const char *public_fault(const E &) { return nullptr; }
inline const char *public_fault(const Digest &) { return nullptr; }
inline const char *public_fault(const std::string &) { return nullptr; }
template <class A, class B> inline const char *public_fault(const std::pair<A, B> &x);
template <class T> inline const char *public_fault(const std::vector<T> &v) {
  for (const auto &x : v)
    if (const char *fault = public_fault((const T &)x)) return fault;
  return nullptr;
}
template <class A, class B> inline const char *public_fault(const std::pair<A, B> &x) {
  const char *fault = public_fault(x.first);
  return fault ? fault : public_fault(x.second);
}

class Binder;
class Rounds;

// Identifies the Binder, and so the first query, that bound a value.
using PublicMark = std::shared_ptr<const char>;

// A bound value (in messages: it "stands in" a first query). Only a Binder, derive() and element()
// make one that is not empty. It has no getter: Rounds::get() returns the value, and ends the
// process unless it was bound by the channel's own Binder or for an earlier chain that this chain
// follows (Binder::follows()).
// A value computed from bound values only is bound too: x.derive(f) is f(x). f is a function or a
// lambda without captures, so that it cannot carry in its caller's local values; it can still read
// globals.
template <class T> class Public {
public:
  Public() = default; // stands in no query: no channel reads it
  template <class Function> auto derive(Function &&f) const {
    static_assert(std::is_empty<std::decay_t<Function>>::value || std::is_function<std::remove_pointer_t<std::decay_t<Function>>>::value,
                  "derive takes a function or a lambda without captures");
    BLK_REQUIRE(value_ != nullptr, "public input: a value that stands in no first query");
    using R = std::decay_t<decltype(f(*value_))>;
    return Public<R>(mark_, f(*value_));
  }

private:
  friend class Binder;
  friend class Rounds;
  template <class U> friend class Public;
  template <class A, class B, class Function> friend auto derive(const Public<A> &, const Public<B> &, Function &&);
  template <class U> friend Public<U> element(const Public<std::vector<U>> &, size_t);
  Public(PublicMark mark, T value) : mark_(std::move(mark)), value_(std::make_shared<const T>(std::move(value))) {}
  PublicMark mark_;
  std::shared_ptr<const T> value_;
};

// f(a, b), for two values bound by one Binder.
template <class A, class B, class Function> inline auto derive(const Public<A> &a, const Public<B> &b, Function &&f) {
  static_assert(std::is_empty<std::decay_t<Function>>::value || std::is_function<std::remove_pointer_t<std::decay_t<Function>>>::value,
                "derive takes a function or a lambda without captures");
  BLK_REQUIRE(a.value_ && b.value_ && a.mark_ == b.mark_, "public input: two values of two first queries");
  using R = std::decay_t<decltype(f(*a.value_, *b.value_))>;
  return Public<R>(a.mark_, f(*a.value_, *b.value_));
}

// Entry i of a bound vector.
template <class U> inline Public<U> element(const Public<std::vector<U>> &v, size_t i) {
  BLK_REQUIRE(v.value_ && i < v.value_->size(), "public input: no such entry");
  return Public<U>(v.mark_, (*v.value_)[i]);
}

// The state of one chain, shared with every chain that follows it. `refused`: its verifier has
// rejected. `ended`: the channel's finish() or finish_inside_a_file() has run. Unless its channel
// has rejected, a chain that finishes before one that it follows ends the process.
struct ChainFate {
  bool refused = false, ended = false;
};

// The end of a chain: its seal (a digest of the entire chain) and the marks of the first queries
// that it depends on. Only Rounds::seal() makes one that is not empty. The Sealed of a rejected
// chain records the rejection, and a verifier's channel that follows it starts rejected: a session
// (`VerifierSession`: the chains of one file's check) is accepted only if all its chains are.
class Sealed {
public:
  Sealed() = default; // of no chain
  const Digest &digest() const {
    BLK_REQUIRE(!marks_.empty(), "public input: the seal of no chain");
    return digest_;
  }

private:
  friend class Binder;
  friend class Rounds;
  // fates: of the sealed chain and of the chains that it follows. A following verifier's channel
  // reads them at every ok(), so that a rejection after the seal (leftover bytes, a failed opening)
  // reaches every chain that follows.
  Sealed(std::vector<PublicMark> marks, const Digest &digest, std::vector<std::shared_ptr<const ChainFate>> fates, const Digest &instance)
      : marks_(std::move(marks)), digest_(digest), instance_(instance), fates_(std::move(fates)) {}
  std::vector<PublicMark> marks_;
  Digest digest_{}, instance_{}; // instance_: the sealed chain's configuration
  std::vector<std::shared_ptr<const ChainFate>> fates_;
};

// Builds a chain's first query. parameter() and statement() append an item (a name and the value's
// canonical bytes) to one of its two lists. The first statement(), follows() or parameters_digest()
// closes the parameters: a later parameter() ends the process. A channel can be constructed only
// from a Binder, which from then on accepts no further item.
// The parameters are what is fixed before a request is known (codes, plan, shapes, tables, keys).
// The run's value is a hash of the label, the parameters' digest and the chain's salt
// (`random_bytes` in identifiers). The hash of every leaf of the trees that the chain sends starts
// from it (`start` in blocks/hash/hash.h), so these trees can be built before the statement is
// complete.
class Binder {
public:
  // label: which protocol the chain runs; below 2^32 (checked by the channel). plan: the list of
  // the chain's rounds. The parameters begin with the version word, `BLK_FIELD`, `BLK_HASH` and the
  // plan: some of its settings, such as the byte widths of a proof's items, and every round's name,
  // sizes, draws and bits of proof of work. The channel takes its plan from plan(), so the rounds
  // that it goes through have the names and numbers that were bound. Not bound, among others: a
  // round's kind (for a name of the proof format, kind_of_round() fixes it), its error bound, and
  // `first_list`, `group`, `table`, `combinations`, `sharing` (`PlanRound`, blocks/numbers/plan.h).
  Binder(u64 label, const Planned &plan) : label_(label), plan_(plan), mark_(std::make_shared<const char>('\0')) {
    item(parameters_, "the format's version", (u64)SERVED_VERSION);
    write_plan(parameters_, plan);
  }
  Binder(const Binder &) = delete;
  Binder &operator=(const Binder &) = delete;

  template <class T> Public<T> parameter(const char *name, T value) {
    BLK_REQUIRE(!parameters_closed_, "binder: the parameters are closed by the first item of the statement and by the run's value");
    other_items_ = true;
    return item(parameters_, name, std::move(value));
  }
  template <class T> Public<T> statement(const char *name, T value) {
    BLK_REQUIRE(!closed_, "binder: the chain has begun: its first query takes nothing more");
    parameters_closed_ = true;
    return item(statement_, name, std::move(value));
  }
  // This chain follows an earlier chain of its session: that chain's seal becomes an item of this
  // statement, so what is bound there is bound here.
  void follows(const Sealed &earlier) {
    statement("the seal of an earlier chain", earlier.digest());
    earlier_.insert(earlier_.end(), earlier.marks_.begin(), earlier.marks_.end());
    earlier_fates_.insert(earlier_fates_.end(), earlier.fates_.begin(), earlier.fates_.end());
    earlier_instances_.push_back(earlier.instance_);
  }

  // The chain's salt: 32 bytes, see `FirstQuery` in chain.h. By default the prover's own, written
  // first in the chain's part of the proof. A chain with no_random_bytes() has none and can send no
  // tree.
  void no_random_bytes() { salted_ = false; }
  // Both sides take the salt from x, a value bound by this Binder. It is to be fresh in every run
  // (not checked), and everyone who sees the proof knows it.
  void random_bytes_are(const Public<Digest> &x) {
    BLK_REQUIRE(x.value_ && x.mark_ == mark_, "binder: these bytes do not stand in this first query");
    salt_in_proof_ = false, salt_ = *x.value_;
  }

  // The first query of every chain that runs on a channel has the item `configuration`: the
  // configuration digest, a hash of all public files (configuration_digest() in
  // blocks/frame/heads.h). The hash of every tree vertex, and of every leaf of a tree made outside
  // a run (a stored tree, the binding tree), starts from it. A chain and the chains that it follows
  // must agree on it, or the process ends. configuration() must be called once, before any
  // parameter().
  Public<Digest> configuration(const Digest &digest) {
    BLK_REQUIRE(!has_configuration_, "binder: a chain names its configuration once");
    BLK_REQUIRE(!other_items_, "binder: the configuration is the first item behind the plan");
    has_configuration_ = true, configuration_ = digest;
    return parameter("configuration", digest);
  }
  Digest configuration_digest() const {
    BLK_REQUIRE(has_configuration_, "binder: every chain names its configuration, as the first item behind the plan");
    const Digest d = configuration_;
    for (const Digest &e : earlier_instances_)
      BLK_REQUIRE(e == d, "binder: two chains of one session name two configurations");
    return d;
  }

  const Planned &plan() const { return plan_; }
  // The two lists as they are hashed, for code that builds or compares first queries itself.
  const std::vector<u8> &parameter_bytes() const { return parameters_.bytes(); }
  const std::vector<u8> &statement_bytes() const { return statement_.bytes(); }
  Digest parameters_digest() const {
    parameters_closed_ = true;
    Hasher h(Tag(Domain::PARAMETERS), 1); // index 1: this list; a parameter file's own text is hashed under index 0
    h.push(parameters_.data(), parameters_.size());
    return h.finish();
  }

private:
  friend class Rounds;
  // Closes the Binder. salt: the prover's; the verifier passes zeros and then reads the salt from
  // the proof.
  FirstQuery first_query(const Digest &salt) const {
    closed_ = true;
    FirstQuery q;
    q.label = label_, q.parameters = parameters_digest(), q.statement = statement_.bytes();
    q.salted = salted_, q.salt_in_proof = salt_in_proof_, q.salt = salt_in_proof_ ? salt : salt_;
    return q;
  }
  static void write_plan(Writer &w, const Planned &plan) {
    const Setting &s = plan->set;
    for (u64 x : {(u64)BLK_FIELD, (u64)BLK_HASH, (u64)s.salt_bytes, (u64)s.nonce_bytes, (u64)s.position_bytes, (u64)s.samples,
                  (u64)s.saved_coefficient, (u64)plan->rounds_.size()})
      w.put_u64(x);
    for (const PlanRound &r : plan->rounds_) {
      public_bytes(w, r.name); // the name says the kind of round
      for (u64 x : {(u64)r.count, (u64)r.work_bits, (u64)r.message_bytes, (u64)r.roots, (u64)r.answers, (u64)r.draw_E, (u64)r.draw_nonzero,
                    (u64)r.draw_positions, (u64)r.log_n, (u64)r.draw_trits, (u64)r.rows})
        w.put_u64(x);
    }
  }
  template <class T> Public<T> item(Writer &w, const char *name, T value) {
    const char *fault = public_fault((const T &)value);
    BLK_REQUIRE(!fault, fault ? fault : "");
    Writer bytes;
    public_bytes(bytes, (const T &)value);
    public_bytes(w, std::string(name));
    w.put_u64(bytes.size());
    w.put_bytes(bytes.data(), bytes.size());
    return Public<T>(mark_, std::move(value));
  }
  u64 label_;
  const Planned plan_;
  PublicMark mark_;
  std::vector<PublicMark> earlier_;
  std::vector<Digest> earlier_instances_;
  Writer parameters_, statement_;
  bool salted_ = true, salt_in_proof_ = true, has_configuration_ = false, other_items_ = false;
  Digest salt_{}, configuration_{};
  std::vector<std::shared_ptr<const ChainFate>> earlier_fates_; // read by the channel, at every ok() and at its end
  mutable bool parameters_closed_ = false, closed_ = false;
};

} // namespace blk
