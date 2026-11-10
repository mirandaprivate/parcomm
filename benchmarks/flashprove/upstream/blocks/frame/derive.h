// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Computes the sizes that the parameter file does not state. For a group (the rows that one
   proximity test reads at one list of positions) these are its slack (the gap below the Johnson
   radius) and distance, the number of positions, the length of its codes' random tails ("random
   coordinates" in messages) and the test's stages. `derive` does this for a request's segments
   (`link` in identifiers: one sub-proof's part of the proof) and sets the committed table's
   distances; lin_frame.h, link_frame.h and wf_frame.h size their groups with this file's
   functions. Prover and verifier compute the sizes here; `configure` (configure.h) rejects the
   parameters unless every round planned from these sizes meets the error target, proof of work
   counted.

   Random tails and positions depend on each other: a row needs a random tail long enough to hide
   the positions at which it is opened, the tail raises the code's rate, and the rate fixes the
   number of positions. So `solve_group` iterates. */
#pragma once
#include <string>
#include <vector>

#include "frame/parameters.h"

namespace blk {

// What the sizes need of the sub-proofs' shapes. One group's codes and distance serve all
// segments: a `link_` number is the largest over them, or a bound (not checked here).
// `link_batched_rows`: the most linear claims that a segment's closing batches into one.
// `link_cube_bits`: the most variables of the point at which an out-of-domain sample reads one of
// a segment's oracle messages (`SampleCube::bits`, defined in blocks/iop/send.h).
struct FrameShape {
  unsigned link_rows_log = 8;     // a segment's group has 2^this main rows, unsent rows counted
  unsigned last_rows_log = 1;     // the same for the message into which the last segment is closed
  unsigned link_cube_bits = 0;    // see above
  u64 link_slots = 1;             // rows that are folded side by side in one main row (kept rows)
  unsigned link_messages = 8;     // the most oracle messages of a segment that share a seed
  u64 link_batched_rows = (u64)1 << 16;
  u64 index_rows = 0;             // rows of the committed matrix, for `lin_lines` (lin_frame.h)
};

// `t`: the number of positions drawn for the group's list. stages[0] is the group itself.
struct TestSizes {
  GroupSize group;
  u64 t = 0;
  std::vector<Stage> stages;
};

struct Derived {
  Setting set;
  TestSizes link;       // a segment's group; `stages` stays empty
  TestSizes last;       // the last closing's message: folded row and carried row
  TableSize table;      // the committed table at its distances
  std::string refused;  // not empty: why sizing failed; other fields may be incomplete
};

inline Setting setting_of(const Parameters &p) {
  Setting s = setting_goldilocks();
  s.target = p.sec;
  s.samples = p.answers;
  s.salt_bytes = p.salt;
  for (int k = 0; k < 5; ++k)
    for (int i = 0; i < 3; ++i) s.work[k][i] = p.work[k][i];
  return s;
}

inline GroupSize shape_of(const std::string &name, const GroupLine &l, unsigned k, bool over_E) {
  GroupSize g;
  g.name = name, g.k = k, g.h = 0;
  g.main.name = name, g.main.log_n = l.log_n, g.main.ell = (u64)1 << l.log_ell, g.main.over_E = over_E;
  g.mask.name = name + " mask", g.mask.log_n = l.mask_log_n, g.mask.ell = (u64)1 << l.mask_log_ell, g.mask.over_E = true;
  return g;
}

/* Solves the group whose shape is in x.group. draws(t) is the number of drawn positions at which a
   main row is opened in all when one list has t; a mask row is opened at one list. x.t goes up from
   its start (0 by default) and stops where the random tails hide the draws that follow from them,
   so a true result is always valid. The last x.t is not proved to be the smallest such number
   under the Johnson rule, where a step of GroupSize::m, which sets the slack, can lower the
   positions needed by one as the rate grows.
   The other error terms of the round that draws a list are far below 2^-target: `margin` allows
   2^-(target + 30) for them, and the plan's exact comparison (`Planned::Data::holds`,
   blocks/numbers/plan.h) decides. */
template <class Draws>
bool solve_group(const Setting &s, Scope scope, const GroupLoad &load, Draws draws, TestSizes &x, std::string &why) {
  const GroupSize shape = x.group;
  for (int turn = 0; turn < 32; ++turn) {
    GroupSize g = shape;
    g.main.r = tail_for(s, draws(x.t)), g.mask.r = tail_for(s, x.t);
    if (g.main.dim(s) > g.main.n() || g.mask.dim(s) > g.mask.n()) return why = shape.name + ": message and random coordinates exceed the code's length", false;
    const Bound margin(Real::two_to(-(long)s.target - 30));
    x.group = size_group(s, scope, g, load);
    const bool serves = group_holds(s, scope, x.group, load);
    const Analysis a = analysis(s, x.group);
    const u64 t = serves ? scope_positions(s, scope, a.at.agree, margin) : a.fault ? positions_from_below(s, scope, g, load, margin) : 0;
    if (!serves && t <= x.t)  // no way up is left
      return why = shape.name + ": " + (a.fault ? a.fault : "no distance under its rule holds the group's rounds at the target"), false;
    if (!t) return why = shape.name + ": no number of positions reaches the target", false;
    if (t <= x.t) return true;
    x.t = t;
  }
  return why = shape.name + ": positions and random coordinates do not settle", false;
}

// Fills x.stages for the proximity test on x.group. x.group and x.t must be solved already.
inline bool solve_test(const Setting &s, Scope scope, const GroupLine &l, TestSizes &x, std::string &why) {
  Plan scratch(s);
  scratch.begin_scope(scope);
  x.stages = solve_stages(scratch, x.group, x.t, l.stages);
  if (scratch.refused()) why = scratch.why();
  return !scratch.refused();
}

// Fills d.table, with its distances; sets d.refused if sizing fails. d.set must be filled first.
inline void derive_table(const Parameters &p, Derived &d) {
  const Setting &s = d.set;
  std::string &why = d.refused;
  TableSize &T = d.table;
  T.code.name = "table", T.code.log_n = p.table_log_n, T.code.ell = p.table_data, T.code.r = p.table_random;
  T.v = p.table_rows_log, T.answers = p.answers;
  if (T.code.dim(s) > T.code.n()) return (void)(why = "table: data and random coordinates exceed the code's length");
  if (!size_recorded_table(s, T, p.slots, p.block_log)) why = "table: no distance holds the fold point's round at the target";
}

inline Derived derive(const Parameters &p, const FrameShape &f) {
  Derived d;
  const Setting &s = d.set = setting_of(p);
  std::string &why = d.refused;
  auto load_of = [](const GroupSize &g, u64 messages, u64 rows) {
    GroupLoad l;
    l.cube_bits = stage_cube_bits(g), l.messages = messages, l.batched_rows = rows;
    return l;
  };
  auto once = [](u64 t) { return t; };

  d.link.group = shape_of("link", p.link, f.link_rows_log, false);
  d.link.group.slots = f.link_slots;
  GroupLoad link_load = load_of(d.link.group, f.link_messages, f.link_batched_rows);
  link_load.cube_bits = std::max(link_load.cube_bits, f.link_cube_bits);
  if (!solve_group(s, Scope::REQUEST, link_load, once, d.link, why)) return d;
  // The last segment is closed into a message with the segments' codes, distance and t; it has at
  // most as many rows (not checked here), all over E, the extension field.
  d.last = d.link;
  d.last.group.name = "last", d.last.group.k = f.last_rows_log, d.last.group.slots = 1, d.last.group.main.over_E = true;
  if (!solve_test(s, Scope::REQUEST, p.link, d.last, why)) return d;

  derive_table(p, d);
  return d;
}

}  // namespace blk
