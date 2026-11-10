// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Sizing of LIN (blocks/iop/lin.h), the test on the committed table (`Index` in identifiers): one
   linear claim per request of a batch, proved with one fold of the table. Derived from the
   parameter file's lines, by the formulas of blocks/numbers/blocks.h: LIN's shapes, codes, numbers
   of positions and columns, stages of proximity tests (`iopp` in identifiers), and rounds in its
   three hash chains' `Plan`s.
   Callers must check `LinSized::refused`: the planning functions do not. */
#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "frame/derive.h"
#include "iop/lin.h"

namespace blk {

// LIN's parameters, without the table's sizes (`TableSize`).
// M: the matrix in the table. The folded row: the table's rows combined with weights from the
// shared point (a challenge common to the batch). Omega: M times that point's equality vector.
// W: a request's row weights times M. A group: oracle messages tested together.
// LINK (blocks/iop/link.h): proof that a further (leaf) table has the first table's entries.
// The one-time radius: the distance at which the committer's one-time test tests the table
// (`TableAnalysis::once`, blocks/numbers/blocks.h).
struct LinLines {
  unsigned k_lo = 0;     // a row of the table holds 2^k_lo columns of the matrix
  u64 matrix_rows = 0;
  GroupLine folded, omega, own;  // the groups of the folded row, of Omega, of a request; own has omega's main code
  u64 requests = 1;              // the most in a batch
  size_t input_mask_rows = 1;    // of the request's oracle message whose entries mask its claim
  // See `VectorLay` (blocks/iop/foldeq.h), `LinShape::masks_ahead`, `IoppChoice::masks_ride`.
  bool rows_high = true, masks_ahead = true, masks_ride = true;
  bool one_time = false;         // committer's one-time test: a batch of one at the one-time radius
  u64 columns_elsewhere = 0;     // columns of this table that LINK proofs to other tables open
};

// one_time: from the lines `once-folded`, `once-omega`, `once-own`.
// Leaves at their defaults: input_mask_rows, columns_elsewhere, rows_high, masks_ahead, masks_ride.
inline LinLines lin_lines(const Parameters &p, const FrameShape &f, bool one_time) {
  LinLines L;
  L.k_lo = p.block_log - p.table_rows_log;  // `block_log` counts all the shared point's coordinates
  L.matrix_rows = f.index_rows;
  L.folded = one_time ? p.once_folded : p.folded, L.omega = one_time ? p.once_omega : p.omega, L.own = one_time ? p.once_own : p.own;
  L.requests = one_time ? 1 : p.slots, L.one_time = one_time;
  return L;
}

struct LinSized {
  LinLines lines;
  LinShape S;
  OracleNotice input;  // the request's mask message, as LIN's group expects it
  TableSize table;
  TestSizes folded, omega, own;  // omega: only if the folded row and Omega differ in code
  u64 batches = 0;               // batches, counted when begun, that one stored table (the first or a leaf) allows
  std::string refused;           // not empty: why these sizes are not valid
};

inline size_t lin_rows_of(u64 entries, u64 ell, bool rows_high) {
  const u64 rows = (entries + ell - 1) / ell;
  return (size_t)(rows_high ? rows : pow2_at_least(rows));
}
// A group of these members has 2^k main and 2^h mask rows: each member takes an aligned
// power-of-two block (`GroupLayout`, blocks/iop/relation.h). k is at least h and at least 1; h = 0
// if no member has mask rows.
inline void lin_group_bits(const std::vector<const OracleShape *> &members, unsigned &k, unsigned &h) {
  u64 main = 0, mask = 0;
  for (const OracleShape *m : members) main += m->main_rows ? pow2_at_least(m->main_rows) : 0, mask += m->mask_rows ? pow2_at_least(m->mask_rows) : 0;
  h = ceil_log2(mask), k = std::max(std::max(ceil_log2(main), h), 1u);
}

// table: must be the same for the one-time test, batches and link_sized() (not checked here): a
// batch's error bound presumes that a one-time test passed at `table.distances`. sized()
// (blocks/frame/files.h) passes one, made once by derive_table().
// Adds LIN's missing codes to `codes`.
inline LinSized lin_sized(const Setting &s, const LinLines &L, const TableSize &table, Codes &codes) {
  LinSized x;
  x.lines = L;
  std::string &why = x.refused;
  LinShape &S = x.S;
  const Scope shared = L.one_time ? Scope::ONCE : Scope::SHARED, request = L.one_time ? Scope::ONCE : Scope::REQUEST;
  const bool split = L.folded.log_n != L.omega.log_n || L.folded.log_ell != L.omega.log_ell;
  if (L.own.log_n != L.omega.log_n || L.own.log_ell != L.omega.log_ell) return why = "own: a request's group has Omega's main code", x;
  if (split && (L.folded.mask_log_n != L.omega.mask_log_n || L.folded.mask_log_ell != L.omega.mask_log_ell))
    return why = "omega: a split's mask message joins the folded row's group and Omega's: they have one mask code", x;

  // ---- the table and its columns ----
  const TableSize &T = x.table = table;
  if (T.code.dim(s) > T.code.n()) return why = "table: data and random coordinates exceed the code's length", x;
  const TableAnalysis A = analysis(s, T);
  if (!table_admitted(T, A)) return why = "table: the distances are not admitted", x;
  const Bound agree = L.one_time ? A.once.agree : A.later.agree;
  S.fold.columns = scope_positions(s, shared, agree);
  if (!S.fold.columns) return why = "table: no number of columns reaches the target", x;

  // ---- shapes ----
  // They do not depend on the numbers of positions.
  IndexShape &I = S.fold.index;
  I.code = Code{T.code.log_n, (size_t)T.code.ell, (size_t)T.code.r}, I.v = T.v, I.answers = s.samples;
  I.salt_bytes = s.salt_bytes, I.distances = T.distances;
  S.k_lo = L.k_lo, S.matrix_rows = L.matrix_rows;
  S.fold.rows_high = S.omega_rows_high = S.w_rows_high = L.rows_high;
  S.masks_ahead = L.masks_ahead, S.choice.masks_ride = L.masks_ride;
  if (index_fault(I)) return why = std::string("table: ") + index_fault(I), x;
  // A row's random coordinates (its random tail) hide a fixed number of columns in all. `spent`:
  // the columns that the one-time test opens before the first batch, plus columns_elsewhere. In a
  // leaf table LINK opens the same number as the one-time test (`columns_further` in link_sized(),
  // blocks/frame/link_frame.h).
  const u64 spent = scope_positions(s, Scope::ONCE, A.once.agree) + L.columns_elsewhere;
  x.batches = hiding_positions(I.code) > spent ? (hiding_positions(I.code) - spent) / S.fold.columns : 0;
  const u64 ell_u = u64(1) << L.folded.log_ell, ell_o = u64(1) << L.omega.log_ell, columns = u64(1) << S.k_col();
  auto message = [&](const char *name, size_t main_rows, size_t mask_rows) {
    OracleShape o;
    o.name = name, o.main_rows = main_rows, o.main_deg = E_DEGREE, o.mask_rows = mask_rows, o.answers = s.samples;
    return o;
  };
  S.fold.u = message("the folded row", lin_rows_of(S.fold.entries() + s.samples, ell_u, L.rows_high), 0);
  S.omega = message("Omega", lin_rows_of(L.matrix_rows + s.samples, ell_o, L.rows_high), 0);
  // Each of the `Setting::samples` answers to an out-of-domain sample is masked by a pad: one
  // random entry at the end of the message's last mask row or, without mask rows, of its last main
  // row (Send::pad(), blocks/iop/send.h). W gets a mask row for its pads in two cases. Its main
  // rows have fewer than `samples` free places. Or `samples` is above 1 and rows_high is false: the
  // end of the last main row then need not be W's last entries.
  S.w = message("W", lin_rows_of(columns, ell_o, L.rows_high), 0);
  if ((u64)S.w.main_rows * ell_o - columns < s.samples || (!L.rows_high && s.samples > 1)) S.w.mask_rows = 1;
  S.shared_split = message("the mask of the split between Omega and the folded row", 0, 1);
  x.input = OracleNotice{message("a request's masks", 0, L.input_mask_rows), 0};

  // Groups are named after their parameter-file lines, so that a rejection names the line to fix.
  auto line_name = [&](const char *group) { return std::string(L.one_time ? "once-" : "") + group; };
  unsigned k, h;
  lin_group_bits(split ? std::vector<const OracleShape *>{&S.fold.u, &S.shared_split} : std::vector<const OracleShape *>{&S.fold.u, &S.omega}, k, h);
  const GroupSize folded_shape = [&] { GroupSize g = shape_of(line_name("folded"), L.folded, k, true); return g.h = h, g; }();
  lin_group_bits({&S.omega, &S.shared_split}, k, h);
  const GroupSize omega_shape = [&] { GroupSize g = shape_of(line_name("omega"), L.omega, k, true); return g.h = h, g; }();
  lin_group_bits({&x.input.shape, &S.w, &S.omega}, k, h);
  const GroupSize own_shape = [&] { GroupSize g = shape_of(line_name("own"), L.own, k, true); return g.h = h, g; }();

  // ---- positions and random coordinates, all groups together ----
  // A row needs random coordinates for all positions at which it is opened; they raise its code's
  // rate, and the rate fixes the number of positions. The groups share oracle messages (Omega is
  // opened by the batch's test and by every request's, a split's mask message by two tests), so all
  // are solved together, upward from 0.
  const Bound margin(Real::two_to(-(long)s.target - 30));  // allows for the other error terms of a round that draws positions; the plan's comparison has the last word
  bool folded_serves = false, omega_serves = false, own_serves = false;  // `group_holds` result of each group's last sizing
  // main_draws, mask_draws: the positions, over all tests, that the main and mask codes' random
  // coordinates must cover. rows: the claims that the test batches.
  auto size = [&](const GroupSize &shape, Scope scope, u64 main_draws, u64 mask_draws, u64 rows, TestSizes &t, bool &serves) {
    GroupSize g = shape;
    g.main.r = tail_for(s, main_draws), g.mask.r = tail_for(s, mask_draws);
    if (g.main.dim(s) > g.main.n() || g.mask.dim(s) > g.mask.n()) return why = shape.name + ": message and random coordinates exceed the code's length", false;
    GroupLoad load;
    load.cube_bits = stage_cube_bits(g), load.messages = 2, load.batched_rows = rows;
    t.group = size_group(s, scope, g, load);
    /* A group need not be valid yet: with too few random coordinates its main code can have a lower
       rate than its mask code. The main code alone then gives a number of positions that is not too
       large; validity is checked after the loop. */
    serves = group_holds(s, scope, t.group, load);
    const u64 more = !serves ? positions_from_below(s, scope, g, load, margin) : scope_positions(s, scope, analysis(s, t.group).at.agree, margin);
    if (!more) return why = shape.name + ": no distance under its rule, or no number of positions, holds the group's rounds at the target", false;
    t.t = std::max(t.t, more);
    return true;
  };
  for (int turn = 0;; ++turn) {
    const u64 a = x.folded.t, b = x.omega.t, c = x.own.t;
    const u64 omega_draws = (split ? b : a) + L.requests * c;
    if (!size(folded_shape, shared, split ? a : omega_draws, split ? a + b : a, I.answers + S.fold.columns + 1 + 2 * s.samples, x.folded, folded_serves)) return x;
    if (split && !size(omega_shape, shared, omega_draws, a + b, 1 + 2 * s.samples, x.omega, omega_serves)) return x;
    if (!size(own_shape, request, omega_draws, c, 2 + 3 * s.samples, x.own, own_serves)) return x;
    if (x.folded.t == a && x.omega.t == b && x.own.t == c) break;
    if (turn == 32) return why = "positions and random coordinates do not settle", x;
  }
  for (const auto &[t, serves] : {std::make_pair(&x.folded, folded_serves), std::make_pair(&x.omega, omega_serves), std::make_pair(&x.own, own_serves)})
    if ((split || t != &x.omega) && !serves) return why = t->group.name + ": no distance under its rule holds the group's rounds at the target", x;
  if (!solve_test(s, shared, L.folded, x.folded, why)) return x;
  if (split && !solve_test(s, shared, L.omega, x.omega, why)) return x;
  if (!solve_test(s, request, L.own, x.own, why)) return x;

  // ---- codes and stages ----
  S.test_u = iopp_stages(x.folded.stages, codes);
  if (split) S.test_omega = iopp_stages(x.omega.stages, codes);
  S.test_own = iopp_stages(x.own.stages, codes);
  S.fold.u.code = S.test_u[0].code, S.omega.code = S.w.code = S.test_own[0].code;
  S.shared_mask_code = S.shared_split.mask_code = S.test_u[0].mask_code;
  S.own_mask_code = S.w.mask_code = x.input.shape.mask_code = S.test_own[0].mask_code;
  BLK_REQUIRE(S.shared_split_needed() == split && (!split || S.test_omega[0].code == S.omega.code), "lin_frame: Omega has not one code in all its groups");
  if (const char *f = lin_fault(codes, S)) return why = std::string("LIN: ") + f, x;
  for (const std::vector<IoppStage> *test : {&S.test_u, &S.test_omega, &S.test_own})
    if (!test->empty())
      if (const char *f = iopp_fault(codes, *test, s.samples)) return why = std::string("a proximity test of LIN: ") + f, x;
  return x;
}

// ---- LIN's share of the three chains' Plans ----
// blocks/iop/lin.h defines each prover message's oracle messages (lin_message_w() etc.).

inline LinSize lin_size_of(const Codes &codes, const LinSized &x) {
  const LinShape &S = x.S;
  // List sizes come from stages[0].g, from which plan_iopp() computes the test's round errors;
  // TestSizes::group is an equal copy.
  const GroupSize &around_omega = S.shared_split_needed() ? x.omega.stages[0].g : x.folded.stages[0].g;
  LinSize z{SampleSize{Send::cube(codes, S.w).bits, {x.own.stages[0].g}}, SampleSize{Send::cube(codes, S.fold.u).bits, {x.folded.stages[0].g}},
            SampleSize{Send::cube(codes, S.omega).bits, {around_omega, x.own.stages[0].g}}};
  z.instances = x.lines.requests, z.one_time = x.lines.one_time;
  z.k_lo = S.k_lo, z.rows_log = S.k_row();
  if (S.shared_split_needed()) z.beside_omega.push_back(SampleSize{Send::cube(codes, S.shared_split).bits, {x.folded.stages[0].g, x.omega.stages[0].g}});
  z.roots_w = (unsigned)lay_trees(codes, lin_message_w(S)).shapes.size();
  z.roots_u = (unsigned)lay_trees(codes, lin_message_u(S)).shapes.size();
  z.roots_omega = (unsigned)lay_trees(codes, lin_message_omega(S)).shapes.size();
  return z;
}
// On the request's first chain, before its seal (the digest that ends a chain: Rounds::seal(),
// blocks/chain/rounds.h).
inline void lin_plan_first(Plan &p, const Codes &codes, const LinSized &x) { plan_lin_first(p, "table", lin_size_of(codes, x)); }

// Before every plan_iopp(): ends the process unless `solved`, from which the plan computes a
// proximity test's errors, matches the shape's `stages` under `codes`.
inline void lin_require_sized(const Codes &codes, const std::vector<IoppStage> &stages, const std::vector<Stage> &solved) {
  const char *fault = iopp_sized_fault(codes, stages, solved);
  BLK_REQUIRE(!fault, fault ? fault : "");
}
// On the shared chain, from its first round to before its seal. table_tree: how the table's columns
// are hashed.
inline void lin_plan_shared(Plan &p, const Codes &codes, const LinSized &x, const TreeSize &table_tree) {
  const LinShape &S = x.S;
  const u64 nu = plan_lin_shared(p, "table", x.table, lin_size_of(codes, x), table_tree);
  if (!p.refused()) BLK_REQUIRE(nu == S.fold.columns, "lin_frame: the plan draws another number of columns");
  std::vector<TreeSize> trees = tree_sizes(codes, lin_message_u(S));
  const std::vector<TreeSize> with_omega = tree_sizes(codes, lin_message_omega(S));
  const u64 samples = p.set.samples;
  if (S.shared_split_needed()) {
    plan_split(p, "Omega against the folded row");
    trees.push_back(with_omega[1]);
    lin_require_sized(codes, S.test_u, x.folded.stages), lin_require_sized(codes, S.test_omega, x.omega.stages);
    plan_iopp(p, "folded row", x.folded.stages, trees, S.fold.index.answers + nu + 1 + 2 * samples, S.masks_ahead ? 0 : 1, S.choice.masks_ride);
    plan_iopp(p, "Omega", x.omega.stages, with_omega, 1 + 2 * samples, 1, S.choice.masks_ride);
  } else {
    trees.push_back(with_omega[0]);
    lin_require_sized(codes, S.test_u, x.folded.stages);
    plan_iopp(p, "folded row and Omega", x.folded.stages, trees, S.fold.index.answers + nu + 1 + 2 * samples, S.masks_ahead ? 0 : 1, S.choice.masks_ride);
  }
}
// On the request's second chain. input_trees: the trees that hold the request's mask message.
inline void lin_plan_second(Plan &p, const Codes &codes, const LinSized &x, const std::vector<TreeSize> &input_trees) {
  const LinShape &S = x.S;
  std::vector<TreeSize> trees = input_trees;
  for (const TreeSize &t : tree_sizes(codes, lin_message_w(S))) trees.push_back(t);
  TreeSize omega = tree_sizes(codes, lin_message_omega(S))[0];
  omega.from_another_chain = true;  // from the shared chain, opened by every request: `plan_open` (blocks/numbers/blocks.h) rejects the plan if `TreeSize::one_list` marks the tree as having a mask for one list of positions only
  trees.push_back(omega);
  lin_require_sized(codes, S.test_own, x.own.stages);
  plan_iopp(p, "a request's own test", x.own.stages, trees, 2 + 3 * (u64)p.set.samples, S.masks_ahead ? 0 : 1, S.choice.masks_ride);
}

}  // namespace blk
