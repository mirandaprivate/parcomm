// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A linear claim (`Row`, relation.h) states that a linear form of the messages encoded in oracle
// messages equals the public `Row::target`. row_value() evaluates that form on the prover's tables.
#pragma once
#include <vector>

#include "base.h"
#include "engine/table.h"
#include "field/field.h"
#include "iop/relation.h"
#include "iop/tensor.h"

namespace blk {

// sum_j v[j] * t.entry(j) on the host. Expands v in full, so it is for mask tables and tests.
inline E table_dot(const Table &t, const Tensor &v) {
  BLK_REQUIRE((u64(1) << v.vars()) == t.entries(), "table_dot: the vector does not fit the table");
  std::vector<E> w(size_t(1) << v.vars(), E::zero());
  v.add_to(w.data());
  E sum = E::zero();
  for (u64 j = 0; j < t.entries(); j++) sum += w[j] * t.entry(j);
  return sum;
}

// Returns the value of `row`'s linear form; does not read row.target. pr is a ProverContext
// (blocks/iop/context.h); pr.behind[o] has oracle message o's tables. Main tables can be long and
// stay in pr.engine (blocks/engine/engine.h): on one, each term, unit or form costs an
// Engine::evaluate() and each run a copy of the table. Oracle ids and the indices of units and
// runs are not range-checked here.
template <class ProverCtx> inline E row_value(ProverCtx &pr, const Row &row) {
  E sum = E::zero();
  for (const Term &t : row.terms)
    sum += t.part == Part::MAIN ? pr.engine.evaluate(pr.behind[t.oracle].main, t.v) : table_dot(pr.behind[t.oracle].mask, t.v);
  for (const Unit &u : row.units) {
    if (u.part == Part::MASK) {
      sum += u.coeff * pr.behind[u.oracle].mask.entry(u.index);
    } else {
      Tensor pick;  // the vector with u.coeff at u.index and 0 elsewhere
      pick.scalar = u.coeff;
      pick.insert_choice(0, pr.st.table_bits(u.oracle, Part::MAIN), u.index);
      sum += pr.engine.evaluate(pr.behind[u.oracle].main, pick);
    }
  }
  for (const Run &r : row.runs) {
    if (r.part == Part::MASK) {
      for (size_t j = 0; j < r.size(); j++) sum += r.at(j) * pr.behind[r.oracle].mask.entry(r.first + r.step * j);
    } else {
      const Table main = pr.engine.get(pr.behind[r.oracle].main);  // a copy of the whole table: for short tables only
      for (size_t j = 0; j < r.size(); j++) sum += r.at(j) * main.entry(r.first + r.step * j);
    }
  }
  for (const MainForm &f : row.forms) sum += pr.engine.evaluate(pr.behind[f.oracle].main, f.c);
  return sum;
}

}  // namespace blk
