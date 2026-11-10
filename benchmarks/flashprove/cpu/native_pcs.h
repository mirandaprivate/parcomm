#pragma once
// Standalone native LIN PCS: public evaluation, one-time proximity test included
// in every opening. No test-only plan or trusted-table bypass is enabled here.
#include "index_stream.h"
#include "frame/lin_frame.h"
#include "engine/engine_cpu.h"
#include <stdexcept>
namespace flashcpu {
struct NativePcs {
  Codes codes; LinSized sized; LinBindingShape binding;
  Digest configuration{}; Planned first, shared, second;
  NativePcs(unsigned row_bits, u64 matrix_rows, unsigned low_bits) {
    Setting set = setting_goldilocks(); set.target = 128;
    TableSize table; table.code.name = "CPU native PCS";
    table.v = row_bits; table.answers = set.samples;
    table.code.ell = (matrix_rows << low_bits) + 3 * set.samples;
    table.code.r = 1024;
    table.code.log_n = std::max(14u, ceil_log2(table.code.ell + table.code.r) + 1);
    if (!size_recorded_table(set, table, 1, row_bits + low_bits)) throw std::runtime_error("table sizing failed");
    LinLines lines; lines.k_lo = low_bits; lines.matrix_rows = matrix_rows;
    lines.one_time = true; lines.requests = 1; lines.input_mask_rows = 1;
    // The same main code for all three small oracle messages; bounded message rows.
    const unsigned ell = std::min(16u, std::max(9u, ceil_log2(table.code.ell)));
    for (GroupLine *g : {&lines.folded, &lines.omega, &lines.own}) {
      g->log_ell = ell; g->log_n = ell + 5;
      g->mask_log_ell = 8; g->mask_log_n = ell + 5;
      g->stages = {StageChoice{2, 1, ell + 5, ell + 5}, StageChoice{0, 0, ell + 7, ell + 7}};
    }
    sized = lin_sized(set, lines, table, codes);
    if (!sized.refused.empty()) throw std::runtime_error(sized.refused);
    binding.log_leaves = 0; binding.salt_bytes = set.salt_bytes;
    configuration.w[0] = 0x43505531; configuration.w[1] = row_bits;
    configuration.w[2] = matrix_rows; configuration.w[3] = matrix_rows >> 32;
    configuration.w[4] = low_bits; // all detailed shapes and plans also enter Binder
    Plan f(set), s(set), t(set);
    f.begin_scope(Scope::ONCE); s.begin_scope(Scope::ONCE); t.begin_scope(Scope::ONCE);
    plan_chain_start(f, "first", true); plan_send(f, "input masks", {SampleSize{Send::cube(codes, sized.input.shape).bits, {sized.own.stages[0].g}}}, (unsigned)tree_sizes(codes, {sized.input}).size()); lin_plan_first(f, codes, sized); plan_seal(f, "first");
    TreeSize tree; tree.name = "table"; tree.log_n = table.code.log_n; tree.leaf_bytes = u64(set.f_bytes) << table.v;
    plan_chain_start(s, "shared", false); lin_plan_shared(s, codes, sized, tree); plan_seal(s, "shared");
    plan_chain_start(t, "second", true); lin_plan_second(t, codes, sized, tree_sizes(codes, {sized.input}));
    first = f.finish(); shared = s.finish(); second = t.finish();
    for (const auto &plan : {first, shared, second})
      if (!plan->holds() || !plan->names_of_the_format()) throw std::runtime_error(plan->why_not());
  }
};
struct Query { std::vector<E> row_point, column_point; E value; };
struct NativeProof { std::vector<u8> first, path, shared, second; };
struct BoundQuery { LinPublic lin; Public<std::vector<E>> rows, columns; Public<E> value; Public<LinBindingShape> binding; };
inline BoundQuery bind_query(Binder &binder, const NativePcs &pcs, const IndexName &table, const Query &query) {
  BoundQuery out; binder.configuration(pcs.configuration);
  lin_bind_parameters(binder, pcs.codes, pcs.sized.S, out.lin);
  out.binding = binder.parameter("binding", pcs.binding);
  lin_bind_table(binder, table, out.lin);
  out.rows = binder.statement("row point", query.row_point);
  out.columns = binder.statement("column point", query.column_point);
  out.value = binder.statement("evaluation", query.value);
  return out;
}
inline LinClaim claim_of(const Query &query) {
  LinClaim claim; claim.g.push_back(Tensor::eq(query.row_point.data(), query.row_point.size()));
  claim.psi = query.column_point; claim.target = query.value; return claim;
}
inline Digest fresh_digest() { u8 bytes[32]; fresh32(bytes); Digest d; std::memcpy(d.w, bytes, 32); wipe(bytes, 32); return d; }
inline NativeProof prove(const NativePcs &pcs, IndexHolder &index, const Query &query, std::vector<E> w) {
  CoinKey seed; u8 bytes[32]; fresh32(bytes); seed = coin_key_from_bytes(bytes); wipe(bytes, 32);
  StreamCoins coins(seed, CoinContext{0, 1, 1, 0}); wipe(seed);
  EngineCpu engine; Writer first, shared_bytes, second_bytes, path;
  const IndexName table{index.commitment(), index.record()};
  std::unique_ptr<LinBegun> begun;
  BLK_REQUIRE(lin_begin(index, pcs.sized.S, begun), "native PCS opening allowance exhausted");
  Binder first_b(1, pcs.first); const auto bound = bind_query(first_b, pcs, table, query);
  ProverRounds first_ch(first_b, fresh_digest(), first);
  ProverContext first_p(bound.lin.codes, first_ch, engine, coins);
  std::vector<OracleDraft> input; input.push_back(lin_draft(first_p, pcs.sized.input.shape, false, pcs.sized.input.leaf_class, {}));
  const auto input_id = send_prove(first_p, std::move(input))[0]; auto claim = claim_of(query);
  claim.add_mask(first_p.st.masks(input_id).take(), E::zero());
  auto first_lin = lin_first_prove(first_p, bound.lin, std::move(claim), std::move(w));
  Sealed first_seal; BLK_REQUIRE(first_ch.seal(R_SEAL, first_seal), "first seal"); first_ch.finish();
  LinBinding binding(pcs.binding, pcs.configuration, coins); binding.put(0, first_seal.digest());
  Binder shared_b(LIN_LABEL_SHARED, pcs.shared); shared_b.configuration(pcs.configuration);
  const auto shared_bound = lin_bind_shared(shared_b, pcs.codes, pcs.sized.S, table, binding.root());
  ProverRounds shared_ch(shared_b, Digest{}, shared_bytes);
  ProverContext shared_p(shared_bound.codes, shared_ch, engine, coins); LinShared shared;
  lin_shared_prove(shared_p, shared_bound, TableTest::ONE_TIME, index, std::move(begun), shared);
  lin_shared_tests_prove(shared_p, shared_bound, shared);
  Sealed shared_seal; BLK_REQUIRE(shared_ch.seal(R_SEAL, shared_seal), "shared seal"); shared_ch.finish();
  shared_p.st.require_all_tested();
  Binder second_b(LIN_LABEL_SECOND, pcs.second); second_b.configuration(pcs.configuration);
  lin_bind_second(second_b, first_seal, shared_seal);
  ProverRounds second_ch(second_b, fresh_digest(), second_bytes);
  auto second_p = lin_go_on(std::move(first_p), second_ch);
  const auto second = lin_second_prove(second_p, bound.lin, first_lin, shared_p, shared);
  lin_second_tests_prove(second_p, bound.lin, first_lin, second); second_ch.finish(); second_p.st.require_all_tested();
  binding.write_path(path, 0); lin_release(shared_p, shared);
  return {first.bytes(), path.bytes(), shared_bytes.bytes(), second_bytes.bytes()};
}
inline bool verify(const NativePcs &pcs, const IndexKey &key, const Query &query, const NativeProof &proof) {
  if (query.row_point.size() != pcs.sized.S.k_row() || query.column_point.size() != pcs.sized.S.k_col()) return false;
  const IndexName table{key.commitment(), key.record()};
  Reader first(proof.first.data(), proof.first.size()); Binder first_b(1, pcs.first);
  auto bound = bind_query(first_b, pcs, table, query); VerifierSession session;
  VerifierRounds first_ch(first_b, first, session); VerifierContext first_v(bound.lin.codes, first_ch);
  std::vector<OracleId> input_ids; if (!send_verify(first_v, std::vector<OracleNotice>{pcs.sized.input}, input_ids)) return false;
  auto claim = claim_of(query); claim.add_mask(first_v.st.masks(input_ids[0]).take(), E::zero());
  LinFirst first_lin; if (!lin_first_verify(first_v, bound.lin, std::move(claim), first_lin)) return false;
  Sealed first_seal; if (!first_ch.seal(R_SEAL, first_seal) || !first_ch.finish()) return false;
  Reader path(proof.path.data(), proof.path.size()); LinBindingRoot root;
  if (!lin_binding_root(path, pcs.binding, pcs.configuration, first_seal.digest(), root) || !path.at_end()) return false;
  Reader shared_bytes(proof.shared.data(), proof.shared.size()); Binder shared_b(LIN_LABEL_SHARED, pcs.shared);
  shared_b.configuration(pcs.configuration);
  auto shared_bound = lin_bind_shared(shared_b, pcs.codes, pcs.sized.S, table, root);
  VerifierRounds shared_ch(shared_b, shared_bytes, session); VerifierContext shared_v(shared_bound.codes, shared_ch);
  LinShared shared;
  if (!lin_shared_verify(shared_v, shared_bound, TableTest::ONE_TIME, TableRead(key), shared)) return false;
  if (!lin_shared_tests_verify(shared_v, shared_bound, shared)) return false;
  Sealed shared_seal; if (!shared_ch.seal(R_SEAL, shared_seal) || !shared_ch.finish()) return false;
  shared_v.st.require_all_tested();
  Reader second_bytes(proof.second.data(), proof.second.size()); Binder second_b(LIN_LABEL_SECOND, pcs.second);
  second_b.configuration(pcs.configuration); lin_bind_second(second_b, first_seal, shared_seal);
  VerifierRounds second_ch(second_b, second_bytes, session); auto second_v = lin_go_on(std::move(first_v), second_ch);
  LinSecond second;
  if (!lin_second_verify(second_v, bound.lin, first_lin, shared_v, shared, second)) return false;
  if (!lin_second_tests_verify(second_v, bound.lin, first_lin, second) || !second_ch.finish()) return false;
  second_v.st.require_all_tested(); return true;
}
} // namespace flashcpu
