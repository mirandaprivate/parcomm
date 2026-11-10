#include "native_pcs.h"
#include <iostream>
using namespace flashcpu;
int main() {
 try {
  NativePcs pcs(3, 100, 3);
  const auto &s = pcs.sized.S;
  std::cout << "plans hold; table queries=" << s.fold.columns << std::endl;
  CoinKey seed{}; seed.w[0] = 194; CoinContext ctx{};
  auto cell = [](u64 row, u64 col) { return from_u64((row * 317 + col * 997) % 256); };
  auto source = [&](u64 table_row, F *out, size_t len) {
    std::fill_n(out, len, F::zero());
    for (u64 row = 0; row < s.matrix_rows; ++row)
      for (u64 pos = 0; pos < (u64(1) << s.k_lo); ++pos) out[(row << s.k_lo) + pos] = cell(row, (table_row << s.k_lo) + pos);
  };
  char path[] = "/tmp/flashprove-native-XXXXXX"; int fd = mkstemp(path); close(fd); unlink(path);
  IndexStream index(s.fold.index, source, seed, ctx, pcs.configuration, 10, 6, 4, path);
  Query q; q.row_point.assign(s.k_row(), E::from(from_u64(17))); q.column_point.assign(s.k_col(), E::from(from_u64(23)));
  std::vector<E> weights(u64(1) << s.k_row()); eq_table(q.row_point.data(), s.k_row(), weights.data());
  std::vector<E> w(u64(1) << s.k_col(), E::zero());
  for (u64 row = 0; row < s.matrix_rows; ++row) for (size_t col = 0; col < w.size(); ++col) w[col] += weights[row] * cell(row, col);
  q.value = mle_eval(w.data(), w.size(), q.column_point.data(), s.k_col());
  auto proof = prove(pcs, index, q, w); auto key = index.key();
  if (!verify(pcs, key, q, proof)) throw std::runtime_error("honest proof rejected");
  Query bad = q; bad.value += E::one();
  if (verify(pcs, key, bad, proof)) throw std::runtime_error("false value accepted");
  auto corrupt = proof; corrupt.shared[corrupt.shared.size() / 2] ^= 1;
  if (verify(pcs, key, q, corrupt)) throw std::runtime_error("corrupt proof accepted");
  unlink(path);
  std::cout << "PASS: native one-time LIN PCS, real 128-bit per-round plans, honest verification and two negative checks; bytes="
            << proof.first.size()+proof.path.size()+proof.shared.size()+proof.second.size() << std::endl;
 } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
