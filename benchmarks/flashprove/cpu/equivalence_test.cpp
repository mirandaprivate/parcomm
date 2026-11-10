// Synthetic differential tests only; never used as the model benchmark.
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#define BLK_TESTED_TABLE_FOR_A_TEST
#include "index_stream.h"
#include "iop/tests/lin_world.h"
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
using namespace blk;
using namespace flashcpu;
static void require(bool ok, const char *why) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); } }
int main() {
  coins_own_test();
  CoinKey process{}; process.w[0] = 123;
  CoinContext ctx{}; ctx.object = 17;
  unsigned tested = 0;
  for (unsigned log_n = 3; log_n <= 11; ++log_n) {
    Code code{log_n, (size_t(1) << (log_n - 1)) - 1, 1};
    std::vector<F> msg(code.ell), tail(code.r), native(code.n());
    for (size_t pos = 0; pos < msg.size(); ++pos) msg[pos] = from_u64(pos * 1927 + 43);
    tail[0] = from_u64(194);
    Encoder(code).encode(msg.data(), tail.data(), native.data());
    for (unsigned bits = 0; bits <= log_n; ++bits) {
      std::vector<F> part(size_t(1) << bits);
      for (u64 first = 0; first < code.n(); first += part.size()) {
        auto tw = interval_twiddles(log_n, bits, first);
        encode_interval(code, msg.data(), tail.data(), bits, first, tw, part.data());
        for (size_t pos = 0; pos < part.size(); ++pos) require(part[pos] == native[first + pos], "interval encoding");
      }
      ++tested;
    }
  }
  const LinCase test{"cpu stream", 2, 2, 3, 1, 1, 4, 3, 2, 8, 8, 8, false, 2, 1, 3, false};
  const auto world = lin_world(test, TableTest::ONE_TIME);
  LinMatrix matrix{world.S.matrix_rows, u64(1) << world.S.k_col(), {}};
  matrix.a.resize(matrix.rows * matrix.columns);
  for (size_t pos = 0; pos < matrix.a.size(); ++pos) matrix.a[pos] = from_u64(pos * 7 + 3);
  Table rows = lin_table_rows(world.S, matrix);
  auto source = [&](u64 row, F *out, size_t n) { std::copy_n(rows.a.data() + row * rows.len, n, out); };
  for (unsigned threads : {1u, 2u, 4u}) for (unsigned part : {1u, 3u, world.S.fold.index.code.log_n}) {
    StreamCoins coins(process, ctx);
    IndexCpu native(world.S.fold.index, rows, coins, world.configuration);
    char path[] = "/tmp/flashprove-counter-XXXXXX";
    // mkstemp reserves a unique name; the keeper itself exclusively creates it.
    int fd = mkstemp(path); require(fd >= 0, "counter name"); close(fd); unlink(path);
    IndexStream cpu(world.S.fold.index, source, process, ctx, world.configuration, part, part / 2, threads, path);
    require(cpu.commitment() == native.commitment(), "commitment equality");
    require(record_bytes(cpu.record()) == record_bytes(native.record()), "record equality");
    std::vector<E> gamma(world.S.fold.index.v, E::from(from_u64(17)));
    std::vector<E> m1, m2, t1, t2;
    native.fold(gamma.data(), m1, t1, nullptr); cpu.fold(gamma.data(), m2, t2, nullptr);
    require(m1 == m2 && t1 == t2, "fold equality");
    auto positions = PositionList::from_draws({0, 1, 3, 3, world.S.fold.index.code.n() - 1}, world.S.fold.index.code.log_n);
    Writer left, right; std::vector<F> c1, c2;
    native.open(positions, left, c1); cpu.open(positions, right, c2);
    require(left.bytes() == right.bytes() && c1 == c2, "opening serialization equality");
    auto key = cpu.key();
    auto reference = lin_session(test, world, native, native.record(), matrix, 12, 13);
    auto actual = lin_session(test, world, cpu, cpu.record(), matrix, 12, 13);
    require(actual.served && reference.served, "LIN served");
    for (size_t request = 0; request < actual.views.size(); ++request) {
      auto &view = actual.views[request]; const auto &ref = reference.views[request];
      require(view.first == ref.first && view.shared == ref.shared && view.second == ref.second && view.path == ref.path,
              "whole LIN proof bytes");
      require(lin_verify_view(world, key, 12, request, view), "upstream verifier");
      auto broken = view; broken.shared[broken.shared.size() / 2] ^= 1;
      require(!lin_verify_view(world, key, 12, request, broken), "tampered proof rejected");
    }
    unlink(path); ++tested;
  }
  std::printf("PASS: %u differential configurations; encoding, root, record, fold, opening and LIN proof bytes agree; upstream verification and tamper rejection pass\n", tested);
}
