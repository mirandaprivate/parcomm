#include "native_pcs.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
using namespace flashcpu;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
static std::vector<u8> read_file(const std::string &name) {
  std::ifstream f(name, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() < 0 || f.tellg() > (1LL << 30)) throw std::runtime_error("invalid proof/statement file");
  std::vector<u8> bytes((size_t)f.tellg()); f.seekg(0); f.read((char *)bytes.data(), bytes.size());
  if (!f) throw std::runtime_error("truncated file"); return bytes;
}
static int verify_saved(const std::string &prefix) {
  auto bytes = read_file(prefix + ".statement"); Reader in(bytes.data(), bytes.size());
  u64 row_bits, matrix_rows;
  if (!in.get_u64(row_bits) || !in.get_u64(matrix_rows) || row_bits > 24 || matrix_rows > (1u << 24) || !matrix_rows)
    throw std::runtime_error("invalid statement dimensions");
  auto start = Clock::now(); NativePcs pcs(row_bits, matrix_rows, 0); const double setup = seconds(start);
  start = Clock::now(); IndexKey key; key.table = pcs.sized.S.fold.index; key.made_under = pcs.configuration;
  if (!in.get(key.root)) throw std::runtime_error("missing root");
  key.rec.y.resize(key.table.rows() * key.table.answers);
  for (E &value : key.rec.y) if (!in.get(value)) throw std::runtime_error("invalid record");
  Query query; query.row_point.resize(pcs.sized.S.k_row()); query.column_point.resize(pcs.sized.S.k_col());
  for (auto *point : {&query.row_point, &query.column_point}) for (E &value : *point)
    if (!in.get(value)) throw std::runtime_error("invalid point");
  if (!in.get(query.value) || !in.at_end()) throw std::runtime_error("invalid evaluation");
  auto encoded = read_file(prefix + ".proof"); Reader proof_in(encoded.data(), encoded.size()); NativeProof proof;
  for (auto *part : {&proof.first, &proof.path, &proof.shared, &proof.second}) {
    u64 n; if (!proof_in.get_u64(n) || n > encoded.size()) throw std::runtime_error("invalid proof framing");
    const u8 *data = proof_in.take(n); if (!data) throw std::runtime_error("truncated proof part"); part->assign(data, data+n);
  }
  if (!proof_in.at_end()) throw std::runtime_error("trailing proof bytes");
  const bool ok = verify(pcs, key, query, proof); const double elapsed = seconds(start);
  std::cout << std::setprecision(12) << "{\"setup_seconds\":" << setup << ",\"verify_decode_io_seconds\":" << elapsed
            << ",\"verified\":" << (ok ? "true" : "false") << "}" << std::endl;
  return ok ? 0 : 2;
}
int main(int argc, char **argv) {
 try {
  if (argc == 3 && std::string(argv[1]) == "verify") return verify_saved(argv[2]);
  if (argc < 6) throw std::runtime_error("usage: model_bench INPUT WIDTH THREADS OUTPUT_PREFIX MATRIX_LOG [plan]");
  const std::string input = argv[1], prefix = argv[4];
  const unsigned width = std::stoul(argv[2]), threads = std::stoul(argv[3]), matrix_log = std::stoul(argv[5]);
  if ((width != 1 && width != 2) || !threads || threads > 64 || matrix_log < 8 || matrix_log > 24) throw std::runtime_error("invalid arguments");
  int fd = ::open(input.c_str(), O_RDONLY); struct stat stat{};
  if (fd < 0 || fstat(fd, &stat) || stat.st_size <= 0 || stat.st_size % width) throw std::runtime_error("input file invalid");
  const u64 entries = stat.st_size / width;
  // Leave room for native pads and random tail in a power-of-two coefficient block.
  const u64 matrix_rows = (u64(1) << matrix_log) - 1027;
  if (matrix_rows < 1 || matrix_log < 11) throw std::runtime_error("matrix_log must be at least 11");
  const unsigned row_bits = ceil_log2((entries + matrix_rows - 1) / matrix_rows);
  auto start = Clock::now(); NativePcs pcs(row_bits, matrix_rows, 0);
  const double setup = seconds(start); const auto &shape = pcs.sized.S;
  std::cout << "setup_seconds=" << setup << " entries=" << entries << " table_rows=" << shape.fold.index.rows()
            << " message_length=" << shape.fold.index.code.ell << " code_length=" << shape.fold.index.code.n()
            << " table_queries=" << shape.fold.columns << " threads=" << threads << std::endl;
  if (argc > 6 && std::string(argv[6]) == "plan") { close(fd); return 0; }
  void *mapped = mmap(nullptr, stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) throw std::runtime_error("input mmap failed");
  auto cell = [&](u64 pos) -> F {
    if (pos >= entries) return F::zero();
    const int64_t value = width == 1 ? ((const int8_t *)mapped)[pos] : ((const int16_t *)mapped)[pos];
    return value < 0 ? -from_u64(u64(-value)) : from_u64(u64(value));
  };
  auto source = [&](u64 table_row, F *out, size_t n) {
    const u64 first = table_row * matrix_rows;
    for (size_t pos = 0; pos < n; ++pos) out[pos] = cell(first + pos);
  };
  coins_own_test(); CoinKey seed; u8 fresh[32]; fresh32(fresh); seed = coin_key_from_bytes(fresh); wipe(fresh, 32);
  const unsigned part = std::min(18u, shape.fold.index.code.log_n), keep = std::min(12u, part);
  start = Clock::now();
  IndexStream index(shape.fold.index, source, seed, CoinContext{0, 1, 2, 0}, pcs.configuration, part, keep, threads, prefix + ".counter", true);
  wipe(seed); const auto key = index.key(); const double commit = seconds(start);
  std::cout << "commit_seconds=" << commit << std::endl;
  Query query; query.row_point.resize(shape.k_row()); query.column_point.resize(shape.k_col());
  fresh32(fresh); seed = coin_key_from_bytes(fresh); wipe(fresh, 32);
  StreamCoins query_coins(seed, CoinContext{0, 1, 3, 0}); wipe(seed);
  std::vector<F> coordinates(3 * (shape.k_row() + shape.k_col())); query_coins.fill(coordinates.data(), coordinates.size());
  size_t offset = 0;
  for (auto *point : {&query.row_point, &query.column_point}) for (E &value : *point) for (unsigned axis = 0; axis < 3; ++axis) value.c[axis] = coordinates[offset++];
  start = Clock::now();
  std::vector<E> row_weights(u64(1) << shape.k_row()); eq_table(query.row_point.data(), shape.k_row(), row_weights.data());
  std::vector<E> w(u64(1) << shape.k_col(), E::zero());
  #pragma omp parallel for num_threads(threads) schedule(static)
  for (u64 col = 0; col < w.size(); ++col) {
    E sum = E::zero(); for (u64 row = 0; row < matrix_rows; ++row) sum += row_weights[row] * cell(col * matrix_rows + row);
    w[col] = sum;
  }
  query.value = mle_eval(w.data(), w.size(), query.column_point.data(), shape.k_col());
  const double evaluation = seconds(start);
  auto proof = prove(pcs, index, query, std::move(w));
  std::vector<u8> serialized;
  for (const auto *part_bytes : {&proof.first, &proof.path, &proof.shared, &proof.second}) {
    u64 n = part_bytes->size(); for (unsigned byte = 0; byte < 8; ++byte) serialized.push_back(n >> (8 * byte));
    serialized.insert(serialized.end(), part_bytes->begin(), part_bytes->end());
  }
  const double opening = seconds(start);
  std::cout << "open_seconds=" << opening << " evaluation_seconds=" << evaluation << " proof_bytes=" << serialized.size() << std::endl;
  Writer statement; statement.put_u64(row_bits); statement.put_u64(matrix_rows); statement.put(key.root);
  for (const E &value : key.rec.y) statement.put(value);
  for (const auto *point : {&query.row_point, &query.column_point}) for (const E &value : *point) statement.put(value);
  statement.put(query.value);
  { std::ofstream file(prefix + ".statement", std::ios::binary); const auto &data = statement.bytes(); file.write((char *)data.data(), data.size()); if (!file) throw std::runtime_error("statement write failed"); }
  { std::ofstream file(prefix + ".proof", std::ios::binary); file.write((char *)serialized.data(), serialized.size()); if (!file) throw std::runtime_error("proof write failed"); }
  start = Clock::now(); const bool accepted = verify(pcs, key, query, proof); const double verification = seconds(start);
  struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
  std::ofstream result(prefix + ".json");
  result << std::setprecision(12) << "{\n\"scope\":\"CPU streaming native LIN PCS; one-time proximity included in opening; public evaluation\",\n"
         << "\"entries\":" << entries << ",\n\"input_width\":" << width << ",\n\"runs\":1,\n\"prover_threads\":" << threads
         << ",\n\"setup_seconds\":" << setup << ",\n\"commit_seconds\":" << commit << ",\n\"open_seconds\":" << opening
         << ",\n\"evaluation_seconds_in_open\":" << evaluation << ",\n\"prover_seconds\":" << commit+opening
         << ",\n\"verify_seconds\":" << verification << ",\n\"proof_bytes\":" << serialized.size()
         << ",\n\"peak_rss_kib\":" << usage.ru_maxrss << ",\n\"verified\":" << (accepted ? "true" : "false") << "\n}\n";
  munmap(mapped, stat.st_size); close(fd);
  std::cout << "verified=" << accepted << " verify_seconds=" << verification << std::endl;
  return accepted ? 0 : 2;
 } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
