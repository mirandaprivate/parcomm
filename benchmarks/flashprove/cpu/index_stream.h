#pragma once
// CPU storage adapter for FlashProve's native IndexHolder. Protocol code is upstream.
#include <functional>
#include <map>
#include <array>
#include <omp.h>
#include <fcntl.h>
#include <unistd.h>
#include "coins/coins.h"
#include "iop/index_cpu.h"

namespace flashcpu {
using namespace blk;

// Evaluate one aligned interval of the native RS code. High coefficient blocks
// are folded at z^width; the low butterflies use the original global positions.
inline std::vector<F> interval_twiddles(unsigned log_n, unsigned bits, u64 first) {
  const size_t width = size_t(1) << bits;
  std::vector<F> tw(width, F::zero());
  if (!bits) return tw;
  F *z = tw.data() + width / 2;
  z[0] = F{gl64_point(log_n, first)};
  for (unsigned step = 0; step + 1 < bits; ++step) {
    const F factor{gl64_pow(GL64_ROOT_32, u64(1) << (30 - step))};
    for (size_t pos = 0; pos < (size_t(1) << step); ++pos)
      z[pos + (size_t(1) << step)] = z[pos] * factor;
  }
  for (unsigned layer = 1; layer < bits; ++layer) {
    size_t at = width >> (layer + 1);
    for (size_t pos = 0; pos < at; ++pos) tw[at + pos] = tw[2 * at + 2 * pos] * tw[2 * at + 2 * pos];
  }
  return tw;
}
inline void encode_interval(const Code &code, const F *message, const F *tail,
                            unsigned bits, u64 first, const std::vector<F> &tw, F *out) {
  BLK_REQUIRE(bits <= 32 && bits <= code.log_n && !(first & ((u64(1) << bits) - 1)), "CPU interval alignment");
  const size_t width = size_t(1) << bits;
  const F high{gl64_pow(gl64_point(code.log_n, first), u64(1) << bits)};
  const size_t groups = (code.dim() + width - 1) / width;
  for (size_t pos = 0; pos < width; ++pos) {
    F acc = F::zero();
    for (size_t group = groups; group-- > 0;) {
      const size_t index = group * width + pos;
      const F value = index < code.r ? tail[index] : index < code.dim() ? message[index - code.r] : F::zero();
      acc = acc * high + value;
    }
    out[pos] = acc;
  }
  for (unsigned layer = bits; layer-- > 0;) {
    const size_t stride = size_t(1) << layer;
    for (size_t group = 0; group < (width >> (layer + 1)); ++group) {
      F *lo = out + 2 * stride * group, *hi = lo + stride;
      for (size_t pos = 0; pos < stride; ++pos) {
        const F left = lo[pos], right = hi[pos] * tw[(width >> (layer + 1)) + group];
        lo[pos] = left + right; hi[pos] = left - right;
      }
    }
  }
}

class IndexStream final : public IndexHolder {
 public:
  // Source fills exactly plain() field entries. It must support simultaneous reads
  // and remain immutable through all openings. No model-sized field buffer is kept.
  using Source = std::function<void(u64, F *, size_t)>;
  IndexStream(IndexShape shape, Source source, const CoinKey &process, const CoinContext &context,
              Digest configuration, unsigned part_bits, unsigned keep_bits, unsigned threads,
              const std::string &counter_path, bool verbose = false)
      : s_(shape), source_(std::move(source)), key_(context_key(process, context)), configuration_(configuration),
        part_(part_bits), keep_(keep_bits), threads_(threads), tree_(index_tree(s_, configuration)), verbose_(verbose) {
    BLK_REQUIRE(!index_fault(s_) && tree_.valid() && threads_ && threads_ <= 64 && keep_ <= part_ && part_ <= s_.code.log_n && part_ <= 30,
                "CPU keeper configuration");
    BLK_REQUIRE(s_.rows() < UINT32_MAX - COIN_FIRST_STREAM - 1, "CPU keeper coin stream limit");
    counter_ = ::open(counter_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    BLK_REQUIRE(counter_ >= 0, "CPU keeper needs a fresh durable counter file");
    upper_.resize(size_t(2) << (s_.code.log_n - keep_));
    const u64 width = u64(1) << part_;
    for (u64 first = 0; first < s_.code.n(); first += width) {
      auto subtree = make_part(first, nullptr, nullptr);
      const u64 at = s_.code.n() >> keep_, count = width >> keep_;
      for (u64 pos = 0; pos < count; ++pos) upper_[at + (first >> keep_) + pos] = subtree[count + pos];
      if (verbose_) std::fprintf(stderr, "stream_commit_part=%llu/%llu\n", (unsigned long long)(first/width+1), (unsigned long long)(s_.code.n()/width));
    }
    const Tag tag = tree_.tree().vertex_tag();
    for (u64 count = s_.code.n() >> keep_; count > 1; count >>= 1) {
      #pragma omp parallel for num_threads(threads_) schedule(static)
      for (u64 pos = count / 2; pos < count; ++pos) upper_[pos] = vertex(tag, upper_[2 * pos], upper_[2 * pos + 1]);
    }
    const auto seeds = record_seeds(s_, name());
    record_.y.resize(s_.rows() * s_.answers);
    #pragma omp parallel for num_threads(threads_) schedule(static)
    for (u64 row = 0; row < s_.rows(); ++row) {
      std::vector<F> message(s_.code.ell), tail(s_.code.r);
      read_row(row, message.data(), tail.data());
      for (unsigned sample = 0; sample < s_.answers; ++sample)
        record_.y[row * s_.answers + sample] = record_value(s_, seeds[sample], sample, message.data());
    }
  }
  ~IndexStream() override { wipe(key_); if (counter_ >= 0) ::close(counter_); }
  IndexStream(const IndexStream &) = delete;
  IndexStream &operator=(const IndexStream &) = delete;
  const IndexShape &shape() const override { return s_; }
  Digest configuration() const override { return configuration_; }
  std::vector<u8> commitment() const override { return digest_bytes(upper_[1]); }
  std::vector<u8> name() const override { return index_name(upper_[1], s_, configuration_); }
  IndexRecord record() override { return record_; }
  IndexKey key() const { return IndexKey(s_, configuration_, upper_[1], record_); }
  bool begin(u64 columns) override {
    if (columns > hiding_positions(s_.code) - spent_) return false;
    const u64 next = spent_ + columns;
    BLK_REQUIRE(::pwrite(counter_, &next, sizeof next, 0) == sizeof next && ::fsync(counter_) == 0, "CPU keeper counter persistence");
    spent_ = next; return true;
  }
  void fold(const E *gamma, std::vector<E> &message, std::vector<E> &tail, BlockSums *blocks) override {
    std::vector<E> weight(s_.rows()); eq_table(gamma, s_.v, weight.data());
    const size_t len = s_.code.ell + s_.code.r;
    std::vector<std::vector<E>> local(threads_, std::vector<E>(len, E::zero()));
    #pragma omp parallel num_threads(threads_)
    {
      auto &sum = local[omp_get_thread_num()];
      std::vector<F> msg(s_.code.ell), rnd(s_.code.r);
      #pragma omp for schedule(static)
      for (u64 row = 0; row < s_.rows(); ++row) {
        read_row(row, msg.data(), rnd.data());
        for (size_t pos = 0; pos < s_.code.ell; ++pos) sum[pos] += weight[row] * msg[pos];
        for (size_t pos = 0; pos < s_.code.r; ++pos) sum[s_.code.ell + pos] += weight[row] * rnd[pos];
      }
    }
    message.assign(s_.code.ell, E::zero()); tail.assign(s_.code.r, E::zero());
    #pragma omp parallel for num_threads(threads_) schedule(static)
    for (size_t pos = 0; pos < len; ++pos) {
      E sum = E::zero(); for (unsigned worker = 0; worker < threads_; ++worker) sum += local[worker][pos];
      if (pos < s_.code.ell) message[pos] = sum; else tail[pos - s_.code.ell] = sum;
    }
    if (blocks) block_sums(message, *blocks);
  }
  void open(const PositionList &at, Writer &file, std::vector<F> &columns) override {
    BLK_REQUIRE(!at.empty() && at.log_n() == s_.code.log_n, "CPU keeper opening positions");
    Opening result; result.main_width = s_.rows();
    result.columns.resize(at.size() * s_.rows()); result.salts.resize(at.size() * s_.salt_bytes);
    std::map<u64, std::vector<Digest>> parts;
    for (u64 pos : at.all()) parts.emplace(pos >> part_, std::vector<Digest>{});
    for (auto &entry : parts) {
      entry.second = make_part(entry.first << part_, &at, &result);
      const u64 width = u64(1) << part_, count = width >> keep_, base = s_.code.n() >> keep_;
      for (u64 pos = 0; pos < count; ++pos)
        BLK_REQUIRE(entry.second[count + pos] == upper_[base + (entry.first << (part_ - keep_)) + pos], "CPU keeper recomputed subtree differs");
      if (verbose_) std::fprintf(stderr, "stream_open_part=%llu\n", (unsigned long long)entry.first);
    }
    struct Fetch {
      struct Value {};
      const IndexStream &self; const std::map<u64, std::vector<Digest>> &parts; std::vector<Digest> &given;
      Value leaf(size_t) { return {}; }
      bool given_node(unsigned depth, u64 place) {
        const unsigned level = self.s_.code.log_n - depth;
        if (level >= self.keep_) given.push_back(self.upper_[(u64(1) << depth) + place]);
        else {
          const unsigned below = self.part_ - level;
          given.push_back(parts.at(place >> below)[(u64(1) << below) + (place & ((u64(1) << below) - 1))]);
        }
        return true;
      }
      bool given_(unsigned depth, u64 place, Value &) { return given_node(depth, place); }
      Value node(unsigned, u64, const Value &, const Value &) { return {}; }
    };
    // Adapter member names follow tree_walk's contract.
    struct Walk : Fetch {
      bool given(unsigned depth, u64 place, Fetch::Value &v) { return this->given_(depth, place, v); }
    } walk{{*this, parts, result.given}};
    Fetch::Value root; tree_walk(tree_.tree(), at, walk, root);
    BLK_REQUIRE(check_opening(upper_[1], tree_, at, result), "CPU keeper opening self-check");
    write_opening(file, tree_, at, result); columns = std::move(result.columns);
  }
 private:
  void read_row(u64 row, F *message, F *tail) const {
    source_(row, message, s_.plain());
    for (size_t pos = 0; pos < s_.code.r; ++pos) tail[pos] = coin_element(key_, COIN_FIRST_STREAM, row * s_.code.r + pos);
    for (size_t pos = 0; pos < s_.pads(); ++pos) message[s_.plain() + pos] = coin_element(key_, COIN_FIRST_STREAM + 1 + row, pos);
  }
  void salt(u64 pos, u8 *out) const {
    const u32 stream = COIN_FIRST_STREAM + 1 + s_.rows();
    for (size_t offset = 0; offset < s_.salt_bytes; offset += 8) {
      const u64 value = coin_word(key_, stream, (pos * s_.salt_bytes + offset) / 8);
      for (unsigned byte = 0; byte < 8; ++byte) out[offset + byte] = value >> (8 * byte);
    }
  }
  std::vector<Digest> make_part(u64 first, const PositionList *queries, Opening *opening) const {
    const size_t width = size_t(1) << part_;
    const auto tw = interval_twiddles(s_.code.log_n, part_, first);
    const LeafHash leaf(tree_);
    std::vector<Hasher> hash; hash.reserve(width);
    for (size_t pos = 0; pos < width; ++pos) hash.emplace_back(leaf.tag, tree_index(s_.code.log_n, first + pos));
    const size_t slab = std::min<u64>(threads_, s_.rows());
    std::vector<F> words(slab * width);
    for (u64 start = 0; start < s_.rows(); start += slab) {
      const size_t rows = std::min<u64>(slab, s_.rows() - start);
      #pragma omp parallel for num_threads(threads_) schedule(static)
      for (size_t row = 0; row < rows; ++row) {
        std::vector<F> msg(s_.code.ell), tail(s_.code.r);
        read_row(start + row, msg.data(), tail.data());
        encode_interval(s_.code, msg.data(), tail.data(), part_, first, tw, words.data() + row * width);
      }
      #pragma omp parallel for num_threads(threads_) schedule(static)
      for (size_t pos = 0; pos < width; ++pos) {
        std::array<u8, 64 * 8> packed;
        for (size_t row = 0; row < rows; ++row) {
          const u64 value = words[row * width + pos].v;
          for (unsigned byte = 0; byte < 8; ++byte) packed[row * 8 + byte] = value >> (8 * byte);
        }
        hash[pos].push(packed.data(), rows * 8);
      }
      if (queries) for (size_t qi = 0; qi < queries->size(); ++qi) {
        const u64 pos = (*queries)[qi];
        if (pos >= first && pos < first + width)
          for (size_t row = 0; row < rows; ++row) opening->columns[qi * s_.rows() + start + row] = words[row * width + pos - first];
      }
    }
    std::vector<Digest> nodes(2 * width);
    #pragma omp parallel for num_threads(threads_) schedule(static)
    for (size_t pos = 0; pos < width; ++pos) {
      std::array<u8, 64> bytes; salt(first + pos, bytes.data());
      nodes[width + pos] = leaf_end(hash[pos], leaf.tag, s_.rows(), bytes.data(), s_.salt_bytes);
    }
    if (queries) for (size_t qi = 0; qi < queries->size(); ++qi)
      if ((*queries)[qi] >= first && (*queries)[qi] < first + width) salt((*queries)[qi], opening->salts.data() + qi * s_.salt_bytes);
    const Tag tag = tree_.tree().vertex_tag();
    for (size_t count = width; count > 1; count >>= 1) {
      #pragma omp parallel for num_threads(threads_) schedule(static)
      for (size_t pos = count / 2; pos < count; ++pos) nodes[pos] = vertex(tag, nodes[2 * pos], nodes[2 * pos + 1]);
    }
    return nodes;
  }
  IndexShape s_; Source source_; CoinKey key_; Digest configuration_;
  unsigned part_, keep_, threads_; OracleTreeShape tree_; std::vector<Digest> upper_;
  IndexRecord record_; u64 spent_ = 0; int counter_ = -1; bool verbose_;
};
} // namespace flashcpu
