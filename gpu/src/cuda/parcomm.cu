// CUDA kernels for the witness-proportional stages of the ParComm prover.
//
// Two conventions are shared with the host side:
//   * The packed witness is a row-major `rows x cols` matrix of signed bytes,
//     so logical index `f` sits at row `f / cols` and column `f % cols`.
//   * Field elements use the same in-memory representation as arkworks
//     `ark_bls12_381::Fr`: four little-endian 64-bit limbs holding the
//     Montgomery form of the value. Buffers therefore move between host and
//     device without any conversion.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// MSM scalar expansion
// ---------------------------------------------------------------------------

// ICICLE consumes MSM scalars as plain little-endian integers, not Montgomery
// forms, so magnitudes are written directly into the low limb.
struct IcicleScalar {
  std::uint32_t limbs[8];
};

static_assert(sizeof(IcicleScalar) == 32, "unexpected BLS12-381 scalar layout");

// Reads a row-major byte matrix and writes one column-major magnitude scalar
// per entry, which is the layout an ICICLE batched MSM expects. Doing the
// transpose here keeps the host from ever materializing a second copy of the
// witness.
__global__ void expand_i8_colmajor_kernel(
  const std::int8_t* input,
  std::size_t cols,
  std::size_t column_start,
  std::size_t shard_cols,
  std::size_t row_start,
  std::size_t shard_rows,
  IcicleScalar* output,
  bool negative)
{
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= shard_rows * shard_cols) { return; }

  // Consecutive threads walk one row so the byte loads coalesce.
  const std::size_t local_row = index / shard_cols;
  const std::size_t local_col = index - local_row * shard_cols;
  const int value = static_cast<int>(input[(row_start + local_row) * cols + column_start + local_col]);
  const std::uint32_t magnitude = negative ? (value < 0 ? static_cast<std::uint32_t>(-value) : 0U)
                                           : (value > 0 ? static_cast<std::uint32_t>(value) : 0U);

  IcicleScalar* slot = &output[local_col * shard_rows + local_row];
  slot->limbs[0] = magnitude;
#pragma unroll
  for (int limb = 1; limb < 8; ++limb) { slot->limbs[limb] = 0U; }
}

// ---------------------------------------------------------------------------
// BLS12-381 scalar field, Montgomery form
// ---------------------------------------------------------------------------

struct Fr {
  std::uint64_t limbs[4];
};

static_assert(sizeof(Fr) == 32, "unexpected BLS12-381 Fr layout");

__constant__ std::uint64_t kModulus[4] = {
  0xffffffff00000001ULL,
  0x53bda402fffe5bfeULL,
  0x3339d80809a1d805ULL,
  0x73eda753299d7d48ULL,
};

// -r^{-1} mod 2^64, the CIOS reduction constant.
static constexpr std::uint64_t kModulusInverse = 0xfffffffeffffffffULL;

__device__ __forceinline__ bool is_less_than_modulus(const std::uint64_t* value)
{
#pragma unroll
  for (int limb = 3; limb >= 0; --limb) {
    if (value[limb] != kModulus[limb]) { return value[limb] < kModulus[limb]; }
  }
  return false;
}

__device__ __forceinline__ void subtract_modulus(std::uint64_t* value)
{
  std::uint64_t borrow = 0;
#pragma unroll
  for (int limb = 0; limb < 4; ++limb) {
    const std::uint64_t subtrahend = kModulus[limb];
    const std::uint64_t intermediate = value[limb] - subtrahend;
    const std::uint64_t next_borrow =
      (value[limb] < subtrahend) || (borrow == 1 && intermediate == 0);
    value[limb] = intermediate - borrow;
    borrow = next_borrow;
  }
}

__device__ __forceinline__ Fr fr_add(const Fr& left, const Fr& right)
{
  Fr result;
  std::uint64_t carry = 0;
#pragma unroll
  for (int limb = 0; limb < 4; ++limb) {
    const std::uint64_t sum = left.limbs[limb] + right.limbs[limb];
    const std::uint64_t carried = sum + carry;
    result.limbs[limb] = carried;
    carry = (sum < left.limbs[limb]) || (carry == 1 && carried == 0);
  }
  // The modulus is under 2^255, so at most one conditional subtraction runs.
  if (carry != 0 || !is_less_than_modulus(result.limbs)) { subtract_modulus(result.limbs); }
  return result;
}

__device__ __forceinline__ Fr fr_negate(const Fr& value)
{
  bool zero = true;
#pragma unroll
  for (int limb = 0; limb < 4; ++limb) { zero = zero && (value.limbs[limb] == 0); }
  if (zero) { return value; }

  Fr result;
  std::uint64_t borrow = 0;
#pragma unroll
  for (int limb = 0; limb < 4; ++limb) {
    const std::uint64_t minuend = kModulus[limb];
    const std::uint64_t intermediate = minuend - value.limbs[limb];
    const std::uint64_t next_borrow =
      (minuend < value.limbs[limb]) || (borrow == 1 && intermediate == 0);
    result.limbs[limb] = intermediate - borrow;
    borrow = next_borrow;
  }
  return result;
}

// CIOS Montgomery multiplication. Both inputs and the result are Montgomery
// forms, so this matches `ark_ff` multiplication bit for bit.
__device__ __forceinline__ Fr fr_mul(const Fr& left, const Fr& right)
{
  std::uint64_t accumulator[5] = {0, 0, 0, 0, 0};

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // accumulator += left * right.limbs[i]
    std::uint64_t carry = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const std::uint64_t low = left.limbs[j] * right.limbs[i];
      const std::uint64_t high = __umul64hi(left.limbs[j], right.limbs[i]);
      std::uint64_t sum = accumulator[j] + low;
      std::uint64_t next_carry = high + (sum < low);
      sum += carry;
      next_carry += (sum < carry);
      accumulator[j] = sum;
      carry = next_carry;
    }
    accumulator[4] = accumulator[4] + carry;
    const std::uint64_t overflow = (accumulator[4] < carry) ? 1ULL : 0ULL;

    // accumulator = (accumulator + modulus * m) / 2^64
    const std::uint64_t m = accumulator[0] * kModulusInverse;
    std::uint64_t reduction_carry = 0;
    {
      const std::uint64_t low = kModulus[0] * m;
      const std::uint64_t high = __umul64hi(kModulus[0], m);
      const std::uint64_t sum = accumulator[0] + low;
      reduction_carry = high + (sum < low);
    }
#pragma unroll
    for (int j = 1; j < 4; ++j) {
      const std::uint64_t low = kModulus[j] * m;
      const std::uint64_t high = __umul64hi(kModulus[j], m);
      std::uint64_t sum = accumulator[j] + low;
      std::uint64_t next_carry = high + (sum < low);
      sum += reduction_carry;
      next_carry += (sum < reduction_carry);
      accumulator[j - 1] = sum;
      reduction_carry = next_carry;
    }
    const std::uint64_t tail = accumulator[4] + reduction_carry;
    accumulator[3] = tail;
    accumulator[4] = overflow + (tail < reduction_carry);
  }

  Fr result;
#pragma unroll
  for (int limb = 0; limb < 4; ++limb) { result.limbs[limb] = accumulator[limb]; }
  if (accumulator[4] != 0 || !is_less_than_modulus(result.limbs)) { subtract_modulus(result.limbs); }
  return result;
}

// ---------------------------------------------------------------------------
// Streaming projections
// ---------------------------------------------------------------------------

// `out[col] = sum_row value[row][col] * weight[row]`, accumulating over a band
// of rows. `magnitudes[k]` holds the Montgomery form of the integer `k`, so a
// signed byte turns into a field element with one table lookup.
__global__ void project_left_kernel(
  const std::int8_t* input,
  std::size_t cols,
  std::size_t row_start,
  std::size_t row_end,
  const Fr* weights,
  const Fr* magnitudes,
  Fr* output)
{
  const std::size_t col = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (col >= cols) { return; }

  Fr accumulator = {{0, 0, 0, 0}};
  for (std::size_t row = row_start; row < row_end; ++row) {
    const int value = static_cast<int>(input[row * cols + col]);
    if (value == 0) { continue; }
    const int magnitude = value > 0 ? value : -value;
    const Fr term = fr_mul(weights[row], magnitudes[magnitude]);
    accumulator = fr_add(accumulator, value > 0 ? term : fr_negate(term));
  }
  output[col] = fr_add(output[col], accumulator);
}

// `out[row] = sum_col value[row][col] * weight[col]`.
//
// One block owns one row and its threads stride across the row, so the byte
// loads stay coalesced; the per-thread partial sums are then reduced in shared
// memory.
constexpr int kProjectRightThreads = 256;

__global__ void project_right_kernel(
  const std::int8_t* input,
  std::size_t cols,
  std::size_t row_start,
  const Fr* weights,
  const Fr* magnitudes,
  Fr* output)
{
  __shared__ Fr partials[kProjectRightThreads];

  const std::size_t row = row_start + blockIdx.x;
  const std::int8_t* row_values = input + row * cols;

  Fr accumulator = {{0, 0, 0, 0}};
  for (std::size_t col = threadIdx.x; col < cols; col += kProjectRightThreads) {
    const int value = static_cast<int>(row_values[col]);
    if (value == 0) { continue; }
    const int magnitude = value > 0 ? value : -value;
    const Fr term = fr_mul(weights[col], magnitudes[magnitude]);
    accumulator = fr_add(accumulator, value > 0 ? term : fr_negate(term));
  }
  partials[threadIdx.x] = accumulator;
  __syncthreads();

  for (int stride = kProjectRightThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < static_cast<unsigned>(stride)) {
      partials[threadIdx.x] = fr_add(partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { output[row] = partials[0]; }
}

// ---------------------------------------------------------------------------
// Packed EO cross terms
// ---------------------------------------------------------------------------

// The packing stride equals the warp width, so one warp covers all lanes of a
// single outer index and its byte loads land on one contiguous 32-byte chunk.
constexpr int kCrossTermsStride = 32;
constexpr int kCrossTermsWarpsPerBlock = 8;
constexpr int kCrossTermsThreads = kCrossTermsStride * kCrossTermsWarpsPerBlock;

// `partial[outer][lane] = sum_{i < inner} value[base + lane + (outer*inner + i)*32]
//                         * inner_weights[i]`.
//
// The bucket weight is a tensor, so the host splits it into an inner half that
// this kernel applies and an outer half it applies to the returned partials.
// That keeps the full-length weight vector, which reaches 2^27 entries for the
// largest bucket, out of device memory entirely.
__global__ void cross_terms_lane_partial_kernel(
  const std::int8_t* input,
  std::ptrdiff_t base,
  std::size_t window_low,
  std::size_t window_high,
  std::size_t inner,
  std::size_t outer_start,
  std::size_t outer_count,
  const Fr* inner_weights,
  const Fr* magnitudes,
  Fr* partial)
{
  const unsigned lane = threadIdx.x & (kCrossTermsStride - 1);
  const unsigned warp = threadIdx.x / kCrossTermsStride;
  const std::size_t local_outer =
    blockIdx.x * static_cast<std::size_t>(kCrossTermsWarpsPerBlock) + warp;
  if (local_outer >= outer_count) { return; }

  const std::size_t outer = outer_start + local_outer;
  std::size_t offset = outer * inner * kCrossTermsStride + lane;

  // Offsets are counted from the bucket's start; this device stores only
  // `[window_low, window_high)` of them, and `base` places that window inside
  // its own row band. Everything outside is either another device's rows or the
  // bucket's implicit zero suffix, and contributes nothing either way.
  Fr accumulator = {{0, 0, 0, 0}};
  for (std::size_t i = 0; i < inner; ++i, offset += kCrossTermsStride) {
    if (offset >= window_high) { break; }
    if (offset < window_low) { continue; }
    const int value = static_cast<int>(*(input + base + offset));
    if (value == 0) { continue; }
    const int magnitude = value > 0 ? value : -value;
    const Fr term = fr_mul(inner_weights[i], magnitudes[magnitude]);
    accumulator = fr_add(accumulator, value > 0 ? term : fr_negate(term));
  }
  partial[local_outer * kCrossTermsStride + lane] = accumulator;
}

// ---------------------------------------------------------------------------
// Packed EO reduced vector
// ---------------------------------------------------------------------------

// `reduced[j] = sum_{t < 32} value[base + j*32 + t] * inverse_powers[t]`.
//
// This is the only point where the reduced vector is created, and it is created
// in device memory. Every later round rewrites it in place, so the witness-sized
// intermediate never crosses the bus.
__global__ void pack_reduced_kernel(
  const std::int8_t* input,
  std::ptrdiff_t base,
  std::size_t first,
  std::size_t limit,
  std::size_t count,
  const Fr* inverse_powers,
  const Fr* magnitudes,
  Fr* reduced)
{
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= count) { return; }

  // `first` is the packed row this device's band starts at, so the output is
  // written from zero while the input is read from the band's own rows.
  const std::size_t row_offset = (first + index) * kCrossTermsStride;
  const std::int8_t* row = input + base + row_offset;
  Fr accumulator = {{0, 0, 0, 0}};
  for (int t = 0; t < kCrossTermsStride; ++t) {
    // The last packed row may run past the bucket's stored entries.
    if (row_offset + t >= limit) { break; }
    const int value = static_cast<int>(row[t]);
    if (value == 0) { continue; }
    const int magnitude = value > 0 ? value : -value;
    const Fr term = fr_mul(inverse_powers[t], magnitudes[magnitude]);
    accumulator = fr_add(accumulator, value > 0 ? term : fr_negate(term));
  }
  reduced[index] = accumulator;
}

constexpr int kEoThreads = 256;

// Even and odd tensor projections of the reduced vector for one EO round:
// `even[h] = sum_{l < inner} values[2*(h*inner + l)]     * inner_weights[l]`
// `odd[h]  = sum_{l < inner} values[2*(h*inner + l) + 1] * inner_weights[l]`
//
// As with the cross terms the round tensor is split, so only its inner half
// reaches the device and the host applies the outer half to these partials.
__global__ void eo_project_kernel(
  const Fr* values,
  std::size_t len,
  std::size_t inner,
  const Fr* inner_weights,
  Fr* even_partial,
  Fr* odd_partial)
{
  __shared__ Fr even_partials[kEoThreads];
  __shared__ Fr odd_partials[kEoThreads];

  // An odd length leaves the last even slot without a partner, so the odd sum
  // simply runs one pair shorter.
  const std::size_t even_pairs = (len + 1) / 2;
  const std::size_t odd_pairs = len / 2;

  const std::size_t outer = blockIdx.x;
  Fr even = {{0, 0, 0, 0}};
  Fr odd = {{0, 0, 0, 0}};
  for (std::size_t l = threadIdx.x; l < inner; l += kEoThreads) {
    const std::size_t pair = outer * inner + l;
    if (pair >= even_pairs) { break; }
    const Fr weight = inner_weights[l];
    even = fr_add(even, fr_mul(values[2 * pair], weight));
    if (pair < odd_pairs) {
      odd = fr_add(odd, fr_mul(values[2 * pair + 1], weight));
    }
  }

  even_partials[threadIdx.x] = even;
  odd_partials[threadIdx.x] = odd;
  __syncthreads();
  for (int stride = kEoThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < static_cast<unsigned>(stride)) {
      even_partials[threadIdx.x] =
        fr_add(even_partials[threadIdx.x], even_partials[threadIdx.x + stride]);
      odd_partials[threadIdx.x] =
        fr_add(odd_partials[threadIdx.x], odd_partials[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    even_partial[outer] = even_partials[0];
    odd_partial[outer] = odd_partials[0];
  }
}

// `output[i] = input[2i] + challenge * input[2i+1]`, halving the vector.
//
// Input and output are distinct buffers because a thread's output slot is also
// another thread's input slot, and CUDA has no grid-wide barrier to order the
// two. The host ping-pongs between a full-length buffer and a half-length one,
// so the extra memory is half the reduced vector rather than all of it.
__global__ void eo_fold_kernel(
  const Fr* input,
  Fr* output,
  std::size_t old_len,
  Fr challenge)
{
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t next_len = (old_len + 1) / 2;
  if (index >= next_len) { return; }

  const std::size_t left = 2 * index;
  const Fr value = input[left];
  output[index] =
    left + 1 < old_len ? fr_add(value, fr_mul(challenge, input[left + 1])) : value;
}

// Multiplies every field element by a magnitude table entry. Used by the host
// tests that check the device arithmetic against arkworks.
__global__ void mul_check_kernel(const Fr* left, const Fr* right, Fr* output, std::size_t count)
{
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= count) { return; }
  output[index] = fr_mul(left[index], right[index]);
}

__global__ void add_check_kernel(const Fr* left, const Fr* right, Fr* output, std::size_t count)
{
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= count) { return; }
  output[index] = fr_add(left[index], right[index]);
}

int launch_status()
{
  cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) { error = cudaDeviceSynchronize(); }
  return static_cast<int>(error);
}

} // namespace

// Moves a folded result back over the source it was folded from, which is what
// lets one scratch buffer per device stand in for a per-vector second half.
extern "C" int parcomm_copy_device(void* destination, const void* source, std::size_t bytes)
{
  if (bytes == 0) { return 0; }
  if (destination == nullptr || source == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(
    cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice));
}

extern "C" int parcomm_expand_i8_colmajor(
  const std::int8_t* input,
  std::size_t cols,
  std::size_t column_start,
  std::size_t shard_cols,
  std::size_t row_start,
  std::size_t shard_rows,
  void* output,
  bool negative)
{
  if (input == nullptr || output == nullptr || shard_cols == 0 || shard_rows == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t count = shard_rows * shard_cols;
  constexpr int threads = 256;
  const std::size_t blocks = (count + threads - 1) / threads;
  expand_i8_colmajor_kernel<<<static_cast<int>(blocks), threads>>>(
    input, cols, column_start, shard_cols, row_start, shard_rows,
    static_cast<IcicleScalar*>(output), negative);
  return launch_status();
}

extern "C" int parcomm_project_left(
  const std::int8_t* input,
  std::size_t cols,
  std::size_t row_start,
  std::size_t row_end,
  const void* weights,
  const void* magnitudes,
  void* output)
{
  if (input == nullptr || weights == nullptr || magnitudes == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr int threads = 256;
  const std::size_t blocks = (cols + threads - 1) / threads;
  project_left_kernel<<<static_cast<int>(blocks), threads>>>(
    input, cols, row_start, row_end, static_cast<const Fr*>(weights),
    static_cast<const Fr*>(magnitudes), static_cast<Fr*>(output));
  return launch_status();
}

extern "C" int parcomm_project_right(
  const std::int8_t* input,
  std::size_t cols,
  std::size_t row_start,
  std::size_t row_end,
  const void* weights,
  const void* magnitudes,
  void* output)
{
  if (input == nullptr || weights == nullptr || magnitudes == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t rows = row_end - row_start;
  if (rows == 0) { return static_cast<int>(cudaSuccess); }
  project_right_kernel<<<static_cast<int>(rows), kProjectRightThreads>>>(
    input, cols, row_start, static_cast<const Fr*>(weights),
    static_cast<const Fr*>(magnitudes), static_cast<Fr*>(output));
  return launch_status();
}

extern "C" int parcomm_cross_terms_lane_partials(
  const std::int8_t* input,
  std::ptrdiff_t base,
  std::size_t window_low,
  std::size_t window_high,
  std::size_t inner,
  std::size_t outer_start,
  std::size_t outer_count,
  const void* inner_weights,
  const void* magnitudes,
  void* partial)
{
  if (input == nullptr || inner_weights == nullptr || magnitudes == nullptr ||
      partial == nullptr || inner == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (outer_count == 0) { return static_cast<int>(cudaSuccess); }
  const std::size_t blocks =
    (outer_count + kCrossTermsWarpsPerBlock - 1) / kCrossTermsWarpsPerBlock;
  cross_terms_lane_partial_kernel<<<static_cast<int>(blocks), kCrossTermsThreads>>>(
    input, base, window_low, window_high, inner, outer_start, outer_count,
    static_cast<const Fr*>(inner_weights), static_cast<const Fr*>(magnitudes),
    static_cast<Fr*>(partial));
  return launch_status();
}

extern "C" int parcomm_pack_reduced(
  const std::int8_t* input,
  std::ptrdiff_t base,
  std::size_t first,
  std::size_t limit,
  std::size_t count,
  const void* inverse_powers,
  const void* magnitudes,
  void* reduced)
{
  if (input == nullptr || inverse_powers == nullptr || magnitudes == nullptr ||
      reduced == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (count == 0) { return static_cast<int>(cudaSuccess); }
  constexpr int threads = 256;
  const std::size_t blocks = (count + threads - 1) / threads;
  pack_reduced_kernel<<<static_cast<int>(blocks), threads>>>(
    input, base, first, limit, count, static_cast<const Fr*>(inverse_powers),
    static_cast<const Fr*>(magnitudes), static_cast<Fr*>(reduced));
  return launch_status();
}

extern "C" int parcomm_eo_project(
  const void* values,
  std::size_t len,
  std::size_t inner,
  std::size_t outer,
  const void* inner_weights,
  void* even_partial,
  void* odd_partial)
{
  if (values == nullptr || inner_weights == nullptr || even_partial == nullptr ||
      odd_partial == nullptr || inner == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (outer == 0) { return static_cast<int>(cudaSuccess); }
  eo_project_kernel<<<static_cast<int>(outer), kEoThreads>>>(
    static_cast<const Fr*>(values), len, inner,
    static_cast<const Fr*>(inner_weights), static_cast<Fr*>(even_partial),
    static_cast<Fr*>(odd_partial));
  return launch_status();
}

extern "C" int parcomm_eo_fold(
  const void* input,
  void* output,
  std::size_t old_len,
  const void* challenge)
{
  if (input == nullptr || output == nullptr || challenge == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t next_len = (old_len + 1) / 2;
  if (next_len == 0) { return static_cast<int>(cudaSuccess); }
  constexpr int threads = 256;
  const std::size_t blocks = (next_len + threads - 1) / threads;
  eo_fold_kernel<<<static_cast<int>(blocks), threads>>>(
    static_cast<const Fr*>(input), static_cast<Fr*>(output), old_len,
    *static_cast<const Fr*>(challenge));
  return launch_status();
}

extern "C" int parcomm_fr_mul_check(
  const void* left, const void* right, void* output, std::size_t count)
{
  constexpr int threads = 256;
  const std::size_t blocks = (count + threads - 1) / threads;
  mul_check_kernel<<<static_cast<int>(blocks), threads>>>(
    static_cast<const Fr*>(left), static_cast<const Fr*>(right),
    static_cast<Fr*>(output), count);
  return launch_status();
}

extern "C" int parcomm_fr_add_check(
  const void* left, const void* right, void* output, std::size_t count)
{
  constexpr int threads = 256;
  const std::size_t blocks = (count + threads - 1) / threads;
  add_check_kernel<<<static_cast<int>(blocks), threads>>>(
    static_cast<const Fr*>(left), static_cast<const Fr*>(right),
    static_cast<Fr*>(output), count);
  return launch_status();
}
