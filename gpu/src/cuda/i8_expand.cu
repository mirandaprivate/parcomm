#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {
struct ScalarField {
  std::uint32_t limbs[8];
};

static_assert(sizeof(ScalarField) == 32, "unexpected BLS12-381 scalar-field layout");

__global__ void expand_i8_kernel(
  const std::int8_t* input,
  std::size_t offset,
  std::size_t count,
  ScalarField* output,
  bool negative)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) { return; }

  const int value = static_cast<int>(input[offset + index]);
  const std::uint32_t magnitude = negative ? (value < 0 ? static_cast<std::uint32_t>(-value) : 0U)
                                           : (value > 0 ? static_cast<std::uint32_t>(value) : 0U);
  output[index].limbs[0] = magnitude;
#pragma unroll
  for (int limb = 1; limb < 8; ++limb) { output[index].limbs[limb] = 0U; }
}
} // namespace

extern "C" int smartpc_expand_i8_device(
  const std::int8_t* input,
  std::size_t offset,
  std::size_t count,
  void* output,
  bool negative)
{
  if (input == nullptr || output == nullptr || count == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  expand_i8_kernel<<<blocks, threads>>>(
    input, offset, count, static_cast<ScalarField*>(output), negative);
  cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) { error = cudaDeviceSynchronize(); }
  return static_cast<int>(error);
}
