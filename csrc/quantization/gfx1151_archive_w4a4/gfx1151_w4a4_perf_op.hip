// SPDX-License-Identifier: Apache-2.0
// Isolated performance-v2 successor to the sealed gfx1151 W4A4 qualification.
// Baseline semantics derive from ciru-ai/vllm commit
// 40027ba1de16212858b92b05190ac89fe4f51d33. Performance candidates add
// explicit forced routes, group-major coalesced wires, cross-row DOT reuse,
// and multi-wave WMMA blocks without modifying the qualified baseline source.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <tuple>

namespace gfx1151_w4a4_perf {

using bf16_t = __hip_bfloat16;
using int2x32 = int __attribute__((ext_vector_type(2)));
using int8x32 = int __attribute__((ext_vector_type(8)));
using float8x32 = float __attribute__((ext_vector_type(8)));

constexpr int kGroupSize = 32;
constexpr uint32_t kUint4B8ToS4Mask = 0x88888888u;
constexpr int64_t kRouteDotBaseline = 1;
constexpr int64_t kRouteWmmaBaselinePadded = 2;
constexpr int64_t kRouteDotMTile4 = 3;
constexpr int64_t kRouteDotMTile8 = 4;
constexpr int64_t kRouteWmma16x64 = 5;
constexpr int64_t kRouteWmma32x64 = 6;
constexpr int64_t kRouteWmma32x32 = 7;

__host__ __device__ constexpr int c_fragment_row(const int lane, const int reg) {
    return 2 * reg + ((lane & 31) >> 4);
}

__host__ __device__ constexpr int c_fragment_col(const int lane) {
    return lane & 15;
}

__device__ __forceinline__ int signed_dot8_i4(const int a,
                                              const int b,
                                              const int accumulator) {
    int result;
    asm volatile("v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[1,1,0]"
                 : "=v"(result)
                 : "v"(a), "v"(b), "v"(accumulator));
    return result;
}

__device__ __forceinline__ void quantize_one_s4(const bf16_t* input,
                                                 const int row,
                                                 const int group,
                                                 const int K,
                                                 uint32_t& word,
                                                 float& scale) {
    const int lane = threadIdx.x & 31;
    const int k = group * kGroupSize + lane;
    const float value = __bfloat162float(input[static_cast<long>(row) * K + k]);
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, 32));
    }
    maximum = __shfl(maximum, 0, 32);
    scale = maximum > 0.0f ? maximum * (1.0f / 7.0f) : 0.0f;
    const double normalized = maximum > 0.0f
                                  ? static_cast<double>(value) * 7.0 /
                                        static_cast<double>(maximum)
                                  : 0.0;
    int q = maximum > 0.0f ? __double2int_rn(normalized) : 0;
    q = q < -7 ? -7 : (q > 7 ? 7 : q);
    const int source_base = (lane >> 3) << 3;
    word = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int code = __shfl(q, source_base + i, 32);
        word |= static_cast<uint32_t>(code & 0xF) << (4 * i);
    }
}

// Qualified row-major activation layout [M,K/8], scales [M,K/32].
__global__ void quantize_pack_s4_g32_rowmajor(const bf16_t* __restrict__ input,
                                              uint32_t* __restrict__ packed,
                                              float* __restrict__ scales,
                                              const int rows,
                                              const int K) {
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) {
        return;
    }
    uint32_t word = 0;
    float scale = 0.0f;
    quantize_one_s4(input, row, group, K, word, scale);
    if ((lane & 7) == 0) {
        const int words_per_row = K / 8;
        packed[static_cast<long>(row) * words_per_row + group * 4 + lane / 8] =
            word;
    }
    if (lane == 0) {
        scales[static_cast<long>(row) * (K / kGroupSize) + group] = scale;
    }
}

// Candidate group-major activation layout [K/32,4,M], scales [K/32,M].
__global__ void quantize_pack_s4_g32_groupmajor(const bf16_t* __restrict__ input,
                                                uint32_t* __restrict__ packed,
                                                float* __restrict__ scales,
                                                const int rows,
                                                const int K) {
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) {
        return;
    }
    uint32_t word = 0;
    float scale = 0.0f;
    quantize_one_s4(input, row, group, K, word, scale);
    if ((lane & 7) == 0) {
        packed[static_cast<long>(group * 4 + lane / 8) * rows + row] = word;
    }
    if (lane == 0) {
        scales[static_cast<long>(group) * rows + row] = scale;
    }
}

// Sealed row-major DOT control: one thread per output.
__global__ void s4s4_g32_dot_baseline(const uint32_t* __restrict__ a_packed,
                                      const uint32_t* __restrict__ b_uint4b8,
                                      const float* __restrict__ a_scale,
                                      const bf16_t* __restrict__ b_scale,
                                      bf16_t* __restrict__ output,
                                      const int M,
                                      const int N,
                                      const int K) {
    const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y;
    if (output_row >= M || output_col >= N) {
        return;
    }
    const int groups = K / kGroupSize;
    const int words_per_row = K / 8;
    float accumulator = 0.0f;
    for (int group = 0; group < groups; ++group) {
        int integer_sum = 0;
        const long a_base = static_cast<long>(output_row) * words_per_row + group * 4;
        const long b_base = static_cast<long>(output_col) * words_per_row + group * 4;
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int a = static_cast<int>(a_packed[a_base + word]);
            const int b = static_cast<int>(
                b_uint4b8[b_base + word] ^ kUint4B8ToS4Mask);
            integer_sum = signed_dot8_i4(a, b, integer_sum);
        }
        const float activation_scale =
            a_scale[static_cast<long>(output_row) * groups + group];
        const float weight_scale = __bfloat162float(
            b_scale[static_cast<long>(output_col) * groups + group]);
        accumulator += static_cast<float>(integer_sum) * activation_scale * weight_scale;
    }
    output[static_cast<long>(output_row) * N + output_col] =
        __float2bfloat16(accumulator);
}

// Sealed one-wave WMMA control, now forceable for padded M<9 crossover evidence.
__global__ void s4s4_g32_wmma_baseline(const uint32_t* __restrict__ a_packed,
                                       const uint32_t* __restrict__ b_uint4b8,
                                       const float* __restrict__ a_scale,
                                       const bf16_t* __restrict__ b_scale,
                                       bf16_t* __restrict__ output,
                                       const int M,
                                       const int N,
                                       const int K) {
    const int lane = threadIdx.x & 31;
    const int source_lane = lane & 15;
    const int tile_m = blockIdx.y * 16;
    const int tile_n = blockIdx.x * 16;
    const int source_row = tile_m + source_lane;
    const int source_col = tile_n + source_lane;
    const int words_per_row = K / 8;
    const int groups = K / kGroupSize;
    float8x32 accumulator;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        accumulator[j] = 0.0f;
    }
    for (int group = 0; group < groups; ++group) {
        int8x32 group_accumulator;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            group_accumulator[j] = 0;
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int2x32 a_fragment = {0, 0};
            int2x32 b_fragment = {0, 0};
            const int word = group * 4 + half * 2;
            if (source_row < M) {
                const long base = static_cast<long>(source_row) * words_per_row + word;
                a_fragment[0] = static_cast<int>(a_packed[base]);
                a_fragment[1] = static_cast<int>(a_packed[base + 1]);
            }
            if (source_col < N) {
                const long base = static_cast<long>(source_col) * words_per_row + word;
                b_fragment[0] = static_cast<int>(
                    b_uint4b8[base] ^ kUint4B8ToS4Mask);
                b_fragment[1] = static_cast<int>(
                    b_uint4b8[base + 1] ^ kUint4B8ToS4Mask);
            }
            group_accumulator = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment, group_accumulator, false);
        }
        const int output_col = tile_n + c_fragment_col(lane);
        const float weight_scale =
            output_col < N
                ? __bfloat162float(
                      b_scale[static_cast<long>(output_col) * groups + group])
                : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int output_row = tile_m + c_fragment_row(lane, j);
            if (output_row < M && output_col < N) {
                const float activation_scale =
                    a_scale[static_cast<long>(output_row) * groups + group];
                accumulator[j] += static_cast<float>(group_accumulator[j]) *
                                  activation_scale * weight_scale;
            }
        }
    }
    const int output_col = tile_n + c_fragment_col(lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int output_row = tile_m + c_fragment_row(lane, j);
        if (output_row < M && output_col < N) {
            output[static_cast<long>(output_row) * N + output_col] =
                __float2bfloat16(accumulator[j]);
        }
    }
}

// Coalesced group-major DOT candidate. Each output thread reuses one weight
// stream and weight-scale stream across kMtile activation rows.
template <int kMtile>
__global__ void s4s4_g32_dot_mtile(const uint32_t* __restrict__ a_packed,
                                   const uint32_t* __restrict__ b_uint4b8,
                                   const float* __restrict__ a_scale,
                                   const bf16_t* __restrict__ b_scale,
                                   bf16_t* __restrict__ output,
                                   const int M,
                                   const int N,
                                   const int K) {
    const int lane = threadIdx.x & 31;
    const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_base = blockIdx.y * kMtile;
    const bool valid_col = output_col < N;
    const int groups = K / kGroupSize;
    float accumulator[kMtile];
#pragma unroll
    for (int row_offset = 0; row_offset < kMtile; ++row_offset) {
        accumulator[row_offset] = 0.0f;
    }
    for (int group = 0; group < groups; ++group) {
        int integer_sum[kMtile];
#pragma unroll
        for (int row_offset = 0; row_offset < kMtile; ++row_offset) {
            integer_sum[row_offset] = 0;
        }
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int b = valid_col
                              ? static_cast<int>(
                                    b_uint4b8[(static_cast<long>(group * 4 + word) * N + output_col)] ^
                                    kUint4B8ToS4Mask)
                              : 0;
#pragma unroll
            for (int row_offset = 0; row_offset < kMtile; ++row_offset) {
                const int output_row = row_base + row_offset;
                int a = 0;
                if (lane == 0 && output_row < M) {
                    a = static_cast<int>(
                        a_packed[(static_cast<long>(group * 4 + word) * M + output_row)]);
                }
                a = __shfl(a, 0, 32);
                integer_sum[row_offset] =
                    signed_dot8_i4(a, b, integer_sum[row_offset]);
            }
        }
        const float weight_scale =
            valid_col
                ? __bfloat162float(
                      b_scale[static_cast<long>(group) * N + output_col])
                : 0.0f;
#pragma unroll
        for (int row_offset = 0; row_offset < kMtile; ++row_offset) {
            const int output_row = row_base + row_offset;
            float activation_scale = 0.0f;
            if (lane == 0 && output_row < M) {
                activation_scale =
                    a_scale[static_cast<long>(group) * M + output_row];
            }
            activation_scale = __shfl(activation_scale, 0, 32);
            accumulator[row_offset] += static_cast<float>(integer_sum[row_offset]) *
                                       activation_scale * weight_scale;
        }
    }
    if (valid_col) {
#pragma unroll
        for (int row_offset = 0; row_offset < kMtile; ++row_offset) {
            const int output_row = row_base + row_offset;
            if (output_row < M) {
                output[static_cast<long>(output_row) * N + output_col] =
                    __float2bfloat16(accumulator[row_offset]);
            }
        }
    }
}

// Group-major multi-wave WMMA block. Packed A/B and scales are cooperatively
// staged once per K32 group and shared by every 16x16 wave tile in the block.
template <int kBlockM, int kBlockN>
__global__ void s4s4_g32_wmma_tiled(const uint32_t* __restrict__ a_packed,
                                    const uint32_t* __restrict__ b_uint4b8,
                                    const float* __restrict__ a_scale,
                                    const bf16_t* __restrict__ b_scale,
                                    bf16_t* __restrict__ output,
                                    const int M,
                                    const int N,
                                    const int K) {
    constexpr int kWavesM = kBlockM / 16;
    constexpr int kWavesN = kBlockN / 16;
    constexpr int kWaves = kWavesM * kWavesN;
    __shared__ uint32_t shared_a[kBlockM * 4];
    __shared__ uint32_t shared_b[kBlockN * 4];
    __shared__ float shared_a_scale[kBlockM];
    __shared__ bf16_t shared_b_scale[kBlockN];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int wave_m = wave / kWavesN;
    const int wave_n = wave % kWavesN;
    const int source_lane = lane & 15;
    const int block_m = blockIdx.y * kBlockM;
    const int block_n = blockIdx.x * kBlockN;
    const int tile_m = block_m + wave_m * 16;
    const int tile_n = block_n + wave_n * 16;
    const int groups = K / kGroupSize;
    static_assert(kWaves * 32 <= 256, "candidate block exceeds 256 threads");

    float8x32 accumulator;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        accumulator[j] = 0.0f;
    }

    for (int group = 0; group < groups; ++group) {
        for (int index = tid; index < kBlockM * 4; index += blockDim.x) {
            const int word = index / kBlockM;
            const int local_row = index % kBlockM;
            const int global_row = block_m + local_row;
            shared_a[local_row * 4 + word] =
                global_row < M
                    ? a_packed[static_cast<long>(group * 4 + word) * M + global_row]
                    : 0;
        }
        for (int index = tid; index < kBlockN * 4; index += blockDim.x) {
            const int word = index / kBlockN;
            const int local_col = index % kBlockN;
            const int global_col = block_n + local_col;
            shared_b[local_col * 4 + word] =
                global_col < N
                    ? b_uint4b8[static_cast<long>(group * 4 + word) * N + global_col] ^
                          kUint4B8ToS4Mask
                    : 0;
        }
        for (int index = tid; index < kBlockM; index += blockDim.x) {
            const int global_row = block_m + index;
            shared_a_scale[index] =
                global_row < M
                    ? a_scale[static_cast<long>(group) * M + global_row]
                    : 0.0f;
        }
        for (int index = tid; index < kBlockN; index += blockDim.x) {
            const int global_col = block_n + index;
            shared_b_scale[index] =
                global_col < N
                    ? b_scale[static_cast<long>(group) * N + global_col]
                    : __float2bfloat16(0.0f);
        }
        __syncthreads();

        int8x32 group_accumulator;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            group_accumulator[j] = 0;
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int local_row = wave_m * 16 + source_lane;
            const int local_col = wave_n * 16 + source_lane;
            int2x32 a_fragment = {
                static_cast<int>(shared_a[local_row * 4 + half * 2]),
                static_cast<int>(shared_a[local_row * 4 + half * 2 + 1]),
            };
            int2x32 b_fragment = {
                static_cast<int>(shared_b[local_col * 4 + half * 2]),
                static_cast<int>(shared_b[local_col * 4 + half * 2 + 1]),
            };
            group_accumulator = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment, group_accumulator, false);
        }

        const int local_output_col = wave_n * 16 + c_fragment_col(lane);
        const int output_col = block_n + local_output_col;
        const float weight_scale =
            __bfloat162float(shared_b_scale[local_output_col]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int local_output_row = wave_m * 16 + c_fragment_row(lane, j);
            const int output_row = block_m + local_output_row;
            if (output_row < M && output_col < N) {
                accumulator[j] += static_cast<float>(group_accumulator[j]) *
                                  shared_a_scale[local_output_row] * weight_scale;
            }
        }
        __syncthreads();
    }

    const int output_col = tile_n + c_fragment_col(lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int output_row = tile_m + c_fragment_row(lane, j);
        if (output_row < M && output_col < N) {
            output[static_cast<long>(output_row) * N + output_col] =
                __float2bfloat16(accumulator[j]);
        }
    }
}

void check_launch(const char* stage) {
    const hipError_t status = hipGetLastError();
    TORCH_CHECK(status == hipSuccess, stage, ": ", hipGetErrorString(status));
}

int64_t resolve_route(const int64_t route, const int M) {
    TORCH_CHECK(route >= kRouteDotBaseline && route <= kRouteWmma32x32,
                "explicit performance route 1..7 is required");
    TORCH_CHECK(route != kRouteDotBaseline || M <= 8,
                "DOT baseline route requires M <= 8");
    TORCH_CHECK((route != kRouteDotMTile4 && route != kRouteDotMTile8) || M <= 8,
                "M-tiled DOT route requires M <= 8");
    return route;
}

void check_common_device(const torch::Tensor& a,
                         const torch::Tensor& a_scale,
                         const torch::Tensor& b,
                         const torch::Tensor& b_scale) {
    TORCH_CHECK(a.is_cuda() && a_scale.is_cuda() && b.is_cuda() && b_scale.is_cuda(),
                "all tensors must be CUDA/HIP tensors");
    TORCH_CHECK(a.scalar_type() == torch::kInt32, "activation packed must be int32");
    TORCH_CHECK(a_scale.scalar_type() == torch::kFloat32,
                "activation scale must be FP32");
    TORCH_CHECK(b.scalar_type() == torch::kInt32, "weight packed must be int32");
    TORCH_CHECK(b_scale.scalar_type() == torch::kBFloat16,
                "weight scale must be BF16");
    TORCH_CHECK(a.is_contiguous() && a_scale.is_contiguous() && b.is_contiguous() &&
                    b_scale.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(a.device() == a_scale.device() && a.device() == b.device() &&
                    a.device() == b_scale.device(),
                "all tensors must share one device");
}

void check_quantized_rowmajor(const torch::Tensor& a,
                              const torch::Tensor& a_scale,
                              const torch::Tensor& b,
                              const torch::Tensor& b_scale) {
    check_common_device(a, a_scale, b, b_scale);
    TORCH_CHECK(a.dim() == 2 && a_scale.dim() == 2 && b.dim() == 2 &&
                    b_scale.dim() == 2,
                "row-major tensors must be rank two");
    const int64_t M = a.size(0);
    const int64_t K = a.size(1) * 8;
    const int64_t N = b.size(0);
    TORCH_CHECK(M > 0 && N > 0 && N % 16 == 0 && K > 0 && K % 32 == 0,
                "invalid row-major M/N/K");
    TORCH_CHECK(b.size(1) * 8 == K, "row-major packed K mismatch");
    TORCH_CHECK(a_scale.sizes() == torch::IntArrayRef({M, K / 32}),
                "row-major activation scale mismatch");
    TORCH_CHECK(b_scale.sizes() == torch::IntArrayRef({N, K / 32}),
                "row-major weight scale mismatch");
}

void check_quantized_groupmajor(const torch::Tensor& a,
                                const torch::Tensor& a_scale,
                                const torch::Tensor& b,
                                const torch::Tensor& b_scale) {
    check_common_device(a, a_scale, b, b_scale);
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a_scale.dim() == 2 &&
                    b_scale.dim() == 2,
                "group-major tensors must be [groups,4,rows/cols]");
    const int64_t groups = a.size(0);
    const int64_t M = a.size(2);
    const int64_t N = b.size(2);
    TORCH_CHECK(groups > 0 && M > 0 && N > 0 && N % 16 == 0,
                "invalid group-major M/N/K");
    TORCH_CHECK(a.size(1) == 4 && b.size(0) == groups && b.size(1) == 4,
                "group-major packed shape mismatch");
    TORCH_CHECK(a_scale.sizes() == torch::IntArrayRef({groups, M}),
                "group-major activation scale mismatch");
    TORCH_CHECK(b_scale.sizes() == torch::IntArrayRef({groups, N}),
                "group-major weight scale mismatch");
}

void check_input(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16,
                "input must be a BF16 CUDA/HIP tensor");
    TORCH_CHECK(input.dim() == 2 && input.is_contiguous() && input.size(0) > 0 &&
                    input.size(1) > 0 && input.size(1) % 32 == 0,
                "input must be contiguous [M,K] with K divisible by 32");
}

void launch_quantize_rowmajor(const torch::Tensor& input,
                              torch::Tensor& packed,
                              torch::Tensor& scales,
                              const at::cuda::CUDAStream& stream) {
    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    quantize_pack_s4_g32_rowmajor<<<dim3(K / 32, M), 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        scales.data_ptr<float>(), M, K);
    check_launch("quantize_s4_rowmajor launch failed");
}

void launch_quantize_groupmajor(const torch::Tensor& input,
                                torch::Tensor& packed,
                                torch::Tensor& scales,
                                const at::cuda::CUDAStream& stream) {
    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    quantize_pack_s4_g32_groupmajor<<<dim3(K / 32, M), 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        scales.data_ptr<float>(), M, K);
    check_launch("quantize_s4_groupmajor launch failed");
}

void launch_rowmajor(const torch::Tensor& a,
                     const torch::Tensor& a_scale,
                     const torch::Tensor& b,
                     const torch::Tensor& b_scale,
                     torch::Tensor& output,
                     const int64_t route,
                     const at::cuda::CUDAStream& stream) {
    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1) * 8);
    const int N = static_cast<int>(b.size(0));
    resolve_route(route, M);
    if (route == kRouteDotBaseline) {
        constexpr int threads = 256;
        s4s4_g32_dot_baseline<<<dim3((N + threads - 1) / threads, M), threads, 0,
                                     stream>>>(
            reinterpret_cast<const uint32_t*>(a.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(b.data_ptr<int32_t>()),
            a_scale.data_ptr<float>(),
            reinterpret_cast<const bf16_t*>(b_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()), M, N, K);
        check_launch("s4s4_g32_dot_baseline launch failed");
        return;
    }
    TORCH_CHECK(route == kRouteWmmaBaselinePadded,
                "row-major API supports routes 1 or 2 only");
    s4s4_g32_wmma_baseline<<<dim3((N + 15) / 16, (M + 15) / 16), 32, 0,
                                  stream>>>(
        reinterpret_cast<const uint32_t*>(a.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(b.data_ptr<int32_t>()),
        a_scale.data_ptr<float>(),
        reinterpret_cast<const bf16_t*>(b_scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()), M, N, K);
    check_launch("s4s4_g32_wmma_baseline launch failed");
}

void launch_groupmajor(const torch::Tensor& a,
                       const torch::Tensor& a_scale,
                       const torch::Tensor& b,
                       const torch::Tensor& b_scale,
                       torch::Tensor& output,
                       const int64_t route,
                       const at::cuda::CUDAStream& stream) {
    const int M = static_cast<int>(a.size(2));
    const int K = static_cast<int>(a.size(0) * 32);
    const int N = static_cast<int>(b.size(2));
    resolve_route(route, M);
    if (route == kRouteDotMTile4) {
        constexpr int threads = 128;
        s4s4_g32_dot_mtile<4><<<dim3((N + threads - 1) / threads, (M + 3) / 4),
                                      threads, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(a.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(b.data_ptr<int32_t>()),
            a_scale.data_ptr<float>(),
            reinterpret_cast<const bf16_t*>(b_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()), M, N, K);
        check_launch("s4s4_g32_dot_mtile4 launch failed");
        return;
    }
    if (route == kRouteDotMTile8) {
        constexpr int threads = 128;
        s4s4_g32_dot_mtile<8><<<dim3((N + threads - 1) / threads, (M + 7) / 8),
                                      threads, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(a.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(b.data_ptr<int32_t>()),
            a_scale.data_ptr<float>(),
            reinterpret_cast<const bf16_t*>(b_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()), M, N, K);
        check_launch("s4s4_g32_dot_mtile8 launch failed");
        return;
    }
#define LAUNCH_WMMA(BM, BN, LABEL)                                                     \
    s4s4_g32_wmma_tiled<BM, BN>                                                       \
        <<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM),                                \
           (BM / 16) * (BN / 16) * 32, 0, stream>>>(                                  \
            reinterpret_cast<const uint32_t*>(a.data_ptr<int32_t>()),                 \
            reinterpret_cast<const uint32_t*>(b.data_ptr<int32_t>()),                 \
            a_scale.data_ptr<float>(),                                                 \
            reinterpret_cast<const bf16_t*>(b_scale.data_ptr<at::BFloat16>()),        \
            reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()), M, N, K);      \
    check_launch(LABEL)
    if (route == kRouteWmma16x64) {
        LAUNCH_WMMA(16, 64, "s4s4_g32_wmma_16x64 launch failed");
        return;
    }
    if (route == kRouteWmma32x64) {
        LAUNCH_WMMA(32, 64, "s4s4_g32_wmma_32x64 launch failed");
        return;
    }
    if (route == kRouteWmma32x32) {
        LAUNCH_WMMA(32, 32, "s4s4_g32_wmma_32x32 launch failed");
        return;
    }
#undef LAUNCH_WMMA
    TORCH_CHECK(false, "group-major API supports routes 3..7 only");
}

std::tuple<torch::Tensor, torch::Tensor> quantize_s4_rowmajor(torch::Tensor input) {
    check_input(input);
    const at::cuda::OptionalCUDAGuard guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    torch::Tensor packed = torch::empty({M, K / 8}, input.options().dtype(torch::kInt32));
    torch::Tensor scales = torch::empty({M, K / 32}, input.options().dtype(torch::kFloat32));
    launch_quantize_rowmajor(input, packed, scales, stream);
    return std::make_tuple(packed, scales);
}

std::tuple<torch::Tensor, torch::Tensor> quantize_s4_groupmajor(torch::Tensor input) {
    check_input(input);
    const at::cuda::OptionalCUDAGuard guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    torch::Tensor packed = torch::empty({K / 32, 4, M}, input.options().dtype(torch::kInt32));
    torch::Tensor scales = torch::empty({K / 32, M}, input.options().dtype(torch::kFloat32));
    launch_quantize_groupmajor(input, packed, scales, stream);
    return std::make_tuple(packed, scales);
}

torch::Tensor linear_quantized_rowmajor(torch::Tensor a,
                                        torch::Tensor a_scale,
                                        torch::Tensor b,
                                        torch::Tensor b_scale,
                                        const int64_t route) {
    check_quantized_rowmajor(a, a_scale, b, b_scale);
    const int M = static_cast<int>(a.size(0));
    const int N = static_cast<int>(b.size(0));
    resolve_route(route, M);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    torch::Tensor output = torch::empty({M, N}, b_scale.options().dtype(torch::kBFloat16));
    launch_rowmajor(a, a_scale, b, b_scale, output, route,
                    at::cuda::getCurrentCUDAStream());
    return output;
}

torch::Tensor linear_quantized_groupmajor(torch::Tensor a,
                                          torch::Tensor a_scale,
                                          torch::Tensor b,
                                          torch::Tensor b_scale,
                                          const int64_t route) {
    check_quantized_groupmajor(a, a_scale, b, b_scale);
    const int M = static_cast<int>(a.size(2));
    const int N = static_cast<int>(b.size(2));
    resolve_route(route, M);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    torch::Tensor output = torch::empty({M, N}, b_scale.options().dtype(torch::kBFloat16));
    launch_groupmajor(a, a_scale, b, b_scale, output, route,
                      at::cuda::getCurrentCUDAStream());
    return output;
}

torch::Tensor linear_rowmajor(torch::Tensor input,
                              torch::Tensor b,
                              torch::Tensor b_scale,
                              const int64_t route) {
    check_input(input);
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(b.dim() == 2 && b.size(1) * 8 == K, "row-major weight shape mismatch");
    TORCH_CHECK(b_scale.sizes() == torch::IntArrayRef({b.size(0), K / 32}),
                "row-major weight scale mismatch");
    resolve_route(route, static_cast<int>(M));
    const at::cuda::OptionalCUDAGuard guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor a = torch::empty({M, K / 8}, input.options().dtype(torch::kInt32));
    torch::Tensor a_scale = torch::empty({M, K / 32}, input.options().dtype(torch::kFloat32));
    torch::Tensor output = torch::empty({M, b.size(0)}, input.options().dtype(torch::kBFloat16));
    launch_quantize_rowmajor(input, a, a_scale, stream);
    launch_rowmajor(a, a_scale, b, b_scale, output, route, stream);
    return output;
}

torch::Tensor linear_groupmajor(torch::Tensor input,
                                torch::Tensor b,
                                torch::Tensor b_scale,
                                const int64_t route) {
    check_input(input);
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    const int64_t groups = K / 32;
    TORCH_CHECK(b.dim() == 3 && b.sizes().slice(0, 2) == torch::IntArrayRef({groups, 4}),
                "group-major weight packed shape mismatch");
    TORCH_CHECK(b_scale.sizes() == torch::IntArrayRef({groups, b.size(2)}),
                "group-major weight scale mismatch");
    resolve_route(route, static_cast<int>(M));
    const at::cuda::OptionalCUDAGuard guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor a = torch::empty({groups, 4, M}, input.options().dtype(torch::kInt32));
    torch::Tensor a_scale = torch::empty({groups, M}, input.options().dtype(torch::kFloat32));
    torch::Tensor output = torch::empty({M, b.size(2)}, input.options().dtype(torch::kBFloat16));
    launch_quantize_groupmajor(input, a, a_scale, stream);
    launch_groupmajor(a, a_scale, b, b_scale, output, route, stream);
    return output;
}

}  // namespace gfx1151_w4a4_perf

TORCH_LIBRARY(gfx1151_w4a4_perf, m) {
    m.def("quantize_s4_rowmajor(Tensor input) -> (Tensor, Tensor)");
    m.def("quantize_s4_groupmajor(Tensor input) -> (Tensor, Tensor)");
    m.def("linear_quantized_rowmajor(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, int route) -> Tensor");
    m.def("linear_quantized_groupmajor(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, int route) -> Tensor");
    m.def("linear_rowmajor(Tensor input, Tensor b, Tensor b_scale, int route) -> Tensor");
    m.def("linear_groupmajor(Tensor input, Tensor b, Tensor b_scale, int route) -> Tensor");
}

TORCH_LIBRARY_IMPL(gfx1151_w4a4_perf, CUDA, m) {
    m.impl("quantize_s4_rowmajor", &gfx1151_w4a4_perf::quantize_s4_rowmajor);
    m.impl("quantize_s4_groupmajor", &gfx1151_w4a4_perf::quantize_s4_groupmajor);
    m.impl("linear_quantized_rowmajor", &gfx1151_w4a4_perf::linear_quantized_rowmajor);
    m.impl("linear_quantized_groupmajor", &gfx1151_w4a4_perf::linear_quantized_groupmajor);
    m.impl("linear_rowmajor", &gfx1151_w4a4_perf::linear_rowmajor);
    m.impl("linear_groupmajor", &gfx1151_w4a4_perf::linear_groupmajor);
}
