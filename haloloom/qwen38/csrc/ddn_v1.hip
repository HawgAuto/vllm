// ddn_v1.hip -- skinny-M draft dense W8A16 (MTPDenseMethod, group 64) for gfx1151.
//
// Replaces the stock fused_moe_kernel_gptq_awq(E=1, top_k=1, BLOCK 16/32/64) launch used by mtp_dense.MTPDenseMethod
// for small M.  Arithmetic contract (identical to sdm_v1/sdm_v2, which emulate the same stock int8_w8a16 kernel):
//   b   = bf16_rne(fma(q, s, -128*s))      q = uint8 offset-128 weight, s = FP32 group-64 scale (exact for finite s>0)
//   acc = v_wmma_f32_16x16x16_bf16 chain over K ascending in 16-chunks from +0, WMMA dual-row layout (A replicated)
//   out = bf16_rne(acc * 1.0f)            NaN-quieting RNE, sign of zero preserved
// Per output element the chain depends only on its own x row, its own W column and the K order, so one wave may
// carry 32 columns (lanes 0-15: n0+j, lanes 16-31: n0+16+j) and MR x rows (one WMMA per row, same B registers).
//
// Layout (unchanged resident layout): x [M][K] bf16 (row-contiguous, 16 B aligned), W [N][K] uint8 (K-contiguous),
// S [N][K/64] f32, out [M][N] bf16.  K % 64 == 0, N >= 1.  Out-of-range columns are clamped for loads and never stored.
// Weight bytes are register-prefetched PG groups ahead (one group = 64 k = 4 b128 per lane); x rows are staged in LDS.
// No new numerics, no split-K, no atomics.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

// ---- helpers copied verbatim from sdm_v2.hip ----------------------------------------------------------------
__device__ __forceinline__ uint32_t rne_hi(float f) {
  uint32_t u = __float_as_uint(f);
  return u + 0x7fffu + ((u >> 16) & 1u);
}
__device__ __forceinline__ uint32_t pack_hi(uint32_t lo_elem, uint32_t hi_elem) {
  return __builtin_amdgcn_perm(hi_elem, lo_elem, 0x07060302u);
}
__device__ __forceinline__ uint32_t f32_to_bf16_rne_nan(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return ((u >> 16) | 0x40u) & 0xffffu;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ float ubyte_f(uint32_t x, int c) { return (float)((x >> (8 * c)) & 0xffu); }
__device__ __forceinline__ f32x8 wmma(const u32x8& a, const u32x8& b, const f32x8& c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(bf16x16, a), __builtin_bit_cast(bf16x16, b), c);
}
__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(const void* p) {
  return __builtin_amdgcn_make_buffer_rsrc((void*)p, (short)0, 0x7FFFFFFF, 0x31004000);
}
__device__ __forceinline__ float bldf(__amdgpu_buffer_rsrc_t r, int vo, int so) {
  return __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(r, vo, so, 0));
}
// ---------------------------------------------------------------------------------------------------------------

template <int MR, int PG, int WPG>
__device__ __forceinline__ void ddn_body(const __bf16* __restrict__ x, const uint8_t* __restrict__ W,
                                         const float* __restrict__ S, __bf16* __restrict__ out, int M, int N, int K) {
  extern __shared__ uint32_t xs[];  // MR rows x K/2 dwords
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int j = lane & 15, h = lane >> 4;
  const int n0 = (blockIdx.x * WPG + wave) * 32;
  const int col = n0 + h * 16 + j;
  const int colc = col < N ? col : N - 1;  // clamped load column; results of clamped lanes are discarded
  const int m0 = blockIdx.y * MR;
  const int G = K >> 6;
  const int KD = K >> 1;
  const __amdgpu_buffer_rsrc_t rw = rsrc(W), rs = rsrc(S);
  const int vo = colc * K;      // this lane's W row (column of the output)
  const int so = colc * G * 4;  // this lane's scale row
  u32x4 buf[PG][4];
  float sb[PG];
  // weight prologue first, so its latency overlaps the x staging below
#pragma unroll
  for (int p = 0; p < PG; ++p) {
    if (p < G) {
#pragma unroll
      for (int c = 0; c < 4; ++c) buf[p][c] = __builtin_amdgcn_raw_buffer_load_b128(rw, vo, p * 64 + c * 16, 0);
      sb[p] = bldf(rs, so, p * 4);
    }
  }
#pragma unroll
  for (int r = 0; r < MR; ++r) {
    const int m = (m0 + r) < M ? (m0 + r) : (M - 1);  // rows past M are computed from a valid row, never stored
    const u32x4* xr = (const u32x4*)(x + (size_t)m * (size_t)K);
    u32x4* dst = (u32x4*)(xs + r * KD);
    for (int i = threadIdx.x; i < (KD >> 2); i += 32 * WPG) dst[i] = xr[i];
  }
  __syncthreads();
  f32x8 acc[MR];
#pragma unroll
  for (int r = 0; r < MR; ++r) acc[r] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
  for (int g0 = 0; g0 < G; g0 += PG) {
#pragma unroll
    for (int p = 0; p < PG; ++p) {
      const int g = g0 + p;
      if (g < G) {
        const float s0 = sb[p], m0f = -128.0f * s0;
        u32x4 w4[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) w4[c] = buf[p][c];
        const int gn = g + PG;
        if (gn < G) {
#pragma unroll
          for (int c = 0; c < 4; ++c) buf[p][c] = __builtin_amdgcn_raw_buffer_load_b128(rw, vo, gn * 64 + c * 16, 0);
          sb[p] = bldf(rs, so, gn * 4);
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const u32x4 rr = w4[kk];
          u32x8 b;
#pragma unroll
          for (int r2 = 0; r2 < 8; ++r2) {
            const uint32_t w = rr[r2 >> 1];
            const float q0 = ubyte_f(w, (2 * r2) & 3), q1 = ubyte_f(w, ((2 * r2) & 3) + 1);
            b[r2] = pack_hi(rne_hi(fmaf(q0, s0, m0f)), rne_hi(fmaf(q1, s0, m0f)));
          }
          const int kc = g * 4 + kk;
#pragma unroll
          for (int r = 0; r < MR; ++r) {
            const u32x4* a4 = (const u32x4*)(xs + r * KD + kc * 8);
            const u32x4 a0 = a4[0], a1 = a4[1];
            const u32x8 a = u32x8{a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]};
            acc[r] = wmma(a, b, acc[r]);
          }
        }
      }
    }
  }
  if (col < N) {
#pragma unroll
    for (int r = 0; r < MR; ++r) {
      const int m = m0 + r;
      if (m < M) ((uint16_t*)out)[(size_t)m * (size_t)N + col] = (uint16_t)f32_to_bf16_rne_nan(acc[r][0] * 1.0f);
    }
  }
}

#define DDN(MR, PG, WPG)                                                                                          \
  extern "C" __global__ __launch_bounds__(32 * WPG) void ddn_m##MR##_p##PG##_w##WPG(                              \
      const __bf16* __restrict__ x, const uint8_t* __restrict__ W, const float* __restrict__ S,                    \
      __bf16* __restrict__ out, int M, int N, int K) {                                                             \
    ddn_body<MR, PG, WPG>(x, W, S, out, M, N, K);                                                                  \
  }

DDN(1, 2, 1)
DDN(1, 4, 1)
DDN(1, 8, 1)
DDN(1, 2, 4)
DDN(1, 4, 4)
DDN(1, 8, 4)
DDN(2, 4, 1)
DDN(2, 4, 4)
DDN(4, 4, 1)
DDN(4, 4, 4)
DDN(4, 8, 4)
