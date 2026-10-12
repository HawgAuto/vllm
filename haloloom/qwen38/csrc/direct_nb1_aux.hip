// nb1 aux kernels (NON-BITWISE arm only), gfx1151.
// silu_mul_bf16_f16 : act[r, j] = fp16_sat( silu(f32(up[r, j])) * f32(up[r, D + j]) ), up bf16 [S, 2D] -> act fp16 [S, D]
//                     (vLLM _C.silu_and_mul semantics, fp32 math, but rounded to fp16 so moe13_down can consume it directly).
// hc_ugm_xn         : y[m, n] = bf16( (1/HC) * sum_s sigmoid(bf16(lora[m,:] . Wup[s*H+n, :])) * xn[m, s*H+n] )
//                     = hc_gate_mix(xn, input_mix_weight_up(lora)) with the [M, HC*H] gate never materialized
//                     (derived from overlays/hc-fused-v2 hc_ugm; reads the production xn instead of recomputing it).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));
#define RS 20

__device__ __forceinline__ int sw4(int n, int ch) { return n * RS + 4 * (ch ^ ((n >> 4) & 3)); }
__device__ __forceinline__ float bf2f(uint32_t bits16) { return __uint_as_float(bits16 << 16); }
__device__ __forceinline__ uint32_t f2bf_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ float bfround(float f) { return __uint_as_float(f2bf_rne(f) << 16); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

extern "C" __global__ void __launch_bounds__(256) silu_mul_bf16_f16(const uint16_t* __restrict__ up, _Float16* __restrict__ act,
                                                                   int S, int D, long long s_up, long long s_act,
                                                                   unsigned int* __restrict__ nsat) {
  const long long i = ((long long)blockIdx.x * 256 + threadIdx.x) * 8;  // flat index into [S, D], D % 8 == 0
  if (i >= (long long)S * D) return;
  const int r = (int)(i / D), j = (int)(i - (long long)r * D);
  const uint16_t* g = up + (long long)r * s_up + j;
  const uint16_t* u = g + D;
  const u32x4 gv = *(const u32x4*)g, uv = *(const u32x4*)u;
  _Float16 o[8];
  unsigned int sat = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float gf = bf2f((gv[k] >> (16 * h)) & 0xffffu), uf = bf2f((uv[k] >> (16 * h)) & 0xffffu);
      float a = gf / (1.0f + expf(-gf)) * uf;
      if (a > 65504.0f) { a = 65504.0f; ++sat; } else if (a < -65504.0f) { a = -65504.0f; ++sat; }
      o[2 * k + h] = (_Float16)a;
    }
  }
  *(u32x4*)(act + (long long)r * s_act + j) = *(const u32x4*)o;
  if (sat && nsat) atomicAdd(nsat, sat);
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(4, 8)))
hc_ugm_xn(const __bf16* __restrict__ lora, const __bf16* __restrict__ wup, const __bf16* __restrict__ xn,
          __bf16* __restrict__ y, int M, int H, int HC, int R, long long s_lora, long long s_xn, long long s_y) {
  __shared__ __attribute__((aligned(16))) uint32_t Bs[2][128 * RS];
  const int n0 = blockIdx.x * 128, m0 = blockIdx.y * 64;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
  const int wm = wave & 3, wn = wave >> 2;
  const int arow = min(m0 + wm * 16 + l16, M - 1);
  const __bf16* ap = lora + (long long)arow * s_lora;
  const int bcol = tid >> 1, bh = tid & 1;
  f32x8 mix[4], acc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) mix[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
  const int KC = R / 32, NT = HC * KC;
  u32x4 bp0, bp1;
  auto gloadB = [&](int t) {
    const int s = t / KC, k0 = (t - s * KC) * 32;
    const u32x4* src = (const u32x4*)(wup + (long long)(s * H + n0 + bcol) * R + k0 + 16 * bh);
    bp0 = src[0]; bp1 = src[1];
  };
  auto sstoreB = [&](int buf) {
    *(u32x4*)(&Bs[buf][sw4(bcol, 2 * bh)]) = bp0;
    *(u32x4*)(&Bs[buf][sw4(bcol, 2 * bh + 1)]) = bp1;
  };
  gloadB(0); sstoreB(0);
  __syncthreads();
  for (int t = 0; t < NT; ++t) {
    const int s = t / KC, kci = t - s * KC, buf = t & 1;
    if (kci == 0) {
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
    }
    u32x4 av[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) av[c] = *(const u32x4*)(ap + kci * 32 + 8 * c);
    if (t + 1 < NT) gloadB(t + 1);
    const uint32_t* bs = Bs[buf];
#pragma unroll
    for (int kc = 0; kc < 2; ++kc) {
      const u32x4 x0 = av[2 * kc], x1 = av[2 * kc + 1];
      const bf16x16 af = __builtin_bit_cast(bf16x16, (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]});
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int n = 64 * wn + 16 * j + l16;
        const u32x4 y0 = *(const u32x4*)(&bs[sw4(n, 2 * kc)]);
        const u32x4 y1 = *(const u32x4*)(&bs[sw4(n, 2 * kc + 1)]);
        const bf16x16 bf = __builtin_bit_cast(bf16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc[j]);
      }
    }
    if (kci == KC - 1) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + wm * 16 + 2 * e + (lane >> 4);
        const int rc = min(r, M - 1);
        const __bf16* xrow = xn + (long long)rc * s_xn + (long long)s * H + n0 + 64 * wn + l16;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float g = bfround(acc[j][e]);
          const float xv = bf2f(*(const uint16_t*)(xrow + 16 * j));
          mix[j][e] = __builtin_fmaf(sigm(g), xv, mix[j][e]);
        }
      }
    }
    if (t + 1 < NT) sstoreB(buf ^ 1);
    __syncthreads();
  }
  const float inv = 1.0f / (float)HC;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int r = m0 + wm * 16 + 2 * e + (lane >> 4);
    if (r >= M) continue;
    __bf16* yrow = y + (long long)r * s_y + n0 + 64 * wn + l16;
#pragma unroll
    for (int j = 0; j < 4; ++j) *(uint16_t*)(yrow + 16 * j) = (uint16_t)f2bf_rne(mix[j][e] * inv);
  }
}
