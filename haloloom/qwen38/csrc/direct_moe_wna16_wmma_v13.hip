// v13 = v6w10 core (BM64 x BN128, 8 waves, 3-set register ring, prefetch distance 2) with NON-BITWISE fp16 operands:
//   A: fp16 copy of the bf16 activations (exact for |a| in fp16 normal range; saturating conversion done by caller).
//   B: fp16 (q - ZP) * s via magic numbers:  bits 0x6400 | q == 1024 + q (exact),  (x - (1024 + ZP)) exact small int
//      (v_pk_add_f16), * s16 (v_pk_mul_f16, one rounding; s16 = fp16(scale)).  ~2.4 VALU/element vs ~4.9 in v6w10.
//   WMMA v_wmma_f32_16x16x16_f16 (same rate as bf16), fp32 accumulate, bf16 RNE output (unchanged).
//   Weight precision: fp16 (11-bit) instead of bf16 (8-bit) rounding of (q - ZP) * s; scale rounded to fp16.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define BM 64
#define BN 128
#define BKS 32
#define RS 20
#define GK 128
#define SPG (GK / BKS)

__device__ __forceinline__ uint32_t pack_hi(uint32_t lo_elem, uint32_t hi_elem) {
  return __builtin_amdgcn_perm(hi_elem, lo_elem, 0x07060302u);
}
__device__ __forceinline__ uint16_t f32_to_bf16_rne_nan(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float ubyte_f(uint32_t x, int c) { return (float)((x >> (8 * c)) & 0xffu); }
__device__ __forceinline__ int sw4(int n, int ch) { return n * RS + 4 * (ch ^ ((n >> 4) & 3)); }
#define MAGIC 0x64006400u
__device__ __forceinline__ uint32_t deq2(uint32_t magic_pair, h2 off, h2 sc) {
  const h2 x = __builtin_bit_cast(h2, magic_pair);
  const h2 y = (x + off) * sc;
  return __builtin_bit_cast(uint32_t, y);
}

struct Ctx {
  uint32_t (*Bs)[BN * RS];
  int32_t* s_tok;
};

// MODE 0: W4 [E][K/2][N]   MODE 1: W8 [E][K][N]   MODE 2: W8 [E][N][K]
template <int MODE>
__device__ __forceinline__ void body(
    Ctx cx, const _Float16* __restrict__ A, const uint8_t* __restrict__ b_base, const float* __restrict__ s_base,
    __bf16* __restrict__ C, int N, int K, int num_valid, int top_k, long long stride_am, long long stride_bk, long long stride_bn,
    long long stride_sg, long long stride_sn, long long stride_cm) {
  constexpr int NBR = (MODE == 0) ? 2 : 4;
  const h2 OFF = (MODE == 0) ? (h2){(_Float16)-1032.0f, (_Float16)-1032.0f} : (h2){(_Float16)-1152.0f, (_Float16)-1152.0f};
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
  const int wm = wave & 3, wn = wave >> 2;
  const int n0 = blockIdx.x * BN;
  int32_t* s_tok = cx.s_tok;
  const int my_tok = s_tok[wm * 16 + l16];
  const bool wave_active = s_tok[wm * 16] < num_valid;
  const _Float16* arow = A + (long long)(my_tok < num_valid ? my_tok / top_k : 0) * stride_am;
  const int q = tid & 31, rg = tid >> 5;
  const int n2 = tid & 127, h2i = tid >> 7;
  const int gq = n0 + 4 * q;
  const int g2 = n0 + n2;

  u32x4 aX[4], aY[4], aZ[4];
  uint32_t bX[NBR], bY[NBR], bZ[NBR];
  auto loadA = [&](u32x4* dst, int k0) {
    if (wave_active) {
#pragma unroll
      for (int c = 0; c < 4; ++c) dst[c] = *(const u32x4*)(arow + k0 + 8 * c);
    }
  };
  auto loadB = [&](uint32_t* braw, int k0) {
    if constexpr (MODE == 0) {
#pragma unroll
      for (int j = 0; j < 2; ++j) braw[j] = *(const uint32_t*)(b_base + (long long)(k0 / 2 + 2 * rg + j) * stride_bk + gq);
    } else if constexpr (MODE == 1) {
#pragma unroll
      for (int j = 0; j < 4; ++j) braw[j] = *(const uint32_t*)(b_base + (long long)(k0 + 4 * rg + j) * stride_bk + gq);
    } else {
      u32x4 v = *(const u32x4*)(b_base + (long long)g2 * stride_bn + k0 + 16 * h2i);
      braw[0] = v[0]; braw[1] = v[1]; braw[2] = v[2]; braw[3] = v[3];
    }
  };
  auto storeB = [&](int buf, int k0, const uint32_t* br) {
    const int g = k0 >> 7;
    uint32_t* bs = cx.Bs[buf];
    if constexpr (MODE == 0) {
      const float4 s4 = *(const float4*)(s_base + (long long)g * stride_sg + gq);
      const float sc[4] = {s4.x, s4.y, s4.z, s4.w};
      const uint32_t l0 = br[0] & 0x0f0f0f0fu, h0 = (br[0] >> 4) & 0x0f0f0f0fu;
      const uint32_t l1 = br[1] & 0x0f0f0f0fu, h1 = (br[1] >> 4) & 0x0f0f0f0fu;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t sel = (uint32_t)c | 0x0c00u | ((uint32_t)(4 + c) << 16) | 0x0c000000u;
        const _Float16 s16 = (_Float16)sc[c];
        const h2 sv = (h2){s16, s16};
        const uint32_t o0 = deq2(__builtin_amdgcn_perm(h0, l0, sel) | MAGIC, OFF, sv);
        const uint32_t o1 = deq2(__builtin_amdgcn_perm(h1, l1, sel) | MAGIC, OFF, sv);
        *(u32x2*)(&bs[sw4(4 * q + c, rg >> 1) + 2 * (rg & 1)]) = (u32x2){o0, o1};
      }
    } else if constexpr (MODE == 1) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t sel = (uint32_t)c | 0x0c00u | ((uint32_t)(4 + c) << 16) | 0x0c000000u;
        const _Float16 s16 = (_Float16)s_base[(long long)(gq + c) * stride_sn + (long long)g * stride_sg];
        const h2 sv = (h2){s16, s16};
        const uint32_t o0 = deq2(__builtin_amdgcn_perm(br[1], br[0], sel) | MAGIC, OFF, sv);
        const uint32_t o1 = deq2(__builtin_amdgcn_perm(br[3], br[2], sel) | MAGIC, OFF, sv);
        *(u32x2*)(&bs[sw4(4 * q + c, rg >> 1) + 2 * (rg & 1)]) = (u32x2){o0, o1};
      }
    } else {
      const _Float16 s16 = (_Float16)s_base[(long long)g2 * stride_sn + (long long)g * stride_sg];
      const h2 sv = (h2){s16, s16};
      uint32_t o[8];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const uint32_t x = br[w];
        o[2 * w] = deq2(__builtin_amdgcn_perm(x, x, 0x0c010c00u) | MAGIC, OFF, sv);
        o[2 * w + 1] = deq2(__builtin_amdgcn_perm(x, x, 0x0c030c02u) | MAGIC, OFF, sv);
      }
      *(u32x4*)(&bs[sw4(n2, 2 * h2i)]) = (u32x4){o[0], o[1], o[2], o[3]};
      *(u32x4*)(&bs[sw4(n2, 2 * h2i + 1)]) = (u32x4){o[4], o[5], o[6], o[7]};
    }
  };

  f32x8 acc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
  const int NS = K / BKS;
  auto step = [&](int s, const u32x4* cur, const uint32_t* b1, u32x4* nx, uint32_t* b2) {
    const int buf = s & 1;
    if (s + 2 < NS) { loadA(nx, (s + 2) * BKS); loadB(b2, (s + 2) * BKS); }
    if (wave_active) {
      const uint32_t* bs = cx.Bs[buf];
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
        u32x4 x0 = cur[2 * kc], x1 = cur[2 * kc + 1];
        f16x16 af = __builtin_bit_cast(f16x16, (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]});
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int n = 64 * wn + 16 * j + l16;
          u32x4 y0 = *(const u32x4*)(&bs[sw4(n, 2 * kc)]);
          u32x4 y1 = *(const u32x4*)(&bs[sw4(n, 2 * kc + 1)]);
          f16x16 bf = __builtin_bit_cast(f16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
          acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc[j]);
        }
      }
    }
    if (s + 1 < NS) storeB(buf ^ 1, (s + 1) * BKS, b1);
    __syncthreads();
  };
  loadA(aX, 0); loadB(bX, 0);
  if (NS > 1) { loadA(aY, BKS); loadB(bY, BKS); }
  storeB(0, 0, bX);
  __syncthreads();
  for (int s = 0; s < NS; s += 3) {
    step(s, aX, bY, aZ, bZ);
    if (s + 1 < NS) step(s + 1, aY, bZ, aX, bX);
    if (s + 2 < NS) step(s + 2, aZ, bX, aY, bY);
  }
  if (!wave_active) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int r = wm * 16 + 2 * e + (lane >> 4);
    const int tok = s_tok[r];
    if (tok >= num_valid) continue;
    __bf16* crow = C + (long long)tok * stride_cm + n0 + 64 * wn + l16;
#pragma unroll
    for (int j = 0; j < 4; ++j) *(uint16_t*)(crow + 16 * j) = f32_to_bf16_rne_nan(acc[j][e]);
  }
}

template <int MODE8>
__device__ __forceinline__ void unified(
    const _Float16* A, const uint8_t* B4, const float* S4, const uint8_t* B8, const float* S8, __bf16* C,
    const int32_t* sorted_ids, const int32_t* expert_ids, const int32_t* ntpp, const int32_t* tab,
    int N, int K, int nv, int tk, long long sam,
    long long b4e, long long b4k, long long b4n, long long s4e, long long s4g, long long s4n,
    long long b8e, long long b8k, long long b8n, long long s8e, long long s8g, long long s8n, long long scm) {
  __shared__ __attribute__((aligned(16))) uint32_t Bs[2][BN * RS];
  __shared__ int32_t s_tok[BM];
  const int tid = threadIdx.x, pid_m = blockIdx.y;
  if (pid_m * BM >= ntpp[0]) return;
  if (tid < BM) s_tok[tid] = sorted_ids[pid_m * BM + tid];
  const int e = expert_ids[pid_m];
  __syncthreads();
  if (e < 0) return;
  const int t = tab[e], pool = t >> 16, local = t & 0xffff;
  Ctx cx{Bs, s_tok};
  if (pool == 0)
    body<0>(cx, A, B4 + (long long)local * b4e, S4 + (long long)local * s4e, C, N, K, nv, tk, sam, b4k, b4n, s4g, s4n, scm);
  else
    body<MODE8>(cx, A, B8 + (long long)local * b8e, S8 + (long long)local * s8e, C, N, K, nv, tk, sam, b8k, b8n, s8g, s8n, scm);
}

#define UARGS                                                                                                                 \
  const _Float16 *A, const uint8_t *B4, const float *S4, const uint8_t *B8, const float *S8, __bf16 *C, const int32_t *sid,  \
      const int32_t *eid, const int32_t *ntpp, const int32_t *tab, int N, int K, int nv, int tk, long long sam, long long b4e, \
      long long b4k, long long b4n, long long s4e, long long s4g, long long s4n, long long b8e, long long b8k, long long b8n,   \
      long long s8e, long long s8g, long long s8n, long long scm
#define UPASS A, B4, S4, B8, S8, C, sid, eid, ntpp, tab, N, K, nv, tk, sam, b4e, b4k, b4n, s4e, s4g, s4n, b8e, b8k, b8n, s8e, s8g, s8n, scm

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10, 10))) moe13_up(UARGS) { unified<1>(UPASS); }
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10, 10))) moe13_down(UARGS) { unified<2>(UPASS); }
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8))) moe13w8_up(UARGS) { unified<1>(UPASS); }
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8))) moe13w8_down(UARGS) { unified<2>(UPASS); }

// bf16 -> fp16 with saturation to +-65504 (NaN preserved); also counts saturated elements into *nsat (atomic).
extern "C" __global__ void __launch_bounds__(256) cvt_bf16_f16_sat(const uint16_t* __restrict__ x, _Float16* __restrict__ y, long long n,
                                                                   unsigned int* __restrict__ nsat) {
  const long long i = ((long long)blockIdx.x * 256 + threadIdx.x) * 8;
  if (i >= n) return;
  unsigned int sat = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    if (i + k >= n) break;
    float f = __uint_as_float((uint32_t)x[i + k] << 16);
    if (f > 65504.0f) { f = 65504.0f; ++sat; } else if (f < -65504.0f) { f = -65504.0f; ++sat; }
    y[i + k] = (_Float16)f;
  }
  if (sat && nsat) atomicAdd(nsat, sat);
}
