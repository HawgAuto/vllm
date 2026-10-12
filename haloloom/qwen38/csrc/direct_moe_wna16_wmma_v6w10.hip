// v6 = v5 + (a) FMA dequant: bf16_rne(fma(q, s, -ZP*s)) == bf16_rne((q - ZP) * s) bit-exactly for finite s >= 0
//   (ZP*s is an exact power-of-two scaling; fma rounds the exact (q-ZP)*s once; q==ZP gives +0 like 0*s for s>=0),
//   with byte-lane v_cvt_f32_ubyte conversions; (b) prefetch distance 2 stages (A and B raw registers ring).
// v5: native gfx1151 WMMA MoE WNA16 — integration kernels on the v4 core (occupancy-first, bit-exact contract):
//   b = bf16_rne((float(q) - ZP) * scale)   acc += WMMA_bf16(a, b) over K ascending in 16-chunks   out = bf16_rne(acc)
// New in v5:
//   * unified pools: ONE launch per projection over the global-expert alignment; per-WG table lookup
//     tab[e] = (pool << 16) | local  (pool 0 = W4 [E][K/2][N], pool 1 = W8); no zero-filled other-pool slots.
//   * fused SiLU-and-mul epilogue for gate_up (FUSE): tile rows 0..63 = gate cols g0.., rows 64..127 = up cols N/2+g0..;
//     out = bf16( float(bf16(silu(float(bf16(gate))))) * float(bf16(up)) ), silu(x) = x / (1 + exp(-x)).
//     SILUV selects exp/div implementation (0: expf + IEEE div, 1: __expf + IEEE div, 2: expf + __fdividef, 3: __expf + __fdividef)
//     so the gate can identify the variant bit-identical to torch.ops._C.silu_and_mul.
//   * kernel-scoped LDS (shared once by both pool bodies).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define BM 64
#define BN 128
#define BKS 32
#define RS 20

__device__ __forceinline__ uint32_t rne_hi(float f) {
  uint32_t u = __float_as_uint(f);
  return u + 0x7fffu + ((u >> 16) & 1u);
}
__device__ __forceinline__ uint32_t pack_hi(uint32_t lo_elem, uint32_t hi_elem) {
  return __builtin_amdgcn_perm(hi_elem, lo_elem, 0x07060302u);
}
__device__ __forceinline__ uint16_t f32_to_bf16_rne_nan(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
// v_cvt_f32_ubyte{0..3}
// (uitofp((x >> 8c) & 0xff) is selected to v_cvt_f32_ubyte{c} by the AMDGPU backend)
__device__ __forceinline__ float ubyte_f(uint32_t x, int c) { return (float)((x >> (8 * c)) & 0xffu); }
__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }
__device__ __forceinline__ int sw4(int n, int ch) { return n * RS + 4 * (ch ^ ((n >> 4) & 3)); }

template <int SILUV>
__device__ __forceinline__ uint16_t silu_mul_bf16(uint16_t g, uint16_t u) {
  const float x = bf16_bits_to_f32(g);
  float e;
  if constexpr (SILUV == 0 || SILUV == 2) e = expf(-x); else e = __expf(-x);
  float s;
  if constexpr (SILUV == 0 || SILUV == 1) s = x / (1.0f + e); else s = __fdividef(x, 1.0f + e);
  const uint16_t sb = f32_to_bf16_rne_nan(s);
  return f32_to_bf16_rne_nan(bf16_bits_to_f32(sb) * bf16_bits_to_f32(u));
}

struct Ctx {
  uint32_t (*Bs)[BN * RS];
  int32_t* s_tok;
};

// MODE 0: W4 [E][K/2][N]   MODE 1: W8 [E][K][N]   MODE 2: W8 [E][N][K]
template <int MODE, bool FUSE, int SILUV>
__device__ __forceinline__ void body(
    Ctx cx, const __bf16* __restrict__ A, const uint8_t* __restrict__ b_base, const float* __restrict__ s_base, __bf16* __restrict__ C,
    int N, int K, int num_valid, int top_k, long long stride_am, long long stride_bk, long long stride_bn,
    long long stride_sg, long long stride_sn, long long stride_cm) {
  constexpr float ZP = (MODE == 0) ? 8.0f : 128.0f;
  constexpr int NBR = (MODE == 0) ? 2 : 4;
  static_assert(!(FUSE && MODE == 2), "fused SiLU only for gate_up");
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
  const int wm = wave & 3, wn = wave >> 2;
  const int half = N >> 1;
  const int n0 = FUSE ? blockIdx.x * 64 : blockIdx.x * BN;
  // global column of tile row nl
  auto gcol = [&](int nl) -> int { return FUSE ? (nl < 64 ? n0 + nl : half + n0 + (nl - 64)) : n0 + nl; };
  int32_t* s_tok = cx.s_tok;

  const int my_tok = s_tok[wm * 16 + l16];
  const bool wave_active = s_tok[wm * 16] < num_valid;
  const __bf16* arow = A + (long long)(my_tok < num_valid ? my_tok / top_k : 0) * stride_am;
  const int q = tid & 31, rg = tid >> 5;
  const int n2 = tid & 127, h2 = tid >> 7;
  const int gq = gcol(4 * q);      // MODE 0/1: 4 consecutive global cols (never straddle halves)
  const int g2 = gcol(n2);         // MODE 2

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
      u32x4 v = *(const u32x4*)(b_base + (long long)g2 * stride_bn + k0 + 16 * h2);
      braw[0] = v[0]; braw[1] = v[1]; braw[2] = v[2]; braw[3] = v[3];
    }
  };
  auto storeB = [&](int buf, int k0, const uint32_t* braw_use) {
    const int g = k0 >> 7;
    uint32_t* bs = cx.Bs[buf];
    if constexpr (MODE == 0) {
      const float4 s4 = *(const float4*)(s_base + (long long)g * stride_sg + gq);
      const float sc[4] = {s4.x, s4.y, s4.z, s4.w};
      const uint32_t* br = braw_use;
      // byte lanes of br[j]: column c's byte; lo/hi nibbles into byte lanes
      const uint32_t l0 = br[0] & 0x0f0f0f0fu, h0 = (br[0] >> 4) & 0x0f0f0f0fu;
      const uint32_t l1 = br[1] & 0x0f0f0f0fu, h1 = (br[1] >> 4) & 0x0f0f0f0fu;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float sz = -ZP * sc[c];
        uint32_t o0 = pack_hi(rne_hi(__builtin_fmaf(ubyte_f(l0, c), sc[c], sz)), rne_hi(__builtin_fmaf(ubyte_f(h0, c), sc[c], sz)));
        uint32_t o1 = pack_hi(rne_hi(__builtin_fmaf(ubyte_f(l1, c), sc[c], sz)), rne_hi(__builtin_fmaf(ubyte_f(h1, c), sc[c], sz)));
        *(u32x2*)(&bs[sw4(4 * q + c, rg >> 1) + 2 * (rg & 1)]) = (u32x2){o0, o1};
      }
    } else if constexpr (MODE == 1) {
      float sc[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) sc[c] = s_base[(long long)(gq + c) * stride_sn + (long long)g * stride_sg];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        float v[4];
        const float sz = -ZP * sc[c];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __builtin_fmaf(ubyte_f(braw_use[j], c), sc[c], sz);
        uint32_t o0 = pack_hi(rne_hi(v[0]), rne_hi(v[1])), o1 = pack_hi(rne_hi(v[2]), rne_hi(v[3]));
        *(u32x2*)(&bs[sw4(4 * q + c, rg >> 1) + 2 * (rg & 1)]) = (u32x2){o0, o1};
      }
    } else {
      const float s = s_base[(long long)g2 * stride_sn + (long long)g * stride_sg];
      uint32_t o[8];
      const float sz = -ZP * s;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        uint32_t x = braw_use[w];
        float v0 = __builtin_fmaf(ubyte_f(x, 0), s, sz), v1 = __builtin_fmaf(ubyte_f(x, 1), s, sz);
        float v2 = __builtin_fmaf(ubyte_f(x, 2), s, sz), v3 = __builtin_fmaf(ubyte_f(x, 3), s, sz);
        o[2 * w] = pack_hi(rne_hi(v0), rne_hi(v1));
        o[2 * w + 1] = pack_hi(rne_hi(v2), rne_hi(v3));
      }
      *(u32x4*)(&bs[sw4(n2, 2 * h2)]) = (u32x4){o[0], o[1], o[2], o[3]};
      *(u32x4*)(&bs[sw4(n2, 2 * h2 + 1)]) = (u32x4){o[4], o[5], o[6], o[7]};
    }
  };

  f32x8 acc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
  const int NS = K / BKS;
  // Prefetch distance 2 with a 3-set register ring (static names; no register copies => no forced s_waitcnt):
  // stage s computes from A set s%3 (B from LDS buf s&1); A/B raw for s+1 already in flight in set (s+1)%3;
  // loads for s+2 are issued into set (s+2)%3 at the start of stage s; B(s+1) is dequantized into LDS at its end.
  auto step = [&](int s, const u32x4* cur, const uint32_t* b1, u32x4* n2, uint32_t* b2) {
    const int buf = s & 1;
    if (s + 2 < NS) { loadA(n2, (s + 2) * BKS); loadB(b2, (s + 2) * BKS); }
    if (wave_active) {
      const uint32_t* bs = cx.Bs[buf];
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
        u32x4 x0 = cur[2 * kc], x1 = cur[2 * kc + 1];
        bf16x16 af = __builtin_bit_cast(bf16x16, (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]});
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int n = 64 * wn + 16 * j + l16;
          u32x4 y0 = *(const u32x4*)(&bs[sw4(n, 2 * kc)]);
          u32x4 y1 = *(const u32x4*)(&bs[sw4(n, 2 * kc + 1)]);
          bf16x16 bf = __builtin_bit_cast(bf16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
          acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc[j]);
        }
      }
    }
    if (s + 1 < NS) storeB(buf ^ 1, (s + 1) * BKS, b1);
    __syncthreads();
  };
  // prologue: stage 0 -> LDS (set X raw), A(0) in X; stage 1 in flight in set Y
  loadA(aX, 0); loadB(bX, 0);
  if (NS > 1) { loadA(aY, BKS); loadB(bY, BKS); }
  storeB(0, 0, bX);
  __syncthreads();
  for (int s = 0; s < NS; s += 3) {
    step(s, aX, bY, aZ, bZ);
    if (s + 1 < NS) step(s + 1, aY, bZ, aX, bX);
    if (s + 2 < NS) step(s + 2, aZ, bX, aY, bY);
  }

  if constexpr (!FUSE) {
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
  } else {
    uint16_t* up = (uint16_t*)cx.Bs[0];  // 64 x 64 bf16 = 8 KB <= 10 KB
    if (wn == 1 && wave_active) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = wm * 16 + 2 * e + (lane >> 4);
#pragma unroll
        for (int j = 0; j < 4; ++j) up[r * 64 + 16 * j + l16] = f32_to_bf16_rne_nan(acc[j][e]);
      }
    }
    __syncthreads();
    if (wn == 0 && wave_active) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = wm * 16 + 2 * e + (lane >> 4);
        const int tok = s_tok[r];
        if (tok >= num_valid) continue;
        __bf16* crow = C + (long long)tok * stride_cm + n0 + l16;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          *(uint16_t*)(crow + 16 * j) = silu_mul_bf16<SILUV>(f32_to_bf16_rne_nan(acc[j][e]), up[r * 64 + 16 * j + l16]);
      }
    }
  }
}

// Unified kernel: global-expert alignment, per-WG pool dispatch. tab[e] = (pool << 16) | local.
template <int MODE8, bool FUSE, int SILUV>
__device__ __forceinline__ void unified(
    const __bf16* A, const uint8_t* B4, const float* S4, const uint8_t* B8, const float* S8, __bf16* C,
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
  if (e < 0) return;  // unreachable with global alignment (no expert_map); defensive
  const int t = tab[e], pool = t >> 16, local = t & 0xffff;
  Ctx cx{Bs, s_tok};
  if (pool == 0)
    body<0, FUSE, SILUV>(cx, A, B4 + (long long)local * b4e, S4 + (long long)local * s4e, C, N, K, nv, tk, sam, b4k, b4n, s4g, s4n, scm);
  else
    body<MODE8, FUSE, SILUV>(cx, A, B8 + (long long)local * b8e, S8 + (long long)local * s8e, C, N, K, nv, tk, sam, b8k, b8n, s8g, s8n, scm);
}

#define UARGS                                                                                                                 \
  const __bf16 *A, const uint8_t *B4, const float *S4, const uint8_t *B8, const float *S8, __bf16 *C, const int32_t *sid,   \
      const int32_t *eid, const int32_t *ntpp, const int32_t *tab, int N, int K, int nv, int tk, long long sam, long long b4e, \
      long long b4k, long long b4n, long long s4e, long long s4g, long long s4n, long long b8e, long long b8k, long long b8n,   \
      long long s8e, long long s8g, long long s8n, long long scm
#define UPASS A, B4, S4, B8, S8, C, sid, eid, ntpp, tab, N, K, nv, tk, sam, b4e, b4k, b4n, s4e, s4g, s4n, b8e, b8k, b8n, s8e, s8g, s8n, scm

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10,10))) moe6w10_up(UARGS) { unified<1, false, 0>(UPASS); }
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10,10))) moe6w10_down(UARGS) { unified<2, false, 0>(UPASS); }
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10,10))) moe6w10_up_silu0(UARGS) { unified<1, true, 0>(UPASS); }
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10,10))) moe6w10_up_silu1(UARGS) { unified<1, true, 1>(UPASS); }
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10,10))) moe6w10_up_silu2(UARGS) { unified<1, true, 2>(UPASS); }
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10,10))) moe6w10_up_silu3(UARGS) { unified<1, true, 3>(UPASS); }
