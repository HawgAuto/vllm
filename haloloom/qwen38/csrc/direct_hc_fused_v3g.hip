// HC fused prefill v3g (= v3e + conflict-free s16 LDS swizzle; NON-BITWISE arm; v3d + exact-out formula + xn layout switch), gfx1151 WMMA bf16 -> fp32.  Packed-xn pipeline (DRAM-locality design).
//
// Measured on gfx1151 (hc-v3-diag-01/02, hc-v3-screen-03): HC GEMMs are DRAM-locality bound when xn is read in
// row-major layout (64-B pieces at a 20-KB stride); packed-block addressing of A cut hc_dn3 4.34 -> 1.51 ms.
// Layout P64:  xn_p[mt][kb][64][32] bf16 (mt = row/64, kb = col/32, KB = K/32) -> every 64x32 tile is a contiguous
// 4-KB block.  Wd_p[kb][336][32] (pre-packed once at load).
//
// hc_cnp      : combine + RMSNorm + pack.  grid (HC, ceil(M/8)) (stream fastest -> block rows shared in L2),
//               256 thr = 8 waves, wave = one (row, stream).  out = bf16(fma(block, 2*sigmoid(inj/HC), res)) stored
//               row-major (production-exact formula, v2-verified); ss per row (lane-sequential fma + xor tree);
//               rr = 1/sqrt(ss/H + eps); xn = bf16(fma(o*rr, w, o*rr)) stored P64.  w_stride 0 (shared [H]) or H.
// hc_dn4_r{RG}_k{KS}: hc_dn3 (2 register + 2 LDS stages, 3 col-waves x RG row-groups) reading xn_p / Wd_p.
// hc_ugmp_{c|r}{MW}  : hc_ugm_xn arithmetic (bit-identical y given identical inputs) with the epilogue reading xn_p;
//               c = column tiles fastest (nb1 order), r = row tiles fastest.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define RS 20
__device__ __forceinline__ int sw4(int n, int ch) { return n * RS + 4 * (ch ^ ((n >> 4) & 3)); }
// v3g: s16 swizzle (r>>2)&3 -> (r>>1)&3.  hc-counters-01 (M=4096): hc_dn5_r8_k1_l1 LDS 105% active, 48% bank-conflict cycles.
// Rows are 64 B apart; a ds_read_b128 pass serves 8 lanes = rows n..n+7 of one logical chunk.  With 128-B bank windows rows r and
// r+2 collided under (r>>2)&3; (r>>1)&3 gives rows 0..7 eight distinct 16-B bank groups.  Layout-only -> outputs bitwise == v3e.
__device__ __forceinline__ int s16(int r, int c) { return r * 16 + 4 * (c ^ ((r >> 1) & 3)); }
__device__ __forceinline__ float bf2f(uint32_t bits16) { return __uint_as_float(bits16 << 16); }
__device__ __forceinline__ uint32_t f2bf_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ float bfround(float f) { return __uint_as_float(f2bf_rne(f) << 16); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ bf16x16 cat16(u32x4 x0, u32x4 x1) {
  return __builtin_bit_cast(bf16x16, (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]});
}

#define DN_N 336

// xn layouts: LY 0 = P64 [mt][col/32][64][32]; LY 1 = B256 [mt][col/256][64][256]; LY 2 = row-major [Mp][K]
template <int LY>
__device__ __forceinline__ long long xaddr(int row, int col, int K) {
  if (LY == 0) return (((long long)(row >> 6) * (K >> 5) + (col >> 5)) * 64 + (row & 63)) * 32 + (col & 31);
  if (LY == 1) return (((long long)(row >> 6) * (K >> 8) + (col >> 8)) * 64 + (row & 63)) * 256 + (col & 255);
  return (long long)row * K + col;
}

// ------------------------------------------------------------------------------------------------ hc_cnp
template <int LY, int RSQ>
__device__ __forceinline__ void hc_cnp_body(
    const __bf16* __restrict__ res, const __bf16* __restrict__ blk, const __bf16* __restrict__ injl,
    const __bf16* __restrict__ w, __bf16* __restrict__ out, __bf16* __restrict__ xnp, int M, int H, int HC,
    long long s_res, long long s_blk, long long s_inj, long long s_out, long long w_stride, float eps) {
  const int s = blockIdx.x, lane = threadIdx.x & 31;
  const int row = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (row >= M) return;
  // production Triton lowering (ISA dump hc-v3d-cnpdiag-01): v_exp_f32 fast exp + precise division
  const float injv = 2.0f * (1.0f / (1.0f + __expf(-(bf2f(*(const uint16_t*)(injl + (long long)row * s_inj + s)) / (float)HC))));
  const __bf16* rp = res + (long long)row * s_res + (long long)s * H;
  const __bf16* bp = blk + (long long)row * s_blk;
  __bf16* op = out + (long long)row * s_out + (long long)s * H;
  const int NCH = H / 256;  // 16-B chunks per lane (H = 2560 -> 10)
  uint32_t ov[10][4];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    if (i < NCH) {
      const int col = 8 * (lane + 32 * i);
      const u32x4 r4 = *(const u32x4*)(rp + col);
      const u32x4 b4 = *(const u32x4*)(bp + col);
      u32x4 o4;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float o0 = bfround(__builtin_fmaf(bf2f(b4[q] & 0xffffu), injv, bf2f(r4[q] & 0xffffu)));
        const float o1 = bfround(__builtin_fmaf(bf2f(b4[q] >> 16), injv, bf2f(r4[q] >> 16)));
        ss = __builtin_fmaf(o0, o0, ss);
        ss = __builtin_fmaf(o1, o1, ss);
        o4[q] = (__float_as_uint(o0) >> 16) | (__float_as_uint(o1) & 0xffff0000u);
        ov[i][q] = o4[q];
      }
      *(u32x4*)(op + col) = o4;
    }
  }
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) ss += __shfl_xor(ss, m, 32);
  const float rr = RSQ ? __builtin_amdgcn_rsqf(ss / (float)H + eps) : 1.0f / sqrtf(ss / (float)H + eps);
  const __bf16* wp = w + (long long)s * w_stride;
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    if (i < NCH) {
      const int col = 8 * (lane + 32 * i);
      const u32x4 w4 = *(const u32x4*)(wp + col);
      u32x4 x4;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float y0 = bf2f(ov[i][q] & 0xffffu) * rr, y1 = bf2f(ov[i][q] >> 16) * rr;
        const uint32_t x0 = f2bf_rne(__builtin_fmaf(y0, bf2f(w4[q] & 0xffffu), y0));
        const uint32_t x1 = f2bf_rne(__builtin_fmaf(y1, bf2f(w4[q] >> 16), y1));
        x4[q] = x0 | (x1 << 16);
      }
      *(u32x4*)(xnp + xaddr<LY>(row, s * H + col, HC * H)) = x4;
    }
  }
}

#define CNP_INST(LY, RSQ)                                                                                             \
  extern "C" __global__ void __launch_bounds__(256) hc_cnp_l##LY##_q##RSQ(                                          \
      const __bf16* __restrict__ res, const __bf16* __restrict__ blk, const __bf16* __restrict__ injl,               \
      const __bf16* __restrict__ w, __bf16* __restrict__ out, __bf16* __restrict__ xnp, int M, int H, int HC,         \
      long long s_res, long long s_blk, long long s_inj, long long s_out, long long w_stride, float eps) {            \
    hc_cnp_body<LY, RSQ>(res, blk, injl, w, out, xnp, M, H, HC, s_res, s_blk, s_inj, s_out, w_stride, eps);          \
  }
CNP_INST(0, 0)
CNP_INST(0, 1)
CNP_INST(1, 1)
CNP_INST(2, 1)

// ------------------------------------------------------------------------------------------------ hc_dn5 (packed A/B)

template <int RG, int KS, int LY>
__device__ __forceinline__ void hc_dn5_body(const __bf16* __restrict__ xn, const __bf16* __restrict__ wd,
                                            __bf16* __restrict__ lora, __bf16* __restrict__ inj, float* __restrict__ part,
                                            unsigned int* __restrict__ cnt, int M, int K, int R, int HC, long long s_xn,
                                            long long s_lora, long long s_inj) {
  constexpr int T = 96 * RG, TM = 16 * RG;
  constexpr int ACH = TM * 4, BCH = DN_N * 4;
  constexpr int NA = (ACH + T - 1) / T, NB = (BCH + T - 1) / T;
  __shared__ __attribute__((aligned(16))) uint32_t As[2][TM * 16];
  __shared__ __attribute__((aligned(16))) uint32_t Bs[2][DN_N * 16];
  __shared__ int s_last;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
  const int cw = wave % 3, rg = wave / 3;
  const int m0 = blockIdx.x * TM, ks = blockIdx.y;
  const int KSL = K / KS, kbeg = ks * KSL, NS = KSL / 32;

  struct Stage { u32x4 a[NA]; u32x4 b[NB]; };
  Stage R0, R1;
  auto gload = [&](Stage& S, int t) {
    const int k0 = kbeg + t * 32;
#pragma unroll
    for (int i = 0; i < NA; ++i) {
      const int c = tid + T * i;
      if (c < ACH) {
        const int row = m0 + (c >> 2);  // rows >= M read the packed padding (never stored)
        S.a[i] = *(const u32x4*)(xn + xaddr<LY>(row, k0 + 8 * (c & 3), K));
      }
    }
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      const int c = tid + T * i;
      if (c < BCH) S.b[i] = *(const u32x4*)(wd + ((long long)(k0 >> 5) * DN_N + (c >> 2)) * 32 + 8 * (c & 3));
    }
  };
  auto sstore = [&](const Stage& S, int buf) {
#pragma unroll
    for (int i = 0; i < NA; ++i) {
      const int c = tid + T * i;
      if (c < ACH) *(u32x4*)(&As[buf][s16(c >> 2, c & 3)]) = S.a[i];
    }
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      const int c = tid + T * i;
      if (c < BCH) *(u32x4*)(&Bs[buf][s16(c >> 2, c & 3)]) = S.b[i];
    }
  };
  f32x8 acc[7];
#pragma unroll
  for (int j = 0; j < 7; ++j) acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
  auto compute = [&](int buf) {
    const uint32_t* as = As[buf];
    const uint32_t* bs = Bs[buf];
    const int ar = 16 * rg + l16;
#pragma unroll
    for (int kc = 0; kc < 2; ++kc) {
      const bf16x16 af = cat16(*(const u32x4*)(&as[s16(ar, 2 * kc)]), *(const u32x4*)(&as[s16(ar, 2 * kc + 1)]));
#pragma unroll
      for (int j = 0; j < 7; ++j) {
        const int n = 16 * (cw * 7 + j) + l16;
        const bf16x16 bf = cat16(*(const u32x4*)(&bs[s16(n, 2 * kc)]), *(const u32x4*)(&bs[s16(n, 2 * kc + 1)]));
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc[j]);
      }
    }
  };

  gload(R0, 0);
  sstore(R0, 0);
  gload(R0, 1);
  __syncthreads();
  for (int t = 0; t < NS; t += 2) {  // NS even (host-checked)
    if (t + 2 < NS) gload(R1, t + 2);
    compute(0);
    sstore(R0, 1);
    __syncthreads();
    if (t + 3 < NS) gload(R0, t + 3);
    compute(1);
    if (t + 2 < NS) sstore(R1, 0);
    __syncthreads();
  }

  if (KS > 1) {
    float* pp = part + (long long)ks * M * DN_N;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = m0 + 16 * rg + 2 * e + (lane >> 4);
      if (row < M) {
#pragma unroll
        for (int j = 0; j < 7; ++j) pp[(long long)row * DN_N + 16 * (cw * 7 + j) + l16] = acc[j][e];
      }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = (atomicAdd(&cnt[blockIdx.x], 1u) == (unsigned)(KS - 1));
    __syncthreads();
    if (!s_last) return;
    __threadfence();
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = m0 + 16 * rg + 2 * e + (lane >> 4);
      if (row < M) {
#pragma unroll
        for (int j = 0; j < 7; ++j) {
          const long long off = (long long)row * DN_N + 16 * (cw * 7 + j) + l16;
          float s = 0.0f;
#pragma unroll
          for (int q = 0; q < KS; ++q)
            s += (q == ks) ? acc[j][e]
                           : __hip_atomic_load(part + (long long)q * M * DN_N + off, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          acc[j][e] = s;
        }
      }
    }
    if (tid == 0) cnt[blockIdx.x] = 0u;
  }

  const float invhc = 1.0f / (float)HC;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int row = m0 + 16 * rg + 2 * e + (lane >> 4);
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < 7; ++j) {
      const int col = 16 * (cw * 7 + j) + l16;
      const float db = bfround(acc[j][e]);
      if (col < R) {
        const float x = db * invhc;
        *(uint16_t*)(lora + (long long)row * s_lora + col) = (uint16_t)f2bf_rne(x * sigm(x));
      } else if (col < R + HC) {
        *(uint16_t*)(inj + (long long)row * s_inj + (col - R)) = (uint16_t)f2bf_rne(db);
      }
    }
  }
}

#define DN5_INST(RG, KS, LY)                                                                                          \
  extern "C" __global__ void __launch_bounds__(96 * RG) hc_dn5_r##RG##_k##KS##_l##LY(                               \
      const __bf16* __restrict__ xn, const __bf16* __restrict__ wd, __bf16* __restrict__ lora,                      \
      __bf16* __restrict__ inj, float* __restrict__ part, unsigned int* __restrict__ cnt, int M, int K, int R, int HC,  \
      long long s_xn, long long s_lora, long long s_inj) {                                                            \
    hc_dn5_body<RG, KS, LY>(xn, wd, lora, inj, part, cnt, M, K, R, HC, s_xn, s_lora, s_inj);                          \
  }
DN5_INST(8, 1, 0)
DN5_INST(4, 4, 0)
DN5_INST(8, 1, 1)
DN5_INST(4, 4, 1)
DN5_INST(8, 2, 1)
DN5_INST(4, 2, 1)

// ------------------------------------------------------------------------------------------------ hc_ugmq (packed xn epilogue)
template <int MW, int RF, int LY>
__device__ __forceinline__ void hc_ugmq_body(const __bf16* __restrict__ lora, const __bf16* __restrict__ wup,
                                               const __bf16* __restrict__ xn, __bf16* __restrict__ y, int M, int H, int HC,
                                               int R, long long s_lora, long long s_xn, long long s_y) {
  constexpr int T = 64 * MW;            // MW m-waves x 2 n-waves
  constexpr int BCH = 128 * 4;          // 16-byte chunks per B stage (128 cols x 32 K)
  constexpr int NB = (BCH + T - 1) / T;
  __shared__ __attribute__((aligned(16))) uint32_t Bs[2][128 * RS];
  const int m0 = (RF ? blockIdx.x : blockIdx.y) * (16 * MW), n0 = (RF ? blockIdx.y : blockIdx.x) * 128;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
  const int wm = wave % MW, wn = wave / MW;
  const int arow = min(m0 + wm * 16 + l16, M - 1);
  const __bf16* ap = lora + (long long)arow * s_lora;
  f32x8 mix[4], acc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) mix[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
  const int KC = R / 32, NT = HC * KC;
  const long long KBX = (long long)(HC * H) / 32;
  u32x4 bp[NB];
  auto gloadB = [&](int t) {
    const int s = t / KC, k0 = (t - s * KC) * 32;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      const int c = tid + T * i;
      if (c < BCH) bp[i] = *(const u32x4*)(wup + (long long)(s * H + n0 + (c >> 2)) * R + k0 + 8 * (c & 3));
    }
  };
  auto sstoreB = [&](int buf) {
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      const int c = tid + T * i;
      if (c < BCH) *(u32x4*)(&Bs[buf][sw4(c >> 2, c & 3)]) = bp[i];
    }
  };
  gloadB(0);
  sstoreB(0);
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
      const bf16x16 af = cat16(av[2 * kc], av[2 * kc + 1]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int n = 64 * wn + 16 * j + l16;
        const bf16x16 bf = cat16(*(const u32x4*)(&bs[sw4(n, 2 * kc)]), *(const u32x4*)(&bs[sw4(n, 2 * kc + 1)]));
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc[j]);
      }
    }
    if (kci == KC - 1) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + wm * 16 + 2 * e + (lane >> 4);
        const int gc0 = s * H + n0 + 64 * wn;  // multiple of 64 -> j-tiles 0,1 in kblock gc0/32, 2,3 in the next
        const __bf16* xrow0 = xn + xaddr<LY>(r, gc0, HC * H) + l16;
        const __bf16* xrow1 = xn + xaddr<LY>(r, gc0 + 32, HC * H) + l16;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float g = bfround(acc[j][e]);
          const float xv = bf2f(*(const uint16_t*)(((j >> 1) ? xrow1 : xrow0) + 16 * (j & 1)));
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

#define UGMQ_INST(TAG, MW, RF, LY)                                                                                  \
  extern "C" __global__ void __launch_bounds__(64 * MW) hc_ugmq_##TAG##MW##_l##LY(                                  \
      const __bf16* __restrict__ lora, const __bf16* __restrict__ wup, const __bf16* __restrict__ xn,                \
      __bf16* __restrict__ y, int M, int H, int HC, int R, long long s_lora, long long s_xn, long long s_y) {         \
    hc_ugmq_body<MW, RF, LY>(lora, wup, xn, y, M, H, HC, R, s_lora, s_xn, s_y);                                      \
  }
UGMQ_INST(c, 8, 0, 0)
UGMQ_INST(c, 8, 0, 1)
UGMQ_INST(c, 4, 0, 1)
UGMQ_INST(r, 8, 1, 1)
