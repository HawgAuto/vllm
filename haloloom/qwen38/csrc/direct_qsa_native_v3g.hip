// qsa_native_v3g.hip = v3e with a PADDED K layout ([tok][264], conflict-free, immediate-offset fragment addresses) instead of XOR swizzle
// qsa_native_v3e.hip = v3d + 4-token V items (v2 layout) + K-staging unroll capped at 2
// qsa_native_v3d.hip = v3b with the V gather/V^T store moved after the P/sum stores (score temporaries dead)
// qsa_native_v3b.hip = v3 without register V prefetch (V gathered after the scores; avoids spills)
// qsa_native_v3.hip = v2 + K tile staged cooperatively in LDS (shared by the 3 row-tile waves; XOR piece swizzle),
//   V prefetched to registers during scores and written as V^T into the SAME 32 KB buffer after scores (8-token items),
//   so LDS stays 43 KB. Operand values / arithmetic identical to v2.
// qsa_native_v2.hip = v1 + PV loop c-outer/t-inner (one live P fragment; per-accumulator chain order unchanged)
//   + amdgpu_waves_per_eu(6,6) so two 12-wave WGs co-reside per WGP. Arithmetic identical to v1.
// (v1 header follows)
// qsa_native_v1.hip — native gfx1151 (RDNA3.5, wave32) WMMA port of the production QSA union attention kernel
// overlays/same-process-moe-abc-v1/direct_qsa_union_kernel_v1.py::_attend (QG=4, GH=12, PH=16, HD=256, BN=64).
//
// FAITHFUL-PORT CONTRACT (derived from Triton's production AMDGCN for _attend, qsa-native-v1/attend_prod.amdgcn):
//  * scores tile (16 tok x 16 rows) = 16 chained v_wmma_f32_16x16x16_bf16 over d-chunks 0..15 from C=0,
//    srcA = keys fragment (M = token), srcB = query fragment (N = row)  [Triton #mma isTranspose=true].
//  * s = acc * scale (fp32 mul, scale = 256^-0.5 * log2(e)); s = use ? s : -1e20.
//  * rowmax exact (any order). alpha = v_exp_f32(maximum - newmax). p = use ? v_exp_f32(s - newmax) : 0.
//  * rowsum: per lane-half a balanced binary tree over 32 leaves ordered (token-tile n = 0..3, vgpr e = 0..7):
//      T_n = ((e0+e1)+(e2+e3)) + ((e4+e5)+(e6+e7));  R_half = (T0+T1) + (T2+T3);  rowsum = R_0 + R_1.
//  * norm = fma(norm, alpha, rowsum)  (v_dual_fmac_f32 in Triton's code).
//  * acc = WMMA chain over token-chunks 0..3 from C = acc * alpha, srcA = V^T fragment (M = d), srcB = P^T (N = row),
//    p converted to bf16 with RNE.
//  * out = norm > 0 ? acc / max(norm, 1e-20) : 0  (IEEE division, correctly rounded), bf16 RNE.
//  * FP mode: f32 denormals preserved (denorm mode 3), IEEE mode — HIP defaults on gfx1151 (verified in descriptor).
// Structural changes (do not alter per-row arithmetic):
//  * 48 real rows (4 queries x 12 heads) in 3 WMMA row-tiles instead of 64 padded rows.
//  * 12 waves: wave (rt, dq) computes score token-tile dq for row-tile rt (no duplicated QK^T) and PV d-tiles 4dq..4dq+3;
//    per-tile subtree sums are exchanged through LDS and combined exactly at the tree's top two levels.
//  * key fragments loaded straight from global as 32 B contiguous per lane; V^T staged once per union tile in LDS with an
//    XOR chunk swizzle (conflict-free, no padding); query fragments register-resident (no scratch).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define HD 256
#define GH 12
#define QG 4
#define ROWS 48
#define BN 64
#define NTHR 384
#define PPITCH 72   // sP row pitch in bf16 elements (144 B: conflict-free 32 B row reads)

__device__ __forceinline__ bf16x16 as_bf16x16(u32x8 v) { return __builtin_bit_cast(bf16x16, v); }

__device__ __forceinline__ uint32_t bf16_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;  // NaN (Triton: v_cmp_o_f32 + 0x7fff); unreachable for p in [0,1]
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}

#define KPITCH (HD + 8)   // 528 B rows: lanes 0..15 hit banks 4i mod 64 -> conflict-free b128 fragment reads
__device__ __forceinline__ int k_index(int tok, int pc) {  // element index into K [64 tok][KPITCH], 16-B piece pc (0..31)
  return tok * KPITCH + (pc << 3);
}

__device__ __forceinline__ int vt_index(int d, int tok) {  // element index into V^T [256][64] with chunk swizzle
  int sw = (d >> 1) & 7;
  return d * BN + ((((tok >> 3) ^ sw)) << 3) + (tok & 7);
}

extern "C" __global__ void __launch_bounds__(NTHR) __attribute__((amdgpu_waves_per_eu(6, 6))) qsa_attend_v3g(
    const uint16_t* __restrict__ Q, const uint16_t* __restrict__ K, const uint16_t* __restrict__ V,
    const int32_t* __restrict__ TAB, const int32_t* __restrict__ U, const int32_t* __restrict__ MASK,
    const int32_t* __restrict__ COUNTS, uint16_t* __restrict__ O,
    int64_t SQ0, int64_t SQ1, int64_t SK0, int64_t SK1, int64_t SK2, int64_t SV0, int64_t SV1, int64_t SV2,
    int64_t SO0, int64_t SO1, int NR, int NP, int PAGE, int TW, int MAXU, int NT) {
  __shared__ __attribute__((aligned(16))) uint16_t sVt[BN * (HD + 8)];    // 33 KB: K tile [tok][264] during scores, then V^T [d][tok] (first 32 KB)
  __shared__ __attribute__((aligned(16))) uint16_t sP[ROWS * PPITCH];     // 6.75 KB
  __shared__ int64_t sOffK[BN];
  __shared__ int64_t sOffV[BN];
  __shared__ uint32_t sBits[BN / 4];
  __shared__ float sMax[4][ROWS];
  __shared__ float sSum[4][2][ROWS];

  const int group = blockIdx.x, kvhead = blockIdx.y;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, lr = lane & 15, hb = lane >> 4;
  const int rt = wave >> 2, dq = wave & 3;
  const int rr = rt * 16 + lr;
  const int qslot = rr / GH, hh = rr - qslot * GH;
  const int grow = group * QG + qslot;
  const bool active = grow < NR;
  const int head = kvhead * GH + hh;
  const float scale = __builtin_bit_cast(float, 0x3db8aa3bu);  // (256^-0.5) * log2(e) as fp32
  const float NEG = -1.0e20f;

  // query fragments (B operand, N = row): 16 d-chunks x 16 bf16, register resident
  u32x8 qf[16];
  {
    const uint16_t* qp = Q + (int64_t)grow * SQ0 + (int64_t)head * SQ1;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      u32x4 a = {0, 0, 0, 0}, b = {0, 0, 0, 0};
      if (active) {
        a = *(const u32x4*)(qp + k * 16);
        b = *(const u32x4*)(qp + k * 16 + 8);
      }
      qf[k] = (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    }
  }
  int count = 0;
  for (int t = 0; t < NT; ++t) count += COUNTS[group * NT + t];
  count = count < MAXU ? count : MAXU;
  const int ntiles = (count * 4 + BN - 1) / BN;

  float maximum = NEG, norm = 0.0f;
  f32x8 acc[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) acc[t] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  for (int tile = 0; tile < ntiles; ++tile) {
    // (a) column metadata
    if (tid < BN) {
      const int col = tile * BN + tid;
      const int ui = col >> 2;
      const bool inu = ui < count;
      const int block = inu ? U[group * MAXU + ui] : 0;
      const int logical = block * 4 + (col & 3);
      const int pg = logical / PAGE;
      const int off = logical - pg * PAGE;
      bool valid = inu && (pg < TW);
      const int physical = valid ? TAB[pg] : -1;
      valid = valid && (physical >= 0) && (physical < NP);
      const int64_t phys = physical > 0 ? (int64_t)physical : 0;
      sOffK[tid] = valid ? phys * SK0 + (int64_t)off * SK1 + (int64_t)kvhead * SK2 : (int64_t)-1;
      sOffV[tid] = valid ? phys * SV0 + (int64_t)off * SV1 + (int64_t)kvhead * SV2 : (int64_t)-1;
    }
    if (tid < BN / 4) {
      const int ui = tile * (BN / 4) + tid;
      sBits[tid] = ui < count ? (uint32_t)MASK[group * MAXU + ui] : 0u;
    }
    __syncthreads();
    // (b) K staging into LDS: 2048 16-B pieces (64 tokens x 32 pieces), consecutive threads walk a token row
#pragma unroll 2
    for (int it = tid; it < BN * 32; it += NTHR) {
      const int tok = it >> 5, pc = it & 31;
      const int64_t o = sOffK[tok];
      const u32x4 kv = o >= 0 ? *(const u32x4*)(K + o + pc * 8) : (u32x4){0, 0, 0, 0};
      *(u32x4*)(sVt + k_index(tok, pc)) = kv;
    }
    __syncthreads();
    // (c) scores for token-tile dq, row-tile rt: A = K fragment from LDS (token 16dq+lr, d-chunk k)
    f32x8 S = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
    {
      const int tok = dq * 16 + lr;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const u32x4 a = *(const u32x4*)(sVt + k_index(tok, 2 * k));
        const u32x4 b = *(const u32x4*)(sVt + k_index(tok, 2 * k + 1));
        const u32x8 kf = (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(kf), as_bf16x16(qf[k]), S);
      }
    }
    // (d) scale + mask; lane holds (token = 16dq + 2e + hb, row = rr)
    float s[8];
    bool use[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int c = dq * 16 + 2 * e + hb;
      const uint32_t bits = sBits[c >> 2];
      const bool mem = ((bits >> (qslot * 4 + (c & 3))) & 1u) != 0u;
      use[e] = active && (sOffK[c] >= 0) && mem;
      const float sc = S[e] * scale;
      s[e] = use[e] ? sc : NEG;
    }
    // (e) partial max over this token-tile (exact), then exchange
    float m = fmaxf(fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3])), fmaxf(fmaxf(s[4], s[5]), fmaxf(s[6], s[7])));
    m = fmaxf(m, __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, m), __builtin_bit_cast(int, m),
                                                                          0x76543210, 0xfedcba98, false, false)));
    if (hb == 0) sMax[dq][rr] = m;
    __syncthreads();   // all waves finished reading K from sVt
    const float newmax = fmaxf(maximum, fmaxf(fmaxf(sMax[0][rr], sMax[1][rr]), fmaxf(sMax[2][rr], sMax[3][rr])));
    const float alpha = __builtin_amdgcn_exp2f(maximum - newmax);
    float p[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) p[e] = use[e] ? __builtin_amdgcn_exp2f(s[e] - newmax) : 0.0f;
    // (g) subtree for this token-tile (leaf order e = 0..7)
    const float T = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
    sSum[dq][hb][rr] = T;
    // (h) P (bf16) to LDS: row rr, tokens 16dq + 2e + hb
#pragma unroll
    for (int e = 0; e < 8; ++e) sP[rr * PPITCH + dq * 16 + 2 * e + hb] = (uint16_t)bf16_rne(p[e]);
    for (int it = tid; it < 512; it += NTHR) {          // 4-token V items (v2 layout): 16 token-quads x 32 d-chunks
      const int c8 = it & 31, tq = it >> 5;
      u32x4 v[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int64_t o = sOffV[tq * 4 + t];
        v[t] = o >= 0 ? *(const u32x4*)(V + o + c8 * 8) : (u32x4){0, 0, 0, 0};
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t sel = (j & 1) ? 0x07060302u : 0x05040100u;
        const uint32_t w01 = __builtin_amdgcn_perm(v[1][j >> 1], v[0][j >> 1], sel);
        const uint32_t w23 = __builtin_amdgcn_perm(v[3][j >> 1], v[2][j >> 1], sel);
        *(uint2*)(sVt + vt_index(c8 * 8 + j, tq * 4)) = make_uint2(w01, w23);
      }
    }
    __syncthreads();
    // (i) exact tree top: R_half = (T0+T1)+(T2+T3); rowsum = R_0 + R_1; norm = fma(norm, alpha, rowsum)
    const float R0 = (sSum[0][0][rr] + sSum[1][0][rr]) + (sSum[2][0][rr] + sSum[3][0][rr]);
    const float R1 = (sSum[0][1][rr] + sSum[1][1][rr]) + (sSum[2][1][rr] + sSum[3][1][rr]);
    const float rowsum = hb == 0 ? (R0 + R1) : (R1 + R0);
    norm = __builtin_fmaf(norm, alpha, rowsum);
    // (j) PV: acc[t] (d-tile 4dq+t) = chain over token-chunks c = 0..3 from C = acc * alpha
#pragma unroll
    for (int t = 0; t < 4; ++t) acc[t] = acc[t] * alpha;   // C seed = acc * alpha (Triton: acc=acc*alpha[:, None])
#pragma unroll
    for (int c = 0; c < 4; ++c) {                           // chunk-outer: each acc[t] chain still runs c = 0..3 in order
      const u32x4 pa = *(const u32x4*)(sP + rr * PPITCH + c * 16);
      const u32x4 pb = *(const u32x4*)(sP + rr * PPITCH + c * 16 + 8);
      const u32x8 pf = (u32x8){pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int d = (dq * 4 + t) * 16 + lr;
        const u32x4 a = *(const u32x4*)(sVt + vt_index(d, c * 16));
        const u32x4 b = *(const u32x4*)(sVt + vt_index(d, c * 16 + 8));
        const u32x8 vf = (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(vf), as_bf16x16(pf), acc[t]);
      }
    }
    maximum = newmax;
    __syncthreads();
  }
  // epilogue: lane holds (d = 16(4dq+t) + 2e + hb, row = rr)
  if (active) {
    uint16_t* op = O + (int64_t)grow * SO0 + (int64_t)head * SO1;
    const float den = fmaxf(norm, 1.0e-20f);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float val = norm > 0.0f ? acc[t][e] / den : 0.0f;
        op[(dq * 4 + t) * 16 + 2 * e + hb] = (uint16_t)bf16_rne(val);
      }
    }
  }
}
